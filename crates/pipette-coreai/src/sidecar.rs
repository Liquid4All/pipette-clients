//! Resolve / build the bundled Swift sidecar.

use std::{
    path::{Path, PathBuf},
    process::Command,
};

use anyhow::Context;
use pipette_plan_types::AppleCoreAiMacosPipette;

/// Env override for the sidecar binary path (bypasses building).
const SIDECAR_BIN_ENV: &str = "PIPETTE_COREAI_SIDECAR";
/// Env override for the Swift package directory (when the binary is off the
/// build machine and `CARGO_MANIFEST_DIR` no longer points at sources).
const SIDECAR_SWIFT_ENV: &str = "PIPETTE_COREAI_SWIFT";
/// Env override for the cache root (tests; operators who want a non-default
/// location). Default: `~/Library/Caches/pipette-coreai`.
const SIDECAR_CACHE_ENV: &str = "PIPETTE_COREAI_CACHE";

/// Resolve the Swift sidecar binary.
///
/// Honors `PIPETTE_COREAI_SIDECAR` (a prebuilt path — used by CI and local
/// dev to skip the Swift build). Otherwise builds the crate's bundled
/// `swift/` package into a **pin-keyed** cache directory and copies the
/// product there. A pin bump therefore cannot keep executing a stale binary.
pub fn require_coreai_sidecar() -> anyhow::Result<PathBuf> {
    if let Some(path) = std::env::var_os(SIDECAR_BIN_ENV) {
        let path = PathBuf::from(path);
        if !path.is_file() {
            anyhow::bail!(
                "{SIDECAR_BIN_ENV}={path} is not a file",
                path = path.display()
            );
        }
        return Ok(path);
    }
    let cache = sidecar_cache_dir()?;
    let bin = cache.join("pipette-coreai-sidecar");
    if bin.is_file() {
        return Ok(bin);
    }
    build_sidecar(&bin)
}

/// Delete the sidecar cache root (every pin). Returns whether anything was
/// removed. Used by `pipette runtimes remove` for this OS-bundled runtime.
pub fn clear_sidecar_cache() -> anyhow::Result<bool> {
    let root = sidecar_cache_root()?;
    if !root.exists() {
        return Ok(false);
    }
    std::fs::remove_dir_all(&root)
        .with_context(|| format!("removing the Core AI sidecar cache at {}", root.display()))?;
    Ok(true)
}

fn bundled_pin() -> String {
    AppleCoreAiMacosPipette::bundled()
        .packages
        .coreai_models
        .repository_version
        .to_string()
}

fn sidecar_cache_root() -> anyhow::Result<PathBuf> {
    if let Some(path) = std::env::var_os(SIDECAR_CACHE_ENV) {
        return Ok(PathBuf::from(path));
    }
    let home = std::env::var_os("HOME").context("cannot resolve $HOME for the sidecar cache")?;
    Ok(PathBuf::from(home).join("Library/Caches/pipette-coreai"))
}

/// `cache_root/<pin>/` — one compiled sidecar per bundled pin.
fn sidecar_cache_dir() -> anyhow::Result<PathBuf> {
    let cache = sidecar_cache_root()?.join(bundled_pin());
    std::fs::create_dir_all(&cache)
        .with_context(|| format!("failed to create {}", cache.display()))?;
    Ok(cache)
}

fn swift_package_dir() -> anyhow::Result<PathBuf> {
    if let Some(path) = std::env::var_os(SIDECAR_SWIFT_ENV) {
        let path = PathBuf::from(path);
        if path.join("Package.swift").is_file() {
            return Ok(path);
        }
        anyhow::bail!(
            "{SIDECAR_SWIFT_ENV}={} has no Package.swift",
            path.display()
        );
    }
    let bundled = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("swift");
    if bundled.join("Package.swift").is_file() {
        return Ok(bundled);
    }
    if let Ok(exe) = std::env::current_exe() {
        for ancestor in exe.ancestors().take(8) {
            for candidate in [
                ancestor.join("crates/pipette-coreai/swift"),
                ancestor.join("swift"),
            ] {
                if candidate.join("Package.swift").is_file() {
                    return Ok(candidate);
                }
            }
        }
    }
    anyhow::bail!(
        "bundled Swift package not found (this pipette binary was built without \
         swift/ nearby, common for a distributed binary). Point {SIDECAR_BIN_ENV} \
         at a prebuilt `pipette-coreai-sidecar`, or {SIDECAR_SWIFT_ENV} at the \
         crate's swift/ directory."
    )
}

fn build_sidecar(bin: &Path) -> anyhow::Result<PathBuf> {
    let src = swift_package_dir()?;
    log::info!(
        "building pipette-coreai sidecar from {} (cache {})",
        src.display(),
        bin.display()
    );
    let mut cmd = Command::new("swift");
    cmd.current_dir(&src).arg("build").arg("-c").arg("release");
    pipette_subprocess::echo_info(&cmd);
    let status = cmd
        .status()
        .context("failed to run `swift build` for the pipette-coreai sidecar")?;
    if !status.success() {
        anyhow::bail!("pipette-coreai sidecar `swift build` exited with {status}");
    }

    let show = Command::new("swift")
        .current_dir(&src)
        .args(["build", "--show-bin-path", "-c", "release"])
        .output()
        .context("failed to run `swift build --show-bin-path`")?;
    if !show.status.success() {
        anyhow::bail!(
            "`swift build --show-bin-path -c release` failed: {}",
            String::from_utf8_lossy(&show.stderr)
        );
    }
    let bin_dir = String::from_utf8(show.stdout)
        .context("swift --show-bin-path stdout was not UTF-8")?
        .trim()
        .to_owned();
    let built = PathBuf::from(bin_dir).join("pipette-coreai-sidecar");
    if !built.is_file() {
        anyhow::bail!("swift build produced no binary at {}", built.display());
    }
    std::fs::copy(&built, bin).with_context(|| {
        format!(
            "failed to copy sidecar from {} to {}",
            built.display(),
            bin.display()
        )
    })?;
    Ok(bin.to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_dir_is_keyed_on_the_bundled_pin() -> anyhow::Result<()> {
        let tmp =
            std::env::temp_dir().join(format!("pipette-coreai-cache-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        // SAFETY: test-only env for this process; not run in parallel with
        // other tests that read the same vars.
        std::env::set_var(SIDECAR_CACHE_ENV, &tmp);
        let dir = sidecar_cache_dir()?;
        assert!(
            dir.ends_with(bundled_pin()),
            "expected pin-keyed cache, got {}",
            dir.display()
        );
        assert!(dir.starts_with(&tmp));
        let _ = std::fs::remove_dir_all(&tmp);
        std::env::remove_var(SIDECAR_CACHE_ENV);
        Ok(())
    }

    #[test]
    fn clear_sidecar_cache_is_a_no_op_when_missing() -> anyhow::Result<()> {
        let tmp = std::env::temp_dir().join(format!(
            "pipette-coreai-cache-missing-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&tmp);
        std::env::set_var(SIDECAR_CACHE_ENV, &tmp);
        assert!(!clear_sidecar_cache()?);
        std::env::remove_var(SIDECAR_CACHE_ENV);
        Ok(())
    }
}
