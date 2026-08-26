//! Resolve / build the bundled Swift sidecar.

use std::{
    path::{Path, PathBuf},
    process::Command,
};

use anyhow::Context;

/// Env override for the sidecar binary path (bypasses building).
const SIDECAR_BIN_ENV: &str = "PIPETTE_COREAI_SIDECAR";

/// Resolve the Swift sidecar binary.
///
/// Honors `PIPETTE_COREAI_SIDECAR` (a prebuilt path — used by CI and local
/// dev to skip the Swift build). Otherwise builds the crate's bundled
/// `swift/` package into `~/Library/Caches/pipette-coreai/` and copies the
/// product there. The product directory is taken from
/// `swift build --show-bin-path -c release` so it is not tied to one
/// toolchain's layout (Xcode-beta uses `.build/out/Products/Release`;
/// stock SwiftPM uses `.build/<triple>/release`).
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

/// `~/Library/Caches/pipette-coreai/` — one compiled sidecar reused across
/// workspaces. Not keyed on the model store: the sidecar is the runtime, not
/// a per-model artifact.
fn sidecar_cache_dir() -> anyhow::Result<PathBuf> {
    let cache = std::env::var_os("HOME")
        .map(|home| PathBuf::from(home).join("Library/Caches/pipette-coreai"))
        .context("cannot resolve $HOME for the sidecar cache")?;
    std::fs::create_dir_all(&cache)
        .with_context(|| format!("failed to create {}", cache.display()))?;
    Ok(cache)
}

fn build_sidecar(bin: &Path) -> anyhow::Result<PathBuf> {
    let src = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("swift");
    if !src.join("Package.swift").is_file() {
        anyhow::bail!(
            "bundled Swift package not found at {}: this pipette binary was \
             built without swift/ (common for a distributed binary). Point \
             {SIDECAR_BIN_ENV} at a prebuilt `pipette-coreai-sidecar` instead.",
            src.display()
        );
    }
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
