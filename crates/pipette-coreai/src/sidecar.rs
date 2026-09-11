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
/// `swift/` package into a **pin-and-source-keyed** cache directory and
/// copies the product there. A pin bump or a sidecar source change therefore
/// cannot keep executing a stale binary.
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

/// True when this host can obtain a sidecar without guessing: a prebuilt
/// override, a previously published cache entry, or `swift` on `PATH` so a
/// first-use build can succeed. Used to advertise `runtime:core_ai` — the OS
/// floor alone is not enough (a macOS 27 host with no toolchain and no cache
/// would be dispatched cells that can only fail).
pub fn sidecar_obtainable() -> bool {
    if let Some(path) = std::env::var_os(SIDECAR_BIN_ENV) {
        return PathBuf::from(path).is_file();
    }
    if let Ok(root) = sidecar_cache_root() {
        if root
            .join(bundled_stack_key())
            .join("pipette-coreai-sidecar")
            .is_file()
        {
            return true;
        }
    }
    swift_on_path()
}

fn swift_on_path() -> bool {
    let Some(path) = std::env::var_os("PATH") else {
        return false;
    };
    std::env::split_paths(&path).any(|dir| dir.join("swift").is_file())
}

/// Cache key for the sidecar: a short digest over the bundled Swift stack
/// *and* the sidecar's own sources. Bumping `swift-transformers` while holding
/// `coreai_models` changes `Display`, so it must change the directory — and a
/// `main.swift` tps-math fix with the pin held must too, otherwise an upgraded
/// pipette keeps executing the pre-fix binary under the same recorded identity.
///
/// Hashed with SHA-256 (not `DefaultHasher`) so the key is stable across
/// Rust releases; a toolchain bump must not silently strand the cache.
fn bundled_stack_key() -> String {
    use sha2::{Digest, Sha256};

    // Compile-time snapshot: a source edit rebuilds this crate and moves the key.
    const SIDECAR_SRC: &str = concat!(
        include_str!("../swift/Sources/pipette-coreai-sidecar/main.swift"),
        include_str!("../swift/Package.swift"),
        include_str!("../swift/Package.resolved"),
    );
    let stack = AppleCoreAiMacosPipette::bundled().to_string();
    let mut hasher = Sha256::new();
    hasher.update(stack.as_bytes());
    hasher.update(SIDECAR_SRC.as_bytes());
    let digest = hasher.finalize();
    let mut hex = String::with_capacity(16);
    for byte in digest.iter().take(8) {
        hex.push_str(&format!("{byte:02x}"));
    }
    let coreai_models = AppleCoreAiMacosPipette::bundled()
        .packages
        .coreai_models
        .repository_version
        .to_string();
    format!("{coreai_models}-{hex}")
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
    let cache = sidecar_cache_root()?.join(bundled_stack_key());
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
    verify_resolved_matches_bundled(&src)?;
    publish_sidecar_atomically(&built, bin)?;
    Ok(bin.to_path_buf())
}

/// Refuse to cache a binary whose `Package.resolved` is not the stack
/// `bundled()` records. A drifted checkout behind `CARGO_MANIFEST_DIR` (or
/// `PIPETTE_COREAI_SWIFT`) would otherwise build under the bundled identity.
fn verify_resolved_matches_bundled(src: &Path) -> anyhow::Result<()> {
    let path = src.join("Package.resolved");
    let text =
        std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
    let value: serde_json::Value =
        serde_json::from_str(&text).context("Package.resolved is not JSON")?;
    let pins = value["pins"]
        .as_array()
        .context("Package.resolved missing pins")?;
    let bundled = AppleCoreAiMacosPipette::bundled().packages;
    let expected = [
        (
            "coreai-models",
            bundled.coreai_models.repository_version.as_ref(),
            bundled.coreai_models.repository_url.to_string(),
        ),
        (
            "swift-transformers",
            bundled.swift_transformers.repository_version.as_ref(),
            bundled.swift_transformers.repository_url.to_string(),
        ),
        (
            "xgrammar",
            bundled.xgrammar.repository_version.as_ref(),
            bundled.xgrammar.repository_url.to_string(),
        ),
        (
            "swift-jinja",
            bundled.swift_jinja.repository_version.as_ref(),
            bundled.swift_jinja.repository_url.to_string(),
        ),
    ];
    for (identity, version, url) in expected {
        let pin = pins
            .iter()
            .find(|p| p["identity"] == identity)
            .with_context(|| format!("Package.resolved missing pin {identity}"))?;
        let resolved = pin["state"]["version"]
            .as_str()
            .or_else(|| pin["state"]["revision"].as_str())
            .with_context(|| {
                format!("Package.resolved pin {identity} has no version or revision")
            })?;
        if resolved != version {
            anyhow::bail!(
                "built sidecar Package.resolved {identity} is {resolved}, \
                 bundled pin is {version}"
            );
        }
        let location = pin["location"].as_str().unwrap_or("");
        let location_norm = location.trim_end_matches(".git");
        if !location_norm.contains(url.as_str()) {
            anyhow::bail!(
                "built sidecar Package.resolved {identity} location {location} \
                 does not match bundled {url}"
            );
        }
    }
    Ok(())
}

/// Stage beside the live path and rename over it. A torn `fs::copy` onto the
/// live name would leave a truncated binary that every later run execs.
fn publish_sidecar_atomically(built: &Path, dest: &Path) -> anyhow::Result<()> {
    let tmp = dest.with_file_name(format!("pipette-coreai-sidecar.{}.tmp", std::process::id()));
    if let Err(err) = std::fs::copy(built, &tmp) {
        let _ = std::fs::remove_file(&tmp);
        return Err(anyhow::Error::new(err).context(format!(
            "failed to stage sidecar from {} to {}",
            built.display(),
            tmp.display()
        )));
    }
    if let Err(err) = std::fs::rename(&tmp, dest) {
        let _ = std::fs::remove_file(&tmp);
        return Err(anyhow::Error::new(err)
            .context(format!("failed to publish sidecar into {}", dest.display())));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;

    // Both tests mutate the process-global `PIPETTE_COREAI_CACHE`. `cargo test`
    // runs them in parallel by default, so serialize env access here.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn cache_dir_is_keyed_on_the_full_stack() -> anyhow::Result<()> {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let tmp =
            std::env::temp_dir().join(format!("pipette-coreai-cache-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::env::set_var(SIDECAR_CACHE_ENV, &tmp);
        let dir = sidecar_cache_dir()?;
        std::env::remove_var(SIDECAR_CACHE_ENV);
        assert!(
            dir.ends_with(bundled_stack_key()),
            "expected full-stack-keyed cache, got {}",
            dir.display()
        );
        // Key changes when any pin in the stack changes: the hash suffix is over
        // the whole `Display`, not only `coreai_models`.
        let key = bundled_stack_key();
        let coreai_only = AppleCoreAiMacosPipette::bundled()
            .packages
            .coreai_models
            .repository_version
            .to_string();
        assert!(key.starts_with(&coreai_only));
        assert_ne!(key, coreai_only, "key must include the stack hash");
        let prefix = format!("{coreai_only}-");
        assert!(key.starts_with(&prefix));
        let suffix = &key[prefix.len()..];
        assert_eq!(suffix.len(), 16, "sha256 prefix is 8 bytes / 16 hex chars");
        assert!(
            suffix.chars().all(|c| c.is_ascii_hexdigit()),
            "hash suffix must be hex, got {suffix}"
        );
        assert_eq!(key, bundled_stack_key(), "key must be stable");
        assert!(dir.starts_with(&tmp));
        let _ = std::fs::remove_dir_all(&tmp);
        Ok(())
    }

    #[test]
    fn clear_sidecar_cache_is_a_no_op_when_missing() -> anyhow::Result<()> {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let tmp = std::env::temp_dir().join(format!(
            "pipette-coreai-cache-missing-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&tmp);
        std::env::set_var(SIDECAR_CACHE_ENV, &tmp);
        let result = clear_sidecar_cache();
        std::env::remove_var(SIDECAR_CACHE_ENV);
        assert!(!result?);
        Ok(())
    }

    #[test]
    fn bundled_package_resolved_matches_identity() -> anyhow::Result<()> {
        let src = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("swift");
        verify_resolved_matches_bundled(&src)
    }
}
