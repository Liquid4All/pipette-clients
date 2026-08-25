//! Bound Apple Core AI runtime projection for a prepared [`RunRequest`].
//!
//! The Core AI engine ships with macOS; the only build artifact pipette owns is
//! the Swift sidecar that drives it. This module resolves the sidecar binary:
//! built (once) from the crate's bundled `swift/` package into the bound
//! runtime directory, then returned as the executable to spawn.

use std::{
    path::{Path, PathBuf},
    process::Command,
};

use anyhow::Context;

use pipette_plan_types::run::RunRequest;

/// The relative path of the bundled Swift package (the sidecar source).
const SWIFT_SOURCE_DIR: &str = "swift";
/// Relative path (from the Swift package dir) of the built sidecar binary.
const SWIFT_BIN_REL: &str = ".build/out/Products/Release/pipette-coreai-sidecar";
/// Env override for the sidecar binary path (bypasses building).
const SIDECAR_BIN_ENV: &str = "PIPETTE_COREAI_SIDECAR";

/// Resolve the Swift sidecar binary for a bound `CoreAiMacosPipette` runtime.
///
/// Honors `PIPETTE_COREAI_SIDECAR` (a prebuilt path — used by CI and local
/// dev to skip the Swift build), otherwise builds the crate's bundled
/// `swift/` package into the bound runtime directory.
pub fn require_coreai_sidecar(req: &RunRequest) -> anyhow::Result<PathBuf> {
    // The runtime carries no source coordinate of ours (the engine ships with
    // the OS), so there is nothing to install in the venv store. `pipette-cli`
    // therefore does not bind an install dir for this runtime; we build the
    // sidecar into a cache under the workspace instead. Resolve the workspace
    // root from the model store like the other macOS runtimes do via their
    // bound path — here we use the model dir's parent as the cache anchor.
    if let Ok(bin) = std::env::var(SIDECAR_BIN_ENV) {
        let path = PathBuf::from(bin);
        if path.is_file() {
            return Ok(path);
        }
        anyhow::bail!(
            "PIPETTE_COREAI_SIDECAR={} is not a file",
            path.display()
        );
    }
    let cache_dir = sidecar_cache_dir(req)?;
    let bin = cache_dir.join("pipette-coreai-sidecar");
    if bin.is_file() {
        return Ok(bin);
    }
    build_sidecar(&cache_dir, &bin)
}

/// A stable cache dir for the built sidecar, under the user cache directory
/// (`~/Library/Caches/pipette-coreai/` on macOS). Building into a fixed cache
/// means the sidecar is compiled once and reused across runs and workspaces,
/// keyed by the bundled source (see [`build_sidecar`]).
fn sidecar_cache_dir(req: &RunRequest) -> anyhow::Result<PathBuf> {
    let _ = req;
    let cache = dirs_cache_dir()
        .context("cannot resolve a user cache directory")?
        .join("pipette-coreai");
    std::fs::create_dir_all(&cache)
        .with_context(|| format!("failed to create {}", cache.display()))?;
    Ok(cache)
}

fn dirs_cache_dir() -> Option<PathBuf> {
    // macOS: ~/Library/Caches (via the HOME env); no external `dirs` crate.
    std::env::var_os("HOME").map(|home| PathBuf::from(home).join("Library/Caches"))
}

/// Build the sidecar from the crate's bundled `swift/` package into `bin`.
fn build_sidecar(cache_dir: &Path, bin: &Path) -> anyhow::Result<PathBuf> {
    let src = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(SWIFT_SOURCE_DIR);
    if !src.join("Package.swift").is_file() {
        anyhow::bail!(
            "bundled Swift package not found at {} (pipette-coreai built without swift/)",
            src.display()
        );
    }
    log::info!(
        "building pipette-coreai sidecar from {} (cache {})",
        src.display(),
        cache_dir.display()
    );
    let mut cmd = Command::new("swift");
    cmd.current_dir(&src)
        .arg("build")
        .arg("-c")
        .arg("release");
    pipette_subprocess::echo_info(&cmd);
    let status = cmd
        .status()
        .with_context(|| "failed to run `swift build` for the pipette-coreai sidecar")?;
    if !status.success() {
        anyhow::bail!("pipette-coreai sidecar `swift build` exited with {status}");
    }
    let built = src.join(SWIFT_BIN_REL);
    if !built.is_file() {
        anyhow::bail!(
            "swift build produced no binary at {}",
            built.display()
        );
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
    use pipette_plan_types::RuntimeType;

    #[test]
    fn runtime_type_is_macos_desktop() {
        assert_eq!(
            RuntimeType::CoreAiMacosPipette.to_string(),
            "core_ai_macos_pipette"
        );
    }
}
