//! Locating and probing the `uv` binary itself. Provisioning a venv with it is
//! [`crate::install`]'s job.

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::Context;

/// A `uv` CLI failure or missing binary.
#[derive(Debug, thiserror::Error)]
pub enum UvError {
    #[error(
        "uv not found on PATH; install it from \
         https://docs.astral.sh/uv/getting-started/installation/"
    )]
    NotOnPath,
    #[error("uv not found at {0}")]
    Missing(PathBuf),
    #[error("failed to run `{0}`")]
    Spawn(String, #[source] std::io::Error),
    #[error("`{cmd}` failed (exit {status})")]
    Exit { cmd: String, status: String },
}

/// Resolve the `uv` binary: explicit path (file or directory containing `uv`),
/// else `PATH`.
pub fn resolve_uv(uv_path: Option<&Path>) -> Result<PathBuf, UvError> {
    match uv_path {
        Some(path) => {
            let uv = if path.is_dir() {
                path.join("uv")
            } else {
                path.to_path_buf()
            };
            if !uv.exists() {
                return Err(UvError::Missing(uv));
            }
            Ok(uv)
        }
        None => pipette_subprocess::which("uv").map_err(|_| UvError::NotOnPath),
    }
}

/// Probe `uv --version` (trimmed stdout).
pub fn verify_uv(uv: &Path) -> anyhow::Result<String> {
    let mut cmd = Command::new(uv);
    cmd.arg("--version");
    pipette_subprocess::echo_debug(&cmd);
    let output = cmd
        .output()
        .with_context(|| format!("failed to run {}", uv.display()))?;
    if !output.status.success() {
        anyhow::bail!("uv --version failed with {}", output.status);
    }
    Ok(String::from_utf8(output.stdout)
        .context("non-UTF-8 uv version")?
        .trim()
        .to_string())
}
