//! In-venv torch GPU probe.
//!
//! A `uv pip install` of a GPU torch wheel can succeed while the resulting
//! torch cannot reach the GPU — a wheel built for the wrong driver, or a driver
//! the host never loaded. Nothing else in the install notices, so the venv is
//! recorded as usable and the failure surfaces mid-benchmark instead.
//!
//! Lives here, next to the code that creates the venv, so the installer can run
//! it as the last step of the install without reaching into a backend crate.

use std::path::Path;
use std::process::Command;

use anyhow::Context;

/// Assert `python`'s torch can reach the GPU, naming `backend_label` ("CUDA" /
/// "ROCm") in the failure. PyTorch's ROCm build answers
/// `torch.cuda.is_available()` through HIP, so one probe covers both.
///
/// Callers that installed a CPU-only torch skip this: there is no GPU backend
/// to assert.
pub fn assert_torch_available(python: &Path, backend_label: &str) -> anyhow::Result<()> {
    let mut cmd = Command::new(python);
    // Prints version + backend so the install log reads
    // "torch.cuda.is_available() == True (cuda 12.1)" or "(hip 6.0)" —
    // operators can tell ROCm from CUDA at a glance.
    cmd.arg("-c").arg(
        "import torch\n\
         assert torch.cuda.is_available(), 'torch.cuda.is_available() is False'\n\
         hip = getattr(torch.version, 'hip', None)\n\
         cuda = getattr(torch.version, 'cuda', None)\n\
         backend = ('hip ' + hip) if hip else (('cuda ' + cuda) if cuda else 'unknown')\n\
         print('torch.cuda.is_available() == True (' + backend + ')')\n",
    );
    pipette_subprocess::echo_debug(&cmd);
    let output = cmd
        .output()
        .with_context(|| format!("failed to spawn {}", python.display()))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        anyhow::bail!(
            "torch {backend_label} probe failed inside the venv:\n\n{stderr}\n\n\
             The installed torch can't see a GPU on this host. Common causes:\n  \
             - wrong `+<flavor>` wheel for the host driver (e.g. `+cu124` on a `+cu121`-only host)\n  \
             - driver / userspace mismatch (re-run `nvidia-smi`/`rocm-smi` on the host)\n\n\
             Fix the host driver stack, then install again."
        );
    }
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if !stdout.is_empty() {
        log::info!("[install] {stdout}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // The install hands over an interpreter it has already checked exists, so a
    // path that doesn't must still surface as an error rather than a success.
    #[test]
    fn assert_torch_available_reports_an_unspawnable_python() -> anyhow::Result<()> {
        let bogus = Path::new("/this/path/does/not/exist/python");
        let err = assert_torch_available(bogus, "CUDA")
            .err()
            .context("expected a spawn failure")?;
        let msg = format!("{err:#}");
        assert!(msg.contains("failed to spawn"), "got {msg}");
        Ok(())
    }
}
