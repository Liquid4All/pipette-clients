//! Shared `uv` venv provisioning for every venv-backed runtime.
//!
//! vLLM, SGLang and desktop MLX all install the same way — a relocatable `uv`
//! venv under the store-owned `blobs/` payload dir, populated from the declared
//! runtime's [`UvRuntimeSource`]. Only the Python version, whether a resolved
//! lockfile is emitted, and the host platform differ, so those are the
//! parameters here and everything else is common.
//!
//! The layout it creates is defined in [`crate::layout`], which every reader
//! shares. Ungated, so all three CI legs type-check it; each caller keeps its
//! own platform gate, since only Linux and macOS have a runtime to install.

use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::Context;

use pipette_plan_types::UvRuntimeSource;

use crate::layout::venv_python;
use crate::uv::UvError;

const VENV_DIRNAME: &str = "venv";
const REQUIREMENTS_FILENAME: &str = "requirements.txt";
const LOCKFILE_FILENAME: &str = "requirements.lock.txt";

/// What differs between the venv-backed runtimes.
pub struct VenvInstall<'a> {
    /// Interpreter passed to `uv venv --python`.
    pub python_version: &'a str,
    /// Requirements + install flags come from here; preinstalled sources are
    /// rejected because there is nothing to fetch.
    pub source: &'a UvRuntimeSource,
    /// Emit `requirements.lock.txt` via `uv pip compile`. Best-effort: a
    /// failure warns and leaves no lockfile rather than failing the install.
    pub compile_lockfile: bool,
}

/// Create the venv and install into it. Returns the in-venv `python`.
pub fn install_venv(uv: &Path, blobs_dir: &Path, spec: VenvInstall<'_>) -> anyhow::Result<PathBuf> {
    let (requirements_text, install_flags) = requirements_from(spec.source)?;

    fs::create_dir_all(blobs_dir)
        .with_context(|| format!("failed to create {}", blobs_dir.display()))?;

    let venv = blobs_dir.join(VENV_DIRNAME);
    log::info!(
        "[install] uv venv {} (python {})",
        venv.display(),
        spec.python_version
    );
    let mut cmd = Command::new(uv);
    cmd.arg("venv")
        .arg("--quiet")
        .arg("--relocatable")
        .arg("--python")
        .arg(spec.python_version)
        .arg(&venv);
    pipette_subprocess::echo_info(&cmd);
    run_status(&mut cmd, "uv venv")?;

    let req_dst = blobs_dir.join(REQUIREMENTS_FILENAME);
    fs::write(&req_dst, requirements_text)
        .with_context(|| format!("failed to write requirements to {}", req_dst.display()))?;

    let python = venv_python(&venv);
    log::info!(
        "[install] uv pip install -r {}",
        req_dst
            .file_name()
            .and_then(OsStr::to_str)
            .unwrap_or(REQUIREMENTS_FILENAME)
    );
    let mut cmd = Command::new(uv);
    cmd.args(["pip", "install", "--python"])
        .arg(&python)
        .args(install_flags)
        .arg("-r")
        .arg(&req_dst);
    pipette_subprocess::echo_info(&cmd);
    run_status(&mut cmd, "uv pip install")?;

    if spec.compile_lockfile {
        compile_lockfile(uv, blobs_dir, &python, &req_dst, install_flags)?;
    }

    if !python.exists() {
        anyhow::bail!("uv install finished but {} is missing", python.display());
    }
    Ok(python)
}

/// Record the resolved dependency set next to the requirements that produced
/// it. Best-effort: an unresolvable set still yields a usable venv.
fn compile_lockfile(
    uv: &Path,
    blobs_dir: &Path,
    python: &Path,
    req_dst: &Path,
    install_flags: &[String],
) -> anyhow::Result<()> {
    let lock_dst = blobs_dir.join(LOCKFILE_FILENAME);
    log::info!("[install] uv pip compile -> {}", lock_dst.display());
    let mut cmd = Command::new(uv);
    cmd.args(["pip", "compile", "--python"])
        .arg(python)
        .args(install_flags)
        .arg(req_dst)
        .arg("-o")
        .arg(&lock_dst);
    pipette_subprocess::echo_info(&cmd);
    let status = cmd
        .status()
        .with_context(|| format!("failed to run `{} pip compile`", uv.display()))?;
    if !status.success() {
        log::warn!("uv pip compile exited with {status}; continuing without {LOCKFILE_FILENAME}");
        let _ = fs::remove_file(&lock_dst);
    }
    Ok(())
}

/// Requirements text + install flags for a fetchable source.
fn requirements_from(source: &UvRuntimeSource) -> anyhow::Result<(&str, &[String])> {
    match source {
        UvRuntimeSource::PipRequirementsText {
            contents,
            install_flags,
        } => Ok((contents.as_ref(), install_flags.as_deref().unwrap_or(&[]))),
        UvRuntimeSource::RelativePreinstalled { .. }
        | UvRuntimeSource::AbsolutePreinstalled { .. } => {
            anyhow::bail!(
                "cannot fetch a preinstalled runtime; pass a pip_requirements_text source"
            )
        }
    }
}

fn run_status(cmd: &mut Command, label: &str) -> anyhow::Result<()> {
    let status = cmd
        .status()
        .map_err(|source| UvError::Spawn(label.to_owned(), source))?;
    if status.success() {
        Ok(())
    } else {
        Err(UvError::Exit {
            cmd: label.to_owned(),
            status: status.to_string(),
        }
        .into())
    }
}

#[cfg(test)]
mod tests {
    use pipette_plan_types::{NonEmptyString, RelativePath};

    use super::*;

    fn pip_source(flags: Option<Vec<String>>) -> anyhow::Result<UvRuntimeSource> {
        Ok(UvRuntimeSource::PipRequirementsText {
            contents: NonEmptyString::try_new("mlx-lm==0.28.4\n".to_string())?,
            install_flags: flags,
        })
    }

    #[test]
    fn requirements_from_reads_pip_text_and_defaults_flags() -> anyhow::Result<()> {
        let source = pip_source(None)?;
        let (text, flags) = requirements_from(&source)?;
        assert_eq!(text, "mlx-lm==0.28.4\n");
        assert!(flags.is_empty());
        Ok(())
    }

    #[test]
    fn requirements_from_passes_install_flags_through() -> anyhow::Result<()> {
        let source = pip_source(Some(vec![
            "--index-strategy".into(),
            "unsafe-best-match".into(),
        ]))?;
        let (_, flags) = requirements_from(&source)?;
        assert_eq!(flags, ["--index-strategy", "unsafe-best-match"]);
        Ok(())
    }

    #[test]
    fn requirements_from_reads_catalog_text() -> anyhow::Result<()> {
        let source = UvRuntimeSource::PipRequirementsText {
            contents: NonEmptyString::try_new("vllm==0.21.0\n".to_string())?,
            install_flags: None,
        };
        let (text, _) = requirements_from(&source)?;
        assert_eq!(text, "vllm==0.21.0\n");
        Ok(())
    }

    // A preinstalled venv is already on disk, so there is nothing for the
    // fetcher to build — this must fail rather than silently make a new venv.
    #[test]
    fn requirements_from_rejects_preinstalled_sources() -> anyhow::Result<()> {
        let source = UvRuntimeSource::RelativePreinstalled {
            dir: RelativePath::try_new("venv".to_string())?,
        };
        let Err(err) = requirements_from(&source) else {
            anyhow::bail!("expected a preinstalled source to be rejected");
        };
        assert!(err.to_string().contains("preinstalled"), "got {err}");
        Ok(())
    }
}
