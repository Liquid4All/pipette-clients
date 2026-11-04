//! `uv` venv provisioning for every venv-backed runtime: a relocatable venv
//! under the store-owned `blobs/` dir, populated from the runtime's
//! [`UvRuntimeSource`].
//!
//! Ungated, so all three CI legs type-check it; a caller whose runtime installs
//! on one OS only keeps its own platform gate.

use std::fs;
use std::path::Path;
use std::process::{Command, Stdio};

use anyhow::Context;

use pipette_plan_types::UvRuntimeSource;

use crate::layout::{venv_dir, venv_python};

const REQUIREMENTS_FILENAME: &str = "requirements.txt";
const LOCKFILE_FILENAME: &str = "requirements.lock.txt";

/// What differs between the venv-backed runtimes.
pub struct VenvInstall<'a> {
    /// Interpreter passed to `uv venv --python`.
    pub python_version: &'a str,
    /// Requirements + install flags come from here; preinstalled sources are
    /// rejected because there is nothing to fetch.
    pub source: &'a UvRuntimeSource,
    /// A directory of wheels passed as `--find-links`, for a
    /// [`UvRuntimeSource::VaultWheelhouse`] the caller has already unpacked.
    /// Required by that arm and ignored by the others.
    pub wheelhouse: Option<&'a Path>,
    /// Emit `requirements.lock.txt` via `uv pip compile`. Best-effort: a
    /// failure warns and leaves no lockfile rather than failing the install.
    pub compile_lockfile: bool,
}

/// Create the venv and install into it.
pub fn install_venv(uv: &Path, blobs_dir: &Path, spec: VenvInstall<'_>) -> anyhow::Result<()> {
    let (requirements_text, install_flags) = requirements_from(spec.source)?;
    let install_flags = with_find_links(spec.source, spec.wheelhouse, install_flags)?;
    let install_flags = install_flags.as_slice();

    fs::create_dir_all(blobs_dir)
        .with_context(|| format!("failed to create {}", blobs_dir.display()))?;

    let venv = venv_dir(blobs_dir);
    tracing::info!(
        path = %venv.display(),
        python_version = %spec.python_version,
        "creating the uv venv"
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
    tracing::info!(
        path = %req_dst.display(),
        "installing the requirements with uv pip install"
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
    Ok(())
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
    tracing::info!(
        path = %lock_dst.display(),
        "compiling the lockfile with uv pip compile"
    );
    let mut cmd = Command::new(uv);
    cmd.args(["pip", "compile", "--python"])
        .arg(python)
        .args(install_flags)
        .arg(req_dst)
        .arg("-o")
        .arg(&lock_dst)
        .stdout(Stdio::null());
    pipette_subprocess::echo_info(&cmd);
    #[expect(
        clippy::disallowed_methods,
        reason = "stdout is discarded: uv prints the whole lockfile there, and the file holds it"
    )]
    let status = cmd
        .status()
        .with_context(|| format!("failed to run `{} pip compile`", uv.display()))?;
    if !status.success() {
        tracing::warn!(
            exit_status = %status,
            lockfile = %LOCKFILE_FILENAME,
            "uv pip compile failed; continuing without a lockfile"
        );
        let _ = fs::remove_file(&lock_dst);
    }
    Ok(())
}

fn requirements_from(source: &UvRuntimeSource) -> anyhow::Result<(&str, &[String])> {
    match source {
        UvRuntimeSource::PipRequirementsText {
            contents,
            install_flags,
        }
        | UvRuntimeSource::VaultWheelhouse {
            contents,
            install_flags,
            ..
        } => Ok((contents.as_ref(), install_flags.as_deref().unwrap_or(&[]))),
        UvRuntimeSource::RelativePreinstalled { .. }
        | UvRuntimeSource::AbsolutePreinstalled { .. } => {
            anyhow::bail!(
                "cannot fetch a preinstalled runtime; pass a pip_requirements_text source"
            )
        }
    }
}

/// The install flags with `--find-links <wheelhouse>` ahead of the plan's own,
/// for a wheelhouse source. The plan's flags come last so `--no-index` or an
/// extra index beside the wheelhouse is the author's call.
fn with_find_links(
    source: &UvRuntimeSource,
    wheelhouse: Option<&Path>,
    install_flags: &[String],
) -> anyhow::Result<Vec<String>> {
    let mut flags = Vec::with_capacity(install_flags.len() + 2);
    match (source, wheelhouse) {
        (UvRuntimeSource::VaultWheelhouse { .. }, Some(dir)) => {
            flags.push("--find-links".to_owned());
            flags.push(dir.display().to_string());
        }
        (UvRuntimeSource::VaultWheelhouse { .. }, None) => {
            anyhow::bail!("a vault wheelhouse source needs its wheelhouse unpacked first")
        }
        (_, _) => {}
    }
    flags.extend(install_flags.iter().cloned());
    Ok(flags)
}

#[derive(Debug, thiserror::Error)]
enum UvError {
    #[error("failed to run `{0}`")]
    Spawn(String, #[source] std::io::Error),
    #[error("`{cmd}` failed ({status})")]
    Exit { cmd: String, status: String },
}

fn run_status(cmd: &mut Command, label: &str) -> anyhow::Result<()> {
    #[expect(clippy::disallowed_methods, reason = "uv's stdout goes to stderr")]
    let status = cmd
        .stdout(std::io::stderr())
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

    fn wheelhouse_source(flags: Option<Vec<String>>) -> anyhow::Result<UvRuntimeSource> {
        Ok(UvRuntimeSource::VaultWheelhouse {
            contents: NonEmptyString::try_new("vllm==0.22.0\n".to_string())?,
            url: "/vault/resources/vault_1/content".parse()?,
            principal_id: "org_1".parse()?,
            custom_label: None,
            install_flags: flags,
        })
    }

    #[test]
    fn a_wheelhouse_source_installs_from_its_directory() -> anyhow::Result<()> {
        let source = wheelhouse_source(Some(vec!["--no-index".into()]))?;
        let (text, flags) = requirements_from(&source)?;
        assert_eq!(text, "vllm==0.22.0\n");
        let flags = with_find_links(&source, Some(Path::new("/w/wheelhouse")), flags)?;
        assert_eq!(flags, ["--find-links", "/w/wheelhouse", "--no-index"]);

        let public = pip_source(None)?;
        assert!(with_find_links(&public, Some(Path::new("/w")), &[])?.is_empty());
        Ok(())
    }

    /// The wheelhouse is the caller's to unpack; forgetting it is refused
    /// rather than resolved against the public index.
    #[test]
    fn a_wheelhouse_source_without_its_directory_is_refused() -> anyhow::Result<()> {
        let source = wheelhouse_source(None)?;
        let Err(err) = with_find_links(&source, None, &[]) else {
            anyhow::bail!("expected a refusal");
        };
        assert!(err.to_string().contains("wheelhouse"), "got {err}");
        Ok(())
    }

    // A preinstalled venv is already on disk: fail rather than silently make a new one.
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
