//! Docker-daemon runtime management. A docker runtime's image lives in the
//! docker daemon, not on the filesystem — so [`docker_image_pull`] shells out to
//! `docker pull` rather than downloading an archive. It still records a
//! `runtimes/<key>/manifest.toml` entry through the shared store, so
//! `runtimes list` sees docker runtimes alongside the filesystem ones. The
//! entry is manifest-only: no binaries land on disk.

use std::path::Path;

use pipette_plan_types::Runtime;

/// A `docker` CLI failure.
#[derive(Debug, thiserror::Error)]
pub enum DockerError {
    /// The `docker` binary couldn't be launched (not installed / not on PATH).
    #[error("failed to run `docker` (is Docker installed and on your PATH?)")]
    Spawn(#[source] std::io::Error),
    /// `docker pull` exited non-zero. Docker's own diagnostics went to the
    /// inherited stderr, so the reference + exit status is enough here.
    #[error("`docker pull {reference}` failed (exit {code})")]
    Pull { reference: String, code: String },
}

/// Install a docker runtime by pulling its image into the daemon via `docker_bin`.
///
/// Writes nothing under the store staging dir (the image isn't a filesystem
/// artifact); the store records a manifest-only entry.
pub fn docker_image_pull(docker_bin: &Path, declared: &Runtime) -> anyhow::Result<()> {
    let (image_name, image_tag) = match declared {
        Runtime::DockerVllm(rt) => (rt.image_name.as_ref(), rt.image_tag.as_ref()),
        Runtime::DockerSglang(rt) => (rt.image_name.as_ref(), rt.image_tag.as_ref()),
        other => anyhow::bail!(
            "docker_image_pull only installs docker runtimes, got `{}`",
            other.headless_token()
        ),
    };
    let reference = format!("{image_name}:{image_tag}");
    let status = std::process::Command::new(docker_bin)
        .arg("pull")
        .arg(&reference)
        .status()
        .map_err(DockerError::Spawn)?;
    if !status.success() {
        return Err(DockerError::Pull {
            reference,
            code: status
                .code()
                .map_or_else(|| "signal".to_owned(), |c| c.to_string()),
        }
        .into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn errors_read_clearly() {
        let err = DockerError::Pull {
            reference: "vllm/vllm-openai:v0.10.0".to_owned(),
            code: "1".to_owned(),
        };
        assert_eq!(
            err.to_string(),
            "`docker pull vllm/vllm-openai:v0.10.0` failed (exit 1)"
        );
    }
}
