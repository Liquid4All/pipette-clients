//! Docker runtime install: the image lives in the docker daemon, so this runs
//! `docker pull` and the store records a manifest-only entry, which keeps
//! docker runtimes in `runtimes list`.

use std::path::Path;

use pipette_plan_types::Runtime;

/// A `docker` CLI failure.
#[derive(Debug, thiserror::Error)]
pub enum DockerError {
    #[error("failed to run `docker` (is Docker installed and on your PATH?)")]
    Spawn(#[source] std::io::Error),
    /// Docker's own diagnostics already went to the inherited stderr.
    #[error("`docker pull {reference}` failed (exit {code})")]
    Pull { reference: String, code: String },
}

/// Pull a docker runtime's image into the daemon via `docker_bin`.
pub fn docker_image_pull(docker_bin: &Path, declared: &Runtime) -> anyhow::Result<()> {
    let (image_name, image_tag) = match declared {
        Runtime::DockerVllm(rt) => (rt.image_name.as_ref(), rt.image_tag.as_ref()),
        Runtime::DockerSglang(rt) => (rt.image_name.as_ref(), rt.image_tag.as_ref()),
        other @ (Runtime::LlamacppCliStockTools(_)
        | Runtime::LlamacppApkPipette(_)
        | Runtime::LlamacppIosPipette(_)
        | Runtime::MlxMacosPipette(_)
        | Runtime::MlxIosPipette(_)
        | Runtime::OnnxGenaiIosPipette(_)
        | Runtime::UvVllm(_)
        | Runtime::UvSglang(_)
        | Runtime::UvOpenvino(_)
        | Runtime::UvOnnxGenai(_)
        | Runtime::Ovms(_)
        | Runtime::TrtEdgellm(_)
        | Runtime::AppleCoreAiMacosPipette(_)
        | Runtime::AppleFoundation(_)) => anyhow::bail!(
            "docker_image_pull only installs docker runtimes, got `{}`",
            other.headless_token()
        ),
    };
    let reference = format!("{image_name}:{image_tag}");
    #[expect(
        clippy::disallowed_methods,
        reason = "docker reports the pull on stdout, sent to stderr here"
    )]
    let status = std::process::Command::new(docker_bin)
        .arg("pull")
        .arg(&reference)
        .stdout(std::io::stderr())
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
