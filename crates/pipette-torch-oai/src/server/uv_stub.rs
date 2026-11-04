//! Non-Linux stub for the uv server, with [`super::uv`]'s signatures so callers
//! compile unchanged. Docker launch works on every host.

use std::path::Path;

use pipette_plan_types::{RuntimeFlags, TorchOaiExtraRuntimeConfiguration};

use super::{Server, ServerKind};

const NOT_LINUX: &str = "uv engine is Linux-only; use the docker engine on this host";

pub fn launch(
    _venv: &Path,
    _kind: ServerKind,
    _model: &Path,
    _server_args: &[String],
    _flags: &RuntimeFlags,
    _client_config: Option<&TorchOaiExtraRuntimeConfiguration>,
) -> anyhow::Result<Server> {
    anyhow::bail!(NOT_LINUX);
}

#[cfg(test)]
mod tests {
    use anyhow::Context;

    use super::*;

    #[test]
    fn launch_reports_the_linux_only_limitation() -> anyhow::Result<()> {
        let flags = RuntimeFlags::EndToEndLatencyUvVllmTorch {
            tensor_parallel_size: None,
            dtype: None,
            max_model_len: None,
            prefix_caching: None,
            envs: vec![],
            raw: vec![],
        };
        let err = launch(
            Path::new("/venv"),
            ServerKind::Vllm,
            Path::new("/models/model"),
            &[],
            &flags,
            None,
        )
        .err()
        .context("expected an error")?;
        assert!(err.to_string().contains("Linux-only"), "got {err:#}");
        Ok(())
    }
}
