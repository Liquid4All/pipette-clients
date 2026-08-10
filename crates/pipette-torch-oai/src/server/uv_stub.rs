//! Non-Linux stub for the uv server. Bails at every callable entrypoint with a
//! clear message; the type surface mirrors [`super::uv`] so callers compile
//! unchanged. Docker launch works on every host, so only uv is stubbed.

use std::path::{Path, PathBuf};

use pipette_subprocess::cleanup::TeardownToken;

use super::{LaunchSpec, ServerState};

pub const ENV_WORKSPACE_LABEL: &str = "PIPETTE_WORKSPACE_LABEL";
pub const ENV_RUNTIME_REF: &str = "PIPETTE_RUNTIME_REF";
pub const ENV_PARENT_PID: &str = "PIPETTE_PARENT_PID";

const NOT_LINUX: &str = "uv engine is Linux-only; use the docker engine on this host";

pub use super::pick_free_port;

#[derive(Debug, Clone)]
pub struct UvLaunchEnv {
    pub workspace_label: String,
    pub hf_home: Option<PathBuf>,
    pub model: String,
    pub runtime_ref: String,
}

pub fn launch(
    _venv: &Path,
    _spec: &LaunchSpec,
    _env: &UvLaunchEnv,
) -> anyhow::Result<(ServerState, TeardownToken)> {
    anyhow::bail!(NOT_LINUX);
}

pub fn stop(_state: &mut ServerState, _grace_secs: u32) -> anyhow::Result<()> {
    anyhow::bail!(NOT_LINUX);
}

/// Always `None`: [`launch`] bails here, so no uv server exists to have
/// exited. Keeps the readiness liveness check's uv arm compiling off Linux.
pub(super) fn exit_status(_pid: u32) -> Option<String> {
    None
}

/// No-op off Linux: there's no `/proc` to walk and uv can't have spawned
/// anything here, so there are no orphans. `Ok(())` keeps `benchmarks run`
/// — which calls this unconditionally — working for docker-only workflows.
pub fn reap_workspace_orphans(_workspace_label: &str) -> anyhow::Result<()> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use anyhow::Context;

    use super::*;

    // `benchmarks run` calls the reap unconditionally, so off Linux it has to
    // succeed rather than report "not supported".
    #[test]
    fn reap_workspace_orphans_is_a_no_op() -> anyhow::Result<()> {
        reap_workspace_orphans("/ws/.pipette")
    }

    #[test]
    fn launch_and_stop_report_the_linux_only_limitation() -> anyhow::Result<()> {
        let mut state = ServerState::Uv {
            executable: std::path::PathBuf::from("python"),
            kind: super::super::ServerKind::Vllm,
            pid: 1,
            pgid: 1,
            port: 8000,
            log_tail: None,
        };
        let err = stop(&mut state, 1).err().context("expected an error")?;
        assert!(err.to_string().contains("Linux-only"), "got {err:#}");
        Ok(())
    }
}
