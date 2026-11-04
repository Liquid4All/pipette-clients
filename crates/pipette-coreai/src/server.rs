//! The Swift sidecar as a [`Process`]: started on a loopback port, ready once
//! it prints its ready marker on stderr.

use std::{path::PathBuf, process::Command, time::Duration};

use anyhow::Context;

use pipette_http::HttpClient;
use pipette_plan_types::{RunRequest, RuntimeOutput};
use pipette_subprocess::{NotReadyError, Process, start_process};

use crate::models::require_coreai_model_dir;

const SIDECAR_NAME: &str = "pipette-coreai-sidecar";
/// First-load specialization runs before the ready marker and can take long.
const READY_TIMEOUT: Duration = Duration::from_secs(3600);
/// The sidecar keeps no state worth a graceful stop.
const STOP_GRACE: Duration = Duration::ZERO;
/// Client-side request timeout for Core AI timing/memory HTTP cells.
const HTTP_TIMEOUT: Duration = Duration::from_secs(3600);
const READY_MARKER: &str = "PIPETTE_COREAI_READY";
const ERROR_MARKER: &str = "PIPETTE_COREAI_ERROR";

pub struct ServerHandle {
    pub process: Process,
    pub base_url: String,
    /// Built once per server so no rep pays for client construction.
    pub http: HttpClient,
}

/// Start the Swift sidecar for the bound Core AI model, and wait for its ready
/// marker (`PIPETTE_COREAI_READY`) on stderr.
///
/// The execute modules resolve `sidecar` through
/// [`crate::sidecar::require_coreai_sidecar`] before the readiness gate, so a
/// first-use `swift build` never runs between the gate and the measurement.
pub fn start_server(req: &RunRequest, sidecar: PathBuf) -> anyhow::Result<ServerHandle> {
    let model_dir = require_coreai_model_dir(req)?;
    let http = HttpClient::loopback("pipette", HTTP_TIMEOUT)
        .context("failed to build Core AI server HTTP client")?;
    let port = pipette_subprocess::pick_server_port()?;

    let mut command = Command::new(&sidecar);
    command
        .arg("--model")
        .arg(&model_dir)
        .arg("--port")
        .arg(port.to_string())
        .arg("--seed")
        .arg("42");
    pipette_subprocess::echo_info(&command);
    let mut process = start_process(SIDECAR_NAME, command, STOP_GRACE)?;
    process.wait_ready(READY_TIMEOUT, ready)?;
    let base_url = format!("http://127.0.0.1:{port}");
    tracing::info!(pid = process.pid(), %base_url, "Core AI sidecar ready");
    Ok(ServerHandle {
        process,
        base_url,
        http,
    })
}

/// Ready once stderr carries the ready marker; the error marker fails the
/// start at once.
fn ready(output: &RuntimeOutput) -> Result<(), NotReadyError> {
    let stderr = output.runtime_stderr.as_deref().unwrap_or_default();
    if stderr.contains(ERROR_MARKER) {
        return Err(NotReadyError::failed(format!("it printed {ERROR_MARKER}")));
    }
    if stderr.contains(READY_MARKER) {
        return Ok(());
    }
    Err(NotReadyError::yet(format!("no {READY_MARKER} yet")))
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    fn sh(script: &str) -> Command {
        let mut command = Command::new("sh");
        command.args(["-c", script]);
        command
    }

    #[test]
    fn a_start_waits_for_the_ready_marker() -> anyhow::Result<()> {
        let script = "sleep 1; echo 'PIPETTE_COREAI_READY port=1 model=m specialization_cache=hit' >&2; exec sleep 60";
        let mut process = start_process(SIDECAR_NAME, sh(script), STOP_GRACE)?;
        process.wait_ready(READY_TIMEOUT, ready)?;
        Ok(())
    }

    #[rstest]
    #[case::exits_before_ready(
        "echo no bundle here >&2; exit 3",
        "exited with exit status: 3 before it was ready",
        "no bundle here"
    )]
    #[case::error_marker(
        "echo 'PIPETTE_COREAI_ERROR bind failed: in use' >&2; exec sleep 30",
        "failed before it was ready",
        "bind failed: in use"
    )]
    fn a_start_that_never_gets_ready_fails(
        #[case] script: &str,
        #[case] why: &str,
        #[case] printed: &str,
    ) -> anyhow::Result<()> {
        let mut process = start_process(SIDECAR_NAME, sh(script), STOP_GRACE)?;
        let Err(err) = process.wait_ready(READY_TIMEOUT, ready) else {
            anyhow::bail!("a sidecar that never got ready was ready");
        };
        let message = format!("{:#}", anyhow::Error::new(err));
        let expected = [format!("{SIDECAR_NAME} {why}"), printed.to_owned()];
        let missing: Vec<_> = expected
            .iter()
            .filter(|part| !message.contains(part.as_str()))
            .collect();
        anyhow::ensure!(missing.is_empty(), "{missing:?} not in: {message}");
        Ok(())
    }
}
