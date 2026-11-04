//! The ONNX GenAI server for one cell: `pipette_onnx_genai_server.py` under
//! the venv's Python, on loopback.
//!
//! It loads the model before it listens, so readiness (`GET /v1/models`)
//! waits out the load and the load is never request time. It exits on its own
//! when this process does, so a harness killed outright leaves no server
//! holding a model.

use std::{path::Path, process::Command, time::Duration};

use anyhow::Context;
use reqwest::Method;

use pipette_http::HttpClient;
use pipette_plan_types::{RuntimeOutput, UvOnnxGenaiExtraRuntimeConfiguration};
use pipette_subprocess::NotReadyError;

use crate::models::genai_name;
use crate::runtimes::ResolvedOnnxGenaiInstall;

const SCRIPT: &str = include_str!("python/pipette_onnx_genai_server.py");
pub(crate) const SCRIPT_FILENAME: &str = "pipette_onnx_genai_server.py";

/// A hang detector for a cold multi-gigabyte weight load, not a performance
/// bound.
pub(crate) const READY_TIMEOUT: Duration = Duration::from_secs(3600);
const PROBE_TIMEOUT: Duration = Duration::from_secs(10);
/// The server holds nothing worth flushing, so a stop kills it at once.
pub(crate) const STOP_GRACE: Duration = Duration::ZERO;

/// A temp dir holding the script, which must outlive the server run from it.
pub(crate) fn script() -> anyhow::Result<tempfile::TempDir> {
    let dir = tempfile::tempdir()
        .with_context(|| format!("could not create a temp dir for {SCRIPT_FILENAME}"))?;
    let script = dir.path().join(SCRIPT_FILENAME);
    std::fs::write(&script, SCRIPT)
        .with_context(|| format!("could not write {}", script.display()))?;
    Ok(dir)
}

/// `client_config` goes first, so it cannot override the environment set here.
pub(crate) fn command(
    install: &ResolvedOnnxGenaiInstall,
    model_dir: &Path,
    script_dir: &Path,
    port: u16,
    client_config: Option<&UvOnnxGenaiExtraRuntimeConfiguration>,
) -> anyhow::Result<Command> {
    let model_dir = model_dir
        .to_str()
        .context("model path is not valid UTF-8")?;
    let mut command = pipette_venv::python_command(&install.python)?;
    pipette_subprocess::apply_client_config(&mut command, client_config);
    command
        .arg(script_dir.join(SCRIPT_FILENAME))
        .arg("--port")
        .arg(port.to_string())
        .arg("--parent-pid")
        .arg(std::process::id().to_string())
        .args(["--model-dir", model_dir])
        .args([
            "--execution-provider",
            genai_name(&install.execution_provider),
        ])
        .env("PYTHONUNBUFFERED", "1")
        // ONNX GenAI prints non-ASCII diagnostics, which a Windows console's
        // legacy code page turns into a fatal UnicodeEncodeError.
        .env("PYTHONIOENCODING", "utf-8");
    Ok(command)
}

pub(crate) fn url(port: u16) -> String {
    format!("http://127.0.0.1:{port}")
}

/// A failed load exits the server, which ends the wait.
pub(crate) fn ready(
    url: &str,
) -> anyhow::Result<impl FnMut(&RuntimeOutput) -> Result<(), NotReadyError>> {
    let http = HttpClient::loopback("pipette", PROBE_TIMEOUT)?;
    let models = format!("{url}/v1/models");
    Ok(move |_: &RuntimeOutput| {
        let _: serde_json::Value = http
            .json_request(Method::GET, &models, None, None)
            .with_context(|| format!("GET {models}"))
            .map_err(NotReadyError::yet)?;
        Ok(())
    })
}

/// Shell scripts stand in for the interpreter, and on Windows `python_command`
/// reads the venv's `pyvenv.cfg`, so these are Unix-only.
#[cfg(all(test, unix))]
mod tests {
    use std::time::Instant;

    use rstest::rstest;

    use pipette_plan_types::OnnxExecutionProvider;
    use pipette_subprocess::start_process;

    use super::*;

    /// A server whose interpreter is a shell script running `body`.
    fn start(body: &str, ready_timeout: Duration) -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let python = dir.path().join("python");
        std::fs::write(&python, format!("#!/bin/sh\n{body}\n"))?;
        let mut permissions = std::fs::metadata(&python)?.permissions();
        std::os::unix::fs::PermissionsExt::set_mode(&mut permissions, 0o755);
        std::fs::set_permissions(&python, permissions)?;
        let install = ResolvedOnnxGenaiInstall {
            python,
            execution_provider: OnnxExecutionProvider::Cpu,
        };
        let script = script()?;
        let port = pipette_subprocess::pick_server_port()?;
        let mut onnx_server = start_process(
            SCRIPT_FILENAME,
            command(&install, dir.path(), script.path(), port, None)?,
            STOP_GRACE,
        )?;
        Ok(onnx_server.wait_ready(ready_timeout, ready(&url(port))?)?)
    }

    fn provider_arg(provider: OnnxExecutionProvider) -> anyhow::Result<std::ffi::OsString> {
        let install = ResolvedOnnxGenaiInstall {
            python: "/venv/bin/python".into(),
            execution_provider: provider,
        };
        let command = command(&install, Path::new("/model"), Path::new("/script"), 1, None)?;
        let value = command
            .get_args()
            .skip_while(|arg| *arg != "--execution-provider")
            .nth(1)
            .context("no --execution-provider value")?;
        Ok(value.to_owned())
    }

    /// A server that dies while loading reports its exit and the traceback it
    /// left, not a readiness timeout.
    #[test]
    fn a_server_that_exits_while_loading_reports_its_stderr() -> anyhow::Result<()> {
        let Err(err) = start("echo 'LOAD_FAILED: no model' >&2\nexit 3", READY_TIMEOUT) else {
            anyhow::bail!("expected the start to fail");
        };
        let message = format!("{err:#}");
        assert!(message.contains("exited with"), "{message}");
        assert!(message.contains("LOAD_FAILED: no model"), "{message}");
        Ok(())
    }

    #[test]
    fn a_server_that_never_answers_times_out() -> anyhow::Result<()> {
        let started = Instant::now();
        let Err(err) = start("exec sleep 30", Duration::from_millis(300)) else {
            anyhow::bail!("expected the start to time out");
        };
        assert!(format!("{err:#}").contains("not ready within"), "{err:#}");
        assert!(started.elapsed() < Duration::from_secs(10));
        Ok(())
    }

    #[rstest]
    #[case::cpu(OnnxExecutionProvider::Cpu, "cpu")]
    #[case::cuda(OnnxExecutionProvider::Cuda, "cuda")]
    #[case::directml(OnnxExecutionProvider::DirectMl, "dml")]
    #[case::custom(OnnxExecutionProvider::Custom("QNN".to_owned()), "QNN")]
    fn the_server_gets_genai_s_name_for_each_provider(
        #[case] provider: OnnxExecutionProvider,
        #[case] expected: &str,
    ) -> anyhow::Result<()> {
        assert_eq!(provider_arg(provider)?, expected);
        Ok(())
    }
}
