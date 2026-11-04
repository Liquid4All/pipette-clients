//! The OpenVINO server for one cell: `pipette_openvino_server.py` under the
//! venv's Python, on loopback.
//!
//! It compiles its one pipeline before it listens, so readiness
//! (`GET /v1/models`) waits out the model's compile and that compile is never
//! request time; on a GPU a request at a new prompt shape still compiles its
//! kernels. It exits on its own when this process does, so a harness killed
//! outright leaves no server holding a model.

use std::{path::Path, process::Command, time::Duration};

use anyhow::Context;
use reqwest::Method;
use serde_json::{Map, Value};

use pipette_http::HttpClient;
use pipette_plan_types::{
    OpenvinoDevice, RuntimeFlagRef, RuntimeFlags, RuntimeOutput,
    UvOpenvinoExtraRuntimeConfiguration,
};
use pipette_subprocess::NotReadyError;

use crate::runtimes::device_property;

const SCRIPT: &str = include_str!("python/pipette_openvino_server.py");
pub(crate) const SCRIPT_FILENAME: &str = "pipette_openvino_server.py";

/// A hang detector, not a performance bound: a large model's NPU compile is
/// legitimately slow (17.7s for a 350M at the default shape, 62.8s at
/// `MAX_PROMPT_LEN` 4096), and the device is documented to wedge.
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

/// Creates the compile cache. `client_config` goes first, so it cannot
/// override the environment set here.
pub(crate) fn command(
    python: &Path,
    model_dir: &Path,
    flags: &RuntimeFlags,
    compile_cache: &Path,
    script_dir: &Path,
    port: u16,
    client_config: Option<&UvOpenvinoExtraRuntimeConfiguration>,
) -> anyhow::Result<Command> {
    let model_dir = model_dir
        .to_str()
        .context("model path is not valid UTF-8")?;
    let device = RuntimeFlagRef::from(flags.clone())
        .device
        .context("resolved flags name no device")?;
    std::fs::create_dir_all(compile_cache).with_context(|| {
        format!(
            "creating the OpenVINO compile cache at {}",
            compile_cache.display()
        )
    })?;
    let mut command = pipette_venv::python_command(python)?;
    pipette_subprocess::apply_client_config(&mut command, client_config);
    command
        .arg(script_dir.join(SCRIPT_FILENAME))
        .arg("--port")
        .arg(port.to_string())
        .arg("--parent-pid")
        .arg(std::process::id().to_string())
        .args(["--model-dir", model_dir])
        .args(["--device", device_property(&device)])
        .arg("--properties")
        .arg(Value::Object(properties(flags, &device, compile_cache)).to_string())
        .env("PYTHONUNBUFFERED", "1")
        // OpenVINO prints non-ASCII diagnostics, which a Windows console's
        // legacy code page turns into a fatal UnicodeEncodeError.
        .env("PYTHONIOENCODING", "utf-8");
    Ok(command)
}

/// `CACHE_DIR` is this host's scratch location, not something the cell
/// declared, so it is added here and never reaches the recorded flags.
///
/// Every device but the NPU also runs SDPA attention: GenAI's paged-attention
/// default prefills a repeated prompt from its prefix cache and stalls a
/// per-token streamer (`docs/openvino-measurement.md`). A custom device gets it
/// too, since it may resolve to a CPU or GPU; GenAI drops the property before
/// an NPU compile.
fn properties(
    flags: &RuntimeFlags,
    device: &OpenvinoDevice,
    compile_cache: &Path,
) -> Map<String, Value> {
    let mut props = crate::flags::pipeline_properties(flags);
    props.insert(
        "CACHE_DIR".to_owned(),
        Value::from(compile_cache.to_string_lossy().into_owned()),
    );
    if *device != OpenvinoDevice::Npu {
        props.insert("ATTENTION_BACKEND".to_owned(), Value::from("SDPA"));
    }
    props
}

pub(crate) fn url(port: u16) -> String {
    format!("http://127.0.0.1:{port}")
}

/// A failed compile exits the server, which ends the wait.
pub(crate) fn ready(
    url: &str,
) -> anyhow::Result<impl FnMut(&RuntimeOutput) -> Result<(), NotReadyError>> {
    let http = HttpClient::loopback("pipette", PROBE_TIMEOUT)?;
    let models = format!("{url}/v1/models");
    Ok(move |_: &RuntimeOutput| {
        let _: Value = http
            .json_request(Method::GET, &models, None, None)
            .with_context(|| format!("GET {models}"))
            .map_err(NotReadyError::yet)?;
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use pipette_plan_types::{BenchmarkType, ModelType, OpenvinoGenerateHint, RuntimeType};

    use super::*;

    fn npu_flags() -> anyhow::Result<RuntimeFlags> {
        let mut r = RuntimeFlagRef::new(
            BenchmarkType::PrefillThroughput,
            RuntimeType::UvOpenvino,
            ModelType::Openvino,
        );
        r.device = Some(OpenvinoDevice::Npu);
        r.generate_hint = Some(OpenvinoGenerateHint::BestPerf);
        Ok(RuntimeFlags::try_from(r)?)
    }

    /// `CACHE_DIR` reaches the pipeline through the server's arguments, never
    /// through the flags a cell records.
    #[test]
    fn the_cache_reaches_the_pipeline_but_not_the_flags() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let cache = dir.path().join("cache");
        let flags = npu_flags()?;
        // Windows runs the interpreter `pyvenv.cfg` names.
        let venv = dir.path().join("venv");
        std::fs::create_dir_all(&venv)?;
        std::fs::write(venv.join("pyvenv.cfg"), "home = /cpython\n")?;
        let command = command(
            &pipette_venv::venv_python(&venv),
            Path::new("/models/ir"),
            &flags,
            &cache,
            dir.path(),
            8000,
            None,
        )?;
        let args: Vec<_> = command
            .get_args()
            .map(|arg| arg.to_string_lossy())
            .collect();
        assert!(args.iter().any(|arg| arg.contains("CACHE_DIR")), "{args:?}");
        assert!(cache.is_dir(), "the compile cache was not created");
        let json = serde_json::to_string(&flags)?;
        assert!(!json.contains("CACHE_DIR"), "got {json}");
        Ok(())
    }

    /// A custom device may resolve to a CPU or GPU, so it is pinned too.
    #[rstest::rstest]
    #[case::cpu(OpenvinoDevice::Cpu, Some("SDPA"))]
    #[case::gpu(OpenvinoDevice::Gpu, Some("SDPA"))]
    #[case::npu(OpenvinoDevice::Npu, None)]
    #[case::custom_gpu(OpenvinoDevice::Custom("GPU.1".to_owned()), Some("SDPA"))]
    #[case::custom_auto(OpenvinoDevice::Custom("AUTO:NPU,CPU".to_owned()), Some("SDPA"))]
    fn sdpa_is_pinned_off_the_npu(
        #[case] device: OpenvinoDevice,
        #[case] expected: Option<&str>,
    ) -> anyhow::Result<()> {
        let props = properties(&npu_flags()?, &device, Path::new("/cache"));
        assert_eq!(
            props.get("ATTENTION_BACKEND").and_then(Value::as_str),
            expected,
            "got {props:?}"
        );
        Ok(())
    }

    /// Shell scripts stand in for the interpreter, so these are Unix-only.
    #[cfg(unix)]
    mod start {
        use std::time::Instant;

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
            let script = script()?;
            let port = pipette_subprocess::pick_server_port()?;
            let mut openvino_server = start_process(
                SCRIPT_FILENAME,
                command(
                    &python,
                    dir.path(),
                    &npu_flags()?,
                    &dir.path().join("cache"),
                    script.path(),
                    port,
                    None,
                )?,
                STOP_GRACE,
            )?;
            Ok(openvino_server.wait_ready(ready_timeout, ready(&url(port))?)?)
        }

        /// A server that dies while loading reports its exit and the traceback
        /// it left, not a readiness timeout.
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
    }
}
