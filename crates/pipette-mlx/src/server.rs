//! `mlx_lm.server` and `mlx_vlm.server`: their command lines and the readiness
//! check [`Process::wait_ready`](pipette_subprocess::Process::wait_ready) runs.

use std::{num::NonZeroUsize, path::Path, process::Command, time::Duration};

use anyhow::Context;
use reqwest::Method;

use pipette_http::HttpClient;
use pipette_plan_types::{MlxMacosPipetteExtraRuntimeConfiguration, RuntimeOutput};
use pipette_subprocess::NotReadyError;

pub(crate) const TEXT_SERVER_MODULE: &str = "mlx_lm.server";
pub(crate) const VISION_SERVER_MODULE: &str = "mlx_vlm.server";
/// Whether each server runs the model repo's own code. The tokenizer helper
/// loads the tokenizer the same way, so its counts agree with the server's.
pub(crate) const TEXT_TRUSTS_REMOTE_CODE: bool = true;
pub(crate) const VISION_TRUSTS_REMOTE_CODE: bool = false;
pub(crate) const READY_TIMEOUT: Duration = Duration::from_secs(3600);
/// The server holds nothing worth flushing, so a stop kills it at once.
pub(crate) const STOP_GRACE: Duration = Duration::ZERO;

const PYTHON_UNBUFFERED_ENV: &str = "PYTHONUNBUFFERED";
/// `mlx_vlm.server`'s prefix cache switch. A launch setting only: the server
/// accepts unknown request fields and ignores them, so a `cache_prompt` in the
/// body would be silently dropped.
const VISION_PREFIX_CACHE_ENV: &str = "APC_ENABLED";
/// One `/health` poll; a socket that accepts and then stalls costs this much.
const HEALTH_TIMEOUT: Duration = Duration::from_secs(10);
/// How Python reports an uncaught exception. `mlx_lm.server` keeps running
/// after a failed bind or model load, with only this on stderr to say so.
const TRACEBACK: &str = "Traceback (most recent call last):";

pub(crate) fn text_command(
    venv_python: &Path,
    model_dir: &Path,
    port: u16,
    client_config: Option<&MlxMacosPipetteExtraRuntimeConfiguration>,
) -> anyhow::Result<Command> {
    let model = model_dir
        .to_str()
        .context("model path is not valid UTF-8")?;
    // `--prompt-cache-size 0` turns off the KV prompt cache: reuse across reps
    // would under-measure prefill.
    let port_arg = port.to_string();
    let mut command = python(venv_python, client_config);
    command.args([
        "-m",
        TEXT_SERVER_MODULE,
        "--model",
        model,
        "--host",
        "127.0.0.1",
        "--port",
        port_arg.as_str(),
    ]);
    command.args(TEXT_TRUSTS_REMOTE_CODE.then_some("--trust-remote-code"));
    command.args(["--prompt-cache-size", "0"]);
    Ok(command)
}

/// Separate from [`text_command`] because `mlx_lm` cannot load a vision model.
/// `draft` is `None` for the baseline arm; each arm is its own process, since a
/// drafter cannot be unloaded from the one holding it.
pub(crate) fn vision_command(
    venv_python: &Path,
    model_dir: &Path,
    port: u16,
    draft: Option<(&Path, u32)>,
    batch_size: NonZeroUsize,
    client_config: Option<&MlxMacosPipetteExtraRuntimeConfiguration>,
) -> anyhow::Result<Command> {
    let mut command = python(venv_python, client_config);
    // Prefix caching off, explicitly rather than by default, so every turn
    // prefills the conversation it carries.
    command
        .args(vision_server_args(model_dir, port, draft, batch_size)?)
        .env(VISION_PREFIX_CACHE_ENV, "0");
    Ok(command)
}

/// The `mlx_vlm.server` argv for one arm.
///
/// The drafter and its block size are launch settings on this server, not
/// request fields, which is why an arm is a process rather than a flag: the
/// baseline is this argv minus these two.
///
/// The server batches continuously and unbounded by default; `--max-num-seqs`
/// caps the batch at what the traversal keeps in flight. It is left out at the
/// reference batch size of 1, where the client sends one request at a time:
/// the command is recorded with the row, and a batch-1 run must not read as a
/// different configuration from the batch-1 rows recorded before the flag.
fn vision_server_args(
    model_dir: &Path,
    port: u16,
    draft: Option<(&Path, u32)>,
    batch_size: NonZeroUsize,
) -> anyhow::Result<Vec<String>> {
    let model = model_dir
        .to_str()
        .context("model path is not valid UTF-8")?;
    let mut args: Vec<String> = ["-m", VISION_SERVER_MODULE, "--model", model, "--port"]
        .into_iter()
        .map(str::to_owned)
        .collect();
    args.push(port.to_string());
    args.extend(VISION_TRUSTS_REMOTE_CODE.then(|| "--trust-remote-code".to_owned()));
    if let Some((draft_dir, draft_length)) = draft {
        let draft_path = draft_dir
            .to_str()
            .context("draft model path is not valid UTF-8")?;
        args.extend([
            "--draft-model".to_owned(),
            draft_path.to_owned(),
            "--draft-block-size".to_owned(),
            draft_length.to_string(),
        ]);
    }
    if batch_size.get() > 1 {
        args.extend(["--max-num-seqs".to_owned(), batch_size.to_string()]);
    }
    Ok(args)
}

/// The venv interpreter with `client_config` applied first, so a host file
/// cannot clobber what a measurement sets, such as its probe channel.
pub(crate) fn python(
    venv_python: &Path,
    client_config: Option<&MlxMacosPipetteExtraRuntimeConfiguration>,
) -> Command {
    let mut command = Command::new(venv_python);
    pipette_subprocess::apply_client_config(&mut command, client_config);
    command.env(PYTHON_UNBUFFERED_ENV, "1");
    command
}

pub(crate) fn url(port: u16) -> String {
    format!("http://127.0.0.1:{port}")
}

/// Ready once `GET /health` answers 200; a Python traceback fails the start at
/// once. `mlx_lm.server` answers while its model still loads, so the warm-up
/// request absorbs the load; `mlx_vlm.server` answers only once loaded.
pub(crate) fn ready(
    url: &str,
) -> anyhow::Result<impl FnMut(&RuntimeOutput) -> Result<(), NotReadyError>> {
    let http = HttpClient::loopback("pipette", HEALTH_TIMEOUT)?;
    let health = format!("{url}/health");
    Ok(move |output: &RuntimeOutput| {
        if let Some(stderr) = output.runtime_stderr.as_deref()
            && stderr.contains(TRACEBACK)
        {
            return Err(NotReadyError::failed("it printed a Python traceback"));
        }
        let _: serde_json::Value = http
            .json_request(Method::GET, &health, None, None)
            .with_context(|| format!("GET {health}"))
            .map_err(NotReadyError::yet)?;
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use httpmock::MockServer;
    use rstest::rstest;
    use serde_json::json;

    use pipette_subprocess::start_process;

    use super::*;

    fn batch(n: usize) -> anyhow::Result<NonZeroUsize> {
        NonZeroUsize::new(n).ok_or_else(|| anyhow::anyhow!("batch size 0"))
    }

    #[rstest]
    #[case::reference(1, None)]
    #[case::batched(8, Some("8"))]
    fn the_servers_batch_follows_the_batch_size(
        #[case] batch_size: usize,
        #[case] max_num_seqs: Option<&str>,
    ) -> anyhow::Result<()> {
        let batch = batch(batch_size)?;
        let baseline = vision_server_args(Path::new("/m/target"), 8080, None, batch)?;
        let speculative = vision_server_args(
            Path::new("/m/target"),
            8080,
            Some((Path::new("/m/draft"), 8)),
            batch,
        )?;
        let reference_baseline = [
            "-m",
            "mlx_vlm.server",
            "--model",
            "/m/target",
            "--port",
            "8080",
        ];
        let reference_speculative = [
            "-m",
            "mlx_vlm.server",
            "--model",
            "/m/target",
            "--port",
            "8080",
            "--draft-model",
            "/m/draft",
            "--draft-block-size",
            "8",
        ];
        let cap = |arm: &[&str]| -> Vec<String> {
            arm.iter()
                .map(|s| (*s).to_owned())
                .chain(
                    max_num_seqs
                        .iter()
                        .flat_map(|n| ["--max-num-seqs".to_owned(), (*n).to_owned()]),
                )
                .collect()
        };
        assert_eq!(baseline, cap(&reference_baseline));
        assert_eq!(speculative, cap(&reference_speculative));
        Ok(())
    }

    /// `sh` rather than a Python fake, which hung without a diagnostic on one
    /// CI image.
    fn sh(script: &str) -> Command {
        let mut command = Command::new("sh");
        command.args(["-c", script]);
        command
    }

    #[test]
    fn a_start_waits_for_health() -> anyhow::Result<()> {
        let stub = MockServer::start();
        let health = stub.mock(|when, then| {
            when.method("GET").path("/health");
            then.status(200).json_body(json!({"status": "ok"}));
        });
        let mut mlx_server = start_process("sh", sh("exec sleep 60"), STOP_GRACE)?;
        mlx_server.wait_ready(READY_TIMEOUT, ready(&stub.base_url())?)?;
        health.assert();
        Ok(())
    }

    #[rstest]
    #[case::exits_before_health(
        "echo no model here >&2; sleep 1; exit 3",
        "exited with exit status: 3 before it was ready",
        "no model here"
    )]
    #[case::traceback(
        "echo 'Traceback (most recent call last):' >&2; \
         echo 'OSError: [Errno 48] Address already in use' >&2; exec sleep 30",
        "failed before it was ready",
        "Address already in use"
    )]
    fn a_start_that_never_gets_ready_fails(
        #[case] script: &str,
        #[case] why: &str,
        #[case] printed: &str,
    ) -> anyhow::Result<()> {
        let url = url(pipette_subprocess::pick_server_port()?);
        let mut mlx_server = start_process(TEXT_SERVER_MODULE, sh(script), STOP_GRACE)?;
        let Err(err) = mlx_server.wait_ready(READY_TIMEOUT, ready(&url)?) else {
            anyhow::bail!("a server that never got ready was ready");
        };
        let message = format!("{:#}", anyhow::Error::new(err));
        let expected = [format!("mlx_lm.server {why}"), printed.to_owned()];
        let missing: Vec<_> = expected
            .iter()
            .filter(|part| !message.contains(part.as_str()))
            .collect();
        anyhow::ensure!(missing.is_empty(), "{missing:?} not in: {message}");
        Ok(())
    }
}
