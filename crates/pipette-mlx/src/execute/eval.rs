use std::{
    collections::{HashMap, HashSet},
    io::{BufRead, BufReader},
    sync::mpsc,
    thread,
    time::Duration,
};

use anyhow::Context;
use reqwest::blocking::Client;
use serde_json::{json, Value};

use pipette_doomloop::format_trigger_log;
use pipette_ops::eval_completions::EvalCompletionsStore;
use pipette_plan_types::benchmark::Temperature;
use pipette_plan_types::result::{
    BenchmarkEvalCompletion, BenchmarkEvalCompletionStopReason, BenchmarkResultData,
};
use pipette_plan_types::run::RunRequest;
use pipette_plan_types::run::RunResponse;

use super::server;
use crate::models::require_mlx_model_dir;
use crate::runtimes::require_mlx_python;

const EVAL_ENDPOINT: &str = "/eval";
const EVAL_ABORT_ENDPOINT: &str = "/eval/abort";
const STREAM_CONNECT_TIMEOUT: Duration = Duration::from_secs(60);
/// Default SSE idle wait between stream events when the plan omits `http_timeout`.
const DEFAULT_STREAM_IDLE_TIMEOUT: Duration = Duration::from_secs(1800);

fn stream_idle_timeout(req: &RunRequest) -> Duration {
    idle_timeout_from_plan_secs(req.benchmark_flags.as_ref().and_then(|f| f.http_timeout()))
}

fn idle_timeout_from_plan_secs(secs: Option<u64>) -> Duration {
    secs.map(|s| Duration::from_secs(s.max(1)))
        .unwrap_or(DEFAULT_STREAM_IDLE_TIMEOUT)
}

pub(super) fn run(
    req: &RunRequest,
    eval_completions: &EvalCompletionsStore,
) -> anyhow::Result<RunResponse> {
    let benchmark = req.benchmark.as_eval().map_err(anyhow::Error::from)?;
    let samples = benchmark
        .samples
        .as_deref()
        .context("eval benchmark missing samples")?;
    let max_tokens = benchmark.parameter_max_tokens;
    let venv_python = require_mlx_python(req)?;
    let model_dir = require_mlx_model_dir(req)?;

    // Sampling temperature is a client-side policy keyed on the eval id
    // (the server does not send one): IFBench/IFStruct sample at 0.6,
    // everything else stays greedy.
    let temperature = req.benchmark.eval_temperature()?;

    let doomloop = pipette_doomloop::plan::pipeline_from_overrides(
        req.benchmark_flags.as_ref().and_then(|bf| bf.doomloop()),
    )
    .map_err(|e| anyhow::anyhow!("invalid doom-loop configuration: {e}"))?;
    let enable_thinking = req.model_flags.as_ref().and_then(|f| f.enable_thinking());

    // Resume keyed by portable RunRequest identity (CLI passes the store).
    let mut checkpoint = eval_completions.open(req)?;
    let done_ids_vec: Vec<String> = checkpoint.done_ids().map(str::to_string).collect();

    let request = build_eval_request_json(
        samples,
        max_tokens,
        temperature,
        &done_ids_vec,
        enable_thinking,
    );

    // Stream chunks from Python so we can run doom-loop detection on the
    // live generation. When a detector fires we log a warning and signal
    // the server to abandon the current sample early; nothing is persisted
    // about the trigger beyond the log line.
    let pipeline = &doomloop;
    let mut buffers: HashMap<String, String> = HashMap::new();
    let mut flagged: HashSet<String> = HashSet::new();
    let total_samples = samples.len();
    let done_set: HashSet<&str> = done_ids_vec.iter().map(String::as_str).collect();
    let sample_positions: HashMap<&str, usize> = samples
        .iter()
        .enumerate()
        .filter_map(|(i, s)| s.get("id").and_then(Value::as_str).map(|id| (id, i + 1)))
        .collect();

    // Emit "skipped, already checkpointed" lines for samples the server
    // will silently pass over. Matches llamacpp's eval log format.
    samples
        .iter()
        .filter_map(|sample| sample.get("id").and_then(Value::as_str))
        .filter(|sample_id| done_set.contains(*sample_id))
        .for_each(|sample_id| {
            let pos = sample_positions.get(sample_id).copied().unwrap_or(0);
            log::info!(
                "eval sample {}/{}: id={} (skipped, already checkpointed)",
                pos,
                total_samples,
                sample_id
            );
        });

    let server = server::start_server(&venv_python, &model_dir, None)?;
    let client = streaming_http_client()?;
    let idle_timeout = stream_idle_timeout(req);
    if idle_timeout != DEFAULT_STREAM_IDLE_TIMEOUT {
        log::info!("eval SSE idle timeout from plan http_timeout: {idle_timeout:?}");
    }
    let mut saw_eval_done = false;
    stream_eval_events(&client, &server.base_url, &request, idle_timeout, |event| {
        match event.kind() {
            Some("eval_sample_start") => {
                let sample_id = event
                    .0
                    .get("sample_id")
                    .and_then(Value::as_str)
                    .context("eval_sample_start missing sample_id")?;
                let prompt = event.0.get("prompt").and_then(Value::as_str).unwrap_or("");
                let pos = sample_positions.get(sample_id).copied().unwrap_or(0);
                log::info!(
                    "eval sample {}/{}: id={} prompt={}",
                    pos,
                    total_samples,
                    sample_id,
                    prompt
                );
                Ok(EvalStreamAction::Continue)
            }
            Some("eval_sample_chunk") => {
                let sample_id = event
                    .0
                    .get("sample_id")
                    .and_then(Value::as_str)
                    .context("eval_sample_chunk missing sample_id")?;
                let delta = event
                    .0
                    .get("delta")
                    .and_then(Value::as_str)
                    .context("eval_sample_chunk missing delta")?;
                let content = buffers.entry(sample_id.to_string()).or_default();
                content.push_str(delta);

                // Don't re-fire the pipeline once we've already flagged the sample.
                if !flagged.contains(sample_id) {
                    if let Some(name) = pipeline.check(content) {
                        log::warn!("{}", format_trigger_log(name, content.len()));
                        flagged.insert(sample_id.to_string());
                        return Ok(EvalStreamAction::AbortSample(sample_id.to_string()));
                    }
                }
                Ok(EvalStreamAction::Continue)
            }
            Some("eval_sample_done") => {
                let sample_id = event
                    .0
                    .get("sample_id")
                    .and_then(Value::as_str)
                    .context("eval_sample_done missing sample_id")?;
                let completion = event
                    .0
                    .get("completion")
                    .and_then(Value::as_str)
                    .unwrap_or("");
                let pos = sample_positions.get(sample_id).copied().unwrap_or(0);
                log::info!(
                    "eval sample {}/{}: id={} completion={}",
                    pos,
                    total_samples,
                    sample_id,
                    completion
                );
                checkpoint.append(BenchmarkEvalCompletion {
                    id: sample_id.to_string(),
                    completion: completion.to_string(),
                    failed: false,
                    failed_reason: None,
                    // stop_reason capture is llama.cpp-only for now (PIP-274);
                    // classifying this runtime is a follow-up, so label it
                    // `unknown` and record why rather than claiming eos/truncated.
                    stop_reason: BenchmarkEvalCompletionStopReason::Unknown,
                    stop_detail: Some("stop_reason capture not implemented for MLX".to_string()),
                    completion_tokens: None,
                })?;
                buffers.remove(sample_id);
                Ok(EvalStreamAction::Continue)
            }
            Some("eval_done") => {
                saw_eval_done = true;
                Ok(EvalStreamAction::Continue)
            }
            Some("eval_error") => {
                let error = event
                    .0
                    .get("error")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown eval stream error");
                anyhow::bail!("pipette_mlx_server /eval failed: {error}")
            }
            None => Ok(EvalStreamAction::Continue),
            Some(other) => {
                log::warn!("unknown eval stream event kind: {other}");
                Ok(EvalStreamAction::Continue)
            }
        }
    })?;

    if !saw_eval_done {
        anyhow::bail!("eval stream ended before eval_done");
    }

    if checkpoint.completions().len() < samples.len() {
        anyhow::bail!(
            "eval incomplete: {} of {} samples persisted in checkpoint at {}",
            checkpoint.completions().len(),
            samples.len(),
            checkpoint.path().display()
        );
    }

    let completions = checkpoint.finalize()?;

    Ok(RunResponse {
        executable: Some(server.executable.clone()),
        command: server.command_preview.clone(),
        ..RunResponse::new(
            BenchmarkResultData::Eval { completions },
            server.stdout(),
            server.stderr(),
        )
    })
}

/// Build the JSON payload handed to the server's `/eval` endpoint.
///
/// `enable_thinking` is intentionally emitted as JSON `null` when the
/// model_flags field is `None` — the server gates with
/// `if enable_thinking is not None:` before forwarding into
/// `tmpl_kwargs`. Switching to `skip_serializing_if = Option::is_none`
/// (or any shape that drops the key) would silently break that contract.
/// The test module pins both ends.
fn build_eval_request_json(
    samples: &[Value],
    max_tokens: u32,
    temperature: Temperature,
    done_ids: &[String],
    enable_thinking: Option<bool>,
) -> Value {
    json!({
        "samples": samples,
        "max_tokens": max_tokens,
        // Sampling temperature for generation. 0.0 is greedy (the prior
        // hardcoded behavior); IFBench/IFStruct send 0.6. No seed is
        // sent, so the server's repeats under temp > 0 are independent.
        "temperature": temperature.as_f64(),
        "completions_done_ids": done_ids,
        // `None` here serializes as JSON null; the server treats
        // null as "do not pass the kwarg" (which is what we want for
        // back-compat with prior runs).
        "enable_thinking": enable_thinking,
    })
}

#[derive(Debug, Clone)]
struct EvalStreamEvent(Value);

impl EvalStreamEvent {
    fn kind(&self) -> Option<&str> {
        self.0.get("kind").and_then(Value::as_str)
    }
}

enum EvalStreamAction {
    Continue,
    AbortSample(String),
}

fn stream_eval_events<F>(
    client: &Client,
    base_url: &str,
    request: &Value,
    idle_timeout: Duration,
    on_event: F,
) -> anyhow::Result<()>
where
    F: FnMut(EvalStreamEvent) -> anyhow::Result<EvalStreamAction>,
{
    let response = client
        .post(format!("{base_url}{EVAL_ENDPOINT}"))
        .json(request)
        .send()
        .context("failed to call pipette_mlx_server /eval")?;
    let status = response.status();
    if !status.is_success() {
        let body = response.text().unwrap_or_default();
        anyhow::bail!("pipette_mlx_server /eval returned HTTP {status}: {body}");
    }

    read_eval_event_lines_with_idle(
        BufReader::new(response),
        idle_timeout,
        on_event,
        |sample_id| abort_eval(client, base_url, sample_id),
    )
}

#[cfg(test)]
fn read_eval_event_lines<R, F, A>(reader: R, mut on_event: F, mut abort: A) -> anyhow::Result<()>
where
    R: BufRead,
    F: FnMut(EvalStreamEvent) -> anyhow::Result<EvalStreamAction>,
    A: FnMut(&str) -> anyhow::Result<()>,
{
    reader.lines().try_for_each(|line| -> anyhow::Result<()> {
        let line = line.context("failed to read /eval stream line")?;
        handle_eval_event_line(&line, &mut on_event, &mut abort)
    })
}

enum EvalLineRead {
    Line(std::io::Result<String>),
    Eof,
}

fn read_eval_event_lines_with_idle<R, F, A>(
    reader: R,
    idle_timeout: Duration,
    mut on_event: F,
    mut abort: A,
) -> anyhow::Result<()>
where
    R: BufRead + Send + 'static,
    F: FnMut(EvalStreamEvent) -> anyhow::Result<EvalStreamAction>,
    A: FnMut(&str) -> anyhow::Result<()>,
{
    let (tx, rx) = mpsc::channel();
    let reader_thread = thread::spawn(move || {
        let read_result = reader.lines().try_for_each(|line| {
            let failed = line.is_err();
            tx.send(EvalLineRead::Line(line)).map_err(|_| ())?;
            if failed {
                Err(())
            } else {
                Ok(())
            }
        });
        if read_result.is_ok() {
            let _ = tx.send(EvalLineRead::Eof);
        }
    });

    loop {
        match rx.recv_timeout(idle_timeout) {
            Ok(EvalLineRead::Line(line)) => {
                let line = line.context("failed to read /eval stream line")?;
                handle_eval_event_line(&line, &mut on_event, &mut abort)?;
            }
            Ok(EvalLineRead::Eof) => {
                let _ = reader_thread.join();
                return Ok(());
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                anyhow::bail!("timed out waiting for /eval stream event after {idle_timeout:?}");
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                anyhow::bail!("eval stream reader stopped unexpectedly");
            }
        }
    }
}

fn handle_eval_event_line<F, A>(line: &str, on_event: &mut F, abort: &mut A) -> anyhow::Result<()>
where
    F: FnMut(EvalStreamEvent) -> anyhow::Result<EvalStreamAction>,
    A: FnMut(&str) -> anyhow::Result<()>,
{
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return Ok(());
    }
    let value: Value = serde_json::from_str(trimmed)
        .with_context(|| format!("parsing /eval stream event: {trimmed}"))?;
    match on_event(EvalStreamEvent(value))? {
        EvalStreamAction::Continue => Ok(()),
        EvalStreamAction::AbortSample(sample_id) => abort(&sample_id),
    }
}

fn abort_eval(client: &Client, base_url: &str, sample_id: &str) -> anyhow::Result<()> {
    let response = client
        .post(format!("{base_url}{EVAL_ABORT_ENDPOINT}"))
        .json(&json!({ "sample_id": sample_id }))
        .send()
        .with_context(|| {
            format!("failed to call pipette_mlx_server /eval/abort for {sample_id}")
        })?;
    let status = response.status();
    if !status.is_success() {
        let body = response.text().unwrap_or_default();
        anyhow::bail!("pipette_mlx_server /eval/abort returned HTTP {status}: {body}");
    }
    Ok(())
}

fn streaming_http_client() -> anyhow::Result<Client> {
    // Long-lived SSE: connect timeout only, no overall body timeout.
    Ok(pipette_http::HttpClient::builder("pipette")
        .preconfigured_tls()
        .connect_timeout(STREAM_CONNECT_TIMEOUT)
        .no_request_timeout()
        .build()
        .context("failed to build MLX eval streaming HTTP client")?
        .client()
        .clone())
}

#[cfg(test)]
mod tests {
    use std::{cell::RefCell, io::Cursor};

    use super::*;

    /// Contract: when `enable_thinking` is `None`, the task JSON must
    /// still contain the key with value `null` — the server's `if
    /// enable_thinking is not None:` gate distinguishes "operator
    /// did not set it" (None → null → skip kwarg) from any present
    /// boolean (True/False → forward kwarg). A `skip_serializing_if`
    /// refactor on the Rust side, or a truthy-check refactor on the
    /// Python side, would silently break this; both ends pin to the
    /// `null` wire form.
    #[test]
    fn enable_thinking_none_serializes_as_explicit_null() -> anyhow::Result<()> {
        let done_ids: Vec<String> = Vec::new();
        let task = build_eval_request_json(&[], 256, Temperature::greedy()?, &done_ids, None);
        assert!(
            task.get("enable_thinking").is_some(),
            "key must be present, not skipped"
        );
        assert_eq!(
            task["temperature"], 0.0,
            "greedy temperature must be forwarded to the server"
        );
        assert!(
            task["enable_thinking"].is_null(),
            "value must be null, got: {}",
            task["enable_thinking"]
        );
        assert!(
            task.get("repo_id").is_none(),
            "server already knows the loaded model"
        );
        assert!(
            task.get("completions_checkpoint_path").is_none(),
            "Rust owns checkpoint appends from the event stream"
        );
        Ok(())
    }

    #[test]
    fn enable_thinking_some_serializes_as_bool() -> anyhow::Result<()> {
        for (input, expected) in [
            (Some(true), Value::Bool(true)),
            (Some(false), Value::Bool(false)),
        ] {
            let done_ids: Vec<String> = Vec::new();
            let task = build_eval_request_json(&[], 256, Temperature::greedy()?, &done_ids, input);
            assert_eq!(task["enable_thinking"], expected);
        }
        Ok(())
    }

    #[test]
    fn eval_event_reader_sends_abort_action() -> anyhow::Result<()> {
        let body = [
            json!({"kind":"eval_sample_chunk","sample_id":"s1","delta":"x"}).to_string(),
            json!({"kind":"eval_done"}).to_string(),
        ]
        .join("\n")
            + "\n";
        let seen = RefCell::new(Vec::<String>::new());
        let aborted = RefCell::new(Vec::<String>::new());

        read_eval_event_lines(
            Cursor::new(body),
            |event| {
                let kind = event.kind().unwrap_or("?").to_string();
                seen.borrow_mut().push(kind.clone());
                if kind == "eval_sample_chunk" {
                    Ok(EvalStreamAction::AbortSample("s1".to_string()))
                } else {
                    Ok(EvalStreamAction::Continue)
                }
            },
            |sample_id| {
                aborted.borrow_mut().push(sample_id.to_string());
                Ok(())
            },
        )?;

        assert_eq!(seen.into_inner(), ["eval_sample_chunk", "eval_done"]);
        assert_eq!(aborted.into_inner(), ["s1"]);
        Ok(())
    }

    #[test]
    fn idle_timeout_from_plan_secs_defaults_and_honors() {
        assert_eq!(
            idle_timeout_from_plan_secs(None),
            DEFAULT_STREAM_IDLE_TIMEOUT
        );
        assert_eq!(
            idle_timeout_from_plan_secs(Some(90)),
            Duration::from_secs(90)
        );
        assert_eq!(
            idle_timeout_from_plan_secs(Some(0)),
            Duration::from_secs(1),
            "zero clamps to 1s"
        );
    }
}
