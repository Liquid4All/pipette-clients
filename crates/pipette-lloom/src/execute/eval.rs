//! `/eval`: the sidecar streams JSONL events per sample; the runner persists
//! each completion to the checkpoint as it lands, runs the doom-loop check
//! on the accumulating text and aborts a sample through `/eval/abort`.
//!
//! One thing this contract carries that the MLX one does not: `lloom-serve`
//! reports `finish_reason` (`stop` | `length`) and `completion_tokens` on
//! `eval_sample_done`, so the stop reason is classified here rather than
//! left `Unknown`.

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
use crate::models::require_lloom_model_dir;
use crate::runtimes::require_lloom_serve;

const EVAL_ENDPOINT: &str = "/eval";
const EVAL_ABORT_ENDPOINT: &str = "/eval/abort";
const STREAM_CONNECT_TIMEOUT: Duration = Duration::from_secs(60);
const DEFAULT_STREAM_IDLE_TIMEOUT: Duration = Duration::from_secs(1800);

fn stream_idle_timeout(req: &RunRequest) -> Duration {
    req.benchmark_flags
        .as_ref()
        .and_then(|f| f.http_timeout())
        .map(|s| Duration::from_secs(s.max(1)))
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
    let binary = require_lloom_serve(req)?;
    let model_dir = require_lloom_model_dir(req)?;
    let temperature = req.benchmark.eval_temperature()?;
    let doomloop = pipette_doomloop::plan::pipeline_from_overrides(
        req.benchmark_flags.as_ref().and_then(|bf| bf.doomloop()),
    )
    .map_err(|e| anyhow::anyhow!("invalid doom-loop configuration: {e}"))?;
    let enable_thinking = req.model_flags.as_ref().and_then(|f| f.enable_thinking());

    let mut checkpoint = eval_completions.open(req)?;
    let done_ids: Vec<String> = checkpoint.done_ids().map(str::to_string).collect();
    let request =
        build_eval_request_json(samples, max_tokens, temperature, &done_ids, enable_thinking);

    let total = samples.len();
    let positions: HashMap<&str, usize> = samples
        .iter()
        .enumerate()
        .filter_map(|(i, s)| s.get("id").and_then(Value::as_str).map(|id| (id, i + 1)))
        .collect();
    for id in &done_ids {
        log::info!(
            "eval sample {}/{total}: id={id} (skipped, already checkpointed)",
            positions.get(id.as_str()).copied().unwrap_or(0)
        );
    }

    let server = server::start_server(&binary, &model_dir, None)?;
    let client = streaming_http_client()?;
    let idle_timeout = stream_idle_timeout(req);
    let mut buffers: HashMap<String, String> = HashMap::new();
    let mut flagged: HashSet<String> = HashSet::new();
    let mut saw_eval_done = false;
    stream_eval_events(
        &client,
        &server.base_url,
        &request,
        idle_timeout,
        |event| match event.kind() {
            Some("eval_sample_start") => {
                let id = event
                    .str("sample_id")
                    .context("eval_sample_start missing sample_id")?;
                log::info!(
                    "eval sample {}/{total}: id={id} prompt={}",
                    positions.get(id).copied().unwrap_or(0),
                    event.str("prompt").unwrap_or("")
                );
                Ok(EvalStreamAction::Continue)
            }
            Some("eval_sample_chunk") => {
                let id = event
                    .str("sample_id")
                    .context("eval_sample_chunk missing sample_id")?;
                let delta = event
                    .str("delta")
                    .context("eval_sample_chunk missing delta")?;
                let content = buffers.entry(id.to_string()).or_default();
                content.push_str(delta);
                if !flagged.contains(id) {
                    if let Some(name) = doomloop.check(content) {
                        log::warn!("{}", format_trigger_log(name, content.len()));
                        flagged.insert(id.to_string());
                        return Ok(EvalStreamAction::AbortSample(id.to_string()));
                    }
                }
                Ok(EvalStreamAction::Continue)
            }
            Some("eval_sample_done") => {
                let id = event
                    .str("sample_id")
                    .context("eval_sample_done missing sample_id")?;
                let completion = event.str("completion").unwrap_or("");
                let stopped_early = event.0.get("stopped_early").and_then(Value::as_bool);
                let (stop_reason, stop_detail) = classify_stop(
                    event.str("finish_reason"),
                    stopped_early,
                    flagged.contains(id),
                );
                log::info!(
                    "eval sample {}/{total}: id={id} {stop_reason:?} completion={completion}",
                    positions.get(id).copied().unwrap_or(0)
                );
                checkpoint.append(BenchmarkEvalCompletion {
                    id: id.to_string(),
                    completion: completion.to_string(),
                    failed: false,
                    failed_reason: None,
                    stop_reason,
                    stop_detail,
                    completion_tokens: event.0.get("completion_tokens").and_then(Value::as_u64),
                })?;
                buffers.remove(id);
                Ok(EvalStreamAction::Continue)
            }
            Some("eval_done") => {
                saw_eval_done = true;
                Ok(EvalStreamAction::Continue)
            }
            Some("eval_error") => {
                anyhow::bail!(
                    "lloom-serve /eval failed: {}",
                    event.str("error").unwrap_or("unknown eval stream error")
                )
            }
            None => Ok(EvalStreamAction::Continue),
            Some(other) => {
                log::warn!("unknown eval stream event kind: {other}");
                Ok(EvalStreamAction::Continue)
            }
        },
    )?;
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

/// The stop reason from what the server said: an abort this runner asked for
/// is a doom loop; `length` is the token budget; `stop` is EOS.
fn classify_stop(
    finish_reason: Option<&str>,
    stopped_early: Option<bool>,
    aborted_by_us: bool,
) -> (BenchmarkEvalCompletionStopReason, Option<String>) {
    if aborted_by_us || stopped_early == Some(true) {
        return (
            BenchmarkEvalCompletionStopReason::DoomLoop,
            Some("aborted through /eval/abort".to_string()),
        );
    }
    match finish_reason {
        Some("stop") => (BenchmarkEvalCompletionStopReason::Eos, None),
        Some("length") => (BenchmarkEvalCompletionStopReason::Truncated, None),
        other => (
            BenchmarkEvalCompletionStopReason::Unknown,
            Some(format!("finish_reason {other:?}")),
        ),
    }
}

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
        "temperature": temperature.as_f64(),
        "completions_done_ids": done_ids,
        "enable_thinking": enable_thinking,
    })
}

#[derive(Debug, Clone)]
struct EvalStreamEvent(Value);

impl EvalStreamEvent {
    fn kind(&self) -> Option<&str> {
        self.str("kind")
    }

    fn str(&self, key: &str) -> Option<&str> {
        self.0.get(key).and_then(Value::as_str)
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
        .context("failed to call lloom-serve /eval")?;
    let status = response.status();
    if !status.is_success() {
        let body = response.text().unwrap_or_default();
        anyhow::bail!("lloom-serve /eval returned HTTP {status}: {body}");
    }
    read_eval_event_lines_with_idle(BufReader::new(response), idle_timeout, on_event, |id| {
        abort_eval(client, base_url, id)
    })
}

enum EvalLineRead {
    Line(std::io::Result<String>),
    Eof,
}

/// Lines arrive on a reader thread so an idle stream (a server that stopped
/// producing) is a timeout, not a hang.
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
        .with_context(|| format!("failed to call lloom-serve /eval/abort for {sample_id}"))?;
    let status = response.status();
    if !status.is_success() {
        let body = response.text().unwrap_or_default();
        anyhow::bail!("lloom-serve /eval/abort returned HTTP {status}: {body}");
    }
    Ok(())
}

fn streaming_http_client() -> anyhow::Result<Client> {
    Ok(pipette_http::HttpClient::builder("pipette")
        .preconfigured_tls()
        .connect_timeout(STREAM_CONNECT_TIMEOUT)
        .no_request_timeout()
        .build()
        .context("failed to build the lloom-serve eval streaming HTTP client")?
        .client()
        .clone())
}

#[cfg(test)]
mod tests {
    use std::{cell::RefCell, io::Cursor};

    use super::*;

    #[test]
    fn the_request_carries_the_sidecar_contract_fields() -> anyhow::Result<()> {
        let task = build_eval_request_json(&[], 256, Temperature::greedy()?, &[], None);
        assert_eq!(task["temperature"], 0.0);
        assert_eq!(task["max_tokens"], 256);
        assert!(task["enable_thinking"].is_null());
        assert!(task["completions_done_ids"]
            .as_array()
            .is_some_and(Vec::is_empty));
        Ok(())
    }

    #[test]
    fn stop_reasons_follow_the_servers_finish_reason() {
        use BenchmarkEvalCompletionStopReason as R;
        assert!(matches!(
            classify_stop(Some("stop"), Some(false), false).0,
            R::Eos
        ));
        assert!(matches!(
            classify_stop(Some("length"), Some(false), false).0,
            R::Truncated
        ));
        assert!(matches!(
            classify_stop(Some("length"), Some(true), false).0,
            R::DoomLoop
        ));
        assert!(matches!(
            classify_stop(Some("stop"), None, true).0,
            R::DoomLoop
        ));
        assert!(matches!(classify_stop(None, None, false).0, R::Unknown));
    }

    #[test]
    fn an_abort_action_calls_the_abort_hook_and_the_stream_continues() -> anyhow::Result<()> {
        let body = [
            json!({"kind":"eval_sample_chunk","sample_id":"s1","delta":"x"}).to_string(),
            json!({"kind":"eval_done"}).to_string(),
        ]
        .join("\n")
            + "\n";
        let seen = RefCell::new(Vec::<String>::new());
        let aborted = RefCell::new(Vec::<String>::new());
        read_eval_event_lines_with_idle(
            Cursor::new(body),
            Duration::from_secs(5),
            |event| {
                let kind = event.kind().unwrap_or("?").to_string();
                seen.borrow_mut().push(kind.clone());
                Ok(if kind == "eval_sample_chunk" {
                    EvalStreamAction::AbortSample("s1".to_string())
                } else {
                    EvalStreamAction::Continue
                })
            },
            |id| {
                aborted.borrow_mut().push(id.to_string());
                Ok(())
            },
        )?;
        assert_eq!(seen.borrow().as_slice(), ["eval_sample_chunk", "eval_done"]);
        assert_eq!(aborted.borrow().as_slice(), ["s1"]);
        Ok(())
    }

    #[test]
    fn a_silent_stream_times_out() {
        struct Never;
        impl std::io::Read for Never {
            fn read(&mut self, _: &mut [u8]) -> std::io::Result<usize> {
                std::thread::sleep(Duration::from_secs(10));
                Ok(0)
            }
        }
        let err = read_eval_event_lines_with_idle(
            BufReader::new(Never),
            Duration::from_millis(100),
            |_| Ok(EvalStreamAction::Continue),
            |_| Ok(()),
        )
        .err()
        .map(|e| e.to_string())
        .unwrap_or_default();
        assert!(err.contains("timed out"), "got: {err}");
    }
}
