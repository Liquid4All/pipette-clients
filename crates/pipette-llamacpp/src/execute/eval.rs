use std::{
    io::{BufRead, BufReader, Write},
    time::Duration,
};

use anyhow::Context;
use reqwest::blocking::Client;
use serde::Deserialize;
use serde_json::{json, Value};
use time::format_description::well_known::Rfc3339;
use time::OffsetDateTime;

use pipette_doomloop::{format_trigger_log, DoomloopPipeline};
use pipette_http::HttpClient;
use pipette_ops::eval_completions::{EvalCompletionSession, EvalCompletionsStore};
use pipette_plan_types::benchmark::Temperature;
use pipette_plan_types::reserved_flags::llamacpp_cli_stock_tools as reserved;
use pipette_plan_types::result::{
    BenchmarkEvalCompletion, BenchmarkEvalCompletionStopReason, BenchmarkResultData,
};
use pipette_plan_types::run::RunRequest;

use super::RunResponse;
use crate::models::require_gguf_text;
use crate::runtime_flags::{self, MmapPolicy};
use crate::runtimes::require_llama_server;
use crate::server;

/// Default HTTP timeout when `benchmark_flags.http_timeout` is unset.
const DEFAULT_HTTP_TIMEOUT_SECS: u64 = 120;

/// Logged-progress step size for the eval loop. Clamps to `[1, 20]`
/// so the log doesn't spam for tiny benchmark sets or fall silent
/// for huge ones.
fn eval_progress_step(total_samples: usize) -> usize {
    total_samples.clamp(1, 20)
}

/// Eval sample loop for a prepared [`RunRequest`] (`benchmark` must be eval).
///
/// Resume identity hashes the portable [`RunRequest`].
pub fn run_eval(
    req: &RunRequest,
    eval_completions: &EvalCompletionsStore,
) -> anyhow::Result<RunResponse> {
    let benchmark = req.benchmark.as_eval().map_err(anyhow::Error::from)?;
    let llama_server = require_llama_server(req)?;
    let model_path = require_gguf_text(req)?;
    let flags = runtime_flags::for_server(
        req,
        8192u32.saturating_add(benchmark.parameter_max_tokens),
        // Eval runs are long enough to amortize page-ins; leave llama.cpp's
        // mmap-on default alone.
        MmapPolicy::AsAuthored,
    )?;
    let extra_flags = server::args_for(&flags).build(reserved::SERVER, "eval")?;

    let doomloop = pipette_doomloop::plan::pipeline_from_overrides(
        req.benchmark_flags.as_ref().and_then(|bf| bf.doomloop()),
    )
    .map_err(|e| anyhow::anyhow!("invalid doom-loop configuration: {e}"))?;
    let enable_thinking = req.model_flags.as_ref().and_then(|f| f.enable_thinking());

    let benchmark_id = benchmark.benchmark_id.as_str();
    let samples = benchmark
        .samples
        .as_deref()
        .context("eval benchmark missing samples")?;
    let max_tokens = u64::from(benchmark.parameter_max_tokens);
    let mcq_choices = benchmark
        .parameter_mcq_choices
        .clone()
        .filter(|items| !items.is_empty());
    // Sampling temperature is a client-side policy keyed on the eval id
    // (the server does not send one): IFBench/IFStruct sample at 0.6,
    // everything else stays greedy.
    let temperature = benchmark.sampling_temperature()?;
    let total_samples = samples.len();
    let progress_every = eval_progress_step(total_samples);
    let request_timeout = request_timeout_from_req(req);

    let server = start_and_wait_ready(&llama_server, &model_path, &extra_flags, request_timeout)?;
    let apply_template_client = HttpClient::blocking_with_timeout("pipette", request_timeout)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let completion_mode = CompletionMode::pick(
        mcq_choices.as_deref(),
        max_tokens,
        temperature,
        &doomloop,
        request_timeout,
    )?;

    // Resume keyed by portable RunRequest identity (EvalCompletionsStore).
    let checkpoint = eval_completions.open(req)?;

    // Thread (checkpoint, server) through each sample. Every arm of
    // every iteration returns a fresh `(checkpoint, server)` pair —
    // the server is replaced after a crash but kept on transport
    // errors; the checkpoint is replaced after each append. The state
    // is owned, not captured by reference, so the closure has no
    // `&mut` upvars.
    let (checkpoint, server) = samples.iter().enumerate().try_fold(
        (checkpoint, server),
        |(checkpoint, mut server), (index, sample)| -> anyhow::Result<(_, _)> {
            let sample_id = sample
                .get("id")
                .and_then(Value::as_str)
                .context("eval sample missing id")?;
            // `eval_completions.open` already logs the resume-count summary; a
            // per-sample skip log was scrollback noise at scale.
            if checkpoint.contains(sample_id) {
                return Ok((checkpoint, server));
            }
            let messages = sample
                .get("messages")
                .cloned()
                .context("eval sample missing messages")?;
            let position = format!("eval sample {}/{total_samples}: id={sample_id}", index + 1);

            match completion_mode.run_sample(
                &apply_template_client,
                &server.base_url,
                &messages,
                &position,
                enable_thinking,
            ) {
                Ok(outcome) => {
                    let checkpoint = checkpoint.with_append(BenchmarkEvalCompletion {
                        id: sample_id.to_string(),
                        completion: outcome.content,
                        failed: false,
                        failed_reason: None,
                        stop_reason: outcome.stop_reason,
                        stop_detail: outcome.stop_detail,
                        completion_tokens: outcome.completion_tokens,
                    })?;
                    log_progress_tick(benchmark_id, &checkpoint, total_samples, progress_every);
                    Ok((checkpoint, server))
                }
                Err(err) => {
                    // Two recovery paths, both record `failed: true`
                    // and continue with the next sample:
                    //
                    //   * child exited → server is gone; spawn a fresh
                    //     one before the next sample.
                    //   * child still alive → transport / HTTP error
                    //     against a live server; keep the same server
                    //     and just skip the sample.
                    //
                    // llama.cpp is known to crash on some prompts; we
                    // want correct per-sample reporting rather than a
                    // whole-cell abort.
                    //
                    // Ordering is load-bearing: append the failed
                    // entry to the checkpoint *before* attempting a
                    // restart. If `start_and_wait_ready` fails (port
                    // collision, OOM, model gone) we still want the
                    // poison sample recorded so a retry skips it
                    // rather than re-crashing in a loop.
                    let exit_status =
                        server::poll_child_exit(&mut server.child, CRASH_OBSERVATION_WINDOW)?;
                    let (live_server, reason, stderr_tail) = match exit_status {
                        Some(status) => {
                            // Crash path: drain the dead server's
                            // stderr; we'll spawn a fresh one after
                            // appending the failed entry.
                            let stderr = server.shutdown_collect_stderr();
                            (None, build_crash_reason(status, &err), stderr)
                        }
                        None => {
                            // Transport error against a live server.
                            // Snapshot recent stderr (non-consuming)
                            // so operators see anything llama-server
                            // logged before dropping the connection
                            // (OOM, tokenizer warnings, etc.). Don't
                            // touch the server itself; the next sample
                            // will hit the same process.
                            let stderr = server.stderr_snapshot();
                            (Some(server), build_transport_failure_reason(&err), stderr)
                        }
                    };
                    log_sample_failure(&position, &reason, &messages, &err, &stderr_tail);
                    let checkpoint = checkpoint.with_append(BenchmarkEvalCompletion {
                        id: sample_id.to_string(),
                        completion: String::new(),
                        failed: true,
                        failed_reason: Some(reason.clone()),
                        // The client is the source of truth for `stop_reason`:
                        // a sample that never produced a completion is a
                        // `failure`, with the crash/transport detail in
                        // `stop_detail`. `failed` / `failed_reason` are
                        // dual-written for legacy consumers until PIP-323
                        // retires them.
                        stop_reason: BenchmarkEvalCompletionStopReason::Failure,
                        stop_detail: Some(reason),
                        completion_tokens: None,
                    })?;
                    let server = match live_server {
                        Some(s) => s,
                        None => {
                            log::info!("restarting llama-server after crash");
                            start_and_wait_ready(
                                &llama_server,
                                &model_path,
                                &extra_flags,
                                request_timeout,
                            )
                            .context("llama-server failed to come back after restart")?
                        }
                    };
                    Ok((checkpoint, server))
                }
            }
        },
    )?;

    // Capture the FAILED-block signal before `finalize` consumes the
    // session. See `EvalCompletionSession::finalize` and
    // `docs/pipette-cli/eval-checkpoint.md` for the on-disk and wire contracts.
    let failed_signal = failed_signal_message(benchmark_id, &checkpoint);
    log_eval_summary(benchmark_id, &checkpoint);
    let completions = checkpoint.finalize()?;
    Ok(RunResponse {
        executable: Some(llama_server.display().to_string()),
        // After a mid-run restart this argv differs from the original
        // only in `--port`.
        command: server.command_preview.clone(),
        runtime_flags: Some(flags.clone()),
        ..RunResponse::new(
            BenchmarkResultData::Eval { completions },
            String::new(),
            failed_signal,
        )
    })
}

/// POST `/apply-template` with the sample's messages; return the
/// rendered prompt string. Runtime-agnostic; the MCQ-vs-streaming
/// dispatch lives on `CompletionMode::run_sample` below.
///
/// `enable_thinking` is forwarded as `chat_template_kwargs.enable_thinking`
/// when set; the field is honored by `/apply-template` on b9119+
/// regardless of whether the server was started with `--jinja`. Omit
/// the kwarg entirely when unset so the request body stays
/// byte-identical to pre-flag callers.
fn fetch_prompt(
    client: &Client,
    base_url: &str,
    messages: &Value,
    enable_thinking: Option<bool>,
) -> anyhow::Result<String> {
    let mut body = serde_json::Map::new();
    body.insert("messages".to_string(), messages.clone());
    if let Some(enable_thinking) = enable_thinking {
        body.insert(
            "chat_template_kwargs".to_string(),
            json!({ "enable_thinking": enable_thinking }),
        );
    }
    let apply: Value = client
        .post(format!("{base_url}/apply-template"))
        .json(&Value::Object(body))
        .send()
        .context("failed to call /apply-template")?
        .error_for_status()
        .context("/apply-template failed")?
        .json()
        .context("failed to parse /apply-template response")?;
    apply
        .get("prompt")
        .and_then(Value::as_str)
        .map(str::to_string)
        .context("/apply-template response missing prompt")
}

/// How long to wait for `llama-server` to surface as exited after a
/// `/completion` failure before deciding the server is still alive.
/// See `server::poll_child_exit` for why this is multi-
/// second rather than a bare `try_wait`.
const CRASH_OBSERVATION_WINDOW: Duration = Duration::from_secs(3);

/// Root cause string from an `anyhow::Error` chain — the deepest
/// `source()`, falling back to the top-level display if the chain is
/// single-link. Used to surface transport/HTTP causes (e.g.
/// `connection error: forcibly closed (os error 10054)`) in the
/// checkpoint's `failed_reason` and downstream mgmt rows.
fn root_cause(err: &anyhow::Error) -> String {
    err.chain()
        .last()
        .map(ToString::to_string)
        .unwrap_or_else(|| err.to_string())
}

/// Build the `failed_reason` for the *child-exited* path. The
/// `[<rfc3339>]` prefix folds the observed-at timestamp into the wire
/// format without a dedicated field (see
/// `BenchmarkEvalCompletion::failed_reason`). The transport-error
/// chain root is appended so mgmt rows carry both the exit status
/// *and* what the client saw.
fn build_crash_reason(exit_status: std::process::ExitStatus, err: &anyhow::Error) -> String {
    let observed_at = OffsetDateTime::now_utc()
        .format(&Rfc3339)
        .unwrap_or_else(|_| "unknown".to_string());
    format!(
        "[{observed_at}] llama-server crashed mid-completion: exit {exit_status}; client saw: {}",
        root_cause(err),
    )
}

/// Build the `failed_reason` for the *server-still-alive* path: a
/// `/completion` call returned an HTTP / transport error but the
/// child process is still running. We keep the existing server and
/// only the sample is recorded as failed.
fn build_transport_failure_reason(err: &anyhow::Error) -> String {
    let observed_at = OffsetDateTime::now_utc()
        .format(&Rfc3339)
        .unwrap_or_else(|_| "unknown".to_string());
    format!(
        "[{observed_at}] llama-server /completion failed (server still alive): {}",
        root_cause(err),
    )
}

/// Emit the operator-visible WARN line for a per-sample failure, plus
/// a stderr-tail line when the server left something useful behind.
/// `stderr` comes from `shutdown_collect_stderr` on the crash path
/// and from `stderr_snapshot` on the live-server path; it may be
/// empty if the server hadn't logged anything yet. `position` is the
/// shared "eval sample N/M: id=X" prefix.
fn log_sample_failure(
    position: &str,
    reason: &str,
    messages: &Value,
    err: &anyhow::Error,
    stderr: &str,
) {
    let prompt_preview = messages
        .as_array()
        .and_then(|m| m.last())
        .and_then(|m| m.get("content"))
        .and_then(Value::as_str)
        .map(|s| s.chars().take(80).collect::<String>())
        .unwrap_or_default();
    log::warn!(
        "{position} FAILED ({reason}); prompt~={prompt_preview:?}; underlying error: {err:#}"
    );
    if !stderr.is_empty() {
        log::warn!("llama-server stderr tail:\n{}", tail_lines(stderr, 20));
    }
}

/// Per-sample progress log. `processed` counts both completed and
/// failed so the end-of-cell tick still fires when samples crashed;
/// the displayed `completed` stays success-only.
fn log_progress_tick(
    benchmark_id: &str,
    checkpoint: &EvalCompletionSession,
    total_samples: usize,
    progress_every: usize,
) {
    let processed = checkpoint.completions().len();
    if processed == 1 || processed == total_samples || processed.is_multiple_of(progress_every) {
        let completed = processed - checkpoint.failed_count();
        log::info!("eval progress: benchmark={benchmark_id} completed={completed}/{total_samples}");
    }
}

/// Spawn `llama-server` and block until `/health` is ready. On failure,
/// drains the server's stderr into the returned error so the operator
/// sees what went wrong without digging through scrollback. The
/// original error's anyhow chain is preserved via `.context(...)`
/// (vs. `bail!("{e}")`, which would flatten causes into a string).
/// Used for both the initial startup and the post-crash restart.
fn start_and_wait_ready(
    llama_server: &std::path::Path,
    model_path: &std::path::Path,
    extra_flags: &[String],
    request_timeout: Duration,
) -> anyhow::Result<server::RunningLlamaServer> {
    let mut server = server::start(llama_server, model_path, None, extra_flags)?;
    if let Err(e) = server::wait_until_ready(&server.base_url, &mut server.child, request_timeout) {
        let stderr = server::shutdown_and_collect_stderr(&mut server);
        return if stderr.is_empty() {
            Err(e)
        } else {
            Err(e.context(format!("server stderr:\n{stderr}")))
        };
    }
    Ok(server)
}

/// Return at most the last `n` lines of `text`, rejoined with `\n`. Used
/// to keep crash logs readable when the runtime emitted thousands of
/// lines before dying.
fn tail_lines(text: &str, n: usize) -> String {
    let lines: Vec<&str> = text.lines().collect();
    let start = lines.len().saturating_sub(n);
    lines[start..].join("\n")
}

/// Final summary line for an eval cell. Mirrors the per-cell view operators
/// look for in the log: how many samples landed in the result payload vs.
/// how many were skipped as failed, with the offending ids inlined.
fn log_eval_summary(benchmark_id: &str, checkpoint: &EvalCompletionSession) {
    let failed_count = checkpoint.failed_count();
    let completed = checkpoint.completions().len() - failed_count;
    if failed_count == 0 {
        log::info!("eval finished: benchmark={benchmark_id} completed={completed} failed=0");
    } else {
        let ids: Vec<&str> = checkpoint.failed_ids().collect();
        log::warn!(
            "eval finished: benchmark={benchmark_id} completed={completed} \
             failed={failed_count} ids={ids:?}"
        );
    }
}

/// Build the FAILED block written to `RunResponse.stderr` — lands
/// in `result_extras_path` so operators can grep historical failures
/// without keeping log scrollback. Empty string when no failed entries.
fn failed_signal_message(benchmark_id: &str, checkpoint: &EvalCompletionSession) -> String {
    use std::fmt::Write;

    let failed = checkpoint.failed_count();
    if failed == 0 {
        return String::new();
    }
    let total = checkpoint.completions().len();
    let mut out = String::new();
    let _ = writeln!(
        out,
        "FAILED: benchmark={benchmark_id} failed={failed} of {total}"
    );
    for entry in checkpoint.completions().iter().filter(|c| c.failed) {
        let reason = entry.failed_reason.as_deref().unwrap_or("(no reason)");
        let _ = writeln!(out, "  id={} reason={}", entry.id, reason);
    }
    out
}

/// A single sample's completion plus the stop metadata captured at
/// generation. `stop_reason` is always classified (the MCQ arm and an
/// unclassifiable stream fall back to `unknown`); `stop_detail` carries the
/// raw why behind a non-clean stop; `completion_tokens` is `None` when the
/// runtime didn't report a count.
struct SampleOutcome {
    content: String,
    stop_reason: BenchmarkEvalCompletionStopReason,
    stop_detail: Option<String>,
    completion_tokens: Option<u64>,
}

/// Fields llama-server emits on the final streaming chunk (`stop: true`),
/// mirroring the completion-response shape parsed by `end_to_end_latency`
/// / `vl_throughput`. `predicted_n` is the generated-token count. Both the
/// boolean flags (older builds) and the `stop_type` string (newer builds)
/// are parsed so classification doesn't depend on which the server sends.
#[derive(Debug, Default, Deserialize)]
struct StreamTermination {
    #[serde(default)]
    stopped_limit: bool,
    #[serde(default)]
    stopped_eos: bool,
    #[serde(default)]
    stop_type: Option<String>,
    #[serde(default)]
    timings: StreamTimings,
}

#[derive(Debug, Default, Deserialize)]
struct StreamTimings {
    #[serde(default)]
    predicted_n: Option<u64>,
}

/// Outcome of reading the SSE stream: the accumulated text, the doom-loop
/// trigger detail (`Some` iff the detector aborted generation), and the
/// terminal event's stop metadata (absent if the stream ended without a
/// `stop: true` event).
struct SseOutcome {
    content: String,
    doom_loop: Option<String>,
    termination: Option<StreamTermination>,
}

/// The classification of a finished stream: the canonical `reason`, the
/// generated-token `completion_tokens` (when the terminal event reported
/// them), and a free-form `detail` recording the raw why for a non-clean
/// stop (so an `unknown` is never a dead end when debugging).
#[derive(Debug, PartialEq, Eq)]
struct StopClassification {
    reason: BenchmarkEvalCompletionStopReason,
    completion_tokens: Option<u64>,
    detail: Option<String>,
}

/// Classify a finished stream. A doom-loop abort wins — it's the only source
/// of `doom_loop`. Otherwise the terminal event's flags decide: `limit` ⇒
/// `truncated` (hit the output-token cap), `eos` ⇒ `eos` (model emitted
/// EOS). Anything else — an unrecognized `stop_type` (e.g. `"word"`), or no
/// terminal event at all (a dropped stream) — is `unknown` rather than a
/// guessed `eos`.
fn classify_stop(sse: &SseOutcome) -> StopClassification {
    if let Some(detail) = &sse.doom_loop {
        return StopClassification {
            reason: BenchmarkEvalCompletionStopReason::DoomLoop,
            completion_tokens: None,
            detail: Some(detail.clone()),
        };
    }
    let Some(term) = &sse.termination else {
        return StopClassification {
            reason: BenchmarkEvalCompletionStopReason::Unknown,
            completion_tokens: None,
            detail: Some("stream ended without a terminal stop event".to_string()),
        };
    };
    let tokens = term.timings.predicted_n;
    if term.stopped_limit || term.stop_type.as_deref() == Some("limit") {
        StopClassification {
            reason: BenchmarkEvalCompletionStopReason::Truncated,
            completion_tokens: tokens,
            detail: None,
        }
    } else if term.stopped_eos || term.stop_type.as_deref() == Some("eos") {
        StopClassification {
            reason: BenchmarkEvalCompletionStopReason::Eos,
            completion_tokens: tokens,
            detail: None,
        }
    } else {
        let detail = match term.stop_type.as_deref() {
            Some(t) => format!("unrecognized stop_type={t}"),
            None => "terminal event carried no stop signal".to_string(),
        };
        StopClassification {
            reason: BenchmarkEvalCompletionStopReason::Unknown,
            completion_tokens: tokens,
            detail: Some(detail),
        }
    }
}

/// How a single sample's `/completion` is fetched. MCQ and streaming
/// differ on three coupled axes — HTTP client, request body, response
/// parser — and a mismatch (e.g. bounded client + SSE body) would
/// silently break. Bundling them into one enum co-locates the
/// decision: `pick` chooses the variant once at run start, and every
/// per-sample call goes through `run_sample`.
enum CompletionMode<'a> {
    /// Grammar-constrained single-token request, returned as a single
    /// JSON response. Uses a bounded-timeout client; an SSE-shaped
    /// client would risk hanging on a wedged server.
    Mcq {
        client: Client,
        grammar: String,
        temperature: Temperature,
    },
    /// Streaming free-form generation. Uses a connect-only-timeout
    /// client so the long-lived SSE response isn't killed mid-stream
    /// by the overall HTTP timeout. Borrows the doom-loop pipeline
    /// from the eval setup because only this path consumes it.
    Streaming {
        client: Client,
        max_tokens: u64,
        temperature: Temperature,
        doomloop: &'a DoomloopPipeline,
    },
}

impl<'a> CompletionMode<'a> {
    fn pick(
        mcq_choices: Option<&[String]>,
        max_tokens: u64,
        temperature: Temperature,
        doomloop: &'a DoomloopPipeline,
        request_timeout: Duration,
    ) -> anyhow::Result<Self> {
        match mcq_choices {
            Some(choices) => Ok(Self::Mcq {
                client: HttpClient::blocking_with_timeout("pipette", request_timeout)
                    .map_err(|e| anyhow::anyhow!("{e}"))?,
                grammar: mcq_grammar(choices),
                temperature,
            }),
            None => Ok(Self::Streaming {
                client: streaming_http_client(request_timeout)?,
                max_tokens,
                temperature,
                doomloop,
            }),
        }
    }

    /// Apply the chat template, then fetch a completion (single-shot
    /// JSON for MCQ, streaming SSE for free-form). `position` is the
    /// shared "eval sample N/M: id=X" prefix used by the prompt and
    /// completion info logs. `enable_thinking` is the operator's declared
    /// value, `None` when unset — see `fetch_prompt` for the
    /// `chat_template_kwargs` contract.
    fn run_sample(
        &self,
        apply_template_client: &Client,
        base_url: &str,
        messages: &Value,
        position: &str,
        enable_thinking: Option<bool>,
    ) -> anyhow::Result<SampleOutcome> {
        let prompt = fetch_prompt(apply_template_client, base_url, messages, enable_thinking)?;
        log::info!("{position} prompt={prompt}");
        let url = format!("{base_url}/completion");
        let outcome = match self {
            Self::Mcq {
                client,
                grammar,
                temperature,
            } => {
                // No `seed` is sent: under temperature > 0 each `#k`
                // repeat must be an independent draw (see PIP-180). A
                // pinned seed would make all repeats identical and
                // collapse pass@1 to the single-shot number.
                let body = json!({
                    "prompt": prompt,
                    "temperature": temperature.as_f64(),
                    "n_predict": 1,
                    "grammar": grammar,
                });
                let completion: Value = client
                    .post(&url)
                    .json(&body)
                    .send()
                    .context("failed to call /completion")?
                    .error_for_status()
                    .context("/completion failed")?
                    .json()
                    .context("failed to parse /completion response")?;
                // MCQ uses `n_predict: 1` + grammar — the server must
                // emit one token from the grammar alternatives, so a
                // missing or non-string `content` field signals a
                // server bug. Bail loudly instead of substituting an
                // empty string (which would score as a wrong answer
                // and bury the real problem).
                let content = completion
                    .get("content")
                    .and_then(Value::as_str)
                    .context("/completion response missing string `content` field")?
                    .to_string();
                // MCQ (`n_predict: 1` + grammar) is out of scope for
                // stop_reason capture — no current eval uses it — so it can't
                // give a meaningful eos/truncated: label it `unknown` and say why.
                SampleOutcome {
                    content,
                    stop_reason: BenchmarkEvalCompletionStopReason::Unknown,
                    stop_detail: Some("mcq arm (n_predict:1, grammar-constrained)".to_string()),
                    completion_tokens: None,
                }
            }
            Self::Streaming {
                client,
                max_tokens,
                temperature,
                doomloop,
            } => {
                // No `seed` is sent here either — see the MCQ arm above.
                let body = json!({
                    "prompt": prompt,
                    "temperature": temperature.as_f64(),
                    "n_predict": max_tokens,
                    "stream": true,
                });
                let sse = stream_completion(client, &url, &body, doomloop)?;
                let stop = classify_stop(&sse);
                SampleOutcome {
                    content: sse.content,
                    stop_reason: stop.reason,
                    stop_detail: stop.detail,
                    completion_tokens: stop.completion_tokens,
                }
            }
        };
        log::info!(
            "{position} completion={} stop_reason={:?} completion_tokens={:?} stop_detail={:?}",
            outcome.content,
            outcome.stop_reason,
            outcome.completion_tokens,
            outcome.stop_detail
        );
        Ok(outcome)
    }
}

/// HTTP client for streaming: connect timeout only, no overall response
/// timeout that would kill the long-lived SSE connection. The server is
/// local and managed by us — if it hangs, the `RunningLlamaServer`
/// `Drop` will kill it and unblock the reader.
fn streaming_http_client(connect_timeout: Duration) -> anyhow::Result<Client> {
    Ok(pipette_http::HttpClient::builder("pipette")
        .preconfigured_tls()
        .connect_timeout(connect_timeout)
        .no_request_timeout()
        .build()
        .context("failed to build streaming HTTP client")?
        .client()
        .clone())
}

/// Send a streaming completion request and read SSE events, printing content
/// to stderr as it arrives. Returns the fully accumulated completion text.
/// If a doom-loop detector fires, generation is stopped early and a
/// `log::warn!` line is emitted.
fn stream_completion(
    client: &Client,
    url: &str,
    body: &Value,
    doomloop: &DoomloopPipeline,
) -> anyhow::Result<SseOutcome> {
    let response = client
        .post(url)
        .json(body)
        .send()
        .context("failed to call /completion")?
        .error_for_status()
        .context("/completion failed")?;
    let reader = BufReader::new(response);
    let mut stderr = std::io::stderr().lock();
    let outcome = read_sse_chunks(reader, doomloop, |pending| {
        let _ = write!(stderr, "{pending}");
        let _ = stderr.flush();
    })?;
    let _ = writeln!(stderr);
    let _ = stderr.flush();
    Ok(outcome)
}

/// Parse SSE events from `reader`, accumulate the `"content"` field, and call
/// `on_flush` every 10 chunks with the pending text.  Returns the full
/// accumulated content.
///
/// If the model enters a doom loop (same substring repeating), generation
/// is stopped early and the content accumulated so far is returned.
///
/// The caller is responsible for ensuring the reader does not block
/// indefinitely (e.g. by killing the server process that feeds it).
fn read_sse_chunks(
    reader: impl BufRead,
    pipeline: &DoomloopPipeline,
    mut on_flush: impl FnMut(&str),
) -> anyhow::Result<SseOutcome> {
    let mut content = String::new();
    let mut chunks: u64 = 0;
    let mut pending = String::new();
    let mut doom_loop: Option<String> = None;
    let mut termination: Option<StreamTermination> = None;
    let events = reader
        .lines()
        .map(|line| line.context("failed to read SSE line"))
        .filter_map(|line| match line {
            Err(e) => Some(Err(e)),
            Ok(line) => line
                .strip_prefix("data: ")
                .map(str::to_owned)
                .or_else(|| {
                    line.strip_prefix("data:")
                        .map(|d| d.trim_start().to_owned())
                })
                .map(Ok),
        })
        .map(|data| {
            let data = data?;
            serde_json::from_str::<Value>(&data)
                .with_context(|| format!("failed to parse SSE data: {data}"))
        });
    for event in events {
        let event = event?;
        if let Some(chunk) = event.get("content").and_then(Value::as_str) {
            if !chunk.is_empty() {
                content.push_str(chunk);
                pending.push_str(chunk);
            }
        }
        chunks += 1;
        if chunks.is_multiple_of(10) && !pending.is_empty() {
            on_flush(&pending);
            pending.clear();
        }
        if chunks.is_multiple_of(50) {
            if let Some(name) = pipeline.check(&content) {
                if !pending.is_empty() {
                    on_flush(&pending);
                    pending.clear();
                }
                let detail = format_trigger_log(name, content.len());
                log::warn!("{detail}");
                doom_loop = Some(detail);
                break;
            }
        }
        if event.get("stop").and_then(Value::as_bool) == Some(true) {
            // Terminal chunk carries the stop metadata (stop_type /
            // stopped_limit / timings). Parse it best-effort; a shape we
            // can't deserialize just leaves the stop unclassified.
            termination = serde_json::from_value(event).ok();
            break;
        }
    }
    if !pending.is_empty() {
        on_flush(&pending);
    }
    Ok(SseOutcome {
        content,
        doom_loop,
        termination,
    })
}

/// HTTP client / readiness wait from plan `benchmark_flags`, else default.
fn request_timeout_from_req(req: &RunRequest) -> Duration {
    Duration::from_secs(
        req.benchmark_flags
            .as_ref()
            .and_then(|f| f.http_timeout())
            .unwrap_or(DEFAULT_HTTP_TIMEOUT_SECS),
    )
}

pub(crate) fn mcq_grammar(choices: &[String]) -> String {
    let alts = choices
        .iter()
        .map(|choice| format!("{choice:?}"))
        .collect::<Vec<_>>()
        .join(" | ");
    format!("root ::= {alts}")
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use rstest::rstest;

    use super::*;

    fn sse_line(content: &str, stop: bool) -> String {
        let event = json!({"content": content, "stop": stop});
        format!("data: {event}\n\n")
    }

    #[test]
    fn read_sse_chunks_accumulates_content() -> anyhow::Result<()> {
        let input = format!("{}{}", sse_line("hello ", false), sse_line("world", true));
        let reader = Cursor::new(input);
        let content = read_sse_chunks(reader, &DoomloopPipeline::disabled(), |_| {})?.content;
        assert_eq!(content, "hello world");
        Ok(())
    }

    #[test]
    fn read_sse_chunks_handles_data_without_space() -> anyhow::Result<()> {
        let input = "data:{\"content\":\"hi\",\"stop\":true}\n\n".to_string();
        let reader = Cursor::new(input);
        let content = read_sse_chunks(reader, &DoomloopPipeline::disabled(), |_| {})?.content;
        assert_eq!(content, "hi");
        Ok(())
    }

    #[test]
    fn read_sse_chunks_skips_non_data_lines() -> anyhow::Result<()> {
        let input = format!(": keepalive\nretry: 100\n{}", sse_line("ok", true));
        let reader = Cursor::new(input);
        let content = read_sse_chunks(reader, &DoomloopPipeline::disabled(), |_| {})?.content;
        assert_eq!(content, "ok");
        Ok(())
    }

    #[test]
    fn read_sse_chunks_flushes_every_10_chunks() -> anyhow::Result<()> {
        let input: String = (0..15)
            .map(|i| sse_line(&format!("{i}"), i == 14))
            .collect();
        let reader = Cursor::new(input);
        let mut flush_count = 0;
        let _ = read_sse_chunks(reader, &DoomloopPipeline::disabled(), |_| flush_count += 1)?;
        assert_eq!(flush_count, 2); // at chunk 10 and remaining 5
        Ok(())
    }

    #[test]
    fn read_sse_chunks_handles_unicode() -> anyhow::Result<()> {
        let input = format!(
            "{}{}{}",
            sse_line("こんにちは", false),
            sse_line("🌍", false),
            sse_line("", true)
        );
        let reader = Cursor::new(input);
        let content = read_sse_chunks(reader, &DoomloopPipeline::disabled(), |_| {})?.content;
        assert_eq!(content, "こんにちは🌍");
        Ok(())
    }

    #[test]
    fn read_sse_chunks_empty_stream() -> anyhow::Result<()> {
        let input = sse_line("", true);
        let reader = Cursor::new(input);
        let content = read_sse_chunks(reader, &DoomloopPipeline::disabled(), |_| {})?.content;
        assert_eq!(content, "");
        Ok(())
    }

    #[test]
    fn read_sse_chunks_stops_on_repetition_loop() -> anyhow::Result<()> {
        // Generate 200 chunks of the same repeating block — well beyond the
        // 50-chunk check interval and the min_chars threshold.
        let block = "the same text repeated over and over again. ";
        let mut input: String = (0..200).map(|_| sse_line(block, false)).collect();
        input.push_str(&sse_line("", true));
        let reader = Cursor::new(input);
        let pipeline = DoomloopPipeline {
            detectors: vec![Box::new(pipette_doomloop::ExactRepeat {
                min_chars: 100,
                window: 512,
                min_period: 32,
                required: 3,
            })],
        };
        let outcome = read_sse_chunks(reader, &pipeline, |_| {})?;
        // Should have stopped before consuming all 200 chunks, and flagged
        // the abort as a doom-loop (the only source of `doom_loop`).
        let full_len = block.len() * 200;
        assert!(
            outcome.content.len() < full_len,
            "expected early stop but got full output ({} chars)",
            outcome.content.len()
        );
        assert!(
            outcome.doom_loop.is_some(),
            "expected doom_loop to be flagged"
        );
        assert!(outcome.termination.is_none());
        Ok(())
    }

    /// Terminal SSE chunk carrying llama-server's stop metadata.
    fn sse_terminal(content: &str, stop_type: &str, predicted_n: u64) -> String {
        let stopped_limit = stop_type == "limit";
        let event = json!({
            "content": content,
            "stop": true,
            "stop_type": stop_type,
            "stopped_limit": stopped_limit,
            "timings": {"predicted_n": predicted_n},
        });
        format!("data: {event}\n\n")
    }

    /// The terminal SSE chunk parses into `termination` and classifies end-to-end.
    #[test]
    fn read_sse_chunks_parses_terminal_metadata() -> anyhow::Result<()> {
        let input = sse_terminal("hi", "limit", 8192);
        let outcome = read_sse_chunks(Cursor::new(input), &DoomloopPipeline::disabled(), |_| {})?;
        let Some(term) = &outcome.termination else {
            anyhow::bail!("expected terminal metadata");
        };
        assert!(term.stopped_limit);
        assert_eq!(term.stop_type.as_deref(), Some("limit"));
        assert_eq!(term.timings.predicted_n, Some(8192));
        assert_eq!(
            classify_stop(&outcome),
            StopClassification {
                reason: BenchmarkEvalCompletionStopReason::Truncated,
                completion_tokens: Some(8192),
                detail: None,
            }
        );
        Ok(())
    }

    /// Classification matrix over the terminal event's flags. `limit` (bool or
    /// `stop_type`) ⇒ truncated; `eos` (bool or `stop_type`) ⇒ eos; an
    /// unrecognized `stop_type` or no signal ⇒ unknown (never a guessed eos).
    #[rstest]
    #[case(true, false, None, BenchmarkEvalCompletionStopReason::Truncated, None)]
    #[case(
        false,
        false,
        Some("limit"),
        BenchmarkEvalCompletionStopReason::Truncated,
        None
    )]
    #[case(
        false,
        false,
        Some("eos"),
        BenchmarkEvalCompletionStopReason::Eos,
        None
    )]
    #[case(false, true, None, BenchmarkEvalCompletionStopReason::Eos, None)]
    #[case(
        false,
        false,
        Some("word"),
        BenchmarkEvalCompletionStopReason::Unknown,
        Some("unrecognized stop_type=word")
    )]
    #[case(
        false,
        false,
        None,
        BenchmarkEvalCompletionStopReason::Unknown,
        Some("terminal event carried no stop signal")
    )]
    fn classify_stop_from_terminal_flags(
        #[case] stopped_limit: bool,
        #[case] stopped_eos: bool,
        #[case] stop_type: Option<&str>,
        #[case] want: BenchmarkEvalCompletionStopReason,
        #[case] want_detail: Option<&str>,
    ) {
        let sse = SseOutcome {
            content: String::new(),
            doom_loop: None,
            termination: Some(StreamTermination {
                stopped_limit,
                stopped_eos,
                stop_type: stop_type.map(str::to_string),
                timings: StreamTimings {
                    predicted_n: Some(7),
                },
            }),
        };
        assert_eq!(
            classify_stop(&sse),
            StopClassification {
                reason: want,
                completion_tokens: Some(7),
                detail: want_detail.map(str::to_string),
            }
        );
    }

    #[test]
    fn doom_loop_abort_classifies_as_doom_loop() {
        let sse = SseOutcome {
            content: "loop".to_string(),
            doom_loop: Some("ExactRepeat fired".to_string()),
            termination: None,
        };
        assert_eq!(
            classify_stop(&sse),
            StopClassification {
                reason: BenchmarkEvalCompletionStopReason::DoomLoop,
                completion_tokens: None,
                detail: Some("ExactRepeat fired".to_string()),
            }
        );
    }

    #[test]
    fn stream_without_terminal_event_falls_back_to_unknown() {
        let sse = SseOutcome {
            content: "partial".to_string(),
            doom_loop: None,
            termination: None,
        };
        assert_eq!(
            classify_stop(&sse),
            StopClassification {
                reason: BenchmarkEvalCompletionStopReason::Unknown,
                completion_tokens: None,
                detail: Some("stream ended without a terminal stop event".to_string()),
            }
        );
    }

    #[test]
    fn mcq_grammar_single() {
        let grammar = mcq_grammar(&["A".to_string()]);
        assert_eq!(grammar, r#"root ::= "A""#);
    }

    #[test]
    fn mcq_grammar_multiple() {
        let grammar = mcq_grammar(&["A".to_string(), "B".to_string(), "C".to_string()]);
        assert_eq!(grammar, r#"root ::= "A" | "B" | "C""#);
    }

    #[test]
    fn request_timeout_from_req_uses_benchmark_flags_or_default() -> anyhow::Result<()> {
        let mut req = fixture_run_request()?;
        assert_eq!(
            request_timeout_from_req(&req),
            Duration::from_secs(DEFAULT_HTTP_TIMEOUT_SECS)
        );
        req.benchmark_flags = Some(
            pipette_plan_types::BenchmarkFlags::EvalLlamacppCliStockToolsGgufText {
                http_timeout_seconds: Some(120),
                doomloop: Default::default(),
            },
        );
        assert_eq!(request_timeout_from_req(&req), Duration::from_secs(120));
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn bounded_http_client_times_out_on_hanging_response() -> anyhow::Result<()> {
        use std::{net::TcpListener, thread, time::Instant};

        let listener =
            TcpListener::bind(("127.0.0.1", 0)).context("failed to bind test listener")?;
        let port = listener
            .local_addr()
            .context("failed to read listener addr")?
            .port();

        let server = thread::spawn(move || -> anyhow::Result<()> {
            let _ = listener
                .accept()
                .context("failed to accept test connection")?;
            thread::sleep(Duration::from_secs(5));
            Ok(())
        });

        let client = HttpClient::blocking_with_timeout("pipette", Duration::from_millis(200))
            .context("failed to build test client")?;
        let url = format!("http://127.0.0.1:{port}/health");
        let started = Instant::now();
        let _err = client
            .get(url)
            .send()
            .err()
            .context("request unexpectedly succeeded")?;

        assert!(started.elapsed() < Duration::from_secs(2));

        let _ = server.join();
        Ok(())
    }

    // Session helpers live on EvalCompletionsStore; digest identity is covered
    // in pipette-ops::eval_completions tests (RunRequest portable fields).

    fn fixture_run_request() -> anyhow::Result<RunRequest> {
        use pipette_plan_types::benchmark::{BenchmarkDefinition, EvalBenchmark};
        use pipette_plan_types::run::DeclaredBound;
        use pipette_plan_types::{
            GgufText, GgufTextSource, HfOrg, HfRepo, HfRepoName, LlamaCppFlavor,
            LlamacppCliStockToolsSource, Model, NonEmptyString, RepoSubpath, RepositoryUrl,
            Runtime, SourceRepository,
        };

        let rt = Runtime::LlamacppCliStockTools(pipette_plan_types::LlamacppCliStockTools {
            source: LlamacppCliStockToolsSource::GithubRelease(SourceRepository {
                repository_url: RepositoryUrl::new("github.com/ggml-org/llama.cpp"),
                repository_version: NonEmptyString::try_new("b1234".to_owned())?,
            }),
            flavor: LlamaCppFlavor::MacosArm64,
        });
        let model = Model::GgufText(GgufText {
            source: GgufTextSource::HuggingFace {
                repo: HfRepo {
                    org: HfOrg::try_new("org".to_owned())?,
                    repo_name: HfRepoName::try_new("model".to_owned())?,
                    revision: None,
                    auth_token: None,
                },
                path: RepoSubpath::try_new("model.gguf".to_owned())?,
                sha256: None,
            },
        });
        Ok(RunRequest {
            runtime: DeclaredBound::already_bound(rt),
            model: DeclaredBound::already_bound(model),
            runtime_flags: None,
            model_flags: None,
            benchmark_flags: None,
            benchmark: BenchmarkDefinition::Eval(EvalBenchmark {
                benchmark_id: "bench".into(),
                parameter_eval_id: "math_500".into(),
                parameter_dataset_name: "ds".into(),
                parameter_max_tokens: 256,
                parameter_mcq_choices: None,
                samples: Some(vec![json!({"id": "a", "messages": []})]),
            }),
        })
    }

    #[test]
    fn failed_and_completion_entries_coexist_in_session() -> anyhow::Result<()> {
        let root = tempfile::tempdir()?;
        let evals = EvalCompletionsStore::new(root.path());
        let req = fixture_run_request()?;
        {
            let mut ckpt = evals.open(&req)?;
            ckpt.append(BenchmarkEvalCompletion {
                id: "done-1".into(),
                completion: "ok".into(),
                ..Default::default()
            })?;
            ckpt.append(BenchmarkEvalCompletion {
                id: "failed-1".into(),
                completion: String::new(),
                failed: true,
                failed_reason: Some("test crash".into()),
                ..Default::default()
            })?;
        }
        let ckpt = evals.open(&req)?;
        assert!(ckpt.contains("done-1"));
        assert!(ckpt.contains("failed-1"));
        assert_eq!(ckpt.failed_count(), 1);
        assert_eq!(ckpt.failed_ids().collect::<Vec<_>>(), vec!["failed-1"]);
        Ok(())
    }

    #[test]
    fn finalize_returns_both_completed_and_failed_entries() -> anyhow::Result<()> {
        let root = tempfile::tempdir()?;
        let evals = EvalCompletionsStore::new(root.path());
        let req = fixture_run_request()?;
        let mut ckpt = evals.open(&req)?;
        ckpt.append(BenchmarkEvalCompletion {
            id: "done-1".into(),
            completion: "ok".into(),
            ..Default::default()
        })?;
        ckpt.append(BenchmarkEvalCompletion {
            id: "failed-1".into(),
            completion: String::new(),
            failed: true,
            failed_reason: Some("boom".into()),
            ..Default::default()
        })?;
        let submitted = ckpt.finalize()?;
        assert_eq!(submitted.len(), 2);
        assert!(submitted.iter().any(|c| c.id == "done-1" && !c.failed));
        assert!(submitted.iter().any(|c| c.id == "failed-1" && c.failed));
        Ok(())
    }
}
