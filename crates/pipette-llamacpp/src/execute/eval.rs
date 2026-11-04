use std::{
    io::{BufReader, Write},
    num::NonZeroUsize,
    path::Path,
    sync::{Mutex, PoisonError},
    time::Duration,
};

use anyhow::Context;
use reqwest::Method;
use reqwest::header::CONNECTION;
use serde::Deserialize;
use serde_json::{Value, json};

use pipette_http::{HttpClient, StreamClient};
use pipette_ops::eval_cell::{EvalCell, Sampling};
use pipette_ops::eval_completions::EvalCompletionSession;
use pipette_ops::measurement::{
    self, EvalMeasurementInFlight, EvalMeasurementReading, EvalMeasurementSample,
};
use pipette_ops::run_response::RunResponseBuilder;
use pipette_ops::sse::{self, Delta};
use pipette_plan_types::{
    BenchmarkEvalCompletionStopReason, BenchmarkFlagRef, EvalBenchmark,
    EvalLlamacppCliStockToolsGgufText, LlamacppCliStockToolsExtraRuntimeConfiguration, ModelFlags,
    ReasoningEffort,
};

use crate::eval_pool::{
    Server, ServerSlot, SlotNeed, check_slot_context, mcq_grammar, read_sample, slot_server,
    spawn_server,
};
use crate::models::ResolvedGgufTextModel;
use crate::runtimes::ResolvedLlamaServer;
use crate::server::{self, LoggingMarker, ServerCommandBuilder};

/// Eval through [`measurement::eval()`], resuming `checkpoint`. The dispatch
/// records the cell's flags as authored.
#[expect(
    clippy::too_many_arguments,
    reason = "the request's parts, and the checkpoint opened on its identity"
)]
pub fn run(
    benchmark: &EvalBenchmark,
    runtime_flags: EvalLlamacppCliStockToolsGgufText,
    model_flags: Option<&ModelFlags>,
    benchmark_flags: &BenchmarkFlagRef,
    program: &ResolvedLlamaServer,
    model: &ResolvedGgufTextModel,
    checkpoint: EvalCompletionSession,
    client_config: Option<&LlamacppCliStockToolsExtraRuntimeConfiguration>,
) -> anyhow::Result<RunResponseBuilder> {
    // `parallel` is the slot count llama-server is started with *and* the
    // number of samples kept in flight. A raw slot flag reaches llama-server
    // without this loop knowing to fill the slots; it is read only so the
    // context check sees the divisor argv will carry.
    let parallel = runtime_flags.parallel;
    let slots = parallel.map_or(1, |n| n.into_inner().unsigned_abs());
    let per_cell_ctx =
        8192u32.saturating_add(benchmark.parameter_max_tokens.into_inner().unsigned_abs());
    check_slot_context(
        LoggingMarker::Eval,
        slots,
        server::raw_parallel(&runtime_flags.raw).and_then(|(_, count)| count),
        runtime_flags.ctx_size,
        per_cell_ctx,
        SlotNeed::Generation {
            max_tokens: benchmark.parameter_max_tokens.into_inner().unsigned_abs(),
        },
    )?;
    // The context spans the slots unless the cell pinned a total. The weights
    // keep llama.cpp's mmap default: an eval runs long enough to amortize the
    // page faults.
    let runtime_flags = EvalLlamacppCliStockToolsGgufText {
        ctx_size: Some(match runtime_flags.ctx_size {
            Some(total) => total,
            None => server::ctx_across_slots(per_cell_ctx, slots)?,
        }),
        ..runtime_flags
    };
    let launch = launch(&runtime_flags, program, model, client_config);
    let argv = launch.args()?;

    let cell = EvalCell::new(benchmark, model_flags, parallel)?;
    let request_timeout = measurement::http_timeout_for(benchmark_flags.http_timeout_seconds, 1);

    // The first start and every restart, with the same argv: a server that
    // came back configured differently from the one that crashed would
    // silently change what the rest of the cell measures.
    let start = || spawn_server(&launch, request_timeout);
    let first = start()?;
    let http = HttpClient::loopback("pipette", request_timeout)?;
    let enable_thinking = cell.enable_thinking;
    let slot = Mutex::new(ServerSlot {
        server: Server::Up(first),
        generation: 0,
    });
    let completion_mode = CompletionMode::pick(
        cell.mcq_choices,
        u64::from(cell.max_tokens),
        cell.sampling,
        measurement::stream_idle_timeout_for(benchmark_flags.http_timeout_seconds),
        // The tee holds the stderr lock for a whole generation, which would
        // serialise a pool. A terminal only: in a log the raw tokens would split
        // the one-event-per-line stream, and the debug `sample completion text`
        // carries it.
        cell.workers <= 1 && std::io::IsTerminal::is_terminal(&std::io::stderr()),
    )?;

    let result_data = measurement::eval(
        benchmark,
        EvalMeasurementInFlight::Parallel(
            NonZeroUsize::new(cell.workers).unwrap_or(NonZeroUsize::MIN),
        ),
        checkpoint,
        &slot,
        |slot, samples, record| {
            samples.iter().enumerate().try_for_each(|(at, sample)| {
                let messages = sample.messages.raw()?;
                let Some((base_url, generation)) = slot_server(slot, &start)? else {
                    return record(
                        at,
                        EvalMeasurementReading::Lost {
                            reason: "llama-server did not come back after a crash".to_owned(),
                        },
                    );
                };
                let reading = read_sample(slot, generation, sample, &argv, "/completion", || {
                    completion_mode.run_sample(
                        &http,
                        &base_url,
                        sample,
                        messages,
                        enable_thinking,
                        cell.reasoning_effort,
                    )
                })?;
                record(at, reading)
            })?;
            Ok(())
        },
    )?;
    // The latest server's argv and log: after a mid-run restart the argv
    // differs only in `--port`. No memory observation, deliberately: a server
    // started after a crash would have watched only the tail of the run.
    let ServerSlot { server, .. } = slot.into_inner().unwrap_or_else(PoisonError::into_inner);
    let response = RunResponseBuilder::new(result_data);
    Ok(match server {
        Server::Up((mut llama_server, _)) | Server::Down(mut llama_server) => {
            response.process(&mut llama_server)
        }
        Server::RestartFailed => response,
    })
}

fn fetch_prompt(
    http: &HttpClient,
    base_url: &str,
    messages: &Value,
    enable_thinking: Option<bool>,
    reasoning_effort: Option<ReasoningEffort>,
) -> anyhow::Result<String> {
    server::apply_template(
        http,
        base_url,
        apply_template_body(messages, enable_thinking, reasoning_effort),
    )
}

/// The `/apply-template` body for `messages` under the cell's model flags,
/// each set only when the flag is, so a cell that sets neither renders the
/// prompt from `messages` alone.
///
/// - `enable_thinking` goes in `chat_template_kwargs`, which `/apply-template`
///   honours on b9119+ whether or not the server was started with `--jinja`.
/// - `reasoning_effort` goes in `chat_template_kwargs` too, which every build
///   this client runs hands the template as a variable: gpt-oss renders it as
///   its `Reasoning:` line. It also goes top-level, the OpenAI spelling. b11349
///   writes that into the same variable; the LFM3 fork (e749ac1) reads it only
///   for `none`, which the flag cannot say, so there the kwarg alone carries it.
fn apply_template_body(
    messages: &Value,
    enable_thinking: Option<bool>,
    reasoning_effort: Option<ReasoningEffort>,
) -> Value {
    let mut kwargs = serde_json::Map::new();
    if let Some(enable_thinking) = enable_thinking {
        kwargs.insert("enable_thinking".to_string(), json!(enable_thinking));
    }
    if let Some(reasoning_effort) = reasoning_effort {
        kwargs.insert("reasoning_effort".to_string(), json!(reasoning_effort));
    }
    let mut body = serde_json::Map::new();
    body.insert("messages".to_string(), messages.clone());
    if !kwargs.is_empty() {
        body.insert("chat_template_kwargs".to_string(), Value::Object(kwargs));
    }
    if let Some(reasoning_effort) = reasoning_effort {
        body.insert("reasoning_effort".to_string(), json!(reasoning_effort));
    }
    Value::Object(body)
}

/// llama-server's final streaming chunk (`stop: true`). Both the boolean flags
/// (older builds) and `stop_type` (newer builds) are read.
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

struct SseOutcome {
    content: String,
    termination: Option<StreamTermination>,
}

/// `detail` records the raw why of a non-clean stop, so an `unknown` is never a
/// dead end when debugging.
#[derive(Debug, PartialEq, Eq)]
struct StopClassification {
    reason: BenchmarkEvalCompletionStopReason,
    completion_tokens: Option<u64>,
    detail: Option<String>,
}

/// Anything but `limit` or `eos`, such as `stop_type: "word"`, is `unknown`
/// rather than a guessed `eos`.
fn classify_stop(sse: &SseOutcome) -> StopClassification {
    // A stream that ends before its stop chunk fails the read, so the only
    // way here is a stop chunk whose shape did not deserialize.
    let Some(term) = &sse.termination else {
        return StopClassification {
            reason: BenchmarkEvalCompletionStopReason::Unknown,
            completion_tokens: None,
            detail: Some("terminal stop event did not carry readable stop metadata".to_string()),
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

/// Control tokens every `/completion` asks to have rendered as text, which
/// `llama-server` otherwise drops. North-Mini-Code (Cohere) brackets its
/// reasoning and its answer with them, and the GGUF converter makes every
/// `<|…|>` token a control token, so without them the scorer gets reasoning
/// and answer with no boundary between. vLLM already keeps them, since the HF
/// tokenizer marks them non-special, so the text matches across runtimes. A
/// string that is not one token of the model's vocabulary is skipped, so every
/// model can carry the list: it changes what is rendered, not what is sampled
/// or where generation stops.
const PRESERVED_TOKENS: [&str; 4] = [
    "<|START_THINKING|>",
    "<|END_THINKING|>",
    "<|START_TEXT|>",
    "<|END_TEXT|>",
];

/// How a sample's `/completion` is fetched. MCQ and streaming differ in client,
/// body and parser together, and a mismatch (a bounded client with an SSE body)
/// would silently break, so one enum holds each set.
enum CompletionMode<'a> {
    /// A grammar-constrained single token, as one JSON response on the bounded
    /// client: a streaming client could hang on a wedged server.
    Mcq {
        grammar: String,
        sampling: Sampling<'a>,
    },
    /// The [`StreamClient`] has a connect-only HTTP timeout, so the long-lived
    /// response is not killed mid-stream, and the idle bound the reader fails on.
    Streaming {
        client: StreamClient,
        max_tokens: u64,
        sampling: Sampling<'a>,
        /// Mirror tokens to stderr as they arrive. Off for a concurrent pool:
        /// the tee holds the stderr lock for a whole generation.
        tee_stderr: bool,
    },
}

impl<'a> CompletionMode<'a> {
    fn pick(
        mcq_choices: Option<&[String]>,
        max_tokens: u64,
        sampling: Sampling<'a>,
        idle_timeout: Duration,
        tee_stderr: bool,
    ) -> anyhow::Result<Self> {
        match mcq_choices {
            Some(choices) => Ok(Self::Mcq {
                grammar: mcq_grammar(choices),
                sampling,
            }),
            None => Ok(Self::Streaming {
                client: streaming_http_client(idle_timeout)?,
                max_tokens,
                sampling,
                tee_stderr,
            }),
        }
    }

    fn run_sample(
        &self,
        http: &HttpClient,
        base_url: &str,
        sample: &EvalMeasurementSample<'_>,
        messages: &Value,
        enable_thinking: Option<bool>,
        reasoning_effort: Option<ReasoningEffort>,
    ) -> anyhow::Result<EvalMeasurementReading> {
        let prompt = fetch_prompt(http, base_url, messages, enable_thinking, reasoning_effort)?;
        // The texts are in the eval checkpoint and the result already; at info
        // they made up nearly all of a pool job's log.
        tracing::debug!(
            sample = sample.position.as_str(),
            sample_id = %sample.id,
            prompt_text = prompt.as_str(),
            "rendered prompt"
        );
        let url = format!("{base_url}/completion");
        let reading = match self {
            Self::Mcq { grammar, sampling } => {
                // The seed is the sample's own, off the `#k` repeat index: the
                // base seed on every repeat would collapse pass@1 to one shot.
                let mut body = json!({
                    "prompt": prompt,
                    "n_predict": 1,
                    "grammar": grammar,
                    "preserved_tokens": PRESERVED_TOKENS,
                });
                sampling.apply(&mut body, sample.id)?;
                let completion: Value = http
                    .json_request(Method::POST, &url, None, Some(body))
                    .context("POST /completion")?;
                // The grammar forces one token, so a missing `content` is a
                // server bug: an empty string would score as a wrong answer.
                let content = completion
                    .get("content")
                    .and_then(Value::as_str)
                    .context("/completion response missing string `content` field")?
                    .to_string();
                // One forced token has no meaningful eos or truncated.
                EvalMeasurementReading::Completed {
                    completion: content,
                    stop_reason: BenchmarkEvalCompletionStopReason::Unknown,
                    stop_detail: Some("mcq arm (n_predict:1, grammar-constrained)".to_string()),
                    completion_tokens: None,
                }
            }
            Self::Streaming {
                client,
                max_tokens,
                sampling,
                tee_stderr,
            } => {
                let mut body = json!({
                    "prompt": prompt,
                    "n_predict": max_tokens,
                    "stream": true,
                    "preserved_tokens": PRESERVED_TOKENS,
                });
                sampling.apply(&mut body, sample.id)?;
                let sse = stream_completion(client, &url, &body, *tee_stderr)?;
                let stop = classify_stop(&sse);
                EvalMeasurementReading::Completed {
                    completion: sse.content,
                    stop_reason: stop.reason,
                    stop_detail: stop.detail,
                    completion_tokens: stop.completion_tokens,
                }
            }
        };
        if let EvalMeasurementReading::Completed {
            completion,
            stop_reason,
            stop_detail,
            completion_tokens,
        } = &reading
        {
            tracing::info!(
                sample = sample.position.as_str(),
                sample_id = %sample.id,
                completion_chars = completion.chars().count(),
                ?stop_reason,
                completion_tokens,
                stop_detail = stop_detail.as_deref(),
                "sample completed"
            );
            tracing::debug!(
                sample_id = %sample.id,
                completion = completion.as_str(),
                "sample completion text"
            );
        }
        Ok(reading)
    }
}

/// No overall response timeout, which would kill the long-lived SSE connection:
/// a wedged server is caught by the reader's idle bound.
fn streaming_http_client(idle_timeout: Duration) -> anyhow::Result<StreamClient> {
    HttpClient::streaming("pipette", idle_timeout).context("failed to build streaming HTTP client")
}

/// The stderr lock is a LEAF and must stay one: it is held for the whole
/// generation, and it blocks across threads, so anything that acquired a
/// pool lock first would stall every worker for the length of a completion.
fn stream_completion(
    client: &StreamClient,
    url: &str,
    body: &Value,
    tee_stderr: bool,
) -> anyhow::Result<SseOutcome> {
    let request = client
        .http()
        .post(url)
        // llama-server closes a streamed reply's connection although its headers keep it alive.
        .header(CONNECTION, "close")
        .json(body);
    let response = pipette_http::send(request).context("failed to call /completion")?;
    let events = sse::data_lines(BufReader::new(response), client.idle_timeout());
    if !tee_stderr {
        // Taking the lock lazily inside the closure would bring back the hold.
        return read_sse_chunks(events, |_| {});
    }
    let mut stderr = std::io::stderr().lock();
    let outcome = read_sse_chunks(events, |pending| {
        let _ = write!(stderr, "{pending}");
        let _ = stderr.flush();
    })?;
    let _ = writeln!(stderr);
    let _ = stderr.flush();
    Ok(outcome)
}

fn completion_delta(event: &Value) -> Delta<'_> {
    Delta {
        text: event.get("content").and_then(Value::as_str).unwrap_or(""),
        terminal: event.get("stop").and_then(Value::as_bool) == Some(true),
    }
}

/// A terminal chunk that does not deserialize is logged and leaves the stop
/// unclassified rather than failing the sample.
fn read_sse_chunks(
    events: impl IntoIterator<Item = anyhow::Result<String>>,
    mut on_flush: impl FnMut(&str),
) -> anyhow::Result<SseOutcome> {
    let stream = sse::read_text_stream(events, &completion_delta, &mut on_flush)?;
    Ok(SseOutcome {
        content: stream.content,
        termination: stream.terminal.and_then(|event| {
            serde_json::from_value(event)
                .inspect_err(
                    |err| tracing::warn!(error = %err, "terminal chunk did not deserialize"),
                )
                .ok()
        }),
    })
}

/// The cell's server, every field named: a flag plan-types adds fails to compile
/// here until this cell renders it or sets it aside.
fn launch(
    flags: &EvalLlamacppCliStockToolsGgufText,
    program: &ResolvedLlamaServer,
    model: &Path,
    client_config: Option<&LlamacppCliStockToolsExtraRuntimeConfiguration>,
) -> ServerCommandBuilder {
    let EvalLlamacppCliStockToolsGgufText {
        threads,
        number_gpu_layers,
        mmap,
        load_mode,
        flash_attention,
        ctx_size,
        no_cache,
        parallel,
        envs,
        raw,
    } = flags.clone();
    ServerCommandBuilder::new(program.clone(), model, LoggingMarker::Eval)
        .threads(threads)
        .number_gpu_layers(number_gpu_layers)
        .mmap(mmap)
        .load_mode(load_mode)
        .flash_attention(flash_attention)
        .raw(raw)
        .ctx_size(ctx_size)
        .no_cache(no_cache)
        .slots(parallel.map(|n| n.into_inner().unsigned_abs()))
        .envs(envs)
        .client_config(client_config.cloned())
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use rstest::rstest;

    use pipette_plan_types::{
        BenchmarkId, LlamacppFlashAttention, LlamacppLoadMode, PositiveCount, PositiveSeconds,
        RunRequest,
    };
    use pipette_subprocess::start_process;

    use super::*;
    use crate::eval_pool::{
        CRASH_OBSERVATION_WINDOW, LlamaServer, allocation_failure_error, is_allocation_failure,
        is_dropped_connection, poll_until_arbitrated,
    };
    use crate::runtimes::test_llama_server;

    fn server_args(
        flags: &EvalLlamacppCliStockToolsGgufText,
    ) -> Result<Vec<String>, server::ServerArgsError> {
        launch(flags, &test_llama_server(), Path::new("model.gguf"), None).args()
    }

    /// Every knob set to its own value, so a knob that lands on another's flag shows.
    fn every_knob_set() -> anyhow::Result<EvalLlamacppCliStockToolsGgufText> {
        Ok(EvalLlamacppCliStockToolsGgufText {
            threads: Some(3),
            number_gpu_layers: Some(7),
            mmap: None,
            load_mode: Some(LlamacppLoadMode::DirectIo),
            flash_attention: Some(LlamacppFlashAttention::On),
            ctx_size: Some(4096),
            no_cache: Some(true),
            parallel: Some(PositiveCount::try_from(2)?),
            envs: Vec::new(),
            raw: vec!["--poll".to_owned(), "0".to_owned()],
        })
    }

    #[test]
    fn the_server_takes_every_knob_and_the_slots() -> anyhow::Result<()> {
        assert_eq!(
            server_args(&every_knob_set()?)?.join(" "),
            "--ctx-checkpoints 0 --ctx-size 4096 --flash-attn on --load-mode dio --n-gpu-layers 7 --no-cache-prompt --no-warmup --parallel 2 --poll 0 --threads 3"
        );
        Ok(())
    }

    #[rstest]
    #[case::allocation_assert(
        "ggml/src/ggml-backend.cpp:348: GGML_ASSERT(tensor->data != NULL && \"tensor not allocated\") failed",
        true
    )]
    #[case::ordinary_crash(
        "terminate called after throwing an instance of 'std::runtime_error'",
        false
    )]
    #[case::quiet_exit("", false)]
    fn allocation_failures_are_told_apart_from_ordinary_crashes(
        #[case] stderr: &str,
        #[case] expected: bool,
    ) {
        assert_eq!(is_allocation_failure(stderr), expected);
    }

    #[test]
    fn allocation_failure_error_names_the_knobs_and_the_flags() -> anyhow::Result<()> {
        let flags = vec!["--ctx-size".to_string(), "1048576".to_string()];
        let msg = allocation_failure_error(&flags, "…tensor not allocated…").to_string();
        let missing = ["parallel", "ctx_size", "--ctx-size 1048576"]
            .into_iter()
            .find(|needle| !msg.contains(needle));
        anyhow::ensure!(missing.is_none(), "{missing:?} missing from: {msg}");
        Ok(())
    }

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

    #[test]
    fn read_sse_chunks_parses_terminal_metadata() -> anyhow::Result<()> {
        let input = sse_terminal("hi", "limit", 8192);
        let outcome = read_sse_chunks(
            sse::data_lines(Cursor::new(input), Duration::from_secs(60)),
            |_| {},
        )?;
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
    fn unreadable_terminal_event_falls_back_to_unknown() {
        let sse = SseOutcome {
            content: "partial".to_string(),
            termination: None,
        };
        assert_eq!(
            classify_stop(&sse),
            StopClassification {
                reason: BenchmarkEvalCompletionStopReason::Unknown,
                completion_tokens: None,
                detail: Some(
                    "terminal stop event did not carry readable stop metadata".to_string()
                ),
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

    #[rstest]
    #[case::no_flags(None, None, json!({"messages": [{"role": "user", "content": "q"}]}))]
    #[case::thinking_alone(
        Some(false),
        None,
        json!({
            "messages": [{"role": "user", "content": "q"}],
            "chat_template_kwargs": {"enable_thinking": false},
        })
    )]
    #[case::effort_alone(
        None,
        Some(ReasoningEffort::Low),
        json!({
            "messages": [{"role": "user", "content": "q"}],
            "chat_template_kwargs": {"reasoning_effort": "low"},
            "reasoning_effort": "low",
        })
    )]
    #[case::both(
        Some(true),
        Some(ReasoningEffort::High),
        json!({
            "messages": [{"role": "user", "content": "q"}],
            "chat_template_kwargs": {"enable_thinking": true, "reasoning_effort": "high"},
            "reasoning_effort": "high",
        })
    )]
    fn the_template_body_carries_each_flag_set(
        #[case] enable_thinking: Option<bool>,
        #[case] reasoning_effort: Option<ReasoningEffort>,
        #[case] expected: Value,
    ) {
        let messages = json!([{"role": "user", "content": "q"}]);
        assert_eq!(
            apply_template_body(&messages, enable_thinking, reasoning_effort),
            expected
        );
    }

    #[test]
    fn request_timeout_from_req_uses_benchmark_flags_or_default() -> anyhow::Result<()> {
        let mut req = fixture_run_request()?;
        assert_eq!(
            pipette_ops::measurement::http_timeout(&req, 1),
            Duration::from_secs(pipette_ops::measurement::DEFAULT_HTTP_TIMEOUT_SECS)
        );
        req.benchmark_flags = Some(
            pipette_plan_types::BenchmarkFlags::EvalLlamacppCliStockToolsGgufText {
                http_timeout_seconds: Some(PositiveSeconds::try_new(120)?),
            },
        );
        assert_eq!(
            pipette_ops::measurement::http_timeout(&req, 1),
            Duration::from_secs(120)
        );
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

        let client = HttpClient::loopback("pipette", Duration::from_millis(200))
            .context("failed to build test client")?
            .client()
            .clone();
        let url = format!("http://127.0.0.1:{port}/health");
        let started = Instant::now();
        let _err = pipette_http::send(client.get(url))
            .err()
            .context("request unexpectedly succeeded")?;

        assert!(started.elapsed() < Duration::from_secs(2));

        let _ = server.join();
        Ok(())
    }

    /// What the test server does with the one request it reads.
    enum Peer {
        Closes,
        Answers500,
        Hangs,
        Absent,
    }

    #[rstest]
    #[case::closed_with_no_response(Peer::Closes, true)]
    #[case::an_http_error(Peer::Answers500, false)]
    #[case::a_timeout(Peer::Hangs, false)]
    #[case::a_refused_connect(Peer::Absent, false)]
    fn only_a_dropped_connection_is_resent(
        #[case] peer: Peer,
        #[case] resent: bool,
    ) -> anyhow::Result<()> {
        use std::{io::Read, net::TcpListener, thread};

        let listener =
            TcpListener::bind(("127.0.0.1", 0)).context("failed to bind test listener")?;
        let url = format!("http://{}/completion", listener.local_addr()?);
        let server = match peer {
            Peer::Absent => {
                drop(listener);
                None
            }
            peer @ (Peer::Closes | Peer::Answers500 | Peer::Hangs) => {
                Some(thread::spawn(move || -> std::io::Result<()> {
                    let (mut stream, _) = listener.accept()?;
                    let _ = stream.read(&mut [0; 4096]);
                    match peer {
                        Peer::Answers500 => stream.write_all(
                            b"HTTP/1.1 500 Internal Server Error\r\n\
                              Content-Length: 0\r\nConnection: close\r\n\r\n",
                        ),
                        Peer::Hangs => {
                            thread::sleep(Duration::from_secs(1));
                            Ok(())
                        }
                        Peer::Closes | Peer::Absent => Ok(()),
                    }
                }))
            }
        };

        let err = HttpClient::loopback("pipette", Duration::from_millis(200))?
            .json_request::<Value>(Method::POST, &url, None, Some(json!({})))
            .context("POST /completion")
            .err()
            .context("request unexpectedly succeeded")?;
        if let Some(server) = server {
            let _ = server.join();
        }

        anyhow::ensure!(is_dropped_connection(&err) == resent, "{err:#}");
        Ok(())
    }

    fn fixture_run_request() -> anyhow::Result<RunRequest> {
        use pipette_plan_types::{
            BenchmarkDefinition, DeclaredBound, EvalBenchmark, GgufText, GgufTextSource, HfOrg,
            HfRepo, HfRepoName, LlamaCppFlavor, LlamacppCliStockToolsRelease,
            LlamacppCliStockToolsSource, Model, NonEmptyString, RepoSubpath, RepositoryUrl,
            Runtime,
        };

        let rt = Runtime::LlamacppCliStockTools(pipette_plan_types::LlamacppCliStockTools {
            source: LlamacppCliStockToolsSource::GithubRelease(LlamacppCliStockToolsRelease {
                repository_url: RepositoryUrl::new("github.com/ggml-org/llama.cpp"),
                repository_version: NonEmptyString::try_new("b1234".to_owned())?,
                asset: None,
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
                benchmark_id: BenchmarkId::try_new("bench".to_owned())?,
                parameter_eval_id: "math_500".into(),
                parameter_dataset_name: "ds".into(),
                parameter_max_tokens: PositiveCount::try_new(256)?,
                parameter_max_context: None,
                parameter_mcq_choices: None,
                parameter_temperature: None,
                parameter_top_p: None,
                parameter_top_k: None,
                parameter_min_p: None,
                parameter_seed: None,
                samples: Some(vec![json!({"id": "a", "messages": []})]),
            }),
        })
    }

    /// The first pass that can answer ends the window, a fatal one ends it where
    /// it failed, and each pass that cannot answer yet costs one poll interval.
    #[rstest]
    #[case::answered_on_the_first_pass(1, true, 0)]
    // The Windows case the window exists for: the exit takes a few hundred
    // milliseconds to surface to the parent.
    #[case::the_exit_surfaces_on_the_fourth_pass(4, true, 3)]
    // Another pass would bury the allocation error behind whatever that one decided.
    #[case::a_fatal_pass_ends_it(1, false, 0)]
    fn the_crash_window(
        #[case] passes: usize,
        #[case] answers: bool,
        #[case] want_sleeps: usize,
    ) -> anyhow::Result<()> {
        // Not `CRASH_POLL_INTERVAL`: a test that hands the loop the same
        // constant it would read cannot tell an injected interval from an
        // ignored one.
        const INTERVAL: Duration = Duration::from_millis(7);
        let slept = std::cell::RefCell::new(Vec::new());
        let sleep = |d: Duration| slept.borrow_mut().push(d);
        let mut deadlines = Vec::new();

        let outcome =
            poll_until_arbitrated(CRASH_OBSERVATION_WINDOW, INTERVAL, &sleep, |deadline| {
                deadlines.push(deadline);
                anyhow::ensure!(answers, "llama-server could not allocate");
                Ok((deadlines.len() >= passes).then_some("answered"))
            });

        anyhow::ensure!(outcome.is_ok() == answers, "{outcome:?}");
        anyhow::ensure!(deadlines.len() == passes, "{} passes", deadlines.len());
        anyhow::ensure!(
            *slept.borrow() == vec![INTERVAL; want_sleeps],
            "slept {:?}",
            slept.borrow(),
        );
        anyhow::ensure!(
            deadlines.windows(2).all(|pair| pair[0] == pair[1]),
            "the window moved between passes: {deadlines:?}",
        );
        Ok(())
    }

    /// The samples queued behind it get `None` instead of each starting a server
    /// and waiting out its readiness.
    #[test]
    fn a_failed_restart_is_tried_once() -> anyhow::Result<()> {
        let mut command = std::process::Command::new(if cfg!(windows) { "cmd" } else { "sh" });
        command.args(if cfg!(windows) {
            ["/C", "exit 0"]
        } else {
            ["-c", "exit 0"]
        });
        let mut exited = start_process("llama-server", command, Duration::ZERO)?;
        exited.wait(Duration::from_secs(10))?;
        let slot = Mutex::new(ServerSlot {
            server: Server::Down(exited),
            generation: 1,
        });
        let starts = std::cell::Cell::new(0);
        let start = || -> anyhow::Result<LlamaServer> {
            starts.set(starts.get() + 1);
            anyhow::bail!("llama-server exited during load")
        };

        let got: Vec<_> = (0..4)
            .map(|_| slot_server(&slot, &start).map(|server| server.is_some()))
            .collect();

        anyhow::ensure!(starts.get() == 1, "{} starts: {got:?}", starts.get());
        anyhow::ensure!(got.first().is_some_and(Result::is_err), "{got:?}");
        anyhow::ensure!(
            got.iter().skip(1).all(|got| matches!(got, Ok(false))),
            "{got:?}"
        );
        Ok(())
    }
}
