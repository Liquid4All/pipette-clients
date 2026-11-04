//! Eval against `mlx_lm.server`'s `/v1/chat/completions`, through
//! [`measurement::eval()`]. The server applies the chat template. A failed
//! sample is a failed row and the cell goes on; a server that went away under
//! one is started again for the next, until [`LOST_IN_A_ROW`] samples in a row
//! have lost one.

use std::io::BufReader;
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::sync::{Mutex, PoisonError};
use std::time::Duration;

use anyhow::Context;
use serde::Serialize;

use pipette_http::{HttpClient, StreamClient};
use pipette_ops::eval_cell::{EvalCell, Sampling};
use pipette_ops::eval_completions::EvalCompletionSession;
use pipette_ops::measurement::{
    self, EvalMeasurementInFlight, EvalMeasurementReading, EvalMeasurementSample,
};
use pipette_ops::run_response::RunResponseBuilder;
use pipette_ops::sse;
use pipette_plan_types::{
    BenchmarkResultData, EvalBenchmark, MinP, MlxMacosPipetteExtraRuntimeConfiguration,
    RunResponse, TopK, TopP,
};
use pipette_subprocess::{Process, start_process};

use crate::models::ResolvedMlxModel;
use crate::runtimes::ResolvedMlxPython;
use crate::{completions, server};

/// How long a failed request waits for the server to surface as exited: the
/// connection can drop before the exit is reaped.
const EXIT_WINDOW: Duration = Duration::from_secs(3);

const FAILURE_TAIL_LINES: usize = 20;

/// Samples in a row that lose their server before the cell ends: past a poison
/// prompt or two, a server lost that often points at the model or the host.
const LOST_IN_A_ROW: usize = 3;

/// The server the samples go to, and its URL, which a restart changes.
type MlxServer = (Process, String);

pub(super) fn run(
    benchmark: &EvalBenchmark,
    cell: &EvalCell<'_>,
    idle_timeout: Duration,
    checkpoint: EvalCompletionSession,
    venv_python: &ResolvedMlxPython,
    model_dir: &ResolvedMlxModel,
    client_config: Option<&MlxMacosPipetteExtraRuntimeConfiguration>,
) -> anyhow::Result<RunResponse> {
    let client = HttpClient::streaming("pipette", idle_timeout)?;
    let start = || start_server(venv_python, model_dir, client_config);
    let server = Mutex::new(start()?);
    tracing::info!(timeout = ?idle_timeout, "eval SSE idle timeout");
    tracing::info!(enable_thinking = cell.enable_thinking, "eval thinking mode");
    let result = eval(benchmark, cell, &client, checkpoint, &server, &start);
    let (mut mlx_server, _) = server.into_inner().unwrap_or_else(PoisonError::into_inner);
    // After a restart this is the replacement: its argv differs only in
    // `--port`, and its log starts at its own launch. No memory observation:
    // a replacement would have watched only the tail of the run.
    let result_data = result.map_err(|err| err.context(mlx_server.stop_with_tail()))?;
    Ok(RunResponseBuilder::new(result_data)
        .process(&mut mlx_server)
        .build())
}

fn start_server(
    venv_python: &ResolvedMlxPython,
    model_dir: &ResolvedMlxModel,
    client_config: Option<&MlxMacosPipetteExtraRuntimeConfiguration>,
) -> anyhow::Result<MlxServer> {
    let port = pipette_subprocess::pick_server_port()?;
    let url = server::url(port);
    let mut mlx_server = start_process(
        server::TEXT_SERVER_MODULE,
        server::text_command(venv_python, model_dir, port, client_config)?,
        server::STOP_GRACE,
    )?;
    mlx_server.wait_ready(server::READY_TIMEOUT, server::ready(&url)?)?;
    Ok((mlx_server, url))
}

/// `start` replaces a server that went away.
fn eval(
    benchmark: &EvalBenchmark,
    cell: &EvalCell<'_>,
    client: &StreamClient,
    checkpoint: EvalCompletionSession,
    server: &Mutex<MlxServer>,
    start: &(dyn Fn() -> anyhow::Result<MlxServer> + Sync),
) -> anyhow::Result<BenchmarkResultData> {
    let lost_in_a_row = AtomicUsize::new(0);
    measurement::eval(
        benchmark,
        // `mlx_lm.server` serves one request at a time.
        EvalMeasurementInFlight::Parallel(NonZeroUsize::MIN),
        checkpoint,
        server,
        |server, samples, record| {
            samples.iter().enumerate().try_for_each(|(at, sample)| {
                let reading = read_sample(
                    &mut server.lock().unwrap_or_else(PoisonError::into_inner),
                    start,
                    cell,
                    client,
                    sample,
                )?;
                let lost = matches!(reading, EvalMeasurementReading::Lost { .. });
                record(at, reading)?;
                let streak = if lost {
                    lost_in_a_row.fetch_add(1, Ordering::Relaxed) + 1
                } else {
                    lost_in_a_row.store(0, Ordering::Relaxed);
                    0
                };
                anyhow::ensure!(
                    streak < LOST_IN_A_ROW,
                    "mlx_lm.server went away under {streak} samples in a row"
                );
                Ok(())
            })?;
            Ok(())
        },
    )
}

/// A server that went away under an earlier sample is started again here,
/// once that sample's row is on disk.
fn read_sample(
    server: &mut MlxServer,
    start: &(dyn Fn() -> anyhow::Result<MlxServer> + Sync),
    cell: &EvalCell<'_>,
    client: &StreamClient,
    sample: &EvalMeasurementSample<'_>,
) -> anyhow::Result<EvalMeasurementReading> {
    if !server.0.is_running() {
        tracing::info!("restarting mlx_lm.server after it went away");
        *server = start().context("mlx_lm.server did not come back after it went away")?;
    }
    let (mlx_server, url) = server;
    tracing::info!(
        sample = sample.position.as_str(),
        sample_id = %sample.id,
        "sample started"
    );
    let request = chat_request(
        sample.messages.raw()?.clone(),
        cell.max_tokens,
        &cell.sampling,
        sample.id,
        cell.enable_thinking,
    );
    let reply = match chat_stream_lines(client, url, &request)
        .and_then(|events| sse::read_chat_stream(events, &mut |_| {}))
    {
        Ok(reply) => reply,
        Err(err) => return failure_reading(mlx_server, &err, sample),
    };
    let (stop_reason, stop_detail) = reply.stop();
    tracing::info!(
        sample = sample.position.as_str(),
        sample_id = %sample.id,
        completion = reply.content.as_str(),
        ?stop_reason,
        completion_tokens = reply.completion_tokens,
        stop_detail = stop_detail.as_deref(),
        "sample completed"
    );
    Ok(EvalMeasurementReading::Completed {
        completion: reply.content,
        stop_reason,
        stop_detail,
        completion_tokens: reply.completion_tokens,
    })
}

/// What a failed request comes to. A refusal is the server answering, so the
/// sample failed on its own. A server that exited, or went silent and is
/// stopped here, is gone: the sample is `Lost`, and the next one starts another.
fn failure_reading(
    mlx_server: &mut Process,
    err: &anyhow::Error,
    sample: &EvalMeasurementSample<'_>,
) -> anyhow::Result<EvalMeasurementReading> {
    let reading = if refused(err) {
        EvalMeasurementReading::Failed {
            reason: format!("mlx_lm.server refused the request: {err:#}"),
        }
    } else if went_silent(err) {
        // Up, but no longer generating: its generation thread can die on its
        // own, and every later request would wait out the bound.
        mlx_server.stop();
        EvalMeasurementReading::Lost {
            reason: format!("mlx_lm.server went silent and was stopped: {err:#}"),
        }
    } else if let Some(status) = mlx_server.wait(EXIT_WINDOW)? {
        EvalMeasurementReading::Lost {
            reason: format!("mlx_lm.server exited mid-completion ({status}): {err:#}"),
        }
    } else {
        EvalMeasurementReading::Failed {
            reason: format!("mlx_lm.server stayed up: {err:#}"),
        }
    };
    // The entry logs a lost sample itself.
    if let EvalMeasurementReading::Failed { reason } = &reading {
        tracing::warn!(
            sample = sample.position.as_str(),
            sample_id = %sample.id,
            reason = reason.as_str(),
            "sample failed"
        );
    }
    if let Some(stderr) = mlx_server.output().runtime_stderr.filter(|s| !s.is_empty()) {
        tracing::warn!(
            stderr_tail = measurement::tail_lines(&stderr, FAILURE_TAIL_LINES).as_str(),
            "mlx_lm.server stderr tail"
        );
    }
    Ok(reading)
}

fn refused(err: &anyhow::Error) -> bool {
    matches!(
        err.downcast_ref::<pipette_http::Error>(),
        Some(pipette_http::Error::Status { .. })
    )
}

/// Whether the response or its stream outlived the idle bound
/// ([`chat_stream_lines`], [`sse::data_lines`]).
fn went_silent(err: &anyhow::Error) -> bool {
    err.chain().any(|cause| {
        cause
            .downcast_ref::<std::io::Error>()
            .is_some_and(|io| io.kind() == std::io::ErrorKind::TimedOut)
    })
}

#[derive(Debug, Serialize)]
struct ChatRequest {
    model: String,
    messages: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    /// The cuts the benchmark pinned. Absent stays absent: `mlx_lm.server`
    /// falls back to its own defaults only for a field the body omits.
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_k: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    min_p: Option<f64>,
    /// The sample's own seed ([`Sampling::seed_for`]); absent unless the entry
    /// pinned one.
    #[serde(skip_serializing_if = "Option::is_none")]
    seed: Option<i64>,
    /// Forwarded to the server's `apply_chat_template`. Absent when unset, so
    /// the server keeps its tokenizer-derived default.
    #[serde(skip_serializing_if = "Option::is_none")]
    chat_template_kwargs: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
    /// `include_usage` adds the trailer that counts the generated tokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    stream_options: Option<serde_json::Value>,
}

fn chat_request(
    messages: serde_json::Value,
    max_tokens: u32,
    sampling: &Sampling,
    sample_id: &str,
    enable_thinking: Option<bool>,
) -> ChatRequest {
    ChatRequest {
        model: completions::DEFAULT_MODEL.to_string(),
        messages,
        max_tokens: Some(max_tokens),
        temperature: Some(sampling.temperature.as_f64()),
        top_p: sampling.top_p.map(TopP::as_f64),
        top_k: sampling.top_k.map(TopK::into_inner),
        min_p: sampling.min_p.map(MinP::as_f64),
        seed: sampling.seed_for(sample_id),
        chat_template_kwargs: enable_thinking
            .map(|enable| serde_json::json!({ "enable_thinking": enable })),
        stream: Some(true),
        stream_options: Some(serde_json::json!({ "include_usage": true })),
    }
}

/// One streamed `/v1/chat/completions` call's `data:` payloads. Dropping the
/// iterator drops the connection, which is the whole cancellation: the server
/// holds no per-request state worth telling. The idle bound covers the wait
/// for the response too: `mlx_lm.server` answers once its generation thread
/// takes the request, which a dead thread never does.
fn chat_stream_lines(
    client: &StreamClient,
    base_url: &str,
    request: &ChatRequest,
) -> anyhow::Result<impl Iterator<Item = anyhow::Result<String>>> {
    let post = client
        .http()
        .post(format!("{base_url}/v1/chat/completions"))
        .json(request);
    let (sent_tx, sent) = mpsc::channel();
    // Detached: on a timeout the caller stops the server, which ends the send.
    std::thread::spawn(move || {
        let _ = sent_tx.send(pipette_http::send(post));
    });
    let response = match sent.recv_timeout(client.idle_timeout()) {
        Ok(response) => response.context("failed to call /v1/chat/completions")?,
        Err(RecvTimeoutError::Timeout) => {
            return Err(std::io::Error::new(
                std::io::ErrorKind::TimedOut,
                format!(
                    "no response to /v1/chat/completions for {:?}",
                    client.idle_timeout()
                ),
            )
            .into());
        }
        Err(RecvTimeoutError::Disconnected) => {
            anyhow::bail!("/v1/chat/completions ended without a response")
        }
    };
    Ok(sse::data_lines(
        BufReader::new(response),
        client.idle_timeout(),
    ))
}

#[cfg(test)]
mod tests {
    use httpmock::{HttpMockResponse, MockServer};
    use rstest::rstest;
    use serde_json::{Value, json};

    use pipette_ops::eval_completions::EvalCompletionsStore;
    use pipette_ops::measurement::EvalMessages;
    use pipette_plan_types::{
        BenchmarkDefinition, BenchmarkEvalCompletionStopReason, BenchmarkId, DeclaredBound,
        EvalBenchmark, Model, PositiveCount, RunRequest, Runtime, SamplingSeed,
        SamplingTemperature,
    };

    use self::Answer::{Cuts, Hangs, Refuses, Stops};
    use super::*;

    /// A fixture body ends at EOF rather than on silence, so the bound a test
    /// passes is unreachable.
    const NO_IDLE: Duration = Duration::from_secs(3600);

    /// A `math_500` entry that pins nothing, so its temperature is the eval-id
    /// policy's `0.6`.
    fn unpinned() -> anyhow::Result<EvalBenchmark> {
        Ok(EvalBenchmark {
            benchmark_id: BenchmarkId::try_new("eval_math_500_2026.06.1".to_owned())?,
            parameter_eval_id: "math_500".into(),
            parameter_dataset_name: "2026.06.1".into(),
            parameter_max_tokens: PositiveCount::try_new(256)?,
            parameter_max_context: None,
            parameter_mcq_choices: None,
            parameter_temperature: None,
            parameter_top_p: None,
            parameter_top_k: None,
            parameter_min_p: None,
            parameter_seed: None,
            samples: None,
        })
    }

    fn pinned() -> anyhow::Result<EvalBenchmark> {
        Ok(EvalBenchmark {
            parameter_temperature: Some(SamplingTemperature::try_new(1.0)?),
            parameter_top_p: Some(TopP::try_new(0.95)?),
            parameter_top_k: Some(TopK::try_new(64)?),
            parameter_min_p: Some(MinP::try_new(0.05)?),
            parameter_seed: Some(SamplingSeed::try_new(7)?),
            ..unpinned()?
        })
    }

    /// The server keeps its tokenizer-derived default only when the key is
    /// absent, not null.
    #[test]
    fn enable_thinking_none_omits_chat_template_kwargs() -> anyhow::Result<()> {
        let benchmark = unpinned()?;
        let sampling = Sampling::from_benchmark(&benchmark)?;
        let request = chat_request(serde_json::json!([]), 256, &sampling, "s#0", None);
        let body = serde_json::to_value(&request)?;
        assert!(
            body.get("chat_template_kwargs").is_none(),
            "key must be absent, not null: {body}"
        );
        assert_eq!(body["temperature"], 0.6);
        Ok(())
    }

    #[rstest]
    #[case::enabled(Some(true), serde_json::Value::Bool(true))]
    #[case::disabled(Some(false), serde_json::Value::Bool(false))]
    fn enable_thinking_some_forwards_the_flag(
        #[case] input: Option<bool>,
        #[case] expected: serde_json::Value,
    ) -> anyhow::Result<()> {
        let benchmark = unpinned()?;
        let sampling = Sampling::from_benchmark(&benchmark)?;
        let request = chat_request(serde_json::json!([]), 256, &sampling, "s#0", input);
        let body = serde_json::to_value(&request)?;
        assert_eq!(body["chat_template_kwargs"]["enable_thinking"], expected);
        Ok(())
    }

    /// What the catalog pins reaches the server and nothing else does: a cut
    /// this client restated would become the server's default in its place.
    /// The seed is the sample's own rather than the entry's base.
    #[rstest]
    #[case::unpinned(unpinned()?, serde_json::json!({"temperature": 0.6}))]
    #[case::pinned(
        pinned()?,
        serde_json::json!({
            "temperature": 1.0, "top_p": 0.95, "top_k": 64, "min_p": 0.05, "seed": 9,
        })
    )]
    fn the_request_carries_what_the_entry_pinned(
        #[case] benchmark: EvalBenchmark,
        #[case] expected: serde_json::Value,
    ) -> anyhow::Result<()> {
        let sampling = Sampling::from_benchmark(&benchmark)?;
        let request = chat_request(serde_json::json!([]), 256, &sampling, "row-3#2", None);
        let body = serde_json::to_value(&request)?;

        let sent = Value::Object(
            ["temperature", "top_p", "top_k", "min_p", "seed"]
                .into_iter()
                .filter_map(|key| body.get(key).map(|value| (key.to_owned(), value.clone())))
                .collect(),
        );
        assert_eq!(sent, expected, "{body}");
        Ok(())
    }

    /// The mobile clients are held to the contract's `sampling.json`, so it must be what
    /// this server is sent.
    #[test]
    fn every_contract_sample_is_sent_as_the_contract_records_it() -> anyhow::Result<()> {
        let fixtures =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../pipette-contract/fixtures");
        let recorded: std::collections::BTreeMap<String, serde_json::Map<String, Value>> =
            serde_json::from_str(&std::fs::read_to_string(fixtures.join("sampling.json"))?)?;
        anyhow::ensure!(!recorded.is_empty(), "sampling.json records no eval");
        for (example, samples) in recorded {
            let definition: BenchmarkDefinition =
                serde_json::from_str(&std::fs::read_to_string(fixtures.join(&example))?)?;
            let sampling = Sampling::from_benchmark(definition.as_eval()?)?;
            for (id, drawn) in samples {
                let request = chat_request(serde_json::json!([]), 256, &sampling, &id, None);
                let body = serde_json::to_value(&request)?;
                let sent = Value::Object(
                    ["temperature", "top_p", "top_k", "min_p", "seed"]
                        .into_iter()
                        .filter_map(|key| {
                            body.get(key).map(|value| (key.to_owned(), value.clone()))
                        })
                        .collect(),
                );
                assert_eq!(sent, drawn, "{example} {id}");
            }
        }
        Ok(())
    }

    /// The trailer that carries `completion_tokens` comes only when asked for.
    #[test]
    fn the_request_asks_for_the_usage_trailer() -> anyhow::Result<()> {
        let benchmark = unpinned()?;
        let sampling = Sampling::from_benchmark(&benchmark)?;
        let request = chat_request(json!([]), 256, &sampling, "s#0", None);
        let body = serde_json::to_value(&request)?;
        assert_eq!(body["stream"], true);
        assert_eq!(body["stream_options"], json!({"include_usage": true}));
        Ok(())
    }

    /// Captured from `mlx_lm.server` 0.31.3 (LFM2.5-350M-MLX-4bit, greedy,
    /// `include_usage`), ids and timestamps dropped. The count includes the EOS
    /// token: "Hello" is one token.
    const STOPPED_ON_EOS: &str = r#": keepalive 16/16

data: {"object": "chat.completion.chunk", "choices": [{"index": 0, "finish_reason": null, "delta": {"role": "assistant", "content": "Hello"}}]}

data: {"object": "chat.completion.chunk", "choices": [{"index": 0, "finish_reason": "stop", "delta": {"role": "assistant"}}]}

data: {"object": "chat.completion", "choices": [], "usage": {"prompt_tokens": 16, "completion_tokens": 2, "total_tokens": 18, "prompt_tokens_details": {"cached_tokens": 0}}}

data: [DONE]

"#;

    /// The same server cut at `max_tokens: 4`.
    const CUT_AT_MAX_TOKENS: &str = r#": keepalive 17/17

data: {"object": "chat.completion.chunk", "choices": [{"index": 0, "finish_reason": null, "delta": {"role": "assistant", "content": "**"}}]}

data: {"object": "chat.completion.chunk", "choices": [{"index": 0, "finish_reason": null, "delta": {"role": "assistant", "content": "Title"}}]}

data: {"object": "chat.completion.chunk", "choices": [{"index": 0, "finish_reason": null, "delta": {"role": "assistant", "content": ":"}}]}

data: {"object": "chat.completion.chunk", "choices": [{"index": 0, "finish_reason": null, "delta": {"role": "assistant", "content": " The"}}]}

data: {"object": "chat.completion.chunk", "choices": [{"index": 0, "finish_reason": "length", "delta": {"role": "assistant"}}]}

data: {"object": "chat.completion", "choices": [], "usage": {"prompt_tokens": 17, "completion_tokens": 4, "total_tokens": 21, "prompt_tokens_details": {"cached_tokens": 0}}}

data: [DONE]

"#;

    /// Captured without `include_usage`: no trailer.
    const NO_USAGE_TRAILER: &str = "data: {\"object\": \"chat.completion.chunk\", \"choices\": [{\"index\": 0, \"finish_reason\": null, \"delta\": {\"role\": \"assistant\", \"content\": \"Hi\"}}]}\n\ndata: {\"object\": \"chat.completion.chunk\", \"choices\": [{\"index\": 0, \"finish_reason\": null, \"delta\": {\"role\": \"assistant\", \"content\": \"!\"}}]}\n\ndata: {\"object\": \"chat.completion.chunk\", \"choices\": [{\"index\": 0, \"finish_reason\": \"stop\", \"delta\": {\"role\": \"assistant\"}}]}\n\ndata: [DONE]\n\n";

    #[rstest]
    #[case::stopped_on_eos(STOPPED_ON_EOS, "Hello", Some("stop"), Some(2))]
    #[case::cut_at_max_tokens(CUT_AT_MAX_TOKENS, "**Title: The", Some("length"), Some(4))]
    #[case::no_usage_trailer(NO_USAGE_TRAILER, "Hi!", Some("stop"), None)]
    fn a_chat_stream_says_how_it_ended(
        #[case] body: &str,
        #[case] content: &str,
        #[case] finish_reason: Option<&str>,
        #[case] completion_tokens: Option<u64>,
    ) -> anyhow::Result<()> {
        let reply = sse::read_chat_stream(
            sse::data_lines(std::io::Cursor::new(body.to_owned()), NO_IDLE),
            &mut |_| {},
        )?;
        assert_eq!(
            (
                reply.content.as_str(),
                reply.finish_reason.as_deref(),
                reply.completion_tokens
            ),
            (content, finish_reason, completion_tokens)
        );
        Ok(())
    }

    /// A stand-in for the server process: `sh`, so the suite needs no Python.
    fn stand_in(script: &str) -> anyhow::Result<Process> {
        let mut command = std::process::Command::new("sh");
        command.args(["-c", script]);
        Ok(start_process(
            server::TEXT_SERVER_MODULE,
            command,
            server::STOP_GRACE,
        )?)
    }

    /// How the stub answers a sample.
    enum Answer {
        Stops,
        /// A 404, as for a sample the chat template cannot take.
        Refuses,
        /// A stream cut before `[DONE]`, and the first server exits.
        Cuts,
        /// Nothing, past any idle bound a case passes.
        Hangs,
    }

    /// What a restart brings up.
    enum Restart {
        Stays,
        Exits,
        Fails,
    }

    /// A failed sample is a row saying why. The cell goes on: on the same server
    /// past a refusal, on a new one past an exit or a server that never
    /// answered. It ends when the new one fails to start, or when
    /// [`LOST_IN_A_ROW`] samples in a row lose their server.
    #[rstest]
    #[case::refused(
        &[Stops, Refuses, Stops],
        Restart::Stays,
        NO_IDLE,
        "refused the request: failed to call /v1/chat/completions: HTTP 404",
        &[false, true, false],
        0,
        None
    )]
    #[case::exited_under_it(
        &[Stops, Cuts, Stops],
        Restart::Stays,
        NO_IDLE,
        "exited mid-completion (exit status: 7)",
        &[false, true, false],
        1,
        None
    )]
    #[case::never_answered(
        &[Stops, Hangs, Stops],
        Restart::Stays,
        Duration::from_secs(1),
        "went silent and was stopped: no response to /v1/chat/completions",
        &[false, true, false],
        1,
        None
    )]
    #[case::restart_failed(
        &[Stops, Cuts, Stops],
        Restart::Fails,
        NO_IDLE,
        "exited mid-completion (exit status: 7)",
        &[false, true],
        1,
        Some("did not come back after it went away: no room")
    )]
    #[case::lost_in_a_row(
        &[Cuts, Cuts, Cuts, Stops],
        Restart::Exits,
        NO_IDLE,
        "exited mid-completion (exit status: 7)",
        &[true, true, true],
        2,
        Some("went away under 3 samples in a row")
    )]
    fn a_failed_sample_is_a_row(
        #[case] answers: &[Answer],
        #[case] restart: Restart,
        #[case] idle: Duration,
        #[case] says: &str,
        #[case] failed: &[bool],
        #[case] restarts: usize,
        #[case] ends: Option<&str>,
    ) -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let exit = dir.path().join("exit");
        let first = stand_in(&format!(
            "while [ ! -e '{}' ]; do sleep 0.02; done; exit 7",
            exit.display()
        ))?;
        let ids: Vec<String> = (0..answers.len()).map(|n| format!("s{n}")).collect();
        let stub = MockServer::start();
        ids.iter().zip(answers).for_each(|(id, answer)| {
            let exit = exit.clone();
            stub.mock(|when, then| {
                when.path("/v1/chat/completions")
                    .body_includes(format!(r#""content":"{id}""#));
                match answer {
                    Stops => {
                        then.status(200)
                            .header("content-type", "text/event-stream")
                            .body(STOPPED_ON_EOS);
                    }
                    Refuses => {
                        then.status(404)
                            .body(r#"{"error": "Only 'text' content type is supported."}"#);
                    }
                    Cuts => {
                        then.respond_with(move |_| {
                            let _ = std::fs::write(&exit, b"");
                            HttpMockResponse::builder()
                                .status(200)
                                .header("content-type", "text/event-stream")
                                .body(": keepalive 16/16\n\n")
                                .build()
                        });
                    }
                    Hangs => {
                        then.delay(Duration::from_secs(60));
                    }
                }
            });
        });
        let url = stub.base_url();
        // The checkpoint keys on the whole request; the runtime it names does
        // not matter here.
        let req = RunRequest {
            runtime: DeclaredBound::already_bound(Runtime::AppleFoundation(Default::default())),
            model: DeclaredBound::already_bound(Model::AppleFoundationText),
            runtime_flags: None,
            model_flags: None,
            benchmark_flags: None,
            benchmark: BenchmarkDefinition::Eval(EvalBenchmark {
                samples: Some(
                    ids.iter()
                        .map(|id| json!({"id": id, "messages": [{"role": "user", "content": id}]}))
                        .collect(),
                ),
                ..unpinned()?
            }),
        };
        let benchmark = req.benchmark.as_eval()?;
        let server = Mutex::new((first, url.clone()));
        let starts = AtomicUsize::new(0);

        let outcome = eval(
            benchmark,
            &EvalCell::new(benchmark, None, None)?,
            &HttpClient::streaming("pipette", idle)?,
            EvalCompletionsStore::new(dir.path()).open(&req)?,
            &server,
            &|| {
                starts.fetch_add(1, Ordering::SeqCst);
                let replacement = match restart {
                    Restart::Stays => stand_in("exec sleep 60")?,
                    Restart::Exits => stand_in("exit 7")?,
                    Restart::Fails => anyhow::bail!("no room"),
                };
                Ok((replacement, url.clone()))
            },
        );

        match (outcome, ends) {
            (Ok(_), None) => {}
            (Err(err), Some(ends)) => {
                anyhow::ensure!(format!("{err:#}").contains(ends), "{err:#}");
            }
            (Ok(data), Some(ends)) => anyhow::bail!("expected the cell to end on {ends}: {data:?}"),
            (Err(err), None) => return Err(err),
        }
        // The rows on disk, which a resume would start from; reloaded by id,
        // so in no particular order.
        let recorded = EvalCompletionsStore::new(dir.path()).open(&req)?;
        let completions = recorded.completions();
        let mut rows: Vec<_> = completions
            .iter()
            .map(|row| {
                let failed = row.failed_reason.as_deref().map(|why| why.contains(says));
                (
                    row.id.as_str(),
                    failed,
                    row.stop_reason,
                    row.completion_tokens,
                )
            })
            .collect();
        rows.sort_by_key(|&(id, ..)| id);
        let eos = BenchmarkEvalCompletionStopReason::Eos;
        let failure = BenchmarkEvalCompletionStopReason::Failure;
        let expected: Vec<_> = ids
            .iter()
            .zip(failed)
            .map(|(id, &failed)| {
                if failed {
                    (id.as_str(), Some(true), failure, None)
                } else {
                    (id.as_str(), None, eos, Some(2))
                }
            })
            .collect();
        assert_eq!(rows, expected, "{completions:?}");
        assert_eq!(starts.into_inner(), restarts);
        Ok(())
    }

    fn sample(messages: &Value) -> EvalMeasurementSample<'_> {
        EvalMeasurementSample {
            id: "s0",
            messages: EvalMessages::Raw(messages),
            chat_template_kwargs: None,
            position: "eval sample 1/1: id=s0".to_owned(),
        }
    }

    /// A stream the idle bound cut: a pipe with a live writer never yields a
    /// line.
    fn silent_stream() -> anyhow::Result<anyhow::Error> {
        let (reader, _writer) = std::io::pipe()?;
        sse::read_chat_stream(
            sse::data_lines(BufReader::new(reader), Duration::from_millis(50)),
            &mut |_| {},
        )
        .err()
        .context("a silent stream was read")
    }

    /// A stream that ends before `[DONE]`.
    fn cut_stream() -> anyhow::Result<anyhow::Error> {
        sse::read_chat_stream(
            sse::data_lines(std::io::Cursor::new(String::new()), NO_IDLE),
            &mut |_| {},
        )
        .err()
        .context("an empty body was read")
    }

    /// A sample is lost only with its server: one that went silent is stopped,
    /// one that exited is gone. A stream cut while the server stays up is the
    /// sample's own failure.
    #[rstest]
    #[case::went_silent("exec sleep 60", silent_stream()?, true, "no SSE stream event")]
    #[case::exited("exit 7", cut_stream()?, true, "exit status: 7")]
    #[case::stayed_up("exec sleep 60", cut_stream()?, false, "stayed up")]
    fn a_sample_is_lost_only_with_its_server(
        #[case] script: &str,
        #[case] err: anyhow::Error,
        #[case] lost: bool,
        #[case] says: &str,
    ) -> anyhow::Result<()> {
        let mut mlx_server = stand_in(script)?;

        let reading = failure_reading(&mut mlx_server, &err, &sample(&json!([])))?;

        let (was_lost, reason) = match &reading {
            EvalMeasurementReading::Lost { reason } => (true, reason),
            EvalMeasurementReading::Failed { reason } => (false, reason),
            EvalMeasurementReading::Completed { .. } | EvalMeasurementReading::Unfit { .. } => {
                anyhow::bail!("{reading:?}")
            }
        };
        assert_eq!(was_lost, lost, "{reason}");
        assert!(reason.contains(says), "{reason}");
        assert_eq!(mlx_server.is_running(), !lost);
        Ok(())
    }
}
