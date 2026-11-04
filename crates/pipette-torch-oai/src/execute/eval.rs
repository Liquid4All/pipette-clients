//! Eval against an OpenAI-compatible server.
//!
//! - **Chat template** is applied server-side: the `messages` array goes
//!   verbatim and the engine assembles the prompt.
//! - **MCQ** uses vLLM's `guided_choice` extra parameter to constrain
//!   generation to one of the listed choices. On engines that don't recognize
//!   `guided_choice` (TGI / older SGLang, strict OpenAI mocks), we fall back
//!   to argmax over `top_logprobs` at the first decoded token.
//! - **Free-text** streams via OpenAI SSE; chunks are accumulated, and the
//!   finish chunk's `finish_reason` is the row's stop reason. On vLLM, the
//!   reasoning its reasoning parser streams apart from the content (always on
//!   gpt-oss) goes back in front of the answer, closed by `</think>`
//!   ([`sse::ChatReply::completion_with_reasoning`]). An SGLang row keeps the
//!   content alone.
//!
//! Samples run through [`measurement::eval()`], which resumes from the
//! [`EvalCompletionsStore`] checkpoint keyed by the portable [`RunRequest`]
//! digest.

use std::io::{BufReader, Write};
use std::num::NonZeroUsize;

use anyhow::Context;
use reqwest::blocking::Client;
use serde_json::{Value, json};

use pipette_http::{HttpClient, StreamClient};
use pipette_ops::eval_cell::{EvalCell, Sampling};
use pipette_ops::eval_completions::EvalCompletionsStore;
use pipette_ops::measurement::{self, EvalMeasurementInFlight, EvalMeasurementReading};
use pipette_ops::run_response::RunResponseBuilder;
use pipette_ops::sse;
use pipette_plan_types::{
    BenchmarkEvalCompletionStopReason, PositiveCount, ReasoningEffort, RunRequest, RuntimeFlags,
    VllmRepetitionDetection,
};

use crate::server::{ServerKind, ServerState};

pub(super) fn run(
    req: &RunRequest,
    model: &str,
    state: &ServerState,
    eval_completions: &EvalCompletionsStore,
) -> anyhow::Result<RunResponseBuilder> {
    // Client-side fan-out, vLLM's repetition stop, and the context the cell
    // pinned. Matched on the typed variant rather than read off the flat
    // `RuntimeFlagRef` so a cell that cannot carry a knob is a compile-time
    // absence, not a silently-ignored field.
    let (parallel, repetition_detection, max_model_len) = match req.runtime_flags.as_ref() {
        Some(
            RuntimeFlags::EvalDockerVllmTorch {
                parallel,
                repetition_detection,
                max_model_len,
                ..
            }
            | RuntimeFlags::EvalUvVllmTorch {
                parallel,
                repetition_detection,
                max_model_len,
                ..
            },
        ) => (*parallel, *repetition_detection, *max_model_len),
        Some(
            RuntimeFlags::EvalDockerSglangTorch { parallel, .. }
            | RuntimeFlags::EvalUvSglangTorch { parallel, .. },
        ) => (*parallel, None, None),
        _ => (None, None, None),
    };
    let cell = EvalCell::from_request(req, parallel)?;
    let context_bound = context_bound(max_model_len, &cell);
    // Eval carries no batch size; its own `parallel` knob is the client-side
    // fan-out, and each request in that pool gets the unscaled budget.
    let request_timeout = measurement::http_timeout(req, 1);
    // The unary MCQ path keeps the request budget above; a stream can only
    // be bounded by its silence, which rides with the client that reads it.
    let idle_timeout = measurement::stream_idle_timeout(req);
    let base_url = state.base_url();
    let chat_url = format!("{base_url}/v1/chat/completions");

    let bounded_client = HttpClient::loopback("pipette", request_timeout)
        .context("failed to build HTTP client")?
        .client()
        .clone();
    let streaming_client = HttpClient::streaming("pipette", idle_timeout)
        .context("failed to build streaming HTTP client")?;

    // Surface the model flags so operators can confirm from the log alone
    // whether each was passed; an absent one is a field the request omits.
    tracing::info!(
        benchmark_id = %cell.benchmark_id,
        enable_thinking = cell.enable_thinking,
        reasoning_effort = cell.reasoning_effort.map(tracing::field::display),
        "eval thinking mode"
    );
    if let Some(detection) = repetition_detection {
        tracing::info!(
            benchmark_id = %cell.benchmark_id,
            min_pattern_size = %detection.min_pattern_size,
            max_pattern_size = %detection.max_pattern_size,
            min_count = %detection.min_count,
            "eval repetition detection on"
        );
    }
    if let Some(max_model_len) = context_bound {
        tracing::info!(
            benchmark_id = %cell.benchmark_id,
            %max_model_len,
            max_tokens = cell.max_tokens,
            "eval sends no max_tokens; max_model_len bounds prompt + output"
        );
    }

    let result_data = measurement::eval(
        req.benchmark.as_eval()?,
        EvalMeasurementInFlight::Parallel(
            NonZeroUsize::new(cell.workers).unwrap_or(NonZeroUsize::MIN),
        ),
        eval_completions.open(req)?,
        (),
        |_, samples, record| {
            samples.iter().enumerate().try_for_each(|(at, sample)| {
                let messages = sample.messages.raw()?;
                // Per sample, not per cell: the seed rides with the `#k` repeat index.
                let sampling = SamplingParams {
                    eval: cell.sampling,
                    sample_id: sample.id,
                    enable_thinking: cell.enable_thinking,
                    reasoning_effort: cell.reasoning_effort,
                    repetition_detection,
                };
                let (content, stop_reason, stop_detail, tokens) = match cell.mcq_choices {
                    // One constrained token says nothing about how an answer
                    // ends: label it `unknown` and say why.
                    Some(choices) => (
                        mcq_completion(
                            &bounded_client,
                            &chat_url,
                            model,
                            messages,
                            choices,
                            sampling,
                        )?,
                        BenchmarkEvalCompletionStopReason::Unknown,
                        Some("mcq arm (guided_choice or first-token logprobs)".to_string()),
                        None,
                    ),
                    None => {
                        let reply = free_text_completion(
                            &streaming_client,
                            &chat_url,
                            model,
                            messages,
                            context_bound.is_none().then_some(cell.max_tokens),
                            sampling,
                            // A terminal only: in a log the raw tokens would
                            // split the one-event-per-line stream.
                            cell.workers <= 1
                                && std::io::IsTerminal::is_terminal(&std::io::stderr()),
                        )?;
                        let (stop_reason, stop_detail) = reply.stop();
                        let tokens = reply.completion_tokens;
                        let completion = match state.kind {
                            ServerKind::Vllm => reply.completion_with_reasoning(),
                            ServerKind::Sglang => reply.content,
                        };
                        (completion, stop_reason, stop_detail, tokens)
                    }
                };
                tracing::info!(
                    sample = sample.position.as_str(),
                    sample_id = %sample.id,
                    completion = truncate_for_log(&content).as_str(),
                    ?stop_reason,
                    completion_tokens = tokens,
                    stop_detail = stop_detail.as_deref(),
                    "sample completed"
                );
                record(
                    at,
                    EvalMeasurementReading::Completed {
                        completion: content,
                        stop_reason,
                        stop_detail,
                        completion_tokens: tokens,
                    },
                )
            })?;
            Ok(())
        },
    )?;

    let mut command = vec![
        format!("POST {chat_url}"),
        match context_bound {
            Some(max_model_len) => format!(
                "model={model} max_tokens=unset (max_model_len={max_model_len} bounds prompt + output)"
            ),
            None => format!("model={model} max_tokens={}", cell.max_tokens),
        },
        match cell.mcq_choices {
            Some(choices) => format!("guided_choice={}", choices.len()),
            None => "stream=true".to_string(),
        },
    ];
    // Only the free-text arm sends it.
    if let (Some(detection), None) = (repetition_detection, cell.mcq_choices) {
        command.push(format!(
            "repetition_detection={}",
            serde_json::to_string(&detection)?
        ));
    }
    Ok(RunResponseBuilder::new(result_data)
        .executable(state.executable.display().to_string())
        .command(command))
}

/// The context a vLLM cell pinned at or under its output budget, which bounds
/// a free-text sample in place of `max_tokens`.
///
/// vLLM refuses (HTTP 400) a request whose prompt plus `max_tokens` exceeds
/// `max_model_len` rather than clamp it. Without `max_tokens` it generates up
/// to `max_model_len` less the prompt, as liquid_evals' offline vLLM does. The
/// MCQ arm asks for a few tokens of its own and never takes this.
fn context_bound(
    max_model_len: Option<PositiveCount>,
    cell: &EvalCell<'_>,
) -> Option<PositiveCount> {
    max_model_len.filter(|len| {
        cell.mcq_choices.is_none() && len.into_inner().unsigned_abs() <= cell.max_tokens
    })
}

/// Write the cell's model flags into a chat-completions request body, each
/// only when set, so engines that don't recognize a field don't reject the
/// request.
///
/// - `enable_thinking` goes in `chat_template_kwargs`, which vLLM (>=0.7) and
///   SGLang (>=0.4.6) forward to `apply_chat_template`.
/// - `reasoning_effort` is the request's own top-level field. vLLM's Harmony
///   path (gpt-oss) reads the effort from nowhere else; on other models vLLM
///   passes it to the template as a kwarg, and turns `enable_thinking` on
///   with it unless `chat_template_kwargs` already says.
fn apply_model_flags(
    body: &mut Value,
    enable_thinking: Option<bool>,
    reasoning_effort: Option<ReasoningEffort>,
) {
    if let Some(enable_thinking) = enable_thinking {
        body["chat_template_kwargs"] = json!({ "enable_thinking": enable_thinking });
    }
    if let Some(reasoning_effort) = reasoning_effort {
        body["reasoning_effort"] = json!(reasoning_effort);
    }
}

/// Per-request sampling knobs shared by the MCQ and free-text builders.
/// `eval` is the cell's resolved sampling (the catalog's where it pinned any,
/// else the policy keyed on the eval id), seeded from `sample_id`.
/// `repetition_detection` is sent by the free-text builder alone.
#[derive(Clone, Copy)]
struct SamplingParams<'a> {
    eval: Sampling<'a>,
    sample_id: &'a str,
    enable_thinking: Option<bool>,
    reasoning_effort: Option<ReasoningEffort>,
    repetition_detection: Option<VllmRepetitionDetection>,
}

impl SamplingParams<'_> {
    /// SGLang refuses `top_k: 0` and spells "no cut" `-1`, which vLLM accepts
    /// too, so a pinned `0` goes out as `-1`.
    fn apply(self, body: &mut Value) -> anyhow::Result<()> {
        self.eval.apply(body, self.sample_id)?;
        if self.eval.top_k.is_some_and(|k| k.into_inner() == 0) {
            body["top_k"] = json!(-1);
        }
        Ok(())
    }
}

/// Send an MCQ request. Prefer vLLM's `guided_choice` (top-level extra field);
/// fall back to `top_logprobs` argmax if the server rejects the extra field.
fn mcq_completion(
    client: &Client,
    url: &str,
    model: &str,
    messages: &Value,
    choices: &[String],
    sampling: SamplingParams<'_>,
) -> anyhow::Result<String> {
    // Attempt 1: guided_choice. vLLM accepts a top-level field; OpenAI-strict
    // servers may reject it. On any 4xx, fall through. The seed is the
    // sample's own, so repeats under temperature > 0 stay independent draws.
    let mut body = json!({
        "model": model,
        "messages": messages,
        "max_tokens": 8,
        "guided_choice": choices,
    });
    sampling.apply(&mut body)?;
    apply_model_flags(
        &mut body,
        sampling.enable_thinking,
        sampling.reasoning_effort,
    );
    match pipette_http::send(client.post(url).json(&body)) {
        Ok(response) => {
            let value: Value = response
                .json()
                .context("failed to parse /v1/chat/completions response (guided_choice)")?;
            return Ok(extract_choice_content(&value).unwrap_or_default());
        }
        Err(pipette_http::Error::Status { status, body, .. }) if status.is_client_error() => {
            tracing::debug!(
                status = status.as_u16(),
                body = body.as_str(),
                "guided_choice rejected; falling back to top_logprobs"
            );
        }
        Err(err) => return Err(err).context("MCQ request failed"),
    }

    // Attempt 2: top_logprobs. Ask for one token + per-token top-K logprobs;
    // pick the listed choice with the highest log-prob.
    //
    // No pinned cuts: a pinned `top_k` would narrow the list the choice is
    // matched against. Temperature scales logits monotonically, so it cannot
    // reorder them.
    let top_logprobs = (choices.len() * 2).max(5);
    let mut body = json!({
        "model": model,
        "messages": messages,
        "max_tokens": 1,
        "temperature": sampling.eval.temperature.as_f64(),
        "logprobs": true,
        "top_logprobs": top_logprobs,
    });
    apply_model_flags(
        &mut body,
        sampling.enable_thinking,
        sampling.reasoning_effort,
    );
    let value: Value = pipette_http::send(client.post(url).json(&body))
        .context("MCQ top_logprobs request failed")?
        .json()
        .context("failed to parse /v1/chat/completions response (top_logprobs)")?;
    Ok(
        pick_choice_from_logprobs(&value, choices).unwrap_or_else(|| {
            tracing::warn!("could not match any choice from top_logprobs; returning empty");
            String::new()
        }),
    )
}

fn extract_choice_content(value: &Value) -> Option<String> {
    let content = value
        .get("choices")?
        .get(0)?
        .get("message")?
        .get("content")?
        .as_str()?
        .trim();
    Some(content.to_string())
}

fn pick_choice_from_logprobs(value: &Value, choices: &[String]) -> Option<String> {
    let top = value
        .get("choices")?
        .get(0)?
        .get("logprobs")?
        .get("content")?
        .get(0)?
        .get("top_logprobs")?
        .as_array()?;
    let mut best: Option<(&str, f64)> = None;
    for entry in top {
        let token = entry.get("token").and_then(Value::as_str)?.trim();
        let logprob = entry
            .get("logprob")
            .and_then(Value::as_f64)
            .unwrap_or(f64::NEG_INFINITY);
        // Case-insensitive exact match against a listed choice.
        if let Some(matched) = choices.iter().find(|c| {
            let c = c.trim();
            !c.is_empty() && token.eq_ignore_ascii_case(c)
        }) && best.is_none_or(|(_, lp)| logprob > lp)
        {
            best = Some((matched.as_str(), logprob));
        }
    }
    best.map(|(s, _)| s.to_string())
}

/// Streaming chat completion. Accumulates `choices[0].delta.content` and the
/// delta's reasoning across SSE events, and notes the finish chunk's
/// `finish_reason`, mirroring the content to stderr when `tee_stderr`. Never
/// under a pool: the tee holds the process-wide stderr lock for a whole
/// generation, so every other slot would wait to read its stream and the
/// server would batch one.
fn free_text_completion(
    client: &StreamClient,
    url: &str,
    model: &str,
    messages: &Value,
    // `None` sends no `max_tokens`, for a cell whose context bounds the
    // output (`context_bound`).
    max_tokens: Option<u32>,
    sampling: SamplingParams<'_>,
    tee_stderr: bool,
) -> anyhow::Result<sse::ChatReply> {
    let mut body = json!({
        "model": model,
        "messages": messages,
        "stream": true,
    });
    if let Some(max_tokens) = max_tokens {
        body.as_object_mut()
            .context("a chat request body must be a JSON object")?
            .insert("max_tokens".to_owned(), json!(max_tokens));
    }
    sampling.apply(&mut body)?;
    apply_model_flags(
        &mut body,
        sampling.enable_thinking,
        sampling.reasoning_effort,
    );
    // vLLM's own request field, as authored (liquid_evals sends
    // `RepetitionDetectionParams(2, 128, 10)`). Not on the MCQ arm: a stop
    // could cut a constrained choice short.
    if let Some(detection) = sampling.repetition_detection {
        body.as_object_mut()
            .context("a chat request body must be a JSON object")?
            .insert("repetition_detection".to_owned(), json!(detection));
    }
    let response = pipette_http::send(client.http().post(url).json(&body))
        .context("failed to call /v1/chat/completions")?;
    let events = sse::data_lines(BufReader::new(response), client.idle_timeout());
    if !tee_stderr {
        return sse::read_chat_stream(events, &mut |_| {});
    }
    let mut stderr = std::io::stderr().lock();
    let reply = sse::read_chat_stream(events, &mut |pending| {
        let _ = write!(stderr, "{pending}");
        let _ = stderr.flush();
    })?;
    let _ = writeln!(stderr);
    let _ = stderr.flush();
    Ok(reply)
}

fn truncate_for_log(s: &str) -> String {
    const LIMIT: usize = 120;
    if s.chars().count() <= LIMIT {
        s.to_string()
    } else {
        let mut out: String = s.chars().take(LIMIT).collect();
        out.push('…');
        out
    }
}

#[cfg(test)]
mod tests {
    use std::io::{BufRead, Read};
    use std::net::TcpListener;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::{Arc, Condvar, Mutex};
    use std::time::Duration;

    use httpmock::MockServer;
    use rstest::rstest;

    use pipette_plan_types::{
        BenchmarkDefinition, BenchmarkId, BenchmarkResultData, BenchmarkType, DeclaredBound,
        EvalBenchmark, EvalId, MinP, Model, ModelFlags, ModelType, PositiveCount, Runtime,
        RuntimeFlagRef, RuntimeType, SamplingSeed, SamplingTemperature, TopK, TopP,
    };

    use super::*;

    /// Like [`FakeChatServer`] but answers each connection on its own thread
    /// and records the high-water mark of simultaneous in-flight requests.
    /// The serial fixture cannot show fan-out at all: it accepts one
    /// connection at a time, so a pool would look identical to the sequential
    /// loop against it.
    struct ConcurrentChatServer {
        port: u16,
        served: Arc<AtomicUsize>,
        peak_in_flight: Arc<AtomicUsize>,
    }

    impl ConcurrentChatServer {
        /// `hold` is how long each request is kept open. Without it a fast
        /// reply can retire before its siblings are dispatched, and the peak
        /// would read 1 on a pool that is genuinely concurrent.
        fn spawn(requests: usize, hold: Duration) -> anyhow::Result<Self> {
            let listener = TcpListener::bind("127.0.0.1:0")?;
            let port = listener.local_addr()?.port();
            let served = Arc::new(AtomicUsize::new(0));
            let peak_in_flight = Arc::new(AtomicUsize::new(0));
            let in_flight = Arc::new(AtomicUsize::new(0));
            let (counter, peak, live) = (
                Arc::clone(&served),
                Arc::clone(&peak_in_flight),
                Arc::clone(&in_flight),
            );
            std::thread::spawn(move || {
                listener
                    .incoming()
                    .take(requests)
                    .filter_map(Result::ok)
                    .for_each(|stream| {
                        let (counter, peak, live) =
                            (Arc::clone(&counter), Arc::clone(&peak), Arc::clone(&live));
                        std::thread::spawn(move || {
                            let now = live.fetch_add(1, Ordering::SeqCst) + 1;
                            peak.fetch_max(now, Ordering::SeqCst);
                            std::thread::sleep(hold);
                            counter.fetch_add(1, Ordering::SeqCst);
                            let _ = FakeChatServer::answer(stream, false);
                            live.fetch_sub(1, Ordering::SeqCst);
                        });
                    });
            });
            Ok(Self {
                port,
                served,
                peak_in_flight,
            })
        }

        fn state(&self) -> ServerState {
            ServerState {
                kind: ServerKind::Vllm,
                port: self.port,
                executable: std::path::PathBuf::from("python"),
            }
        }

        fn served(&self) -> usize {
            self.served.load(Ordering::SeqCst)
        }

        fn peak(&self) -> usize {
            self.peak_in_flight.load(Ordering::SeqCst)
        }
    }

    fn vllm_request(n: usize, parallel: Option<i32>) -> anyhow::Result<RunRequest> {
        let samples = (0..n)
            .map(|i| json!({"id": format!("s{i}"), "messages": [{"role": "user", "content": "q"}]}))
            .collect();
        let mut req = mcq_request()?;
        req.benchmark = BenchmarkDefinition::Eval(EvalBenchmark {
            benchmark_id: BenchmarkId::try_new("eval_pool".to_owned())?,
            samples: Some(samples),
            ..repeats(true)?
        });
        req.runtime_flags = Some(RuntimeFlags::EvalUvVllmTorch {
            tensor_parallel_size: None,
            dtype: None,
            max_model_len: None,
            prefix_caching: None,
            parallel: parallel.map(PositiveCount::try_new).transpose()?,
            repetition_detection: None,
            envs: Vec::new(),
            raw: Vec::new(),
        });
        Ok(req)
    }

    /// Every sample is dispatched once whatever the value; unset stays at one
    /// in flight, a pool overlaps, and a count above the sample total clamps.
    #[rstest]
    #[case::pool_overlaps(8, Some(4), 2, 8)]
    #[case::unset_stays_sequential(4, None, 1, 1)]
    #[case::clamped_to_sample_count(2, Some(16), 1, 2)]
    fn parallel_controls_in_flight_requests(
        #[case] samples: usize,
        #[case] parallel: Option<i32>,
        #[case] peak_min: usize,
        #[case] peak_max: usize,
    ) -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let evals = EvalCompletionsStore::new(dir.path());
        let req = vllm_request(samples, parallel)?;

        let server = ConcurrentChatServer::spawn(samples, Duration::from_millis(150))?;
        let out = run(&req, "m", &server.state(), &evals)?.build();

        assert_eq!(server.served(), samples, "every sample dispatched once");
        let peak = server.peak();
        assert!(
            (peak_min..=peak_max).contains(&peak),
            "parallel = {parallel:?} over {samples} samples: peak in-flight {peak} outside {peak_min}..={peak_max}"
        );

        let BenchmarkResultData::Eval { completions } = out.result_data else {
            anyhow::bail!("expected Eval result data");
        };
        let mut ids: Vec<&str> = completions.iter().map(|c| c.id.as_str()).collect();
        ids.sort_unstable();
        assert_eq!(ids.len(), samples, "one completion per sample");
        ids.dedup();
        assert_eq!(ids.len(), samples, "no sample completed twice");
        Ok(())
    }

    /// Asserted through resume, the property operators depend on. The samples
    /// in flight when the 500 lands are lost, so the count is not
    /// deterministic; what holds regardless is that progress is kept, nothing
    /// is scored twice, and the retry finishes the dataset.
    ///
    /// The failing worker raises its abort only after its request returns, so
    /// the server holds every answer after the 500: a sibling that claims
    /// meanwhile stalls long enough for the abort to land, instead of draining
    /// the dataset while the failing worker is descheduled.
    #[test]
    fn a_failing_sample_aborts_the_pool_but_keeps_earlier_rows() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let evals = EvalCompletionsStore::new(dir.path());
        let req = vllm_request(8, Some(2))?;

        // 8 connections offered, the 4th answers 500.
        let crashing =
            FakeChatServer::spawn_holding_after_failure(8, Some(4), Duration::from_millis(500))?;
        assert!(
            run(&req, "m", &crashing.state(), &evals).is_err(),
            "a 500 from the server must fail the cell"
        );
        let served_first = crashing.served();
        assert!(
            served_first < 8,
            "the pool must abort rather than run the whole dataset; served {served_first}"
        );

        let healthy = ConcurrentChatServer::spawn(8, Duration::from_millis(10))?;
        let out = run(&req, "m", &healthy.state(), &evals)?.build();
        assert!(
            healthy.served() < 8,
            "rows kept by the aborted run must not be re-dispatched; served {}",
            healthy.served()
        );

        let BenchmarkResultData::Eval { completions } = out.result_data else {
            anyhow::bail!("expected Eval result data");
        };
        let mut ids: Vec<&str> = completions.iter().map(|c| c.id.as_str()).collect();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), 8, "every sample accounted for exactly once");
        Ok(())
    }

    /// `resume_relevant_flags` strips `parallel` from the resume digest, so the
    /// serial pass and the pool pass open the same checkpoint. The first pass
    /// has to fail to leave part of the dataset.
    #[test]
    fn switching_to_the_pool_resumes_a_partial_serial_run() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let evals = EvalCompletionsStore::new(dir.path());

        // Serial, and the third request 500s: s0 and s1 land, s2 kills the run.
        let serial_req = vllm_request(6, None)?;
        let crashing = FakeChatServer::spawn(3, Some(3))?;
        assert!(
            run(&serial_req, "m", &crashing.state(), &evals).is_err(),
            "sample 3 returned 500; the run must not succeed"
        );
        assert_eq!(crashing.served(), 3, "stops at the failing sample");

        // Same dataset, now with a pool. Only the four unfinished samples
        // should reach the server.
        let pool_req = vllm_request(6, Some(3))?;
        let healthy = ConcurrentChatServer::spawn(4, Duration::from_millis(20))?;
        let out = run(&pool_req, "m", &healthy.state(), &evals)?.build();
        assert_eq!(
            healthy.served(),
            4,
            "s0 and s1 were checkpointed by the serial pass; only 4 remain"
        );

        let BenchmarkResultData::Eval { completions } = out.result_data else {
            anyhow::bail!("expected Eval result data");
        };
        let mut ids: Vec<&str> = completions.iter().map(|c| c.id.as_str()).collect();
        ids.sort_unstable();
        assert_eq!(
            ids,
            ["s0", "s1", "s2", "s3", "s4", "s5"],
            "every sample accounted for exactly once across both passes"
        );
        Ok(())
    }

    #[test]
    fn parallel_is_clamped_to_the_sample_count() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let evals = EvalCompletionsStore::new(dir.path());
        // 16 workers over 2 samples must not spawn 16 threads racing an empty
        // index; the run should simply finish both.
        let req = vllm_request(2, Some(16))?;

        let server = ConcurrentChatServer::spawn(2, Duration::from_millis(10))?;
        let out = run(&req, "m", &server.state(), &evals)?.build();
        assert_eq!(server.served(), 2);
        let BenchmarkResultData::Eval { completions } = out.result_data else {
            anyhow::bail!("expected Eval result data");
        };
        assert_eq!(completions.len(), 2);
        Ok(())
    }

    /// A chat-completions endpoint that can fail a nominated request with a
    /// 500, the status `mcq_completion` bails on (a 4xx falls through to its
    /// `top_logprobs` attempt instead).
    ///
    /// Accepts at most `requests` connections, one request each
    /// (`Connection: close`), so the counter is the number of samples
    /// dispatched. A test with headroom catches an over-dispatching run in its
    /// count assertion rather than in a connection error, which says far less.
    struct FakeChatServer {
        port: u16,
        served: Arc<AtomicUsize>,
        /// Every request body the server was sent, in arrival order.
        bodies: Arc<Mutex<Vec<String>>>,
    }

    impl FakeChatServer {
        fn spawn(requests: usize, fail_on: Option<usize>) -> anyhow::Result<Self> {
            Self::spawn_holding_after_failure(requests, fail_on, Duration::ZERO)
        }

        /// Like [`Self::spawn`], but every answer after the failing one is
        /// delayed by `hold`.
        fn spawn_holding_after_failure(
            requests: usize,
            fail_on: Option<usize>,
            hold: Duration,
        ) -> anyhow::Result<Self> {
            let listener = TcpListener::bind("127.0.0.1:0")?;
            let port = listener.local_addr()?.port();
            let served = Arc::new(AtomicUsize::new(0));
            let bodies = Arc::new(Mutex::new(Vec::new()));
            let (counter, seen) = (Arc::clone(&served), Arc::clone(&bodies));
            std::thread::spawn(move || {
                listener
                    .incoming()
                    .take(requests)
                    .filter_map(Result::ok)
                    .for_each(|stream| {
                        let n = counter.fetch_add(1, Ordering::SeqCst) + 1;
                        if fail_on.is_some_and(|f| n > f) {
                            std::thread::sleep(hold);
                        }
                        let _ = Self::answer_full(stream, fail_on == Some(n), false, Some(&seen));
                    });
            });
            Ok(Self {
                port,
                served,
                bodies,
            })
        }

        fn bodies(&self) -> anyhow::Result<Vec<Value>> {
            self.bodies
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .iter()
                .map(|body| {
                    serde_json::from_str(body)
                        .with_context(|| format!("a recorded body is not JSON: {body}"))
                })
                .collect()
        }

        /// Replies `400` to a `guided_choice` request and answers anything
        /// else: the OpenAI-strict server the `top_logprobs` fallback exists
        /// for. `requests` counts both attempts of every sample.
        fn spawn_rejecting_guided_choice(requests: usize) -> anyhow::Result<Self> {
            let listener = TcpListener::bind("127.0.0.1:0")?;
            let port = listener.local_addr()?.port();
            let served = Arc::new(AtomicUsize::new(0));
            let bodies = Arc::new(Mutex::new(Vec::new()));
            let (counter, seen) = (Arc::clone(&served), Arc::clone(&bodies));
            std::thread::spawn(move || {
                listener
                    .incoming()
                    .take(requests)
                    .filter_map(Result::ok)
                    .for_each(|stream| {
                        counter.fetch_add(1, Ordering::SeqCst);
                        let _ = Self::answer_full(stream, false, true, Some(&seen));
                    });
            });
            Ok(Self {
                port,
                served,
                bodies,
            })
        }

        fn answer(stream: std::net::TcpStream, fail: bool) -> std::io::Result<()> {
            Self::answer_full(stream, fail, false, None)
        }

        fn answer_full(
            mut stream: std::net::TcpStream,
            fail: bool,
            reject_guided: bool,
            seen: Option<&Arc<Mutex<Vec<String>>>>,
        ) -> std::io::Result<()> {
            let body = read_request(&stream)?;
            if let Some(seen) = seen {
                seen.lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .push(body.clone());
            }

            // The free-text arm reads SSE and the MCQ arm reads one JSON
            // document, so the reply follows what the request asked for.
            let (status, body) = if fail {
                (
                    "500 Internal Server Error",
                    "{\"error\":\"injected\"}".to_string(),
                )
            } else if reject_guided && body.contains("guided_choice") {
                (
                    "400 Bad Request",
                    "{\"error\":\"unknown field guided_choice\"}".to_string(),
                )
            } else if reject_guided {
                // The fallback reads one token's top logprobs and argmaxes the
                // listed choices over them.
                (
                    "200 OK",
                    json!({"choices": [{"logprobs": {"content": [{
                        "token": "A",
                        "top_logprobs": [
                            {"token": "A", "logprob": -0.1},
                            {"token": "B", "logprob": -2.0},
                        ],
                    }]}}]})
                    .to_string(),
                )
            } else if body.contains("\"stream\":true") {
                (
                    "200 OK",
                    format!(
                        "data: {}\n\ndata: [DONE]\n\n",
                        json!({"choices": [{"delta": {"content": "A"}}]})
                    ),
                )
            } else {
                (
                    "200 OK",
                    json!({"choices": [{"message": {"content": "A"}}]}).to_string(),
                )
            };
            write!(
                stream,
                "HTTP/1.1 {status}\r\nContent-Type: application/json\r\n\
                 Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            )?;
            stream.flush()
        }

        fn state(&self) -> ServerState {
            ServerState {
                kind: ServerKind::Vllm,
                port: self.port,
                executable: std::path::PathBuf::from("python"),
            }
        }

        fn served(&self) -> usize {
            self.served.load(Ordering::SeqCst)
        }
    }

    /// The request's body. Drained before any reply: replying without reading
    /// it can surface as a write error on the client instead of the status
    /// under test.
    fn read_request(stream: &std::net::TcpStream) -> std::io::Result<String> {
        let mut reader = BufReader::new(stream.try_clone()?);
        let mut length = 0usize;
        loop {
            let mut line = String::new();
            if reader.read_line(&mut line)? == 0 || line == "\r\n" {
                break;
            }
            if let Some(v) = line.to_ascii_lowercase().strip_prefix("content-length:") {
                length = v.trim().parse().unwrap_or(0);
            }
        }
        let mut body = String::new();
        reader.take(length as u64).read_to_string(&mut body)?;
        Ok(body)
    }

    /// Streams every chat completion, but holds the one whose prompt is `held`
    /// open after its first token until `others` other requests have arrived,
    /// or ten seconds pass: a long generation the server keeps batching beside.
    struct HeldStreamServer {
        port: u16,
        /// Whether the held stream was let go by the arrivals, not the bound.
        released: Arc<AtomicBool>,
    }

    impl HeldStreamServer {
        fn spawn(held: &'static str, others: usize) -> anyhow::Result<Self> {
            let listener = TcpListener::bind("127.0.0.1:0")?;
            let port = listener.local_addr()?.port();
            let arrived = Arc::new((Mutex::new(0usize), Condvar::new()));
            let released = Arc::new(AtomicBool::new(false));
            let (arrivals, release) = (Arc::clone(&arrived), Arc::clone(&released));
            std::thread::spawn(move || {
                listener
                    .incoming()
                    .filter_map(Result::ok)
                    .for_each(|stream| {
                        let (arrived, released) = (Arc::clone(&arrivals), Arc::clone(&release));
                        std::thread::spawn(move || {
                            let _ = Self::answer(stream, held, others, &arrived, &released);
                        });
                    });
            });
            Ok(Self { port, released })
        }

        fn answer(
            mut stream: std::net::TcpStream,
            held: &str,
            others: usize,
            arrived: &(Mutex<usize>, Condvar),
            released: &AtomicBool,
        ) -> std::io::Result<()> {
            let body: Value = serde_json::from_str(&read_request(&stream)?).unwrap_or_default();
            let prompt = body["messages"][0]["content"].as_str().unwrap_or_default();
            let (count, all) = arrived;
            if prompt != held {
                *count
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner) += 1;
                all.notify_all();
            }
            let token = |text: &str| {
                format!(
                    "data: {}\n\n",
                    json!({"choices": [{"delta": {"content": text}}]})
                )
            };
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n{}",
                token("A")
            )?;
            stream.flush()?;
            if prompt == held {
                let (arrivals, waited) = all
                    .wait_timeout_while(
                        count
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner),
                        Duration::from_secs(10),
                        |arrivals| *arrivals < others,
                    )
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                drop(arrivals);
                released.store(!waited.timed_out(), Ordering::SeqCst);
            }
            write!(stream, "{}data: [DONE]\n\n", token("B"))?;
            stream.flush()
        }

        fn state(&self) -> ServerState {
            ServerState {
                kind: ServerKind::Vllm,
                port: self.port,
                executable: std::path::PathBuf::from("python"),
            }
        }
    }

    /// A free-text pool reads its streams side by side. While one sample is
    /// still generating, the other slot finishes its samples and sends the
    /// next, so the server always has `parallel` requests to batch. Reading
    /// one stream at a time would park every other slot behind the long one.
    #[test]
    fn a_free_text_pool_keeps_sending_while_one_sample_generates() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let evals = EvalCompletionsStore::new(dir.path());
        let mut req = vllm_request(5, Some(2))?;
        let BenchmarkDefinition::Eval(benchmark) = &mut req.benchmark else {
            anyhow::bail!("expected an eval benchmark");
        };
        benchmark.parameter_mcq_choices = None;
        benchmark.samples = Some(
            (0..5)
                .map(|i| json!({"id": format!("s{i}"), "messages": [{"role": "user", "content": format!("s{i}")}]}))
                .collect(),
        );

        let server = HeldStreamServer::spawn("s0", 4)?;
        let out = run(&req, "m", &server.state(), &evals)?.build();

        assert!(
            server.released.load(Ordering::SeqCst),
            "the other slot stalled behind the long stream instead of sending s1 to s4"
        );
        let BenchmarkResultData::Eval { completions } = out.result_data else {
            anyhow::bail!("expected Eval result data");
        };
        assert!(
            completions.iter().all(|c| c.completion == "AB"),
            "{completions:?}"
        );
        assert_eq!(completions.len(), 5);
        Ok(())
    }

    /// A free-text row's stop reason is the finish chunk's `finish_reason`,
    /// and its text is every delta's content, the finish chunk's included.
    /// The stream is vLLM 0.31's shape: the last token rides in the finish
    /// chunk, beside its own `stop_reason`, which says nothing here.
    #[rstest]
    #[case::eos(Some("stop"), BenchmarkEvalCompletionStopReason::Eos, None)]
    #[case::token_cap(Some("length"), BenchmarkEvalCompletionStopReason::Truncated, None)]
    #[case::repetition_detection(
        Some("repetition"),
        BenchmarkEvalCompletionStopReason::DoomLoop,
        Some("finish_reason=repetition")
    )]
    #[case::other(
        Some("abort"),
        BenchmarkEvalCompletionStopReason::Unknown,
        Some("unrecognized finish_reason=abort")
    )]
    #[case::missing(
        None,
        BenchmarkEvalCompletionStopReason::Unknown,
        Some("the stream carried no finish_reason")
    )]
    fn a_free_text_row_records_the_finish_reason(
        #[case] finish_reason: Option<&str>,
        #[case] stop_reason: BenchmarkEvalCompletionStopReason,
        #[case] stop_detail: Option<&str>,
    ) -> anyhow::Result<()> {
        let completions = free_text_rows(
            ServerKind::Vllm,
            &[
                json!({"choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": null}]}),
                json!({"choices": [{"index": 0, "delta": {"content": "The answer"}, "finish_reason": null}]}),
                json!({"choices": [{"index": 0, "delta": {"content": " is"}, "finish_reason": finish_reason, "stop_reason": 2}]}),
            ],
        )?;
        let rows: Vec<_> = completions
            .iter()
            .map(|c| {
                (
                    c.completion.as_str(),
                    c.failed,
                    c.stop_reason,
                    c.stop_detail.as_deref(),
                )
            })
            .collect();
        assert_eq!(rows, [("The answer is", false, stop_reason, stop_detail)]);
        Ok(())
    }

    /// One free-text sample run against a `kind` server that streams `events`,
    /// and the rows it recorded.
    fn free_text_rows(
        kind: ServerKind,
        events: &[Value],
    ) -> anyhow::Result<Vec<pipette_plan_types::BenchmarkEvalCompletion>> {
        let dir = tempfile::tempdir()?;
        let evals = EvalCompletionsStore::new(dir.path());
        let mut req = vllm_request(1, None)?;
        let BenchmarkDefinition::Eval(benchmark) = &mut req.benchmark else {
            anyhow::bail!("expected an eval benchmark");
        };
        benchmark.parameter_mcq_choices = None;

        let stream: String = events
            .iter()
            .map(|event| format!("data: {event}\n\n"))
            .chain(["data: [DONE]\n\n".to_owned()])
            .collect();
        let stub = MockServer::start();
        stub.mock(|when, then| {
            when.method("POST").path("/v1/chat/completions");
            then.status(200)
                .header("content-type", "text/event-stream")
                .body(&stream);
        });
        let state = ServerState {
            kind,
            port: stub.port(),
            executable: std::path::PathBuf::from("python"),
        };
        let out = run(&req, "m", &state, &evals)?.build();
        let BenchmarkResultData::Eval { completions } = out.result_data else {
            anyhow::bail!("expected Eval result data");
        };
        Ok(completions)
    }

    /// A reply whose reasoning a parser split off into `delta.reasoning`.
    /// `answer` adds a content chunk after it, and the last chunk carries
    /// `finish_reason`.
    fn reasoning_reply(answer: bool, finish_reason: &str) -> Vec<Value> {
        let mut events = vec![
            json!({"choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": null}]}),
            json!({"choices": [{"index": 0, "delta": {"reasoning": "Two and"}, "finish_reason": null}]}),
            json!({"choices": [{"index": 0, "delta": {"reasoning": " two."}, "finish_reason": null}]}),
        ];
        if answer {
            events.push(json!({"choices": [{"index": 0, "delta": {"content": "4."}, "finish_reason": null}]}));
        }
        events
            .push(json!({"choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}]}));
        events
    }

    /// A vLLM row puts the reasoning back in front of the answer, closed by
    /// `</think>`, or keeps it alone when the stop came inside it. An SGLang
    /// row keeps the content alone, whatever the stream carries.
    #[rstest]
    #[case::vllm_answered(
        ServerKind::Vllm,
        reasoning_reply(true, "stop"),
        "Two and two.</think>4.",
        BenchmarkEvalCompletionStopReason::Eos
    )]
    #[case::vllm_cut_in_the_reasoning(
        ServerKind::Vllm,
        reasoning_reply(false, "length"),
        "Two and two.",
        BenchmarkEvalCompletionStopReason::Truncated
    )]
    #[case::vllm_repetition_in_the_reasoning(
        ServerKind::Vllm,
        reasoning_reply(false, "repetition"),
        "Two and two.",
        BenchmarkEvalCompletionStopReason::DoomLoop
    )]
    #[case::sglang_answered(
        ServerKind::Sglang,
        reasoning_reply(true, "stop"),
        "4.",
        BenchmarkEvalCompletionStopReason::Eos
    )]
    #[case::sglang_cut_in_the_reasoning(
        ServerKind::Sglang,
        reasoning_reply(false, "length"),
        "",
        BenchmarkEvalCompletionStopReason::Truncated
    )]
    fn only_a_vllm_row_carries_the_reasoning(
        #[case] kind: ServerKind,
        #[case] events: Vec<Value>,
        #[case] completion: &str,
        #[case] stop_reason: BenchmarkEvalCompletionStopReason,
    ) -> anyhow::Result<()> {
        let completions = free_text_rows(kind, &events)?;
        let rows: Vec<_> = completions
            .iter()
            .map(|c| (c.completion.as_str(), c.failed, c.stop_reason))
            .collect();
        assert_eq!(rows, [(completion, false, stop_reason)]);
        Ok(())
    }

    /// Three MCQ samples, so the store is keyed by a stable portable identity
    /// across both runs below.
    fn mcq_request() -> anyhow::Result<RunRequest> {
        let samples = ["s1", "s2", "s3"]
            .iter()
            .map(|id| json!({"id": id, "messages": [{"role": "user", "content": "q"}]}))
            .collect();
        let runtime = Runtime::AppleFoundation(Default::default());
        let model = Model::AppleFoundationText;
        Ok(RunRequest {
            runtime: DeclaredBound::already_bound(runtime),
            model: DeclaredBound::already_bound(model),
            runtime_flags: None,
            model_flags: None,
            benchmark_flags: None,
            benchmark: BenchmarkDefinition::Eval(EvalBenchmark {
                benchmark_id: BenchmarkId::try_new("eval_resume".to_owned())?,
                samples: Some(samples),
                ..repeats(true)?
            }),
        })
    }

    /// A `math_500` cell of two `#k` repeats that pins nothing.
    fn repeats(mcq: bool) -> anyhow::Result<EvalBenchmark> {
        Ok(EvalBenchmark {
            benchmark_id: BenchmarkId::try_new("eval_sampling".to_owned())?,
            parameter_eval_id: EvalId::from("math_500"),
            parameter_dataset_name: "d".to_string(),
            parameter_max_tokens: PositiveCount::try_new(8)?,
            parameter_max_context: None,
            parameter_mcq_choices: mcq.then(|| vec!["A".to_string(), "B".to_string()]),
            parameter_temperature: None,
            parameter_top_p: None,
            parameter_top_k: None,
            parameter_min_p: None,
            parameter_seed: None,
            samples: Some(vec![
                json!({"id": "q#0", "messages": [{"role": "user", "content": "q"}]}),
                json!({"id": "q#1", "messages": [{"role": "user", "content": "q"}]}),
            ]),
        })
    }

    fn pinned(benchmark: EvalBenchmark) -> anyhow::Result<EvalBenchmark> {
        Ok(EvalBenchmark {
            parameter_temperature: Some(SamplingTemperature::try_new(1.0)?),
            parameter_top_p: Some(TopP::try_new(0.95)?),
            parameter_top_k: Some(TopK::try_new(64)?),
            parameter_min_p: Some(MinP::try_new(0.05)?),
            parameter_seed: Some(SamplingSeed::try_new(7)?),
            ..benchmark
        })
    }

    /// Run `benchmark` under `model_flags` against `server` and hand back the
    /// chat bodies it got.
    fn chat_bodies(
        benchmark: EvalBenchmark,
        model_flags: Option<ModelFlags>,
        server: &FakeChatServer,
    ) -> anyhow::Result<Vec<Value>> {
        let dir = tempfile::tempdir()?;
        let evals = EvalCompletionsStore::new(dir.path());
        let mut req = mcq_request()?;
        req.benchmark = BenchmarkDefinition::Eval(benchmark);
        req.model_flags = model_flags;
        drop(run(&req, "m", &server.state(), &evals)?);
        server.bodies()
    }

    /// The sampling keys of a body, the absent ones left out, so an
    /// assertion catches a knob that should not be there as well as one that
    /// should. The seed is read separately: it differs per `#k`.
    fn sampling_of(body: &Value) -> Value {
        Value::Object(
            ["temperature", "top_p", "top_k", "min_p"]
                .into_iter()
                .filter_map(|key| body.get(key).map(|value| (key.to_owned(), value.clone())))
                .collect(),
        )
    }

    /// Both request builders send what the entry pinned, under the spellings
    /// vLLM and SGLang read, with each `#k` repeat carrying its own seed, and
    /// an unpinned cell sends the policy's temperature alone. A dropped call
    /// would be invisible until a row was filed under an id it did not run at.
    #[rstest]
    #[case::guided_choice_pinned(
        pinned(repeats(true)?)?,
        json!({"temperature": 1.0, "top_p": 0.95, "top_k": 64, "min_p": 0.05}),
        vec![Some(7), Some(8)]
    )]
    #[case::free_text_pinned(
        pinned(repeats(false)?)?,
        json!({"temperature": 1.0, "top_p": 0.95, "top_k": 64, "min_p": 0.05}),
        vec![Some(7), Some(8)]
    )]
    #[case::free_text_unpinned(repeats(false)?, json!({"temperature": 0.6}), vec![None, None])]
    #[case::top_k_off_is_minus_one(
        EvalBenchmark { parameter_top_k: Some(TopK::try_new(0)?), ..repeats(false)? },
        json!({"temperature": 0.6, "top_k": -1}),
        vec![None, None]
    )]
    fn the_chat_request_carries_what_the_entry_pinned(
        #[case] benchmark: EvalBenchmark,
        #[case] expected: Value,
        #[case] expected_seeds: Vec<Option<i64>>,
    ) -> anyhow::Result<()> {
        let bodies = chat_bodies(benchmark, None, &FakeChatServer::spawn(2, None)?)?;
        assert_eq!(bodies.len(), 2, "{bodies:?}");
        bodies
            .iter()
            .for_each(|body| assert_eq!(sampling_of(body), expected, "{body}"));

        let mut seeds: Vec<Option<i64>> = bodies
            .iter()
            .map(|body| body.get("seed").and_then(Value::as_i64))
            .collect();
        seeds.sort_unstable();
        assert_eq!(seeds, expected_seeds, "{bodies:?}");
        Ok(())
    }

    /// The `top_logprobs` fallback reads the distribution rather than drawing
    /// from it, so it carries the temperature and none of the cuts: a pinned
    /// `top_k` would narrow the very list the choice is matched against.
    #[test]
    fn the_logprobs_fallback_sends_no_cuts() -> anyhow::Result<()> {
        // Two requests per sample: the refused `guided_choice` and the
        // fallback behind it.
        let server = FakeChatServer::spawn_rejecting_guided_choice(4)?;
        let bodies = chat_bodies(pinned(repeats(true)?)?, None, &server)?;
        assert_eq!(bodies.len(), 4, "{bodies:?}");

        let fallbacks: Vec<&Value> = bodies
            .iter()
            .filter(|body| body.get("top_logprobs").is_some())
            .collect();
        assert_eq!(fallbacks.len(), 2, "{bodies:?}");
        fallbacks.iter().for_each(|fallback| {
            assert_eq!(fallback["temperature"], 1.0, "{fallback}");
            ["top_p", "top_k", "min_p", "seed"].iter().for_each(|cut| {
                assert!(
                    fallback.get(cut).is_none(),
                    "{cut} must not ride on the logprobs read: {fallback}"
                );
            });
        });
        Ok(())
    }

    /// The model-flag keys of a body, the absent ones left out, as
    /// [`sampling_of`] reads the sampling ones.
    fn model_flags_of(body: &Value) -> Value {
        Value::Object(
            ["chat_template_kwargs", "reasoning_effort"]
                .into_iter()
                .filter_map(|key| body.get(key).map(|value| (key.to_owned(), value.clone())))
                .collect(),
        )
    }

    /// Every request a cell sends carries its model flags, both MCQ attempts
    /// included (the refused `guided_choice` and the `top_logprobs` fallback
    /// behind it), and a cell that sets none sends neither field. vLLM's
    /// Harmony path reads gpt-oss's effort off the request and nowhere else,
    /// so a builder that dropped it would answer at the default effort under
    /// the cell's name.
    #[rstest]
    #[case::no_flags(None, json!({}))]
    #[case::effort_alone(
        Some(ModelFlags::EvalTorch { enable_thinking: None, reasoning_effort: Some(ReasoningEffort::Low) }),
        json!({"reasoning_effort": "low"})
    )]
    #[case::thinking_alone(
        Some(ModelFlags::EvalTorch { enable_thinking: Some(false), reasoning_effort: None }),
        json!({"chat_template_kwargs": {"enable_thinking": false}})
    )]
    #[case::both(
        Some(ModelFlags::EvalTorch { enable_thinking: Some(true), reasoning_effort: Some(ReasoningEffort::High) }),
        json!({"chat_template_kwargs": {"enable_thinking": true}, "reasoning_effort": "high"})
    )]
    fn every_chat_request_carries_the_model_flags(
        #[case] model_flags: Option<ModelFlags>,
        #[case] expected: Value,
        #[values(false, true)] mcq: bool,
    ) -> anyhow::Result<()> {
        let (server, requests) = if mcq {
            (FakeChatServer::spawn_rejecting_guided_choice(4)?, 4)
        } else {
            (FakeChatServer::spawn(2, None)?, 2)
        };
        let bodies = chat_bodies(repeats(mcq)?, model_flags, &server)?;
        assert_eq!(bodies.len(), requests, "{bodies:?}");
        bodies
            .iter()
            .for_each(|body| assert_eq!(model_flags_of(body), expected, "{body}"));
        Ok(())
    }

    /// liquid_evals' setting, `RepetitionDetectionParams(min_pattern_size=2,
    /// max_pattern_size=128, min_count=10)`.
    fn liquid_evals_repetition_detection() -> anyhow::Result<VllmRepetitionDetection> {
        Ok(VllmRepetitionDetection {
            min_pattern_size: PositiveCount::try_new(2)?,
            max_pattern_size: PositiveCount::try_new(128)?,
            min_count: PositiveCount::try_new(10)?,
        })
    }

    /// The same setting as vLLM's request takes it.
    fn liquid_evals_repetition_detection_wire() -> Value {
        json!({"min_pattern_size": 2, "max_pattern_size": 128, "min_count": 10})
    }

    /// An eval cell's flags on `runtime`, setting the repetition stop and the
    /// pinned context alone.
    fn eval_flags(
        runtime: RuntimeType,
        repetition_detection: Option<VllmRepetitionDetection>,
        max_model_len: Option<i32>,
    ) -> anyhow::Result<RuntimeFlags> {
        let mut flags = RuntimeFlagRef::new(BenchmarkType::Eval, runtime, ModelType::Torch);
        flags.repetition_detection = repetition_detection;
        flags.max_model_len = max_model_len.map(PositiveCount::try_new).transpose()?;
        Ok(RuntimeFlags::try_from(flags)?)
    }

    /// Run `benchmark` under `runtime_flags` against `server`, and hand back
    /// the recorded command beside the chat bodies the server got.
    fn bodies_and_command(
        benchmark: EvalBenchmark,
        runtime_flags: Option<RuntimeFlags>,
        server: &FakeChatServer,
    ) -> anyhow::Result<(Vec<Value>, Vec<String>)> {
        let dir = tempfile::tempdir()?;
        let evals = EvalCompletionsStore::new(dir.path());
        let mut req = mcq_request()?;
        req.benchmark = BenchmarkDefinition::Eval(benchmark);
        req.runtime_flags = runtime_flags;
        let out = run(&req, "m", &server.state(), &evals)?.build();
        Ok((server.bodies()?, out.command))
    }

    /// The free-text body carries `repetition_detection` only when the cell
    /// sets it, and `max_tokens` unless a vLLM cell pins `max_model_len` at or
    /// under the output budget, whose command then names the bound instead. A
    /// context pinned one token above the budget keeps `max_tokens`.
    #[rstest]
    #[case::no_flags(None, Some(8), "model=m max_tokens=8", None)]
    #[case::vllm_unset(
        Some(eval_flags(RuntimeType::UvVllm, None, None)?),
        Some(8),
        "model=m max_tokens=8",
        None
    )]
    #[case::sglang(
        Some(eval_flags(RuntimeType::UvSglang, None, None)?),
        Some(8),
        "model=m max_tokens=8",
        None
    )]
    #[case::uv_vllm_repetition(
        Some(eval_flags(RuntimeType::UvVllm, Some(liquid_evals_repetition_detection()?), None)?),
        Some(8),
        "model=m max_tokens=8",
        Some(liquid_evals_repetition_detection_wire())
    )]
    #[case::docker_vllm_repetition(
        Some(eval_flags(RuntimeType::DockerVllm, Some(liquid_evals_repetition_detection()?), None)?),
        Some(8),
        "model=m max_tokens=8",
        Some(liquid_evals_repetition_detection_wire())
    )]
    #[case::vllm_above_the_budget(
        Some(eval_flags(RuntimeType::UvVllm, None, Some(9))?),
        Some(8),
        "model=m max_tokens=8",
        None
    )]
    #[case::uv_vllm_at_the_budget(
        Some(eval_flags(RuntimeType::UvVllm, None, Some(8))?),
        None,
        "model=m max_tokens=unset (max_model_len=8 bounds prompt + output)",
        None
    )]
    #[case::docker_vllm_at_the_budget(
        Some(eval_flags(RuntimeType::DockerVllm, None, Some(8))?),
        None,
        "model=m max_tokens=unset (max_model_len=8 bounds prompt + output)",
        None
    )]
    #[case::vllm_under_the_budget(
        Some(eval_flags(RuntimeType::UvVllm, None, Some(4))?),
        None,
        "model=m max_tokens=unset (max_model_len=4 bounds prompt + output)",
        None
    )]
    #[case::bound_with_repetition(
        Some(eval_flags(RuntimeType::UvVllm, Some(liquid_evals_repetition_detection()?), Some(8))?),
        None,
        "model=m max_tokens=unset (max_model_len=8 bounds prompt + output)",
        Some(liquid_evals_repetition_detection_wire())
    )]
    fn the_free_text_request_follows_the_cell_flags(
        #[case] runtime_flags: Option<RuntimeFlags>,
        #[case] max_tokens: Option<u32>,
        #[case] model_line: &str,
        #[case] repetition_detection: Option<Value>,
    ) -> anyhow::Result<()> {
        let (bodies, command) = bodies_and_command(
            repeats(false)?,
            runtime_flags,
            &FakeChatServer::spawn(2, None)?,
        )?;
        assert_eq!(bodies.len(), 2, "{bodies:?}");
        let mut expected = json!({
            "model": "m",
            "messages": [{"role": "user", "content": "q"}],
            "stream": true,
            "temperature": 0.6,
        });
        if let Some(max_tokens) = max_tokens {
            expected["max_tokens"] = json!(max_tokens);
        }
        // After the `POST <url>`, whose port is the fixture's.
        let mut expected_command = vec![model_line.to_owned(), "stream=true".to_owned()];
        if let Some(detection) = repetition_detection {
            // The struct's field order, which is vLLM's.
            expected_command.push(
                r#"repetition_detection={"min_pattern_size":2,"max_pattern_size":128,"min_count":10}"#
                    .to_owned(),
            );
            expected["repetition_detection"] = detection;
        }
        bodies
            .iter()
            .for_each(|body| assert_eq!(body, &expected, "{body}"));
        assert_eq!(command.get(1..), Some(&expected_command[..]), "{command:?}");
        Ok(())
    }

    /// The MCQ arm asks for its few tokens whatever the context: a context
    /// pinned at the budget changes neither its body nor its command.
    #[test]
    fn an_mcq_cell_ignores_the_context_bound() -> anyhow::Result<()> {
        let benchmark = EvalBenchmark {
            parameter_max_tokens: PositiveCount::try_new(16)?,
            ..repeats(true)?
        };
        let flags = eval_flags(RuntimeType::UvVllm, None, Some(16))?;
        let (bodies, command) =
            bodies_and_command(benchmark, Some(flags), &FakeChatServer::spawn(2, None)?)?;
        assert_eq!(bodies.len(), 2, "{bodies:?}");
        bodies
            .iter()
            .for_each(|body| assert_eq!(body["max_tokens"], 8, "{body}"));
        assert_eq!(
            command.get(1..),
            Some(
                &[
                    "model=m max_tokens=16".to_owned(),
                    "guided_choice=2".to_owned()
                ][..]
            ),
            "{command:?}"
        );
        Ok(())
    }

    /// The MCQ arm never carries the stop, the `top_logprobs` fallback
    /// included: its answer is a few tokens, too short to repeat.
    #[rstest]
    #[case::guided_choice(false)]
    #[case::top_logprobs_fallback(true)]
    fn an_mcq_request_never_carries_repetition_detection(
        #[case] fallback: bool,
    ) -> anyhow::Result<()> {
        let server = if fallback {
            FakeChatServer::spawn_rejecting_guided_choice(4)?
        } else {
            FakeChatServer::spawn(2, None)?
        };
        let flags = eval_flags(
            RuntimeType::UvVllm,
            Some(liquid_evals_repetition_detection()?),
            None,
        )?;
        let (bodies, command) = bodies_and_command(repeats(true)?, Some(flags), &server)?;
        assert_eq!(bodies.len(), if fallback { 4 } else { 2 }, "{bodies:?}");
        bodies.iter().for_each(|body| {
            assert!(body.get("repetition_detection").is_none(), "{body}");
        });
        assert!(
            !command
                .iter()
                .any(|part| part.contains("repetition_detection")),
            "{command:?}"
        );
        Ok(())
    }

    /// A failed sample aborts the run (torch-oai propagates rather than
    /// recording `failed: true`), so the second run must dispatch only the
    /// samples the first never finished. Every MCQ row, the checkpointed one
    /// included, is `unknown` and says why.
    #[test]
    fn eval_resumes_from_the_checkpoint_after_a_failed_sample() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let evals = EvalCompletionsStore::new(dir.path());
        let req = mcq_request()?;

        let crashing = FakeChatServer::spawn(2, Some(2))?;
        let first = run(&req, "m", &crashing.state(), &evals);
        assert!(
            first.is_err(),
            "sample 2 returned 500; run must not succeed"
        );
        assert_eq!(crashing.served(), 2, "should stop at the failing sample");

        let healthy = FakeChatServer::spawn(3, None)?;
        let resumed = run(&req, "m", &healthy.state(), &evals)?.build();
        assert_eq!(
            healthy.served(),
            2,
            "s1 was checkpointed; only s2 and s3 should be dispatched"
        );
        let BenchmarkResultData::Eval { completions } = resumed.result_data else {
            anyhow::bail!("expected Eval result data");
        };
        let ids: Vec<&str> = completions.iter().map(|c| c.id.as_str()).collect();
        assert_eq!(ids, ["s1", "s2", "s3"], "every sample accounted for once");
        let stops: Vec<_> = completions
            .iter()
            .map(|c| (c.stop_reason, c.stop_detail.as_deref()))
            .collect();
        assert_eq!(
            stops,
            [(
                BenchmarkEvalCompletionStopReason::Unknown,
                Some("mcq arm (guided_choice or first-token logprobs)")
            ); 3]
        );
        Ok(())
    }

    #[test]
    fn pick_choice_from_logprobs_returns_highest_match() {
        let value = json!({
            "choices": [{
                "logprobs": {
                    "content": [{
                        "token": "X",
                        "top_logprobs": [
                            {"token": "A", "logprob": -2.0},
                            {"token": "B", "logprob": -0.5},
                            {"token": "C", "logprob": -3.0},
                        ]
                    }]
                }
            }]
        });
        let choices = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        assert_eq!(
            pick_choice_from_logprobs(&value, &choices).as_deref(),
            Some("B")
        );
    }

    #[test]
    fn pick_choice_from_logprobs_case_insensitive() {
        let value = json!({
            "choices": [{
                "logprobs": {
                    "content": [{
                        "token": "X",
                        "top_logprobs": [{"token": "a", "logprob": -1.0}]
                    }]
                }
            }]
        });
        let choices = vec!["A".to_string()];
        assert_eq!(
            pick_choice_from_logprobs(&value, &choices).as_deref(),
            Some("A")
        );
    }

    #[test]
    fn pick_choice_from_logprobs_returns_none_on_no_match() {
        let value = json!({
            "choices": [{
                "logprobs": {
                    "content": [{
                        "token": "X",
                        "top_logprobs": [{"token": "Z", "logprob": -0.1}]
                    }]
                }
            }]
        });
        let choices = vec!["A".to_string(), "B".to_string()];
        assert!(pick_choice_from_logprobs(&value, &choices).is_none());
    }

    #[test]
    fn extract_choice_content_trims() {
        let value = json!({
            "choices": [{"message": {"content": "  hello \n"}}]
        });
        assert_eq!(extract_choice_content(&value).as_deref(), Some("hello"));
    }

    #[test]
    fn truncate_for_log_short_passthrough() {
        assert_eq!(truncate_for_log("hi"), "hi");
    }

    #[test]
    fn truncate_for_log_long_truncates() {
        let s: String = "a".repeat(200);
        let out = truncate_for_log(&s);
        assert!(out.ends_with('…'));
        assert_eq!(out.chars().count(), 121);
    }

    #[test]
    fn apply_model_flags_omits_both_fields_when_unset() {
        let mut body = json!({"model": "m", "messages": []});
        apply_model_flags(&mut body, None, None);
        assert_eq!(body, json!({"model": "m", "messages": []}));
    }

    #[test]
    fn apply_model_flags_injects_enable_thinking_true() {
        let mut body = json!({"model": "m", "messages": []});
        apply_model_flags(&mut body, Some(true), None);
        assert_eq!(
            body["chat_template_kwargs"],
            json!({"enable_thinking": true})
        );
        assert!(body.get("reasoning_effort").is_none());
    }

    #[test]
    fn apply_model_flags_injects_enable_thinking_false() {
        let mut body = json!({"model": "m", "messages": []});
        apply_model_flags(&mut body, Some(false), None);
        assert_eq!(
            body["chat_template_kwargs"],
            json!({"enable_thinking": false})
        );
    }

    /// The effort is the request's own field, in the lowercase word vLLM's
    /// `reasoning_effort` takes, and never a template kwarg here: vLLM writes
    /// the kwarg itself.
    #[rstest]
    #[case::low(ReasoningEffort::Low, "low")]
    #[case::medium(ReasoningEffort::Medium, "medium")]
    #[case::high(ReasoningEffort::High, "high")]
    fn apply_model_flags_sends_the_effort_top_level(
        #[case] effort: ReasoningEffort,
        #[case] wire: &str,
    ) {
        let mut body = json!({"model": "m", "messages": []});
        apply_model_flags(&mut body, None, Some(effort));
        assert_eq!(
            body,
            json!({"model": "m", "messages": [], "reasoning_effort": wire})
        );
    }
}
