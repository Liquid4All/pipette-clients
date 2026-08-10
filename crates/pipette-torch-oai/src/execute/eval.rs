//! Eval against an OpenAI-compatible server.
//!
//! - **Chat template** is applied server-side by the engine — we send the
//!   `messages` array verbatim and let the engine assemble the prompt.
//! - **MCQ** uses vLLM's `guided_choice` extra parameter to constrain
//!   generation to one of the listed choices. On engines that don't recognize
//!   `guided_choice` (TGI / older SGLang, strict OpenAI mocks), we fall back
//!   to argmax over `top_logprobs` at the first decoded token.
//! - **Free-text** streams via OpenAI SSE; chunks are accumulated and the
//!   `DoomloopPipeline` is consulted on each flush to early-abort on
//!   degenerate output.
//!
//! Resume via [`EvalCompletionsStore`] keyed by portable [`RunRequest`] digest.

use std::{
    io::{BufRead, BufReader, Write},
    time::Duration,
};

use anyhow::Context;
use reqwest::blocking::Client;
use serde_json::{json, Value};

use pipette_doomloop::{format_trigger_log, DoomloopPipeline};
use pipette_http::HttpClient;
use pipette_ops::eval_completions::EvalCompletionsStore;
use pipette_plan_types::benchmark::Temperature;
use pipette_plan_types::result::{
    BenchmarkEvalCompletion, BenchmarkEvalCompletionStopReason, BenchmarkResultData,
};
use pipette_plan_types::run::RunRequest;

use super::{http_timeout, RunResponse};
use crate::server::ServerState;

pub(super) fn run(
    req: &RunRequest,
    model: &str,
    state: &ServerState,
    eval_completions: &EvalCompletionsStore,
) -> anyhow::Result<RunResponse> {
    let benchmark = req.benchmark.as_eval().map_err(anyhow::Error::from)?;
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
    let temperature = req.benchmark.eval_temperature()?;
    let total = samples.len();
    let request_timeout = http_timeout(req);
    let enable_thinking = req.model_flags.as_ref().and_then(|f| f.enable_thinking());
    let doomloop = pipette_doomloop::plan::pipeline_from_overrides(
        req.benchmark_flags.as_ref().and_then(|bf| bf.doomloop()),
    )
    .map_err(|e| anyhow::anyhow!("invalid doom-loop configuration: {e}"))?;
    let base_url = state.base_url();
    let chat_url = format!("{base_url}/v1/chat/completions");

    let bounded_client = HttpClient::blocking_with_timeout("pipette", request_timeout)
        .context("failed to build HTTP client")?;
    let streaming_client = streaming_http_client(request_timeout)?;

    // Portable RunRequest digest (CLI passes the store).
    let mut checkpoint = eval_completions.open(req)?;
    let resume_count = checkpoint.completions().len();
    if resume_count > 0 {
        log::info!(
            "resuming eval {benchmark_id}: {resume_count}/{total} samples already checkpointed"
        );
    }
    // Surface enable_thinking so operators can confirm from the log alone
    // whether the kwarg was passed; None means no chat_template_kwargs field.
    log::info!("eval {benchmark_id}: enable_thinking={enable_thinking:?}");

    for (index, sample) in samples.iter().enumerate() {
        let sample_id = sample
            .get("id")
            .and_then(Value::as_str)
            .context("eval sample missing id")?;
        if checkpoint.contains(sample_id) {
            continue;
        }
        let messages = sample
            .get("messages")
            .cloned()
            .context("eval sample missing messages")?;

        let sampling = SamplingParams {
            temperature,
            enable_thinking,
        };
        let content = if let Some(choices) = &mcq_choices {
            mcq_completion(
                &bounded_client,
                &chat_url,
                model,
                &messages,
                choices,
                sampling,
            )?
        } else {
            free_text_completion(
                &streaming_client,
                &chat_url,
                model,
                &messages,
                max_tokens,
                sampling,
                &doomloop,
            )?
        };

        log::info!(
            "eval sample {}/{}: id={} completion={}",
            index + 1,
            total,
            sample_id,
            truncate_for_log(&content)
        );
        checkpoint.append(BenchmarkEvalCompletion {
            id: sample_id.to_string(),
            completion: content,
            failed: false,
            failed_reason: None,
            // stop_reason capture is llama.cpp-only for now (PIP-274);
            // classifying this runtime is a follow-up, so label it `unknown`
            // and record why rather than claiming eos/truncated.
            stop_reason: BenchmarkEvalCompletionStopReason::Unknown,
            stop_detail: Some("stop_reason capture not implemented for torch-oai".to_string()),
            completion_tokens: None,
        })?;
    }

    let completions = checkpoint.finalize()?;
    Ok(RunResponse {
        executable: Some(state.executable().display().to_string()),
        command: vec![
            format!("POST {chat_url}"),
            format!("model={model} max_tokens={max_tokens}"),
            match &mcq_choices {
                Some(choices) => format!("guided_choice={}", choices.len()),
                None => "stream=true".to_string(),
            },
        ],
        ..RunResponse::new(
            BenchmarkResultData::Eval { completions },
            String::new(),
            String::new(),
        )
    })
}

/// Inject `chat_template_kwargs.enable_thinking` into a chat-completions
/// request body when set. Forwarded by both vLLM (>=0.7) and SGLang (>=0.4.6)
/// to `apply_chat_template`. When `enable_thinking` is `None`, the body is
/// left unchanged so the request stays byte-identical to pre-flag callers
/// (and engines that don't recognize the field don't reject the request).
fn apply_chat_template_kwargs(body: &mut Value, enable_thinking: Option<bool>) {
    let Some(enable_thinking) = enable_thinking else {
        return;
    };
    body["chat_template_kwargs"] = json!({ "enable_thinking": enable_thinking });
}

/// Per-request sampling knobs shared by the MCQ and free-text builders.
/// `temperature` is the client-side eval policy (greedy unless
/// IFBench/IFStruct); `enable_thinking` is the resolved `ModelFlags`
/// value forwarded as `chat_template_kwargs`.
#[derive(Clone, Copy)]
struct SamplingParams {
    temperature: Temperature,
    enable_thinking: Option<bool>,
}

/// Send an MCQ request. Prefer vLLM's `guided_choice` (top-level extra field);
/// fall back to `top_logprobs` argmax if the server rejects the extra field.
fn mcq_completion(
    client: &Client,
    url: &str,
    model: &str,
    messages: &Value,
    choices: &[String],
    sampling: SamplingParams,
) -> anyhow::Result<String> {
    // Attempt 1: guided_choice. vLLM accepts a top-level field; OpenAI-strict
    // servers may reject it. On any 4xx, fall through. No `seed` is sent, so
    // repeats under temperature > 0 are independent draws.
    let mut body = json!({
        "model": model,
        "messages": messages,
        "max_tokens": 8,
        "temperature": sampling.temperature.as_f64(),
        "guided_choice": choices,
    });
    apply_chat_template_kwargs(&mut body, sampling.enable_thinking);
    let resp = client.post(url).json(&body).send();
    match resp {
        Ok(r) if r.status().is_success() => {
            let value: Value = r
                .json()
                .context("failed to parse /v1/chat/completions response (guided_choice)")?;
            return Ok(extract_choice_content(&value).unwrap_or_default());
        }
        Ok(r) if r.status().is_client_error() => {
            log::debug!(
                "guided_choice rejected ({}); falling back to top_logprobs",
                r.status()
            );
        }
        Ok(r) => anyhow::bail!(
            "MCQ request failed: {} {}",
            r.status(),
            r.text().unwrap_or_default()
        ),
        Err(err) => return Err(anyhow::Error::from(err)).context("MCQ HTTP request failed"),
    }

    // Attempt 2: top_logprobs. Ask for one token + per-token top-K logprobs;
    // pick the listed choice with the highest log-prob.
    let top_k = (choices.len() * 2).max(5);
    let mut body = json!({
        "model": model,
        "messages": messages,
        "max_tokens": 1,
        "temperature": sampling.temperature.as_f64(),
        "logprobs": true,
        "top_logprobs": top_k,
    });
    apply_chat_template_kwargs(&mut body, sampling.enable_thinking);
    let resp = client
        .post(url)
        .json(&body)
        .send()
        .context("MCQ fallback HTTP request failed")?;
    if !resp.status().is_success() {
        anyhow::bail!(
            "MCQ top_logprobs request failed: {} {}",
            resp.status(),
            resp.text().unwrap_or_default()
        );
    }
    let value: Value = resp
        .json()
        .context("failed to parse /v1/chat/completions response (top_logprobs)")?;
    Ok(
        pick_choice_from_logprobs(&value, choices).unwrap_or_else(|| {
            log::warn!("could not match any choice from top_logprobs; returning empty");
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
        // Match the token (case-insensitive prefix) against any listed choice.
        if let Some(matched) = choices.iter().find(|c| {
            let c = c.trim();
            !c.is_empty() && token.eq_ignore_ascii_case(c)
        }) {
            if best.is_none_or(|(_, lp)| logprob > lp) {
                best = Some((matched.as_str(), logprob));
            }
        }
    }
    best.map(|(s, _)| s.to_string())
}

/// Streaming chat completion. Accumulates `choices[0].delta.content` across
/// SSE events and runs the doomloop pipeline every 50 chunks. Bails early if
/// a detector fires.
fn free_text_completion(
    client: &Client,
    url: &str,
    model: &str,
    messages: &Value,
    max_tokens: u64,
    sampling: SamplingParams,
    pipeline: &DoomloopPipeline,
) -> anyhow::Result<String> {
    // No `seed` is sent, so repeats under temperature > 0 are independent draws.
    let mut body = json!({
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": sampling.temperature.as_f64(),
        "stream": true,
    });
    apply_chat_template_kwargs(&mut body, sampling.enable_thinking);
    let response = client
        .post(url)
        .json(&body)
        .send()
        .context("failed to call /v1/chat/completions")?
        .error_for_status()
        .context("/v1/chat/completions failed")?;
    let reader = BufReader::new(response);
    let mut stderr = std::io::stderr().lock();
    let content = read_sse_chat(reader, pipeline, |pending| {
        let _ = write!(stderr, "{pending}");
        let _ = stderr.flush();
    })?;
    let _ = writeln!(stderr);
    let _ = stderr.flush();
    Ok(content)
}

/// Parse OpenAI-style SSE chat-completion events, accumulate
/// `choices[0].delta.content`. `[DONE]` terminates; doomloop pipeline
/// checked every 50 chunks.
fn read_sse_chat(
    reader: impl BufRead,
    pipeline: &DoomloopPipeline,
    mut on_flush: impl FnMut(&str),
) -> anyhow::Result<String> {
    let mut content = String::new();
    let mut pending = String::new();
    let mut chunks: u64 = 0;
    for line in reader.lines() {
        let line = line.context("failed to read SSE line")?;
        let Some(data) = line
            .strip_prefix("data: ")
            .or_else(|| line.strip_prefix("data:").map(str::trim_start))
        else {
            continue;
        };
        if data == "[DONE]" {
            break;
        }
        let event: Value = match serde_json::from_str(data) {
            Ok(v) => v,
            Err(e) => {
                log::warn!("failed to parse SSE data: {data}: {e}");
                continue;
            }
        };
        if let Some(chunk) = event
            .get("choices")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("delta"))
            .and_then(|d| d.get("content"))
            .and_then(Value::as_str)
        {
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
                log::warn!("{}", format_trigger_log(name, content.len()));
                break;
            }
        }
    }
    if !pending.is_empty() {
        on_flush(&pending);
    }
    Ok(content)
}

/// Streaming client: connect timeout only, no overall response timeout
/// (would otherwise kill a long SSE connection mid-decode).
fn streaming_http_client(connect_timeout: Duration) -> anyhow::Result<Client> {
    Ok(HttpClient::builder("pipette")
        .preconfigured_tls()
        .connect_timeout(connect_timeout)
        .no_request_timeout()
        .build()
        .context("failed to build streaming HTTP client")?
        .client()
        .clone())
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
    use std::io::Read;
    use std::net::TcpListener;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    use pipette_plan_types::benchmark::eval_id::EvalId;
    use pipette_plan_types::benchmark::{BenchmarkDefinition, EvalBenchmark};
    use pipette_plan_types::run::DeclaredBound;
    use pipette_plan_types::{Model, Runtime};

    use super::*;
    use crate::server::ServerKind;

    fn pipeline_disabled() -> DoomloopPipeline {
        DoomloopPipeline::disabled()
    }

    /// A chat-completions endpoint that answers `guided_choice` requests and can
    /// fail a nominated one with a 500 — the status `mcq_completion` bails on
    /// (a 4xx would fall through to its `top_logprobs` attempt instead).
    ///
    /// Accepts at most `requests` connections. The resume test gives itself
    /// headroom so an over-dispatching run is caught by the count assertion
    /// rather than by a connection error, which says far less; the surplus
    /// accept parks the thread until the process exits. `Connection: close` on
    /// every response keeps it to one request per accept, so the counter is the
    /// number of samples the eval actually dispatched.
    struct FakeChatServer {
        port: u16,
        served: Arc<AtomicUsize>,
    }

    impl FakeChatServer {
        fn spawn(requests: usize, fail_on: Option<usize>) -> anyhow::Result<Self> {
            let listener = TcpListener::bind("127.0.0.1:0")?;
            let port = listener.local_addr()?.port();
            let served = Arc::new(AtomicUsize::new(0));
            let counter = Arc::clone(&served);
            std::thread::spawn(move || {
                listener
                    .incoming()
                    .take(requests)
                    .filter_map(Result::ok)
                    .for_each(|stream| {
                        let n = counter.fetch_add(1, Ordering::SeqCst) + 1;
                        let _ = Self::answer(stream, fail_on == Some(n));
                    });
            });
            Ok(Self { port, served })
        }

        fn answer(mut stream: std::net::TcpStream, fail: bool) -> std::io::Result<()> {
            // Drain the request: replying without reading the body can surface as
            // a write error on the client instead of the status under test.
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
            std::io::copy(&mut reader.take(length as u64), &mut std::io::sink())?;

            let (status, body) = if fail {
                (
                    "500 Internal Server Error",
                    "{\"error\":\"injected\"}".to_string(),
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

        /// Any `ServerState` variant works: `eval::run` reads `base_url()` and
        /// nothing else off it, so the pid here is never consulted.
        fn state(&self) -> ServerState {
            ServerState::Uv {
                executable: std::path::PathBuf::from("python"),
                kind: ServerKind::Vllm,
                pid: 0,
                pgid: 0,
                port: self.port,
                log_tail: None,
            }
        }

        fn served(&self) -> usize {
            self.served.load(Ordering::SeqCst)
        }
    }

    /// Three MCQ samples, so the store is keyed by a stable portable identity
    /// across both runs below.
    fn mcq_request() -> RunRequest {
        let samples = ["s1", "s2", "s3"]
            .iter()
            .map(|id| json!({"id": id, "messages": [{"role": "user", "content": "q"}]}))
            .collect();
        let runtime = Runtime::AppleFoundation(Default::default());
        let model = Model::AppleFoundationText;
        RunRequest {
            runtime: DeclaredBound::already_bound(runtime),
            model: DeclaredBound::already_bound(model),
            runtime_flags: None,
            model_flags: None,
            benchmark_flags: None,
            benchmark: BenchmarkDefinition::Eval(EvalBenchmark {
                benchmark_id: "eval_resume".to_string(),
                parameter_eval_id: EvalId::from("math_500"),
                parameter_dataset_name: "d".to_string(),
                parameter_max_tokens: 8,
                parameter_mcq_choices: Some(vec!["A".to_string(), "B".to_string()]),
                samples: Some(samples),
            }),
        }
    }

    /// A sample that fails aborts the run — torch-oai propagates rather than
    /// recording `failed: true` — so the completions already appended are all
    /// that stand between a re-run and repeating the work. The second run must
    /// dispatch only the samples the first never finished.
    #[test]
    fn eval_resumes_from_the_checkpoint_after_a_failed_sample() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let evals = EvalCompletionsStore::new(dir.path());
        let req = mcq_request();

        let crashing = FakeChatServer::spawn(2, Some(2))?;
        let first = run(&req, "m", &crashing.state(), &evals);
        assert!(
            first.is_err(),
            "sample 2 returned 500; run must not succeed"
        );
        assert_eq!(crashing.served(), 2, "should stop at the failing sample");

        let healthy = FakeChatServer::spawn(3, None)?;
        let resumed = run(&req, "m", &healthy.state(), &evals)?;
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
        Ok(())
    }

    #[test]
    fn read_sse_chat_accumulates_content() -> anyhow::Result<()> {
        let stream = "data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n\
                      data: {\"choices\":[{\"delta\":{\"content\":\" world\"}}]}\n\
                      data: [DONE]\n";
        let p = pipeline_disabled();
        let out = read_sse_chat(stream.as_bytes(), &p, |_| {})?;
        assert_eq!(out, "Hello world");
        Ok(())
    }

    #[test]
    fn read_sse_chat_skips_non_data_lines() -> anyhow::Result<()> {
        let stream = "event: ping\n\
                      data: {\"choices\":[{\"delta\":{\"content\":\"X\"}}]}\n\
                      data: [DONE]\n";
        let p = pipeline_disabled();
        let out = read_sse_chat(stream.as_bytes(), &p, |_| {})?;
        assert_eq!(out, "X");
        Ok(())
    }

    #[test]
    fn read_sse_chat_terminates_on_done() -> anyhow::Result<()> {
        let stream = "data: {\"choices\":[{\"delta\":{\"content\":\"A\"}}]}\n\
                      data: [DONE]\n\
                      data: {\"choices\":[{\"delta\":{\"content\":\"B\"}}]}\n";
        let p = pipeline_disabled();
        let out = read_sse_chat(stream.as_bytes(), &p, |_| {})?;
        assert_eq!(out, "A");
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
    fn apply_chat_template_kwargs_omits_field_when_unset() {
        let mut body = json!({"model": "m", "messages": []});
        apply_chat_template_kwargs(&mut body, None);
        assert!(body.get("chat_template_kwargs").is_none());
    }

    #[test]
    fn apply_chat_template_kwargs_injects_enable_thinking_true() {
        let mut body = json!({"model": "m", "messages": []});
        apply_chat_template_kwargs(&mut body, Some(true));
        assert_eq!(
            body["chat_template_kwargs"],
            json!({"enable_thinking": true})
        );
    }

    #[test]
    fn apply_chat_template_kwargs_injects_enable_thinking_false() {
        let mut body = json!({"model": "m", "messages": []});
        apply_chat_template_kwargs(&mut body, Some(false));
        assert_eq!(
            body["chat_template_kwargs"],
            json!({"enable_thinking": false})
        );
    }
}
