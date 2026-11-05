//! VLM accuracy eval (`vlm_eval`) against vLLM's OpenAI-compatible server.
//!
//! The cell the llama.cpp runner runs, on a Hugging Face repo's own weights:
//! the entry's corpus and recipe ([`cell_rules`]), each sample sent as its
//! `messages`, images included, to `/v1/chat/completions`, and every sample
//! answered or the cell refused ([`measurement::eval_complete()`]). What vLLM
//! does itself is not repeated: it holds each sample's context up to the
//! model's own (or the cell's `max_model_len`) and batches the samples in
//! flight, so there is no calibration and no slot arithmetic.
//!
//! - **Sampling** is the entry's, every knob it leaves out at its neutral
//!   value, on a server started at `--generation-config vllm`, so the repo's
//!   `generation_config.json` sets no default behind the recipe.
//! - **Thinking** is the request's `chat_template_kwargs`, when the model's
//!   chat template has a switch ([`thinking_switch`]); the cell's declared mode
//!   is held to the template exactly as llama.cpp's is.
//! - **The image budget** (`parameter_image_tokens`) is the request's
//!   `mm_processor_kwargs` `max_soft_tokens`, checked against the image tokens
//!   the server reports before any sample runs ([`check_image_budget`]).
//! - **The reply** is the model's text as written, special tokens included,
//!   as llama.cpp returns it. The plan types reserve vLLM's reasoning parser;
//!   reasoning a server streams apart anyway goes back in front of the
//!   answer, closed by `</think>`, as on the text eval.

use std::io::BufReader;
use std::num::NonZeroUsize;

use anyhow::Context;
use base64::Engine as _;
use reqwest::Method;
use serde_json::{Value, json};

use pipette_http::{HttpClient, StreamClient};
use pipette_ops::eval_completions::EvalCompletionsStore;
use pipette_ops::measurement::{
    self, EvalMeasurementDataset, EvalMeasurementInFlight, EvalMeasurementReading,
    EvalMeasurementSample,
};
use pipette_ops::run_response::RunResponseBuilder;
use pipette_ops::sse::{self, ChatReply};
use pipette_plan_types::{
    BenchmarkEvalCompletionStopReason, EvalSample, MinP, ModelFlags, PositiveCount,
    PresencePenalty, RepeatPenalty, RunRequest, RuntimeFlags, TopP, VlmEvalBenchmark,
};
use pipette_vl_spec_decode_corpus::vlm_eval::{self as cell_rules, VlmEvalCorpus};

use crate::server::{ServerKind, ServerState};

/// The multimodal chat endpoint every sample goes to.
const CHAT_ENDPOINT: &str = "/v1/chat/completions";
/// vLLM's tokenizer, which renders a chat through the model's template (and
/// expands each image into its tokens) without generating.
const TOKENIZE_ENDPOINT: &str = "/tokenize";
/// The images the budget check sends: a large square and a small, tall one.
/// A processor that sizes every image to the budget gives both the budget; one
/// that sizes by pixels gives them counts far apart.
const BUDGET_PROBES: [(u32, u32); 2] = [(1024, 1024), (256, 512)];
/// Tokens an image adds beyond its soft tokens (begin/end markers, a line
/// break): room the budget check allows above the budget.
const IMAGE_MARKER_TOKENS: usize = 16;

/// What a cell is, once everything that needs no server has been checked:
/// `run_with` checks it before the launch, so a refusal costs no model load,
/// and [`run`] again for what it reads.
pub(super) struct Cell<'a> {
    samples: &'a [EvalSample],
    enable_thinking: Option<bool>,
    prompt_thinks: bool,
    max_tokens: u32,
    workers: NonZeroUsize,
}

pub(super) fn preflight<'a>(
    req: &RunRequest,
    benchmark: &'a VlmEvalBenchmark,
    corpus: Option<&'a VlmEvalCorpus>,
    flags: &RuntimeFlags,
) -> anyhow::Result<Cell<'a>> {
    let benchmark_id = benchmark.benchmark_id.as_str();
    // The plan types define the cell on uv vLLM alone; this is the message
    // for a run that reached here some other way.
    let RuntimeFlags::VlmEvalUvVllmTorch {
        parallel,
        max_model_len,
        ..
    } = flags
    else {
        anyhow::bail!(
            "vlm_eval `{benchmark_id}` runs on uv vLLM alone here, with a repo's own weights"
        );
    };
    let samples = cell_rules::samples_of(benchmark, corpus)?;
    cell_rules::check_samples(benchmark, samples)?;
    let enable_thinking = req
        .model_flags
        .as_ref()
        .and_then(ModelFlags::enable_thinking);
    let prompt_thinks = cell_rules::prompt_thinks(benchmark, enable_thinking)?;
    let max_tokens = benchmark.parameter_max_tokens.into_inner().unsigned_abs();
    check_room(benchmark_id, *max_model_len, max_tokens)?;
    // Constrained answers have no vLLM arm yet: refused rather than answered
    // free-form under an entry that asks for one choice.
    anyhow::ensure!(
        benchmark
            .parameter_mcq_choices
            .as_deref()
            .is_none_or(<[String]>::is_empty),
        "vlm_eval `{benchmark_id}` constrains its answers to listed choices \
         (parameter_mcq_choices), which the vLLM runner does not apply; run it on llama.cpp"
    );
    // More workers than samples would only ever claim an exhausted list.
    let workers = parallel
        .map_or(Ok(1), |n| usize::try_from(n.into_inner()))?
        .clamp(1, samples.len().max(1));
    Ok(Cell {
        samples,
        enable_thinking,
        prompt_thinks,
        max_tokens,
        workers: NonZeroUsize::new(workers).unwrap_or(NonZeroUsize::MIN),
    })
}

pub(super) fn run(
    req: &RunRequest,
    benchmark: &VlmEvalBenchmark,
    corpus: Option<&VlmEvalCorpus>,
    model: &str,
    state: &ServerState,
    eval_completions: &EvalCompletionsStore,
    flags: &RuntimeFlags,
) -> anyhow::Result<RunResponseBuilder> {
    let benchmark_id = benchmark.benchmark_id.as_str();
    anyhow::ensure!(
        matches!(state.kind, ServerKind::Vllm),
        "vlm_eval `{benchmark_id}` runs on vLLM; this server is not one"
    );
    let Cell {
        samples,
        enable_thinking,
        prompt_thinks,
        max_tokens,
        workers,
    } = preflight(req, benchmark, corpus, flags)?;

    let request_timeout = measurement::http_timeout(req, 1);
    let idle_timeout = measurement::stream_idle_timeout(req);
    let base_url = state.base_url();
    let http = HttpClient::loopback("pipette", request_timeout)?;
    let streaming = HttpClient::streaming("pipette", idle_timeout)
        .context("failed to build streaming HTTP client")?;

    // Before any sample: a mode the template cannot honour, or an image budget
    // the processor ignores, would score settings the recipe never had.
    let switch = thinking_switch(&http, &base_url, model)?;
    let thinking = thinking_kwarg(enable_thinking, switch, prompt_thinks)
        .map_err(|refused| refused.error(model))?;
    let mm_processor_kwargs = benchmark
        .parameter_image_tokens
        .map(|tokens| json!({ "max_soft_tokens": tokens.into_inner() }));
    if let Some(budget) = benchmark.parameter_image_tokens {
        check_image_budget(&http, &base_url, model, budget)?;
    }
    tracing::info!(
        benchmark_id,
        thinking_switch = switch,
        enable_thinking = ?enable_thinking,
        prompt_thinks,
        sent_enable_thinking = ?thinking,
        image_tokens = ?benchmark.parameter_image_tokens,
        max_tokens,
        "vlm_eval request settings"
    );

    let chat_url = format!("{base_url}{CHAT_ENDPOINT}");
    let request = Request {
        benchmark,
        model,
        max_tokens,
        thinking,
        mm_processor_kwargs: mm_processor_kwargs.as_ref(),
    };
    let result_data = measurement::eval_complete(
        EvalMeasurementDataset::vlm_eval(benchmark, samples),
        EvalMeasurementInFlight::Parallel(workers),
        eval_completions.open(req)?,
        (),
        |_, samples, record| {
            samples.iter().enumerate().try_for_each(|(at, sample)| {
                record(at, read_sample(&streaming, &chat_url, &request, sample))
            })?;
            Ok(())
        },
    )?;

    let mut command = vec![
        format!("POST {chat_url}"),
        format!("model={model} max_tokens={max_tokens} stream=true skip_special_tokens=false"),
    ];
    if let Some(thinking) = thinking {
        command.push(format!(
            "chat_template_kwargs={{\"enable_thinking\":{thinking}}}"
        ));
    }
    if let Some(kwargs) = &mm_processor_kwargs {
        command.push(format!("mm_processor_kwargs={kwargs}"));
    }
    let mut response = RunResponseBuilder::new(result_data)
        .executable(state.executable.display().to_string())
        .command(command);
    // Which corpus the answers came from, as the row records it.
    if let Some(corpus) = corpus {
        response = response.dataset_digest(corpus.digest().clone());
    }
    Ok(response)
}

/// A pinned context at or under the output budget leaves no room for a
/// prompt: vLLM refuses (HTTP 400) every request whose prompt plus
/// `max_tokens` exceeds `max_model_len`. Unpinned, the model's own context
/// holds them.
fn check_room(
    benchmark_id: &str,
    max_model_len: Option<PositiveCount>,
    max_tokens: u32,
) -> anyhow::Result<()> {
    if let Some(len) = max_model_len {
        anyhow::ensure!(
            len.into_inner().unsigned_abs() > max_tokens,
            "vlm_eval `{benchmark_id}` pins max_model_len={len}, which is no more than its \
             parameter_max_tokens={max_tokens}: vLLM would refuse every sample; leave it unset \
             (the model's own context) or raise it past the output budget"
        );
    }
    Ok(())
}

/// What every sample's request shares.
struct Request<'a> {
    benchmark: &'a VlmEvalBenchmark,
    model: &'a str,
    max_tokens: u32,
    thinking: Option<bool>,
    mm_processor_kwargs: Option<&'a Value>,
}

impl Request<'_> {
    /// The streamed chat request for one sample: the entry's sampling, every
    /// knob it leaves out neutral (`top_k` `-1` is vLLM's "no cut"), its
    /// per-sample seed when it pins one.
    ///
    /// The text keeps the special tokens the model wrote, as llama.cpp's
    /// does: Gemma 4's thought channel is one (`<|channel>thought…<channel|>`),
    /// and the scorer finds the reasoning by those markers. vLLM drops them by
    /// default, which would leave the reasoning in the answer it grades.
    fn body(&self, sample: &EvalMeasurementSample<'_>) -> Value {
        let benchmark = self.benchmark;
        let mut body = json!({
            "model": self.model,
            "messages": sample.messages,
            "max_tokens": self.max_tokens,
            "stream": true,
            "stream_options": { "include_usage": true },
            "temperature": benchmark.sampling_temperature().as_f64(),
            "top_p": benchmark.parameter_top_p.map_or(1.0, TopP::as_f64),
            "top_k": benchmark
                .parameter_top_k
                .map(i32::from)
                .filter(|&k| k > 0)
                .unwrap_or(-1),
            "min_p": benchmark.parameter_min_p.map_or(0.0, MinP::as_f64),
            "presence_penalty": benchmark
                .parameter_presence_penalty
                .map_or(0.0, PresencePenalty::as_f64),
            "frequency_penalty": 0.0,
            "repetition_penalty": benchmark
                .parameter_repeat_penalty
                .map_or(1.0, RepeatPenalty::as_f64),
            "skip_special_tokens": false,
        });
        if let Some(fields) = body.as_object_mut() {
            if let Some(seed) = benchmark.sample_seed(sample.id) {
                fields.insert("seed".to_owned(), json!(seed));
            }
            if let Some(thinking) = self.thinking {
                fields.insert(
                    "chat_template_kwargs".to_owned(),
                    json!({ "enable_thinking": thinking }),
                );
            }
            if let Some(kwargs) = self.mm_processor_kwargs {
                fields.insert("mm_processor_kwargs".to_owned(), kwargs.clone());
            }
        }
        body
    }
}

/// One attempt at a sample. A request the server refused (a 4xx, whose body
/// says why) is the sample's own failure. One that broke on the way (a reset,
/// a stream cut short, its silence outlasting the idle bound, an overloaded
/// server's 429 or 5xx) is lost, as
/// llama.cpp files a transport break: sent again alone, and only then failed,
/// so a blip beside a pool of samples does not spend their attempts. Either
/// way the shared entry sends a failed sample again on a later run while it
/// has attempts left, and refuses the cell if it never answers.
fn read_sample(
    client: &StreamClient,
    url: &str,
    request: &Request<'_>,
    sample: &EvalMeasurementSample<'_>,
) -> EvalMeasurementReading {
    let reading = match stream_chat(client, url, &request.body(sample)) {
        Ok(reply) => {
            let (stop_reason, stop_detail) = reply.stop();
            let completion_tokens = reply.completion_tokens;
            match cut_short(stop_reason, completion_tokens, request.max_tokens) {
                Some(reason) => EvalMeasurementReading::Failed { reason },
                None => EvalMeasurementReading::Completed {
                    completion: reply.completion_with_reasoning(),
                    stop_reason,
                    stop_detail,
                    completion_tokens,
                },
            }
        }
        Err(err) if refused(&err) => EvalMeasurementReading::Failed {
            reason: format!("{err:#}"),
        },
        Err(err) => EvalMeasurementReading::Lost {
            reason: format!("{err:#}"),
        },
    };
    match &reading {
        EvalMeasurementReading::Completed {
            completion,
            stop_reason,
            completion_tokens,
            ..
        } => tracing::info!(
            sample = sample.position.as_str(),
            sample_id = %sample.id,
            completion_chars = completion.chars().count(),
            ?stop_reason,
            ?completion_tokens,
            "sample completed"
        ),
        EvalMeasurementReading::Failed { reason }
        | EvalMeasurementReading::Lost { reason }
        | EvalMeasurementReading::Unfit { reason } => {
            tracing::warn!(
                sample = sample.position.as_str(),
                sample_id = %sample.id,
                reason = reason.as_str(),
                "sample failed"
            );
        }
    }
    reading
}

/// Why a reply cut at a length bound is not an answer: the bound was the
/// context, not the recipe's output budget. vLLM from 0.32 treats `max_tokens`
/// as an upper bound rather than reserving it, so a prompt that leaves less
/// room than the budget is generated into the context's end and reported as
/// `length`, where an earlier vLLM refused it. Either way the sample gets no
/// answer, as on llama.cpp, whose slots hold every prompt plus the whole
/// budget: a cut answer is one that ran the full `max_tokens`.
fn cut_short(
    stop_reason: BenchmarkEvalCompletionStopReason,
    completion_tokens: Option<u64>,
    max_tokens: u32,
) -> Option<String> {
    if stop_reason != BenchmarkEvalCompletionStopReason::Truncated
        || completion_tokens == Some(u64::from(max_tokens))
    {
        return None;
    }
    Some(match completion_tokens {
        Some(tokens) => format!(
            "the reply was cut after {tokens} of its {max_tokens} output tokens: the server's \
             context (max_model_len) ran out first, which leaves no room for the recipe's \
             output budget; raise max_model_len or leave it unset"
        ),
        None => format!(
            "the reply was cut at a length bound and the stream reported no token count, so \
             whether it ran its {max_tokens} output tokens cannot be told"
        ),
    })
}

/// Whether the server refused the request for good: a status that is not an
/// overload (a 4xx but 429). A 429 or a 5xx is the server's state, not the
/// sample's, and is sent again as a transport break is.
fn refused(err: &anyhow::Error) -> bool {
    err.chain().any(|cause| {
        cause
            .downcast_ref::<pipette_http::Error>()
            .is_some_and(|err| {
                matches!(err, pipette_http::Error::Status { .. })
                    && matches!(
                        err.disposition(),
                        pipette_http::retry::Disposition::Permanent
                    )
            })
    })
}

/// A streamed chat completion, read to its finish chunk. A stream that ends
/// without one is an error, not an answer: the text was cut short, and
/// scoring it would publish a truncated reply as complete.
fn stream_chat(client: &StreamClient, url: &str, body: &Value) -> anyhow::Result<ChatReply> {
    let response = pipette_http::send(client.http().post(url).json(body))
        .with_context(|| format!("failed to call {CHAT_ENDPOINT}"))?;
    let events = sse::data_lines(BufReader::new(response), client.idle_timeout());
    let reply = sse::read_chat_stream(events, &mut |_| {})?;
    anyhow::ensure!(
        reply.finish_reason.is_some(),
        "{CHAT_ENDPOINT} stream ended without finish_reason (connection closed early?); \
         discarding {} chars of partial content",
        reply.content.len()
    );
    Ok(reply)
}

/// The token ids vLLM renders `body` to through the model's chat template.
fn tokenize(http: &HttpClient, base_url: &str, body: Value) -> anyhow::Result<Vec<Value>> {
    let reply: Value = http
        .json_request(
            Method::POST,
            &format!("{base_url}{TOKENIZE_ENDPOINT}"),
            None,
            Some(body),
        )
        .with_context(|| format!("POST {TOKENIZE_ENDPOINT}"))?;
    reply
        .get("tokens")
        .and_then(Value::as_array)
        .cloned()
        .with_context(|| format!("{TOKENIZE_ENDPOINT} replied without `tokens`"))
}

/// Whether the model's chat template has a thinking switch: a probe renders
/// differently with `enable_thinking` false than true. Two probes, as on
/// llama.cpp: a template may switch only once a system prompt is present.
fn thinking_switch(http: &HttpClient, base_url: &str, model: &str) -> anyhow::Result<bool> {
    let conversations = [
        json!([{"role": "user", "content": "Hi"}]),
        json!([
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hi"},
        ]),
    ];
    for messages in &conversations {
        let render = |enable_thinking: bool| {
            tokenize(
                http,
                base_url,
                json!({
                    "model": model,
                    "messages": messages,
                    "add_generation_prompt": true,
                    "chat_template_kwargs": { "enable_thinking": enable_thinking },
                }),
            )
        };
        let (on, off, again) = (render(true)?, render(false)?, render(true)?);
        anyhow::ensure!(
            on == again,
            "the chat template rendered the same probe differently with thinking on twice, so \
             whether it has a thinking switch cannot be told; it likely prints something that \
             changes per render"
        );
        if on != off {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Why a cell's declared thinking mode cannot run on its model's template:
/// llama.cpp's rule ([`pipette_llamacpp`'s `server::thinking`]), so a recipe
/// reads the same on every engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ThinkingRefused {
    /// The template has a switch and the cell did not say which way.
    Unstated,
    /// Thinking asked of a template with no switch and no thinking prompt.
    NoSwitch,
    /// `false` asked of a model with no reasoning mode to turn off.
    NoMode,
}

impl ThinkingRefused {
    fn error(self, model: &str) -> anyhow::Error {
        match self {
            Self::Unstated => anyhow::anyhow!(
                "{model} has a thinking switch in its chat template, so the cell must say which \
                 mode it scores: set its model flags' enable_thinking (true runs its reasoning, \
                 false turns it off)"
            ),
            Self::NoSwitch => anyhow::anyhow!(
                "{model} has no thinking switch in its chat template, so enable_thinking = true \
                 would score it without thinking: leave enable_thinking unset, or run a template \
                 that has the switch"
            ),
            Self::NoMode => anyhow::anyhow!(
                "{model} has no thinking switch in its chat template, so it has no reasoning mode \
                 to turn off: leave enable_thinking unset and it runs as its template renders"
            ),
        }
    }
}

/// The `enable_thinking` each request sends, if any. A template with a switch
/// takes the cell's mode, which must be stated; one without takes none, unless
/// the prompt itself is the model's thinking mode (InternVL3.5's R1 prompt).
fn thinking_kwarg(
    enable_thinking: Option<bool>,
    thinking_switch: bool,
    prompt_thinks: bool,
) -> Result<Option<bool>, ThinkingRefused> {
    match (enable_thinking, thinking_switch) {
        (Some(mode), true) => Ok(Some(mode)),
        (None, true) => Err(ThinkingRefused::Unstated),
        (None, false) => Ok(None),
        (Some(true), false) if prompt_thinks => Ok(None),
        (Some(true), false) => Err(ThinkingRefused::NoSwitch),
        (Some(false), false) => Err(ThinkingRefused::NoMode),
    }
}

/// That the server's image processor takes the recipe's budget: an image is
/// as many tokens as the budget (and a few markers) whatever its size, and a
/// different number at another budget. A processor that ignores
/// `max_soft_tokens` (a family that sizes images by their pixels, or to a
/// fixed count) refuses the cell rather than score images sized off the
/// recipe.
fn check_image_budget(
    http: &HttpClient,
    base_url: &str,
    model: &str,
    budget: PositiveCount,
) -> anyhow::Result<()> {
    let want = usize::try_from(budget.into_inner())?;
    // Half the budget, or twice the smallest: Gemma 4 takes 70 to 1,120 in
    // doublings, so the contrast is one of its budgets too.
    let contrast = if want >= 140 { want / 2 } else { want * 2 };
    let text = json!({"type": "text", "text": "Describe the image."});
    let count = |content: Value, budget: Option<usize>| -> anyhow::Result<usize> {
        let mut body = json!({
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "add_generation_prompt": true,
        });
        if let (Some(fields), Some(budget)) = (body.as_object_mut(), budget) {
            fields.insert(
                "mm_processor_kwargs".to_owned(),
                json!({ "max_soft_tokens": budget }),
            );
        }
        Ok(tokenize(http, base_url, body)?.len())
    };
    let without_image = count(json!([text.clone()]), None)?;
    for (width, height) in BUDGET_PROBES {
        let png = pipette_ops::png::gray(width, height)?;
        let image = json!({
            "type": "image_url",
            "image_url": {
                "url": format!(
                    "data:image/png;base64,{}",
                    base64::engine::general_purpose::STANDARD.encode(png)
                ),
            },
        });
        let image_tokens = |budget| -> anyhow::Result<usize> {
            Ok(count(json!([image.clone(), text.clone()]), Some(budget))?
                .saturating_sub(without_image))
        };
        let (tokens, at_contrast) = (image_tokens(want)?, image_tokens(contrast)?);
        tracing::info!(
            model,
            budget = want,
            width,
            height,
            tokens,
            contrast,
            at_contrast,
            "vlm_eval image budget check"
        );
        anyhow::ensure!(
            tokens >= want * 4 / 5 && tokens <= want + IMAGE_MARKER_TOKENS && at_contrast != tokens,
            "{model}'s image processor does not take the recipe's image budget: a {width}x{height} \
             image is {tokens} tokens with max_soft_tokens={want} and {at_contrast} with \
             {contrast}, so the cell would score images sized off its recipe"
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::io::{BufRead, Read, Write};
    use std::net::{TcpListener, TcpStream};
    use std::sync::{Arc, Mutex, PoisonError};

    use rstest::rstest;

    use pipette_plan_types::{
        AbsolutePath, BenchmarkDefinition, BenchmarkId, BenchmarkResultData, DeclaredBound,
        EvalSample, Model, PositiveCount, Runtime, SamplingSeed, SamplingTemperature, TopK,
        UvBuild, UvPythonVersion, UvRuntimeSource, UvServerVersion, UvVllm,
    };

    use super::*;

    /// A reply the fake server sends.
    enum Reply {
        Json(Value),
        Stream(Vec<Value>),
        Status(&'static str, Value),
    }

    type Handler = dyn Fn(&str, &Value) -> Reply + Send + Sync;

    /// A vLLM stand-in: each request's path and body go to `handler`, and are
    /// kept for the test to read.
    struct FakeVllm {
        port: u16,
        seen: Arc<Mutex<Vec<(String, Value)>>>,
    }

    impl FakeVllm {
        fn spawn(
            handler: impl Fn(&str, &Value) -> Reply + Send + Sync + 'static,
        ) -> anyhow::Result<Self> {
            let listener = TcpListener::bind("127.0.0.1:0")?;
            let port = listener.local_addr()?.port();
            let seen = Arc::new(Mutex::new(Vec::new()));
            let (log, handler): (_, Arc<Handler>) = (Arc::clone(&seen), Arc::new(handler));
            std::thread::spawn(move || {
                for stream in listener.incoming().filter_map(Result::ok) {
                    let (log, handler) = (Arc::clone(&log), Arc::clone(&handler));
                    std::thread::spawn(move || {
                        let _ = Self::answer(stream, &log, handler.as_ref());
                    });
                }
            });
            Ok(Self { port, seen })
        }

        fn answer(
            mut stream: TcpStream,
            log: &Mutex<Vec<(String, Value)>>,
            handler: &Handler,
        ) -> std::io::Result<()> {
            let mut reader = std::io::BufReader::new(stream.try_clone()?);
            let mut request_line = String::new();
            reader.read_line(&mut request_line)?;
            let path = request_line
                .split_whitespace()
                .nth(1)
                .unwrap_or_default()
                .to_owned();
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
            let mut raw = String::new();
            reader.take(length as u64).read_to_string(&mut raw)?;
            let body: Value = serde_json::from_str(&raw).unwrap_or(Value::Null);
            let reply = handler(&path, &body);
            log.lock()
                .unwrap_or_else(PoisonError::into_inner)
                .push((path, body));
            let (status, content_type, text) = match reply {
                Reply::Json(value) => ("200 OK", "application/json", value.to_string()),
                Reply::Stream(chunks) => (
                    "200 OK",
                    "text/event-stream",
                    chunks
                        .iter()
                        .map(|chunk| format!("data: {chunk}\n\n"))
                        .chain(std::iter::once("data: [DONE]\n\n".to_owned()))
                        .collect(),
                ),
                Reply::Status(status, value) => (status, "application/json", value.to_string()),
            };
            write!(
                stream,
                "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\n\
                 Content-Length: {}\r\nConnection: close\r\n\r\n{text}",
                text.len()
            )?;
            stream.flush()
        }

        fn state(&self) -> ServerState {
            ServerState {
                kind: ServerKind::Vllm,
                port: self.port,
                executable: std::path::PathBuf::from("vllm"),
            }
        }

        /// The bodies sent to `path`, in arrival order.
        fn bodies(&self, path: &str) -> Vec<Value> {
            self.seen
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .iter()
                .filter(|(at, _)| at == path)
                .map(|(_, body)| body.clone())
                .collect()
        }
    }

    /// A template's rendering of a probe: one token more with thinking on
    /// when it has a switch.
    fn tokens(body: &Value, switch: bool) -> Reply {
        let thinking = body
            .pointer("/chat_template_kwargs/enable_thinking")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let image = body.to_string().contains("image_url");
        let budgeted = body.pointer("/mm_processor_kwargs/max_soft_tokens");
        let count = 10
            + usize::from(switch && thinking)
            + match (image, budgeted.and_then(Value::as_u64)) {
                (false, _) => 0,
                (true, Some(budget)) => usize::try_from(budget).unwrap_or(0) + 2,
                (true, None) => 258,
            };
        Reply::Json(json!({"count": count, "tokens": vec![1; count]}))
    }

    /// A reasoning parser's stream: the reasoning apart from the answer.
    fn answer() -> Reply {
        Reply::Stream(vec![
            json!({"choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}}]}),
            json!({"choices": [{"index": 0, "delta": {"reasoning": "Look."}}]}),
            json!({"choices": [{"index": 0, "delta": {"content": "B"}, "finish_reason": "stop"}]}),
            json!({"choices": [], "usage": {"prompt_tokens": 9, "completion_tokens": 3}}),
        ])
    }

    /// A vLLM whose template has a thinking switch or not, answering every sample.
    fn server(switch: bool) -> anyhow::Result<FakeVllm> {
        FakeVllm::spawn(move |path, body| match path {
            TOKENIZE_ENDPOINT => tokens(body, switch),
            _ => answer(),
        })
    }

    fn sample(id: &str) -> anyhow::Result<EvalSample> {
        Ok(serde_json::from_value(json!({
            "id": id,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                {"type": "text", "text": "Which?"},
            ]}],
        }))?)
    }

    fn benchmark(ids: &[&str]) -> anyhow::Result<VlmEvalBenchmark> {
        Ok(VlmEvalBenchmark {
            benchmark_id: BenchmarkId::try_new("vlm_eval_mmstar_vllm".to_owned())?,
            parameter_lmms_task: "mmstar".into(),
            parameter_dataset_version: "2026.10.1".into(),
            parameter_sample_count: PositiveCount::try_new(i32::try_from(ids.len())?)?,
            parameter_max_tokens: PositiveCount::try_new(64)?,
            parameter_mcq_choices: None,
            parameter_temperature: None,
            parameter_top_p: None,
            parameter_top_k: None,
            parameter_min_p: None,
            parameter_presence_penalty: None,
            parameter_repeat_penalty: None,
            parameter_seed: None,
            parameter_seed_rule: None,
            parameter_prompt_profile: None,
            parameter_image_tokens: None,
            dataset: None,
            recipe: None,
            samples: Some(
                ids.iter()
                    .map(|id| sample(id))
                    .collect::<anyhow::Result<_>>()?,
            ),
        })
    }

    fn flags(parallel: Option<i32>, max_model_len: Option<i32>) -> anyhow::Result<RuntimeFlags> {
        Ok(RuntimeFlags::VlmEvalUvVllmTorch {
            tensor_parallel_size: None,
            dtype: None,
            max_model_len: max_model_len.map(PositiveCount::try_new).transpose()?,
            prefix_caching: Some(false),
            parallel: parallel.map(PositiveCount::try_new).transpose()?,
            envs: Vec::new(),
            raw: Vec::new(),
        })
    }

    fn request(
        benchmark: &VlmEvalBenchmark,
        enable_thinking: Option<bool>,
    ) -> anyhow::Result<RunRequest> {
        let model: Model = serde_json::from_value(json!({
            "type": "torch", "source": "huggingface", "org": "Qwen", "repo_name": "Qwen3.5-2B",
        }))?;
        let runtime = Runtime::UvVllm(UvVllm {
            server_version: UvServerVersion::try_new("0.31.0".to_owned())?,
            build: UvBuild::try_new("rocm723".to_owned())?,
            python_version: UvPythonVersion::try_new("3.12".to_owned())?,
            source: UvRuntimeSource::AbsolutePreinstalled {
                dir: AbsolutePath::try_new("/venv".to_owned())?,
            },
        });
        Ok(RunRequest {
            runtime: DeclaredBound::already_bound(runtime),
            model: DeclaredBound::already_bound(model),
            runtime_flags: Some(flags(None, None)?),
            model_flags: enable_thinking.map(|enable_thinking| ModelFlags::VlmEvalTorch {
                enable_thinking: Some(enable_thinking),
            }),
            benchmark_flags: None,
            benchmark: BenchmarkDefinition::VlmEval(benchmark.clone()),
        })
    }

    /// Run `benchmark` against `server` in a fresh checkpoint.
    fn run_on(
        server: &FakeVllm,
        benchmark: &VlmEvalBenchmark,
        enable_thinking: Option<bool>,
        flags: &RuntimeFlags,
    ) -> anyhow::Result<RunResponseBuilder> {
        let dir = tempfile::tempdir()?;
        let evals = EvalCompletionsStore::new(dir.path());
        let req = request(benchmark, enable_thinking)?;
        run(&req, benchmark, None, "m", &server.state(), &evals, flags)
    }

    /// [`run_on`] with no mode, and the rows its checkpoint kept, which say
    /// why a refused cell's samples failed.
    fn refused_rows(
        server: &FakeVllm,
        benchmark: &VlmEvalBenchmark,
        flags: &RuntimeFlags,
    ) -> anyhow::Result<(
        anyhow::Result<RunResponseBuilder>,
        Vec<pipette_plan_types::BenchmarkEvalCompletion>,
    )> {
        let dir = tempfile::tempdir()?;
        let evals = EvalCompletionsStore::new(dir.path());
        let req = request(benchmark, None)?;
        let outcome = run(&req, benchmark, None, "m", &server.state(), &evals, flags);
        Ok((outcome, evals.open(&req)?.into_completions()))
    }

    /// Every sample answered once, its reasoning back in front of its answer
    /// as the scorer strips it, and each request carrying the entry's sampling
    /// with every knob it left out at its neutral value.
    #[test]
    fn every_sample_is_answered_with_the_entrys_sampling() -> anyhow::Result<()> {
        let server = server(true)?;
        let pinned = VlmEvalBenchmark {
            parameter_temperature: Some(SamplingTemperature::try_new(0.6)?),
            parameter_top_k: Some(TopK::try_new(20)?),
            parameter_seed: Some(SamplingSeed::try_new(0)?),
            ..benchmark(&["a", "b", "c"])?
        };
        let out = run_on(&server, &pinned, Some(true), &flags(Some(2), None)?)?.build();
        let BenchmarkResultData::Eval { completions } = out.result_data else {
            anyhow::bail!("expected Eval result data");
        };
        let mut ids: Vec<_> = completions.iter().map(|c| c.id.as_str()).collect();
        ids.sort_unstable();
        assert_eq!(ids, ["a", "b", "c"]);
        assert!(
            completions
                .iter()
                .all(|c| !c.failed && c.completion == "Look.</think>B"),
            "{completions:?}"
        );
        let bodies = server.bodies(CHAT_ENDPOINT);
        assert_eq!(bodies.len(), 3);
        for body in &bodies {
            let knobs: Value = [
                "temperature",
                "top_p",
                "top_k",
                "min_p",
                "presence_penalty",
                "frequency_penalty",
                "repetition_penalty",
                "max_tokens",
                "stream",
                "skip_special_tokens",
                "chat_template_kwargs",
            ]
            .into_iter()
            .map(|key| (key.to_owned(), body[key].clone()))
            .collect::<serde_json::Map<_, _>>()
            .into();
            assert_eq!(
                knobs,
                json!({
                    "temperature": 0.6, "top_p": 1.0, "top_k": 20, "min_p": 0.0,
                    "presence_penalty": 0.0, "frequency_penalty": 0.0,
                    "repetition_penalty": 1.0, "max_tokens": 64, "stream": true,
                    "skip_special_tokens": false,
                    "chat_template_kwargs": {"enable_thinking": true},
                })
            );
            assert!(body["seed"].is_i64(), "a pinned seed goes out per sample");
            assert!(body.get("mm_processor_kwargs").is_none());
        }
        Ok(())
    }

    /// The declared mode against the template, before any sample is sent: a
    /// refusal costs no generation.
    #[rstest]
    #[case::unstated_on_a_switch(true, None, Some("must say which mode"))]
    #[case::thinking_without_a_switch(false, Some(true), Some("no thinking switch"))]
    #[case::off_without_a_mode(false, Some(false), Some("no reasoning mode"))]
    #[case::as_rendered(false, None, None)]
    fn the_template_decides_what_is_sent(
        #[case] switch: bool,
        #[case] enable_thinking: Option<bool>,
        #[case] refused: Option<&str>,
    ) -> anyhow::Result<()> {
        let server = server(switch)?;
        let outcome = run_on(
            &server,
            &benchmark(&["a"])?,
            enable_thinking,
            &flags(None, None)?,
        );
        let chats = server.bodies(CHAT_ENDPOINT);
        match refused {
            Some(says) => {
                let error = outcome.err().map(|e| format!("{e:#}")).unwrap_or_default();
                assert!(error.contains(says), "{error}");
                assert!(chats.is_empty(), "nothing generates under a refused mode");
            }
            None => {
                drop(outcome?);
                assert!(
                    chats
                        .iter()
                        .all(|b| b.get("chat_template_kwargs").is_none()),
                    "a template without a switch is sent no mode: {chats:?}"
                );
            }
        }
        Ok(())
    }

    /// The recipe's image budget goes out with every sample once the processor
    /// is seen to take it; a processor that ignores it refuses the cell.
    #[rstest]
    #[case::taken(true)]
    #[case::ignored(false)]
    fn an_image_budget_is_checked_before_it_is_sent(#[case] taken: bool) -> anyhow::Result<()> {
        let server = FakeVllm::spawn(move |path, body| match path {
            TOKENIZE_ENDPOINT if taken => tokens(body, false),
            // A fixed count that happens to sit at the budget: the budget
            // plays no part.
            TOKENIZE_ENDPOINT if body.to_string().contains("image_url") => {
                Reply::Json(json!({"count": 292, "tokens": vec![1; 292]}))
            }
            TOKENIZE_ENDPOINT => tokens(body, false),
            _ => answer(),
        })?;
        let budgeted = VlmEvalBenchmark {
            parameter_image_tokens: Some(PositiveCount::try_new(280)?),
            ..benchmark(&["a"])?
        };
        let outcome = run_on(&server, &budgeted, None, &flags(None, None)?);
        let chats = server.bodies(CHAT_ENDPOINT);
        if taken {
            drop(outcome?);
            assert_eq!(chats.len(), 1);
            assert_eq!(
                chats[0]["mm_processor_kwargs"],
                json!({"max_soft_tokens": 280})
            );
        } else {
            let error = outcome.err().map(|e| format!("{e:#}")).unwrap_or_default();
            assert!(
                error.contains("does not take the recipe's image budget"),
                "{error}"
            );
            assert!(chats.is_empty());
        }
        Ok(())
    }

    /// A sample the server refuses is failed at once, and the cell with it: an
    /// eval result answers every sample or is not published.
    #[test]
    fn a_refused_sample_refuses_the_cell() -> anyhow::Result<()> {
        let server = FakeVllm::spawn(|path, body| match path {
            TOKENIZE_ENDPOINT => tokens(body, false),
            _ => Reply::Status(
                "400 Bad Request",
                json!({"error": {"message": "maximum context length is 4096 tokens"}}),
            ),
        })?;
        let (outcome, rows) =
            refused_rows(&server, &benchmark(&["a", "b"])?, &flags(Some(2), None)?)?;
        let error = outcome.err().map(|e| format!("{e:#}")).unwrap_or_default();
        assert!(error.contains("incomplete"), "{error}");
        assert_eq!(
            server.bodies(CHAT_ENDPOINT).len(),
            2,
            "a refusal is not resent"
        );
        assert!(
            rows.iter().all(|row| row.failed
                && row
                    .failed_reason
                    .as_deref()
                    .is_some_and(|r| r.contains("maximum context length"))),
            "{rows:?}"
        );
        Ok(())
    }

    /// An overloaded server's 429 or 503 is its state, not the sample's: the
    /// sample is sent again alone, and answers.
    #[rstest]
    #[case::too_many_requests("429 Too Many Requests")]
    #[case::unavailable("503 Service Unavailable")]
    fn an_overload_is_sent_again(#[case] status: &'static str) -> anyhow::Result<()> {
        let refused = std::sync::atomic::AtomicBool::new(false);
        let server = FakeVllm::spawn(move |path, body| match path {
            TOKENIZE_ENDPOINT => tokens(body, false),
            _ if !refused.swap(true, std::sync::atomic::Ordering::SeqCst) => {
                Reply::Status(status, json!({"error": "busy"}))
            }
            _ => answer(),
        })?;
        let out = run_on(
            &server,
            &benchmark(&["a", "b"])?,
            None,
            &flags(Some(2), None)?,
        )?
        .build();
        let BenchmarkResultData::Eval { completions } = out.result_data else {
            anyhow::bail!("expected Eval result data");
        };
        assert!(completions.iter().all(|c| !c.failed), "{completions:?}");
        assert_eq!(server.bodies(CHAT_ENDPOINT).len(), 3, "one resend");
        Ok(())
    }

    /// A stream that ends without its finish chunk is not an answer: lost, sent
    /// again alone, and failed when it breaks again.
    #[test]
    fn a_stream_cut_short_is_not_an_answer() -> anyhow::Result<()> {
        let server = FakeVllm::spawn(|path, body| match path {
            TOKENIZE_ENDPOINT => tokens(body, false),
            _ => Reply::Stream(vec![json!({"choices": [{"delta": {"content": "B"}}]})]),
        })?;
        let (outcome, rows) =
            refused_rows(&server, &benchmark(&["a", "b"])?, &flags(Some(2), None)?)?;
        assert!(outcome.is_err());
        assert_eq!(server.bodies(CHAT_ENDPOINT).len(), 4, "each sent twice");
        assert!(
            rows.iter().all(|row| row.failed
                && row
                    .failed_reason
                    .as_deref()
                    .is_some_and(|r| r.contains("without finish_reason"))),
            "{rows:?}"
        );
        Ok(())
    }

    /// A reply cut at the length bound is an answer only when it ran the
    /// whole output budget (64 here); one the context cut first fails.
    #[rstest]
    #[case::the_budget(64, false)]
    #[case::the_context(40, true)]
    fn a_reply_cut_by_the_context_is_not_an_answer(
        #[case] completion_tokens: u64,
        #[case] fails: bool,
    ) -> anyhow::Result<()> {
        let server = FakeVllm::spawn(move |path, body| match path {
            TOKENIZE_ENDPOINT => tokens(body, false),
            _ => Reply::Stream(vec![
                json!({"choices": [{"index": 0, "delta": {"content": "Because"}, "finish_reason": "length"}]}),
                json!({"choices": [], "usage": {"prompt_tokens": 9, "completion_tokens": completion_tokens}}),
            ]),
        })?;
        let (outcome, rows) = refused_rows(&server, &benchmark(&["a"])?, &flags(None, None)?)?;
        assert_eq!(outcome.is_err(), fails);
        let row = rows.first().context("a row")?;
        assert_eq!(row.failed, fails, "{row:?}");
        if fails {
            assert!(
                row.failed_reason
                    .as_deref()
                    .is_some_and(|r| r.contains("cut after 40 of its 64")),
                "{row:?}"
            );
        } else {
            assert_eq!(
                row.stop_reason,
                BenchmarkEvalCompletionStopReason::Truncated
            );
        }
        Ok(())
    }

    #[test]
    fn constrained_answers_are_refused() -> anyhow::Result<()> {
        let server = server(false)?;
        let mcq = VlmEvalBenchmark {
            parameter_mcq_choices: Some(vec!["A".to_owned(), "B".to_owned()]),
            ..benchmark(&["a"])?
        };
        let error = run_on(&server, &mcq, None, &flags(None, None)?)
            .err()
            .map(|e| format!("{e:#}"))
            .unwrap_or_default();
        assert!(error.contains("parameter_mcq_choices"), "{error}");
        assert!(server.bodies(TOKENIZE_ENDPOINT).is_empty());
        Ok(())
    }

    #[rstest]
    #[case::on(Some(true), true, false, Ok(Some(true)))]
    #[case::off(Some(false), true, false, Ok(Some(false)))]
    #[case::unstated_on_a_switch(None, true, false, Err(ThinkingRefused::Unstated))]
    #[case::no_mode(None, false, false, Ok(None))]
    #[case::prompt_thinks(Some(true), false, true, Ok(None))]
    #[case::no_switch(Some(true), false, false, Err(ThinkingRefused::NoSwitch))]
    #[case::nothing_to_turn_off(Some(false), false, false, Err(ThinkingRefused::NoMode))]
    fn the_declared_mode_is_held_to_the_template(
        #[case] enable_thinking: Option<bool>,
        #[case] switch: bool,
        #[case] prompt_thinks: bool,
        #[case] expected: Result<Option<bool>, ThinkingRefused>,
    ) {
        assert_eq!(
            thinking_kwarg(enable_thinking, switch, prompt_thinks),
            expected
        );
    }

    #[rstest]
    #[case::unpinned(None, 32768, true)]
    #[case::room(Some(40960), 32768, true)]
    #[case::equal(Some(32768), 32768, false)]
    #[case::under(Some(8192), 32768, false)]
    fn a_pinned_context_must_hold_a_prompt(
        #[case] max_model_len: Option<i32>,
        #[case] max_tokens: u32,
        #[case] fits: bool,
    ) -> anyhow::Result<()> {
        let len = max_model_len.map(PositiveCount::try_new).transpose()?;
        assert_eq!(check_room("vlm-x", len, max_tokens).is_ok(), fits);
        Ok(())
    }
}
