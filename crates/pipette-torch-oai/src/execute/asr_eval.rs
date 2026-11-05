//! Speech-recognition accuracy eval (`asr_eval`) against vLLM's
//! OpenAI-compatible server.
//!
//! The cell the llama.cpp runner runs, on a Hugging Face repo's own weights:
//! the entry's corpus and recipe ([`pipette_vl_spec_decode_corpus::asr_eval`]),
//! every clip answered or the cell refused
//! ([`measurement::eval_complete_chunked`]), the clips read a chunk at a time
//! as on llama.cpp. vLLM holds each clip's context up to the
//! model's own (or the cell's `max_model_len`) and batches the clips in
//! flight, so there is no context sizing and no slot arithmetic.
//!
//! - **Decoding** is the entry's effective one (plan-types'
//!   [`AsrEvalBenchmark::effective_decoding`], the table mgmt and the
//!   warehouse read), every knob sent, on a server started at
//!   `--generation-config vllm` so the repo's `generation_config.json` sets no
//!   default behind it. vLLM's repetition penalty covers the whole sequence,
//!   which is MiniCPM-o's card (Transformers'); its sampler order is
//!   temperature then top_p, which is Voxtral's. Beams never reach here: vLLM's
//!   beam search has no repetition penalty for MiniCPM-o's beam recipe, and
//!   plan-types refuses them on vLLM ([`AsrEvalBenchmark::check_runtime`]).
//! - **The prompt** is each repo's own chat template, as vLLM renders it with
//!   the clip's audio, and the profile's template switches: MiniCPM-o's TTS
//!   template with thinking off, ended at `<|tts_eos|>` as OpenBMB's `chat()`
//!   cuts it; Gemma 4's thinking off. A Voxtral transcription clip goes to
//!   `/v1/audio/transcriptions` with its language, which renders
//!   mistral-common's transcription request, the format the llama.cpp profile
//!   writes by hand.
//! - **The reply** is the model's text as written, special tokens included, as
//!   llama.cpp returns it: the score server extracts each profile's transcript.

use std::io::BufReader;
use std::num::NonZeroUsize;

use anyhow::Context;
use base64::Engine as _;
use serde_json::{Map, Value, json};

use pipette_http::{HttpClient, StreamClient};
use pipette_ops::eval_completions::EvalCompletionSession;
use pipette_ops::measurement::{
    self, EvalMeasurementInFlight, EvalMeasurementReading, EvalMeasurementSample, EvalMessages,
};
use pipette_ops::run_response::RunResponseBuilder;
use pipette_ops::sse::{self, ChatReply};
use pipette_plan_types::{
    AsrDecoding, AsrEvalBenchmark, AsrPromptProfile, AsrSampling, AudioFormat,
    BenchmarkEvalCompletionStopReason, ContentPart, KnownAsrPromptProfile, MessageContent,
    PositiveCount, RunRequest, RuntimeFlags, RuntimeType,
};
use pipette_vl_spec_decode_corpus::asr_eval::AsrEvalCorpus;

use crate::server::{ServerKind, ServerState};

/// The chat endpoint every clip but a Voxtral transcription goes to.
const CHAT_ENDPOINT: &str = "/v1/chat/completions";
/// vLLM's transcription endpoint, for Voxtral's transcription clips.
const TRANSCRIPTION_ENDPOINT: &str = "/v1/audio/transcriptions";
/// MiniCPM-o 4.5's `<|tts_eos|>`, which its ASR answer ends on: OpenBMB's
/// `chat()` cuts the answer there (`modeling_minicpmo.py`,
/// `answer.split("<|tts_eos|>")[0]`), and the llama.cpp profile ends the turn
/// on it. Sent as a stop token, so the server stops where `chat()` cuts.
const MINICPM_O_TTS_EOS: u32 = 151_704;
const MINICPM_O_TTS_EOS_TEXT: &str = "<|tts_eos|>";
/// The samples a chunk carries at the most, and per worker, so every worker
/// has several to work through before the chunk's last one; and the audio a
/// chunk holds at the most (one clip past it, alone), as on llama.cpp.
const CHUNK_SAMPLES: usize = 32;
const CHUNK_SAMPLES_PER_WORKER: usize = 8;
const CHUNK_AUDIO_BYTES: u64 = 32 * 1024 * 1024;
/// The multipart boundary of a transcription request: a fixed string that no
/// base64 clip or field value can hold, since it has characters neither uses.
const BOUNDARY: &str = "pipette-asr-eval-boundary-7f3c";

/// What a cell is, once everything that needs neither a server nor the corpus
/// has been checked: `run_with` checks it before the download and the launch,
/// so a refusal costs neither.
pub(super) struct Cell {
    profile: KnownAsrPromptProfile,
    sampling: AsrSampling,
    max_tokens: u32,
    declared: usize,
    workers: NonZeroUsize,
}

pub(super) fn preflight(
    benchmark: &AsrEvalBenchmark,
    flags: &RuntimeFlags,
) -> anyhow::Result<Cell> {
    let benchmark_id = benchmark.benchmark_id.as_str();
    // The plan types define the cell on uv vLLM alone; this is the message
    // for a run that reached here some other way.
    let RuntimeFlags::AsrEvalUvVllmTorch {
        parallel,
        max_model_len,
        ..
    } = flags
    else {
        anyhow::bail!(
            "asr_eval `{benchmark_id}` runs on uv vLLM alone here, with a repo's own weights"
        );
    };
    // What vLLM cannot run of the entry (a model it has no implementation of,
    // beams), as mgmt refuses it before queueing.
    benchmark
        .check_runtime(RuntimeType::UvVllm)
        .with_context(|| format!("asr_eval `{benchmark_id}`"))?;
    let profile = match &benchmark.parameter_prompt_profile {
        AsrPromptProfile::Known(profile) => *profile,
        AsrPromptProfile::Unknown(name) => {
            anyhow::bail!("asr_eval `{benchmark_id}`: no vLLM recipe for prompt profile `{name}`")
        }
    };
    let sampling = match benchmark
        .effective_decoding()
        .with_context(|| format!("asr_eval `{benchmark_id}`"))?
    {
        AsrDecoding::Sampling(sampling) => sampling,
        AsrDecoding::Beams(_) => {
            anyhow::bail!("asr_eval `{benchmark_id}` decodes by beam search, which vLLM cannot run")
        }
    };
    let max_tokens = benchmark.parameter_max_tokens.into_inner().unsigned_abs();
    check_room(benchmark_id, *max_model_len, max_tokens)?;
    let declared = usize::try_from(benchmark.parameter_sample_count.into_inner().unsigned_abs())
        .context("parameter_sample_count does not fit in usize")?;
    // More workers than samples would only ever claim an exhausted list.
    let workers = parallel
        .map_or(Ok(1), |n| usize::try_from(n.into_inner()))?
        .clamp(1, declared.max(1));
    Ok(Cell {
        profile,
        sampling,
        max_tokens,
        declared,
        workers: NonZeroUsize::new(workers).unwrap_or(NonZeroUsize::MIN),
    })
}

impl Cell {
    /// The samples the entry declares.
    pub(super) fn declared(&self) -> usize {
        self.declared
    }
}

/// What `run_with` prepares before the launch for the run to read: the cell,
/// its corpus and its checkpoint.
pub(super) struct AsrRun {
    pub(super) cell: Cell,
    pub(super) corpus: AsrEvalCorpus,
    pub(super) checkpoint: EvalCompletionSession,
}

/// The dataset's id at `position`, plan-types' derivation, the one the evals
/// server's ids follow.
pub(super) fn id_at(benchmark: &AsrEvalBenchmark) -> impl Fn(usize) -> String + '_ {
    |position| benchmark.parameter_asr_task.sample_id(position as u64)
}

/// The response of a run whose checkpoint already settles every sample: no
/// corpus is downloaded and no server started, as on llama.cpp.
pub(super) fn settled(
    benchmark: &AsrEvalBenchmark,
    cell: &Cell,
    checkpoint: EvalCompletionSession,
) -> anyhow::Result<RunResponseBuilder> {
    let result_data = measurement::eval_complete_chunked(
        &benchmark.benchmark_id,
        cell.declared,
        id_at(benchmark),
        |offset| anyhow::bail!("no sample is left to read, asked for {offset}"),
        EvalMeasurementInFlight::Parallel(NonZeroUsize::MIN),
        checkpoint,
        (),
        |(), _, _| Ok(()),
    )?;
    let mut response = RunResponseBuilder::new(result_data);
    // The checkpoint is keyed by the corpus digest, so its answers came from
    // the corpus the entry pins.
    if let Some(dataset) = &benchmark.dataset {
        response = response.dataset_digest(dataset.digest.clone());
    }
    Ok(response)
}

pub(super) fn run(
    req: &RunRequest,
    benchmark: &AsrEvalBenchmark,
    cell: &Cell,
    corpus: &AsrEvalCorpus,
    checkpoint: EvalCompletionSession,
    model: &str,
    state: &ServerState,
) -> anyhow::Result<RunResponseBuilder> {
    let benchmark_id = benchmark.benchmark_id.as_str();
    anyhow::ensure!(
        matches!(state.kind, ServerKind::Vllm),
        "asr_eval `{benchmark_id}` runs on vLLM; this server is not one"
    );
    let idle_timeout = measurement::stream_idle_timeout(req);
    let streaming = HttpClient::streaming("pipette", idle_timeout)
        .context("failed to build streaming HTTP client")?;
    let base_url = state.base_url();
    let request = Request {
        benchmark,
        profile: cell.profile,
        sampling: cell.sampling,
        model,
        max_tokens: cell.max_tokens,
    };
    tracing::info!(
        benchmark_id,
        profile = %cell.profile,
        sampling = ?cell.sampling,
        max_tokens = cell.max_tokens,
        workers = cell.workers,
        "asr_eval request settings"
    );
    let chunk = CHUNK_SAMPLES.max(cell.workers.get().saturating_mul(CHUNK_SAMPLES_PER_WORKER));
    let result_data = measurement::eval_complete_chunked(
        &benchmark.benchmark_id,
        cell.declared,
        id_at(benchmark),
        |offset| corpus.chunk(offset, chunk, CHUNK_AUDIO_BYTES),
        EvalMeasurementInFlight::Parallel(cell.workers),
        checkpoint,
        (),
        |(), samples, record| {
            samples.iter().enumerate().try_for_each(|(at, sample)| {
                record(at, read_sample(&streaming, &base_url, &request, sample))
            })?;
            Ok(())
        },
    )?;
    Ok(RunResponseBuilder::new(result_data)
        .executable(state.executable.display().to_string())
        .command(request.command(&base_url))
        .dataset_digest(corpus.digest().clone()))
}

/// A pinned context at or under the output budget leaves no room for a clip:
/// vLLM refuses every request whose prompt plus `max_tokens` exceeds
/// `max_model_len`. Unpinned, the model's own context holds them.
fn check_room(
    benchmark_id: &str,
    max_model_len: Option<PositiveCount>,
    max_tokens: u32,
) -> anyhow::Result<()> {
    if let Some(len) = max_model_len {
        anyhow::ensure!(
            len.into_inner().unsigned_abs() > max_tokens,
            "asr_eval `{benchmark_id}` pins max_model_len={len}, which is no more than its \
             parameter_max_tokens={max_tokens}: vLLM would refuse every clip; leave it unset \
             (the model's own context) or raise it past the output budget"
        );
    }
    Ok(())
}

/// What every clip's request shares.
struct Request<'a> {
    benchmark: &'a AsrEvalBenchmark,
    profile: KnownAsrPromptProfile,
    sampling: AsrSampling,
    model: &'a str,
    max_tokens: u32,
}

/// Where a clip goes.
#[derive(Debug, Clone, PartialEq)]
enum Route {
    Chat(Value),
    /// A Voxtral transcription clip: the multipart body, as bytes.
    Transcription(Vec<u8>),
}

impl Request<'_> {
    /// The entry's sampling as vLLM's request spells it: `top_k` `-1` is its
    /// "no cut", which the effective decoding writes `0`.
    fn sampling(&self, sample_id: &str) -> Map<String, Value> {
        let s = self.sampling;
        let mut fields = Map::new();
        fields.insert("temperature".to_owned(), json!(s.temperature.as_f64()));
        fields.insert("top_p".to_owned(), json!(s.top_p.as_f64()));
        let top_k = i32::from(s.top_k);
        fields.insert(
            "top_k".to_owned(),
            json!(if top_k > 0 { top_k } else { -1 }),
        );
        fields.insert("min_p".to_owned(), json!(s.min_p.as_f64()));
        fields.insert(
            "repetition_penalty".to_owned(),
            json!(s.repeat_penalty.as_f64()),
        );
        fields.insert("presence_penalty".to_owned(), json!(0.0));
        fields.insert("frequency_penalty".to_owned(), json!(0.0));
        if s.seed.is_some()
            && let Some(seed) = self.benchmark.sample_seed(sample_id)
        {
            fields.insert("seed".to_owned(), json!(seed));
        }
        fields
    }

    /// The template arguments a chat request carries: the profile's switches,
    /// then the clip's own, with thinking closed whatever either says.
    fn template_kwargs(&self, sample: Option<&Map<String, Value>>) -> Option<Map<String, Value>> {
        let mut kwargs = Map::new();
        if self.profile == KnownAsrPromptProfile::MinicpmO {
            kwargs.insert("use_tts_template".to_owned(), Value::Bool(true));
        }
        if let Some(sample) = sample {
            kwargs.extend(sample.clone());
        }
        let closes_thinking = matches!(
            self.profile,
            KnownAsrPromptProfile::Gemma4 | KnownAsrPromptProfile::MinicpmO
        );
        if closes_thinking || kwargs.contains_key("enable_thinking") {
            kwargs.insert("enable_thinking".to_owned(), Value::Bool(false));
        }
        (!kwargs.is_empty()).then_some(kwargs)
    }

    /// Where one clip goes and what it sends.
    fn route(&self, sample: &EvalMeasurementSample<'_>) -> anyhow::Result<Route> {
        let transcription = self.profile == KnownAsrPromptProfile::Voxtral
            && sample
                .chat_template_kwargs
                .and_then(|kwargs| kwargs.get("transcription"))
                .and_then(Value::as_bool)
                == Some(true);
        if transcription {
            return self.transcription(sample).map(Route::Transcription);
        }
        let mut body = json!({
            "model": self.model,
            "messages": sample.messages,
            "max_tokens": self.max_tokens,
            "stream": true,
            "stream_options": { "include_usage": true },
            "skip_special_tokens": false,
        });
        if let Some(fields) = body.as_object_mut() {
            fields.extend(self.sampling(sample.id));
            if let Some(kwargs) = self.template_kwargs(sample.chat_template_kwargs) {
                fields.insert("chat_template_kwargs".to_owned(), Value::Object(kwargs));
            }
            if self.profile == KnownAsrPromptProfile::MinicpmO {
                fields.insert("stop_token_ids".to_owned(), json!([MINICPM_O_TTS_EOS]));
            }
        }
        Ok(Route::Chat(body))
    }

    /// A Voxtral transcription clip's multipart body: the clip, its language,
    /// the entry's sampling and the output cap, streamed with its usage. The
    /// clip must be the user turn's only content, as a transcription request
    /// carries no text.
    fn transcription(&self, sample: &EvalMeasurementSample<'_>) -> anyhow::Result<Vec<u8>> {
        let EvalMessages::Chat(messages) = sample.messages else {
            anyhow::bail!("a Voxtral transcription clip arrived as raw messages");
        };
        let mut clips = Vec::new();
        for message in messages {
            match &message.content {
                MessageContent::Text(text) => {
                    anyhow::ensure!(
                        text.trim().is_empty(),
                        "a Voxtral transcription clip carries text, which a transcription \
                         request has no place for"
                    );
                }
                MessageContent::Parts(parts) => {
                    for part in parts {
                        match part {
                            ContentPart::InputAudio { input_audio } => clips.push(input_audio),
                            ContentPart::Text { text } => anyhow::ensure!(
                                text.trim().is_empty(),
                                "a Voxtral transcription clip carries text, which a \
                                 transcription request has no place for"
                            ),
                            ContentPart::ImageUrl { .. } => {
                                anyhow::bail!("a Voxtral transcription clip carries an image")
                            }
                        }
                    }
                }
            }
        }
        let [clip] = clips.as_slice() else {
            anyhow::bail!(
                "a Voxtral transcription request takes one clip; this sample holds {}",
                clips.len()
            );
        };
        let audio = base64::engine::general_purpose::STANDARD
            .decode(clip.data.as_bytes())
            .context("the clip's base64 does not decode")?;
        let (extension, mime) = match clip.format {
            AudioFormat::Wav => ("wav", "audio/wav"),
            AudioFormat::Flac => ("flac", "audio/flac"),
            AudioFormat::Mp3 => ("mp3", "audio/mpeg"),
            AudioFormat::Ogg => ("ogg", "audio/ogg"),
        };
        let mut fields: Vec<(String, String)> = vec![
            ("model".to_owned(), self.model.to_owned()),
            ("response_format".to_owned(), "json".to_owned()),
            ("stream".to_owned(), "true".to_owned()),
            ("stream_include_usage".to_owned(), "true".to_owned()),
            (
                "max_completion_tokens".to_owned(),
                self.max_tokens.to_string(),
            ),
        ];
        if let Some(language) = sample
            .chat_template_kwargs
            .and_then(|kwargs| kwargs.get("language"))
            .and_then(Value::as_str)
        {
            fields.push(("language".to_owned(), language.to_owned()));
        }
        // Voxtral's chain is temperature then top_p, the only knobs its card
        // sets: plan-types refuses the rest on its profile.
        let s = self.sampling;
        fields.push(("temperature".to_owned(), s.temperature.as_f64().to_string()));
        fields.push(("top_p".to_owned(), s.top_p.as_f64().to_string()));
        if s.seed.is_some()
            && let Some(seed) = self.benchmark.sample_seed(sample.id)
        {
            fields.push(("seed".to_owned(), seed.to_string()));
        }
        Ok(multipart(
            &fields,
            &format!("clip.{extension}"),
            mime,
            &audio,
        ))
    }

    /// The command the response records: each endpoint and what every request
    /// to it fixes.
    fn command(&self, base_url: &str) -> Vec<String> {
        let mut command = vec![
            format!("POST {base_url}{CHAT_ENDPOINT}"),
            format!(
                "model={} max_tokens={} stream=true skip_special_tokens=false sampling={:?}",
                self.model, self.max_tokens, self.sampling
            ),
        ];
        if let Some(kwargs) = self.template_kwargs(None) {
            command.push(format!("chat_template_kwargs={}", Value::Object(kwargs)));
        }
        match self.profile {
            KnownAsrPromptProfile::MinicpmO => {
                command.push(format!("stop_token_ids=[{MINICPM_O_TTS_EOS}]"));
            }
            KnownAsrPromptProfile::Voxtral => {
                command.push(format!(
                    "POST {base_url}{TRANSCRIPTION_ENDPOINT} (transcription clips): \
                     language, temperature, top_p, max_completion_tokens, stream=true"
                ));
            }
            KnownAsrPromptProfile::Gemma4
            | KnownAsrPromptProfile::Qwen25Omni
            | KnownAsrPromptProfile::Lfm25Audio
            | KnownAsrPromptProfile::Lfm3 => {}
        }
        command
    }
}

/// A `multipart/form-data` body of text `fields` and one file part.
fn multipart(fields: &[(String, String)], filename: &str, mime: &str, file: &[u8]) -> Vec<u8> {
    let mut body = Vec::with_capacity(file.len() + 1024);
    for (name, value) in fields {
        body.extend_from_slice(
            format!(
                "--{BOUNDARY}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n"
            )
            .as_bytes(),
        );
    }
    body.extend_from_slice(
        format!(
            "--{BOUNDARY}\r\nContent-Disposition: form-data; name=\"file\"; \
             filename=\"{filename}\"\r\nContent-Type: {mime}\r\n\r\n"
        )
        .as_bytes(),
    );
    body.extend_from_slice(file);
    body.extend_from_slice(format!("\r\n--{BOUNDARY}--\r\n").as_bytes());
    body
}

/// One attempt at a clip. A request the server refused (a 4xx, whose body
/// says why) is the clip's own failure; one that broke on the way (a reset, a
/// stream cut short, an overloaded server's 429 or 5xx) is lost, sent again
/// alone and only then failed, as on vlm_eval. Either way the shared entry
/// sends a failed clip again on a later run while it has attempts left, and
/// refuses the cell if it never answers.
fn read_sample(
    client: &StreamClient,
    base_url: &str,
    request: &Request<'_>,
    sample: &EvalMeasurementSample<'_>,
) -> EvalMeasurementReading {
    let reply = request.route(sample).and_then(|route| match route {
        Route::Chat(body) => stream(
            client,
            CHAT_ENDPOINT,
            client
                .http()
                .post(format!("{base_url}{CHAT_ENDPOINT}"))
                .json(&body),
        ),
        Route::Transcription(body) => stream(
            client,
            TRANSCRIPTION_ENDPOINT,
            client
                .http()
                .post(format!("{base_url}{TRANSCRIPTION_ENDPOINT}"))
                .header(
                    reqwest::header::CONTENT_TYPE,
                    format!("multipart/form-data; boundary={BOUNDARY}"),
                )
                .body(body),
        ),
    });
    let reading = match reply {
        Ok(reply) => {
            let (stop_reason, stop_detail) = reply.stop();
            let completion_tokens = reply.completion_tokens;
            match cut_short(stop_reason, completion_tokens, request.max_tokens) {
                Some(reason) => EvalMeasurementReading::Failed { reason },
                None => EvalMeasurementReading::Completed {
                    completion: answer(request.profile, reply.completion_with_reasoning()),
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
        | EvalMeasurementReading::Unfit { reason } => tracing::warn!(
            sample = sample.position.as_str(),
            sample_id = %sample.id,
            reason = reason.as_str(),
            "sample failed"
        ),
    }
    reading
}

/// The answer as llama.cpp's profile returns it. vLLM leaves a special stop
/// token out of the text; should it ever be kept, MiniCPM-o's `<|tts_eos|>`
/// is dropped, as llama.cpp's end-of-turn override ends the answer before it.
fn answer(profile: KnownAsrPromptProfile, completion: String) -> String {
    match (profile, completion.strip_suffix(MINICPM_O_TTS_EOS_TEXT)) {
        (KnownAsrPromptProfile::MinicpmO, Some(answer)) => answer.to_owned(),
        _ => completion,
    }
}

/// Why a reply cut at a length bound is not an answer: the bound was the
/// context, not the entry's output budget. vLLM from 0.32 treats the cap as an
/// upper bound rather than reserving it, so a clip that leaves less room than
/// the budget is generated into the context's end and reported as `length`; as
/// on vlm_eval, a cut answer is one that ran the full budget.
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
             context (max_model_len) ran out first, which leaves no room for the entry's \
             output budget; raise max_model_len or leave it unset"
        ),
        None => format!(
            "the reply was cut at a length bound and the stream reported no token count, so \
             whether it ran its {max_tokens} output tokens cannot be told"
        ),
    })
}

/// Whether the server refused the request for good: a status that is not an
/// overload (a 4xx but 429), as on vlm_eval.
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

/// A streamed reply, chat or transcription (whose chunks carry the same
/// `choices[0].delta`, `finish_reason` and `usage`), read to its finish chunk.
/// A stream that ends without one is an error, not an answer.
fn stream(
    client: &StreamClient,
    endpoint: &str,
    request: reqwest::blocking::RequestBuilder,
) -> anyhow::Result<ChatReply> {
    let response =
        pipette_http::send(request).with_context(|| format!("failed to call {endpoint}"))?;
    let events = sse::data_lines(BufReader::new(response), client.idle_timeout());
    let reply = sse::read_chat_stream(events, &mut |_| {})?;
    anyhow::ensure!(
        reply.finish_reason.is_some(),
        "{endpoint} stream ended without finish_reason (connection closed early?); \
         discarding {} chars of partial content",
        reply.content.len()
    );
    Ok(reply)
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use pipette_plan_types::{
        BenchmarkId, ChatMessage, ChatRole, InputAudio, NumBeams, PositiveCount, SamplingSeed,
    };

    use super::*;

    /// The profile's entry on FLEURS Dutch.
    fn entry(profile: &str) -> anyhow::Result<AsrEvalBenchmark> {
        Ok(AsrEvalBenchmark {
            benchmark_id: BenchmarkId::try_new(format!("asr_eval_fleurs_nl_nl_{profile}"))?,
            parameter_asr_task: "asr_fleurs_nl_nl".to_owned().into(),
            parameter_dataset_version: "2026.10.1".into(),
            parameter_sample_count: PositiveCount::try_new(364)?,
            parameter_prompt_profile: profile.into(),
            parameter_max_tokens: PositiveCount::try_new(256)?,
            parameter_temperature: None,
            parameter_top_p: None,
            parameter_top_k: None,
            parameter_min_p: None,
            parameter_repeat_penalty: None,
            parameter_seed: None,
            parameter_seed_rule: None,
            parameter_num_beams: None,
            parameter_beam_repetition_penalty: None,
            parameter_beam_length_penalty: None,
            dataset: None,
            recipe: None,
        })
    }

    fn flags(parallel: Option<i32>, max_model_len: Option<i32>) -> anyhow::Result<RuntimeFlags> {
        Ok(RuntimeFlags::AsrEvalUvVllmTorch {
            tensor_parallel_size: None,
            dtype: None,
            max_model_len: max_model_len.map(PositiveCount::try_new).transpose()?,
            prefix_caching: None,
            parallel: parallel.map(PositiveCount::try_new).transpose()?,
            envs: Vec::new(),
            raw: Vec::new(),
        })
    }

    fn messages(text: Option<&str>) -> Vec<ChatMessage> {
        let mut parts = vec![ContentPart::InputAudio {
            input_audio: InputAudio {
                data: base64::engine::general_purpose::STANDARD.encode(b"RIFF-not-really"),
                format: AudioFormat::Wav,
            },
        }];
        if let Some(text) = text {
            parts.push(ContentPart::Text {
                text: text.to_owned(),
            });
        }
        vec![ChatMessage {
            role: ChatRole::User,
            content: MessageContent::Parts(parts),
        }]
    }

    fn sample<'a>(
        messages: &'a [ChatMessage],
        kwargs: Option<&'a Map<String, Value>>,
    ) -> EvalMeasurementSample<'a> {
        EvalMeasurementSample {
            id: "asr_fleurs_nl_nl_0",
            messages: EvalMessages::Chat(messages),
            chat_template_kwargs: kwargs,
            position: "eval sample 1/364: id=asr_fleurs_nl_nl_0".to_owned(),
        }
    }

    fn request<'a>(benchmark: &'a AsrEvalBenchmark, cell: &Cell) -> Request<'a> {
        Request {
            benchmark,
            profile: cell.profile,
            sampling: cell.sampling,
            model: "/models/repo",
            max_tokens: cell.max_tokens,
        }
    }

    /// The four families vLLM runs take the cell; what vLLM cannot run of an
    /// entry, or a cell another engine's flags describe, is refused before
    /// the corpus downloads.
    #[rstest]
    #[case::gemma(entry("gemma4"), None)]
    #[case::qwen(entry("qwen25_omni"), None)]
    #[case::voxtral(entry("voxtral"), None)]
    #[case::minicpm(entry("minicpm_o"), None)]
    #[case::lfm(entry("lfm25_audio"), Some("vLLM has no implementation of its model"))]
    #[case::beams(
        entry("minicpm_o").map(|e| AsrEvalBenchmark {
            parameter_num_beams: NumBeams::try_new(3).ok(),
            ..e
        }),
        Some("its beam search takes no repetition penalty")
    )]
    fn the_cell_runs_what_vllm_can(
        #[case] benchmark: anyhow::Result<AsrEvalBenchmark>,
        #[case] refused: Option<&str>,
    ) -> anyhow::Result<()> {
        let checked = preflight(&benchmark?, &flags(Some(8), None)?);
        match refused {
            None => {
                checked?;
            }
            Some(names) => {
                let error = checked.err().map(|e| format!("{e:#}")).unwrap_or_default();
                assert!(error.contains(names), "{error}");
            }
        }
        Ok(())
    }

    /// A pinned context no larger than the output budget is refused, and the
    /// workers never outnumber the samples.
    #[test]
    fn the_room_and_the_workers_are_checked() -> anyhow::Result<()> {
        let benchmark = entry("qwen25_omni")?;
        let error = preflight(&benchmark, &flags(None, Some(256))?)
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default();
        assert!(error.contains("max_model_len=256"), "{error}");
        let cell = preflight(&benchmark, &flags(Some(1024), Some(4096))?)?;
        assert_eq!(cell.workers.get(), 364);
        Ok(())
    }

    /// MiniCPM-o's chat request: the TTS template with thinking off, stopped
    /// at `<|tts_eos|>`, greedy through top-k with its card's whole-sequence
    /// repetition penalty, special tokens kept.
    #[test]
    fn minicpm_o_runs_its_card_on_the_chat_endpoint() -> anyhow::Result<()> {
        let benchmark = entry("minicpm_o")?;
        let cell = preflight(&benchmark, &flags(None, None)?)?;
        let msgs = messages(Some("Transcribe the audio."));
        let Route::Chat(body) = request(&benchmark, &cell).route(&sample(&msgs, None))? else {
            anyhow::bail!("MiniCPM-o is a chat request");
        };
        assert_eq!(
            body["chat_template_kwargs"],
            json!({"use_tts_template": true, "enable_thinking": false})
        );
        assert_eq!(body["stop_token_ids"], json!([MINICPM_O_TTS_EOS]));
        assert_eq!(body["temperature"], json!(0.0));
        assert_eq!(body["top_k"], json!(1));
        assert_eq!(body["repetition_penalty"], json!(1.02));
        assert_eq!(body["skip_special_tokens"], json!(false));
        assert_eq!(body["messages"], json!(msgs));
        Ok(())
    }

    /// Gemma 4 samples at its card's settings with thinking closed; an entry's
    /// seed reaches each clip as its own derived seed.
    #[test]
    fn gemma_4_samples_at_its_card_with_thinking_off() -> anyhow::Result<()> {
        let benchmark = AsrEvalBenchmark {
            parameter_seed: Some(SamplingSeed::try_new(7)?),
            ..entry("gemma4")?
        };
        let cell = preflight(&benchmark, &flags(None, None)?)?;
        let msgs = messages(Some("Transcribe."));
        let Route::Chat(body) = request(&benchmark, &cell).route(&sample(&msgs, None))? else {
            anyhow::bail!("Gemma 4 is a chat request");
        };
        assert_eq!(
            body["chat_template_kwargs"],
            json!({"enable_thinking": false})
        );
        assert_eq!(body["temperature"], json!(1.0));
        assert_eq!(body["top_p"], json!(0.95));
        assert_eq!(body["top_k"], json!(64));
        assert_eq!(
            body["seed"],
            json!(benchmark.sample_seed("asr_fleurs_nl_nl_0"))
        );
        assert!(body.get("stop_token_ids").is_none());
        Ok(())
    }

    /// A Voxtral transcription clip goes to the transcription endpoint with its
    /// language and the entry's temperature; one that carries text, which such a
    /// request cannot, is refused; a clip that is not a transcription is a chat.
    #[test]
    fn a_voxtral_transcription_clip_goes_to_the_transcription_endpoint() -> anyhow::Result<()> {
        let benchmark = entry("voxtral")?;
        let cell = preflight(&benchmark, &flags(None, None)?)?;
        let kwargs = json!({"transcription": true, "language": "nl"});
        let kwargs = kwargs.as_object();
        let clip = messages(None);
        let Route::Transcription(body) =
            request(&benchmark, &cell).route(&sample(&clip, kwargs))?
        else {
            anyhow::bail!("a transcription clip is a transcription request");
        };
        let text = String::from_utf8_lossy(&body);
        for field in [
            "name=\"language\"\r\n\r\nnl\r\n",
            "name=\"temperature\"\r\n\r\n0\r\n",
            "name=\"max_completion_tokens\"\r\n\r\n256\r\n",
            "name=\"stream\"\r\n\r\ntrue\r\n",
            "name=\"stream_include_usage\"\r\n\r\ntrue\r\n",
            "filename=\"clip.wav\"\r\nContent-Type: audio/wav\r\n\r\nRIFF-not-really\r\n",
        ] {
            assert!(text.contains(field), "{field:?} not in {text}");
        }
        assert!(text.ends_with(&format!("--{BOUNDARY}--\r\n")));
        let worded = messages(Some("Transcribe this."));
        let error = request(&benchmark, &cell)
            .route(&sample(&worded, kwargs))
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default();
        assert!(error.contains("carries text"), "{error}");
        let chat = request(&benchmark, &cell).route(&sample(&worded, None))?;
        assert!(matches!(chat, Route::Chat(_)));
        Ok(())
    }

    /// MiniCPM-o's answer ends before `<|tts_eos|>`, as llama.cpp's does;
    /// another profile's text is kept as written.
    #[rstest]
    #[case::minicpm_with(
        KnownAsrPromptProfile::MinicpmO,
        "<|tts_bos|>hallo<|tts_eos|>",
        "<|tts_bos|>hallo"
    )]
    #[case::minicpm_without(
        KnownAsrPromptProfile::MinicpmO,
        "<|tts_bos|>hallo",
        "<|tts_bos|>hallo"
    )]
    #[case::gemma(KnownAsrPromptProfile::Gemma4, "hallo<|tts_eos|>", "hallo<|tts_eos|>")]
    fn the_answer_is_llama_cpps(
        #[case] profile: KnownAsrPromptProfile,
        #[case] completion: &str,
        #[case] expected: &str,
    ) {
        assert_eq!(answer(profile, completion.to_owned()), expected);
    }
}
