use std::{
    num::NonZeroUsize,
    sync::{Mutex, PoisonError},
    time::Duration,
};

use anyhow::Context;
use base64::Engine;
use reqwest::Method;
use serde_json::{Map, Value, json};

use pipette_http::{HttpClient, StreamClient};
use pipette_ops::eval_completions::EvalCompletionSession;
use pipette_ops::measurement::{
    self, EvalMeasurementInFlight, EvalMeasurementReading, EvalMeasurementSample, EvalMessages,
    MeasurementObserver,
};
use pipette_ops::readiness::ReadinessGate;
use pipette_ops::run_context::RunContext;
use pipette_ops::run_response::RunResponseBuilder;
use pipette_ops::sse::ChatReply;
use pipette_plan_types::{
    AsrDecoding, AsrEvalBenchmark, AsrEvalLlamacppCliStockToolsGgufVision, AsrPromptProfile,
    AsrRepeatWindow, AsrSamplerChain, AudioFormat, BenchmarkFlagRef, ChatMessage, ContentPart,
    KnownAsrPromptProfile, LlamacppCliStockToolsExtraRuntimeConfiguration, MessageContent,
    RunResponse, RuntimeFlags,
};
use pipette_subprocess::start_process;
use pipette_vl_spec_decode_corpus::asr_eval as asr_corpus;

use crate::eval_pool::{
    self, LlamaServer, Server, ServerSlot, log_completed, read_sample, slot_server,
    stream_chat_completion,
};
use crate::gguf_meta;
use crate::models::ResolvedGgufVisionModel;
use crate::mtmd;
use crate::runtimes::{ResolvedLlamaServer, get_load_mode};
use crate::server::{self, LoggingMarker, ServerCommandBuilder};
use crate::thinking_controls;

/// The multimodal chat endpoint every clip is sent to.
const CHAT_ENDPOINT: &str = "/v1/chat/completions";

/// llama.cpp's default sampler chain, by the names llama-server's `samplers`
/// reads: `common_params_sampling::samplers` in `common/common.h`, the same in
/// the vendored llama.cpp, pipette's ASR build (362d8dc3e) and its LFM3 build
/// (863f8884). Every profile but Voxtral runs it, spelled on each request so
/// that neither a GGUF's `general.sampling.sequence` nor a server default can
/// drop or reorder a stage: a chain without `penalties` would turn MiniCPM-o's
/// repetition penalty off, and one without `top_k` Gemma 4's cut.
const LLAMA_CPP_DEFAULT_SAMPLERS: [&str; 9] = [
    "penalties",
    "dry",
    "top_n_sigma",
    "top_k",
    "typ_p",
    "top_p",
    "min_p",
    "xtc",
    "temperature",
];

/// The audio a server's context is sized for at the least: a 60-second clip,
/// whose bound is three 30-second encoder windows (a clip of exactly 60 s
/// starts a third, see [`audio_tokens`]), so a clip whose length its header
/// does not say still fits if it is no longer.
const CLIP_FLOOR_SECONDS: f64 = 60.0;
/// The audio token rate the context bound assumes: 25 per second, the densest
/// of the supported encoders (Whisper-style 50 Hz frames pooled by two; Gemma
/// 4's encoder is held to it as well).
const AUDIO_TOKENS_PER_SECOND: f64 = 25.0;
/// The Whisper-family encoders (Qwen2.5-Omni, Voxtral) run whole 30-second
/// windows, and stock mtmd starts an extra, silent window at an exact
/// multiple of 30 s; so a short clip can cost more than its rate says. 750 is
/// a window at [`AUDIO_TOKENS_PER_SECOND`]; Voxtral's encoder pools further,
/// so 750 bounds it too. Like the rate, a conservative bound of pipette's own,
/// not a published figure.
const AUDIO_TOKENS_PER_WINDOW: u32 = 750;
const AUDIO_WINDOW_SECONDS: f64 = 30.0;
/// The most context an unpinned asr_eval cell sizes itself, across its slots:
/// the eval cells' shared ceiling.
const ASR_EVAL_CTX_CEILING: u32 = mtmd::VLM_EVAL_CTX_CEILING;
/// The chat template's own tokens per sample (role markers, BOS, the
/// generation prompt), on top of the prompt's text, which is counted as one
/// token per byte, an upper bound.
const TEMPLATE_MARGIN_TOKENS: u32 = 256;

/// Base64 characters decoded to read a clip's header, a multiple of four:
/// 6 KiB of audio, where a WAV `fmt `/`data` pair or FLAC's STREAMINFO sits.
const HEADER_BASE64_CHARS: usize = 8192;

// Provenance. Each profile's recipe (its "card") follows its family's public
// source, as pipette-plan-types' `KnownAsrPromptProfile` lists them: Google's
// gemma-4-E2B-it card (3e22461f), Qwen's ASR cookbook (QwenLM/Qwen2.5-Omni
// d8a31ca5), Mistral's transcription usage (the Voxtral-Mini-3B-2507 card and
// mistral-common), OpenBMB's MiniCPM-o-4_5 `modeling_minicpmo.py` (HF revision
// d59117d19ddb8a28a62f1b5f4e3b34ac48d510a4) and OpenBMB/OmniEvalKit@1adac057,
// and the LiquidAI/LFM2.5-Audio-1.5B card. LFM3 is a placeholder, to be
// replaced before release: the model is not published and has no public card,
// so its recipe is pipette's own, its template rendering the model's native
// ASR chat format. A public source says nothing about llama-server, so its
// spelling there (stop words, template switches, the EOT override,
// `cache_prompt`) is pipette's own, and each item below says so.

/// MiniCPM-o 4.5's `<|tts_eos|>`, which its ASR answer ends on: OpenBMB's
/// `chat()` cuts the answer there (`modeling_minicpmo.py`,
/// `answer.split("<|tts_eos|>")[0]`). The GGUF's own end-of-turn token is
/// `<|im_end|>`, which never comes after a TTS-template answer, so stopping
/// llama-server there by overriding `tokenizer.ggml.eot_token_id` is pipette's
/// own spelling of that cut. The id is checked against the GGUF's vocabulary
/// before the server starts ([`MINICPM_O_TOKENS`]).
const MINICPM_O_EOT_TOKEN: u32 = 151_704;
/// The tokens that end a MiniCPM-o beam: `<|im_end|>` and `<|endoftext|>`, the
/// model's `terminators` in OpenBMB's `modeling_minicpmo.py` (and the
/// `eos_token_id` of its `generation_config.json`), at d59117d1. The beam
/// search is that file's `prepare_generation_config` beam branch (three beams,
/// repetition penalty 1.2) under Transformers' beam search, whose length
/// penalty defaults to 1.0; its logits are bf16-rounded because OpenBMB's card
/// runs the model in `torch.bfloat16`.
const MINICPM_O_BEAM_EOS: [u32; 2] = [151_645, 151_643];
/// The pieces MiniCPM-o's hard-coded ids must name in the GGUF it runs: another
/// release or a re-tokenized conversion that numbers them differently would
/// end its answers on the wrong token, and nothing would fail.
const MINICPM_O_TOKENS: [(u32, &str); 3] = [
    (MINICPM_O_EOT_TOKEN, "<|tts_eos|>"),
    (151_645, "<|im_end|>"),
    (151_643, "<|endoftext|>"),
];

/// [`minicpm_o_template_kwargs`] as the server flag spells them, in a fixed
/// order: a fixed string, so the recorded argv is the same whatever
/// key order a build's JSON maps keep (a test holds the two equal).
const MINICPM_O_SERVER_KWARGS: &str = r#"{"use_tts_template":true,"enable_thinking":false}"#;

/// MiniCPM-o's chat template switches, on the server and on every request:
/// the TTS template its ASR answer is rendered with, and thinking closed, as
/// OmniEvalKit@1adac057's MiniCPM-o adapter calls `chat(use_tts_template=True,
/// enable_thinking=False)` (`o_e_Kit/models/minicpm/minicpmo.py`).
/// `--reasoning off` closes thinking too; sending both is harmless (pipette's
/// ASR build of llama.cpp logs `enable_thinking` here as deprecated).
fn minicpm_o_template_kwargs() -> Map<String, Value> {
    let mut kwargs = Map::new();
    kwargs.insert("use_tts_template".to_owned(), Value::Bool(true));
    kwargs.insert("enable_thinking".to_owned(), Value::Bool(false));
    kwargs
}

/// Voxtral's transcription template, pipette's own, written to Mistral's
/// documented transcription format (`[INST]...[/INST]lang:<code>[TRANSCRIBE]`,
/// mistral-common's `TranscriptionRequest`; see `asr_templates/README.md`).
/// The published GGUFs carry a text-only Devstral template, which renders
/// neither the audio nor the `[TRANSCRIBE]` turn.
const VOXTRAL_CHAT_TEMPLATE: &str = include_str!("asr_templates/voxtral.jinja");

/// LFM3's ASR template, pipette's own, which renders the model's native ASR
/// chat format: `<|startoftext|>`, then `<|im_start|>role\n ... <|im_end|>\n`
/// per turn (the score server's system turn, and a user turn of the clip
/// alone), and an open assistant turn with no think block; the GGUF's own
/// template opens a think block. llama-server turns each audio part
/// into its media marker inside a text part, so only text parts are rendered.
const LFM3_ASR_TEMPLATE: &str = concat!(
    "{{- bos_token -}}",
    "{%- for message in messages -%}",
    "{{- '<|im_start|>' + message['role'] + '\\n' -}}",
    "{%- if message['content'] is string -%}{{- message['content'] -}}",
    "{%- else -%}{%- for part in message['content'] -%}",
    "{%- if part['type'] == 'text' -%}{{- part['text'] -}}{%- endif -%}",
    "{%- endfor -%}{%- endif -%}",
    "{{- '<|im_end|>\\n' -}}",
    "{%- endfor -%}",
    "{%- if add_generation_prompt -%}{{- '<|im_start|>assistant\\n' -}}{%- endif -%}"
);

/// Speech-recognition accuracy eval via `/v1/chat/completions`, through the
/// chunked form of the shared eval entry ([`measurement::eval_complete_chunked`]):
/// resumable checkpoint, failed rows sent again while they have attempts left,
/// crash recovery against one shared server, `parallel` slots, and no result
/// unless every sample has an answer. Where it differs from `vlm_eval`:
///
///   * The clips are the entry's `asr-eval-corpus/1` corpus, downloaded to disk
///     and checked against both its pins before the first prompt
///     ([`pipette_vl_spec_decode_corpus::asr_eval`]); they are read back a
///     chunk at a time, never held whole. A row's id is its position's
///     (`AsrEvalTask::sample_id`), so a resume that has settled every row
///     downloads nothing.
///   * Each prompt profile's llama.cpp recipe (its public card, spelled for
///     llama-server; see the provenance note above) is applied here: the server's chat template and template switches, and
///     each request's stop words, neutral samplers, beams and template
///     arguments ([`AsrRecipe`]).
///   * Thinking is always off: the cell's model flags state
///     `enable_thinking = false`, every server starts with `--reasoning off`,
///     and no raw flag or env may say otherwise.
///   * The context is sized from every clip's duration, read from its WAV
///     header as the corpus is checked, with no calibration run.
#[expect(
    clippy::too_many_arguments,
    reason = "the request's parts, the checkpoint opened on its identity, and the seams"
)]
pub fn run(
    benchmark: &AsrEvalBenchmark,
    runtime_flags: AsrEvalLlamacppCliStockToolsGgufVision,
    benchmark_flags: &BenchmarkFlagRef,
    program: &ResolvedLlamaServer,
    model: &ResolvedGgufVisionModel,
    checkpoint: EvalCompletionSession,
    run_context: &dyn RunContext,
    readiness_gate: ReadinessGate,
    observer: MeasurementObserver<'_>,
    client_config: Option<&LlamacppCliStockToolsExtraRuntimeConfiguration>,
) -> anyhow::Result<RunResponse> {
    let recipe = AsrRecipe::for_entry(benchmark, &runtime_flags)?;
    let benchmark_id = benchmark.benchmark_id.as_str();
    // The model the profile is run on, before anything is fetched: a
    // projector of another family, or a MiniCPM-o vocabulary that numbers its
    // turn ends differently, would run the recipe on the wrong model.
    check_model(&recipe, model).with_context(|| format!("asr_eval `{benchmark_id}`"))?;
    let declared = usize::try_from(benchmark.parameter_sample_count.into_inner().unsigned_abs())
        .context("parameter_sample_count does not fit in usize")?;
    let max_tokens = benchmark.parameter_max_tokens.into_inner().unsigned_abs();
    let generation = recipe.generation_tokens(max_tokens);
    let slots = recipe.slots;

    // Before the corpus, which can be gigabytes of audio: a control the cell
    // may not carry, or a launch the builder would refuse (a pinned context
    // past what llama-server parses), is refused for the price of a file
    // read. The floor stands in for the corpus's context.
    let provisional = with_ctx(runtime_flags.clone(), floor_ctx(0, generation), slots);
    let provisional = launch(&provisional, &recipe, program, model, client_config);
    provisional.check_thinking_controls()?;
    provisional.args()?;

    // A row's id is its position's, so the checkpoint alone says what is
    // left: a run with nothing left downloads nothing and starts no server.
    let task = &benchmark.parameter_asr_task;
    // plan-types' derivation, the one the evals server's ids follow.
    let id_at = |position: usize| task.sample_id(position as u64);
    if measurement::chunked_pending(&checkpoint, declared, id_at).is_empty() {
        tracing::info!(
            benchmark_id,
            "every sample is settled in the checkpoint; the corpus is not downloaded"
        );
        // The run that settled them was killed, and may have left its corpus.
        asr_corpus::remove_leftover(run_context, benchmark)?;
        (observer.on_before_rep_start)()?;
        let result_data = measurement::eval_complete_chunked(
            &benchmark.benchmark_id,
            declared,
            id_at,
            |offset| anyhow::bail!("no sample is left to read, asked for {offset}"),
            EvalMeasurementInFlight::Parallel(NonZeroUsize::MIN),
            checkpoint,
            (),
            |(), _, _| Ok(()),
        )?;
        (observer.on_after_rep_end)()?;
        let mut response = RunResponseBuilder::new(result_data).runtime_flags(Some(
            RuntimeFlags::AsrEvalLlamacppCliStockToolsGgufVision(
                AsrEvalLlamacppCliStockToolsGgufVision {
                    load_mode: get_load_mode(runtime_flags.mmap, runtime_flags.load_mode),
                    ..runtime_flags
                },
            ),
        ));
        // The checkpoint is keyed by the corpus digest, so its answers came
        // from the corpus the entry pins.
        if let Some(dataset) = &benchmark.dataset {
            response = response.dataset_digest(dataset.digest.clone());
        }
        return Ok(response.build());
    }

    // The corpus, checked against both its pins, every clip's context bound
    // read as the identity pass rebuilds it.
    let mut sizing = Sizing::default();
    let corpus = asr_corpus::download(run_context, benchmark, |sample| {
        sizing.add(&sample.messages, generation);
        Ok(())
    })?;
    asr_corpus::check_positions(benchmark_id, &corpus, declared, id_at)?;
    let budget = slot_budget(
        benchmark_id,
        &sizing,
        generation,
        runtime_flags.ctx_size,
        slots,
    )?;
    tracing::info!(
        benchmark_id,
        per_slot_budget = budget,
        generation,
        slots,
        pinned = runtime_flags.ctx_size.is_some(),
        unsized_clips = sizing.unreadable,
        "asr_eval context sized from the corpus's clips"
    );
    let runtime_flags = with_ctx(runtime_flags, budget, slots);
    let launch = launch(&runtime_flags, &recipe, program, model, client_config);
    // Refused before anything is spawned.
    let argv = launch.args()?;

    let request_timeout = measurement::http_timeout_for(benchmark_flags.http_timeout_seconds, 1);
    let idle_timeout = measurement::stream_idle_timeout_for(benchmark_flags.http_timeout_seconds);

    // Device protection, not score protection, as on vlm_eval: the audio tower
    // runs thousands of clips back to back. Best-effort, and only at the start.
    eval_pool::readiness_best_effort(readiness_gate, "asr_eval start");
    (observer.on_before_rep_start)()?;

    let spawn = || -> anyhow::Result<LlamaServer> {
        let port = pipette_subprocess::pick_server_port()?;
        let url = server::url(port);
        let ready = server::health(&url, request_timeout)?;
        let command = launch.build(port)?;
        let mut llama_server = start_process("llama-server", command, Duration::ZERO)?;
        llama_server.wait_ready(request_timeout, ready)?;
        Ok((llama_server, url))
    };
    let http = HttpClient::loopback("pipette", request_timeout)?;
    let slot = Mutex::new(ServerSlot {
        server: Server::Up(spawn()?),
        generation: 0,
    });
    let sender = AsrSender {
        recipe: &recipe,
        benchmark,
        max_tokens,
        generation,
        budget,
        slots,
        stream: if recipe.beams.is_some() {
            None
        } else {
            Some(
                HttpClient::streaming("pipette", idle_timeout)
                    .context("failed to build streaming HTTP client")?,
            )
        },
        // A terminal only, and only one slot, as on vlm_eval.
        tee_stderr: slots <= 1 && std::io::IsTerminal::is_terminal(&std::io::stderr()),
    };
    let workers = usize::try_from(slots)
        .context("parallel does not fit in usize")?
        .clamp(1, declared.max(1));
    // Enough samples per chunk that the slots rarely wait on a chunk's
    // slowest clip, within a bound on the audio held.
    let chunk = CHUNK_SAMPLES.max(workers.saturating_mul(CHUNK_SAMPLES_PER_SLOT));

    let result_data = measurement::eval_complete_chunked(
        &benchmark.benchmark_id,
        declared,
        id_at,
        |offset| corpus.chunk(offset, chunk, CHUNK_AUDIO_BYTES),
        EvalMeasurementInFlight::Parallel(NonZeroUsize::new(workers).unwrap_or(NonZeroUsize::MIN)),
        checkpoint,
        &slot,
        |slot, samples, record| {
            samples.iter().enumerate().try_for_each(|(at, sample)| {
                if let Some(reason) = sender.does_not_fit(sample) {
                    // Never sent, so no attempt: a larger context runs it.
                    return record(at, EvalMeasurementReading::Unfit { reason });
                }
                let Some((base_url, generation)) = slot_server(slot, &spawn)? else {
                    return record(
                        at,
                        EvalMeasurementReading::Lost {
                            reason: "llama-server did not come back after a crash".to_owned(),
                        },
                    );
                };
                let reading = read_sample(slot, generation, sample, &argv, CHAT_ENDPOINT, || {
                    sender.run_sample(&http, &base_url, sample)
                })?;
                record(at, reading)
            })?;
            Ok(())
        },
    )?;
    (observer.on_after_rep_end)()?;
    let ServerSlot { server, .. } = slot.into_inner().unwrap_or_else(PoisonError::into_inner);
    // Which corpus the answers came from, as the row records it.
    let response = RunResponseBuilder::new(result_data)
        .runtime_flags(Some(RuntimeFlags::AsrEvalLlamacppCliStockToolsGgufVision(
            runtime_flags,
        )))
        .dataset_digest(corpus.digest().clone());
    Ok(match server {
        Server::Up((mut llama_server, _)) | Server::Down(mut llama_server) => {
            response.process(&mut llama_server).build()
        }
        Server::RestartFailed => response.build(),
    })
}

/// The samples a chunk carries at the most, and per slot, so that every
/// slot has several to work through before the chunk's last one.
const CHUNK_SAMPLES: usize = 32;
const CHUNK_SAMPLES_PER_SLOT: usize = 8;
/// The audio a chunk holds at the most (one clip past it, alone): its
/// rebuilt base64 is a third more.
const CHUNK_AUDIO_BYTES: u64 = 32 * 1024 * 1024;

/// The recipe rule, shared with every engine's asr_eval.
pub(super) use pipette_vl_spec_decode_corpus::asr_eval::on_recipe;

/// What a corpus's clips need of the context, gathered one sample at a time.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
struct Sizing {
    /// The largest bound of a sample whose clips' durations could be read.
    needed: Option<u32>,
    /// The longest prompt text, in its bound's tokens.
    text: u32,
    /// Samples a clip's duration could not be read for: the server says
    /// whether they fit.
    unreadable: usize,
}

impl Sizing {
    fn add(&mut self, messages: &[ChatMessage], generation: u32) {
        match sample_ctx(messages, generation) {
            Some(needed) => self.needed = Some(self.needed.map_or(needed, |n| n.max(needed))),
            None => self.unreadable += 1,
        }
        self.text = self.text.max(text_tokens(messages));
    }
}

/// The context one slot holds: an operator's pinned total divided across the
/// slots, as llama-server divides it, once it is known to hold the corpus's
/// longest clip; else that sample's bound, floored at a
/// [`CLIP_FLOOR_SECONDS`] clip, within the ceiling across the slots. A clip
/// the context cannot hold is refused here, before any server starts: the run
/// would end with no result.
fn slot_budget(
    benchmark_id: &str,
    sizing: &Sizing,
    generation: u32,
    pinned: Option<u32>,
    slots: u32,
) -> anyhow::Result<u32> {
    let needed = sizing.needed;
    if let Some(total) = pinned {
        let budget = total / slots.max(1);
        if let Some(needed) = needed.filter(|&needed| needed > budget) {
            anyhow::bail!(
                "asr_eval `{benchmark_id}`: the pinned ctx_size {total} gives each of {slots} \
                 slot(s) {budget} tokens, and the corpus's longest sample needs up to {needed} \
                 (its audio, text, template margin and {generation} generated); pin a ctx_size \
                 of at least {}, or drop it to size the context from the clips",
                needed.saturating_mul(slots),
            );
        }
        return Ok(budget);
    }
    let per_slot = needed.unwrap_or(0).max(floor_ctx(sizing.text, generation));
    let ceiling = ASR_EVAL_CTX_CEILING;
    anyhow::ensure!(
        per_slot.saturating_mul(slots) <= ceiling,
        "asr_eval `{benchmark_id}`: {per_slot} tokens of context per slot x {slots} slot(s) is \
         past the {ceiling} ceiling; lower `parallel`, lower parameter_max_tokens, or pin \
         ctx_size explicitly",
    );
    Ok(per_slot)
}

/// The slot count the cell opens: `parallel`, or one.
fn typed_slots(runtime_flags: &AsrEvalLlamacppCliStockToolsGgufVision) -> u32 {
    runtime_flags
        .parallel
        .map_or(1, |n| n.into_inner().unsigned_abs())
}

/// The cell's flags with the weights pinned in RAM unless it said how to load
/// them, and, when it pinned no context, `per_slot` for every slot: what
/// llama-server divides is the total. An operator's own total is not
/// multiplied, as on vlm_eval.
fn with_ctx(
    runtime_flags: AsrEvalLlamacppCliStockToolsGgufVision,
    per_slot: u32,
    slots: u32,
) -> AsrEvalLlamacppCliStockToolsGgufVision {
    AsrEvalLlamacppCliStockToolsGgufVision {
        load_mode: get_load_mode(runtime_flags.mmap, runtime_flags.load_mode),
        ctx_size: runtime_flags
            .ctx_size
            .or_else(|| Some(per_slot.saturating_mul(slots))),
        ..runtime_flags
    }
}

/// The cell's server, every field named: a flag plan-types adds fails to
/// compile here until this cell renders it or sets it aside. The profile's
/// template and beam flags, and `--reasoning off`, ride in `extra`. Every clip
/// is a new prompt, so llama-server's prompt caches would only copy state out
/// between clips, holding up every slot's decoding, as on vlm_eval: they are
/// off ([`ServerCommandBuilder::without_prompt_caches`]).
fn launch(
    flags: &AsrEvalLlamacppCliStockToolsGgufVision,
    recipe: &AsrRecipe,
    program: &ResolvedLlamaServer,
    model: &ResolvedGgufVisionModel,
    client_config: Option<&LlamacppCliStockToolsExtraRuntimeConfiguration>,
) -> ServerCommandBuilder {
    let AsrEvalLlamacppCliStockToolsGgufVision {
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
    ServerCommandBuilder::new(program.clone(), &model.backbone, LoggingMarker::AsrEval)
        .mmproj(&model.mmproj)
        .threads(threads)
        .number_gpu_layers(number_gpu_layers)
        .mmap(mmap)
        .load_mode(load_mode)
        .flash_attention(flash_attention)
        .raw(raw)
        .ctx_size(ctx_size)
        .no_cache(no_cache)
        .without_prompt_caches()
        .slots(parallel.map(|n| n.into_inner().unsigned_abs()))
        .one_slot_spelled(recipe.beams.is_some())
        .extra(recipe.server_flags())
        .envs(envs)
        .client_config(client_config.cloned())
}

/// The audio encoders a prompt profile's model carries, as its projector
/// names them (`clip.audio.projector_type`, or `clip.projector_type` in a
/// single-encoder file): the names in pipette's ASR build of llama.cpp
/// (`tools/mtmd/clip-impl.h`), and LFM3's `lfm3a` from pipette's LFM3 build of
/// llama.cpp, as the ASR timing cells read them (`asr::AudioEncoder` there).
/// OpenBMB's whisper-architecture MiniCPM-o audio GGUF names none; the ASR
/// build loads it as `minicpmoa`, and so it is read
/// ([`gguf_meta::OPENBMB_MINICPMO_AUDIO_ENCODER`]).
fn profile_encoders(profile: KnownAsrPromptProfile) -> &'static [&'static str] {
    match profile {
        KnownAsrPromptProfile::Gemma4 => &["gemma4a", "gemma4ua"],
        KnownAsrPromptProfile::Qwen25Omni => &["qwen2.5o", "qwen2.5oa"],
        KnownAsrPromptProfile::Voxtral => &["voxtral"],
        KnownAsrPromptProfile::MinicpmO => &["minicpmoa"],
        KnownAsrPromptProfile::Lfm25Audio => &["lfm2a"],
        KnownAsrPromptProfile::Lfm3 => &["lfm3a"],
    }
}

/// The model is the one the profile's recipe is for: its projector holds the
/// profile's audio encoder, and, for MiniCPM-o, its vocabulary names the
/// token ids the recipe hard-codes as the pieces it means. Either mismatch
/// would run one family's template, stops and decoding on another's model,
/// and transcribe without failing.
fn check_model(recipe: &AsrRecipe, model: &ResolvedGgufVisionModel) -> anyhow::Result<()> {
    let profile = recipe.profile;
    let mmproj = model.mmproj.as_path();
    let metadata = gguf_meta::projector_metadata(mmproj)?;
    anyhow::ensure!(
        metadata.holds_audio_encoder(),
        "the projector {} holds no audio encoder (`clip.has_audio_encoder`), and the `{profile}` \
         prompt profile transcribes with one",
        mmproj.display(),
    );
    let encoder = metadata.audio_encoder_name().unwrap_or_default();
    let expected = profile_encoders(profile);
    anyhow::ensure!(
        expected.contains(&encoder),
        "the projector {} holds the `{encoder}` audio encoder, and the `{profile}` prompt \
         profile is the recipe for {}; the entry would run another family's template, stops \
         and decoding on this model. Run the entry on its family's model",
        mmproj.display(),
        expected.join(" or "),
    );
    if profile == KnownAsrPromptProfile::MinicpmO {
        let backbone = model.backbone.as_path();
        let ids: Vec<u32> = MINICPM_O_TOKENS.iter().map(|(id, _)| *id).collect();
        let pieces = gguf_meta::token_pieces(backbone, &ids)?.with_context(|| {
            format!(
                "{} carries no vocabulary (`tokenizer.ggml.tokens`) to check MiniCPM-o's token \
                 ids against",
                backbone.display()
            )
        })?;
        for ((id, want), got) in MINICPM_O_TOKENS.iter().zip(pieces) {
            anyhow::ensure!(
                got.as_deref() == Some(*want),
                "token {id} of {} is {}, not `{want}`: the `minicpm_o` recipe ends its answer on \
                 token {MINICPM_O_EOT_TOKEN} (`<|tts_eos|>`) and its beams on {:?} \
                 (`<|im_end|>`, `<|endoftext|>`), as MiniCPM-o 4.5 numbers them; this GGUF \
                 numbers them otherwise",
                backbone.display(),
                got.map_or_else(
                    || "past its vocabulary".to_owned(),
                    |got| format!("`{got}`")
                ),
                MINICPM_O_BEAM_EOS,
            );
        }
    }
    Ok(())
}

/// A prompt profile's llama.cpp recipe, its public card spelled for
/// llama-server: what the server starts with, and what each request adds to
/// the entry's decoding.
#[derive(Debug, Clone, PartialEq)]
struct AsrRecipe {
    profile: KnownAsrPromptProfile,
    /// The entry's decoding with every knob it leaves unset at the profile
    /// card's value: plan-types' [`AsrEvalBenchmark::effective_decoding`],
    /// the one table mgmt and the warehouse read too.
    decoding: AsrDecoding,
    /// The beam width when the entry decodes by beam search (MiniCPM-o only).
    beams: Option<u32>,
    /// The slot count the server opens.
    slots: u32,
}

impl AsrRecipe {
    /// The recipe for an entry on a cell, once both are known to be one this
    /// client can run as the profile's card does. Everything here fails before
    /// the corpus is downloaded or a server spawned.
    fn for_entry(
        benchmark: &AsrEvalBenchmark,
        runtime_flags: &AsrEvalLlamacppCliStockToolsGgufVision,
    ) -> anyhow::Result<Self> {
        let benchmark_id = benchmark.benchmark_id.as_str();
        let profile = match &benchmark.parameter_prompt_profile {
            AsrPromptProfile::Known(profile) => *profile,
            AsrPromptProfile::Unknown(name) => anyhow::bail!(
                "asr_eval `{benchmark_id}`: this client has no llama.cpp recipe for prompt profile \
                 `{name}` (it runs {})",
                KnownAsrPromptProfile::ALL
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join(", "),
            ),
        };
        if let Some((flag, _)) = server::raw_parallel(&runtime_flags.raw) {
            anyhow::bail!(
                "asr_eval `{benchmark_id}`: raw {flag} in --runtime-flags; set the cell's typed \
                 `parallel`, which the context is sized for"
            );
        }
        // A second template or metadata override would decide which one
        // renders or ends the turn; a raw beam search would run beside the
        // entry's `parameter_num_beams` rather than as it; a raw sampler
        // chain is the request's to spell; a per-slot limit or output cap
        // would cut a transcript the context was sized to hold.
        if let Some(flag) =
            thinking_controls::raw_flag(&runtime_flags.raw, thinking_controls::ASR_OWNED_FLAGS)
        {
            anyhow::bail!(
                "asr_eval `{benchmark_id}`: raw {flag} in --runtime-flags; the `{profile}` prompt \
                 profile owns the chat template, the metadata overrides, the beam search and the \
                 sampler chain, and the entry the output cap (`parameter_max_tokens`) the context is sized for. \
                 Drop the raw flag"
            );
        }
        // What the profile's card can run (plan-types' check_decoding: beams
        // only on MiniCPM-o and alone, no knob Voxtral's chain ignores), with
        // every knob the entry leaves unset at the card's value.
        let decoding = benchmark
            .effective_decoding()
            .with_context(|| format!("asr_eval `{benchmark_id}`"))?;
        let beams = match decoding {
            AsrDecoding::Beams(beams) => Some(beams.num_beams.into_inner().unsigned_abs()),
            AsrDecoding::Sampling(_) => None,
        };
        let slots = typed_slots(runtime_flags);
        if beams.is_some() {
            anyhow::ensure!(
                slots == 1,
                "asr_eval `{benchmark_id}`: a beam search runs on one slot (the ASR build's \
                 --beam-search needs --parallel 1); drop `parallel` from the cell",
            );
        }
        Ok(Self {
            profile,
            decoding,
            beams,
            slots,
        })
    }

    /// The tokens one sample's generation holds in its slot: `max_tokens`,
    /// once per beam. Pipette's ASR build of llama.cpp splits a beam search's
    /// room among its beams (`beam_search_and_respond` in
    /// `tools/server/server-context.cpp` at 362d8dc3e: `room = (n_ctx - prompt) / n_beams`, and the cap is
    /// `min(n_predict, room)`), so a slot short of the prompt plus
    /// `n_beams * max_tokens` would stop the best beam early with an ordinary
    /// length stop.
    fn generation_tokens(&self, max_tokens: u32) -> u32 {
        max_tokens.saturating_mul(self.beams.unwrap_or(1))
    }

    /// The argv the profile adds to every launch and restart.
    fn server_flags(&self) -> Vec<String> {
        let mut flags = vec!["--reasoning", "off"];
        match self.profile {
            KnownAsrPromptProfile::Gemma4
            | KnownAsrPromptProfile::Qwen25Omni
            | KnownAsrPromptProfile::Lfm25Audio => {}
            KnownAsrPromptProfile::Voxtral => {
                flags.extend(["--jinja", "--chat-template", VOXTRAL_CHAT_TEMPLATE]);
            }
            KnownAsrPromptProfile::Lfm3 => {
                flags.extend(["--jinja", "--chat-template", LFM3_ASR_TEMPLATE]);
            }
            KnownAsrPromptProfile::MinicpmO => {}
        }
        let mut flags: Vec<String> = flags.into_iter().map(str::to_owned).collect();
        if self.profile == KnownAsrPromptProfile::MinicpmO {
            flags.extend([
                "--chat-template-kwargs".to_owned(),
                MINICPM_O_SERVER_KWARGS.to_owned(),
                "--override-kv".to_owned(),
                format!("tokenizer.ggml.eot_token_id=int:{MINICPM_O_EOT_TOKEN}"),
            ]);
        }
        if let Some(beams) = self.beams {
            flags.extend(["--beam-search".to_owned(), beams.to_string()]);
        }
        flags
    }

    /// The stop words, pipette's own: the turn and tool-response markers of
    /// Gemma 4's chat format and Qwen2.5-Omni's `<|im_end|>`, which their
    /// GGUFs do not all mark end-of-generation.
    fn stop(&self) -> Option<&'static [&'static str]> {
        match self.profile {
            KnownAsrPromptProfile::Gemma4 => Some(&["<eos>", "<turn|>", "<|tool_response>"]),
            KnownAsrPromptProfile::Qwen25Omni => Some(&["<|im_end|>"]),
            KnownAsrPromptProfile::Voxtral
            | KnownAsrPromptProfile::MinicpmO
            | KnownAsrPromptProfile::Lfm25Audio
            | KnownAsrPromptProfile::Lfm3 => None,
        }
    }

    /// The decoding a request carries: the beams, or the entry's effective
    /// sampling ([`Self::decoding`]) with every other sampler neutral and the
    /// sampler chain spelled out. A GGUF's `general.sampling.*` defaults, and
    /// a server's own chain (`general.sampling.sequence`, `--samplers`), apply
    /// only to what a request leaves unset, so it leaves nothing unset. A
    /// whole-sequence repeat window is spelled as the slot's context,
    /// `per_slot`; the temperature-then-top_p chain, and llama.cpp's default
    /// chain on every other profile, as llama-server's `samplers` list.
    fn request_decoding(
        &self,
        benchmark: &AsrEvalBenchmark,
        sample_id: &str,
        per_slot: u32,
    ) -> Value {
        let sampling = match self.decoding {
            AsrDecoding::Beams(beams) => {
                return json!({
                    "n_beams": beams.num_beams.into_inner().unsigned_abs(),
                    "beam_repetition_penalty": beams.repetition_penalty.into_inner(),
                    "beam_length_penalty": beams.length_penalty.into_inner(),
                    "beam_eos_token_ids": MINICPM_O_BEAM_EOS,
                    "beam_bf16_logits": true,
                });
            }
            AsrDecoding::Sampling(sampling) => sampling,
        };
        let mut body = json!({
            "temperature": sampling.temperature.as_f64(),
            "top_k": i32::from(sampling.top_k),
            "top_p": sampling.top_p.as_f64(),
            "min_p": sampling.min_p.as_f64(),
            "typical_p": 1.0,
            "top_n_sigma": -1.0,
            "repeat_penalty": sampling.repeat_penalty.as_f64(),
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "dry_multiplier": 0.0,
            "xtc_probability": 0.0,
            "mirostat": 0,
        });
        if let Some(body) = body.as_object_mut() {
            match sampling.repeat_window {
                // Transformers' repetition penalty looks back over the whole
                // sequence: MiniCPM-o's chat() greedy branch sets 1.02
                // (`prepare_generation_config` in OpenBMB's
                // `modeling_minicpmo.py` at d59117d1), and llama-server spells
                // the whole sequence as every token a slot can hold.
                AsrRepeatWindow::WholeSequence => {
                    body.insert("repeat_last_n".to_owned(), json!(per_slot));
                }
                AsrRepeatWindow::RuntimeDefault => {}
            }
            match sampling.sampler_chain {
                // vLLM applies temperature before top_p; llama.cpp's default
                // order changes Voxtral's sampled distribution. A top_p of 1.0
                // is a no-op, so an entry that pins none runs temperature
                // alone, the only knob Mistral's transcription usage sets.
                AsrSamplerChain::TemperatureThenTopP => {
                    let samplers: &[&str] = match benchmark.parameter_top_p {
                        Some(_) => &["temperature", "top_p"],
                        None => &["temperature"],
                    };
                    body.insert("samplers".to_owned(), json!(samplers));
                }
                AsrSamplerChain::RuntimeDefault => {
                    body.insert("samplers".to_owned(), json!(LLAMA_CPP_DEFAULT_SAMPLERS));
                }
            }
            if sampling.seed.is_some()
                && let Some(seed) = benchmark.sample_seed(sample_id)
            {
                body.insert("seed".to_owned(), json!(seed));
            }
        }
        body
    }

    /// The template arguments a request carries: the card's, then the
    /// sample's own (Voxtral's `transcription` and `language`), with thinking
    /// closed whatever either says.
    fn template_kwargs(&self, sample: Option<&Map<String, Value>>) -> Option<Map<String, Value>> {
        let closes_thinking = matches!(
            self.profile,
            KnownAsrPromptProfile::Gemma4 | KnownAsrPromptProfile::MinicpmO
        );
        let mut kwargs = match self.profile {
            KnownAsrPromptProfile::MinicpmO => minicpm_o_template_kwargs(),
            KnownAsrPromptProfile::Gemma4
            | KnownAsrPromptProfile::Qwen25Omni
            | KnownAsrPromptProfile::Voxtral
            | KnownAsrPromptProfile::Lfm25Audio
            | KnownAsrPromptProfile::Lfm3 => Map::new(),
        };
        if let Some(sample) = sample {
            kwargs.extend(sample.clone());
        }
        if closes_thinking || kwargs.contains_key("enable_thinking") {
            kwargs.insert("enable_thinking".to_owned(), Value::Bool(false));
        }
        (!kwargs.is_empty()).then_some(kwargs)
    }

    /// One sample's request body.
    fn request(
        &self,
        benchmark: &AsrEvalBenchmark,
        sample: &EvalMeasurementSample<'_>,
        max_tokens: u32,
        per_slot: u32,
        stream: bool,
    ) -> Value {
        let mut body = self.request_decoding(benchmark, sample.id, per_slot);
        if let Some(body) = body.as_object_mut() {
            body.insert("messages".to_owned(), json!(sample.messages));
            body.insert("max_tokens".to_owned(), json!(max_tokens));
            body.insert("reasoning_format".to_owned(), json!("none"));
            if let Some(stop) = self.stop() {
                body.insert("stop".to_owned(), json!(stop));
            }
            if let Some(kwargs) = self.template_kwargs(sample.chat_template_kwargs) {
                body.insert("chat_template_kwargs".to_owned(), Value::Object(kwargs));
            }
            if self.profile == KnownAsrPromptProfile::MinicpmO {
                // Pipette's own: each clip a fresh prompt, as OpenBMB's chat()
                // processes each call alone; no prefix of one clip's prompt is
                // reused for the next.
                body.insert("cache_prompt".to_owned(), json!(false));
            }
            if stream {
                body.insert("stream".to_owned(), json!(true));
                // The usage trailer, so completion_tokens pairs with the stop.
                body.insert(
                    "stream_options".to_owned(),
                    json!({ "include_usage": true }),
                );
            }
        }
        body
    }
}

/// What every attempt of the cell reads.
struct AsrSender<'a> {
    recipe: &'a AsrRecipe,
    benchmark: &'a AsrEvalBenchmark,
    max_tokens: u32,
    /// What a sample's generation holds in its slot
    /// ([`AsrRecipe::generation_tokens`]).
    generation: u32,
    /// The context one slot holds.
    budget: u32,
    slots: u32,
    /// The streaming client, or `None` for a beam search, which answers in one
    /// response.
    stream: Option<StreamClient>,
    tee_stderr: bool,
}

impl AsrSender<'_> {
    /// Why a clip cannot be sent: its bound is past what one slot holds, so
    /// the server would refuse it or cut its transcript. `None` when it fits,
    /// or when its duration cannot be read (the server then says, and
    /// [`Self::cut_short`] reads a cut answer as not fitting).
    fn does_not_fit(&self, sample: &EvalMeasurementSample<'_>) -> Option<String> {
        let EvalMessages::Chat(messages) = sample.messages else {
            return None;
        };
        let needed = sample_ctx(messages, self.generation)?;
        (needed > self.budget).then(|| {
            format!(
                "the clip needs up to {needed} tokens of context per slot ({} of them \
                 generated) and the server holds {}; pin a ctx_size of at least {} to run it",
                self.generation,
                self.budget,
                needed.saturating_mul(self.slots),
            )
        })
    }

    /// A reply the server cut short of `max_tokens` for want of context, as
    /// `Unfit`: a length stop with fewer tokens than asked is the slot running
    /// out (or a beam's room shrinking), not the cap, and scoring it would
    /// checkpoint a partial transcript under an identity that does not name
    /// the context.
    fn cut_short(&self, reply: &ChatReply) -> Option<EvalMeasurementReading> {
        let generated = reply.completion_tokens?;
        (reply.finish_reason.as_deref() == Some("length") && generated < u64::from(self.max_tokens))
            .then(|| EvalMeasurementReading::Unfit {
                reason: format!(
                    "the server stopped the answer at {generated} of {} tokens for want of \
                     context: one slot holds {} tokens, and this clip's length could not be \
                     read to size it; pin a larger ctx_size to run it",
                    self.max_tokens, self.budget,
                ),
            })
    }

    /// One attempt at the sample. A request the server refuses for want of
    /// context, or an answer it cuts short for want of it, is `Unfit`: the
    /// duration of a clip the server refuses could not be read to size it.
    fn run_sample(
        &self,
        http: &HttpClient,
        base_url: &str,
        sample: &EvalMeasurementSample<'_>,
    ) -> anyhow::Result<EvalMeasurementReading> {
        let url = format!("{base_url}{CHAT_ENDPOINT}");
        let body = self.recipe.request(
            self.benchmark,
            sample,
            self.max_tokens,
            self.budget,
            self.stream.is_some(),
        );
        let reply = match &self.stream {
            Some(client) => stream_chat_completion(client, &url, &body, self.tee_stderr),
            None => http
                .json_request::<Value>(Method::POST, &url, None, Some(body))
                .with_context(|| format!("POST {CHAT_ENDPOINT}"))
                .and_then(|response| chat_reply(&response)),
        };
        let reply = match reply {
            Ok(reply) => reply,
            Err(err) if is_context_refusal(&err) => {
                return Ok(EvalMeasurementReading::Unfit {
                    reason: format!(
                        "llama-server refused the clip for want of context ({}): one slot holds \
                         {} tokens, and this clip's length could not be read to size it; pin a \
                         larger ctx_size to run it",
                        eval_pool::root_cause(&err),
                        self.budget,
                    ),
                });
            }
            Err(err) => return Err(err),
        };
        if let Some(unfit) = self.cut_short(&reply) {
            return Ok(unfit);
        }
        let (stop_reason, stop_detail) = reply.stop();
        // Stored as the server returned it, as on every eval: the score
        // server extracts each profile's transcript. MiniCPM-o's answer so
        // keeps the TTS template's opening, an empty think block and
        // `<|tts_bos|>`, which the score server drops (it reads the text
        // after the last `<|tts_bos|>`, as OpenBMB's `_generate_mel_spec`
        // does), so one rule, in one place, decides what is scored.
        let reading = EvalMeasurementReading::Completed {
            completion: reply.content,
            stop_reason,
            stop_detail,
            completion_tokens: reply.completion_tokens,
        };
        log_completed(sample, &reading);
        Ok(reading)
    }
}

/// Whether `err` is llama-server refusing a request for want of context: its
/// `exceed_context_size_error`, or the ASR build's beam search finding no room.
fn is_context_refusal(err: &anyhow::Error) -> bool {
    let text = format!("{err:#}");
    text.contains("exceed_context_size_error")
        || text.contains("exceeds the available context size")
        || text.contains("no room in the context for beam search")
}

/// A one-response chat completion read as the stream would be.
fn chat_reply(response: &Value) -> anyhow::Result<ChatReply> {
    let choice = response
        .get("choices")
        .and_then(|c| c.get(0))
        .context("/v1/chat/completions response carries no choice")?;
    let content = choice
        .get("message")
        .and_then(|m| m.get("content"))
        .and_then(Value::as_str)
        .context("/v1/chat/completions response missing message content")?;
    Ok(ChatReply {
        content: content.to_owned(),
        finish_reason: choice
            .get("finish_reason")
            .and_then(Value::as_str)
            .map(str::to_owned),
        completion_tokens: response
            .get("usage")
            .and_then(|u| u.get("completion_tokens"))
            .and_then(Value::as_u64),
        ..ChatReply::default()
    })
}

/// The context one slot is sized for at the least: a [`CLIP_FLOOR_SECONDS`]
/// clip beside `text` tokens of prompt, the template margin and `generation`.
fn floor_ctx(text: u32, generation: u32) -> u32 {
    audio_tokens(CLIP_FLOOR_SECONDS)
        .saturating_add(text)
        .saturating_add(TEMPLATE_MARGIN_TOKENS)
        .saturating_add(generation)
}

/// One sample's context bound: its clips' audio tokens, its text, the
/// template's margin and the `generation` it holds
/// ([`AsrRecipe::generation_tokens`]). `None` when a clip's duration cannot be
/// read.
fn sample_ctx(messages: &[ChatMessage], generation: u32) -> Option<u32> {
    let audio = messages
        .iter()
        .flat_map(|message| parts(&message.content))
        .filter_map(|part| match part {
            ContentPart::InputAudio { input_audio } => Some(input_audio),
            ContentPart::Text { .. } | ContentPart::ImageUrl { .. } => None,
        })
        .map(|clip| clip_seconds(&clip.data, clip.format).map(audio_tokens))
        .try_fold(0u32, |sum, tokens| tokens.map(|t| sum.saturating_add(t)))?;
    Some(
        audio
            .saturating_add(text_tokens(messages))
            .saturating_add(TEMPLATE_MARGIN_TOKENS)
            .saturating_add(generation),
    )
}

/// A message's parts; a string content has none to scan for audio.
fn parts(content: &MessageContent) -> &[ContentPart] {
    match content {
        MessageContent::Text(_) => &[],
        MessageContent::Parts(parts) => parts,
    }
}

/// An upper bound on the prompt text's tokens: one per byte.
fn text_tokens(messages: &[ChatMessage]) -> u32 {
    let bytes: usize = messages
        .iter()
        .map(|message| match &message.content {
            MessageContent::Text(text) => text.len(),
            MessageContent::Parts(parts) => parts
                .iter()
                .map(|part| match part {
                    ContentPart::Text { text } => text.len(),
                    ContentPart::ImageUrl { .. } | ContentPart::InputAudio { .. } => 0,
                })
                .sum(),
        })
        .sum();
    u32::try_from(bytes).unwrap_or(u32::MAX)
}

/// An upper bound on a clip's audio tokens across the supported encoders: 25
/// a second, or 750 (Qwen2.5-Omni's window, twice Voxtral's) for every
/// started 30-second window, whichever is more.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "a clip's seconds are non-negative and far below u32::MAX tokens; `as` saturates"
)]
fn audio_tokens(seconds: f64) -> u32 {
    let by_rate = (seconds.max(0.0) * AUDIO_TOKENS_PER_SECOND).ceil() as u32;
    let windows = (seconds.max(0.0) / AUDIO_WINDOW_SECONDS).floor() as u32 + 1;
    by_rate.max(AUDIO_TOKENS_PER_WINDOW.saturating_mul(windows))
}

/// A clip's duration in seconds, from the header of its base64 bytes: a WAV's
/// `fmt ` and `data` chunks, or a FLAC's STREAMINFO. `None` for another
/// container or a header this cannot read.
fn clip_seconds(data: &str, format: AudioFormat) -> Option<f64> {
    let head = data.get(..data.len().min(HEADER_BASE64_CHARS) / 4 * 4)?;
    let header = base64::engine::general_purpose::STANDARD
        .decode(head)
        .ok()?;
    match format {
        AudioFormat::Wav => wav_seconds(&header, decoded_len(data)),
        AudioFormat::Flac => flac_seconds(&header),
        AudioFormat::Mp3 | AudioFormat::Ogg => None,
    }
}

/// The byte length `data` decodes to, padding excluded.
fn decoded_len(data: &str) -> usize {
    let padding = data.bytes().rev().take_while(|&b| b == b'=').count();
    (data.len() / 4 * 3).saturating_sub(padding)
}

fn u16_at(bytes: &[u8], at: usize) -> Option<u16> {
    Some(u16::from_le_bytes(bytes.get(at..at + 2)?.try_into().ok()?))
}

fn u32_at(bytes: &[u8], at: usize) -> Option<u32> {
    Some(u32::from_le_bytes(bytes.get(at..at + 4)?.try_into().ok()?))
}

/// A RIFF/WAVE clip's seconds: its `data` bytes over the `fmt ` byte rate. A
/// `data` size of zero or past the file (a streamed WAV) reads as the rest of
/// the file.
fn wav_seconds(header: &[u8], file_len: usize) -> Option<f64> {
    if header.get(..4)? != b"RIFF" || header.get(8..12)? != b"WAVE" {
        return None;
    }
    let mut at = 12;
    let mut byte_rate = None;
    while let (Some(id), Some(size)) = (header.get(at..at + 4), u32_at(header, at + 4)) {
        let body = at + 8;
        match id {
            b"fmt " => {
                let channels = u16_at(header, body + 2)?;
                let rate = u32_at(header, body + 4)?;
                let bits = u16_at(header, body + 14)?;
                byte_rate =
                    Some(u64::from(rate) * u64::from(channels) * u64::from(bits).div_ceil(8));
            }
            b"data" => {
                let rate = byte_rate.filter(|&r| r > 0)?;
                let rest = u64::try_from(file_len.saturating_sub(body)).ok()?;
                let size = u64::from(size);
                let bytes = if size == 0 || size > rest { rest } else { size };
                #[expect(clippy::cast_precision_loss, reason = "byte counts far below 2^53")]
                return Some(bytes as f64 / rate as f64);
            }
            _ => {}
        }
        // Chunks are padded to an even size.
        at = body + usize::try_from(size).ok()? + usize::try_from(size % 2).ok()?;
    }
    None
}

/// A FLAC clip's seconds: STREAMINFO's total samples over its sample rate.
/// `None` when the encoder left the total unknown (zero).
fn flac_seconds(header: &[u8]) -> Option<f64> {
    if header.get(..4)? != b"fLaC" || header.get(4)? & 0x7f != 0 {
        return None;
    }
    // STREAMINFO's body starts at 8; the rate, channels, depth and total
    // samples are packed into its bytes 10..18.
    let packed: [u8; 8] = header.get(18..26)?.try_into().ok()?;
    let packed = u64::from_be_bytes(packed);
    let rate = packed >> 44;
    let total = packed & 0x000f_ffff_ffff;
    #[expect(clippy::cast_precision_loss, reason = "sample counts far below 2^53")]
    (rate > 0 && total > 0).then(|| total as f64 / rate as f64)
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use pipette_plan_types::{
        BeamLengthPenalty, BeamRepetitionPenalty, BenchmarkEvalCompletionStopReason, BenchmarkId,
        ChatRole, EvalSample, InputAudio, NumBeams, PositiveCount, SamplingSeed,
        SamplingTemperature, TopP,
    };

    use super::*;
    use crate::runtimes::test_llama_server;

    /// A mono 16-bit WAV of `seconds` at 16 kHz, base64.
    fn wav(seconds: f64) -> String {
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "test durations are small and positive"
        )]
        let samples = (seconds * 16_000.0) as u32;
        let data = samples * 2;
        let mut bytes = Vec::new();
        bytes.extend(b"RIFF");
        bytes.extend((36 + data).to_le_bytes());
        bytes.extend(b"WAVEfmt ");
        bytes.extend(16u32.to_le_bytes());
        bytes.extend(1u16.to_le_bytes());
        bytes.extend(1u16.to_le_bytes());
        bytes.extend(16_000u32.to_le_bytes());
        bytes.extend(32_000u32.to_le_bytes());
        bytes.extend(2u16.to_le_bytes());
        bytes.extend(16u16.to_le_bytes());
        bytes.extend(b"data");
        bytes.extend(data.to_le_bytes());
        bytes.resize(bytes.len() + data as usize, 0);
        base64::engine::general_purpose::STANDARD.encode(bytes)
    }

    /// A mono 32-bit float WAV of `seconds` at 16 kHz, base64, as soundfile
    /// writes the scores server's `subtype="FLOAT"` clips: an 18-byte IEEE
    /// float `fmt `, a `fact` and a `PEAK` chunk before `data`, and with
    /// `list_first` a `LIST` chunk (odd-sized, so padded) before `fmt `.
    fn float_wav(seconds: f64, list_first: bool) -> String {
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "test durations are small and positive"
        )]
        let samples = (seconds * 16_000.0) as u32;
        let data = samples * 4;
        let mut chunks = Vec::new();
        if list_first {
            chunks.extend(b"LIST");
            chunks.extend(5u32.to_le_bytes());
            chunks.extend(b"INFOx\0");
        }
        chunks.extend(b"fmt ");
        chunks.extend(18u32.to_le_bytes());
        chunks.extend(3u16.to_le_bytes()); // WAVE_FORMAT_IEEE_FLOAT
        chunks.extend(1u16.to_le_bytes());
        chunks.extend(16_000u32.to_le_bytes());
        chunks.extend(64_000u32.to_le_bytes());
        chunks.extend(4u16.to_le_bytes());
        chunks.extend(32u16.to_le_bytes());
        chunks.extend(0u16.to_le_bytes()); // cbSize
        chunks.extend(b"fact");
        chunks.extend(4u32.to_le_bytes());
        chunks.extend(samples.to_le_bytes());
        chunks.extend(b"PEAK");
        chunks.extend(16u32.to_le_bytes());
        chunks.extend(1u32.to_le_bytes());
        chunks.extend(0u32.to_le_bytes());
        chunks.extend(0.5f32.to_le_bytes());
        chunks.extend(0u32.to_le_bytes());
        chunks.extend(b"data");
        chunks.extend(data.to_le_bytes());
        let mut bytes = b"RIFF".to_vec();
        let chunks_len = u32::try_from(chunks.len()).unwrap_or(u32::MAX);
        bytes.extend((4 + chunks_len + data).to_le_bytes());
        bytes.extend(b"WAVE");
        bytes.extend(chunks);
        bytes.resize(bytes.len() + data as usize, 0);
        base64::engine::general_purpose::STANDARD.encode(bytes)
    }

    /// A FLAC header whose STREAMINFO says `total` samples at `rate`.
    fn flac(rate: u64, total: u64) -> String {
        let mut bytes = b"fLaC".to_vec();
        bytes.extend([0x80, 0, 0, 34]);
        bytes.extend([0u8; 10]);
        // One channel (stored as 0) at 16 bits (stored as 15).
        let packed = (rate << 44) | (15 << 36) | total;
        bytes.extend(packed.to_be_bytes());
        bytes.extend([0u8; 16]);
        base64::engine::general_purpose::STANDARD.encode(bytes)
    }

    fn sample_with(id: &str, data: String, kwargs: Option<Value>) -> EvalSample {
        EvalSample {
            id: id.to_owned(),
            messages: vec![ChatMessage {
                role: ChatRole::User,
                content: MessageContent::Parts(vec![
                    ContentPart::InputAudio {
                        input_audio: InputAudio {
                            data,
                            format: AudioFormat::Wav,
                        },
                    },
                    ContentPart::Text {
                        text: "Transcribe the audio.".to_owned(),
                    },
                ]),
            }],
            chat_template_kwargs: kwargs.and_then(|v| v.as_object().cloned()),
        }
    }

    /// The profile's entry on FLEURS Dutch, except LFM2.5-Audio's, on
    /// LibriSpeech test-clean (its card's one language).
    fn entry(profile: &str) -> anyhow::Result<AsrEvalBenchmark> {
        let (task, samples) = if profile == "lfm25_audio" {
            ("asr_esb_librispeech_test_clean", 2620)
        } else {
            ("asr_fleurs_nl_nl", 364)
        };
        Ok(AsrEvalBenchmark {
            benchmark_id: BenchmarkId::try_new(format!("asr_eval_{task}_{profile}"))?,
            parameter_asr_task: task.to_owned().into(),
            parameter_dataset_version: "2026.10.1".into(),
            parameter_sample_count: PositiveCount::try_new(samples)?,
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

    fn beam_entry() -> anyhow::Result<AsrEvalBenchmark> {
        Ok(AsrEvalBenchmark {
            parameter_num_beams: Some(NumBeams::try_new(3)?),
            parameter_beam_repetition_penalty: Some(BeamRepetitionPenalty::try_new(1.2)?),
            parameter_beam_length_penalty: Some(BeamLengthPenalty::try_new(1.0)?),
            ..entry("minicpm_o")?
        })
    }

    fn cell(
        ctx_size: Option<u32>,
        parallel: Option<i32>,
        raw: &[&str],
    ) -> anyhow::Result<AsrEvalLlamacppCliStockToolsGgufVision> {
        Ok(AsrEvalLlamacppCliStockToolsGgufVision {
            threads: None,
            number_gpu_layers: None,
            mmap: None,
            load_mode: None,
            flash_attention: None,
            ctx_size,
            no_cache: None,
            parallel: parallel.map(PositiveCount::try_new).transpose()?,
            envs: Vec::new(),
            raw: raw.iter().map(|s| (*s).to_owned()).collect(),
        })
    }

    fn argv(
        benchmark: &AsrEvalBenchmark,
        flags: AsrEvalLlamacppCliStockToolsGgufVision,
    ) -> anyhow::Result<Vec<String>> {
        let recipe = AsrRecipe::for_entry(benchmark, &flags)?;
        let slots = recipe.slots;
        let model = ResolvedGgufVisionModel {
            backbone: "model.gguf".into(),
            mmproj: "mmproj.gguf".into(),
        };
        Ok(launch(
            &with_ctx(flags, 4096, slots),
            &recipe,
            &test_llama_server(),
            &model,
            None,
        )
        .args()?)
    }

    fn value_of<'a>(argv: &'a [String], flag: &str) -> Option<&'a str> {
        argv.windows(2)
            .find(|w| w.first().is_some_and(|f| f == flag))
            .and_then(|w| w.get(1))
            .map(String::as_str)
    }

    fn request_for(
        benchmark: &AsrEvalBenchmark,
        sample: &EvalSample,
        stream: bool,
    ) -> anyhow::Result<Value> {
        let recipe = AsrRecipe::for_entry(benchmark, &cell(None, None, &[])?)?;
        let measured = EvalMeasurementSample {
            id: &sample.id,
            messages: EvalMessages::Chat(&sample.messages),
            chat_template_kwargs: sample.chat_template_kwargs.as_ref(),
            position: "eval sample 1/1".to_owned(),
        };
        Ok(recipe.request(benchmark, &measured, 256, 4096, stream))
    }

    /// Every prompt cache is off on every profile, MiniCPM-o's one beam slot
    /// included, unless the cell's `raw` sets it.
    #[rstest]
    #[case::gemma4("gemma4", &[], &["--cache-ram", "0", "--ctx-checkpoints", "0", "--no-cache-prompt"])]
    #[case::voxtral("voxtral", &[], &["--cache-ram", "0", "--ctx-checkpoints", "0", "--no-cache-prompt"])]
    #[case::minicpm_o("minicpm_o", &[], &["--cache-ram", "0", "--ctx-checkpoints", "0", "--no-cache-prompt"])]
    #[case::lfm3("lfm3", &[], &["--cache-ram", "0", "--ctx-checkpoints", "0", "--no-cache-prompt"])]
    #[case::raw_size("lfm25_audio", &["--cache-ram", "512"], &["--cache-ram", "512", "--ctx-checkpoints", "0", "--no-cache-prompt"])]
    fn every_prompt_cache_is_off_unless_raw_sets_it(
        #[case] profile: &str,
        #[case] raw: &[&str],
        #[case] expected: &[&str],
    ) -> anyhow::Result<()> {
        let argv = argv(&entry(profile)?, cell(None, None, raw)?)?;
        let caches: Vec<&str> = argv
            .iter()
            .enumerate()
            .filter(|(_, t)| {
                ["--cache-ram", "--ctx-checkpoints", "--no-cache-prompt"].contains(&t.as_str())
            })
            .flat_map(|(i, t)| {
                let value = argv.get(i + 1).filter(|v| !v.starts_with('-'));
                std::iter::once(t.as_str()).chain(value.map(String::as_str))
            })
            .collect();
        assert_eq!(caches, expected, "{argv:?}");
        Ok(())
    }

    /// Every profile's server starts with thinking off and the profile's own
    /// template flags, and no other profile's.
    #[rstest]
    #[case::gemma4("gemma4", None, false)]
    #[case::qwen25_omni("qwen25_omni", None, false)]
    #[case::lfm25_audio("lfm25_audio", None, false)]
    #[case::voxtral("voxtral", Some(VOXTRAL_CHAT_TEMPLATE), false)]
    #[case::lfm3("lfm3", Some(LFM3_ASR_TEMPLATE), false)]
    #[case::minicpm_o("minicpm_o", None, true)]
    fn each_profile_renders_its_server_flags(
        #[case] profile: &str,
        #[case] template: Option<&str>,
        #[case] minicpm: bool,
    ) -> anyhow::Result<()> {
        let got = argv(&entry(profile)?, cell(None, None, &[])?)?;
        assert_eq!(value_of(&got, "--reasoning"), Some("off"), "{got:?}");
        assert_eq!(value_of(&got, "--chat-template"), template, "{got:?}");
        assert_eq!(
            got.iter().any(|t| t == "--jinja"),
            template.is_some(),
            "{got:?}"
        );
        let kwargs = value_of(&got, "--chat-template-kwargs")
            .map(serde_json::from_str::<Value>)
            .transpose()?;
        assert_eq!(
            kwargs,
            minicpm.then(|| json!({"use_tts_template": true, "enable_thinking": false})),
            "{got:?}"
        );
        assert_eq!(
            value_of(&got, "--override-kv"),
            minicpm.then_some("tokenizer.ggml.eot_token_id=int:151704"),
            "{got:?}"
        );
        assert!(!got.iter().any(|t| t == "--beam-search"), "{got:?}");
        assert!(!got.iter().any(|t| t == "--parallel"), "{got:?}");
        Ok(())
    }

    /// Every asr_eval recipe version's fingerprint, oldest first; the last is
    /// this build's ([`recipe_fingerprint`]).
    const RECIPE_FINGERPRINTS: &[&str] = &[
        "948fd0d561b5ac7312331bfd0a33a671a753f05701790a9c84151d04dc85ee9a",
        // 2: every unpinned knob is its card's (plan-types 852a4bf):
        // Gemma 4 samples at 1.0 / 0.95 / 64.
        "bff52240ff76243d013234b0f23ebb5b91056e14497b65a0fbe028710d6ecb92",
        // 3: every request spells its sampler chain, and an unpinned beam
        // penalty is the card's (plan-types 70993d0: repetition 1.2); from
        // here on the fingerprint also renders a pinned entry per profile
        // and a beam entry with unpinned penalties.
        "94eb417247e5d9d0df7a58575dde3a4e7106a8136c646dd13fcb9b92034948b7",
    ];

    /// An entry on `profile` that pins every sampling knob the profile's
    /// card takes but `top_k` (so the card's cut under a pinned temperature
    /// is rendered too): temperature, top_p, seed, and the repeat penalty
    /// where the card applies one.
    fn pinned_entry(profile: KnownAsrPromptProfile) -> anyhow::Result<AsrEvalBenchmark> {
        use pipette_plan_types::RepeatPenalty;
        Ok(AsrEvalBenchmark {
            parameter_temperature: Some(SamplingTemperature::try_new(0.7)?),
            parameter_top_p: Some(TopP::try_new(0.9)?),
            parameter_seed: Some(SamplingSeed::try_new(7)?),
            parameter_repeat_penalty: (profile != KnownAsrPromptProfile::Voxtral)
                .then(|| RepeatPenalty::try_new(1.1))
                .transpose()?,
            ..entry(profile.as_ref())?
        })
    }

    /// What each profile's server and request render, hashed: every profile
    /// unpinned and with its sampling knobs pinned ([`pinned_entry`]), and
    /// MiniCPM-o's beams, pinned and unpinned penalties.
    fn recipe_fingerprint() -> anyhow::Result<String> {
        use sha2::Digest;
        let sample = sample_with("c1", wav(1.0), Some(json!({"language": "nl"})));
        let mut rendered = Vec::new();
        let unpinned_beams = beam_entry().map(|beams| AsrEvalBenchmark {
            parameter_beam_repetition_penalty: None,
            parameter_beam_length_penalty: None,
            ..beams
        });
        let entries = KnownAsrPromptProfile::ALL
            .iter()
            .map(|profile| entry(profile.as_ref()))
            .chain(KnownAsrPromptProfile::ALL.iter().map(|p| pinned_entry(*p)))
            .chain([beam_entry(), unpinned_beams]);
        for benchmark in entries {
            let benchmark = benchmark?;
            let recipe = AsrRecipe::for_entry(&benchmark, &cell(None, None, &[])?)?;
            rendered.push(json!({
                "profile": recipe.profile.to_string(),
                "server": recipe.server_flags(),
                "streamed": request_for(&benchmark, &sample, true)?,
                "one_response": request_for(&benchmark, &sample, false)?,
            }));
        }
        let mut canonical = String::new();
        write_sorted(&Value::Array(rendered), &mut canonical);
        Ok(hex::encode(sha2::Sha256::digest(canonical.as_bytes())))
    }

    /// `value` as JSON with every object's keys sorted, so the fingerprint
    /// does not move with the key order a build's JSON maps keep.
    fn write_sorted(value: &Value, out: &mut String) {
        match value {
            Value::Array(items) => {
                out.push('[');
                for (at, item) in items.iter().enumerate() {
                    if at > 0 {
                        out.push(',');
                    }
                    write_sorted(item, out);
                }
                out.push(']');
            }
            Value::Object(map) => {
                let mut keys: Vec<&String> = map.keys().collect();
                keys.sort();
                out.push('{');
                for (at, key) in keys.into_iter().enumerate() {
                    if at > 0 {
                        out.push(',');
                    }
                    out.push_str(&Value::String(key.clone()).to_string());
                    out.push(':');
                    if let Some(item) = map.get(key) {
                        write_sorted(item, out);
                    }
                }
                out.push('}');
            }
            other @ (Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_)) => {
                out.push_str(&other.to_string());
            }
        }
    }

    /// The server flag's fixed spelling is the request's kwargs.
    #[test]
    fn minicpm_os_server_kwargs_are_its_request_kwargs() -> anyhow::Result<()> {
        assert_eq!(
            serde_json::from_str::<Map<String, Value>>(MINICPM_O_SERVER_KWARGS)?,
            minicpm_o_template_kwargs()
        );
        Ok(())
    }

    /// The recipe this client runs is the one its version names in the resume
    /// identity: a change to any profile's server flags or request is a new
    /// version, or a checkpoint written under the old recipe would resume
    /// under the new one and submit transcripts of both as one cell.
    #[test]
    fn the_recipe_fingerprint_is_pinned_to_its_version() -> anyhow::Result<()> {
        let got = recipe_fingerprint()?;
        assert_eq!(
            RECIPE_FINGERPRINTS.last().copied(),
            Some(got.as_str()),
            "the asr_eval recipe changed: append {got} to RECIPE_FINGERPRINTS and bump \
             pipette_ops::eval_completions::ASR_EVAL_RECIPE_VERSION"
        );
        assert_eq!(
            u32::try_from(RECIPE_FINGERPRINTS.len())?,
            pipette_ops::eval_completions::ASR_EVAL_RECIPE_VERSION,
            "one fingerprint per recipe version"
        );
        Ok(())
    }

    /// The template shipped is the reviewed one, byte for byte: a stray edit
    /// would change every Voxtral prompt with no review of what changed.
    #[test]
    fn the_voxtral_template_is_the_reference_copy() {
        use sha2::Digest;
        assert_eq!(
            hex::encode(sha2::Sha256::digest(VOXTRAL_CHAT_TEMPLATE.as_bytes())),
            "3d7b2f9d649544f6fe3223620a84d91eb64bd9ff4a7467b0edc843c20b90f5de"
        );
    }

    /// A beam entry starts the ASR build's beam search on one spelled-out slot.
    #[test]
    fn a_beam_entry_starts_one_slot_with_the_beam_search() -> anyhow::Result<()> {
        let got = argv(&beam_entry()?, cell(None, None, &[])?)?;
        assert_eq!(value_of(&got, "--beam-search"), Some("3"), "{got:?}");
        assert_eq!(value_of(&got, "--parallel"), Some("1"), "{got:?}");
        assert_eq!(value_of(&got, "--ctx-size"), Some("4096"), "{got:?}");
        Ok(())
    }

    /// The beams travel on every request with MiniCPM-o's end tokens, and no
    /// sampler beside them.
    #[test]
    fn a_beam_request_carries_the_beam_keys() -> anyhow::Result<()> {
        let body = request_for(&beam_entry()?, &sample_with("c1", wav(1.0), None), false)?;
        assert_eq!(body["n_beams"], json!(3));
        assert_eq!(body["beam_repetition_penalty"], json!(1.2));
        assert_eq!(body["beam_length_penalty"], json!(1.0));
        assert_eq!(body["beam_eos_token_ids"], json!([151_645, 151_643]));
        assert_eq!(body["beam_bf16_logits"], json!(true));
        assert_eq!(body["cache_prompt"], json!(false));
        assert_eq!(
            body["chat_template_kwargs"],
            json!({"use_tts_template": true, "enable_thinking": false})
        );
        for absent in ["temperature", "top_k", "seed", "stream", "stop"] {
            assert!(body.get(absent).is_none(), "{absent} in {body}");
        }
        Ok(())
    }

    /// Each profile's request when the entry pins nothing: its stop words and
    /// the card's sampling (Gemma 4 samples at 1.0 / top_p 0.95 / top_k 64,
    /// the others transcribe greedily), and the sample's template arguments
    /// passed through, thinking closed.
    #[rstest]
    #[case::gemma4(
        "gemma4",
        Some(json!(["<eos>", "<turn|>", "<|tool_response>"])),
        (1.0, 0.95, 64),
        json!({"enable_thinking": false})
    )]
    #[case::qwen25_omni("qwen25_omni", Some(json!(["<|im_end|>"])), (0.0, 1.0, 1), Value::Null)]
    #[case::lfm25_audio("lfm25_audio", None, (0.0, 1.0, 1), Value::Null)]
    #[case::lfm3("lfm3", None, (0.0, 1.0, 1), Value::Null)]
    #[case::voxtral("voxtral", None, (0.0, 1.0, 0), Value::Null)]
    #[case::minicpm_o(
        "minicpm_o",
        None,
        (0.0, 1.0, 1),
        json!({"use_tts_template": true, "enable_thinking": false})
    )]
    fn each_profile_shapes_its_request(
        #[case] profile: &str,
        #[case] stop: Option<Value>,
        #[case] (temperature, top_p, top_k): (f64, f64, i32),
        #[case] kwargs: Value,
    ) -> anyhow::Result<()> {
        let body = request_for(&entry(profile)?, &sample_with("c1", wav(1.0), None), true)?;
        assert_eq!(body.get("stop").cloned(), stop);
        assert_eq!(body["temperature"], json!(temperature), "{body}");
        assert_eq!(body["top_p"], json!(top_p), "{body}");
        assert_eq!(body["top_k"], json!(top_k), "{body}");
        assert_eq!(body["min_p"], json!(0.0));
        assert_eq!(body["max_tokens"], json!(256));
        assert_eq!(body["stream"], json!(true));
        assert_eq!(body["stream_options"], json!({"include_usage": true}));
        assert_eq!(body["reasoning_format"], json!("none"));
        // The chain spelled out, so a GGUF's `general.sampling.sequence` or
        // a server's `--samplers` cannot drop or reorder a stage.
        let chain = if profile == "voxtral" {
            json!(["temperature"])
        } else {
            json!([
                "penalties",
                "dry",
                "top_n_sigma",
                "top_k",
                "typ_p",
                "top_p",
                "min_p",
                "xtc",
                "temperature"
            ])
        };
        assert_eq!(body["samplers"], chain, "{body}");
        // Neutral, so a GGUF's `general.sampling.*` cannot change the decode.
        for (key, neutral) in [
            ("typical_p", json!(1.0)),
            ("top_n_sigma", json!(-1.0)),
            ("presence_penalty", json!(0.0)),
            ("frequency_penalty", json!(0.0)),
            ("dry_multiplier", json!(0.0)),
            ("xtc_probability", json!(0.0)),
            ("mirostat", json!(0)),
        ] {
            assert_eq!(body[key], neutral, "{key} in {body}");
        }
        assert_eq!(
            body.get("chat_template_kwargs")
                .cloned()
                .unwrap_or(Value::Null),
            kwargs
        );
        Ok(())
    }

    /// Every entry's request samples exactly as plan-types' effective
    /// decoding says, the one table mgmt and the warehouse read: each knob
    /// the entry pins, each it leaves unset at its card's value.
    #[rstest]
    #[case::gemma4_unpinned("gemma4", None, None, None, None, None)]
    #[case::gemma4_card("gemma4", Some(1.0), Some(0.95), Some(64), None, Some(7))]
    #[case::qwen_greedy("qwen25_omni", None, None, None, None, None)]
    #[case::qwen_pinned_temperature("qwen25_omni", Some(0.7), None, None, None, Some(1))]
    #[case::voxtral("voxtral", None, None, None, None, None)]
    #[case::voxtral_top_p("voxtral", Some(0.2), Some(0.9), None, None, None)]
    #[case::minicpm_o("minicpm_o", None, None, None, None, None)]
    #[case::minicpm_o_pinned_penalty("minicpm_o", None, None, None, Some(1.1), None)]
    #[case::lfm25_audio("lfm25_audio", None, None, Some(5), None, None)]
    #[case::lfm3("lfm3", None, None, None, None, None)]
    fn the_request_samples_as_the_effective_decoding(
        #[case] profile: &str,
        #[case] temperature: Option<f64>,
        #[case] top_p: Option<f64>,
        #[case] top_k: Option<i32>,
        #[case] repeat_penalty: Option<f64>,
        #[case] seed: Option<i64>,
    ) -> anyhow::Result<()> {
        use pipette_plan_types::{AsrDecoding, RepeatPenalty, TopK};
        let benchmark = AsrEvalBenchmark {
            parameter_temperature: temperature.map(SamplingTemperature::try_new).transpose()?,
            parameter_top_p: top_p.map(TopP::try_new).transpose()?,
            parameter_top_k: top_k.map(TopK::try_new).transpose()?,
            parameter_repeat_penalty: repeat_penalty.map(RepeatPenalty::try_new).transpose()?,
            parameter_seed: seed.map(SamplingSeed::try_new).transpose()?,
            ..entry(profile)?
        };
        let AsrDecoding::Sampling(want) = benchmark.effective_decoding()? else {
            anyhow::bail!("a sampling entry");
        };
        let body = request_for(&benchmark, &sample_with("c1", wav(1.0), None), true)?;
        assert_eq!(
            body["temperature"],
            json!(want.temperature.as_f64()),
            "{body}"
        );
        assert_eq!(body["top_p"], json!(want.top_p.as_f64()), "{body}");
        assert_eq!(body["top_k"], json!(i32::from(want.top_k)), "{body}");
        assert_eq!(body["min_p"], json!(want.min_p.as_f64()), "{body}");
        assert_eq!(
            body["repeat_penalty"],
            json!(want.repeat_penalty.as_f64()),
            "{body}"
        );
        assert_eq!(
            body.get("repeat_last_n").cloned(),
            (want.repeat_window == pipette_plan_types::AsrRepeatWindow::WholeSequence)
                .then(|| json!(4096)),
            "{body}"
        );
        let chain = match (want.sampler_chain, benchmark.parameter_top_p) {
            (pipette_plan_types::AsrSamplerChain::TemperatureThenTopP, Some(_)) => {
                json!(["temperature", "top_p"])
            }
            (pipette_plan_types::AsrSamplerChain::TemperatureThenTopP, None) => {
                json!(["temperature"])
            }
            (pipette_plan_types::AsrSamplerChain::RuntimeDefault, _) => {
                json!(LLAMA_CPP_DEFAULT_SAMPLERS)
            }
        };
        assert_eq!(body["samplers"], chain, "{body}");
        assert_eq!(
            body.get("seed").cloned(),
            want.seed
                .and_then(|_| benchmark.sample_seed("c1"))
                .map(|seed| json!(seed)),
            "{body}"
        );
        Ok(())
    }

    /// A beam entry's request carries plan-types' beam search, its unset
    /// penalties at MiniCPM-o's card values (repetition 1.2, length 1.0).
    #[test]
    fn a_beam_request_carries_the_effective_beams() -> anyhow::Result<()> {
        use pipette_plan_types::AsrDecoding;
        let benchmark = AsrEvalBenchmark {
            parameter_beam_repetition_penalty: None,
            parameter_beam_length_penalty: None,
            ..beam_entry()?
        };
        let AsrDecoding::Beams(want) = benchmark.effective_decoding()? else {
            anyhow::bail!("a beam entry");
        };
        let body = request_for(&benchmark, &sample_with("c1", wav(1.0), None), false)?;
        assert_eq!(body["n_beams"], json!(want.num_beams.into_inner()));
        assert_eq!(
            body["beam_repetition_penalty"],
            json!(want.repetition_penalty.into_inner())
        );
        assert_eq!(body["beam_repetition_penalty"], json!(1.2));
        assert_eq!(body["beam_length_penalty"], json!(1.0));
        Ok(())
    }

    /// A temperature pinned above 0 on a card greedy through top_k 1 samples:
    /// the unpinned cut is off; a pinned cut still wins, and a pinned 0.0
    /// stays greedy.
    #[rstest]
    #[case::qwen("qwen25_omni", 0.7, None, 0)]
    #[case::minicpm_o("minicpm_o", 0.7, None, 0)]
    #[case::lfm25_audio("lfm25_audio", 0.7, None, 0)]
    #[case::lfm3("lfm3", 0.7, None, 0)]
    #[case::pinned_cut("qwen25_omni", 0.7, Some(5), 5)]
    #[case::pinned_greedy("qwen25_omni", 0.0, None, 1)]
    fn a_pinned_temperature_samples_on_a_greedy_card(
        #[case] profile: &str,
        #[case] temperature: f64,
        #[case] top_k: Option<i32>,
        #[case] sent: i32,
    ) -> anyhow::Result<()> {
        let benchmark = AsrEvalBenchmark {
            parameter_temperature: Some(SamplingTemperature::try_new(temperature)?),
            parameter_top_k: top_k.map(pipette_plan_types::TopK::try_new).transpose()?,
            ..entry(profile)?
        };
        let body = request_for(&benchmark, &sample_with("c1", wav(1.0), None), true)?;
        assert_eq!(body["temperature"], json!(temperature), "{body}");
        assert_eq!(body["top_k"], json!(sent), "{body}");
        Ok(())
    }

    /// MiniCPM-o greedy runs chat()'s repetition penalty over the whole slot.
    #[test]
    fn minicpm_greedy_penalises_repeats_over_the_slot() -> anyhow::Result<()> {
        let body = request_for(
            &entry("minicpm_o")?,
            &sample_with("c1", wav(1.0), None),
            true,
        )?;
        assert_eq!(body["repeat_penalty"], json!(1.02));
        assert_eq!(body["repeat_last_n"], json!(4096));
        Ok(())
    }

    /// Voxtral's sample carries its language; it reaches the request as sent,
    /// beside the card's sampler chain.
    #[test]
    fn voxtral_passes_the_samples_template_arguments() -> anyhow::Result<()> {
        let sample = sample_with(
            "c1",
            wav(1.0),
            Some(json!({"transcription": true, "language": "nl"})),
        );
        let body = request_for(&entry("voxtral")?, &sample, true)?;
        assert_eq!(
            body["chat_template_kwargs"],
            json!({"transcription": true, "language": "nl"})
        );
        assert_eq!(body["samplers"], json!(["temperature"]));
        assert_eq!(body["top_k"], json!(0));
        let pinned = AsrEvalBenchmark {
            parameter_top_p: Some(TopP::try_new(0.9)?),
            ..entry("voxtral")?
        };
        let body = request_for(&pinned, &sample, true)?;
        assert_eq!(body["samplers"], json!(["temperature", "top_p"]));
        Ok(())
    }

    /// The entry's pins win over the card's defaults, and a seed is derived
    /// per sample.
    #[test]
    fn the_entrys_sampling_wins() -> anyhow::Result<()> {
        let pinned = AsrEvalBenchmark {
            parameter_temperature: Some(SamplingTemperature::try_new(1.0)?),
            parameter_top_p: Some(TopP::try_new(0.95)?),
            parameter_top_k: Some(pipette_plan_types::TopK::try_new(64)?),
            parameter_seed: Some(SamplingSeed::try_new(0)?),
            ..entry("gemma4")?
        };
        let body = request_for(&pinned, &sample_with("c1", wav(1.0), None), true)?;
        assert_eq!(body["temperature"], json!(1.0));
        assert_eq!(body["top_p"], json!(0.95));
        assert_eq!(body["top_k"], json!(64));
        assert_eq!(body["seed"], json!(pinned.sample_seed("c1")));
        Ok(())
    }

    /// What the runner refuses before the corpus is downloaded.
    #[rstest]
    #[case::unknown_profile(entry("whisper"), cell(None, None, &[]), "no llama.cpp recipe")]
    #[case::beams_off_minicpm(
        Ok(AsrEvalBenchmark { parameter_num_beams: Some(NumBeams::try_new(3)?), ..entry("gemma4")? }),
        cell(None, None, &[]),
        "`minicpm_o`'s recipe"
    )]
    #[case::beams_on_slots(beam_entry(), cell(None, Some(2), &[]), "one slot")]
    #[case::raw_slots(entry("gemma4"), cell(None, None, &["-np", "2"]), "typed")]
    #[case::raw_template(entry("lfm3"), cell(None, None, &["--chat-template", "x"]), "owns the chat template")]
    #[case::raw_beams(entry("minicpm_o"), cell(None, None, &["--beam-search", "3"]), "raw --beam-search")]
    #[case::raw_template_file(entry("gemma4"), cell(None, None, &["--chat_template_file", "t.jinja"]), "raw --chat-template-file")]
    #[case::raw_override_kv(entry("minicpm_o"), cell(None, None, &["--override-kv", "tokenizer.ggml.eot_token_id=int:2"]), "raw --override-kv")]
    #[case::raw_predict(entry("qwen25_omni"), cell(None, None, &["-n", "16"]), "raw -n")]
    #[case::raw_per_slot(entry("voxtral"), cell(None, None, &["--kv-unified-per-slot", "1024"]), "raw --kv-unified-per-slot")]
    #[case::raw_samplers(entry("minicpm_o"), cell(None, None, &["--samplers", "top_k;temperature"]), "raw --samplers")]
    #[case::raw_sampling_seq(entry("gemma4"), cell(None, None, &["--sampling_seq", "kt"]), "raw --sampling-seq")]
    #[case::voxtral_top_k(
        Ok(AsrEvalBenchmark { parameter_top_k: Some(pipette_plan_types::TopK::try_new(5)?), ..entry("voxtral")? }),
        cell(None, None, &[]),
        "parameter_top_k"
    )]
    #[case::incoherent_decoding(
        Ok(AsrEvalBenchmark { parameter_beam_length_penalty: Some(BeamLengthPenalty::try_new(1.0)?), ..entry("minicpm_o")? }),
        cell(None, None, &[]),
        "parameter_num_beams"
    )]
    fn an_entry_the_runner_cannot_run_as_its_card_is_refused(
        #[case] benchmark: anyhow::Result<AsrEvalBenchmark>,
        #[case] flags: anyhow::Result<AsrEvalLlamacppCliStockToolsGgufVision>,
        #[case] needle: &str,
    ) -> anyhow::Result<()> {
        let Err(err) = AsrRecipe::for_entry(&benchmark?, &flags?) else {
            anyhow::bail!("expected a refusal naming {needle}");
        };
        let msg = format!("{err:#}");
        assert!(msg.contains(needle), "{msg}");
        Ok(())
    }

    /// Every profile runs every task: LFM2.5-Audio, whose card lists English
    /// alone, on FLEURS (English or not) as on an English set, and another
    /// profile on FLEURS. A cell outside its card's languages is marked by its
    /// entry's recipe, never refused.
    #[rstest]
    #[case::lfm25_audio_on_english(entry("lfm25_audio"))]
    #[case::lfm25_audio_on_fleurs_english(Ok(AsrEvalBenchmark {
        parameter_asr_task: "asr_fleurs_en_us".to_owned().into(),
        ..entry("lfm25_audio")?
    }))]
    #[case::lfm25_audio_on_fleurs_german(Ok(AsrEvalBenchmark {
        parameter_asr_task: "asr_fleurs_de_de".to_owned().into(),
        ..entry("lfm25_audio")?
    }))]
    #[case::lfm3_on_fleurs(entry("lfm3"))]
    fn every_profile_runs_every_task(
        #[case] benchmark: anyhow::Result<AsrEvalBenchmark>,
    ) -> anyhow::Result<()> {
        AsrRecipe::for_entry(&benchmark?, &cell(None, None, &[])?)?;
        Ok(())
    }

    /// A raw thinking flag is refused on the asr_eval server as on the other
    /// eval cells: thinking is always off.
    #[rstest]
    #[case::reasoning(&["--reasoning", "on"])]
    #[case::short(&["-rea", "on"])]
    #[case::budget(&["--reasoning-budget", "-1"])]
    #[case::template_kwargs(&["--chat-template-kwargs", "{}"])]
    fn a_raw_thinking_flag_is_refused(#[case] raw: &[&str]) -> anyhow::Result<()> {
        let Err(err) = argv(&entry("gemma4")?, cell(None, None, raw)?) else {
            anyhow::bail!("{raw:?} must be refused");
        };
        let err = err
            .downcast::<server::ServerArgsError>()
            .context("refused by the server's thinking check, not another")?;
        assert!(
            matches!(
                err,
                server::ServerArgsError::RawReasoningFlag {
                    logging_marker: LoggingMarker::AsrEval,
                    ..
                }
            ),
            "{err}"
        );
        assert!(
            err.to_string().contains("sets how llama-server thinks"),
            "{err}"
        );
        Ok(())
    }

    /// A llama.cpp option variable, set by the cell, is refused before the
    /// corpus is downloaded, as the thinking envs are.
    #[rstest]
    #[case::template_file("LLAMA_ARG_CHAT_TEMPLATE_FILE=/tmp/t.jinja")]
    #[case::template("LLAMA_ARG_CHAT_TEMPLATE=chatml")]
    #[case::parallel("LLAMA_ARG_N_PARALLEL=4")]
    #[case::beams("LLAMA_ARG_BEAM_SEARCH=3")]
    #[case::samplers("LLAMA_ARG_SAMPLERS=top_k;temperature")]
    #[case::reasoning("LLAMA_ARG_REASONING=on")]
    fn an_owned_env_is_refused(#[case] env: &str) -> anyhow::Result<()> {
        let flags = AsrEvalLlamacppCliStockToolsGgufVision {
            envs: vec![pipette_plan_types::EnvForward::try_new(env.to_owned())?],
            ..cell(None, None, &[])?
        };
        let Err(err) = argv(&entry("qwen25_omni")?, flags) else {
            anyhow::bail!("{env} must be refused");
        };
        let name = env.split_once('=').map_or(env, |(name, _)| name);
        assert!(format!("{err:#}").contains(name), "{err:#}");
        Ok(())
    }

    /// A model under `dir`: the projector `projector` and the backbone
    /// `backbone`, as GGUF bytes.
    fn model_files(
        dir: &std::path::Path,
        projector: &[u8],
        backbone: &[u8],
    ) -> anyhow::Result<ResolvedGgufVisionModel> {
        let model = ResolvedGgufVisionModel {
            backbone: dir.join("model.gguf"),
            mmproj: dir.join("mmproj.gguf"),
        };
        std::fs::write(&model.backbone, backbone)?;
        std::fs::write(&model.mmproj, projector)?;
        Ok(model)
    }

    /// The projector must hold the profile's audio encoder: another family's,
    /// or none, is refused before anything is fetched.
    #[rstest]
    #[case::qwen("qwen25_omni", Some("qwen2.5oa"), None)]
    #[case::qwen_legacy_name("qwen25_omni", Some("qwen2.5o"), None)]
    #[case::gemma("gemma4", Some("gemma4a"), None)]
    #[case::voxtral("voxtral", Some("voxtral"), None)]
    #[case::lfm25("lfm25_audio", Some("lfm2a"), None)]
    #[case::lfm3("lfm3", Some("lfm3a"), None)]
    #[case::voxtral_on_qwen(
        "voxtral",
        Some("qwen2.5oa"),
        Some("holds the `qwen2.5oa` audio encoder")
    )]
    #[case::lfm3_on_lfm25("lfm3", Some("lfm2a"), Some("the recipe for lfm3a"))]
    #[case::image_only("gemma4", None, Some("holds no audio encoder"))]
    #[case::openbmb_minicpm_o("minicpm_o", Some("openbmb"), None)]
    #[case::openbmb_on_qwen(
        "qwen25_omni",
        Some("openbmb"),
        Some("holds the `minicpmoa` audio encoder")
    )]
    fn the_projector_is_the_profiles(
        #[case] profile: &str,
        #[case] encoder: Option<&str>,
        #[case] needle: Option<&str>,
    ) -> anyhow::Result<()> {
        use crate::gguf_meta::fixture;
        let dir = tempfile::tempdir()?;
        let projector = match encoder {
            Some("openbmb") => fixture::openbmb_audio(
                true,
                &["encoder.conv1.weight", "audio_projector.linear2.weight"],
            ),
            Some(encoder) => fixture::projector(encoder),
            None => fixture::gguf(&[(
                "clip.projector_type",
                fixture::TYPE_STRING,
                fixture::string("gemma4v"),
            )]),
        };
        // MiniCPM-o's vocabulary is checked too: its own, here.
        let backbone = if profile == "minicpm_o" {
            fixture::model(151_705, &MINICPM_O_TOKENS)
        } else {
            Vec::new()
        };
        let model = model_files(dir.path(), &projector, &backbone)?;
        let recipe = AsrRecipe::for_entry(&entry(profile)?, &cell(None, None, &[])?)?;
        match (check_model(&recipe, &model), needle) {
            (Ok(()), None) => Ok(()),
            (Err(err), Some(needle)) => {
                assert!(format!("{err:#}").contains(needle), "{err:#}");
                Ok(())
            }
            (got, needle) => anyhow::bail!("{got:?}, expected {needle:?}"),
        }
    }

    /// MiniCPM-o's hard-coded ids must be the pieces it means in the GGUF it
    /// runs: a vocabulary that numbers them otherwise, or is too short, is
    /// refused.
    #[rstest]
    #[case::minicpm_o_4_5(&[(151_704, "<|tts_eos|>"), (151_645, "<|im_end|>"), (151_643, "<|endoftext|>")], 151_705, None)]
    #[case::another_eot(&[(151_704, "<|tts_bos|>"), (151_645, "<|im_end|>"), (151_643, "<|endoftext|>")], 151_705, Some("token 151704 of"))]
    #[case::beams_end_elsewhere(&[(151_704, "<|tts_eos|>"), (151_645, "<|im_start|>"), (151_643, "<|endoftext|>")], 151_705, Some("not `<|im_end|>`"))]
    #[case::short_vocabulary(&[(151_645, "<|im_end|>"), (151_643, "<|endoftext|>")], 151_700, Some("past its vocabulary"))]
    fn minicpm_os_token_ids_are_checked_against_its_vocabulary(
        #[case] pieces: &[(u32, &str)],
        #[case] len: u32,
        #[case] needle: Option<&str>,
    ) -> anyhow::Result<()> {
        use crate::gguf_meta::fixture;
        let dir = tempfile::tempdir()?;
        let model = model_files(
            dir.path(),
            &fixture::projector("minicpmoa"),
            &fixture::model(len, pieces),
        )?;
        let recipe = AsrRecipe::for_entry(&beam_entry()?, &cell(None, None, &[])?)?;
        match (check_model(&recipe, &model), needle) {
            (Ok(()), None) => Ok(()),
            (Err(err), Some(needle)) => {
                assert!(format!("{err:#}").contains(needle), "{err:#}");
                Ok(())
            }
            (got, needle) => anyhow::bail!("{got:?}, expected {needle:?}"),
        }
    }

    /// Durations read from both containers the evals server sends.
    #[rstest]
    #[case::wav_short(wav(1.5), AudioFormat::Wav, Some(1.5))]
    #[case::wav_long(wav(42.0), AudioFormat::Wav, Some(42.0))]
    #[case::float_wav(float_wav(3.5, false), AudioFormat::Wav, Some(3.5))]
    #[case::float_wav_list_first(float_wav(12.25, true), AudioFormat::Wav, Some(12.25))]
    #[case::float_wav_long(float_wav(95.0, true), AudioFormat::Wav, Some(95.0))]
    #[case::flac(flac(16_000, 160_000), AudioFormat::Flac, Some(10.0))]
    #[case::flac_unknown_total(flac(16_000, 0), AudioFormat::Flac, None)]
    #[case::mp3(wav(1.0), AudioFormat::Mp3, None)]
    #[case::garbage("bm90IGF1ZGlv".to_owned(), AudioFormat::Wav, None)]
    fn a_clips_duration_is_read_from_its_header(
        #[case] data: String,
        #[case] format: AudioFormat,
        #[case] expected: Option<f64>,
    ) {
        let got = clip_seconds(&data, format);
        match (got, expected) {
            (Some(got), Some(expected)) => assert!((got - expected).abs() < 1e-6, "{got}"),
            (got, expected) => assert_eq!(got, expected),
        }
    }

    /// The bound is the larger of the rate and the started windows.
    #[rstest]
    #[case::short_clip_pays_a_window(1.0, 750)]
    #[case::thirty_seconds_starts_a_second_window(30.0, 1500)]
    #[case::sixty_seconds(60.0, 2250)]
    #[case::rate_wins_never_at_25_per_second(29.0, 750)]
    fn audio_tokens_bound_both_encoders(#[case] seconds: f64, #[case] expected: u32) {
        assert_eq!(audio_tokens(seconds), expected);
    }

    /// What the corpus's `samples` need, as the identity pass gathers it.
    fn sized(samples: &[EvalSample], generation: u32) -> Sizing {
        let mut sizing = Sizing::default();
        for sample in samples {
            sizing.add(&sample.messages, generation);
        }
        sizing
    }

    /// The context is sized for the corpus's longest clip, floored at a
    /// 60-second one; a clip whose length cannot be read counts toward neither.
    #[test]
    fn the_context_follows_the_longest_clip() -> anyhow::Result<()> {
        let short = vec![sample_with("a", wav(2.0), None)];
        let text = 21; // "Transcribe the audio."
        let floor = audio_tokens(60.0) + text + TEMPLATE_MARGIN_TOKENS + 256;
        assert_eq!(slot_budget("b", &sized(&short, 256), 256, None, 1)?, floor);
        let long = vec![
            sample_with("a", wav(2.0), None),
            sample_with("b", wav(95.0), None),
            sample_with("c", "AAAA".to_owned(), None),
        ];
        let sizing = sized(&long, 256);
        assert_eq!(sizing.unreadable, 1);
        assert_eq!(
            slot_budget("b", &sizing, 256, None, 1)?,
            audio_tokens(95.0) + text + TEMPLATE_MARGIN_TOKENS + 256
        );
        Ok(())
    }

    /// A beam search holds `max_tokens` once per beam, as the ASR build divides
    /// the room after the prompt among the beams: the bound, the budget and
    /// the pin a too-long clip is told to set all count every beam.
    #[test]
    fn a_beam_search_budgets_every_beam() -> anyhow::Result<()> {
        let recipe = AsrRecipe::for_entry(&beam_entry()?, &cell(None, None, &[])?)?;
        assert_eq!(recipe.generation_tokens(128), 3 * 128);
        let greedy = AsrRecipe::for_entry(&entry("minicpm_o")?, &cell(None, None, &[])?)?;
        assert_eq!(greedy.generation_tokens(128), 128);
        let text = 21;
        let clips = vec![sample_with("a", wav(70.0), None)];
        let needed = audio_tokens(70.0) + text + TEMPLATE_MARGIN_TOKENS + 3 * 128;
        assert_eq!(
            slot_budget("b", &sized(&clips, 3 * 128), 3 * 128, None, 1)?,
            needed
        );
        // With room for one beam's tokens only, the ASR build's beam would be cut
        // to a third of them: refused, with the pin that holds all three.
        let benchmark = beam_entry()?;
        let sender = AsrSender {
            recipe: &recipe,
            benchmark: &benchmark,
            max_tokens: 128,
            generation: recipe.generation_tokens(128),
            budget: needed - 1,
            slots: 1,
            stream: None,
            tee_stderr: false,
        };
        let reason = sender
            .does_not_fit(&measured(&clips[0]))
            .context("a clip short of its beams' room must not be sent")?;
        assert!(reason.contains(&format!("at least {needed}")), "{reason}");
        let fits = AsrSender {
            budget: needed,
            ..sender
        };
        assert_eq!(fits.does_not_fit(&measured(&clips[0])), None);
        Ok(())
    }

    /// A pinned context too small for the corpus's longest clip is refused
    /// before any server starts, with the pin that would hold it; one that
    /// holds it is divided across the slots as written.
    #[test]
    fn a_pinned_context_short_of_the_longest_clip_is_refused() -> anyhow::Result<()> {
        let clips = vec![
            sample_with("a", wav(2.0), None),
            sample_with("b", wav(40.0), None),
        ];
        let needed = audio_tokens(40.0) + 21 + TEMPLATE_MARGIN_TOKENS + 256;
        let sizing = sized(&clips, 256);
        let Err(err) = slot_budget("b", &sizing, 256, Some(2 * needed - 2), 2) else {
            anyhow::bail!("a pin short of the longest clip must be refused");
        };
        assert!(
            format!("{err:#}").contains(&format!("at least {}", 2 * needed)),
            "{err:#}"
        );
        assert_eq!(slot_budget("b", &sizing, 256, Some(2 * needed), 2)?, needed);
        // A pin is the operator's: no floor is added under it.
        assert_eq!(
            slot_budget("b", &sized(&clips[..1], 256), 256, Some(4000), 1)?,
            4000,
            "a pin above the clip but below the floor stands"
        );
        Ok(())
    }

    /// The sample as an attempt receives it.
    fn measured(sample: &EvalSample) -> EvalMeasurementSample<'_> {
        EvalMeasurementSample {
            id: &sample.id,
            messages: EvalMessages::Chat(&sample.messages),
            chat_template_kwargs: None,
            position: String::new(),
        }
    }

    /// A clip whose bound is past the slot's budget fails alone, with the
    /// context that would fit it; one that fits, or whose length is unknown,
    /// is sent.
    #[test]
    fn a_clip_past_the_budget_fails_with_the_fix() -> anyhow::Result<()> {
        let benchmark = entry("qwen25_omni")?;
        let recipe = AsrRecipe::for_entry(&benchmark, &cell(None, Some(2), &[])?)?;
        let sender = AsrSender {
            recipe: &recipe,
            benchmark: &benchmark,
            max_tokens: 256,
            generation: 256,
            budget: 3000,
            slots: 2,
            stream: None,
            tee_stderr: false,
        };
        let reason = |sample: &EvalSample| sender.does_not_fit(&measured(sample));
        assert_eq!(reason(&sample_with("fits", wav(10.0), None)), None);
        let long = reason(&sample_with("long", wav(120.0), None)).unwrap_or_default();
        let needed = audio_tokens(120.0) + 21 + TEMPLATE_MARGIN_TOKENS + 256;
        assert!(long.contains(&needed.to_string()), "{long}");
        assert!(long.contains(&(needed * 2).to_string()), "{long}");
        assert_eq!(
            reason(&sample_with("unknown", "AAAA".to_owned(), None)),
            None
        );
        Ok(())
    }

    /// A length stop short of `max_tokens` is the slot running out, not the
    /// cap: it is `Unfit`, never a transcript. A length stop at the cap, a
    /// natural stop, or a reply that carries no count, is the answer.
    #[rstest]
    #[case::cut_by_the_context(Some("length"), Some(40), true)]
    #[case::at_the_cap(Some("length"), Some(128), false)]
    #[case::eos(Some("stop"), Some(12), false)]
    #[case::no_count(Some("length"), None, false)]
    fn an_answer_cut_short_for_want_of_context_is_unfit(
        #[case] finish_reason: Option<&str>,
        #[case] completion_tokens: Option<u64>,
        #[case] unfit: bool,
    ) -> anyhow::Result<()> {
        let benchmark = beam_entry()?;
        let recipe = AsrRecipe::for_entry(&benchmark, &cell(None, None, &[])?)?;
        let sender = AsrSender {
            recipe: &recipe,
            benchmark: &benchmark,
            max_tokens: 128,
            generation: 384,
            budget: 3000,
            slots: 1,
            stream: None,
            tee_stderr: false,
        };
        let reply = ChatReply {
            content: "partial".to_owned(),
            finish_reason: finish_reason.map(str::to_owned),
            completion_tokens,
            ..ChatReply::default()
        };
        let got = sender.cut_short(&reply);
        assert_eq!(
            matches!(got, Some(EvalMeasurementReading::Unfit { .. })),
            unfit,
            "{got:?}"
        );
        Ok(())
    }

    /// The server's own context refusals, plain or the ASR build's beam room, are
    /// told apart from every other failure.
    #[rstest]
    #[case::exceeds(
        r#"400 Bad Request: {"error":{"code":400,"message":"request (5000 tokens) exceeds the available context size (4096 tokens), try increasing it","type":"exceed_context_size_error"}}"#,
        true
    )]
    #[case::beam_room("no room in the context for beam search", true)]
    #[case::other("400 Bad Request: failed to decode audio", false)]
    fn a_context_refusal_is_recognised(#[case] text: &str, #[case] refusal: bool) {
        let err = anyhow::anyhow!(text.to_owned()).context("POST /v1/chat/completions");
        assert_eq!(is_context_refusal(&err), refusal);
    }

    /// A beam reply is read from its one response, its stop with it.
    #[test]
    fn a_one_response_reply_reads_its_stop() -> anyhow::Result<()> {
        let reply = chat_reply(&json!({
            "choices": [{"message": {"content": "hallo"}, "finish_reason": "stop"}],
            "usage": {"completion_tokens": 3}
        }))?;
        assert_eq!(reply.content, "hallo");
        assert_eq!(reply.completion_tokens, Some(3));
        assert_eq!(reply.stop().0, BenchmarkEvalCompletionStopReason::Eos);
        Ok(())
    }
}
