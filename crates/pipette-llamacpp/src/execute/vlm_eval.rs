use std::{
    num::NonZeroUsize,
    path::Path,
    sync::{Mutex, PoisonError},
    time::Duration,
};

use anyhow::Context;
use reqwest::Method;
use serde_json::{Value, json};

use pipette_http::{HttpClient, StreamClient};
use pipette_ops::eval_completions::EvalCompletionSession;
use pipette_ops::measurement::{
    self, EvalMeasurementDataset, EvalMeasurementInFlight, EvalMeasurementReading,
    EvalMeasurementSample, MeasurementObserver,
};
use pipette_ops::readiness::ReadinessGate;
use pipette_ops::run_response::RunResponseBuilder;
use pipette_plan_types::{
    BenchmarkEvalCompletionStopReason, BenchmarkFlagRef, EvalSample,
    LlamacppCliStockToolsExtraRuntimeConfiguration, MinP, ModelFlags, PositiveCount,
    PresencePenalty, RepeatPenalty, RunResponse, RuntimeFlags, TopP, VlmEvalBenchmark,
    VlmEvalLlamacppCliStockToolsGgufVision,
};

use pipette_vl_spec_decode_corpus::vlm_eval::{self, VlmEvalCorpus, prompt_thinks, samples_of};

/// The recipe rule, shared with every engine's vlm_eval.
pub(super) use pipette_vl_spec_decode_corpus::vlm_eval::on_recipe;

use crate::eval_pool::{
    LlamaServer, Server, ServerSlot, SlotNeed, check_slot_context, log_completed, mcq_grammar,
    read_sample, readiness_best_effort, slot_server, spawn_server, stream_chat_completion,
};
use crate::models::ResolvedGgufVisionModel;
use crate::mtmd::{self, MtmdCommandBuilder};
use crate::runtimes::{ResolvedLlamaServer, get_load_mode};
use crate::server::{self, LoggingMarker, ServerCommandBuilder};

const CHAT_ENDPOINT: &str = "/v1/chat/completions";

/// The order the sampling a model card gives is applied in by Hugging Face and
/// vLLM, where those numbers are tuned: penalties, then temperature, then the
/// top-k, top-p and min-p cuts. llama.cpp's default order divides by the
/// temperature last, so the same top_p would keep a different set of tokens.
const SAMPLERS: [&str; 5] = ["penalties", "temperature", "top_k", "top_p", "min_p"];

/// The sampling the entry pins (VLMEvalKit's or the model card's), else greedy,
/// with every sampler it leaves unset at its neutral value. A GGUF's
/// `general.sampling.*` defaults apply unless the request overrides them, and a
/// repeat penalty, XTC or mirostat would change the decode.
///
/// The numbers are applied as Hugging Face and vLLM apply them: in [`SAMPLERS`]
/// order, with the presence and repeat penalties counting every token of the
/// answer. llama.cpp's penalties otherwise see only the last 64 tokens, which
/// lets a reasoning model loop on a passage longer than that until it hits the
/// token cap. llama-server also counts the prompt's text in this window; Hugging
/// Face does the same for its repetition penalty, vLLM does not for presence.
fn sampled_request(benchmark: &VlmEvalBenchmark, sample_id: &str, fields: Value) -> Value {
    let mut body = json!({
        "samplers": SAMPLERS,
        "repeat_last_n": benchmark.parameter_max_tokens.into_inner(),
        "temperature": benchmark.sampling_temperature().as_f64(),
        "top_k": benchmark.parameter_top_k.map_or(0, i32::from),
        "top_p": benchmark.parameter_top_p.map_or(1.0, TopP::as_f64),
        "min_p": benchmark.parameter_min_p.map_or(0.0, MinP::as_f64),
        "typical_p": 1.0,
        "top_n_sigma": -1.0,
        "repeat_penalty": benchmark.parameter_repeat_penalty.map_or(1.0, RepeatPenalty::as_f64),
        "presence_penalty": benchmark
            .parameter_presence_penalty
            .map_or(0.0, PresencePenalty::as_f64),
        "frequency_penalty": 0.0,
        "dry_multiplier": 0.0,
        "xtc_probability": 0.0,
        "mirostat": 0,
    });
    if let Some(body) = body.as_object_mut() {
        if let Some(seed) = benchmark.sample_seed(sample_id) {
            body.insert("seed".to_owned(), json!(seed));
        }
        if let Value::Object(fields) = fields {
            body.extend(fields);
        }
    }
    body
}

/// VLM accuracy eval through [`measurement::eval_complete()`]: the text `eval`
/// runner's entry plus bounded resends of failed samples and a refusal to
/// publish an incomplete cell. Each sample's `messages` already embed their
/// `image_url` data URIs and are forwarded as-is, straight to
/// `/v1/chat/completions` (not `/apply-template` and `/completion`), so the
/// server renders the multimodal chat template and runs the vision tower.
#[expect(
    clippy::too_many_arguments,
    reason = "the request's parts, the checkpoint opened on its identity, and the seams"
)]
pub fn run(
    benchmark: &VlmEvalBenchmark,
    corpus: Option<&VlmEvalCorpus>,
    runtime_flags: VlmEvalLlamacppCliStockToolsGgufVision,
    model_flags: Option<&ModelFlags>,
    benchmark_flags: &BenchmarkFlagRef,
    program: &ResolvedLlamaServer,
    model: &ResolvedGgufVisionModel,
    checkpoint: EvalCompletionSession,
    readiness_gate: ReadinessGate,
    observer: MeasurementObserver<'_>,
    client_config: Option<&LlamacppCliStockToolsExtraRuntimeConfiguration>,
) -> anyhow::Result<RunResponse> {
    let samples = validate(benchmark, samples_of(benchmark, corpus)?, &runtime_flags)?;
    let enable_thinking = model_flags.and_then(ModelFlags::enable_thinking);
    let prompt_thinks = prompt_thinks(benchmark, enable_thinking)?;
    // Before the calibration, which can load the model: a thinking control the
    // cell may not carry is refused for the price of reading two files, a bad
    // flag set for the price of rendering the argv.
    let image_tokens = benchmark.parameter_image_tokens;
    let unsized_launch = launch(
        &runtime_flags,
        program,
        &model.backbone,
        &model.mmproj,
        image_tokens,
        client_config,
    );
    unsized_launch.check_thinking_controls()?;
    unsized_launch.args()?;
    let benchmark_id = benchmark.benchmark_id.as_str();
    let max_tokens_u32 = benchmark.parameter_max_tokens.into_inner().unsigned_abs();
    let ctx_pinned = runtime_flags.ctx_size.is_some();
    let parallel = slots(&runtime_flags);
    let raw_slots = server::raw_parallel(&runtime_flags.raw).and_then(|(_, count)| count);

    let mcq_choices = benchmark
        .parameter_mcq_choices
        .clone()
        .filter(|items| !items.is_empty());

    // A pinned ctx_size discards the derived default, so it skips the
    // calibration (a model load on a cache miss), and an over-ceiling estimate
    // only warns: the operator already did what the error would ask.
    let derived_ctx = if ctx_pinned {
        mtmd::vlm_eval_fallback_ctx_size(
            samples,
            max_tokens_u32,
            &with_image_budget(runtime_flags.raw.clone(), image_tokens),
            parallel,
        )
        .unwrap_or_else(|e| {
            tracing::warn!(
                benchmark_id,
                error = %format_args!("{e:#}"),
                "vlm_eval ctx estimate is over the ceiling; ignored because the cell's \
                 pinned ctx_size wins"
            );
            None
        })
    } else {
        // Over-ceiling is an error here: clamping would undersize a real need
        // (see `bound_vlm_eval_ctx`).
        mtmd::vlm_eval_ctx_size(
            &calibration(&runtime_flags, program, model, image_tokens, client_config),
            samples,
            max_tokens_u32,
            parallel,
        )?
    }
    .unwrap_or_else(|| mtmd::VLM_EVAL_CTX_FLOOR.saturating_add(max_tokens_u32));
    match runtime_flags.ctx_size {
        Some(pinned) => {
            let budget_raw = with_image_budget(runtime_flags.raw.clone(), image_tokens);
            check_pinned_ctx(
                pinned,
                parallel.max(raw_slots.unwrap_or(1)),
                max_tokens_u32,
                mtmd::vlm_eval_sample_need(samples, max_tokens_u32, &budget_raw, None)
                    .map(|(need, _)| need),
                || {
                    mtmd::vlm_eval_sample_need(
                        samples,
                        max_tokens_u32,
                        &budget_raw,
                        Some(&calibration(
                            &runtime_flags,
                            program,
                            model,
                            image_tokens,
                            client_config,
                        )),
                    )
                },
            )?;
        }
        None => check_slot_context(
            LoggingMarker::VlmEval,
            parallel,
            raw_slots,
            None,
            derived_ctx,
            // The whole per-sample context, not max_tokens: for an MCQ cell
            // max_tokens is 1, so it would accept almost any division.
            SlotNeed::Sample { ctx: derived_ctx },
        )?,
    }
    let runtime_flags = with_ctx(runtime_flags, derived_ctx, parallel);
    let launch = launch(
        &runtime_flags,
        program,
        &model.backbone,
        &model.mmproj,
        image_tokens,
        client_config,
    );
    // Refused before anything is spawned. The reasoning flags come later and
    // add no flag the checks read.
    launch.args()?;

    let request_timeout = measurement::http_timeout_for(benchmark_flags.http_timeout_seconds, 1);
    let idle_timeout = measurement::stream_idle_timeout_for(benchmark_flags.http_timeout_seconds);

    // Text eval is not gated, but thousands of images through the vision tower
    // can overheat an edge device: the gate protects the device, not the score.
    // Only at the start: the eval entry's attempts must be `Sync` and the gate
    // is not, and a cooldown between samples cools nothing at `parallel = N`.
    readiness_best_effort(readiness_gate, "vlm_eval start");
    if parallel > 1 {
        tracing::warn!(
            benchmark_id,
            parallel,
            "vlm_eval running samples concurrently; set parallel=1 on thermally constrained \
             hardware"
        );
    }
    // One thermal pair brackets the whole run: accuracy does not depend on
    // clock speed, so a per-sample series would say nothing.
    (observer.on_before_rep_start)()?;

    let spawn = |launch: &ServerCommandBuilder| spawn_server(launch, request_timeout);
    let http = HttpClient::loopback("pipette", request_timeout)?;
    let (first, launch) = with_thinking(
        enable_thinking,
        prompt_thinks,
        &model.backbone,
        &http,
        launch,
        &spawn,
    )?;
    // The flags an allocation failure quotes.
    let argv = launch.args()?;
    // Restarts reuse the reasoning flags too: a server that came back
    // configured differently would change what the rest of the cell measures.
    let start = || spawn(&launch);
    let slot = Mutex::new(ServerSlot {
        server: Server::Up(first),
        generation: 0,
    });
    let completion_mode = VlmCompletionMode::pick(
        mcq_choices.as_deref(),
        u64::from(max_tokens_u32),
        idle_timeout,
        // A terminal only: in a log the raw tokens would split the
        // one-event-per-line stream.
        parallel <= 1 && std::io::IsTerminal::is_terminal(&std::io::stderr()),
    )?;
    let workers = usize::try_from(parallel)
        .context("parallel does not fit in usize")?
        .clamp(1, samples.len().max(1));

    let result_data = measurement::eval_complete(
        EvalMeasurementDataset::vlm_eval(benchmark, samples),
        EvalMeasurementInFlight::Parallel(NonZeroUsize::new(workers).unwrap_or(NonZeroUsize::MIN)),
        checkpoint,
        &slot,
        |slot, samples, record| {
            samples.iter().enumerate().try_for_each(|(at, sample)| {
                let Some((base_url, generation)) = slot_server(slot, &start)? else {
                    return record(
                        at,
                        EvalMeasurementReading::Lost {
                            reason: "llama-server did not come back after a crash".to_owned(),
                        },
                    );
                };
                let reading = read_sample(slot, generation, sample, &argv, CHAT_ENDPOINT, || {
                    completion_mode.run_sample(&http, &base_url, benchmark, sample)
                })?;
                record(at, reading)
            })?;
            Ok(())
        },
    )?;
    (observer.on_after_rep_end)()?;
    // No memory observation, as on text eval: after a mid-run restart no one
    // server watched the whole run.
    let ServerSlot { server, .. } = slot.into_inner().unwrap_or_else(PoisonError::into_inner);
    let mut response = RunResponseBuilder::new(result_data).runtime_flags(Some(
        RuntimeFlags::VlmEvalLlamacppCliStockToolsGgufVision(runtime_flags),
    ));
    if let Some(corpus) = corpus {
        response = response.dataset_digest(corpus.digest().clone());
    }
    Ok(match server {
        Server::Up((mut llama_server, _)) | Server::Down(mut llama_server) => {
            response.process(&mut llama_server).build()
        }
        Server::RestartFailed => response.build(),
    })
}

/// Fails before anything is calibrated or spawned.
fn validate<'a>(
    benchmark: &VlmEvalBenchmark,
    samples: &'a [EvalSample],
    runtime_flags: &VlmEvalLlamacppCliStockToolsGgufVision,
) -> anyhow::Result<&'a [EvalSample]> {
    let benchmark_id = benchmark.benchmark_id.as_str();
    vlm_eval::check_samples(benchmark, samples)?;
    // The slot check would always refuse this after calibration: only the
    // typed count multiplies the derived one-slot context. Refused here, before
    // a calibration that can load the model. A raw flag beside the typed knob
    // is the builder's `RawSlotConflict`.
    let raw_slots = server::raw_parallel(&runtime_flags.raw)
        .and_then(|(_, count)| count)
        .unwrap_or(1);
    anyhow::ensure!(
        raw_slots <= 1 || runtime_flags.parallel.is_some() || runtime_flags.ctx_size.is_some(),
        "vlm_eval `{benchmark_id}`: raw `-np {raw_slots}` would split a context sized for one \
         slot across {raw_slots}; set the typed `parallel`, which sizes the context for every \
         slot, or pin ctx_size",
    );
    // Past the ceiling, no derivable n_ctx satisfies the cell, so an unpinned
    // one fails here rather than after a calibration. The floor is per slot, and
    // the ceiling bounds the total (see `bound_vlm_eval_ctx`). A pinned
    // ctx_size already did what the error asks, so it only warns.
    let max_tokens = benchmark.parameter_max_tokens.into_inner().unsigned_abs();
    let parallel = slots(runtime_flags);
    let floor_per_slot = mtmd::VLM_EVAL_CTX_FLOOR.saturating_add(max_tokens);
    let floor_total = floor_per_slot.saturating_mul(parallel);
    let ceiling = mtmd::VLM_EVAL_CTX_CEILING;
    if floor_total <= ceiling {
        return Ok(samples);
    }
    anyhow::ensure!(
        runtime_flags.ctx_size.is_some(),
        "vlm_eval `{benchmark_id}`: parameter_max_tokens={max_tokens} needs an n_ctx of at least \
         {floor_per_slot} per slot x {parallel} slot(s) = {floor_total}, past the {ceiling} \
         ceiling; fix the cell's max_tokens, lower `parallel`, or pin --ctx-size explicitly",
    );
    tracing::warn!(
        benchmark_id,
        max_tokens,
        floor_per_slot,
        parallel,
        floor_total,
        ceiling,
        "vlm_eval ctx floor is past the ceiling; proceeding because the cell's pinned ctx_size \
         wins over any derived default"
    );
    Ok(samples)
}

fn slots(runtime_flags: &VlmEvalLlamacppCliStockToolsGgufVision) -> u32 {
    runtime_flags
        .parallel
        .map_or(1, |n| n.into_inner().unsigned_abs())
}

/// llama-server divides `--ctx-size` across its slots, so the calibrated
/// one-slot budget is multiplied by the slot count. An operator's `ctx_size` is
/// a stated total and is not, as on the text eval cell.
fn with_ctx(
    runtime_flags: VlmEvalLlamacppCliStockToolsGgufVision,
    per_slot_ctx: u32,
    slots: u32,
) -> VlmEvalLlamacppCliStockToolsGgufVision {
    VlmEvalLlamacppCliStockToolsGgufVision {
        load_mode: get_load_mode(runtime_flags.mmap, runtime_flags.load_mode),
        // Cannot saturate: `validate` or `mtmd::bound_vlm_eval_ctx` has
        // already refused a product that large.
        ctx_size: runtime_flags
            .ctx_size
            .or_else(|| Some(per_slot_ctx.saturating_mul(slots))),
        ..runtime_flags
    }
}

/// The thinking flags depend on the chat template, which only a running server
/// can report, so the first server starts with none; a template with a switch
/// costs a second model load.
fn with_thinking(
    enable_thinking: Option<bool>,
    prompt_thinks: bool,
    model: &Path,
    http: &HttpClient,
    launch: ServerCommandBuilder,
    spawn: &dyn Fn(&ServerCommandBuilder) -> anyhow::Result<LlamaServer>,
) -> anyhow::Result<(LlamaServer, ServerCommandBuilder)> {
    let first = spawn(&launch)?;
    let thinking_switch = server::thinking_switch(http, &first.1)?;
    let extra = server::thinking(enable_thinking, thinking_switch, prompt_thinks)
        .map_err(|refused| refused.error(model))?;
    tracing::info!(
        model = %model.display(),
        thinking_switch,
        enable_thinking = ?enable_thinking,
        prompt_thinks,
        flags = %extra.join(" "),
        restart = !extra.is_empty(),
        "thinking mode applied through the model's chat template"
    );
    if extra.is_empty() {
        return Ok((first, launch));
    }
    // Stopped first, so two copies of the model never have to fit.
    drop(first);
    let launch = launch.extra(extra);
    Ok((spawn(&launch)?, launch))
}

/// Every field named, so a flag plan-types adds does not compile until it is
/// rendered or set aside. The prompt caches are off: every sample is a new
/// prompt, so they would only copy state out between samples, stalling every
/// slot, and no answer may depend on which samples ran before.
fn launch(
    flags: &VlmEvalLlamacppCliStockToolsGgufVision,
    program: &ResolvedLlamaServer,
    model: &Path,
    mmproj: &Path,
    image_tokens: Option<PositiveCount>,
    client_config: Option<&LlamacppCliStockToolsExtraRuntimeConfiguration>,
) -> ServerCommandBuilder {
    let VlmEvalLlamacppCliStockToolsGgufVision {
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
    ServerCommandBuilder::new(program.clone(), model, LoggingMarker::VlmEval)
        .mmproj(mmproj)
        .threads(threads)
        .number_gpu_layers(number_gpu_layers)
        .mmap(mmap)
        .load_mode(load_mode)
        .flash_attention(flash_attention)
        .raw(with_image_budget(raw, image_tokens))
        .ctx_size(ctx_size)
        .no_cache(no_cache)
        .without_prompt_caches()
        .slots(parallel.map(|n| n.into_inner().unsigned_abs()))
        .envs(envs)
        .client_config(client_config.cloned())
}

/// Every field named, as in [`launch`]. The ignored ones change how the server
/// serves, not the token count or where the model loads.
fn calibration(
    flags: &VlmEvalLlamacppCliStockToolsGgufVision,
    program: &ResolvedLlamaServer,
    model: &ResolvedGgufVisionModel,
    image_tokens: Option<PositiveCount>,
    client_config: Option<&LlamacppCliStockToolsExtraRuntimeConfiguration>,
) -> MtmdCommandBuilder {
    let VlmEvalLlamacppCliStockToolsGgufVision {
        threads: _,
        number_gpu_layers,
        mmap: _,
        load_mode: _,
        flash_attention: _,
        ctx_size: _,
        no_cache: _,
        parallel: _,
        envs,
        raw,
    } = flags.clone();
    MtmdCommandBuilder::new(program.clone(), &model.backbone, &model.mmproj)
        .number_gpu_layers(number_gpu_layers)
        .raw(with_image_budget(raw, image_tokens))
        .envs(envs)
        .client_config(client_config.cloned())
        .without_inherited_llama_args()
}

/// Both bounds at the budget, so the server and its calibration size every
/// image to it. plan-types refuses these flags in a vlm_eval cell's `raw`.
fn with_image_budget(mut raw: Vec<String>, image_tokens: Option<PositiveCount>) -> Vec<String> {
    if let Some(tokens) = image_tokens {
        let tokens = tokens.into_inner().to_string();
        raw.extend([
            "--image-min-tokens".to_owned(),
            tokens.clone(),
            "--image-max-tokens".to_owned(),
            tokens,
        ]);
    }
    raw
}

/// Refuse a pinned `ctx_size` whose slots cannot each hold the largest sample.
/// The resume identity leaves ctx_size out as a throughput knob, which holds
/// only while every sample fits: a resume under a larger context would mix a
/// cut-short answer with whole ones.
///
/// `estimate` is the patch-grid over-estimate, so only a slot below it pays for
/// `measure` (a model load on a cache miss): a budgeted projector can give far
/// fewer tokens than the grid.
fn check_pinned_ctx(
    pinned: u32,
    slots: u32,
    max_tokens: u32,
    estimate: Option<u32>,
    measure: impl FnOnce() -> Option<(u32, &'static str)>,
) -> anyhow::Result<()> {
    let slots = slots.max(1);
    let per_slot = pinned / slots;
    let (need, how) = match estimate {
        None => (
            max_tokens,
            "its output budget; no image size could be read".to_owned(),
        ),
        Some(estimate) if per_slot >= estimate => return Ok(()),
        Some(estimate) => {
            let (need, source) = measure().unwrap_or((estimate, "estimated"));
            (need, format!("image, prompt and output, {source}"))
        }
    };
    anyhow::ensure!(
        per_slot >= need,
        "vlm_eval: ctx_size {pinned} across {slots} slot(s) leaves {per_slot} tokens per slot, \
         under the {need} tokens one sample needs ({how}), so a sample would overflow or have \
         its answer cut short. Raise ctx_size to at least {}, lower parallel, or leave \
         ctx_size unset to size it from the dataset",
        need.saturating_mul(slots),
    );
    Ok(())
}

enum VlmCompletionMode {
    /// Grammar-constrained single-token request, answered as one JSON response.
    Mcq { grammar: String },
    Streaming {
        /// A connect-only timeout, so a large image prefill does not kill the
        /// stream, plus the idle bound the reader fails on.
        client: StreamClient,
        max_tokens: u64,
        /// Off for a concurrent pool: the tee holds the process-wide stderr
        /// lock for a whole generation, which would serialise the workers.
        tee_stderr: bool,
    },
}

impl VlmCompletionMode {
    fn pick(
        mcq_choices: Option<&[String]>,
        max_tokens: u64,
        idle_timeout: Duration,
        tee_stderr: bool,
    ) -> anyhow::Result<Self> {
        match mcq_choices {
            Some(choices) => Ok(Self::Mcq {
                grammar: mcq_grammar(choices),
            }),
            None => Ok(Self::Streaming {
                client: HttpClient::streaming("pipette", idle_timeout)
                    .context("failed to build streaming HTTP client")?,
                max_tokens,
                tee_stderr,
            }),
        }
    }

    /// One attempt at the sample. The caller resends one the server dropped
    /// ([`read_sample`]): a vlm_eval sample carries no timing measurement, so a
    /// resend costs latency nobody records.
    fn run_sample(
        &self,
        http: &HttpClient,
        base_url: &str,
        benchmark: &VlmEvalBenchmark,
        sample: &EvalMeasurementSample<'_>,
    ) -> anyhow::Result<EvalMeasurementReading> {
        let url = format!("{base_url}{CHAT_ENDPOINT}");
        let messages = sample.messages;
        let reading = match self {
            Self::Mcq { grammar } => {
                // llama.cpp's OpenAI-compatible endpoint takes a top-level
                // `grammar` as an extension.
                let body = sampled_request(
                    benchmark,
                    sample.id,
                    json!({
                        "messages": messages,
                        "max_tokens": 1,
                        "grammar": grammar,
                        "reasoning_format": "none",
                    }),
                );
                // The error keeps the response body: a ctx-overflow or
                // image-decode 400 carries the server's message.
                let response: Value = http
                    .json_request(Method::POST, &url, None, Some(body))
                    .with_context(|| format!("POST {CHAT_ENDPOINT}"))?;
                let content = response
                    .get("choices")
                    .and_then(|c| c.get(0))
                    .and_then(|c| c.get("message"))
                    .and_then(|m| m.get("content"))
                    .and_then(Value::as_str)
                    .context("/v1/chat/completions response missing message content")?
                    .to_string();
                EvalMeasurementReading::Completed {
                    completion: content,
                    stop_reason: BenchmarkEvalCompletionStopReason::Unknown,
                    stop_detail: Some("mcq arm (max_tokens:1, grammar-constrained)".to_string()),
                    completion_tokens: None,
                }
            }
            Self::Streaming {
                client,
                max_tokens,
                tee_stderr,
            } => {
                let body = sampled_request(
                    benchmark,
                    sample.id,
                    json!({
                        "messages": messages,
                        "max_tokens": max_tokens,
                        "reasoning_format": "none",
                        "stream": true,
                        // The final usage chunk gives completion_tokens to pair
                        // with stop_reason.
                        "stream_options": { "include_usage": true },
                    }),
                );
                let reply = stream_chat_completion(client, &url, &body, *tee_stderr)?;
                let (stop_reason, stop_detail) = reply.stop();
                EvalMeasurementReading::Completed {
                    completion: reply.content,
                    stop_reason,
                    stop_detail,
                    completion_tokens: reply.completion_tokens,
                }
            }
        };
        log_completed(sample, &reading);
        Ok(reading)
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use rstest::rstest;

    use pipette_ops::sse::{self, ChatReply};
    use pipette_plan_types::{Model, VlmPromptProfile};

    use pipette_plan_types::{BenchmarkId, PositiveCount};

    use super::*;
    use crate::eval_pool::read_chat_stream;
    use crate::runtimes::test_llama_server;

    /// What [`run`] falls back to for a `max_tokens = 16` cell with no
    /// parseable image.
    const PER_SLOT: u32 = mtmd::VLM_EVAL_CTX_FLOOR + 16;

    fn cell(
        ctx_size: Option<u32>,
        parallel: Option<i32>,
        raw: Vec<String>,
    ) -> anyhow::Result<VlmEvalLlamacppCliStockToolsGgufVision> {
        Ok(VlmEvalLlamacppCliStockToolsGgufVision {
            threads: None,
            number_gpu_layers: None,
            mmap: None,
            load_mode: None,
            flash_attention: None,
            ctx_size,
            no_cache: None,
            parallel: parallel.map(PositiveCount::try_new).transpose()?,
            envs: Vec::new(),
            raw,
        })
    }

    fn sample(id: &str) -> EvalSample {
        EvalSample {
            id: id.to_owned(),
            messages: Vec::new(),
            chat_template_kwargs: None,
        }
    }

    fn benchmark(
        count: u32,
        max_tokens: u32,
        samples: Vec<EvalSample>,
    ) -> anyhow::Result<VlmEvalBenchmark> {
        Ok(VlmEvalBenchmark {
            benchmark_id: BenchmarkId::try_new("vlm_eval_mmstar_2026.07.1".to_owned())?,
            parameter_lmms_task: "mmstar".into(),
            parameter_dataset_version: "2026.07.1".into(),
            parameter_sample_count: PositiveCount::try_new(i32::try_from(count)?)?,
            parameter_max_tokens: PositiveCount::try_new(i32::try_from(max_tokens)?)?,
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
            samples: Some(samples),
        })
    }

    #[test]
    fn a_catalog_entry_without_a_recipe_does_not_run() -> anyhow::Result<()> {
        let model: Model = serde_json::from_value(json!({
            "type": "gguf_vision", "source": "huggingface", "org": "o", "repo_name": "r",
            "model": "m.gguf", "mmproj": "p.gguf",
        }))?;
        let local = benchmark(1, 16, vec![sample("a")])?;
        on_recipe(&local, &model, None)?;
        let mut catalog = local.clone();
        catalog.samples = None;
        catalog.dataset = Some(Box::new(serde_json::from_value(json!({
            "format": "vlm-eval-corpus/1", "adapter": "evals-server@1",
            "digest": "1".repeat(64), "bytes": 1,
            "upstream": {"source": "evals_server", "eval_id": "mmstar",
                         "dataset_version": "2026.07.1", "identity": "2".repeat(64)},
        }))?));
        let error = on_recipe(&catalog, &model, None)
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default();
        assert!(error.contains("names no recipe"), "{error}");
        catalog.recipe = Some(serde_json::from_value(json!({
            "family": "F", "models": [{"repo": "o/r"}], "sources": ["https://a"],
        }))?);
        on_recipe(&catalog, &model, None)?;
        Ok(())
    }

    #[test]
    fn a_request_applies_the_sampling_as_hugging_face_and_vllm_do() -> anyhow::Result<()> {
        let mut entry = benchmark(1, 32768, vec![sample("a")])?;
        entry.parameter_temperature = Some(pipette_plan_types::SamplingTemperature::try_new(0.6)?);
        entry.parameter_top_p = Some(TopP::try_new(0.95)?);
        entry.parameter_top_k = Some(pipette_plan_types::TopK::try_new(20)?);
        entry.parameter_presence_penalty = Some(PresencePenalty::try_new(1.5)?);
        let body = sampled_request(&entry, "a", json!({"max_tokens": 32768}));
        assert_eq!(
            body["samplers"],
            json!(["penalties", "temperature", "top_k", "top_p", "min_p"])
        );
        assert_eq!(body["repeat_last_n"], json!(32768));
        assert_eq!(body["presence_penalty"], json!(1.5));
        assert_eq!(body["temperature"], json!(0.6));
        Ok(())
    }

    fn argv(runtime_flags: VlmEvalLlamacppCliStockToolsGgufVision) -> anyhow::Result<Vec<String>> {
        let slots = slots(&runtime_flags);
        let flags = with_ctx(runtime_flags, PER_SLOT, slots);
        Ok(launch(
            &flags,
            &test_llama_server(),
            Path::new("model.gguf"),
            Path::new("mmproj.gguf"),
            None,
            None,
        )
        .args()?)
    }

    #[rstest]
    #[case::unset(Vec::new(), None, &["--cache-ram", "0", "--ctx-checkpoints", "0", "--no-cache-prompt"])]
    #[case::slots(Vec::new(), Some(3), &["--cache-ram", "0", "--ctx-checkpoints", "0", "--no-cache-prompt"])]
    #[case::raw_size(
        vec!["--cache-ram".to_owned(), "512".to_owned()], None,
        &["--cache-ram", "512", "--ctx-checkpoints", "0", "--no-cache-prompt"]
    )]
    fn every_prompt_cache_is_off_unless_raw_sets_it(
        #[case] raw: Vec<String>,
        #[case] parallel: Option<i32>,
        #[case] expected: &[&str],
    ) -> anyhow::Result<()> {
        let argv = argv(cell(None, parallel, raw)?)?;
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
        let slots = parallel.map(|n| n.to_string());
        let rendered = argv
            .windows(2)
            .find(|w| w[0] == "--parallel")
            .map(|w| w[1].clone());
        assert_eq!(rendered, slots.filter(|n| n != "1"), "{argv:?}");
        Ok(())
    }

    #[test]
    fn the_image_budget_reaches_the_server_and_the_calibration() -> anyhow::Result<()> {
        let budget = Some(PositiveCount::try_new(1120)?);
        let server = launch(
            &cell(None, None, Vec::new())?,
            &test_llama_server(),
            Path::new("model.gguf"),
            Path::new("mmproj.gguf"),
            budget,
            None,
        )
        .args()?;
        let bounds = |argv: &[String]| -> std::collections::BTreeSet<(String, String)> {
            argv.windows(2)
                .filter(|w| w[0] == "--image-min-tokens" || w[0] == "--image-max-tokens")
                .map(|w| (w[0].clone(), w[1].clone()))
                .collect()
        };
        let expected = [
            ("--image-min-tokens".to_owned(), "1120".to_owned()),
            ("--image-max-tokens".to_owned(), "1120".to_owned()),
        ]
        .into_iter()
        .collect();
        assert_eq!(bounds(&server), expected);
        assert_eq!(
            with_image_budget(vec!["-ub".to_owned(), "512".to_owned()], budget),
            [
                "-ub",
                "512",
                "--image-min-tokens",
                "1120",
                "--image-max-tokens",
                "1120"
            ]
        );
        assert!(with_image_budget(Vec::new(), None).is_empty());
        Ok(())
    }

    fn rendered_ctx(argv: &[String]) -> anyhow::Result<&str> {
        argv.windows(2)
            .find(|w| w[0] == "--ctx-size")
            .map(|w| w[1].as_str())
            .ok_or_else(|| anyhow::anyhow!("argv must carry a context: {argv:?}"))
    }

    #[rstest]
    #[case::calibrated_default_is_one_slots_budget(None, None, PER_SLOT)]
    #[case::calibrated_default_scales_with_the_slots(None, Some(4), PER_SLOT * 4)]
    // `Some(1)` must render exactly what `None` does: a stray difference here
    // would make otherwise-identical runs compare as different configurations.
    #[case::one_slot_is_the_serial_number(None, Some(1), PER_SLOT)]
    #[case::operator_total_is_verbatim(Some(40_000), None, 40_000)]
    #[case::operator_total_is_verbatim_at_n_slots(Some(40_000), Some(4), 40_000)]
    fn only_the_calibrated_ctx_is_multiplied_by_the_slot_count(
        #[case] ctx_size: Option<u32>,
        #[case] parallel: Option<i32>,
        #[case] expected_ctx: u32,
    ) -> anyhow::Result<()> {
        assert_eq!(
            rendered_ctx(&argv(cell(ctx_size, parallel, Vec::new())?)?)?,
            expected_ctx.to_string()
        );
        Ok(())
    }

    /// The rendered flags are persisted with the result, so a stray
    /// `--parallel 1` would fork the configuration a run is compared under.
    #[rstest]
    #[case::unset(None)]
    #[case::explicit_one(Some(1))]
    fn one_slot_renders_no_parallel_flag(#[case] parallel: Option<i32>) -> anyhow::Result<()> {
        let got = argv(cell(None, parallel, Vec::new())?)?;
        let baseline = argv(cell(None, None, Vec::new())?)?;
        assert_eq!(got, baseline);
        assert!(
            !got.iter().any(|t| t == "--parallel" || t == "-np"),
            "a single slot must emit no slot flag: {got:?}"
        );
        Ok(())
    }

    #[rstest]
    #[case::unset(None, "1", PER_SLOT)]
    #[case::one(Some(1), "1", PER_SLOT)]
    #[case::four(Some(4), "4", PER_SLOT * 4)]
    fn no_cache_keeps_the_cells_slot_count(
        #[case] parallel: Option<i32>,
        #[case] expected_slots: &str,
        #[case] expected_ctx: u32,
    ) -> anyhow::Result<()> {
        let flags = VlmEvalLlamacppCliStockToolsGgufVision {
            no_cache: Some(true),
            ..cell(None, parallel, Vec::new())?
        };
        let got = argv(flags)?;
        assert!(
            got.windows(2).any(|w| w == ["--parallel", expected_slots]),
            "{got:?}"
        );
        assert_eq!(rendered_ctx(&got)?, expected_ctx.to_string());
        Ok(())
    }

    #[test]
    fn the_first_server_carries_no_reasoning_flag() -> anyhow::Result<()> {
        let got = argv(cell(None, None, Vec::new())?)?;
        assert!(!got.iter().any(|t| t.starts_with("--reasoning")), "{got:?}");
        Ok(())
    }

    #[rstest]
    #[case::r1_declared(Some("internvl_r1"), Some(true), Ok(true))]
    #[case::r1_unstated(Some("internvl_r1"), None, Err(()))]
    #[case::r1_off(Some("internvl_r1"), Some(false), Err(()))]
    #[case::other_profile(Some("internvl"), Some(true), Ok(false))]
    #[case::no_profile(None, None, Ok(false))]
    fn the_internvl_r1_profile_must_be_declared_thinking(
        #[case] profile: Option<&str>,
        #[case] enable_thinking: Option<bool>,
        #[case] expected: Result<bool, ()>,
    ) -> anyhow::Result<()> {
        let mut entry = benchmark(1, 16, vec![sample("a")])?;
        entry.parameter_prompt_profile = profile.map(VlmPromptProfile::from);
        assert_eq!(
            prompt_thinks(&entry, enable_thinking).map_err(|_| ()),
            expected
        );
        Ok(())
    }

    #[test]
    fn the_weights_are_pinned_unless_the_cell_said() -> anyhow::Result<()> {
        let got = argv(cell(None, None, Vec::new())?)?;
        assert!(
            got.windows(2)
                .any(|w| w[0] == "--load-mode" && w[1] == "none"),
            "{got:?}"
        );
        Ok(())
    }

    /// With two slot flags llama-server would honor one while the context was
    /// sized for the other.
    #[rstest]
    #[case::long_form(&["--parallel", "8"])]
    #[case::short_form(&["-np", "8"])]
    #[case::glued_short_form(&["-np8"])]
    fn a_raw_slot_flag_beside_the_typed_knob_is_refused(
        #[case] raw: &[&str],
    ) -> anyhow::Result<()> {
        let raw = raw.iter().map(|s| (*s).to_string()).collect();
        let err = match argv(cell(None, Some(4), raw)?) {
            Ok(argv) => anyhow::bail!("two slot flags must be refused, rendered {argv:?}"),
            Err(e) => format!("{e:#}"),
        };
        assert!(err.contains("conflicts with raw"), "got {err}");
        Ok(())
    }

    /// A raw slot flag with no typed knob passes through unscaled, so plans
    /// that work keep working, and divides a one-slot context.
    #[test]
    fn a_raw_slot_flag_still_divides_the_calibrated_context() -> anyhow::Result<()> {
        let err = match check_slot_context(
            LoggingMarker::VlmEval,
            1,
            Some(8),
            None,
            PER_SLOT,
            SlotNeed::Sample { ctx: PER_SLOT },
        ) {
            Ok(()) => anyhow::bail!("8 raw slots over one slot's budget must be refused"),
            Err(e) => format!("{e:#}"),
        };
        assert!(err.starts_with("vlm_eval:"), "got {err}");
        assert!(err.contains("per slot"), "got {err}");
        assert!(
            err.contains(&format!("per-sample context of {PER_SLOT}")),
            "got {err}"
        );
        assert!(!err.contains("max_tokens"), "got {err}");
        Ok(())
    }

    #[rstest]
    #[case::holds_the_estimate(16_384, 1, Some(9_000), None, true, false)]
    #[case::one_slot_too_small(4_096, 1, Some(9_000), Some(8_000), false, true)]
    #[case::measured_need_fits(32_768, 2, Some(40_000), Some(2_300), true, true)]
    #[case::measured_need_does_not(32_768, 2, Some(40_000), Some(20_000), false, true)]
    #[case::unmeasurable_falls_back(32_768, 2, Some(40_000), None, false, true)]
    #[case::no_image_checks_the_output(8_192, 1, None, None, true, false)]
    #[case::no_image_output_too_long(4_096, 2, None, None, false, false)]
    fn a_pinned_context_must_hold_every_sample(
        #[case] pinned: u32,
        #[case] slots: u32,
        #[case] estimate: Option<u32>,
        #[case] measured: Option<u32>,
        #[case] accepted: bool,
        #[case] measures: bool,
    ) {
        let asked = std::cell::Cell::new(false);
        let got = check_pinned_ctx(pinned, slots, 4_096, estimate, || {
            asked.set(true);
            measured.map(|need| (need, "calibrated"))
        });
        assert_eq!(got.is_ok(), accepted, "{got:?}");
        assert_eq!(asked.get(), measures, "measured only below the estimate");
        if let Err(e) = got {
            let msg = format!("{e:#}");
            assert!(msg.starts_with("vlm_eval: ctx_size"), "{msg}");
            assert!(msg.contains("per slot"), "{msg}");
        }
    }

    /// `too_many_slots`: (8192 + 16) x 32 = 262656, past the 131072 ceiling.
    #[rstest]
    #[case::too_many_slots(
        1, 16, &["s1"], Some(32), &[],
        &["8208", "32", "262656", "131072", "parallel"]
    )]
    #[case::sample_count_mismatch(
        3, 16, &["s1"], None, &[],
        &["parameter_sample_count=3", "1 samples"]
    )]
    #[case::duplicate_sample_ids(
        3, 16, &["s1", "s2", "s1"], None, &[],
        &["duplicate sample id `s1`"]
    )]
    #[case::max_tokens_past_the_ctx_ceiling(
        1, 131_072, &["s1"], None, &[],
        &["parameter_max_tokens=131072", "ceiling"]
    )]
    #[case::raw_slots_without_the_typed_knob(
        1, 16, &["s1"], None, &["-np", "4"],
        &["raw `-np 4`", "typed `parallel`"]
    )]
    fn validate_refuses(
        #[case] count: u32,
        #[case] max_tokens: u32,
        #[case] ids: &[&str],
        #[case] parallel: Option<i32>,
        #[case] raw: &[&str],
        #[case] needles: &[&str],
    ) -> anyhow::Result<()> {
        let benchmark = benchmark(count, max_tokens, ids.iter().map(|id| sample(id)).collect())?;
        let raw = raw.iter().map(|s| (*s).to_string()).collect();
        let Err(err) = validate(
            &benchmark,
            samples_of(&benchmark, None)?,
            &cell(None, parallel, raw)?,
        ) else {
            anyhow::bail!("validate must refuse this cell")
        };
        let msg = format!("{err:#}");
        for needle in needles {
            assert!(msg.contains(needle), "error must name {needle}; got {msg}");
        }
        Ok(())
    }

    /// Neither is refused up front: a pinned ctx_size's division is judged by the
    /// slot check once the context is known, and one raw slot divides nothing.
    #[rstest]
    #[case::pinned_ctx(Some(65_536), &["-np", "4"])]
    #[case::one_raw_slot(None, &["-np", "1"])]
    fn validate_leaves_a_raw_slot_flag_it_cannot_judge(
        #[case] ctx_size: Option<u32>,
        #[case] raw: &[&str],
    ) -> anyhow::Result<()> {
        let benchmark = benchmark(1, 16, vec![sample("s1")])?;
        let raw = raw.iter().map(|s| (*s).to_string()).collect();
        validate(
            &benchmark,
            samples_of(&benchmark, None)?,
            &cell(ctx_size, None, raw)?,
        )?;
        Ok(())
    }

    #[test]
    fn max_tokens_past_the_ctx_ceiling_is_accepted_when_ctx_is_pinned() -> anyhow::Result<()> {
        let benchmark = benchmark(1, 130_000, vec![sample("s1")])?;
        validate(
            &benchmark,
            samples_of(&benchmark, None)?,
            &cell(Some(140_000), None, Vec::new())?,
        )?;
        Ok(())
    }

    /// A fixture body ends at EOF, never on silence, so the bound a test
    /// passes is unreachable.
    const NO_IDLE: Duration = Duration::from_secs(3600);

    fn read(lines: &[&str]) -> anyhow::Result<ChatReply> {
        let body = lines.join("\n\n") + "\n\n";
        read_chat_stream(
            sse::data_lines(Cursor::new(body.into_bytes()), NO_IDLE),
            |_| {},
        )
    }

    #[test]
    fn chat_sse_accumulates_delta_content_and_finish_reason() -> anyhow::Result<()> {
        let out = read(&[
            r#"data: {"choices":[{"delta":{"content":"Hel"},"finish_reason":null}]}"#,
            r#"data: {"choices":[{"delta":{"content":"lo"},"finish_reason":null}]}"#,
            r#"data: {"choices":[{"delta":{},"finish_reason":"stop"}]}"#,
            r#"data: {"choices":[],"usage":{"completion_tokens":2,"prompt_tokens":10}}"#,
            "data: [DONE]",
        ])?;
        assert_eq!(out.content, "Hello");
        assert_eq!(out.finish_reason.as_deref(), Some("stop"));
        assert_eq!(out.completion_tokens, Some(2));
        let (reason, detail) = out.stop();
        assert_eq!(reason, BenchmarkEvalCompletionStopReason::Eos);
        assert_eq!(detail, None);
        Ok(())
    }

    #[test]
    fn chat_sse_length_finish_classifies_truncated() -> anyhow::Result<()> {
        let out = read(&[
            r#"data: {"choices":[{"delta":{"content":"abc"},"finish_reason":null}]}"#,
            r#"data: {"choices":[{"delta":{},"finish_reason":"length"}]}"#,
            "data: [DONE]",
        ])?;
        let (reason, _) = out.stop();
        assert_eq!(reason, BenchmarkEvalCompletionStopReason::Truncated);
        Ok(())
    }

    #[rstest]
    #[case::llamacpp_object(
        r#"data: {"error":{"code":400,"message":"image exceeds context window"}}"#,
        "image exceeds context window"
    )]
    #[case::bare_string(r#"data: {"error":"decode failed"}"#, "decode failed")]
    fn chat_sse_in_band_error_fails_with_server_message(
        #[case] error_event: &str,
        #[case] expected: &str,
    ) -> anyhow::Result<()> {
        let Err(err) = read(&[
            r#"data: {"choices":[{"delta":{"content":"par"},"finish_reason":null}]}"#,
            error_event,
            "data: [DONE]",
        ]) else {
            anyhow::bail!("an in-band error event must fail the read")
        };
        let msg = format!("{err:#}");
        assert!(msg.contains(expected), "server message lost: {msg}");
        Ok(())
    }

    /// A failed read checkpoints the sample failed, so a resume retries it.
    #[rstest]
    #[case::eof_after_content(
        &[r#"data: {"choices":[{"delta":{"content":"abc"},"finish_reason":null}]}"#]
    )]
    #[case::done_after_a_usage_only_tail(&[
        r#"data: {"choices":[{"delta":{"content":"abc"},"finish_reason":null}]}"#,
        r#"data: {"choices":[],"usage":{"completion_tokens":1,"prompt_tokens":5}}"#,
        "data: [DONE]",
    ])]
    fn chat_sse_stream_without_finish_reason_fails_the_read(
        #[case] lines: &[&str],
    ) -> anyhow::Result<()> {
        let Err(err) = read(lines) else {
            anyhow::bail!("a stream without finish_reason must fail the read")
        };
        let msg = format!("{err:#}");
        assert!(
            msg.contains("before its terminal event") || msg.contains("without finish_reason"),
            "error must say the stream was cut: {msg}"
        );
        Ok(())
    }
}
