//! Peak memory for one audio clip, driven through `llama-server` under the
//! flavor's peak sampler ([`crate::memory::peak`])
//! (`docs/methodology/asr-max-memory.md`).
//!
//! Measured as `vl_max_memory` is, behind a warm-up server whose reading is
//! dropped. The KV cache is allocated for the whole context up front, so the
//! context is sized from the exact audio token count (counted by the runtime's
//! `llama-mtmd-cli`, confirmed by the server) and the cell's text: the peak
//! carries no cache for audio that is not there.

use std::cell::Cell;
use std::time::{Duration, Instant};

use anyhow::Context;
use base64::Engine;

use pipette_http::HttpClient;
use pipette_memprobe::{Peak, Sampler, SupportedWatch};
use pipette_ops::measurement::{
    self, MaxMemoryMeasurementReading, MeasurementObserver, MemoryMeasurementWarmup,
};
use pipette_ops::run_response::RunResponseBuilder;
use pipette_ops::wav;
use pipette_plan_types::{
    AsrClientExtras, AsrContextSizing, AsrMaxMemoryBenchmark,
    AsrMaxMemoryLlamacppCliStockToolsGgufVision, AudioDurationMillis, BenchmarkFlagRef,
    EngineExtraRuntimeConfiguration, LlamaCppFlavor,
    LlamacppCliStockToolsExtraRuntimeConfiguration, MemoryObservation, RunResponse, RuntimeFlags,
};
use pipette_subprocess::{Process, start_process};

use crate::asr::{self, CtxInputs, Recipe};
use crate::memory;
use crate::models::ResolvedGgufVisionModel;
use crate::mtmd::MtmdCommandBuilder;
use crate::runtimes::{self, ResolvedLlamaServer};
use crate::server::{self, LoggingMarker, ServerCommandBuilder};
#[cfg(any(target_os = "linux", target_os = "android"))]
use crate::swap_aware;

struct Clip<'a> {
    duration: AudioDurationMillis,
    format: wav::ClipFormat,
    text_tokens: u32,
    /// Decoded with end-of-sequence ignored, as `asr_throughput` decodes, so
    /// the peak holds the output's KV and decode buffers of the same workload.
    decode_tokens: u32,
    recipe: &'a Recipe,
    /// Only a count that sized the context is one for the server to confirm.
    calibrated: Option<&'a asr::Calibrated>,
    /// The prompt the first server checked, which every later server's must
    /// repeat.
    confirmed_prompt: Cell<Option<u32>>,
}

#[expect(
    clippy::too_many_arguments,
    reason = "the request's parts, and the seams its entry takes"
)]
pub fn run(
    benchmark: &AsrMaxMemoryBenchmark,
    runtime_flags: AsrMaxMemoryLlamacppCliStockToolsGgufVision,
    benchmark_flags: &BenchmarkFlagRef,
    flavor: &LlamaCppFlavor,
    program: &ResolvedLlamaServer,
    model: &ResolvedGgufVisionModel,
    observer: MeasurementObserver<'_>,
    client_config: Option<&LlamacppCliStockToolsExtraRuntimeConfiguration>,
) -> anyhow::Result<RunResponse> {
    // Before the calibration, which loads the model: a flavor refused here
    // runs nothing.
    let watch = memory::peak(flavor, |name| {
        runtimes::launch_env(name, program, &runtime_flags.envs, client_config)
    })?;
    let duration = benchmark.parameter_audio_duration_ms;
    let text_tokens = benchmark.parameter_text_tokens.into_inner().unsigned_abs();
    let decode_tokens = benchmark.parameter_decode_tokens.as_u32();
    asr::refuse_raw_parallel(&runtime_flags.raw, "asr_max_memory")?;
    let recipe = Recipe::for_mmproj(&model.mmproj)?;
    let format = wav::ClipFormat::of(
        benchmark.parameter_audio_sample_rate_hz,
        benchmark.parameter_audio_channels,
    );
    recipe.check_duration(duration.as_u32(), format.sample_rate_hz())?;
    let beyond_limit = recipe.beyond_documented_limit(duration.as_u32());
    let recipe_args = recipe.server_args(
        &runtime_flags.raw,
        &runtime_flags.envs,
        client_config.map_or(&[], EngineExtraRuntimeConfiguration::envs),
    )?;
    // The config files llama.cpp applies to every launch, for the price of
    // reading them; the server's own launch checks them again.
    launch(
        &runtime_flags,
        recipe_args.clone(),
        program,
        model,
        client_config,
    )
    .check_thinking_controls()?;
    // Derived only for a cell that pins no ctx_size: behind a pinned one the
    // calibration's cache miss would be a full model load (up to 600s) paid for
    // a number that is never used.
    let ctx_pinned = runtime_flags.ctx_size.is_some();
    let calibrated = (!ctx_pinned)
        .then(|| {
            observer.outside(|| {
                asr::calibrate(
                    &recipe,
                    duration,
                    format,
                    &calibration(&runtime_flags, program, model, client_config),
                )
            })
        })
        .flatten();
    let audio_bound = asr::fallback_audio_tokens(duration.as_u32());
    if !ctx_pinned && calibrated.is_none() {
        // Named, since it is the one way this peak carries a KV cache sized for
        // more audio than the clip holds.
        tracing::warn!(
            duration_ms = duration.as_u32(),
            "the audio tokens could not be counted; sizing the context from the upper bound, so \
             the peak includes KV for up to {audio_bound} audio tokens"
        );
    }
    let ctx_size = CtxInputs {
        audio_bound,
        text_tokens,
        decode_tokens,
        text_chars: recipe.text_chars(),
    }
    .ctx_size(calibrated.as_ref().map(|c| c.tokens));
    // Final: a count the server does not confirm fails the cell ([`drive`]).
    let sizing = asr::context_sizing(ctx_pinned, calibrated.is_some());
    let clip = Clip {
        duration,
        format,
        text_tokens,
        decode_tokens,
        recipe: &recipe,
        calibrated: calibrated.as_ref(),
        confirmed_prompt: Cell::new(None),
    };
    let runtime_flags = AsrMaxMemoryLlamacppCliStockToolsGgufVision {
        load_mode: runtimes::get_load_mode(runtime_flags.mmap, runtime_flags.load_mode),
        ctx_size: runtime_flags.ctx_size.or(Some(ctx_size)),
        ..runtime_flags
    };
    let launch = launch(&runtime_flags, recipe_args, program, model, client_config);
    let timeout = measurement::http_timeout_for(benchmark_flags.http_timeout_seconds, 1);
    // One budget per server from spawn through the workload: the model load,
    // then the audio encode, prefill and decode.
    let deadline = memory::attempt_timeout(benchmark_flags.attempt_timeout_seconds);
    let mut served = None;
    // How the context was sized, so only a peak sized from the exact audio count
    // is read as the clip's own, and whether the clip is past the model's
    // documented audio limit.
    #[cfg_attr(
        not(any(target_os = "linux", target_os = "android")),
        expect(
            unused_mut,
            reason = "the swap-aware readings join it on Linux and Android"
        )
    )]
    let mut extras = serde_json::Map::from_iter([
        (
            AsrContextSizing::KEY.to_owned(),
            serde_json::Value::from(sizing.to_string()),
        ),
        (
            AsrClientExtras::BEYOND_MODEL_AUDIO_LIMIT_KEY.to_owned(),
            serde_json::Value::from(beyond_limit),
        ),
    ]);
    let result = measurement::max_memory(
        &benchmark.benchmark_id,
        observer,
        MemoryMeasurementWarmup::Once,
        &mut served,
        |served| {
            let peak = serve(served, &launch, watch, &clip, timeout, deadline)?;
            #[cfg(any(target_os = "linux", target_os = "android"))]
            let peak = {
                extras.extend(swap_aware::extras(&peak.footprint));
                peak.counting_swap()
            };
            Ok(MaxMemoryMeasurementReading {
                max_host_bytes: peak.host_bytes,
                max_gpu_bytes: peak.gpu_bytes,
                max_npu_bytes: peak.npu_bytes,
            })
        },
    );
    // Only an unpinned cell calibrates, so a count here sized the context, and a
    // failed run may be its fault.
    let data = result
        .map_err(|err| match &calibrated {
            Some(calibrated) => calibrated.drop_on_failure(err),
            None => err,
        })
        .map_err(|err| match &mut served {
            Some((llama_server, _)) => asr::with_server_tail(err, llama_server.stop_with_tail()),
            None => err,
        })?;
    let response = served
        .iter_mut()
        .fold(
            RunResponseBuilder::new(data),
            |response, (llama_server, memory)| response.process(llama_server).memory(*memory),
        )
        .runtime_flags(RuntimeFlags::AsrMaxMemoryLlamacppCliStockToolsGgufVision(
            runtime_flags,
        ))
        .client_extras(extras);
    Ok(response.build())
}

/// One server from its launch through the workload, within `deadline`, left in
/// `served`. One an earlier attempt left there stops first; the observation
/// covers both.
fn serve(
    served: &mut Option<(Process, MemoryObservation)>,
    launch: &ServerCommandBuilder,
    watch: SupportedWatch,
    clip: &Clip<'_>,
    timeout: Duration,
    deadline: Duration,
) -> anyhow::Result<Peak> {
    let earlier = match served.take() {
        Some((mut llama_server, memory)) => {
            llama_server.stop();
            memory
        }
        None => MemoryObservation::default(),
    };
    let port = pipette_subprocess::pick_server_port()?;
    let url = server::url(port);
    let ready = server::health(&url, timeout)?;
    let started = Instant::now();
    let mut llama_server = start_process("llama-server", launch.build(port)?, Duration::ZERO)?;
    let sampler = Sampler::start(llama_server.pid(), watch)?;
    llama_server.wait_ready(deadline, ready)?;
    let left = deadline.saturating_sub(started.elapsed());
    let worked = drive(&url, clip, timeout.min(left)).map_err(|err| {
        if started.elapsed() < deadline {
            return err;
        }
        err.context(format!(
            "the workload exceeded {deadline:?}; a wedge inside llama.cpp or a GPU driver is \
             the usual cause"
        ))
    });
    let (observed, peak) = sampler.stop();
    // Before the checks, so the cell quotes it on any failure from here.
    *served = Some((llama_server, earlier.merge_max(observed)));
    worked?;
    Ok(peak?)
}

/// The peak covers the audio encode, the prompt's and the output's KV, and the
/// decode buffers: the throughput cell's workload.
///
/// The first server (the warm-up's) checks the prompt, as `asr_throughput` does
/// after its reps: without the clip the same turns must prefill fewer tokens,
/// and for a cell with text, without the text exactly `parameter_text_tokens`
/// fewer. Only a calibrated count that sized the context is also held to the
/// clip's share ([`asr::audio_tokens`]); one the server does not confirm fails
/// the cell, as the peak is not the clip's own. Later servers send no probes,
/// so the measured window holds the clip's request alone; each must prefill
/// the first's prompt instead.
fn drive(url: &str, clip: &Clip<'_>, timeout: Duration) -> anyhow::Result<()> {
    let http = HttpClient::loopback("pipette", timeout)?;
    clip.recipe
        .check_served_template(|| server::chat_template(&http, url))?;
    // Before the clip's request, so the peak's window holds no tokenization
    // after it.
    let text = asr::build_text(clip.recipe, clip.text_tokens, |part| {
        server::tokenize_message_text(&http, url, part)
    })?;
    let user_text = clip.recipe.user_text(&text);
    let audio =
        base64::engine::general_purpose::STANDARD.encode(wav::tone(clip.duration, clip.format)?);
    // The text-only request that checks the clip sends the same text.
    let chat_with = |user_text: &str, audio: Option<&str>, tokens| {
        server::audio_chat(
            &http,
            url,
            clip.recipe.system,
            user_text,
            clip.recipe.order,
            audio,
            tokens,
        )
    };
    let chat = |audio: Option<&str>, tokens| chat_with(&user_text, audio, tokens);
    let prompt_of = |generation: anyhow::Result<pipette_ops::generation::Generation>,
                     what: &str| {
        anyhow::Ok(
            generation?
                .counts
                .with_context(|| format!("the {what} reply carried no token counts"))?
                .prompt_tokens,
        )
    };
    // The peak is the workload's only if the request decoded all its tokens:
    // the reply must stop at the limit with that many, as a throughput rep
    // must. Its times are not read, so nothing stricter is asked of it.
    let generation = chat(Some(&audio), clip.decode_tokens)?;
    measurement::decoded_in_full(&generation, clip.decode_tokens)?;
    let prompt = prompt_of(Ok(generation), "clip's")?;
    match clip.confirmed_prompt.get() {
        None => {
            let text_only = Cell::new(None);
            // `false`: only an unpinned cell calibrates, so a count here sized
            // the context.
            asr::audio_tokens(clip.calibrated, false, prompt, || {
                let tokens = prompt_of(chat(None, 1), "text-only")?;
                text_only.set(Some(tokens));
                Ok(tokens)
            })?;
            // The cell's text is exactly its tokens.
            let with_text = text_only.get().context("the text-only request ran")?;
            asr::confirm_served_text(clip.recipe, clip.text_tokens, with_text, |user_text| {
                prompt_of(chat_with(user_text, None, 1), "text-free")
            })?;
            clip.confirmed_prompt.set(Some(prompt));
        }
        Some(confirmed) => anyhow::ensure!(
            prompt == confirmed,
            "the server prefilled {prompt} tokens where the warm-up server, which checked the \
             prompt, prefilled {confirmed}: the peak would not be the checked prompt's"
        ),
    }
    Ok(())
}

/// The cell's server, every field named: a flag plan-types adds fails to
/// compile here until this cell renders it or sets it aside. `extra` is the
/// recipe's server options.
fn launch(
    flags: &AsrMaxMemoryLlamacppCliStockToolsGgufVision,
    extra: Vec<String>,
    program: &ResolvedLlamaServer,
    model: &ResolvedGgufVisionModel,
    client_config: Option<&LlamacppCliStockToolsExtraRuntimeConfiguration>,
) -> ServerCommandBuilder {
    let AsrMaxMemoryLlamacppCliStockToolsGgufVision {
        threads,
        number_gpu_layers,
        mmap,
        load_mode,
        flash_attention,
        ctx_size,
        no_cache,
        envs,
        raw,
    } = flags.clone();
    ServerCommandBuilder::new(
        program.clone(),
        &model.backbone,
        LoggingMarker::AsrMaxMemory,
    )
    .mmproj(&model.mmproj)
    .threads(threads)
    .number_gpu_layers(number_gpu_layers)
    .mmap(mmap)
    .load_mode(load_mode)
    .flash_attention(flash_attention)
    .raw(raw)
    .ctx_size(ctx_size)
    .no_cache(no_cache)
    .extra(extra)
    .envs(envs)
    .client_config(client_config.cloned())
}

/// The cell's calibration helper, every field named: a flag plan-types adds
/// fails to compile here until the helper takes it or sets it aside. The
/// threads, load knobs, flash attention, context and cache change how the
/// server serves, not the count or where the model loads.
fn calibration(
    flags: &AsrMaxMemoryLlamacppCliStockToolsGgufVision,
    program: &ResolvedLlamaServer,
    model: &ResolvedGgufVisionModel,
    client_config: Option<&LlamacppCliStockToolsExtraRuntimeConfiguration>,
) -> MtmdCommandBuilder {
    let AsrMaxMemoryLlamacppCliStockToolsGgufVision {
        threads: _,
        number_gpu_layers,
        mmap: _,
        load_mode: _,
        flash_attention: _,
        ctx_size: _,
        no_cache: _,
        envs,
        raw,
    } = flags.clone();
    MtmdCommandBuilder::new(program.clone(), &model.backbone, &model.mmproj)
        .number_gpu_layers(number_gpu_layers)
        .raw(raw)
        .envs(envs)
        .client_config(client_config.cloned())
}

#[cfg(test)]
asr_launch_tests!(AsrMaxMemoryLlamacppCliStockToolsGgufVision);
