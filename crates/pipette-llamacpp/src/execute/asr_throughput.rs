//! `asr_throughput` on llama-server: prefill one audio clip and decode after it
//! (`docs/methodology/asr-throughput.md`). The audio counterpart of
//! `vl_throughput`.

use std::num::NonZeroUsize;
use std::time::Duration;

use anyhow::Context;
use base64::Engine;

use pipette_http::HttpClient;
use pipette_memprobe::{Sampler, SupportedWatch};
use pipette_ops::generation::Clock;
use pipette_ops::measurement::{
    self, AsrThroughputMeasurementReading, MeasurementObserver, MeasurementWarmup,
};
use pipette_ops::readiness::ReadinessGate;
use pipette_ops::run_response::RunResponseBuilder;
use pipette_ops::wav;
use pipette_plan_types::{
    AsrAudioTokensSource, AsrClientExtras, AsrContextSizing, AsrThroughputBenchmark,
    AsrThroughputLlamacppCliStockToolsGgufVision, BenchmarkFlagRef,
    EngineExtraRuntimeConfiguration, LlamacppCliStockToolsExtraRuntimeConfiguration, PositiveCount,
    RunResponse, RuntimeFlags,
};
use pipette_subprocess::start_process;

use crate::asr::{self, CtxInputs, Recipe};
use crate::models::ResolvedGgufVisionModel;
use crate::mtmd::MtmdCommandBuilder;
use crate::runtimes::{self, ResolvedLlamaServer};
use crate::server::{self, LoggingMarker, ServerCommandBuilder};

#[expect(
    clippy::too_many_arguments,
    reason = "the request's parts, and the seams its entry takes"
)]
pub fn run(
    benchmark: &AsrThroughputBenchmark,
    runtime_flags: AsrThroughputLlamacppCliStockToolsGgufVision,
    benchmark_flags: &BenchmarkFlagRef,
    program: &ResolvedLlamaServer,
    model: &ResolvedGgufVisionModel,
    readiness_gate: ReadinessGate,
    observer: MeasurementObserver<'_>,
    client_config: Option<&LlamacppCliStockToolsExtraRuntimeConfiguration>,
) -> anyhow::Result<RunResponse> {
    let duration = benchmark.parameter_audio_duration_ms;
    let format = wav::ClipFormat::of(
        benchmark.parameter_audio_sample_rate_hz,
        benchmark.parameter_audio_channels,
    );
    let decode_tokens = benchmark.parameter_decode_tokens.as_u32();
    let text_tokens = benchmark.parameter_text_tokens.into_inner().unsigned_abs();
    // Every refusal before the calibration and the server, which load the
    // model: the recipe's server flags are rendered here for that reason.
    asr::refuse_raw_parallel(&runtime_flags.raw, "asr_throughput")?;
    let recipe = Recipe::for_mmproj(&model.mmproj)?;
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
    // Calibrated once, before the server starts: a second model load beside a
    // resident server is an OOM risk.
    let calibrated = asr::calibrate(
        &recipe,
        duration,
        format,
        &calibration(&runtime_flags, program, model, client_config),
    );
    let ctx_size = CtxInputs {
        audio_bound: asr::fallback_audio_tokens(duration.as_u32()),
        text_tokens,
        decode_tokens,
        text_chars: recipe.text_chars(),
    }
    .ctx_size(calibrated.as_ref().map(|c| c.tokens));
    let ctx_pinned = runtime_flags.ctx_size.is_some();
    // Final: a count the server does not confirm fails the cell rather than
    // relabel it (`asr::audio_tokens`).
    let sizing = asr::context_sizing(ctx_pinned, calibrated.is_some());
    // The count sized the context, so a failed run may be its fault.
    let sized_by = calibrated.as_ref().filter(|_| !ctx_pinned);
    let runtime_flags = AsrThroughputLlamacppCliStockToolsGgufVision {
        load_mode: runtimes::get_load_mode(runtime_flags.mmap, runtime_flags.load_mode),
        ctx_size: runtime_flags.ctx_size.or(Some(ctx_size)),
        ..runtime_flags
    };
    let launch = launch(&runtime_flags, recipe_args, program, model, client_config);
    let timeout = measurement::http_timeout_for(benchmark_flags.http_timeout_seconds, 1);
    let http = HttpClient::loopback("pipette", timeout)?;
    let port = pipette_subprocess::pick_server_port()?;
    let url = server::url(port);
    let ready = server::health(&url, timeout)?;
    let command = launch.build(port)?;

    let mut llama_server = start_process("llama-server", command, Duration::ZERO)?;
    let sampler = Sampler::start(llama_server.pid(), SupportedWatch::OBSERVE)?;
    llama_server.wait_ready(timeout, ready)?;
    let source = std::cell::Cell::new(None);
    let worked = (|| {
        recipe.check_served_template(|| server::chat_template(&http, &url))?;
        let text = asr::build_text(&recipe, text_tokens, |part| {
            server::tokenize_message_text(&http, &url, part)
        })?;
        let user_text = recipe.user_text(&text);
        let clip = base64::engine::general_purpose::STANDARD.encode(wav::tone(duration, format)?);
        // The text-only request that measures the audio sends the same text.
        let chat_with = |user_text: &str, clip: Option<&str>, decode_tokens| {
            server::audio_chat(
                &http,
                &url,
                recipe.system,
                user_text,
                recipe.order,
                clip,
                decode_tokens,
            )
        };
        let chat = |clip: Option<&str>, decode_tokens| chat_with(&user_text, clip, decode_tokens);
        let prompt_of = |generation: anyhow::Result<pipette_ops::generation::Generation>| {
            anyhow::Ok(
                generation?
                    .counts
                    .context("the reply carried no token counts")?
                    .prompt_tokens,
            )
        };
        // Every rep sends the same prompt with `cache_prompt: false`, so the
        // entry can require the same token counts from each.
        measurement::asr_throughput(
            benchmark,
            NonZeroUsize::try_from(measurement::REPS)?,
            readiness_gate,
            observer,
            MeasurementWarmup::Once,
            (),
            |()| {
                let generation = chat(Some(&clip), decode_tokens)?;
                Ok(AsrThroughputMeasurementReading::from_generation(
                    &generation,
                    Clock::Server,
                )?)
            },
            |prompt_tokens| {
                let text_only = std::cell::Cell::new(None);
                let (audio_tokens, from) = asr::audio_tokens(
                    calibrated.as_ref(),
                    ctx_pinned,
                    prompt_tokens.into_inner().unsigned_abs(),
                    || {
                        let tokens = prompt_of(chat(None, 1))?;
                        text_only.set(Some(tokens));
                        Ok(tokens)
                    },
                )?;
                // The cell's text is exactly its tokens.
                let with_text = text_only.get().context("the text-only request ran")?;
                asr::confirm_served_text(&recipe, text_tokens, with_text, |user_text| {
                    prompt_of(chat_with(user_text, None, 1))
                })?;
                source.set(Some(from));
                PositiveCount::try_new(i32::try_from(audio_tokens)?)
                    .context("the audio token count")
            },
        )
    })();
    let (memory, _) = sampler.stop();
    let data = worked
        .map_err(|err| match sized_by {
            Some(calibrated) => calibrated.drop_on_failure(err),
            None => err,
        })
        .map_err(|err| asr::with_server_tail(err, llama_server.stop_with_tail()))?;
    // All three on every row: a measured run always settled its count.
    let source = source
        .get()
        .context("the run settled no audio token count")?;
    let extras = [
        (
            AsrContextSizing::KEY.to_owned(),
            serde_json::Value::from(sizing.to_string()),
        ),
        (
            AsrAudioTokensSource::KEY.to_owned(),
            serde_json::Value::from(source.to_string()),
        ),
        (
            AsrClientExtras::BEYOND_MODEL_AUDIO_LIMIT_KEY.to_owned(),
            serde_json::Value::from(beyond_limit),
        ),
    ];
    Ok(RunResponseBuilder::new(data)
        .process(&mut llama_server)
        .memory(memory)
        .client_extras(extras)
        .runtime_flags(RuntimeFlags::AsrThroughputLlamacppCliStockToolsGgufVision(
            runtime_flags,
        ))
        .build())
}

/// The cell's server, every field named: a flag plan-types adds fails to
/// compile here until this cell renders it or sets it aside. `extra` is the
/// recipe's server options.
fn launch(
    flags: &AsrThroughputLlamacppCliStockToolsGgufVision,
    extra: Vec<String>,
    program: &ResolvedLlamaServer,
    model: &ResolvedGgufVisionModel,
    client_config: Option<&LlamacppCliStockToolsExtraRuntimeConfiguration>,
) -> ServerCommandBuilder {
    let AsrThroughputLlamacppCliStockToolsGgufVision {
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
        LoggingMarker::AsrThroughput,
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
    flags: &AsrThroughputLlamacppCliStockToolsGgufVision,
    program: &ResolvedLlamaServer,
    model: &ResolvedGgufVisionModel,
    client_config: Option<&LlamacppCliStockToolsExtraRuntimeConfiguration>,
) -> MtmdCommandBuilder {
    let AsrThroughputLlamacppCliStockToolsGgufVision {
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
asr_launch_tests!(AsrThroughputLlamacppCliStockToolsGgufVision);
