use std::num::NonZeroUsize;
use std::path::Path;
use std::time::Duration;

use base64::Engine;

use pipette_http::HttpClient;
use pipette_memprobe::{Sampler, SupportedWatch};
use pipette_ops::generation::Clock;
use pipette_ops::measurement::{
    self, MeasurementObserver, MeasurementWarmup, VlThroughputMeasurementReading,
};
use pipette_ops::readiness::ReadinessGate;
use pipette_ops::run_response::RunResponseBuilder;
use pipette_plan_types::{
    BenchmarkFlagRef, BenchmarkResultData, LlamacppCliStockToolsExtraRuntimeConfiguration,
    RunResponse, RuntimeFlags, VlThroughputBenchmark, VlThroughputLlamacppCliStockToolsGgufVision,
};
use pipette_subprocess::start_process;

use crate::models::ResolvedGgufVisionModel;
use crate::mtmd::{self, CtxInputs, MtmdCommandBuilder};
use crate::runtimes::{self, ResolvedLlamaServer};
use crate::server::{self, LoggingMarker, ServerCommandBuilder};

#[expect(
    clippy::too_many_arguments,
    reason = "the request's parts, and the seams its entry takes"
)]
pub fn run(
    benchmark: &VlThroughputBenchmark,
    runtime_flags: VlThroughputLlamacppCliStockToolsGgufVision,
    benchmark_flags: &BenchmarkFlagRef,
    program: &ResolvedLlamaServer,
    model: &ResolvedGgufVisionModel,
    readiness_gate: ReadinessGate,
    observer: MeasurementObserver<'_>,
    client_config: Option<&LlamacppCliStockToolsExtraRuntimeConfiguration>,
) -> anyhow::Result<RunResponse> {
    // Before the model loads.
    measurement::require_decode_steps(benchmark.parameter_decode_tokens)?;
    let inputs = CtxInputs::new(
        benchmark.parameter_image_width.into_inner().unsigned_abs(),
        benchmark.parameter_image_height.into_inner().unsigned_abs(),
        benchmark.parameter_text_tokens.into_inner().unsigned_abs(),
        benchmark
            .parameter_decode_tokens
            .into_inner()
            .unsigned_abs(),
        benchmark.parameter_num_images.into_inner().unsigned_abs(),
    )?;
    // Calibrated once, before the server starts: a second mtmd model load beside
    // a resident server is an OOM/jetsam risk on the low-RAM devices this targets.
    // The cell's image-token overrides ride along so the helper counts what the
    // server will see.
    let calibration = calibration(&runtime_flags, program, model, client_config);
    let calibrated_per_image = mtmd::calibrate(&inputs, &calibration);
    // The 8192 floor is this benchmark's headroom on top of the shared fallback
    // (vl_max_memory has none).
    let ctx_size = inputs.ctx_size(calibrated_per_image, &runtime_flags.raw, 8192);
    // Several frames are counted together, since some llama.cpp builds merge
    // frames, in a context for the frames alone: the helper's prompt is `hi`.
    // Not after one frame found no count: the same helper would not find one
    // for more frames, and a wedged one would hold the cell for a second
    // deadline.
    let calibrated_image_tokens = if inputs.num_images > 1 {
        calibrated_per_image.and_then(|per_image| {
            let frames = CtxInputs {
                text_tokens: 0,
                decode_tokens: 0,
                ..inputs
            };
            let frames_ctx = frames.ctx_size(Some(per_image), &runtime_flags.raw, 0);
            mtmd::calibrate_frames(&inputs, &calibration.ctx_size(Some(frames_ctx)))
        })
    } else {
        calibrated_per_image
    };
    let runtime_flags = VlThroughputLlamacppCliStockToolsGgufVision {
        load_mode: runtimes::get_load_mode(runtime_flags.mmap, runtime_flags.load_mode),
        ctx_size: runtime_flags.ctx_size.or(Some(ctx_size)),
        ..runtime_flags
    };
    let launch = launch(
        &runtime_flags,
        program,
        &model.backbone,
        &model.mmproj,
        client_config,
    );
    let timeout = measurement::http_timeout_for(benchmark_flags.http_timeout_seconds, 1);
    let http = HttpClient::loopback("pipette", timeout)?;
    let port = pipette_subprocess::pick_server_port()?;
    let url = server::url(port);
    let ready = server::health(&url, timeout)?;
    let command = launch.build(port)?;

    let mut llama_server = start_process("llama-server", command, Duration::ZERO)?;
    let sampler = Sampler::start(llama_server.pid(), SupportedWatch::OBSERVE)?;
    llama_server.wait_ready(timeout, ready)?;
    let worked = (|| {
        let text = server::message_filler(&http, &url, inputs.text_tokens)?;
        let image = base64::engine::general_purpose::STANDARD
            .encode(pipette_ops::png::gray(inputs.width, inputs.height)?);
        let chat = |decode_tokens, num_images| {
            server::chat(&http, &url, &text, &image, decode_tokens, num_images)
        };
        // Every rep sends the same prompt with `cache_prompt: false`, so the
        // entry can require the same token counts from each.
        let mut result_data = measurement::vl_throughput(
            benchmark,
            NonZeroUsize::try_from(measurement::REPS)?,
            readiness_gate,
            observer,
            MeasurementWarmup::Once,
            (),
            |()| {
                let generation = chat(inputs.decode_tokens, inputs.num_images)?;
                // The image tokens are settled after the reps, below.
                Ok(VlThroughputMeasurementReading::from_generation(
                    &generation,
                    None,
                    Clock::Server,
                )?)
            },
        )?;
        let BenchmarkResultData::VlThroughput {
            prompt_tokens,
            image_tokens: settled,
            ..
        } = &mut result_data
        else {
            anyhow::bail!("the VL throughput entry returned another benchmark's result");
        };
        *settled = image_tokens(calibrated_image_tokens, *prompt_tokens);
        Ok(result_data)
    })();
    let (memory, _) = sampler.stop();
    let data =
        worked.map_err(|err| server::with_server_tail(err, llama_server.stop_with_tail()))?;
    Ok(RunResponseBuilder::new(data)
        .process(&mut llama_server)
        .memory(memory)
        .runtime_flags(RuntimeFlags::VlThroughputLlamacppCliStockToolsGgufVision(
            runtime_flags,
        ))
        .build())
}

/// The calibrated image-only share of `prompt_tokens`, or `None` if it is missing or
/// cannot fit in the prompt.
fn image_tokens(calibrated: Option<u32>, prompt_tokens: u32) -> Option<u32> {
    let Some(tokens) = calibrated else {
        tracing::warn!(
            prompt_tokens,
            "no calibrated image_tokens (is llama-mtmd-cli in the runtime?); the row reports \
             none"
        );
        return None;
    };
    if !(1..=prompt_tokens).contains(&tokens) {
        tracing::warn!(
            image_tokens = tokens,
            prompt_tokens,
            "calibrated image_tokens is not within the measured prompt_tokens; stale \
             calibration cache row? the row reports none"
        );
        return None;
    }
    tracing::info!(prompt_tokens, image_tokens = tokens, "token counts");
    Some(tokens)
}

/// The cell's server, every field named: a flag plan-types adds fails to compile
/// here until this cell renders it or sets it aside.
/// The slot count is the server's unless `no_cache` pins one.
fn launch(
    flags: &VlThroughputLlamacppCliStockToolsGgufVision,
    program: &ResolvedLlamaServer,
    model: &Path,
    mmproj: &Path,
    client_config: Option<&LlamacppCliStockToolsExtraRuntimeConfiguration>,
) -> ServerCommandBuilder {
    let VlThroughputLlamacppCliStockToolsGgufVision {
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
    ServerCommandBuilder::new(program.clone(), model, LoggingMarker::VlThroughput)
        .mmproj(mmproj)
        .threads(threads)
        .number_gpu_layers(number_gpu_layers)
        .mmap(mmap)
        .load_mode(load_mode)
        .flash_attention(flash_attention)
        .raw(raw)
        .ctx_size(ctx_size)
        .no_cache(no_cache)
        .envs(envs)
        .client_config(client_config.cloned())
}

/// The cell's calibration helper, every field named: a flag plan-types adds
/// fails to compile here until the helper takes it or sets it aside. The
/// threads, load knobs, flash attention, context and cache change how the
/// server serves, not the count or where the model loads.
fn calibration(
    flags: &VlThroughputLlamacppCliStockToolsGgufVision,
    program: &ResolvedLlamaServer,
    model: &ResolvedGgufVisionModel,
    client_config: Option<&LlamacppCliStockToolsExtraRuntimeConfiguration>,
) -> MtmdCommandBuilder {
    let VlThroughputLlamacppCliStockToolsGgufVision {
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
mod tests {
    use std::time::Duration;

    use pipette_plan_types::{LlamacppFlashAttention, LlamacppLoadMode, PositiveSeconds};

    use super::*;
    use crate::runtimes::test_llama_server;

    fn server_args(
        flags: &VlThroughputLlamacppCliStockToolsGgufVision,
    ) -> Result<Vec<String>, server::ServerArgsError> {
        launch(
            flags,
            &test_llama_server(),
            Path::new("model.gguf"),
            Path::new("mmproj.gguf"),
            None,
        )
        .args()
    }

    /// Every knob set to its own value, so a knob that lands on another's flag shows.
    fn every_knob_set() -> VlThroughputLlamacppCliStockToolsGgufVision {
        VlThroughputLlamacppCliStockToolsGgufVision {
            threads: Some(3),
            number_gpu_layers: Some(7),
            mmap: None,
            load_mode: Some(LlamacppLoadMode::DirectIo),
            flash_attention: Some(LlamacppFlashAttention::On),
            ctx_size: Some(4096),
            no_cache: Some(true),
            envs: Vec::new(),
            raw: vec!["--poll".to_owned(), "0".to_owned()],
        }
    }

    #[test]
    fn the_server_takes_every_knob() -> anyhow::Result<()> {
        assert_eq!(
            server_args(&every_knob_set())?.join(" "),
            "--ctx-checkpoints 0 --ctx-size 4096 --flash-attn on --load-mode dio --n-gpu-layers 7 --no-cache-prompt --no-warmup --parallel 1 --poll 0 --threads 3"
        );
        Ok(())
    }

    /// Every case measured a 300-token prompt.
    #[rstest::rstest]
    #[case::calibrated(Some(64), Some(64))]
    #[case::the_whole_prompt(Some(300), Some(300))]
    #[case::calibration_past_the_prompt(Some(400), None)]
    #[case::calibrated_zero(Some(0), None)]
    #[case::uncalibrated(None, None)]
    fn image_tokens_are_the_calibrated_count_or_none(
        #[case] calibrated: Option<u32>,
        #[case] expected: Option<u32>,
    ) {
        assert_eq!(image_tokens(calibrated, 300), expected);
    }

    /// A single token is the prefill's own, with no decode after it, so the
    /// cell is refused before the gate, the calibration and the server start.
    #[test]
    fn a_single_decode_token_is_refused_before_the_model_loads() -> anyhow::Result<()> {
        use std::cell::Cell;

        use pipette_plan_types::{
            AbsolutePath, BenchmarkId, LlamaCppFlavor, LlamacppCliStockTools,
            LlamacppCliStockToolsSource, NonNegativeCount, PositiveCount, Runtime,
        };

        let tmp = tempfile::tempdir()?;
        // Never run: it only resolves.
        std::fs::write(
            tmp.path()
                .join(format!("llama-server{}", std::env::consts::EXE_SUFFIX)),
            "",
        )?;
        let program = ResolvedLlamaServer::try_from(&Runtime::LlamacppCliStockTools(
            LlamacppCliStockTools {
                source: LlamacppCliStockToolsSource::AbsoluteDir {
                    dir: AbsolutePath::try_new(tmp.path().display().to_string())?,
                },
                flavor: LlamaCppFlavor::MacosArm64,
            },
        ))?;
        let benchmark = VlThroughputBenchmark {
            benchmark_id: BenchmarkId::try_new("vl_throughput_224x224_t0_d1_f1")?,
            parameter_image_width: PositiveCount::try_new(224)?,
            parameter_image_height: PositiveCount::try_new(224)?,
            parameter_text_tokens: NonNegativeCount::try_new(0)?,
            parameter_decode_tokens: PositiveCount::try_new(1)?,
            parameter_num_images: NonNegativeCount::try_new(1)?,
        };
        let runtime_flags = VlThroughputLlamacppCliStockToolsGgufVision {
            threads: None,
            number_gpu_layers: None,
            mmap: None,
            load_mode: None,
            flash_attention: None,
            ctx_size: None,
            no_cache: None,
            envs: Vec::new(),
            raw: Vec::new(),
        };
        let gated = Cell::new(false);
        let Err(err) = run(
            &benchmark,
            runtime_flags,
            &BenchmarkFlagRef::from(
                pipette_plan_types::BenchmarkFlags::VlThroughputLlamacppCliStockToolsGgufVision {
                    http_timeout_seconds: None,
                    readiness: None,
                },
            ),
            &program,
            &ResolvedGgufVisionModel {
                backbone: tmp.path().join("model.gguf"),
                mmproj: tmp.path().join("mmproj.gguf"),
            },
            &|| {
                gated.set(true);
                Ok(())
            },
            MeasurementObserver {
                on_before_rep_start: &|| Ok(()),
                on_after_rep_end: &|| Ok(()),
            },
            None,
        ) else {
            anyhow::bail!("a one-token decode cell was measured");
        };
        assert!(
            matches!(
                err.downcast_ref::<measurement::VlThroughputMeasurementError>(),
                Some(measurement::VlThroughputMeasurementError::TooFewTokens { .. })
            ),
            "{err:#}"
        );
        assert!(!gated.get(), "refused only after the gate");
        Ok(())
    }

    #[test]
    fn http_timeout_flag_propagates_and_defaults() -> anyhow::Result<()> {
        use pipette_plan_types::BenchmarkFlags;

        let pinned = |flags: Option<BenchmarkFlags>| {
            flags
                .map(BenchmarkFlagRef::from)
                .and_then(|flags| flags.http_timeout_seconds)
        };
        let with_flag = pinned(Some(
            BenchmarkFlags::VlThroughputLlamacppCliStockToolsGgufVision {
                http_timeout_seconds: Some(PositiveSeconds::try_new(7)?),
                readiness: None,
            },
        ));
        assert_eq!(
            measurement::http_timeout_for(with_flag, 1),
            Duration::from_secs(7)
        );
        assert_eq!(
            measurement::http_timeout_for(pinned(None), 1),
            Duration::from_secs(measurement::DEFAULT_HTTP_TIMEOUT_SECS)
        );
        Ok(())
    }
}
