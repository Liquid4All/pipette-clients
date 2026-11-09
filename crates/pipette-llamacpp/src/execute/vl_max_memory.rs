//! Peak memory for a vision-language workload, driven through `llama-server`
//! under the flavor's peak sampler ([`crate::memory::peak`]). A server of its
//! own on the same command serves the request first, its reading dropped, so a
//! build that compiles into a cache (a GPU driver's kernels) is measured on a
//! filled one; the observation covers that server too.

use std::path::Path;
use std::time::{Duration, Instant};

use base64::Engine;

use pipette_http::HttpClient;
use pipette_memprobe::{Peak, Sampler, SupportedWatch};
use pipette_ops::measurement::{
    self, MeasurementObserver, MemoryMeasurementWarmup, VlMaxMemoryMeasurementReading,
};
use pipette_ops::run_response::RunResponseBuilder;
use pipette_plan_types::{
    BenchmarkFlagRef, LlamaCppFlavor, LlamacppCliStockToolsExtraRuntimeConfiguration,
    MemoryObservation, RunResponse, RuntimeFlags, VlMaxMemoryBenchmark,
    VlMaxMemoryLlamacppCliStockToolsGgufVision,
};
use pipette_subprocess::{Process, start_process};

use crate::memory;
use crate::models::ResolvedGgufVisionModel;
use crate::mtmd::{self, CtxInputs, MtmdCommandBuilder};
use crate::runtimes::{self, ResolvedLlamaServer};
use crate::server::{self, LoggingMarker, ServerCommandBuilder};
#[cfg(any(target_os = "linux", target_os = "android"))]
use crate::swap_aware;

#[expect(
    clippy::too_many_arguments,
    reason = "the request's parts, and the seams its entry takes"
)]
pub fn run(
    benchmark: &VlMaxMemoryBenchmark,
    runtime_flags: VlMaxMemoryLlamacppCliStockToolsGgufVision,
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
    let inputs = inputs(benchmark)?;
    // Derived only for a cell that pins no ctx_size: behind a pinned one the
    // calibration's cache miss would be a full model load (up to 600s) paid for
    // a number that is never used.
    let ctx_pinned = runtime_flags.ctx_size.is_some();
    let calibrated_per_image = (!ctx_pinned)
        .then(|| {
            observer.outside(|| {
                mtmd::calibrate(
                    &inputs,
                    &calibration(&runtime_flags, program, model, client_config),
                )
            })
        })
        .flatten();
    // No floor, unlike vl_throughput: the fallback already over-estimates, so
    // the peak carries a small bounded KV over-allocation where the helper is
    // absent.
    let ctx_size = inputs.ctx_size(calibrated_per_image, &runtime_flags.raw, 0);
    let runtime_flags = VlMaxMemoryLlamacppCliStockToolsGgufVision {
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
    // One budget per server from spawn through the workload: the model load,
    // then the vision encode and prefill.
    let deadline = memory::attempt_timeout(benchmark_flags.attempt_timeout_seconds);
    let mut served = None;
    #[cfg(any(target_os = "linux", target_os = "android"))]
    let mut extras = serde_json::Map::new();
    let result = measurement::vl_max_memory(
        benchmark,
        observer,
        MemoryMeasurementWarmup::Once,
        &mut served,
        |served| {
            let peak = serve(served, &launch, watch, &inputs, timeout, deadline)?;
            #[cfg(any(target_os = "linux", target_os = "android"))]
            let peak = {
                extras = swap_aware::extras(&peak.footprint);
                peak.counting_swap()
            };
            Ok(VlMaxMemoryMeasurementReading {
                max_host_bytes: peak.host_bytes,
                max_gpu_bytes: peak.gpu_bytes,
                max_npu_bytes: peak.npu_bytes,
            })
        },
    );
    let data = result.map_err(|err| match &mut served {
        Some((llama_server, _)) => server::with_server_tail(err, llama_server.stop_with_tail()),
        None => err,
    })?;
    let response = served
        .iter_mut()
        .fold(
            RunResponseBuilder::new(data),
            |response, (llama_server, memory)| response.process(llama_server).memory(*memory),
        )
        .runtime_flags(RuntimeFlags::VlMaxMemoryLlamacppCliStockToolsGgufVision(
            runtime_flags,
        ));
    #[cfg(any(target_os = "linux", target_os = "android"))]
    let response = response.client_extras(extras);
    Ok(response.build())
}

/// A cell without an output length decodes one token, so older cells keep
/// measuring what they did.
fn inputs(benchmark: &VlMaxMemoryBenchmark) -> anyhow::Result<CtxInputs> {
    CtxInputs::new(
        benchmark.parameter_image_width.into_inner().unsigned_abs(),
        benchmark.parameter_image_height.into_inner().unsigned_abs(),
        benchmark.parameter_text_tokens.into_inner().unsigned_abs(),
        benchmark
            .parameter_decode_tokens
            .map_or(1, |tokens| tokens.into_inner().unsigned_abs()),
        benchmark.parameter_num_images.into_inner().unsigned_abs(),
    )
}

/// One server from its launch through the workload, within `deadline`, left in
/// `served`. One an earlier attempt left there stops first; the observation
/// covers both.
fn serve(
    served: &mut Option<(Process, MemoryObservation)>,
    launch: &ServerCommandBuilder,
    watch: SupportedWatch,
    inputs: &CtxInputs,
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
    let worked = drive(&url, inputs, timeout.min(left)).map_err(|err| {
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

/// The peak covers the vision encode, the image-token KV cache, the prefill's
/// buffers and the decode. The first token comes off the prefill, so a
/// one-token reply runs no decode step.
fn drive(url: &str, inputs: &CtxInputs, timeout: Duration) -> anyhow::Result<()> {
    let http = HttpClient::loopback("pipette", timeout)?;
    let text = server::message_filler(&http, url, inputs.text_tokens)?;
    let image = base64::engine::general_purpose::STANDARD
        .encode(pipette_ops::png::gray(inputs.width, inputs.height)?);
    let generation = server::chat(
        &http,
        url,
        &text,
        &image,
        inputs.decode_tokens,
        inputs.num_images,
    )?;
    measurement::decoded_in_full(&generation, inputs.decode_tokens)
}

/// The cell's server, every field named: a flag plan-types adds fails to compile
/// here until this cell renders it or sets it aside.
/// The slot count is the server's unless `no_cache` pins one.
fn launch(
    flags: &VlMaxMemoryLlamacppCliStockToolsGgufVision,
    program: &ResolvedLlamaServer,
    model: &Path,
    mmproj: &Path,
    client_config: Option<&LlamacppCliStockToolsExtraRuntimeConfiguration>,
) -> ServerCommandBuilder {
    let VlMaxMemoryLlamacppCliStockToolsGgufVision {
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
    ServerCommandBuilder::new(program.clone(), model, LoggingMarker::VlMaxMemory)
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
    flags: &VlMaxMemoryLlamacppCliStockToolsGgufVision,
    program: &ResolvedLlamaServer,
    model: &ResolvedGgufVisionModel,
    client_config: Option<&LlamacppCliStockToolsExtraRuntimeConfiguration>,
) -> MtmdCommandBuilder {
    let VlMaxMemoryLlamacppCliStockToolsGgufVision {
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
    use rstest::rstest;

    use pipette_plan_types::{
        BenchmarkId, LlamacppFlashAttention, LlamacppLoadMode, NonNegativeCount, PositiveCount,
    };

    use super::*;
    use crate::runtimes::test_llama_server;

    #[rstest]
    #[case::absent(None, 1)]
    #[case::decode256(Some(256), 256)]
    fn the_output_length_sizes_the_reply_and_the_context(
        #[case] decode: Option<i32>,
        #[case] expected: u32,
    ) -> anyhow::Result<()> {
        let benchmark = |decode: Option<i32>| -> anyhow::Result<VlMaxMemoryBenchmark> {
            Ok(VlMaxMemoryBenchmark {
                benchmark_id: BenchmarkId::try_new("vl_max_memory_224x224_text0_img1")?,
                parameter_image_width: PositiveCount::try_new(224)?,
                parameter_image_height: PositiveCount::try_new(224)?,
                parameter_text_tokens: NonNegativeCount::try_new(0)?,
                parameter_decode_tokens: decode.map(PositiveCount::try_new).transpose()?,
                parameter_num_images: NonNegativeCount::try_new(1)?,
            })
        };
        let one = inputs(&benchmark(None)?)?;
        let cell = inputs(&benchmark(decode)?)?;
        assert_eq!(cell.decode_tokens, expected);
        assert_eq!(
            cell.ctx_size(None, &[], 0) - one.ctx_size(None, &[], 0),
            expected - 1
        );
        Ok(())
    }

    fn server_args(
        flags: &VlMaxMemoryLlamacppCliStockToolsGgufVision,
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
    fn runtime_flags() -> VlMaxMemoryLlamacppCliStockToolsGgufVision {
        VlMaxMemoryLlamacppCliStockToolsGgufVision {
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
            server_args(&runtime_flags())?.join(" "),
            "--ctx-checkpoints 0 --ctx-size 4096 --flash-attn on --load-mode dio --n-gpu-layers 7 --no-cache-prompt --no-warmup --parallel 1 --poll 0 --threads 3"
        );
        Ok(())
    }
}
