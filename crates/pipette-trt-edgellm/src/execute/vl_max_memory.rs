//! `vl_max_memory` on TensorRT Edge-LLM: `pipette-llamacpp`'s cell
//! (`docs/methodology/vl-max-memory.md`) under the GPU peak watch, which on a
//! Jetson reads the server's `VmHWM` alone: its CUDA buffers live in that
//! process's RSS (`docs/methodology/peak-memory-linux.md`). The engines are
//! built first, outside the sampler; a server of its own then serves the
//! request with its reading dropped, as llama.cpp's warm-up does.

use std::time::{Duration, Instant};

use base64::Engine;

use pipette_completions::llama_server;
use pipette_http::HttpClient;
use pipette_memprobe::{Peak, Sampler, SupportedWatch, Watch};
use pipette_ops::measurement::{self, MemoryMeasurementWarmup, VlMaxMemoryMeasurementReading};
use pipette_ops::run_response::RunResponseBuilder;
use pipette_plan_types::{
    BenchmarkFlagRef, MemoryObservation, PositiveSeconds, RunResponse, RuntimeFlags,
    VlMaxMemoryBenchmark,
};
use pipette_subprocess::{Process, start_process};

use crate::engines;
use crate::server::{self, Launch};

/// As llama.cpp's peak-memory cells, when the cell sets no
/// `bench_timeout_seconds`: one server from spawn through its workload.
const BENCH_TIMEOUT: Duration = Duration::from_secs(600);

/// The request one server serves.
struct Inputs {
    width: u32,
    height: u32,
    text_tokens: u32,
    decode_tokens: u32,
    num_images: u32,
}

pub fn run(
    benchmark: &VlMaxMemoryBenchmark,
    benchmark_flags: &BenchmarkFlagRef,
    launch: &Launch,
) -> anyhow::Result<RunResponse> {
    // Before the build: a host that cannot read the peak builds nothing.
    let watch = Watch::PeakWithGpu.supported()?;
    let inputs = Inputs {
        width: benchmark.parameter_image_width.into_inner().unsigned_abs(),
        height: benchmark.parameter_image_height.into_inner().unsigned_abs(),
        text_tokens: benchmark.parameter_text_tokens.into_inner().unsigned_abs(),
        // A cell without an output length decodes one token, as llama.cpp's.
        decode_tokens: benchmark
            .parameter_decode_tokens
            .map_or(1, |tokens| tokens.into_inner().unsigned_abs()),
        num_images: benchmark.parameter_num_images.into_inner().unsigned_abs(),
    };
    engines::build(launch)?;

    let timeout = measurement::http_timeout_for(benchmark_flags.http_timeout_seconds, 1);
    let deadline = benchmark_flags
        .bench_timeout_seconds
        .map_or(BENCH_TIMEOUT, |secs: PositiveSeconds| {
            Duration::from_secs(secs.into_inner())
        });
    let mut served = None;
    let mut extras = serde_json::Map::new();
    let result = measurement::vl_max_memory(
        benchmark,
        MemoryMeasurementWarmup::Once,
        &mut served,
        |served| {
            let (peak, stack) = serve(served, launch, watch, &inputs, timeout, deadline)?;
            extras = stack;
            #[cfg(any(target_os = "linux", target_os = "android"))]
            let peak = {
                extras.extend(swap_aware_extras(&peak.footprint));
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
        Some((server, _)) => llama_server::with_server_tail(err, server.stop_with_tail()),
        None => err,
    })?;
    let response = served
        .iter_mut()
        .fold(
            RunResponseBuilder::new(data),
            |response, (server, memory)| response.process(server).memory(*memory),
        )
        .runtime_flags(RuntimeFlags::VlMaxMemoryTrtEdgellmGgufVision(
            launch.flags(),
        ))
        .client_extras(extras);
    Ok(response.build())
}

/// One server from its launch through the workload, within `deadline`, left in
/// `served`, with the engine stack it reports. One an earlier attempt left
/// there stops first; the observation covers both.
fn serve(
    served: &mut Option<(Process, MemoryObservation)>,
    launch: &Launch,
    watch: SupportedWatch,
    inputs: &Inputs,
    timeout: Duration,
    deadline: Duration,
) -> anyhow::Result<(Peak, serde_json::Map<String, serde_json::Value>)> {
    let earlier = match served.take() {
        Some((mut server, memory)) => {
            server.stop();
            memory
        }
        None => MemoryObservation::default(),
    };
    let port = pipette_subprocess::pick_server_port()?;
    let url = llama_server::url(port);
    let ready = llama_server::health(&url, timeout)?;
    let started = Instant::now();
    let mut server = start_process("trt-edgellm-server", launch.serve(port), Duration::ZERO)?;
    let sampler = Sampler::start(server.pid(), watch)?;
    server.wait_ready(deadline, ready)?;
    let left = deadline.saturating_sub(started.elapsed());
    let worked = drive(&url, inputs, timeout.min(left)).map_err(|err| {
        if started.elapsed() < deadline {
            return err;
        }
        err.context(format!("the workload exceeded {deadline:?}"))
    });
    let (observed, peak) = sampler.stop();
    // Before the checks, so the cell quotes it on any failure from here.
    *served = Some((server, earlier.merge_max(observed)));
    let stack = worked?;
    Ok((peak?, stack))
}

/// The peak covers the vision encode, the prefill and the decode. The first
/// token comes off the prefill, so a one-token reply runs no decode step.
/// The server's engine stack comes back.
fn drive(
    url: &str,
    inputs: &Inputs,
    timeout: Duration,
) -> anyhow::Result<serde_json::Map<String, serde_json::Value>> {
    let http = HttpClient::loopback("pipette", timeout)?;
    let stack = server::engine_stack(&http, url)?;
    let text = llama_server::message_filler(&http, url, inputs.text_tokens)?;
    let image = base64::engine::general_purpose::STANDARD
        .encode(pipette_ops::png::gray(inputs.width, inputs.height)?);
    let generation = llama_server::chat(
        &http,
        url,
        &text,
        &image,
        inputs.decode_tokens,
        inputs.num_images,
    )?;
    measurement::decoded_in_full(&generation, inputs.decode_tokens)?;
    Ok(stack)
}

/// The `/proc` terms behind the swap-aware peak, under llama.cpp's keys.
#[cfg(any(target_os = "linux", target_os = "android"))]
fn swap_aware_extras(
    footprint: &pipette_memprobe::Footprint,
) -> serde_json::Map<String, serde_json::Value> {
    serde_json::Map::from_iter([
        ("peak_rss_kib".to_owned(), footprint.peak_rss_kib.into()),
        (
            "peak_committed_kib".to_owned(),
            footprint.peak_committed_kib.into(),
        ),
        ("max_swap_kib".to_owned(), footprint.max_swap_kib.into()),
        ("major_faults".to_owned(), footprint.major_faults.into()),
    ])
}
