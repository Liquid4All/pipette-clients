//! `vl_throughput` on TensorRT Edge-LLM: `pipette-llamacpp`'s cell, request for
//! request (`docs/methodology/vl-throughput.md`), against a server whose engines
//! were built beforehand. The image tokens are the server's own count
//! (`timings.image_n`), the figure llama.cpp's calibration helper prints.

use std::num::NonZeroUsize;
use std::time::Duration;

use base64::Engine;

use pipette_completions::llama_server::{self, ChatReply};
use pipette_http::HttpClient;
use pipette_memprobe::{Sampler, SupportedWatch};
use pipette_ops::generation::Clock;
use pipette_ops::measurement::{
    self, MeasurementObserver, MeasurementWarmup, VlThroughputMeasurementReading,
};
use pipette_ops::readiness::ReadinessGate;
use pipette_ops::run_response::RunResponseBuilder;
use pipette_plan_types::{
    BenchmarkFlagRef, PositiveCount, RunResponse, RuntimeFlags, VlThroughputBenchmark,
};
use pipette_subprocess::start_process;

use crate::engines;
use crate::server::{self, Launch, READY_TIMEOUT};

pub fn run(
    benchmark: &VlThroughputBenchmark,
    benchmark_flags: &BenchmarkFlagRef,
    launch: &Launch,
    readiness_gate: ReadinessGate,
    observer: MeasurementObserver<'_>,
) -> anyhow::Result<RunResponse> {
    // Before the build, which can take minutes.
    measurement::require_decode_steps(benchmark.parameter_decode_tokens)?;
    let width = benchmark.parameter_image_width.into_inner().unsigned_abs();
    let height = benchmark.parameter_image_height.into_inner().unsigned_abs();
    let text_tokens = benchmark.parameter_text_tokens.into_inner().unsigned_abs();
    let decode_tokens = benchmark
        .parameter_decode_tokens
        .into_inner()
        .unsigned_abs();
    let num_images = benchmark.parameter_num_images.into_inner().unsigned_abs();
    // The build is not an attempt: a cold one takes minutes, before any server.
    observer.outside(|| engines::build(launch))?;

    let timeout = measurement::http_timeout_for(benchmark_flags.http_timeout_seconds, 1);
    let http = HttpClient::loopback("pipette", timeout)?;
    let port = pipette_subprocess::pick_server_port()?;
    let url = llama_server::url(port);
    let ready = llama_server::health(&url, timeout)?;
    let mut server = start_process("trt-edgellm-server", launch.serve(port), Duration::ZERO)?;
    let sampler = Sampler::start(server.pid(), SupportedWatch::OBSERVE)?;
    server.wait_ready(READY_TIMEOUT, ready)?;
    let worked = (|| {
        let stack = server::engine_stack(&http, &url)?;
        let text = llama_server::message_filler(&http, &url, text_tokens)?;
        let image = base64::engine::general_purpose::STANDARD
            .encode(pipette_ops::png::gray(width, height)?);
        // Every rep sends the same prompt and the server keeps no prompt or
        // vision cache, so the entry can require the same counts from each.
        let result_data = measurement::vl_throughput(
            benchmark,
            NonZeroUsize::try_from(measurement::REPS)?,
            readiness_gate,
            observer,
            MeasurementWarmup::Once,
            (),
            |()| {
                let ChatReply {
                    generation,
                    image_tokens,
                } = llama_server::chat_reply(
                    &http,
                    &url,
                    &text,
                    &image,
                    decode_tokens,
                    num_images,
                )?;
                let image_tokens = image_tokens
                    .filter(|&tokens| tokens > 0)
                    .map(PositiveCount::try_from)
                    .transpose()
                    .map_err(anyhow::Error::from)?;
                Ok(VlThroughputMeasurementReading::from_generation(
                    &generation,
                    image_tokens,
                    Clock::Server,
                )?)
            },
        )?;
        Ok((result_data, stack))
    })();
    let (memory, _) = sampler.stop();
    let (data, extras) =
        worked.map_err(|err| llama_server::with_server_tail(err, server.stop_with_tail()))?;
    Ok(RunResponseBuilder::new(data)
        .process(&mut server)
        .memory(memory)
        .client_extras(extras)
        .runtime_flags(RuntimeFlags::VlThroughputTrtEdgellmGgufVision(
            launch.flags(),
        ))
        .build())
}
