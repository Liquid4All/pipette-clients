//! `vl_throughput` on GenAI's `VLMPipeline`: `pipette-llamacpp`'s cell, request
//! for request (`docs/methodology/vl-throughput.md`), against the server's VL
//! routes, which report GenAI's timings as llama-server's counters. The image
//! tokens are this client's count from the model's image rule.

use std::num::NonZeroUsize;
use std::path::Path;
use std::time::Duration;

use base64::Engine;

use pipette_completions::llama_server;
use pipette_http::HttpClient;
use pipette_memprobe::{Sampler, SupportedWatch};
use pipette_ops::generation::Clock;
use pipette_ops::measurement::{
    self, MeasurementObserver, MeasurementWarmup, VlThroughputMeasurementReading,
};
use pipette_ops::readiness::ReadinessGate;
use pipette_ops::run_response::RunResponseBuilder;
use pipette_plan_types::{
    BenchmarkResultData, PositiveCount, RunResponse, RuntimeFlags,
    UvOpenvinoExtraRuntimeConfiguration, VlThroughputBenchmark, VlUvOpenvinoOpenvino,
};
use pipette_subprocess::{OnStop, start_process};

use crate::models::ResolvedOpenvinoVlModel;
use crate::response;
use crate::runtimes::ResolvedOpenvinoPython;
use crate::server;

#[expect(
    clippy::too_many_arguments,
    reason = "the request's parts, and the seams its entry takes"
)]
pub(super) fn run(
    benchmark: &VlThroughputBenchmark,
    runtime_flags: VlUvOpenvinoOpenvino,
    python: &ResolvedOpenvinoPython,
    model: &ResolvedOpenvinoVlModel,
    compile_cache: &Path,
    timeout: Duration,
    readiness_gate: ReadinessGate,
    observer: MeasurementObserver<'_>,
    client_config: Option<&UvOpenvinoExtraRuntimeConfiguration>,
) -> anyhow::Result<RunResponse> {
    // Before the gate and the compile, which takes minutes on an NPU.
    measurement::require_decode_steps(benchmark.parameter_decode_tokens)?;
    let flags = RuntimeFlags::VlThroughputUvOpenvinoOpenvino(runtime_flags);
    readiness_gate()?;
    let script = server::script()?;
    let port = pipette_subprocess::pick_server_port()?;
    let url = server::url(port);
    let mut openvino_server = start_process(
        server::SCRIPT_FILENAME,
        server::command(
            python,
            model,
            &flags,
            compile_cache,
            script.path(),
            port,
            client_config,
        )?,
        OnStop::SIGKILL,
    )?;
    let sampler = Sampler::start(openvino_server.pid(), SupportedWatch::OBSERVE)?;
    let ready = server::ready(&url)?;
    // Off the clock: readiness waits out the model's compile.
    observer.outside(|| openvino_server.wait_ready(server::READY_TIMEOUT, ready))?;
    let result = measure(benchmark, model, &url, timeout, readiness_gate, observer);
    let (memory, _) = sampler.stop();
    let result_data = result.map_err(|err| err.context(openvino_server.stop_with_tail()))?;
    Ok(RunResponseBuilder::new(result_data)
        .process(&mut openvino_server)
        .command(response::command(python))
        .memory(memory)
        .runtime_flags(flags)
        .build())
}

fn measure(
    benchmark: &VlThroughputBenchmark,
    model: &ResolvedOpenvinoVlModel,
    url: &str,
    timeout: Duration,
    readiness_gate: ReadinessGate,
    observer: MeasurementObserver<'_>,
) -> anyhow::Result<BenchmarkResultData> {
    let width = benchmark.parameter_image_width.into_inner().unsigned_abs();
    let height = benchmark.parameter_image_height.into_inner().unsigned_abs();
    // A 0 would drop the image the cell measures; llama.cpp reads it as 1 too.
    let num_images = benchmark
        .parameter_num_images
        .into_inner()
        .unsigned_abs()
        .max(1);
    let text_tokens = benchmark.parameter_text_tokens.into_inner().unsigned_abs();
    let decode_tokens = benchmark
        .parameter_decode_tokens
        .into_inner()
        .unsigned_abs();
    let http = HttpClient::loopback("pipette", timeout)?;
    let text = llama_server::message_filler(&http, url, text_tokens)?;
    let image =
        base64::engine::general_purpose::STANDARD.encode(pipette_ops::png::gray(width, height)?);
    let image_tokens = model
        .image_tokens(width, height, num_images)
        .map(PositiveCount::try_from)
        .transpose()?;
    // GenAI reports no image count, so its prompt count vouches for this
    // client's rule on the build the plan installed.
    let expected_prompt = model.prompt_tokens(text_tokens, width, height, num_images);
    // The server keeps no prompt or image cache, so the entry can require the
    // same counts from each rep.
    measurement::vl_throughput(
        benchmark,
        NonZeroUsize::try_from(measurement::REPS)?,
        readiness_gate,
        observer,
        MeasurementWarmup::Once,
        (),
        |()| {
            let generation =
                llama_server::chat(&http, url, &text, &image, decode_tokens, num_images)?;
            let reading = VlThroughputMeasurementReading::from_generation(
                &generation,
                image_tokens,
                Clock::Server,
            )?;
            if let Some(expected) = expected_prompt
                && reading.prompt_tokens.into_inner().unsigned_abs() != expected
            {
                return Err(anyhow::anyhow!(
                    "GenAI prompted with {} tokens where this client's LFM3-VL rule counts \
                     {expected}, so the image_tokens it would record are wrong for this build",
                    reading.prompt_tokens.into_inner()
                )
                .into());
            }
            Ok(reading)
        },
    )
}
