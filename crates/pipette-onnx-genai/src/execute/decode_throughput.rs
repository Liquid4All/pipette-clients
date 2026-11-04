//! The reading is the interval from the first token to the last of a streamed
//! `/v1/completions`, on the client's clock.

use std::num::NonZeroUsize;
use std::time::Duration;

use pipette_completions::TimingRequest;
use pipette_http::HttpClient;
use pipette_memprobe::{Sampler, SupportedWatch};
use pipette_ops::measurement::{
    self, DecodeMeasurementPerPromptReading, DecodeMeasurementReading, MeasurementObserver,
    MeasurementWarmup,
};
use pipette_ops::readiness::ReadinessGate;
use pipette_ops::run_response::RunResponseBuilder;
use pipette_plan_types::{
    BenchmarkResultData, DecodeThroughputBenchmark, RunResponse,
    UvOnnxGenaiExtraRuntimeConfiguration,
};
use pipette_subprocess::start_process;

use crate::completions;
use crate::models::ResolvedOnnxGenaiModel;
use crate::response;
use crate::runtimes::ResolvedOnnxGenaiInstall;
use crate::server;

pub(super) fn run(
    benchmark: &DecodeThroughputBenchmark,
    install: &ResolvedOnnxGenaiInstall,
    model: &ResolvedOnnxGenaiModel,
    timeout: Duration,
    readiness_gate: ReadinessGate,
    observer: MeasurementObserver<'_>,
    client_config: Option<&UvOnnxGenaiExtraRuntimeConfiguration>,
) -> anyhow::Result<RunResponse> {
    // Before the model loads: the entry refuses a single token only once the
    // warm-up has run.
    anyhow::ensure!(
        benchmark.parameter_decode_tokens.into_inner() >= 2,
        "parameter_decode_tokens must be at least 2: the first token is the prefill's, so a \
         decode rate starts at the second; got {}",
        benchmark.parameter_decode_tokens
    );
    readiness_gate()?;
    let script = server::script()?;
    let port = pipette_subprocess::pick_server_port()?;
    let url = server::url(port);
    let mut onnx_server = start_process(
        server::SCRIPT_FILENAME,
        server::command(install, model, script.path(), port, client_config)?,
        server::STOP_GRACE,
    )?;
    let sampler = Sampler::start(onnx_server.pid(), SupportedWatch::OBSERVE)?;
    onnx_server.wait_ready(server::READY_TIMEOUT, server::ready(&url)?)?;
    let result = measure(benchmark, &url, timeout, readiness_gate, observer);
    let (memory, _) = sampler.stop();
    let result_data = result.map_err(|err| err.context(onnx_server.stop_with_tail()))?;
    Ok(RunResponseBuilder::new(result_data)
        .process(&mut onnx_server)
        .command(response::command(&install.python))
        .memory(memory)
        .build())
}

fn measure(
    benchmark: &DecodeThroughputBenchmark,
    base_url: &str,
    timeout: Duration,
    readiness_gate: ReadinessGate,
    observer: MeasurementObserver<'_>,
) -> anyhow::Result<BenchmarkResultData> {
    let http = HttpClient::loopback("pipette", timeout)?;
    let prompt = pipette_completions::build_prompt_text(
        &http,
        base_url,
        completions::MODEL,
        benchmark
            .parameter_prefill_tokens
            .into_inner()
            .unsigned_abs(),
    )?;
    let url = completions::url(base_url);
    let request = TimingRequest::text(
        Some(completions::MODEL),
        &prompt,
        benchmark
            .parameter_decode_tokens
            .into_inner()
            .unsigned_abs(),
    );
    measurement::decode(
        benchmark,
        NonZeroUsize::try_from(measurement::REPS)?,
        readiness_gate,
        observer,
        MeasurementWarmup::Once,
        (),
        |()| {
            let generation = pipette_completions::stream(&http, &url, &request)?;
            let reading = DecodeMeasurementPerPromptReading::from_generation(&generation)?;
            Ok(DecodeMeasurementReading::PerPrompt(vec![reading]))
        },
    )
}
