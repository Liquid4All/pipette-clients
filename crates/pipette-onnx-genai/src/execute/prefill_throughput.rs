//! The reading is the time to the first token of a streamed one-token
//! `/v1/completions`, on the client's clock.

use std::num::NonZeroUsize;
use std::time::Duration;

use pipette_completions::TimingRequest;
use pipette_http::HttpClient;
use pipette_memprobe::{Sampler, SupportedWatch};
use pipette_ops::measurement::{
    self, MeasurementObserver, MeasurementWarmup, PrefillMeasurementPerPromptReading,
    PrefillMeasurementReading,
};
use pipette_ops::readiness::ReadinessGate;
use pipette_ops::run_response::RunResponseBuilder;
use pipette_plan_types::{
    BenchmarkResultData, PrefillThroughputBenchmark, RunResponse,
    UvOnnxGenaiExtraRuntimeConfiguration,
};
use pipette_subprocess::start_process;

use crate::completions;
use crate::models::ResolvedOnnxGenaiModel;
use crate::response;
use crate::runtimes::ResolvedOnnxGenaiInstall;
use crate::server;

pub(super) fn run(
    benchmark: &PrefillThroughputBenchmark,
    install: &ResolvedOnnxGenaiInstall,
    model: &ResolvedOnnxGenaiModel,
    timeout: Duration,
    readiness_gate: ReadinessGate,
    observer: MeasurementObserver<'_>,
    client_config: Option<&UvOnnxGenaiExtraRuntimeConfiguration>,
) -> anyhow::Result<RunResponse> {
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
    benchmark: &PrefillThroughputBenchmark,
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
    let request = TimingRequest::text(Some(completions::MODEL), &prompt, 1);
    measurement::prefill(
        benchmark,
        NonZeroUsize::try_from(measurement::REPS)?,
        readiness_gate,
        observer,
        MeasurementWarmup::Once,
        (),
        |()| {
            let generation = pipette_completions::stream(&http, &url, &request)?;
            let reading = PrefillMeasurementPerPromptReading::from_generation(&generation)?;
            Ok(PrefillMeasurementReading::PerPrompt(vec![reading]))
        },
    )
}
