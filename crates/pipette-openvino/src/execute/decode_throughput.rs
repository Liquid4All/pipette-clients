//! The reading is the interval from the first token to the last of a streamed
//! `/v1/completions`, on the client's clock.

use std::num::NonZeroUsize;
use std::path::Path;
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
    BenchmarkResultData, DecodeThroughputBenchmark, DecodeThroughputUvOpenvinoOpenvino,
    RunResponse, RuntimeFlags, UvOpenvinoExtraRuntimeConfiguration,
};
use pipette_subprocess::start_process;

use crate::completions;
use crate::models::ResolvedOpenvinoModel;
use crate::response;
use crate::runtimes::ResolvedOpenvinoPython;
use crate::server;

#[expect(
    clippy::too_many_arguments,
    reason = "the request's parts, and the seams its entry takes"
)]
pub(super) fn run(
    benchmark: &DecodeThroughputBenchmark,
    runtime_flags: DecodeThroughputUvOpenvinoOpenvino,
    python: &ResolvedOpenvinoPython,
    model: &ResolvedOpenvinoModel,
    compile_cache: &Path,
    timeout: Duration,
    readiness_gate: ReadinessGate,
    observer: MeasurementObserver<'_>,
    client_config: Option<&UvOpenvinoExtraRuntimeConfiguration>,
) -> anyhow::Result<RunResponse> {
    // Before the pipeline compiles: the entry refuses a single token only once
    // the warm-up has run.
    anyhow::ensure!(
        benchmark.parameter_decode_tokens.into_inner() >= 2,
        "parameter_decode_tokens must be at least 2: the first token is the prefill's, so a \
         decode rate starts at the second; got {}",
        benchmark.parameter_decode_tokens
    );
    let flags = RuntimeFlags::DecodeThroughputUvOpenvinoOpenvino(runtime_flags);
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
        server::STOP_GRACE,
    )?;
    let sampler = Sampler::start(openvino_server.pid(), SupportedWatch::OBSERVE)?;
    let ready = server::ready(&url)?;
    // Off the clock: readiness waits out the model's compile.
    observer.outside(|| openvino_server.wait_ready(server::READY_TIMEOUT, ready))?;
    let result = measure(benchmark, &url, timeout, readiness_gate, observer);
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
