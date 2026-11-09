//! The reading is the time to the first chunk of a streamed one-token request,
//! on this client's clock (`docs/ovms.md`). A batched entry keeps
//! `parameter_batch_size` of these in flight, each timed from its own send.

use std::num::NonZeroUsize;
use std::path::Path;

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
    BenchmarkResultData, OvmsExtraRuntimeConfiguration, PositiveSeconds,
    PrefillThroughputBenchmark, PrefillThroughputOvmsOpenvino, RunResponse, RuntimeFlags,
};
use pipette_subprocess::start_process;

use crate::completions;
use crate::models::ResolvedOpenvinoModel;
use crate::runtimes::ResolvedOvmsInstall;
use crate::server;

#[expect(
    clippy::too_many_arguments,
    reason = "the request's parts, and the seams its entry takes"
)]
pub(super) fn run(
    benchmark: &PrefillThroughputBenchmark,
    runtime_flags: PrefillThroughputOvmsOpenvino,
    install: &ResolvedOvmsInstall,
    model: &ResolvedOpenvinoModel,
    compile_cache: &Path,
    http_timeout: Option<PositiveSeconds>,
    readiness_gate: ReadinessGate,
    observer: MeasurementObserver<'_>,
    client_config: Option<&OvmsExtraRuntimeConfiguration>,
) -> anyhow::Result<RunResponse> {
    let flags = RuntimeFlags::PrefillThroughputOvmsOpenvino(runtime_flags);
    let batch_size = benchmark.parameter_batch_size;
    let http = completions::timing_client(http_timeout, batch_size)?;
    let sizing_port = pipette_subprocess::pick_server_port()?;
    let prompt = server::size_prompt(
        server::command(
            install,
            model,
            &flags,
            batch_size,
            compile_cache,
            sizing_port,
            client_config,
        )?,
        &server::url(sizing_port),
        &http,
        benchmark
            .parameter_prefill_tokens
            .into_inner()
            .unsigned_abs(),
        observer,
    )?;

    readiness_gate()?;
    let port = pipette_subprocess::pick_server_port()?;
    let url = server::url(port);
    let mut ovms = start_process(
        server::PROGRAM,
        server::command(
            install,
            model,
            &flags,
            batch_size,
            compile_cache,
            port,
            client_config,
        )?,
        server::STOP_GRACE,
    )?;
    let sampler = Sampler::start(ovms.pid(), SupportedWatch::OBSERVE)?;
    ovms.wait_ready(server::READY_TIMEOUT, server::ready(&url)?)?;
    let result = measure(benchmark, readiness_gate, observer, &http, &url, &prompt);
    let (memory, _) = sampler.stop();
    let result_data = result.map_err(|err| err.context(ovms.stop_with_tail()))?;
    Ok(RunResponseBuilder::new(result_data)
        .process(&mut ovms)
        .memory(memory)
        .runtime_flags(flags)
        .build())
}

fn measure(
    benchmark: &PrefillThroughputBenchmark,
    readiness_gate: ReadinessGate,
    observer: MeasurementObserver<'_>,
    http: &HttpClient,
    base_url: &str,
    prompt: &str,
) -> anyhow::Result<BenchmarkResultData> {
    completions::check_opening_chunk(http, base_url, prompt)?;
    let url = completions::url(base_url);
    let request = TimingRequest::text(Some(server::MODEL_NAME), prompt, 1);
    let in_flight = benchmark.parameter_batch_size.in_flight()?;
    measurement::prefill(
        benchmark,
        NonZeroUsize::try_from(measurement::REPS)?,
        readiness_gate,
        observer,
        MeasurementWarmup::Once,
        (),
        |()| {
            let readings = measurement::in_flight(in_flight, |_| {
                let generation = pipette_completions::stream(http, &url, &request)?;
                PrefillMeasurementPerPromptReading::from_generation(&generation)
            })?;
            Ok(PrefillMeasurementReading::PerPrompt(readings))
        },
    )
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use httpmock::MockServer;

    use pipette_plan_types::{BatchSize, BenchmarkId, NonNegativeCount};

    use super::*;

    /// A one-token stream the way OVMS sends it: an empty opening chunk, the
    /// finish chunk, then the usage trailer.
    const ONE_TOKEN_STREAM: &str = "data: {\"choices\":[{\"index\":0,\"text\":\"\",\"finish_reason\":null}]}\n\n\
data: {\"choices\":[{\"index\":0,\"text\":\"\",\"finish_reason\":\"length\"}]}\n\n\
data: {\"choices\":[],\"usage\":{\"prompt_tokens\":512,\"completion_tokens\":1}}\n\n\
data: [DONE]\n\n";

    /// One opening-chunk check, then the warm-up and each rep send one request
    /// per slot.
    #[rstest::rstest]
    #[case::reference(1)]
    #[case::batched(2)]
    fn every_request_of_a_batch_is_read(#[case] batch_size: u32) -> anyhow::Result<()> {
        let server = MockServer::start();
        let completions = server.mock(|when, then| {
            when.method("POST").path("/v3/completions");
            // Stands in for the prefill: the opening-chunk check needs the
            // first chunk to trail the send by more than a scheduler stall.
            then.status(200)
                .header("content-type", "text/event-stream")
                .delay(Duration::from_millis(100))
                .body(ONE_TOKEN_STREAM);
        });
        let benchmark = PrefillThroughputBenchmark {
            benchmark_id: BenchmarkId::try_new(format!("prefill_throughput_512_bs{batch_size}"))?,
            parameter_prefill_tokens: NonNegativeCount::try_new(512)?,
            parameter_batch_size: BatchSize::try_new(batch_size)?,
        };
        let observer = MeasurementObserver::QUIET;
        let result_data = measure(
            &benchmark,
            &|| Ok(()),
            observer,
            &HttpClient::loopback("pipette", Duration::from_secs(5))?,
            &server.base_url(),
            "prompt",
        )?;
        assert!(
            matches!(
                result_data,
                BenchmarkResultData::PrefillThroughput { prefill_time_ms, .. }
                    if prefill_time_ms > 0.0
            ),
            "{result_data:?}"
        );
        let slots = usize::try_from(batch_size)?;
        assert_eq!(completions.calls(), 1 + (measurement::REPS + 1) * slots);
        Ok(())
    }
}
