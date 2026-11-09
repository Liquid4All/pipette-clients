//! The reading is one streamed request on this client's clock, from the send
//! to its finish chunk. A batched entry keeps `parameter_batch_size` of these
//! in flight, each timed from its own send.

use std::num::NonZeroUsize;
use std::path::Path;

use pipette_completions::TimingRequest;
use pipette_http::HttpClient;
use pipette_memprobe::{Sampler, SupportedWatch};
use pipette_ops::measurement::{
    self, EndToEndMeasurementPerPromptReading, EndToEndMeasurementReading, MeasurementObserver,
    MeasurementWarmup,
};
use pipette_ops::readiness::ReadinessGate;
use pipette_ops::run_response::RunResponseBuilder;
use pipette_plan_types::{
    BenchmarkResultData, EndToEndLatencyBenchmark, EndToEndLatencyOvmsOpenvino,
    OvmsExtraRuntimeConfiguration, PositiveSeconds, RunResponse, RuntimeFlags,
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
    benchmark: &EndToEndLatencyBenchmark,
    runtime_flags: EndToEndLatencyOvmsOpenvino,
    install: &ResolvedOvmsInstall,
    model: &ResolvedOpenvinoModel,
    compile_cache: &Path,
    http_timeout: Option<PositiveSeconds>,
    readiness_gate: ReadinessGate,
    observer: MeasurementObserver<'_>,
    client_config: Option<&OvmsExtraRuntimeConfiguration>,
) -> anyhow::Result<RunResponse> {
    let flags = RuntimeFlags::EndToEndLatencyOvmsOpenvino(runtime_flags);
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
    benchmark: &EndToEndLatencyBenchmark,
    readiness_gate: ReadinessGate,
    observer: MeasurementObserver<'_>,
    http: &HttpClient,
    base_url: &str,
    prompt: &str,
) -> anyhow::Result<BenchmarkResultData> {
    let url = completions::url(base_url);
    let request = TimingRequest::text(
        Some(server::MODEL_NAME),
        prompt,
        benchmark
            .parameter_decode_tokens
            .into_inner()
            .unsigned_abs(),
    );
    let in_flight = benchmark.parameter_batch_size.in_flight()?;
    measurement::end_to_end(
        benchmark,
        NonZeroUsize::try_from(measurement::REPS)?,
        readiness_gate,
        observer,
        MeasurementWarmup::Once,
        (),
        |()| {
            let readings = measurement::in_flight(in_flight, |_| {
                let generation = pipette_completions::stream(http, &url, &request)?;
                EndToEndMeasurementPerPromptReading::from_generation(&generation)
            })?;
            Ok(EndToEndMeasurementReading::PerPrompt(readings))
        },
    )
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use anyhow::Context;
    use httpmock::MockServer;

    use pipette_completions::test_support::stream_at_limit;
    use pipette_plan_types::{BatchSize, BenchmarkId, NonNegativeCount, PositiveCount};

    use super::*;

    /// The warm-up and each rep send one request per slot; a request that ran
    /// another shape fails the cell at the warm-up.
    #[rstest::rstest]
    #[case::reference(1, 256, None)]
    #[case::batched(2, 256, None)]
    #[case::short(2, 255, Some("decode_tokens 255, expected 256"))]
    fn every_request_of_a_batch_is_read(
        #[case] batch_size: u32,
        #[case] completion_tokens: u32,
        #[case] refused: Option<&str>,
    ) -> anyhow::Result<()> {
        let server = MockServer::start();
        let completions = server.mock(|when, then| {
            when.method("POST")
                .path("/v3/completions")
                .json_body_includes(r#"{"stream":true}"#);
            then.status(200)
                .header("content-type", "text/event-stream")
                .body(stream_at_limit(512, completion_tokens));
        });
        let benchmark = EndToEndLatencyBenchmark {
            benchmark_id: BenchmarkId::try_new(format!(
                "end_to_end_latency_512_256_bs{batch_size}"
            ))?,
            parameter_prefill_tokens: NonNegativeCount::try_new(512)?,
            parameter_decode_tokens: PositiveCount::try_new(256)?,
            parameter_batch_size: BatchSize::try_new(batch_size)?,
        };
        let observer = MeasurementObserver::QUIET;
        let result = measure(
            &benchmark,
            &|| Ok(()),
            observer,
            &HttpClient::loopback("pipette", Duration::from_secs(5))?,
            &server.base_url(),
            "prompt",
        );
        let slots = usize::try_from(batch_size)?;
        match refused {
            None => {
                let result_data = result?;
                assert!(
                    matches!(
                        result_data,
                        BenchmarkResultData::EndToEndLatency { total_time_ms, .. }
                            if total_time_ms > 0.0
                    ),
                    "{result_data:?}"
                );
                assert_eq!(completions.calls(), (measurement::REPS + 1) * slots);
            }
            Some(reason) => {
                let err = result.err().context("a short generation was measured")?;
                assert!(format!("{err:#}").contains(reason), "{err:#}");
                assert_eq!(completions.calls(), slots, "failed at the warm-up");
            }
        }
        Ok(())
    }
}
