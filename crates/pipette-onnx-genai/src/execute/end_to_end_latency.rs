//! The reading is one streamed `/v1/completions` on the client's clock, from
//! the send to its finish chunk.

use std::num::NonZeroUsize;
use std::time::Duration;

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
    BenchmarkResultData, EndToEndLatencyBenchmark, RunResponse,
    UvOnnxGenaiExtraRuntimeConfiguration,
};
use pipette_subprocess::start_process;

use crate::completions;
use crate::models::ResolvedOnnxGenaiModel;
use crate::response;
use crate::runtimes::ResolvedOnnxGenaiInstall;
use crate::server;

pub(super) fn run(
    benchmark: &EndToEndLatencyBenchmark,
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
    benchmark: &EndToEndLatencyBenchmark,
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
    measurement::end_to_end(
        benchmark,
        NonZeroUsize::try_from(measurement::REPS)?,
        readiness_gate,
        observer,
        MeasurementWarmup::Once,
        (),
        |()| {
            let generation = pipette_completions::stream(&http, &url, &request)?;
            let reading = EndToEndMeasurementPerPromptReading::from_generation(&generation)?;
            Ok(EndToEndMeasurementReading::PerPrompt(vec![reading]))
        },
    )
}

#[cfg(test)]
mod tests {
    use httpmock::MockServer;

    use pipette_completions::test_support::stream_at_limit;
    use pipette_plan_types::{BatchSize, BenchmarkId, NonNegativeCount, PositiveCount};

    use super::*;

    #[test]
    fn every_request_is_streamed() -> anyhow::Result<()> {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method("POST").path("/tokenize");
            then.status(200)
                .json_body(serde_json::json!({ "count": 64 }));
        });
        let completions = server.mock(|when, then| {
            when.method("POST")
                .path("/v1/completions")
                .json_body_includes(r#"{"stream":true}"#);
            then.status(200)
                .header("content-type", "text/event-stream")
                .body(stream_at_limit(64, 16));
        });
        let benchmark = EndToEndLatencyBenchmark {
            benchmark_id: BenchmarkId::try_new("end_to_end_latency_64_16".to_owned())?,
            parameter_prefill_tokens: NonNegativeCount::try_new(64)?,
            parameter_decode_tokens: PositiveCount::try_new(16)?,
            parameter_batch_size: BatchSize::one(),
        };
        let observer = MeasurementObserver::QUIET;
        let result = measure(
            &benchmark,
            &server.base_url(),
            Duration::from_secs(5),
            &|| Ok(()),
            observer,
        )?;
        assert!(
            matches!(
                result,
                BenchmarkResultData::EndToEndLatency { total_time_ms, .. } if total_time_ms > 0.0
            ),
            "{result:?}"
        );
        assert_eq!(completions.calls(), measurement::REPS + 1);
        Ok(())
    }
}
