//! End-to-end latency over `mlx_lm.server`, through
//! [`measurement::end_to_end()`].
//!
//! The reading is one streamed `/v1/completions` on the client's clock, from
//! the send to its finish chunk. EOS is biased away so the full `max_tokens`
//! comes out.

use std::num::NonZeroUsize;

use serde::Serialize;

use pipette_http::HttpClient;
use pipette_memprobe::{Sampler, SupportedWatch};
use pipette_ops::measurement::{
    self, EndToEndMeasurementPerPromptReading, EndToEndMeasurementReading, MeasurementObserver,
    MeasurementWarmup,
};
use pipette_ops::readiness::ReadinessGate;
use pipette_ops::run_response::RunResponseBuilder;
use pipette_plan_types::{
    BenchmarkResultData, EndToEndLatencyBenchmark, MlxMacosPipetteExtraRuntimeConfiguration,
    RunResponse,
};
use pipette_subprocess::start_process;

use crate::models::ResolvedMlxModel;
use crate::runtimes::ResolvedMlxPython;
use crate::{completions, server, tokenizer};

pub(super) fn run(
    benchmark: &EndToEndLatencyBenchmark,
    venv_python: &ResolvedMlxPython,
    model_dir: &ResolvedMlxModel,
    readiness_gate: ReadinessGate,
    observer: MeasurementObserver<'_>,
    client_config: Option<&MlxMacosPipetteExtraRuntimeConfiguration>,
) -> anyhow::Result<RunResponse> {
    let (prompt, eos_ids) = tokenizer::prompt(
        venv_python,
        model_dir,
        benchmark
            .parameter_prefill_tokens
            .into_inner()
            .unsigned_abs(),
        server::TEXT_TRUSTS_REMOTE_CODE,
    )?;
    let request = completions::completion_request(
        &prompt,
        benchmark
            .parameter_decode_tokens
            .into_inner()
            .unsigned_abs(),
        &eos_ids,
    );
    let http = HttpClient::loopback("pipette", completions::TIMING_TIMEOUT)?;

    readiness_gate()?;
    let port = pipette_subprocess::pick_server_port()?;
    let url = server::url(port);
    let completions_url = completions::url(&url);
    let mut mlx_server = start_process(
        server::TEXT_SERVER_MODULE,
        server::text_command(venv_python, model_dir, port, client_config)?,
        server::STOP_GRACE,
    )?;
    let sampler = Sampler::start(mlx_server.pid(), SupportedWatch::OBSERVE)?;
    mlx_server.wait_ready(server::READY_TIMEOUT, server::ready(&url)?)?;
    let result = measure(
        benchmark,
        readiness_gate,
        observer,
        &http,
        &completions_url,
        &request,
    );
    let (memory, _) = sampler.stop();
    let result_data = result.map_err(|err| err.context(mlx_server.stop_with_tail()))?;
    Ok(RunResponseBuilder::new(result_data)
        .process(&mut mlx_server)
        .memory(memory)
        .build())
}

fn measure(
    benchmark: &EndToEndLatencyBenchmark,
    readiness_gate: ReadinessGate,
    observer: MeasurementObserver<'_>,
    http: &HttpClient,
    url: &str,
    request: &impl Serialize,
) -> anyhow::Result<BenchmarkResultData> {
    measurement::end_to_end(
        benchmark,
        NonZeroUsize::try_from(measurement::REPS)?,
        readiness_gate,
        observer,
        MeasurementWarmup::Once,
        (),
        |()| {
            let generation = pipette_completions::stream(http, url, request)?;
            let reading = EndToEndMeasurementPerPromptReading::from_generation(&generation)?;
            Ok(EndToEndMeasurementReading::PerPrompt(vec![reading]))
        },
    )
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use httpmock::MockServer;

    use pipette_completions::test_support::stream_at_limit;
    use pipette_plan_types::{BatchSize, BenchmarkId, NonNegativeCount, PositiveCount};

    use super::*;

    #[test]
    fn every_request_is_streamed() -> anyhow::Result<()> {
        let server = MockServer::start();
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
        let observer = MeasurementObserver {
            on_before_rep_start: &|| Ok(()),
            on_after_rep_end: &|| Ok(()),
        };
        let result = measure(
            &benchmark,
            &|| Ok(()),
            observer,
            &HttpClient::loopback("pipette", Duration::from_secs(5))?,
            &completions::url(&server.base_url()),
            &completions::completion_request("prompt", 16, &[]),
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
