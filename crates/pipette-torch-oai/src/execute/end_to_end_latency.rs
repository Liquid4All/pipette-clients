//! The reading is one streamed `/v1/completions` on the client's clock, from
//! the send to its finish chunk. A batched entry keeps `parameter_batch_size`
//! of these in flight, each timed from its own send.

use std::num::NonZeroUsize;

use pipette_completions::TimingRequest;
use pipette_http::HttpClient;
use pipette_ops::measurement::{
    self, EndToEndMeasurementPerPromptReading, EndToEndMeasurementReading, MeasurementObserver,
    MeasurementWarmup,
};
use pipette_ops::readiness::ReadinessGate;
use pipette_ops::run_response::RunResponseBuilder;
use pipette_plan_types::{BenchmarkResultData, EndToEndLatencyBenchmark, PositiveSeconds};

use crate::{completions, response, server::ServerState};

pub(super) fn run(
    benchmark: &EndToEndLatencyBenchmark,
    http_timeout: Option<PositiveSeconds>,
    model: &str,
    state: &ServerState,
    readiness_gate: ReadinessGate,
    observer: MeasurementObserver<'_>,
) -> anyhow::Result<RunResponseBuilder> {
    let batch_size = benchmark.parameter_batch_size;
    readiness_gate()?;
    let timeout = measurement::http_timeout_for(http_timeout, batch_size.as_u32());
    let base_url = state.base_url();
    let http = HttpClient::loopback("pipette", timeout)?;
    let prompt_text = completions::build_prompt_text(
        &http,
        &base_url,
        model,
        benchmark
            .parameter_prefill_tokens
            .into_inner()
            .unsigned_abs(),
    )?;
    let request = TimingRequest::text(
        Some(model),
        &prompt_text,
        benchmark
            .parameter_decode_tokens
            .into_inner()
            .unsigned_abs(),
    );
    let result_data = measure(
        benchmark,
        readiness_gate,
        observer,
        &http,
        &completions::url(&base_url),
        &request,
    )?;
    Ok(response::timing_response(
        result_data,
        state,
        format!(
            "model={model} max_tokens={} prefill_target_tokens={}",
            benchmark.parameter_decode_tokens, benchmark.parameter_prefill_tokens
        ),
        batch_size,
    ))
}

fn measure(
    benchmark: &EndToEndLatencyBenchmark,
    readiness_gate: ReadinessGate,
    observer: MeasurementObserver<'_>,
    http: &HttpClient,
    url: &str,
    request: &TimingRequest<'_>,
) -> anyhow::Result<BenchmarkResultData> {
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
                let generation = pipette_completions::stream(http, url, request)?;
                EndToEndMeasurementPerPromptReading::from_generation(&generation)
            })?;
            Ok(EndToEndMeasurementReading::PerPrompt(readings))
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

    /// The warm-up and every rep stream one request per slot.
    #[rstest::rstest]
    #[case::reference(1)]
    #[case::batched(2)]
    fn every_request_is_streamed(#[case] batch_size: u32) -> anyhow::Result<()> {
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
            benchmark_id: BenchmarkId::try_new(format!("end_to_end_latency_64_16_bs{batch_size}"))?,
            parameter_prefill_tokens: NonNegativeCount::try_new(64)?,
            parameter_decode_tokens: PositiveCount::try_new(16)?,
            parameter_batch_size: BatchSize::try_new(batch_size)?,
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
            &TimingRequest::text(Some("m"), "prompt", 16),
        )?;
        assert!(
            matches!(
                result,
                BenchmarkResultData::EndToEndLatency { total_time_ms, .. } if total_time_ms > 0.0
            ),
            "{result:?}"
        );
        let slots = usize::try_from(batch_size)?;
        assert_eq!(completions.calls(), (measurement::REPS + 1) * slots);
        Ok(())
    }
}
