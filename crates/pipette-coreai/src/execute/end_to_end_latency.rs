use std::{num::NonZeroUsize, path::PathBuf};

use serde::{Deserialize, Serialize};

use pipette_ops::measurement::{self as shared, MeasurementObserver, MeasurementWarmup};
use pipette_ops::readiness::ReadinessGate;
use pipette_ops::run_response::RunResponseBuilder;
use pipette_plan_types::EndToEndLatencyBenchmark;
use pipette_plan_types::RunRequest;
use pipette_plan_types::RunResponse;

use crate::{server, throughput_http};

const ENDPOINT: &str = "/end_to_end_latency";

#[derive(Debug, Serialize)]
struct EndToEndLatencyRequest {
    prompt_tokens: u32,
    decode_tokens: u32,
}

#[derive(Debug, Deserialize)]
struct EndToEndLatencyResponse {
    total_ms: f64,
    prompt_tokens: u32,
    completion_tokens: u32,
}

pub(super) fn run(
    req: &RunRequest,
    benchmark: &EndToEndLatencyBenchmark,
    sidecar: PathBuf,
    readiness_gate: ReadinessGate,
    observer: &MeasurementObserver<'_>,
) -> anyhow::Result<RunResponse> {
    let prefill_tokens = benchmark
        .parameter_prefill_tokens
        .into_inner()
        .unsigned_abs();
    let decode_tokens = benchmark
        .parameter_decode_tokens
        .into_inner()
        .unsigned_abs();

    readiness_gate()?;
    let mut server = observer.outside(|| server::start_server(req, sidecar))?;

    let result = shared::end_to_end(
        benchmark,
        NonZeroUsize::try_from(shared::REPS)?,
        readiness_gate,
        *observer,
        MeasurementWarmup::Once,
        &server,
        |server| {
            throughput_http::prepare(server)?;
            let started = std::time::Instant::now();
            let response = run_latency_request(server, prefill_tokens, decode_tokens)?;
            let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
            let reading = reading(response, elapsed_ms)?;
            Ok(shared::EndToEndMeasurementReading::PerPrompt(vec![reading]))
        },
    )?;
    Ok(RunResponseBuilder::new(result)
        .process(&mut server.process)
        .build())
}

fn run_latency_request(
    server: &server::ServerHandle,
    prefill_tokens: u32,
    decode_tokens: u32,
) -> anyhow::Result<EndToEndLatencyResponse> {
    throughput_http::post_json(
        server,
        ENDPOINT,
        &EndToEndLatencyRequest {
            prompt_tokens: prefill_tokens,
            decode_tokens,
        },
    )
}

fn reading(
    response: EndToEndLatencyResponse,
    elapsed_ms: f64,
) -> anyhow::Result<shared::EndToEndMeasurementPerPromptReading> {
    shared::positive_finite("total_ms", response.total_ms)?;
    Ok(shared::EndToEndMeasurementPerPromptReading::from_request(
        response.prompt_tokens.try_into()?,
        response.completion_tokens.try_into()?,
        elapsed_ms,
    )?)
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    fn valid_response() -> EndToEndLatencyResponse {
        EndToEndLatencyResponse {
            total_ms: 10.0,
            prompt_tokens: 4,
            completion_tokens: 3,
        }
    }

    #[test]
    fn uses_the_client_elapsed_time() -> anyhow::Result<()> {
        let r = reading(valid_response(), 25.0)?;
        assert_eq!(r.total_time_ms.into_inner(), 25.0);
        Ok(())
    }

    #[rstest]
    #[case::ok(valid_response(), true)]
    #[case::zero_total_ms(
        EndToEndLatencyResponse {
            total_ms: 0.0,
            ..valid_response()
        },
        false
    )]
    #[case::zero_completion_tokens(
        EndToEndLatencyResponse {
            completion_tokens: 0,
            ..valid_response()
        },
        false
    )]
    fn validates_the_response(#[case] response: EndToEndLatencyResponse, #[case] ok: bool) {
        assert_eq!(reading(response, 25.0).is_ok(), ok);
    }
}
