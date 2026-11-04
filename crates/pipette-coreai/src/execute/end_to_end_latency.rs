use std::path::PathBuf;

use anyhow::Context;
use serde::{Deserialize, Serialize};

use pipette_ops::measurement::MeasurementObserver;
use pipette_ops::readiness::ReadinessGate;
use pipette_ops::run_response::RunResponseBuilder;
use pipette_plan_types::BenchmarkResultData;
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
    let mut server = server::start_server(req, sidecar)?;

    tracing::info!(
        cell = ENDPOINT,
        prefill_tokens,
        decode_tokens,
        "Core AI end-to-end warm-up"
    );
    throughput_http::prepare(&server)?;
    validate_response(
        &run_latency_request(&server, prefill_tokens, decode_tokens)?,
        prefill_tokens,
        decode_tokens,
    )
    .context("invalid /end_to_end_latency warmup")?;

    let measured = pipette_ops::measurement::run(
        ENDPOINT,
        readiness_gate,
        *observer,
        |_| throughput_http::prepare(&server),
        |_| run_latency_request(&server, prefill_tokens, decode_tokens),
        |idx, rep| {
            validate_response(&rep.value, prefill_tokens, decode_tokens)
                .with_context(|| format!("invalid {ENDPOINT} trial {idx}"))?;
            Ok(rep.elapsed_ms())
        },
    )?;
    let stats = measured.stats();

    Ok(
        RunResponseBuilder::new(BenchmarkResultData::EndToEndLatency {
            total_time_ms: stats.mean_ms,
            total_time_ms_stddev: Some(stats.stddev_ms),
        })
        .process(&mut server.process)
        .build(),
    )
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

fn validate_response(
    response: &EndToEndLatencyResponse,
    expected_prompt_tokens: u32,
    expected_completion_tokens: u32,
) -> anyhow::Result<()> {
    pipette_ops::measurement::positive_finite("total_ms", response.total_ms)?;
    pipette_ops::measurement::expect_tokens(
        "prompt_tokens",
        response.prompt_tokens,
        expected_prompt_tokens,
    )?;
    pipette_ops::measurement::expect_tokens(
        "completion_tokens",
        response.completion_tokens,
        expected_completion_tokens,
    )
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    fn valid_response() -> EndToEndLatencyResponse {
        EndToEndLatencyResponse {
            total_ms: 10.0,
            prompt_tokens: 100,
            completion_tokens: 256,
        }
    }

    #[rstest]
    #[case::ok(valid_response(), 100, 256, true)]
    #[case::prompt_mismatch(
        EndToEndLatencyResponse {
            prompt_tokens: 99,
            ..valid_response()
        },
        100,
        256,
        false
    )]
    #[case::completion_mismatch(
        EndToEndLatencyResponse {
            completion_tokens: 255,
            ..valid_response()
        },
        100,
        256,
        false
    )]
    #[case::zero_ms(
        EndToEndLatencyResponse {
            total_ms: 0.0,
            ..valid_response()
        },
        100,
        256,
        false
    )]
    fn validates_response(
        #[case] response: EndToEndLatencyResponse,
        #[case] prompt: u32,
        #[case] completion: u32,
        #[case] ok: bool,
    ) {
        assert_eq!(validate_response(&response, prompt, completion).is_ok(), ok);
    }
}
