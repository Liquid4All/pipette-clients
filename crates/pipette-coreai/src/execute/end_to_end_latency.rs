use anyhow::Context;
use serde::{Deserialize, Serialize};

use pipette_ops::measurement;
use pipette_ops::readiness::{ReadinessGate, RepObserver};
use pipette_plan_types::result::BenchmarkResultData;
use pipette_plan_types::run::RunRequest;
use pipette_plan_types::run::RunResponse;

use super::{server, throughput_http};

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
    readiness_gate: ReadinessGate,
    observer: &RepObserver,
) -> anyhow::Result<RunResponse> {
    let benchmark = req
        .benchmark
        .as_end_to_end_latency()
        .map_err(anyhow::Error::from)?;
    let prefill_tokens = benchmark.parameter_prefill_tokens;
    let decode_tokens = benchmark.parameter_decode_tokens;

    readiness_gate()?;
    let server = server::start_server(req, None)?;

    log::info!("{ENDPOINT}: warm-up run ({prefill_tokens}p/{decode_tokens}g)");
    validate_response(
        &run_latency_request(&server.base_url, prefill_tokens, decode_tokens)?,
        prefill_tokens,
        decode_tokens,
    )
    .context("invalid /end_to_end_latency warmup")?;

    let measured = measurement::run(
        "end_to_end_latency",
        readiness_gate,
        observer,
        |_| Ok(()),
        |_| run_latency_request(&server.base_url, prefill_tokens, decode_tokens),
        |idx, rep| {
            validate_response(&rep.value, prefill_tokens, decode_tokens)
                .with_context(|| format!("invalid {ENDPOINT} trial {idx}"))?;
            Ok(rep.elapsed_ms())
        },
    )?;
    let stats = measured.stats();

    Ok(RunResponse {
        executable: Some(server.executable.clone()),
        command: server.command_preview.clone(),
        ..RunResponse::new(
            BenchmarkResultData::EndToEndLatency {
                total_time_ms: stats.mean_ms,
                total_time_ms_stddev: Some(stats.stddev_ms),
            },
            server.stdout(),
            server.stderr(),
        )
    })
}

fn run_latency_request(
    base_url: &str,
    prefill_tokens: u32,
    decode_tokens: u32,
) -> anyhow::Result<EndToEndLatencyResponse> {
    throughput_http::post_json(
        base_url,
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
    validate_ms("total_ms", response.total_ms)?;
    validate_token_count(
        "prompt_tokens",
        response.prompt_tokens,
        expected_prompt_tokens,
    )?;
    validate_token_count(
        "completion_tokens",
        response.completion_tokens,
        expected_completion_tokens,
    )
}

fn validate_ms(metric: &str, value: f64) -> anyhow::Result<()> {
    if !value.is_finite() || value <= 0.0 {
        anyhow::bail!("invalid {metric}: {value}");
    }
    Ok(())
}

fn validate_token_count(metric: &str, actual: u32, expected: u32) -> anyhow::Result<()> {
    if actual != expected {
        anyhow::bail!("{ENDPOINT} returned {metric} {actual}, expected {expected}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_response() -> EndToEndLatencyResponse {
        EndToEndLatencyResponse {
            total_ms: 10.0,
            prompt_tokens: 100,
            completion_tokens: 256,
        }
    }

    #[test]
    fn accepts_valid_response() {
        assert!(validate_response(&valid_response(), 100, 256).is_ok());
    }

    #[test]
    fn rejects_token_mismatches() {
        assert!(validate_response(
            &EndToEndLatencyResponse {
                prompt_tokens: 99,
                ..valid_response()
            },
            100,
            256
        )
        .is_err());
        assert!(validate_response(
            &EndToEndLatencyResponse {
                completion_tokens: 255,
                ..valid_response()
            },
            100,
            256
        )
        .is_err());
    }

    #[test]
    fn rejects_invalid_ms() {
        assert!(validate_response(
            &EndToEndLatencyResponse {
                total_ms: 0.0,
                ..valid_response()
            },
            100,
            256
        )
        .is_err());
    }
}
