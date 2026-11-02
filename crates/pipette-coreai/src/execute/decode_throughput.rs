use anyhow::Context;
use serde::{Deserialize, Serialize};

use pipette_ops::measurement;
use pipette_ops::readiness::{ReadinessGate, RepObserver};
use pipette_plan_types::result::BenchmarkResultData;
use pipette_plan_types::run::RunRequest;
use pipette_plan_types::run::RunResponse;

use super::{server, throughput_http};

const ENDPOINT: &str = "/decode_throughput";

#[derive(Debug, Serialize)]
struct DecodeThroughputRequest {
    prompt_tokens: u32,
    decode_tokens: u32,
}

#[derive(Debug, Deserialize)]
struct DecodeThroughputResponse {
    generation_tps: f64,
    decode_tokens: u32,
}

pub(super) fn run(
    req: &RunRequest,
    readiness_gate: ReadinessGate,
    observer: &RepObserver,
) -> anyhow::Result<RunResponse> {
    let benchmark = req
        .benchmark
        .as_decode_throughput()
        .map_err(anyhow::Error::from)?;
    let prefill_tokens = benchmark.parameter_prefill_tokens;
    let decode_tokens = benchmark.parameter_decode_tokens;
    require_decode_interval(decode_tokens)?;

    // Resolve (and, on first use, build) the sidecar BEFORE the readiness
    // gate: a first-use `swift build -c release` saturates every core for
    // minutes and would sit between the gate certifying the device as thermally
    // idle and the measurement that certification is for.
    let sidecar = crate::sidecar::require_coreai_sidecar()?;
    readiness_gate()?;
    let server = server::start_server(req, Some(sidecar))?;

    log::info!("{ENDPOINT}: warm-up run ({prefill_tokens}p/{decode_tokens}g)");
    throughput_http::prepare(&server.base_url)?;
    let warmup: DecodeThroughputResponse = throughput_http::post_json(
        &server.base_url,
        ENDPOINT,
        &DecodeThroughputRequest {
            prompt_tokens: prefill_tokens,
            decode_tokens,
        },
    )?;
    measurement::expect_tokens(
        &format!("{ENDPOINT} warmup decode_tokens"),
        warmup.decode_tokens,
        decode_tokens,
    )?;
    let measured = measurement::run(
        ENDPOINT,
        readiness_gate,
        observer,
        |_| throughput_http::prepare(&server.base_url),
        |_| {
            throughput_http::post_json::<_, DecodeThroughputResponse>(
                &server.base_url,
                ENDPOINT,
                &DecodeThroughputRequest {
                    prompt_tokens: prefill_tokens,
                    decode_tokens,
                },
            )
        },
        |idx, rep| {
            let response = &rep.value;
            measurement::expect_tokens("decode_tokens", response.decode_tokens, decode_tokens)?;
            measurement::validate_tps("generation_tps", response.generation_tps)
                .with_context(|| format!("invalid {ENDPOINT} rep {idx}"))?;
            measurement::time_ms_from_tps(ENDPOINT, decode_tokens, response.generation_tps)
        },
    )?;
    let stats = measured.stats();
    Ok(RunResponse {
        executable: Some(server.executable.clone()),
        command: server.command_preview.clone(),
        ..RunResponse::new(
            BenchmarkResultData::DecodeThroughput {
                decode_time_ms: stats.mean_ms,
                decode_time_ms_stddev: Some(stats.stddev_ms),
            },
            server.stdout(),
            server.stderr(),
        )
    })
}

/// Decode tps is counted over N-1 inter-token intervals (first token starts
/// the clock). A 1-token cell would report generation_tps=0 and fail mid-run
/// with a vacuous metric error.
fn require_decode_interval(decode_tokens: u32) -> anyhow::Result<()> {
    if decode_tokens < 2 {
        anyhow::bail!(
            "decode_throughput requires decode_tokens >= 2 \
             (generation_tps is counted over N-1 inter-token intervals; \
             a 1-token decode has no interval)"
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_single_token_decode() {
        let result = require_decode_interval(1);
        assert!(result.is_err(), "1-token decode must fail");
        if let Err(err) = result {
            let msg = format!("{err:#}");
            assert!(msg.contains("decode_tokens >= 2"), "{msg}");
            assert!(msg.contains("N-1"), "{msg}");
        }
    }

    #[test]
    fn accepts_two_token_decode() {
        assert!(require_decode_interval(2).is_ok());
    }
}
