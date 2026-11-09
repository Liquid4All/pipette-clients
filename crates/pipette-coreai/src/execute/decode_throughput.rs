use std::{num::NonZeroUsize, path::PathBuf};

use serde::{Deserialize, Serialize};

use pipette_ops::measurement::{self as shared, MeasurementObserver, MeasurementWarmup};
use pipette_ops::readiness::ReadinessGate;
use pipette_ops::run_response::RunResponseBuilder;
use pipette_plan_types::DecodeThroughputBenchmark;
use pipette_plan_types::RunRequest;
use pipette_plan_types::RunResponse;

use crate::{measurement, server, throughput_http};

const ENDPOINT: &str = "/decode_throughput";

#[derive(Debug, Serialize)]
struct DecodeThroughputRequest {
    prompt_tokens: u32,
    decode_tokens: u32,
}

#[derive(Debug, Deserialize)]
struct DecodeThroughputResponse {
    generation_tps: f64,
    prompt_tokens: u32,
    decode_tokens: u32,
}

pub(super) fn run(
    req: &RunRequest,
    benchmark: &DecodeThroughputBenchmark,
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
    require_decode_interval(decode_tokens)?;

    readiness_gate()?;
    let mut server = observer.outside(|| server::start_server(req, sidecar))?;

    let result = shared::decode(
        benchmark,
        NonZeroUsize::try_from(shared::REPS)?,
        readiness_gate,
        *observer,
        MeasurementWarmup::Once,
        &server,
        |server| {
            throughput_http::prepare(server)?;
            let response: DecodeThroughputResponse = throughput_http::post_json(
                server,
                ENDPOINT,
                &DecodeThroughputRequest {
                    prompt_tokens: prefill_tokens,
                    decode_tokens,
                },
            )?;
            Ok(shared::DecodeMeasurementReading::PerPrompt(vec![reading(
                response,
            )?]))
        },
    )?;
    Ok(RunResponseBuilder::new(result)
        .process(&mut server.process)
        .build())
}

/// The decode rate (tokens per second, tps) uses the N-1 intervals between
/// tokens; the first token starts the clock. A cell with one decode token
/// reports `generation_tps=0`, which fails validation with a misleading error.
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
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::one_token(1, Some(["decode_tokens >= 2", "N-1"].as_slice()))]
    #[case::two_tokens(2, None)]
    fn require_decode_interval_by_token_count(
        #[case] decode_tokens: u32,
        #[case] expected_error: Option<&[&str]>,
    ) -> anyhow::Result<()> {
        let result = require_decode_interval(decode_tokens);
        match (result, expected_error) {
            (Ok(()), None) => Ok(()),
            (Err(err), Some(fragments)) => {
                let msg = format!("{err:#}");
                for fragment in fragments {
                    anyhow::ensure!(msg.contains(fragment), "missing {fragment:?} in {msg}");
                }
                Ok(())
            }
            (Ok(()), Some(_)) => anyhow::bail!("{decode_tokens}-token decode must fail"),
            (Err(err), None) => Err(err.context(format!("{decode_tokens}-token decode must pass"))),
        }
    }
}

fn reading(
    response: DecodeThroughputResponse,
) -> anyhow::Result<shared::DecodeMeasurementPerPromptReading> {
    require_decode_interval(response.decode_tokens)?;
    // Core AI reports N-1 intervals. Preserve that span without from_stream's scaling.
    let ms = measurement::time_ms_from_tps(
        ENDPOINT,
        response.decode_tokens - 1,
        response.generation_tps,
    )?;
    Ok(shared::DecodeMeasurementPerPromptReading::from_millis(
        response.prompt_tokens,
        response.decode_tokens.try_into()?,
        ms,
    )?)
}

#[cfg(test)]
mod reading_tests {
    use rstest::rstest;

    use super::*;

    #[test]
    fn preserves_first_to_last_token_span() -> anyhow::Result<()> {
        let r = reading(DecodeThroughputResponse {
            prompt_tokens: 4,
            decode_tokens: 3,
            generation_tps: 2.0,
        })?;
        assert_eq!(r.decode_time_ms.into_inner(), 1000.0);
        assert_eq!(r.prefill_tokens, 4);
        Ok(())
    }

    #[rstest]
    #[case::zero(0.0)]
    #[case::negative(-1.0)]
    #[case::nan(f64::NAN)]
    #[case::infinite(f64::INFINITY)]
    fn rejects_an_invalid_rate(#[case] generation_tps: f64) {
        assert!(
            reading(DecodeThroughputResponse {
                prompt_tokens: 4,
                decode_tokens: 3,
                generation_tps,
            })
            .is_err()
        );
    }
}
