use std::{num::NonZeroUsize, path::PathBuf};

use serde::{Deserialize, Serialize};

use pipette_ops::measurement::{self as shared, MeasurementObserver, MeasurementWarmup};
use pipette_ops::readiness::ReadinessGate;
use pipette_ops::run_response::RunResponseBuilder;
use pipette_plan_types::PrefillThroughputBenchmark;
use pipette_plan_types::RunRequest;
use pipette_plan_types::RunResponse;

use crate::{measurement, server, throughput_http};

const ENDPOINT: &str = "/prefill_throughput";

#[derive(Debug, Serialize)]
struct PrefillThroughputRequest {
    prompt_tokens: u32,
}

#[derive(Debug, Deserialize)]
struct PrefillThroughputResponse {
    prompt_tps: f64,
    prompt_tokens: u32,
}

pub(super) fn run(
    req: &RunRequest,
    benchmark: &PrefillThroughputBenchmark,
    sidecar: PathBuf,
    readiness_gate: ReadinessGate,
    observer: &MeasurementObserver<'_>,
) -> anyhow::Result<RunResponse> {
    let prefill_tokens = benchmark
        .parameter_prefill_tokens
        .into_inner()
        .unsigned_abs();

    readiness_gate()?;
    let mut server = observer.outside(|| server::start_server(req, sidecar))?;

    let result = shared::prefill(
        benchmark,
        NonZeroUsize::try_from(shared::REPS)?,
        readiness_gate,
        *observer,
        MeasurementWarmup::Once,
        &server,
        |server| {
            throughput_http::prepare(server)?;
            let response: PrefillThroughputResponse = throughput_http::post_json(
                server,
                ENDPOINT,
                &PrefillThroughputRequest {
                    prompt_tokens: prefill_tokens,
                },
            )?;
            Ok(shared::PrefillMeasurementReading::PerPrompt(vec![reading(
                response,
            )?]))
        },
    )?;
    Ok(RunResponseBuilder::new(result)
        .process(&mut server.process)
        .build())
}

fn reading(
    response: PrefillThroughputResponse,
) -> anyhow::Result<shared::PrefillMeasurementPerPromptReading> {
    let ms = measurement::time_ms_from_tps(ENDPOINT, response.prompt_tokens, response.prompt_tps)?;
    Ok(shared::PrefillMeasurementPerPromptReading::from_millis(
        response.prompt_tokens.try_into()?,
        ms,
    )?)
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[test]
    fn converts_the_rate_to_a_prefill_time() -> anyhow::Result<()> {
        let r = reading(PrefillThroughputResponse {
            prompt_tokens: 4,
            prompt_tps: 2.0,
        })?;
        assert_eq!(r.prefill_time_ms.into_inner(), 2000.0);
        Ok(())
    }

    #[rstest]
    #[case::zero(0.0)]
    #[case::negative(-1.0)]
    #[case::nan(f64::NAN)]
    #[case::infinite(f64::INFINITY)]
    fn rejects_an_invalid_rate(#[case] prompt_tps: f64) {
        assert!(
            reading(PrefillThroughputResponse {
                prompt_tokens: 4,
                prompt_tps,
            })
            .is_err()
        );
    }
}
