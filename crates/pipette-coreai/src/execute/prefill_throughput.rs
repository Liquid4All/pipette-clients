use std::path::PathBuf;

use anyhow::Context;
use serde::{Deserialize, Serialize};

use pipette_ops::measurement::MeasurementObserver;
use pipette_ops::readiness::ReadinessGate;
use pipette_ops::run_response::RunResponseBuilder;
use pipette_plan_types::BenchmarkResultData;
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
    let mut server = server::start_server(req, sidecar)?;

    tracing::info!(cell = ENDPOINT, prefill_tokens, "Core AI prefill warm-up");
    throughput_http::prepare(&server)?;
    let warmup: PrefillThroughputResponse = throughput_http::post_json(
        &server,
        ENDPOINT,
        &PrefillThroughputRequest {
            prompt_tokens: prefill_tokens,
        },
    )?;
    pipette_ops::measurement::expect_tokens(
        "warmup prompt_tokens",
        warmup.prompt_tokens,
        prefill_tokens,
    )?;

    let measured = pipette_ops::measurement::run(
        ENDPOINT,
        readiness_gate,
        *observer,
        |_| throughput_http::prepare(&server),
        |_| {
            throughput_http::post_json::<_, PrefillThroughputResponse>(
                &server,
                ENDPOINT,
                &PrefillThroughputRequest {
                    prompt_tokens: prefill_tokens,
                },
            )
        },
        |idx, rep| {
            let response = &rep.value;
            pipette_ops::measurement::expect_tokens(
                "prompt_tokens",
                response.prompt_tokens,
                prefill_tokens,
            )?;
            pipette_ops::measurement::positive_finite("prompt_tps", response.prompt_tps)
                .with_context(|| format!("invalid {ENDPOINT} rep {idx}"))?;
            measurement::time_ms_from_tps(ENDPOINT, prefill_tokens, response.prompt_tps)
        },
    )?;
    let stats = measured.stats();
    Ok(
        RunResponseBuilder::new(BenchmarkResultData::PrefillThroughput {
            prefill_time_ms: stats.mean_ms,
            prefill_time_ms_stddev: Some(stats.stddev_ms),
        })
        .process(&mut server.process)
        .build(),
    )
}
