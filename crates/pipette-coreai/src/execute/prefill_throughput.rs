use anyhow::Context;
use serde::{Deserialize, Serialize};

use pipette_ops::measurement;
use pipette_ops::readiness::{ReadinessGate, RepObserver};
use pipette_plan_types::result::BenchmarkResultData;
use pipette_plan_types::run::RunRequest;
use pipette_plan_types::run::RunResponse;

use super::{server, throughput_http};

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
    readiness_gate: ReadinessGate,
    observer: &RepObserver,
) -> anyhow::Result<RunResponse> {
    let benchmark = req
        .benchmark
        .as_prefill_throughput()
        .map_err(anyhow::Error::from)?;
    let prefill_tokens = benchmark.parameter_prefill_tokens;

    // Resolve (and, on first use, build) the sidecar BEFORE the readiness
    // gate: a first-use `swift build -c release` saturates every core for
    // minutes and would sit between the gate certifying the device as thermally
    // idle and the measurement that certification is for.
    let sidecar = crate::sidecar::require_coreai_sidecar()?;
    readiness_gate()?;
    let server = server::start_server(req, Some(sidecar))?;

    log::info!("{ENDPOINT}: warm-up run ({prefill_tokens}p)");
    throughput_http::prepare(&server.base_url)?;
    let warmup: PrefillThroughputResponse = throughput_http::post_json(
        &server.base_url,
        ENDPOINT,
        &PrefillThroughputRequest {
            prompt_tokens: prefill_tokens,
        },
    )?;
    measurement::expect_tokens("warmup prompt_tokens", warmup.prompt_tokens, prefill_tokens)?;

    let measured = measurement::run(
        ENDPOINT,
        readiness_gate,
        observer,
        |_| throughput_http::prepare(&server.base_url),
        |_| {
            throughput_http::post_json::<_, PrefillThroughputResponse>(
                &server.base_url,
                ENDPOINT,
                &PrefillThroughputRequest {
                    prompt_tokens: prefill_tokens,
                },
            )
        },
        |idx, rep| {
            let response = &rep.value;
            measurement::expect_tokens("prompt_tokens", response.prompt_tokens, prefill_tokens)?;
            measurement::validate_tps("prompt_tps", response.prompt_tps)
                .with_context(|| format!("invalid {ENDPOINT} rep {idx}"))?;
            measurement::time_ms_from_tps(ENDPOINT, prefill_tokens, response.prompt_tps)
        },
    )?;
    let stats = measured.stats();
    Ok(RunResponse {
        executable: Some(server.executable.clone()),
        command: server.command_preview.clone(),
        ..RunResponse::new(
            BenchmarkResultData::PrefillThroughput {
                prefill_time_ms: stats.mean_ms,
                prefill_time_ms_stddev: Some(stats.stddev_ms),
            },
            server.stdout(),
            server.stderr(),
        )
    })
}
