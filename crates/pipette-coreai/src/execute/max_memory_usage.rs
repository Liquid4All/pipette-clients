use std::path::PathBuf;
use std::time::Duration;

use anyhow::Context;
use serde::{Deserialize, Serialize};

use pipette_memprobe::{Sampler, Watch};
use pipette_ops::measurement::{MeasurementObserver, PEAK_MEMORY_MEASURED};
use pipette_ops::run_response::RunResponseBuilder;
use pipette_plan_types::BenchmarkResultData;
use pipette_plan_types::MaxMemoryUsageBenchmark;
use pipette_plan_types::RunRequest;
use pipette_plan_types::RunResponse;

use crate::{server, throughput_http};

const ENDPOINT: &str = "/max_memory_usage";
const SHUTDOWN_ENDPOINT: &str = "/shutdown";
const DECODE_TOKENS: u32 = 1;
const SERVER_EXIT_TIMEOUT: Duration = Duration::from_secs(60);

#[derive(Debug, Serialize)]
struct MaxMemoryUsageRequest {
    prompt_tokens: u32,
    decode_tokens: u32,
}

#[derive(Debug, Deserialize)]
struct MaxMemoryUsageResponse {
    prompt_tokens: u32,
    completion_tokens: u32,
}

pub(super) fn run(
    req: &RunRequest,
    benchmark: &MaxMemoryUsageBenchmark,
    sidecar: PathBuf,
    observer: &MeasurementObserver<'_>,
) -> anyhow::Result<RunResponse> {
    let prefill_tokens = benchmark
        .parameter_prefill_tokens
        .into_inner()
        .unsigned_abs();

    // The sampler reports the lifetime peak of the sidecar process, so the
    // number includes model load. On a cold specialization cache it also
    // includes the specialization peak; `specialization_cache=hit|miss` on the
    // READY line in `runtime_stderr` records which case applies. See
    // docs/methodology/coreai-specialization.md.
    (observer.on_attempt_start)(PEAK_MEMORY_MEASURED);
    let mut server = observer.outside(|| server::start_server(req, sidecar))?;
    let phys_poller = Sampler::start(server.process.pid(), Watch::Peak.supported()?)?;

    throughput_http::prepare(&server)?;
    let response_result: anyhow::Result<MaxMemoryUsageResponse> = throughput_http::post_json(
        &server,
        ENDPOINT,
        &MaxMemoryUsageRequest {
            prompt_tokens: prefill_tokens,
            decode_tokens: DECODE_TOKENS,
        },
    );
    let (_, peak) = phys_poller.stop();
    let max_host_bytes = peak.context("phys_footprint poller failed")?.host_bytes;

    let shutdown_result: anyhow::Result<serde_json::Value> =
        throughput_http::post_json(&server, SHUTDOWN_ENDPOINT, &serde_json::json!({}));
    let exit_result = server.process.wait_success(SERVER_EXIT_TIMEOUT);

    let response = response_result?;
    validate_response(&response, prefill_tokens, DECODE_TOKENS)?;
    shutdown_result?;
    exit_result?;

    // Apple Silicon is unified memory: Metal allocations are billed to
    // phys_footprint, so max_host_bytes subsumes the GPU allocation and there
    // is no separate pool to report (mirrors pipette-mlx).
    Ok(
        RunResponseBuilder::new(BenchmarkResultData::MaxMemoryUsage {
            max_host_bytes,
            max_gpu_bytes: None,
            max_npu_bytes: None,
        })
        .process(&mut server.process)
        .build(),
    )
}

fn validate_response(
    response: &MaxMemoryUsageResponse,
    expected_prompt_tokens: u32,
    expected_completion_tokens: u32,
) -> anyhow::Result<()> {
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

    fn valid_response() -> MaxMemoryUsageResponse {
        MaxMemoryUsageResponse {
            prompt_tokens: 8,
            completion_tokens: 1,
        }
    }

    #[rstest]
    #[case::ok(valid_response(), 8, 1, true)]
    #[case::prompt_mismatch(
        MaxMemoryUsageResponse {
            prompt_tokens: 7,
            ..valid_response()
        },
        8,
        1,
        false
    )]
    #[case::completion_mismatch(
        MaxMemoryUsageResponse {
            completion_tokens: 2,
            ..valid_response()
        },
        8,
        1,
        false
    )]
    fn validates_response(
        #[case] response: MaxMemoryUsageResponse,
        #[case] prompt: u32,
        #[case] completion: u32,
        #[case] ok: bool,
    ) {
        assert_eq!(validate_response(&response, prompt, completion).is_ok(), ok);
    }
}
