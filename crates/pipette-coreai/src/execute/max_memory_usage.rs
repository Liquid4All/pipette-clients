use std::time::Duration;

use anyhow::Context;
use serde::{Deserialize, Serialize};

use pipette_memprobe_metal::host;
use pipette_plan_types::result::BenchmarkResultData;
use pipette_plan_types::run::RunRequest;
use pipette_plan_types::run::RunResponse;

use super::{server, throughput_http};

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

pub(super) fn run(req: &RunRequest) -> anyhow::Result<RunResponse> {
    let benchmark = req
        .benchmark
        .as_max_memory_usage()
        .map_err(anyhow::Error::from)?;
    let prefill_tokens = benchmark.parameter_prefill_tokens;

    // Specialization (first load of this `.aimodel` on this OS) happens inside
    // sidecar construction, *before* READY. The poller starts after READY, so
    // a cold-cache compile peak is excluded by policy. See
    // docs/methodology/coreai-specialization.md.
    let mut server = server::start_server(req)?;
    let phys_poller = host::spawn_phys_footprint_poller(server.pid() as i32);

    let response_result: anyhow::Result<MaxMemoryUsageResponse> = throughput_http::post_json(
        &server.base_url,
        ENDPOINT,
        &MaxMemoryUsageRequest {
            prompt_tokens: prefill_tokens,
            decode_tokens: DECODE_TOKENS,
        },
    );
    let max_host_bytes = phys_poller
        .stop_and_join()
        .context("phys_footprint poller failed; max_host_bytes is unreliable")?;

    let shutdown_result: anyhow::Result<serde_json::Value> =
        throughput_http::post_json(&server.base_url, SHUTDOWN_ENDPOINT, &serde_json::json!({}));
    let exit_result = server.wait_for_exit(SERVER_EXIT_TIMEOUT);

    let response = response_result?;
    validate_response(&response, prefill_tokens, DECODE_TOKENS)?;
    shutdown_result?;
    exit_result?;

    // Apple Silicon is unified memory: Metal allocations are billed to
    // phys_footprint, so max_host_bytes subsumes the GPU allocation and there
    // is no separate pool to report (mirrors pipette-mlx).
    Ok(RunResponse {
        executable: Some(server.executable.clone()),
        command: server.command_preview.clone(),
        ..RunResponse::new(
            BenchmarkResultData::MaxMemoryUsage {
                max_host_bytes,
                max_gpu_bytes: None,
                max_npu_bytes: None,
            },
            server.stdout(),
            server.stderr(),
        )
    })
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
    use super::*;

    #[test]
    fn validates_response_shape() -> anyhow::Result<()> {
        validate_response(
            &MaxMemoryUsageResponse {
                prompt_tokens: 8,
                completion_tokens: 1,
            },
            8,
            1,
        )?;

        assert!(validate_response(
            &MaxMemoryUsageResponse {
                prompt_tokens: 7,
                completion_tokens: 1,
            },
            8,
            1,
        )
        .is_err());
        assert!(validate_response(
            &MaxMemoryUsageResponse {
                prompt_tokens: 8,
                completion_tokens: 2,
            },
            8,
            1,
        )
        .is_err());
        Ok(())
    }
}
