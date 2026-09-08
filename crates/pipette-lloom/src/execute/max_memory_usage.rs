use std::time::Duration;

use anyhow::Context;
use serde::{Deserialize, Serialize};

use pipette_memprobe_metal::{host, metal};
use pipette_plan_types::result::BenchmarkResultData;
use pipette_plan_types::run::RunRequest;
use pipette_plan_types::run::RunResponse;

use super::{server, throughput_http};
use crate::models::require_lloom_model_dir;
use crate::runtimes::require_lloom_serve;

const ENDPOINT: &str = "/max_memory_usage";
const SHUTDOWN_ENDPOINT: &str = "/shutdown";
const DECODE_TOKENS: u32 = 1;
const SERVER_EXIT_TIMEOUT: Duration = Duration::from_secs(60);
const READY_TIMEOUT: Duration = Duration::from_secs(3600);

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

/// The process's peak `phys_footprint` over load + prefill + one decode step,
/// polled from the outside, with the Metal shim attached for the diagnostic
/// device-side peak (unified memory: not reported as `max_gpu_bytes`).
pub(super) fn run(req: &RunRequest) -> anyhow::Result<RunResponse> {
    let benchmark = req
        .benchmark
        .as_max_memory_usage()
        .map_err(anyhow::Error::from)?;
    let prefill_tokens = benchmark.parameter_prefill_tokens;
    let binary = require_lloom_serve(req)?;
    let model_dir = require_lloom_model_dir(req)?;

    let mut probe = None;
    let mut server = server::start_server_with_command_config(
        &binary,
        &model_dir,
        None,
        READY_TIMEOUT,
        |command| {
            probe = Some(metal::MetalProbeChannel::attach(command)?);
            Ok(())
        },
    )?;
    let probe = probe.context("Metal probe channel was not attached to lloom-serve")?;
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

    let metal_peak = probe.read_peak()?;
    log::debug!(
        "lloom-serve peak: phys_footprint={max_host_bytes} (max_host_bytes); metal={} \
         (diagnostic — unified memory, not reported as max_gpu_bytes); unified={:?} devices={:?}",
        metal_peak.bytes,
        metal_peak.unified,
        metal_peak.n_devices,
    );
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
    if response.prompt_tokens != expected_prompt_tokens {
        anyhow::bail!(
            "{ENDPOINT} returned prompt_tokens {}, expected {expected_prompt_tokens}",
            response.prompt_tokens
        );
    }
    if response.completion_tokens != expected_completion_tokens {
        anyhow::bail!(
            "{ENDPOINT} returned completion_tokens {}, expected {expected_completion_tokens}",
            response.completion_tokens
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_response_shape() {
        let ok = MaxMemoryUsageResponse {
            prompt_tokens: 8,
            completion_tokens: 1,
        };
        assert!(validate_response(&ok, 8, 1).is_ok());
        assert!(validate_response(&ok, 7, 1).is_err());
        assert!(validate_response(&ok, 8, 2).is_err());
    }
}
