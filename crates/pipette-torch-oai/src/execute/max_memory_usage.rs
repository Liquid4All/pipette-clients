use std::path::Path;
use std::time::Duration;

use anyhow::Context;

use pipette_plan_types::result::BenchmarkResultData;
use pipette_plan_types::run::RunRequest;
use pipette_plan_types::run::RunResponse;

use super::{build_prompt_text, http_timeout, validate_completion_usage};
use crate::{
    memprobe,
    openai::{self, CompletionPrompt, CompletionRequest},
    server::ServerState,
};

/// Max-memory cell: peak host/GPU bytes during a short completion (docker only).
pub(super) fn run(
    req: &RunRequest,
    model: &str,
    docker_bin: &Path,
    state: &ServerState,
) -> anyhow::Result<RunResponse> {
    let body = req
        .benchmark
        .as_max_memory_usage()
        .map_err(anyhow::Error::from)?;
    let prefill_tokens = body.parameter_prefill_tokens;

    // memprobe is docker-only — it relies on `docker inspect`'s
    // cgroup-path output to read `memory.peak`. Doing the same for the
    // uv engine would need a /proc-walked cgroup lookup keyed off the
    // uv server's PGID; that's a separate feature, not part of T5.
    // (Caller rejects uv launch before this runs.)

    let decode_tokens = 16;
    let timeout = http_timeout(req);
    let prompt_text = build_prompt_text(&state.base_url(), model, prefill_tokens, timeout)?;
    let probe = memprobe::Probe::start(
        docker_bin,
        state
            .container_id()
            .context("max_memory_usage benchmark on docker runtime requires container id")?
            .as_str(),
        Duration::from_millis(500),
    )?;
    let request = CompletionRequest {
        model: model.to_string(),
        prompt: CompletionPrompt::Text(prompt_text),
        // A small decode is enough to load the model + flow a forward pass;
        // the peak typically tracks prefill, not decode length.
        max_tokens: Some(decode_tokens),
        temperature: Some(0.0),
        ignore_eos: Some(true),
    };
    let response = openai::complete(&state.base_url(), &request, timeout);

    let peak = probe.stop()?;
    // Report the request error after the probe so we still surface any partial
    // peaks the probe collected before failure.
    let response = response?;
    let usage = validate_completion_usage(response.usage.as_ref(), prefill_tokens, decode_tokens)?;

    let stdout = format!(
        "prefill_target_tokens={prefill_tokens} prompt_tokens={} completion_tokens={}\n\
         max_host_bytes={} max_gpu_bytes={:?} samples={}\n",
        usage.prompt_tokens,
        usage.completion_tokens,
        peak.max_host_bytes,
        peak.max_gpu_bytes,
        peak.samples,
    );
    log::info!("{}", stdout.trim_end());

    Ok(RunResponse {
        executable: Some(state.executable().display().to_string()),
        command: vec![
            format!("POST {}/v1/completions", state.base_url()),
            format!("model={model} prefill_target_tokens={prefill_tokens}"),
            "memprobe: cgroup memory.peak + nvidia-smi compute-apps".to_string(),
        ],
        ..RunResponse::new(
            BenchmarkResultData::MaxMemoryUsage {
                max_host_bytes: peak.max_host_bytes,
                max_gpu_bytes: peak.max_gpu_bytes,
                max_npu_bytes: None,
            },
            stdout,
            String::new(),
        )
    })
}
