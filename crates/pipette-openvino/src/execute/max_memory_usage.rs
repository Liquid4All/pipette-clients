//! Peak memory for one OpenVINO cell: the server process's peak resident set
//! across its load and one request at the cell's prompt size with one output
//! token.
//!
//! A warm-up server runs the same request first: on a cold cache its load
//! compiles the model blob, and on a GPU its request compiles the kernels for
//! the cell's prompt shape, both into the cache, so the measured one loads
//! them. A compile's peak is the compiler's, not the model's
//! (`docs/openvino-measurement.md`). The observation covers the warm-up too,
//! since the cell started it.
//!
//! Off the CPU a run must show device memory (`max_gpu_bytes` or
//! `max_npu_bytes`). A device whose memory this OS cannot read (an NPU or a
//! custom device on Linux) is refused before anything starts rather than
//! recorded as the host's alone.

use std::path::Path;
use std::time::Duration;

use anyhow::Context;

use pipette_completions::TimingRequest;
use pipette_http::HttpClient;
use pipette_ops::measurement::{
    self, MaxMemoryMeasurementReading, MeasurementObserver, MemoryMeasurementWarmup,
};
use pipette_ops::run_response::RunResponseBuilder;
use pipette_plan_types::{
    MaxMemoryUsageBenchmark, MaxMemoryUsageUvOpenvinoOpenvino, RunResponse, RuntimeFlags,
    UvOpenvinoExtraRuntimeConfiguration,
};

use crate::completions;
use crate::memory::{serve, watch};
use crate::models::ResolvedOpenvinoModel;
use crate::response;
use crate::runtimes::ResolvedOpenvinoPython;
use crate::server;

const DECODE_TOKENS: u32 = 1;

#[expect(
    clippy::too_many_arguments,
    reason = "the request's parts, and the seams its entry takes"
)]
pub(super) fn run(
    benchmark: &MaxMemoryUsageBenchmark,
    runtime_flags: MaxMemoryUsageUvOpenvinoOpenvino,
    python: &ResolvedOpenvinoPython,
    model: &ResolvedOpenvinoModel,
    compile_cache: &Path,
    timeout: Duration,
    observer: MeasurementObserver<'_>,
    client_config: Option<&UvOpenvinoExtraRuntimeConfiguration>,
) -> anyhow::Result<RunResponse> {
    let watch = watch(
        runtime_flags
            .device
            .as_ref()
            .context("resolved flags name no device")?,
    )?;
    let flags = RuntimeFlags::MaxMemoryUsageUvOpenvinoOpenvino(runtime_flags);
    let prefill_tokens = benchmark
        .parameter_prefill_tokens
        .into_inner()
        .unsigned_abs();
    let script = server::script()?;
    let command = |port| {
        server::command(
            python,
            model,
            &flags,
            compile_cache,
            script.path(),
            port,
            client_config,
        )
    };
    let http = HttpClient::loopback("pipette", timeout)?;
    let mut served = None;
    // The `/proc` terms behind the swap-aware peak, for the row.
    #[cfg(any(target_os = "linux", target_os = "android"))]
    let mut extras = serde_json::Map::new();
    let result = measurement::max_memory(
        &benchmark.benchmark_id,
        observer,
        MemoryMeasurementWarmup::Once,
        &mut served,
        |served| {
            let peak = serve(served, command, watch, observer, |url| {
                measure(&http, url, prefill_tokens)
            })?;
            // The peak counts swap: swap can hold part of the model mid-run.
            #[cfg(any(target_os = "linux", target_os = "android"))]
            let peak = {
                extras = peak.footprint.extras();
                peak.counting_swap()
            };
            Ok(MaxMemoryMeasurementReading {
                max_host_bytes: peak.host_bytes,
                max_gpu_bytes: peak.gpu_bytes,
                max_npu_bytes: peak.npu_bytes,
            })
        },
    );
    let result_data = result.map_err(|err| match &mut served {
        Some((openvino_server, _)) => err.context(openvino_server.stop_with_tail()),
        None => err,
    })?;
    let response = served
        .iter_mut()
        .fold(
            RunResponseBuilder::new(result_data),
            |response, (openvino_server, memory)| response.process(openvino_server).memory(*memory),
        )
        .command(response::command(python))
        .runtime_flags(flags);
    #[cfg(any(target_os = "linux", target_os = "android"))]
    let response = response.client_extras(extras);
    Ok(response.build())
}

/// One request at the cell's shape.
fn measure(http: &HttpClient, url: &str, prefill_tokens: u32) -> anyhow::Result<()> {
    let prompt =
        pipette_completions::build_prompt_text(http, url, completions::MODEL, prefill_tokens)?;
    let counts = pipette_completions::reply(
        http,
        &completions::url(url),
        &TimingRequest::text(Some(completions::MODEL), &prompt, DECODE_TOKENS),
    )?
    .counts_at_limit()?;
    measurement::expect_tokens("prompt_tokens", counts.prompt_tokens, prefill_tokens)?;
    measurement::expect_tokens("completion_tokens", counts.completion_tokens, DECODE_TOKENS)?;
    Ok(())
}
