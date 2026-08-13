//! macOS measurement path — direct orchestration.
//!
//! Attach the Metal probe channel to the `llama-bench` Command, run
//! a `phys_footprint` poller alongside, wait, read the shim's
//! snapshot. The two peaks are reported independently: `max_gpu_bytes`
//! is the Metal allocator's peak, `max_host_bytes` is `phys_footprint`'s
//! peak. They may overlap on Apple UMA (Metal allocations live inside
//! `phys_footprint`); each is the peak of its own counter.
//!
//! See `pipette-mgmt/docs/methodology/peak-memory.md` §3.1 for the
//! methodology.
//!
//! **Parallel implementation**:
//! `pipette-mlx/src/execute/max_memory_usage.rs` runs the same
//! probe-attach → spawn server → phys_footprint poller → workload →
//! server exit → read flow for the Python server process. Both consumers
//! use [`pipette_memprobe_metal::metal::MetalProbeChannel`] so the attach
//! + tempdir lifecycle are owned by the crate.

use std::{
    path::Path,
    process::{Command, Stdio},
};

use anyhow::Context;

use pipette_memprobe_metal::{host, metal};
use pipette_plan_types::result::{BenchmarkResultData, MemoryObservation};
use pipette_plan_types::RuntimeFlags;
use pipette_subprocess::{argv, echo_info};

use super::super::RunResponse;
use super::Params;
use crate::common::{
    apply_dylib_search_env, deadline_error_message, spawn_timeout_killer, MAX_MEMORY_USAGE_TIMEOUT,
};

pub(super) fn run(
    llama_bench: &Path,
    params: Params,
    model_path: &Path,
    extra_flags: &[String],
    flags: &RuntimeFlags,
) -> anyhow::Result<RunResponse> {
    // 1. Build llama-bench Command and attach the Metal probe
    //    channel. Attach extracts the bundled dylib, allocates a
    //    per-run output tempfile, and wires DYLD_INSERT_LIBRARIES +
    //    PIPETTE_MEMPROBE_OUT into the env.
    let mut cmd = Command::new(llama_bench);
    apply_dylib_search_env(&mut cmd, llama_bench);
    cmd.arg("--output").arg("json");
    cmd.arg("--model").arg(model_path);
    cmd.args(extra_flags);
    // --n-gen 1: exercise one decode step so we count Metal decode
    // kernels + sampling buffers in the peak. Prefill alone misses
    // those — they're created on first decode and persist for the
    // run. See pipette-mgmt/docs/methodology/peak-memory.md §2.4
    // "Workload phases captured" for rationale.
    cmd.arg("--n-prompt")
        .arg(params.parameter_prefill_tokens.to_string())
        .arg("--n-gen")
        .arg("1");
    let probe = metal::MetalProbeChannel::attach(&mut cmd)?;
    cmd.stdout(Stdio::piped()).stderr(Stdio::piped());
    let preview = argv(&cmd);
    log::info!("measuring peak memory via peakmtl");
    echo_info(&cmd);

    // 2. Spawn child + phys_footprint poller + deadline killer;
    //    wait_with_output to drain pipes and reap. Stop+join the
    //    poller and drop the killer after wait. The killer's drop
    //    cancels the watchdog thread; if it had already fired we
    //    detect that via `fired()` below and surface a clear error.
    let child = cmd
        .spawn()
        .with_context(|| format!("failed to spawn {}", llama_bench.display()))?;
    let pid = child.id();
    let phys_poller = host::spawn_phys_footprint_poller(pid as i32);
    let killer = spawn_timeout_killer(MAX_MEMORY_USAGE_TIMEOUT, move || unsafe {
        libc::kill(pid as libc::pid_t, libc::SIGKILL);
    });
    let output = child
        .wait_with_output()
        .context("failed to wait for llama-bench")?;
    let killer_fired = killer.fired();
    drop(killer);
    let phys_peak = phys_poller
        .stop_and_join()
        .context("phys_footprint poller failed; max_host_bytes is unreliable")?;

    if killer_fired {
        anyhow::bail!(
            "{}",
            deadline_error_message(
                MAX_MEMORY_USAGE_TIMEOUT,
                &String::from_utf8_lossy(&output.stderr)
            )
        );
    }
    if !output.status.success() {
        anyhow::bail!(
            "llama-bench failed: {}; stderr tail:\n{}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        );
    }

    // 3. Read the shim's snapshot. `read_peak` already returns
    //    operator-actionable errors for the DYLD-blocked and
    //    shim-anomaly cases.
    let metal_peak = probe.read_peak()?;

    // 4. Apple Silicon (M1–M5, the only macOS target) is unified memory with no
    //    host/GPU accounting split: Metal allocations are billed to
    //    phys_footprint, so the host counter already subsumes them. Per the
    //    peak-memory methodology's unified-vs-split rule, report the whole cost
    //    as max_host_bytes and leave max_gpu_bytes null — the Metal allocator
    //    peak is a diagnostic, not a separate capacity dimension.
    let metal_peak_bytes = metal_peak.bytes;
    let max_host_bytes = phys_peak;
    log::debug!(
        "mac peak: phys_footprint={phys_peak} (max_host_bytes); metal={metal_peak_bytes} \
         (diagnostic — unified memory, not reported as max_gpu_bytes); \
         metal_unified={:?} metal_devices={:?}",
        metal_peak.unified,
        metal_peak.n_devices,
    );

    // 5. Capture llama-bench output verbatim. The JSON shape is
    //    inert here — `max_memory_usage` reports memory counters,
    //    not the timing fields the timing benchmarks parse.
    let stdout = String::from_utf8(output.stdout).context("llama-bench returned non-utf8")?;
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();

    Ok(RunResponse {
        executable: Some(llama_bench.display().to_string()),
        command: preview,
        runtime_flags: Some(flags.clone()),
        // `phys_footprint` bills compressed pages to the process, so the peak
        // already counts them and there is no separate swap term to observe.
        memory: MemoryObservation {
            max_host_bytes: Some(phys_peak),
            max_swap_bytes: None,
        },
        ..RunResponse::new(
            BenchmarkResultData::MaxMemoryUsage {
                max_host_bytes,
                // Unified memory (Apple Silicon): the GPU allocation lives inside the
                // host footprint, so there is no separate pool to report.
                max_gpu_bytes: None,
                // No platform implements per-process NPU memory yet.
                max_npu_bytes: None,
            },
            stdout,
            stderr,
        )
    })
}
