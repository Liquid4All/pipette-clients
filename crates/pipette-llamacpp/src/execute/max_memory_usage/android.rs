//! Android measurement path: runs `llama-bench` under a `/proc` footprint
//! sampler and reports `max(VmHWM, max(VmRSS + VmSwap))`.
//!
//! The swap term is why this samples rather than reading a counter at exit:
//! Android carries more zram than RAM, and pinning the weights in anonymous
//! memory makes them swap-eligible, so a resident-only peak misses whatever the
//! kernel compressed away while the run still needed it. Rationale, measured
//! numbers, and why a `toybox time -v` wrapper cannot supply the figure are in
//! `docs/methodology/peak-memory-android.md`.
//!
//! `max_gpu_bytes` is `null` on this flavor: Mali GPUs don't expose
//! DRM fdinfo, and there's no in-process Mali probe. Adreno + MSM-DRM
//! could populate it via DRM fdinfo in a future build; the work
//! belongs in `pipette-memprobe::os_counters::linux` if/when wanted.

use std::os::unix::process::CommandExt;
use std::{
    path::Path,
    process::{Command, Stdio},
};

use anyhow::Context;

use pipette_plan_types::result::{BenchmarkResultData, MemoryObservation};
use pipette_plan_types::RuntimeFlags;
use pipette_subprocess::{argv, echo_info};

use super::super::RunResponse;
use super::Params;
use crate::common::{
    apply_dylib_search_env, deadline_error_message, spawn_timeout_killer, MAX_MEMORY_USAGE_TIMEOUT,
};
use crate::run_memory::proc_footprint::{peak_ram_bytes, spawn_footprint_poller};

pub(super) fn run(
    llama_bench: &Path,
    params: Params,
    model_path: &Path,
    extra_flags: &[String],
    flags: &RuntimeFlags,
) -> anyhow::Result<RunResponse> {
    let mut cmd = Command::new(llama_bench);
    apply_dylib_search_env(&mut cmd, llama_bench);
    cmd.arg("--output").arg("json");
    cmd.arg("--model").arg(model_path);
    cmd.args(extra_flags);
    // --n-gen 1: exercise one decode step so we count decode-path
    // backend kernels + sampling buffers in the peak. Prefill alone
    // misses those — they're created on first decode and persist.
    // See pipette-mgmt/docs/methodology/peak-memory.md §2.4
    // "Workload phases captured" for rationale.
    cmd.arg("--n-prompt")
        .arg(params.parameter_prefill_tokens.to_string());
    cmd.arg("--n-gen").arg("1");
    cmd.stdout(Stdio::piped()).stderr(Stdio::piped());

    // Run llama-bench in its own process group so the deadline can SIGKILL
    // `-pid` and take any helper it spawns with it.
    //
    // Propagate setpgid's errno: if it failed and we then sent
    // `kill(-pid, SIGKILL)`, we'd target whatever pgid happens to
    // equal the child's pid (most likely no group, but possibly the
    // wrong one). Returning Err from `pre_exec` aborts the exec —
    // the safe failure mode.
    //
    // SAFETY: `setpgid(0, 0)` is async-signal-safe and the only call
    // we make between fork and exec. No allocators, no locks.
    unsafe {
        cmd.pre_exec(|| {
            if libc::setpgid(0, 0) != 0 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }

    let preview = argv(&cmd);
    echo_info(&cmd);

    let child = cmd
        .spawn()
        .with_context(|| format!("failed to spawn {}", llama_bench.display()))?;
    let pid = child.id();
    let pgid = pid as libc::pid_t;
    let killer = spawn_timeout_killer(MAX_MEMORY_USAGE_TIMEOUT, move || unsafe {
        // Negative pid → kill the whole process group.
        libc::kill(-pgid, libc::SIGKILL);
    });
    let poller = spawn_footprint_poller(pid);
    let output = child
        .wait_with_output()
        .with_context(|| format!("failed to wait for {}", llama_bench.display()))?;
    let footprint = poller.stop_and_join();
    let killer_fired = killer.fired();

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
            "llama-bench failed: {}\nstderr:\n{}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        );
    }

    let stdout = String::from_utf8(output.stdout).context("llama-bench returned non-utf8")?;
    let max_host_bytes = peak_ram_bytes(&footprint, model_path, flags)?;
    log::debug!(
        "android max_host_bytes = {max_host_bytes} (peak_rss {} KiB, peak_committed {} KiB, \
         max_swap {} KiB, major_faults {})",
        footprint.peak_rss_kib,
        footprint.peak_committed_kib,
        footprint.max_swap_kib,
        footprint.major_faults,
    );
    // The footprint terms explain the number the row carries, and the wire
    // schema has nowhere to put them, so they ride along in the captured
    // stderr that `extras.json` already preserves verbatim.
    let stderr = format!(
        "{}\n[pipette probe] peak_rss_kib={} peak_committed_kib={} max_swap_kib={} \
         major_faults={}\n",
        String::from_utf8_lossy(&output.stderr),
        footprint.peak_rss_kib,
        footprint.peak_committed_kib,
        footprint.max_swap_kib,
        footprint.major_faults,
    );

    Ok(RunResponse {
        executable: Some(llama_bench.display().to_string()),
        command: preview,
        runtime_flags: Some(flags.clone()),
        // Same sampler the metric came from, so this arm's observation and its
        // metric agree. Reported anyway: a consumer reading observations across
        // benchmarks should not have to special-case the memory one.
        memory: MemoryObservation {
            max_host_bytes: Some(max_host_bytes),
            max_swap_bytes: Some(footprint.max_swap_bytes()),
        },
        ..RunResponse::new(
            BenchmarkResultData::MaxMemoryUsage {
                max_host_bytes,
                max_gpu_bytes: None,
                max_npu_bytes: None,
            },
            stdout,
            stderr,
        )
    })
}
