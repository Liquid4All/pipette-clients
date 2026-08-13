//! Linux measurement path: runs `llama-bench` directly and reads peak
//! resident set from `/proc/<pid>/status` `VmHWM`.
//!
//! `VmHWM` is the same peak-RSS figure `wait4`'s `ru_maxrss` reports,
//! but reading it avoids fighting `std` for the zombie: `wait_with_output`
//! reaps the child itself, so a competing `wait4` would race it. `VmHWM`
//! only grows, so any sample taken while the child lives — after the peak
//! at model load + first decode — captures it.
//!
//! Note this reports peak *resident* set, which under swap pressure is less than
//! the run required: the Android arm reports `VmRSS + VmSwap` for that reason.
//! Adopting the same term here needs the model-file floor too, so it is left as
//! a follow-up rather than changed as a side effect.
//!
//! `max_gpu_bytes` is `null`: only CPU flavors dispatch here. A GPU
//! flavor would need a DRM-fdinfo probe in
//! `pipette-memprobe::os_counters::linux` before routing here without
//! under-reporting.

use std::os::unix::process::CommandExt;
use std::{
    path::Path,
    process::{Command, Stdio},
};

// `Context` is a trait; its `.context()` / `.with_context()` methods need it
// in scope. `bail!` and `Result` are written `anyhow::`-qualified at use sites.
use anyhow::Context;

use pipette_plan_types::result::BenchmarkResultData;
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

    // Own process group so a SIGKILL to -pid takes down llama-bench
    // and anything it spawns. See android.rs for the full rationale on
    // propagating setpgid's errno out of pre_exec.
    //
    // SAFETY: `setpgid(0, 0)` is async-signal-safe and the only call
    // we make between fork and exec.
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
        libc::kill(-pgid, libc::SIGKILL);
    });

    let poller = crate::run_memory::proc_footprint::spawn_footprint_poller(pid);

    let output = child
        .wait_with_output()
        .with_context(|| format!("failed to wait for {}", llama_bench.display()))?;
    // Resident peak only, as before. The sampler also collects the swap-aware
    // footprint the Android arm reports; adopting it here needs the same
    // model-file floor, so it is left to a follow-up rather than changing what
    // Linux rows mean as a side effect.
    let footprint = poller.stop_and_join();
    let max_kib = footprint.peak_rss_kib;
    let killer_fired = killer.fired();
    drop(killer);

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

    if max_kib == 0 {
        anyhow::bail!("could not read VmHWM from /proc/{pid}/status during the run");
    }
    let max_host_bytes = max_kib.saturating_mul(1024);
    log::debug!("linux max_host_bytes = {max_host_bytes} ({max_kib} KiB) from /proc VmHWM");

    let stdout = String::from_utf8(output.stdout).context("llama-bench returned non-utf8")?;
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();

    Ok(RunResponse {
        executable: Some(llama_bench.display().to_string()),
        command: preview,
        runtime_flags: Some(flags.clone()),
        // Both resident-only, so the observation restates this arm's metric and
        // adds the swap term beside it. That term is the point here: it names
        // the hosts where this metric is being suppressed by reclaim, which is
        // what adopting the swap-aware peak on Linux would have to be argued
        // from. Through the same rule the timing benchmarks use, so one sampler
        // cannot mean two things.
        memory: crate::run_memory::observation_from(&footprint),
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
