//! Linux measurement path: runs `llama-bench` directly and reads peak
//! resident set from `/proc/<pid>/status` `VmHWM`.
//!
//! `VmHWM` is the same peak-RSS figure `wait4`'s `ru_maxrss` reports,
//! but reading it avoids fighting `std` for the zombie: `wait_with_output`
//! reaps the child itself, so a competing `wait4` would race it. `VmHWM`
//! only grows, so any sample taken while the child lives — after the peak
//! at model load + first decode — captures it, with no vendored `time`
//! binary like Android's `toybox` wrapper.
//!
//! `max_gpu_bytes` is `null`: only CPU flavors dispatch here. A GPU
//! flavor would need a DRM-fdinfo probe in
//! `pipette-memprobe::os_counters::linux` before routing here without
//! under-reporting.

use std::os::unix::process::CommandExt;
use std::{
    fs,
    path::Path,
    process::{Command, Stdio},
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc,
    },
    thread,
    time::Duration,
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

/// `VmHWM` only grows, so the cadence doesn't affect correctness — it
/// just bounds how late the first post-peak sample lands. 25ms catches
/// the peak on a sub-second run; the read cost over the deadline is
/// negligible.
const POLL_INTERVAL: Duration = Duration::from_millis(25);

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

    // Seed with one synchronous read so a child that exits before the
    // poller thread is scheduled still yields a non-zero peak — its
    // `/proc` entry exists the moment `spawn` returns.
    let peak_kib = Arc::new(AtomicU64::new(read_vmhwm_kib(pid).unwrap_or(0)));
    let stop = Arc::new(AtomicBool::new(false));
    let poller = {
        let peak_kib = Arc::clone(&peak_kib);
        let stop = Arc::clone(&stop);
        thread::spawn(move || {
            // Reads after the child is reaped return `None` (its `/proc`
            // entry is gone); the only gap is pid reuse in the tiny
            // window between reap and `stop`, not worth guarding.
            while !stop.load(Ordering::Relaxed) {
                if let Some(kib) = read_vmhwm_kib(pid) {
                    peak_kib.fetch_max(kib, Ordering::Relaxed);
                }
                thread::sleep(POLL_INTERVAL);
            }
        })
    };

    let output = child
        .wait_with_output()
        .with_context(|| format!("failed to wait for {}", llama_bench.display()))?;
    stop.store(true, Ordering::Relaxed);
    let _ = poller.join();
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

    let max_kib = peak_kib.load(Ordering::Relaxed);
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

fn read_vmhwm_kib(pid: u32) -> Option<u64> {
    let status = fs::read_to_string(format!("/proc/{pid}/status")).ok()?;
    parse_vmhwm_kib(&status)
}

/// Pull the KiB value out of the `VmHWM:` line of `/proc/<pid>/status`
/// (`VmHWM:\t  347644 kB` → `347644`).
fn parse_vmhwm_kib(status: &str) -> Option<u64> {
    status
        .lines()
        .find_map(|line| line.strip_prefix("VmHWM:"))
        .and_then(|rest| rest.split_whitespace().next())
        .and_then(|v| v.parse::<u64>().ok())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_vmhwm_from_status() {
        let status = "\
Name:\tllama-bench
VmPeak:\t  512000 kB
VmHWM:\t  347644 kB
VmRSS:\t  301200 kB
";
        assert_eq!(parse_vmhwm_kib(status), Some(347_644));
    }

    #[test]
    fn returns_none_when_vmhwm_absent() {
        assert!(parse_vmhwm_kib("Name:\tllama-bench\nVmRSS:\t 100 kB\n").is_none());
    }
}
