//! Windows measurement path — wraps `llama-bench.exe` with two
//! concurrent measurement channels, both purely parent-side (no DLL
//! injection, no env-var dance, no layer registration):
//!
//! 1. **`max_host_bytes` via PSAPI** — `PROCESS_MEMORY_COUNTERS.PeakWorkingSetSize`,
//!    read post-exit through a separately-held `PROCESS_QUERY_LIMITED_INFORMATION`
//!    handle. This is the kernel's lifetime peak working-set, the
//!    counter Task Manager calls "Peak Working Set" and what
//!    methodology §3.3 names as the host source.
//! 2. **`max_gpu_bytes` via PDH** — a polling thread samples
//!    `\GPU Process Memory(pid_<PID>_*)\Total Committed` every 20 ms
//!    and tracks the running maximum. PDH is GPU-API-agnostic, so
//!    the same path covers Vulkan, HIP, SYCL, and D3D12 flavors
//!    without any per-runtime DLL injection. `Total Committed` is the
//!    per-tick joint sum of Dedicated + Shared usage — the right
//!    "what does the OS attribute to this process" reading. See
//!    `pdh_poller.rs` for empirical justification (verified against
//!    `Dedicated`, `Shared`, `Local`, and `Non Local` counters on
//!    Strix Halo).
//!
//! Host and GPU are separate pools on every Windows flavor: PSAPI's
//! working-set counter does not include GPU-driver-managed memory
//! (GPU-allocator-managed bytes on AMD UMA, dedicated VRAM on
//! discrete). The wire schema's two peaks are disjoint by physical
//! structure.
//!
//! `WindowsArm64Cpu` (no GPU) skips the PDH poller entirely;
//! `max_gpu_bytes` is `null` on the wire.
//!
//! ## Why PDH and not an in-process Vulkan layer
//!
//! Earlier versions of this path injected a Vulkan layer DLL into
//! `llama-bench` and hooked `vkAllocateMemory`/`vkFreeMemory`. That
//! approach was byte-exact to the Vulkan allocator but had three
//! material drawbacks:
//!
//! - **Vulkan-only**: each GPU API (HIP, SYCL, D3D12, …) would have
//!   needed its own interposer crate and DLL.
//! - **Elevation-sensitive**: the Vulkan loader's `loader_secure_getenv`
//!   filters `VK_LAYER_PATH` and `VK_INSTANCE_LAYERS` under High
//!   integrity level (e.g. SSH-as-admin on Windows OpenSSH), so the
//!   layer had to be registered via `HKLM\SOFTWARE\Khronos\Vulkan\
//!   ExplicitLayers` — a system-wide write visible to every Vulkan
//!   app on the machine for the duration of the bench, with stale
//!   entries surviving parent crashes.
//! - **Build complexity**: required a C++ toolchain at every build
//!   site (MSVC on CI / dev hosts) and per-target compile of the
//!   layer DLL.
//!
//! PDH eliminates all three: pure Win32 API from the parent process,
//! works at any integrity level, no system-state pollution, no C++
//! build dependency. The ~3% drift on UMA is the same magnitude as
//! the driver-state overhead the in-process probe shows above the
//! runtime's announced sum — within the methodology's existing
//! tolerance for "what the GPU allocator reports."
//!
//! See `pipette-mgmt/docs/methodology/peak-memory.md` §3.3 for the
//! host-counter methodology; the PDH-as-`max_gpu_bytes` choice is a
//! deliberate departure from the methodology's "PDH stays sidecar"
//! recommendation, justified by empirical match to the Vulkan
//! allocator on Strix Halo (see commit message for the validation
//! table).

use std::{
    path::Path,
    process::{Command, Stdio},
};

use anyhow::Context;
use windows_sys::Win32::{
    Foundation::{CloseHandle, HANDLE},
    System::{
        ProcessStatus::{GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS},
        Threading::{
            OpenProcess, TerminateProcess, PROCESS_QUERY_LIMITED_INFORMATION, PROCESS_TERMINATE,
        },
    },
};

use pipette_plan_types::result::{BenchmarkResultData, MemoryObservation};
use pipette_plan_types::{LlamaCppFlavor, RuntimeFlags};
use pipette_subprocess::{argv, echo_info};

use super::super::RunResponse;
use super::pdh_poller::spawn_pdh_gpu_memory_poller;
use super::Params;
use crate::common::{
    apply_dylib_search_env, deadline_error_message, spawn_timeout_killer, MAX_MEMORY_USAGE_TIMEOUT,
};

/// RAII wrapper around a Win32 `HANDLE` so every early-return path
/// (`wait_with_output` Err, `GetProcessMemoryInfo` Err, deadline fire,
/// status fail) closes the duplicated probe handle. Without this, a
/// `wait_with_output` Err would leak the handle on the `?` operator,
/// which matters in long-running parents (e.g. `pipette-plan` running
/// many benchmarks in sequence).
struct OwnedProcessHandle(HANDLE);

impl Drop for OwnedProcessHandle {
    fn drop(&mut self) {
        if !self.0.is_null() {
            // SAFETY: we own this handle (returned by `OpenProcess`)
            // and Drop runs at most once per instance.
            unsafe {
                let _ = CloseHandle(self.0);
            }
        }
    }
}

pub(super) fn run(
    llama_bench: &Path,
    flavor: &pipette_plan_types::LlamaCppFlavor,
    params: Params,
    model_path: &Path,
    extra_flags: &[String],
    flags: &RuntimeFlags,
) -> anyhow::Result<RunResponse> {
    // 1. Build the llama-bench command. Same shape as the macOS /
    //    Android paths so the captured `outcome.stderr` is directly
    //    comparable across platforms. `--n-gen 1` exercises one
    //    decode step; see methodology §2.4 for why prefill alone
    //    misses decode-path kernels + sampling buffers.
    let mut cmd = Command::new(llama_bench);
    apply_dylib_search_env(&mut cmd, llama_bench);
    cmd.arg("--output").arg("json");
    cmd.arg("--model").arg(model_path);
    cmd.args(extra_flags);
    cmd.arg("--n-prompt")
        .arg(params.parameter_prefill_tokens.to_string())
        .arg("--n-gen")
        .arg("1");
    cmd.stdout(Stdio::piped()).stderr(Stdio::piped());

    let preview = argv(&cmd);
    log::info!("measuring peak memory via PSAPI + PDH");
    echo_info(&cmd);

    // 2. Spawn, then immediately duplicate the process handle via
    //    OpenProcess. std::process::Child closes its own HANDLE on
    //    reap inside `wait_with_output`, but the PROCESS_MEMORY_COUNTERS
    //    snapshot we want is the *post-exit* peak — the kernel keeps
    //    the EPROCESS object (and its lifetime-max counters) alive
    //    while any handle still refers to it.
    let child = cmd
        .spawn()
        .with_context(|| format!("failed to spawn {}", llama_bench.display()))?;
    let pid = child.id();
    let probe_handle = unsafe { OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid) };
    if probe_handle.is_null() {
        let err = std::io::Error::last_os_error();
        // Still need to drain pipes + reap the child to avoid a
        // zombie; we just have no peak to report.
        let _ = child.wait_with_output();
        anyhow::bail!(
            "OpenProcess(child={}) failed: {err}",
            "PROCESS_QUERY_LIMITED_INFORMATION"
        );
    }
    let probe_handle = OwnedProcessHandle(probe_handle);

    // 3. Whether to poll the GPU counter at all. ARM64-CPU has no
    //    GPU runtime; x64-CPU is likewise host-only. The other Windows
    //    flavors do (Vulkan / HIP / SYCL / CUDA / OpenCL), and PDH
    //    `\GPU Process Memory\Total Committed` is API-agnostic — it
    //    surfaces whatever the WDDM driver reports the process is
    //    using.
    let gpu_poller = match flavor {
        LlamaCppFlavor::WindowsArm64Cpu | LlamaCppFlavor::WindowsX64Cpu => None,
        _ => Some(spawn_pdh_gpu_memory_poller(pid)),
    };

    // 4. Deadline killer: opens its own PROCESS_TERMINATE handle on
    //    timeout fire (closes-over only the pid, which is Send).
    let killer = spawn_timeout_killer(MAX_MEMORY_USAGE_TIMEOUT, move || unsafe {
        let h = OpenProcess(PROCESS_TERMINATE, 0, pid);
        if !h.is_null() {
            TerminateProcess(h, 1);
            CloseHandle(h);
        }
    });

    let output = child
        .wait_with_output()
        .context("failed to wait for llama-bench")?;
    let killer_fired = killer.fired();
    drop(killer);

    // 5. Error ordering matches macos.rs / android.rs:
    //    deadline-fire first (most informative root cause), then
    //    GPU poller read, then PSAPI read, then exit-status.
    if killer_fired {
        // Stop the GPU poller (peak is discarded — we're failing).
        if let Some(poller) = gpu_poller {
            let _ = poller.stop_and_join();
        }
        anyhow::bail!(
            "{}",
            deadline_error_message(
                MAX_MEMORY_USAGE_TIMEOUT,
                &String::from_utf8_lossy(&output.stderr)
            )
        );
    }

    let max_gpu_bytes = match gpu_poller {
        Some(poller) => Some(
            poller
                .stop_and_join()
                .context("PDH GPU-memory poller failed; max_gpu_bytes is unreliable")?,
        ),
        None => None,
    };

    // 6. Read PROCESS_MEMORY_COUNTERS once. PeakWorkingSetSize is the
    //    kernel's lifetime maximum for this process.
    let mut counters: PROCESS_MEMORY_COUNTERS = unsafe { std::mem::zeroed() };
    counters.cb = std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32;
    let ok = unsafe { GetProcessMemoryInfo(probe_handle.0, &mut counters, counters.cb) };
    if ok == 0 {
        let err = std::io::Error::last_os_error();
        anyhow::bail!("GetProcessMemoryInfo failed after llama-bench exit: {err}");
    }

    if !output.status.success() {
        anyhow::bail!(
            "llama-bench failed: {}; stderr tail:\n{}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        );
    }

    let max_host_bytes = counters.PeakWorkingSetSize as u64;
    log::debug!(
        "windows max_host_bytes={max_host_bytes} (PSAPI PeakWorkingSetSize); \
         max_gpu_bytes={max_gpu_bytes:?} (PDH Total Committed)",
    );

    // 7. Capture llama-bench output verbatim so the runtime's announced
    //    buffer sizes (`load_tensors:`, `llama_kv_cache:`,
    //    `sched_reserve:`) land in `extras.json` for cross-checking.
    let stdout = String::from_utf8(output.stdout).context("llama-bench returned non-utf8")?;
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();

    Ok(RunResponse {
        executable: Some(llama_bench.display().to_string()),
        command: preview,
        runtime_flags: Some(flags.clone()),
        // No swap term: `PeakWorkingSetSize` is resident-only and PSAPI exposes
        // no paged-out watermark yet.
        memory: MemoryObservation::host_only(max_host_bytes),
        ..RunResponse::new(
            BenchmarkResultData::MaxMemoryUsage {
                max_host_bytes,
                max_gpu_bytes,
                // No platform implements per-process NPU memory yet.
                max_npu_bytes: None,
            },
            stdout,
            stderr,
        )
    })
}
