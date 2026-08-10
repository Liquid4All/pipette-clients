//! Peak memory benchmark — per-OS execution paths.
//!
//! Currently implemented for **macOS** (Metal probe), **Android**
//! (toybox `time -v` wrapper), **Windows** (PSAPI host counter, no GPU
//! probe yet), and **Linux** (`/proc` `VmHWM`, CPU flavors only).
//! Other flavors bail at the dispatcher.
//!
//! - [`macos`]   — `pipette-memprobe-metal` Metal probe (DYLD shim)
//!   gives in-process `[MTLDevice currentAllocatedSize]`. Host
//!   counter is `phys_footprint`.
//! - [`android`] — wraps `llama-bench` with a vendored `toybox time -v`
//!   binary; parses `Max RSS (KiB)` from its stderr summary. No GPU
//!   probe (CPU-only Android build).
//! - [`windows`] — reads `PROCESS_MEMORY_COUNTERS.PeakWorkingSetSize`
//!   via PSAPI for `max_host_bytes` and polls
//!   `\GPU Process Memory(pid_<PID>_*)\Total Committed` via PDH for
//!   `max_gpu_bytes`. PDH is GPU-API-agnostic so the same path covers
//!   every Windows flavor (Vulkan, HIP, SYCL, D3D12); no per-runtime
//!   in-process probe is needed.
//! - [`linux`]   — runs `llama-bench` directly and polls
//!   `/proc/<pid>/status` `VmHWM` (peak RSS) for `max_host_bytes`. No
//!   GPU probe; only CPU flavors dispatch here.
//!
//! GPU/accelerator Linux flavors and Custom flavors are not
//! implemented; the dispatcher's catch-all `bail!` covers them. A GPU
//! Linux flavor needs a DRM-fdinfo probe
//! (`pipette-memprobe::os_counters::linux`) before it can route to
//! [`linux`] without under-reporting — the methodology
//! (`pipette-mgmt/docs/methodology/peak-memory.md` §3.2)
//! describes the host counter to read on that platform.
//!
//! Entry [`run`] takes [`RunRequest`] + [`MaxMemoryUsage`]; per-OS modules
//! take paths + [`Params`] distilled from the typed body.
//!
//! Each per-OS module reads top-to-bottom for one purpose: collect
//! peak memory, emit `MaxMemoryUsage`. The mem benchmark does not
//! parse `llama-bench`'s JSON or pick a result row — those concerns
//! belong to the timing benchmarks; mem reports memory counters and
//! captures bench output verbatim into `extras.json` for operator
//! visibility.

use pipette_plan_types::reserved_flags::llamacpp_cli_stock_tools as reserved;
use pipette_plan_types::run::RunRequest;
use pipette_plan_types::Runtime;

use super::RunResponse;
use crate::bench;
use crate::models::require_gguf_text;
use crate::runtime_flags;
use crate::runtimes::require_llama_bench;

// LlamaCppFlavor is referenced only by the cfg-gated dispatcher arms
// below, so it's only needed when at least one per-OS module is built
// in. Without this gate, Linux builds emit `unused_imports`.
#[cfg(any(
    target_os = "macos",
    target_os = "android",
    target_os = "windows",
    target_os = "linux"
))]
use pipette_plan_types::LlamaCppFlavor;

#[cfg(target_os = "android")]
mod android;
#[cfg(target_os = "linux")]
mod linux;
#[cfg(target_os = "macos")]
mod macos;
#[cfg(target_os = "windows")]
mod pdh_poller;
#[cfg(target_os = "windows")]
mod windows;

/// Prefill shape for peak-memory llama-bench invocations.
#[cfg_attr(
    not(any(
        target_os = "macos",
        target_os = "android",
        target_os = "windows",
        target_os = "linux"
    )),
    allow(dead_code)
)]
#[derive(Debug, Clone, Copy)]
pub(super) struct Params {
    pub parameter_prefill_tokens: u32,
}

/// Peak-memory cell: bound `llama-bench` + GGUF text, typed [`MaxMemoryUsage`].
#[cfg_attr(
    not(any(
        target_os = "macos",
        target_os = "android",
        target_os = "windows",
        target_os = "linux"
    )),
    allow(unused_variables)
)]
pub fn run(req: &RunRequest) -> anyhow::Result<RunResponse> {
    let benchmark = req
        .benchmark
        .as_max_memory_usage()
        .map_err(anyhow::Error::from)?;
    let llama_bench = require_llama_bench(req)?;
    let model_path = require_gguf_text(req)?;
    let flags = runtime_flags::for_bench(req)?;
    let extra_flags = bench::args_for(&flags).build(reserved::MAX_MEMORY, "max_memory_usage")?;
    let params = Params {
        parameter_prefill_tokens: benchmark.parameter_prefill_tokens,
    };
    let Runtime::LlamacppCliStockTools(inner) = &req.runtime.bound else {
        anyhow::bail!(
            "max_memory_usage expected llamacpp_cli_stock_tools, got `{}`",
            req.runtime.bound.headless_token()
        );
    };
    let flavor = &inner.flavor;

    match flavor {
        #[cfg(target_os = "macos")]
        LlamaCppFlavor::MacosX64
        | LlamaCppFlavor::MacosArm64
        | LlamaCppFlavor::MacosArm64Kleidiai => {
            macos::run(&llama_bench, params, &model_path, &extra_flags, &flags)
        }

        #[cfg(target_os = "android")]
        LlamaCppFlavor::AndroidArm64Cpu => {
            android::run(&llama_bench, params, &model_path, &extra_flags, &flags)
        }

        // CPU flavors only: host RSS (`VmHWM`) is the whole picture.
        #[cfg(target_os = "linux")]
        LlamaCppFlavor::LinuxArm64Cpu
        | LlamaCppFlavor::LinuxX64Cpu
        | LlamaCppFlavor::LinuxS390xCpu => {
            linux::run(&llama_bench, params, &model_path, &extra_flags, &flags)
        }

        #[cfg(target_os = "windows")]
        LlamaCppFlavor::WindowsX64Vulkan
        | LlamaCppFlavor::WindowsX64Hip
        | LlamaCppFlavor::WindowsX64Sycl
        | LlamaCppFlavor::WindowsX64Cuda124
        | LlamaCppFlavor::WindowsX64Cuda131
        | LlamaCppFlavor::WindowsX64Cpu
        | LlamaCppFlavor::WindowsArm64OpenclAdreno
        | LlamaCppFlavor::WindowsArm64Cpu => windows::run(
            &llama_bench,
            flavor,
            params,
            &model_path,
            &extra_flags,
            &flags,
        ),

        flavor => anyhow::bail!(
            "max_memory_usage benchmark is only implemented for macOS, \
             Android, Windows, and Linux CPU flavors today; cannot run \
             flavor {flavor:?} on this host"
        ),
    }
}
