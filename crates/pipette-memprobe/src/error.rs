//! What a [`Sampler`](crate::Sampler) fails with. Each OS has its own reasons,
//! so a variant exists only where its sampler can produce it.

#[cfg(any(target_os = "linux", target_os = "windows"))]
use std::io;

use crate::Watch;

/// A [`Watch`] this OS's sampler has no reader for, refused before anything
/// spawns rather than read as less than it asks for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error("{} has no reader for {watch}", std::env::consts::OS)]
pub struct UnsupportedError {
    pub watch: Watch,
}

/// Why [`Watch::supported`](crate::Watch::supported) refused a watch, before
/// anything spawns.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum WatchError {
    #[error(transparent)]
    Unsupported(#[from] UnsupportedError),
    /// Read as no device tree instead, a Jetson's GPU watch would look for a
    /// separate GPU pool its iGPU never bills.
    #[cfg(target_os = "linux")]
    #[error(
        "could not read /proc/device-tree/compatible, which tells a Jetson's GPU memory from a \
         card's: {kind}"
    )]
    DeviceTree { kind: io::ErrorKind },
}

/// Why [`Sampler::start`](crate::Sampler::start) failed. Empty where starting
/// cannot fail.
#[derive(Debug, thiserror::Error)]
pub enum StartError {
    #[cfg(target_os = "windows")]
    #[error("OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION) failed for pid {pid}")]
    Open { pid: u32, source: io::Error },
}

/// Why NVML could not be opened.
#[cfg(target_os = "linux")]
#[derive(Debug, thiserror::Error)]
pub enum NvmlError {
    #[error("dlopen: {0}")]
    Load(String),
    #[error("{0} is missing from libnvidia-ml.so.1")]
    Symbol(String),
    #[error("{call} failed: {message}")]
    Call { call: &'static str, message: String },
    #[error("NVML lists no GPU")]
    NoGpu,
}

/// Why `Nvml::processes` has no list.
#[cfg(target_os = "linux")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum NvmlReadError {
    /// Read without that list, the sum would leave its processes out.
    #[error("a GPU refused one of NVML's process lists")]
    Unanswered,
}

/// Why a run has no [`Peak`](crate::Peak).
#[derive(Debug, thiserror::Error)]
pub enum PeakError {
    #[error("the run was sampled for its observation, not its peak")]
    NotWatched,
    #[cfg(any(target_os = "linux", target_os = "android"))]
    #[error("could not read VmHWM from /proc/{pid}/status during the run")]
    NoVmHwm { pid: u32 },
    /// The stop found the process neither running nor exited and unreaped, so
    /// the reads before it would pass for the whole run without its last
    /// interval. Finding it proves the stop came first, not that the interval
    /// was read: Linux sets aside an exit figure no larger than pipette's own.
    #[cfg(any(target_os = "linux", target_os = "android", target_os = "macos"))]
    #[error(
        "pid {pid} was not read when the sampler stopped, so its peak is unknown: it was reaped \
         before the sampler stopped"
    )]
    UnreadAtStop { pid: u32 },
    /// Read as none instead, it would hide the GPU and NPU memory the watch
    /// exists to report.
    #[cfg(any(target_os = "linux", target_os = "android"))]
    #[error("could not read /proc/{pid}/maps and its dmabufs during the run")]
    NoAccelerators { pid: u32 },
    #[cfg(target_os = "linux")]
    #[error(
        "no DRM client of pid {pid} held GPU memory during the run: did the run reach the GPU?"
    )]
    NoDrmMemory { pid: u32 },
    #[cfg(target_os = "linux")]
    #[error(
        "pid {pid} holds an NVIDIA GPU open, but NVML attributed it no memory during the run: \
         the run did not reach the GPU, or NVML sees another pid namespace (pipette inside a \
         container)"
    )]
    NoGpuMemory { pid: u32 },
    /// Read as none instead, it would hide the GPU memory the watch exists to
    /// report.
    #[cfg(target_os = "linux")]
    #[error(
        "could not read the memory of pid {pid}'s DRM clients from fdinfo during the run: an fd \
         prints neither a drm-client-id with a drm-resident-, drm-memory- or drm-total-<region> \
         size it can read, nor amdgpu's older pdev, pasid and vram, gtt and cpu mem lines (i915 \
         before Linux 6.8 prints no memory, amdgpu before 5.14 prints nothing; or it is an \
         NVIDIA node whose driver sysfs hides?)"
    )]
    DrmUnread { pid: u32 },
    #[cfg(target_os = "linux")]
    #[error(
        "pid {pid} holds an NVIDIA GPU open, whose memory only NVML reads, and NVML is \
         unavailable"
    )]
    NvmlUnavailable { pid: u32, source: NvmlError },
    #[cfg(target_os = "linux")]
    #[error(
        "pid {pid} holds an NVIDIA GPU open, and on every read during the run a GPU refused one \
         of NVML's process lists, so its memory there is unknown"
    )]
    NvmlUnanswered { pid: u32 },
    #[cfg(target_os = "linux")]
    #[error(
        "NVML reports pid {pid}'s GPU memory as not available (MIG, vGPU, or no permission), \
         so it cannot be read"
    )]
    GpuMemoryUnavailable { pid: u32 },
    #[cfg(target_os = "linux")]
    #[error("could not list /proc/{pid}/fd during the run, so its GPU memory is unknown")]
    GpuUnread { pid: u32 },
    #[cfg(target_os = "macos")]
    #[error("the phys_footprint sampler thread panicked; max_host_bytes is unreliable")]
    SamplerPanicked,
    #[cfg(target_os = "macos")]
    #[error(
        "phys_footprint poller captured no samples for pid {pid}; max_host_bytes would be \
         meaningless"
    )]
    NoFootprint { pid: u32 },
    #[cfg(target_os = "windows")]
    #[error("the PDH poller thread panicked; the GPU and NPU peaks cannot be reported")]
    GpuPollerPanicked,
    #[cfg(target_os = "windows")]
    #[error("{call} failed (0x{code:08X}), so the run's GPU and NPU memory cannot be read")]
    Pdh { call: &'static str, code: u32 },
    /// A zero is a failure, not a measurement: `None` stays reserved for "not
    /// collected".
    #[cfg(target_os = "windows")]
    #[error(
        "PDH observed no GPU-committed bytes for pid {pid}: the run did not reach a GPU, \
         its driver bills the memory to no adapter, or the GPU runs as a compute-only \
         adapter, which reads as an NPU; the result would be mislabeled"
    )]
    NoGpuCommit { pid: u32 },
    #[cfg(target_os = "windows")]
    #[error(
        "PDH observed no NPU-committed bytes for pid {pid}: the run did not reach the NPU, \
         or its driver bills the memory to no adapter; the result would be mislabeled"
    )]
    NoNpuCommit { pid: u32 },
    #[cfg(target_os = "windows")]
    #[error("GetProcessMemoryInfo failed for pid {pid}")]
    MemoryInfo { pid: u32, source: io::Error },
    #[cfg(target_os = "windows")]
    #[error("GetProcessMemoryInfo reported a zero peak working set for pid {pid}")]
    ZeroWorkingSet { pid: u32 },
    #[cfg(target_os = "windows")]
    #[error("a PSAPI size does not fit its integer type")]
    Size(#[from] std::num::TryFromIntError),
}
