//! The Windows sampler: two channels on the process, both read from the parent
//! (no DLL injection, no environment, no registration):
//!
//! 1. The host peak via PSAPI: `PROCESS_MEMORY_COUNTERS.PeakWorkingSetSize`, the
//!    kernel's lifetime peak working set (Task Manager's "Peak Working Set"),
//!    read at stop through a `PROCESS_QUERY_LIMITED_INFORMATION` handle opened at
//!    start. The kernel keeps the process object, and with it the counter, while
//!    any handle refers to it, so the read works after the process is reaped.
//! 2. The GPU and NPU peaks via PDH ([`crate::pdh`]), on the accelerator
//!    watches: what WDDM bills the process per adapter.
//!
//! Whether the device bytes sit inside the working set is the driver's choice:
//! a discrete GPU's VRAM never does, AMD's iGPU allocations mostly do not, and
//! on the Intel Arc iGPU and NPU measured they do. So the peaks are reported
//! apart, never summed.
//!
//! [`Watch::Observe`] makes the same PSAPI read, which costs the run nothing
//! before the stop; a process it cannot open goes unobserved.

use std::os::windows::io::{AsRawHandle, FromRawHandle, OwnedHandle};

use windows_sys::Win32::System::{
    ProcessStatus::{GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS},
    Threading::{OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION},
};

use pipette_plan_types::MemoryObservation;

use super::pdh::{AdapterPeaks, PdhGpuMemoryPoller, spawn_pdh_gpu_memory_poller};
use super::{Peak, PeakError, StartError, SupportedWatch, UnsupportedError, Watch};

/// The process's PSAPI peak working set, and on an accelerator watch its PDH
/// GPU and NPU peaks.
pub struct Sampler {
    pid: u32,
    watch: Watch,
    /// Held from the start so the counter can be read after the process exits.
    /// `None` when an observation could not open the process.
    process: Option<OwnedHandle>,
    gpu: Option<PdhGpuMemoryPoller>,
}

impl Sampler {
    pub(crate) const OBSERVATION: Watch = Watch::Observe;

    /// Every watch: PDH reads each adapter, GPU or NPU.
    pub(crate) fn reads(watch: Watch) -> Result<Watch, UnsupportedError> {
        match watch {
            Watch::Observe
            | Watch::Peak
            | Watch::PeakWithGpu
            | Watch::PeakWithNpu
            | Watch::PeakWithAccelerators => Ok(watch),
        }
    }

    pub fn start(pid: u32, SupportedWatch(watch): SupportedWatch) -> Result<Self, StartError> {
        let (process, gpu) = match watch {
            Watch::Observe => (open(pid).ok(), None),
            Watch::Peak => (Some(open(pid)?), None),
            Watch::PeakWithGpu | Watch::PeakWithNpu | Watch::PeakWithAccelerators => {
                (Some(open(pid)?), Some(spawn_pdh_gpu_memory_poller(pid)))
            }
        };
        Ok(Self {
            pid,
            watch,
            process,
            gpu,
        })
    }

    /// The run's observation, and its peak when watched for one.
    pub fn stop(self) -> (MemoryObservation, Result<Peak, PeakError>) {
        let Some(process) = self.process else {
            return (MemoryObservation::default(), Err(PeakError::NotWatched));
        };
        let peak = peak(self.pid, self.watch, &process, self.gpu);
        // No swap term: `PeakWorkingSetSize` is resident-only and PSAPI exposes
        // no paged-out watermark.
        let observation = peak.as_ref().map_or_else(
            |_| MemoryObservation::default(),
            |peak| MemoryObservation::host_only(peak.host_bytes),
        );
        match self.watch {
            Watch::Observe => (observation, Err(PeakError::NotWatched)),
            Watch::Peak | Watch::PeakWithGpu | Watch::PeakWithNpu | Watch::PeakWithAccelerators => {
                (observation, peak)
            }
        }
    }
}

fn open(pid: u32) -> Result<OwnedHandle, StartError> {
    // SAFETY: `OpenProcess` takes no pointers; a null return is handled below.
    let handle = unsafe { OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid) };
    if handle.is_null() {
        return Err(StartError::Open {
            pid,
            source: std::io::Error::last_os_error(),
        });
    }
    // SAFETY: `OpenProcess` returned a handle nothing else owns.
    Ok(unsafe { OwnedHandle::from_raw_handle(handle) })
}

fn peak(
    pid: u32,
    watch: Watch,
    process: &OwnedHandle,
    gpu: Option<PdhGpuMemoryPoller>,
) -> Result<Peak, PeakError> {
    // Stopped before anything can fail, so the PDH query is closed on every path.
    let adapters = gpu
        .map(PdhGpuMemoryPoller::stop_and_join)
        .transpose()?
        .unwrap_or_default();
    required_pool(pid, watch, adapters)?;
    // SAFETY: `PROCESS_MEMORY_COUNTERS` is plain integers, so all-zero is valid.
    let mut counters: PROCESS_MEMORY_COUNTERS = unsafe { std::mem::zeroed() };
    counters.cb = u32::try_from(std::mem::size_of::<PROCESS_MEMORY_COUNTERS>())?;
    // SAFETY: `process` is a live query handle and `counters` is correctly sized.
    let ok =
        unsafe { GetProcessMemoryInfo(process.as_raw_handle(), &raw mut counters, counters.cb) };
    if ok == 0 {
        return Err(PeakError::MemoryInfo {
            pid,
            source: std::io::Error::last_os_error(),
        });
    }
    let host_bytes = u64::try_from(counters.PeakWorkingSetSize)?;
    if host_bytes == 0 {
        return Err(PeakError::ZeroWorkingSet { pid });
    }
    Ok(Peak {
        host_bytes,
        gpu_bytes: (adapters.gpu_bytes > 0).then_some(adapters.gpu_bytes),
        npu_bytes: (adapters.npu_bytes > 0).then_some(adapters.npu_bytes),
    })
}

/// A watch for the device a run was put on fails when that device showed
/// nothing: its reading would be mislabeled.
fn required_pool(pid: u32, watch: Watch, adapters: AdapterPeaks) -> Result<(), PeakError> {
    match watch {
        Watch::PeakWithGpu if adapters.gpu_bytes == 0 => Err(PeakError::NoGpuCommit { pid }),
        Watch::PeakWithNpu if adapters.npu_bytes == 0 => Err(PeakError::NoNpuCommit { pid }),
        Watch::Observe
        | Watch::Peak
        | Watch::PeakWithGpu
        | Watch::PeakWithNpu
        | Watch::PeakWithAccelerators => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[rstest::rstest]
    #[case::gpu_held(Watch::PeakWithGpu, 4096, 0, None)]
    #[case::gpu_held_none(Watch::PeakWithGpu, 0, 4096, Some("no GPU-committed"))]
    #[case::npu_held(Watch::PeakWithNpu, 0, 4096, None)]
    #[case::npu_held_none(Watch::PeakWithNpu, 4096, 0, Some("no NPU-committed"))]
    #[case::either(Watch::PeakWithAccelerators, 0, 0, None)]
    fn a_watch_requires_its_devices_pool(
        #[case] watch: Watch,
        #[case] gpu_bytes: u64,
        #[case] npu_bytes: u64,
        #[case] refused: Option<&str>,
    ) {
        let checked = required_pool(
            42,
            watch,
            AdapterPeaks {
                gpu_bytes,
                npu_bytes,
            },
        );
        match refused {
            None => assert!(checked.is_ok(), "{checked:?}"),
            Some(cause) => assert!(
                checked
                    .as_ref()
                    .is_err_and(|err| err.to_string().contains(cause)),
                "{checked:?}"
            ),
        }
    }
}
