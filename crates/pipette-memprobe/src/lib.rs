//! A process's memory, read by pid from the parent: one [`Sampler`] per OS,
//! started as the process spawns and stopped with what it saw ([`Watch`]).
//!
//! The observation ([`pipette_plan_types::MemoryObservation`]) is the host peak,
//! with the swap term beside it on Linux and Android; an observation alone reads
//! nothing on an OS with no sampler. The peak ([`Peak`]) is a memory benchmark's
//! result, and never zero: a sampler that read nothing fails instead
//! ([`PeakError`]).
//!
//! Every [`Watch`] exists on every OS. [`Watch::supported`] says whether this
//! OS's sampler reads it, refusing one it does not with an
//! [`UnsupportedError`] (or, on Linux, one whose device tree it cannot read,
//! with a [`WatchError`]), and returns the [`SupportedWatch`] that
//! [`Sampler::start`] takes, so an engine asks before anything spawns. Every
//! OS reads [`SupportedWatch::OBSERVE`].
//!
//! | OS | host | swap term | GPU, NPU | source |
//! |---|---|---|---|---|
//! | Linux | yes | yes | `Watch::PeakWithGpu` | `/proc/<pid>/status`, sampled (`Footprint`); the DRM clients' fdinfo for a DRM GPU; NVML for an NVIDIA GPU |
//! | Android | yes | yes | `Watch::PeakWithAccelerators` | `/proc/<pid>/status`, sampled; `/proc/<pid>/maps` and the open dmabufs for the GPU and NPU |
//! | macOS | yes | no | inside the host figure: `Watch::PeakWithAccelerators` reads it | `phys_footprint`, which already bills compressed pages and, on unified memory, the Metal allocations |
//! | Windows | yes | no | `Watch::PeakWithGpu`, `PeakWithNpu`, `PeakWithAccelerators` | PSAPI's `PeakWorkingSetSize`, read once at stop; PDH per adapter for the GPU and NPU |
//!
//! `metal` is the macOS Metal allocator shim, a diagnostic beside the peak.
//! [`cgroup`] reads a cgroup's memory on Linux, for a server that runs as
//! several processes, such as a container's or a systemd scope's.
//!
//! Methodology per OS: `docs/methodology/peak-memory-*.md`.

#[cfg(target_os = "linux")]
pub mod amd;
pub mod cgroup;
#[cfg(target_os = "linux")]
mod drm;
mod error;
#[cfg(any(target_os = "linux", target_os = "android"))]
mod linux;
#[cfg(target_os = "macos")]
mod macos;
#[cfg(target_os = "macos")]
pub mod metal;
#[cfg(target_os = "linux")]
mod nvml;
#[cfg(target_os = "windows")]
mod pdh;
#[cfg(target_os = "windows")]
mod windows;

#[cfg(target_os = "linux")]
pub use error::{NvmlError, NvmlReadError};
pub use error::{PeakError, StartError, UnsupportedError, WatchError};
#[cfg(any(target_os = "linux", target_os = "android"))]
use linux::Reads;
#[cfg(any(target_os = "linux", target_os = "android"))]
pub use linux::{Footprint, Sampler};
#[cfg(target_os = "macos")]
pub use macos::Sampler;
#[cfg(target_os = "linux")]
pub use nvml::{NotAvailable, Pids};
#[cfg(target_os = "windows")]
pub use windows::Sampler;

/// How a [`Sampler`] reads a run. The observation polls slowly, since a
/// benchmark that reports time must not move because it was watched; a peak
/// polls fast enough to catch a short-lived high. Each exists on every OS;
/// what one reads there, if anything, is its sampler's ([`Watch::supported`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Watch {
    /// The observation alone, read on every OS. Starting one never fails, so
    /// watching cannot fail a run.
    Observe,
    /// The peak, and the observation.
    Peak,
    /// The peak with the GPU's, for a process that runs on one, so a GPU that
    /// shows no memory fails the peak: on Windows what WDDM bills the process on
    /// each GPU adapter (and on an NPU, if it held some), on Linux what the
    /// DRM drivers bill its clients plus, on an NVIDIA GPU, whose driver bills
    /// none there, what NVML attributes to it. macOS has no reader for it: its
    /// unified memory bills Metal's allocations to the host figure, which cannot
    /// tell whether the run reached the GPU, so an engine on unified memory
    /// asks for [`Watch::Peak`]. Android has no reader for it either.
    PeakWithGpu,
    /// The peak with the NPU's, for a process that runs on one: what WDDM bills
    /// it on each compute-only adapter (and on a GPU, if it held some). An NPU
    /// that shows no memory fails the peak. Only Windows reads it.
    PeakWithNpu,
    /// The peak with whatever GPU and NPU memory the OS bills the process, for a
    /// run that may hold memory on either device, so none is required. On
    /// Windows WDDM's per-adapter billing; on Android the Adreno memory the
    /// KGSL driver maps into the process, and the dmabufs it holds, which on
    /// the Snapdragon build are the Hexagon NPU's buffers; on macOS the host
    /// figure, which holds whatever Metal allocates. Linux reads no NPU, so it
    /// has no reader for it.
    PeakWithAccelerators,
}

impl Watch {
    /// This watch, checked for the start, or why this OS's sampler has no
    /// reader for it. An engine asks before anything spawns.
    pub fn supported(self) -> Result<SupportedWatch, WatchError> {
        Ok(SupportedWatch(Sampler::reads(self)?))
    }
}

/// In words, for a refusal that reaches a job's record.
impl std::fmt::Display for Watch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Observe => "the host's memory",
            Self::Peak => "the host's peak memory",
            Self::PeakWithGpu => "a GPU's own memory",
            Self::PeakWithNpu => "an NPU's own memory",
            Self::PeakWithAccelerators => "whatever GPU or NPU memory a run holds",
        })
    }
}

/// A [`Watch`] this OS's sampler reads, as it reads it: from
/// [`Watch::supported`], or [`SupportedWatch::OBSERVE`], which every OS reads.
/// So [`Sampler::start`] is never handed a watch it would refuse after the
/// process spawned:
///
/// ```compile_fail
/// let sampler = pipette_memprobe::Sampler::start(1, pipette_memprobe::Watch::Peak);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SupportedWatch(Reads);

impl SupportedWatch {
    /// The observation, read on every OS.
    pub const OBSERVE: Self = Self(Sampler::OBSERVATION);
}

/// What a sampler reads for a watch: the watch itself where the sampler
/// matches on it at the start.
#[cfg(not(any(target_os = "linux", target_os = "android")))]
type Reads = Watch;

/// What a peak sampler saw.
#[derive(Clone, Copy, Debug)]
pub struct Peak {
    pub host_bytes: u64,
    /// `None` unless the watch reads a separate GPU pool and it held memory.
    pub gpu_bytes: Option<u64>,
    /// `None` unless the watch reads an NPU and it held memory.
    pub npu_bytes: Option<u64>,
    /// The `/proc` terms behind the peak, for a swap-aware reading.
    #[cfg(any(target_os = "linux", target_os = "android"))]
    pub footprint: Footprint,
}

#[cfg(any(target_os = "linux", target_os = "android"))]
impl Peak {
    /// This peak with its host figure swap-aware ([`Footprint::peak_ram_kib`]):
    /// weights in anonymous memory are swap-eligible, so `host_bytes`, the
    /// resident watermark, misses what the kernel swapped out while the run
    /// still needed it (`docs/methodology/peak-memory-android.md`).
    #[must_use]
    pub fn counting_swap(self) -> Self {
        Self {
            host_bytes: self.footprint.peak_ram_kib().saturating_mul(1024),
            ..self
        }
    }
}

/// The sampler of an OS with none: it observes nothing and refuses a peak.
#[cfg(not(any(
    target_os = "linux",
    target_os = "android",
    target_os = "macos",
    target_os = "windows"
)))]
pub struct Sampler;

#[cfg(not(any(
    target_os = "linux",
    target_os = "android",
    target_os = "macos",
    target_os = "windows"
)))]
impl Sampler {
    pub(crate) const OBSERVATION: Reads = Watch::Observe;

    /// Only the observation, which reads nothing here.
    pub(crate) fn reads(watch: Watch) -> Result<Reads, UnsupportedError> {
        match watch {
            Watch::Observe => Ok(watch),
            Watch::Peak | Watch::PeakWithGpu | Watch::PeakWithNpu | Watch::PeakWithAccelerators => {
                Err(UnsupportedError { watch })
            }
        }
    }

    pub fn start(_pid: u32, _watch: SupportedWatch) -> Result<Self, StartError> {
        Ok(Self)
    }

    pub fn stop(
        self,
    ) -> (
        pipette_plan_types::MemoryObservation,
        Result<Peak, PeakError>,
    ) {
        (
            pipette_plan_types::MemoryObservation::default(),
            Err(PeakError::NotWatched),
        )
    }
}

#[cfg(all(test, any(target_os = "linux", target_os = "android")))]
#[path = "../tests/exiting_child/mod.rs"]
mod exiting_child;

/// A child that idles until dropped, then is killed and reaped, so an early
/// assertion cannot leave it running.
#[cfg(test)]
pub(crate) struct IdleChild(std::process::Child);

#[cfg(test)]
impl IdleChild {
    #[expect(clippy::disallowed_methods, reason = "its stdout goes to null")]
    pub(crate) fn spawn() -> anyhow::Result<Self> {
        let mut command = if cfg!(windows) {
            let mut ping = std::process::Command::new("ping");
            ping.args(["-n", "30", "127.0.0.1"]);
            ping
        } else {
            let mut sleep = std::process::Command::new("sleep");
            sleep.arg("30");
            sleep
        };
        Ok(Self(command.stdout(std::process::Stdio::null()).spawn()?))
    }

    pub(crate) fn id(&self) -> u32 {
        self.0.id()
    }
}

#[cfg(test)]
impl Drop for IdleChild {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

#[cfg(test)]
mod tests {
    use pipette_plan_types::MemoryObservation;

    use super::*;

    /// The host figure is the larger of the resident watermark and the largest
    /// resident-plus-swap sample; the GPU and NPU figures pass through.
    #[cfg(any(target_os = "linux", target_os = "android"))]
    #[rstest::rstest]
    #[case::resident(8, 6, 8)]
    #[case::swapped(5, 9, 9)]
    fn counting_swap_takes_the_larger_term(
        #[case] peak_rss_kib: u64,
        #[case] peak_committed_kib: u64,
        #[case] expected_kib: u64,
    ) {
        let peak = Peak {
            host_bytes: peak_rss_kib * 1024,
            gpu_bytes: Some(7),
            npu_bytes: None,
            footprint: Footprint {
                peak_rss_kib,
                peak_committed_kib,
                ..Footprint::default()
            },
        }
        .counting_swap();
        assert_eq!(peak.host_bytes, expected_kib * 1024);
        assert_eq!(peak.gpu_bytes, Some(7));
        assert_eq!(peak.npu_bytes, None);
    }

    /// The observation must never invent a figure: populated, or absent with no
    /// swap term either.
    #[test]
    fn an_observation_is_either_populated_or_absent_but_never_zero() -> anyhow::Result<()> {
        let child = IdleChild::spawn()?;
        let sampler = Sampler::start(child.id(), SupportedWatch::OBSERVE)?;
        std::thread::sleep(std::time::Duration::from_millis(350));
        let (observed, peak) = sampler.stop();

        assert!(peak.is_err(), "an observation has no peak: {peak:?}");
        if let Some(bytes) = observed.max_host_bytes {
            assert!(bytes > 0, "a reported peak must never be zero");
        } else {
            assert_eq!(
                observed,
                MemoryObservation::default(),
                "an OS with no host peak must report no swap either"
            );
        }
        Ok(())
    }

    #[cfg(any(
        target_os = "linux",
        target_os = "android",
        target_os = "macos",
        target_os = "windows"
    ))]
    #[test]
    fn a_live_process_is_observed() -> anyhow::Result<()> {
        let child = IdleChild::spawn()?;
        let sampler = Sampler::start(child.id(), SupportedWatch::OBSERVE)?;
        std::thread::sleep(std::time::Duration::from_millis(350));
        let (observed, peak) = sampler.stop();

        assert!(peak.is_err(), "an observation has no peak: {peak:?}");
        assert!(observed.max_host_bytes.is_some(), "{observed:?}");
        Ok(())
    }

    /// Watching cannot fail a run: an observation of a pid that cannot be read
    /// starts, and observes nothing.
    #[test]
    fn an_unreadable_process_is_observed_as_nothing() -> anyhow::Result<()> {
        let (observed, _) = Sampler::start(u32::MAX, SupportedWatch::OBSERVE)?.stop();
        assert_eq!(observed, MemoryObservation::default());
        Ok(())
    }

    /// Whether `os` (as `std::env::consts::OS` names it) reads the watch.
    fn read_on(watch: Watch, os: &str) -> bool {
        match watch {
            Watch::Observe => true,
            Watch::Peak => matches!(os, "linux" | "android" | "macos" | "windows"),
            Watch::PeakWithGpu => matches!(os, "linux" | "windows"),
            Watch::PeakWithNpu => os == "windows",
            Watch::PeakWithAccelerators => matches!(os, "android" | "macos" | "windows"),
        }
    }

    #[rstest::rstest]
    #[case::observe(Watch::Observe)]
    #[case::peak(Watch::Peak)]
    #[case::gpu(Watch::PeakWithGpu)]
    #[case::npu(Watch::PeakWithNpu)]
    #[case::accelerators(Watch::PeakWithAccelerators)]
    fn each_os_reads_its_watches(#[case] watch: Watch) {
        let os = std::env::consts::OS;
        let refused = (!read_on(watch, os)).then_some(UnsupportedError { watch }.into());
        assert_eq!(watch.supported().err(), refused);
    }

    #[test]
    fn the_observation_is_read_on_every_os() {
        assert_eq!(Watch::Observe.supported(), Ok(SupportedWatch::OBSERVE));
    }

    #[test]
    fn a_refusal_names_the_watch_in_words_and_the_os() {
        let refused = UnsupportedError {
            watch: Watch::PeakWithNpu,
        };
        assert_eq!(
            refused.to_string(),
            format!(
                "{} has no reader for an NPU's own memory",
                std::env::consts::OS
            )
        );
    }

    #[rstest::rstest]
    #[case::observe(Watch::Observe)]
    #[case::peak(Watch::Peak)]
    #[case::gpu(Watch::PeakWithGpu)]
    #[case::npu(Watch::PeakWithNpu)]
    #[case::accelerators(Watch::PeakWithAccelerators)]
    fn a_sampler_starts_on_every_watch_this_os_reads(#[case] watch: Watch) -> anyhow::Result<()> {
        let Ok(checked) = watch.supported() else {
            return Ok(());
        };
        let child = IdleChild::spawn()?;
        let sampler = Sampler::start(child.id(), checked)?;
        drop(sampler.stop());
        Ok(())
    }

    /// The process holds no GPU or NPU memory; on macOS a watch for any device
    /// reads the footprint alone.
    #[cfg(any(
        target_os = "linux",
        target_os = "android",
        target_os = "macos",
        target_os = "windows"
    ))]
    #[rstest::rstest]
    #[case::host(Watch::Peak)]
    #[cfg_attr(
        any(target_os = "android", target_os = "macos"),
        case::with_gpu_and_npu(Watch::PeakWithAccelerators)
    )]
    fn a_live_process_has_a_peak(#[case] watch: Watch) -> anyhow::Result<()> {
        let child = IdleChild::spawn()?;
        let sampler = Sampler::start(child.id(), watch.supported()?)?;
        std::thread::sleep(std::time::Duration::from_millis(120));
        let (observed, peak) = sampler.stop();

        let peak = peak?;
        assert!(peak.host_bytes > 0);
        assert_eq!((peak.gpu_bytes, peak.npu_bytes), (None, None));
        assert!(
            observed.max_host_bytes.is_some(),
            "a peak is observed too: {observed:?}"
        );
        Ok(())
    }

    /// A process on no GPU has no GPU peak, so the watch for one fails: it
    /// holds no DRM client and no NVIDIA device, whatever NVML says. On a
    /// Jetson the watch reads the resident set alone, so it has a peak.
    #[cfg(target_os = "linux")]
    #[test]
    fn a_process_off_the_gpu_fails_a_gpu_watch() -> anyhow::Result<()> {
        let child = IdleChild::spawn()?;
        let sampler = Sampler::start(child.id(), Watch::PeakWithGpu.supported()?)?;
        std::thread::sleep(std::time::Duration::from_millis(120));
        let (_, peak) = sampler.stop();
        if linux::gpu_memory_is_resident()? {
            assert!(
                matches!(
                    peak,
                    Ok(Peak {
                        gpu_bytes: None,
                        ..
                    })
                ),
                "{peak:?}"
            );
        } else {
            assert!(
                matches!(peak, Err(PeakError::NoDrmMemory { .. })),
                "{peak:?}"
            );
        }
        Ok(())
    }

    /// A pid that cannot be read is a failed peak, not a zero one, whether
    /// `start` refuses it (Windows) or `stop` does, or the OS reads no peak.
    #[test]
    fn a_process_that_cannot_be_read_has_no_peak() {
        let started = Watch::Peak
            .supported()
            .ok()
            .and_then(|watch| Sampler::start(u32::MAX, watch).ok());
        let peak = started.map(|sampler| {
            let (observed, peak) = sampler.stop();
            assert_eq!(observed, MemoryObservation::default());
            peak
        });
        assert!(!matches!(peak, Some(Ok(_))), "{peak:?}");
    }
}
