//! The macOS sampler: `proc_pid_rusage(RUSAGE_INFO_V4)` polled for
//! `max(ri_phys_footprint, ri_lifetime_max_phys_footprint)`.
//!
//! `phys_footprint` is the kernel's per-process physical-pages-billed counter. It
//! bills compressed pages to the process, so there is no separate swap term, and
//! Apple Silicon is unified memory, so it bills the Metal allocations too: its
//! peak is the whole cost and there is no GPU pool to report. The lifetime
//! maximum is a kernel high-water mark, so the cadence is detection latency, not
//! accuracy: [`Sampler::stop`] reads once more, which covers a process still
//! alive then, or exited and not yet reaped: the kernel keeps an exited
//! process's usage for `proc_pid_rusage` until the reap
//! (`Process::wait_success_before_reap`). A reaped one cannot be read, so a
//! peak in its last interval could be missing, and the peak is refused
//! ([`PeakError::UnreadAtStop`]).
//!
//! Like the `/proc` sampler, it pins the process's start time
//! (`ri_proc_start_abstime`, which an exited process keeps until the reap) and
//! skips a read whose start time differs, so a pid recycled after the reap is
//! never read as the process, at the stop either.
//!
//! See `docs/methodology/peak-memory-macos.md` for the methodology.

use std::mem::MaybeUninit;
use std::sync::mpsc::{self, RecvTimeoutError};
use std::thread::JoinHandle;
use std::time::Duration;

use pipette_plan_types::MemoryObservation;

use super::{Peak, PeakError, StartError, SupportedWatch, UnsupportedError, Watch};

/// Cadence when the peak is the benchmark's result. Sufficient for steady-state
/// peaks; tighten only if a workload allocates and frees within a frame.
const PEAK_INTERVAL: Duration = Duration::from_millis(20);

/// Cadence for an observation riding along on a benchmark that reports time,
/// rare enough (~0.1% of a core) that watching cannot move the number.
const OBSERVE_INTERVAL: Duration = Duration::from_millis(100);

/// What the thread saw.
#[derive(Clone, Copy, Debug, Default)]
struct PhysFootprint {
    /// Largest reading, in bytes. Zero means no read succeeded.
    peak_bytes: u64,
    /// Whether the last read, at the stop, landed on the process. Without it a
    /// peak in the run's last interval can be missing while the earlier reads
    /// pass for the whole run.
    read_at_stop: bool,
}

impl PhysFootprint {
    /// Whether `usage` is the pinned process's (`identity`, a start time), and
    /// so counted.
    fn observe(&mut self, identity: Option<u64>, usage: Usage) -> bool {
        if identity.is_some_and(|pinned| pinned != usage.start_abstime) {
            return false;
        }
        self.peak_bytes = self.peak_bytes.max(usage.footprint_bytes);
        true
    }
}

/// A `phys_footprint` poller on the process, at the cadence of its [`Watch`].
pub struct Sampler {
    pid: u32,
    watch: Watch,
    /// Dropping it ends the thread within one interval.
    stop: mpsc::Sender<()>,
    thread: JoinHandle<PhysFootprint>,
}

impl Sampler {
    pub(crate) const OBSERVATION: Watch = Watch::Observe;

    /// `watch`, or why this sampler has no reader for it. Unified memory bills
    /// Metal's allocations to the footprint, so a watch for whatever devices a
    /// run holds memory on reads the host peak. One for a GPU or an NPU needs
    /// that device's own figure, which macOS does not report: the footprint
    /// cannot tell whether the run reached it.
    pub(crate) fn reads(watch: Watch) -> Result<Watch, UnsupportedError> {
        match watch {
            Watch::Observe | Watch::Peak | Watch::PeakWithAccelerators => Ok(watch),
            Watch::PeakWithGpu | Watch::PeakWithNpu => Err(UnsupportedError { watch }),
        }
    }

    pub fn start(pid: u32, SupportedWatch(watch): SupportedWatch) -> Result<Self, StartError> {
        let interval = if watch == Watch::Observe {
            OBSERVE_INTERVAL
        } else {
            PEAK_INTERVAL
        };
        // Pins the process, so a read of its pid recycled is skipped.
        let identity = read_usage(pid).map(|usage| usage.start_abstime);
        let (stop, stopped) = mpsc::channel();
        let thread = std::thread::spawn(move || {
            let mut footprint = PhysFootprint::default();
            // Reads after each wait, the one a stop ends too, so the last read is
            // taken at the stop.
            loop {
                let stopping = match stopped.recv_timeout(interval) {
                    Err(RecvTimeoutError::Timeout) => false,
                    Ok(()) | Err(RecvTimeoutError::Disconnected) => true,
                };
                let landed =
                    read_usage(pid).is_some_and(|usage| footprint.observe(identity, usage));
                if stopping {
                    footprint.read_at_stop = landed;
                    break footprint;
                }
            }
        });
        Ok(Self {
            pid,
            watch,
            stop,
            thread,
        })
    }

    /// The run's observation, and its peak when watched for one. Its last read
    /// needs the process running, or exited and not yet reaped by the caller,
    /// whose child it must be: one already seen to exit (`Process::is_running`,
    /// `wait`, `wait_ready`) has been reaped, and its peak is refused.
    pub fn stop(self) -> (MemoryObservation, Result<Peak, PeakError>) {
        // Fails only when the thread already exited, which `join` reports.
        let _ = self.stop.send(());
        let footprint = self.thread.join().map_err(|_| PeakError::SamplerPanicked);
        if self.watch == Watch::Observe {
            let observation = footprint.map(observation).unwrap_or_default();
            return (observation, Err(PeakError::NotWatched));
        }
        let peak = footprint.and_then(|footprint| peak(self.pid, footprint));
        let observation = peak.as_ref().map_or_else(
            |_| MemoryObservation::default(),
            |peak| MemoryObservation::host_only(peak.host_bytes),
        );
        (observation, peak)
    }
}

/// The peak, once the read at the stop has landed.
fn peak(pid: u32, footprint: PhysFootprint) -> Result<Peak, PeakError> {
    match footprint {
        PhysFootprint { peak_bytes: 0, .. } => Err(PeakError::NoFootprint { pid }),
        PhysFootprint {
            read_at_stop: false,
            ..
        } => Err(PeakError::UnreadAtStop { pid }),
        PhysFootprint {
            peak_bytes: host_bytes,
            read_at_stop: true,
        } => Ok(Peak {
            host_bytes,
            gpu_bytes: None,
            npu_bytes: None,
        }),
    }
}

/// What a reading is worth reporting as, or nothing: the `/proc` sampler's rule,
/// for the same reason (`PhysFootprint::read_at_stop`).
fn observation(footprint: PhysFootprint) -> MemoryObservation {
    if !footprint.read_at_stop {
        return MemoryObservation::default();
    }
    MemoryObservation::host_only(footprint.peak_bytes)
}

/// One `proc_pid_rusage` read.
#[derive(Clone, Copy, Debug)]
struct Usage {
    /// `max(ri_phys_footprint, ri_lifetime_max_phys_footprint)`.
    footprint_bytes: u64,
    /// `ri_proc_start_abstime`, which only identifies the process.
    start_abstime: u64,
}

/// `pid`'s [`Usage`], or `None` if the process cannot be read. A pid past
/// `pid_t` reads as `None` like a reaped one, so it yields no samples, which
/// every caller rejects.
fn read_usage(pid: u32) -> Option<Usage> {
    let pid = libc::pid_t::try_from(pid).ok()?;
    let mut info = MaybeUninit::<libc::rusage_info_v4>::zeroed();
    // SAFETY: `info` is a live, writable `rusage_info_v4`, the struct
    // `RUSAGE_INFO_V4` tells the kernel to fill.
    let rc = unsafe {
        libc::proc_pid_rusage(
            pid,
            libc::RUSAGE_INFO_V4,
            info.as_mut_ptr().cast::<libc::rusage_info_t>(),
        )
    };
    if rc != 0 {
        return None;
    }
    // SAFETY: `rc == 0` means the kernel filled `info`, and it started zeroed,
    // which is a valid value for this plain C struct.
    let info = unsafe { info.assume_init() };
    Some(Usage {
        footprint_bytes: info
            .ri_phys_footprint
            .max(info.ri_lifetime_max_phys_footprint),
        start_abstime: info.ri_proc_start_abstime,
    })
}

#[cfg(test)]
mod tests {
    use anyhow::Context;

    use super::*;

    #[test]
    fn reads_own_footprint() -> anyhow::Result<()> {
        let usage = read_usage(std::process::id()).context("own pid unreadable")?;
        anyhow::ensure!(usage.footprint_bytes > 0, "own footprint read as 0");
        anyhow::ensure!(usage.start_abstime > 0, "own start time read as 0");
        Ok(())
    }

    /// A read counts only for the process pinned at the start: a pid recycled
    /// after the reap is another process, and its read does not land.
    #[rstest::rstest]
    #[case::same_process(Some(987_654), 987_654, true)]
    #[case::pid_recycled(Some(987_654), 111_111, false)]
    #[case::identity_unknown(None, 111_111, true)]
    fn a_read_counts_only_for_the_pinned_process(
        #[case] identity: Option<u64>,
        #[case] start_abstime: u64,
        #[case] expected: bool,
    ) {
        let mut footprint = PhysFootprint::default();
        let usage = Usage {
            footprint_bytes: 6_000_000,
            start_abstime,
        };
        assert_eq!(footprint.observe(identity, usage), expected);
        assert_eq!(footprint.peak_bytes, if expected { 6_000_000 } else { 0 });
    }

    #[rstest::rstest]
    #[case::nothing_read(0, false, Err("captured no samples"))]
    #[case::unread_at_the_stop(3_000_000, false, Err("was not read when the sampler stopped"))]
    #[case::read_at_the_stop(6_594_494_464, true, Ok(6_594_494_464))]
    fn a_peak_needs_its_read_at_the_stop(
        #[case] peak_bytes: u64,
        #[case] read_at_stop: bool,
        #[case] expected: Result<u64, &str>,
    ) -> anyhow::Result<()> {
        let footprint = PhysFootprint {
            peak_bytes,
            read_at_stop,
        };
        match (peak(42, footprint), expected) {
            (Ok(peak), Ok(bytes)) => assert_eq!(peak.host_bytes, bytes),
            (Err(err), Err(cause)) => assert!(err.to_string().contains(cause), "{err}"),
            (got, want) => anyhow::bail!("expected {want:?}, got {got:?}"),
        }
        Ok(())
    }

    #[test]
    fn pid_past_pid_t_reads_as_none() {
        assert!(read_usage(u32::MAX).is_none());
    }

    /// The same rule as the `/proc` sampler's observation, and a zero withheld on
    /// top of it.
    #[rstest::rstest]
    #[case::unread_at_the_stop_is_withheld(3_000_000, false, MemoryObservation::default())]
    #[case::nothing_read_is_withheld(0, false, MemoryObservation::default())]
    #[case::a_zero_peak_is_withheld_even_when_read(0, true, MemoryObservation::default())]
    #[case::read_at_the_stop_is_reported(
        6_594_494_464,
        true,
        MemoryObservation::host_only(6_594_494_464)
    )]
    fn an_observation_needs_its_read_at_the_stop(
        #[case] peak_bytes: u64,
        #[case] read_at_stop: bool,
        #[case] expected: MemoryObservation,
    ) {
        let footprint = PhysFootprint {
            peak_bytes,
            read_at_stop,
        };
        assert_eq!(observation(footprint), expected);
    }
}
