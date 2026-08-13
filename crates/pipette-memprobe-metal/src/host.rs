//! macOS host counter that pairs with the Metal probe in
//! [`crate::metal`].
//!
//! `phys_footprint` (the kernel's per-process physical-pages-billed
//! counter, polled via `proc_pid_rusage(RUSAGE_INFO_V4)`) is reported
//! directly as `max_host_bytes`. Apple UMA means Metal allocations
//! are billed to `phys_footprint`, so the host bucket is a superset
//! of `max_gpu_bytes` on Mac — each is its own dimension's peak,
//! reported independently with no cross-subtraction.
//!
//! The Android/Windows host counters (`/proc/<pid>/status` sampling,
//! `GetProcessMemoryInfo.PeakWorkingSetSize`) are inlined directly
//! in `pipette-llamacpp`'s per-OS `max_memory_usage::{android,
//! windows}` modules — they share no abstraction with this poller
//! (different syscall families, different lifetimes), and each has
//! a single consumer.

use std::{
    mem::MaybeUninit,
    sync::mpsc::{self, RecvTimeoutError},
    thread::{self, JoinHandle},
    time::Duration,
};

/// Handle for a macOS phys_footprint polling thread. Call
/// [`Self::stop_and_join`] after `wait_with_output` returns to
/// retrieve the peak.
///
/// The struct is an actor handle: a stop-signal channel plus the
/// thread's join handle. The peak is the thread's return value
/// (single writer, no shared state). Dropping the handle without
/// calling `stop_and_join` is also clean — the sender drops, the
/// channel disconnects, the poller exits within one tick — but
/// the peak is silently discarded, which is exactly what the
/// `#[must_use]` warning is designed to catch.
#[must_use = "PhysFootprintPoller::stop_and_join must be called to retrieve the peak; \
              dropping the handle silently discards the measurement"]
pub struct PhysFootprintPoller {
    stop: mpsc::Sender<()>,
    handle: JoinHandle<u64>,
}

impl PhysFootprintPoller {
    /// Signal the poller to stop, join the thread, and return the
    /// peak `ri_phys_footprint` it observed (in bytes). Use this
    /// **after** `wait_with_output` (or equivalent) returns —
    /// `proc_pid_rusage` returns ESRCH after wait4 reaps the child.
    ///
    /// Returns `Err` if the poller thread panicked. The peak feeds
    /// `max_host_bytes` (a wire-schema field), so a panicked poller
    /// must fail the bench loudly rather than silently report a
    /// partial peak — the caller can't tell the difference between
    /// "zero peak observed" and "panic before any observation."
    pub fn stop_and_join(self) -> anyhow::Result<u64> {
        // Send is non-blocking; the poller's recv_timeout returns Ok
        // on the next tick. Send may fail (Disconnected) if the
        // poller already exited — ignore here; join below surfaces
        // any panic.
        let _ = self.stop.send(());
        self.handle
            .join()
            .map_err(|_| anyhow::anyhow!("phys_footprint poller thread panicked"))
    }
}

/// Spawn a thread that polls `proc_pid_rusage(pid,
/// RUSAGE_INFO_V4).ri_phys_footprint` every 20 ms and tracks the
/// running max. Returns immediately; the caller should
/// [`PhysFootprintPoller::stop_and_join`] after the child is reaped.
///
/// 20 ms cadence is sufficient for steady-state peaks; tighten only
/// if a future workload allocates and frees within a frame. See
/// `pipette-mgmt/docs/methodology/peak-memory.md` §3.1 for the
/// failure-detection signal.
pub fn spawn_phys_footprint_poller(pid: i32) -> PhysFootprintPoller {
    let (stop_tx, stop_rx) = mpsc::channel::<()>();
    let handle = thread::spawn(move || -> u64 {
        let mut peak: u64 = 0;
        loop {
            let mut info = MaybeUninit::<libc::rusage_info_v4>::zeroed();
            let rc = unsafe {
                libc::proc_pid_rusage(
                    pid,
                    libc::RUSAGE_INFO_V4,
                    info.as_mut_ptr() as *mut libc::rusage_info_t,
                )
            };
            if rc == 0 {
                let info = unsafe { info.assume_init() };
                let observed = info
                    .ri_phys_footprint
                    .max(info.ri_lifetime_max_phys_footprint);
                if observed > peak {
                    peak = observed;
                }
            }
            // proc_pid_rusage returns ESRCH after wait4 reaps the
            // child. Fine — peak is captured; loop until stop.
            //
            // Combined sleep + stop-check: recv_timeout sleeps up to
            // 20 ms, returns early on stop signal or sender drop.
            match stop_rx.recv_timeout(Duration::from_millis(20)) {
                Ok(_) | Err(RecvTimeoutError::Disconnected) => break peak,
                Err(RecvTimeoutError::Timeout) => continue,
            }
        }
    });
    PhysFootprintPoller {
        stop: stop_tx,
        handle,
    }
}
