//! What memory a run held while it ran, observed for **every** benchmark rather
//! than only the memory one.
//!
//! This is the observation channel, not the metric one: a
//! [`MemoryObservation`] qualifies a row instead of scoring it, the way
//! `observation_vl_throughput_prefill_tokens` records the tokens a VL run
//! actually processed. A decode-throughput number measured while zram held part
//! of the model is a different fact from one measured resident, and without this
//! nothing in the row distinguishes them.
//!
//! The host term is the **resident** watermark, so the field means one thing
//! wherever it appears, and the swap term rides beside it rather than inside it.
//! A run that swapped is recognised by a non-zero swap term, not by an inflated
//! peak. That keeps the two terms independently checkable and leaves the
//! swap-aware *sum* where it is already scored: the `max_memory_usage` metric,
//! whose Android arm reports it. An Android row where the metric exceeds this
//! observation is showing exactly that difference.
//!
//! Attaching is per OS and by design incomplete: an arm with no sampler
//! observes nothing and contributes no keys, which is why every field is
//! optional and absence never means zero.
//!
//! | Platform | host peak | swap | source |
//! |---|---|---|---|
//! | Android, Linux | yes | yes | [`proc_footprint`] sampling `/proc/<pid>/status` |
//! | macOS | yes | no | `phys_footprint`, which already bills compressed pages |
//! | Windows | no | no | PSAPI reads once after exit through a duplicated handle, so it is not a poller yet |
//!
//! The table is this observer's coverage. The `max_memory_usage` benchmark does
//! not attach one — it already measures memory itself and fills the observation
//! from that — so its Windows rows do carry a host term, the one place a Windows
//! row has `observation_max_host_bytes` at all.

#[cfg(any(target_os = "linux", target_os = "android"))]
pub(crate) mod proc_footprint;

use pipette_plan_types::result::MemoryObservation;

/// A `sleep` child, killed and reaped on drop.
///
/// Several tests here and in [`proc_footprint`] need a live process to point a
/// sampler at. Cleanup belongs on `Drop` rather than at the end of each test: an
/// assertion that fires early would otherwise leave the child running for the
/// rest of its duration.
#[cfg(all(test, unix))]
pub(crate) struct SleepChild(std::process::Child);

#[cfg(all(test, unix))]
impl SleepChild {
    pub(crate) fn spawn(seconds: &str) -> anyhow::Result<Self> {
        Ok(Self(
            std::process::Command::new("sleep").arg(seconds).spawn()?,
        ))
    }

    pub(crate) fn id(&self) -> u32 {
        self.0.id()
    }
}

#[cfg(all(test, unix))]
impl Drop for SleepChild {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

/// Observes a spawned child until [`Self::finish`]. Attach right after spawn;
/// finish after the child is waited for.
///
/// Dropping without finishing stops the sampler but discards what it saw, which
/// is the right outcome on a path that will not report a row.
pub(crate) struct RunMemoryObserver {
    inner: Inner,
}

/// Per-OS backing. A platform with no sampler still gets an observer so callers
/// need no `cfg` of their own.
enum Inner {
    #[cfg(any(target_os = "linux", target_os = "android"))]
    Proc(proc_footprint::FootprintPoller),
    #[cfg(target_os = "macos")]
    PhysFootprint(pipette_memprobe_metal::host::PhysFootprintPoller),
    /// Windows today, plus any target without a sampler. Gated to exactly those
    /// targets: on an arm that does sample, an `Unobserved` that can never be
    /// constructed is a variant claiming a state the code cannot reach.
    #[cfg(not(any(target_os = "linux", target_os = "android", target_os = "macos")))]
    Unobserved,
}

impl RunMemoryObserver {
    /// Start observing `pid`.
    pub(crate) fn attach(pid: u32) -> Self {
        // Both arms poll at the observation cadence, not the memory benchmark's:
        // a benchmark that reports time must not move because we watched it. See
        // `OBSERVATION_INTERVAL` for what that costs and why 100 ms rather than
        // something rarer.
        #[cfg(any(target_os = "linux", target_os = "android"))]
        let inner = Inner::Proc(proc_footprint::spawn_observation_poller(pid));
        // Unlike the `/proc` arm this one has no identity guard: a caller that
        // reaps before finishing leaves a window of at most one interval in which
        // a read could land on a recycled pid and be folded into the peak. Known
        // and accepted — `proc_pid_rusage` carries no start time to compare, the
        // metric path has always had the same window, and macOS would have to
        // recycle a pid within ~100 ms of the reap to hit it.
        #[cfg(target_os = "macos")]
        let inner = Inner::PhysFootprint(
            pipette_memprobe_metal::host::spawn_phys_footprint_poller_for_observation(pid as i32),
        );
        #[cfg(not(any(target_os = "linux", target_os = "android", target_os = "macos")))]
        let inner = {
            let _ = pid;
            Inner::Unobserved
        };
        Self { inner }
    }

    /// Stop observing and report what was seen.
    ///
    /// A sampler that failed yields an empty observation rather than an error: an
    /// observation missing from a row costs a diagnostic, while failing the run
    /// would discard a benchmark result that is itself perfectly good. The
    /// memory *metric* makes the opposite trade and refuses to submit a figure
    /// it cannot trust.
    pub(crate) fn finish(self) -> MemoryObservation {
        match self.inner {
            #[cfg(any(target_os = "linux", target_os = "android"))]
            Inner::Proc(poller) => observation_from(&poller.stop_and_join()),
            // `host_only` because `phys_footprint` bills compressed pages to the
            // process: the peak already counts them and no separate term exists.
            #[cfg(target_os = "macos")]
            Inner::PhysFootprint(poller) => poller
                .stop_and_join()
                .map(observation_from_phys)
                .unwrap_or_default(),
            #[cfg(not(any(target_os = "linux", target_os = "android", target_os = "macos")))]
            Inner::Unobserved => MemoryObservation::default(),
        }
    }
}

/// What a `phys_footprint` reading is worth reporting as, or nothing.
///
/// The macOS counterpart of [`observation_from`], applying the same rule for the
/// same reason (`PhysFootprint::samples`). Weaker here than on `/proc`: that
/// sampler seeds synchronously, so one sample provably means startup, while this
/// poller's first read lands whenever its thread is scheduled. The guard only
/// ever withholds, so the difference costs a short run its observation rather
/// than putting a wrong figure on a row.
#[cfg(target_os = "macos")]
fn observation_from_phys(
    footprint: pipette_memprobe_metal::host::PhysFootprint,
) -> MemoryObservation {
    if footprint.samples < 2 {
        return MemoryObservation::default();
    }
    MemoryObservation::host_only(footprint.peak_bytes)
}

/// What a `/proc` footprint is worth reporting as, or nothing.
///
/// The single rule for this sampler, used by [`RunMemoryObserver::finish`] and
/// by the `max_memory_usage` arms that hold a footprint of their own. Separate
/// from `finish` so it is decidable from a hand-built footprint instead of only
/// from a live process.
#[cfg(any(target_os = "linux", target_os = "android"))]
pub(crate) fn observation_from(footprint: &proc_footprint::Footprint) -> MemoryObservation {
    // Withheld on top of the zero rule the constructor applies: see
    // `Footprint::samples`.
    if footprint.samples < 2 {
        return MemoryObservation::default();
    }
    // `peak_rss_kib` (VmHWM), deliberately not `peak_ram_kib()`: the host term is
    // the resident watermark, so the field means the same thing on every arm that
    // can report it. The swap-aware sum stays out of it and rides beside it as
    // the swap term, which is what says the resident figure was suppressed. An
    // unreadable VmHWM therefore withholds the swap reading too, rather than
    // reporting a term whose peak is missing.
    MemoryObservation::with_swap(
        footprint.peak_rss_kib.saturating_mul(1024),
        footprint.max_swap_bytes(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// What reaches a row from a `/proc` footprint: the resident peak, only once
    /// a read has landed after the seed, and never half an observation.
    ///
    /// Asserts the whole observation rather than the peak alone, since a
    /// withheld peak has to take the swap term with it.
    #[cfg(any(target_os = "linux", target_os = "android"))]
    #[rstest::rstest]
    #[case::seed_only_is_withheld(3_000, 0, 0, 1, MemoryObservation::default())]
    #[case::nothing_read_is_withheld(0, 0, 0, 0, MemoryObservation::default())]
    #[case::a_post_seed_read_is_reported(
        6_440_000, 0, 0, 2,
        MemoryObservation::with_swap(6_440_000 * 1024, 0)
    )]
    // A measured S26 Ultra footprint: `VmHWM` 5004 MiB while zram held 4105 MiB,
    // the two summing to the 6138 MiB the Android metric scores. Only the
    // resident term may appear here, or the field means one thing on an arm that
    // swapped and another on one that did not.
    #[case::swapping_reports_the_resident_term(
        5_124_688, 6_285_312, 4_203_520, 64,
        MemoryObservation::with_swap(5_124_688 * 1024, 4_203_520 * 1024)
    )]
    fn an_observation_reports_the_resident_peak_after_the_seed(
        #[case] peak_rss_kib: u64,
        #[case] peak_committed_kib: u64,
        #[case] max_swap_kib: u64,
        #[case] samples: u32,
        #[case] expected: MemoryObservation,
    ) {
        let footprint = proc_footprint::Footprint {
            peak_rss_kib,
            peak_committed_kib,
            max_swap_kib,
            samples,
            ..Default::default()
        };
        assert_eq!(observation_from(&footprint), expected);
    }

    /// The macOS arm needs the same rule as the `/proc` one, and withholds a
    /// zero on top of it.
    #[cfg(target_os = "macos")]
    #[rstest::rstest]
    #[case::seed_only_is_withheld(3_000_000, 1, MemoryObservation::default())]
    #[case::nothing_read_is_withheld(0, 0, MemoryObservation::default())]
    #[case::a_zero_peak_is_withheld_even_when_sampled(0, 5, MemoryObservation::default())]
    #[case::a_post_seed_read_is_reported(
        6_594_494_464,
        2,
        MemoryObservation::host_only(6_594_494_464)
    )]
    fn a_phys_observation_needs_a_read_after_the_first(
        #[case] peak_bytes: u64,
        #[case] samples: u32,
        #[case] expected: MemoryObservation,
    ) {
        let footprint = pipette_memprobe_metal::host::PhysFootprint {
            peak_bytes,
            samples,
        };
        assert_eq!(observation_from_phys(footprint), expected);
    }

    /// The observer must work without the caller knowing which platform it is
    /// on, and must never invent a figure on one that samples nothing.
    ///
    /// Unix-gated because it needs a live child to observe and `sleep` is not on
    /// the PATH of a stock Windows runner — the sampler-less arm this would cover
    /// there is pinned by `an_arm_with_no_sampler_observes_nothing` instead.
    #[cfg(unix)]
    #[test]
    fn an_observation_is_either_populated_or_absent_but_never_zero() -> anyhow::Result<()> {
        let child = SleepChild::spawn("2")?;
        let observer = RunMemoryObserver::attach(child.id());
        std::thread::sleep(std::time::Duration::from_millis(350));
        let observed = observer.finish();

        if let Some(peak) = observed.max_host_bytes {
            assert!(peak > 0, "a reported peak must never be zero");
        } else {
            assert_eq!(
                observed,
                MemoryObservation::default(),
                "an arm with no host peak must report no swap either"
            );
        }
        Ok(())
    }

    /// Windows today: attaching still has to succeed so callers need no `cfg`,
    /// and finishing has to contribute no keys rather than a zero that reads as
    /// a measurement. Needs no child, since nothing is sampled.
    #[cfg(not(any(target_os = "linux", target_os = "android", target_os = "macos")))]
    #[test]
    fn an_arm_with_no_sampler_observes_nothing() {
        assert_eq!(
            RunMemoryObserver::attach(std::process::id()).finish(),
            MemoryObservation::default()
        );
    }
}
