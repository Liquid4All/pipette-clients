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
//! Because it never feeds a scored figure, this is free to report the
//! swap-aware peak wherever the sampler supplies one. The `max_memory_usage`
//! metric keeps its own per-arm definition (Linux still reports resident-only
//! there); the two are allowed to disagree, and that disagreement is exactly
//! what the swap term explains.
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
            #[cfg(target_os = "macos")]
            Inner::PhysFootprint(poller) => match poller.stop_and_join() {
                Ok(0) | Err(_) => MemoryObservation::default(),
                Ok(bytes) => MemoryObservation {
                    max_host_bytes: Some(bytes),
                    // `phys_footprint` bills compressed pages to the process, so
                    // the peak already counts them and no separate term exists.
                    max_swap_bytes: None,
                },
            },
            #[cfg(not(any(target_os = "linux", target_os = "android", target_os = "macos")))]
            Inner::Unobserved => MemoryObservation::default(),
        }
    }
}

/// What a `/proc` footprint is worth reporting as, or nothing.
///
/// Separate from [`RunMemoryObserver::finish`] so the rule is decidable from a
/// hand-built footprint instead of only from a live process.
#[cfg(any(target_os = "linux", target_os = "android"))]
fn observation_from(footprint: &proc_footprint::Footprint) -> MemoryObservation {
    // A zero peak means nobody looked. A seed-only footprint is worse: it
    // describes process startup in a few plausible-looking MB that a consumer
    // cannot tell apart from a real measurement. Absence is filterable, a wrong
    // number is not.
    if footprint.peak_ram_kib() == 0 || footprint.samples < 2 {
        return MemoryObservation::default();
    }
    MemoryObservation {
        max_host_bytes: Some(footprint.peak_ram_kib().saturating_mul(1024)),
        max_swap_bytes: Some(footprint.max_swap_bytes()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The seed is taken before the child has loaded anything, so a footprint
    /// built only from it describes startup, not the run. Reporting it would put
    /// a few MB on a row as though measured; these cases pin that it is withheld.
    #[cfg(any(target_os = "linux", target_os = "android"))]
    #[rstest::rstest]
    #[case::seed_only_is_withheld(3_000, 1, None)]
    #[case::nothing_read_is_withheld(0, 0, None)]
    #[case::a_post_seed_read_is_reported(6_440_000, 2, Some(6_440_000 * 1024))]
    fn an_observation_needs_a_read_after_the_seed(
        #[case] peak_rss_kib: u64,
        #[case] samples: u32,
        #[case] expected_host_bytes: Option<u64>,
    ) {
        let footprint = proc_footprint::Footprint {
            peak_rss_kib,
            samples,
            ..Default::default()
        };
        assert_eq!(
            observation_from(&footprint).max_host_bytes,
            expected_host_bytes
        );
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
        let mut child = std::process::Command::new("sleep").arg("2").spawn()?;
        let observer = RunMemoryObserver::attach(child.id());
        std::thread::sleep(std::time::Duration::from_millis(350));
        let observed = observer.finish();
        child.kill()?;
        child.wait()?;

        if let Some(peak) = observed.max_host_bytes {
            assert!(peak > 0, "a reported peak must never be zero");
            if let Some(swap) = observed.max_swap_bytes {
                assert!(
                    swap <= peak,
                    "swap is contained in the peak, not added to it"
                );
            }
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
