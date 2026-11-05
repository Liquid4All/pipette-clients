//! The poll loop Android, Linux and macOS share: probe, verdict, deadline,
//! top-up sleep. Each platform supplies only the probe.
//!
//! `windows.rs` keeps its own loop (it keeps the poll series for its
//! stuck-sensor report and runs a ceiling budget beside the deadline) behind
//! the same [`ReadinessCallbacks`] and [`ReadinessStatus`], and shares
//! `on_probe_failure` for a failed probe.

// Compiled on every target, not only the three that use it, so its tests (the
// only loop coverage those gates have) run everywhere.
#![cfg_attr(
    not(any(target_os = "android", target_os = "linux", target_os = "macos")),
    allow(
        dead_code,
        reason = "this target drives its own loop or none; compiled for the tests"
    )
)]

use std::ops::ControlFlow;
use std::time::{Duration, Instant};

use crate::{
    ReadinessCallbacks, ReadinessError, ReadinessPhase, ReadinessPlatform, ReadinessStatus,
};

pub(crate) struct Probe {
    pub ready: bool,
    pub summary: String,
}

/// Poll `probe` until it reports ready, the deadline passes, or the caller
/// cancels.
///
/// After a failed probe, the loop polls again on the next interval. It stops
/// after [`crate::PROBE_RETRY_POLLS`] consecutive failures or at the deadline,
/// and returns the last probe error unchanged. That error names what broke,
/// and a `TimedOut` does not. A refused configuration fails the wait on the
/// first poll, because more polls do not change it.
pub(crate) fn wait_loop(
    platform: ReadinessPlatform,
    max_wait: Duration,
    callbacks: &ReadinessCallbacks<'_>,
    mut probe: impl FnMut() -> Result<Probe, ReadinessError>,
) -> Result<(), ReadinessError> {
    let started = Instant::now();
    let deadline = started + max_wait;
    // Before the first probe, not after: a caller that has already given up
    // should not pay for a `dumpsys` fork or a 1 s CPU sample first.
    if callbacks.is_cancelled() {
        return Err(ReadinessError::Cancelled);
    }
    let mut failed_probes = 0usize;
    loop {
        let polled = Instant::now();
        let probed = probe();
        callbacks.polled();
        // Top up: the probe may already have spent ~1 s sampling CPU counters.
        // Held to the deadline, or a timeout lands an interval late.
        let rest = pipette_wait::clamp_to_deadline(
            crate::SENSOR_POLL_INTERVAL.saturating_sub(polled.elapsed()),
            deadline,
        );
        let Probe { ready, summary } = match probed {
            Ok(probe) => {
                failed_probes = 0;
                probe
            }
            // A failed poll read nothing, so it reports no status.
            Err(err) => {
                failed_probes += 1;
                if on_probe_failure(platform, &err, failed_probes, deadline, rest).is_break() {
                    return Err(err);
                }
                if callbacks.sleep_cancellably(rest).is_break() {
                    return Err(ReadinessError::Cancelled);
                }
                continue;
            }
        };
        // A probe can spend a second or more sampling, which is long enough
        // for a cancel to arrive inside one.
        if callbacks.is_cancelled() {
            return Err(ReadinessError::Cancelled);
        }
        let status = |phase| ReadinessStatus {
            phase,
            platform,
            observed: summary.clone(),
            elapsed: started.elapsed(),
            max_wait,
        };
        if ready {
            callbacks.report(&status(ReadinessPhase::Proceeding));
            return Ok(());
        }
        if Instant::now() >= deadline {
            callbacks.report(&status(ReadinessPhase::TimedOut));
            return Err(ReadinessError::TimedOut {
                max_wait,
                observed: summary,
            });
        }
        callbacks.report(&status(ReadinessPhase::Waiting { poll_in: rest }));
        if callbacks.sleep_cancellably(rest).is_break() {
            return Err(ReadinessError::Cancelled);
        }
    }
}

/// The retry policy for a failed probe, which `wait_loop` and the Windows
/// loop share. `failed` is the count of consecutive failed probes, with this
/// one included.
///
/// Returns `Break` if the wait fails with `err`: the error is not
/// [`retriable`], the count reached [`crate::PROBE_RETRY_POLLS`], or the
/// deadline passed. Returns `Continue` if the loop polls again after
/// `retry_in`. The caller sleeps; the sleep differs between the two loops.
pub(crate) fn on_probe_failure(
    platform: ReadinessPlatform,
    err: &ReadinessError,
    failed: usize,
    deadline: Instant,
    retry_in: Duration,
) -> ControlFlow<()> {
    // The file and the errno of a failed read, which `ReadinessError::Io`
    // leaves out of its own text.
    let source = || std::error::Error::source(err).map(tracing::field::display);
    if !retriable(err) || failed >= crate::PROBE_RETRY_POLLS || Instant::now() >= deadline {
        // The caller prints only this error, so without a line here the
        // operator cannot tell one bad poll from a probe that never answered.
        tracing::warn!(
            platform = %platform,
            failed,
            error = %err,
            source = source(),
            "readiness probe failed; failing the wait"
        );
        return ControlFlow::Break(());
    }
    tracing::warn!(
        platform = %platform,
        error = %err,
        source = source(),
        attempt = failed,
        attempts = crate::PROBE_RETRY_POLLS,
        retry_in = ?retry_in,
        "readiness probe failed; polling again"
    );
    ControlFlow::Continue(())
}

/// Returns `true` if another poll can clear `err`.
///
/// The match has no wildcard arm, so a new variant does not compile until it
/// has a class here.
fn retriable(err: &ReadinessError) -> bool {
    match err {
        // A read that can succeed on the next poll.
        ReadinessError::Io(_)
        | ReadinessError::Probe(_)
        | ReadinessError::ParseFailure { .. }
        | ReadinessError::Unavailable(_) => true,
        // A configuration refusal, which is wrong on every poll. For example, a
        // configuration table that bounds a sensor the host does not have.
        ReadinessError::BoundWithoutThermalZones
        | ReadinessError::BoundWithoutSkinSensor
        | ReadinessError::UnknownSensor { .. } => false,
        // A held touch that the probe already saw on several polls in a row.
        // It stays until an operator removes the cause.
        ReadinessError::HeldTouch { .. } => false,
        // A terminal outcome of a wait, which a probe does not return.
        ReadinessError::TimedOut { .. }
        | ReadinessError::Cancelled
        | ReadinessError::SensorUnread { .. } => false,
    }
}

#[cfg(test)]
mod tests {
    use std::cell::{Cell, RefCell};

    use anyhow::Context;
    use rstest::rstest;

    use pipette_wait::CANCEL_CHECK_INTERVAL;

    use super::*;

    /// A scripted gate: the verdicts a probe will report in order, with every
    /// status and every sleep recorded instead of spent.
    struct Scripted {
        verdicts: RefCell<std::vec::IntoIter<Result<bool, ReadinessError>>>,
        polls: Cell<usize>,
        observed_polls: Cell<usize>,
        statuses: RefCell<Vec<ReadinessStatus>>,
        slept: RefCell<Vec<Duration>>,
        cancel_after_polls: Cell<Option<usize>>,
        cancel_after_chunks: Cell<Option<usize>>,
    }

    impl Scripted {
        fn new(verdicts: Vec<Result<bool, ReadinessError>>) -> Self {
            Self {
                verdicts: RefCell::new(verdicts.into_iter()),
                polls: Cell::new(0),
                observed_polls: Cell::new(0),
                statuses: RefCell::new(Vec::new()),
                slept: RefCell::new(Vec::new()),
                cancel_after_polls: Cell::new(None),
                cancel_after_chunks: Cell::new(None),
            }
        }

        /// Raise the cancel flag once `polls` probes have run. `0` is a caller
        /// who cancelled before the wait started.
        fn cancelling_after_polls(self, polls: usize) -> Self {
            self.cancel_after_polls.set(Some(polls));
            self
        }

        /// Raise the cancel flag part-way through a poll's sleep, once
        /// `chunks` of it have been spent: the ^C case.
        fn cancelling_after_chunks(self, chunks: usize) -> Self {
            self.cancel_after_chunks.set(Some(chunks));
            self
        }

        fn probe(&self) -> Result<Probe, ReadinessError> {
            self.polls.set(self.polls.get() + 1);
            let verdict = self
                .verdicts
                .borrow_mut()
                .next()
                // A loop that polls past its script is the bug this reports.
                .unwrap_or(Ok(false))?;
            Ok(Probe {
                ready: verdict,
                summary: format!("poll {}", self.polls.get()),
            })
        }

        fn cancelled(&self) -> bool {
            let by_polls = self
                .cancel_after_polls
                .get()
                .is_some_and(|after| self.polls.get() >= after);
            let by_chunks = self
                .cancel_after_chunks
                .get()
                .is_some_and(|after| self.slept.borrow().len() >= after);
            by_polls || by_chunks
        }

        fn final_phase(&self) -> anyhow::Result<ReadinessPhase> {
            Ok(self
                .statuses
                .borrow()
                .last()
                .context("the wait reported no status at all")?
                .phase)
        }

        /// Every status but the last, which is what a held poll reports.
        fn held_polls(&self) -> Vec<ReadinessStatus> {
            let statuses = self.statuses.borrow();
            statuses
                .split_last()
                .map(|(_, held)| held.to_vec())
                .unwrap_or_default()
        }

        fn run(&self, max_wait: Duration) -> Result<(), ReadinessError> {
            let on_status = |status: &ReadinessStatus| {
                self.statuses.borrow_mut().push(status.clone());
            };
            let on_poll = || self.observed_polls.set(self.observed_polls.get() + 1);
            let cancelled = || self.cancelled();
            let sleep = |d: Duration| self.slept.borrow_mut().push(d);
            let callbacks = ReadinessCallbacks::new()
                .with_status(&on_status)
                .with_poll(&on_poll)
                .with_cancel(&cancelled)
                .with_sleep(&sleep);
            wait_loop(ReadinessPlatform::Other, max_wait, &callbacks, || {
                self.probe()
            })
        }
    }

    /// One probe and no sleep at all.
    #[test]
    fn a_ready_host_proceeds_on_the_first_poll() -> anyhow::Result<()> {
        let script = Scripted::new(vec![Ok(true)]);
        script.run(Duration::from_secs(300))?;

        assert_eq!(script.polls.get(), 1);
        assert_eq!(script.final_phase()?, ReadinessPhase::Proceeding);
        assert!(script.held_polls().is_empty(), "no poll was held");
        assert!(script.slept.borrow().is_empty(), "nothing to wait for");
        Ok(())
    }

    #[test]
    fn a_cooling_host_reports_every_held_poll_then_proceeds() -> anyhow::Result<()> {
        let script = Scripted::new(vec![Ok(false), Ok(false), Ok(true)]);
        script.run(Duration::from_secs(300))?;

        let held = script.held_polls();
        assert_eq!(held.len(), 2, "one status per held poll");
        assert!(
            held.iter().all(ReadinessStatus::is_waiting),
            "a held poll reports `Waiting`: {held:?}",
        );
        assert_eq!(script.final_phase()?, ReadinessPhase::Proceeding);
        let slept = script.slept.borrow();
        assert!(
            slept.iter().sum::<Duration>() <= 2 * crate::SENSOR_POLL_INTERVAL,
            "two held polls spend at most two intervals: {slept:?}",
        );
        assert!(
            slept.iter().all(|chunk| *chunk <= CANCEL_CHECK_INTERVAL),
            "and spend them in cancel-checkable pieces: {slept:?}",
        );
        let chunks_per_interval = crate::SENSOR_POLL_INTERVAL
            .as_millis()
            .div_ceil(CANCEL_CHECK_INTERVAL.as_millis());
        assert_eq!(
            u128::try_from(slept.len())?,
            2 * chunks_per_interval,
            "each interval is cut into as many pieces as the cancel check needs: {slept:?}",
        );
        Ok(())
    }

    #[test]
    fn a_host_that_never_clears_times_out_with_its_last_reading() -> anyhow::Result<()> {
        let script = Scripted::new(vec![Ok(false)]);
        let err = script
            .run(Duration::ZERO)
            .err()
            .context("a zero deadline must time out")?;

        assert!(
            matches!(err, ReadinessError::TimedOut { ref observed, .. } if observed == "poll 1"),
            "got {err:?}",
        );
        assert_eq!(script.final_phase()?, ReadinessPhase::TimedOut);
        Ok(())
    }

    #[test]
    fn a_cancel_before_the_first_poll_never_probes() -> anyhow::Result<()> {
        let script = Scripted::new(vec![Ok(true)]).cancelling_after_polls(0);
        let err = script
            .run(Duration::from_secs(300))
            .err()
            .context("a pre-cancelled wait must not report ready")?;

        assert!(matches!(err, ReadinessError::Cancelled), "got {err:?}");
        assert_eq!(script.polls.get(), 0);
        assert!(script.statuses.borrow().is_empty());
        Ok(())
    }

    #[test]
    fn a_cancel_mid_sleep_is_honored_within_one_chunk() -> anyhow::Result<()> {
        let script = Scripted::new(vec![Ok(false), Ok(true)]).cancelling_after_chunks(1);
        let err = script
            .run(Duration::from_secs(300))
            .err()
            .context("a cancelled wait must not report ready")?;

        assert!(matches!(err, ReadinessError::Cancelled), "got {err:?}");
        assert_eq!(script.polls.get(), 1, "the second poll must never run");
        assert_eq!(
            script.slept.borrow().len(),
            1,
            "one chunk spent, then the flag is read and the rest is not",
        );
        Ok(())
    }

    fn probe_failed() -> Result<bool, ReadinessError> {
        Err(ReadinessError::Unavailable("no sensors"))
    }

    /// A probe that fails and then recovers must not fail the cell. The
    /// allowance applies to each series of consecutive failures, and a good
    /// poll resets it to the full count.
    #[rstest]
    #[case::one_bad_series(vec![probe_failed(), probe_failed(), Ok(true)])]
    // The held reading keeps the gate in the loop. The second series of
    // failures starts from a count of zero.
    #[case::two_series_split_by_a_good_poll(vec![
        probe_failed(),
        probe_failed(),
        Ok(false),
        probe_failed(),
        probe_failed(),
        Ok(true),
    ])]
    fn the_gate_rides_out_transient_probe_failures(
        #[case] verdicts: Vec<Result<bool, ReadinessError>>,
    ) -> anyhow::Result<()> {
        let polls = verdicts.len();
        let script = Scripted::new(verdicts);
        script.run(Duration::from_secs(300))?;

        assert_eq!(script.polls.get(), polls);
        assert_eq!(script.observed_polls.get(), polls);
        assert_eq!(script.final_phase()?, ReadinessPhase::Proceeding);
        Ok(())
    }

    /// The script of a probe that never answers: twice the allowance of
    /// failures, then a ready verdict.
    fn never_answers() -> Vec<Result<bool, ReadinessError>> {
        // The script answers after the allowance and releases the gate. A
        // loop that retries too often then fails the assertions, and does not
        // run until the 300 s deadline.
        let mut verdicts: Vec<_> = (0..crate::PROBE_RETRY_POLLS * 2)
            .map(|_| probe_failed())
            .collect();
        verdicts.push(Ok(true));
        verdicts
    }

    fn is_no_sensors(err: &ReadinessError) -> bool {
        matches!(err, ReadinessError::Unavailable("no sensors"))
    }

    fn is_bound_without_thermal_zones(err: &ReadinessError) -> bool {
        matches!(err, ReadinessError::BoundWithoutThermalZones)
    }

    fn is_held_touch(err: &ReadinessError) -> bool {
        matches!(err, ReadinessError::HeldTouch { .. })
    }

    /// The wait fails with the probe's own error, not a `TimedOut`. An i/o or
    /// parse failure and a timeout say different things about the host, and
    /// only a timeout is about heat. A poll that read nothing reports no status.
    #[rstest]
    // A retry after a transient failure must not make a probe that never
    // answers use the whole deadline.
    #[case::never_answers(
        never_answers(),
        Duration::from_secs(300),
        crate::PROBE_RETRY_POLLS,
        is_no_sensors
    )]
    // If the deadline passed before the probe failed, the wait returns the
    // probe error at once. It does not retry.
    #[case::spent_deadline(vec![probe_failed(), Ok(true)], Duration::ZERO, 1, is_no_sensors)]
    // A refused configuration is wrong on every poll, so a retry only delays
    // the error.
    #[case::configuration_refusal(
        vec![Err(ReadinessError::BoundWithoutThermalZones), Ok(true)],
        Duration::from_secs(300),
        1,
        is_bound_without_thermal_zones
    )]
    #[case::held_touch(
        vec![Err(ReadinessError::HeldTouch { pointers: 1, irq_per_sec: 120 }), Ok(true)],
        Duration::from_secs(300),
        1,
        is_held_touch
    )]
    fn a_failed_probe_ends_the_wait_with_its_own_error(
        #[case] verdicts: Vec<Result<bool, ReadinessError>>,
        #[case] max_wait: Duration,
        #[case] want_polls: usize,
        #[case] is_expected: fn(&ReadinessError) -> bool,
    ) -> anyhow::Result<()> {
        let script = Scripted::new(verdicts);
        let err = script
            .run(max_wait)
            .err()
            .context("the failed probe must fail the wait")?;

        assert!(is_expected(&err), "got {err:?}");
        assert_eq!(script.polls.get(), want_polls);
        assert_eq!(script.observed_polls.get(), want_polls);
        assert!(
            script.statuses.borrow().is_empty(),
            "a poll that read nothing has nothing to report",
        );
        // Each failed poll but the last sleeps before the next one.
        assert_eq!(
            script.slept.borrow().is_empty(),
            want_polls == 1,
            "slept {:?}",
            script.slept.borrow(),
        );
        Ok(())
    }

    /// The held poll also reports the remainder it actually sleeps, not the
    /// whole interval.
    #[test]
    fn a_slow_probe_shortens_the_sleep_instead_of_extending_the_interval() -> anyhow::Result<()> {
        let spent = Duration::from_millis(200);
        let budget = crate::SENSOR_POLL_INTERVAL - spent;
        let slept: RefCell<Vec<Duration>> = RefCell::new(Vec::new());
        let statuses: RefCell<Vec<ReadinessStatus>> = RefCell::new(Vec::new());
        let polls = Cell::new(0usize);
        let sleep = |d: Duration| slept.borrow_mut().push(d);
        let on_status = |s: &ReadinessStatus| statuses.borrow_mut().push(s.clone());
        let callbacks = ReadinessCallbacks::new()
            .with_status(&on_status)
            .with_sleep(&sleep);

        wait_loop(
            ReadinessPlatform::Other,
            Duration::from_secs(300),
            &callbacks,
            || {
                polls.set(polls.get() + 1);
                std::thread::sleep(spent);
                Ok(Probe {
                    ready: polls.get() > 1,
                    summary: "slow".to_string(),
                })
            },
        )?;

        let total: Duration = slept.borrow().iter().sum();
        assert!(
            total <= budget,
            "a probe that spent {spent:?} should leave at most {budget:?} to sleep, got {total:?}",
        );
        let held = statuses
            .borrow()
            .first()
            .map(|status| status.phase)
            .context("the held poll reported no status")?;
        assert_eq!(
            held,
            ReadinessPhase::Waiting { poll_in: total },
            "the held poll must report the remainder it actually slept",
        );
        Ok(())
    }
}
