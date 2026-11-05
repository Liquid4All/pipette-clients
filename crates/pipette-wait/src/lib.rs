//! The caller's side of a blocking wait: how each poll is narrated, when the
//! wait gives up, and how it spends the gap between polls. The loops are not
//! shared: each wait has its own probe and its own deadline.
//!
//! [`sleep_cancellably`] is what makes the cancel honest: a wait that read its
//! flag only between polls would still hold an operator for a whole interval.

use std::ops::ControlFlow;
use std::time::{Duration, Instant};

/// Longest a wait sleeps before re-reading its cancel callback: the delay a
/// person deserves after pressing ^C, whether the wait was a 3 s sensor poll or
/// a 60 s claim backoff. Not finer, because below a second the operator cannot
/// tell and the wake-ups stop being free on a phone; the iOS port takes the
/// same second.
pub const CANCEL_CHECK_INTERVAL: Duration = Duration::from_secs(1);

/// How a status narrates itself when the caller has not said otherwise.
///
/// The default lives with the domain because callers narrate differently
/// (`tracing` events in the CLI and its runtime crates, `eprintln!` in the plan
/// runner), and a shared one would drag a facade into every dependent. The
/// *level* is domain knowledge too: a spent reindex budget is a warning, a
/// readiness poll is an info line.
pub trait WaitStatus {
    /// Called only when no [`WaitCallbacks::with_status`] was installed.
    fn report_default(&self);
}

/// The cancel a caller with no stop signal of its own passes: the default, and
/// what a caller that takes a `&dyn Fn() -> bool` writes rather than spelling
/// `&|| false` at each call site.
pub fn never_cancelled() -> bool {
    false
}

/// These callbacks let the caller observe probes, receive statuses, cancel the
/// wait, and control sleep between polls.
///
/// These callbacks do not define the wait criteria. The plan defines the
/// deadline and poll interval. Status, cancel and sleep match the mobile
/// readiness gates: the iOS port uses these names, and Android's `Readiness`
/// uses `sleepMillis` and `waitUntilReady(cancelFlag, onStatus)`.
///
/// Borrowed (`&dyn`) because these waits are synchronous and nested inside the
/// caller's frame, so the callbacks can be closures over its stack.
///
/// **Installing [`Self::with_status`] takes over the narration**: a terminal
/// drawing a live line has to stop the per-poll log line, not compete with it
/// for the cursor.
pub struct WaitCallbacks<'a, S> {
    /// `None` is "narrate as you always did". An `Option` rather than a default
    /// closure because the default is the domain's, and reaching it through
    /// [`WaitStatus`] is what keeps this crate free of a logging facade.
    on_status: Option<&'a dyn Fn(&S)>,
    on_poll: Option<&'a dyn Fn()>,
    cancelled: &'a dyn Fn() -> bool,
    sleep: &'a dyn Fn(Duration),
}

impl<S> Default for WaitCallbacks<'_, S> {
    fn default() -> Self {
        Self {
            on_status: None,
            on_poll: None,
            cancelled: &never_cancelled,
            sleep: &std::thread::sleep,
        }
    }
}

impl<'a, S> WaitCallbacks<'a, S> {
    /// The domain's own narration, never cancel, sleep on the calling thread.
    pub fn new() -> Self {
        Self::default()
    }

    /// Report every poll here instead of to the domain's default narration.
    ///
    /// The loop calls this callback on the caller's thread when it has a
    /// status. A failed probe can have no status; use [`Self::with_poll`] to
    /// count every attempt.
    #[must_use]
    pub fn with_status(mut self, on_status: &'a dyn Fn(&S)) -> Self {
        self.on_status = Some(on_status);
        self
    }

    /// Observe a completed probe even when it produced no status. The loop
    /// calls [`Self::polled`] once after each probe. A call to [`Self::report`]
    /// does not call it again.
    #[must_use]
    pub fn with_poll(mut self, on_poll: &'a dyn Fn()) -> Self {
        self.on_poll = Some(on_poll);
        self
    }

    /// Notify the caller that one probe completed.
    pub fn polled(&self) {
        if let Some(on_poll) = self.on_poll {
            on_poll();
        }
    }

    /// Abandon the wait when this reads `true`.
    ///
    /// Read before the first probe, after each probe, and between the
    /// [`CANCEL_CHECK_INTERVAL`] pieces a sleep is cut into, so a flag raised
    /// during a five-minute wait is honored in about a second rather than at
    /// the deadline. It cannot interrupt a probe already in flight: several
    /// readiness platforms spend ~1 s sampling CPU counters, and an HTTP probe
    /// is bound by its own request timeout.
    #[must_use]
    pub fn with_cancel(mut self, cancelled: &'a dyn Fn() -> bool) -> Self {
        self.cancelled = cancelled;
        self
    }

    /// Spend the interval between polls here instead of in
    /// [`std::thread::sleep`].
    ///
    /// A test drives the real loop over scripted probe results and records the
    /// durations instead of spending them. Handed the already-computed
    /// remainder, so a recorder sees the interval the loop actually chose.
    #[must_use]
    pub fn with_sleep(mut self, sleep: &'a dyn Fn(Duration)) -> Self {
        self.sleep = sleep;
        self
    }

    /// Whether the caller has asked the wait to give up.
    pub fn is_cancelled(&self) -> bool {
        (self.cancelled)()
    }

    /// [`sleep_cancellably`] over this wait's own cancel and sleep.
    pub fn sleep_cancellably(&self, total: Duration) -> ControlFlow<()> {
        sleep_cancellably(total, self.cancelled, self.sleep)
    }

    /// [`Self::sleep_cancellably`] for the shorter of `interval` and the time
    /// left to `deadline`.
    ///
    /// A loop that checks its deadline and then sleeps a whole interval
    /// reports its timeout an interval late; the overshoot grows with the
    /// interval. Report the clamped value as the loop's `poll_in`, since that
    /// is the gap it chose.
    pub fn sleep_cancellably_until(
        &self,
        deadline: Instant,
        interval: Duration,
    ) -> ControlFlow<()> {
        self.sleep_cancellably(clamp_to_deadline(interval, deadline))
    }
}

/// The shorter of `interval` and what is left before `deadline`; zero once it
/// has passed.
pub fn clamp_to_deadline(interval: Duration, deadline: Instant) -> Duration {
    interval.min(deadline.saturating_duration_since(Instant::now()))
}

impl<S: WaitStatus> WaitCallbacks<'_, S> {
    /// Narrate a status: through [`Self::with_status`] if the caller installed
    /// one, otherwise through [`WaitStatus::report_default`].
    pub fn report(&self, status: &S) {
        match self.on_status {
            Some(on_status) => on_status(status),
            None => status.report_default(),
        }
    }
}

/// Sleep `total` in pieces no longer than [`CANCEL_CHECK_INTERVAL`], reading
/// `cancelled` between them. [`ControlFlow::Break`] means the caller has
/// cancelled; any remainder was left unspent.
///
/// A free function as well as a [`WaitCallbacks`] method because the waits that
/// need it most have no status to report: a claim backoff is a duration and
/// nothing else, and building a `WaitCallbacks<()>` to reach this would be
/// ceremony around two closures.
///
/// A zero or already-elapsed `total` still reads the flag once, so a poll whose
/// probe outran its own interval is not a poll that cannot be cancelled.
///
/// The remainder is decremented rather than measured against a deadline, so a
/// test that injects a `sleep` which does not actually sleep still terminates
/// in `total / CANCEL_CHECK_INTERVAL` pieces instead of spinning until the wall
/// clock catches up.
pub fn sleep_cancellably(
    total: Duration,
    cancelled: &dyn Fn() -> bool,
    sleep: &dyn Fn(Duration),
) -> ControlFlow<()> {
    let mut remaining = total;
    while remaining > Duration::ZERO {
        if cancelled() {
            return ControlFlow::Break(());
        }
        let chunk = remaining.min(CANCEL_CHECK_INTERVAL);
        sleep(chunk);
        remaining -= chunk;
    }
    if cancelled() {
        ControlFlow::Break(())
    } else {
        ControlFlow::Continue(())
    }
}

#[cfg(test)]
mod tests {
    use std::cell::{Cell, RefCell};

    use rstest::rstest;

    use super::*;

    /// Stands in for the `tracing::info!` / `tracing::warn!` the real statuses
    /// do, so the default narration is observable.
    struct Spoken {
        said: Cell<usize>,
    }

    impl WaitStatus for Spoken {
        fn report_default(&self) {
            self.said.set(self.said.get() + 1);
        }
    }

    #[test]
    fn a_status_narrates_itself_when_no_reporter_is_installed() {
        let status = Spoken { said: Cell::new(0) };
        WaitCallbacks::new().report(&status);
        assert_eq!(status.said.get(), 1);
    }

    #[test]
    fn an_installed_reporter_takes_over_the_narration() {
        let status = Spoken { said: Cell::new(0) };
        let seen = Cell::new(0);
        let report = |_: &Spoken| seen.set(seen.get() + 1);
        WaitCallbacks::new().with_status(&report).report(&status);
        assert_eq!(seen.get(), 1, "the installed reporter was not called");
        assert_eq!(status.said.get(), 0, "the default narrated as well");
    }

    #[test]
    fn a_poll_observer_is_independent_of_status_reporting() {
        let status = Spoken { said: Cell::new(0) };
        let polls = Cell::new(0usize);
        let on_poll = || polls.set(polls.get() + 1);
        let callbacks = WaitCallbacks::new().with_poll(&on_poll);
        callbacks.polled();
        callbacks.report(&status);
        assert_eq!(polls.get(), 1);
        assert_eq!(status.said.get(), 1);
    }

    /// The last piece is the remainder rather than a whole interval, or a wait
    /// would overrun the span it was given.
    #[rstest]
    #[case::under_one_interval(Duration::from_millis(250), &[Duration::from_millis(250)])]
    #[case::exactly_one_interval(CANCEL_CHECK_INTERVAL, &[CANCEL_CHECK_INTERVAL])]
    #[case::two_and_a_half(
        Duration::from_millis(2_500),
        &[CANCEL_CHECK_INTERVAL, CANCEL_CHECK_INTERVAL, Duration::from_millis(500)]
    )]
    fn a_wait_is_spent_in_pieces_no_longer_than_the_check_interval(
        #[case] total: Duration,
        #[case] want: &[Duration],
    ) {
        let slept = RefCell::new(Vec::new());
        let flow = sleep_cancellably(total, &|| false, &|d| slept.borrow_mut().push(d));
        assert_eq!(flow, ControlFlow::Continue(()));
        assert_eq!(*slept.borrow(), want);
    }

    #[test]
    fn a_cancel_mid_wait_leaves_the_remainder_unspent() {
        let slept = Cell::new(0usize);
        // Set during the second piece of a ten-piece wait.
        let cancelled = || slept.get() >= 2;
        let flow = sleep_cancellably(CANCEL_CHECK_INTERVAL * 10, &cancelled, &|_| {
            slept.set(slept.get() + 1)
        });
        assert_eq!(flow, ControlFlow::Break(()));
        assert_eq!(slept.get(), 2, "the wait slept past its cancel");
    }

    /// An hour, because the point is that the span no longer decides when the
    /// wait returns.
    #[test]
    fn a_wait_that_is_already_cancelled_spends_nothing() {
        let slept = Cell::new(false);
        let flow = sleep_cancellably(Duration::from_secs(3_600), &|| true, &|_| slept.set(true));
        assert_eq!(flow, ControlFlow::Break(()));
        assert!(!slept.get(), "a cancelled wait slept anyway");
    }

    #[rstest]
    #[case::budget_is_the_longer(
        Duration::from_secs(3),
        Duration::from_secs(60),
        Duration::from_secs(3)
    )]
    #[case::interval_is_the_longer(
        Duration::from_secs(3),
        Duration::from_secs(1),
        Duration::from_secs(1)
    )]
    #[case::deadline_already_passed(Duration::from_secs(3), Duration::ZERO, Duration::ZERO)]
    fn a_gap_is_clamped_to_what_is_left_of_the_budget(
        #[case] interval: Duration,
        #[case] left: Duration,
        #[case] want: Duration,
    ) {
        let deadline = Instant::now() + left;
        let clamped = clamp_to_deadline(interval, deadline);
        // The clock moves between `now()`s: a hair under `want` is fine, over is not.
        assert!(clamped <= want, "clamped {clamped:?} past {want:?}");
        assert!(
            want.saturating_sub(clamped) < Duration::from_millis(50),
            "clamped {clamped:?} far short of {want:?}"
        );
    }

    /// A poll whose probe outran its interval computes a zero top-up.
    #[rstest]
    #[case::not_cancelled(false, ControlFlow::Continue(()))]
    #[case::cancelled(true, ControlFlow::Break(()))]
    fn a_zero_wait_still_answers_the_cancel(
        #[case] cancelled: bool,
        #[case] want: ControlFlow<()>,
    ) {
        let slept = Cell::new(false);
        let flow = sleep_cancellably(Duration::ZERO, &|| cancelled, &|_| slept.set(true));
        assert_eq!(flow, want);
        assert!(!slept.get(), "a zero wait slept");
    }
}
