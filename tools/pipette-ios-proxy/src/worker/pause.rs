//! A phone that cannot run a job pauses claiming. A pause is logged again every
//! [`STILL_PAUSED_EVERY`] while it lasts, so an alarm can tell a pause of hours
//! from one of seconds (DESIGN section 7).

use std::time::{Duration, Instant};

use super::fleet::Phone;
use crate::device::Reachability;
use crate::timing;

pub(crate) const STILL_PAUSED_EVERY: Duration = Duration::from_secs(10 * 60);

/// A pause, from the first probe that found the phone unable to run a job.
#[derive(Debug)]
pub(crate) struct Pause {
    since: Instant,
    /// The intervals `phone still paused` has covered.
    said: u64,
}

impl Pause {
    pub(crate) fn new() -> Self {
        Self {
            since: Instant::now(),
            said: 0,
        }
    }

    pub(crate) fn minutes(&self) -> u64 {
        self.since.elapsed().as_secs() / 60
    }

    /// After each probe that finds the phone still unable to run a job: logs
    /// `phone still paused` once every [`STILL_PAUSED_EVERY`].
    pub(crate) fn remind(&mut self, reason: &str) {
        self.remind_at(Instant::now(), reason);
    }

    fn remind_at(&mut self, now: Instant, reason: &str) {
        let paused = now.saturating_duration_since(self.since);
        // A probe that outlasts an interval says it once, not once per interval.
        let due = paused.as_secs() / STILL_PAUSED_EVERY.as_secs();
        if due <= self.said {
            return;
        }
        self.said = due;
        tracing::warn!(
            paused_mins = %(paused.as_secs() / 60),
            reason = %reason,
            "phone still paused"
        );
    }
}

/// The phone cannot run a job: pause claims, or say again that they are.
pub(crate) fn pause_claims(paused: &mut Option<Pause>, reason: &str) {
    pause_or_remind(paused, reason, || {
        tracing::warn!(
            reason = %reason,
            "the phone cannot run a job; pausing claims (no job is failed for this)"
        );
    });
}

/// Start a pause with `announce`, or remind of the one under way.
pub(crate) fn pause_or_remind(paused: &mut Option<Pause>, reason: &str, announce: impl FnOnce()) {
    match paused {
        Some(pause) => pause.remind(reason),
        None => {
            announce();
            *paused = Some(Pause::new());
        }
    }
}

/// The phone can run a job: end the pause, if there is one.
pub(crate) fn resume_claims(paused: &mut Option<Pause>) {
    if let Some(pause) = paused.take() {
        tracing::info!(
            paused_mins = %pause.minutes(),
            "the phone can run a job again; resuming claims"
        );
    }
}

/// Probe the phone until it can run a job, pausing claims meanwhile. `false`
/// on a stop.
pub(crate) fn wait_until_runnable(phone: &Phone) -> bool {
    let pacing = &phone.config.loop_config;
    let mut paused = None;
    loop {
        match phone.control.reachability(&phone.entry.bundle_id) {
            Reachability::Reachable { .. } => {
                resume_claims(&mut paused);
                return true;
            }
            Reachability::Unreachable(reason) => pause_claims(&mut paused, &reason),
        }
        let wait = timing::idle_wait_with_jitter(pacing.idle_wait, pacing.idle_jitter);
        if !timing::sleep_interruptible(wait, &phone.stop) {
            return false;
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::Ordering;

    use anyhow::Context;

    use super::*;
    use crate::device::fake::FakeDevice;
    use crate::logging::testing::{captured, said};
    use crate::worker::testing::{fast_paced, phone};

    /// One line at each ten minutes the pause reaches, the first after ten,
    /// and one for a probe that took longer than an interval.
    #[test]
    fn a_pause_is_said_again_every_ten_minutes() {
        let since = Instant::now();
        let mut pause = Pause { since, said: 0 };
        let at = |mins: u64, secs: u64| since + Duration::from_secs(mins * 60 + secs);

        let ((), log) = captured(|| {
            [
                at(0, 30),
                at(9, 59),
                at(10, 0),
                at(10, 30),
                at(19, 59),
                at(20, 5),
                at(41, 0),
                at(49, 59),
                at(50, 0),
            ]
            .into_iter()
            .for_each(|now| pause.remind_at(now, "devicectl: CoreDeviceError 1011"));
        });

        assert_eq!(
            said(&log),
            [
                "WARN phone still paused paused_mins=10 reason=devicectl: CoreDeviceError 1011",
                "WARN phone still paused paused_mins=20 reason=devicectl: CoreDeviceError 1011",
                "WARN phone still paused paused_mins=41 reason=devicectl: CoreDeviceError 1011",
                "WARN phone still paused paused_mins=50 reason=devicectl: CoreDeviceError 1011",
            ]
        );
    }

    /// A phone that can run a job passes at once and says nothing; one that
    /// cannot pauses claims until a stop.
    #[rstest::rstest]
    #[case::runnable(FakeDevice::new("phone-a"), true, &[])]
    #[case::gone(
        FakeDevice::new("phone-a").gone("CoreDeviceError 1011"),
        false,
        &["WARN the phone cannot run a job; pausing claims (no job is failed for this) \
           reason=CoreDeviceError 1011"]
    )]
    fn the_wait_passes_only_a_phone_that_can_run_a_job(
        #[case] device: FakeDevice,
        #[case] runnable: bool,
        #[case] expected: &[&str],
    ) -> anyhow::Result<()> {
        let state = tempfile::tempdir()?;
        let (phone, _) = phone(device, state.path())?;
        phone.stop.store(true, Ordering::SeqCst);

        let (waited, log) = captured(|| wait_until_runnable(&phone));

        assert_eq!(waited, runnable);
        assert_eq!(said(&log), expected);
        Ok(())
    }

    /// A probe that finds the phone still unable to run a job, 11 minutes
    /// into a pause, says so.
    #[test]
    fn a_pause_under_way_is_said_again() -> anyhow::Result<()> {
        let since = Instant::now()
            .checked_sub(Duration::from_secs(11 * 60))
            .context("the clock is too young to go back 11 minutes")?;
        let mut paused = Some(Pause { since, said: 0 });

        let ((), log) = captured(|| pause_claims(&mut paused, "developer mode is disabled"));

        assert_eq!(
            said(&log),
            ["WARN phone still paused paused_mins=11 reason=developer mode is disabled"]
        );
        Ok(())
    }

    /// A phone gone for two probes pauses claims once, then resumes them and
    /// passes.
    #[test]
    fn the_wait_resumes_claims_when_the_phone_is_back() -> anyhow::Result<()> {
        let state = tempfile::tempdir()?;
        let device = FakeDevice::new("phone-a")
            .then_gone("CoreDeviceError 1011")
            .then_gone("CoreDeviceError 1011")
            .then_back();
        let (mut phone, _) = phone(device, state.path())?;
        fast_paced(&mut phone);

        let (waited, log) = captured(|| wait_until_runnable(&phone));

        assert!(waited);
        assert_eq!(
            said(&log),
            [
                "WARN the phone cannot run a job; pausing claims (no job is failed for this) \
                 reason=CoreDeviceError 1011",
                "INFO the phone can run a job again; resuming claims paused_mins=0",
            ]
        );
        Ok(())
    }
}
