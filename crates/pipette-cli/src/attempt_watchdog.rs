//! Stops a run whose attempt passes `attempt_timeout_seconds`: a thread watches
//! the run's [`Clock`] and, past the limit, stops every child the process runs,
//! so whatever the engine waits on fails at once.

use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::sync::{Mutex, MutexGuard, PoisonError};
use std::thread::{Scope, ScopedJoinHandle};
use std::time::{Duration, Instant};

use pipette_ops::attempt::{AttemptTimeoutError, Overrun};

/// How late past the limit an overrun is caught.
const POLL: Duration = Duration::from_millis(250);

/// One run's attempt clock, marked through the run's observer and gate.
pub(crate) struct Clock {
    /// `None`: the cell has no limit, so nothing overruns.
    limit: Option<Duration>,
    attempt: Mutex<Attempt>,
}

#[derive(Debug)]
struct Attempt {
    what: String,
    /// `None` while held.
    started: Option<Instant>,
}

impl Clock {
    /// Held until the first [`Self::begin`].
    pub(crate) fn new(limit: Option<Duration>) -> Self {
        Self {
            limit,
            attempt: Mutex::new(Attempt {
                what: String::new(),
                started: None,
            }),
        }
    }

    /// A new attempt starts now, with the whole limit.
    pub(crate) fn begin(&self, what: &str) {
        *self.attempt() = Attempt::starting(what, Instant::now());
    }

    /// The clock stops until the next [`Self::resume`] or [`Self::begin`].
    pub(crate) fn hold(&self) {
        self.attempt().started = None;
    }

    /// The held attempt goes on with the whole limit.
    pub(crate) fn resume(&self) {
        self.attempt().started = Some(Instant::now());
    }

    fn overrun(&self) -> Option<Overrun> {
        self.attempt().overrun(self.limit?, Instant::now())
    }

    fn attempt(&self) -> MutexGuard<'_, Attempt> {
        self.attempt.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

impl Attempt {
    fn starting(what: &str, now: Instant) -> Self {
        Self {
            what: what.to_owned(),
            started: Some(now),
        }
    }

    /// `now` injected, so a test needs no sleep past the limit.
    fn overrun(&self, limit: Duration, now: Instant) -> Option<Overrun> {
        let elapsed = now.saturating_duration_since(self.started?);
        (elapsed > limit).then(|| Overrun {
            what: self.what.clone(),
            limit,
            elapsed,
        })
    }
}

/// Run `engine` with `clock` watched. An overrun fails the run with
/// [`AttemptTimeoutError`] whatever the engine said: its children were
/// stopped under it.
pub(crate) fn watched<T>(
    clock: &Clock,
    engine: impl FnOnce() -> anyhow::Result<T>,
) -> anyhow::Result<T> {
    if clock.limit.is_none() {
        return engine();
    }
    std::thread::scope(|scope| {
        let watchdog = Watchdog::start(scope, clock)?;
        let outcome = engine();
        match watchdog.finish() {
            Some(overrun) => Err(AttemptTimeoutError {
                overrun,
                engine_error: outcome.err().map(|err| format!("{err:#}")),
            }
            .into()),
            None => outcome,
        }
    })
}

struct Watchdog<'scope> {
    clock: &'scope Clock,
    stop: mpsc::Sender<()>,
    thread: ScopedJoinHandle<'scope, Option<Overrun>>,
}

impl<'scope> Watchdog<'scope> {
    fn start(scope: &'scope Scope<'scope, '_>, clock: &'scope Clock) -> anyhow::Result<Self> {
        let (stop, stopped) = mpsc::channel();
        let thread = std::thread::Builder::new()
            .name("attempt-watchdog".into())
            .spawn_scoped(scope, move || watch(clock, &stopped))?;
        Ok(Self {
            clock,
            stop,
            thread,
        })
    }

    /// The first overrun, or the attempt past the limit now: an engine whose
    /// own deadline is the limit fails before the next poll.
    fn finish(self) -> Option<Overrun> {
        drop(self.stop);
        let first = self.thread.join().ok().flatten();
        first.or_else(|| self.clock.overrun())
    }
}

/// Poll `clock` until the run ends. From the first overrun on, every poll
/// stops the children, so one started after the first stop goes too.
fn watch(clock: &Clock, stopped: &Receiver<()>) -> Option<Overrun> {
    let mut first = None;
    while stopped.recv_timeout(POLL) == Err(RecvTimeoutError::Timeout) {
        let looking = first.is_none();
        if looking {
            first = clock.overrun();
        }
        let Some(overrun) = &first else {
            continue;
        };
        let children = pipette_subprocess::stop_children();
        if looking || children > 0 {
            tracing::error!(
                attempt = overrun.what.as_str(),
                limit_secs = overrun.limit.as_secs(),
                elapsed_secs = overrun.elapsed.as_secs(),
                children,
                "attempt past attempt_timeout_seconds; stopped the run's children"
            );
        }
    }
    first
}

#[cfg(test)]
mod tests {
    use super::*;

    const LIMIT: Duration = Duration::from_secs(300);

    #[test]
    fn an_attempt_overruns_only_past_the_limit() -> anyhow::Result<()> {
        let start = Instant::now();
        let attempt = Attempt::starting(pipette_ops::attempt::LOAD_AND_WARM_UP, start);
        anyhow::ensure!(attempt.overrun(LIMIT, start + LIMIT).is_none());
        let overrun = attempt.overrun(LIMIT, start + Duration::from_secs(301));
        anyhow::ensure!(
            overrun
                == Some(Overrun {
                    what: pipette_ops::attempt::LOAD_AND_WARM_UP.to_string(),
                    limit: LIMIT,
                    elapsed: Duration::from_secs(301),
                }),
            "{overrun:?}"
        );
        Ok(())
    }

    #[test]
    fn a_new_attempt_gets_the_whole_limit_again() -> anyhow::Result<()> {
        let start = Instant::now();
        let attempt = Attempt::starting("rep 1 of 5", start + Duration::from_secs(290));
        anyhow::ensure!(
            attempt
                .overrun(LIMIT, start + Duration::from_secs(580))
                .is_none()
        );
        let overrun = attempt.overrun(LIMIT, start + Duration::from_secs(591));
        anyhow::ensure!(
            overrun.as_ref().map(|o| o.what.as_str()) == Some("rep 1 of 5"),
            "{overrun:?}"
        );
        Ok(())
    }

    #[test]
    fn a_held_attempt_never_overruns() -> anyhow::Result<()> {
        let start = Instant::now();
        let held = Attempt {
            what: "rep 1".into(),
            started: None,
        };
        anyhow::ensure!(
            held.overrun(LIMIT, start + Duration::from_secs(10_000))
                .is_none()
        );
        Ok(())
    }

    #[test]
    fn a_cell_without_a_limit_never_overruns() -> anyhow::Result<()> {
        let clock = Clock::new(None);
        clock.begin("rep 1");
        anyhow::ensure!(clock.overrun().is_none());
        watched(&clock, || Ok(()))
    }

    #[test]
    fn a_run_inside_the_limit_reports_no_overrun() -> anyhow::Result<()> {
        let clock = Clock::new(Some(LIMIT));
        clock.begin("rep 1");
        watched(&clock, || Ok(()))
    }

    /// A step held past the limit leaves the attempt its whole limit after.
    #[test]
    fn a_step_outside_the_limit_does_not_overrun() -> anyhow::Result<()> {
        let clock = Clock::new(Some(Duration::from_millis(300)));
        clock.begin("rep 1");
        let observer = pipette_ops::measurement::MeasurementObserver {
            on_hold: &|| clock.hold(),
            on_resume: &|| clock.resume(),
            ..pipette_ops::measurement::MeasurementObserver::QUIET
        };
        watched(&clock, || {
            observer.outside(|| std::thread::sleep(Duration::from_millis(900)));
            Ok(())
        })
    }

    /// The engine's own deadline at the limit fails it before a poll lands;
    /// the run still fails as an overrun, with the engine's error beside it.
    #[test]
    fn an_engine_error_past_the_limit_is_an_overrun() -> anyhow::Result<()> {
        let clock = Clock::new(Some(Duration::from_millis(20)));
        clock.begin("the measured peak-memory attempt");
        let Err(err) = watched(&clock, || -> anyhow::Result<()> {
            std::thread::sleep(Duration::from_millis(60));
            anyhow::bail!("llama-server did not get ready within 0.02s")
        }) else {
            anyhow::bail!("an engine failure past the limit was the engine's");
        };
        let failure = err
            .downcast_ref::<AttemptTimeoutError>()
            .ok_or_else(|| anyhow::anyhow!("not an attempt timeout: {err:#}"))?;
        anyhow::ensure!(
            failure.overrun.what == "the measured peak-memory attempt",
            "{failure:?}"
        );
        anyhow::ensure!(
            failure.engine_error.as_deref() == Some("llama-server did not get ready within 0.02s"),
            "{failure:?}"
        );
        Ok(())
    }

    /// Unix-only: the children are `sleep`s.
    #[cfg(unix)]
    mod children {
        use super::*;

        fn sleeper() -> anyhow::Result<pipette_subprocess::Process> {
            let mut sleep = std::process::Command::new("sleep");
            sleep.arg("60");
            Ok(pipette_subprocess::start_process(
                "sleep",
                sleep,
                Duration::ZERO,
            )?)
        }

        /// A child that would sleep a minute is stopped at the limit, and so
        /// is one started after that first stop; the overrun names the
        /// attempt that ran over.
        #[test]
        fn an_overrun_stops_every_child_until_the_run_ends() -> anyhow::Result<()> {
            let clock = Clock::new(Some(Duration::from_millis(300)));
            clock.begin("rep 1");
            let started = Instant::now();
            let Err(err) = watched(&clock, || -> anyhow::Result<()> {
                let first = sleeper()?.wait_success(Duration::from_secs(30));
                anyhow::ensure!(first.is_err(), "the first child exited cleanly");
                let second = sleeper()?.wait_success(Duration::from_secs(30));
                anyhow::ensure!(second.is_err(), "the later child exited cleanly");
                anyhow::bail!("both children were stopped")
            }) else {
                anyhow::bail!("an overrun passed");
            };
            anyhow::ensure!(
                started.elapsed() < Duration::from_secs(20),
                "a child ran to its own end"
            );
            let failure = err
                .downcast_ref::<AttemptTimeoutError>()
                .ok_or_else(|| anyhow::anyhow!("not an attempt timeout: {err:#}"))?;
            anyhow::ensure!(failure.overrun.what == "rep 1", "{failure:?}");
            anyhow::ensure!(
                failure.engine_error.as_deref() == Some("both children were stopped"),
                "{failure:?}"
            );
            Ok(())
        }
    }
}
