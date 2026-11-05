//! Device-alive heartbeat: the probe a worker takes before claiming a cell,
//! and the wait when a box stops answering mid-run. How long the wait lasts
//! ([`RejoinPolicy`]) is the caller's: five minutes suits a rebooting phone
//! and wastes five on an ssh host that is gone.

use std::time::Duration;

use pipette_wait::{WaitCallbacks, WaitStatus};

use crate::transport::Transport;

/// How long a worker waits on a box that stopped answering, and how often it
/// asks. [`Default`] is ten probes thirty seconds apart, five minutes in all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RejoinPolicy {
    /// Spent before *every* probe, the first included: the caller arrives
    /// having just watched a probe fail.
    pub retry_delay: Duration,
    /// Probes before giving up. `0` gives up without probing at all.
    pub max_retries: usize,
}

impl RejoinPolicy {
    pub const DEFAULT_RETRY_DELAY: Duration = Duration::from_secs(30);
    pub const DEFAULT_MAX_RETRIES: usize = 10;

    /// Each override lands on its own field, so `--rejoin-max-retries` alone
    /// keeps the 30 s spacing and vice versa.
    pub fn from_overrides(retry_delay_secs: Option<u64>, max_retries: Option<usize>) -> Self {
        Self {
            retry_delay: retry_delay_secs.map_or(Self::DEFAULT_RETRY_DELAY, Duration::from_secs),
            max_retries: max_retries.unwrap_or(Self::DEFAULT_MAX_RETRIES),
        }
    }
}

impl Default for RejoinPolicy {
    fn default() -> Self {
        Self {
            retry_delay: Self::DEFAULT_RETRY_DELAY,
            max_retries: Self::DEFAULT_MAX_RETRIES,
        }
    }
}

/// What a single probe found. Separate from [`RejoinPhase`], which has two
/// states (`Waiting`, `GaveUp`) no probe can report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum Probe {
    Online,
    /// The box answered with a non-zero exit code.
    Refused {
        exit_code: i32,
    },
    /// The transport itself failed.
    Unreachable {
        error: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum RejoinPhase {
    /// Reported *before* the sleep, so the line explains a pause that is
    /// starting.
    Waiting {
        retry_in: Duration,
    },
    Refused {
        exit_code: i32,
    },
    Unreachable {
        error: String,
    },
    Online,
    GaveUp,
}

impl RejoinPhase {
    /// The `state` a `--json` consumer branches on.
    pub(super) fn as_str(&self) -> &'static str {
        match self {
            Self::Waiting { .. } => "waiting",
            Self::Refused { .. } => "refused",
            Self::Unreachable { .. } => "unreachable",
            Self::Online => "online",
            Self::GaveUp => "gave_up",
        }
    }
}

/// One turn of a rejoin wait: the parts of the stderr line, which
/// [`Display`](std::fmt::Display) reassembles, so `--json` reads the parts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct RejoinStatus<'a> {
    pub label: &'a str,
    pub phase: RejoinPhase,
    /// Which retry this is, 1-based, out of [`RejoinPolicy::max_retries`].
    pub retry: usize,
    pub policy: RejoinPolicy,
}

/// Pinned by tests: fleet logs are grepped for these lines.
impl std::fmt::Display for RejoinStatus<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}] ", self.label)?;
        match &self.phase {
            RejoinPhase::Waiting { retry_in } => write!(
                f,
                "waiting {retry_in:?} before retry {}/{}...",
                self.retry, self.policy.max_retries
            ),
            RejoinPhase::Refused { exit_code } => write!(f, "probe exit code {exit_code}"),
            RejoinPhase::Unreachable { error } => write!(f, "still unreachable: {error}"),
            RejoinPhase::Online => f.write_str("device back online"),
            RejoinPhase::GaveUp => {
                write!(f, "giving up after {} retries", self.policy.max_retries)
            }
        }
    }
}

/// Here rather than in `pipette-wait`: this binary narrates on stderr, where
/// the runtime crates use `log`.
impl WaitStatus for RejoinStatus<'_> {
    fn report_default(&self) {
        eprintln!("{self}");
    }
}

/// A plain run passes the defaults; `--json` installs a reporter.
pub(super) type RejoinCallbacks<'a, 'label> = WaitCallbacks<'a, RejoinStatus<'label>>;

fn probe_once(transport: &Transport) -> Probe {
    match transport.probe() {
        Ok(o) if o.status == 0 => Probe::Online,
        Ok(o) => Probe::Refused {
            exit_code: o.status,
        },
        Err(e) => Probe::Unreachable {
            error: format!("{e:#}"),
        },
    }
}

/// One probe, for a worker deciding whether to claim the next cell. Silent on
/// success.
pub(super) fn probe_device(transport: &Transport, label: &str) -> bool {
    match probe_once(transport) {
        Probe::Online => true,
        Probe::Refused { exit_code } => {
            eprintln!("[{label}] device probe failed with exit code {exit_code}");
            false
        }
        Probe::Unreachable { error } => {
            eprintln!("[{label}] device probe failed: {error}");
            false
        }
    }
}

/// `true` means the box came back and the worker should re-sync and resume;
/// `false` means the retries ran out or the caller cancelled. Reported on the
/// calling thread, so a slow reporter lengthens the wait it describes.
pub(super) fn wait_for_device<'label>(
    transport: &Transport,
    label: &'label str,
    policy: RejoinPolicy,
    callbacks: &RejoinCallbacks<'_, 'label>,
) -> bool {
    wait_loop(label, policy, callbacks, || probe_once(transport))
}

/// Takes its probe so tests can drive it over scripted answers.
fn wait_loop<'label>(
    label: &'label str,
    policy: RejoinPolicy,
    callbacks: &RejoinCallbacks<'_, 'label>,
    mut probe: impl FnMut() -> Probe,
) -> bool {
    let status = |phase, retry| RejoinStatus {
        label,
        phase,
        retry,
        policy,
    };
    for retry in 1..=policy.max_retries {
        // Sleep first: the caller just watched a probe fail, and a box worth
        // waiting on is rebooting.
        callbacks.report(&status(
            RejoinPhase::Waiting {
                retry_in: policy.retry_delay,
            },
            retry,
        ));
        // In pieces, so a cancel mid-gap is honored within a second.
        if callbacks.sleep_cancellably(policy.retry_delay).is_break() {
            // No closing line: the wait was called off, not given up on.
            return false;
        }
        match probe() {
            Probe::Online => {
                callbacks.report(&status(RejoinPhase::Online, retry));
                return true;
            }
            Probe::Refused { exit_code } => {
                callbacks.report(&status(RejoinPhase::Refused { exit_code }, retry));
            }
            Probe::Unreachable { error } => {
                callbacks.report(&status(RejoinPhase::Unreachable { error }, retry));
            }
        }
    }
    callbacks.report(&status(RejoinPhase::GaveUp, policy.max_retries));
    false
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use anyhow::Context;
    use rstest::rstest;

    use super::*;

    const LABEL: &str = "adb:R5CY";

    /// A trace rather than tallies because the ordering is under test.
    #[derive(Debug, Clone, PartialEq, Eq)]
    enum Step {
        Reported(String),
        Slept(Duration),
        Probed,
    }

    /// A box whose probe gives these answers in order, with every line and
    /// sleep recorded instead of written or spent.
    struct Scripted {
        answers: RefCell<std::vec::IntoIter<Probe>>,
        trace: RefCell<Vec<Step>>,
    }

    impl Scripted {
        fn new(answers: Vec<Probe>) -> Self {
            Self {
                answers: RefCell::new(answers.into_iter()),
                trace: RefCell::new(Vec::new()),
            }
        }

        fn never_answers() -> Self {
            Self::new(Vec::new())
        }

        fn run(&self, policy: RejoinPolicy) -> bool {
            self.run_until(policy, &pipette_wait::never_cancelled)
        }

        fn run_until(&self, policy: RejoinPolicy, cancelled: &dyn Fn() -> bool) -> bool {
            let report = |status: &RejoinStatus<'_>| {
                self.trace
                    .borrow_mut()
                    .push(Step::Reported(status.to_string()));
            };
            let sleep = |d: Duration| self.trace.borrow_mut().push(Step::Slept(d));
            let callbacks = RejoinCallbacks::new()
                .with_status(&report)
                .with_cancel(cancelled)
                .with_sleep(&sleep);
            wait_loop(LABEL, policy, &callbacks, || {
                self.trace.borrow_mut().push(Step::Probed);
                self.answers
                    .borrow_mut()
                    .next()
                    // Running past the script is a box that never answers.
                    .unwrap_or_else(|| Probe::Unreachable {
                        error: "no route to host".to_string(),
                    })
            })
        }

        fn steps(&self) -> Vec<Step> {
            self.trace.borrow().clone()
        }

        fn lines(&self) -> Vec<String> {
            self.trace
                .borrow()
                .iter()
                .filter_map(|step| match step {
                    Step::Reported(line) => Some(line.clone()),
                    Step::Slept(_) | Step::Probed => None,
                })
                .collect()
        }

        /// One duration per gap: `sleep_cancellably` cuts a gap into
        /// [`pipette_wait::CANCEL_CHECK_INTERVAL`] pieces, summed back here.
        fn slept(&self) -> Vec<Duration> {
            let mut gaps: Vec<Duration> = Vec::new();
            let mut spending = false;
            for step in self.trace.borrow().iter() {
                match step {
                    Step::Slept(piece) => {
                        match gaps.last_mut().filter(|_| spending) {
                            Some(gap) => *gap += *piece,
                            None => gaps.push(*piece),
                        }
                        spending = true;
                    }
                    Step::Reported(_) | Step::Probed => spending = false,
                }
            }
            gaps
        }

        fn sleep_pieces(&self) -> usize {
            self.trace
                .borrow()
                .iter()
                .filter(|step| matches!(step, Step::Slept(_)))
                .count()
        }

        fn probes(&self) -> usize {
            self.trace
                .borrow()
                .iter()
                .filter(|step| **step == Step::Probed)
                .count()
        }
    }

    /// The `0` case: the loop reads as though it always probes once, and it
    /// does not.
    #[rstest]
    #[case::back_on_the_first_retry(
        vec![Probe::Online],
        RejoinPolicy::default(),
        true,
        1,
        vec![
            "[adb:R5CY] waiting 30s before retry 1/10...",
            "[adb:R5CY] device back online",
        ],
    )]
    #[case::refused_once_then_back(
        vec![Probe::Refused { exit_code: 1 }, Probe::Online],
        RejoinPolicy::default(),
        true,
        2,
        vec![
            "[adb:R5CY] waiting 30s before retry 1/10...",
            "[adb:R5CY] probe exit code 1",
            "[adb:R5CY] waiting 30s before retry 2/10...",
            "[adb:R5CY] device back online",
        ],
    )]
    #[case::a_policy_that_will_not_wait(
        vec![Probe::Online],
        RejoinPolicy { retry_delay: RejoinPolicy::DEFAULT_RETRY_DELAY, max_retries: 0 },
        false,
        0,
        vec!["[adb:R5CY] giving up after 0 retries"],
    )]
    fn the_wait_narrates_what_it_did(
        #[case] answers: Vec<Probe>,
        #[case] policy: RejoinPolicy,
        #[case] want_online: bool,
        #[case] want_probes: usize,
        #[case] want_lines: Vec<&str>,
    ) {
        let script = Scripted::new(answers);

        assert_eq!(script.run(policy), want_online);
        assert_eq!(script.probes(), want_probes);
        // Never sleeps without then probing, nor probes without first sleeping.
        assert_eq!(script.slept().len(), want_probes);
        assert_eq!(script.lines(), want_lines);
    }

    /// A wait that probed first would spend its first retry re-reading the
    /// failure that sent the caller here.
    #[test]
    fn the_wait_sleeps_before_its_first_probe() -> anyhow::Result<()> {
        let script = Scripted::new(vec![Probe::Online]);
        script.run(RejoinPolicy::default());

        let steps = script.steps();
        let first_probe = steps
            .iter()
            .position(|step| *step == Step::Probed)
            .context("the wait never probed at all")?;
        let first_sleep = steps
            .iter()
            .position(|step| matches!(step, Step::Slept(_)))
            .context("the wait never slept")?;
        assert!(
            first_sleep < first_probe,
            "the interval is spent before the first probe: {steps:?}",
        );
        Ok(())
    }

    #[rstest]
    #[case::the_default(RejoinPolicy::default(), 10, Duration::from_secs(30))]
    #[case::an_ssh_host_that_is_not_coming_back(
        RejoinPolicy { retry_delay: Duration::from_secs(5), max_retries: 2 },
        2,
        Duration::from_secs(5),
    )]
    #[case::a_phone_given_longer(
        RejoinPolicy { retry_delay: Duration::from_secs(60), max_retries: 20 },
        20,
        Duration::from_secs(60),
    )]
    fn the_policy_decides_how_long_a_worker_waits(
        #[case] policy: RejoinPolicy,
        #[case] want_probes: usize,
        #[case] want_delay: Duration,
    ) {
        let script = Scripted::never_answers();

        assert!(!script.run(policy));
        assert_eq!(script.probes(), want_probes);
        assert_eq!(script.slept(), vec![want_delay; want_probes]);
    }

    /// No probe and no closing line, which would claim the retries ran out
    /// when one of ten had been started.
    #[test]
    fn a_worker_called_off_mid_gap_stops_where_it_stands() {
        let script = Scripted::never_answers();
        let cancelled = || script.sleep_pieces() >= 2;

        assert!(!script.run_until(RejoinPolicy::default(), &cancelled));
        assert_eq!(script.probes(), 0);
        assert_eq!(
            script.slept(),
            vec![pipette_wait::CANCEL_CHECK_INTERVAL * 2]
        );
        assert_eq!(
            script.lines(),
            vec!["[adb:R5CY] waiting 30s before retry 1/10..."],
        );
    }

    #[rstest]
    #[case::neither(None, None, RejoinPolicy::DEFAULT_RETRY_DELAY, 10)]
    #[case::delay_only(Some(5), None, Duration::from_secs(5), 10)]
    #[case::retries_only(None, Some(3), RejoinPolicy::DEFAULT_RETRY_DELAY, 3)]
    #[case::both(Some(5), Some(3), Duration::from_secs(5), 3)]
    fn each_override_lands_on_its_own_field(
        #[case] retry_delay_secs: Option<u64>,
        #[case] max_retries: Option<usize>,
        #[case] want_delay: Duration,
        #[case] want_retries: usize,
    ) {
        let policy = RejoinPolicy::from_overrides(retry_delay_secs, max_retries);

        assert_eq!(policy.retry_delay, want_delay);
        assert_eq!(policy.max_retries, want_retries);
    }

    /// Fleet logs are grepped for these lines.
    #[rstest]
    #[case::waiting(
        RejoinPhase::Waiting { retry_in: Duration::from_secs(30) },
        3,
        "[adb:R5CY] waiting 30s before retry 3/10..."
    )]
    #[case::refused(RejoinPhase::Refused { exit_code: 127 }, 1, "[adb:R5CY] probe exit code 127")]
    #[case::unreachable(
        RejoinPhase::Unreachable { error: "ssh: connect to host: No route to host".to_string() },
        1,
        "[adb:R5CY] still unreachable: ssh: connect to host: No route to host"
    )]
    #[case::online(RejoinPhase::Online, 4, "[adb:R5CY] device back online")]
    #[case::gave_up(RejoinPhase::GaveUp, 10, "[adb:R5CY] giving up after 10 retries")]
    fn the_status_renders_the_line_the_runner_has_always_written(
        #[case] phase: RejoinPhase,
        #[case] retry: usize,
        #[case] expected: &str,
    ) {
        let status = RejoinStatus {
            label: LABEL,
            phase,
            retry,
            policy: RejoinPolicy::default(),
        };
        assert_eq!(status.to_string(), expected);
    }

    #[rstest]
    #[case(RejoinPhase::Waiting { retry_in: Duration::ZERO }, "waiting")]
    #[case(RejoinPhase::Refused { exit_code: 1 }, "refused")]
    #[case(RejoinPhase::Unreachable { error: String::new() }, "unreachable")]
    #[case(RejoinPhase::Online, "online")]
    #[case(RejoinPhase::GaveUp, "gave_up")]
    fn each_phase_keeps_its_own_word(#[case] phase: RejoinPhase, #[case] expected: &str) {
        assert_eq!(phase.as_str(), expected);
    }
}
