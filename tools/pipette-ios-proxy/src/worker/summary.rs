//! The `job finished` line every claimed job ends with (DESIGN section 7): how
//! it ended, the build it ran on, and where its time went.

use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use pipette_mgmt_client::types::ClaimedJob;

use crate::app::Signals;
use crate::releases::BuildVersion;
use crate::server::submit::FailureCause;

/// How a claimed job ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq, strum::IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
pub(crate) enum Outcome {
    /// Every result the run left was accepted.
    Accepted,
    /// The run succeeded, but a result was refused or could not be submitted.
    NotAccepted,
    /// A failure was reported, or its report was tried.
    Failed(FailureCause),
    /// The lease was lost; nothing was reported.
    LeaseLost,
    /// A stop signal ended the job; nothing was reported.
    Stopped,
}

/// Why a job ended with nothing reported.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Unreported {
    LeaseLost,
    Stopped,
}

impl Unreported {
    /// A lost lease wins over a stop. A stop never loses the lease (the
    /// heartbeat renews it until the step returns), so with both raised the
    /// server took the job back.
    pub(crate) fn of(signals: Signals) -> Self {
        if signals.abort.load(Ordering::SeqCst) {
            Self::LeaseLost
        } else {
            Self::Stopped
        }
    }
}

impl From<Unreported> for Outcome {
    fn from(unreported: Unreported) -> Self {
        match unreported {
            Unreported::LeaseLost => Self::LeaseLost,
            Unreported::Stopped => Self::Stopped,
        }
    }
}

/// `chosen_by` in pool mode: the pool build that provides the runtime.
pub(crate) const CHOSEN_BY_POOL: &str = "pool";

/// What the line reports besides the outcome, filled in as the job goes. A
/// phase the job never reached stays `None`.
#[derive(Debug, Default)]
pub(crate) struct JobSummary {
    /// The build version mode chose.
    pub(crate) version: Option<BuildVersion>,
    /// `default` or `newest` in version mode, [`CHOSEN_BY_POOL`] in pool mode.
    pub(crate) chosen_by: Option<&'static str>,
    /// `device install app` put a build on the phone for this job.
    pub(crate) installed: bool,
    pub(crate) choose: Option<Duration>,
    pub(crate) pull: Option<Duration>,
    pub(crate) sign: Option<Duration>,
    pub(crate) install: Option<Duration>,
    pub(crate) run: Option<Duration>,
    pub(crate) submit: Option<Duration>,
}

impl JobSummary {
    /// One line, every field always present and in this order, `-` for none,
    /// and no value holding a space, so a metric filter can match any field as
    /// a term.
    pub(crate) fn log(&self, job: &ClaimedJob, outcome: Outcome, total: Duration) {
        let failed = match outcome {
            Outcome::Failed(cause) => Some(cause),
            Outcome::Accepted | Outcome::NotAccepted | Outcome::LeaseLost | Outcome::Stopped => {
                None
            }
        };
        tracing::info!(
            outcome = %<&str>::from(outcome),
            cause = %failed.map_or("-", <&str>::from),
            infra = %failed.map_or("-", |cause| if cause.infrastructure() { "yes" } else { "no" }),
            benchmark = %job.benchmark_id,
            version = %self.version.as_ref().map_or("-", BuildVersion::as_str),
            chosen_by = %self.chosen_by.unwrap_or("-"),
            installed = %if self.installed { "yes" } else { "no" },
            choose_secs = %secs(self.choose),
            pull_secs = %secs(self.pull),
            sign_secs = %secs(self.sign),
            install_secs = %secs(self.install),
            run_secs = %secs(self.run),
            submit_secs = %secs(self.submit),
            total_secs = %secs(Some(total)),
            "job finished"
        );
    }
}

/// Run `step`, adding its time to `phase` whatever it returns.
pub(crate) fn timed<T>(phase: &mut Option<Duration>, step: impl FnOnce() -> T) -> T {
    let started = Instant::now();
    let result = step();
    *phase = Some(phase.unwrap_or_default() + started.elapsed());
    result
}

fn secs(phase: Option<Duration>) -> String {
    phase.map_or_else(
        || "-".to_owned(),
        |took| format!("{:.1}", took.as_secs_f64()),
    )
}

#[cfg(test)]
pub(crate) mod testing {
    use anyhow::Context;

    /// The `key=value` fields of the one `job finished` line in `log`, in order.
    pub(crate) fn summary_fields(log: &str) -> anyhow::Result<Vec<String>> {
        let mut lines = log.lines().filter(|line| line.contains(" job finished "));
        let line = lines.next().context("no `job finished` line")?;
        anyhow::ensure!(lines.next().is_none(), "more than one `job finished` line");
        let (_, fields) = line
            .split_once(" job finished ")
            .context("the line has no fields")?;
        Ok(fields.split(' ').map(str::to_owned).collect())
    }

    /// Whether `field` is `expected`, where `name=#` stands for any seconds.
    pub(crate) fn field_matches(field: &str, expected: &str) -> bool {
        match expected.strip_suffix('#') {
            Some(name) => field
                .strip_prefix(name)
                .is_some_and(|secs| secs.parse::<f64>().is_ok()),
            None => field == expected,
        }
    }

    /// Every one of `expected` is among `fields`, as [`field_matches`] reads it.
    pub(crate) fn has_fields(fields: &[String], expected: &[&str]) {
        expected.iter().for_each(|expected| {
            assert!(
                fields.iter().any(|field| field_matches(field, expected)),
                "{expected}: {fields:?}"
            );
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::logging::testing::captured;

    fn claim() -> ClaimedJob {
        ClaimedJob {
            job_id: "job-1".to_owned(),
            benchmark_id: "decode_throughput_256".to_owned(),
            time_window: "PT10M".to_owned(),
            expires_at: None,
            client_versions: None,
            spec: serde_json::Value::Null,
        }
    }

    /// The line the alarms read: every field, in order, `-` where the job never
    /// got.
    #[test]
    fn the_line_names_every_field_in_order() -> anyhow::Result<()> {
        let summary = JobSummary {
            version: Some("2026.10.02-e6a9434-internal".parse()?),
            chosen_by: Some("default"),
            installed: true,
            choose: Some(Duration::from_millis(40)),
            pull: Some(Duration::from_millis(12_340)),
            install: Some(Duration::from_secs(16)),
            ..JobSummary::default()
        };

        let ((), log) = captured(|| {
            summary.log(
                &claim(),
                Outcome::Failed(FailureCause::DevicectlFailure),
                Duration::from_millis(31_240),
            );
        });

        assert_eq!(
            testing::summary_fields(&log)?,
            [
                "outcome=failed",
                "cause=devicectl_failure",
                "infra=yes",
                "benchmark=decode_throughput_256",
                "version=2026.10.02-e6a9434-internal",
                "chosen_by=default",
                "installed=yes",
                "choose_secs=0.0",
                "pull_secs=12.3",
                "sign_secs=-",
                "install_secs=16.0",
                "run_secs=-",
                "submit_secs=-",
                "total_secs=31.2",
            ]
        );
        assert!(log.contains(" INFO job finished outcome="), "{log}");
        Ok(())
    }

    #[test]
    fn a_phase_timed_again_adds_to_its_time() {
        let step = Duration::from_millis(5);
        let mut phase = Some(Duration::from_secs(2));
        let value = timed(&mut phase, || {
            std::thread::sleep(step);
            7
        });
        assert_eq!(value, 7);
        assert!(
            phase.is_some_and(|took| took >= Duration::from_secs(2) + step),
            "{phase:?}"
        );
    }

    /// A lost lease names the job's end even when a stop came too.
    #[rstest::rstest]
    #[case::lost(true, false, Unreported::LeaseLost)]
    #[case::stopped(false, true, Unreported::Stopped)]
    #[case::both(true, true, Unreported::LeaseLost)]
    fn a_lost_lease_wins_over_a_stop(
        #[case] lost: bool,
        #[case] stop: bool,
        #[case] expected: Unreported,
    ) {
        let lost = std::sync::atomic::AtomicBool::new(lost);
        let stop = std::sync::atomic::AtomicBool::new(stop);
        let signals = Signals {
            abort: &lost,
            stop: &stop,
        };
        assert_eq!(Unreported::of(signals), expected);
    }
}
