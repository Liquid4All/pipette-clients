//! Retries for a download whose failure may pass: a dropped connection, a
//! `5xx`, a `429`, a body that broke part-way.
//!
//! Opt-in, and for downloads only. [`send`](crate::send) itself never retries:
//! a benchmark request that was silently sent twice would change what the
//! measurement records.
//!
//! An attempt is the whole transfer (the request, the body and the write),
//! because a body that breaks part-way is as transient as a refused connect.
//! The caller says what one attempt is and how to classify its error; this
//! module owns how many attempts there are and how long to wait between them.

use std::error::Error as StdError;
use std::time::Duration;

/// How many attempts a download gets and how long it waits between them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RetryPolicy {
    /// Attempts in total, the first included.
    pub attempts: u32,
    /// The wait after the first failed attempt, before jitter.
    pub first: Duration,
    /// The longest wait the schedule reaches, before jitter.
    pub cap: Duration,
    /// The longest `Retry-After` the retry obeys. A server that asks for a
    /// longer wait gets no retry: the job fails, and the worker claims again
    /// later.
    pub retry_after_limit: Duration,
}

impl RetryPolicy {
    /// The policy for artifact downloads: five attempts, waits from a second
    /// to thirty, and a `Retry-After` of at most two minutes.
    pub const DOWNLOAD: Self = Self {
        attempts: 5,
        first: Duration::from_secs(1),
        cap: Duration::from_secs(30),
        retry_after_limit: Duration::from_secs(120),
    };

    /// No waits, for a test that drives the retries through a mock server.
    pub const IMMEDIATE: Self = Self {
        first: Duration::ZERO,
        cap: Duration::ZERO,
        ..Self::DOWNLOAD
    };
}

/// Whether another attempt can succeed where this one failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Disposition {
    /// Every attempt fails the same way: a `404`, a digest mismatch, a full disk.
    Permanent,
    /// The failure can pass. `retry_after` is the wait the server asked for,
    /// which replaces the schedule's wait for this attempt.
    Transient { retry_after: Option<Duration> },
}

impl Disposition {
    /// A transient failure with no wait from the server.
    pub const TRANSIENT: Self = Self::Transient { retry_after: None };
}

/// The wait after `attempt` consecutive transient failures: `first`, doubling,
/// capped at `cap`.
///
/// A function of the attempt number rather than a loop counter, so the
/// schedule can be asserted for any N without a server that fails on cue.
pub fn exponential_backoff(attempt: u32, first: Duration, cap: Duration) -> Duration {
    // Thirty-one doublings of any first wait dwarf any cap; clamping the
    // exponent keeps the shift in range for a daemon left retrying for days.
    let doublings = attempt.saturating_sub(1).min(31);
    first.saturating_mul(1 << doublings).min(cap)
}

/// A wait in `[0, wait]`, picked by `roll` to the millisecond.
///
/// Full jitter: clients that fail together, such as ten devices on one host
/// that start the same download, retry at different times.
pub fn full_jitter(wait: Duration, roll: u64) -> Duration {
    let max = u64::try_from(wait.as_millis()).unwrap_or(u64::MAX);
    Duration::from_millis(roll % max.saturating_add(1))
}

/// A random wait in `[0, wait]`.
pub fn jittered(wait: Duration) -> Duration {
    full_jitter(wait, random_u64())
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "the clock fallback keeps the low 64 bits"
)]
fn random_u64() -> u64 {
    let mut buf = [0u8; 8];
    // getrandom failure is vanishingly rare; fall back to a time-derived
    // value so the retry still waits rather than panicking.
    if getrandom::fill(&mut buf).is_err() {
        return std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);
    }
    u64::from_le_bytes(buf)
}

/// `error` and each of its sources, joined with `": "`: what `{:#}` prints
/// for an `anyhow::Error`, for an error that is not one yet.
pub fn error_chain(error: &(dyn StdError + 'static)) -> String {
    std::iter::successors(Some(error), |&cause| cause.source())
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(": ")
}

/// Run `attempt` until it succeeds, fails permanently, or runs out of
/// attempts, and sleep a jittered [`exponential_backoff`] between attempts.
///
/// `attempt` gets the attempt number, counted from one. The error returned is
/// the last attempt's, unchanged, so its source chain is intact.
pub fn retrying<T, E: StdError + 'static>(
    policy: &RetryPolicy,
    url: &str,
    attempt: impl FnMut(u32) -> Result<T, E>,
    classify: impl Fn(&E) -> Disposition,
) -> Result<T, E> {
    with_retries(policy, url, attempt, classify, jittered, std::thread::sleep)
}

/// [`retrying`], with the jitter and the sleep injected, so a test can read
/// the waits without living through them.
pub fn with_retries<T, E: StdError + 'static>(
    policy: &RetryPolicy,
    url: &str,
    mut attempt: impl FnMut(u32) -> Result<T, E>,
    classify: impl Fn(&E) -> Disposition,
    jitter: impl Fn(Duration) -> Duration,
    mut sleep: impl FnMut(Duration),
) -> Result<T, E> {
    let mut n = 1;
    loop {
        let error = match attempt(n) {
            Ok(value) => {
                if n > 1 {
                    tracing::info!(%url, attempts = n, "download succeeded after retries");
                }
                return Ok(value);
            }
            Err(error) => error,
        };
        let retry_in = match classify(&error) {
            Disposition::Permanent => return Err(error),
            Disposition::Transient { .. } if n >= policy.attempts => return Err(error),
            Disposition::Transient {
                retry_after: Some(asked),
            } if asked > policy.retry_after_limit => {
                tracing::warn!(
                    %url,
                    attempt = n,
                    retry_after = ?asked,
                    limit = ?policy.retry_after_limit,
                    error = %error_chain(&error),
                    "download not retried: Retry-After exceeds the limit"
                );
                return Err(error);
            }
            Disposition::Transient {
                retry_after: Some(asked),
            } => asked,
            Disposition::Transient { retry_after: None } => {
                jitter(exponential_backoff(n, policy.first, policy.cap))
            }
        };
        tracing::warn!(
            %url,
            attempt = n,
            retry_in = ?retry_in,
            error = %error_chain(&error),
            "download attempt failed; retrying"
        );
        sleep(retry_in);
        n += 1;
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::io;
    use std::sync::{Arc, Mutex};

    use super::*;

    /// The schedules the workspace runs: the worker's claim, from a second to
    /// a minute; its submit, from a second to thirty; and a download's, the
    /// same as the submit's. A worker left retrying against a dead server for
    /// days still gets the cap, not an overflowed shift.
    #[rstest::rstest]
    #[case::claim_first(1, 60, 1)]
    #[case::claim_second(2, 60, 2)]
    #[case::claim_sixth(6, 60, 32)]
    #[case::claim_capped(7, 60, 60)]
    #[case::claim_still_capped(8, 60, 60)]
    #[case::submit_fifth(5, 30, 16)]
    #[case::submit_capped(6, 30, 30)]
    #[case::far_past_the_cap(4_000_000_000, 60, 60)]
    fn exponential_backoff_doubles_from_the_first_wait_to_the_cap(
        #[case] attempt: u32,
        #[case] cap_secs: u64,
        #[case] expected_secs: u64,
    ) {
        assert_eq!(
            exponential_backoff(
                attempt,
                Duration::from_secs(1),
                Duration::from_secs(cap_secs)
            ),
            Duration::from_secs(expected_secs)
        );
    }

    /// The roll picks a wait from zero to the whole wait, both ends included.
    #[rstest::rstest]
    #[case::zero_roll(0, 0)]
    #[case::middle(1_500, 1_500)]
    #[case::top(4_000, 4_000)]
    #[case::wraps(4_001, 0)]
    #[case::max_roll(u64::MAX, u64::MAX % 4_001)]
    fn full_jitter_stays_within_the_wait(#[case] roll: u64, #[case] expected_ms: u64) {
        assert_eq!(
            full_jitter(Duration::from_secs(4), roll),
            Duration::from_millis(expected_ms)
        );
    }

    #[test]
    fn full_jitter_of_no_wait_is_no_wait() {
        assert_eq!(full_jitter(Duration::ZERO, u64::MAX), Duration::ZERO);
    }

    #[derive(Debug, thiserror::Error)]
    #[error("attempt {0} failed")]
    struct AttemptFailedError(u32, #[source] io::Error);

    /// Drive [`with_retries`] with no jitter: attempt `n` fails as
    /// `outcome(n)` says, or succeeds when it says `None`. Returns the result,
    /// the attempts made and the waits slept.
    fn drive(
        outcome: impl Fn(u32) -> Option<Disposition>,
    ) -> (Result<u32, AttemptFailedError>, Vec<u32>, Vec<Duration>) {
        let tries = RefCell::new(Vec::new());
        let mut sleeps = Vec::new();
        let result = with_retries(
            &RetryPolicy::DOWNLOAD,
            "http://example.test/archive.tar.gz",
            |n| {
                tries.borrow_mut().push(n);
                match outcome(n) {
                    None => Ok(n),
                    Some(_) => Err(AttemptFailedError(n, io::Error::other("connection reset"))),
                }
            },
            |AttemptFailedError(n, _)| outcome(*n).unwrap_or(Disposition::Permanent),
            |wait| wait,
            |wait| sleeps.push(wait),
        );
        (result, tries.into_inner(), sleeps)
    }

    #[test]
    fn a_transient_failure_is_retried_on_the_schedule() -> anyhow::Result<()> {
        let (result, tries, sleeps) = drive(|n| (n < 3).then_some(Disposition::TRANSIENT));
        anyhow::ensure!(result? == 3);
        assert_eq!(tries, [1, 2, 3]);
        assert_eq!(sleeps, [1, 2].map(Duration::from_secs));
        Ok(())
    }

    /// Every attempt fails: the policy's attempts are made, with a wait
    /// between each pair and none after the last, and the last error returns.
    #[test]
    fn the_attempts_run_out() {
        let (result, tries, sleeps) = drive(|_| Some(Disposition::TRANSIENT));
        assert!(
            matches!(result, Err(AttemptFailedError(5, _))),
            "{result:?}"
        );
        assert_eq!(tries, [1, 2, 3, 4, 5]);
        assert_eq!(sleeps, [1, 2, 4, 8].map(Duration::from_secs));
    }

    #[test]
    fn a_permanent_failure_is_not_retried() {
        let (result, tries, sleeps) = drive(|_| Some(Disposition::Permanent));
        assert!(
            matches!(result, Err(AttemptFailedError(1, _))),
            "{result:?}"
        );
        assert_eq!(tries, [1]);
        assert!(sleeps.is_empty());
    }

    /// The server's `Retry-After` replaces the schedule's wait, up to the
    /// limit; past it the retry stops.
    #[rstest::rstest]
    #[case::obeyed(7, &[1, 2], &[7])]
    #[case::at_the_limit(120, &[1, 2], &[120])]
    #[case::past_the_limit(121, &[1], &[])]
    fn retry_after_replaces_the_wait(
        #[case] asked_secs: u64,
        #[case] expected_tries: &[u32],
        #[case] expected_sleeps: &[u64],
    ) {
        let asked = Disposition::Transient {
            retry_after: Some(Duration::from_secs(asked_secs)),
        };
        let (_, tries, sleeps) = drive(|n| (n < 2).then_some(asked));
        assert_eq!(tries, expected_tries);
        assert_eq!(
            sleeps,
            expected_sleeps
                .iter()
                .copied()
                .map(Duration::from_secs)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn the_chain_names_every_cause() {
        let error = AttemptFailedError(2, io::Error::other("connection reset"));
        assert_eq!(error_chain(&error), "attempt 2 failed: connection reset");
    }

    #[derive(Clone, Default)]
    struct Captured(Arc<Mutex<Vec<u8>>>);

    impl io::Write for Captured {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            self.0
                .lock()
                .map_err(|_| io::Error::other("poisoned"))?
                .extend_from_slice(buf);
            Ok(buf.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    /// N transient failures log N warnings, each with its cause, and one
    /// success line.
    #[test]
    fn each_retry_is_logged() -> anyhow::Result<()> {
        let captured = Captured::default();
        let writer = captured.clone();
        let subscriber = tracing_subscriber::fmt()
            .with_writer(move || writer.clone())
            .with_ansi(false)
            .finish();
        tracing::subscriber::with_default(subscriber, || {
            drive(|n| (n < 4).then_some(Disposition::TRANSIENT))
        })
        .0?;
        let log = String::from_utf8(
            captured
                .0
                .lock()
                .map_err(|_| anyhow::anyhow!("poisoned"))?
                .clone(),
        )?;
        let warnings: Vec<_> = log
            .lines()
            .filter(|line| line.contains("download attempt failed; retrying"))
            .collect();
        assert_eq!(warnings.len(), 3, "{log}");
        assert!(
            warnings
                .iter()
                .all(|line| line.contains("connection reset") && line.contains("retry_in")),
            "{log}"
        );
        let successes: Vec<_> = log
            .lines()
            .filter(|line| line.contains("download succeeded after retries"))
            .collect();
        assert!(
            matches!(successes.as_slice(), [line] if line.contains("attempts=4")),
            "{log}"
        );
        Ok(())
    }
}
