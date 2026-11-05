//! Claim-loop timing: how long to wait between polls and between heartbeats.
//! No I/O: every value comes from the server's `time_window` or local config.

use std::ops::ControlFlow;
use std::time::Duration;

use pipette_http::retry::{exponential_backoff, jittered};

/// Idle wait when `claim` returns `204` (the mgmt repo's
/// `docs/client-integration.md`, "Idle polling cadence").
pub const DEFAULT_IDLE_WAIT: Duration = Duration::from_secs(5 * 60);
/// Jitter range added to the idle wait (0..=this).
pub const DEFAULT_IDLE_JITTER: Duration = Duration::from_secs(60);

/// Parse the ISO 8601 form the server emits for `time_window`: `PTnHnMnS`
/// only, no days, weeks or months.
fn parse_iso8601_duration(raw: &str) -> Option<Duration> {
    let s = raw.trim();
    let rest = s.strip_prefix("PT").or_else(|| s.strip_prefix("pt"))?;
    if rest.is_empty() {
        return None;
    }
    let mut total_secs: u64 = 0;
    let mut num = 0u64;
    let mut saw_digit = false;
    for ch in rest.chars() {
        match ch {
            '0'..='9' => {
                saw_digit = true;
                num = num
                    .saturating_mul(10)
                    .saturating_add(u64::from(ch as u8 - b'0'));
            }
            'H' | 'h' if saw_digit => {
                total_secs = total_secs.saturating_add(num.saturating_mul(3600));
                num = 0;
                saw_digit = false;
            }
            'M' | 'm' if saw_digit => {
                total_secs = total_secs.saturating_add(num.saturating_mul(60));
                num = 0;
                saw_digit = false;
            }
            'S' | 's' if saw_digit => {
                total_secs = total_secs.saturating_add(num);
                num = 0;
                saw_digit = false;
            }
            _ => return None,
        }
    }
    // Trailing number without a unit is invalid.
    if saw_digit {
        return None;
    }
    Some(Duration::from_secs(total_secs))
}

/// Half of `time_window` (10 min when unparseable), floored at 1 s, unless
/// overridden. An override longer than half the window can let the lease lapse;
/// it is honored anyway and the worker warns.
pub fn resolve_heartbeat_interval(time_window: &str, override_secs: Option<u64>) -> Duration {
    let window = parse_iso8601_duration(time_window).unwrap_or(Duration::from_secs(10 * 60));
    let half = {
        let h = window / 2;
        if h.is_zero() {
            Duration::from_secs(1)
        } else {
            h
        }
    };
    match override_secs {
        Some(secs) => Duration::from_secs(secs.max(1)),
        None => half,
    }
}

/// Idle wait after a `204` claim: base + uniform jitter in `[0, jitter]`.
pub fn idle_wait_with_jitter(base: Duration, jitter: Duration) -> Duration {
    base + jittered(jitter)
}

/// Run `attempt` once per entry of `attempts` until it breaks with an answer,
/// sleeping [`exponential_backoff`] through `sleep` after each try that gives
/// none. `None` when the attempts run out, or when `sleep` breaks.
///
/// The sleep is a callback because it is the only part long enough to
/// interrupt: one that breaks (the process is shutting down) ends the retries.
pub fn retry_with_backoff<T>(
    attempts: impl IntoIterator<Item = u32>,
    first: Duration,
    cap: Duration,
    mut attempt: impl FnMut(u32) -> ControlFlow<T>,
    mut sleep: impl FnMut(Duration) -> ControlFlow<()>,
) -> Option<T> {
    for n in attempts {
        match attempt(n) {
            ControlFlow::Break(answer) => return Some(answer),
            ControlFlow::Continue(()) => {
                if sleep(exponential_backoff(n, first, cap)).is_break() {
                    return None;
                }
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[rstest::rstest]
    #[case::ten_min("PT10M", 600)]
    #[case::five_min("PT5M", 300)]
    #[case::mixed("PT1H2M3S", 3723)]
    #[case::seconds_only("PT30S", 30)]
    #[case::lowercase("pt10m", 600)]
    fn parse_iso8601_duration_ok(#[case] raw: &str, #[case] secs: u64) {
        assert_eq!(parse_iso8601_duration(raw), Some(Duration::from_secs(secs)));
    }

    #[rstest::rstest]
    #[case::empty("")]
    #[case::no_pt("10M")]
    #[case::trailing_num("PT10")]
    #[case::garbage("PTfoo")]
    fn parse_iso8601_duration_err(#[case] raw: &str) {
        assert!(parse_iso8601_duration(raw).is_none());
    }

    #[test]
    fn resolve_heartbeat_interval_default_and_override() {
        assert_eq!(
            resolve_heartbeat_interval("PT10M", None),
            Duration::from_secs(300)
        );
        assert_eq!(
            resolve_heartbeat_interval("bogus", None),
            Duration::from_secs(300)
        ); // default 10m / 2
        assert_eq!(
            resolve_heartbeat_interval("PT10M", Some(60)),
            Duration::from_secs(60)
        );
        // Floor at 1 s.
        assert_eq!(
            resolve_heartbeat_interval("PT10M", Some(0)),
            Duration::from_secs(1)
        );
        // Override longer than half-window is still accepted (caller warns).
        assert_eq!(
            resolve_heartbeat_interval("PT10M", Some(900)),
            Duration::from_secs(900)
        );
    }

    /// Tries count from one, so the first wait is `first`.
    #[test]
    fn retry_with_backoff_stops_at_the_first_answer() {
        let mut tries = Vec::new();
        let mut sleeps = Vec::new();
        let answer = retry_with_backoff(
            1..=32,
            Duration::from_secs(1),
            Duration::from_secs(30),
            |n| {
                tries.push(n);
                if n < 3 {
                    ControlFlow::Continue(())
                } else {
                    ControlFlow::Break("accepted")
                }
            },
            |d| {
                sleeps.push(d);
                ControlFlow::Continue(())
            },
        );
        assert_eq!(answer, Some("accepted"));
        assert_eq!(tries, [1, 2, 3]);
        assert_eq!(sleeps, [1, 2].map(Duration::from_secs));
    }

    /// Running out of attempts is `None`, not an answer: the caller decides what
    /// giving up means.
    #[test]
    fn retry_with_backoff_reports_exhaustion_as_none() {
        let mut sleeps = 0;
        let answer: Option<()> = retry_with_backoff(
            1..=3,
            Duration::from_secs(1),
            Duration::from_secs(30),
            |_| ControlFlow::Continue(()),
            |_| {
                sleeps += 1;
                ControlFlow::Continue(())
            },
        );
        assert_eq!(answer, None);
        assert_eq!(sleeps, 3, "one wait per try that gave no answer");
    }

    #[test]
    fn retry_with_backoff_gives_up_when_the_sleep_breaks() {
        let mut tries = Vec::new();
        let mut waits = 0;
        let answer: Option<()> = retry_with_backoff(
            1..=32,
            Duration::from_secs(1),
            Duration::from_secs(30),
            |n| {
                tries.push(n);
                ControlFlow::Continue(())
            },
            // Stopped during the second wait, having served the first in full.
            |_| {
                waits += 1;
                if waits < 2 {
                    ControlFlow::Continue(())
                } else {
                    ControlFlow::Break(())
                }
            },
        );
        assert_eq!(answer, None);
        assert_eq!(tries, [1, 2], "no third try after the stop");
    }

    #[test]
    fn idle_wait_stays_within_jitter_bounds() {
        let base = Duration::from_secs(300);
        let jitter = Duration::from_secs(60);
        for _ in 0..20 {
            let w = idle_wait_with_jitter(base, jitter);
            assert!(w >= base);
            assert!(w <= base + jitter);
        }
    }
}
