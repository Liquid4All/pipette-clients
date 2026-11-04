//! Claim-loop timing and the clock format the logs print. The heartbeat and
//! `time_window` parsing mirror `crates/pipette-cli/src/client/worker/timing.rs`.

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use time::OffsetDateTime;

pub(crate) const DEFAULT_IDLE_WAIT: Duration = Duration::from_secs(30);
pub(crate) const DEFAULT_IDLE_JITTER: Duration = Duration::from_secs(10);
pub(crate) const DEFAULT_HEARTBEAT: Duration = Duration::from_secs(20);
pub(crate) const DEFAULT_CLAIM_BACKOFF: Duration = Duration::from_secs(5);

pub(crate) const DEFAULT_CALL_TIMEOUT: Duration = Duration::from_secs(60);
pub(crate) const DEFAULT_RUN_TIMEOUT: Duration = Duration::from_secs(1800);

/// `[devicectl] first_line_timeout_secs` when unset. A build with the launch marker
/// prints it within a second of starting. On 6119 lab runs of builds without it
/// (2026-10-12 to 10-22), the first line came at most 48 s after the spawn, nearly
/// all of it devicectl's own launch, and at most 30 s (the app's request timeout)
/// after devicectl's `Launched application` line.
pub(crate) const DEFAULT_FIRST_LINE_TIMEOUT: Duration = Duration::from_secs(60);

/// `[signing] export_timeout_secs` when unset; a fleet export takes about 10 s.
pub(crate) const DEFAULT_EXPORT_TIMEOUT: Duration = Duration::from_secs(300);

/// How long a claim waits for another process's pull to free `releases.lock`
/// before reporting `build_not_ready`. The heartbeat runs meanwhile.
pub(crate) const LOCK_WAIT_BUDGET: Duration = Duration::from_secs(5 * 60);

pub(crate) const LOCK_POLL: Duration = Duration::from_secs(1);

/// How old the advertised catalog may get before the claim loop lists the
/// bucket again, holding no lease.
pub(crate) const CATALOG_REFRESH: Duration = Duration::from_secs(15 * 60);

/// Consecutive `build_not_ready` or `devicectl_failure` jobs back off from
/// `[loop] claim_backoff_secs` doubling to this, so an S3 or Apple outage or a
/// full phone burns a handful of jobs, not one a minute.
pub(crate) const PREPARE_BACKOFF_CAP: Duration = Duration::from_secs(15 * 60);

/// The lease window assumed when `time_window` is unparseable.
const FALLBACK_TIME_WINDOW: Duration = Duration::from_secs(10 * 60);

/// Claim backoff doubles from `[loop] claim_backoff_secs` to this cap.
pub(crate) const CLAIM_BACKOFF_CAP: Duration = Duration::from_secs(60);

/// Submit retry: 1 s doubling to a 30 s cap, 32 attempts (as `pipette-cli`).
pub(crate) const SUBMIT_BACKOFF_BASE: Duration = Duration::from_secs(1);
pub(crate) const SUBMIT_BACKOFF_CAP: Duration = Duration::from_secs(30);
pub(crate) const SUBMIT_MAX_ATTEMPTS: u32 = 32;

/// How long a heartbeat waits after a `Retry` disposition before trying again.
pub(crate) const HEARTBEAT_RETRY_WAIT: Duration = Duration::from_secs(2);

/// How often the startup gate re-reads `GET /clients/me`.
pub(crate) const REINDEX_POLL_INTERVAL: Duration = Duration::from_secs(5);

/// How long that gate waits for `status` Approved AND `reindex_pending` clear.
pub(crate) const APPROVAL_WAIT_BUDGET: Duration = Duration::from_secs(300);

/// Longest single sleep taken by [`sleep_interruptible`].
pub(crate) const SLEEP_SLICE: Duration = Duration::from_secs(1);

/// Whole-second RFC 3339 UTC with `Z`. A fixed description rather than `time`'s
/// `Rfc3339`, which emits fractional digits when the nanosecond field is non-zero.
pub(crate) fn rfc3339_now() -> String {
    rfc3339(OffsetDateTime::now_utc())
}

pub(crate) fn rfc3339(at: OffsetDateTime) -> String {
    // A server-supplied time can fall past year 9999 in UTC.
    at.checked_to_offset(time::UtcOffset::UTC)
        .and_then(|utc| {
            utc.format(&time::macros::format_description!(
                "[year]-[month]-[day]T[hour]:[minute]:[second]Z"
            ))
            .ok()
        })
        .unwrap_or_default()
}

/// Parse the server's `time_window` (`PTnHnMnS`, H/M/S only, case-insensitive).
/// A trailing unitless number is INVALID: `PT10` is `None`, not ten seconds.
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
    if saw_digit {
        return None;
    }
    Some(Duration::from_secs(total_secs))
}

/// Heartbeat period for one claim: half the lease window floored at 1 s, or the
/// override, which is honoured even past half the window (the caller warns).
pub(crate) fn resolve_heartbeat_interval(
    time_window: &str,
    override_secs: Option<u64>,
) -> Duration {
    let window = parse_iso8601_duration(time_window).unwrap_or(FALLBACK_TIME_WINDOW);
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

/// Idle wait after a `204` claim: `base` with SYMMETRIC whole-second jitter in
/// `[base - jitter, base + jitter]`, so the mean is `base`. (The CLI worker's
/// jitter is one-sided; deliberately not copied.)
pub(crate) fn idle_wait_with_jitter(base: Duration, jitter: Duration) -> Duration {
    let jitter_secs = jitter.as_secs();
    if jitter_secs == 0 {
        return base;
    }
    let offset = fastrand::u64(0..=jitter_secs.saturating_mul(2));
    // Clamp after the offset so a jitter wider than the base floors at zero.
    base.saturating_add(Duration::from_secs(offset))
        .saturating_sub(Duration::from_secs(jitter_secs))
}

pub(crate) fn next_backoff(current: Duration, cap: Duration) -> Duration {
    let doubled = current.saturating_mul(2);
    if doubled > cap { cap } else { doubled }
}

/// Sleep in [`SLEEP_SLICE`] slices, returning early once `stop` is set. `true`
/// when the whole wait elapsed.
pub(crate) fn sleep_interruptible(total: Duration, stop: &AtomicBool) -> bool {
    let mut left = total;
    while !left.is_zero() {
        if stop.load(Ordering::SeqCst) {
            return false;
        }
        let slice = if left > SLEEP_SLICE {
            SLEEP_SLICE
        } else {
            left
        };
        std::thread::sleep(slice);
        left -= slice;
    }
    !stop.load(Ordering::SeqCst)
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::ten_min("PT10M", 600)]
    #[case::five_min("PT5M", 300)]
    #[case::hour_and_a_half("PT1H30M", 5400)]
    #[case::mixed("PT1H2M3S", 3723)]
    #[case::seconds_only("PT30S", 30)]
    #[case::lowercase("pt10m", 600)]
    #[case::padded("  PT10M  ", 600)]
    fn a_window_the_server_sends_parses(#[case] raw: &str, #[case] secs: u64) {
        assert_eq!(parse_iso8601_duration(raw), Some(Duration::from_secs(secs)));
    }

    #[rstest]
    #[case::empty("")]
    #[case::pt_only("PT")]
    #[case::no_pt("10M")]
    // `PT10` read as seconds would heartbeat every 5 s against a 10-minute lease.
    #[case::trailing_number("PT10")]
    #[case::garbage("PTfoo")]
    #[case::days("P1D")]
    #[case::unit_with_no_number("PTM")]
    fn a_window_this_parser_does_not_accept_is_no_window(#[case] raw: &str) {
        assert_eq!(parse_iso8601_duration(raw), None);
    }

    #[rstest]
    #[case::half_of_ten_minutes("PT10M", None, 300)]
    #[case::half_of_ninety_minutes("PT1H30M", None, 2700)]
    // The fallback is the window, not the interval.
    #[case::trailing_number_falls_back("PT10", None, 300)]
    #[case::garbage_falls_back("bogus", None, 300)]
    #[case::lowercase("pt10m", None, 300)]
    // Floored at 1 s: a zero-length sleep is a spin.
    #[case::floor_from_a_zero_window("PT0S", None, 1)]
    #[case::floor_from_the_override("PT10M", Some(0), 1)]
    #[case::override_honoured("PT10M", Some(60), 60)]
    #[case::longer_override_honoured("PT10M", Some(900), 900)]
    fn the_heartbeat_interval(
        #[case] window: &str,
        #[case] override_secs: Option<u64>,
        #[case] secs: u64,
    ) {
        assert_eq!(
            resolve_heartbeat_interval(window, override_secs),
            Duration::from_secs(secs)
        );
    }

    #[test]
    fn half_a_window_keeps_its_sub_second_remainder() {
        assert_eq!(
            resolve_heartbeat_interval("PT1S", None),
            Duration::from_millis(500)
        );
    }

    fn samples(base: Duration, jitter: Duration, n: usize) -> Vec<Duration> {
        (0..n)
            .map(|_| idle_wait_with_jitter(base, jitter))
            .collect()
    }

    #[rstest]
    #[case::shipped_defaults(DEFAULT_IDLE_WAIT, DEFAULT_IDLE_JITTER, 20, 40)]
    #[case::wider_than_the_base(Duration::from_secs(2), Duration::from_secs(30), 0, 32)]
    fn the_idle_wait_stays_inside_its_jitter_band(
        #[case] base: Duration,
        #[case] jitter: Duration,
        #[case] lo: u64,
        #[case] hi: u64,
    ) {
        let waited = samples(base, jitter, 256);
        assert!(
            waited
                .iter()
                .all(|w| *w >= Duration::from_secs(lo) && *w <= Duration::from_secs(hi)),
            "{waited:?}"
        );
        assert!(waited.iter().all(|w| w.subsec_nanos() == 0), "{waited:?}");
    }

    #[test]
    fn the_jitter_reaches_below_the_base_as_well_as_above_it() {
        let waited = samples(DEFAULT_IDLE_WAIT, DEFAULT_IDLE_JITTER, 512);
        assert!(
            waited.iter().any(|w| *w < DEFAULT_IDLE_WAIT),
            "the band never went under idle_wait_secs"
        );
        assert!(
            waited.iter().any(|w| *w > DEFAULT_IDLE_WAIT),
            "the band never went over idle_wait_secs"
        );
    }

    #[test]
    fn no_jitter_is_the_base_exactly() {
        assert_eq!(
            idle_wait_with_jitter(DEFAULT_IDLE_WAIT, Duration::ZERO),
            DEFAULT_IDLE_WAIT
        );
    }

    #[test]
    fn the_jitter_actually_varies() {
        // A constant would satisfy the bounds test above.
        let seen: std::collections::BTreeSet<Duration> =
            samples(DEFAULT_IDLE_WAIT, DEFAULT_IDLE_JITTER, 64)
                .into_iter()
                .collect();
        assert!(seen.len() > 1, "the idle wait never moved: {seen:?}");
    }

    #[rstest]
    #[case::doubles(Duration::from_secs(5), Duration::from_secs(10))]
    #[case::caps(Duration::from_secs(40), CLAIM_BACKOFF_CAP)]
    #[case::stays_capped(CLAIM_BACKOFF_CAP, CLAIM_BACKOFF_CAP)]
    fn the_backoff_doubles_to_its_cap(#[case] current: Duration, #[case] expected: Duration) {
        assert_eq!(next_backoff(current, CLAIM_BACKOFF_CAP), expected);
    }

    #[test]
    fn the_timestamp_is_whole_second_utc_with_a_z() {
        assert_eq!(
            rfc3339(time::macros::datetime!(2026-09-13 16:19:37 UTC)),
            "2026-09-13T16:19:37Z"
        );
        // A non-zero nanosecond field must not add fractional digits.
        assert_eq!(
            rfc3339(time::macros::datetime!(2026-09-13 16:19:37.123456 UTC)),
            "2026-09-13T16:19:37Z"
        );
        // Past year 9999 once in UTC: nothing, not a panic.
        assert_eq!(rfc3339(time::macros::datetime!(9999-12-31 23:30 -1)), "");
    }

    #[test]
    fn an_already_set_stop_flag_skips_the_wait() {
        let stop = AtomicBool::new(true);
        assert!(!sleep_interruptible(Duration::from_secs(3600), &stop));
    }

    #[test]
    fn a_short_wait_runs_to_completion() {
        let stop = AtomicBool::new(false);
        assert!(sleep_interruptible(Duration::from_millis(5), &stop));
    }
}
