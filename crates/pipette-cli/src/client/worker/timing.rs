//! Claim-loop timing: how long to wait between polls, and how long a lease
//! lasts. No I/O — every value here is derived from the server's `time_window`
//! or from local configuration.

use std::time::Duration;

/// Default idle wait when `claim` returns `204` (client-integration §4).
pub const DEFAULT_IDLE_WAIT: Duration = Duration::from_secs(5 * 60);
/// Jitter range added to the idle wait (0..=this).
pub const DEFAULT_IDLE_JITTER: Duration = Duration::from_secs(60);
/// How often to poll `GET /clients/me` while waiting for `reindex_pending`
/// to clear (at most one queue-maintenance cron interval).
/// Parse an ISO 8601 duration of the form the server emits for `time_window`
/// (`PTnHnMnS`, hours/minutes/seconds only — no days/weeks/months). Returns
/// `None` when the string is empty or not a recognized `PT…` duration.
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

/// Resolve the heartbeat period for a claim.
///
/// - `override_secs = None` → half of `time_window` (protocol default),
///   floored at 1 s. The protocol requires heartbeating at least this often
///   so the lease does not lapse.
/// - `override_secs = Some(n)` → `n` seconds (minimum 1). When `n` is longer
///   than half the lease window the lease may expire between ticks; the value
///   is still honored so operators can deliberately slow heartbeats, but
///   callers should log a warning (see the CLI worker).
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
    if jitter.is_zero() {
        return base;
    }
    let max = jitter.as_millis() as u64;
    let roll = random_u64() % (max + 1);
    base + Duration::from_millis(roll)
}

fn random_u64() -> u64 {
    let mut buf = [0u8; 8];
    // getrandom failure is vanishingly rare; fall back to a time-derived
    // value so the loop still progresses rather than panicking.
    if getrandom::fill(&mut buf).is_err() {
        return std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);
    }
    u64::from_le_bytes(buf)
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
