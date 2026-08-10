//! Raspberry Pi 5 readiness: gate on `vcgencmd get_throttled` (the firmware
//! signal — covers thermal soft/hard limit AND under-voltage, which a
//! thermal-zone threshold can't), plus the 80 °C soft limit (the Pi 5
//! throttles 80→85 °C; there's no `temp_soft_limit` knob on the 5-series)
//! and the shared load check.

use std::process::Command;
use std::time::Duration;

use super::common;
use crate::{ReadinessError, ThermalGate};

const PI5_SOFT_LIMIT_C: i32 = 80;
const PI5_HARD_LIMIT_C: i32 = 85;
/// `vcgencmd get_throttled` "active now" bits: under-voltage (0x1), arm
/// frequency capped (0x2), currently throttled (0x4), soft temperature
/// limit active (0x8). The high "occurred since boot" bits (0x10000+) are
/// sticky and unclearable without a reboot, so the gate ignores them —
/// they'd otherwise wedge the wait forever once the Pi had ever throttled.
const THROTTLE_ACTIVE_NOW_MASK: u32 = 0xF;

pub(super) fn wait_until_ready(
    max_wait: Duration,
    thermal: ThermalGate,
) -> Result<(), ReadinessError> {
    common::wait_loop(max_wait, || {
        let hottest = common::read_hottest_zone_celsius()?;
        let cpu_load = common::read_cpu_load()?;
        // `None` (vcgencmd/`/dev/vcio` unavailable) degrades to the
        // soft-limit temp gate alone — see `read_throttle_active`.
        let throttle = read_throttle_active();
        // `ThermalGate::Skip` waives the temperature limit but NOT the
        // firmware throttle bits: those also report under-voltage, which is
        // how an undersized power supply shows up, and that is a power fault
        // rather than heat. Skipping it would hide a rig problem.
        let temp_ok = thermal == ThermalGate::Skip || hottest < PI5_SOFT_LIMIT_C;
        let ready = throttle.is_none_or(|mask| mask == 0)
            && temp_ok
            && cpu_load < common::CPU_LOAD_THRESHOLD;
        Ok(common::Probe {
            ready,
            summary: format_summary(hottest, cpu_load, throttle),
        })
    })
}

fn format_summary(hottest_c: i32, cpu_load: f64, throttle: Option<u32>) -> String {
    let verdict = if hottest_c < PI5_SOFT_LIMIT_C {
        "ok"
    } else {
        "hot"
    };
    format!(
        "pi5 thermal=hottest:{hottest_c}°C({verdict};soft{PI5_SOFT_LIMIT_C}/hard{PI5_HARD_LIMIT_C}) {} {}",
        common::cpu_part(cpu_load),
        format_throttle(throttle),
    )
}

/// Render the throttle mask for the readiness summary. `None` = the probe
/// wasn't usable on this host (see [`read_throttle_active`]).
fn format_throttle(throttle: Option<u32>) -> String {
    match throttle {
        None => "throttle=n/a".to_string(),
        Some(0) => "throttle=0x0(ok)".to_string(),
        Some(mask) => format!("throttle={mask:#x}({})", describe_throttle(mask)),
    }
}

fn describe_throttle(mask: u32) -> String {
    let mut flags = Vec::new();
    if mask & 0x1 != 0 {
        flags.push("under-voltage");
    }
    if mask & 0x2 != 0 {
        flags.push("freq-capped");
    }
    if mask & 0x4 != 0 {
        flags.push("throttled");
    }
    if mask & 0x8 != 0 {
        flags.push("soft-temp-limit");
    }
    if flags.is_empty() {
        "ok".to_string()
    } else {
        flags.join(",")
    }
}

/// Pi throttle/under-voltage state via `vcgencmd get_throttled`, reduced
/// to the [`THROTTLE_ACTIVE_NOW_MASK`] "active now" bits.
///
/// Returns `None` when the probe isn't usable — `vcgencmd` not on PATH, or
/// present but `/dev/vcio` unreadable (the invoking user lacks access). A
/// `None` degrades to the soft-limit temp gate; a permission failure is
/// logged so the operator can grant access (add the user to `video`).
fn read_throttle_active() -> Option<u32> {
    let raw = run_vcgencmd_get_throttled()?;
    match parse_throttled_hex(&raw) {
        Some(bits) => Some(bits & THROTTLE_ACTIVE_NOW_MASK),
        None => {
            log::warn!("readiness: pi5 unparseable `vcgencmd get_throttled` output: {raw:?}");
            None
        }
    }
}

fn run_vcgencmd_get_throttled() -> Option<String> {
    // Spawn failure (NotFound) → tool absent → no signal.
    let output = Command::new("vcgencmd")
        .arg("get_throttled")
        .output()
        .ok()?;
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    // vcgencmd prints "Can't open device file: /dev/vcio" (often with a
    // zero exit) when /dev/vcio isn't accessible; that stdout won't carry
    // a `throttled=` field, so parse_throttled_hex returns None and the
    // caller logs + degrades. Surface a non-zero exit's stderr too.
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        log::warn!(
            "readiness: pi5 `vcgencmd get_throttled` exited {}: {}",
            output.status,
            stderr.trim()
        );
    }
    Some(stdout)
}

/// Parse a `vcgencmd get_throttled` line — `throttled=0x50005` → `0x50005`.
fn parse_throttled_hex(raw: &str) -> Option<u32> {
    let value = raw.trim().strip_prefix("throttled=")?.trim();
    let hex = value.strip_prefix("0x").unwrap_or(value);
    u32::from_str_radix(hex, 16).ok()
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case("throttled=0x0", 0x0)]
    #[case("throttled=0x50005\n", 0x50005)]
    #[case("  throttled=0x4  ", 0x4)]
    #[case("throttled=50000", 0x50000)] // bare hex, no 0x prefix
    fn parses_throttled_hex(#[case] raw: &str, #[case] want: u32) {
        assert_eq!(parse_throttled_hex(raw), Some(want));
    }

    #[rstest]
    #[case("")]
    #[case("Can't open device file: /dev/vcio")]
    #[case("throttled=notanumber")]
    fn unparseable_throttled_is_none(#[case] raw: &str) {
        assert_eq!(parse_throttled_hex(raw), None);
    }

    #[test]
    fn active_now_mask_ignores_sticky_occurred_bits() {
        // 0x5 = bits 0 and 2 (under-voltage + throttled); 0x50000 = sticky
        // "occurred" bits. The gate must keep only the low nibble.
        assert_eq!(0x50005 & THROTTLE_ACTIVE_NOW_MASK, 0x5);
        assert_eq!(0x50000 & THROTTLE_ACTIVE_NOW_MASK, 0x0);
    }

    #[rstest]
    #[case(0x0, "ok")]
    #[case(0x1, "under-voltage")]
    #[case(0x4, "throttled")]
    #[case(0x8, "soft-temp-limit")]
    #[case(0x5, "under-voltage,throttled")]
    #[case(0xF, "under-voltage,freq-capped,throttled,soft-temp-limit")]
    fn describes_throttle_flags(#[case] mask: u32, #[case] want: &str) {
        assert_eq!(describe_throttle(mask), want);
    }

    #[test]
    fn format_throttle_renders_states() {
        assert_eq!(format_throttle(None), "throttle=n/a");
        assert_eq!(format_throttle(Some(0)), "throttle=0x0(ok)");
        assert_eq!(format_throttle(Some(0x4)), "throttle=0x4(throttled)");
    }

    #[test]
    fn summary_marks_soft_limit_band() {
        assert!(format_summary(66, 0.05, Some(0)).contains("66°C(ok"));
        assert!(format_summary(82, 0.05, Some(0)).contains("82°C(hot"));
        assert!(format_summary(70, 0.05, Some(0x8)).contains("throttle=0x8(soft-temp-limit)"));
    }
}
