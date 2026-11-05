//! Raspberry Pi 5 readiness: `vcgencmd get_throttled` (the firmware's signal,
//! which also reports under-voltage that no thermal zone shows), the 80 °C soft
//! limit (the Pi 5 throttles from 80 to 85 °C and has no `temp_soft_limit`
//! knob), and the shared load check.

use std::process::Command;
use std::time::Duration;

use super::common::{self, CpuLoad};
use crate::wait::{Probe, wait_loop};
use crate::{
    CpuPercent, ReadinessCallbacks, ReadinessConfiguration, ReadinessError, ReadinessPlatform,
    ThermalGate,
};

const PI5_SOFT_LIMIT_C: i32 = 80;
const PI5_HARD_LIMIT_C: i32 = 85;
/// `vcgencmd get_throttled` "active now" bits: under-voltage (0x1), arm
/// frequency capped (0x2), throttled (0x4), soft temperature limit (0x8). The
/// "occurred since boot" bits (0x10000+) clear only on reboot, so gating on
/// them would wedge the wait forever once the Pi had ever throttled.
const THROTTLE_ACTIVE_NOW_MASK: u32 = 0xF;

/// What the gate compares against: its own numbers, or the ones the host's
/// `[readiness.pi5]` table sets.
#[derive(Debug, Clone, Copy)]
struct Bounds {
    hottest_below_c: i32,
    cpu_load_below: f64,
}

impl Bounds {
    fn new(configuration: Option<&ReadinessConfiguration>) -> Self {
        let config = configuration.and_then(ReadinessConfiguration::pi5);
        let hottest = config.and_then(|c| c.hottest_below_c);
        let load = config.and_then(|c| c.cpu_load_below_pct);
        Self {
            hottest_below_c: hottest.map_or(PI5_SOFT_LIMIT_C, |c| c.get().cast_signed()),
            cpu_load_below: load.map_or(common::CPU_LOAD_THRESHOLD, CpuPercent::ratio),
        }
    }

    /// `ThermalGate::Skip` waives the temperature bound, not the throttle bits:
    /// they also report under-voltage, a power fault rather than heat. The
    /// table cannot waive them either.
    fn ready(
        self,
        hottest: i32,
        cpu_load: f64,
        throttle: Option<u32>,
        thermal: ThermalGate,
    ) -> bool {
        let temp_ok = thermal == ThermalGate::Skip || hottest < self.hottest_below_c;
        throttle.is_none_or(|mask| mask == 0) && temp_ok && cpu_load < self.cpu_load_below
    }
}

pub(super) fn wait_until_ready(
    max_wait: Duration,
    thermal: ThermalGate,
    configuration: Option<&ReadinessConfiguration>,
    callbacks: &ReadinessCallbacks<'_>,
) -> Result<(), ReadinessError> {
    let bounds = Bounds::new(configuration);
    wait_loop(ReadinessPlatform::RaspberryPi5, max_wait, callbacks, || {
        let hottest = common::read_hottest_zone_celsius()?;
        let cpu = common::read_cpu_load()?;
        let throttle = read_throttle_active();
        Ok(Probe {
            ready: bounds.ready(hottest, cpu.ratio(), throttle, thermal),
            summary: format_summary(hottest, cpu, throttle, bounds),
        })
    })
}

fn format_summary(hottest_c: i32, cpu: CpuLoad, throttle: Option<u32>, bounds: Bounds) -> String {
    let verdict = if hottest_c < bounds.hottest_below_c {
        "ok"
    } else {
        "hot"
    };
    format!(
        "thermal=hottest:{hottest_c}°C({verdict};soft{PI5_SOFT_LIMIT_C}/hard{PI5_HARD_LIMIT_C}) {} {}",
        common::cpu_part(cpu, bounds.cpu_load_below),
        format_throttle(throttle),
    )
}

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

/// `None` when `vcgencmd` is not on PATH or cannot open `/dev/vcio`, and the
/// gate then relies on the temperature bound. The second case is logged so the
/// operator can add the user to `video`.
fn read_throttle_active() -> Option<u32> {
    let raw = run_vcgencmd_get_throttled()?;
    match parse_throttled_hex(&raw) {
        Some(bits) => Some(bits & THROTTLE_ACTIVE_NOW_MASK),
        None => {
            tracing::warn!(
                stdout = raw.as_str(),
                "unparseable `vcgencmd get_throttled` output"
            );
            None
        }
    }
}

fn run_vcgencmd_get_throttled() -> Option<String> {
    let mut cmd = Command::new("vcgencmd");
    cmd.arg("get_throttled");
    let output = crate::probe_output(cmd).ok()?;
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    // Without access, vcgencmd prints "Can't open device file: /dev/vcio",
    // often with a zero exit; parse_throttled_hex rejects it and the caller
    // warns. A non-zero exit's stderr is logged here too.
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        tracing::warn!(
            exit_status = %output.status,
            stderr = stderr.trim(),
            "`vcgencmd get_throttled` failed"
        );
    }
    Some(stdout)
}

/// `throttled=0x50005` to `0x50005`.
fn parse_throttled_hex(raw: &str) -> Option<u32> {
    let value = raw.trim().strip_prefix("throttled=")?.trim();
    let hex = value.strip_prefix("0x").unwrap_or(value);
    u32::from_str_radix(hex, 16).ok()
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;
    use crate::configuration::table;

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

    /// The verdicts follow the table's numbers; the firmware's limits stay
    /// printed whatever it sets.
    #[rstest]
    #[case::own_ok(None, 66, 0.05, Some(0), "66°C(ok")]
    #[case::own_hot(None, 82, 0.05, Some(0), "82°C(hot")]
    #[case::throttle_named(None, 70, 0.05, Some(0x8), "throttle=0x8(soft-temp-limit)")]
    #[case::table_hot(
        Some("hottest_below_c = 70\ncpu_load_below_pct = 50"),
        70,
        0.40,
        Some(0),
        "thermal=hottest:70°C(hot;soft80/hard85) cpu=load:40%(ok) throttle=0x0(ok)"
    )]
    #[case::table_busy(
        Some("hottest_below_c = 70\ncpu_load_below_pct = 50"),
        69,
        0.50,
        Some(0),
        "thermal=hottest:69°C(ok;soft80/hard85) cpu=load:50%(busy) throttle=0x0(ok)"
    )]
    fn summary_marks_soft_limit_band(
        #[case] keys: Option<&str>,
        #[case] hottest_c: i32,
        #[case] cpu_load: f64,
        #[case] throttle: Option<u32>,
        #[case] needle: &str,
    ) -> Result<(), toml::de::Error> {
        let config = keys
            .map(|keys| table(ReadinessPlatform::RaspberryPi5, keys))
            .transpose()?;
        let cpu = CpuLoad::LoadAverage(cpu_load);
        let line = format_summary(hottest_c, cpu, throttle, Bounds::new(config.as_ref()));
        assert!(line.contains(needle), "{line}");
        Ok(())
    }

    #[rstest]
    #[case::unconfined(CpuLoad::LoadAverage(0.05), "cpu=load:5%(ok)")]
    #[case::confined(CpuLoad::BusyShare(0.05), "cpu=busy:5%(ok)")]
    #[case::confined_and_busy(CpuLoad::BusyShare(0.30), "cpu=busy:30%(busy)")]
    fn summary_names_the_cpu_reading(#[case] cpu: CpuLoad, #[case] expected: &str) {
        let line = format_summary(66, cpu, Some(0), Bounds::new(None));
        assert_eq!(
            line,
            format!("thermal=hottest:66°C(ok;soft80/hard85) {expected} throttle=0x0(ok)")
        );
    }

    /// Each key replaces its own default under the same strict comparison; the
    /// firmware's throttle bits hold whatever the table says.
    #[rstest]
    #[case::own_numbers(None, ThermalGate::Enforce, 79, 0.29, Some(0), true)]
    #[case::own_temp_holds(None, ThermalGate::Enforce, 80, 0.05, Some(0), false)]
    #[case::own_load_holds(None, ThermalGate::Enforce, 60, 0.30, Some(0), false)]
    #[case::empty_table(Some(""), ThermalGate::Enforce, 80, 0.05, Some(0), false)]
    #[case::restated(
        Some("hottest_below_c = 80\ncpu_load_below_pct = 30"),
        ThermalGate::Enforce,
        80,
        0.05,
        Some(0),
        false
    )]
    #[case::temp_lowered(
        Some("hottest_below_c = 70"),
        ThermalGate::Enforce,
        72,
        0.05,
        Some(0),
        false
    )]
    #[case::temp_lowered_passes_below(
        Some("hottest_below_c = 70"),
        ThermalGate::Enforce,
        69,
        0.05,
        Some(0),
        true
    )]
    #[case::temp_key_leaves_load(
        Some("hottest_below_c = 70"),
        ThermalGate::Enforce,
        60,
        0.40,
        Some(0),
        false
    )]
    #[case::load_raised(
        Some("cpu_load_below_pct = 50"),
        ThermalGate::Enforce,
        60,
        0.40,
        Some(0),
        true
    )]
    #[case::load_raised_is_strict(
        Some("cpu_load_below_pct = 50"),
        ThermalGate::Enforce,
        60,
        0.50,
        Some(0),
        false
    )]
    #[case::load_key_leaves_temp(
        Some("cpu_load_below_pct = 50"),
        ThermalGate::Enforce,
        80,
        0.05,
        Some(0),
        false
    )]
    #[case::throttle_holds_a_raised_bound(
        Some("hottest_below_c = 90"),
        ThermalGate::Enforce,
        82,
        0.05,
        Some(0x8),
        false
    )]
    #[case::no_vcgencmd_leaves_the_bound(
        Some("hottest_below_c = 70"),
        ThermalGate::Enforce,
        72,
        0.05,
        None,
        false
    )]
    // Skipping thermal skips the temperature bound, never the throttle bits.
    #[case::skip_ignores_temp(
        Some("hottest_below_c = 70"),
        ThermalGate::Skip,
        84,
        0.05,
        Some(0),
        true
    )]
    #[case::skip_keeps_throttle(
        Some("hottest_below_c = 70"),
        ThermalGate::Skip,
        60,
        0.05,
        Some(0x1),
        false
    )]
    #[case::skip_keeps_load(
        Some("cpu_load_below_pct = 50"),
        ThermalGate::Skip,
        84,
        0.60,
        Some(0),
        false
    )]
    fn the_hosts_table_moves_the_gate(
        #[case] keys: Option<&str>,
        #[case] thermal: ThermalGate,
        #[case] hottest: i32,
        #[case] cpu_load: f64,
        #[case] throttle: Option<u32>,
        #[case] want_ready: bool,
    ) -> Result<(), toml::de::Error> {
        let config = keys
            .map(|keys| table(ReadinessPlatform::RaspberryPi5, keys))
            .transpose()?;
        let bounds = Bounds::new(config.as_ref());
        assert_eq!(
            bounds.ready(hottest, cpu_load, throttle, thermal),
            want_ready
        );
        Ok(())
    }
}
