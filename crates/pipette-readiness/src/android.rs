//! Android readiness — three signals:
//!
//! 1. `dumpsys thermalservice` for the OS thermal-status enum.
//! 2. Hottest CPU-cluster die temperature read from
//!    `/sys/class/thermal/thermal_zone*/{type,temp}`.
//! 3. `/proc/stat` delta for instantaneous CPU `%busy`.
//!
//! All three must pass.
//!
//! The OS thermal-status enum follows `android.os.PowerManager`'s
//! `THERMAL_STATUS_*` constants (0 = NONE … 6 = SHUTDOWN); anything
//! above NONE means the OS is asking apps to back off. It's a
//! power-management judgment, not a die-temp readout — empirically
//! the enum stays at `0` even when CPU dies reach 75–80 °C on
//! Snapdragon 8 Elite. That's fine for app-level throttling but not
//! fine for benchmark correctness: starting a measurement on hot
//! silicon guarantees the reported throughput averages across a
//! thermal collapse. The raw zone-temp check closes that gap.
//!
//! CPU pressure used to read `/proc/loadavg`, but the 1-minute EMA
//! includes Samsung One UI's background-AI stack and `adbd` traffic
//! as historical run-queue depth, parking the resting normalized
//! load around 0.35–0.45 on tethered S25-class phones. A `/proc/stat`
//! delta over a short window yields the instantaneous `%busy`, which
//! sits in the high-single-digit percent range when the SoC is
//! genuinely idle and only crosses the threshold when something is
//! actively eating cores.

use std::fs;
use std::path::Path;
use std::time::{Duration, Instant};

use super::{ReadinessError, ThermalGate};

/// Default deadline for the wait: the shared cross-platform value
/// (see [`super::DEFAULT_MAX_WAIT`]). Phones can sit in `LIGHT` /
/// `MODERATE` for minutes under load before the SoC works back to
/// nominal, and 300 s covers that with room to spare against the
/// ~30-40 s a typical post-rep cooldown actually takes (below).
///
/// This was 600 s, chosen by feel, while the app-side gate used
/// 180 s, so the same phone was held to two deadlines three times
/// apart depending on which binary launched the run (PIP-278).
/// Neither number had a measurement behind it; this one at least has
/// one definition.
pub(super) const DEFAULT_MAX_WAIT: Duration = super::DEFAULT_MAX_WAIT;
/// Window over which `/proc/stat` is sampled to derive an
/// instantaneous CPU `%busy`. One second is long enough to smooth
/// out 100 ms jiffy noise while not noticeably extending the wait.
const CPU_SAMPLE_INTERVAL: Duration = Duration::from_secs(1);
/// Highest acceptable CPU-cluster die temperature, in °C. Snapdragon
/// 8 Elite *hard* trip is ~95–105 °C; performance-throttling kicks
/// in much earlier, around 75–80 °C.
///
/// The threshold determines how much thermal budget the next rep
/// gets before it runs into throttle. Lowering it buys more
/// headroom (more thermal capacity available to the rep) at the
/// cost of a longer cooldown wait between reps. The release-v1
/// verification at 50 °C left catastrophic stddev on 20–40 s reps
/// (±29.8–34.6 %); tightening to 35 °C dropped most of the band to
/// ±1–8 %; 34 °C aims at the residual ±5–8 % on the borderline
/// 10–15 s reps without parking the gate against the resting floor.
///
/// 34 °C gives ~41 °C of headroom below throttle onset. Field data
/// on tethered S25-class phones (USB-charging, adb attached) shows
/// the sustained-operation floor sits at ~32 °C — not the 27–29 °C
/// we initially measured from a brief idle sample. The gate uses a
/// strict `<` comparison, so 34 °C releases at ≤33 °C, leaving
/// ~1 °C above that floor.
///
/// Going lower (32–33 °C) risks the gate parking on the floor and
/// timing out [`DEFAULT_MAX_WAIT`]: a v1 stddev5 run at 32 °C stalled
/// all three phones at exactly 32 °C with no further cooldown
/// observed across multiple minutes. That observation is what the
/// deadline is really for, and it argues about the threshold rather
/// than the deadline: a gate parked on the resting floor never
/// releases, so no deadline rescues it.
///
/// Cooldown from a typical post-rep peak (~75 °C) to 34 °C is
/// empirically ~30–40 s.
///
/// Won't rescue 20+ s reps — those throttle mid-rep regardless of
/// entry temp and need a different rep architecture (e.g. an
/// HTTP-driven server transport where each "rep" is short enough
/// to fit any reasonable budget).
const THERMAL_THRESHOLD_C: i32 = 34;
/// Highest acceptable instantaneous CPU `%busy` across all cores
/// during the sample window. 0.30 leaves comfortable headroom over
/// the resting floor (typically well under 0.10 on Snapdragon-class
/// SoCs) while still catching another benchmark that's already
/// running.
const CPU_LOAD_THRESHOLD: f64 = 0.30;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CpuTimes {
    /// `idle + iowait` jiffies since boot (kernels 2.5.41+, all
    /// supported Android targets).
    idle: u64,
    /// Sum of all aggregate `cpu` fields since boot.
    total: u64,
}

pub(super) fn wait_until_ready(
    max_wait: Duration,
    thermal: ThermalGate,
) -> Result<(), ReadinessError> {
    let deadline = Instant::now() + max_wait;
    loop {
        let dumpsys = read_dumpsys_thermal()?;
        let status = parse_dumpsys_status(&dumpsys)?;
        // `/sys/class/thermal` is the primary die-temp source, but it is
        // unreadable to apps on some devices — SELinux denies `/sys/class/thermal`
        // on Pixel/Tensor, so every probe would otherwise error and the gate would
        // never release (it just burns the full deadline before proceeding). Fall
        // back to the CPU temperatures dumpsys already reported (the same HAL
        // values, `mType=0`) so the die-temp signal still works there.
        let hottest_c = match read_hottest_cpu_zone_celsius() {
            Ok(c) => c,
            Err(sysfs_err) => parse_hottest_cpu_celsius_from_dumpsys(&dumpsys).ok_or(sysfs_err)?,
        };
        let cpu_load = read_cpu_load()?;
        // Both the OS status enum and the die zone are thermal signals, so
        // `ThermalGate::Skip` waives both; CPU load is unaffected.
        let status_ok = thermal == ThermalGate::Skip || status == 0;
        let temp_ok = thermal == ThermalGate::Skip || hottest_c < THERMAL_THRESHOLD_C;
        let cpu_ok = cpu_load < CPU_LOAD_THRESHOLD;
        let summary = format_summary(status, hottest_c, cpu_load);
        if status_ok && temp_ok && cpu_ok {
            log::info!("readiness: android {summary} → proceeding");
            return Ok(());
        }
        if Instant::now() >= deadline {
            log::info!("readiness: android {summary} → timed out after {max_wait:?}");
            return Err(ReadinessError::TimedOut {
                max_wait,
                observed: summary,
            });
        }
        log::info!(
            "readiness: android {summary} → waiting {:?}",
            super::SENSOR_POLL_INTERVAL
        );
        std::thread::sleep(super::SENSOR_POLL_INTERVAL);
    }
}

fn format_summary(status: u8, hottest_c: i32, cpu_load: f64) -> String {
    let temp_verdict = if hottest_c < THERMAL_THRESHOLD_C {
        "ok"
    } else {
        "hot"
    };
    let cpu_pct = (cpu_load * 100.0).round() as u32;
    let cpu_verdict = if cpu_load < CPU_LOAD_THRESHOLD {
        "ok"
    } else {
        "busy"
    };
    format!(
        "thermal=status:{status}({}),hottest:{hottest_c}°C({temp_verdict}) cpu=busy:{cpu_pct}%({cpu_verdict})",
        format_status_word(status),
    )
}

fn format_status_word(status: u8) -> &'static str {
    match status {
        0 => "nominal",
        1 => "light",
        2 => "moderate",
        3 => "severe",
        4 => "critical",
        5 => "emergency",
        6 => "shutdown",
        _ => "unknown",
    }
}

fn read_dumpsys_thermal() -> Result<String, ReadinessError> {
    let output = std::process::Command::new("/system/bin/dumpsys")
        .arg("thermalservice")
        .output()?;
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

fn read_hottest_cpu_zone_celsius() -> Result<i32, ReadinessError> {
    let dir = fs::read_dir("/sys/class/thermal")?;
    dir.filter_map(Result::ok)
        .filter(|entry| {
            entry
                .file_name()
                .to_str()
                .is_some_and(|name| name.starts_with("thermal_zone"))
        })
        .filter_map(|entry| {
            let path = entry.path();
            let zone_type = fs::read_to_string(path.join("type")).ok()?;
            if !is_cpu_zone(zone_type.trim()) {
                return None;
            }
            read_zone_celsius(&path.join("temp"))
        })
        .max()
        .ok_or(ReadinessError::Unavailable(
            "no readable CPU thermal zones in /sys/class/thermal",
        ))
}

/// Snapdragon SoCs expose `cpu-<cluster>-<core>` (per-core die temps)
/// and `cpullc-<cluster>-<core>` (cluster low-level cache) zones,
/// both of which track CPU-cluster temperature. Everything else
/// (`gpuss-*`, `nsphvx-*` NPU, `qmx-*` modem, `battery`, …) reflects
/// unrelated blocks and is excluded.
///
/// Notably **`cpu-hw-trip-*` is excluded** — those are pseudo-zones
/// that report the HW throttle threshold (105 °C on Snapdragon 8
/// Elite) as a *constant* "current" value, not a real-time die
/// temperature. A naive `starts_with("cpu")` filter picks them up
/// and pins the gate at "always hot". Match against the digit that
/// follows `cpu-` / `cpullc-` to keep only real per-cluster zones.
///
/// **Other static-constant pseudo-zones observed on the S25-class
/// phones** (do not naively add these if widening the gate to PMIC
/// junctions): `pmh0104_tz` and `pmr735d_tz` both report a fixed
/// 37 000 mC across consecutive reads on all sampled units — they
/// look like threshold values bleeding through the kernel interface,
/// same failure mode as `cpu-hw-trip-*`. The remaining PMIC zones
/// (`pmh0101_tz`, `pmh0110_{d,f,g,i}_tz`) and the skin sensor
/// `sys-therm-0` do vary in real time and are safe candidates if a
/// future change extends thermal gating beyond CPU dies.
fn is_cpu_zone(zone_type: &str) -> bool {
    let body = zone_type
        .strip_prefix("cpu-")
        .or_else(|| zone_type.strip_prefix("cpullc-"));
    body.is_some_and(|b| b.starts_with(|c: char| c.is_ascii_digit()))
}

fn read_zone_celsius(temp_path: &Path) -> Option<i32> {
    let raw = fs::read_to_string(temp_path).ok()?;
    let millicelsius: i32 = raw.trim().parse().ok()?;
    Some(millicelsius / 1000)
}

fn read_cpu_load() -> Result<f64, ReadinessError> {
    let first = read_proc_stat_cpu()?;
    std::thread::sleep(CPU_SAMPLE_INTERVAL);
    let second = read_proc_stat_cpu()?;
    Ok(busy_ratio(first, second))
}

fn read_proc_stat_cpu() -> Result<CpuTimes, ReadinessError> {
    let raw = fs::read_to_string("/proc/stat")?;
    parse_proc_stat_cpu(&raw)
}

fn parse_proc_stat_cpu(raw: &str) -> Result<CpuTimes, ReadinessError> {
    let line = raw
        .lines()
        .find(|l| l.starts_with("cpu "))
        .ok_or(ReadinessError::Unavailable(
            "no aggregate `cpu` line in /proc/stat",
        ))?;
    let parsed: Result<Vec<u64>, _> = line.split_whitespace().skip(1).map(str::parse).collect();
    let fields = parsed.map_err(|_| ReadinessError::ParseFailure {
        raw: line.to_string(),
    })?;
    // user, nice, system, idle, iowait, [irq, softirq, steal, guest, guest_nice].
    // We need through `iowait` (index 4) to compute idle time.
    if fields.len() < 5 {
        return Err(ReadinessError::ParseFailure {
            raw: line.to_string(),
        });
    }
    Ok(CpuTimes {
        idle: fields[3] + fields[4],
        total: fields.iter().sum(),
    })
}

fn busy_ratio(first: CpuTimes, second: CpuTimes) -> f64 {
    let total_delta = second.total.saturating_sub(first.total);
    if total_delta == 0 {
        return 0.0;
    }
    let idle_delta = second.idle.saturating_sub(first.idle);
    let busy_delta = total_delta.saturating_sub(idle_delta);
    busy_delta as f64 / total_delta as f64
}

fn parse_dumpsys_status(stdout: &str) -> Result<u8, ReadinessError> {
    let rest = stdout
        .lines()
        .find_map(|line| line.trim().strip_prefix("Thermal Status:").map(str::trim))
        .ok_or(ReadinessError::Unavailable(
            "no `Thermal Status:` line in dumpsys output",
        ))?;
    rest.parse::<u8>()
        .map_err(|_| ReadinessError::ParseFailure {
            raw: rest.to_string(),
        })
}

/// Hottest CPU-cluster die temperature (°C) parsed from `dumpsys
/// thermalservice` output, or `None` if no live CPU temperature is present.
/// Used as a fallback when `/sys/class/thermal` is inaccessible: dumpsys
/// surfaces the same HAL readings as `Temperature{mValue=..,mType=0,..}`
/// lines (`mType` 0 == CPU). Sensors with no reading report `-FLT_MAX`
/// (~-3.4e38); the sane-band filter drops those and other noise.
///
/// dumpsys prints both a stale `Cached temperatures:` block and the live
/// `Current temperatures from HAL:` block — we must read only the latter, or a
/// stale peak (e.g. 82 °C cached while the SoC is actually at 33 °C) would pin
/// the gate hot forever.
fn parse_hottest_cpu_celsius_from_dumpsys(stdout: &str) -> Option<i32> {
    let mut in_current = false;
    stdout
        .lines()
        .filter_map(|line| {
            let line = line.trim();
            if line == "Current temperatures from HAL:" {
                in_current = true;
                return None;
            }
            if !in_current {
                return None;
            }
            if !line.starts_with("Temperature{") {
                in_current = false; // left the live section
                return None;
            }
            if dumpsys_field(line, "mType=")? != "0" {
                return None;
            }
            let value: f64 = dumpsys_field(line, "mValue=")?.parse().ok()?;
            if !value.is_finite() || !(-50.0..=200.0).contains(&value) {
                return None;
            }
            Some(value as i32)
        })
        .max()
}

/// Extract the value of `key` (e.g. `"mType="`) from a `dumpsys`
/// `Temperature{..}` line, up to the next `,` or `}`.
fn dumpsys_field<'a>(line: &'a str, key: &str) -> Option<&'a str> {
    let start = line.find(key)? + key.len();
    let rest = &line[start..];
    let end = rest.find([',', '}']).unwrap_or(rest.len());
    Some(rest[..end].trim())
}

#[cfg(test)]
mod tests {
    use anyhow::Context;
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case("Thermal Status: 0\nIsStatusOverride: false\n", 0)]
    #[case("IsStatusOverride: false\nThermal Status: 2\nHAL Ready: true\n", 2)]
    #[case("   Thermal Status:   4   ", 4)]
    fn dumpsys_parses_status_line(
        #[case] stdout: &str,
        #[case] want: u8,
    ) -> Result<(), Box<dyn std::error::Error>> {
        assert_eq!(parse_dumpsys_status(stdout)?, want);
        Ok(())
    }

    #[test]
    fn dumpsys_missing_line_is_unavailable() -> anyhow::Result<()> {
        let err = parse_dumpsys_status("HAL Ready: true\n")
            .err()
            .context("expected missing Thermal Status line to be Unavailable")?;
        assert!(
            matches!(
                err,
                ReadinessError::Unavailable("no `Thermal Status:` line in dumpsys output")
            ),
            "got {err:?}",
        );
        Ok(())
    }

    #[test]
    fn dumpsys_unparseable_is_parse_failure() -> anyhow::Result<()> {
        let err = parse_dumpsys_status("Thermal Status: hot\n")
            .err()
            .context("expected unparseable Thermal Status to be a ParseFailure")?;
        assert!(
            matches!(err, ReadinessError::ParseFailure { ref raw } if raw == "hot"),
            "got {err:?}",
        );
        Ok(())
    }

    // Real `dumpsys thermalservice` shape (Pixel/Tensor): a stale `Cached
    // temperatures:` block precedes the live `Current temperatures from HAL:`
    // block. CPU clusters are `mType=0`; GPU/TPU report the -FLT_MAX no-reading
    // sentinel; battery is `mType=2`. Only the live block counts, and within it
    // the hottest CPU cluster wins (non-CPU + sentinel lines drop).
    const DUMPSYS_TEMPS: &str = "\
Thermal Status: 0
Cached temperatures:
\tTemperature{mValue=82.0, mType=0, mName=BIG, mStatus=0}
\tTemperature{mValue=83.0, mType=0, mName=MID, mStatus=0}
HAL Ready: true
Current temperatures from HAL:
\tTemperature{mValue=31.000002, mType=0, mName=BIG, mStatus=0}
\tTemperature{mValue=35.0, mType=0, mName=MID, mStatus=0}
\tTemperature{mValue=30.0, mType=0, mName=LITTLE, mStatus=0}
\tTemperature{mValue=26.0, mType=2, mName=battery, mStatus=0}
\tTemperature{mValue=-3.4028235E38, mType=1, mName=GPU, mStatus=0}
Current cooling devices from HAL:
\tCoolingDevice{mValue=0, mType=1, mName=x}
";

    #[test]
    fn dumpsys_hottest_cpu_uses_live_block_max() {
        // Hottest live CPU is MID=35; the stale cached 82/83 must be ignored,
        // as must battery (mType=2) and the GPU -FLT_MAX sentinel.
        assert_eq!(
            parse_hottest_cpu_celsius_from_dumpsys(DUMPSYS_TEMPS),
            Some(35)
        );
    }

    #[test]
    fn dumpsys_hottest_cpu_ignores_non_cpu_and_sentinel() {
        let out = "\
Current temperatures from HAL:
\tTemperature{mValue=99.0, mType=2, mName=battery, mStatus=0}
\tTemperature{mValue=-3.4028235E38, mType=0, mName=BIG, mStatus=0}
\tTemperature{mValue=42.0, mType=0, mName=LITTLE, mStatus=0}
";
        assert_eq!(parse_hottest_cpu_celsius_from_dumpsys(out), Some(42));
    }

    #[test]
    fn dumpsys_no_live_cpu_temperature_is_none() {
        // Only a cached block, no live section → None (don't gate on stale data).
        let out = "\
Cached temperatures:
\tTemperature{mValue=82.0, mType=0, mName=BIG, mStatus=0}
";
        assert_eq!(parse_hottest_cpu_celsius_from_dumpsys(out), None);
    }

    #[rstest]
    // Full 10-field aggregate line (modern kernels).
    #[case("cpu  100 0 50 800 50 0 0 0 0 0\nintr ...\n", 850, 1000)]
    // Minimum 5-field shape (Android kernels include at least through `iowait`).
    #[case("cpu 1 2 3 4 5\n", 7, 15)]
    fn proc_stat_parses_aggregate_cpu_line(
        #[case] raw: &str,
        #[case] want_idle: u64,
        #[case] want_total: u64,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let parsed = parse_proc_stat_cpu(raw)?;
        assert_eq!(parsed.idle, want_idle);
        assert_eq!(parsed.total, want_total);
        Ok(())
    }

    #[test]
    fn proc_stat_missing_cpu_line_is_unavailable() -> anyhow::Result<()> {
        let err = parse_proc_stat_cpu("intr 12345\nctxt 6789\n")
            .err()
            .context("expected missing cpu line to be Unavailable")?;
        assert!(
            matches!(
                err,
                ReadinessError::Unavailable("no aggregate `cpu` line in /proc/stat")
            ),
            "got {err:?}",
        );
        Ok(())
    }

    #[test]
    fn proc_stat_too_few_fields_is_parse_failure() -> anyhow::Result<()> {
        let err = parse_proc_stat_cpu("cpu 1 2 3\n")
            .err()
            .context("expected too-few-fields cpu line to be a ParseFailure")?;
        assert!(
            matches!(err, ReadinessError::ParseFailure { ref raw } if raw == "cpu 1 2 3"),
            "got {err:?}",
        );
        Ok(())
    }

    #[test]
    fn proc_stat_garbage_field_is_parse_failure() -> anyhow::Result<()> {
        let err = parse_proc_stat_cpu("cpu 1 nope 3 4 5\n")
            .err()
            .context("expected garbage cpu field to be a ParseFailure")?;
        assert!(
            matches!(err, ReadinessError::ParseFailure { ref raw } if raw == "cpu 1 nope 3 4 5"),
            "got {err:?}",
        );
        Ok(())
    }

    #[rstest]
    // 30% busy: total +1000, idle +700.
    #[case(CpuTimes { idle: 5000, total: 10_000 }, CpuTimes { idle: 5700, total: 11_000 }, 0.30)]
    // 0% busy: idle absorbs all the delta.
    #[case(CpuTimes { idle: 5000, total: 10_000 }, CpuTimes { idle: 6000, total: 11_000 }, 0.0)]
    // 100% busy: idle doesn't move.
    #[case(CpuTimes { idle: 5000, total: 10_000 }, CpuTimes { idle: 5000, total: 11_000 }, 1.0)]
    fn busy_ratio_computes_expected(
        #[case] first: CpuTimes,
        #[case] second: CpuTimes,
        #[case] want: f64,
    ) {
        let got = busy_ratio(first, second);
        assert!((got - want).abs() < 1e-9, "got {got}, want {want}");
    }

    #[test]
    fn busy_ratio_returns_zero_on_no_delta() {
        let snapshot = CpuTimes {
            idle: 100,
            total: 200,
        };
        assert_eq!(busy_ratio(snapshot, snapshot), 0.0);
    }

    #[rstest]
    #[case(0, "nominal")]
    #[case(1, "light")]
    #[case(2, "moderate")]
    #[case(3, "severe")]
    #[case(4, "critical")]
    #[case(5, "emergency")]
    #[case(6, "shutdown")]
    #[case(7, "unknown")]
    fn status_word_covers_all_thermal_levels(#[case] status: u8, #[case] want: &str) {
        assert_eq!(format_status_word(status), want);
    }

    #[rstest]
    // CPU-cluster zones we want to gate on.
    #[case("cpu-0-0-0", true)]
    #[case("cpu-1-1-1", true)]
    #[case("cpullc-0-0", true)]
    #[case("cpullc-1-1", true)]
    // HW throttle-threshold pseudo-zones report 105 000 mC as a
    // constant; they share the `cpu-` prefix but must NOT be gated on.
    #[case("cpu-hw-trip-0", false)]
    #[case("cpu-hw-trip-1", false)]
    // Defensive: any cpu-prefixed zone that doesn't fit the
    // `cpu-<digit>` / `cpullc-<digit>` shape is excluded.
    #[case("cpu-anything-else", false)]
    #[case("cpullc-anything-else", false)]
    // Non-CPU blocks — these have their own thermal envelopes and
    // can be hot without throttling the CPU; excluded.
    #[case("gpuss-0", false)]
    #[case("nsphvx-2", false)]
    #[case("qmx-0-0", false)]
    #[case("battery", false)]
    #[case("ddr", false)]
    #[case("camera-0", false)]
    fn cpu_zone_predicate_filters_to_cpu_clusters(#[case] zone_type: &str, #[case] want: bool) {
        assert_eq!(is_cpu_zone(zone_type), want);
    }
}
