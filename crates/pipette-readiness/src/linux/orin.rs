//! Jetson Orin readiness: the hottest thermal zone and the shared load check,
//! plus three signals a generic Linux gate cannot see. Inference runs on the
//! GPU, so its load is gated beside the CPU's; a cooling device that is
//! clamping a clock right now holds the gate; and so does an over-current event
//! counted since the previous poll, of this wait or an earlier one. Every one
//! is a sysfs read the non-root fleet user can make, so the gate spawns nothing.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, PoisonError};
use std::time::Duration;

use super::common::{self, CpuLoad};
use crate::wait::{Probe, wait_loop};
use crate::{
    CpuPercent, ReadinessCallbacks, ReadinessConfiguration, ReadinessError, ReadinessPlatform,
    ThermalGate,
};

/// The board's `hot-surface-alert` trip, bound to nothing that throttles:
/// the same 70 °C the generic gate holds to, and well below the 99 °C passive
/// trip where the CPU, GPU and SoC clocks are cut (104.5 °C critical).
const ORIN_HOT_SURFACE_C: i32 = 70;
const ORIN_THROTTLE_C: i32 = 99;

/// GPU busy share that still counts as idle: the 5 % the Windows gate holds
/// its GPU-compute counter to.
const GPU_LOAD_THRESHOLD: f64 = 0.05;

/// Where JetPack 6 exposes the GPU's busy share. A later kernel may move or
/// drop it, so the first readable one wins and none at all leaves GPU load
/// ungated.
const GPU_LOAD_PATHS: [&str; 3] = [
    "/sys/devices/platform/bus@0/17000000.gpu/load",
    "/sys/devices/platform/17000000.gpu/load",
    "/sys/class/devfreq/17000000.gpu/device/load",
];

/// Whether the missing-GPU-load warning has been written yet: the fact is
/// fixed for the life of the process, and the gate polls every few seconds.
static GPU_LOAD_WARNED: AtomicBool = AtomicBool::new(false);

/// The over-current counters at the last poll of any wait in this process. The
/// counters are cumulative since boot, so only a change between two polls is an
/// event; keeping the snapshot across waits lets a wait's first poll see the
/// events of the rep before it. Only the process's first poll counts as clear.
static PREVIOUS_OC: Mutex<Option<BTreeMap<String, u64>>> = Mutex::new(None);

/// What the gate compares against: its own numbers, or the ones the host's
/// `[readiness.orin]` table sets.
#[derive(Debug, Clone, Copy)]
struct Bounds {
    hottest_below_c: i32,
    cpu_load_below: f64,
    gpu_load_below: f64,
}

/// One poll's readings. `None` is a probe this host does not expose.
#[derive(Debug, Clone, PartialEq)]
struct Readings {
    hottest_c: i32,
    cpu: CpuLoad,
    gpu_load: Option<f64>,
    /// `type:cur_state` of each throttling cooling device that is active.
    throttling: Option<Vec<String>>,
    /// Over-current events since the previous poll.
    oc_events: Option<u64>,
}

impl Bounds {
    fn new(configuration: Option<&ReadinessConfiguration>) -> Self {
        let config = configuration.and_then(ReadinessConfiguration::orin);
        let hottest = config.and_then(|c| c.hottest_below_c);
        let cpu = config.and_then(|c| c.cpu_load_below_pct);
        let gpu = config.and_then(|c| c.gpu_load_below_pct);
        Self {
            hottest_below_c: hottest.map_or(ORIN_HOT_SURFACE_C, |c| c.get().cast_signed()),
            cpu_load_below: cpu.map_or(common::CPU_LOAD_THRESHOLD, CpuPercent::ratio),
            gpu_load_below: gpu.map_or(GPU_LOAD_THRESHOLD, CpuPercent::ratio),
        }
    }

    /// `ThermalGate::Skip` waives the temperature bound, not the throttling
    /// cooling devices or the over-current events: either means the clocks are
    /// being cut now, which no table can waive.
    fn ready(self, readings: &Readings, thermal: ThermalGate) -> bool {
        let temp_ok = thermal == ThermalGate::Skip || readings.hottest_c < self.hottest_below_c;
        readings.throttling.as_ref().is_none_or(Vec::is_empty)
            && readings.oc_events.is_none_or(|n| n == 0)
            && temp_ok
            && readings.cpu.ratio() < self.cpu_load_below
            && readings.gpu_load.is_none_or(|g| g < self.gpu_load_below)
    }
}

pub(super) fn wait_until_ready(
    max_wait: Duration,
    thermal: ThermalGate,
    configuration: Option<&ReadinessConfiguration>,
    callbacks: &ReadinessCallbacks<'_>,
) -> Result<(), ReadinessError> {
    let bounds = Bounds::new(configuration);
    wait_loop(ReadinessPlatform::JetsonOrin, max_wait, callbacks, || {
        let readings = Readings {
            hottest_c: common::read_hottest_zone_celsius()?,
            cpu: common::read_cpu_load()?,
            gpu_load: read_gpu_load(),
            throttling: active_throttling(read_cooling_devices()),
            oc_events: oc_events_since_last_poll(),
        };
        Ok(Probe {
            ready: bounds.ready(&readings, thermal),
            summary: format_summary(&readings, bounds),
        })
    })
}

fn format_summary(readings: &Readings, bounds: Bounds) -> String {
    let hottest_c = readings.hottest_c;
    let verdict = if hottest_c < bounds.hottest_below_c {
        "ok"
    } else {
        "hot"
    };
    let gpu = match readings.gpu_load {
        None => "gpu=n/a".to_string(),
        Some(load) => {
            let verdict = if load < bounds.gpu_load_below {
                "ok"
            } else {
                "busy"
            };
            format!("gpu=load:{}%({verdict})", (load * 100.0).round())
        }
    };
    let throttle = match &readings.throttling {
        None => "throttle=n/a".to_string(),
        Some(active) if active.is_empty() => "throttle=none(ok)".to_string(),
        Some(active) => format!("throttle={}(throttled)", active.join(",")),
    };
    let oc = match readings.oc_events {
        None => "oc=n/a".to_string(),
        Some(0) => "oc=+0(ok)".to_string(),
        Some(n) => format!("oc=+{n}(over-current)"),
    };
    format!(
        "thermal=hottest:{hottest_c}°C({verdict};throttle{ORIN_THROTTLE_C}) {} {gpu} {throttle} {oc}",
        common::cpu_part(readings.cpu, bounds.cpu_load_below),
    )
}

/// The GPU's busy share from the first readable of [`GPU_LOAD_PATHS`], or
/// `None` (warned once) when none reads as one.
fn read_gpu_load() -> Option<f64> {
    let load = GPU_LOAD_PATHS
        .iter()
        .find_map(|path| fs::read_to_string(path).ok())
        .and_then(|raw| parse_gpu_load(&raw));
    if load.is_none() && !GPU_LOAD_WARNED.swap(true, Ordering::Relaxed) {
        tracing::warn!(
            paths = ?GPU_LOAD_PATHS,
            "no readable Jetson GPU load; every wait in this process leaves GPU load ungated"
        );
    }
    load
}

/// The GPU `load` file is permille, 0-1000.
fn parse_gpu_load(raw: &str) -> Option<f64> {
    let permille: u32 = raw.trim().parse().ok()?;
    (permille <= 1000).then(|| f64::from(permille) / 1000.0)
}

/// `(type, cur_state)` of every readable `/sys/class/thermal/cooling_device*`.
fn read_cooling_devices() -> Vec<(String, u32)> {
    let Ok(dir) = fs::read_dir("/sys/class/thermal") else {
        return Vec::new();
    };
    dir.filter_map(Result::ok)
        .filter(|entry| {
            entry
                .file_name()
                .to_str()
                .is_some_and(|name| name.starts_with("cooling_device"))
        })
        .filter_map(|entry| {
            let path = entry.path();
            let kind = fs::read_to_string(path.join("type")).ok()?;
            let state = fs::read_to_string(path.join("cur_state")).ok()?;
            Some((kind.trim().to_string(), state.trim().parse().ok()?))
        })
        .collect()
}

/// The devices that cut a clock when their trip fires. `pwm-fan` only cools,
/// and `hot-surface-alert` (the 70 °C trip) is bound to nothing that throttles.
fn is_throttling_cooling_device(kind: &str) -> bool {
    kind.starts_with("cpufreq-cpu")
        || (kind.starts_with("devfreq-") && kind.contains("gpu"))
        || kind.ends_with("-throttle-alert")
}

/// `None` when no throttling device is there to watch.
fn active_throttling(mut devices: Vec<(String, u32)>) -> Option<Vec<String>> {
    devices.retain(|(kind, _)| is_throttling_cooling_device(kind));
    if devices.is_empty() {
        return None;
    }
    let mut active: Vec<String> = devices
        .into_iter()
        .filter(|(_, state)| *state > 0)
        .map(|(kind, state)| format!("{kind}:{state}"))
        .collect();
    // `read_dir` order is the directory's, not a stable one.
    active.sort();
    Some(active)
}

/// `ocN_event_cnt` of each over-current channel with `ocN_throt_en = 1`, from
/// the `soctherm_oc` hwmon; `None` when no such hwmon or channel exists.
fn read_oc_counts() -> Option<BTreeMap<String, u64>> {
    let hwmon = fs::read_dir("/sys/class/hwmon")
        .ok()?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .find(|path| {
            fs::read_to_string(path.join("name")).is_ok_and(|name| name.trim() == "soctherm_oc")
        })?;
    let read = |path: &Path| fs::read_to_string(path).ok()?.trim().parse::<u64>().ok();
    let counts = fs::read_dir(&hwmon)
        .ok()?
        .filter_map(Result::ok)
        .filter_map(|entry| {
            let name = entry.file_name().into_string().ok()?;
            let channel = name.strip_suffix("_throt_en")?.to_string();
            if read(&entry.path())? != 1 {
                return None;
            }
            let count = read(&hwmon.join(format!("{channel}_event_cnt")))?;
            Some((channel, count))
        })
        .collect::<BTreeMap<_, _>>();
    (!counts.is_empty()).then_some(counts)
}

/// Events since [`PREVIOUS_OC`], which this poll's counters then replace.
fn oc_events_since_last_poll() -> Option<u64> {
    let now = read_oc_counts();
    let mut previous = PREVIOUS_OC.lock().unwrap_or_else(PoisonError::into_inner);
    let events = now.as_ref().map(|now| oc_delta(previous.as_ref(), now));
    *previous = now;
    events
}

/// Events counted since `previous`. A channel `previous` lacks, or the
/// process's first poll, counts none.
fn oc_delta(previous: Option<&BTreeMap<String, u64>>, now: &BTreeMap<String, u64>) -> u64 {
    let Some(previous) = previous else {
        return 0;
    };
    now.iter()
        .filter_map(|(channel, count)| Some(count.saturating_sub(*previous.get(channel)?)))
        .sum()
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;
    use crate::configuration::table;

    fn idle() -> Readings {
        Readings {
            hottest_c: 48,
            cpu: CpuLoad::LoadAverage(0.04),
            gpu_load: Some(0.0),
            throttling: Some(Vec::new()),
            oc_events: Some(0),
        }
    }

    #[rstest]
    #[case("0\n", Some(0.0))]
    #[case("  999 ", Some(0.999))]
    #[case("1000", Some(1.0))]
    #[case("1001", None)]
    #[case("", None)]
    #[case("-1", None)]
    #[case("busy", None)]
    fn parses_gpu_load_permille(#[case] raw: &str, #[case] want: Option<f64>) {
        assert_eq!(parse_gpu_load(raw), want);
    }

    #[rstest]
    #[case("cpufreq-cpu0", true)]
    #[case("cpufreq-cpu8", true)]
    #[case("devfreq-17000000.gpu", true)]
    #[case("gpu-throttle-alert", true)]
    #[case("cv0-throttle-alert", true)]
    #[case("hot-surface-alert", false)]
    #[case("pwm-fan", false)]
    #[case("devfreq-emc", false)]
    fn only_clock_cutting_devices_count(#[case] kind: &str, #[case] want: bool) {
        assert_eq!(is_throttling_cooling_device(kind), want);
    }

    #[rstest]
    #[case::engaged_in_order(
        &[
            ("pwm-fan", 2),
            ("gpu-throttle-alert", 1),
            ("hot-surface-alert", 1),
            ("cpufreq-cpu4", 0),
            ("cpufreq-cpu0", 3),
        ],
        Some(&["cpufreq-cpu0:3", "gpu-throttle-alert:1"][..])
    )]
    #[case::none_engaged(&[("cpufreq-cpu0", 0)], Some(&[][..]))]
    #[case::no_throttler(&[("pwm-fan", 2), ("hot-surface-alert", 1)], None)]
    #[case::no_devices(&[], None)]
    fn active_throttling_keeps_engaged_throttlers(
        #[case] devices: &[(&str, u32)],
        #[case] want: Option<&[&str]>,
    ) {
        let devices = devices
            .iter()
            .map(|(kind, state)| (kind.to_string(), *state))
            .collect();
        let want = want.map(|w| w.iter().map(ToString::to_string).collect::<Vec<_>>());
        assert_eq!(active_throttling(devices), want);
    }

    #[rstest]
    #[case::first_poll(None, &[("oc3", 477_505)], 0)]
    #[case::unchanged(Some(&[("oc3", 477_505)][..]), &[("oc3", 477_505)], 0)]
    #[case::fired(Some(&[("oc3", 477_505)][..]), &[("oc3", 477_508)], 3)]
    #[case::summed(Some(&[("oc1", 1), ("oc3", 5)][..]), &[("oc1", 2), ("oc3", 7)], 3)]
    #[case::new_channel(Some(&[("oc3", 5)][..]), &[("oc1", 9), ("oc3", 5)], 0)]
    #[case::counter_reset(Some(&[("oc3", 5)][..]), &[("oc3", 0)], 0)]
    fn oc_delta_counts_events_since_the_previous_poll(
        #[case] previous: Option<&[(&str, u64)]>,
        #[case] now: &[(&str, u64)],
        #[case] want: u64,
    ) {
        let map = |pairs: &[(&str, u64)]| -> BTreeMap<String, u64> {
            pairs.iter().map(|(k, v)| (k.to_string(), *v)).collect()
        };
        let previous = previous.map(map);
        assert_eq!(oc_delta(previous.as_ref(), &map(now)), want);
    }

    #[rstest]
    #[case::idle(
        idle(),
        "thermal=hottest:48°C(ok;throttle99) cpu=load:4%(ok) gpu=load:0%(ok) throttle=none(ok) oc=+0(ok)"
    )]
    #[case::everything_holds(
        Readings {
            hottest_c: 99,
            cpu: CpuLoad::BusyShare(0.50),
            gpu_load: Some(0.42),
            throttling: Some(vec!["cpufreq-cpu0:3".into(), "gpu-throttle-alert:1".into()]),
            oc_events: Some(2),
        },
        "thermal=hottest:99°C(hot;throttle99) cpu=busy:50%(busy) gpu=load:42%(busy) \
         throttle=cpufreq-cpu0:3,gpu-throttle-alert:1(throttled) oc=+2(over-current)"
    )]
    #[case::probes_missing(
        Readings { gpu_load: None, throttling: None, oc_events: None, ..idle() },
        "thermal=hottest:48°C(ok;throttle99) cpu=load:4%(ok) gpu=n/a throttle=n/a oc=n/a"
    )]
    fn summary_names_every_reading(#[case] readings: Readings, #[case] expected: &str) {
        assert_eq!(format_summary(&readings, Bounds::new(None)), expected);
    }

    /// The host's numbers replace the gate's own in the same strict
    /// comparison; throttling and over-current hold whatever the table says.
    #[rstest]
    #[case::own_numbers(None, ThermalGate::Enforce, idle(), true)]
    #[case::own_temp_holds(None, ThermalGate::Enforce, Readings { hottest_c: 70, ..idle() }, false)]
    #[case::own_cpu_holds(
        None,
        ThermalGate::Enforce,
        Readings { cpu: CpuLoad::LoadAverage(0.30), ..idle() },
        false
    )]
    #[case::own_gpu_holds(None, ThermalGate::Enforce, Readings { gpu_load: Some(0.05), ..idle() }, false)]
    #[case::no_gpu_load_is_ungated(
        None,
        ThermalGate::Enforce,
        Readings { gpu_load: None, ..idle() },
        true
    )]
    #[case::no_throttler_is_ungated(
        None,
        ThermalGate::Enforce,
        Readings { throttling: None, ..idle() },
        true
    )]
    #[case::no_oc_hwmon_is_ungated(
        None,
        ThermalGate::Enforce,
        Readings { oc_events: None, ..idle() },
        true
    )]
    #[case::empty_table(Some(""), ThermalGate::Enforce, Readings { hottest_c: 70, ..idle() }, false)]
    #[case::temp_raised(
        Some("hottest_below_c = 80"),
        ThermalGate::Enforce,
        Readings { hottest_c: 79, ..idle() },
        true
    )]
    #[case::gpu_raised(
        Some("gpu_load_below_pct = 20"),
        ThermalGate::Enforce,
        Readings { gpu_load: Some(0.19), ..idle() },
        true
    )]
    #[case::gpu_raised_is_strict(
        Some("gpu_load_below_pct = 20"),
        ThermalGate::Enforce,
        Readings { gpu_load: Some(0.20), ..idle() },
        false
    )]
    #[case::cpu_raised(
        Some("cpu_load_below_pct = 50"),
        ThermalGate::Enforce,
        Readings { cpu: CpuLoad::LoadAverage(0.40), ..idle() },
        true
    )]
    #[case::throttle_holds_a_raised_bound(
        Some("hottest_below_c = 100"),
        ThermalGate::Enforce,
        Readings { throttling: Some(vec!["cpufreq-cpu0:1".into()]), ..idle() },
        false
    )]
    #[case::oc_holds(None, ThermalGate::Enforce, Readings { oc_events: Some(1), ..idle() }, false)]
    // Skipping thermal skips the temperature bound, never throttling or
    // over-current.
    #[case::skip_ignores_temp(None, ThermalGate::Skip, Readings { hottest_c: 95, ..idle() }, true)]
    #[case::skip_keeps_throttle(
        None,
        ThermalGate::Skip,
        Readings { throttling: Some(vec!["devfreq-17000000.gpu:2".into()]), ..idle() },
        false
    )]
    #[case::skip_keeps_oc(None, ThermalGate::Skip, Readings { oc_events: Some(4), ..idle() }, false)]
    #[case::skip_keeps_gpu(None, ThermalGate::Skip, Readings { gpu_load: Some(0.5), ..idle() }, false)]
    fn the_hosts_table_moves_the_gate(
        #[case] keys: Option<&str>,
        #[case] thermal: ThermalGate,
        #[case] readings: Readings,
        #[case] want_ready: bool,
    ) -> Result<(), toml::de::Error> {
        let config = keys
            .map(|keys| table(ReadinessPlatform::JetsonOrin, keys))
            .transpose()?;
        let bounds = Bounds::new(config.as_ref());
        assert_eq!(bounds.ready(&readings, thermal), want_ready);
        Ok(())
    }
}
