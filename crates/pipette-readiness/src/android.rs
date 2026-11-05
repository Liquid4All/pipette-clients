//! Android readiness: the `dumpsys thermalservice` status must be 0, the
//! hottest CPU-cluster zone under [`THERMAL_THRESHOLD_C`], the `/proc/stat`
//! busy share under [`CPU_LOAD_THRESHOLD`], and no detected held touch on the
//! touchscreen unless [`HeldTouch::Off`] disables that criterion.
//!
//! The held-touch check has two stages. The 1 s CPU sample also counts the
//! touch controller's interrupts. If they reach [`TOUCH_IRQ_HELD_PER_SEC`],
//! the summed count decreases, or a known controller has malformed counts,
//! the gate reads `dumpsys input` to confirm that a pointer is down. The check
//! recognizes only the controller names in
//! [`TOUCH_IRQ_NAMES`]. It cannot
//! detect a held touch below the rate threshold or on an unlisted controller.
//! The contact interrupt rate has not been measured on the listed phones.
//!
//! The status enum (`PowerManager.THERMAL_STATUS_*`) is a power-management
//! judgment, not a temperature: it stays 0 with CPU dies at 75-80 °C on
//! Snapdragon 8 Elite, so the zone check is what keeps reps off hot silicon.
//!
//! Not `/proc/loadavg`: its 1-minute EMA carries One UI's background work and
//! `adbd` traffic, parking resting load at 0.35-0.45 on tethered S25-class
//! phones. The instantaneous busy share reads high single digits at idle.

#![cfg_attr(
    not(target_os = "android"),
    allow(dead_code, reason = "off Android only the tests reach this")
)]

use std::fs;
use std::path::Path;
use std::time::{Duration, Instant};

use super::wait::{Probe, wait_loop};
use super::{
    CeilingCelsius, CpuPercent, HeldTouch, ReadinessCallbacks, ReadinessConfiguration,
    ReadinessError, ReadinessPlatform, ThermalGate,
};

pub(super) const DEFAULT_MAX_WAIT: Duration = super::DEFAULT_MAX_WAIT;
/// Long enough to smooth 100 ms jiffy noise without noticeably extending the
/// wait.
const CPU_SAMPLE_INTERVAL: Duration = Duration::from_secs(1);
/// Highest acceptable CPU-cluster die temperature, °C (strict `<`, so it
/// releases at 33 °C). Snapdragon 8 Elite throttles at ~75-80 °C.
///
/// Measured: 50 °C left ±29.8-34.6 % stddev on 20-40 s reps, 35 °C cut most to
/// ±1-8 %, and 34 °C targets the residual ±5-8 % on 10-15 s reps. Tethered
/// S25-class phones rest at ~32 °C, and a 32 °C threshold parked all three
/// phones there for minutes, so a lower threshold risks a gate that never
/// releases. Cooldown from a ~75 °C post-rep peak takes ~30-40 s.
///
/// Reps of 20 s or more throttle mid-rep whatever the entry temperature.
const THERMAL_THRESHOLD_C: i32 = 34;
/// Highest acceptable busy share across all cores. Idle Snapdragon-class SoCs
/// rest well under 0.10, and a concurrent benchmark is well over.
const CPU_LOAD_THRESHOLD: f64 = 0.30;

/// The touch controller's line in `/proc/interrupts`, by the name in its last
/// column: `synaptics_tcm` on the Pixel 11 Pro and `synaptics_ts` on the Galaxy
/// S26 Ultra. The gate cannot detect a held touch on a phone with neither line.
const TOUCH_IRQ_NAMES: &[&str] = &["synaptics_tcm", "synaptics_ts"];
/// The touch interrupt rate, per second, at or above which the gate reads
/// `dumpsys input` for a held pointer.
///
/// Provisional. Both phones above read 0/s at idle. We have not measured a
/// contact rate yet, so a held touch below this rate passes the gate.
const TOUCH_IRQ_HELD_PER_SEC: f64 = 20.0;
/// Under [`HeldTouch::Fail`], the number of consecutive polls with a held
/// touch that fails the wait: about 9 s at the 3 s poll interval. One brief
/// operator touch does not fail a cell.
const HELD_TOUCH_POLLS: usize = 3;

/// The probe error when `/sys/class/thermal` is unreadable and the `dumpsys`
/// fallback also has no live CPU temperature.
const NO_DUMPSYS_CPU_TEMPERATURE: &str = "`dumpsys thermalservice` reports no live CPU \
     temperature, and /sys/class/thermal is unreadable";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CpuTimes {
    /// `idle + iowait` jiffies since boot.
    idle: u64,
    total: u64,
}

/// The gate's own numbers, or the ones the host's `[readiness.android]` table
/// sets.
#[derive(Debug, Clone, Copy)]
struct Bounds {
    hottest_below_c: i32,
    skin_below_c: Option<CeilingCelsius>,
    cpu_busy_below: f64,
}

impl Bounds {
    fn new(configuration: Option<&ReadinessConfiguration>) -> Self {
        let config = configuration.and_then(ReadinessConfiguration::android);
        let hottest = config.and_then(|c| c.hottest_below_c);
        let busy = config.and_then(|c| c.cpu_busy_below_pct);
        Self {
            hottest_below_c: hottest.map_or(THERMAL_THRESHOLD_C, |c| c.get().cast_signed()),
            skin_below_c: config.and_then(|c| c.skin_below_c),
            cpu_busy_below: busy.map_or(CPU_LOAD_THRESHOLD, CpuPercent::ratio),
        }
    }

    fn with_skin_below_c(mut self, threshold: Option<CeilingCelsius>) -> Self {
        self.skin_below_c = threshold;
        self
    }

    /// Both the OS status enum and the die zone are thermal signals, so
    /// `ThermalGate::Skip` waives both; CPU load is unaffected. If the gate is
    /// enforced and `thermal` is `None`, the device is not ready, because an
    /// unknown temperature is not a safe temperature.
    fn ready(self, thermal: Option<ThermalReading>, cpu_load: f64, gate: ThermalGate) -> bool {
        let thermal_ok = gate == ThermalGate::Skip
            || thermal.is_some_and(|t| {
                t.status == 0
                    && t.hottest_c < self.hottest_below_c
                    && self
                        .skin_below_c
                        .is_none_or(|c| t.skin_c.is_some_and(|skin| skin < f64::from(c.get())))
            });
        thermal_ok && cpu_load < self.cpu_busy_below
    }
}

/// The CPU busy share and the touch interrupt rate, from one 1 s sample.
#[derive(Debug, Clone, Copy)]
struct LoadSample {
    cpu_load: f64,
    /// `None` if `/proc/interrupts` has no line in [`TOUCH_IRQ_NAMES`].
    touch_irq_per_sec: Option<f64>,
    touch_irq_count_decreased: bool,
    touch_irq_count_malformed: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TouchIrqCount {
    Unchecked,
    Count(u64),
    Malformed,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum TouchReading {
    /// The host disabled the held-touch criterion.
    Disabled,
    /// This phone has no known touch interrupt line.
    Unchecked,
    /// The interrupt rate, and the pointer count from `dumpsys input` if the
    /// rate reached [`TOUCH_IRQ_HELD_PER_SEC`], the count decreased, or the
    /// counts were malformed.
    Read {
        irq_per_sec: f64,
        pointers: Option<u32>,
    },
}

impl TouchReading {
    fn held(self) -> bool {
        matches!(self, Self::Read { pointers: Some(p), .. } if p > 0)
    }
}

/// The OS thermal status and the hottest CPU temperature, from one thermal
/// read.
#[derive(Debug, Clone, Copy)]
struct ThermalReading {
    status: u8,
    hottest_c: i32,
    skin_c: Option<f64>,
    skin_block_complete: bool,
}

pub(super) fn wait_until_ready(
    max_wait: Duration,
    thermal: ThermalGate,
    configuration: Option<&ReadinessConfiguration>,
    skin_below_c: Option<CeilingCelsius>,
    on_held_touch: HeldTouch,
    callbacks: &ReadinessCallbacks<'_>,
) -> Result<(), ReadinessError> {
    let mut probe = AndroidProbe {
        bounds: Bounds::new(configuration).with_skin_below_c(skin_below_c),
        thermal,
        on_held_touch,
        skipped_unreadable: false,
        missing_skin_polls: 0,
        held_touch_polls: 0,
        warned_unchecked: false,
        read_thermal,
        read_load: read_load_sample,
        read_touch_pointers,
    };
    wait_loop(ReadinessPlatform::Android, max_wait, callbacks, || {
        probe.poll()
    })
}

/// The probe for one wait. The readers are fields, so the tests can run a poll
/// without a phone.
struct AndroidProbe<T, L, P> {
    bounds: Bounds,
    thermal: ThermalGate,
    on_held_touch: HeldTouch,
    /// Set by the first failed thermal read under a skipped gate. After it, the
    /// probe does not read the thermal state again in this wait: a hung
    /// `dumpsys` costs `PROBE_TIMEOUT` on each read, and each failed read logs
    /// the full `dumpsys` output. The next wait reads the thermal state again.
    skipped_unreadable: bool,
    /// Complete live HAL blocks without a valid `mType=3` sensor in a row.
    missing_skin_polls: usize,
    held_touch_polls: usize,
    /// Set after the one warning, per wait, that this phone has no known
    /// touch interrupt line.
    warned_unchecked: bool,
    read_thermal: T,
    read_load: L,
    /// The highest pointer count of a touchscreen in `dumpsys input`.
    read_touch_pointers: P,
}

impl<T, L, P> AndroidProbe<T, L, P>
where
    T: FnMut() -> Result<ThermalReading, ReadinessError>,
    L: FnMut(bool) -> Result<LoadSample, ReadinessError>,
    P: FnMut() -> Result<u32, ReadinessError>,
{
    fn poll(&mut self) -> Result<Probe, ReadinessError> {
        // Read the thermal state even if the gate is skipped, so the log line
        // shows the thermal state of a run that does not gate on it. A skipped
        // gate does not use the reading, so a failed read does not fail the
        // poll. `macos.rs` and `linux/generic.rs` make the same split.
        let reading = if self.skipped_unreadable {
            None
        } else {
            match (self.read_thermal)() {
                Ok(reading) => Some(reading),
                Err(err) if self.thermal == ThermalGate::Skip => {
                    self.missing_skin_polls = 0;
                    tracing::warn!(
                        error = %err,
                        source = std::error::Error::source(&err).map(tracing::field::display),
                        "android thermal probe unreadable; the thermal gate is skipped, so \
                         continuing without it for the rest of this wait"
                    );
                    self.skipped_unreadable = true;
                    None
                }
                Err(err) => {
                    self.missing_skin_polls = 0;
                    self.held_touch_polls = 0;
                    return Err(err);
                }
            }
        };
        if self.thermal != ThermalGate::Skip && self.bounds.skin_below_c.is_some() {
            match reading {
                Some(ThermalReading {
                    skin_block_complete: false,
                    ..
                }) => {
                    self.missing_skin_polls = 0;
                    self.held_touch_polls = 0;
                    return Err(ReadinessError::Unavailable(
                        "incomplete live HAL temperature block in `dumpsys thermalservice`",
                    ));
                }
                Some(ThermalReading {
                    skin_c: Some(_),
                    skin_block_complete: true,
                    ..
                }) => self.missing_skin_polls = 0,
                Some(ThermalReading {
                    skin_c: None,
                    skin_block_complete: true,
                    ..
                }) => {
                    self.missing_skin_polls += 1;
                    if self.missing_skin_polls >= super::PROBE_RETRY_POLLS {
                        self.held_touch_polls = 0;
                        tracing::warn!(
                            skin = "unavailable",
                            verdict = "unsupported",
                            "android skin readiness poll"
                        );
                        return Err(ReadinessError::BoundWithoutSkinSensor);
                    }
                }
                None => {}
            }
        }
        let check_touch = self.on_held_touch != HeldTouch::Off;
        let load = (self.read_load)(check_touch).inspect_err(|_| {
            self.missing_skin_polls = 0;
            self.held_touch_polls = 0;
        })?;
        let touch = if check_touch {
            self.read_touch(
                load.touch_irq_per_sec,
                load.touch_irq_count_decreased,
                load.touch_irq_count_malformed,
            )
            .inspect_err(|_| {
                self.missing_skin_polls = 0;
                self.held_touch_polls = 0;
            })?
        } else {
            TouchReading::Disabled
        };
        Ok(Probe {
            ready: self.bounds.ready(reading, load.cpu_load, self.thermal) && !touch.held(),
            summary: format_summary(reading, load.cpu_load, touch, self.thermal, self.bounds),
        })
    }

    /// The held-touch stage of a poll. `ThermalGate::Skip` does not waive it,
    /// because a held touch is load, not heat.
    fn read_touch(
        &mut self,
        irq_per_sec: Option<f64>,
        count_decreased: bool,
        count_malformed: bool,
    ) -> Result<TouchReading, ReadinessError> {
        let Some(irq_per_sec) = irq_per_sec else {
            self.held_touch_polls = 0;
            if !self.warned_unchecked {
                self.warned_unchecked = true;
                tracing::warn!(
                    known_lines = ?TOUCH_IRQ_NAMES,
                    "no known touch interrupt line in /proc/interrupts; the held-touch check \
                     is off for this wait"
                );
            }
            return Ok(TouchReading::Unchecked);
        };
        if count_decreased {
            tracing::warn!("touch interrupt count decreased; checking touchscreen pointers");
        }
        if count_malformed {
            tracing::warn!("touch interrupt count malformed; checking touchscreen pointers");
        }
        let pointers =
            (irq_per_sec >= TOUCH_IRQ_HELD_PER_SEC || count_decreased || count_malformed)
                .then(|| (self.read_touch_pointers)())
                .transpose()?;
        let reading = TouchReading::Read {
            irq_per_sec,
            pointers,
        };
        let Some(pointers @ 1..) = pointers else {
            if self.held_touch_polls > 0 {
                tracing::info!(
                    held_polls = self.held_touch_polls,
                    irq_per_sec,
                    "held touch cleared"
                );
                self.held_touch_polls = 0;
            }
            return Ok(reading);
        };
        self.held_touch_polls += 1;
        if self.held_touch_polls == 1 {
            tracing::warn!(
                irq_per_sec,
                pointers,
                mode = %self.on_held_touch,
                "held touch on the touchscreen; holding the gate"
            );
        }
        if self.on_held_touch == HeldTouch::Fail && self.held_touch_polls >= HELD_TOUCH_POLLS {
            return Err(ReadinessError::HeldTouch {
                pointers,
                irq_per_sec: whole_rate(irq_per_sec),
            });
        }
        Ok(reading)
    }
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the rate is a count delta over a ~1 s window: finite, non-negative, and far \
              below u32::MAX"
)]
fn whole_rate(per_sec: f64) -> u32 {
    per_sec.round() as u32
}

fn read_thermal() -> Result<ThermalReading, ReadinessError> {
    let dumpsys = read_dumpsys_thermal()?;
    let status = parse_dumpsys_status(&dumpsys)?;
    let hottest_c = hottest_cpu_celsius(read_hottest_cpu_zone_celsius(), &dumpsys)?;
    let (skin_c, skin_block_complete) = parse_live_skin_celsius(&dumpsys);
    Ok(ThermalReading {
        status,
        hottest_c,
        skin_c,
        skin_block_complete,
    })
}

/// The hottest CPU die temperature, from sysfs if it reads and from `dumpsys`
/// otherwise.
///
/// `/sys/class/thermal` is the primary source, but SELinux refuses it to apps
/// and to the adb shell user on Pixel/Tensor. There, every poll reads the CPU
/// temperatures that `dumpsys` already reported (the same Hardware
/// Abstraction Layer (HAL) values, `mType=0`). The sysfs refusal is expected
/// on those phones. A fallback that finds no temperature returns
/// [`NO_DUMPSYS_CPU_TEMPERATURE`] and logs the `dumpsys` output, because that
/// output shows the cause.
fn hottest_cpu_celsius(
    sysfs: Result<i32, ReadinessError>,
    dumpsys: &str,
) -> Result<i32, ReadinessError> {
    let sysfs_err = match sysfs {
        Ok(c) => return Ok(c),
        Err(err) => err,
    };
    parse_hottest_cpu_celsius_from_dumpsys(dumpsys).ok_or_else(|| {
        tracing::warn!(
            sysfs_error = %sysfs_err,
            // The file and the errno, which `ReadinessError::Io` leaves out of
            // its own text.
            sysfs_source = std::error::Error::source(&sysfs_err).map(tracing::field::display),
            dumpsys,
            "dumpsys fallback has no live CPU temperature"
        );
        ReadinessError::Unavailable(NO_DUMPSYS_CPU_TEMPERATURE)
    })
}

/// The summary line for the log and for `TimedOut`. `thermal` is `None` only
/// if the read failed and the gate was skipped. A skipped gate is marked
/// `[SKIPPED]`, because a skipped run is not comparable to a gated one.
fn format_summary(
    thermal: Option<ThermalReading>,
    cpu_load: f64,
    touch: TouchReading,
    gate: ThermalGate,
    bounds: Bounds,
) -> String {
    let thermal_field = thermal.map_or_else(
        || "unreadable".to_string(),
        |ThermalReading { status, hottest_c, skin_c, .. }| {
            let temp_verdict = if hottest_c < bounds.hottest_below_c {
                "ok"
            } else {
                "hot"
            };
            let skin_verdict = match (skin_c, bounds.skin_below_c) {
                (Some(skin), Some(bound)) if skin < f64::from(bound.get()) => "ok",
                (Some(_), Some(_)) => "hot",
                (Some(_), None) | (None, None) => "off",
                (None, Some(_)) => "held",
            };
            let skin = skin_c.map_or_else(|| "unavailable".to_owned(), |c| format!("{c}°C"));
            format!(
                "status:{status}({}),hottest:{hottest_c}°C({temp_verdict}),skin:{skin}({skin_verdict})",
                format_status_word(status),
            )
        },
    );
    let note = if gate == ThermalGate::Skip {
        "[SKIPPED]"
    } else {
        ""
    };
    let cpu_pct = (cpu_load * 100.0).round();
    let cpu_verdict = if cpu_load < bounds.cpu_busy_below {
        "ok"
    } else {
        "busy"
    };
    let touch_field = match touch {
        TouchReading::Disabled => "off".to_owned(),
        TouchReading::Unchecked => "unchecked".to_owned(),
        TouchReading::Read {
            irq_per_sec,
            pointers,
        } => {
            let verdict = if touch.held() { "held" } else { "ok" };
            let pointers =
                pointers.map_or_else(String::new, |pointers| format!(",pointers:{pointers}"));
            format!("irq:{irq_per_sec:.0}/s{pointers}({verdict})")
        }
    };
    format!("thermal={thermal_field}{note} cpu=busy:{cpu_pct}%({cpu_verdict}) touch={touch_field}")
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
    let mut cmd = std::process::Command::new("/system/bin/dumpsys");
    cmd.arg("thermalservice");
    let output = crate::probe_output(cmd)?;
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

/// Snapdragon's `cpu-<cluster>-<core>` and `cpullc-<cluster>-<core>` zones.
///
/// The digit after the prefix excludes `cpu-hw-trip-*`, pseudo-zones that
/// report the 105 °C hardware trip as a constant and would pin the gate
/// hot. On S25-class phones `pmh0104_tz` and `pmr735d_tz` are constant too
/// (37 000 mC); `pmh0101_tz`, `pmh0110_{d,f,g,i}_tz` and `sys-therm-0` vary in
/// real time, should the gate ever widen beyond CPU dies.
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

/// Both counters bracket the same sleep. With touch checks disabled, read only
/// CPU load, so an unreadable `/proc/interrupts` cannot fail the poll.
fn read_load_sample(check_touch: bool) -> Result<LoadSample, ReadinessError> {
    let first = read_proc_stat_cpu()?;
    let first_irqs = if check_touch {
        read_touch_irq_count()?
    } else {
        TouchIrqCount::Unchecked
    };
    let started = Instant::now();
    std::thread::sleep(CPU_SAMPLE_INTERVAL);
    let second = read_proc_stat_cpu()?;
    let second_irqs = if check_touch {
        read_touch_irq_count()?
    } else {
        TouchIrqCount::Unchecked
    };
    let secs = started.elapsed().as_secs_f64();
    let touch_irq_count_malformed =
        first_irqs == TouchIrqCount::Malformed || second_irqs == TouchIrqCount::Malformed;
    let touch_irqs = match (first_irqs, second_irqs) {
        (TouchIrqCount::Count(first), TouchIrqCount::Count(second)) => {
            Some(touch_irq_rate(first, second, secs))
        }
        _ => None,
    };
    Ok(LoadSample {
        cpu_load: busy_ratio(first, second),
        touch_irq_per_sec: touch_irqs
            .map(|(rate, _)| rate)
            .or_else(|| touch_irq_count_malformed.then_some(0.0)),
        touch_irq_count_decreased: touch_irqs.is_some_and(|(_, decreased)| decreased),
        touch_irq_count_malformed,
    })
}

/// A failed read propagates: if the gate read it as "no touch", a held touch
/// would pass.
fn read_touch_irq_count() -> Result<TouchIrqCount, ReadinessError> {
    Ok(parse_touch_irq_count(&fs::read_to_string(
        "/proc/interrupts",
    )?))
}

/// The touch line's count summed over the CPU columns. Use the header to
/// distinguish an invalid count from the controller metadata after the counts.
fn parse_touch_irq_count(raw: &str) -> TouchIrqCount {
    let Some(line) = raw.lines().find(|line| {
        line.split_whitespace()
            .last()
            .is_some_and(|name| TOUCH_IRQ_NAMES.contains(&name))
    }) else {
        return TouchIrqCount::Unchecked;
    };
    let Some(cpu_columns) = raw.lines().find_map(|line| {
        let mut tokens = line.split_whitespace();
        (tokens.next() == Some("CPU0")).then(|| 1 + tokens.count())
    }) else {
        return TouchIrqCount::Malformed;
    };
    // `755:`, then one count per online CPU, then the controller, the
    // hardware IRQ number, the trigger type and the name.
    let mut counts = line.split_whitespace().skip(1);
    let mut total = 0_u64;
    for _ in 0..cpu_columns {
        let Some(count) = counts.next().and_then(|count| count.parse::<u64>().ok()) else {
            return TouchIrqCount::Malformed;
        };
        total = total.saturating_add(count);
    }
    TouchIrqCount::Count(total)
}

/// Clamp a decrease to zero and mark it for a direct pointer check. An offline
/// CPU can drop its column from the sum.
fn touch_irq_rate(first: u64, second: u64, secs: f64) -> (f64, bool) {
    let decreased = second < first;
    let delta = second.saturating_sub(first) as f64;
    (if secs > 0.0 { delta / secs } else { 0.0 }, decreased)
}

fn read_touch_pointers() -> Result<u32, ReadinessError> {
    let mut cmd = std::process::Command::new("/system/bin/dumpsys");
    cmd.arg("input");
    let output = crate::probe_output(cmd)?;
    parse_touch_pointers(&String::from_utf8_lossy(&output.stdout)).ok_or(
        ReadinessError::Unavailable("`dumpsys input` reports no touchscreen pointer count"),
    )
}

/// The highest `Last Cooked Touch: pointerCount=` of a `DeviceType:
/// TOUCH_SCREEN` device in `dumpsys input`. A stylus digitizer reports
/// `DeviceType: POINTER`, so a pen that hovers does not count.
fn parse_touch_pointers(dump: &str) -> Option<u32> {
    let mut in_touchscreen = false;
    dump.lines()
        .map(str::trim)
        .filter_map(|line| {
            // `Device 3: synaptics_tcm_touch` or `Device -1: Virtual` starts a
            // device's section.
            let device_header = line
                .strip_prefix("Device ")
                .and_then(|rest| rest.split_once(':'))
                .is_some_and(|(id, _)| id.parse::<i32>().is_ok());
            if device_header {
                in_touchscreen = false;
                return None;
            }
            if line == "DeviceType: TOUCH_SCREEN" {
                in_touchscreen = true;
                return None;
            }
            if !in_touchscreen {
                return None;
            }
            line.strip_prefix("Last Cooked Touch: pointerCount=")?
                .parse()
                .ok()
        })
        .max()
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
    let [_user, _nice, _system, idle, iowait, ..] = fields.as_slice() else {
        return Err(ReadinessError::ParseFailure {
            raw: line.to_string(),
        });
    };
    Ok(CpuTimes {
        idle: idle + iowait,
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

/// Hottest CPU temperature (`mType=0`) in dumpsys's live `Current temperatures
/// from HAL:` block. Not the `Cached temperatures:` block, whose stale peak
/// would pin the gate hot. The band filter drops the `-FLT_MAX` no-reading
/// sentinel.
#[expect(
    clippy::cast_possible_truncation,
    reason = "the band check bounds `value` to -50..=200"
)]
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

/// The hottest valid skin reading in the live HAL block, with a completion
/// marker for deciding whether repeated absence means this host has no sensor.
/// A partial dump cannot establish that absence.
fn parse_live_skin_celsius(stdout: &str) -> (Option<f64>, bool) {
    let mut in_current = false;
    let mut hottest: Option<f64> = None;
    for line in stdout.lines().map(str::trim) {
        if line == "Current temperatures from HAL:" {
            in_current = true;
            continue;
        }
        if !in_current || line.is_empty() {
            continue;
        }
        if !line.starts_with("Temperature{") {
            return (hottest, true);
        }
        let Some(m_type) = dumpsys_field(line, "mType=").and_then(|s| s.parse::<i32>().ok()) else {
            continue;
        };
        let Some(value) = dumpsys_field(line, "mValue=").and_then(|s| s.parse::<f64>().ok()) else {
            continue;
        };
        if !value.is_finite() || !(-50.0..=200.0).contains(&value) {
            continue;
        }
        if m_type == 3 {
            hottest = Some(hottest.map_or(value, |current| current.max(value)));
        }
    }
    (hottest, false)
}

/// The value of `key` (e.g. `"mType="`) in a `Temperature{..}` line.
fn dumpsys_field<'a>(line: &'a str, key: &str) -> Option<&'a str> {
    let (_, rest) = line.split_once(key)?;
    let value = rest.split_once([',', '}']).map_or(rest, |(value, _)| value);
    Some(value.trim())
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use anyhow::Context;
    use rstest::rstest;

    use super::*;
    use crate::configuration::table;

    /// The host's numbers replace the gate's own in the same strict
    /// comparison, each key alone; the OS status stays required.
    #[rstest]
    #[case::own_numbers(None, ThermalGate::Enforce, 0, 33, 0.29, true)]
    #[case::own_temp_holds(None, ThermalGate::Enforce, 0, 34, 0.05, false)]
    #[case::own_busy_holds(None, ThermalGate::Enforce, 0, 30, 0.30, false)]
    #[case::own_status_holds(None, ThermalGate::Enforce, 1, 30, 0.05, false)]
    #[case::empty_table(Some(""), ThermalGate::Enforce, 0, 34, 0.05, false)]
    #[case::restated_temp(
        Some("hottest_below_c = 34\ncpu_busy_below_pct = 30"),
        ThermalGate::Enforce,
        0,
        34,
        0.05,
        false
    )]
    #[case::restated_busy(
        Some("hottest_below_c = 34\ncpu_busy_below_pct = 30"),
        ThermalGate::Enforce,
        0,
        30,
        0.30,
        false
    )]
    #[case::temp_raised(Some("hottest_below_c = 36"), ThermalGate::Enforce, 0, 35, 0.05, true)]
    #[case::temp_raised_is_strict(
        Some("hottest_below_c = 36"),
        ThermalGate::Enforce,
        0,
        36,
        0.05,
        false
    )]
    #[case::temp_lowered(Some("hottest_below_c = 33"), ThermalGate::Enforce, 0, 33, 0.05, false)]
    #[case::temp_key_leaves_busy(
        Some("hottest_below_c = 36"),
        ThermalGate::Enforce,
        0,
        30,
        0.35,
        false
    )]
    #[case::busy_raised(
        Some("cpu_busy_below_pct = 40"),
        ThermalGate::Enforce,
        0,
        30,
        0.35,
        true
    )]
    #[case::busy_raised_is_strict(
        Some("cpu_busy_below_pct = 40"),
        ThermalGate::Enforce,
        0,
        30,
        0.40,
        false
    )]
    #[case::busy_key_leaves_temp(
        Some("cpu_busy_below_pct = 40"),
        ThermalGate::Enforce,
        0,
        35,
        0.05,
        false
    )]
    #[case::status_still_required(
        Some("hottest_below_c = 36\ncpu_busy_below_pct = 40"),
        ThermalGate::Enforce,
        1,
        30,
        0.05,
        false
    )]
    // Skipping thermal skips the temperature bound and the status; busy stays.
    #[case::skip_ignores_temp(Some("hottest_below_c = 33"), ThermalGate::Skip, 2, 60, 0.05, true)]
    #[case::skip_keeps_busy(Some("cpu_busy_below_pct = 40"), ThermalGate::Skip, 0, 60, 0.45, false)]
    fn the_hosts_table_moves_the_gate(
        #[case] keys: Option<&str>,
        #[case] thermal: ThermalGate,
        #[case] status: u8,
        #[case] hottest_c: i32,
        #[case] cpu_load: f64,
        #[case] want_ready: bool,
    ) -> Result<(), toml::de::Error> {
        let config = keys
            .map(|keys| table(ReadinessPlatform::Android, keys))
            .transpose()?;
        let bounds = Bounds::new(config.as_ref());
        let reading = ThermalReading {
            status,
            hottest_c,
            skin_c: None,
            skin_block_complete: false,
        };
        assert_eq!(bounds.ready(Some(reading), cpu_load, thermal), want_ready);
        Ok(())
    }

    /// A failed thermal read reaches `ready` only if the gate is skipped. A
    /// skipped gate waives the read, and an enforced gate stays closed. CPU
    /// load still counts.
    #[rstest]
    #[case::skipped(ThermalGate::Skip, 0.05, true)]
    #[case::skipped_still_busy(ThermalGate::Skip, 0.35, false)]
    #[case::enforced(ThermalGate::Enforce, 0.05, false)]
    fn an_unread_thermal_signal(
        #[case] thermal: ThermalGate,
        #[case] cpu_load: f64,
        #[case] want_ready: bool,
    ) {
        assert_eq!(Bounds::new(None).ready(None, cpu_load, thermal), want_ready);
    }

    #[rstest]
    #[case::below(
        Some("skin_below_c = 34"),
        Some(33.9),
        ThermalGate::Enforce,
        0.05,
        true
    )]
    #[case::equal(
        Some("skin_below_c = 34"),
        Some(34.0),
        ThermalGate::Enforce,
        0.05,
        false
    )]
    #[case::hot(
        Some("skin_below_c = 34"),
        Some(35.2),
        ThermalGate::Enforce,
        0.05,
        false
    )]
    #[case::missing(Some("skin_below_c = 34"), None, ThermalGate::Enforce, 0.05, false)]
    #[case::off(None, Some(35.2), ThermalGate::Enforce, 0.05, true)]
    #[case::skipped(Some("skin_below_c = 34"), Some(35.2), ThermalGate::Skip, 0.05, true)]
    #[case::skipped_still_busy(Some("skin_below_c = 34"), None, ThermalGate::Skip, 0.35, false)]
    fn skin_bound_is_strict_and_skippable(
        #[case] keys: Option<&str>,
        #[case] skin_c: Option<f64>,
        #[case] gate: ThermalGate,
        #[case] cpu_load: f64,
        #[case] expected: bool,
    ) -> anyhow::Result<()> {
        let config = keys
            .map(|keys| table(ReadinessPlatform::Android, keys))
            .transpose()?;
        let bounds = Bounds::new(config.as_ref());
        let reading = ThermalReading {
            status: 0,
            hottest_c: 33,
            skin_c,
            skin_block_complete: true,
        };
        assert_eq!(bounds.ready(Some(reading), cpu_load, gate), expected);
        Ok(())
    }

    #[rstest]
    #[case(
        None,
        Some(33),
        ThermalGate::Enforce,
        0.05,
        "thermal=status:0(nominal),hottest:33°C(ok),skin:unavailable(off) cpu=busy:5%(ok) touch=irq:0/s(ok)"
    )]
    #[case(
        None,
        Some(35),
        ThermalGate::Enforce,
        0.35,
        "thermal=status:0(nominal),hottest:35°C(hot),skin:unavailable(off) cpu=busy:35%(busy) touch=irq:0/s(ok)"
    )]
    #[case(
        Some("hottest_below_c = 36\ncpu_busy_below_pct = 40"),
        Some(35),
        ThermalGate::Enforce,
        0.35,
        "thermal=status:0(nominal),hottest:35°C(ok),skin:unavailable(off) cpu=busy:35%(ok) touch=irq:0/s(ok)"
    )]
    #[case(
        Some("hottest_below_c = 36\ncpu_busy_below_pct = 40"),
        Some(36),
        ThermalGate::Enforce,
        0.40,
        "thermal=status:0(nominal),hottest:36°C(hot),skin:unavailable(off) cpu=busy:40%(busy) touch=irq:0/s(ok)"
    )]
    #[case(
        None,
        Some(33),
        ThermalGate::Skip,
        0.05,
        "thermal=status:0(nominal),hottest:33°C(ok),skin:unavailable(off)[SKIPPED] cpu=busy:5%(ok) touch=irq:0/s(ok)"
    )]
    #[case(
        None,
        None,
        ThermalGate::Skip,
        0.05,
        "thermal=unreadable[SKIPPED] cpu=busy:5%(ok) touch=irq:0/s(ok)"
    )]
    fn summary_judges_against_the_bounds(
        #[case] keys: Option<&str>,
        #[case] hottest_c: Option<i32>,
        #[case] thermal: ThermalGate,
        #[case] cpu_load: f64,
        #[case] want: &str,
    ) -> Result<(), toml::de::Error> {
        let config = keys
            .map(|keys| table(ReadinessPlatform::Android, keys))
            .transpose()?;
        let bounds = Bounds::new(config.as_ref());
        let reading = hottest_c.map(|hottest_c| ThermalReading {
            status: 0,
            hottest_c,
            skin_c: None,
            skin_block_complete: false,
        });
        assert_eq!(
            format_summary(reading, cpu_load, QUIET_TOUCH, thermal, bounds),
            want
        );
        Ok(())
    }

    #[rstest]
    #[case::below(33.9, "skin:33.9°C(ok)")]
    #[case::equal(34.0, "skin:34°C(hot)")]
    #[case::above(35.2, "skin:35.2°C(hot)")]
    fn summary_shows_skin_reading_and_verdict(
        #[case] skin_c: f64,
        #[case] expected: &str,
    ) -> anyhow::Result<()> {
        let config = table(ReadinessPlatform::Android, "skin_below_c = 34")?;
        let reading = ThermalReading {
            skin_c: Some(skin_c),
            ..COOL
        };
        let summary = format_summary(
            Some(reading),
            0.05,
            QUIET_TOUCH,
            ThermalGate::Enforce,
            Bounds::new(Some(&config)),
        );
        assert!(summary.contains(expected), "{summary}");
        Ok(())
    }

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

    // Real `dumpsys thermalservice` shape (Pixel/Tensor).
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

    const COOL: ThermalReading = ThermalReading {
        status: 0,
        hottest_c: 33,
        skin_c: Some(33.0),
        skin_block_complete: true,
    };

    fn thermal_failed() -> Result<ThermalReading, ReadinessError> {
        Err(ReadinessError::Unavailable("dumpsys failed"))
    }

    const QUIET_TOUCH: TouchReading = TouchReading::Read {
        irq_per_sec: 0.0,
        pointers: None,
    };

    fn quiet_load(_: bool) -> Result<LoadSample, ReadinessError> {
        Ok(LoadSample {
            cpu_load: 0.05,
            touch_irq_per_sec: Some(0.0),
            touch_irq_count_decreased: false,
            touch_irq_count_malformed: false,
        })
    }

    /// For a probe whose script never reaches the `dumpsys input` stage.
    fn no_pointers_read() -> Result<u32, ReadinessError> {
        Err(ReadinessError::Unavailable(
            "read `dumpsys input` at a quiet rate",
        ))
    }

    /// Each case polls once for each entry in `want`. An entry is the summary
    /// of the poll, or `None` if the poll failed. A skipped gate continues
    /// after a failed thermal read and does not read the thermal state again
    /// in that wait. An enforced gate fails the poll.
    #[rstest]
    #[case::skipped_failed_read_is_not_read_again(
        ThermalGate::Skip,
        vec![thermal_failed(), Ok(COOL)],
        &[
            Some("thermal=unreadable[SKIPPED] cpu=busy:5%(ok) touch=irq:0/s(ok)"),
            Some("thermal=unreadable[SKIPPED] cpu=busy:5%(ok) touch=irq:0/s(ok)"),
        ],
        1
    )]
    #[case::enforced_failed_read_fails_the_poll(
        ThermalGate::Enforce,
        vec![thermal_failed(), Ok(COOL)],
        &[None, Some("thermal=status:0(nominal),hottest:33°C(ok),skin:33°C(off) cpu=busy:5%(ok) touch=irq:0/s(ok)")],
        2
    )]
    #[case::good_read(
        ThermalGate::Enforce,
        vec![Ok(COOL)],
        &[Some("thermal=status:0(nominal),hottest:33°C(ok),skin:33°C(off) cpu=busy:5%(ok) touch=irq:0/s(ok)")],
        1
    )]
    fn a_poll_reads_thermal_until_a_skipped_read_fails(
        #[case] thermal: ThermalGate,
        #[case] thermal_reads: Vec<Result<ThermalReading, ReadinessError>>,
        #[case] want: &[Option<&str>],
        #[case] want_thermal_reads: usize,
    ) {
        let reads = Cell::new(0usize);
        let mut script = thermal_reads.into_iter();
        let mut probe = AndroidProbe {
            bounds: Bounds::new(None),
            thermal,
            on_held_touch: HeldTouch::Wait,
            skipped_unreadable: false,
            missing_skin_polls: 0,
            held_touch_polls: 0,
            warned_unchecked: false,
            read_thermal: || {
                reads.set(reads.get() + 1);
                // A probe that reads past its script is the bug this reports.
                script
                    .next()
                    .unwrap_or(Err(ReadinessError::Unavailable("read past the script")))
            },
            read_load: quiet_load,
            read_touch_pointers: no_pointers_read,
        };
        let got: Vec<Option<String>> = want
            .iter()
            .map(|_| probe.poll().ok().map(|p| p.summary))
            .collect();
        let want: Vec<Option<String>> = want.iter().map(|w| w.map(str::to_string)).collect();
        assert_eq!(got, want);
        assert_eq!(reads.get(), want_thermal_reads);
    }

    #[test]
    fn three_complete_skinless_polls_fail_but_incomplete_or_failed_polls_reset_them()
    -> anyhow::Result<()> {
        let config = table(ReadinessPlatform::Android, "skin_below_c = 34")?;
        let skinless = ThermalReading {
            skin_c: None,
            skin_block_complete: true,
            ..COOL
        };
        let partial = ThermalReading {
            skin_c: Some(33.0),
            skin_block_complete: false,
            ..skinless
        };
        let mut script = [
            Ok(skinless),
            Ok(partial),
            Ok(skinless),
            thermal_failed(),
            Ok(skinless),
            Ok(skinless),
            Ok(skinless),
        ]
        .into_iter();
        let mut probe = AndroidProbe {
            bounds: Bounds::new(Some(&config)),
            thermal: ThermalGate::Enforce,
            on_held_touch: HeldTouch::Wait,
            skipped_unreadable: false,
            missing_skin_polls: 0,
            held_touch_polls: 0,
            warned_unchecked: false,
            read_thermal: || {
                script
                    .next()
                    .ok_or(ReadinessError::Unavailable("read past the script"))?
            },
            read_load: quiet_load,
            read_touch_pointers: no_pointers_read,
        };
        assert!(!probe.poll()?.ready);
        assert!(matches!(probe.poll(), Err(ReadinessError::Unavailable(_))));
        assert!(!probe.poll()?.ready);
        assert!(matches!(probe.poll(), Err(ReadinessError::Unavailable(_))));
        assert!(!probe.poll()?.ready);
        assert!(!probe.poll()?.ready);
        assert!(matches!(
            probe.poll(),
            Err(ReadinessError::BoundWithoutSkinSensor)
        ));
        Ok(())
    }

    /// If sysfs reads, the function uses its value. If sysfs does not read,
    /// the function uses `dumpsys`. If `dumpsys` has no temperature either, the
    /// error names the `dumpsys` fallback, not the sysfs refusal.
    #[rstest]
    #[case::sysfs_reads(Some(40), DUMPSYS_TEMPS, Some(40))]
    #[case::dumpsys_answers(None, DUMPSYS_TEMPS, Some(35))]
    #[case::neither_answers(None, "Thermal Status: 0\nHAL Ready: true\n", None)]
    fn hottest_cpu_prefers_sysfs_then_dumpsys(
        #[case] sysfs: Option<i32>,
        #[case] dumpsys: &str,
        #[case] want: Option<i32>,
    ) -> anyhow::Result<()> {
        let sysfs = sysfs.ok_or_else(|| {
            ReadinessError::Io(std::io::Error::from(std::io::ErrorKind::PermissionDenied))
        });
        match (hottest_cpu_celsius(sysfs, dumpsys), want) {
            (Ok(got), Some(want)) => assert_eq!(got, want),
            (Err(err), None) => assert!(
                matches!(err, ReadinessError::Unavailable(NO_DUMPSYS_CPU_TEMPERATURE)),
                "got {err:?}",
            ),
            (got, want) => anyhow::bail!("got {got:?}, want {want:?}"),
        }
        Ok(())
    }

    #[test]
    fn dumpsys_no_live_cpu_temperature_is_none() {
        let out = "\
Cached temperatures:
\tTemperature{mValue=82.0, mType=0, mName=BIG, mStatus=0}
";
        assert_eq!(parse_hottest_cpu_celsius_from_dumpsys(out), None);
    }

    #[rstest]
    #[case::pixel(
        "Cached temperatures:\nTemperature{mValue=99, mType=3, mName=VIRTUAL-SKIN, mStatus=0}\nCurrent temperatures from HAL:\nTemperature{mValue=33, mType=0, mName=MID, mStatus=0}\nTemperature{mValue=35.2, mType=3, mName=VIRTUAL-SKIN, mStatus=0}\nTemperature{mValue=34.1, mType=3, mName=SKIN-2, mStatus=0}\nCurrent cooling devices from HAL:\n",
        Some(35.2),
        true
    )]
    #[case::galaxy(
        "Current temperatures from HAL:\nTemperature{mValue=32.7, mType=3, mName=SKIN, mStatus=0}\nCurrent cooling devices from HAL:\n",
        Some(32.7),
        true
    )]
    #[case::no_skin(
        "Current temperatures from HAL:\nTemperature{mValue=33, mType=0, mName=CPU, mStatus=0}\nCurrent cooling devices from HAL:\n",
        None,
        true
    )]
    #[case::empty_complete_block(
        "Current temperatures from HAL:\nCurrent cooling devices from HAL:\n",
        None,
        true
    )]
    #[case::sentinel(
        "Current temperatures from HAL:\nTemperature{mValue=33, mType=0, mName=CPU, mStatus=0}\nTemperature{mValue=-3.4028235E38, mType=3, mName=SKIN, mStatus=0}\nCurrent cooling devices from HAL:\n",
        None,
        true
    )]
    #[case::partial(
        "Current temperatures from HAL:\nTemperature{mValue=33, mType=0, mName=CPU, mStatus=0}\n",
        None,
        false
    )]
    #[case::partial_with_skin(
        "Current temperatures from HAL:\nTemperature{mValue=33, mType=3, mName=SKIN, mStatus=0}\n",
        Some(33.0),
        false
    )]
    #[case::cached_only(
        "Cached temperatures:\nTemperature{mValue=99, mType=3, mName=SKIN, mStatus=0}\n",
        None,
        false
    )]
    fn skin_parser_uses_the_live_hal_block(
        #[case] dump: &str,
        #[case] expected: Option<f64>,
        #[case] complete: bool,
    ) {
        assert_eq!(parse_live_skin_celsius(dump), (expected, complete));
    }

    #[rstest]
    #[case("cpu  100 0 50 800 50 0 0 0 0 0\nintr ...\n", 850, 1000)]
    #[case("cpu 1 2 3 4 5\n", 9, 15)]
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
    #[case(CpuTimes { idle: 5000, total: 10_000 }, CpuTimes { idle: 5700, total: 11_000 }, 0.30)]
    #[case(CpuTimes { idle: 5000, total: 10_000 }, CpuTimes { idle: 6000, total: 11_000 }, 0.0)]
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
    #[case("cpu-0-0-0", true)]
    #[case("cpu-1-1-1", true)]
    #[case("cpullc-0-0", true)]
    #[case("cpullc-1-1", true)]
    #[case("cpu-hw-trip-0", false)]
    #[case("cpu-hw-trip-1", false)]
    #[case("cpu-anything-else", false)]
    #[case("cpullc-anything-else", false)]
    #[case("gpuss-0", false)]
    #[case("nsphvx-2", false)]
    #[case("qmx-0-0", false)]
    #[case("battery", false)]
    #[case("ddr", false)]
    #[case("camera-0", false)]
    fn cpu_zone_predicate_filters_to_cpu_clusters(#[case] zone_type: &str, #[case] want: bool) {
        assert_eq!(is_cpu_zone(zone_type), want);
    }

    // Real `/proc/interrupts` lines: Pixel 11 Pro (7 CPUs) and Galaxy S26
    // Ultra (8 CPUs, with its S Pen lines).
    const PIXEL_INTERRUPTS: &str = "\
           CPU0       CPU1       CPU2       CPU3       CPU4       CPU5       CPU6
755:     202082        149          0          0          0          0          0  e500000.pinctrl  54 Level     synaptics_tcm
IPI0:  73448106   75984895   17736072   13080999   15534692   23430840    2916460       Rescheduling interrupts
";
    const S26_INTERRUPTS: &str = "\
           CPU0       CPU1       CPU2       CPU3       CPU4       CPU5       CPU6       CPU7
363:     108060      63287          0          0          0          0          0          0   msmgpio  15 Level     sec_epen_irq
372:     165351      31568          0          0          0          0          0          0   msmgpio 158 Level     synaptics_ts
IPI0:  12878039    5452976     955824    1005415    6515703    8052324     338467     220269       Rescheduling interrupts
";

    #[rstest]
    #[case::pixel(PIXEL_INTERRUPTS, TouchIrqCount::Count(202_231))]
    #[case::s26_skips_the_pen(S26_INTERRUPTS, TouchIrqCount::Count(196_919))]
    #[case::no_touch_line(
        "IPI0:  73448106   75984895       Rescheduling interrupts\n",
        TouchIrqCount::Unchecked
    )]
    // A name that only contains a known one is another device.
    #[case::other_device(
        "12:  5  5  msmgpio 1 Level  synaptics_ts_wake\n",
        TouchIrqCount::Unchecked
    )]
    #[case::counter_sum_overflows(
        "CPU0 CPU1\n12:  18446744073709551615  1  msmgpio 1 Level synaptics_ts\n",
        TouchIrqCount::Count(u64::MAX)
    )]
    #[case::missing_count(
        "CPU0 CPU1\n12:  msmgpio 1 Level synaptics_ts\n",
        TouchIrqCount::Malformed
    )]
    #[case::malformed_count(
        "CPU0 CPU1\n12:  12 bad msmgpio 1 Level synaptics_ts\n",
        TouchIrqCount::Malformed
    )]
    fn touch_irq_count_sums_the_touch_line(#[case] raw: &str, #[case] want: TouchIrqCount) {
        assert_eq!(parse_touch_irq_count(raw), want);
    }

    #[rstest]
    #[case::one_second(100, 300, 1.0, 200.0, false)]
    #[case::longer_window(100, 300, 2.0, 100.0, false)]
    // An offline CPU drops its column, so the sum can fall.
    #[case::cpu_went_offline(300, 100, 1.0, 0.0, true)]
    fn touch_irq_rate_is_per_second(
        #[case] first: u64,
        #[case] second: u64,
        #[case] secs: f64,
        #[case] want: f64,
        #[case] want_decreased: bool,
    ) {
        let (got, decreased) = touch_irq_rate(first, second, secs);
        assert!((got - want).abs() < 1e-9, "got {got}, want {want}");
        assert_eq!(decreased, want_decreased);
    }

    /// This sample shows the `dumpsys input` format on a Galaxy S26 Ultra.
    /// The S Pen has a `POINTER` mapper and must not count as a touchscreen.
    fn s26_input_dump(touchscreen: u32, pen: u32) -> String {
        format!(
            "\
INPUT MANAGER (dumpsys input)
Input Reader State (Nums of device: 9):
  Device 4: sec_touchscreen
    Sources: KEYBOARD | TOUCHSCREEN
    Touch Input Mapper (mode - DIRECT):
      Parameters:
        DeviceType: TOUCH_SCREEN
      Last Raw Touch:
        Raw pointer data:
      Last Cooked Button State: 0x00000000
      Last Cooked Touch: pointerCount={touchscreen}
  Device 9: sec_e-pen
    Sources: KEYBOARD | TOUCHSCREEN | STYLUS
    Touch Input Mapper (mode - POINTER):
      Parameters:
        DeviceType: POINTER
      Last Cooked Touch: pointerCount={pen}
  Device -1: Virtual
    Sources: KEYBOARD
"
        )
    }

    #[rstest]
    #[case::idle(s26_input_dump(0, 0), Some(0))]
    #[case::held(s26_input_dump(1, 0), Some(1))]
    #[case::two_fingers(s26_input_dump(2, 0), Some(2))]
    #[case::hovering_pen(s26_input_dump(0, 1), Some(0))]
    #[case::no_touchscreen(
        "  Device 2: gpio-keys\n    Sources: KEYBOARD\n      Last Cooked Touch: pointerCount=1\n"
            .to_owned(),
        None
    )]
    fn touch_pointers_come_from_touchscreens_only(#[case] dump: String, #[case] want: Option<u32>) {
        assert_eq!(parse_touch_pointers(&dump), want);
    }

    #[rstest]
    #[case::quiet(QUIET_TOUCH, "touch=irq:0/s(ok)")]
    #[case::storm_without_pointer(
        TouchReading::Read { irq_per_sec: 212.4, pointers: Some(0) },
        "touch=irq:212/s,pointers:0(ok)"
    )]
    #[case::held(
        TouchReading::Read { irq_per_sec: 212.4, pointers: Some(1) },
        "touch=irq:212/s,pointers:1(held)"
    )]
    #[case::unchecked(TouchReading::Unchecked, "touch=unchecked")]
    #[case::disabled(TouchReading::Disabled, "touch=off")]
    fn summary_shows_the_touch_stage(#[case] touch: TouchReading, #[case] want: &str) {
        let summary = format_summary(
            Some(COOL),
            0.05,
            touch,
            ThermalGate::Enforce,
            Bounds::new(None),
        );
        assert!(summary.ends_with(want), "{summary}");
    }

    #[derive(Clone, Copy)]
    struct TouchPoll {
        irq_per_sec: Option<f64>,
        count_decreased: bool,
        count_malformed: bool,
        pointers: u32,
    }

    const QUIET: TouchPoll = TouchPoll {
        irq_per_sec: Some(0.0),
        count_decreased: false,
        count_malformed: false,
        pointers: 0,
    };
    const HELD: TouchPoll = TouchPoll {
        irq_per_sec: Some(200.0),
        count_decreased: false,
        count_malformed: false,
        pointers: 1,
    };
    const STORM: TouchPoll = TouchPoll {
        irq_per_sec: Some(200.0),
        count_decreased: false,
        count_malformed: false,
        pointers: 0,
    };
    const NO_TOUCH_LINE: TouchPoll = TouchPoll {
        irq_per_sec: None,
        count_decreased: false,
        count_malformed: false,
        pointers: 1,
    };
    const CPU_OFFLINE_WITH_TOUCH: TouchPoll = TouchPoll {
        irq_per_sec: Some(0.0),
        count_decreased: true,
        count_malformed: false,
        pointers: 1,
    };
    const MALFORMED_COUNT_WITH_TOUCH: TouchPoll = TouchPoll {
        irq_per_sec: Some(0.0),
        count_decreased: false,
        count_malformed: true,
        pointers: 1,
    };

    #[derive(Debug, PartialEq)]
    enum Polled {
        Ready,
        Held,
        HeldTouchError,
        /// An error the case does not expect, shown in the assertion.
        Failed(String),
    }
    use Polled::{Held, HeldTouchError, Ready};

    /// Each case polls once per script entry. `want_dumpsys` counts pointer
    /// reads triggered by a high rate, decreasing count, or malformed count.
    #[rstest]
    #[case::quiet_phone(HeldTouch::Wait, ThermalGate::Enforce, vec![QUIET], vec![Ready], 0)]
    #[case::wait_holds_while_held(
        HeldTouch::Wait,
        ThermalGate::Enforce,
        vec![HELD, HELD, HELD, HELD, QUIET],
        vec![Held, Held, Held, Held, Ready],
        4
    )]
    #[case::fail_after_three_held_polls(
        HeldTouch::Fail,
        ThermalGate::Enforce,
        vec![HELD, HELD, HELD],
        vec![Held, Held, HeldTouchError],
        3
    )]
    #[case::a_clear_poll_resets_the_count(
        HeldTouch::Fail,
        ThermalGate::Enforce,
        vec![HELD, HELD, QUIET, HELD, HELD, HELD],
        vec![Held, Held, Ready, Held, Held, HeldTouchError],
        5
    )]
    #[case::a_storm_without_a_pointer_is_ready(
        HeldTouch::Fail,
        ThermalGate::Enforce,
        vec![STORM, STORM, STORM, STORM],
        vec![Ready, Ready, Ready, Ready],
        4
    )]
    #[case::no_touch_line_is_unchecked(
        HeldTouch::Fail,
        ThermalGate::Enforce,
        vec![NO_TOUCH_LINE, NO_TOUCH_LINE, NO_TOUCH_LINE],
        vec![Ready, Ready, Ready],
        0
    )]
    #[case::skipped_thermal_still_holds(
        HeldTouch::Wait,
        ThermalGate::Skip,
        vec![HELD],
        vec![Held],
        1
    )]
    #[case::decreasing_irq_count_checks_pointers(
        HeldTouch::Wait,
        ThermalGate::Enforce,
        vec![CPU_OFFLINE_WITH_TOUCH],
        vec![Held],
        1
    )]
    #[case::malformed_irq_count_checks_pointers(
        HeldTouch::Wait,
        ThermalGate::Enforce,
        vec![MALFORMED_COUNT_WITH_TOUCH],
        vec![Held],
        1
    )]
    fn a_held_touch_holds_or_fails_the_gate(
        #[case] on_held_touch: HeldTouch,
        #[case] thermal: ThermalGate,
        #[case] script: Vec<TouchPoll>,
        #[case] want: Vec<Polled>,
        #[case] want_dumpsys: usize,
    ) {
        let current = Cell::new(QUIET);
        let dumpsys_reads = Cell::new(0usize);
        let mut probe = AndroidProbe {
            bounds: Bounds::new(None),
            thermal,
            on_held_touch,
            skipped_unreadable: false,
            missing_skin_polls: 0,
            held_touch_polls: 0,
            warned_unchecked: false,
            read_thermal: || Ok(COOL),
            read_load: |_| {
                Ok(LoadSample {
                    cpu_load: 0.05,
                    touch_irq_per_sec: current.get().irq_per_sec,
                    touch_irq_count_decreased: current.get().count_decreased,
                    touch_irq_count_malformed: current.get().count_malformed,
                })
            },
            read_touch_pointers: || {
                dumpsys_reads.set(dumpsys_reads.get() + 1);
                Ok(current.get().pointers)
            },
        };
        let got: Vec<Polled> = script
            .into_iter()
            .map(|poll| {
                current.set(poll);
                match probe.poll() {
                    Ok(polled) if polled.ready => Ready,
                    Ok(_) => Held,
                    Err(ReadinessError::HeldTouch { .. }) => HeldTouchError,
                    Err(err) => Polled::Failed(err.to_string()),
                }
            })
            .collect();
        assert_eq!(got, want);
        assert_eq!(dumpsys_reads.get(), want_dumpsys);
    }

    #[test]
    fn off_skips_touch_reads_and_keeps_cpu_gate() -> Result<(), ReadinessError> {
        let cpu_load = Cell::new(0.5);
        let mut probe = AndroidProbe {
            bounds: Bounds::new(None),
            thermal: ThermalGate::Enforce,
            on_held_touch: HeldTouch::Off,
            skipped_unreadable: false,
            missing_skin_polls: 0,
            held_touch_polls: 0,
            warned_unchecked: false,
            read_thermal: || Ok(COOL),
            read_load: |check_touch: bool| {
                assert!(!check_touch, "disabled touch gate must not read interrupts");
                Ok(LoadSample {
                    cpu_load: cpu_load.get(),
                    touch_irq_per_sec: Some(200.0),
                    touch_irq_count_decreased: false,
                    touch_irq_count_malformed: false,
                })
            },
            read_touch_pointers: || Err(ReadinessError::Unavailable("pointer read attempted")),
        };
        let busy = probe.poll()?;
        assert!(!busy.ready);
        assert!(busy.summary.contains("touch=off"));
        cpu_load.set(0.05);
        assert!(probe.poll()?.ready);
        Ok(())
    }

    #[test]
    fn a_failed_dumpsys_input_read_fails_the_poll() {
        let mut probe = AndroidProbe {
            bounds: Bounds::new(None),
            thermal: ThermalGate::Enforce,
            on_held_touch: HeldTouch::Wait,
            skipped_unreadable: false,
            missing_skin_polls: 0,
            held_touch_polls: 0,
            warned_unchecked: false,
            read_thermal: || Ok(COOL),
            read_load: |_| {
                Ok(LoadSample {
                    cpu_load: 0.05,
                    touch_irq_per_sec: Some(200.0),
                    touch_irq_count_decreased: false,
                    touch_irq_count_malformed: false,
                })
            },
            read_touch_pointers: no_pointers_read,
        };
        assert!(matches!(probe.poll(), Err(ReadinessError::Unavailable(_))));
    }

    #[test]
    fn a_failed_poll_breaks_the_held_touch_streak() -> Result<(), ReadinessError> {
        let poll_number = Cell::new(0);
        let mut probe = AndroidProbe {
            bounds: Bounds::new(None),
            thermal: ThermalGate::Enforce,
            on_held_touch: HeldTouch::Fail,
            skipped_unreadable: false,
            missing_skin_polls: 0,
            held_touch_polls: 0,
            warned_unchecked: false,
            read_thermal: || Ok(COOL),
            read_load: |_| {
                let next = poll_number.get() + 1;
                poll_number.set(next);
                if next == 3 {
                    return Err(ReadinessError::Unavailable("load sample failed"));
                }
                Ok(LoadSample {
                    cpu_load: 0.05,
                    touch_irq_per_sec: Some(200.0),
                    touch_irq_count_decreased: false,
                    touch_irq_count_malformed: false,
                })
            },
            read_touch_pointers: || Ok(1),
        };

        assert!(!probe.poll()?.ready);
        assert!(!probe.poll()?.ready);
        assert!(matches!(probe.poll(), Err(ReadinessError::Unavailable(_))));
        assert!(!probe.poll()?.ready);
        assert_eq!(probe.held_touch_polls, 1);
        Ok(())
    }
}
