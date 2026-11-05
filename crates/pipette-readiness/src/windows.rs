//! Windows readiness: gates on CPU busy share, GPU compute, the throttle-reason
//! flags and the platform's critical trip, plus any sensor the host's
//! `[readiness.windows]` table holds to a ceiling. Only CPU load is mandatory;
//! the rest gate only where the counter exists. CPU load comes from
//! `GetSystemTimes`, GPU temperature from `D3DKMTQueryAdapterInfo`, and the rest
//! from one PowerShell invocation (~500 ms of interpreter start-up).
//!
//! # No relative temperature test
//!
//! Waiting for the hottest zone to stop moving (measured 2026-09-30) changed no
//! GMKtec cell by more than 0.2 %, took 3.4x as long, and released the Zenbook
//! S 16 with its firmware power cut still on (next prefill 25 % slow).
//!
//! # Ceilings
//!
//! The Zenbook S 16 cuts socket power from 28 W toward 17 W once THZ0 reaches
//! 68 °C; held to 43 °C, the next prefill ran at full speed. A sensor still over
//! its ceiling after [`CEILING_MAX_WAIT`] is let through with a warning, so a
//! warm room costs time, not cells.
//!
//! # No sensor outranks another
//!
//! Which sensor works depends on the box: ESIF is absent on the AMD boxes, the
//! 258V's ACPI zone reads exactly 301.00 K whatever the load, and the SER9's
//! only zone is stuck at 293.2 K, leaving its GPU sensor. Ranking them would
//! mean deciding per chassis, wrongly on hardware nobody has measured. Only a
//! ceiling or the critical trip holds the gate on temperature; stuck sensors
//! are reported.
//!
//! The AMD zone is a real junction temperature, tracking LibreHardwareMonitor's
//! `Tctl/Tdie` within 1-4 °C; the misreporting sensor that
//! `docs/fleet-perf-troubleshooting/gmktec-evo-x2.md` warns about is
//! `ryzenadj`'s `THM VALUE CORE` (42.7 °C with the die at 98 °C).
//!
//! The gate holds at [`CRITICAL_MARGIN_C`] below `CriticalTripPoint`, a
//! hardware-declared trip that means the same on every chassis that reports
//! one ([`CRITICAL_FALLBACK_C`] otherwise); `PassiveTripPoint` reads 0 on the
//! measured boxes.
//!
//! # Known gaps
//!
//! Throttle reasons read 0 on every measured box, even with the gmktec die at
//! 98 °C or the Zenbook held at 17 W; kept because a 0 costs nothing. Nothing
//! catches an OpenVINO NPU graph compile (~90 % of an NPU cell's wall clock):
//! GPU compute reads 0 % and the die is cold.

use std::time::{Duration, Instant};

use super::wait::on_probe_failure;
use super::{
    CeilingCelsius, CpuPercent, ReadinessCallbacks, ReadinessError, ReadinessPhase,
    ReadinessPlatform, ReadinessStatus, ThermalGate, WindowsReadinessConfiguration, WindowsSensor,
};

#[cfg(target_os = "windows")]
pub(super) const DEFAULT_MAX_WAIT: Duration = super::DEFAULT_MAX_WAIT;
/// A load-only gate releases on its first poll, so this only paces a box still
/// busy or a zone still over its ceiling.
const POLL_INTERVAL: Duration = Duration::from_secs(5);
/// Highest acceptable busy share of all logical CPUs, unless
/// [`WindowsReadinessConfiguration::max_cpu_busy_pct`] sets one.
///
/// Idle reads 0-1 % on a gmktec and the HX 370; half the logical CPUs spinning
/// reads 47-54 %. Not `Win32_Processor.LoadPercentage`, which read 1-19 % on the
/// 258V under a 16-job burn and alternated 100 / 3 / 100 / 90 on a steady
/// gmktec burn.
const CPU_LOAD_THRESHOLD_PCT: u32 = 40;
/// The 1 s [`CPU_LOAD_THRESHOLD_PCT`] was calibrated over.
#[cfg(target_os = "windows")]
const CPU_SAMPLE_WINDOW: Duration = Duration::from_secs(1);
/// Highest acceptable summed GPU-compute utilization. Idle reads exactly 0 and
/// Vulkan inference 71-105 %, with nothing observed in between.
const GPU_COMPUTE_THRESHOLD_PCT: u32 = 5;
/// Margin below `CriticalTripPoint` (a shutdown threshold, 105-110 °C
/// measured), °C: enough to keep clear of runaway, not of normal operation.
const CRITICAL_MARGIN_C: u32 = 15;
/// Critical limit, °C, for a box that reports no trip point.
const CRITICAL_FALLBACK_C: u32 = 95;
/// Cap on waiting for a sensor to reach its ceiling. The Zenbook's power came
/// back 133-137 s after a 6-minute run with the box idle, and ~163 s when it
/// kept working.
const CEILING_MAX_WAIT: Duration = Duration::from_secs(240);

#[derive(Debug, Clone)]
struct WindowsReading {
    cpu_load_pct: u32,
    gpu_compute_pct: Option<u32>,
    throttle_reasons: Option<u32>,
    /// Every ACPI thermal zone by name, °C (the counter reports Kelvin).
    zones: Vec<ZoneReading>,
    /// Intel Dynamic Tuning (DPTF/ESIF), already °C.
    esif_temp_c: Option<u32>,
    /// The GPU's own sensor, °C.
    gpu_temp_c: Option<u32>,
    /// ACPI `CriticalTripPoint`, °C (the class reports deci-Kelvin).
    critical_c: Option<u32>,
}

/// One ACPI thermal zone, named as its instance ends: `THZ0` for `\_TZ.THZ0`.
#[derive(Debug, Clone, PartialEq, Eq)]
struct ZoneReading {
    name: String,
    temp_c: u32,
}

/// What one poll says about a configured sensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SensorState {
    At(u32),
    /// No reading this poll: the zone read failed, or the GPU did not answer.
    /// Unknown, not cold, so it holds the gate like a hot reading.
    Unread,
    /// The zones read, and this one is not among them.
    Absent,
}

fn sensor_state(reading: &WindowsReading, sensor: &WindowsSensor) -> SensorState {
    match sensor {
        WindowsSensor::AcpiZone(_) if reading.zones.is_empty() => SensorState::Unread,
        WindowsSensor::AcpiZone(zone) => reading
            .zones
            .iter()
            .find(|z| z.name.eq_ignore_ascii_case(zone))
            .map_or(SensorState::Absent, |z| SensorState::At(z.temp_c)),
        WindowsSensor::Gpu => reading
            .gpu_temp_c
            .map_or(SensorState::Unread, SensorState::At),
    }
}

fn ceilings_met(reading: &WindowsReading, config: &WindowsReadinessConfiguration) -> bool {
    config.ceilings.iter().all(|(sensor, max_c)| {
        matches!(sensor_state(reading, sensor), SensorState::At(c) if c <= max_c.get())
    })
}

/// The first ceiling naming a zone this poll's zone list does not include.
fn absent_sensor(
    reading: &WindowsReading,
    config: &WindowsReadinessConfiguration,
) -> Option<ReadinessError> {
    config
        .ceilings
        .keys()
        .find(|sensor| sensor_state(reading, sensor) == SensorState::Absent)
        .map(|sensor| ReadinessError::UnknownSensor {
            sensor: sensor.clone(),
            available: reading
                .zones
                .iter()
                .map(|z| format!("acpi:{}", z.name))
                .chain(reading.gpu_temp_c.map(|_| "gpu".to_string()))
                .collect::<Vec<_>>()
                .join(", "),
        })
}

/// A temperature counter's log label and its reading.
type TempSource = (&'static str, fn(&WindowsReading) -> Option<u32>);

/// What the critical and stuck-sensor checks read. Order is cosmetic.
const TEMP_SOURCES: [TempSource; 3] = [
    ("esif", |r| r.esif_temp_c),
    ("acpi", |r| r.zone_temp_c()),
    ("gpu", |r| r.gpu_temp_c),
];

impl WindowsReading {
    /// The hottest ACPI thermal zone, °C.
    fn zone_temp_c(&self) -> Option<u32> {
        self.zones.iter().map(|z| z.temp_c).max()
    }

    fn temperatures(&self) -> impl Iterator<Item = (&'static str, u32)> + '_ {
        TEMP_SOURCES
            .into_iter()
            .filter_map(|(name, read)| read(self).map(|c| (name, c)))
    }
}

fn over_critical(reading: &WindowsReading) -> bool {
    let limit = reading
        .critical_c
        .map_or(CRITICAL_FALLBACK_C, |c| c.saturating_sub(CRITICAL_MARGIN_C));
    reading.temperatures().any(|(_, c)| c >= limit)
}

/// Polls needed before a never-moving temperature is a series rather than a
/// coincidence. Says nothing about whether the box was ever busy, which
/// [`held_temperature_if_stuck`] checks separately.
const CONSTANT_TEMP_MIN_POLLS: usize = 3;

#[cfg(target_os = "windows")]
pub(super) fn wait_until_ready(
    max_wait: Duration,
    thermal: ThermalGate,
    config: Option<&WindowsReadinessConfiguration>,
    callbacks: &ReadinessCallbacks<'_>,
) -> Result<(), ReadinessError> {
    let default = WindowsReadinessConfiguration::default();
    wait_for(
        max_wait,
        thermal,
        config.unwrap_or(&default),
        &mut read_status,
        callbacks,
    )
}

/// The gate over an injected probe, so its tests run off Windows. Not
/// [`crate::wait::wait_loop`]: this keeps the poll series for the stuck-sensor
/// report and holds ceilings to their own budget.
fn wait_for(
    max_wait: Duration,
    thermal: ThermalGate,
    config: &WindowsReadinessConfiguration,
    probe: &mut dyn FnMut() -> Result<WindowsReading, ReadinessError>,
    callbacks: &ReadinessCallbacks<'_>,
) -> Result<(), ReadinessError> {
    let started = Instant::now();
    let deadline = started + max_wait;
    // Before the first probe, so a caller that gave up skips a PowerShell start.
    if callbacks.is_cancelled() {
        return Err(ReadinessError::Cancelled);
    }
    let status = |phase, observed: &str| ReadinessStatus {
        phase,
        platform: ReadinessPlatform::Windows,
        observed: observed.to_string(),
        elapsed: started.elapsed(),
        max_wait,
    };
    // The whole series: the stuck-sensor report judges it, and a ceiling's
    // sensor that never read in it fails the wait at the ceiling budget.
    let mut samples: Vec<WindowsReading> = Vec::new();
    let cpu_max = config
        .max_cpu_busy_pct
        .map_or(CPU_LOAD_THRESHOLD_PCT, CpuPercent::get);
    // Never outlives the caller's own deadline: a short `max_wait` is a
    // caller saying the whole wait is worth less than this cap.
    let ceiling_cap = CEILING_MAX_WAIT.min(max_wait);
    let skip = thermal == ThermalGate::Skip;
    let mut failed_probes = 0usize;
    loop {
        let probed = probe();
        callbacks.polled();
        let reading = match probed {
            Ok(reading) => {
                failed_probes = 0;
                reading
            }
            // Returned rather than a `TimedOut`, since it names what broke.
            Err(err) => {
                failed_probes += 1;
                let platform = ReadinessPlatform::Windows;
                if on_probe_failure(platform, &err, failed_probes, deadline, POLL_INTERVAL)
                    .is_break()
                {
                    return Err(err);
                }
                if callbacks.sleep_cancellably(POLL_INTERVAL).is_break() {
                    return Err(ReadinessError::Cancelled);
                }
                continue;
            }
        };
        // A cancel can arrive during the seconds a probe spends in PowerShell.
        if callbacks.is_cancelled() {
            return Err(ReadinessError::Cancelled);
        }
        // A refused configuration is not a timeout: no terminal status.
        if !skip && let Some(err) = absent_sensor(&reading, config) {
            return Err(err);
        }
        samples.push(reading.clone());
        let cpu_ok = reading.cpu_load_pct <= cpu_max;
        let gpu_ok = reading
            .gpu_compute_pct
            .is_none_or(|v| v < GPU_COMPUTE_THRESHOLD_PCT);
        let throttle_ok = skip || reading.throttle_reasons.is_none_or(|v| v == 0);
        let critical = !skip && over_critical(&reading);
        let cool_zones = skip || ceilings_met(&reading, config);
        // Its own budget: a busy box may free up within `max_wait`, but a zone
        // in a warm room may never reach its ceiling.
        let ceiling_expired = started.elapsed() >= ceiling_cap;
        if !cool_zones
            && ceiling_expired
            && let Some(sensor) = config.ceilings.keys().find(|sensor| {
                !samples
                    .iter()
                    .any(|r| matches!(sensor_state(r, sensor), SensorState::At(_)))
            })
        {
            return Err(ReadinessError::SensorUnread {
                sensor: sensor.clone(),
                waited: ceiling_cap,
            });
        }
        let summary = with_ceilings(format_summary(&reading, cpu_max), &reading, config);
        // Terminal status before the warnings: a renderer erases its last frame
        // on that report, and would erase a warning written first.
        if cpu_ok && gpu_ok && throttle_ok && !critical && (cool_zones || ceiling_expired) {
            callbacks.report(&status(ReadinessPhase::Proceeding, &summary));
            warn_about_stuck_sensors(&samples, cpu_max);
            if !cool_zones {
                tracing::warn!(
                    budget = ?ceiling_cap,
                    observed = summary.as_str(),
                    "a ceiling's sensor was still over it, or unread, when the budget ran \
                     out; proceeding anyway, so this cell may run under the firmware's power cut"
                );
            }
            return Ok(());
        }
        if Instant::now() >= deadline {
            callbacks.report(&status(ReadinessPhase::TimedOut, &summary));
            // A sensor stuck high times out every time, so say so here too.
            warn_about_stuck_sensors(&samples, cpu_max);
            return Err(ReadinessError::TimedOut {
                max_wait,
                observed: summary,
            });
        }
        callbacks.report(&status(
            ReadinessPhase::Waiting {
                poll_in: POLL_INTERVAL,
            },
            &summary,
        ));
        if callbacks.sleep_cancellably(POLL_INTERVAL).is_break() {
            return Err(ReadinessError::Cancelled);
        }
    }
}

/// The value a sensor held across the series while the load went from busy to
/// idle (or back). The 258V's ACPI zone reads exactly 301.00 K through idle,
/// 75 s of saturation and cooldown, as does `MSAcpi_ThermalZoneTemperature`.
fn held_temperature_if_stuck(
    temps: &[Option<u32>],
    loads: &[(u32, Option<u32>)],
    cpu_max: u32,
) -> Option<u32> {
    if temps.len() < CONSTANT_TEMP_MIN_POLLS {
        return None;
    }
    let first = temps.first().copied().flatten()?;
    let held = temps.iter().all(|t| *t == Some(first));
    // "Moved" means the busy verdict changed, not the number: idle CPU samples
    // on a 258V ran 3, 1, 1, 2, 1, 8, and an idle box holding steady is what a
    // working sensor reports.
    let load_moved =
        loads.iter().any(|l| is_busy(l, cpu_max)) && loads.iter().any(|l| !is_busy(l, cpu_max));
    (held && load_moved).then_some(first)
}

fn is_busy((cpu, gpu): &(u32, Option<u32>), cpu_max: u32) -> bool {
    *cpu > cpu_max || gpu.is_some_and(|v| v >= GPU_COMPUTE_THRESHOLD_PCT)
}

/// Diagnostic only, so a constant reading is not mistaken for a cold box.
/// Silence proves nothing: a box that settles before
/// [`CONSTANT_TEMP_MIN_POLLS`] never qualifies.
fn warn_about_stuck_sensors(samples: &[WindowsReading], cpu_max: u32) {
    let loads: Vec<(u32, Option<u32>)> = samples
        .iter()
        .map(|s| (s.cpu_load_pct, s.gpu_compute_pct))
        .collect();
    TEMP_SOURCES.into_iter().for_each(|(name, read)| {
        let series: Vec<Option<u32>> = samples.iter().map(read).collect();
        if let Some(held) = held_temperature_if_stuck(&series, &loads, cpu_max) {
            tracing::warn!(
                sensor = %name,
                temp_c = held,
                polls = samples.len(),
                "a sensor held one temperature while the load signals crossed their \
                 thresholds; probably a BIOS constant, so its reading measures nothing"
            );
        }
    });
}

/// Puts the ceilings first, where a terminal's 80 columns still show them.
fn with_ceilings(
    summary: String,
    reading: &WindowsReading,
    config: &WindowsReadinessConfiguration,
) -> String {
    if config.ceilings.is_empty() {
        return summary;
    }
    let parts: Vec<String> = config
        .ceilings
        .iter()
        .map(|(sensor, max_c): (&WindowsSensor, &CeilingCelsius)| {
            match sensor_state(reading, sensor) {
                SensorState::At(c) => format!(
                    "{sensor}={c}°C/{max_c}({})",
                    if c > max_c.get() { "hot" } else { "ok" }
                ),
                SensorState::Unread => format!("{sensor}=unread/{max_c}"),
                SensorState::Absent => format!("{sensor}=absent/{max_c}"),
            }
        })
        .collect();
    format!("ceiling={} {summary}", parts.join(","))
}

fn format_summary(r: &WindowsReading, cpu_max: u32) -> String {
    let cpu_verdict = if r.cpu_load_pct <= cpu_max {
        "ok"
    } else {
        "busy"
    };
    let gpu = match r.gpu_compute_pct {
        Some(v) => format!(
            "compute:{v}%({})",
            if v < GPU_COMPUTE_THRESHOLD_PCT {
                "ok"
            } else {
                "busy"
            }
        ),
        None => "n/a".to_string(),
    };
    let thermal = match r.throttle_reasons {
        Some(v) => format!("throttle:{v}({})", if v == 0 { "ok" } else { "throttling" }),
        None => "n/a".to_string(),
    };
    // Every sensor, under the names a ceiling uses, for whoever tunes one.
    let sensors: Vec<String> = r
        .esif_temp_c
        .map(|c| format!("esif={c}°C"))
        .into_iter()
        .chain(
            r.zones
                .iter()
                .map(|z| format!("acpi:{}={}°C", z.name, z.temp_c)),
        )
        .chain(r.gpu_temp_c.map(|c| format!("gpu={c}°C")))
        .collect();
    let temp = if sensors.is_empty() {
        "n/a".to_string()
    } else {
        sensors.join(",")
    };
    format!(
        "cpu=busy:{}%({cpu_verdict}) gpu={gpu} thermal={thermal} temp={temp}",
        r.cpu_load_pct,
    )
}

#[cfg(target_os = "windows")]
fn read_status() -> Result<WindowsReading, ReadinessError> {
    // Before PowerShell starts, so its start-up is not in the CPU window.
    let cpu_load_pct = read_cpu_load()?;
    let gpu_temp_c = pipette_device::gpu_temp_max_c();
    let mut cmd = std::process::Command::new("powershell");
    cmd.args([
        "-NoProfile",
        "-Command",
        // Prints `gpu_compute_pct throttle_reasons esif_temp_c critical_c
        // zones`, each `-` where the counter is absent; `zones` is
        // `NAME=C,...`. Instance names come back lowercased (`\_tz.thz0`), so
        // the tail after the last `.` is upper-cased to the `THZ0` a ceiling names.
        //
        // `try { measure } catch {}` with `-ErrorAction Stop`: the
        // equivalent `if (counter-present) { measure }` hangs PowerShell
        // silently on a real GMKTec box. The catches stay silent because
        // absence is normal (ESIF throws on every gmktec poll), and a
        // transient failure costs a poll: a ceiling's unread zone holds.
        //
        // GPU compute is a sum, since each instance is one engine of one
        // process. The throttle field is the `Maximum` across zones, enough
        // for the gate's `== 0` but not a faithful bitmask. Zone temperatures
        // are Kelvin; ESIF is already °C, and its sensorless instances read
        // exactly 0 (7 of 10 on a 258V), so zeros are dropped and an
        // all-zero box reads as absent rather than stone cold.
        // `CriticalTripPoint` is deci-Kelvin (3832 is 110 °C on the
        // gmktec) and 0 where unpopulated; `Minimum`, because the gate must
        // clear the earliest shutdown trip.
        //
        // Unverified: the absent-counter path for the GPU and zone signals.
        // ESIF's throws in 105-448 ms on first call and 0 ms after on the
        // gmktecs, rather than hanging on enumeration.
        "$gpuPct = '-';\
         try { \
           $m = (Get-Counter -Counter '\\GPU Engine(*engtype_compute*)\\Utilization Percentage' \
            -ErrorAction Stop).CounterSamples | \
            Where-Object { $_.CookedValue -gt 0 } | \
            Measure-Object -Property CookedValue -Sum;\
           $gpuPct = if ($m.Sum) { [int]$m.Sum } else { 0 } } catch {};\
         $throttleV = '-';\
         try { \
           $m = (Get-Counter -Counter '\\Thermal Zone Information(*)\\Throttle Reasons' \
            -ErrorAction Stop).CounterSamples | \
            Measure-Object -Property CookedValue -Maximum;\
           $throttleV = if ($null -eq $m.Maximum) { 0 } else { [int]$m.Maximum } } catch {};\
         $zones = '-';\
         try { \
           $z = (Get-Counter -Counter '\\Thermal Zone Information(*)\\Temperature' \
            -ErrorAction Stop).CounterSamples;\
           $n = @($z | ForEach-Object { \
            ($_.InstanceName -replace '^.*\\.', '').ToUpper() + '=' + \
            [int]($_.CookedValue - 273) });\
           if ($n.Count -gt 0) { $zones = $n -join ',' } } catch {};\
         $esifC = '-';\
         try { \
           $m = (Get-Counter -Counter '\\EsifDeviceInformation(*)\\Temperature' \
            -ErrorAction Stop).CounterSamples | \
            Where-Object { $_.CookedValue -gt 0 } | \
            Measure-Object -Property CookedValue -Maximum;\
           if ($null -ne $m.Maximum) { $esifC = [int]$m.Maximum } } catch {};\
         $critC = '-';\
         try { \
           $m = Get-CimInstance -Namespace root/wmi \
            -ClassName MSAcpi_ThermalZoneTemperature -ErrorAction Stop | \
            Where-Object { $_.CriticalTripPoint -gt 0 } | \
            Measure-Object -Property CriticalTripPoint -Minimum;\
           if ($null -ne $m.Minimum) { $critC = [int]($m.Minimum / 10 - 273) } } catch {};\
         \"$gpuPct $throttleV $esifC $critC $zones\"",
    ]);
    let output = crate::probe_output(cmd)?;
    // Every `catch {}` is silent, so anything here is unexpected.
    let reason = String::from_utf8_lossy(&output.stderr);
    let reason = reason.trim();
    if !reason.is_empty() {
        tracing::warn!(stderr = reason, "probe reported an error");
    }
    parse_windows_reading(
        cpu_load_pct,
        gpu_temp_c,
        String::from_utf8_lossy(&output.stdout).trim(),
    )
}

/// Cumulative CPU time over every logical CPU, in 100 ns units. `total` is
/// kernel plus user time, and kernel time includes idle time.
#[derive(Debug, Clone, Copy)]
struct CpuTimes {
    idle: u64,
    total: u64,
}

/// Busy share of all logical CPUs over [`CPU_SAMPLE_WINDOW`], in percent.
///
/// Not PDH's `% Processor Time` (or CIM over it): it credits idle time
/// unevenly, refusing 10-20 % of samples (`PDH_CALC_NEGATIVE_DENOMINATOR`) and
/// reading up to 47 % on an idle gmktec. Here idle and total share one
/// accounting.
#[cfg(target_os = "windows")]
fn read_cpu_load() -> Result<u32, ReadinessError> {
    let before = system_times()?;
    std::thread::sleep(CPU_SAMPLE_WINDOW);
    let after = system_times()?;
    busy_pct(before, after).ok_or(ReadinessError::Unavailable(
        "GetSystemTimes accounted no CPU time across the sample window",
    ))
}

#[cfg(target_os = "windows")]
fn system_times() -> Result<CpuTimes, ReadinessError> {
    use windows_sys::Win32::Foundation::FILETIME;
    use windows_sys::Win32::System::Threading::GetSystemTimes;

    let mut idle = FILETIME::default();
    let mut kernel = FILETIME::default();
    let mut user = FILETIME::default();
    // SAFETY: three `FILETIME`s we own; the call only writes them.
    let ok = unsafe { GetSystemTimes(&raw mut idle, &raw mut kernel, &raw mut user) };
    if ok == 0 {
        let err = std::io::Error::last_os_error();
        // The `Io` variant displays without its source, so name the call here.
        tracing::warn!(error = %err, "GetSystemTimes failed");
        return Err(err.into());
    }
    let ticks = |t: FILETIME| (u64::from(t.dwHighDateTime) << 32) | u64::from(t.dwLowDateTime);
    Ok(CpuTimes {
        idle: ticks(idle),
        total: ticks(kernel).saturating_add(ticks(user)),
    })
}

/// Rounded to the nearest point; `None` if no CPU time was accounted.
fn busy_pct(before: CpuTimes, after: CpuTimes) -> Option<u32> {
    let total = after.total.checked_sub(before.total).filter(|t| *t > 0)?;
    // Idle is part of kernel time, so it cannot outrun the total; `min` keeps
    // a torn pair of snapshots from underflowing anyway.
    let idle = after.idle.saturating_sub(before.idle).min(total);
    let busy = u128::from(total - idle);
    let total = u128::from(total);
    u32::try_from((busy * 100 + total / 2) / total).ok()
}

fn parse_windows_reading(
    cpu_load_pct: u32,
    gpu_temp_c: Option<u32>,
    line: &str,
) -> Result<WindowsReading, ReadinessError> {
    let mut parts = line.split_whitespace();
    let gpu_raw = parts.next().ok_or(ReadinessError::Unavailable(
        "empty powershell readiness output",
    ))?;
    let throttle_raw = parts.next().ok_or(ReadinessError::Unavailable(
        "missing throttle field in readiness output",
    ))?;
    // Trailing fields default to absent rather than failing every check.
    let esif_raw = parts.next().unwrap_or("-");
    let crit_raw = parts.next().unwrap_or("-");
    let zones_raw = parts.next().unwrap_or("-");
    Ok(WindowsReading {
        cpu_load_pct,
        gpu_compute_pct: parse_optional(gpu_raw)?,
        throttle_reasons: parse_optional(throttle_raw)?,
        zones: parse_zones(zones_raw)?,
        gpu_temp_c,
        // Backstop for the script's zero filter: a sensorless ESIF reads 0 °C.
        esif_temp_c: parse_optional(esif_raw)?.filter(|c| *c > 0),
        critical_c: parse_optional(crit_raw)?.filter(|c| *c > 0),
    })
}

/// `NAME=C,...`, or `-` for no zones. A zone below 0 °C is left out: an
/// unpopulated zone reads 0 K and would otherwise fail every poll.
fn parse_zones(raw: &str) -> Result<Vec<ZoneReading>, ReadinessError> {
    if raw == "-" {
        return Ok(Vec::new());
    }
    raw.split(',')
        .map(|entry| {
            let garbage = || ReadinessError::ParseFailure {
                raw: entry.to_string(),
            };
            let (name, temp) = entry.split_once('=').ok_or_else(garbage)?;
            if name.is_empty() {
                return Err(garbage());
            }
            let temp: i32 = temp.parse().map_err(|_| garbage())?;
            Ok(u32::try_from(temp).ok().map(|temp_c| ZoneReading {
                name: name.to_string(),
                temp_c,
            }))
        })
        .filter_map(Result::transpose)
        .collect()
}

/// `-` is an absent counter; garbage is a `ParseFailure`, not absent.
fn parse_optional(raw: &str) -> Result<Option<u32>, ReadinessError> {
    if raw == "-" {
        return Ok(None);
    }
    raw.parse()
        .map(Some)
        .map_err(|_| ReadinessError::ParseFailure {
            raw: raw.to_string(),
        })
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use rstest::rstest;

    use super::*;

    /// Builds a reading from `"cpu gpu throttle esif crit zones"`: the CPU load
    /// `read_cpu_load` measured, then the line PowerShell printed.
    fn reading(line: &str) -> Result<WindowsReading, ReadinessError> {
        let (cpu, powershell) = line.split_once(' ').unwrap_or((line, ""));
        let cpu = cpu.parse().map_err(|_| ReadinessError::ParseFailure {
            raw: cpu.to_string(),
        })?;
        parse_windows_reading(cpu, None, powershell)
    }

    /// Builds a series from `"cpu gpu throttle esif crit zones"` lines.
    fn series(lines: &[&str]) -> Result<Vec<WindowsReading>, ReadinessError> {
        lines.iter().map(|l| reading(l)).collect()
    }

    /// A poll whose probe failed: PowerShell printed nothing.
    fn probe_failed() -> ReadinessError {
        ReadinessError::Unavailable("empty powershell readiness output")
    }

    /// Builds a probe script from `"cpu gpu throttle esif crit zones"` lines,
    /// where `None` is a poll whose probe failed.
    fn scripted(
        steps: &[Option<&str>],
    ) -> Result<Vec<Result<WindowsReading, ReadinessError>>, ReadinessError> {
        steps
            .iter()
            .map(|step| match step {
                Some(line) => reading(line).map(Ok),
                None => Ok(Err(probe_failed())),
            })
            .collect()
    }

    fn no_sleep(_: Duration) {}

    /// Callbacks that spend no wall clock and print nothing.
    fn silent<'a>() -> ReadinessCallbacks<'a> {
        fn no_status(_: &ReadinessStatus) {}
        ReadinessCallbacks::new()
            .with_sleep(&no_sleep)
            .with_status(&no_status)
    }

    /// The allowance is per run of failures: a good poll restores it.
    #[rstest]
    #[case::one_bad_poll(&[None, None, Some("0 0 0 41 105 TZ01=40")])]
    // The busy reading keeps the gate polling, so the second run of failures
    // meets a clean count rather than one carried over from the first.
    #[case::two_runs_split_by_a_good_poll(
        &[None, None, Some("95 0 0 41 105 TZ01=40"), None, None, Some("0 0 0 41 105 TZ01=40")]
    )]
    fn the_gate_rides_out_transient_probe_failures(
        #[case] steps: &[Option<&str>],
    ) -> Result<(), Box<dyn std::error::Error>> {
        let polled = std::cell::Cell::new(0usize);
        let on_poll = || polled.set(polled.get() + 1);
        let mut results = scripted(steps)?.into_iter();
        wait_for(
            Duration::from_secs(300),
            ThermalGate::Skip,
            &WindowsReadinessConfiguration::default(),
            &mut || {
                results
                    .next()
                    .unwrap_or(Err(ReadinessError::Unavailable("probe script exhausted")))
            },
            &silent().with_poll(&on_poll),
        )?;
        assert_eq!(polled.get(), steps.len());
        Ok(())
    }

    #[test]
    fn a_probe_that_never_answers_still_fails_the_wait() -> Result<(), Box<dyn std::error::Error>> {
        let mut probes = 0usize;
        let polled = std::cell::Cell::new(0usize);
        let on_poll = || polled.set(polled.get() + 1);
        let err = wait_for(
            Duration::from_secs(300),
            ThermalGate::Skip,
            &WindowsReadinessConfiguration::default(),
            &mut || {
                probes += 1;
                // Answers past the allowance, so a regression fails the
                // assertion instead of spinning to the deadline.
                if probes > crate::PROBE_RETRY_POLLS * 2 {
                    return reading("0 0 0 41 105 TZ01=40");
                }
                Err(probe_failed())
            },
            &silent().with_poll(&on_poll),
        )
        .err()
        .ok_or("expected the wait to fail")?;
        assert!(matches!(err, ReadinessError::Unavailable(_)), "got {err:?}");
        assert_eq!(probes, crate::PROBE_RETRY_POLLS);
        assert_eq!(polled.get(), probes);
        Ok(())
    }

    #[test]
    fn a_spent_deadline_skips_the_retry() -> Result<(), Box<dyn std::error::Error>> {
        let mut probes = 0usize;
        wait_for(
            Duration::ZERO,
            ThermalGate::Skip,
            &WindowsReadinessConfiguration::default(),
            &mut || {
                probes += 1;
                Err(probe_failed())
            },
            &silent(),
        )
        .err()
        .ok_or("expected the wait to fail")?;
        assert_eq!(probes, 1);
        Ok(())
    }

    #[rstest]
    #[case::all_present(
        "0 0 0 41 105 TZ01=40",
        0,
        Some(0),
        Some(0),
        Some(40),
        Some(41),
        Some(105)
    )]
    #[case::all_present_busy(
        "62 0 4 72 110 TZ01=71",
        62,
        Some(0),
        Some(4),
        Some(71),
        Some(72),
        Some(110)
    )]
    #[case::no_gpu("5 - 0 - - TZ01=52", 5, None, Some(0), Some(52), None, None)]
    #[case::no_thermal_zones("3 91 - - - -", 3, Some(91), None, None, None, None)]
    #[case::cpu_only("4 - - - - -", 4, None, None, None, None, None)]
    #[case::no_trailing_fields("5 0 0", 5, Some(0), Some(0), None, None, None)]
    #[case::no_trip_point("5 0 0 44 - TZ01=52", 5, Some(0), Some(0), Some(52), Some(44), None)]
    // An unpopulated `CriticalTripPoint` reads 0, not a 0 °C trip.
    #[case::zero_trip_point_dropped(
        "5 0 0 44 0 TZ01=52",
        5,
        Some(0),
        Some(0),
        Some(52),
        Some(44),
        None
    )]
    #[case::hottest_of_several_zones(
        "2 0 0 - 103 THRM=48,THZ0=43,TZ01=20",
        2,
        Some(0),
        Some(0),
        Some(48),
        None,
        Some(103)
    )]
    fn windows_reading_parses(
        #[case] line: &str,
        #[case] want_cpu: u32,
        #[case] want_gpu: Option<u32>,
        #[case] want_throttle: Option<u32>,
        #[case] want_temp: Option<u32>,
        #[case] want_esif: Option<u32>,
        #[case] want_crit: Option<u32>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let r = reading(line)?;
        assert_eq!(r.cpu_load_pct, want_cpu);
        assert_eq!(r.gpu_compute_pct, want_gpu);
        assert_eq!(r.throttle_reasons, want_throttle);
        assert_eq!(r.zone_temp_c(), want_temp);
        assert_eq!(r.esif_temp_c, want_esif);
        assert_eq!(r.critical_c, want_crit);
        Ok(())
    }

    /// Lines as PowerShell prints them: `"gpu throttle esif crit zones"`.
    #[rstest]
    #[case("", "empty powershell readiness output")]
    #[case("0", "missing throttle field in readiness output")]
    fn unavailable_errors_are_distinct(
        #[case] line: &str,
        #[case] want_msg: &'static str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let err = parse_windows_reading(5, None, line)
            .err()
            .ok_or("expected error")?;
        let ReadinessError::Unavailable(msg) = err else {
            return Err(format!("expected Unavailable, got {err:?}").into());
        };
        assert_eq!(msg, want_msg);
        Ok(())
    }

    #[rstest]
    #[case::garbage_gpu("notanumber 0", "notanumber")]
    #[case::garbage_throttle("0 weird", "weird")]
    #[case::garbage_esif("0 0 notanumber", "notanumber")]
    #[case::garbage_trip_point("0 0 44 notanumber", "notanumber")]
    fn garbage_field_is_parse_failure(
        #[case] line: &str,
        #[case] want_raw: &'static str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let err = parse_windows_reading(5, None, line)
            .err()
            .ok_or("expected error")?;
        let ReadinessError::ParseFailure { raw } = err else {
            return Err(format!("expected ParseFailure, got {err:?}").into());
        };
        assert_eq!(raw, want_raw);
        Ok(())
    }

    /// Each verdict is independent; absent signals render `n/a`.
    #[rstest]
    #[case::all_ok(
        "0 0 0 - - TZ01=40",
        "cpu=busy:0%(ok) gpu=compute:0%(ok) thermal=throttle:0(ok) temp=acpi:TZ01=40°C"
    )]
    #[case::gpu_busy(
        "5 91 0 - - TZ01=55",
        "cpu=busy:5%(ok) gpu=compute:91%(busy) thermal=throttle:0(ok) temp=acpi:TZ01=55°C"
    )]
    #[case::cpu_idle_high(
        "31 0 0 - - TZ01=45",
        "cpu=busy:31%(ok) gpu=compute:0%(ok) thermal=throttle:0(ok) temp=acpi:TZ01=45°C"
    )]
    #[case::cpu_busy(
        "50 0 0 - - TZ01=60",
        "cpu=busy:50%(busy) gpu=compute:0%(ok) thermal=throttle:0(ok) temp=acpi:TZ01=60°C"
    )]
    #[case::throttling(
        "5 0 8 - - TZ01=50",
        "cpu=busy:5%(ok) gpu=compute:0%(ok) thermal=throttle:8(throttling) temp=acpi:TZ01=50°C"
    )]
    #[case::both_sensors(
        "2 0 0 57 - TZ01=28",
        "cpu=busy:2%(ok) gpu=compute:0%(ok) thermal=throttle:0(ok) temp=esif=57°C,acpi:TZ01=28°C"
    )]
    #[case::pathologically_hot(
        "2 0 0 88 - TZ01=28",
        "cpu=busy:2%(ok) gpu=compute:0%(ok) thermal=throttle:0(ok) temp=esif=88°C,acpi:TZ01=28°C"
    )]
    #[case::cpu_only_box("4 - - - - -", "cpu=busy:4%(ok) gpu=n/a thermal=n/a temp=n/a")]
    fn summary_renders_per_signal_verdict(
        #[case] line: &str,
        #[case] expected: &str,
    ) -> Result<(), ReadinessError> {
        assert_eq!(
            format_summary(&reading(line)?, CPU_LOAD_THRESHOLD_PCT),
            expected
        );
        Ok(())
    }

    /// Measured trips: 105 °C on the 258V, 110 °C on the gmktec.
    #[rstest]
    #[case::under_both("2 0 0 - 105 TZ01=88", false)]
    #[case::at_the_258v_limit("2 0 0 - 105 TZ01=90", true)]
    #[case::same_reading_under_the_gmktec_limit("2 0 0 - 110 TZ01=90", false)]
    #[case::at_the_gmktec_limit("2 0 0 - 110 TZ01=95", true)]
    #[case::fallback_under("2 0 0 - - TZ01=94", false)]
    #[case::fallback_over("2 0 0 - - TZ01=95", true)]
    #[case::esif_over_zone_stuck_low("2 0 0 95 105 TZ01=28", true)]
    #[case::no_sensor_at_all("2 0 0 - 105", false)]
    fn critical_uses_the_platform_trip_point(
        #[case] line: &str,
        #[case] expected: bool,
    ) -> Result<(), ReadinessError> {
        assert_eq!(over_critical(&reading(line)?), expected);
        Ok(())
    }

    /// A held temperature is only suspicious when something else moved.
    #[rstest]
    #[case::stuck_while_load_moved(
        &[Some(28), Some(28), Some(28)],
        &[(60, Some(80)), (2, Some(5)), (1, Some(0))],
        Some(28)
    )]
    #[case::steady_idle_box(
        &[Some(28), Some(28), Some(28)],
        &[(1, Some(0)), (1, Some(0)), (1, Some(0))],
        None
    )]
    #[case::temperature_moved(
        &[Some(58), Some(55), Some(52)],
        &[(60, Some(80)), (2, Some(5)), (1, Some(0))],
        None
    )]
    #[case::too_few_polls(
        &[Some(28), Some(28)],
        &[(60, Some(80)), (1, Some(0))],
        None
    )]
    #[case::no_sensor(
        &[None, None, None],
        &[(60, Some(80)), (2, Some(5)), (1, Some(0))],
        None
    )]
    #[case::sensor_dropped_out(
        &[Some(28), None, Some(28)],
        &[(60, Some(80)), (2, Some(5)), (1, Some(0))],
        None
    )]
    // Jitter under the threshold is not the box going from busy to idle.
    #[case::load_only_jittered(
        &[Some(45), Some(45), Some(45), Some(45)],
        &[(20, Some(0)), (14, Some(0)), (18, Some(0)), (31, Some(0))],
        None
    )]
    #[case::busy_throughout(
        &[Some(45), Some(45), Some(45)],
        &[(60, Some(80)), (55, Some(75)), (57, Some(90))],
        None
    )]
    fn a_stuck_sensor_is_one_that_held_while_the_load_moved(
        #[case] temps: &[Option<u32>],
        #[case] loads: &[(u32, Option<u32>)],
        #[case] expected: Option<u32>,
    ) {
        assert_eq!(
            held_temperature_if_stuck(temps, loads, CPU_LOAD_THRESHOLD_PCT),
            expected
        );
    }

    #[rstest]
    #[case::built_in(CPU_LOAD_THRESHOLD_PCT, None)]
    #[case::tightened(15, Some(28))]
    fn the_stuck_check_uses_the_gates_cpu_threshold(
        #[case] cpu_max: u32,
        #[case] expected: Option<u32>,
    ) {
        let loads = [(30, Some(0)), (2, Some(0)), (1, Some(0))];
        assert_eq!(
            held_temperature_if_stuck(&[Some(28); 3], &loads, cpu_max),
            expected
        );
    }

    #[rstest]
    #[case::esif_missing("2 0 0 - - TZ01=88", vec![("acpi", 88)])]
    #[case::zone_missing("2 0 0 57 -", vec![("esif", 57)])]
    #[case::both_present("2 0 0 57 - TZ01=28", vec![("esif", 57), ("acpi", 28)])]
    #[case::neither("2 0 0 - -", vec![])]
    #[case::zero_esif_dropped("2 0 0 0 - TZ01=28", vec![("acpi", 28)])]
    fn temperatures_lists_only_the_sensors_that_answered(
        #[case] line: &str,
        #[case] expected: Vec<(&str, u32)>,
    ) -> Result<(), ReadinessError> {
        let got: Vec<_> = reading(line)?.temperatures().collect();
        assert_eq!(got, expected);
        Ok(())
    }

    #[test]
    fn a_stuck_sensor_is_reported_even_beside_a_working_one() -> Result<(), ReadinessError> {
        let samples = series(&[
            "60 80 0 44 - TZ01=28",
            "2 0 0 58 - TZ01=28",
            "1 0 0 46 - TZ01=28",
        ])?;
        let loads: Vec<(u32, Option<u32>)> = samples
            .iter()
            .map(|s| (s.cpu_load_pct, s.gpu_compute_pct))
            .collect();
        let zone: Vec<Option<u32>> = samples.iter().map(WindowsReading::zone_temp_c).collect();
        let esif: Vec<Option<u32>> = samples.iter().map(|s| s.esif_temp_c).collect();
        assert_eq!(
            held_temperature_if_stuck(&zone, &loads, CPU_LOAD_THRESHOLD_PCT),
            Some(28)
        );
        assert_eq!(
            held_temperature_if_stuck(&esif, &loads, CPU_LOAD_THRESHOLD_PCT),
            None
        );
        Ok(())
    }

    #[rstest]
    #[case::three_zones("THRM=48,THZ0=43,TZ01=20", vec![("THRM", 48), ("THZ0", 43), ("TZ01", 20)])]
    #[case::absent("-", vec![])]
    #[case::placeholder_left_out("THRM=48,TZ02=-273", vec![("THRM", 48)])]
    fn zones_parse(
        #[case] raw: &str,
        #[case] expected: Vec<(&str, u32)>,
    ) -> Result<(), ReadinessError> {
        let got: Vec<(String, u32)> = parse_zones(raw)?
            .into_iter()
            .map(|z| (z.name, z.temp_c))
            .collect();
        let expected: Vec<(String, u32)> = expected
            .into_iter()
            .map(|(z, c)| (z.to_string(), c))
            .collect();
        assert_eq!(got, expected);
        Ok(())
    }

    #[rstest]
    #[case::no_equals("THZ0")]
    #[case::not_a_number("THZ0=warm")]
    #[case::no_name("=43")]
    fn a_garbled_zone_is_parse_failure(#[case] raw: &str) {
        assert!(matches!(
            parse_zones(raw),
            Err(ReadinessError::ParseFailure { .. })
        ));
    }

    #[test]
    fn the_zone_list_is_optional() -> Result<(), ReadinessError> {
        assert!(reading("2 0 0 - 103")?.zones.is_empty());
        Ok(())
    }

    fn config(
        ceilings: &[(&str, u32)],
        max_cpu_busy_pct: Option<u32>,
    ) -> Result<WindowsReadinessConfiguration, Box<dyn std::error::Error>> {
        Ok(WindowsReadinessConfiguration {
            ceilings: ceilings
                .iter()
                .map(|(sensor, c)| {
                    Ok((
                        sensor.parse::<WindowsSensor>()?,
                        CeilingCelsius::try_from(*c)?,
                    ))
                })
                .collect::<Result<_, Box<dyn std::error::Error>>>()?,
            max_cpu_busy_pct: max_cpu_busy_pct.map(CpuPercent::try_from).transpose()?,
        })
    }

    fn polls_to_release(
        lines: &[&str],
        max_wait_secs: u64,
        thermal: ThermalGate,
        config: &WindowsReadinessConfiguration,
    ) -> Result<usize, ReadinessError> {
        let mut scripted = series(lines)?.into_iter();
        let mut polls = 0usize;
        wait_for(
            Duration::from_secs(max_wait_secs),
            thermal,
            config,
            &mut || {
                polls += 1;
                scripted
                    .next()
                    .ok_or(ReadinessError::Unavailable("probe script exhausted"))
            },
            &silent(),
        )?;
        Ok(polls)
    }

    /// Polled on the Zenbook S 16 after 6 minutes of prefill, box idle, ~9.5 s
    /// apart. The firmware's power cut lifted only once THZ0 read 43 °C.
    const ZENBOOK_AFTER_A_CUT: [&str; 14] = [
        "2 0 0 - 103 THRM=78,THZ0=65,TZ01=20",
        "2 0 0 - 103 THRM=64,THZ0=63,TZ01=20",
        "2 0 0 - 103 THRM=59,THZ0=59,TZ01=20",
        "2 0 0 - 103 THRM=56,THZ0=56,TZ01=20",
        "2 0 0 - 103 THRM=54,THZ0=54,TZ01=20",
        "2 0 0 - 103 THRM=52,THZ0=52,TZ01=20",
        "2 0 0 - 103 THRM=50,THZ0=50,TZ01=20",
        "2 0 0 - 103 THRM=49,THZ0=48,TZ01=20",
        "2 0 0 - 103 THRM=48,THZ0=47,TZ01=20",
        "2 0 0 - 103 THRM=47,THZ0=46,TZ01=20",
        "2 0 0 - 103 THRM=46,THZ0=45,TZ01=20",
        "2 0 0 - 103 THRM=46,THZ0=44,TZ01=20",
        "2 0 0 - 103 THRM=45,THZ0=44,TZ01=20",
        "2 0 0 - 103 THRM=45,THZ0=43,TZ01=20",
    ];

    /// One idle poll with THZ0 over a 43 °C ceiling.
    const HOT: &str = "2 0 0 - 103 THRM=50,THZ0=50,TZ01=20";

    const CPU_12_THEN_3: [&str; 2] = ["12 0 0 - 110 TZ01=40", "3 0 0 - 110 TZ01=40"];

    #[rstest]
    // Without a ceiling the gate releases at once, with the cut still on.
    #[case::load_only(&ZENBOOK_AFTER_A_CUT, 300, ThermalGate::Enforce, &[], None, 1)]
    #[case::held_to_43(
        &ZENBOOK_AFTER_A_CUT, 300, ThermalGate::Enforce, &[("acpi:THZ0", 43)], None, 14
    )]
    #[case::lowercase_name(
        &ZENBOOK_AFTER_A_CUT, 300, ThermalGate::Enforce, &[("acpi:thz0", 43)], None, 14
    )]
    #[case::spent_budget(&[HOT], 0, ThermalGate::Enforce, &[("acpi:THZ0", 43)], None, 1)]
    #[case::unread_holds(
        &["2 0 0 - 103", "2 0 0 - 103 THRM=45,THZ0=43,TZ01=20"],
        300,
        ThermalGate::Enforce,
        &[("acpi:THZ0", 43)],
        None,
        2
    )]
    #[case::skip_ignores_ceilings(&[HOT], 300, ThermalGate::Skip, &[("acpi:THZ0", 43)], None, 1)]
    #[case::skip_ignores_an_absent_zone(
        &[HOT], 300, ThermalGate::Skip, &[("acpi:THZ1", 43)], None, 1
    )]
    // With no ceiling, temperature holds nothing however fast the die cools.
    #[case::cooling(
        &["2 0 0 - 110 TZ01=85", "2 0 0 - 110 TZ01=62", "2 0 0 - 110 TZ01=58"],
        300,
        ThermalGate::Enforce,
        &[],
        None,
        1
    )]
    #[case::busy_then_idle(
        &["60 90 0 - 110 TZ01=85", "2 0 0 - 110 TZ01=62"],
        300,
        ThermalGate::Enforce,
        &[],
        None,
        2
    )]
    #[case::cpu_built_in(&CPU_12_THEN_3, 300, ThermalGate::Enforce, &[], None, 1)]
    #[case::cpu_tighter_holds(&CPU_12_THEN_3, 300, ThermalGate::Enforce, &[], Some(10), 2)]
    #[case::cpu_inclusive(&CPU_12_THEN_3, 300, ThermalGate::Enforce, &[], Some(12), 1)]
    fn the_gate_releases_after(
        #[case] lines: &[&str],
        #[case] max_wait_secs: u64,
        #[case] thermal: ThermalGate,
        #[case] ceilings: &[(&str, u32)],
        #[case] max_cpu_busy_pct: Option<u32>,
        #[case] expected_polls: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let config = config(ceilings, max_cpu_busy_pct)?;
        assert_eq!(
            polls_to_release(lines, max_wait_secs, thermal, &config)?,
            expected_polls
        );
        Ok(())
    }

    /// Fails at once with no terminal status: it is no timeout.
    #[test]
    fn a_ceiling_on_an_absent_zone_fails_naming_the_zones() -> Result<(), Box<dyn std::error::Error>>
    {
        let phases = RefCell::new(Vec::new());
        let record = |status: &ReadinessStatus| phases.borrow_mut().push(status.phase);
        let mut scripted = series(&ZENBOOK_AFTER_A_CUT)?.into_iter();
        let err = wait_for(
            Duration::from_secs(300),
            ThermalGate::Enforce,
            &config(&[("acpi:THZ1", 43)], None)?,
            &mut || {
                scripted
                    .next()
                    .ok_or(ReadinessError::Unavailable("probe script exhausted"))
            },
            &ReadinessCallbacks::new()
                .with_sleep(&no_sleep)
                .with_status(&record),
        )
        .err()
        .ok_or("expected the wait to fail")?;
        let ReadinessError::UnknownSensor { sensor, available } = err else {
            return Err(format!("expected UnknownSensor, got {err:?}").into());
        };
        assert_eq!(sensor.to_string(), "acpi:THZ1");
        assert_eq!(available, "acpi:THRM, acpi:THZ0, acpi:TZ01");
        assert!(phases.borrow().is_empty(), "{:?}", phases.borrow());
        Ok(())
    }

    /// Unlike a sensor that read hot, which the spent budget lets through.
    #[test]
    fn a_sensor_never_read_fails_at_the_budget() -> Result<(), Box<dyn std::error::Error>> {
        let err = polls_to_release(
            &["2 0 0 - 103"],
            0,
            ThermalGate::Enforce,
            &config(&[("gpu", 70)], None)?,
        );
        assert!(
            matches!(
                &err,
                Err(ReadinessError::SensorUnread {
                    sensor: WindowsSensor::Gpu,
                    ..
                })
            ),
            "{err:?}"
        );
        Ok(())
    }

    #[rstest]
    #[case::ok("2 0 0 - 103 THRM=48,THZ0=43,TZ01=20", "ceiling=acpi:THZ0=43°C/43(ok)")]
    #[case::hot(
        "2 0 0 - 103 THRM=48,THZ0=47,TZ01=20",
        "ceiling=acpi:THZ0=47°C/43(hot)"
    )]
    #[case::no_such_zone("2 0 0 - 103 THRM=48,TZ01=20", "ceiling=acpi:THZ0=absent/43")]
    #[case::zones_unread("2 0 0 - 103", "ceiling=acpi:THZ0=unread/43")]
    fn the_summary_leads_with_each_ceiling(
        #[case] line: &str,
        #[case] prefix: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let r = reading(line)?;
        let summary = format_summary(&r, CPU_LOAD_THRESHOLD_PCT);
        assert_eq!(
            with_ceilings(summary.clone(), &r, &config(&[("acpi:THZ0", 43)], None)?),
            format!("{prefix} {summary}")
        );
        Ok(())
    }

    /// The SER9's only zone is a constant; its GPU sensor is the real one.
    #[test]
    fn the_gpu_sensor_is_reported_beside_a_constant_zone() -> Result<(), ReadinessError> {
        let r = WindowsReading {
            gpu_temp_c: Some(34),
            ..reading("2 0 0 - 110 TZ01=20")?
        };
        assert_eq!(
            format_summary(&r, CPU_LOAD_THRESHOLD_PCT),
            "cpu=busy:2%(ok) gpu=compute:0%(ok) thermal=throttle:0(ok) temp=acpi:TZ01=20°C,gpu=34°C"
        );
        Ok(())
    }

    fn times(idle: u64, total: u64) -> CpuTimes {
        CpuTimes { idle, total }
    }

    #[rstest]
    #[case::idle(times(5_000, 9_000), times(6_000, 10_000), Some(0))]
    // 32 logical CPUs over 1 s, half of them spinning.
    #[case::half_busy(times(0, 0), times(160_000_000, 320_000_000), Some(50))]
    #[case::saturated(times(5_000, 9_000), times(5_000, 10_000), Some(100))]
    #[case::rounds_half_up(times(0, 0), times(395, 1_000), Some(61))]
    #[case::rounds_down(times(0, 0), times(396, 1_000), Some(60))]
    #[case::no_overflow(times(0, 0), times(u64::MAX / 2, u64::MAX), Some(50))]
    // A torn pair of snapshots reads as idle rather than underflowing.
    #[case::idle_past_total(times(0, 0), times(1_200, 1_000), Some(0))]
    #[case::nothing_accounted(times(5_000, 9_000), times(5_000, 9_000), None)]
    #[case::total_went_backwards(times(5_000, 9_000), times(5_000, 8_000), None)]
    fn busy_pct_is_the_non_idle_share(
        #[case] before: CpuTimes,
        #[case] after: CpuTimes,
        #[case] expected: Option<u32>,
    ) {
        assert_eq!(busy_pct(before, after), expected);
    }

    /// The whole probe on this host, on the Windows CI leg.
    #[cfg(target_os = "windows")]
    #[test]
    fn the_probe_reads_this_host() -> Result<(), ReadinessError> {
        read_status()?;
        Ok(())
    }
}
