//! Windows readiness — gates on CPU load, GPU compute utilization,
//! thermal-throttling reason flags, and die temperature.
//!
//! Only CPU load is mandatory; the rest gate **only when the
//! underlying counter exists on the box**. Signals are sampled in a
//! single PowerShell invocation (interpreter spin-up costs ~500 ms
//! and dominates the syscall budget).
//!
//! # The temperature criterion is a decay test, not a ceiling
//!
//! A fixed ceiling cannot serve both boxes in the fleet, because
//! "hot" is not portable between chassis. Measured, idle to
//! saturated:
//!
//! | | resting | saturated | span |
//! |---|---|---|---|
//! | gmktec EVO-X2 (Ryzen AI MAX+ 395), ACPI zone | 33–36 °C | 98 °C | 65 °C |
//! | devcloud (Core Ultra 7 258V), ESIF | 42–46 °C | 55 °C | 13 °C |
//!
//! One box rests 9 °C hotter than the other and uses a fifth of the
//! range. Any single ceiling is therefore either unreachable on one
//! box or a no-op on the other: the 70 °C ceiling this replaced was
//! *both* — the 258V never reaches it under full saturation, so the
//! criterion never fired there, while the gmktec falls under it 10 s
//! after load stops.
//!
//! What *is* portable is the shape of the curve. Both boxes shed the
//! junction heat fast and then crawl:
//!
//! | | fast phase | then |
//! |---|---|---|
//! | gmktec | 98 → 44 °C in 30 s (83 % of span) | 44 → 33 °C over ~600 s |
//! | devcloud | 55 → 46 °C in 42 s (69 % of span) | 46 → 42 °C over ~700 s |
//!
//! So the gate waits for the reading to go *flat* — a spread within
//! [`FLAT_SPAN_C`] across [`FLAT_WINDOW_POLLS`] polls — rather than
//! for it to reach a number. A derivative is scale-free, so the
//! sensor spanning 65 °C and the one spanning 13 °C are judged by
//! the same rule with nothing to calibrate per chassis. Replayed
//! against the measured curves at [`POLL_INTERVAL`] it releases 30 s
//! after load ends on the gmktec and 10 s on the devcloud, against
//! the 161 s / 301 s an absolute threshold low enough to mean
//! anything would have cost.
//!
//! On the devcloud that 10 s is the floor the window itself costs,
//! not an observed plateau: its whole cooldown is 51 → 43 °C over
//! ~13 minutes, so it never moves more than 2.4 °C across a 10 s
//! window and the span is satisfied on the first full one. The
//! criterion does real work on the gmktec (30 s against the same
//! 10 s floor) and degrades to a fixed settle on the devcloud.
//! Tightening [`FLAT_SPAN_C`] to catch that box is not available —
//! its sensor jitters ±2 °C at rest, so a 1 °C span would never be
//! satisfied and every cell would run to [`FLAT_MAX_WAIT`].
//!
//! Flatness cannot be faked by a box on its way down: the gmktec
//! passes through 70 °C between consecutive samples, and every
//! 3-sample window across its descent spans 30–41 °C.
//!
//! # Which sensors are read, and why none outranks another
//!
//! `\EsifDeviceInformation(*)\Temperature` (Intel Dynamic Tuning,
//! already °C) and `\Thermal Zone Information(*)\Temperature`
//! (Kelvin, max across zones). Which one works is a property of the
//! box: ESIF is absent on the AMD boxes, while on the 258V the ACPI
//! zone reports exactly 301.00 K through idle, saturation and
//! cooldown alike. Ranking them means deciding per chassis and being
//! wrong on hardware nobody has measured.
//!
//! Every sensor must be flat, which needs no such decision and errs
//! toward waiting. A sensor stuck at a constant is trivially flat and
//! so contributes nothing — it cannot hold the gate, and it cannot
//! wave a hot box through either, because the sensors that do move
//! are still judged.
//!
//! The AMD zone is a real junction temperature, not the board sensor
//! `docs/fleet-perf-troubleshooting/gmktec-evo-x2.md` warns about: it
//! tracks LibreHardwareMonitor's `Tctl/Tdie` within 1–4 °C across its
//! whole range (98.3/98, 40.4/40, 33.4/33 °C). `ryzenadj`'s `THM
//! VALUE CORE` is the one that misreports on Strix Halo — it read
//! 42.7 °C while the die was at 98 °C under 73–103 W of measured
//! package power, consistent with its clocks reading `nan` there.
//!
//! # The two absolute thresholds, and what each is for
//!
//! Flatness says "at steady state", not "cool" — a box whose fan has
//! failed sits flat and hot. So two absolute checks sit above it,
//! with deliberately different consequences:
//!
//! - [`CRITICAL_MARGIN_C`] below the platform's own `CriticalTripPoint`
//!   **holds the gate**. That is a hardware-declared limit (measured:
//!   110 °C on the gmktec, 105 °C on the 258V), so it means the same
//!   thing on every chassis — unlike `PassiveTripPoint`, which reads 0
//!   on both.
//! - [`PATHOLOGY_C`] only **warns**. Blocking is futile at steady
//!   state — the box is not going to cool further — and failing the
//!   cell loses the measurement instead of flagging it. The reading is
//!   logged so the cell stays attributable.
//!
//! # Known gaps
//!
//! **Throttle reasons never fire.** `\Thermal Zone Information(*)\
//! Throttle Reasons` read 0 on both boxes across every sample,
//! including while the gmktec die sat pinned at 98 °C — its own
//! limit. The signal is kept because a 0 costs nothing, but nothing
//! has ever been observed to set it. A sensor pinned to one value
//! while package power swings is the more reliable indicator that
//! silicon is at a limiter, and is not implemented here.
//!
//! **A sensor at its limiter is flat.** A pinned reading is a plateau
//! like any other, so saturation and rest are indistinguishable to
//! the decay test — observed directly: the gmktec held exactly 98 °C
//! across eight consecutive polls under load. What separates them
//! here is that the plateau sits above the critical threshold
//! (98 ≥ 110 − [`CRITICAL_MARGIN_C`]), so the gate holds. That is
//! measured, not guaranteed: a box that saturated below its critical
//! margin *and* whose load counter dipped under
//! [`CPU_LOAD_THRESHOLD_PCT`] for a full window would read as
//! settled. During real saturation that counter ran 47–78 %, and the
//! dips to 29–31 % appeared only as the load wound down, so the
//! overlap needs both to go wrong at once.
//!
//! **Nothing catches an OpenVINO NPU graph compile**, ~90 % of an NPU
//! cell's wall clock: GPU compute reads 0 % (the work is on the NPU)
//! and the die is still cold. `\Energy Meter(*)\Power` does see it —
//! 6.6 W against a 3.3 W floor — but is verified on one laptop, and
//! its absent / transient / all-zero paths all read as idle.
//!
//! **A box with no temperature counter gets no cooldown**, only the
//! [`FLAT_WINDOW_POLLS`]-poll minimum the algorithm already costs.
//! Inventing a longer delay for hardware nobody has measured would be
//! superstition, not caution.

// The decision logic is arithmetic over sampled readings, so it is portable
// even though the probe is not. CI has no Windows runner, so compiling this
// module into test builds everywhere is the only way these tests run at all —
// which leaves the Windows-only entry points reading as dead code elsewhere.
#![cfg_attr(not(target_os = "windows"), allow(dead_code))]

use std::time::{Duration, Instant};

use super::{ReadinessError, ThermalGate};

/// Default deadline for the wait: the shared cross-platform value
/// (see [`super::DEFAULT_MAX_WAIT`]). Mini PCs / NUCs with small case
/// fans take longer to dump heat than thinner laptops, and 5 minutes
/// is a comfortable upper bound for our hardware. Same number as
/// before, now stated once.
pub(super) const DEFAULT_MAX_WAIT: Duration = super::DEFAULT_MAX_WAIT;
/// 5 s so the flatness window spans 10 s of wall clock. The decay knee falls at
/// 14–30 s on the measured boxes, and a 20 s window would straddle it — the gate
/// would still be watching the fast phase when it was meant to be confirming the
/// slow one.
///
/// **Not [`crate::SENSOR_POLL_INTERVAL`], and not an override of it.** Elsewhere
/// the interval is granularity, and shortening it only removes latency; here it is
/// half the criterion, because the gate judges a spread across
/// [`FLAT_WINDOW_POLLS`] samples and this spacing is what gives that window its
/// duration. At the shared 3 s the window would span 6 s, which on the gmktec's
/// measured curve is short enough to read the fast phase as a plateau — the gate
/// would release on cooling silicon and the run would measure a throttle. Change
/// this only against a re-measured curve, and move [`FLAT_WINDOW_POLLS`] with it.
const POLL_INTERVAL: Duration = Duration::from_secs(5);
/// Highest acceptable `\Processor Information(_Total)\% Processor Time`.
///
/// Idle ran 14–31 % on the 258V and 0–9 % on the gmktec; saturation ran
/// 50–57 % and 45–51 %. 40 % sits in the gap on both, closer to the idle side
/// so a burst of Windows housekeeping on the noisier box does not trip it.
///
/// This replaced `Win32_Processor.LoadPercentage`, which cannot separate the
/// two states on either box: it read 1–19 % on the 258V under a 16-job burn
/// against 1–6 % idle, and alternated 100 / 3 / 100 / 90 during a steady burn
/// on the gmktec.
const CPU_LOAD_THRESHOLD_PCT: u32 = 40;
/// Highest acceptable summed GPU-compute utilization. Idle baseline
/// is exactly 0 and active Vulkan inference sits at 71–105 %; we
/// never observed any in-between values on this hardware. 5 % is
/// well above zero-noise but far enough below the active band to
/// catch any non-trivial compute workload.
const GPU_COMPUTE_THRESHOLD_PCT: u32 = 5;
/// Polls a temperature must stay within [`FLAT_SPAN_C`] to count as settled.
///
/// Three is the shortest series that distinguishes a plateau from two readings
/// that happen to match, and at [`POLL_INTERVAL`] it costs 10 s — the floor on
/// how fast this gate can ever release.
const FLAT_WINDOW_POLLS: usize = 3;
/// Widest spread across the window that still counts as settled, °C.
///
/// The measured sensors quantize to 1 °C and jitter by up to 2 °C at rest, so
/// a tighter span would never be satisfied. Wide enough to clear that noise,
/// narrow enough that the fast phase never fits inside it: the descending
/// windows span 30–41 °C on the gmktec.
const FLAT_SPAN_C: u32 = 3;
/// Steady-state temperature too hot to be a resting box, °C. Warns; see the
/// module docs for why it does not block.
///
/// Resting is 33–36 °C on the gmktec and 42–46 °C on the 258V, so this clears
/// the hotter box by 14 °C and cannot fire on a healthy machine. It is not
/// enforcing cooldown — flatness does that — which is exactly why it can sit
/// this far above both without being a no-op.
const PATHOLOGY_C: u32 = 60;
/// Margin below the platform's `CriticalTripPoint`, °C. That trip is a
/// shutdown threshold (110 °C / 105 °C measured), so the margin only has to
/// keep the gate clear of thermal runaway, not of normal operation.
const CRITICAL_MARGIN_C: u32 = 15;
/// Critical limit for a box that reports no trip point, °C. Above anything
/// either measured box reaches, so it only ever fires on hardware whose limits
/// we cannot read.
const CRITICAL_FALLBACK_C: u32 = 95;
/// Cap on waiting for flatness, independent of the overall deadline.
///
/// Past the knee, waiting buys fractions of a degree per minute — the gmktec
/// needs another ~600 s to give up its last 11 °C. Capping here and recording
/// the temperature is worth more than spending that on every cell.
const FLAT_MAX_WAIT: Duration = Duration::from_secs(60);

#[derive(Debug, Clone, Copy)]
struct WindowsReading {
    cpu_load_pct: u32,
    gpu_compute_pct: Option<u32>,
    throttle_reasons: Option<u32>,
    /// ACPI thermal zone, °C (the counter reports Kelvin).
    zone_temp_c: Option<u32>,
    /// Intel Dynamic Tuning (DPTF/ESIF), already °C.
    esif_temp_c: Option<u32>,
    /// ACPI `CriticalTripPoint`, °C (the class reports deci-Kelvin).
    critical_c: Option<u32>,
}

/// A named temperature counter: its log label and how to read it off a poll.
type TempSource = (&'static str, fn(&WindowsReading) -> Option<u32>);

/// Every temperature counter this module samples, in log order.
///
/// Adding one is a row here plus a block in the PowerShell below; the gate,
/// the summary and the stuck-sensor check all read from this table. Order is
/// cosmetic — no entry outranks another.
const TEMP_SOURCES: [TempSource; 2] = [("esif", |r| r.esif_temp_c), ("acpi", |r| r.zone_temp_c)];

impl WindowsReading {
    /// Every temperature this box reported this poll, labelled with its sensor.
    fn temperatures(&self) -> impl Iterator<Item = (&'static str, u32)> + '_ {
        TEMP_SOURCES
            .into_iter()
            .filter_map(|(name, read)| read(self).map(|c| (name, c)))
    }
}

/// Whether every sensor has gone flat across the most recent window.
///
/// False until the window is full: with fewer samples than that there is no
/// series to judge, only readings.
fn is_settled(samples: &[WindowsReading]) -> bool {
    let Some(start) = samples.len().checked_sub(FLAT_WINDOW_POLLS) else {
        return false;
    };
    let window = &samples[start..];
    TEMP_SOURCES
        .into_iter()
        .all(|(_, read)| sensor_is_flat(window, read))
}

/// Whether one sensor's spread across the window is within [`FLAT_SPAN_C`].
///
/// A sensor that missed any poll in the window is not judged rather than
/// assumed to be moving — a counter that drops out intermittently would
/// otherwise hold the gate for the full cap on every cell. A sensor absent
/// from the box entirely takes the same path, which is what makes a box with
/// no temperature counter cost only the window itself.
fn sensor_is_flat(window: &[WindowsReading], read: fn(&WindowsReading) -> Option<u32>) -> bool {
    let readings: Vec<u32> = window.iter().filter_map(read).collect();
    if readings.len() < window.len() {
        return true;
    }
    match (readings.iter().min(), readings.iter().max()) {
        (Some(low), Some(high)) => high.saturating_sub(*low) <= FLAT_SPAN_C,
        _ => true,
    }
}

/// Whether any sensor is close enough to the platform's shutdown trip to hold
/// the gate regardless of whether it has settled.
fn over_critical(reading: &WindowsReading) -> bool {
    let limit = reading
        .critical_c
        .map_or(CRITICAL_FALLBACK_C, |c| c.saturating_sub(CRITICAL_MARGIN_C));
    reading.temperatures().any(|(_, c)| c >= limit)
}

/// Polls needed before a never-moving temperature is worth reporting. Three is
/// enough to be a series rather than a coincidence. It says nothing on its own
/// about whether the box was ever busy — a wait can reach three polls and time
/// out still working — which is why the load check below is a separate
/// condition rather than an assumption.
const CONSTANT_TEMP_MIN_POLLS: usize = 3;

pub(super) fn wait_until_ready(
    max_wait: Duration,
    thermal: ThermalGate,
) -> Result<(), ReadinessError> {
    let started = Instant::now();
    let deadline = started + max_wait;
    // Flatness is judged over a window, so the series has to outlive the poll.
    let mut samples: Vec<WindowsReading> = Vec::new();
    // Never outlives the caller's own deadline: a short `max_wait` is a
    // caller saying the whole wait is worth less than this cap.
    let settle_cap = FLAT_MAX_WAIT.min(max_wait);
    loop {
        let reading = read_status()?;
        samples.push(reading);
        let cpu_ok = reading.cpu_load_pct < CPU_LOAD_THRESHOLD_PCT;
        let gpu_ok = reading
            .gpu_compute_pct
            .is_none_or(|v| v < GPU_COMPUTE_THRESHOLD_PCT);
        let skip = thermal == ThermalGate::Skip;
        let throttle_ok = skip || reading.throttle_reasons.is_none_or(|v| v == 0);
        let critical = !skip && over_critical(&reading);
        let settled = skip || is_settled(&samples);
        // Two budgets, because the criteria fail for different reasons. A busy
        // box may genuinely free up within `max_wait`; a box past the knee will
        // not get meaningfully cooler no matter how long the wait is.
        let settle_expired = started.elapsed() >= settle_cap;
        let summary = format_summary(&reading);
        if cpu_ok && gpu_ok && throttle_ok && !critical && (settled || settle_expired) {
            warn_about_stuck_sensors(&samples);
            if !settled {
                log::warn!(
                    "readiness: windows temperature still moving after {settle_cap:?} \
                     ({summary}) — proceeding anyway; this cell started hotter than a \
                     settled box would"
                );
            }
            warn_about_pathological_heat(&reading);
            log::info!("readiness: windows {summary} → proceeding");
            return Ok(());
        }
        if Instant::now() >= deadline {
            // Also on this path: a sensor stuck *high* times out every single
            // time, which is precisely when the reader needs to know the
            // number they are being blocked by is not measuring anything.
            warn_about_stuck_sensors(&samples);
            log::info!("readiness: windows {summary} → timed out after {max_wait:?}");
            return Err(ReadinessError::TimedOut {
                max_wait,
                observed: summary,
            });
        }
        log::info!("readiness: windows {summary} → waiting {POLL_INTERVAL:?}");
        std::thread::sleep(POLL_INTERVAL);
    }
}

/// Report a sensor resting above [`PATHOLOGY_C`], which a healthy box in this
/// fleet cannot do — a failed fan, a clogged heatsink, an extreme ambient, or
/// a sensor stuck high all land here.
fn warn_about_pathological_heat(reading: &WindowsReading) {
    reading
        .temperatures()
        .filter(|(_, c)| *c >= PATHOLOGY_C)
        .for_each(|(name, c)| {
            log::warn!(
                "readiness: windows `{name}` sensor reads {c}°C at the point the gate released, \
                 above the {PATHOLOGY_C}°C a resting box should hold — this cell's numbers are \
                 not comparable with one started from a cool box"
            );
        });
}

/// The value a stuck sensor is holding, if the series looks stuck: one
/// temperature across enough polls to be a series, while the load signals
/// demonstrably moved.
///
/// Observed on a Core Ultra 7 258V: the ACPI zone reports exactly 301.00 K
/// through idle, 75 s of full-core saturation and cooldown alike, and the
/// legacy `MSAcpi_ThermalZoneTemperature` returns the same constant.
fn held_temperature_if_stuck(temps: &[Option<u32>], loads: &[(u32, Option<u32>)]) -> Option<u32> {
    if temps.len() < CONSTANT_TEMP_MIN_POLLS {
        return None;
    }
    let first = temps.first().copied().flatten()?;
    let held = temps.iter().all(|t| *t == Some(first));
    // Only meaningful if something else moved; an idle box holding steady is
    // exactly what a working sensor should report.
    //
    // "Moved" has to mean the *verdict* changed, not the number. Raw counters
    // jitter — idle CPU samples on a 258V ran 3, 1, 1, 2, 1, 8 — so comparing
    // consecutive readings for inequality fires on noise, and the whole check
    // collapses into "the temperature was constant", which would accuse a
    // working sensor on any box blocked by GPU for a few polls.
    let load_moved = loads.iter().any(is_busy) && loads.iter().any(|l| !is_busy(l));
    (held && load_moved).then_some(first)
}

/// Whether one poll's load signals put the box over any gate threshold.
fn is_busy((cpu, gpu): &(u32, Option<u32>)) -> bool {
    *cpu >= CPU_LOAD_THRESHOLD_PCT || gpu.is_some_and(|v| v >= GPU_COMPUTE_THRESHOLD_PCT)
}

/// Report each sensor that never moved, so a constant reading isn't mistaken
/// for a cold box.
///
/// Purely diagnostic. Under a flatness gate a constant sensor is already
/// harmless — it is trivially flat, so it neither holds the gate nor releases
/// it while another sensor is still moving — but an operator comparing two
/// hosts still needs to know that one of the numbers in the log is furniture.
///
/// Silence is not a clean bill of health: a box that settles quickly never
/// reaches [`CONSTANT_TEMP_MIN_POLLS`] busy polls, and a host whose load
/// signals under-report can fail the busy-then-idle condition even while
/// working hard.
fn warn_about_stuck_sensors(samples: &[WindowsReading]) {
    let loads: Vec<(u32, Option<u32>)> = samples
        .iter()
        .map(|s| (s.cpu_load_pct, s.gpu_compute_pct))
        .collect();
    TEMP_SOURCES.into_iter().for_each(|(name, read)| {
        let series: Vec<Option<u32>> = samples.iter().map(read).collect();
        if let Some(held) = held_temperature_if_stuck(&series, &loads) {
            log::warn!(
                "readiness: windows `{name}` sensor held {held}°C across {} polls while the \
                 load signals crossed their thresholds — probably a BIOS constant, so it is \
                 contributing nothing to the temperature gate",
                samples.len()
            );
        }
    });
}

fn format_summary(r: &WindowsReading) -> String {
    let cpu_verdict = if r.cpu_load_pct < CPU_LOAD_THRESHOLD_PCT {
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
    // Every sensor, not just the decisive one: which sensors a host has, and
    // how far apart they read, is the first thing anyone debugging this needs.
    let sensors: Vec<String> = r
        .temperatures()
        .map(|(name, c)| {
            format!(
                "{name}:{c}°C({})",
                if c < PATHOLOGY_C { "ok" } else { "hot" }
            )
        })
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

fn read_status() -> Result<WindowsReading, ReadinessError> {
    let output = std::process::Command::new("powershell")
        .args([
            "-NoProfile",
            "-Command",
            // Emit six whitespace-separated tokens:
            //   cpu_load_pct  gpu_compute_pct  throttle_reasons
            //   temp_c  esif_temp_c  critical_c
            //
            // Every one is either an integer or the sentinel `-`,
            // meaning "the underlying counter does not exist on this
            // box" (e.g. no GPU, no exposed thermal zones). The Rust
            // side treats `-` as `None` and skips that signal; only
            // CPU load is mandatory and surfaces `-` as an error.
            //
            // Each signal initializes to `-`, then a `try { measure }
            // catch {}` overwrites it on success. `-ErrorAction Stop`
            // turns the cmdlet's non-terminating error into a
            // terminating one so the `catch` fires. The structurally-
            // equivalent `if (counter-present-check) { measure }`
            // shape hangs PowerShell silently on a real GMKTec box
            // for reasons we couldn't pin down.
            //
            // The `catch {}` also swallows transient Get-Counter
            // failures (perfmon catalog briefly locked, antivirus
            // inspecting the namespace, etc.), not just genuine
            // absence. That's intentional: the readiness loop polls
            // every few seconds, so a transient miss recovers on the
            // next pass. A genuinely absent counter stays `None` for
            // the lifetime of this process.
            //
            // GPU compute is summed across processes because each
            // `engtype_compute N` instance reports per-engine
            // utilization. The throttle field aggregates with
            // `Maximum` across thermal zones — sufficient because the
            // gate only checks `== 0`; the logged value is not a
            // faithful bitmask if multiple zones throttle for
            // different reasons. The zone temperature counter is in
            // Kelvin (as a double); we take the max across zones and
            // subtract 273.
            //
            // The ESIF (Intel Dynamic Tuning) temperature is already in
            // Celsius, unlike the Kelvin thermal-zone counter — do not
            // subtract 273 from it. Instances that carry no sensor report
            // exactly 0 (7 of 10 on a 258V), so they are filtered out before
            // the max; a box where *every* instance reads 0 must come back as
            // absent rather than 0 °C, or the gate would read "stone cold" and
            // wave everything through.
            //
            // `CriticalTripPoint` is deci-Kelvin (3832 → 110 °C on the
            // gmktec, 3782 → 105 °C on the 258V) and reads 0 where
            // unpopulated, so zeros are dropped the same way. `Minimum`
            // across zones is the conservative choice — the gate must
            // clear the *earliest* shutdown trip, not the latest.
            //
            // **Unverified**: the absent-counter path for the GPU and
            // thermal-zone signals. ESIF's is verified — it is absent
            // on all three reachable gmktecs and `Get-Counter
            // -ErrorAction Stop` throws there in 105–448 ms on first
            // call and 0 ms after, rather than hanging on enumeration.
            "$cpu = '-';\
             try { \
               $m = (Get-Counter -Counter '\\Processor Information(_Total)\\% Processor Time' \
                -ErrorAction Stop).CounterSamples | \
                Measure-Object -Property CookedValue -Maximum;\
               if ($null -ne $m.Maximum) { $cpu = [int]$m.Maximum } } catch {};\
             $gpuPct = '-';\
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
             $tempC = '-';\
             try { \
               $m = (Get-Counter -Counter '\\Thermal Zone Information(*)\\Temperature' \
                -ErrorAction Stop).CounterSamples | \
                Measure-Object -Property CookedValue -Maximum;\
               if ($null -ne $m.Maximum) { $tempC = [int]($m.Maximum - 273) } } catch {};\
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
             \"$cpu $gpuPct $throttleV $tempC $esifC $critC\"",
        ])
        .output()?;
    parse_windows_reading(String::from_utf8_lossy(&output.stdout).trim())
}

fn parse_windows_reading(line: &str) -> Result<WindowsReading, ReadinessError> {
    let mut parts = line.split_whitespace();
    let cpu_raw = parts.next().ok_or(ReadinessError::Unavailable(
        "empty powershell readiness output",
    ))?;
    let gpu_raw = parts.next().ok_or(ReadinessError::Unavailable(
        "missing gpu field in readiness output",
    ))?;
    let throttle_raw = parts.next().ok_or(ReadinessError::Unavailable(
        "missing throttle field in readiness output",
    ))?;
    let temp_raw = parts.next().ok_or(ReadinessError::Unavailable(
        "missing temp field in readiness output",
    ))?;
    // Trailing fields: an older client's four-token output is still parsed, so
    // a mixed fleet mid-rollout degrades to "no ESIF sensor, no trip point"
    // rather than failing every readiness check.
    let esif_raw = parts.next().unwrap_or("-");
    let crit_raw = parts.next().unwrap_or("-");
    // The load gate is the one signal with no fallback, so a box that cannot
    // report it is unassessable rather than idle.
    if cpu_raw == "-" {
        return Err(ReadinessError::Unavailable(
            "cpu load counter unavailable on this box",
        ));
    }
    Ok(WindowsReading {
        cpu_load_pct: cpu_raw.parse().map_err(|_| ReadinessError::ParseFailure {
            raw: cpu_raw.to_string(),
        })?,
        gpu_compute_pct: parse_optional(gpu_raw)?,
        throttle_reasons: parse_optional(throttle_raw)?,
        zone_temp_c: parse_optional(temp_raw)?,
        // A 0 °C die is not physical on a running box; it is what an ESIF
        // instance with no sensor behind it reports. The PowerShell filters
        // those out, and so do we, because this parser also accepts output
        // from older clients that predate that filter. Left unfiltered, a
        // zero reads as "stone cold" and waves every cell through.
        esif_temp_c: parse_optional(esif_raw)?.filter(|c| *c > 0),
        critical_c: parse_optional(crit_raw)?.filter(|c| *c > 0),
    })
}

/// `-` → counter set absent on this box. Any other token must parse
/// as `u32`; garbage surfaces as `ParseFailure` rather than being
/// silently treated as absent.
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
    use rstest::rstest;

    use super::*;

    /// Builds a series from `"cpu gpu throttle zone esif crit"` lines.
    fn series(lines: &[&str]) -> Result<Vec<WindowsReading>, ReadinessError> {
        lines.iter().map(|l| parse_windows_reading(l)).collect()
    }

    #[rstest]
    #[case::all_present("0 0 0 40 41 105", 0, Some(0), Some(0), Some(40), Some(41), Some(105))]
    #[case::all_present_busy(
        "62 0 4 71 72 110",
        62,
        Some(0),
        Some(4),
        Some(71),
        Some(72),
        Some(110)
    )]
    #[case::no_gpu("5 - 0 52 - -", 5, None, Some(0), Some(52), None, None)]
    #[case::no_thermal_zones("3 91 - - - -", 3, Some(91), None, None, None, None)]
    #[case::cpu_only("4 - - - - -", 4, None, None, None, None, None)]
    // An older client emits four tokens. A mixed fleet mid-rollout should lose
    // the ESIF signal and the trip point, not fail every readiness check.
    #[case::four_token_client("5 0 0 52", 5, Some(0), Some(0), Some(52), None, None)]
    #[case::five_token_client("5 0 0 52 44", 5, Some(0), Some(0), Some(52), Some(44), None)]
    // An unpopulated `CriticalTripPoint` reads 0; that is not a 0 °C shutdown
    // trip, which would block every cell forever.
    #[case::zero_trip_point_dropped("5 0 0 52 44 0", 5, Some(0), Some(0), Some(52), Some(44), None)]
    fn windows_reading_parses(
        #[case] line: &str,
        #[case] want_cpu: u32,
        #[case] want_gpu: Option<u32>,
        #[case] want_throttle: Option<u32>,
        #[case] want_temp: Option<u32>,
        #[case] want_esif: Option<u32>,
        #[case] want_crit: Option<u32>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let r = parse_windows_reading(line)?;
        assert_eq!(r.cpu_load_pct, want_cpu);
        assert_eq!(r.gpu_compute_pct, want_gpu);
        assert_eq!(r.throttle_reasons, want_throttle);
        assert_eq!(r.zone_temp_c, want_temp);
        assert_eq!(r.esif_temp_c, want_esif);
        assert_eq!(r.critical_c, want_crit);
        Ok(())
    }

    #[rstest]
    #[case("", "empty powershell readiness output")]
    #[case("5", "missing gpu field in readiness output")]
    #[case("5 0", "missing throttle field in readiness output")]
    #[case("5 0 0", "missing temp field in readiness output")]
    // CPU is the one signal with no fallback: absent is unassessable, and must
    // not be confused with the `ParseFailure` a garbage token gets.
    #[case("- 0 0 50 - -", "cpu load counter unavailable on this box")]
    fn unavailable_errors_are_distinct(
        #[case] line: &str,
        #[case] want_msg: &'static str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let err = parse_windows_reading(line).err().ok_or("expected error")?;
        let ReadinessError::Unavailable(msg) = err else {
            return Err(format!("expected Unavailable, got {err:?}").into());
        };
        assert_eq!(msg, want_msg);
        Ok(())
    }

    /// `-` is the sentinel for "absent counter" on the optional signals;
    /// garbage in any slot must surface as `ParseFailure` rather than being
    /// silently treated as absent.
    #[rstest]
    #[case::garbage_cpu("notanumber 0 0 50", "notanumber")]
    #[case::garbage_gpu("5 notanumber 0 50", "notanumber")]
    #[case::garbage_throttle("5 0 weird 50", "weird")]
    #[case::garbage_temp("5 0 0 notanumber", "notanumber")]
    #[case::garbage_trip_point("5 0 0 50 44 notanumber", "notanumber")]
    fn garbage_field_is_parse_failure(
        #[case] line: &str,
        #[case] want_raw: &'static str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let err = parse_windows_reading(line).err().ok_or("expected error")?;
        let ReadinessError::ParseFailure { raw } = err else {
            return Err(format!("expected ParseFailure, got {err:?}").into());
        };
        assert_eq!(raw, want_raw);
        Ok(())
    }

    /// Each verdict is independent: any signal above its threshold flips that
    /// field's label, absent signals render `n/a` and don't gate.
    #[rstest]
    #[case::all_ok(
        "0 0 0 40 - -",
        "cpu=busy:0%(ok) gpu=compute:0%(ok) thermal=throttle:0(ok) temp=acpi:40°C(ok)"
    )]
    #[case::gpu_busy(
        "5 91 0 55 - -",
        "cpu=busy:5%(ok) gpu=compute:91%(busy) thermal=throttle:0(ok) temp=acpi:55°C(ok)"
    )]
    // 31 % is the top of the measured idle band on the 258V and must read ok;
    // 50 % is the bottom of its saturated band and must read busy.
    #[case::cpu_idle_high(
        "31 0 0 45 - -",
        "cpu=busy:31%(ok) gpu=compute:0%(ok) thermal=throttle:0(ok) temp=acpi:45°C(ok)"
    )]
    #[case::cpu_busy(
        "50 0 0 60 - -",
        "cpu=busy:50%(busy) gpu=compute:0%(ok) thermal=throttle:0(ok) temp=acpi:60°C(hot)"
    )]
    #[case::throttling(
        "5 0 8 50 - -",
        "cpu=busy:5%(ok) gpu=compute:0%(ok) thermal=throttle:8(throttling) temp=acpi:50°C(ok)"
    )]
    #[case::both_sensors(
        "2 0 0 28 57 -",
        "cpu=busy:2%(ok) gpu=compute:0%(ok) thermal=throttle:0(ok) temp=esif:57°C(ok),acpi:28°C(ok)"
    )]
    #[case::pathologically_hot(
        "2 0 0 28 88 -",
        "cpu=busy:2%(ok) gpu=compute:0%(ok) thermal=throttle:0(ok) temp=esif:88°C(hot),acpi:28°C(ok)"
    )]
    #[case::cpu_only_box("4 - - - - -", "cpu=busy:4%(ok) gpu=n/a thermal=n/a temp=n/a")]
    fn summary_renders_per_signal_verdict(
        #[case] line: &str,
        #[case] expected: &str,
    ) -> Result<(), ReadinessError> {
        assert_eq!(format_summary(&parse_windows_reading(line)?), expected);
        Ok(())
    }

    /// The real gmktec cooldown, sampled at 5 s: 98 °C to resting. Every
    /// window across the fast phase spans far more than [`FLAT_SPAN_C`], so a
    /// box on its way down cannot be mistaken for a settled one — including
    /// across 70 °C, which the descent passes between samples.
    #[rstest]
    #[case::descending_through_70(&["2 0 0 97 - -", "2 0 0 79 - -", "2 0 0 56 - -"], false)]
    #[case::still_falling(&["2 0 0 79 - -", "2 0 0 56 - -", "2 0 0 49 - -"], false)]
    #[case::knee(&["2 0 0 56 - -", "2 0 0 49 - -", "2 0 0 47 - -"], false)]
    // t=20/25/30 s on the measured curve: 47, 45, 44 — spread 3, the first
    // window that settles, releasing the gate 30 s after load stopped.
    #[case::settled_at_30s(&["2 0 0 47 - -", "2 0 0 45 - -", "2 0 0 44 - -"], true)]
    #[case::resting(&["2 0 0 34 - -", "2 0 0 33 - -", "2 0 0 33 - -"], true)]
    fn gmktec_cooldown_settles_only_past_the_knee(
        #[case] lines: &[&str],
        #[case] expected: bool,
    ) -> Result<(), ReadinessError> {
        assert_eq!(is_settled(&series(lines)?), expected);
        Ok(())
    }

    /// The real 258V cooldown, sampled at the ~7 s spacing it was captured at.
    /// Its ACPI zone is the constant 301 K (28 °C) throughout and is trivially
    /// flat, so it neither holds the gate nor releases it early.
    ///
    /// This box only clears the span because its decay is slow, not because it
    /// reached a plateau — see the module docs.
    #[rstest]
    #[case::at_load(&["2 0 0 28 55 -", "2 0 0 28 51 -", "2 0 0 28 49 -"], false)]
    #[case::settled_at_14s(&["2 0 0 28 51 -", "2 0 0 28 49 -", "2 0 0 28 48 -"], true)]
    #[case::resting(&["2 0 0 28 43 -", "2 0 0 28 42 -", "2 0 0 28 42 -"], true)]
    fn devcloud_cooldown_settles_beside_a_stuck_zone(
        #[case] lines: &[&str],
        #[case] expected: bool,
    ) -> Result<(), ReadinessError> {
        assert_eq!(is_settled(&series(lines)?), expected);
        Ok(())
    }

    /// A settled sensor cannot release the gate while another is still moving.
    #[test]
    fn every_sensor_must_be_flat() -> Result<(), ReadinessError> {
        let lines = ["2 0 0 33 88 -", "2 0 0 33 61 -", "2 0 0 33 49 -"];
        assert!(!is_settled(&series(&lines)?));
        Ok(())
    }

    #[rstest]
    // Fewer samples than the window is not a series, only readings.
    #[case::one_sample(&["2 0 0 33 - -"], false)]
    #[case::two_samples(&["2 0 0 33 - -", "2 0 0 33 - -"], false)]
    // A box with no temperature counter at all costs the window and nothing
    // more — there is no decay to observe.
    #[case::no_sensors(&["2 0 0 - - -", "2 0 0 - - -", "2 0 0 - - -"], true)]
    // A counter that drops out mid-window is not judged; assuming it was
    // moving would hold the gate for the full cap on every cell.
    #[case::intermittent(&["2 0 0 90 - -", "2 0 0 - - -", "2 0 0 40 - -"], true)]
    // Only the most recent window counts, so an earlier descent does not keep
    // a since-settled box waiting.
    #[case::older_samples_ignored(
        &["2 0 0 98 - -", "2 0 0 60 - -", "2 0 0 44 - -", "2 0 0 43 - -", "2 0 0 43 - -"],
        true
    )]
    fn settling_needs_a_full_window_of_answers(
        #[case] lines: &[&str],
        #[case] expected: bool,
    ) -> Result<(), ReadinessError> {
        assert_eq!(is_settled(&series(lines)?), expected);
        Ok(())
    }

    /// Boundary: the span is inclusive, so exactly [`FLAT_SPAN_C`] settles and
    /// one degree more does not.
    #[rstest]
    #[case::exactly_the_span(&["2 0 0 47 - -", "2 0 0 45 - -", "2 0 0 44 - -"], true)]
    #[case::one_over(&["2 0 0 48 - -", "2 0 0 45 - -", "2 0 0 44 - -"], false)]
    fn flat_span_boundary(
        #[case] lines: &[&str],
        #[case] expected: bool,
    ) -> Result<(), ReadinessError> {
        assert_eq!(is_settled(&series(lines)?), expected);
        Ok(())
    }

    /// The critical check reads the box's own trip point, so the same reading
    /// blocks on one chassis and not another. Measured trips: 105 °C on the
    /// 258V, 110 °C on the gmktec.
    #[rstest]
    #[case::under_both("2 0 0 88 - 105", false)]
    #[case::at_the_258v_limit("2 0 0 90 - 105", true)]
    // 90 °C clears the gmktec's higher trip but not the 258V's.
    #[case::same_reading_under_the_gmktec_limit("2 0 0 90 - 110", false)]
    #[case::at_the_gmktec_limit("2 0 0 95 - 110", true)]
    // A box that reports no trip point falls back to a fixed limit rather than
    // losing the check entirely.
    #[case::fallback_under("2 0 0 94 - -", false)]
    #[case::fallback_over("2 0 0 95 - -", true)]
    // Either sensor can trip it; the stuck-low zone must not mask a hot die.
    #[case::esif_over_zone_stuck_low("2 0 0 28 95 105", true)]
    #[case::no_sensor_at_all("2 0 0 - - 105", false)]
    fn critical_uses_the_platform_trip_point(
        #[case] line: &str,
        #[case] expected: bool,
    ) -> Result<(), ReadinessError> {
        assert_eq!(over_critical(&parse_windows_reading(line)?), expected);
        Ok(())
    }

    /// A sensor pinned at its limiter is a plateau, so the decay test on its
    /// own calls it settled — the gmktec held exactly 98 °C across eight
    /// consecutive polls under load, and its CPU counter dipped to 29 % as the
    /// load wound down. Only the critical check separates that from rest.
    #[test]
    fn saturation_reads_as_settled_and_is_held_by_the_critical_check() -> Result<(), ReadinessError>
    {
        let pinned = series(&["75 0 0 98 - 110", "76 0 0 98 - 110", "29 0 0 98 - 110"])?;
        assert!(is_settled(&pinned));
        let last = pinned
            .last()
            .ok_or(ReadinessError::Unavailable("empty series"))?;
        assert!(over_critical(last));
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
    /// A sensor that drops out mid-series is not stuck, it is intermittent.
    #[case::sensor_dropped_out(
        &[Some(28), None, Some(28)],
        &[(60, Some(80)), (2, Some(5)), (1, Some(0))],
        None
    )]
    /// Counter jitter is not the box going from busy to idle. These are real
    /// idle CPU samples from a 258V; comparing consecutive readings for
    /// inequality would call this "moved" and accuse a working sensor.
    #[case::load_only_jittered(
        &[Some(45), Some(45), Some(45), Some(45)],
        &[(20, Some(0)), (14, Some(0)), (18, Some(0)), (31, Some(0))],
        None
    )]
    /// Busy for the whole wait is not busy-then-idle either — a box that never
    /// settled tells us nothing about whether the sensor tracks cooling.
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
        assert_eq!(held_temperature_if_stuck(temps, loads), expected);
    }

    /// A missing sensor drops out of that poll; the others still gate.
    #[rstest]
    #[case::esif_missing("2 0 0 88 - -", vec![("acpi", 88)])]
    #[case::zone_missing("2 0 0 - 57 -", vec![("esif", 57)])]
    #[case::both_present("2 0 0 28 57 -", vec![("esif", 57), ("acpi", 28)])]
    #[case::neither("2 0 0 - - -", vec![])]
    // A sensorless ESIF instance reports exactly 0; the parser drops it, so it
    // never reaches the gate as a spuriously cold reading.
    #[case::zero_esif_dropped("2 0 0 28 0 -", vec![("acpi", 28)])]
    fn temperatures_lists_only_the_sensors_that_answered(
        #[case] line: &str,
        #[case] expected: Vec<(&str, u32)>,
    ) -> Result<(), ReadinessError> {
        let got: Vec<_> = parse_windows_reading(line)?.temperatures().collect();
        assert_eq!(got, expected);
        Ok(())
    }

    /// Each sensor is judged on its own series, so a working sensor next to a
    /// stuck one does not mask it.
    #[test]
    fn a_stuck_sensor_is_reported_even_beside_a_working_one() -> Result<(), ReadinessError> {
        let samples = series(&["60 80 0 28 44 -", "2 0 0 28 58 -", "1 0 0 28 46 -"])?;
        let loads: Vec<(u32, Option<u32>)> = samples
            .iter()
            .map(|s| (s.cpu_load_pct, s.gpu_compute_pct))
            .collect();
        let zone: Vec<Option<u32>> = samples.iter().map(|s| s.zone_temp_c).collect();
        let esif: Vec<Option<u32>> = samples.iter().map(|s| s.esif_temp_c).collect();
        assert_eq!(held_temperature_if_stuck(&zone, &loads), Some(28));
        assert_eq!(held_temperature_if_stuck(&esif, &loads), None);
        Ok(())
    }
}
