//! macOS readiness: the thermal-pressure level must read `nominal`, the die
//! must be under [`MAX_DIE_C`], and the machine under [`MAX_BUSY_CORES`].
//!
//! ## The pressure level is a poor signal, kept because it is conservative
//!
//! `com.apple.system.thermalpressurelevel` is not a temperature. Measured with
//! `tools/macos-thermal-probe` on a MacBook Neo (A18 Pro), it engages 3-4 s into
//! a load and clears a fixed ~318 s after the load stops, whatever the heat
//! input (10 s or 123 s of load) or cooling. On an M4 Max it never left
//! `nominal` while the die rose 49 to 60 °C.
//!
//! ## The die criterion
//!
//! On the M5 Max the level engages only after minutes of continuous load, and
//! the gaps between cells reset that clock, so a batch heats underneath it.
//! Die temperature is hysteretic, but this gate samples only the cooling
//! branch: it runs between reps on a machine [`MAX_BUSY_CORES`] calls idle.
//! Measurements: `docs/methodology/macbook-m5-thermal-behavior.md`.
//!
//! ## Reading the signals
//!
//! The level is `OSThermalPressureLevel` (0 nominal ... 4 sleeping), which maps
//! neither to iOS's numbering nor to `ProcessInfo.thermalState`, so it has no
//! mapping to the recorded `apple_thermal_state` series. It is read through
//! notify(3) rather than `swift -e` (~120 ms warm, ~0.7 s cold), which would
//! itself load the CPU the sibling check measures.
//!
//! CPU load is a busy ratio from Mach tick counters, not `vm.loadavg`, whose
//! 1-minute EMA sits near the threshold on an idle developer Mac.

use std::ffi::{CStr, c_char, c_int};
use std::time::Duration;

use super::wait::{Probe, wait_loop};
use super::{
    ReadinessCallbacks, ReadinessConfiguration, ReadinessError, ReadinessPlatform, ThermalGate,
};

/// `NOTIFY_STATUS_OK` from `<notify.h>`.
const NOTIFY_STATUS_OK: u32 = 0;
/// `kOSThermalPressureLevelNominal`.
const PRESSURE_NOMINAL: u64 = 0;

// libSystem is always linked on macOS, so these need no `#[link]`.
unsafe extern "C" {
    fn notify_register_check(name: *const c_char, out_token: *mut c_int) -> u32;
    fn notify_get_state(token: c_int, state: *mut u64) -> u32;
    fn notify_cancel(token: c_int) -> u32;

    /// From `<libkern/OSThermalNotification.h>`; see [`thermal_pressure_name`].
    static kOSThermalNotificationPressureLevelName: *const c_char;
}

/// The thermal-pressure notify(3) name, from Apple's exported symbol rather
/// than a literal.
///
/// `notify_register_check` accepts any name and `notify_get_state` then reads
/// 0 (`nominal`) forever, so a typo would make the gate fail open; linking the
/// symbol turns that into a link error. The similar undocumented
/// `com.apple.system.thermalpressure` stayed 0 on the Neo while this one
/// reached `moderate`.
fn thermal_pressure_name() -> &'static CStr {
    // SAFETY: the symbol is a NUL-terminated C string constant with static
    // storage duration in libSystem, so the pointer is valid for the
    // process lifetime and the `'static` lifetime is sound.
    unsafe { CStr::from_ptr(kOSThermalNotificationPressureLevelName) }
}

/// The one override of [`crate::DEFAULT_MAX_WAIT`]: the pressure level's
/// measured ~318 s hold-off (`docs/methodology/macbook-neo-thermal-behavior.md`)
/// exceeds the shared 300 s, which timed out and failed cells on the fanless Neo.
pub(super) const DEFAULT_MAX_WAIT: Duration = Duration::from_secs(420);
/// Busy cores (`busy_ratio × ncpu`) a machine must stay under.
///
/// Not a fraction of capacity: 0.30 all-core on an 18-core M5 Max needs ~5.4
/// saturated cores to trip, and one host read `ok` at 13 % for five days with a
/// stray `llama-cli` pinning a core (2.36 busy cores). Idle on these hosts is
/// 0.23-0.37 busy cores.
const MAX_BUSY_CORES: f64 = 1.0;

/// Die temperature a machine must be under. 50 °C is where full throughput
/// returns, measured on three M5 Max hosts.
///
/// Calibrated for the M5 Max only: `die_temp_max_c` is a max over a per-host
/// sensor set. Other machine types set `[readiness.macos] die_below_c`.
const MAX_DIE_C: f64 = 50.0;

/// The gate's own numbers, or the ones the host's `[readiness.macos]` table
/// sets.
#[derive(Debug, Clone, Copy)]
struct Bounds {
    die_below_c: f64,
    /// The host set `die_below_c` itself, so an unreadable die holds the gate.
    die_set: bool,
    busy_below_cores: f64,
}

impl Bounds {
    fn new(configuration: Option<&ReadinessConfiguration>) -> Self {
        let config = configuration.and_then(ReadinessConfiguration::macos);
        let die = config.and_then(|c| c.die_below_c);
        let cores = config.and_then(|c| c.cpu_busy_below_cores);
        Self {
            die_below_c: die.map_or(MAX_DIE_C, |c| f64::from(c.get())),
            die_set: die.is_some(),
            busy_below_cores: cores.map_or(MAX_BUSY_CORES, |n| f64::from(n.get())),
        }
    }

    /// An unreadable die passes the gate's own [`MAX_DIE_C`] (calibrated for
    /// one machine type) and holds a bound the host set (its own calibration).
    fn die_ok(self, die_c: Option<f64>) -> bool {
        die_c.map_or(!self.die_set, |c| c < self.die_below_c)
    }

    fn cpu_ok(self, busy_cores: f64) -> bool {
        busy_cores < self.busy_below_cores
    }

    fn ready(
        self,
        level: Option<u64>,
        die_c: Option<f64>,
        busy_cores: f64,
        thermal: ThermalGate,
    ) -> bool {
        let thermal_ok =
            thermal == ThermalGate::Skip || (level == Some(PRESSURE_NOMINAL) && self.die_ok(die_c));
        thermal_ok && self.cpu_ok(busy_cores)
    }
}

pub(super) fn wait_until_ready(
    max_wait: Duration,
    thermal: ThermalGate,
    configuration: Option<&ReadinessConfiguration>,
    callbacks: &ReadinessCallbacks<'_>,
) -> Result<(), ReadinessError> {
    let bounds = Bounds::new(configuration);
    wait_loop(ReadinessPlatform::MacOs, max_wait, callbacks, || {
        // Read even when skipped, for the log line. The recorded telemetry is
        // sampled separately, so a skipped gate's probe failure must not fail
        // the cell.
        let level = match read_thermal_pressure() {
            Ok(level) => Some(level),
            Err(e) if thermal == ThermalGate::Skip => {
                tracing::warn!(
                    error = %e,
                    "thermal probe unreadable; the thermal gate is skipped for this wait, so \
                     continuing without it"
                );
                None
            }
            Err(e) => return Err(e),
        };
        let busy_cores = read_busy_cores()?;
        let die_c = pipette_device::die_temp_max_c();
        Ok(Probe {
            ready: bounds.ready(level, die_c, busy_cores, thermal),
            summary: format_summary(level, die_c, busy_cores, thermal, bounds),
        })
    })
}

/// Carries the raw level beside its name, marks a waived gate, and keeps an
/// unreadable level (`None`, only when skipped) distinct from `nominal`.
fn format_summary(
    level: Option<u64>,
    die_c: Option<f64>,
    busy_cores: f64,
    thermal: ThermalGate,
    bounds: Bounds,
) -> String {
    let cpu_verdict = if bounds.cpu_ok(busy_cores) {
        "ok"
    } else {
        "busy"
    };
    let skipped = if thermal == ThermalGate::Skip {
        "[SKIPPED]"
    } else {
        ""
    };
    let thermal_field = match level {
        Some(level) => format!("pressure:{level}({})", format_pressure_word(level)),
        None => "pressure:unreadable".to_string(),
    };
    let die_field = match die_c {
        Some(c) if bounds.die_ok(Some(c)) => format!(" die:{c:.0}C(ok)"),
        Some(c) => format!(" die:{c:.0}C(hot)"),
        None if thermal == ThermalGate::Skip || bounds.die_ok(None) => {
            " die:unreadable".to_string()
        }
        None => " die:unreadable(held)".to_string(),
    };
    format!(
        "thermal={thermal_field}{skipped}{die_field} cpu=busy:{busy_cores:.1}cores({cpu_verdict})"
    )
}

/// `OSThermalPressureLevel` names in the macOS numbering.
fn format_pressure_word(level: u64) -> &'static str {
    match level {
        0 => "nominal",
        1 => "moderate",
        2 => "heavy",
        3 => "trapping",
        4 => "sleeping",
        _ => "unknown",
    }
}

/// The current thermal-pressure level, 0 (coolest) to 4.
///
/// A fresh token per call still reads the live value, since the state belongs
/// to the name. `notify_get_state`, not `notify_check`, which reports only
/// whether the notification fired since the last check.
fn read_thermal_pressure() -> Result<u64, ReadinessError> {
    let mut token: c_int = 0;
    // SAFETY: the name is a NUL-terminated static C string valid for the
    // call, and `token` is an initialized `c_int` we own.
    let rc = unsafe { notify_register_check(thermal_pressure_name().as_ptr(), &raw mut token) };
    if rc != NOTIFY_STATUS_OK {
        tracing::warn!(
            notify_status = rc,
            "notify_register_check(thermalpressurelevel) failed"
        );
        return Err(ReadinessError::Unavailable(
            "notify_register_check(com.apple.system.thermalpressurelevel) failed",
        ));
    }
    let mut level: u64 = 0;
    // SAFETY: `token` was registered successfully above, and `level` is
    // an initialized `u64` we own.
    let rc = unsafe { notify_get_state(token, &raw mut level) };
    // Released even on a failed read, or a failing probe leaks one per poll.
    // SAFETY: `token` was registered successfully above.
    let _ = unsafe { notify_cancel(token) };
    if rc != NOTIFY_STATUS_OK {
        tracing::warn!(
            notify_status = rc,
            "notify_get_state(thermalpressurelevel) failed"
        );
        return Err(ReadinessError::Unavailable(
            "notify_get_state(com.apple.system.thermalpressurelevel) failed",
        ));
    }
    Ok(level)
}

/// Do not shorten below ~800 ms: `HOST_CPU_LOAD_INFO` refreshes about once a
/// second, so a shorter window often sees no tick delta, and the samples that
/// do are biased high (a 200 ms window read 42 % on a machine at 23 %).
/// Measured zero-delta rates: 17/30 at 200 ms, 10/30 at 400 ms, 4/30 at 600 ms,
/// 0/30 at 800 ms and above. 1 s gave the tightest spread (sd 0.9 %).
const CPU_SAMPLE_WINDOW: Duration = Duration::from_secs(1);

/// One `HOST_CPU_LOAD_INFO` snapshot, or `None` if the call failed.
fn cpu_ticks() -> Option<libc::host_cpu_load_info> {
    let mut info = libc::host_cpu_load_info {
        cpu_ticks: [0; libc::CPU_STATE_MAX as usize],
    };
    let mut count = libc::mach_msg_type_number_t::try_from(
        std::mem::size_of::<libc::host_cpu_load_info>() / std::mem::size_of::<libc::integer_t>(),
    )
    .ok()?;
    // SAFETY: `mach_host_self` has no preconditions; it returns a send right
    // we own and release below.
    let host = unsafe { mach_host_self() };
    // SAFETY: `host_statistics` writes `count` integers into `info`, and
    // `count` is derived from `info`'s own size, so the buffer cannot be
    // overrun.
    let status = unsafe {
        libc::host_statistics(
            host,
            libc::HOST_CPU_LOAD_INFO,
            std::ptr::from_mut(&mut info).cast::<libc::integer_t>(),
            &raw mut count,
        )
    };
    // `mach_host_self` hands out a reference per call; not releasing it leaks
    // one port right per check.
    // SAFETY: `mach_task_self_` is initialized by libSystem before `main`
    // and never written after.
    let task = unsafe { mach_task_self_ };
    // SAFETY: `host` is a send right in `task` that we own and do not use
    // after this.
    unsafe { mach_port_deallocate(task, host) };
    (status == libc::KERN_SUCCESS).then_some(info)
}

// Not `libc`'s Mach wrappers, which are deprecated in favor of the `mach2`
// crate; three symbols are not worth a dependency.
unsafe extern "C" {
    fn mach_host_self() -> libc::mach_port_t;
    fn mach_port_deallocate(
        task: libc::mach_port_t,
        name: libc::mach_port_t,
    ) -> libc::kern_return_t;
    /// The global behind C's `mach_task_self()` macro; never written after
    /// process start.
    static mach_task_self_: libc::mach_port_t;
}

/// The all-core busy ratio times the online CPU count, the unit of
/// [`MAX_BUSY_CORES`].
fn read_busy_cores() -> Result<f64, ReadinessError> {
    Ok(read_cpu_load()? * online_cpus()?)
}

/// An unreadable count is an error, not a default: assuming 1 caps busy cores
/// at 1.0, so the gate would stop rejecting busy machines.
fn online_cpus() -> Result<f64, ReadinessError> {
    // SAFETY: `sysconf` takes an integer name and returns a long; no pointers.
    let n = unsafe { libc::sysconf(libc::_SC_NPROCESSORS_ONLN) };
    if n > 0 {
        Ok(n as f64)
    } else {
        Err(ReadinessError::Unavailable(
            "sysconf(_SC_NPROCESSORS_ONLN) reported no online CPUs",
        ))
    }
}

/// Busy ratio across all cores over [`CPU_SAMPLE_WINDOW`], from Mach tick
/// counters rather than `top -l 2` (~1.88 s per check against ~1.0 s).
///
/// Resamples once, for a window that spanned no counter refresh. A failing
/// `host_statistics` gets no fallback: it is public, unprivileged API, so a
/// host where it breaks should be heard about.
fn read_cpu_load() -> Result<f64, ReadinessError> {
    mach_cpu_busy()
        .or_else(|| {
            tracing::debug!("the CPU tick counters did not advance; resampling");
            mach_cpu_busy()
        })
        .ok_or(ReadinessError::Unavailable(
            "HOST_CPU_LOAD_INFO ticks did not advance across two windows",
        ))
}

/// `None` if either read failed or the counters did not advance.
fn mach_cpu_busy() -> Option<f64> {
    let before = cpu_ticks()?;
    std::thread::sleep(CPU_SAMPLE_WINDOW);
    let after = cpu_ticks()?;
    busy_from_ticks(&before.cpu_ticks, &after.cpu_ticks)
}

fn busy_from_ticks(
    before: &[u32; libc::CPU_STATE_MAX as usize],
    after: &[u32; libc::CPU_STATE_MAX as usize],
) -> Option<f64> {
    const IDLE: usize = libc::CPU_STATE_IDLE as usize;
    // 32-bit monotonic counters: `wrapping_sub` reads a wrap correctly.
    let delta = |after: &u32, before: &u32| u64::from(after.wrapping_sub(*before));
    let total: u64 = after.iter().zip(before).map(|(a, b)| delta(a, b)).sum();
    if total == 0 {
        return None; // counters didn't refresh inside the window
    }
    let idle = delta(&after[IDLE], &before[IDLE]);
    Some((1.0 - idle as f64 / total as f64).clamp(0.0, 1.0))
}

#[cfg(test)]
mod tests {
    use anyhow::Context;
    use rstest::rstest;

    use super::*;
    use crate::configuration::table;

    /// The FFI end to end against the live kernel; only the read's success is
    /// asserted, since the level depends on the host.
    #[test]
    fn thermal_pressure_reads_from_kernel() -> anyhow::Result<()> {
        let _level = read_thermal_pressure()?;
        Ok(())
    }

    /// Linking prevents a typo, not Apple repointing the symbol at a
    /// differently encoded notification.
    #[test]
    fn linked_name_is_the_pressure_level_notification() -> anyhow::Result<()> {
        assert_eq!(
            thermal_pressure_name().to_str()?,
            "com.apple.system.thermalpressurelevel",
        );
        Ok(())
    }

    /// The hazard the linked symbol removes. If this starts failing, notify
    /// grew name validation.
    #[test]
    fn notify_accepts_a_bogus_name_and_reports_nominal() -> anyhow::Result<()> {
        let mut token: c_int = 0;
        // SAFETY: a NUL-terminated literal and a `c_int` we own.
        let rc = unsafe {
            notify_register_check(
                c"com.example.pipette.no.such.notification".as_ptr(),
                &raw mut token,
            )
        };
        assert_eq!(rc, NOTIFY_STATUS_OK, "notify rejected a bogus name");
        let mut level = u64::MAX;
        // SAFETY: `token` was registered successfully above.
        let read = unsafe { notify_get_state(token, &raw mut level) };
        // SAFETY: `token` was registered successfully above.
        let _ = unsafe { notify_cancel(token) };
        assert_eq!(read, NOTIFY_STATUS_OK, "get_state failed on a bogus name");
        assert_eq!(
            level, PRESSURE_NOMINAL,
            "a bogus name no longer reads as nominal",
        );
        Ok(())
    }

    #[test]
    fn deadline_clears_the_measured_hold_off() {
        assert!(
            DEFAULT_MAX_WAIT >= Duration::from_secs(318),
            "DEFAULT_MAX_WAIT {DEFAULT_MAX_WAIT:?} is below the measured 318s hold-off",
        );
    }

    /// Judged in cores: every 18-core reading here is under 0.30 all-core.
    #[rstest]
    #[case(4.0, 1.0, false)]
    #[case(18.0, 1.0, false)]
    #[case(18.0, 2.36, false)]
    #[case(18.0, 0.23, true)]
    #[case(18.0, 0.37, true)]
    fn one_pinned_core_fails_readiness_at_any_core_count(
        #[case] cores: f64,
        #[case] busy_cores: f64,
        #[case] want_ok: bool,
    ) {
        assert_eq!(Bounds::new(None).cpu_ok(busy_cores), want_ok);
        let as_ratio = busy_cores / cores;
        if cores >= 18.0 {
            assert!(as_ratio < 0.30, "{as_ratio} would have passed the old gate");
        }
    }

    /// With no table, an unreadable die (`None`) passes.
    #[rstest]
    #[case(Some(35.0), true)]
    #[case(Some(49.0), true)]
    #[case(Some(50.0), false)]
    #[case(Some(62.0), false)]
    #[case(Some(72.0), false)]
    #[case(None, true)]
    fn die_criterion_rejects_a_hot_machine_the_enum_calls_nominal(
        #[case] die_c: Option<f64>,
        #[case] want_ok: bool,
    ) {
        assert_eq!(Bounds::new(None).die_ok(die_c), want_ok);
    }

    /// The host's numbers replace the gate's own in the same strict
    /// comparison, each key alone; the pressure level stays required.
    #[rstest]
    #[case::own_die_holds(None, ThermalGate::Enforce, 0, Some(52.0), 0.4, false)]
    #[case::empty_table(Some(""), ThermalGate::Enforce, 0, Some(52.0), 0.4, false)]
    #[case::restated_die(
        Some("die_below_c = 50\ncpu_busy_below_cores = 1"),
        ThermalGate::Enforce,
        0,
        Some(50.0),
        0.4,
        false
    )]
    #[case::restated_cores(
        Some("die_below_c = 50\ncpu_busy_below_cores = 1"),
        ThermalGate::Enforce,
        0,
        Some(40.0),
        1.0,
        false
    )]
    #[case::die_raised(
        Some("die_below_c = 60"),
        ThermalGate::Enforce,
        0,
        Some(52.0),
        0.4,
        true
    )]
    #[case::die_raised_is_strict(
        Some("die_below_c = 60"),
        ThermalGate::Enforce,
        0,
        Some(60.0),
        0.4,
        false
    )]
    #[case::die_lowered(
        Some("die_below_c = 45"),
        ThermalGate::Enforce,
        0,
        Some(47.0),
        0.4,
        false
    )]
    #[case::die_key_leaves_cores(
        Some("die_below_c = 60"),
        ThermalGate::Enforce,
        0,
        Some(52.0),
        1.8,
        false
    )]
    #[case::cores_raised(
        Some("cpu_busy_below_cores = 4"),
        ThermalGate::Enforce,
        0,
        Some(40.0),
        1.8,
        true
    )]
    #[case::cores_raised_is_strict(
        Some("cpu_busy_below_cores = 4"),
        ThermalGate::Enforce,
        0,
        Some(40.0),
        4.0,
        false
    )]
    #[case::cores_key_leaves_die(
        Some("cpu_busy_below_cores = 4"),
        ThermalGate::Enforce,
        0,
        Some(52.0),
        1.8,
        false
    )]
    #[case::pressure_still_required(
        Some("die_below_c = 60\ncpu_busy_below_cores = 4"),
        ThermalGate::Enforce,
        1,
        Some(40.0),
        0.4,
        false
    )]
    // An unreadable die waives the gate's own number, and holds the host's.
    #[case::unread_die_waives_own(None, ThermalGate::Enforce, 0, None, 0.4, true)]
    #[case::unread_die_holds_a_set_bound(
        Some("die_below_c = 60"),
        ThermalGate::Enforce,
        0,
        None,
        0.4,
        false
    )]
    #[case::unread_die_waives_an_empty_table(Some(""), ThermalGate::Enforce, 0, None, 0.4, true)]
    #[case::unread_die_waives_a_cores_only_table(
        Some("cpu_busy_below_cores = 4"),
        ThermalGate::Enforce,
        0,
        None,
        0.4,
        true
    )]
    // Skipping thermal skips the die bound, set or not; cores still apply.
    #[case::skip_ignores_die(Some("die_below_c = 45"), ThermalGate::Skip, 2, Some(70.0), 0.4, true)]
    #[case::skip_ignores_unread(Some("die_below_c = 45"), ThermalGate::Skip, 0, None, 0.4, true)]
    #[case::skip_keeps_cores(
        Some("cpu_busy_below_cores = 4"),
        ThermalGate::Skip,
        0,
        Some(70.0),
        4.2,
        false
    )]
    fn the_hosts_table_moves_the_gate(
        #[case] keys: Option<&str>,
        #[case] thermal: ThermalGate,
        #[case] level: u64,
        #[case] die_c: Option<f64>,
        #[case] busy_cores: f64,
        #[case] want_ready: bool,
    ) -> anyhow::Result<()> {
        let config = keys
            .map(|keys| table(ReadinessPlatform::MacOs, keys))
            .transpose()?;
        let bounds = Bounds::new(config.as_ref());
        assert_eq!(
            bounds.ready(Some(level), die_c, busy_cores, thermal),
            want_ready
        );
        Ok(())
    }

    #[rstest]
    #[case(0, "nominal")]
    #[case(1, "moderate")]
    #[case(2, "heavy")]
    #[case(3, "trapping")]
    #[case(4, "sleeping")]
    #[case(5, "unknown")]
    fn pressure_word_covers_all_macos_levels(#[case] level: u64, #[case] want: &str) {
        assert_eq!(format_pressure_word(level), want);
    }

    #[rstest]
    #[case::nominal(
        Some(0),
        Some(40.0),
        0.4,
        ThermalGate::Enforce,
        None,
        "thermal=pressure:0(nominal) die:40C(ok) cpu=busy:0.4cores(ok)"
    )]
    #[case::heavy_and_busy(
        Some(2),
        Some(40.0),
        6.2,
        ThermalGate::Enforce,
        None,
        "thermal=pressure:2(heavy) die:40C(ok) cpu=busy:6.2cores(busy)"
    )]
    #[case::unknown_level(
        Some(9),
        Some(40.0),
        0.4,
        ThermalGate::Enforce,
        None,
        "thermal=pressure:9(unknown) die:40C(ok) cpu=busy:0.4cores(ok)"
    )]
    #[case::skipped(
        Some(2),
        Some(40.0),
        0.4,
        ThermalGate::Skip,
        None,
        "thermal=pressure:2(heavy)[SKIPPED] die:40C(ok) cpu=busy:0.4cores(ok)"
    )]
    #[case::unreadable_pressure(
        None,
        Some(40.0),
        0.4,
        ThermalGate::Skip,
        None,
        "thermal=pressure:unreadable[SKIPPED] die:40C(ok) cpu=busy:0.4cores(ok)"
    )]
    #[case::unreadable_die_waived(
        Some(0),
        None,
        0.4,
        ThermalGate::Enforce,
        None,
        "thermal=pressure:0(nominal) die:unreadable cpu=busy:0.4cores(ok)"
    )]
    #[case::under_the_table(
        Some(0),
        Some(52.0),
        1.8,
        ThermalGate::Enforce,
        Some("die_below_c = 60\ncpu_busy_below_cores = 4"),
        "thermal=pressure:0(nominal) die:52C(ok) cpu=busy:1.8cores(ok)"
    )]
    #[case::over_the_table(
        Some(0),
        Some(61.0),
        4.2,
        ThermalGate::Enforce,
        Some("die_below_c = 60\ncpu_busy_below_cores = 4"),
        "thermal=pressure:0(nominal) die:61C(hot) cpu=busy:4.2cores(busy)"
    )]
    #[case::unreadable_die_held(
        Some(0),
        None,
        0.4,
        ThermalGate::Enforce,
        Some("die_below_c = 60\ncpu_busy_below_cores = 4"),
        "thermal=pressure:0(nominal) die:unreadable(held) cpu=busy:0.4cores(ok)"
    )]
    #[case::unreadable_die_skipped(
        Some(0),
        None,
        0.4,
        ThermalGate::Skip,
        Some("die_below_c = 60\ncpu_busy_below_cores = 4"),
        "thermal=pressure:0(nominal)[SKIPPED] die:unreadable cpu=busy:0.4cores(ok)"
    )]
    fn the_summary_line(
        #[case] level: Option<u64>,
        #[case] die_c: Option<f64>,
        #[case] busy_cores: f64,
        #[case] thermal: ThermalGate,
        #[case] keys: Option<&str>,
        #[case] want: &str,
    ) -> anyhow::Result<()> {
        let config = keys
            .map(|keys| table(ReadinessPlatform::MacOs, keys))
            .transpose()?;
        assert_eq!(
            format_summary(
                level,
                die_c,
                busy_cores,
                thermal,
                Bounds::new(config.as_ref())
            ),
            want
        );
        Ok(())
    }

    /// Order is `[user, system, idle, nice]` per `CPU_STATE_*`.
    #[rstest]
    #[case([0, 0, 0, 0], [50, 50, 100, 0], Some(0.5))]
    #[case([0, 0, 0, 0], [0, 0, 100, 0], Some(0.0))]
    #[case([0, 0, 0, 0], [70, 30, 0, 0], Some(1.0))]
    // Nice time counts as busy, not idle.
    #[case([0, 0, 0, 0], [0, 0, 50, 50], Some(0.5))]
    // Counters did not refresh inside the window: no ratio, so the caller
    // resamples.
    #[case([9, 9, 9, 9], [9, 9, 9, 9], None)]
    // 3 busy ticks and 6 idle across a 32-bit wrap: 1/3 busy.
    #[case([u32::MAX - 1, 0, u32::MAX - 4, 0], [1, 0, 1, 0], Some(1.0 / 3.0))]
    fn busy_from_ticks_handles_deltas_wraps_and_stalls(
        #[case] before: [u32; libc::CPU_STATE_MAX as usize],
        #[case] after: [u32; libc::CPU_STATE_MAX as usize],
        #[case] want: Option<f64>,
    ) -> anyhow::Result<()> {
        let got = busy_from_ticks(&before, &after);
        match want {
            Some(want) => {
                let got = got.context("expected a busy ratio")?;
                assert!((got - want).abs() < 1e-9, "got {got}, want {want}");
            }
            None => assert!(got.is_none(), "expected no ratio, got {got:?}"),
        }
        Ok(())
    }

    #[test]
    fn cpu_sample_window_clears_the_counter_refresh_cadence() {
        assert!(
            CPU_SAMPLE_WINDOW >= Duration::from_millis(800),
            "CPU_SAMPLE_WINDOW {CPU_SAMPLE_WINDOW:?} is below the measured refresh cadence",
        );
    }

    /// The live Mach path, resample included.
    #[test]
    fn cpu_load_returns_a_plausible_ratio_on_this_host() -> anyhow::Result<()> {
        let busy = read_cpu_load()?;
        assert!(
            (0.0..=1.0).contains(&busy),
            "CPU busy ratio out of range: {busy}",
        );
        Ok(())
    }
}
