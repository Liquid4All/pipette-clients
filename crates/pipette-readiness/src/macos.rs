//! macOS readiness — waits for the OS thermal-pressure level to read
//! `nominal` and for Mach CPU tick counters to show the machine idle. Both
//! before a measurement starts.
//!
//! ## The thermal criterion is known to be poor, and deliberately kept
//!
//! `com.apple.system.thermalpressurelevel` (the notification behind
//! `ProcessInfo.thermalState`) is **not a temperature signal**. Measured with
//! `tools/macos-thermal-probe` on a MacBook Neo (A18 Pro), it is a fixed
//! hold-off anchored to the moment the CPU went quiet.
//!
//! (That Neo is thermally modified — a passive pad inside the case and an
//! active pad under it — so the *temperatures* below describe the rig, not a
//! stock machine. The enum's behavior does not depend on any of that, which is
//! rather the point.)
//!
//! | load | peak die | die back at baseline | enum cleared |
//! |-------|---------:|---------------------:|-----------------:|
//! | 10 s | 51.5 °C | — | 317 s after stop |
//! | 123 s | 59.8 °C | 147 s after stop | **318 s after stop** |
//!
//! A 12× difference in heat input moved the clear time by one second. Running
//! the same schedule with and without an external cooler cleared at the
//! *identical sample* — die at 34.84 °C in one arm, 38.52 °C in the other. It
//! engages 3–4 s into a load and releases 318 s after, and one observed
//! engagement happened on a die change of 0.10 °C, a single sensor
//! quantization step. It cuts the other way too: on a stock-cooled MacBook Pro
//! (M4 Max) the die rose 49 → 60 °C under full load while the enum never left
//! `nominal` at all.
//!
//! So this gate cannot pass mid-batch on the Neo and never engages on the
//! M4 Max — uninformative on both, for opposite reasons. It is kept
//! because it is conservative and its deadline now accommodates the hold-off,
//! **not because it is a good signal.**
//!
//! ## The M5 Max needs a temperature criterion too
//!
//! The enum alone cannot deliver a comparable starting state there: it engages
//! only after minutes of *continuous* load, and the gaps between cells reset
//! that clock while heat keeps accumulating, so a batch heats underneath it.
//! [`MAX_DIE_C`] is the criterion added for it.
//!
//! Die temperature is hysteretic — the same reading means one thing warming and
//! another cooling — but this gate only ever samples the cooling branch, where
//! the relation is monotonic: it runs between repetitions, on an idle machine,
//! and [`MAX_BUSY_CORES`] rejects a machine something else is loading.
//!
//! Measurements, cost and open questions:
//! `docs/methodology/macbook-m5-thermal-behavior.md`.
//!
//! ## Why the deadline is 7 minutes
//!
//! [`DEFAULT_MAX_WAIT`] exists because of the 318 s measurement above: the
//! previous 300 s sat *below* the hold-off, so on the Neo this gate
//! did not merely over-wait — it timed out and **failed the cell** after any
//! repetition that warmed the machine. That is the one unambiguous defect the
//! measurements turned up, and 420 s is the fix.
//!
//! ## Reading the level
//!
//! Pressure levels are `OSThermalPressureLevel` from
//! `<libkern/OSThermalNotification.h>`. The macOS numbering (0 nominal,
//! 1 moderate, 2 heavy, 3 trapping, 4 sleeping) is *not* the iOS numbering
//! (0/10/20/30/40/50), and neither matches `ProcessInfo.thermalState`'s four
//! levels — so there is no mapping between this gate and the recorded
//! `apple_thermal_state` series, which reports `ProcessInfo` unchanged.
//!
//! Reading the notification directly rather than shelling out to `swift -e`
//! for `ProcessInfo.thermalState` costs microseconds instead of ~120 ms warm
//! (~0.7 s cold) of Swift-compiler spawn. That mattered here beyond speed:
//! the *other* readiness criterion is "the CPU is quiet", so a probe that
//! burns CPU was a confound for its own sibling check. The two signals track
//! in lock-step in both directions, so this is the same criterion read more
//! cheaply, not a different one.
//!
//! ## Reading CPU load
//!
//! CPU pressure used to read `vm.loadavg`, but XNU's 1-minute EMA carries
//! minutes of historical run-queue depth — on a developer Mac with Spotlight /
//! Time Machine / Slack / browser at rest the normalized load drifts near the
//! threshold. What replaced it is an instantaneous busy ratio over a
//! one-second window, which only crosses the threshold when something is
//! actively eating cores.
//!
//! That window is sampled from Mach tick counters
//! (`host_statistics(HOST_CPU_LOAD_INFO)`), resampled once if the counters did
//! not advance (see [`read_cpu_load`]).
//!
//! The motive is the same one that moved the thermal read off `swift -e`: a
//! probe that spawns a subprocess is burning CPU inside the window it is
//! measuring, in service of deciding whether the CPU is quiet. It is also
//! ~1.88 s per check against ~1.0 s, which across a 1000-cell plan is hours.

use std::ffi::{c_char, c_int, CStr};
use std::time::{Duration, Instant};

use super::{ReadinessError, ThermalGate};

/// `NOTIFY_STATUS_OK` from `<notify.h>`.
const NOTIFY_STATUS_OK: u32 = 0;
/// `kOSThermalPressureLevelNominal` — the only level we start a
/// measurement at.
const PRESSURE_NOMINAL: u64 = 0;

// libSystem is always linked on macOS, so these need no `#[link]`
// attribute and no `libc` dependency.
extern "C" {
    fn notify_register_check(name: *const c_char, out_token: *mut c_int) -> u32;
    fn notify_get_state(token: c_int, state: *mut u64) -> u32;
    fn notify_cancel(token: c_int) -> u32;

    /// The notify(3) name for the thermal-pressure level, exported by
    /// `<libkern/OSThermalNotification.h>`. Keeps its C spelling so the
    /// linker resolves it; see [`thermal_pressure_name`] for why it's
    /// linked instead of typed as a literal.
    #[allow(non_upper_case_globals)]
    static kOSThermalNotificationPressureLevelName: *const c_char;
}

/// The notify(3) name to read the thermal-pressure level from, taken from
/// Apple's exported symbol rather than spelled out here.
///
/// This matters more than it looks. `notify_register_check` succeeds for
/// *any* name — verified, including pure nonsense and near-miss typos —
/// and `notify_get_state` then reports 0 for it indefinitely. Because 0 is
/// `nominal`, a mistyped or later-renamed notification would make this
/// gate silently pass every check rather than error: fail-open, in the one
/// crate that exists to fail closed. Linking Apple's symbol demotes that
/// whole class of mistake to a link error at build time.
///
/// It also resolves which of the two similar names is correct. There's an
/// undocumented `com.apple.system.thermalpressure` alongside this
/// `…thermalpressurelevel`; registering the former proves nothing (see
/// above), and it stayed at 0 on the Neo throughout a run where the linked
/// one reached `moderate`. Only this one is the documented pairing for the
/// `OSThermalPressureLevel` values [`format_pressure_word`] decodes.
fn thermal_pressure_name() -> &'static CStr {
    // SAFETY: the symbol is a NUL-terminated C string constant with static
    // storage duration in libSystem, so the pointer is valid for the
    // process lifetime and the `'static` lifetime is sound.
    unsafe { CStr::from_ptr(kOSThermalNotificationPressureLevelName) }
}

/// Default deadline for the wait, sized so the measured ~318 s hold-off fits
/// with margin. The previous 300 s sat below it, which is why this gate used
/// to time out and fail cells on fanless hosts — see the module docs.
///
/// **The one platform that overrides [`crate::DEFAULT_MAX_WAIT`]**, and the
/// reason that constant asks for a measured basis rather than a judgment: the
/// 318 s figure is a measurement (`tools/macos-thermal-probe/README.md`), and it
/// lands above the shared 300 s, so this is a case where the shared value
/// provably does not work. The assertion in this module's tests keeps it above
/// the hold-off.
pub(super) const DEFAULT_MAX_WAIT: Duration = Duration::from_secs(420);
/// The CPU budget a machine must stay *under* during [`CPU_SAMPLE_WINDOW`],
/// expressed in **busy cores** (`busy_ratio × ncpu`) rather than a fraction of
/// total capacity; a reading equal to it fails, as on every other platform.
///
/// A fraction does not survive core count. The previous 0.30 all-core was sized
/// for smaller machines: on an 18-core M5 Max it takes ~5.4 saturated cores to
/// trip, so a single competing process passes. One host ran a stray `llama-cli`
/// pinning a core for five days and read `ok` at 13% all-core throughout,
/// benchmarking against a competitor the whole time.
///
/// One core is the smallest unit of competition that matters, and it clears
/// idle noise with room to spare: idle on these hosts measures 0.23–0.37 busy
/// cores, against 2.36 for that pinned-core case.
const MAX_BUSY_CORES: f64 = 1.0;

/// Die temperature a machine must be *under* to start a measurement.
///
/// 50 °C is where full throughput returns, measured on three M5 Max hosts —
/// far enough above their idle baseline to clear the sensor noise that rules
/// out tighter thresholds.
///
/// **Calibrated for the M5 Max; it does not transfer.** `die_temp_max_c` is a
/// max over a per-host sensor count, so the same number means different things
/// on different hardware; a portable version needs a per-host baseline plus a
/// delta. See `docs/methodology/macbook-m5-thermal-behavior.md`.
const MAX_DIE_C: f64 = 50.0;

/// Whether a die reading clears [`MAX_DIE_C`].
///
/// `None` waives the criterion rather than failing: the threshold is calibrated
/// per machine, so a host whose sensors are unreadable falls back to the enum
/// alone instead of failing every cell.
fn die_ok(die_c: Option<f64>) -> bool {
    die_c.is_none_or(|c| c < MAX_DIE_C)
}

pub(super) fn wait_until_ready(
    max_wait: Duration,
    thermal: ThermalGate,
) -> Result<(), ReadinessError> {
    let deadline = Instant::now() + max_wait;
    loop {
        let started = Instant::now();
        // Read the level even when the gate is skipped: it costs microseconds
        // and it makes the log line say what the machine's thermal state was on
        // a run that chose not to wait for it. It is *not* the recorded
        // telemetry — `detect_thermal()` samples that separately once this
        // returns — so when the gate is skipped a probe failure must not fail
        // the cell for the sake of a log line.
        let level = match read_thermal_pressure() {
            Ok(level) => Some(level),
            Err(e) if thermal == ThermalGate::Skip => {
                log::warn!(
                    "readiness: macos thermal probe unreadable ({e}); the thermal gate is \
                     skipped for this wait, so continuing without it"
                );
                None
            }
            Err(e) => return Err(e),
        };
        let busy_cores = read_busy_cores()?;
        let die_c = pipette_device::die_temp_max_c();
        let thermal_ok =
            thermal == ThermalGate::Skip || (level == Some(PRESSURE_NOMINAL) && die_ok(die_c));
        let cpu_ok = busy_cores < MAX_BUSY_CORES;
        let summary = format_summary(level, die_c, busy_cores, thermal);
        if thermal_ok && cpu_ok {
            log::info!("readiness: macos {summary} → proceeding");
            return Ok(());
        }
        if Instant::now() >= deadline {
            log::info!("readiness: macos {summary} → timed out after {max_wait:?}");
            return Err(ReadinessError::TimedOut {
                max_wait,
                observed: summary,
            });
        }
        // `read_cpu_load` already spent ~1 s inside this interval (~2 s if it
        // had to resample), so top up rather than sleeping a further full
        // interval on top.
        let rest = super::SENSOR_POLL_INTERVAL.saturating_sub(started.elapsed());
        log::info!("readiness: macos {summary} → waiting {rest:?}");
        std::thread::sleep(rest);
    }
}

/// The log / `TimedOut` summary line. Carries the raw pressure level
/// alongside its name so a fleet log records exactly what the kernel
/// reported, not just our interpretation of it — including when the gate was
/// waived, since a skipped run is not comparable to a gated one and the log
/// should not be silent about which it was.
/// `level` is `None` only when the probe was unreadable *and* the gate was
/// skipped, which the line has to distinguish from a healthy reading — a run
/// that never learned its thermal state is not the same evidence as one that
/// read `nominal`.
fn format_summary(
    level: Option<u64>,
    die_c: Option<f64>,
    busy_cores: f64,
    thermal: ThermalGate,
) -> String {
    let cpu_verdict = if busy_cores < MAX_BUSY_CORES {
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
        Some(c) if die_ok(Some(c)) => format!(" die:{c:.0}C(ok)"),
        Some(c) => format!(" die:{c:.0}C(hot)"),
        // Distinct from a healthy reading, for the same reason the pressure
        // field is: a waived criterion is not a passed one.
        None => " die:unreadable".to_string(),
    };
    format!(
        "thermal={thermal_field}{skipped}{die_field} cpu=busy:{busy_cores:.1}cores({cpu_verdict})"
    )
}

/// `OSThermalPressureLevel` names in the macOS numbering. An
/// unrecognized value reads as `unknown` rather than being coerced to a
/// neighbor, so a level added by a future kernel shows up in the logs
/// instead of silently passing as nominal.
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

/// Read the current thermal-pressure level, coolest (0) → hottest (4).
///
/// Registers a token, reads, then cancels on every call: the state
/// belongs to the notify *name* rather than the token, so a fresh token
/// still reads the live value, and at one call per poll a register /
/// cancel pair is cheaper than reasoning about a cached token's lifetime.
///
/// This is `notify_get_state`, deliberately not `notify_check` —
/// `notify_check` reports whether the notification has *fired* since the
/// last check on that token (and always reports "changed" on a token's
/// first check), which says nothing about the current level.
fn read_thermal_pressure() -> Result<u64, ReadinessError> {
    let mut token: c_int = 0;
    // SAFETY: the name is a NUL-terminated static C string valid for the
    // call, and `token` is an initialized `c_int` we own.
    let rc = unsafe { notify_register_check(thermal_pressure_name().as_ptr(), &mut token) };
    if rc != NOTIFY_STATUS_OK {
        log::warn!(
            "readiness: notify_register_check(thermalpressurelevel) failed with status {rc}"
        );
        return Err(ReadinessError::Unavailable(
            "notify_register_check(com.apple.system.thermalpressurelevel) failed",
        ));
    }
    let mut level: u64 = 0;
    // SAFETY: `token` was registered successfully above, and `level` is
    // an initialized `u64` we own.
    let rc = unsafe { notify_get_state(token, &mut level) };
    // Release the token whether or not the read succeeded, so a probe
    // that starts failing mid-run doesn't leak one per poll.
    // SAFETY: `token` was registered successfully above.
    let _ = unsafe { notify_cancel(token) };
    if rc != NOTIFY_STATUS_OK {
        log::warn!("readiness: notify_get_state(thermalpressurelevel) failed with status {rc}");
        return Err(ReadinessError::Unavailable(
            "notify_get_state(com.apple.system.thermalpressurelevel) failed",
        ));
    }
    Ok(level)
}

/// Width of the CPU sampling window. **Do not shorten below ~800 ms.**
///
/// `HOST_CPU_LOAD_INFO` refreshes on a coarse (~1 s) cadence, so a shorter
/// window frequently spans no update at all and yields a zero tick delta. That
/// failure is not merely noisy, it is *biased*: the samples that do land on an
/// update are selected for having had activity in them, so a 200 ms window
/// measured 42% busy on a machine genuinely at 23% — enough to fail readiness
/// spuriously against [`CPU_LOAD_THRESHOLD`]. Measured zero-delta rates: 17/30
/// at 200 ms, 10/30 at 400 ms, 4/30 at 600 ms, 0/30 at 800 ms and above. 1 s
/// also gave the tightest spread (sd 0.9%).
const CPU_SAMPLE_WINDOW: Duration = Duration::from_secs(1);

/// One `HOST_CPU_LOAD_INFO` snapshot, or `None` if the call failed.
fn cpu_ticks() -> Option<libc::host_cpu_load_info> {
    let mut info = libc::host_cpu_load_info {
        cpu_ticks: [0; libc::CPU_STATE_MAX as usize],
    };
    let mut count = (std::mem::size_of::<libc::host_cpu_load_info>()
        / std::mem::size_of::<libc::integer_t>())
        as libc::mach_msg_type_number_t;
    // SAFETY: `mach_host_self` returns a send right we own and release below.
    // `host_statistics` writes `count` integers into `info`, and `count` is
    // derived from `info`'s own size, so the buffer cannot be overrun.
    let status = unsafe {
        let host = mach_host_self();
        let status = libc::host_statistics(
            host,
            libc::HOST_CPU_LOAD_INFO,
            std::ptr::from_mut(&mut info).cast::<libc::integer_t>(),
            &mut count,
        );
        // Release the send right. `mach_host_self` hands out a reference per
        // call, so skipping this leaks one port right per readiness check —
        // thousands over a long plan.
        mach_port_deallocate(mach_task_self_, host);
        status
    };
    (status == libc::KERN_SUCCESS).then_some(info)
}

// Declared here rather than taken from `libc`, which deprecates its Mach
// wrappers in favor of the `mach2` crate. Three plain libSystem symbols aren't
// worth a new dependency when this file already hand-declares its notify(3)
// bindings; `libc` is still used for the types and constants, which are not
// deprecated.
extern "C" {
    fn mach_host_self() -> libc::mach_port_t;
    fn mach_port_deallocate(
        task: libc::mach_port_t,
        name: libc::mach_port_t,
    ) -> libc::kern_return_t;
    /// `mach_task_self()` is a macro over this global in C. Declared immutable
    /// because nothing mutates it after process start and we only read it.
    static mach_task_self_: libc::mach_port_t;
}

/// Busy ratio across all cores over [`CPU_SAMPLE_WINDOW`].
///
/// Reads Mach CPU tick counters directly rather than spawning
/// `top -l 2 -s 1 -n 0`, which cost ~1.88 s per check against ~1.0 s here —
/// worth ~1.2 h across a 1000-cell plan, and it takes a subprocess spawn out of
/// a loop whose whole job is deciding whether the machine is quiet.
///
/// Resamples once, then reports unready. The only failure mode ever measured is
/// a window that spanned no counter refresh — transient, and cleared by a second
/// sample (0/30 at this window, though not 0-in-infinity). A `host_statistics`
/// call that genuinely fails gets no fallback: it is public, unprivileged Darwin
/// API, so a host where it breaks is one worth hearing about loudly rather than
/// quietly measuring anyway.
/// Busy cores: the all-core busy ratio scaled by the online CPU count, which is
/// what [`MAX_BUSY_CORES`] is expressed in.
fn read_busy_cores() -> Result<f64, ReadinessError> {
    Ok(read_cpu_load()? * online_cpus()?)
}

/// Online logical CPUs.
///
/// An unreadable count is an error rather than a default, because every
/// plausible default is wrong in the unsafe direction: assuming 1 makes
/// `ratio × 1` at most 1.0, which is *always* under the threshold, so the gate
/// would silently stop rejecting busy machines. `sysconf` cannot realistically
/// fail here, so this costs nothing in practice.
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

fn read_cpu_load() -> Result<f64, ReadinessError> {
    mach_cpu_busy()
        .or_else(|| {
            log::debug!("readiness: macos CPU tick delta was zero; resampling");
            mach_cpu_busy()
        })
        .ok_or(ReadinessError::Unavailable(
            "HOST_CPU_LOAD_INFO ticks did not advance across two windows",
        ))
}

/// Busy ratio from two tick snapshots, or `None` if either read failed or the
/// counters did not advance between them.
fn mach_cpu_busy() -> Option<f64> {
    let before = cpu_ticks()?;
    std::thread::sleep(CPU_SAMPLE_WINDOW);
    let after = cpu_ticks()?;
    busy_from_ticks(&before.cpu_ticks, &after.cpu_ticks)
}

/// Reduce two tick arrays to a busy ratio. Split out from the sampling so the
/// arithmetic — including the wrap and zero-delta cases — is testable without
/// waiting a second on real hardware.
fn busy_from_ticks(
    before: &[u32; libc::CPU_STATE_MAX as usize],
    after: &[u32; libc::CPU_STATE_MAX as usize],
) -> Option<f64> {
    // `natural_t` is 32-bit and these are monotonic counters, so `wrapping_sub`
    // is the correct reading across a wrap rather than a panic or a negative.
    // Indexing is in-bounds by construction: both arrays are `CPU_STATE_MAX`
    // long and every caller below indexes with a `CPU_STATE_*` constant.
    let delta = |state: usize| u64::from(after[state].wrapping_sub(before[state]));
    let total: u64 = (0..libc::CPU_STATE_MAX as usize).map(delta).sum();
    if total == 0 {
        return None; // counters didn't refresh inside the window
    }
    let idle = delta(libc::CPU_STATE_IDLE as usize);
    Some((1.0 - idle as f64 / total as f64).clamp(0.0, 1.0))
}

#[cfg(test)]
mod tests {
    use anyhow::Context;
    use rstest::rstest;

    use super::*;

    /// Exercises the FFI end to end — signatures, registration, state
    /// read, cancel — against the live kernel. No subprocess, so unlike
    /// the old `swift -e` path this is cheap enough to run unconditionally
    /// in CI, which only compiles this module on macOS anyway. Asserts
    /// only that the read succeeds: the level depends on how hot the host
    /// actually is.
    #[test]
    fn thermal_pressure_reads_from_kernel() -> anyhow::Result<()> {
        let _level = read_thermal_pressure()?;
        Ok(())
    }

    /// Pins the value behind Apple's exported symbol. Linking it prevents a
    /// *typo*, but not Apple repointing the constant at a differently
    /// encoded notification — if that happens this fails loudly instead of
    /// [`format_pressure_word`] quietly decoding the wrong scale.
    #[test]
    fn linked_name_is_the_pressure_level_notification() -> anyhow::Result<()> {
        assert_eq!(
            thermal_pressure_name().to_str()?,
            "com.apple.system.thermalpressurelevel",
        );
        Ok(())
    }

    /// Documents the hazard the linked symbol exists to remove: notify
    /// accepts *any* name and reports `nominal` for it forever, so a
    /// hand-typed name with a typo would silently make this gate fail open.
    /// If this test ever starts failing, notify grew name validation and
    /// the linked symbol became a convenience rather than a safeguard.
    #[test]
    fn notify_accepts_a_bogus_name_and_reports_nominal() -> anyhow::Result<()> {
        let mut token: c_int = 0;
        // SAFETY: a NUL-terminated literal and a `c_int` we own.
        let rc = unsafe {
            notify_register_check(
                c"com.example.pipette.no.such.notification".as_ptr(),
                &mut token,
            )
        };
        assert_eq!(rc, NOTIFY_STATUS_OK, "notify rejected a bogus name");
        let mut level = u64::MAX;
        // SAFETY: `token` was registered successfully above.
        let read = unsafe { notify_get_state(token, &mut level) };
        // SAFETY: `token` was registered successfully above.
        let _ = unsafe { notify_cancel(token) };
        assert_eq!(read, NOTIFY_STATUS_OK, "get_state failed on a bogus name");
        assert_eq!(
            level, PRESSURE_NOMINAL,
            "a bogus name no longer reads as nominal",
        );
        Ok(())
    }

    /// The deadline has to clear the measured hold-off, or the gate fails
    /// cells instead of waiting. This is the regression guard on that
    /// number: 318 s measured on a fanless MacBook Neo, invariant across a
    /// 12x load range.
    #[test]
    fn deadline_clears_the_measured_hold_off() {
        assert!(
            DEFAULT_MAX_WAIT >= Duration::from_secs(318),
            "DEFAULT_MAX_WAIT {DEFAULT_MAX_WAIT:?} is below the measured 318s hold-off",
        );
    }

    /// The bug this threshold replaced: one saturated core on a many-core host
    /// is a small *fraction* but a full core of competition. Expressed in
    /// cores it fails on any host; expressed as the old 0.30 all-core ratio it
    /// passed on everything from 4 cores up.
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
        assert_eq!(busy_cores < MAX_BUSY_CORES, want_ok);
        // The same reading as a fraction of capacity, which is what the gate
        // used to compare: every one of these passed 0.30 on 18 cores.
        let as_ratio = busy_cores / cores;
        if cores >= 18.0 {
            assert!(as_ratio < 0.30, "{as_ratio} would have passed the old gate");
        }
    }

    /// The failure this criterion exists for: the enum reads `nominal` at every
    /// one of these temperatures while a batch ratchets the die up underneath
    /// it. `None` is the waiver for a host whose sensors are unreadable.
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
        assert_eq!(die_ok(die_c), want_ok);
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

    /// The raw kernel level must survive into the summary line — that's
    /// what makes a fleet log diagnosable when a host times out.
    #[rstest]
    #[case(
        0,
        0.4,
        "thermal=pressure:0(nominal) die:40C(ok) cpu=busy:0.4cores(ok)"
    )]
    #[case(
        2,
        6.2,
        "thermal=pressure:2(heavy) die:40C(ok) cpu=busy:6.2cores(busy)"
    )]
    #[case(
        9,
        0.4,
        "thermal=pressure:9(unknown) die:40C(ok) cpu=busy:0.4cores(ok)"
    )]
    fn summary_reports_raw_pressure_level(
        #[case] level: u64,
        #[case] busy_cores: f64,
        #[case] want: &str,
    ) {
        assert_eq!(
            format_summary(Some(level), Some(40.0), busy_cores, ThermalGate::Enforce),
            want,
        );
    }

    /// A waived gate must be visible in the log, and it must still report the
    /// level it declined to wait for — otherwise a skipped run is
    /// indistinguishable from a gated one after the fact.
    #[test]
    fn summary_marks_a_skipped_thermal_gate() {
        assert_eq!(
            format_summary(Some(2), Some(40.0), 0.4, ThermalGate::Skip),
            "thermal=pressure:2(heavy)[SKIPPED] die:40C(ok) cpu=busy:0.4cores(ok)",
        );
    }

    /// An unreadable probe is only tolerated when the gate is skipped, and the
    /// line must not let that pass for a healthy `nominal` — "we never found
    /// out" and "we checked, it was fine" are different evidence about a run.
    #[test]
    fn summary_distinguishes_an_unreadable_probe_from_nominal() {
        assert_eq!(
            format_summary(None, Some(40.0), 0.4, ThermalGate::Skip),
            "thermal=pressure:unreadable[SKIPPED] die:40C(ok) cpu=busy:0.4cores(ok)",
        );
        assert_ne!(
            format_summary(None, Some(40.0), 0.4, ThermalGate::Skip),
            format_summary(Some(PRESSURE_NOMINAL), Some(40.0), 0.4, ThermalGate::Skip),
        );
        // Same rule for the die reading: waived is not passed.
        assert_eq!(
            format_summary(Some(PRESSURE_NOMINAL), None, 0.4, ThermalGate::Enforce),
            "thermal=pressure:0(nominal) die:unreadable cpu=busy:0.4cores(ok)",
        );
    }

    /// The tick arithmetic, including the two cases real hardware produced.
    ///
    /// Order is `[user, system, idle, nice]` per `CPU_STATE_*`.
    #[rstest]
    // Half idle, half busy.
    #[case([0, 0, 0, 0], [50, 50, 100, 0], Some(0.5))]
    // Fully idle.
    #[case([0, 0, 0, 0], [0, 0, 100, 0], Some(0.0))]
    // Fully busy.
    #[case([0, 0, 0, 0], [70, 30, 0, 0], Some(1.0))]
    // Nice time counts as busy, not idle.
    #[case([0, 0, 0, 0], [0, 0, 50, 50], Some(0.5))]
    // Counters did not refresh inside the window: no answer, so the caller
    // falls back to `top` rather than reporting a made-up ratio. Observed
    // 17 times in 30 at a 200 ms window on real hardware.
    #[case([9, 9, 9, 9], [9, 9, 9, 9], None)]
    // `natural_t` is 32-bit; a counter that wrapped must read as the small
    // forward delta it is, not a huge one or a panic.
    // 3 busy ticks and 6 idle across the wrap => 1/3 busy.
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

    /// The window must stay at or above the ~800 ms refresh cadence of
    /// `HOST_CPU_LOAD_INFO`. Below it, samples frequently see no tick delta,
    /// and the ones that do land on an update are biased *high* — 42% measured
    /// on a machine genuinely at 23% — which would fail readiness spuriously.
    #[test]
    fn cpu_sample_window_clears_the_counter_refresh_cadence() {
        assert!(
            CPU_SAMPLE_WINDOW >= Duration::from_millis(800),
            "CPU_SAMPLE_WINDOW {CPU_SAMPLE_WINDOW:?} is below the measured refresh cadence",
        );
    }

    /// The live Mach path must produce a usable ratio on real hardware; the
    /// synthetic-tick cases above pin the arithmetic, not the syscall. Goes
    /// through `read_cpu_load` so the resample-on-zero-delta path is covered
    /// too, rather than only the happy first sample.
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
