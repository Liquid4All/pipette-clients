//! Pre-benchmark readiness wait: hold each throughput-class rep until the host
//! is not throttling and not loaded, so the numbers reflect steady state.
//!
//! [`resolve_readiness`] fixes the policy once per cell; [`wait_until_ready`]
//! applies it through the running platform's submodule. A timeout is an error,
//! so the cell fails instead of recording polluted numbers.

use std::time::Duration;

mod callbacks;
mod configuration;
mod wait;

pub use callbacks::{ReadinessCallbacks, ReadinessPhase, ReadinessPlatform, ReadinessStatus};
pub use configuration::{
    AndroidReadinessConfiguration, BusyCores, CeilingCelsius, CpuPercent, HeldTouch,
    HeldTouchError, JetsonOrinReadinessConfiguration, LinuxReadinessConfiguration,
    MacOsReadinessConfiguration, RangeError, RaspberryPi5ReadinessConfiguration,
    ReadinessConfiguration, WindowsReadinessConfiguration, WindowsSensor, WindowsSensorError,
};

// In every test build: no CI leg runs tests on Android.
#[cfg(any(target_os = "android", test))]
mod android;
#[cfg(target_os = "linux")]
mod linux;
#[cfg(target_os = "macos")]
mod macos;
#[cfg(not(any(
    target_os = "android",
    target_os = "linux",
    target_os = "macos",
    target_os = "windows"
)))]
mod other;
// In every test build, so the Windows poll loop's tests run on every host.
#[cfg(any(target_os = "windows", test))]
mod windows;

/// Environment override for the readiness deadline, in whole seconds. An
/// explicit `max_wait` to [`resolve_readiness`] beats it. The plan runner sets
/// it on the invocations it spawns, so an override reaches remote runners
/// (ssh, adb, slurm, iOS), not only local ones that inherit its environment.
pub const MAX_WAIT_ENV: &str = "PIPETTE_READINESS_MAX_WAIT_SECS";

/// An unparseable value is ignored with a warning, so a typo cannot silently
/// disable the wait.
fn env_max_wait() -> Option<Duration> {
    let raw = std::env::var(MAX_WAIT_ENV).ok()?;
    match raw.trim().parse::<u64>() {
        Ok(secs) => Some(Duration::from_secs(secs)),
        Err(_) => {
            tracing::warn!(
                var = %MAX_WAIT_ENV,
                value = raw.as_str(),
                "ignoring an unparseable max wait override; using the per-platform default"
            );
            None
        }
    }
}

/// Environment switch that disables the thermal criterion, in the grammar of
/// [`skip_thermal_from_str`]. It decides only when the [`ThermalGate`] passed
/// to [`resolve_readiness`] is `Unset`. Forwarded by the plan runner as
/// [`MAX_WAIT_ENV`] is.
pub const SKIP_THERMAL_ENV: &str = "PIPETTE_READINESS_SKIP_THERMAL";

fn env_skip_thermal() -> bool {
    std::env::var(SKIP_THERMAL_ENV)
        .as_deref()
        .map(skip_thermal_from_str)
        .unwrap_or(false)
}

/// Whether a textual value asks for the thermal criterion to be skipped.
///
/// Empty and every spelling of "no" enforce: exporting `FOO=` is a common way
/// to unset a variable, and `off` plainly means "don't skip".
///
/// The only grammar for the flag: it backs [`SKIP_THERMAL_ENV`] and is
/// `pipette-ctl`'s `--readiness-skip-thermal` value parser. Not clap's
/// `FalseyValueParser`, which disagrees on `off`/`no` and whitespace, so a
/// value could mean "enforce" to the plan driver and "skip" to its runner.
pub fn skip_thermal_from_str(raw: &str) -> bool {
    !matches!(
        raw.trim().to_ascii_lowercase().as_str(),
        "" | "0" | "false" | "f" | "no" | "n" | "off"
    )
}

/// Whether a wait must satisfy the platform's thermal criterion. Three-valued
/// so "no opinion" (defer to [`SKIP_THERMAL_ENV`]) differs from "enforce".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ThermalGate {
    /// No opinion; [`SKIP_THERMAL_ENV`] decides, and enforces when it is unset.
    /// What a cell whose flags do not say passes.
    #[default]
    Unset,
    /// Require the thermal criterion even if [`SKIP_THERMAL_ENV`] asks to skip:
    /// a cell that authored `skip_thermal = false` must not be undone by a
    /// stale export.
    Enforce,
    /// Skip the thermal criterion; the load criterion still applies, since it
    /// catches a concurrent benchmark, which has nothing to do with heat.
    ///
    /// A skipped run is not comparable to a gated one. Results carry
    /// `benchmark_flags.readiness.skip_thermal` where the flag schema models
    /// the cell, and every rep records its starting thermal state
    /// (`ThermalTelemetry`); a reading outside the gate's band betrays a waiver
    /// only while the host's `[readiness.<gate>]` table does not raise the
    /// bounds, since no result carries the table.
    Skip,
}

impl ThermalGate {
    fn skipped(self) -> bool {
        self == Self::Skip
    }

    /// The gate a platform applies, never `Unset`. When both the argument and
    /// the environment speak, the argument (a cell) beats the environment (the
    /// whole process).
    fn resolve(self, env_asks_skip: bool) -> Self {
        match self {
            Self::Skip => Self::Skip,
            Self::Enforce => Self::Enforce,
            Self::Unset if env_asks_skip => Self::Skip,
            Self::Unset => Self::Enforce,
        }
    }
}

/// Block until the device is ready for a measurement under `resolved`.
pub fn wait_until_ready(resolved: &ResolvedReadiness) -> Result<(), ReadinessError> {
    wait_until_ready_with(resolved, &ReadinessCallbacks::default())
}

/// [`wait_until_ready`] with the caller's own progress, cancellation and sleep.
/// `resolved` still decides every criterion and the deadline.
pub fn wait_until_ready_with(
    resolved: &ResolvedReadiness,
    callbacks: &ReadinessCallbacks<'_>,
) -> Result<(), ReadinessError> {
    dispatch(resolved, callbacks)
}

/// The readiness policy a run applies, after the request, the environment and
/// the platform default are folded together. A caller records this, not the
/// request, so the result describes the run that happened.
///
/// `#[non_exhaustive]` so only [`resolve_readiness`] builds one: a literal
/// could carry [`ThermalGate::Unset`] and skip the environment, the platform
/// default and the clamp unseen.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct ResolvedReadiness {
    /// The deadline actually used, after the env override, the platform
    /// default and [`clamp_max_wait`].
    pub max_wait: Duration,
    /// [`ThermalGate::Skip`] or [`ThermalGate::Enforce`], never `Unset`.
    pub thermal: ThermalGate,
    /// The client configuration file's `[readiness]` table, as parsed. The CLI
    /// refuses an arm for a gate this host does not run.
    pub configuration: Option<ReadinessConfiguration>,
    /// Android skin threshold after the cell override and host table are combined.
    pub skin_below_c: Option<CeilingCelsius>,
    /// The Android CLI gate's held-touch mode, as supplied by the caller or
    /// host table. `None` is [`HeldTouch::Wait`].
    pub held_touch: Option<HeldTouch>,
}

impl ResolvedReadiness {
    /// The deadline in whole seconds, as the plan's `readiness.max_wait_secs`
    /// spells it. Truncates, which loses nothing: every source is whole seconds.
    pub fn max_wait_secs(&self) -> u64 {
        self.max_wait.as_secs()
    }

    /// Whether the thermal criterion was waived (`readiness.skip_thermal`).
    pub fn skip_thermal(&self) -> bool {
        self.thermal.skipped()
    }

    /// The skin threshold actually applied, including a host-configured value.
    pub fn skin_below_c(&self) -> Option<CeilingCelsius> {
        self.skin_below_c
    }

    /// Return [`HeldTouch::Wait`] when no mode is configured.
    pub fn held_touch(&self) -> HeldTouch {
        self.held_touch.unwrap_or_default()
    }

    /// The same policy under the client configuration's `[readiness]` table.
    #[must_use]
    pub fn with_configuration(self, configuration: Option<ReadinessConfiguration>) -> Self {
        let android = configuration
            .as_ref()
            .and_then(ReadinessConfiguration::android);
        let skin_below_c = self
            .skin_below_c
            .or_else(|| android.and_then(|android| android.skin_below_c));
        let held_touch = self
            .held_touch
            .or_else(|| android.and_then(|android| android.held_touch));
        Self {
            configuration,
            skin_below_c,
            held_touch,
            ..self
        }
    }

    /// A caller's held-touch override takes precedence over the host
    /// configuration.
    #[must_use]
    pub fn with_held_touch(self, override_mode: Option<HeldTouch>) -> Self {
        Self {
            held_touch: override_mode.or(self.held_touch),
            ..self
        }
    }

    /// A cell-authored skin threshold takes precedence over host configuration.
    #[must_use]
    pub fn with_skin_below_c(self, override_c: Option<CeilingCelsius>) -> Self {
        Self {
            skin_below_c: override_c.or(self.skin_below_c),
            ..self
        }
    }
}

/// The policy a run applies. The deadline is `max_wait`, else [`MAX_WAIT_ENV`],
/// else the platform default; the thermal gate resolves against
/// [`SKIP_THERMAL_ENV`]. The caller supplies a held-touch override through
/// [`ResolvedReadiness::with_held_touch`]. The deadline never changes the
/// criteria; `thermal` does.
///
/// Resolve once per cell and reuse it: resolving twice could straddle an
/// environment change and record a policy that disagrees with the gate.
pub fn resolve_readiness(max_wait: Option<Duration>, thermal: ThermalGate) -> ResolvedReadiness {
    let max_wait = clamp_max_wait(
        max_wait
            .or_else(env_max_wait)
            .unwrap_or_else(platform_default_max_wait),
    );
    let thermal = thermal.resolve(env_skip_thermal());
    if thermal.skipped() {
        tracing::warn!(
            "thermal criterion DISABLED for this wait; only the load check applies, and \
             results are not comparable to thermally gated runs"
        );
    }
    ResolvedReadiness {
        max_wait,
        thermal,
        configuration: None,
        skin_below_c: None,
        held_touch: None,
    }
}

/// Ceiling on any resolved deadline. Every platform computes
/// `Instant::now() + max_wait`, which panics when unrepresentable, and both
/// inputs accept any `u64` of seconds. A day is far past the longest platform
/// default (7 minutes).
const MAX_WAIT_CEILING: Duration = Duration::from_secs(24 * 60 * 60);

/// Clamps rather than rejects: an absurd deadline is a typo whose intent
/// ("wait a long time") the ceiling still serves. Logged, so the operator's
/// number is not ignored silently.
fn clamp_max_wait(requested: Duration) -> Duration {
    if requested > MAX_WAIT_CEILING {
        tracing::warn!(
            max_wait = ?requested,
            ceiling = ?MAX_WAIT_CEILING,
            "requested max wait exceeds the ceiling; using the ceiling instead"
        );
        return MAX_WAIT_CEILING;
    }
    requested
}

/// The deadline for one readiness wait (one rep, not a cell), shared by every
/// platform without a measured reason to differ.
///
/// 300 s is ~8x the ~30-40 s a phone takes to cool from a post-rep peak to its
/// entry threshold (`THERMAL_THRESHOLD_C` in `android.rs`), so it bounds only a
/// device that will never get there: giving up beats reporting the throttle as
/// throughput. Android's `Readiness.COOLDOWN_MAX_MILLIS` and iOS's
/// `defaultMaxWait` carry the same number.
///
/// A platform overrides this only with a measured basis stated at the override;
/// today only macOS does (`DEFAULT_MAX_WAIT` in `macos.rs`). Those are code
/// spans, not intra-doc links, because the platform modules are `#[cfg]`-gated.
pub const DEFAULT_MAX_WAIT: Duration = Duration::from_secs(300);

/// How often a waiting gate re-reads its sensors: latency on every gated rep,
/// not a criterion.
///
/// 10 s was measurably expensive: an iOS `decode_throughput_1024_256` cell spent
/// 46 s of 59 s gating against ~6.5 s of decode, mostly after the die was already
/// under the setpoint. The reads are cheap. iOS uses the same 3 s, the
/// Android app 2 s; Windows runs its own loop at `windows.rs`'s `POLL_INTERVAL`.
pub const SENSOR_POLL_INTERVAL: Duration = Duration::from_secs(3);

/// The number of consecutive failed probes that ends a wait. The shared wait
/// loop and the Windows loop both use it.
///
/// One bad poll must not fail a measurement. For example, PowerShell can fail
/// to start, or `dumpsys` can print a partial report when the phone has little
/// free memory. A good poll resets the count.
///
/// **Three is a bound, not a measurement.** A probe that fails at once costs
/// two poll intervals before the wait fails. A probe command that hangs costs
/// up to `PROBE_TIMEOUT` (30 s) on each attempt. Without the bound, a probe
/// that never answers uses the whole deadline.
pub(crate) const PROBE_RETRY_POLLS: usize = 3;

/// Longest a probe's command may run: the gate checks its deadline and cancel
/// only between probes, so a wedged WMI or `dumpsys` would otherwise hang it.
#[cfg(any(
    target_os = "android",
    target_os = "linux",
    target_os = "windows",
    test
))]
const PROBE_TIMEOUT: Duration = Duration::from_secs(30);

/// `cmd`'s output, killed at [`PROBE_TIMEOUT`].
#[cfg(any(
    target_os = "android",
    target_os = "linux",
    target_os = "windows",
    test
))]
fn probe_output(cmd: std::process::Command) -> Result<std::process::Output, ReadinessError> {
    Ok(pipette_subprocess::output_within(cmd, PROBE_TIMEOUT)?)
}

/// The gate this host runs, as its log line names it.
pub fn platform() -> ReadinessPlatform {
    #[cfg(target_os = "android")]
    return ReadinessPlatform::Android;
    #[cfg(target_os = "linux")]
    return linux::platform();
    #[cfg(target_os = "macos")]
    return ReadinessPlatform::MacOs;
    #[cfg(target_os = "windows")]
    return ReadinessPlatform::Windows;
    #[cfg(not(any(
        target_os = "android",
        target_os = "linux",
        target_os = "macos",
        target_os = "windows"
    )))]
    return ReadinessPlatform::Other;
}

fn platform_default_max_wait() -> Duration {
    #[cfg(target_os = "android")]
    return android::DEFAULT_MAX_WAIT;
    #[cfg(target_os = "linux")]
    return linux::DEFAULT_MAX_WAIT;
    #[cfg(target_os = "macos")]
    return macos::DEFAULT_MAX_WAIT;
    #[cfg(target_os = "windows")]
    return windows::DEFAULT_MAX_WAIT;
    #[cfg(not(any(
        target_os = "android",
        target_os = "linux",
        target_os = "macos",
        target_os = "windows"
    )))]
    return other::DEFAULT_MAX_WAIT;
}

fn dispatch(
    resolved: &ResolvedReadiness,
    callbacks: &ReadinessCallbacks<'_>,
) -> Result<(), ReadinessError> {
    let (max_wait, thermal) = (resolved.max_wait, resolved.thermal);
    #[cfg(target_os = "android")]
    return android::wait_until_ready(
        max_wait,
        thermal,
        resolved.configuration.as_ref(),
        resolved.skin_below_c,
        resolved.held_touch(),
        callbacks,
    );
    // On Linux the board decides which gate.
    #[cfg(target_os = "linux")]
    return linux::wait_until_ready(
        max_wait,
        thermal,
        resolved.configuration.as_ref(),
        callbacks,
    );
    #[cfg(target_os = "macos")]
    return macos::wait_until_ready(
        max_wait,
        thermal,
        resolved.configuration.as_ref(),
        callbacks,
    );
    #[cfg(target_os = "windows")]
    return windows::wait_until_ready(
        max_wait,
        thermal,
        resolved
            .configuration
            .as_ref()
            .and_then(ReadinessConfiguration::windows),
        callbacks,
    );
    #[cfg(not(any(
        target_os = "android",
        target_os = "linux",
        target_os = "macos",
        target_os = "windows"
    )))]
    return other::wait_until_ready(max_wait, thermal, callbacks);
}

#[derive(Debug, thiserror::Error)]
pub enum ReadinessError {
    /// The device stayed non-ready until the deadline. A hard failure for the
    /// cell: proceeding would record numbers under non-steady conditions.
    #[error("readiness wait timed out after {max_wait:?}: last seen {observed}")]
    TimedOut {
        max_wait: Duration,
        /// The platform's summary of the last poll, as its log line prints it.
        observed: String,
    },
    #[error("readiness probe i/o failure")]
    Io(#[from] std::io::Error),
    /// A probe's command could not start, could not be waited on, or timed out.
    #[error("readiness probe command failed")]
    Probe(#[from] pipette_subprocess::OutputError),
    /// `raw` is the offending input verbatim.
    #[error("readiness probe could not parse output: {raw:?}")]
    ParseFailure { raw: String },
    /// The probe ran but reported no usable data.
    #[error("readiness probe unavailable: {0}")]
    Unavailable(&'static str),
    /// The caller's cancel callback stopped the wait. An error, not `Ok`: the rep
    /// never passed the gate, so it must not be recorded as one that did.
    #[error("readiness wait cancelled by the caller")]
    Cancelled,
    /// A ceiling names a sensor this host's successful reads never include: a
    /// typo, or a sensor another device type has. Refused at once rather than
    /// waited on, since no amount of waiting makes it appear.
    #[error(
        "readiness ceiling names {sensor}, which this host does not report; it reports {available}"
    )]
    UnknownSensor {
        sensor: WindowsSensor,
        available: String,
    },
    /// A ceiling's sensor gave no reading for the whole ceiling budget. Failed
    /// rather than let through: the rep would run under a condition the gate
    /// was told to check and could not.
    #[error("readiness ceiling sensor {sensor} gave no reading in {waited:?}")]
    SensorUnread {
        sensor: WindowsSensor,
        waited: Duration,
    },
    /// The table holds the hottest zone to a bound and the host has no zone:
    /// the setting would apply to nothing.
    #[error("[readiness.linux] sets hottest_below_c, and this host has no thermal zones")]
    BoundWithoutThermalZones,
    /// Three complete live HAL reads lacked a usable skin sensor.
    #[error(
        "skin_below_c is set, but `dumpsys thermalservice` reports no live HAL mType=3 skin sensor"
    )]
    BoundWithoutSkinSensor,
    /// The Android gate saw a touch held on the touchscreen on several polls in
    /// a row, under [`HeldTouch::Fail`]. A phantom touch stays until an
    /// operator removes the cause, so a retry does not clear it.
    #[error(
        "the touchscreen has a held touch ({pointers} pointer(s), {irq_per_sec} touch \
         interrupts/s); remove the cause (for example, a phantom touch from a screen cooler). \
         Then run again"
    )]
    HeldTouch { pointers: u32, irq_per_sec: u32 },
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    /// A literal, so retuning the deadline fails here: Android and iOS carry the
    /// same number, and it rides the wire as
    /// `benchmark_flags.readiness.max_wait_secs`.
    #[test]
    fn shared_default_deadline_is_five_minutes() {
        assert_eq!(DEFAULT_MAX_WAIT, Duration::from_secs(300));
    }

    /// Back at the shared default, the gate times out on fanless hosts behind
    /// the measured 318 s hold-off.
    #[cfg(target_os = "macos")]
    #[test]
    fn macos_overrides_the_shared_default_upward() {
        assert!(
            macos::DEFAULT_MAX_WAIT > DEFAULT_MAX_WAIT,
            "macOS default {:?} must stay above the shared {:?} for the measured 318s hold-off",
            macos::DEFAULT_MAX_WAIT,
            DEFAULT_MAX_WAIT,
        );
    }

    #[rstest]
    #[case("", false)]
    #[case("   ", false)]
    #[case("0", false)]
    #[case("  0  ", false)]
    #[case("false", false)]
    #[case("False", false)]
    #[case("FALSE", false)]
    #[case("f", false)]
    #[case("no", false)]
    #[case("NO", false)]
    #[case("n", false)]
    #[case("off", false)]
    #[case("OFF", false)]
    #[case("1", true)]
    #[case("true", true)]
    #[case("TRUE", true)]
    #[case("t", true)]
    #[case("yes", true)]
    #[case("y", true)]
    #[case("on", true)]
    fn only_explicit_values_skip_the_thermal_gate(#[case] raw: &str, #[case] want_skip: bool) {
        assert_eq!(
            skip_thermal_from_str(raw),
            want_skip,
            "{raw:?} should {} the thermal gate",
            if want_skip { "skip" } else { "enforce" },
        );
    }

    #[test]
    fn the_ceiling_keeps_deadline_arithmetic_representable() {
        let absurd = Duration::from_secs(u64::MAX);
        assert!(
            std::time::Instant::now().checked_add(absurd).is_none(),
            "premise failed: if u64::MAX seconds is representable, this ceiling guards nothing",
        );
        assert!(
            std::time::Instant::now()
                .checked_add(clamp_max_wait(absurd))
                .is_some(),
            "a clamped deadline must be representable",
        );
    }

    /// The ceiling itself is inclusive.
    #[rstest]
    #[case(1)]
    #[case(300)]
    #[case(420)]
    #[case(3600)]
    #[case(24 * 60 * 60)]
    fn clamp_leaves_reasonable_deadlines_alone(#[case] secs: u64) {
        let requested = Duration::from_secs(secs);
        assert_eq!(clamp_max_wait(requested), requested);
    }

    #[test]
    fn thermal_gate_defaults_to_no_opinion_that_resolves_to_enforcing() {
        assert_eq!(ThermalGate::default(), ThermalGate::Unset);
        assert!(!ThermalGate::default().skipped());
        assert!(ThermalGate::Skip.skipped());
        assert_eq!(
            ThermalGate::default().resolve(false),
            ThermalGate::Enforce,
            "an unset gate with a silent env must enforce",
        );
    }

    #[rstest]
    #[case(ThermalGate::Skip, false, ThermalGate::Skip)]
    #[case(ThermalGate::Skip, true, ThermalGate::Skip)]
    #[case(ThermalGate::Enforce, false, ThermalGate::Enforce)]
    #[case(ThermalGate::Enforce, true, ThermalGate::Enforce)]
    #[case(ThermalGate::Unset, false, ThermalGate::Enforce)]
    #[case(ThermalGate::Unset, true, ThermalGate::Skip)]
    fn resolve_lets_an_explicit_argument_beat_the_environment(
        #[case] requested: ThermalGate,
        #[case] env_asks_skip: bool,
        #[case] want: ThermalGate,
    ) {
        assert_eq!(requested.resolve(env_asks_skip), want);
    }

    /// Platforms treat "not Skip" as enforcing, so `Unset` would work only by
    /// accident.
    #[rstest]
    #[case(ThermalGate::Unset)]
    #[case(ThermalGate::Enforce)]
    #[case(ThermalGate::Skip)]
    fn resolve_always_produces_a_decided_gate(#[case] requested: ThermalGate) {
        assert_ne!(requested.resolve(false), ThermalGate::Unset);
        assert_ne!(requested.resolve(true), ThermalGate::Unset);
    }

    #[test]
    fn platform_names_the_gate_this_target_runs() {
        let platform = platform();
        #[cfg(target_os = "android")]
        assert_eq!(platform, ReadinessPlatform::Android);
        #[cfg(target_os = "linux")]
        assert!(
            matches!(
                platform,
                ReadinessPlatform::Linux
                    | ReadinessPlatform::RaspberryPi5
                    | ReadinessPlatform::JetsonOrin
            ),
            "{platform:?}"
        );
        #[cfg(target_os = "macos")]
        assert_eq!(platform, ReadinessPlatform::MacOs);
        #[cfg(target_os = "windows")]
        assert_eq!(platform, ReadinessPlatform::Windows);
        #[cfg(not(any(
            target_os = "android",
            target_os = "linux",
            target_os = "macos",
            target_os = "windows"
        )))]
        assert_eq!(platform, ReadinessPlatform::Other);
    }

    #[rstest]
    #[case::default(None, None, None, HeldTouch::Wait)]
    #[case::host(None, None, Some(HeldTouch::Fail), HeldTouch::Fail)]
    #[case::host_off(None, None, Some(HeldTouch::Off), HeldTouch::Off)]
    #[case::env_beats_host(None, Some(HeldTouch::Wait), Some(HeldTouch::Fail), HeldTouch::Wait)]
    #[case::flag_beats_env(Some(HeldTouch::Fail), Some(HeldTouch::Wait), None, HeldTouch::Fail)]
    #[case::flag_beats_host(Some(HeldTouch::Wait), None, Some(HeldTouch::Fail), HeldTouch::Wait)]
    #[case::off_beats_host(Some(HeldTouch::Off), None, Some(HeldTouch::Fail), HeldTouch::Off)]
    fn held_touch_resolves_flag_then_env_then_host(
        #[case] flag: Option<HeldTouch>,
        #[case] env: Option<HeldTouch>,
        #[case] host: Option<HeldTouch>,
        #[case] want: HeldTouch,
    ) {
        // The environment's value as `resolve_readiness` would read it, set
        // directly so the test does not race other tests on the process env.
        let resolved = ResolvedReadiness {
            held_touch: env,
            ..resolve_readiness(Some(Duration::from_secs(60)), ThermalGate::Enforce)
        };
        let table = ReadinessConfiguration::Android(AndroidReadinessConfiguration {
            held_touch: host,
            ..Default::default()
        });
        let resolved = resolved
            .with_configuration(Some(table))
            .with_held_touch(flag);
        assert_eq!(resolved.held_touch(), want);
    }

    #[rstest]
    #[case("wait", Some(HeldTouch::Wait))]
    #[case(" FAIL ", Some(HeldTouch::Fail))]
    #[case(" Off ", Some(HeldTouch::Off))]
    #[case("", None)]
    #[case("skip", None)]
    fn held_touch_parses_mode(#[case] raw: &str, #[case] want: Option<HeldTouch>) {
        assert_eq!(raw.parse::<HeldTouch>().ok(), want);
    }

    #[test]
    fn with_configuration_keeps_the_resolved_policy() {
        let resolved = resolve_readiness(Some(Duration::from_secs(60)), ThermalGate::Enforce);
        let table = ReadinessConfiguration::Windows(WindowsReadinessConfiguration::default());
        let tuned = resolved.clone().with_configuration(Some(table.clone()));
        assert_eq!(
            (tuned.max_wait, tuned.thermal),
            (resolved.max_wait, resolved.thermal)
        );
        assert_eq!(tuned.configuration, Some(table));
        assert_eq!(resolved.configuration, None);
    }
}
