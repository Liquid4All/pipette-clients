//! Pre-benchmark readiness wait. Throughput-class benchmarks (prefill
//! / decode / e2e / vl) are sensitive to the host's runtime state:
//! starting a measurement while the OS is actively throttling, or
//! while a background process is hogging the CPU, produces
//! high-variance numbers that don't reflect steady-state performance.
//!
//! [`wait_until_ready`] is the single entry point — the benchmark calls
//! it without knowing what platform it's on. Each platform's submodule
//! owns its own probes (thermal + CPU load today, more signals later if
//! needed), poll loop, and "ready" criteria, plus a `DEFAULT_MAX_WAIT`
//! it considers reasonable. The deadline resolves by precedence:
//! an explicit `Some(max_wait)` (threaded from a cell's flags) wins,
//! else `PIPETTE_READINESS_MAX_WAIT_SECS`, else the platform default.
//!
//! Errors propagate via [`ReadinessError`]: a timeout is signalled
//! (not silently ignored) so the benchmark can fail the cell rather
//! than record polluted numbers. Probe-level failures (i/o, parse,
//! missing data) are reported separately so the caller can decide
//! whether a broken probe is fatal in their context.

use std::time::Duration;

#[cfg(target_os = "android")]
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
// Compiled into every test build, not just Windows ones: the Windows gate is
// the only platform here whose decision logic is a series test rather than a
// single-reading comparison, and CI has no Windows runner. Building it
// everywhere is what makes those tests run at all.
#[cfg(any(target_os = "windows", test))]
mod windows;

/// Name of the environment variable that overrides the per-platform readiness
/// deadline (whole seconds). Read by [`wait_until_ready`], which prefers an
/// explicit `max_wait` argument over it — that argument carries a cell's
/// `benchmarks run --readiness-max-wait-secs`, threaded through
/// `BenchmarkFlags`. This variable is the fleet-wide channel instead: the plan
/// runner sets it on the invocations it spawns, so an override reaches remote
/// runners over ssh/adb/slurm and not just the local ones that inherit its
/// environment.
pub const MAX_WAIT_ENV: &str = "PIPETTE_READINESS_MAX_WAIT_SECS";

/// Environment override for the readiness deadline, in whole seconds,
/// via `PIPETTE_READINESS_MAX_WAIT_SECS`. Returns `None` when unset; an
/// unparseable value is ignored (with a warning) so a typo can't
/// silently disable the wait. Honored on every runner CLI except iOS
/// (see [`wait_until_ready`]): lets a host that recovers unusually
/// slowly — e.g. the fanless MacBook Neo (A18 Pro) — wait for true
/// `nominal` instead of timing out, without recompiling.
#[cfg(not(target_os = "ios"))]
fn env_max_wait() -> Option<Duration> {
    let raw = std::env::var(MAX_WAIT_ENV).ok()?;
    match raw.trim().parse::<u64>() {
        Ok(secs) => Some(Duration::from_secs(secs)),
        Err(_) => {
            log::warn!(
                "readiness: ignoring unparseable PIPETTE_READINESS_MAX_WAIT_SECS={raw:?}; \
                 using the per-platform default"
            );
            None
        }
    }
}

/// iOS has no operator-set-env CLI surface (it drives readiness timing
/// through the host app harness), so the env override is never consulted.
#[cfg(target_os = "ios")]
fn env_max_wait() -> Option<Duration> {
    None
}

/// Name of the environment variable that disables the thermal criterion.
/// [`skip_thermal_from_str`] defines which values mean "skip". Read by
/// [`wait_until_ready`], which ORs it with its `thermal` argument — that
/// argument carries a cell's `benchmarks run --readiness-skip-thermal`,
/// threaded through `BenchmarkFlags`. This variable is the fleet-wide channel
/// instead, set by the plan runner on the invocations it spawns, exactly as
/// [`MAX_WAIT_ENV`] is.
pub const SKIP_THERMAL_ENV: &str = "PIPETTE_READINESS_SKIP_THERMAL";

#[cfg(not(target_os = "ios"))]
fn env_skip_thermal() -> bool {
    std::env::var(SKIP_THERMAL_ENV)
        .as_deref()
        .map(skip_thermal_from_str)
        .unwrap_or(false)
}

/// Whether a textual value asks for the thermal criterion to be skipped.
///
/// Absent, empty, and every spelling of "no" enforce — the default has to be
/// the safe one, because a shell that exports the variable empty (`FOO=`) is a
/// common way to *unset* it in practice, and an operator who writes `off`
/// plainly means "don't skip" rather than "skip".
///
/// This is the *only* grammar for the flag: it backs both [`SKIP_THERMAL_ENV`]
/// and `pipette-plan`'s `--readiness-skip-thermal`, which passes it to clap as
/// a `value_parser`. Keeping one function rather than pairing this with clap's
/// near-equivalent `FalseyValueParser` is deliberate — the two disagreed on
/// `off`/`no` and on surrounding whitespace, so a value could mean "enforce"
/// to the plan driver and "skip" to the runner it spawned.
pub fn skip_thermal_from_str(raw: &str) -> bool {
    !matches!(
        raw.trim().to_ascii_lowercase().as_str(),
        "" | "0" | "false" | "f" | "no" | "n" | "off"
    )
}

/// iOS drives readiness through the host app harness, not an env-set CLI.
#[cfg(target_os = "ios")]
fn env_skip_thermal() -> bool {
    false
}

/// Whether a wait must satisfy the platform's thermal criterion.
///
/// Named rather than a bare `bool` because the call sites read as policy
/// decisions, and `wait_until_ready(max_wait, true)` would not say which
/// way `true` points. Three-valued rather than two so "no opinion" is
/// distinguishable from "definitely enforce" — see [`wait_until_ready`] for
/// how each resolves against [`SKIP_THERMAL_ENV`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ThermalGate {
    /// No opinion; [`SKIP_THERMAL_ENV`] decides, and enforces when it is unset.
    ///
    /// The default, and the right value for a caller with no flag surface of
    /// its own — the Android JNI entry point takes no plan, so passing this is
    /// what lets a forwarded `PIPETTE_READINESS_SKIP_THERMAL` still reach it.
    #[default]
    Unset,
    /// Explicitly require the platform's thermal criterion, overriding
    /// [`SKIP_THERMAL_ENV`]. A cell that authored `skip_thermal = false` said
    /// something, and a stale export in the operator's shell should not undo it.
    Enforce,
    /// Skip the thermal criterion; the load criterion still applies.
    ///
    /// For hosts or workloads where the thermal signal costs more than it
    /// buys — most sharply on macOS, whose enum is a fixed ~318 s hold-off
    /// rather than a temperature (see `macos.rs`). The load check is
    /// deliberately still enforced: it catches a second benchmark running
    /// concurrently, which is a correctness problem unrelated to heat.
    ///
    /// A run that skips it is not comparable to one that did, so whether it was
    /// skipped has to be recoverable from the result. On the plan path it is
    /// not submitted at all, and leaves a trace only in the local `extras.json`
    /// argv; the Android app does submit it, as
    /// `benchmark_flags.readiness.skip_thermal`. Recorded either way is the
    /// thermal state each rep started at (`ThermalTelemetry`), which is the
    /// evidence that matters where the flag is absent: a gated run cannot have
    /// started outside the band, so a recorded reading outside it is itself the
    /// tell that it was waived.
    Skip,
}

impl ThermalGate {
    fn skipped(self) -> bool {
        self == Self::Skip
    }

    /// Fold this request together with [`SKIP_THERMAL_ENV`] into the gate a
    /// platform will actually apply — never [`ThermalGate::Unset`].
    ///
    /// An explicit value on either side wins over silence on the other; when
    /// both speak, the argument does, because it is the more specific
    /// authority (a cell, versus the whole process's environment).
    fn resolve(self, env_asks_skip: bool) -> Self {
        match self {
            Self::Skip => Self::Skip,
            Self::Enforce => Self::Enforce,
            Self::Unset if env_asks_skip => Self::Skip,
            Self::Unset => Self::Enforce,
        }
    }
}

/// Block until the device is ready for a measurement.
///
/// The deadline resolves by precedence: an explicit `Some(max_wait)`
/// (e.g. threaded from a cell's benchmark flags) wins; otherwise
/// `PIPETTE_READINESS_MAX_WAIT_SECS` (honored on every runner CLI
/// including the Android binary over `adb shell`, but not iOS);
/// otherwise the per-platform default. The deadline never changes the
/// criteria, so measurements stay comparable — it only trades wall-clock
/// patience.
///
/// `thermal` *does* change the criteria, and resolves against
/// [`SKIP_THERMAL_ENV`] by `ThermalGate::resolve`:
///
/// | argument | env asks skip | result |
/// |---|---|---|
/// | [`ThermalGate::Skip`] | either | skip |
/// | [`ThermalGate::Enforce`] | either | **enforce** |
/// | [`ThermalGate::Unset`] | yes | skip |
/// | [`ThermalGate::Unset`] | no | enforce |
///
/// An explicit argument beats the environment in both directions: a cell that
/// authored `skip_thermal = false` should not be undone by a forgotten export,
/// and a caller with no flag surface passes `Unset` so a forwarded env still
/// reaches it. Enforce winning is the safe direction of the two.
pub fn wait_until_ready(resolved: &ResolvedReadiness) -> Result<(), ReadinessError> {
    dispatch(resolved.max_wait, resolved.thermal)
}

/// The readiness policy a run will actually apply, once the request, the
/// environment and the platform default have been folded together.
///
/// Separate from [`wait_until_ready`] so a caller can both gate on it and
/// *report* it. The values that matter are not knowable from the request
/// alone — an absent deadline becomes the platform's, an absent thermal
/// opinion resolves against [`SKIP_THERMAL_ENV`] — so a result that recorded
/// the request would describe a run that may not have happened.
/// `#[non_exhaustive]` so [`resolve_readiness`] is the only way to build one:
/// a hand-written literal could carry [`ThermalGate::Unset`], and `Unset`
/// enforces silently — skipping the environment waiver, the platform default
/// and the clamp without any of it being visible.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct ResolvedReadiness {
    /// The deadline actually used, after the env override, the platform
    /// default and [`clamp_max_wait`].
    pub max_wait: Duration,
    /// [`ThermalGate::Skip`] or [`ThermalGate::Enforce`] — never `Unset`,
    /// which is a question rather than an answer.
    pub thermal: ThermalGate,
}

impl ResolvedReadiness {
    /// The deadline in whole seconds.
    ///
    /// Named and scaled to match the plan's `readiness.max_wait_secs`, so a
    /// caller recording what ran copies a value rather than converting one —
    /// a wrong unit here would be silent in the result. [`Self::max_wait`]
    /// stays the honest value; this is only the shape the plan and the result
    /// schema use.
    ///
    /// Truncating, and reachable only if a sub-second deadline ever appears:
    /// every source today is whole seconds (the plan's `max_wait_secs`, the
    /// env override, the platform defaults).
    pub fn max_wait_secs(&self) -> u64 {
        self.max_wait.as_secs()
    }

    /// Whether the thermal criterion was waived, under the plan's own name for
    /// it (`readiness.skip_thermal`) rather than a second word for the same
    /// thing.
    pub fn skip_thermal(&self) -> bool {
        self.thermal.skipped()
    }
}

/// Fold a request together with the environment and the platform default into
/// the policy a run will apply. Resolve once per cell and reuse it: calling
/// this twice could straddle an environment change and produce a record that
/// disagrees with the gate.
pub fn resolve_readiness(max_wait: Option<Duration>, thermal: ThermalGate) -> ResolvedReadiness {
    let max_wait = clamp_max_wait(
        max_wait
            .or_else(env_max_wait)
            .unwrap_or_else(platform_default_max_wait),
    );
    let thermal = thermal.resolve(env_skip_thermal());
    // Once per cell, where the decision is made — not once per rep, which is
    // where it used to land when the wait resolved its own arguments.
    if thermal.skipped() {
        log::warn!(
            "readiness: thermal criterion DISABLED for this wait; only the load \
             check applies. Results are not comparable to thermally gated runs."
        );
    }
    ResolvedReadiness { max_wait, thermal }
}

/// Ceiling on any resolved readiness deadline.
///
/// Every platform does `Instant::now() + max_wait`, which *panics* when the
/// result can't be represented — and both channels into this crate accept an
/// arbitrary `u64` of seconds, so a mistyped or pasted
/// `PIPETTE_READINESS_MAX_WAIT_SECS` (or a plan literal) could abort the runner
/// outright. A day is orders of magnitude past any real wait — the longest
/// platform default is 7 minutes — while leaving the arithmetic nowhere near
/// its limit.
const MAX_WAIT_CEILING: Duration = Duration::from_secs(24 * 60 * 60);

/// Hold a resolved deadline to [`MAX_WAIT_CEILING`], warning when it bites.
///
/// Clamping rather than rejecting: an absurd deadline is a typo, and the
/// intent behind it ("wait a long time") is still served by the ceiling, so
/// there's no reason to fail a cell over it. It is logged because silently
/// ignoring an operator's number is its own kind of surprise.
fn clamp_max_wait(requested: Duration) -> Duration {
    if requested > MAX_WAIT_CEILING {
        log::warn!(
            "readiness: requested max wait {requested:?} exceeds the \
             {MAX_WAIT_CEILING:?} ceiling; using the ceiling instead"
        );
        return MAX_WAIT_CEILING;
    }
    requested
}

/// The default deadline one readiness wait may spend before it gives up, shared
/// by every platform that has not measured a reason to differ.
///
/// It bounds a **single** wait, not a cell: the gate runs before each measured
/// rep, so a multi-rep cell can spend a multiple of this in total.
///
/// 300 s is roughly 8x the ~30-40 s a phone empirically takes to fall from a
/// post-rep peak back to its entry threshold (see the `THERMAL_THRESHOLD_C`
/// notes in `android.rs`), so it does not bind normal cooling. What it
/// bounds is the other case: a device that will not reach the entry condition at
/// all, in a room above the setpoint or under external load. Giving up there is
/// the point, because the alternative is measuring on hot silicon and reporting
/// the throttle as throughput.
///
/// The clients carry this same number in their own languages
/// (`Readiness.COOLDOWN_MAX_MILLIS` on Android, `readinessTimeoutSeconds` on
/// iOS), since neither can read a Rust constant. They cite this one.
///
/// **A platform overrides this only with a measured basis stated at the
/// override.** Today exactly one does: macOS, whose thermal-state enum was
/// measured holding for 318 s after load stops, above this deadline (see
/// `macos.rs`'s `DEFAULT_MAX_WAIT` and `tools/macos-thermal-probe/README.md`). A
/// number chosen by feel is what left the Android CLI at 600 s while the Android
/// app used 180 s, so the same phone was judged by two different deadlines
/// depending on which binary launched the run (PIP-278).
///
/// The per-platform modules are `#[cfg]`-gated, so the references above are plain
/// code spans rather than intra-doc links: a link into `android.rs` cannot resolve
/// when docs are built on Linux, and `cargo doc --workspace` runs on all three.
pub const DEFAULT_MAX_WAIT: Duration = Duration::from_secs(300);

/// How often a waiting gate re-reads its sensors.
///
/// **Granularity, not criteria.** The thresholds decide whether a rep may start
/// and have to match every other client; this only decides how soon that is
/// noticed. A host that cools in a second still waits a whole interval, so this
/// is pure latency added to every gated rep — and there are several gates per
/// cell.
///
/// 3 s because the previous 10 s was measurably expensive and bought nothing.
/// On iOS a `decode_throughput_1024_256` cell took 59 s of which 46 s was
/// gating, against ~6.5 s of actual decode, most of the wait spent after the die
/// was already under the setpoint (#1142). The reads here are a sysfs or SMC
/// read — cheap enough that polling more often costs nothing measurable. The iOS
/// client uses this same 3 s and the Android app 2 s.
///
/// Windows does not use this, and is not an override of it: its gate judges
/// flatness across `FLAT_WINDOW_POLLS` samples, so its spacing is an input to the
/// criterion rather than latency around one. Two different quantities, kept under
/// two names — see `windows.rs`'s `POLL_INTERVAL`.
pub const SENSOR_POLL_INTERVAL: Duration = Duration::from_secs(3);

/// The reasonable deadline the running platform's submodule ships.
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

/// Route to the running platform's poll loop with the resolved deadline and
/// thermal policy.
fn dispatch(max_wait: Duration, thermal: ThermalGate) -> Result<(), ReadinessError> {
    #[cfg(target_os = "android")]
    return android::wait_until_ready(max_wait, thermal);
    #[cfg(target_os = "linux")]
    return linux::wait_until_ready(max_wait, thermal);
    #[cfg(target_os = "macos")]
    return macos::wait_until_ready(max_wait, thermal);
    #[cfg(target_os = "windows")]
    return windows::wait_until_ready(max_wait, thermal);
    #[cfg(not(any(
        target_os = "android",
        target_os = "linux",
        target_os = "macos",
        target_os = "windows"
    )))]
    return other::wait_until_ready(max_wait, thermal);
}

#[derive(Debug, thiserror::Error)]
pub enum ReadinessError {
    /// Probes were readable but the device stayed non-ready (still
    /// throttled, still loaded, …) until the deadline passed. Caller
    /// should treat this as a hard failure for the cell — proceeding
    /// would record numbers under non-steady conditions.
    #[error("readiness wait timed out after {max_wait:?}: last seen {observed}")]
    TimedOut {
        max_wait: Duration,
        /// Platform-specific summary of the latest probe readings
        /// (e.g. `"android status=2 cpu_load=0.10"`,
        /// `"linux hottest=85°C cpu_load=0.70"`).
        observed: String,
    },
    /// I/O failure invoking a probe (process spawn, sysfs read).
    /// Source error is captured via `#[from]` so callers can drill in.
    #[error("readiness probe i/o failure")]
    Io(#[from] std::io::Error),
    /// Probe ran but its output didn't parse into a number we
    /// recognize. `raw` is the offending input verbatim.
    #[error("readiness probe could not parse output: {raw:?}")]
    ParseFailure { raw: String },
    /// Probe ran successfully but reported no usable data
    /// (missing expected field, empty result set). The message is
    /// `&'static str` because the cases are enumerable per platform.
    #[error("readiness probe unavailable: {0}")]
    Unavailable(&'static str),
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    /// A literal, not `DEFAULT_MAX_WAIT`, so retuning the shared deadline has to
    /// fail here and be looked at. The number is carried independently by two
    /// other languages that cannot reference this constant
    /// (`Readiness.COOLDOWN_MAX_MILLIS` on Android, `readinessTimeoutSeconds` on
    /// iOS) and it rides the wire as `benchmark_flags.readiness.max_wait_secs`,
    /// so a change here is a change to what a submitted result claims it waited
    /// for.
    #[test]
    fn shared_default_deadline_is_five_minutes() {
        assert_eq!(DEFAULT_MAX_WAIT, Duration::from_secs(300));
    }

    /// macOS is the one platform that overrides the shared default, and only
    /// because its thermal-state enum was measured holding for 318 s after load
    /// stops. Pinned as *greater than* the shared value: if a future edit brought
    /// it back down to the default, the gate would time out on fanless hosts
    /// exactly as it used to.
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

    /// The default must be "enforce" for anything ambiguous. An exported-empty
    /// variable (`PIPETTE_READINESS_SKIP_THERMAL=`) is how a shell commonly
    /// clears a setting, so it must not read as a request to skip.
    ///
    /// `off`/`no`/`n`/`f` are pinned as *enforce*: they are the spellings clap
    /// treats as false, and this function is the value parser behind
    /// `--readiness-skip-thermal`, so `--readiness-skip-thermal=off` and
    /// `PIPETTE_READINESS_SKIP_THERMAL=off` have to agree.
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

    /// The hazard the ceiling exists for is real, and the ceiling closes it.
    /// Every platform's wait loop does `Instant::now() + max_wait`, which
    /// panics on an unrepresentable sum; both halves are checked with
    /// `checked_add` so neither has to actually panic to prove the point.
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

    /// The clamp must not quietly shorten a legitimate wait — a fanless host
    /// genuinely needs ~7 minutes, and the ceiling itself is inclusive.
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

    /// The default must be "no opinion", not "enforce" — a caller that passes
    /// nothing has to leave `SKIP_THERMAL_ENV` able to speak, which is the only
    /// channel the Android JNI bridge has. It must still *resolve* to enforcing
    /// when the environment is silent.
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

    /// The full precedence table from [`wait_until_ready`]'s doc. The row that
    /// matters most is `(Enforce, env says skip) -> Enforce`: a cell that
    /// authored `skip_thermal = false` must not be ungated by a stale export.
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

    /// Resolution must never hand a platform `Unset` — every platform module
    /// treats "not Skip" as enforcing, so an unresolved value would work by
    /// accident rather than by construction.
    #[rstest]
    #[case(ThermalGate::Unset)]
    #[case(ThermalGate::Enforce)]
    #[case(ThermalGate::Skip)]
    fn resolve_always_produces_a_decided_gate(#[case] requested: ThermalGate) {
        assert_ne!(requested.resolve(false), ThermalGate::Unset);
        assert_ne!(requested.resolve(true), ThermalGate::Unset);
    }
}
