import Foundation

// Port of `crates/pipette-readiness/src/lib.rs`. `Callbacks`, `Wait` and `Other` port the crate
// modules of the same name; `Ios.swift` is this client's platform module.
//
// The values here decide which runs are comparable across the fleet, so they are the crate's:
// a platform overrides the deadline only with a measured basis stated at the override
// (PIP-278), and iOS has none.
//
// `Ops/Readiness.swift` is the seam an engine is handed; this is the wait behind it.

/// The fleet-wide deadline override, in whole seconds. Only a `devicectl` launch sets it;
/// one from the UI or a deep link keeps the default.
nonisolated let maxWaitEnv = "PIPETTE_READINESS_MAX_WAIT_SECS"

/// An unparseable value is ignored with a warning, so a typo cannot disable the wait.
nonisolated func envMaxWait() -> Duration? {
    guard let raw = ProcessInfo.processInfo.environment[maxWaitEnv] else { return nil }
    guard let secs = UInt64(raw.trimmingCharacters(in: .whitespacesAndNewlines)) else {
        AppLog.jobRun.warning(
            "readiness: ignoring unparseable \(maxWaitEnv)=\"\(raw)\"; "
                + "using the per-platform default")
        return nil
    }
    return .fromSecs(secs)
}

nonisolated let skipThermalEnv = "PIPETTE_READINESS_SKIP_THERMAL"

nonisolated func envSkipThermal() -> Bool {
    ProcessInfo.processInfo.environment[skipThermalEnv].map(skipThermalFromStr) ?? false
}

/// The one grammar for this variable and `readiness-skip-thermal=`: empty and every
/// spelling of "no" enforce, since a shell clears a variable by exporting it empty.
nonisolated func skipThermalFromStr(_ raw: String) -> Bool {
    !["", "0", "false", "f", "no", "n", "off"]
        .contains(raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased())
}

/// Three-valued so "no opinion" is distinguishable from "enforce": `unset` defers to
/// `PIPETTE_READINESS_SKIP_THERMAL`, while an explicit value overrides it.
nonisolated enum ThermalGate: Sendable, Equatable {
    /// The environment decides, and enforces when it is silent.
    case unset
    case enforce
    /// The CPU-load check still applies. A run that skips is not comparable to one that did
    /// not.
    case skip

    func skipped() -> Bool {
        self == .skip
    }

    /// Never `unset`.
    func resolve(envAsksSkip: Bool) -> ThermalGate {
        switch self {
        case .skip, .enforce: self
        case .unset: envAsksSkip ? .skip : .enforce
        }
    }
}

/// Block until the device is ready for a measurement, with the caller's own progress,
/// cancellation and sleep.
nonisolated func waitUntilReadyWith(_ resolved: ResolvedReadiness,
                                    callbacks: ReadinessCallbacks) -> ReadinessOutcome {
    waitUntilReady(maxWait: resolved.maxWait, thermal: resolved.thermal, callbacks: callbacks)
}

/// The readiness policy a wait will actually apply. Only `resolveReadiness` builds one (the
/// crate's `#[non_exhaustive]`): a hand-built value could skip the ceiling or carry `unset`.
nonisolated struct ResolvedReadiness: Sendable, Equatable {
    /// After the environment, the default and the ceiling.
    let maxWait: Duration
    /// Never `.unset`.
    let thermal: ThermalGate

    /// Nothing requested: the environment's overrides, else the defaults.
    init() {
        self = resolveReadiness(maxWait: nil, thermal: .unset)
    }

    fileprivate init(resolvedMaxWait: Duration, thermal: ThermalGate) {
        self.maxWait = resolvedMaxWait
        self.thermal = thermal
    }

    /// The deadline in whole seconds, as the plan's `readiness.max_wait_secs` spells it.
    func maxWaitSecs() -> UInt64 {
        UInt64(max(0, maxWait.components.seconds))
    }

    /// Whether the thermal criterion was waived, under the plan's name for it.
    func skipThermal() -> Bool {
        thermal.skipped()
    }
}

/// Resolve once per cell and reuse it: resolving again could straddle an environment change,
/// and each resolve of a skipped gate logs the warning again.
nonisolated func resolveReadiness(maxWait: Duration?, thermal: ThermalGate) -> ResolvedReadiness {
    let resolved = ResolvedReadiness(
        resolvedMaxWait: clampMaxWait(maxWait ?? envMaxWait() ?? defaultMaxWait),
        thermal: thermal.resolve(envAsksSkip: envSkipThermal()))
    if resolved.skipThermal() {
        AppLog.jobRun.warning(
            "readiness: thermal criterion DISABLED for this wait; only the load check "
                + "applies. Results are not comparable to thermally gated runs.")
    }
    return resolved
}

/// Every channel in (a plan's `max_wait_secs`, `readiness-max-wait-secs=`, the environment)
/// takes an arbitrary `u64` of seconds; a day is far past any real wait and keeps the
/// arithmetic on it (and the progress UI's clock) nowhere near a trap.
nonisolated let maxWaitCeiling: Duration = .seconds(24 * 60 * 60)

/// Clamps rather than rejects: an absurd deadline is a typo whose intent ("wait a long time")
/// the ceiling still serves.
nonisolated func clampMaxWait(_ requested: Duration) -> Duration {
    guard requested > maxWaitCeiling else { return requested }
    AppLog.jobRun.warning(
        "readiness: requested max wait \(requested.components.seconds)s exceeds the "
            + "\(maxWaitCeiling.components.seconds)s ceiling; using the ceiling instead")
    return maxWaitCeiling
}

/// How long one readiness wait (one rep) may take before failing the rep. Restates the crate's
/// `DEFAULT_MAX_WAIT` (Android: `Readiness.COOLDOWN_MAX_MILLIS`); changing it here makes this
/// client's runs incomparable with every other (PIP-278).
nonisolated let defaultMaxWait: Duration = .seconds(300)

/// Latency on every gated rep, not a criterion. At 10 s a measured cell spent 46 of 59 seconds
/// gating, mostly after the die was already under the setpoint (#1142).
nonisolated let sensorPollInterval: Duration = .seconds(3)

extension Duration {
    /// Swift's `Duration.seconds` traps past `Int64.max`, which a plan's `max_wait_secs` can
    /// carry; every such value is far past `maxWaitCeiling`, so saturating loses nothing.
    nonisolated static func fromSecs(_ secs: UInt64) -> Duration {
        .seconds(Int64(clamping: secs))
    }
}
