import Foundation

// Per-iteration thermal telemetry, mirroring `pipette-plan-types/src/thermal.rs`.

/// Apple `ProcessInfo.thermalState` as the warehouse names it — the crate's
/// `AppleThermalState`, `snake_case` on the wire. A device-level enum: no temperature,
/// no headroom.
///
/// A type rather than a token string because the mapping has to be strict: a future OS
/// level (`@unknown`) has no member here, and dropping the field beats emitting a
/// spelling the server's enum would reject, which fails the whole submission.
///
/// TODO: review — mirrors `thermal.rs:62`; four variants and the snake_case wire
/// spelling checked against it. Nothing outstanding.
nonisolated enum AppleThermalState: String, Sendable, Codable {
    case nominal, fair, serious, critical

    /// The reading for a `ProcessInfo` state, or `nil` for one this build cannot name.
    init?(_ state: ProcessInfo.ThermalState) {
        switch state {
        case .nominal: self = .nominal
        case .fair: self = .fair
        case .serious: self = .serious
        case .critical: self = .critical
        @unknown default: return nil
        }
    }
}

/// Run-environment power state at benchmark time — the crate's `DevicePowerState`,
/// `snake_case` on the wire.
///
/// A type rather than a token string for the reason the crate gives: the management
/// server's enum rejects a misspelling, and hand-spelling the value at each detection
/// site is where that typo gets in. Replaces the earlier boolean "is charging", which
/// could not tell "plugged in and topping up" from "plugged in but holding" (battery
/// full or charge-limited) — both remove the battery current-limiting that can throttle
/// the SoC, and both differ from running on battery.
///
/// TODO: review — mirrors `thermal.rs:29`; three variants and the snake_case wire
/// spelling checked against it.
nonisolated enum DevicePowerState: String, Sendable, Codable {
    /// On external power and the battery is charging.
    case charging
    /// Running on battery (unplugged), discharging.
    case notCharging = "not_charging"
    /// On external power but not adding charge (battery full or charge-limited).
    case pluggedInNotCharging = "plugged_in_not_charging"
}

/// Volatile run-environment power state captured per benchmark submission — the crate's
/// `PowerState`, produced by `DeviceProbe.detectPowerState()`.
///
/// `powerSaveMode` is non-optional where the crate has `Option<bool>`: the crate leaves
/// room for a platform whose detector cannot answer, while `ProcessInfo` always does.
///
/// TODO: review — mirrors `thermal.rs:340`.
nonisolated struct PowerState: Equatable, Sendable {
    /// Battery charge percent (0–100); nil where unavailable (the simulator).
    var batteryLevel: Int32?
    /// Nil only when the state is genuinely unknown.
    var powerState: DevicePowerState?
    /// OS Low Power Mode active.
    var powerSaveMode: Bool
}

/// One sensor snapshot — the crate's `ThermalReading`, narrowed to the Apple family.
///
/// The crate carries an Android and a Linux family beside this one; each is populated
/// only by the platform that exposes it, so a phone fills these two and nothing else.
///
/// TODO: review — mirrors `thermal.rs:149`, narrowed to the Apple family. The crate's
/// Android and Linux fields are omitted deliberately (platform-populated), not missed.
nonisolated struct ThermalReading: Equatable, Sendable {
    var appleThermalState: AppleThermalState?
    /// Raw SoC die temperature (°C), from the private `PMU tdie*` sensors. Present only
    /// on a `PIPETTE_PRIVATE_THERMAL` build; `nil` marks a rep whose read failed.
    var appleSocTempC: Float?
}

/// Snapshots bracketing each measured repetition: `before[i]` at rep `i`'s gate-pass,
/// `after[i]` once its timed work completes. The crate's `RunThermal`.
///
/// Grouping the two series here is what lets them travel on a run's result instead of
/// out of band, and the submission builder flattens them back into the per-iteration
/// wire lists — as the crate's does.
///
/// TODO: review — mirrors `thermal.rs:171`. `series(states:socTemps:)` replaces the
/// crate's `from_pairs` because our two sides differ in length (the SoC one is compiled
/// out on a public build); confirm that divergence is acceptable.
nonisolated struct RunThermal: Equatable, Sendable {
    var before: [ThermalReading] = []
    var after: [ThermalReading] = []

    var isEmpty: Bool { before.isEmpty && after.isEmpty }

    /// Build one side's series from the parallel arrays the collector fills.
    ///
    /// The two are not the same length: the SoC series is compiled out entirely on a
    /// public build, so it is empty while the state series has one entry per rep. Index
    /// past either end reads as absent rather than truncating the other.
    static func series(states: [AppleThermalState], socTemps: [Float?]) -> [ThermalReading] {
        (0..<max(states.count, socTemps.count)).map { i in
            ThermalReading(
                appleThermalState: i < states.count ? states[i] : nil,
                appleSocTempC: i < socTemps.count ? socTemps[i] : nil)
        }
    }
}
