import Foundation

// The run-environment power state: pipette-mgmt's
// `crates/pipette-plan-types/src/thermal.rs`. Per-rep thermal readings are in
// `RepTelemetry.swift`.

/// The crate's `DevicePowerState`. Plugged in at all removes the battery current limit
/// that can throttle the SoC, which is why the two plugged-in states differ from battery.
nonisolated enum DevicePowerState: String, Sendable, Codable {
    case charging
    /// Running on battery.
    case notCharging = "not_charging"
    /// Battery full or charge-limited.
    case pluggedInNotCharging = "plugged_in_not_charging"
}

/// The crate's `PowerReading`, from `detectPowerState()`. `powerSaveMode` is
/// non-optional where the crate's is `Option<bool>`: `ProcessInfo` always answers.
nonisolated struct PowerReading: Equatable, Sendable {
    /// Percent, 0 to 100; nil in the simulator.
    var batteryLevel: Int32?
    var powerState: DevicePowerState?
    /// OS Low Power Mode.
    var powerSaveMode: Bool
}
