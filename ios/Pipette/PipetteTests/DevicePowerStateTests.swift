import Foundation
import Testing

@testable import Pipette

/// `DevicePowerState` in `PlanTypes/Thermal.swift`.
struct DevicePowerStateTests {
    /// The exact snake_case strings the management server's enum accepts.
    @Test func devicePowerStateWireFormMatchesServer() throws {
        for (state, wire) in [(DevicePowerState.charging, "charging"),
                              (.notCharging, "not_charging"),
                              (.pluggedInNotCharging, "plugged_in_not_charging")] {
            #expect(String(decoding: try JSONEncoder().encode(state), as: UTF8.self) == "\"\(wire)\"")
        }
    }
}
