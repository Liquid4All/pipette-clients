import Testing
import Foundation
import SwiftUI
@testable import Pipette

/// `JobRunner.deviceTemperatureC` picks the temperature the UI shows: the gate's
/// live reading while cooling, else a direct SoC read, else nil (never -1).
@Suite @MainActor struct DeviceTemperatureTests {
    @Test func usesGateReadingWhileCooling() {
        let runner = JobRunner()
        runner.readinessStatus = ReadinessStatus(
            phase: .waiting, temperatureC: 41.6, thermalStateLabel: "fair",
            thresholdC: 36, elapsedSeconds: 5, maxSeconds: 300, action: "waiting")
        #expect(runner.deviceTemperatureC == 41.6)
    }

    @Test func nilWithoutSensorSignal() {
        let runner = JobRunner()
        // No gate reading, and the test host has no private SoC sensor
        // (`pipette_soc_temp` → -1 without PIPETTE_PRIVATE_THERMAL), so there is
        // no temperature to display.
        #expect(runner.deviceTemperatureC == nil)
    }

    // MARK: - How the "Device temperature" row renders

    @Test func coolingShowsReadingArrowTargetInOrangeAboveSetpoint() {
        let cooling = JobCoolingState(since: Date(), deadline: 300, targetC: 36)
        let display = DeviceThermalDisplay(temperatureC: 43, cooling: cooling, state: .nominal)
        #expect(display.text == "43°C → 36°C")
        #expect(display.iconName == "thermometer.high")
        #expect(display.color == .orange)   // above target, even though OS state is nominal
    }

    @Test func coolingTurnsGreenOnceAtSetpoint() {
        let cooling = JobCoolingState(since: Date(), deadline: 300, targetC: 36)
        let display = DeviceThermalDisplay(temperatureC: 36, cooling: cooling, state: .nominal)
        #expect(display.text == "36°C → 36°C")
        #expect(display.color == .green)
    }

    @Test func notCoolingShowsBareReadingWithStateTint() {
        let display = DeviceThermalDisplay(temperatureC: 40, cooling: nil, state: .fair)
        #expect(display.text == "40°C")
        #expect(display.iconName == ProcessInfo.ThermalState.fair.iconName)
        #expect(display.color == ProcessInfo.ThermalState.fair.indicatorColor)
    }

    @Test func noSensorFallsBackToStateLabel() {
        let display = DeviceThermalDisplay(temperatureC: nil, cooling: nil, state: .nominal)
        #expect(display.text == "Nominal")
        #expect(display.color == ProcessInfo.ThermalState.nominal.indicatorColor)
    }
}
