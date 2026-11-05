import Testing
import Foundation
import SwiftUI
@testable import Pipette

/// How the "Device temperature" row renders a reading: `DeviceThermalDisplay`. The
/// reading it is handed comes from `JobRunner.deviceTemperatureC` (see `JobRunnerTests`).
@Suite @MainActor struct DeviceThermalDisplayTests {
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
