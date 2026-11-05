import Foundation
import Testing

@testable import Pipette

/// `Readiness/Ios.swift`, the platform module, tested as `macos.rs` is where the two share a
/// criterion.
struct ReadinessIosTests {

    /// The state reads `nominal` at every one of these, and a nil reading is the waiver. The
    /// IMU estimate is held to its own, margin-padded setpoint.
    @Test(arguments: [
        (TemperatureSignal.die(35), true), (.die(35.9), true), (.die(36), false),
        (.die(42), false), (.imu(37), true), (.imu(38), false), (nil, true),
    ] as [(TemperatureSignal?, Bool)])
    func dieCriterionRejectsAHotDeviceTheStateCallsNominal(signal: TemperatureSignal?,
                                                           wantOk: Bool) {
        #expect(dieOk(signal) == wantOk)
        #expect(thermalOk(state: .nominal, signal: signal, thermal: .enforce) == wantOk)
    }

    @Test func thermalPressureWaitsWhateverTheTemperature() {
        for state in [ProcessInfo.ThermalState.fair, .serious, .critical] {
            for signal in [nil, TemperatureSignal.die(30)] {
                #expect(!thermalOk(state: state, signal: signal, thermal: .enforce))
            }
        }
    }

    /// `skip` waives every thermal signal, the state as well as the temperature, as
    /// `macos.rs` and `android.rs` do.
    @Test func skipThermalWaivesEveryThermalSignal() {
        for state in [ProcessInfo.ThermalState.nominal, .fair, .serious, .critical] {
            for signal in [nil, TemperatureSignal.die(30), .die(80)] {
                #expect(thermalOk(state: state, signal: signal, thermal: .skip))
            }
        }
    }

    /// One saturated core is a full core of competition however many cores the device has,
    /// yet as a fraction of capacity it passes a 0.30 ratio from 4 cores up, so the gate
    /// counts busy cores, not a ratio.
    @Test(arguments: [(4.0, 1.0, false), (18.0, 1.0, false), (18.0, 2.36, false),
                      (18.0, 0.23, true), (18.0, 0.37, true)] as [(Double, Double, Bool)])
    func onePinnedCoreFailsReadinessAtAnyCoreCount(cores: Double, busyCores: Double,
                                                   wantOk: Bool) {
        #expect(cpuOk(busyCores: busyCores) == wantOk)
        if cores >= 18 {
            #expect(busyCores / cores < 0.30)
        }
    }

    /// An unreadable load fails, since every default is wrong in the unsafe direction.
    @Test func anUnreadableLoadFailsReadiness() {
        #expect(!cpuOk(busyCores: nil))
    }

    /// The shape of `macos.rs`'s summary, with iOS's signals: the OS thermal state by name,
    /// the reading in whole degrees, and the load in busy cores.
    @Test func summaryReportsEverySignal() {
        #expect(formatSummary(state: .nominal, signal: .die(30), busyCores: 0.4, thermal: .enforce)
            == "thermal=state:nominal die:30C(ok) cpu=busy:0.4cores(ok)")
        #expect(formatSummary(state: .serious, signal: .die(40), busyCores: 6.2, thermal: .enforce)
            == "thermal=state:serious die:40C(hot) cpu=busy:6.2cores(busy)")
        // The IMU estimate is named for what it is, and judged against its own setpoint.
        #expect(formatSummary(state: .nominal, signal: .imu(37), busyCores: 0.4, thermal: .enforce)
            == "thermal=state:nominal imu:37C(ok) cpu=busy:0.4cores(ok)")
    }

    /// A waived gate is visible in the log, and still reports the state it declined to wait
    /// for.
    @Test func summaryMarksASkippedThermalGate() {
        #expect(formatSummary(state: .serious, signal: .die(30), busyCores: 0.4, thermal: .skip)
            == "thermal=state:serious[SKIPPED] die:30C(ok) cpu=busy:0.4cores(ok)")
    }

    /// "We never found out" and "we checked, it was fine" are different evidence.
    @Test func summaryDistinguishesAnUnreadableProbeFromNominal() {
        let unreadable = formatSummary(state: .nominal, signal: nil, busyCores: 0.4,
                                       thermal: .enforce)
        #expect(unreadable == "thermal=state:nominal die:unreadable cpu=busy:0.4cores(ok)")
        #expect(unreadable != formatSummary(state: .nominal, signal: .die(30), busyCores: 0.4,
                                            thermal: .enforce))
        #expect(formatSummary(state: .nominal, signal: .die(30), busyCores: nil, thermal: .enforce)
            == "thermal=state:nominal die:30C(ok) cpu=busy:unreadable")
    }

    /// Order is `[user, system, idle, nice]`, per `CPU_STATE_*`.
    @Test(arguments: [
        ([0, 0, 0, 0], [50, 50, 100, 0], 0.5),
        ([0, 0, 0, 0], [0, 0, 100, 0], 0.0),
        ([0, 0, 0, 0], [70, 30, 0, 0], 1.0),
        // Nice time counts as busy, not idle.
        ([0, 0, 0, 0], [0, 0, 50, 50], 0.5),
        // The counters did not refresh inside the window: no answer, not a made-up one.
        ([9, 9, 9, 9], [9, 9, 9, 9], nil),
        // 32-bit counters that wrapped read as the small forward delta they are.
        ([UInt32.max - 1, 0, UInt32.max - 4, 0], [1, 0, 1, 0], 1.0 / 3.0),
    ] as [([UInt32], [UInt32], Double?)])
    func busyFromTicksHandlesDeltasWrapsAndStalls(before: [UInt32], after: [UInt32],
                                                  want: Double?) throws {
        let got = busyFromTicks(before: before, after: after)
        guard let want else {
            #expect(got == nil)
            return
        }
        #expect(abs(try #require(got) - want) < 1e-9)
    }

    /// Below ~800 ms the samples that see a tick delta are biased high, failing readiness
    /// spuriously.
    @Test func cpuSampleWindowClearsTheCounterRefreshCadence() {
        #expect(cpuSampleWindow >= .milliseconds(800))
    }

    /// The arithmetic is pinned above; this pins the Mach call, through the resample.
    @Test func cpuLoadReturnsAPlausibleRatioOnThisHost() throws {
        let busy = try #require(readCpuLoad())
        #expect((0.0...1.0).contains(busy))
    }
}
