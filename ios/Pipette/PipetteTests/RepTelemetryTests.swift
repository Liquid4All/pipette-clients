import Foundation
import Testing

@testable import Pipette

/// Ports `crates/pipette-cli/src/rep_telemetry.rs`; the first three cases are its tests of
/// the same names.
struct RepTelemetryTests {
    private func status(_ phase: ReadinessPhase, _ observed: String,
                        elapsedMs: Int64) -> ReadinessStatus {
        ReadinessStatus(phase: phase, platform: .ios, observed: observed,
                        elapsed: .milliseconds(elapsedMs), maxWait: .seconds(300))
    }

    private func sample(_ hardware: [String: JsonValue], _ gate: GateWait?) -> RepSample {
        RepSample(device: DeviceReadings(hardware: hardware), gate: gate)
    }

    private func json(_ value: some Encodable) throws -> NSObject {
        let data = try JSONEncoder().encode(value)
        return try #require(try JSONSerialization.jsonObject(with: data) as? NSObject)
    }

    private let released = "thermal=state:nominal die:34C(ok) cpu=busy:0.4cores(ok)"

    /// A wait's record is its own polls and its last report, held for the rep it released.
    @Test func aPassedWaitIsHeldForItsRep() throws {
        let recorder = GateWaitRecorder()
        recorder.begin()
        recorder.observe(status(.waiting(pollIn: .seconds(5)), "die:42C(hot)", elapsedMs: 1_200))
        recorder.observe(status(.proceeding, released, elapsedMs: 10_700))
        recorder.pass()

        let wait = try #require(recorder.take())
        #expect(wait.element(iteration: 0)
            == GateWaitElement(iteration: 0, pollCount: 2, platform: "ios",
                               phase: "proceeding", elapsedMs: 10_700, observed: released))
        #expect(recorder.take() == nil, "taken once")
    }

    /// A gate inside a rep releases none, so the next rep gets the next wait, counted from
    /// its own first poll.
    @Test func aWaitInsideARepIsReplacedByTheNext() throws {
        let recorder = GateWaitRecorder()
        for polls in [3, 1] {
            recorder.begin()
            for _ in 0 ..< polls { recorder.observe(status(.proceeding, "ok", elapsedMs: 5)) }
            recorder.pass()
        }

        #expect(try #require(recorder.take()).pollCount == 1)
    }

    /// One element per rep and sample point that read something, each naming both; the
    /// gate's report once per rep.
    @Test func repsBecomeOneElementPerSamplePoint() throws {
        let sensors: JsonValue = .array([.object(["Product": .string("PMU tdie1"),
                                                  "TemperatureLevel": .double(45.5)])])
        let hardware = ["apple_iohid_temperature_sensors": sensors]
        let gate = GateWait(pollCount: 1, last: status(.proceeding, "ok", elapsedMs: 40))
        let telemetry = RepTelemetry.fromReps([
            (start: sample(hardware, gate), end: sample(hardware, nil)),
            // A rep whose end read nothing posts its start alone.
            (start: sample(hardware, nil), end: sample([:], nil)),
        ])

        #expect(telemetry.hardwareReadings.map(\.iteration) == [0, 0, 1])
        #expect(telemetry.hardwareReadings.map(\.samplePoint) == [.beforeRep, .afterRep, .beforeRep])
        // On the wire the facilities sit beside the rep and point, not under a key of
        // their own.
        #expect(try json(telemetry.hardwareReadings[0]) == [
            "iteration": 0, "sample_point": "before_rep",
            "apple_iohid_temperature_sensors": [["Product": "PMU tdie1", "TemperatureLevel": 45.5]],
        ] as NSDictionary)
        #expect(telemetry.osThrottlingReadings.isEmpty)
        #expect(telemetry.gateWaits.map(\.iteration) == [0])
    }

    /// IOHID's records keep its keys, and a service missing one keeps the other.
    @Test func eachIohidSensorKeepsItsOwnKeys() throws {
        let sensors = appleIohidTemperatureSensors([
            ["Product": "PMU tdie1", "TemperatureLevel": NSNumber(value: 45.5)],
            ["Product": "gas gauge battery"],
            ["TemperatureLevel": NSNumber(value: 30.25)],
        ])

        #expect(try json(sensors) == [
            ["Product": "PMU tdie1", "TemperatureLevel": 45.5],
            ["Product": "gas gauge battery"],
            ["TemperatureLevel": 30.25],
        ] as NSArray)
    }

    /// The OS's thermal state goes out as its raw code, nothing mapped to a name.
    @Test func theThermalStateIsItsRawValue() {
        #expect(detectDeviceReadings().osThrottling["apple_process_info_thermal_state"]
            == .int(ProcessInfo.processInfo.thermalState.rawValue))
    }

    @Test func phasesReadAsTheCrateWritesThem() {
        #expect(ReadinessPhase.waiting(pollIn: .seconds(1)).asStr() == "waiting")
        #expect(ReadinessPhase.proceeding.asStr() == "proceeding")
        #expect(ReadinessPhase.timedOut.asStr() == "timed_out")
    }
}
