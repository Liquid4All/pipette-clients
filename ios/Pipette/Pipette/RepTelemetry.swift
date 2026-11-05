import Foundation

// `crates/pipette-cli/src/rep_telemetry.rs`: the device's readings at each rep's two ends
// and the readiness gate's report on the wait before it.

/// What this side samples at one end of a measured rep.
nonisolated struct RepSample: Sendable {
    let device: DeviceReadings
    /// The gate's report on the wait that released the rep; on a start only.
    let gate: GateWait?
}

nonisolated struct GateWait: Sendable {
    let pollCount: Int
    let last: ReadinessStatus

    func element(iteration: Int) -> GateWaitElement {
        GateWaitElement(iteration: iteration,
                        pollCount: pollCount,
                        platform: last.platform.asStr(),
                        phase: last.phase.asStr(),
                        elapsedMs: UInt64(max(0, durationMs(last.elapsed))),
                        observed: last.observed)
    }
}

/// Follows the gate's polls, and holds each passed wait until the rep it released starts.
///
/// `@unchecked Sendable` as `ThermalSeries` is: the gate reports on the thread that waits,
/// and the observer fires there too, one rep at a time.
nonisolated final class GateWaitRecorder: @unchecked Sendable {
    private var polls = 0
    private var last: ReadinessStatus?
    private var passed: GateWait?

    func begin() {
        polls = 0
        last = nil
    }

    func observe(_ status: ReadinessStatus) {
        polls += 1
        last = status
    }

    func pass() {
        passed = last.map { GateWait(pollCount: polls, last: $0) }
        last = nil
    }

    /// The wait that released the rep now starting. A gate inside a rep (between
    /// `vl_spec_decode`'s samples) releases none, so the next wait replaces it.
    func take() -> GateWait? {
        defer { passed = nil }
        return passed
    }
}

nonisolated enum SamplePoint: String, Encodable, Sendable {
    /// When the gate released the rep.
    case beforeRep = "before_rep"
    /// When the rep's timed work ended.
    case afterRep = "after_rep"
}

/// One element of `device_hardware_readings` or `device_os_throttling_readings`: each
/// facility read at one sample point, under its own key, as the OS returned it.
nonisolated struct ReadingsElement: Encodable, Equatable, Sendable {
    let iteration: Int
    let samplePoint: SamplePoint
    /// Beside `iteration` and `sample_point` on the wire, as the crate's `flatten` puts them.
    let facilities: [String: JsonValue]

    private struct Key: CodingKey {
        let stringValue: String
        var intValue: Int? { nil }

        init(_ stringValue: String) { self.stringValue = stringValue }
        init?(stringValue: String) { self.stringValue = stringValue }
        init?(intValue: Int) { nil }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: Key.self)
        try container.encode(iteration, forKey: Key("iteration"))
        try container.encode(samplePoint, forKey: Key("sample_point"))
        for (facility, value) in facilities {
            try container.encode(value, forKey: Key(facility))
        }
    }
}

/// One element of `readiness_gate_waits`: the gate's last report on the wait that released
/// a rep.
nonisolated struct GateWaitElement: Encodable, Equatable, Sendable {
    let iteration: Int
    let pollCount: Int
    let platform: String
    let phase: String
    let elapsedMs: UInt64
    let observed: String

    enum CodingKeys: String, CodingKey {
        case iteration
        case pollCount = "poll_count"
        case platform
        case phase
        case elapsedMs = "elapsed_ms"
        case observed
    }
}

/// Returned beside the response.
nonisolated struct RepTelemetry: Sendable {
    /// `device_hardware_readings`: one element per rep and sample point.
    var hardwareReadings: [ReadingsElement] = []
    /// `device_os_throttling_readings`, likewise.
    var osThrottlingReadings: [ReadingsElement] = []
    /// `readiness_gate_waits`: one element per rep the gate released.
    var gateWaits: [GateWaitElement] = []

    /// From each rep's `(start, end)` samples, in rep order.
    static func fromReps(_ reps: [(start: RepSample, end: RepSample)]) -> RepTelemetry {
        let points = reps.enumerated().flatMap { iteration, rep in
            [(iteration: iteration, samplePoint: SamplePoint.beforeRep, device: rep.start.device),
             (iteration: iteration, samplePoint: SamplePoint.afterRep, device: rep.end.device)]
        }
        return RepTelemetry(
            hardwareReadings: points.compactMap {
                element($0.iteration, $0.samplePoint, $0.device.hardware)
            },
            osThrottlingReadings: points.compactMap {
                element($0.iteration, $0.samplePoint, $0.device.osThrottling)
            },
            gateWaits: reps.enumerated().compactMap { iteration, rep in
                rep.start.gate?.element(iteration: iteration)
            })
    }
}

/// One sample point's facilities, or none where it read nothing.
private nonisolated func element(_ iteration: Int, _ samplePoint: SamplePoint,
                                 _ facilities: [String: JsonValue]) -> ReadingsElement? {
    facilities.isEmpty
        ? nil
        : ReadingsElement(iteration: iteration, samplePoint: samplePoint, facilities: facilities)
}
