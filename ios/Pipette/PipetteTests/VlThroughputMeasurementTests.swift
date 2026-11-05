import Foundation
import Testing

@testable import Pipette

/// `BenchmarkMeasurement.vlThroughput`, the crate's
/// `pipette-ops/src/measurement/vl_throughput.rs`. Errors carry no context chain here, so
/// a refusal is checked by its type.
struct VlThroughputMeasurementTests {
    private let body = VlThroughputBenchmark(
        benchmarkId: BenchmarkId(validated: "vl_throughput_512x512_text0_decode64_img1"),
        parameterImageWidth: 512, parameterImageHeight: 512, parameterTextTokens: 0,
        parameterDecodeTokens: 64, parameterNumImages: 1)

    /// A reading whose whole request took as long as its two phases.
    private func reading(_ promptTokens: Int, _ imageTokens: Int?, _ decodeTokens: Int,
                         _ promptMs: Double, _ predictedMs: Double) throws
        -> VlThroughputMeasurementReading {
        try timedReading(promptTokens, imageTokens, decodeTokens, promptMs, predictedMs,
                         promptMs + predictedMs)
    }

    private func timedReading(_ promptTokens: Int, _ imageTokens: Int?, _ decodeTokens: Int,
                              _ promptMs: Double, _ predictedMs: Double, _ totalTimeMs: Double)
        throws -> VlThroughputMeasurementReading {
        try .fromMillis(promptTokens: try PositiveCount(promptTokens),
                        imageTokens: try imageTokens.map { try PositiveCount($0) },
                        decodeTokens: try PositiveCount(decodeTokens), promptMs: promptMs,
                        predictedMs: predictedMs, totalTimeMs: totalTimeMs)
    }

    /// The entry behind an open gate, with no observer and no warm-up.
    private func unobserved(reps: Int, _ attempt: () throws -> VlThroughputMeasurementReading)
        async throws -> BenchmarkResultData {
        try await BenchmarkMeasurement.vlThroughput(
            body, reps: reps, readinessGate: {}, observer: noObserver(), warmup: .inline, ctx: ()
        ) { _ in try attempt() }
    }

    /// Any time zero, non-finite or past `Float` is no reading, named by field.
    @Test(arguments: [
        (0.0, 40.0, 160.0, "prompt_ms"),
        (Double.nan, 40.0, 160.0, "prompt_ms"),
        (300.0, -1.0, 160.0, "predicted_ms"),
        (300.0, Double.infinity, 160.0, "predicted_ms"),
        (1e300, 40.0, 160.0, "prompt_ms"),
        (300.0, 40.0, 0.0, "total_time_ms"),
        (300.0, 40.0, Double.infinity, "total_time_ms"),
    ] as [(Double, Double, Double, String)])
    func aTimeWithoutARateIsRefused(_ promptMs: Double, _ predictedMs: Double,
                                    _ totalTimeMs: Double, _ field: String) throws {
        let error = #expect(throws: VlThroughputMeasurementError.self) {
            try timedReading(300, nil, 64, promptMs, predictedMs, totalTimeMs)
        }
        guard case let .invalidTime(got, _, _) = error else {
            Issue.record("expected invalidTime, got \(String(describing: error))")
            return
        }
        #expect(got == field)
    }

    /// The engine's times and counts are taken as given.
    @Test(arguments: [256, nil, 300] as [Int?])
    func aReadingBecomesTheResult(_ imageTokens: Int?) async throws {
        let given = try reading(300, imageTokens, 64, 120, 40)
        let resultData = try await unobserved(reps: 5) { given }
        #expect(resultData == .vlThroughput(
            promptTokens: 300, imageTokens: imageTokens.map(UInt32.init), promptMs: 120,
            promptMsStddev: 0, predictedMs: 40, predictedMsStddev: 0, totalTimeMs: 160,
            totalTimeMsStddev: 0))
    }

    /// A reading off the decode length the entry asked for, or with an image larger than
    /// its prompt, is refused.
    @Test(arguments: [(63, 256, "decode"), (64, 301, "image")] as [(Int, Int, String)])
    func aReadingOffTheEntryIsRefused(_ decodeTokens: Int, _ imageTokens: Int,
                                      _ kind: String) async throws {
        let given = try reading(300, imageTokens, decodeTokens, 120, 40)
        let error = await #expect(throws: VlThroughputMeasurementError.self) {
            try await unobserved(reps: 5) { given }
        }
        switch (error, kind) {
        case (.mismatch(field: "decode_tokens", _, _), "decode"),
             (.imageTokensExceedPrompt, "image"):
            break
        default:
            Issue.record("expected \(kind), got \(String(describing: error))")
        }
    }

    /// Prefills of 1, 2, … ms, decodes of 2, 4, … ms and totals of 4, 8, … ms, past the two
    /// phases' sum, reduce to their means and sample stddevs; one rep has no spread to report.
    @Test(arguments: [(5, 3.0, 2.5.squareRoot()), (1, 1.0, nil)] as [(Int, Double, Double?)])
    func theRepsReduceToMeanAndStddev(_ reps: Int, _ mean: Double, _ stddev: Double?)
        async throws {
        var ms = 0.0
        let resultData = try await unobserved(reps: reps) {
            ms += 1
            return try timedReading(300, 256, 64, ms, 2 * ms, 4 * ms)
        }
        guard case let .vlThroughput(_, _, promptMs, promptMsStddev, predictedMs,
                                     predictedMsStddev, totalTimeMs?,
                                     totalTimeMsStddev) = resultData
        else {
            Issue.record("not a timed VL throughput result: \(resultData)")
            return
        }
        #expect(abs(promptMs - mean) < 1e-9)
        #expect(abs(predictedMs - 2 * mean) < 1e-9)
        #expect(abs(totalTimeMs - 4 * mean) < 1e-9)
        switch (promptMsStddev, predictedMsStddev, totalTimeMsStddev, stddev) {
        case let (prompt?, predicted?, total?, want?):
            #expect(abs(prompt - want) < 1e-9)
            #expect(abs(predicted - 2 * want) < 1e-9)
            #expect(abs(total - 4 * want) < 1e-9)
        case (nil, nil, nil, nil):
            break
        default:
            let got = (promptMsStddev, predictedMsStddev, totalTimeMsStddev)
            Issue.record("stddev \(String(describing: got))")
        }
    }

    /// A rep whose prompt or image count differs from the first attempt's stops the cell at
    /// that rep. With a warm-up that attempt is the warm-up, so a prompt it left cached
    /// fails rep 1; without one, rep 2.
    @Test(arguments: [
        PromptChange(firstImageTokens: 256, promptTokens: 280, imageTokens: 256, kind: "prompt"),
        PromptChange(firstImageTokens: 256, promptTokens: 10, imageTokens: 256, kind: "prompt"),
        PromptChange(firstImageTokens: 256, promptTokens: 300, imageTokens: 255, kind: "image"),
        PromptChange(firstImageTokens: 256, promptTokens: 300, imageTokens: nil, kind: "image"),
        PromptChange(firstImageTokens: nil, promptTokens: 300, imageTokens: 256, kind: "image"),
    ], [MeasurementWarmup.once, .inline])
    func aPromptThatChangesAfterTheFirstAttemptIsRefused(_ change: PromptChange,
                                                         _ warmup: MeasurementWarmup)
        async throws {
        let kind = change.kind
        var attempts = [try reading(300, change.firstImageTokens, 64, 120, 40),
                        try reading(change.promptTokens, change.imageTokens, 64, 120, 40)][...]
        let log = StepLog()
        let error = await #expect(throws: VlThroughputMeasurementError.self) {
            try await BenchmarkMeasurement.vlThroughput(
                body, reps: 5, readinessGate: {}, observer: log.observer, warmup: warmup,
                ctx: ()
            ) { _ in try next(&attempts) }
        }
        switch (error, kind) {
        case (.promptVaried, "prompt"), (.imageTokensVaried, "image"):
            break
        default:
            Issue.record("expected \(kind), got \(String(describing: error))")
        }
        let reps = warmup == .once ? 1 : 2
        #expect(log.events == Array(repeating: ["start", "end"], count: reps).flatMap { $0 })
    }

    /// A `once` warm-up is one more attempt, and its reading is not in the result; an
    /// `inline` one adds none.
    @Test(arguments: [
        (MeasurementWarmup.once, [1000.0, 120, 120]),
        (.inline, [120.0, 120]),
    ] as [(MeasurementWarmup, [Double])])
    func onlyTheRepsAreReduced(_ warmup: MeasurementWarmup, _ times: [Double]) async throws {
        var times = times[...]
        let resultData = try await BenchmarkMeasurement.vlThroughput(
            body, reps: 2, readinessGate: {}, observer: noObserver(), warmup: warmup, ctx: ()
        ) { _ in
            try reading(300, 256, 64, try next(&times), 40)
        }
        #expect(times.isEmpty, "an attempt was not made")
        guard case let .vlThroughput(_, _, promptMs, _, _, _, _, _) = resultData else {
            Issue.record("not a VL throughput result: \(resultData)")
            return
        }
        #expect(abs(promptMs - 120) < 1e-9)
    }

    /// A `once` warm-up, checked, before the first gate; then gate, start, attempt, end
    /// and the check per rep. A failing step abandons the cell there.
    @Test(arguments: [
        (nil, ["warm-up", "gate", "start", "attempt", "end", "gate", "start", "attempt", "end"]),
        ("warm-up", ["warm-up"]),
        ("warm-up check", ["warm-up"]),
        ("gate", ["warm-up", "gate"]),
        ("start", ["warm-up", "gate", "start"]),
        ("attempt", ["warm-up", "gate", "start", "attempt"]),
        ("end", ["warm-up", "gate", "start", "attempt", "end"]),
        ("check", ["warm-up", "gate", "start", "attempt", "end"]),
    ] as [(String?, [String])])
    func theStepsRunInOrderUntilOneFails(_ failing: String?, _ expected: [String]) async throws {
        let log = StepLog(failing: failing)
        var warming = true
        let got = await outcome {
            try await BenchmarkMeasurement.vlThroughput(
                body, reps: 2, readinessGate: { try log.step("gate") }, observer: log.observer,
                warmup: .once, ctx: ()
            ) { _ in
                let (event, refused) = warming ? ("warm-up", "warm-up check") : ("attempt", "check")
                warming = false
                try log.step(event)
                return try reading(300, 256, failing == refused ? 63 : 64, 120, 40)
            }
        }
        switch (got, failing) {
        case (.success, nil):
            break
        case let (.failure(error), "check"?), let (.failure(error), "warm-up check"?):
            #expect(error as? VlThroughputMeasurementError == .mismatch(
                field: "decode_tokens", actual: try PositiveCount(63), expected: 64))
        case let (.failure(error), failing?):
            #expect(error as? StepFailed == StepFailed(event: failing))
        default:
            Issue.record("failing \(String(describing: failing)), got \(got)")
        }
        #expect(log.events == expected)
    }

    /// The context is lent to the warm-up and to each attempt in turn, and what they
    /// recorded is still the caller's after one fails; the attempt's error keeps its type.
    @Test(arguments: [1, 3])
    func theContextOutlivesAFailedAttempt(_ failsOn: Int) async throws {
        let recorded = Recorded()
        await #expect(throws: ServerWentAway.self) {
            try await BenchmarkMeasurement.vlThroughput(
                body, reps: 5, readinessGate: {}, observer: noObserver(), warmup: .once,
                ctx: recorded
            ) { recorded in
                recorded.attempts.append("attempt \(recorded.attempts.count)")
                if recorded.attempts.count == failsOn { throw ServerWentAway() }
                return try reading(300, 256, 64, 120, 40)
            }
        }
        #expect(recorded.attempts == (0 ..< failsOn).map { "attempt \($0)" })
    }
}

/// The second attempt of `aPromptThatChangesAfterTheFirstAttemptIsRefused`, against a
/// first one of 300 prompt tokens.
struct PromptChange: Sendable {
    let firstImageTokens: Int?
    let promptTokens: Int
    let imageTokens: Int?
    let kind: String
}
