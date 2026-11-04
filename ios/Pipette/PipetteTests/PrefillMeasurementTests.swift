import Foundation
import Testing

@testable import Pipette

/// `BenchmarkMeasurement.prefill`, the crate's `pipette-ops/src/measurement/prefill.rs`,
/// case for case. Errors carry no context chain here, so a refusal is checked by its own
/// message.
struct PrefillMeasurementTests {
    private func reading(_ prefillTokens: Int, _ ms: Double) throws -> PrefillMeasurementReading {
        .perPrompt([try prompt(prefillTokens, ms)])
    }

    private func body(_ batchSize: UInt32) -> PrefillThroughputBenchmark {
        PrefillThroughputBenchmark(benchmarkId: BenchmarkId(validated: "prefill_throughput_512"),
                                   parameterPrefillTokens: 512,
                                   parameterBatchSize: BatchSize(validated: batchSize))
    }

    /// The entry behind an open gate, with no observer and no warm-up.
    private func unobserved(_ benchmark: PrefillThroughputBenchmark, reps: Int,
                            _ attempt: () throws -> PrefillMeasurementReading)
        async throws -> BenchmarkResultData {
        try await BenchmarkMeasurement.prefill(
            benchmark, reps: reps, readinessGate: {}, observer: noObserver(), warmup: .inline,
            ctx: ()) { _ in try attempt() }
    }

    private func prompt(_ prefillTokens: Int, _ ms: Double) throws
        -> PrefillMeasurementPerPromptReading {
        try .fromMillis(prefillTokens: try PositiveCount(prefillTokens), prefillTimeMs: ms)
    }

    private func prompts(_ prefillTokens: [Int]) throws -> PrefillMeasurementReading {
        .perPrompt(try prefillTokens.map { try prompt($0, 40) })
    }

    private func totals(_ prefillTokens: Int, _ ms: Float) throws -> PrefillMeasurementReading {
        .totals(PrefillMeasurementTotalsReading(prefillTokens: try PositiveCount(prefillTokens),
                                                prefillTimeMs: try PositiveMillis(ms)))
    }

    private func meanAndStddev(_ resultData: BenchmarkResultData) throws -> (Double, Double?) {
        guard case let .prefillThroughput(mean, stddev) = resultData else {
            throw StepFailed(event: "not a prefill result: \(resultData)")
        }
        return (mean, stddev)
    }

    /// A time the result could not divide by, or past `Float`, is no reading.
    @Test(arguments: [0.0, -1.0, Double.nan, Double.infinity, 1e300])
    func aTimeWithoutARateIsRefused(_ ms: Double) throws {
        let error = #expect(throws: PrefillMeasurementError.self) {
            try PrefillMeasurementPerPromptReading.fromMillis(
                prefillTokens: try PositiveCount(512), prefillTimeMs: ms)
        }
        guard case .invalidTime = error else {
            Issue.record("expected invalidTime, got \(String(describing: error))")
            return
        }
    }

    /// A stream's first token is the reading only when it is the only one.
    @Test(arguments: [(1, nil), (2, 2)] as [(Int, Int?)])
    func aStreamIsReadAtItsOnlyToken(_ completionTokens: Int, _ refused: Int?) throws {
        let got = Result {
            try PrefillMeasurementPerPromptReading.fromStream(
                prefillTokens: try PositiveCount(512),
                completionTokens: try PositiveCount(completionTokens), ttftMs: 40)
        }
        switch (got, refused) {
        case let (.success(reading), nil):
            #expect(reading == (try prompt(512, 40)))
        case let (.failure(error), tokens?):
            #expect(error as? PrefillMeasurementError == .mismatch(
                field: "completion_tokens", actual: try PositiveCount(tokens), expected: 1))
        default:
            Issue.record("refused \(String(describing: refused)), got \(got)")
        }
    }

    /// The engine's time is taken as given, per prompt or in totals, on the batch and
    /// prompt the entry asked for.
    @Test(arguments: [
        (1, "reading 512", "40"),
        (1, "reading 511", "engine returned prefill_tokens 511, expected 512"),
        (3, "prompts 512 511 512", "engine returned prefill_tokens 511, expected 512"),
        (1, "prompts", "engine answered 0 prompts, expected a batch of 1"),
        (1, "prompts 512 512", "engine answered 2 prompts, expected a batch of 1"),
        (4, "prompts 512 512 512", "engine answered 3 prompts, expected a batch of 4"),
        (4, "prompts 512 512 512 512 512", "engine answered 5 prompts, expected a batch of 4"),
        (4, "totals 2048", "40"),
        (1, "totals 512", "40"),
        (4, "totals 2044", "engine returned prefill_tokens 2044 over 4 prompts, expected 512 each"),
        (4, "totals 512", "engine returned prefill_tokens 512 over 4 prompts, expected 512 each"),
    ] as [(UInt32, String, String)])
    func aReadingBecomesThePrefillTime(_ batchSize: UInt32, _ shape: String,
                                       _ expected: String) async throws {
        let words = shape.split(separator: " ")
        let counts = words.dropFirst().compactMap { Int($0) }
        let given = switch words[0] {
        case "reading": try reading(counts[0], 40)
        case "prompts": try prompts(counts)
        default: try totals(counts[0], 40)
        }
        do {
            let resultData = try await unobserved(body(batchSize), reps: 5) { given }
            let (prefillTimeMs, _) = try meanAndStddev(resultData)
            #expect(Double(expected).map { abs(prefillTimeMs - $0) < 1e-9 } == true,
                    "\(prefillTimeMs), expected \(expected)")
        } catch let error as PrefillMeasurementError {
            #expect(error.errorDescription == expected)
        }
    }

    /// Readings of 1, 2, … ms reduce to their mean and sample stddev; one rep has no spread
    /// to report.
    @Test(arguments: [(5, 3.0, 2.5.squareRoot()), (1, 1.0, nil)] as [(Int, Double, Double?)])
    func theRepsReduceToMeanAndStddev(_ reps: Int, _ mean: Double, _ stddev: Double?)
        async throws {
        var ms = 0.0
        let resultData = try await unobserved(body(1), reps: reps) {
            ms += 1
            return try reading(512, ms)
        }
        let (prefillTimeMs, prefillTimeMsStddev) = try meanAndStddev(resultData)
        #expect(abs(prefillTimeMs - mean) < 1e-9)
        switch (prefillTimeMsStddev, stddev) {
        case let (got?, want?): #expect(abs(got - want) < 1e-9)
        case (nil, nil): break
        default: Issue.record("stddev \(String(describing: prefillTimeMsStddev))")
        }
    }

    /// A `once` warm-up is one more attempt, a whole batch of it, and its readings are not
    /// in the result; an `inline` one adds none.
    @Test(arguments: [
        (MeasurementWarmup.once, 1, [1000.0, 40, 40]),
        (.inline, 1, [40.0, 40]),
        (.once, 3, [1000.0, 40, 40]),
    ] as [(MeasurementWarmup, UInt32, [Double])])
    func onlyTheRepsAreReduced(_ warmup: MeasurementWarmup, _ batchSize: UInt32,
                               _ times: [Double]) async throws {
        var times = times[...]
        let resultData = try await BenchmarkMeasurement.prefill(
            body(batchSize), reps: 2, readinessGate: {}, observer: noObserver(), warmup: warmup,
            ctx: ()
        ) { _ in
            let ms = try next(&times)
            return .perPrompt(Array(repeating: try prompt(512, ms), count: Int(batchSize)))
        }
        #expect(times.isEmpty, "an attempt was not made")
        let (prefillTimeMs, _) = try meanAndStddev(resultData)
        #expect(abs(prefillTimeMs - 40) < 1e-9)
    }

    /// A `once` warm-up, checked, before the first gate, and none with `inline`; then gate,
    /// start, attempt, end and the check per rep. A failing step abandons the cell there.
    @Test(arguments: [
        (MeasurementWarmup.once, nil,
         ["warm-up", "gate", "start", "attempt", "end", "gate", "start", "attempt", "end"]),
        (.inline, nil, ["gate", "start", "attempt", "end", "gate", "start", "attempt", "end"]),
        (.once, "warm-up", ["warm-up"]),
        (.once, "warm-up check", ["warm-up"]),
        (.once, "gate", ["warm-up", "gate"]),
        (.once, "start", ["warm-up", "gate", "start"]),
        (.once, "attempt", ["warm-up", "gate", "start", "attempt"]),
        (.once, "end", ["warm-up", "gate", "start", "attempt", "end"]),
        (.once, "check", ["warm-up", "gate", "start", "attempt", "end"]),
    ] as [(MeasurementWarmup, String?, [String])])
    func theStepsRunInOrderUntilOneFails(_ warmup: MeasurementWarmup, _ failing: String?,
                                         _ expected: [String]) async throws {
        let log = StepLog(failing: failing)
        // Nothing marks the warm-up to the attempt; it is the first call.
        var warming = warmup == .once
        let got = await outcome {
            try await BenchmarkMeasurement.prefill(
                body(1), reps: 2, readinessGate: { try log.step("gate") },
                observer: log.observer, warmup: warmup, ctx: ()
            ) { _ in
                let (event, refused) = warming ? ("warm-up", "warm-up check") : ("attempt", "check")
                warming = false
                try log.step(event)
                return try reading(failing == refused ? 511 : 512, 40)
            }
        }
        switch (got, failing) {
        case (.success, nil):
            break
        case let (.failure(error), "check"?), let (.failure(error), "warm-up check"?):
            #expect(error as? PrefillMeasurementError == .mismatch(
                field: "prefill_tokens", actual: try PositiveCount(511), expected: 512))
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
            try await BenchmarkMeasurement.prefill(
                body(1), reps: 5, readinessGate: {}, observer: noObserver(), warmup: .once,
                ctx: recorded
            ) { recorded in
                recorded.attempts.append("attempt \(recorded.attempts.count)")
                if recorded.attempts.count == failsOn { throw ServerWentAway() }
                return try reading(512, 40)
            }
        }
        #expect(recorded.attempts == (0 ..< failsOn).map { "attempt \($0)" })
    }

    /// Rep `r` reads `10r + 0..3`, so it averages to `10r + 1.5`; the stddev is across reps
    /// only.
    @Test func aBatchReducesToItsPerRequestMean() async throws {
        var rep = 0.0
        let resultData = try await unobserved(body(4), reps: 5) {
            rep += 10
            return .perPrompt(try [0.0, 1, 2, 3].map { try prompt(512, rep + $0) })
        }
        let (prefillTimeMs, prefillTimeMsStddev) = try meanAndStddev(resultData)
        #expect(abs(prefillTimeMs - 31.5) < 1e-9)
        let stddev = try #require(prefillTimeMsStddev, "five reps have a spread")
        #expect(abs(stddev - 250.0.squareRoot()) < 1e-9)
    }
}
