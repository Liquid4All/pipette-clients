import Foundation
import Testing

@testable import Pipette

/// `BenchmarkMeasurement.endToEnd`, the crate's
/// `pipette-ops/src/measurement/end_to_end.rs`. Errors carry no context chain here, so a
/// refusal is checked by its own message.
struct EndToEndMeasurementTests {
    private func body(_ batchSize: UInt32) -> EndToEndLatencyBenchmark {
        EndToEndLatencyBenchmark(benchmarkId: BenchmarkId(validated: "end_to_end_latency_512_256"),
                                 parameterPrefillTokens: 512, parameterDecodeTokens: 256,
                                 parameterBatchSize: BatchSize(validated: batchSize))
    }

    /// The entry behind an open gate, with no observer and no warm-up.
    private func unobserved(_ benchmark: EndToEndLatencyBenchmark, reps: Int,
                            _ attempt: () throws -> EndToEndMeasurementReading)
        async throws -> BenchmarkResultData {
        try await BenchmarkMeasurement.endToEnd(
            benchmark, reps: reps, readinessGate: {}, observer: noObserver(), warmup: .inline,
            ctx: ()) { _ in try attempt() }
    }

    private func prompt(_ prefillTokens: Int, _ decodeTokens: Int, _ ms: Double) throws
        -> EndToEndMeasurementPerPromptReading {
        try .fromRequest(prefillTokens: try PositiveCount(prefillTokens),
                         decodeTokens: try PositiveCount(decodeTokens), elapsedMs: ms)
    }

    private func reading(_ prefillTokens: Int, _ decodeTokens: Int, _ ms: Double) throws
        -> EndToEndMeasurementReading {
        .perPrompt([try prompt(prefillTokens, decodeTokens, ms)])
    }

    private func prompts(_ decodeTokens: [Int]) throws -> EndToEndMeasurementReading {
        .perPrompt(try decodeTokens.map { try prompt(512, $0, 900) })
    }

    private func totals(_ prefillTokens: Int, _ decodeTokens: Int) throws
        -> EndToEndMeasurementReading {
        .totals(EndToEndMeasurementTotalsReading(prefillTokens: try PositiveCount(prefillTokens),
                                                 decodeTokens: try PositiveCount(decodeTokens),
                                                 totalTimeMs: try PositiveMillis(900)))
    }

    private func meanAndStddev(_ resultData: BenchmarkResultData) throws -> (Double, Double?) {
        guard case let .endToEndLatency(mean, stddev) = resultData else {
            throw StepFailed(event: "not an end-to-end result: \(resultData)")
        }
        return (mean, stddev)
    }

    /// A time the result could not divide by, or past `Float`, is no reading.
    @Test(arguments: [0.0, -1.0, Double.nan, Double.infinity, 1e300])
    func aRequestWithoutAValidTimeIsRefused(_ ms: Double) throws {
        let error = #expect(throws: EndToEndMeasurementError.self) {
            try EndToEndMeasurementPerPromptReading.fromRequest(
                prefillTokens: try PositiveCount(512), decodeTokens: try PositiveCount(256),
                elapsedMs: ms)
        }
        guard case .invalidTime = error else {
            Issue.record("expected invalidTime, got \(String(describing: error))")
            return
        }
    }

    /// The engine's time is taken as given, per prompt or in totals, on the batch and
    /// workload the entry asked for.
    @Test(arguments: [
        (1, "reading 512 256", "900"),
        (1, "reading 512 255", "engine returned decode_tokens 255, expected 256"),
        (1, "reading 511 256", "engine returned prefill_tokens 511, expected 512"),
        (3, "prompts 256 255 256", "engine returned decode_tokens 255, expected 256"),
        (1, "prompts", "engine answered 0 prompts, expected a batch of 1"),
        (1, "prompts 256 256", "engine answered 2 prompts, expected a batch of 1"),
        (4, "prompts 256 256 256", "engine answered 3 prompts, expected a batch of 4"),
        (4, "prompts 256 256 256 256 256", "engine answered 5 prompts, expected a batch of 4"),
        (4, "totals 2048 1024", "900"),
        (1, "totals 512 256", "900"),
        (4, "totals 2048 1023", "engine returned decode_tokens 1023 over 4 prompts, expected 256 each"),
        (4, "totals 2044 1024", "engine returned prefill_tokens 2044 over 4 prompts, expected 512 each"),
        (4, "totals 512 256", "engine returned prefill_tokens 512 over 4 prompts, expected 512 each"),
    ] as [(UInt32, String, String)])
    func aReadingBecomesTheTotalTime(_ batchSize: UInt32, _ shape: String,
                                     _ expected: String) async throws {
        let words = shape.split(separator: " ")
        let counts = words.dropFirst().compactMap { Int($0) }
        let given = switch words[0] {
        case "reading": try reading(counts[0], counts[1], 900)
        case "prompts": try prompts(counts)
        default: try totals(counts[0], counts[1])
        }
        do {
            let resultData = try await unobserved(body(batchSize), reps: 5) { given }
            let (totalTimeMs, _) = try meanAndStddev(resultData)
            #expect(Double(expected).map { abs(totalTimeMs - $0) < 1e-9 } == true,
                    "\(totalTimeMs), expected \(expected)")
        } catch let error as EndToEndMeasurementError {
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
            return try reading(512, 256, ms)
        }
        let (totalTimeMs, totalTimeMsStddev) = try meanAndStddev(resultData)
        #expect(abs(totalTimeMs - mean) < 1e-9)
        switch (totalTimeMsStddev, stddev) {
        case let (got?, want?): #expect(abs(got - want) < 1e-9)
        case (nil, nil): break
        default: Issue.record("stddev \(String(describing: totalTimeMsStddev))")
        }
    }

    /// A `once` warm-up is one more attempt, and its reading is not in the result; an
    /// `inline` one adds none.
    @Test(arguments: [
        (MeasurementWarmup.once, [1000.0, 900, 900]),
        (.inline, [900.0, 900]),
    ] as [(MeasurementWarmup, [Double])])
    func onlyTheRepsAreReduced(_ warmup: MeasurementWarmup, _ times: [Double]) async throws {
        var times = times[...]
        let resultData = try await BenchmarkMeasurement.endToEnd(
            body(1), reps: 2, readinessGate: {}, observer: noObserver(), warmup: warmup, ctx: ()
        ) { _ in
            try reading(512, 256, try next(&times))
        }
        #expect(times.isEmpty, "an attempt was not made")
        let (totalTimeMs, _) = try meanAndStddev(resultData)
        #expect(abs(totalTimeMs - 900) < 1e-9)
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
            try await BenchmarkMeasurement.endToEnd(
                body(1), reps: 2, readinessGate: { try log.step("gate") },
                observer: log.observer, warmup: .once, ctx: ()
            ) { _ in
                let (event, refused) = warming ? ("warm-up", "warm-up check") : ("attempt", "check")
                warming = false
                try log.step(event)
                return try reading(512, failing == refused ? 255 : 256, 900)
            }
        }
        switch (got, failing) {
        case (.success, nil):
            break
        case let (.failure(error), "check"?), let (.failure(error), "warm-up check"?):
            #expect(error as? EndToEndMeasurementError == .mismatch(
                field: "decode_tokens", actual: try PositiveCount(255), expected: 256))
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
            try await BenchmarkMeasurement.endToEnd(
                body(1), reps: 5, readinessGate: {}, observer: noObserver(), warmup: .once,
                ctx: recorded
            ) { recorded in
                recorded.attempts.append("attempt \(recorded.attempts.count)")
                if recorded.attempts.count == failsOn { throw ServerWentAway() }
                return try reading(512, 256, 900)
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
            return .perPrompt(try [0.0, 1, 2, 3].map { try prompt(512, 256, rep + $0) })
        }
        let (totalTimeMs, totalTimeMsStddev) = try meanAndStddev(resultData)
        #expect(abs(totalTimeMs - 31.5) < 1e-9)
        let stddev = try #require(totalTimeMsStddev, "five reps have a spread")
        #expect(abs(stddev - 250.0.squareRoot()) < 1e-9)
    }
}
