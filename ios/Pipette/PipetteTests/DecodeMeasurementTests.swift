import Foundation
import Testing

@testable import Pipette

/// `BenchmarkMeasurement.decode`, the crate's `pipette-ops/src/measurement/decode.rs`.
/// Errors carry no context chain here, so a refusal is checked by its own message.
struct DecodeMeasurementTests {
    private func body(_ batchSize: UInt32) -> DecodeThroughputBenchmark {
        DecodeThroughputBenchmark(benchmarkId: BenchmarkId(validated: "decode_throughput_512_10"),
                                  parameterPrefillTokens: 512, parameterDecodeTokens: 10,
                                  parameterBatchSize: BatchSize(validated: batchSize))
    }

    /// The entry behind an open gate, with no observer and no warm-up.
    private func unobserved(_ benchmark: DecodeThroughputBenchmark, reps: Int,
                            _ attempt: () throws -> DecodeMeasurementReading)
        async throws -> BenchmarkResultData {
        try await BenchmarkMeasurement.decode(
            benchmark, reps: reps, readinessGate: {}, observer: noObserver(), warmup: .inline,
            ctx: ()) { _ in try attempt() }
    }

    private func prompt(_ prefillTokens: UInt32, _ decodeTokens: Int, _ ms: Double) throws
        -> DecodeMeasurementPerPromptReading {
        try .fromMillis(prefillTokens: prefillTokens,
                        decodeTokens: try PositiveCount(decodeTokens), decodeTimeMs: ms)
    }

    private func reading(_ prefillTokens: UInt32, _ decodeTokens: Int, _ ms: Double) throws
        -> DecodeMeasurementReading {
        .perPrompt([try prompt(prefillTokens, decodeTokens, ms)])
    }

    private func prompts(_ decodeTokens: [Int]) throws -> DecodeMeasurementReading {
        .perPrompt(try decodeTokens.map { try prompt(512, $0, 40) })
    }

    private func totals(_ prefillTokens: Int, _ decodeTokens: Int) throws
        -> DecodeMeasurementReading {
        .totals(DecodeMeasurementTotalsReading(prefillTokens: try PositiveCount(prefillTokens),
                                               decodeTokens: try PositiveCount(decodeTokens),
                                               decodeTimeMs: try PositiveMillis(40)))
    }

    private func fromStream(_ decodeTokens: Int, _ ttftMs: Double, _ lastTokenMs: Double)
        throws -> Result<DecodeMeasurementPerPromptReading, DecodeMeasurementError> {
        let decodeTokens = try PositiveCount(decodeTokens)
        return Result {
            try DecodeMeasurementPerPromptReading.fromStream(
                prefillTokens: try PositiveCount(512), decodeTokens: decodeTokens,
                ttftMs: ttftMs, lastTokenMs: lastTokenMs)
        }.mapError { $0 as? DecodeMeasurementError ?? .tooFewTokens(tokens: decodeTokens) }
    }

    private func meanAndStddev(_ resultData: BenchmarkResultData) throws -> (Double, Double?) {
        guard case let .decodeThroughput(mean, stddev) = resultData else {
            throw StepFailed(event: "not a decode result: \(resultData)")
        }
        return (mean, stddev)
    }

    /// A time the result could not divide by, or past `Float`, is no reading.
    @Test(arguments: [0.0, -1.0, Double.nan, Double.infinity, 1e300])
    func aTimeWithoutARateIsRefused(_ ms: Double) throws {
        let error = #expect(throws: DecodeMeasurementError.self) {
            try DecodeMeasurementPerPromptReading.fromMillis(
                prefillTokens: 512, decodeTokens: try PositiveCount(10), decodeTimeMs: ms)
        }
        guard case .invalidTime = error else {
            Issue.record("expected invalidTime, got \(String(describing: error))")
            return
        }
    }

    /// A stream's 100 ms interval over the 9 steps after its first token is 100/9*10 over
    /// the full count.
    @Test func aStreamBecomesTheDecodeTime() throws {
        let reading = try fromStream(10, 100, 200).get()
        #expect(reading.prefillTokens == 512)
        #expect(reading.decodeTokens.value == 10)
        #expect(abs(Double(reading.decodeTimeMs.value) - 100.0 / 9.0 * 10.0) < 1e-4)
    }

    /// One token has no step to rate; no TTFT or a non-positive interval is no reading.
    @Test(arguments: [
        (1, 100.0, 100.0, "too few"),
        (10, 0.0, 100.0, "ttft"),
        (10, Double.nan, 100.0, "ttft"),
        (10, 1.0, 1e300, "time"),
        (10, 200.0, 200.0, "interval"),
        (10, 300.0, 200.0, "interval"),
    ] as [(Int, Double, Double, String)])
    func aStreamWithoutARateIsRefused(_ decodeTokens: Int, _ ttftMs: Double,
                                      _ lastTokenMs: Double, _ kind: String) throws {
        let got = try fromStream(decodeTokens, ttftMs, lastTokenMs)
        switch (got, kind) {
        case (.failure(.tooFewTokens), "too few"), (.failure(.invalidTtft), "ttft"),
             (.failure(.invalidTime), "time"), (.failure(.invalidInterval), "interval"):
            break
        default:
            Issue.record("expected \(kind), got \(got)")
        }
    }

    /// The engine's time is taken as given, per prompt or in totals, on the batch and
    /// workload the entry asked for.
    @Test(arguments: [
        (1, "reading 512 10", "40"),
        (1, "reading 512 9", "engine returned decode_tokens 9, expected 10"),
        (1, "reading 256 10", "engine returned prefill_tokens 256, expected 512"),
        (1, "reading 0 10", "engine returned prefill_tokens 0, expected 512"),
        (3, "prompts 10 9 10", "engine returned decode_tokens 9, expected 10"),
        (1, "prompts", "engine answered 0 prompts, expected a batch of 1"),
        (1, "prompts 10 10", "engine answered 2 prompts, expected a batch of 1"),
        (4, "prompts 10 10 10", "engine answered 3 prompts, expected a batch of 4"),
        (4, "prompts 10 10 10 10 10", "engine answered 5 prompts, expected a batch of 4"),
        (4, "totals 2048 40", "40"),
        (1, "totals 512 10", "40"),
        (4, "totals 2048 39", "engine returned decode_tokens 39 over 4 prompts, expected 10 each"),
        (4, "totals 1024 40", "engine returned prefill_tokens 1024 over 4 prompts, expected 512 each"),
        (4, "totals 512 10", "engine returned prefill_tokens 512 over 4 prompts, expected 512 each"),
    ] as [(UInt32, String, String)])
    func aReadingBecomesTheDecodeTime(_ batchSize: UInt32, _ shape: String,
                                      _ expected: String) async throws {
        let words = shape.split(separator: " ")
        let counts = words.dropFirst().compactMap { Int($0) }
        let given = switch words[0] {
        case "reading": try reading(UInt32(counts[0]), counts[1], 40)
        case "prompts": try prompts(counts)
        default: try totals(counts[0], counts[1])
        }
        do {
            let resultData = try await unobserved(body(batchSize), reps: 5) { given }
            let (decodeTimeMs, _) = try meanAndStddev(resultData)
            #expect(Double(expected).map { abs(decodeTimeMs - $0) < 1e-9 } == true,
                    "\(decodeTimeMs), expected \(expected)")
        } catch let error as DecodeMeasurementError {
            #expect(error.errorDescription == expected)
        }
    }

    /// A benchmark that asks for no context (llama-bench `--n-depth 0`) takes a reading of
    /// none.
    @Test func aDecodeWithoutAPromptIsMeasured() async throws {
        let benchmark = DecodeThroughputBenchmark(
            benchmarkId: BenchmarkId(validated: "decode_throughput_0_32"),
            parameterPrefillTokens: 0, parameterDecodeTokens: 32,
            parameterBatchSize: BatchSize(validated: 1))
        let resultData = try await unobserved(benchmark, reps: 5) { try reading(0, 32, 40) }
        let (decodeTimeMs, _) = try meanAndStddev(resultData)
        #expect(abs(decodeTimeMs - 40) < 1e-9)
    }

    /// Readings of 1, 2, … ms reduce to their mean and sample stddev; one rep has no spread
    /// to report.
    @Test(arguments: [(5, 3.0, 2.5.squareRoot()), (1, 1.0, nil)] as [(Int, Double, Double?)])
    func theRepsReduceToMeanAndStddev(_ reps: Int, _ mean: Double, _ stddev: Double?)
        async throws {
        var ms = 0.0
        let resultData = try await unobserved(body(1), reps: reps) {
            ms += 1
            return try reading(512, 10, ms)
        }
        let (decodeTimeMs, decodeTimeMsStddev) = try meanAndStddev(resultData)
        #expect(abs(decodeTimeMs - mean) < 1e-9)
        switch (decodeTimeMsStddev, stddev) {
        case let (got?, want?): #expect(abs(got - want) < 1e-9)
        case (nil, nil): break
        default: Issue.record("stddev \(String(describing: decodeTimeMsStddev))")
        }
    }

    /// A `once` warm-up is one more attempt, and its reading is not in the result; an
    /// `inline` one adds none.
    @Test(arguments: [
        (MeasurementWarmup.once, [1000.0, 40, 40]),
        (.inline, [40.0, 40]),
    ] as [(MeasurementWarmup, [Double])])
    func onlyTheRepsAreReduced(_ warmup: MeasurementWarmup, _ times: [Double]) async throws {
        var times = times[...]
        let resultData = try await BenchmarkMeasurement.decode(
            body(1), reps: 2, readinessGate: {}, observer: noObserver(), warmup: warmup, ctx: ()
        ) { _ in
            try reading(512, 10, try next(&times))
        }
        #expect(times.isEmpty, "an attempt was not made")
        let (decodeTimeMs, _) = try meanAndStddev(resultData)
        #expect(abs(decodeTimeMs - 40) < 1e-9)
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
            try await BenchmarkMeasurement.decode(
                body(1), reps: 2, readinessGate: { try log.step("gate") },
                observer: log.observer, warmup: .once, ctx: ()
            ) { _ in
                let (event, refused) = warming ? ("warm-up", "warm-up check") : ("attempt", "check")
                warming = false
                try log.step(event)
                return try reading(512, failing == refused ? 9 : 10, 40)
            }
        }
        switch (got, failing) {
        case (.success, nil):
            break
        case let (.failure(error), "check"?), let (.failure(error), "warm-up check"?):
            #expect(error as? DecodeMeasurementError == .mismatch(
                field: "decode_tokens", actual: 9, expected: 10))
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
            try await BenchmarkMeasurement.decode(
                body(1), reps: 5, readinessGate: {}, observer: noObserver(), warmup: .once,
                ctx: recorded
            ) { recorded in
                recorded.attempts.append("attempt \(recorded.attempts.count)")
                if recorded.attempts.count == failsOn { throw ServerWentAway() }
                return try reading(512, 10, 40)
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
            return .perPrompt(try [0.0, 1, 2, 3].map { try prompt(512, 10, rep + $0) })
        }
        let (decodeTimeMs, decodeTimeMsStddev) = try meanAndStddev(resultData)
        #expect(abs(decodeTimeMs - 31.5) < 1e-9)
        let stddev = try #require(decodeTimeMsStddev, "five reps have a spread")
        #expect(abs(stddev - 250.0.squareRoot()) < 1e-9)
    }
}
