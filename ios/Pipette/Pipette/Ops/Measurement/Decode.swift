import Foundation

// Decode throughput: the time to generate the requested tokens after the prompt, reduced
// over the reps: the crate's `pipette-ops/src/measurement/decode.rs`.

/// A batch or a reading a decode cell cannot use.
nonisolated enum DecodeMeasurementError: Error, Equatable, LocalizedError {
    case promptCount(actual: Int, expected: BatchSize)
    case mismatch(field: String, actual: UInt32, expected: UInt32)
    case totalsMismatch(field: String, actual: PositiveCount, prompts: BatchSize, each: UInt32)
    case tooFewTokens(tokens: PositiveCount)
    case invalidTtft(ttftMs: Double)
    case invalidInterval(ttftMs: Double, lastTokenMs: Double)
    case invalidTime(ms: Double, source: PositiveMillisError)

    var errorDescription: String? {
        switch self {
        case let .promptCount(actual, expected):
            "engine answered \(actual) prompts, expected a batch of \(expected)"
        case let .mismatch(field, actual, expected):
            "engine returned \(field) \(actual), expected \(expected)"
        case let .totalsMismatch(field, actual, prompts, each):
            "engine returned \(field) \(actual) over \(prompts) prompts, expected \(each) each"
        case let .tooFewTokens(tokens):
            "a decode rate needs at least 2 tokens, got \(tokens)"
        case let .invalidTtft(ttftMs):
            "invalid ttft_ms: \(ttftMs)"
        case let .invalidInterval(ttftMs, lastTokenMs):
            "invalid decode interval: last token at \(lastTokenMs) ms, TTFT \(ttftMs) ms"
        case let .invalidTime(ms, _):
            "invalid decode_time_ms: \(ms)"
        }
    }
}

/// The shape every engine converts a batched reply to.
nonisolated enum DecodeMeasurementReading: Equatable, Sendable {
    case perPrompt([DecodeMeasurementPerPromptReading])
    case totals(DecodeMeasurementTotalsReading)
}

/// Each count is checked against the parameter of the same name.
nonisolated struct DecodeMeasurementPerPromptReading: Equatable, Sendable {
    /// Context tokens before the first decode step.
    let prefillTokens: UInt32
    /// Tokens generated, the first one included.
    let decodeTokens: PositiveCount
    /// Time to generate all `decodeTokens`.
    let decodeTimeMs: PositiveMillis

    /// A reading from a stream's client-clock stamps. The first token comes from the
    /// prefill, so the interval after it covers `decodeTokens - 1` steps; it is scaled to
    /// the full count, which is what llama-bench times.
    static func fromStream(prefillTokens: PositiveCount, decodeTokens: PositiveCount,
                           ttftMs: Double, lastTokenMs: Double)
        throws(DecodeMeasurementError) -> Self {
        let tokens = decodeTokens.value
        guard tokens >= 2 else { throw .tooFewTokens(tokens: decodeTokens) }
        guard ttftMs.isFinite, ttftMs > 0 else { throw .invalidTtft(ttftMs: ttftMs) }
        let intervalMs = lastTokenMs - ttftMs
        guard intervalMs > 0, intervalMs.isFinite else {
            throw .invalidInterval(ttftMs: ttftMs, lastTokenMs: lastTokenMs)
        }
        return try fromMillis(prefillTokens: UInt32(prefillTokens.value),
                              decodeTokens: decodeTokens,
                              decodeTimeMs: intervalMs / Double(tokens - 1) * Double(tokens))
    }

    /// A reading from a decode the engine timed itself, over all `decodeTokens`.
    static func fromMillis(prefillTokens: UInt32, decodeTokens: PositiveCount,
                           decodeTimeMs: Double) throws(DecodeMeasurementError) -> Self {
        Self(prefillTokens: prefillTokens, decodeTokens: decodeTokens,
             decodeTimeMs: try millis(decodeTimeMs))
    }
}

/// Counts summed over the streams and one time they all share, which any batch reply gives
/// exactly. The prompts are identical and generation is capped, so sums of
/// `parameter_batch_size` times each parameter mean each stream ran them.
nonisolated struct DecodeMeasurementTotalsReading: Equatable, Sendable {
    let prefillTokens: PositiveCount
    let decodeTokens: PositiveCount
    let decodeTimeMs: PositiveMillis
}

nonisolated extension BenchmarkMeasurement {
    /// Measure a decode cell: `reps` attempts, each one engine call, checked against the
    /// entry and reduced to a mean and stddev. Each rep waits on `readinessGate` first.
    ///
    /// `ctx` is lent to every attempt, the warm-up's included, for what the engine records
    /// while they run. An attempt runs the `parameter_batch_size` prompts together; a rep
    /// reads their mean time, or the totals' shared one.
    static func decode<C>(
        _ benchmark: DecodeThroughputBenchmark,
        reps: Int,
        readinessGate: ReadinessGate,
        observer: MeasurementObserver,
        warmup: MeasurementWarmup,
        ctx: C,
        attempt: (inout C) async throws -> DecodeMeasurementReading
    ) async throws -> BenchmarkResultData {
        precondition(reps > 0, "a cell measures at least one rep")
        let marker = benchmark.benchmarkId.value
        var ctx = ctx

        if warmup == .once {
            try await warmUp("\(marker) warm-up", &ctx, attempt, benchmark)
        }
        var times: [Double] = []
        for rep in 1 ... reps {
            times.append(try await runRep("\(marker) rep \(rep)/\(reps)", readinessGate,
                                          observer, &ctx, attempt, benchmark))
        }
        let stats = meanStddev(times)
        AppLog.jobRun.info("\(reductionLine(label: marker, samples: times, stats: stats))")

        return .decodeThroughput(decodeTimeMs: stats.meanMs,
                                 // One rep has no spread to report.
                                 decodeTimeMsStddev: reps > 1 ? stats.stddevMs : nil)
    }
}

/// Ungated and unobserved: the warm-up is not a rep.
private nonisolated func warmUp<C>(
    _ label: String, _ ctx: inout C,
    _ attempt: (inout C) async throws -> DecodeMeasurementReading,
    _ benchmark: DecodeThroughputBenchmark
) async throws {
    AppLog.jobRun.info("\(label)")
    let reading = try await attempt(&ctx)
    _ = try sample(label, reading, benchmark)
}

private nonisolated func runRep<C>(
    _ label: String, _ readinessGate: ReadinessGate, _ observer: MeasurementObserver,
    _ ctx: inout C, _ attempt: (inout C) async throws -> DecodeMeasurementReading,
    _ benchmark: DecodeThroughputBenchmark
) async throws -> Double {
    try readinessGate()

    try observer.onBeforeRepStart()

    AppLog.jobRun.info("\(label)")
    let reading = try await attempt(&ctx)

    try observer.onAfterRepEnd()

    return try sample(label, reading, benchmark)
}

/// Each shape is logged before its checks, so a rejected reading is in the log too.
private nonisolated func sample(_ label: String, _ reading: DecodeMeasurementReading,
                                _ benchmark: DecodeThroughputBenchmark) throws -> Double {
    switch reading {
    case let .perPrompt(readings): try perPrompt(label, readings, benchmark)
    case let .totals(reading): try totals(label, reading, benchmark)
    }
}

private nonisolated func perPrompt(_ label: String,
                                   _ readings: [DecodeMeasurementPerPromptReading],
                                   _ benchmark: DecodeThroughputBenchmark) throws -> Double {
    let count = readings.count
    let times = readings.enumerated().map { index, reading in
        let ms = Double(reading.decodeTimeMs.value)
        AppLog.jobRun.info(
            "\(promptLabel(label, index, count)): decode \(BenchmarkMeasurement.format(ms)) ms "
                + "(prefill_tokens=\(reading.prefillTokens), "
                + "decode_tokens=\(reading.decodeTokens))")
        return ms
    }
    try answered(count, benchmark)
    for reading in readings { try check(reading, benchmark) }

    let ms = try BenchmarkMeasurement.perRequestMean(times)
    if count > 1 {
        AppLog.jobRun.info(
            "\(label): mean \(BenchmarkMeasurement.format(ms)) ms over \(count) prompts")
    }
    return ms
}

private nonisolated func totals(_ label: String, _ reading: DecodeMeasurementTotalsReading,
                                _ benchmark: DecodeThroughputBenchmark) throws -> Double {
    let ms = Double(reading.decodeTimeMs.value)
    let batchSize = benchmark.parameterBatchSize
    AppLog.jobRun.info(
        "\(label): decode \(BenchmarkMeasurement.format(ms)) ms "
            + "(prefill_tokens=\(reading.prefillTokens), decode_tokens=\(reading.decodeTokens) "
            + "over \(batchSize) prompts)")
    try expectTotals("prefill_tokens", reading.prefillTokens, batchSize,
                     benchmark.parameterPrefillTokens.value.magnitude)
    try expectTotals("decode_tokens", reading.decodeTokens, batchSize,
                     benchmark.parameterDecodeTokens.value.magnitude)
    return ms
}

private nonisolated func answered(_ prompts: Int, _ benchmark: DecodeThroughputBenchmark)
    throws(DecodeMeasurementError) {
    let expected = benchmark.parameterBatchSize
    guard prompts == Int(expected.asU32) else {
        throw .promptCount(actual: prompts, expected: expected)
    }
}

/// One answer logs under the rep's own label.
private nonisolated func promptLabel(_ label: String, _ index: Int, _ count: Int) -> String {
    count == 1 ? label : "\(label) prompt \(index + 1)/\(count)"
}

/// The workload the benchmark asked for; the time is valid by its type.
private nonisolated func check(_ reading: DecodeMeasurementPerPromptReading,
                               _ benchmark: DecodeThroughputBenchmark)
    throws(DecodeMeasurementError) {
    try expect("prefill_tokens", reading.prefillTokens, benchmark.parameterPrefillTokens.value.magnitude)
    try expect("decode_tokens", UInt32(reading.decodeTokens.value),
               benchmark.parameterDecodeTokens.value.magnitude)
}

private nonisolated func expect(_ field: String, _ actual: UInt32, _ expected: UInt32)
    throws(DecodeMeasurementError) {
    guard actual == expected else {
        throw .mismatch(field: field, actual: actual, expected: expected)
    }
}

/// In `Int128`, so the product cannot overflow.
private nonisolated func expectTotals(_ field: String, _ actual: PositiveCount,
                                      _ prompts: BatchSize, _ each: UInt32)
    throws(DecodeMeasurementError) {
    guard Int128(actual.value) == Int128(prompts.asU32) * Int128(each) else {
        throw .totalsMismatch(field: field, actual: actual, prompts: prompts, each: each)
    }
}

/// Narrowed to the `Float` a reading carries its time in.
private nonisolated func millis(_ ms: Double) throws(DecodeMeasurementError) -> PositiveMillis {
    do {
        return try PositiveMillis(Float(ms))
    } catch {
        throw .invalidTime(ms: ms, source: error)
    }
}
