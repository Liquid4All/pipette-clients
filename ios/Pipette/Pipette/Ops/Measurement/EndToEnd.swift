import Foundation

// End-to-end latency: the time a caller waits for one whole completion, reduced over the
// reps: the crate's `pipette-ops/src/measurement/end_to_end.rs`.

/// A batch or a reading an end-to-end cell cannot use.
nonisolated enum EndToEndMeasurementError: Error, Equatable, LocalizedError {
    case promptCount(actual: Int, expected: BatchSize)
    case mismatch(field: String, actual: PositiveCount, expected: UInt32)
    case totalsMismatch(field: String, actual: PositiveCount, prompts: BatchSize, each: UInt32)
    case invalidTime(ms: Double, source: PositiveMillisError)

    var errorDescription: String? {
        switch self {
        case let .promptCount(actual, expected):
            "engine answered \(actual) prompts, expected a batch of \(expected)"
        case let .mismatch(field, actual, expected):
            "engine returned \(field) \(actual), expected \(expected)"
        case let .totalsMismatch(field, actual, prompts, each):
            "engine returned \(field) \(actual) over \(prompts) prompts, expected \(each) each"
        case let .invalidTime(ms, _):
            "invalid total_time_ms: \(ms)"
        }
    }
}

/// The shape every engine converts a batched reply to.
nonisolated enum EndToEndMeasurementReading: Equatable, Sendable {
    case perPrompt([EndToEndMeasurementPerPromptReading])
    case totals(EndToEndMeasurementTotalsReading)
}

/// Each count is checked against the parameter of the same name.
nonisolated struct EndToEndMeasurementPerPromptReading: Equatable, Sendable {
    /// Prompt tokens, as the engine's tokenizer counted the text.
    let prefillTokens: PositiveCount
    let decodeTokens: PositiveCount
    /// The request's send to this prompt's whole answer.
    let totalTimeMs: PositiveMillis

    /// On the client's clock (`BenchmarkMeasurement.timed`). A non-streamed request answers
    /// every prompt as it returns, so each takes the request's time.
    static func fromRequest(prefillTokens: PositiveCount, decodeTokens: PositiveCount,
                            elapsedMs: Double) throws(EndToEndMeasurementError) -> Self {
        Self(prefillTokens: prefillTokens, decodeTokens: decodeTokens,
             totalTimeMs: try millis(elapsedMs))
    }
}

/// What an in-process cell's timed request ran, before `fromRequest` checks it.
nonisolated struct EndToEndCounts: Sendable {
    let prefillTokens: Int
    let decodeTokens: Int
}

/// Counts summed over the prompts and the request's time, which answered them all. The
/// prompts are identical and generation is capped, so sums of `parameter_batch_size` times
/// each parameter mean each prompt ran them.
nonisolated struct EndToEndMeasurementTotalsReading: Equatable, Sendable {
    let prefillTokens: PositiveCount
    let decodeTokens: PositiveCount
    let totalTimeMs: PositiveMillis
}

nonisolated extension BenchmarkMeasurement {
    /// Measure an end-to-end cell: `reps` attempts, each one engine call, checked against
    /// the entry and reduced to a mean and stddev. Each rep waits on `readinessGate` first.
    ///
    /// `ctx` is lent to every attempt, the warm-up's included, for what the engine records
    /// while they run. An attempt runs the `parameter_batch_size` prompts together; a rep
    /// reads their mean time, or the totals' shared one.
    static func endToEnd<C>(
        _ benchmark: EndToEndLatencyBenchmark,
        reps: Int,
        readinessGate: ReadinessGate,
        observer: MeasurementObserver,
        warmup: MeasurementWarmup,
        ctx: C,
        attempt: (inout C) async throws -> EndToEndMeasurementReading
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

        return .endToEndLatency(totalTimeMs: stats.meanMs,
                                // One rep has no spread to report.
                                totalTimeMsStddev: reps > 1 ? stats.stddevMs : nil)
    }
}

/// Ungated and unobserved: the warm-up is not a rep.
private nonisolated func warmUp<C>(
    _ label: String, _ ctx: inout C,
    _ attempt: (inout C) async throws -> EndToEndMeasurementReading,
    _ benchmark: EndToEndLatencyBenchmark
) async throws {
    AppLog.jobRun.info("\(label)")
    let reading = try await attempt(&ctx)
    _ = try sample(label, reading, benchmark)
}

private nonisolated func runRep<C>(
    _ label: String, _ readinessGate: ReadinessGate, _ observer: MeasurementObserver,
    _ ctx: inout C, _ attempt: (inout C) async throws -> EndToEndMeasurementReading,
    _ benchmark: EndToEndLatencyBenchmark
) async throws -> Double {
    try readinessGate()

    try observer.onBeforeRepStart()

    AppLog.jobRun.info("\(label)")
    let reading = try await attempt(&ctx)

    try observer.onAfterRepEnd()

    return try sample(label, reading, benchmark)
}

/// Each shape is logged before its checks, so a rejected reading is in the log too.
private nonisolated func sample(_ label: String, _ reading: EndToEndMeasurementReading,
                                _ benchmark: EndToEndLatencyBenchmark) throws -> Double {
    switch reading {
    case let .perPrompt(readings): try perPrompt(label, readings, benchmark)
    case let .totals(reading): try totals(label, reading, benchmark)
    }
}

private nonisolated func perPrompt(_ label: String,
                                   _ readings: [EndToEndMeasurementPerPromptReading],
                                   _ benchmark: EndToEndLatencyBenchmark) throws -> Double {
    let count = readings.count
    let times = readings.enumerated().map { index, reading in
        let ms = Double(reading.totalTimeMs.value)
        AppLog.jobRun.info(
            "\(promptLabel(label, index, count)): total \(BenchmarkMeasurement.format(ms)) ms "
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

private nonisolated func totals(_ label: String, _ reading: EndToEndMeasurementTotalsReading,
                                _ benchmark: EndToEndLatencyBenchmark) throws -> Double {
    let ms = Double(reading.totalTimeMs.value)
    let batchSize = benchmark.parameterBatchSize
    AppLog.jobRun.info(
        "\(label): total \(BenchmarkMeasurement.format(ms)) ms "
            + "(prefill_tokens=\(reading.prefillTokens), decode_tokens=\(reading.decodeTokens) "
            + "over \(batchSize) prompts)")
    try expectTotals("prefill_tokens", reading.prefillTokens, batchSize,
                     benchmark.parameterPrefillTokens.value.magnitude)
    try expectTotals("decode_tokens", reading.decodeTokens, batchSize,
                     benchmark.parameterDecodeTokens.value.magnitude)
    return ms
}

private nonisolated func answered(_ prompts: Int, _ benchmark: EndToEndLatencyBenchmark)
    throws(EndToEndMeasurementError) {
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
private nonisolated func check(_ reading: EndToEndMeasurementPerPromptReading,
                               _ benchmark: EndToEndLatencyBenchmark)
    throws(EndToEndMeasurementError) {
    try expect("prefill_tokens", reading.prefillTokens, benchmark.parameterPrefillTokens.value.magnitude)
    try expect("decode_tokens", reading.decodeTokens, benchmark.parameterDecodeTokens.value.magnitude)
}

/// Widened to `Int64` so a signed count and an unsigned parameter compare.
private nonisolated func expect(_ field: String, _ actual: PositiveCount, _ expected: UInt32)
    throws(EndToEndMeasurementError) {
    guard Int64(actual.value) == Int64(expected) else {
        throw .mismatch(field: field, actual: actual, expected: expected)
    }
}

/// In `Int128`, so the product cannot overflow.
private nonisolated func expectTotals(_ field: String, _ actual: PositiveCount,
                                      _ prompts: BatchSize, _ each: UInt32)
    throws(EndToEndMeasurementError) {
    guard Int128(actual.value) == Int128(prompts.asU32) * Int128(each) else {
        throw .totalsMismatch(field: field, actual: actual, prompts: prompts, each: each)
    }
}

/// Narrowed to the `Float` a reading carries its time in.
private nonisolated func millis(_ ms: Double) throws(EndToEndMeasurementError) -> PositiveMillis {
    do {
        return try PositiveMillis(Float(ms))
    } catch {
        throw .invalidTime(ms: ms, source: error)
    }
}
