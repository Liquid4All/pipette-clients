import Foundation

// Prefill throughput: the time to process the prompt, reduced over the reps: the crate's
// `pipette-ops/src/measurement/prefill.rs`.

/// A batch or a reading a prefill cell cannot use.
nonisolated enum PrefillMeasurementError: Error, Equatable, LocalizedError {
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
            "invalid prefill_time_ms: \(ms)"
        }
    }
}

/// The shape every engine converts a batched reply to.
nonisolated enum PrefillMeasurementReading: Equatable, Sendable {
    case perPrompt([PrefillMeasurementPerPromptReading])
    case totals(PrefillMeasurementTotalsReading)
}

nonisolated struct PrefillMeasurementPerPromptReading: Equatable, Sendable {
    /// Prompt tokens, checked against `parameter_prefill_tokens`.
    let prefillTokens: PositiveCount
    /// Time to process all `prefillTokens`.
    let prefillTimeMs: PositiveMillis

    /// A reading from a stream's time to its first token. The cell asks for one token; more
    /// means the engine ignored the request's shape.
    static func fromStream(prefillTokens: PositiveCount, completionTokens: PositiveCount,
                           ttftMs: Double) throws(PrefillMeasurementError) -> Self {
        try expect("completion_tokens", completionTokens, 1)
        return try fromMillis(prefillTokens: prefillTokens, prefillTimeMs: ttftMs)
    }

    /// A reading from a prefill the engine timed itself, which generates no token to count.
    static func fromMillis(prefillTokens: PositiveCount,
                           prefillTimeMs: Double) throws(PrefillMeasurementError) -> Self {
        Self(prefillTokens: prefillTokens, prefillTimeMs: try millis(prefillTimeMs))
    }
}

/// Counts summed over the prompts and one time they all share, which any batch reply gives
/// exactly. The prompts are identical, so a sum of `parameter_batch_size` times
/// `parameter_prefill_tokens` means each prompt ran it.
nonisolated struct PrefillMeasurementTotalsReading: Equatable, Sendable {
    let prefillTokens: PositiveCount
    let prefillTimeMs: PositiveMillis
}

nonisolated extension BenchmarkMeasurement {
    /// Measure a prefill cell: `reps` attempts, each one engine call, checked against the
    /// entry and reduced to a mean and stddev. Each rep waits on `readinessGate` first.
    ///
    /// `ctx` is lent to every attempt, the warm-up's included, for what the engine records
    /// while they run. An attempt runs the `parameter_batch_size` prompts together; a rep
    /// reads their mean time, or the totals' shared one.
    static func prefill<C>(
        _ benchmark: PrefillThroughputBenchmark,
        reps: Int,
        readinessGate: ReadinessGate,
        observer: MeasurementObserver,
        warmup: MeasurementWarmup,
        ctx: C,
        attempt: (inout C) async throws -> PrefillMeasurementReading
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

        return .prefillThroughput(prefillTimeMs: stats.meanMs,
                                  // One rep has no spread to report.
                                  prefillTimeMsStddev: reps > 1 ? stats.stddevMs : nil)
    }
}

/// Ungated and unobserved: the warm-up is not a rep.
private nonisolated func warmUp<C>(
    _ label: String, _ ctx: inout C,
    _ attempt: (inout C) async throws -> PrefillMeasurementReading,
    _ benchmark: PrefillThroughputBenchmark
) async throws {
    AppLog.jobRun.info("\(label)")
    let reading = try await attempt(&ctx)
    _ = try sample(label, reading, benchmark)
}

private nonisolated func runRep<C>(
    _ label: String, _ readinessGate: ReadinessGate, _ observer: MeasurementObserver,
    _ ctx: inout C, _ attempt: (inout C) async throws -> PrefillMeasurementReading,
    _ benchmark: PrefillThroughputBenchmark
) async throws -> Double {
    try readinessGate()

    try observer.onBeforeRepStart()

    AppLog.jobRun.info("\(label)")
    let reading = try await attempt(&ctx)

    try observer.onAfterRepEnd()

    return try sample(label, reading, benchmark)
}

/// Each shape is logged before its checks, so a rejected reading is in the log too.
private nonisolated func sample(_ label: String, _ reading: PrefillMeasurementReading,
                                _ benchmark: PrefillThroughputBenchmark) throws -> Double {
    switch reading {
    case let .perPrompt(readings): try perPrompt(label, readings, benchmark)
    case let .totals(reading): try totals(label, reading, benchmark)
    }
}

private nonisolated func perPrompt(_ label: String,
                                   _ readings: [PrefillMeasurementPerPromptReading],
                                   _ benchmark: PrefillThroughputBenchmark) throws -> Double {
    let count = readings.count
    let times = readings.enumerated().map { index, reading in
        let ms = Double(reading.prefillTimeMs.value)
        AppLog.jobRun.info(
            "\(promptLabel(label, index, count)): prefill \(BenchmarkMeasurement.format(ms)) ms "
                + "(prefill_tokens=\(reading.prefillTokens))")
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

private nonisolated func totals(_ label: String, _ reading: PrefillMeasurementTotalsReading,
                                _ benchmark: PrefillThroughputBenchmark) throws -> Double {
    let ms = Double(reading.prefillTimeMs.value)
    let batchSize = benchmark.parameterBatchSize
    AppLog.jobRun.info(
        "\(label): prefill \(BenchmarkMeasurement.format(ms)) ms "
            + "(prefill_tokens=\(reading.prefillTokens) over \(batchSize) prompts)")
    try expectTotals("prefill_tokens", reading.prefillTokens, batchSize,
                     benchmark.parameterPrefillTokens.value.magnitude)
    return ms
}

private nonisolated func answered(_ prompts: Int, _ benchmark: PrefillThroughputBenchmark)
    throws(PrefillMeasurementError) {
    let expected = benchmark.parameterBatchSize
    guard prompts == Int(expected.asU32) else {
        throw .promptCount(actual: prompts, expected: expected)
    }
}

/// One answer logs under the rep's own label.
private nonisolated func promptLabel(_ label: String, _ index: Int, _ count: Int) -> String {
    count == 1 ? label : "\(label) prompt \(index + 1)/\(count)"
}

/// The prompt the benchmark asked for; the time is valid by its type.
private nonisolated func check(_ reading: PrefillMeasurementPerPromptReading,
                               _ benchmark: PrefillThroughputBenchmark)
    throws(PrefillMeasurementError) {
    try expect("prefill_tokens", reading.prefillTokens, benchmark.parameterPrefillTokens.value.magnitude)
}

/// Widened to `Int64` so a signed count and an unsigned parameter compare.
private nonisolated func expect(_ field: String, _ actual: PositiveCount, _ expected: UInt32)
    throws(PrefillMeasurementError) {
    guard Int64(actual.value) == Int64(expected) else {
        throw .mismatch(field: field, actual: actual, expected: expected)
    }
}

/// In `Int128`, so the product cannot overflow.
private nonisolated func expectTotals(_ field: String, _ actual: PositiveCount,
                                      _ prompts: BatchSize, _ each: UInt32)
    throws(PrefillMeasurementError) {
    guard Int128(actual.value) == Int128(prompts.asU32) * Int128(each) else {
        throw .totalsMismatch(field: field, actual: actual, prompts: prompts, each: each)
    }
}

/// Narrowed to the `Float` a reading carries its time in.
private nonisolated func millis(_ ms: Double) throws(PrefillMeasurementError) -> PositiveMillis {
    do {
        return try PositiveMillis(Float(ms))
    } catch {
        throw .invalidTime(ms: ms, source: error)
    }
}
