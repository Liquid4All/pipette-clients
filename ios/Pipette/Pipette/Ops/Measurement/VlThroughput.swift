import Foundation

// VL throughput: the times to prefill an image-plus-text prompt, to decode after it and to
// answer the whole request, reduced over the reps: the crate's
// `pipette-ops/src/measurement/vl_throughput.rs` (`docs/methodology/vl-throughput.md`).

/// A reading a VL throughput cell cannot use.
nonisolated enum VlThroughputMeasurementError: Error, Equatable, LocalizedError {
    case mismatch(field: String, actual: PositiveCount, expected: UInt32)
    case promptVaried(first: PositiveCount, actual: PositiveCount)
    case imageTokensVaried(first: PositiveCount?, actual: PositiveCount?)
    case imageTokensExceedPrompt(imageTokens: PositiveCount, promptTokens: PositiveCount)
    case invalidTime(field: String, ms: Double, source: PositiveMillisError)

    var errorDescription: String? {
        switch self {
        case let .mismatch(field, actual, expected):
            "engine returned \(field) \(actual), expected \(expected)"
        case let .promptVaried(first, actual):
            "engine returned prompt_tokens \(actual), \(first) on the first attempt: prompt "
                + "caching must be off"
        case let .imageTokensVaried(first, actual):
            "engine returned image_tokens \(orNone(actual)), \(orNone(first)) on the first "
                + "attempt"
        case let .imageTokensExceedPrompt(imageTokens, promptTokens):
            "engine returned image_tokens \(imageTokens) of prompt_tokens \(promptTokens)"
        case let .invalidTime(field, ms, _):
            "invalid \(field): \(ms)"
        }
    }
}

/// What one attempt ran, as the engine reports it.
nonisolated struct VlThroughputMeasurementReading: Equatable, Sendable {
    /// The whole prompt: image, text and template tokens. The same on every rep.
    let promptTokens: PositiveCount
    /// The image-only part of `promptTokens`, where the engine can tell
    /// (`docs/methodology/vl-image-tokens.md`). The same on every rep.
    let imageTokens: PositiveCount?
    /// Tokens generated, checked against `parameter_decode_tokens`.
    let decodeTokens: PositiveCount
    /// Time to process all `promptTokens`, the image encode included.
    let promptMs: PositiveMillis
    /// Time to generate all `decodeTokens`.
    let predictedMs: PositiveMillis
    /// The whole request on the client's clock, measured rather than `promptMs + predictedMs`
    /// (`docs/methodology/vl-throughput.md#end-to-end-latency`).
    let totalTimeMs: PositiveMillis

    /// From a decode time already covering all `decodeTokens` steps.
    static func fromMillis(promptTokens: PositiveCount, imageTokens: PositiveCount?,
                           decodeTokens: PositiveCount, promptMs: Double, predictedMs: Double,
                           totalTimeMs: Double)
        throws(VlThroughputMeasurementError) -> Self {
        Self(promptTokens: promptTokens, imageTokens: imageTokens, decodeTokens: decodeTokens,
             promptMs: try millis("prompt_ms", promptMs),
             predictedMs: try millis("predicted_ms", predictedMs),
             totalTimeMs: try millis("total_time_ms", totalTimeMs))
    }
}

nonisolated extension BenchmarkMeasurement {
    /// Measure a VL throughput cell: `reps` attempts, each one engine call, checked against
    /// the entry and the first attempt's prompt, and each time reduced to a mean and
    /// stddev. Each rep waits on `readinessGate` first.
    ///
    /// `ctx` is lent to every attempt, the warm-up's included, for what the engine records
    /// while they run.
    static func vlThroughput<C>(
        _ benchmark: VlThroughputBenchmark,
        reps: Int,
        readinessGate: ReadinessGate,
        observer: MeasurementObserver,
        warmup: MeasurementWarmup,
        ctx: C,
        attempt: (inout C) async throws -> VlThroughputMeasurementReading
    ) async throws -> BenchmarkResultData {
        let (first, prompt, predicted, total) = try await throughput(
            marker: benchmark.benchmarkId.value,
            decodeTokens: benchmark.parameterDecodeTokens.value.magnitude, reps: reps,
            readinessGate: readinessGate, observer: observer, warmup: warmup, ctx: ctx,
            attempt: attempt)
        // One rep has no spread to report.
        func stddev(_ stats: Stats) -> Double? { reps > 1 ? stats.stddevMs : nil }

        return .vlThroughput(promptTokens: count(first.promptTokens),
                             imageTokens: first.imageTokens.map(count),
                             promptMs: prompt.meanMs,
                             promptMsStddev: stddev(prompt),
                             predictedMs: predicted.meanMs,
                             predictedMsStddev: stddev(predicted),
                             totalTimeMs: total.meanMs,
                             totalTimeMsStddev: stddev(total))
    }

    /// The reps both media cells run: the crate's `vl_throughput::throughput`.
    static func throughput<C>(
        marker: String,
        decodeTokens: UInt32,
        reps: Int,
        readinessGate: ReadinessGate,
        observer: MeasurementObserver,
        warmup: MeasurementWarmup,
        ctx: C,
        attempt: (inout C) async throws -> VlThroughputMeasurementReading
    ) async throws -> (first: VlThroughputMeasurementReading, prompt: Stats, predicted: Stats,
                       total: Stats) {
        precondition(reps > 0, "a cell measures at least one rep")
        var ctx = ctx

        // The prompt every rep must repeat is the first the engine saw: the warm-up's where
        // there is one, so a cache it primed shows from rep 1.
        let warm: VlThroughputMeasurementReading? = switch warmup {
        case .once: try await warmUp("\(marker) warm-up", &ctx, attempt, decodeTokens)
        case .inline: nil
        }
        let first = try await runRep("\(marker) rep 1/\(reps)", readinessGate, observer, &ctx,
                                     attempt, decodeTokens, warm)
        let reference = warm ?? first
        var readings = [first]
        for rep in stride(from: 2, through: reps, by: 1) {
            readings.append(try await runRep("\(marker) rep \(rep)/\(reps)", readinessGate,
                                             observer, &ctx, attempt, decodeTokens, reference))
        }
        return (first,
                reduce(marker, "prompt_ms", readings.map(\.promptMs)),
                reduce(marker, "predicted_ms", readings.map(\.predictedMs)),
                reduce(marker, "total_time_ms", readings.map(\.totalTimeMs)))
    }
}

/// Ungated and unobserved: the warm-up is not a rep.
private nonisolated func warmUp<C>(
    _ label: String, _ ctx: inout C,
    _ attempt: (inout C) async throws -> VlThroughputMeasurementReading,
    _ decodeTokens: UInt32
) async throws -> VlThroughputMeasurementReading {
    AppLog.jobRun.info("\(label)")
    let reading = try await attempt(&ctx)
    try sample(label, reading, decodeTokens, nil)
    return reading
}

private nonisolated func runRep<C>(
    _ label: String, _ readinessGate: ReadinessGate, _ observer: MeasurementObserver,
    _ ctx: inout C, _ attempt: (inout C) async throws -> VlThroughputMeasurementReading,
    _ decodeTokens: UInt32, _ first: VlThroughputMeasurementReading?
) async throws -> VlThroughputMeasurementReading {
    try readinessGate()

    try observer.onBeforeRepStart()

    AppLog.jobRun.info("\(label)")
    let reading = try await attempt(&ctx)

    try observer.onAfterRepEnd()

    try sample(label, reading, decodeTokens, first)
    return reading
}

private nonisolated func sample(_ label: String, _ reading: VlThroughputMeasurementReading,
                                _ decodeTokens: UInt32,
                                _ first: VlThroughputMeasurementReading?) throws {
    // Before the check, so a rejected reading is in the log too.
    AppLog.jobRun.info(
        "\(label): prompt \(BenchmarkMeasurement.format(Double(reading.promptMs.value))) ms, "
            + "predicted \(BenchmarkMeasurement.format(Double(reading.predictedMs.value))) ms, "
            + "total \(BenchmarkMeasurement.format(Double(reading.totalTimeMs.value))) ms "
            + "(prompt_tokens=\(reading.promptTokens), "
            + (reading.imageTokens.map { "image_tokens=\($0), " } ?? "")
            + "decode_tokens=\(reading.decodeTokens))")
    try check(reading, decodeTokens, first)
}

/// The workload the benchmark asked for, on the prompt the first attempt ran; the times are
/// valid by their type.
private nonisolated func check(_ reading: VlThroughputMeasurementReading,
                               _ decodeTokens: UInt32,
                               _ first: VlThroughputMeasurementReading?)
    throws(VlThroughputMeasurementError) {
    try expect("decode_tokens", reading.decodeTokens, decodeTokens)
    // Before the image check: a prompt cut short by a cache can fall below the image
    // count, and the cache is the cause to name.
    if let first {
        guard reading.promptTokens == first.promptTokens else {
            throw .promptVaried(first: first.promptTokens, actual: reading.promptTokens)
        }
        guard reading.imageTokens == first.imageTokens else {
            throw .imageTokensVaried(first: first.imageTokens, actual: reading.imageTokens)
        }
    }
    if let imageTokens = reading.imageTokens, imageTokens > reading.promptTokens {
        throw .imageTokensExceedPrompt(imageTokens: imageTokens,
                                       promptTokens: reading.promptTokens)
    }
}

/// Widened to `Int64` so a signed count and an unsigned parameter compare.
private nonisolated func expect(_ field: String, _ actual: PositiveCount, _ expected: UInt32)
    throws(VlThroughputMeasurementError) {
    guard Int64(actual.value) == Int64(expected) else {
        throw .mismatch(field: field, actual: actual, expected: expected)
    }
}

private nonisolated func reduce(_ marker: String, _ metric: String,
                                _ times: [PositiveMillis]) -> BenchmarkMeasurement.Stats {
    let samples = times.map { Double($0.value) }
    let stats = BenchmarkMeasurement.meanStddev(samples)
    AppLog.jobRun.info(
        "\(BenchmarkMeasurement.reductionLine(label: "\(marker) \(metric)", samples: samples, stats: stats))")
    return stats
}

/// A count on the wire. `PositiveCount` is never negative, so this is exact.
private nonisolated func count(_ tokens: PositiveCount) -> UInt32 {
    UInt32(tokens.value)
}

private nonisolated func orNone(_ tokens: PositiveCount?) -> String {
    tokens.map { "\($0)" } ?? "none"
}

/// Narrowed to the `Float` a reading carries its time in.
private nonisolated func millis(_ field: String, _ ms: Double)
    throws(VlThroughputMeasurementError) -> PositiveMillis {
    do {
        return try PositiveMillis(Float(ms))
    } catch {
        throw .invalidTime(field: field, ms: ms, source: error)
    }
}
