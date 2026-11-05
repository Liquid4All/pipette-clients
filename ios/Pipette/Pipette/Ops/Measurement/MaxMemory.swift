import Foundation

// Peak memory: the most memory one model load and one short generation held, read once per
// cell: the crate's `pipette-ops/src/measurement/max_memory.rs`
// (`docs/methodology/peak-memory-usage.md`).

/// A reading a peak-memory cell cannot use.
nonisolated enum MaxMemoryMeasurementError: Error, Equatable, LocalizedError {
    case zeroPeak(field: String)
    case tooLarge(field: String, bytes: UInt64)

    var errorDescription: String? {
        switch self {
        case let .zeroPeak(field): "\(field) is 0: a peak reads zero only when no read landed"
        case let .tooLarge(field, bytes):
            "\(field) is \(bytes), past the largest byte count a result holds"
        }
    }
}

/// What the attempt read, as the engine reports it.
nonisolated struct MaxMemoryMeasurementReading: Equatable, Sendable {
    let maxHostBytes: UInt64
    /// `nil` on Apple's unified memory, where the host figure already holds it.
    let maxGpuBytes: UInt64?
    /// Where the engine probes an NPU pool; none does yet.
    let maxNpuBytes: UInt64?
}

nonisolated extension BenchmarkMeasurement {
    /// Measure a peak-memory cell: one attempt, which loads the model and runs
    /// `parameter_prefill_tokens` and a short decode, and whose peaks are the result. With
    /// `.once` the same attempt runs first, checked and its reading dropped, so the peak is
    /// the model's and not a compiler's.
    ///
    /// One measured attempt because a second in the same process would inherit the first's
    /// allocator caches and high-water marks. No readiness gate either: a peak does not
    /// move with the device's temperature. The attempt owns its setup and its checks.
    ///
    /// `ctx` is lent to the attempt, for what the engine records while it runs.
    static func maxMemory<C>(
        _ benchmark: MaxMemoryUsageBenchmark,
        warmup: MemoryMeasurementWarmup,
        ctx: C,
        attempt: (inout C) async throws -> MaxMemoryMeasurementReading
    ) async throws -> BenchmarkResultData {
        let marker = benchmark.benchmarkId.value
        var ctx = ctx

        if warmup == .once {
            _ = try await measure("\(marker) warm-up", &ctx, attempt)
        }
        return try await measure(marker, &ctx, attempt)
    }
}

/// One attempt, its reading logged and then checked.
private nonisolated func measure<C>(
    _ label: String, _ ctx: inout C,
    _ attempt: (inout C) async throws -> MaxMemoryMeasurementReading
) async throws -> BenchmarkResultData {
    AppLog.jobRun.info("\(label)")
    let reading = try await attempt(&ctx)
    // Before the refusals, so a rejected reading is in the log too.
    AppLog.jobRun.info(
        "\(label): max_host_bytes=\(reading.maxHostBytes) "
            + "max_gpu_bytes=\(orNone(reading.maxGpuBytes)) "
            + "max_npu_bytes=\(orNone(reading.maxNpuBytes))")
    return try resultData(reading)
}

private nonisolated func resultData(_ reading: MaxMemoryMeasurementReading)
    throws(MaxMemoryMeasurementError) -> BenchmarkResultData {
    .maxMemoryUsage(maxHostBytes: try bytes("max_host_bytes", reading.maxHostBytes),
                    maxGpuBytes: try pool("max_gpu_bytes", reading.maxGpuBytes),
                    maxNpuBytes: try pool("max_npu_bytes", reading.maxNpuBytes))
}

private nonisolated func orNone(_ peak: UInt64?) -> String {
    peak.map { "\($0)" } ?? "none"
}

/// A peak on the wire, refused at zero and past what a `ByteCount` holds, the server's bound
/// on a byte count.
private nonisolated func bytes(_ field: String, _ peak: UInt64)
    throws(MaxMemoryMeasurementError) -> UInt64 {
    guard peak != 0 else { throw .zeroPeak(field: field) }
    guard Int64(exactly: peak).flatMap({ try? ByteCount($0) }) != nil else {
        throw .tooLarge(field: field, bytes: peak)
    }
    return peak
}

/// `bytes` for a pool the engine may not probe.
private nonisolated func pool(_ field: String, _ peak: UInt64?)
    throws(MaxMemoryMeasurementError) -> UInt64? {
    guard let peak else { return nil }
    return try bytes(field, peak)
}
