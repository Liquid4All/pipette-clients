import Foundation

// VL peak memory: the most memory one model and projector load and one image workload held,
// read once per cell: the crate's `pipette-ops/src/measurement/vl_max_memory.rs`
// (`docs/methodology/vl-max-memory.md`).

/// A reading a VL peak-memory cell cannot use.
nonisolated enum VlMaxMemoryMeasurementError: Error, Equatable, LocalizedError {
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
nonisolated struct VlMaxMemoryMeasurementReading: Equatable, Sendable {
    let maxHostBytes: UInt64
    /// `nil` on Apple's unified memory, where the host figure already holds it.
    let maxGpuBytes: UInt64?
    /// Where the engine probes an NPU pool; none does yet.
    let maxNpuBytes: UInt64?
}

nonisolated extension BenchmarkMeasurement {
    /// Measure a VL peak-memory cell: one attempt, which loads the model and its projector,
    /// runs `parameter_num_images` images and `parameter_text_tokens` text, and decodes
    /// `parameter_decode_tokens` (one when unset), and whose peaks are the result.
    ///
    /// One measured attempt, a warm-up before it only with `.once`, and no readiness gate,
    /// as in `maxMemory`. The attempt owns its checks.
    ///
    /// `ctx` is lent to the attempt, for what the engine records while it runs.
    static func vlMaxMemory<C>(
        _ benchmark: VlMaxMemoryBenchmark,
        warmup: MemoryMeasurementWarmup,
        ctx: C,
        attempt: (inout C) async throws -> VlMaxMemoryMeasurementReading
    ) async throws -> BenchmarkResultData {
        try await mediaMaxMemory(marker: benchmark.benchmarkId.value, warmup: warmup, ctx: ctx,
                                 attempt: attempt)
    }

    /// The attempts both media cells run, the cell named by `marker`.
    static func mediaMaxMemory<C>(
        marker: String,
        warmup: MemoryMeasurementWarmup,
        ctx: C,
        attempt: (inout C) async throws -> VlMaxMemoryMeasurementReading
    ) async throws -> BenchmarkResultData {
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
    _ attempt: (inout C) async throws -> VlMaxMemoryMeasurementReading
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

private nonisolated func resultData(_ reading: VlMaxMemoryMeasurementReading)
    throws(VlMaxMemoryMeasurementError) -> BenchmarkResultData {
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
    throws(VlMaxMemoryMeasurementError) -> UInt64 {
    guard peak != 0 else { throw .zeroPeak(field: field) }
    guard Int64(exactly: peak).flatMap({ try? ByteCount($0) }) != nil else {
        throw .tooLarge(field: field, bytes: peak)
    }
    return peak
}

/// `bytes` for a pool the engine may not probe.
private nonisolated func pool(_ field: String, _ peak: UInt64?)
    throws(VlMaxMemoryMeasurementError) -> UInt64? {
    guard let peak else { return nil }
    return try bytes(field, peak)
}
