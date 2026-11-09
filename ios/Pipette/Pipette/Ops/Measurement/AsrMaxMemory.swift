import Foundation

// ASR peak memory: the crate's `measurement::max_memory` (`docs/methodology/asr-max-memory.md`),
// with `vl_max_memory`'s checks.

nonisolated extension BenchmarkMeasurement {
    static func asrMaxMemory<C>(
        _ benchmark: AsrMaxMemoryBenchmark,
        observer: MeasurementObserver,
        warmup: MemoryMeasurementWarmup,
        ctx: C,
        attempt: (inout C) async throws -> VlMaxMemoryMeasurementReading
    ) async throws -> BenchmarkResultData {
        try await mediaMaxMemory(marker: benchmark.benchmarkId.value, observer: observer,
                                 warmup: warmup, ctx: ctx, attempt: attempt)
    }
}
