import Foundation

// The crate's `pipette-mlx/src/execute/max_memory_usage.rs`: one prefill and one decode on a
// fresh load, under the process-footprint sampler.

/// The crate's `DECODE_TOKENS`: one token, so the peak covers real generation.
private nonisolated let decodeTokens = 1

nonisolated extension MlxExecute {
    static func maxMemoryUsage(
        _ benchmark: MaxMemoryUsageBenchmark,
        model: ResolvedMlxModel,
        chunk: Int,
        observer: MeasurementObserver
    ) async throws -> BenchmarkResultData {
        try await BenchmarkMeasurement.maxMemory(benchmark, observer: observer, warmup: .cold,
                                                 ctx: ()) { _ in
            try await ProcessMemory.maxMemoryBracket(label: "mlx") {
                try await MlxRuntime.withFreshModel(path: model.dir) { lm in
                    try MlxRuntime.generate(lm, prefill: Int(benchmark.parameterPrefillTokens.value),
                                            decode: decodeTokens, chunk: chunk)
                }
            }
        }
    }
}
