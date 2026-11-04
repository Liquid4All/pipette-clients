import Foundation
import MLX
import MLXLMCommon

// The crate's `pipette-mlx/src/execute/prefill_throughput.rs`: the cell's prompt into a
// fresh cache.

nonisolated extension MlxExecute {
    static func prefillThroughput(
        _ benchmark: PrefillThroughputBenchmark,
        model: ResolvedMlxModel,
        chunk: Int,
        readinessGate: ReadinessGate,
        observer: MeasurementObserver
    ) async throws -> BenchmarkResultData {
        let prompt = PromptSeed.syntheticTokenIds(Int(benchmark.parameterPrefillTokens.value))

        // Before the load, which would warm the device the gate waited to cool.
        try readinessGate()
        return try await MlxRuntime.withFreshModel(path: model.dir) { lm in
            let ids = MLXArray(prompt).reshaped(1, -1)
            return try await measuringCell(benchmark.benchmarkId, chunk: chunk) {
                try await BenchmarkMeasurement.prefill(
                    benchmark, reps: BenchmarkMeasurement.reps, readinessGate: readinessGate,
                    observer: observer, warmup: .once, ctx: lm
                ) { lm in
                    let cache = try lm.newCache(parameters: nil)
                    let (ms, _) = await BenchmarkMeasurement.timed {
                        MlxRuntime.prefill(lm, ids, cache: cache, chunk: chunk)
                    }
                    return .perPrompt([try PrefillMeasurementPerPromptReading.fromMillis(
                        prefillTokens: try PositiveCount(ids.dim(1)), prefillTimeMs: ms)])
                }
            }
        }
    }
}
