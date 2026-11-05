import Foundation
import MLX
import MLXLMCommon

// The crate's `pipette-mlx/src/execute/decode_throughput.rs`: the prompt prefilled untimed,
// then the decode alone timed.

nonisolated extension MlxExecute {
    static func decodeThroughput(
        _ benchmark: DecodeThroughputBenchmark,
        model: ResolvedMlxModel,
        chunk: Int,
        readinessGate: ReadinessGate,
        observer: MeasurementObserver
    ) async throws -> BenchmarkResultData {
        let prompt = PromptSeed.syntheticTokenIds(Int(benchmark.parameterPrefillTokens.value))
        let decodeTokens = Int(benchmark.parameterDecodeTokens.value)

        // Before the load, which would warm the device the gate waited to cool.
        try readinessGate()
        return try await MlxRuntime.withFreshModel(path: model.dir) { lm in
            let ids = MLXArray(prompt).reshaped(1, -1)
            return try await measuringCell(benchmark.benchmarkId, chunk: chunk) {
                try await BenchmarkMeasurement.decode(
                    benchmark, reps: BenchmarkMeasurement.reps, readinessGate: readinessGate,
                    observer: observer, warmup: .once, ctx: lm
                ) { lm in
                    let cache = try lm.newCache(parameters: nil)
                    let seed = MlxRuntime.greedyToken(
                        MlxRuntime.prefill(lm, ids, cache: cache, chunk: chunk))
                    // Qualified: inside this type, a bare `eval` is the eval benchmark.
                    MLX.eval(seed)
                    let (ms, decoded) = await BenchmarkMeasurement.timed {
                        MlxRuntime.decodeGreedy(lm, from: seed, cache: cache, count: decodeTokens)
                    }
                    return .perPrompt([try DecodeMeasurementPerPromptReading.fromMillis(
                        prefillTokens: UInt32(ids.dim(1)),
                        decodeTokens: try PositiveCount(decoded), decodeTimeMs: ms)])
                }
            }
        }
    }
}
