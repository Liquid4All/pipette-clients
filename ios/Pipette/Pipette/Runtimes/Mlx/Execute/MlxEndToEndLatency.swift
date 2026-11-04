import Foundation
import MLX
import MLXHuggingFace
import MLXLMCommon
import Tokenizers

// The crate's `pipette-mlx/src/execute/end_to_end_latency.rs`: tokenize, prefill and decode
// in one timed span; building the prompt text is untimed.

nonisolated extension MlxExecute {
    static func endToEndLatency(
        _ benchmark: EndToEndLatencyBenchmark,
        model: ResolvedMlxModel,
        chunk: Int,
        readinessGate: ReadinessGate,
        observer: MeasurementObserver
    ) async throws -> BenchmarkResultData {
        let tokenizer = try await #huggingFaceTokenizerLoader().load(
            from: URL(fileURLWithPath: model.dir))
        let promptText = try await PromptSeed.buildPromptText(
            targetTokens: Int(benchmark.parameterPrefillTokens.value)
        ) { tokenizer.encode(text: $0).count }
        let decodeTokens = Int(benchmark.parameterDecodeTokens.value)

        // Before the load, which would warm the device the gate waited to cool.
        try readinessGate()
        return try await MlxRuntime.withFreshModel(path: model.dir) { lm in
            try await measuringCell(benchmark.benchmarkId, chunk: chunk) {
                try await BenchmarkMeasurement.endToEnd(
                    benchmark, reps: BenchmarkMeasurement.reps, readinessGate: readinessGate,
                    observer: observer, warmup: .once, ctx: lm
                ) { lm in
                    let cache = try lm.newCache(parameters: nil)
                    let (ms, counts) = await BenchmarkMeasurement.timed {
                        let prompt = tokenizer.encode(text: promptText).map { Int32($0) }
                        let logits = MlxRuntime.prefill(lm, MLXArray(prompt).reshaped(1, -1),
                                                        cache: cache, chunk: chunk)
                        let seed = MlxRuntime.greedyToken(logits)
                        asyncEval(seed)
                        let decoded = MlxRuntime.decodeGreedy(
                            lm, from: seed, cache: cache, count: decodeTokens)
                        return EndToEndCounts(prefillTokens: prompt.count, decodeTokens: decoded)
                    }
                    return .perPrompt([try EndToEndMeasurementPerPromptReading.fromRequest(
                        prefillTokens: try PositiveCount(counts.prefillTokens),
                        decodeTokens: try PositiveCount(counts.decodeTokens), elapsedMs: ms)])
                }
            }
        }
    }
}
