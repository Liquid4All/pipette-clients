import Foundation

// The crate's `pipette-onnx-genai/src/execute/prefill_throughput.rs`.

nonisolated extension OnnxGenaiExecute {
    /// The time to the first token of a one-token generation over the cell's prompt
    /// (`GenerationTiming.ttftMs`).
    static func prefillThroughput(
        _ benchmark: PrefillThroughputBenchmark,
        model: ResolvedOnnxGenaiModel,
        readinessGate: ReadinessGate,
        observer: MeasurementObserver
    ) async throws -> BenchmarkResultData {
        try readinessGate()
        return try await OnnxGenaiRuntime.withFreshModel(path: model.dir) { loaded in
            let ids = try await OnnxGenaiRuntime.promptIds(
                targetTokens: Int(benchmark.parameterPrefillTokens.value), tokenizer: loaded.tokenizer)
            return try await BenchmarkMeasurement.prefill(
                benchmark, reps: BenchmarkMeasurement.reps, readinessGate: readinessGate,
                observer: observer, warmup: .once, ctx: loaded
            ) { loaded in
                let timing = try OnnxGenaiRuntime.generate(loaded, promptIds: ids, decodeTokens: 1)
                return .perPrompt([try PrefillMeasurementPerPromptReading.fromStream(
                    prefillTokens: try PositiveCount(timing.promptTokens),
                    completionTokens: try PositiveCount(timing.generatedTokens),
                    ttftMs: timing.ttftMs)])
            }
        }
    }
}
