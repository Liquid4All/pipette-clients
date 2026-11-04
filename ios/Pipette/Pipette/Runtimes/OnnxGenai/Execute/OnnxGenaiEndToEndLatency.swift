import Foundation

// The crate's `pipette-onnx-genai/src/execute/end_to_end_latency.rs`.

nonisolated extension OnnxGenaiExecute {
    /// `ttft + tpot * (n - 1)` (`GenerationTiming.endToEndMs`): reconstructed, so tokenizing
    /// and building the generator are left out.
    static func endToEndLatency(
        _ benchmark: EndToEndLatencyBenchmark,
        model: ResolvedOnnxGenaiModel,
        readinessGate: ReadinessGate,
        observer: MeasurementObserver
    ) async throws -> BenchmarkResultData {
        let decodeTokens = Int(benchmark.parameterDecodeTokens.value)
        // Before the load, which would warm the device the gate waited to cool.
        try readinessGate()
        return try await OnnxGenaiRuntime.withFreshModel(path: model.dir) { loaded in
            let ids = try await OnnxGenaiRuntime.promptIds(
                targetTokens: Int(benchmark.parameterPrefillTokens.value), tokenizer: loaded.tokenizer)
            return try await BenchmarkMeasurement.endToEnd(
                benchmark, reps: BenchmarkMeasurement.reps, readinessGate: readinessGate,
                observer: observer, warmup: .once, ctx: loaded
            ) { loaded in
                let timing = try OnnxGenaiRuntime.generate(loaded, promptIds: ids,
                                                           decodeTokens: decodeTokens)
                // Built from its fields rather than `fromRequest`, whose time is the client's
                // clock around a request.
                return .perPrompt([EndToEndMeasurementPerPromptReading(
                    prefillTokens: try PositiveCount(timing.promptTokens),
                    decodeTokens: try PositiveCount(timing.generatedTokens),
                    totalTimeMs: try PositiveMillis(Float(timing.endToEndMs)))])
            }
        }
    }
}
