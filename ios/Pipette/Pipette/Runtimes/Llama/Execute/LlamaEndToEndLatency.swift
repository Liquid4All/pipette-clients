import Foundation

// The crate's `pipette-llamacpp/src/execute/end_to_end_latency.rs`, which posts a string to
// `/completion`: tokenizing it is timed, building it is not. The generation stops where the
// server stops a request, after sampling the last token.

nonisolated extension LlamaExecute {
    static func endToEndLatency(
        _ benchmark: EndToEndLatencyBenchmark,
        _ inf: Inference,
        readinessGate: ReadinessGate,
        observer: MeasurementObserver
    ) async throws -> BenchmarkResultData {
        let promptText = try await PromptSeed.buildPromptText(
            targetTokens: Int(benchmark.parameterPrefillTokens.value)
        ) { try inf.tokenize($0, true).count }
        let decodeTokens = Int(benchmark.parameterDecodeTokens.value)
        return try await BenchmarkMeasurement.endToEnd(
            benchmark, reps: BenchmarkMeasurement.reps, readinessGate: readinessGate,
            observer: observer, warmup: .once, ctx: inf
        ) { inf in
            inf.resetContext()
            inf.resetSampler()
            let (ms, counts) = try await BenchmarkMeasurement.timed {
                let prompt = try inf.tokenize(promptText, true)
                try inf.prefill(prompt)
                let generated = try inf.generate(decodeTokens, .afterLastSample).tokens
                return EndToEndCounts(prefillTokens: prompt.count, decodeTokens: generated)
            }
            return .perPrompt([try EndToEndMeasurementPerPromptReading.fromRequest(
                prefillTokens: try PositiveCount(counts.prefillTokens),
                decodeTokens: try PositiveCount(counts.decodeTokens), elapsedMs: ms)])
        }
    }
}
