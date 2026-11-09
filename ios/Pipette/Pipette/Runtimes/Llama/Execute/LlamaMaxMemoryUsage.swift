import Foundation

// The crate's `pipette-llamacpp/src/execute/max_memory_usage.rs`: the process footprint across
// a fresh load, one prefill and one decode.

nonisolated extension LlamaExecute {
    static func maxMemoryUsage(
        _ benchmark: MaxMemoryUsageBenchmark,
        load: LoadSettings,
        model: ResolvedGgufTextModel,
        observer: MeasurementObserver
    ) async throws -> BenchmarkResultData {
        try await BenchmarkMeasurement.maxMemory(benchmark, observer: observer, warmup: .cold,
                                                 ctx: ()) { _ in
            try await ProcessMemory.maxMemoryBracket(label: "llama") {
                try await load.withInference(model) {
                    try maxMemoryUsage($0, prefillTokens: Int(benchmark.parameterPrefillTokens.value))
                }
            }
        }
    }

    /// Apart from the bracket so tests run it on a hand-built `Inference`.
    static func maxMemoryUsage(_ inf: Inference, prefillTokens: Int) throws {
        let tokens = try seedTokens(inf, target: prefillTokens, addSpecial: true)
        inf.resetContext()
        inf.resetSampler()
        try inf.prefill(tokens)
        // Touches the KV path for the memory sampler to see.
        _ = try inf.decode(1)
    }
}
