import Foundation

// The crate's `pipette-llamacpp/src/execute/prefill_throughput.rs`
// (`llama-bench --n-prompt N --n-gen 0`): the cell's prompt into an empty KV cache.

nonisolated extension LlamaExecute {
    static func prefillThroughput(
        _ benchmark: PrefillThroughputBenchmark,
        _ inf: Inference,
        readinessGate: ReadinessGate,
        observer: MeasurementObserver
    ) async throws -> BenchmarkResultData {
        let tokens = try seedTokens(inf, target: Int(benchmark.parameterPrefillTokens.value),
                                    addSpecial: true)
        // Metal compiles pipelines per shape, so the warm-up runs the cell's own.
        return try await BenchmarkMeasurement.prefill(
            benchmark, reps: BenchmarkMeasurement.reps, readinessGate: readinessGate,
            observer: observer, warmup: .once, ctx: inf
        ) { inf in
            inf.resetContext()
            inf.resetSampler()
            let (ms, _) = try await BenchmarkMeasurement.timed { try inf.prefill(tokens) }
            return .perPrompt([try PrefillMeasurementPerPromptReading.fromMillis(
                prefillTokens: try PositiveCount(tokens.count), prefillTimeMs: ms)])
        }
    }
}
