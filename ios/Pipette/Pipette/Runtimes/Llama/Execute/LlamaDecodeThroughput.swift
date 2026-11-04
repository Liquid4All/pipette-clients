import Foundation

// The crate's `pipette-llamacpp/src/execute/decode_throughput.rs`
// (`llama-bench --n-prompt 0 --n-gen D --n-depth P`): only the decode is timed.

nonisolated extension LlamaExecute {
    static func decodeThroughput(
        _ benchmark: DecodeThroughputBenchmark,
        _ inf: Inference,
        readinessGate: ReadinessGate,
        observer: MeasurementObserver
    ) async throws -> BenchmarkResultData {
        let tokens = try seedTokens(inf, target: Int(benchmark.parameterPrefillTokens.value),
                                    addSpecial: true)
        let decodeTokens = Int(benchmark.parameterDecodeTokens.value)
        return try await BenchmarkMeasurement.decode(
            benchmark, reps: BenchmarkMeasurement.reps, readinessGate: readinessGate,
            observer: observer, warmup: .once, ctx: inf
        ) { inf in
            inf.resetContext()
            inf.resetSampler()
            try inf.prefill(tokens)
            let (ms, decoded) = try await BenchmarkMeasurement.timed { try inf.decode(decodeTokens) }
            return .perPrompt([try DecodeMeasurementPerPromptReading.fromMillis(
                prefillTokens: UInt32(tokens.count),
                decodeTokens: try PositiveCount(decoded), decodeTimeMs: ms)])
        }
    }
}
