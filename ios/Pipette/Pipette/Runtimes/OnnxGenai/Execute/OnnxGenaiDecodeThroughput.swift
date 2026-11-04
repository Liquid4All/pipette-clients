import Foundation

// The crate's `pipette-onnx-genai/src/execute/decode_throughput.rs`.

nonisolated extension OnnxGenaiExecute {
    /// The mean interval after the first token, times the cell's decode count: the span the
    /// other runtimes report.
    static func decodeThroughput(
        _ benchmark: DecodeThroughputBenchmark,
        model: ResolvedOnnxGenaiModel,
        readinessGate: ReadinessGate,
        observer: MeasurementObserver
    ) async throws -> BenchmarkResultData {
        let decodeTokens = Int(benchmark.parameterDecodeTokens.value)
        try readinessGate()
        return try await OnnxGenaiRuntime.withFreshModel(path: model.dir) { loaded in
            let ids = try await OnnxGenaiRuntime.promptIds(
                targetTokens: Int(benchmark.parameterPrefillTokens.value), tokenizer: loaded.tokenizer)
            return try await BenchmarkMeasurement.decode(
                benchmark, reps: BenchmarkMeasurement.reps, readinessGate: readinessGate,
                observer: observer, warmup: .once, ctx: loaded
            ) { loaded in
                let timing = try OnnxGenaiRuntime.generate(loaded, promptIds: ids,
                                                           decodeTokens: decodeTokens)
                // `nil` only for a single-token generation, which has no interval.
                guard let tpotMs = timing.tpotMs else {
                    throw RuntimeError.engine(
                        "a \(decodeTokens)-token generation reported no per-token interval")
                }
                return .perPrompt([try DecodeMeasurementPerPromptReading.fromMillis(
                    prefillTokens: UInt32(timing.promptTokens),
                    decodeTokens: try PositiveCount(timing.generatedTokens),
                    decodeTimeMs: tpotMs * Double(decodeTokens))])
            }
        }
    }
}
