import Foundation

// The crate's `pipette-onnx-genai/src/execute/max_memory_usage.rs`: one prefill and one
// decode on a fresh load, under the process-footprint sampler counting the mapped weights too
// (`OnnxGenaiRuntime.footprintWithMappedWeights`). Unlike AFM's model, this engine runs in this
// process, where `ProcessMemory` sees its arenas.

nonisolated extension OnnxGenaiExecute {
    static func maxMemoryUsage(
        _ benchmark: MaxMemoryUsageBenchmark,
        model: ResolvedOnnxGenaiModel,
        observer: MeasurementObserver
    ) async throws -> BenchmarkResultData {
        try await BenchmarkMeasurement.maxMemory(benchmark, observer: observer, warmup: .cold,
                                                 ctx: ()) { _ in
            try await ProcessMemory.maxMemoryBracket(
                label: "onnx", counter: OnnxGenaiRuntime.footprintWithMappedWeights
            ) {
                try await OnnxGenaiRuntime.withFreshModel(path: model.dir) { loaded in
                    let ids = try await OnnxGenaiRuntime.promptIds(
                        targetTokens: Int(benchmark.parameterPrefillTokens.value),
                        tokenizer: loaded.tokenizer)
                    _ = try OnnxGenaiRuntime.generate(loaded, promptIds: ids, decodeTokens: 1)
                }
            }
        }
    }
}
