import Foundation

// The crate's `pipette-llamacpp/src/execute/vl_max_memory.rs`, which posts one image request
// to `llama-server` under a peak poller; this brackets the model and projector load, the
// image prefill and the cell's decode in process.

nonisolated extension LlamaExecute {
    /// Answers the flags the cell ran with: the context is sized inside the bracket.
    static func vlMaxMemory(
        _ benchmark: VlMaxMemoryBenchmark,
        _ request: RunRequest,
        flags: RuntimeFlags,
        model: ResolvedGgufVisionModel
    ) async throws -> (BenchmarkResultData, RuntimeFlags) {
        var ran = flags
        let resultData = try await BenchmarkMeasurement.vlMaxMemory(
            benchmark, warmup: .cold, ctx: ()
        ) { _ in
            let peak = try await ProcessMemory.maxMemoryBracket(label: "llama vl") {
                (_, ran) = try await withCalibratedVision(
                    request, model, flags: flags, width: benchmark.parameterImageWidth.value.magnitude,
                    height: benchmark.parameterImageHeight.value.magnitude
                ) { try await vlMaxMemory(benchmark, $0) }
            }
            return VlMaxMemoryMeasurementReading(maxHostBytes: peak.maxHostBytes,
                                                 maxGpuBytes: peak.maxGpuBytes,
                                                 maxNpuBytes: peak.maxNpuBytes)
        }
        return (resultData, ran)
    }

    /// Apart from the bracket so tests run it on a hand-built `VisionInference`.
    ///
    /// The decode is not counted: `generate` runs every step it is asked for, end of
    /// generation masked. The image chunk is, since without one the prompt was text.
    static func vlMaxMemory(_ benchmark: VlMaxMemoryBenchmark, _ vis: VisionInference) async throws {
        let inputs = try LlamaMtmd.CtxInputs(benchmark)
        let images = Int(inputs.numImages)
        let prompt = try await visionPrompt(vis, textTokens: inputs.textTokens, images: images)
        vis.resetContext()
        vis.resetSampler()
        let chunks = try vis.tokenizeChunks(prompt, inputs.width, inputs.height, images)
        defer { chunks.free() }
        try requireImageChunk(chunks, .vlMaxMemory)
        try vis.evalChunks(chunks)
        // N decodes after the prefill, as the VL throughput cell runs them; the crate's
        // request runs N-1, its first token the prefill's.
        _ = try vis.generate(Int(inputs.decodeTokens), .afterLastDecode)
    }
}

nonisolated extension LlamaMtmd.CtxInputs {
    /// A cell without an output length decodes one token, so older cells keep measuring
    /// what they did.
    init(_ benchmark: VlMaxMemoryBenchmark) throws {
        try self.init(width: benchmark.parameterImageWidth.value.magnitude,
                      height: benchmark.parameterImageHeight.value.magnitude,
                      textTokens: benchmark.parameterTextTokens.value.magnitude,
                      decodeTokens: benchmark.parameterDecodeTokens?.value.magnitude ?? 1,
                      numImages: benchmark.parameterNumImages.value.magnitude)
    }
}
