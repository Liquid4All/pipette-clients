import Foundation

// The crate's `pipette-llamacpp/src/execute/vl_throughput.rs`, which posts a PNG to
// `llama-server`; this drives mtmd directly. `prompt_ms` times the vision encode and the
// prefill, `predicted_ms` the generation (every decode step, the last token's included);
// `image_tokens` is mtmd's per-chunk count. `total_time_ms` is one window from
// `tokenizeChunks` (the synthetic frames, text tokenization and image preprocessing) to the
// last sample, the scope of `LlamaEndToEndLatency.swift`, ending where `llama-server` ends
// a request.

nonisolated extension LlamaExecute {
    static func vlThroughput(
        _ benchmark: VlThroughputBenchmark,
        _ vis: VisionInference,
        readinessGate: ReadinessGate,
        observer: MeasurementObserver
    ) async throws -> BenchmarkResultData {
        let inputs = try LlamaMtmd.CtxInputs(benchmark)
        let images = Int(inputs.numImages)
        let width = inputs.width
        let height = inputs.height
        let decodeTokens = Int(inputs.decodeTokens)
        let prompt = try await visionPrompt(vis, textTokens: inputs.textTokens, images: images)

        let clock = ContinuousClock()
        return try await BenchmarkMeasurement.vlThroughput(
            benchmark, reps: BenchmarkMeasurement.reps, readinessGate: readinessGate,
            observer: observer, warmup: .once, ctx: vis
        ) { vis in
            vis.resetContext()
            vis.resetSampler()
            // Resetting stays outside the window, as in `LlamaEndToEndLatency`.
            let requestStarted = clock.now
            let chunks = try vis.tokenizeChunks(prompt, width, height, images)
            defer { chunks.free() }
            let promptStarted = clock.now
            try vis.evalChunks(chunks)
            let promptMs = durationMs(clock.now - promptStarted)
            let decodeStarted = clock.now
            let generation = try vis.generate(decodeTokens, .afterLastDecode)
            let predictedMs = durationMs(clock.now - decodeStarted)
            try requireImageChunk(chunks, .vlThroughput)
            return try VlThroughputMeasurementReading.fromMillis(
                promptTokens: try PositiveCount(chunks.promptTokens),
                imageTokens: try PositiveCount(chunks.imageTokens),
                decodeTokens: try PositiveCount(generation.tokens),
                promptMs: promptMs, predictedMs: predictedMs,
                totalTimeMs: durationMs(generation.lastSampledAt - requestStarted))
        }
    }

    /// The crate's single user message, which `vl_max_memory` sends too: the text sized
    /// under this model's tokenizer, as the crate sizes it through `/tokenize` (`addSpecial`
    /// on, as that endpoint counts BOS), then one marker per frame, templated so
    /// `prompt_tokens` counts the same scaffolding. Untimed.
    static func visionPrompt(_ vis: VisionInference, textTokens: UInt32,
                             images: Int) async throws -> String {
        let text = try await PromptSeed.buildPromptText(targetTokens: Int(textTokens)) {
            try vis.tokenize($0, true).count
        }
        return try vis.applyChatTemplate(text + String(repeating: vis.mediaMarker, count: images))
    }

    /// A zero means the markers never expanded, so the prompt was plain text and the cell
    /// would report it as a vision measurement.
    static func requireImageChunk(_ chunks: VisionChunks, _ kind: BenchmarkType) throws {
        guard chunks.imageTokens > 0 else {
            throw RuntimeError.engine(
                "\(kind.rawValue) prompt tokenized to \(chunks.promptTokens) tokens with no "
                    + "image chunk among them; the media marker did not expand")
        }
    }

}

nonisolated extension LlamaMtmd.CtxInputs {
    /// The workload a `vl_throughput` cell sizes its context for, its frame count bounded.
    init(_ benchmark: VlThroughputBenchmark) throws {
        try self.init(width: benchmark.parameterImageWidth.value.magnitude,
                      height: benchmark.parameterImageHeight.value.magnitude,
                      textTokens: benchmark.parameterTextTokens.value.magnitude,
                      decodeTokens: benchmark.parameterDecodeTokens.value.magnitude,
                      numImages: benchmark.parameterNumImages.value.magnitude)
    }
}
