import Foundation

// The crate's `pipette-llamacpp/src/execute/asr_max_memory.rs`, which posts one clip request
// to `llama-server` under a peak poller; this brackets the load, count, prefill and decode in
// process. The CLI's warm-up prompt checks run here as text-only tokenizations after the
// request, which hold nothing the request did not.

nonisolated extension LlamaExecute {
    /// The context is sized inside the bracket.
    static func asrMaxMemory(
        _ benchmark: AsrMaxMemoryBenchmark, _ request: RunRequest, cell: AsrCell,
        model: ResolvedGgufVisionModel, flags: RuntimeFlags, observer: MeasurementObserver
    ) async throws -> (BenchmarkResultData, RuntimeFlags, AsrClientExtras) {
        // Behind a pinned context the count would size nothing, as the CLI skips its helper.
        let pinned = request.runtimeFlags?.ctxSize != nil
        var ran = flags
        let resultData = try await withAsrClip(cell) { clip in
            try await BenchmarkMeasurement.asrMaxMemory(benchmark, observer: observer,
                                                        warmup: .cold, ctx: ()) { _ in
                let peak = try await ProcessMemory.maxMemoryBracket(label: "llama asr") {
                    (_, ran) = try await withCalibratedAudio(
                        request, model, cell, flags: flags, clipPath: clip, calibrate: !pinned,
                        observer: observer
                    ) { try await asrMaxMemory(cell, $0, clipPath: clip, calibrated: $1) }
                }
                return VlMaxMemoryMeasurementReading(maxHostBytes: peak.maxHostBytes,
                                                     maxGpuBytes: peak.maxGpuBytes,
                                                     maxNpuBytes: peak.maxNpuBytes)
            }
        }
        return (resultData, ran,
                AsrClientExtras(contextSizing: AsrContextSizing(pinned: pinned, calibrated: !pinned),
                                audioTokensSource: nil, beyondModelAudioLimit: cell.beyondLimit))
    }

    /// Apart from the bracket so tests run it on a hand-built `AudioInference`.
    static func asrMaxMemory(_ cell: AsrCell, _ aud: AudioInference, clipPath: String,
                             calibrated: Int?) async throws {
        let (userText, prompt) = try await asrPrompt(aud, cell)
        aud.resetContext()
        aud.resetSampler()
        let chunks = try aud.tokenizeClip(prompt, clipPath)
        defer { chunks.free() }
        try requireAudioChunk(chunks, calibrated: calibrated)
        try aud.evalChunks(chunks)
        // N-1 decodes after the prefill, as the crate's request runs them.
        let generation = try aud.generate(Int(cell.decodeTokens), .afterLastSample)
        guard generation.tokens == Int(cell.decodeTokens) else {
            throw RuntimeError.engine(
                "decoded \(generation.tokens) tokens, expected \(cell.decodeTokens)")
        }
        // `pinned` false: only an unpinned cell counts, so a count here sized the context.
        _ = try settleAudioTokens(aud, cell, userText: userText,
                                  promptTokens: chunks.promptTokens, calibrated: calibrated,
                                  pinned: false)
    }
}
