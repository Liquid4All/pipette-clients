import Foundation

// The ONNX Runtime GenAI generation loop and the timing taken around it. Ported from
// PIP-589's definitions, so an iOS number and a host number measure the same thing.

nonisolated extension OnnxGenaiRuntime {
    /// The engine has no prefill/decode split: `appendTokens` may defer the prefill to the
    /// first `generateNextToken`. So this records what is observable either way.
    struct GenerationTiming: Sendable, Equatable {
        let promptTokens: Int
        let generatedTokens: Int
        /// `appendTokens` and the first `generateNextToken` together, since the prefill may
        /// run in either.
        let ttftMs: Double
        /// Mean per token after the first; `nil` for a one-token generation.
        let tpotMs: Double?

        /// Reconstructed from the intervals, as the crate composes it, rather than wall
        /// clock that would include tokenizing and teardown.
        var endToEndMs: Double {
            ttftMs + (tpotMs ?? 0) * Double(max(0, generatedTokens - 1))
        }
    }

    /// `min_length` is this engine's `ignore_eos`; with `max_length` it pins the generation
    /// to exactly the cell's length.
    static func generate(_ loaded: Loaded, promptIds: [Int32],
                         decodeTokens: Int) throws -> GenerationTiming {
        guard decodeTokens > 0 else {
            throw RuntimeError.engine("a generation of \(decodeTokens) tokens measures nothing")
        }
        let params = try OGAGeneratorParams(model: loaded.model)
        let total = Double(promptIds.count + decodeTokens)
        try params.setSearchOption("min_length", doubleValue: total)
        try params.setSearchOption("max_length", doubleValue: total)
        let generator = try OGAGenerator(model: loaded.model, params: params)

        let clock = ContinuousClock()
        let started = clock.now
        try generator.appendTokens(promptIds.map { NSNumber(value: $0) })
        try generator.generateNextToken()
        let firstToken = clock.now

        // A fixed count, not `isDone`, which would make a silent early stop look complete.
        for _ in 1 ..< decodeTokens {
            try generator.generateNextToken()
        }
        let finished = clock.now

        let ttftMs = milliseconds(firstToken - started)
        let remaining = decodeTokens - 1
        let tpotMs = remaining > 0
            ? milliseconds(finished - firstToken) / Double(remaining)
            : nil
        return GenerationTiming(promptTokens: promptIds.count, generatedTokens: decodeTokens,
                                ttftMs: ttftMs, tpotMs: tpotMs)
    }

    static func encode(_ text: String, with tokenizer: OGATokenizer) throws -> [Int32] {
        let sequences = try tokenizer.encode(text)
        // `sequenceCount` returns a size, so ObjC import leaves its `NSError` out-parameter;
        // `sequenceData` returns a nullable pointer and so becomes throwing.
        var error: NSError?
        let length = sequences.sequenceCount(at: 0, error: &error)
        if let error { throw error }
        let data = try sequences.sequenceData(at: 0)
        guard length > 0 else {
            throw RuntimeError.engine("the tokenizer produced an empty sequence for the prompt")
        }
        // `data` points into `sequences`, which ARC could otherwise release before the copy.
        return withExtendedLifetime(sequences) {
            Array(UnsafeBufferPointer(start: data, count: length))
        }
    }

    /// The shared corpus text under this model's own tokenizer, not synthetic ids and not
    /// chat-templated: a prefill number only compares with llama.cpp's and MLX's if all three
    /// ran the same text through their own tokenizers.
    static func promptIds(targetTokens: Int, tokenizer: OGATokenizer) async throws -> [Int32] {
        let text = try await PromptSeed.buildPromptText(targetTokens: targetTokens) {
            try encode($0, with: tokenizer).count
        }
        return try encode(text, with: tokenizer)
    }
}

// MARK: - Timing intervals

/// A free function: an extension member would pick up the module's default actor isolation
/// and be uncallable from the nonisolated loop above.
private nonisolated func milliseconds(_ duration: Duration) -> Double {
    let (seconds, attoseconds) = duration.components
    return Double(seconds) * 1000 + Double(attoseconds) / 1_000_000_000_000_000
}
