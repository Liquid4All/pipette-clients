import Foundation

/// The text and vision cells time different shapes of the same loop.
nonisolated enum GenerationEnd: CaseIterable, Sendable {
    /// After sampling the last token, as `llama-server` releases a request's slot: `count`
    /// samples and `count - 1` decodes after the prefill.
    case afterLastSample
    /// After decoding the last token too: `count` samples and `count` decodes.
    case afterLastDecode
}

nonisolated struct LlamaGeneration: Sendable {
    let tokens: Int
    /// Right after the last sample, before its decode even under `.afterLastDecode`: where a
    /// `llama-server` request ends.
    let lastSampledAt: ContinuousClock.Instant
    /// Right after the first sample, where `llama-server`'s `prompt_ms` ends.
    var firstSampledAt: ContinuousClock.Instant?
}

/// The seam `LlamaExecute` runs against: bound to a loaded model by `live`, or hand-built
/// in tests.
nonisolated struct Inference {
    var tokenize: (_ text: String, _ addSpecial: Bool) throws -> [Int32]
    var resetContext: () -> Void
    var resetSampler: () -> Void
    var prefill: (_ tokens: [Int32]) throws -> Void
    /// Decodes without sampling, as `llama-bench`'s `test_gen` does. Returns the count
    /// decoded, so a cell can check the workload it asked for.
    var decode: (_ count: Int) throws -> Int
    /// Greedy, with every end-of-generation token masked.
    var generate: (_ count: Int, _ end: GenerationEnd) throws -> LlamaGeneration
    var chatCompletion: (_ messagesJSON: String, _ maxTokens: Int, _ mcqChoices: [String]?,
                         _ sampling: Sampling, _ seed: UInt32?) throws -> EvalGeneration
}

extension Inference {
    nonisolated static func live(_ m: LlamaModel) -> Inference {
        Inference(
            tokenize: { try LlamaCpp.tokenize(m, $0, addSpecial: $1) },
            resetContext: { LlamaCpp.resetContext(m) },
            resetSampler: { LlamaCpp.resetSampler(m) },
            prefill: { try LlamaCpp.prefill(m, $0) },
            decode: { try LlamaCpp.decode(m, count: $0) },
            generate: { try LlamaCpp.generate(m, count: $0, end: $1) },
            chatCompletion: {
                try LlamaCpp.chatCompletion(m, messagesJSON: $0, maxTokens: $1, mcqChoices: $2,
                                            sampling: $3, seed: $4)
            })
    }
}

/// Loads, runs `body`, and frees on exit; the `LlamaModel` never escapes. Takes a path:
/// resolving the model is `ResolvedGgufTextModel`'s job, as the crate splits `models.rs`.
nonisolated func withInference<T>(path: String, nGpuLayers: UInt32,
                                  contextSize: UInt32, nUbatch: UInt32, threads: UInt32,
                                  swaFull: Bool,
                                  loadMode: LlamacppLoadMode,
                                  _ body: (Inference) async throws -> T) async throws -> T {
    let m = try LlamaCpp.load(path: path, nGpuLayers: nGpuLayers,
                              contextSize: contextSize, nUbatch: nUbatch, threads: threads,
                              swaFull: swaFull, loadMode: loadMode)
    defer { m.free() }
    return try await body(.live(m))
}
