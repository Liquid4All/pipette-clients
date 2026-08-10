import Foundation

/// The inference capabilities the benchmark methodology needs, as data — bound to
/// a loaded model at the edge (`live`) or hand-built in tests. This is the
/// detachment seam: `LlamaBenchmark` depends on this value, never on a `LlamaModel`
/// or a C pointer. The model is captured by these closures inside `withInference`
/// and dies with that scope.
nonisolated struct Inference {
    var tokenize: (_ text: String, _ addSpecial: Bool) throws -> [Int32]
    var resetContext: () -> Void
    var resetSampler: () -> Void
    var prefill: (_ tokens: [Int32]) throws -> Void
    var decodeIgnoringEoG: (_ count: Int) throws -> Void
    var chatCompletion: (_ messagesJSON: String, _ maxTokens: Int, _ mcqChoices: [String]?) throws -> EvalGeneration
}

extension Inference {
    /// Bind the live llama.cpp ops to a loaded model. The model is captured here
    /// and nowhere else — every `LlamaBenchmark` function stays model-agnostic.
    nonisolated static func live(_ m: LlamaModel) -> Inference {
        Inference(
            tokenize: { try LlamaCpp.tokenize(m, $0, addSpecial: $1) },
            resetContext: { LlamaCpp.resetContext(m) },
            resetSampler: { LlamaCpp.resetSampler(m) },
            prefill: { try LlamaCpp.prefill(m, $0) },
            decodeIgnoringEoG: { try LlamaCpp.decodeIgnoringEoG(m, count: $0) },
            chatCompletion: { try LlamaCpp.chatCompletion(m, messagesJSON: $0, maxTokens: $1, mcqChoices: $2) })
    }
}

/// Assemble the short-lived llama.cpp resources the weights at `path` need, run `body`
/// against the bound `Inference`, and free them on exit — the llama twin of
/// `MLXRuntime.withFreshModel`. The loaded `LlamaModel` never escapes this scope.
///
/// Takes a path, not a bound `Model`: deciding *which* arm is a GGUF one and that its
/// file exists is `LlamaModels.requireGgufText`, as the crate separates `models.rs` from
/// execute. A model of another format can no longer be handed here at all, so the two
/// defensive arms this used to carry are gone rather than unreachable.
nonisolated func withInference<T>(path: String, nGpuLayers: UInt32,
                                  contextSize: UInt32, nUbatch: UInt32, threads: UInt32,
                                  swaFull: Bool,
                                  _ body: (Inference) async throws -> T) async throws -> T {
    let m = try LlamaCpp.load(path: path, nGpuLayers: nGpuLayers,
                              contextSize: contextSize, nUbatch: nUbatch, threads: threads,
                              swaFull: swaFull)
    defer { m.free() }
    return try await body(.live(m))
}
