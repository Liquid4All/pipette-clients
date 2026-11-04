import Foundation

/// `VisionInference` for the ASR cells.
nonisolated struct AudioInference {
    var mediaMarker: String
    /// The text filler's counter, as the CLI counts it through `/tokenize`.
    var tokenize: (_ text: String, _ addSpecial: Bool) throws -> [Int32]
    var render: (_ userContent: String) throws -> String
    var resetContext: () -> Void
    var resetSampler: () -> Void
    /// Clip decode and mel run inside the whole request's window, as in llama-server's handler.
    var tokenizeClip: (_ prompt: String, _ clipPath: String) throws -> VisionChunks
    /// What the server's `prompt_n` reads for a clip-free prompt.
    var promptTokens: (_ prompt: String) throws -> Int
    /// The `prompt_ms` window.
    var evalChunks: (_ chunks: VisionChunks) throws -> Void
    /// The `predicted_ms` window.
    var generate: (_ count: Int, _ end: GenerationEnd) throws -> LlamaGeneration
}

extension AudioInference {
    nonisolated static func live(_ m: LlamaModel, _ mtmd: MtmdContext,
                                 _ recipe: AsrRecipe) -> AudioInference {
        AudioInference(
            mediaMarker: LlamaMtmd.defaultMediaMarker,
            tokenize: { try LlamaCpp.tokenize(m, $0, addSpecial: $1) },
            render: { try recipe.render(m.model, userContent: $0) },
            resetContext: { LlamaCpp.resetContext(m) },
            resetSampler: { LlamaCpp.resetSampler(m) },
            tokenizeClip: { try LlamaMtmd.tokenizeClip(mtmd, text: $0, clipPath: $1) },
            promptTokens: { try LlamaMtmd.promptTokens(mtmd, text: $0) },
            evalChunks: { try LlamaMtmd.evalChunks(mtmd, m, $0) },
            generate: { try LlamaCpp.generate(m, count: $0, end: $1) })
    }
}
