import Foundation

// Bound GGUF path projection for a prepared `RunRequest` — the Swift counterpart of
// `pipette-llamacpp/src/models.rs`, named for its engine as every file in this folder is.
//
// Install is `ensureModel`. This module only checks the bound model and returns host paths
// for execute, which is why it sits beside the engine rather than in the store.
//
// Matches `req.model.bound` against the `absolute*` arms `bindUnder` produces, exactly as
// the crate does. Two guarantees per call: the arm is this engine's, and the file is
// really there.

nonisolated enum LlamaModels {
    /// Bound main GGUF for text cells — the crate's `require_gguf_text`.
    static func requireGgufText(_ req: RunRequest) throws -> String {
        guard case let .ggufText(m) = req.model.bound,
              case let .absoluteFile(path) = m.source else {
            throw UnexpectedBoundModel(expected: "GgufText AbsoluteFile",
                                       got: req.model.declared.artifactName)
        }
        return try requireFile(path.value, "GGUF text model")
    }

    /// Bound weights + projector for vision cells — the crate's `require_gguf_vision`.
    ///
    /// Returns `(model, mmproj)`. The projector is checked, not assumed: an entry bound
    /// without one has no `absoluteFiles` arm to match, which is what this refuses by name.
    ///
    /// Called from the `vl_throughput` arm, as upstream calls it from `vl_throughput.rs`
    /// alone — the one benchmark plan-types gives gguf_vision a variant for. That arm is
    /// still unsupported here (image-embed is future work), so this is the half of the
    /// module that lands with it.
    static func requireGgufVision(_ req: RunRequest) throws -> (model: String, mmproj: String) {
        guard case let .ggufVision(m) = req.model.bound,
              case let .absoluteFiles(path, mmprojPath) = m.source else {
            throw UnexpectedBoundModel(expected: "GgufVision AbsoluteFiles",
                                       got: req.model.declared.artifactName)
        }
        return (try requireFile(path.value, "GGUF vision model"),
                try requireFile(mmprojPath.value, "GGUF mmproj"))
    }

    private static func requireFile(_ path: String, _ what: String) throws -> String {
        guard FileManager.default.fileExists(atPath: path) else {
            throw BoundPathMissing(what: what, path: path)
        }
        return path
    }
}
