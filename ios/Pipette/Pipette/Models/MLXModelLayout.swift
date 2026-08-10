import Foundation

/// The install gate: validates a directory is a *complete*, loadable MLX model.
/// Nothing is relocated or recorded unless this passes, so a partial/empty download
/// (e.g. an empty-match or a poisoned HubApi cache) can never masquerade as an
/// installed model — the silent failure this pipeline was built to prevent.
nonisolated enum MLXModelLayout {
    /// The pieces an MLX model directory must contain to load: model config, at
    /// least one weight shard, and a tokenizer.
    static func missing(in dir: URL) -> [String] {
        let names = Set((try? FileManager.default.contentsOfDirectory(atPath: dir.path)) ?? [])
        var missing: [String] = []
        if !names.contains("config.json") { missing.append("config.json") }
        if !names.contains(where: { $0.hasSuffix(".safetensors") }) { missing.append("*.safetensors") }
        let tokenizers = ["tokenizer.json", "tokenizer.model", "tokenizer_config.json"]
        if !names.contains(where: tokenizers.contains) { missing.append("tokenizer") }
        return missing
    }

    /// Throws `.incompleteModel` (naming the missing pieces) if `dir` isn't loadable.
    static func validate(_ dir: URL, ref: HFModelRef) throws {
        let missing = missing(in: dir)
        guard missing.isEmpty else { throw DownloadError.incompleteModel(ref, missing: missing) }
    }
}
