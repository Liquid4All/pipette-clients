import Foundation

/// The install gate for an MLX directory: nothing is published unless it passes, so a
/// partial or empty pull cannot pass as an installed model.
nonisolated enum MlxModelLayout {
    static func missing(in dir: URL) -> [String] {
        let names = Set((try? FileManager.default.contentsOfDirectory(atPath: dir.path)) ?? [])
        var missing: [String] = []
        if !names.contains("config.json") { missing.append("config.json") }
        if !names.contains(where: { $0.hasSuffix(".safetensors") }) { missing.append("*.safetensors") }
        let tokenizers = ["tokenizer.json", "tokenizer.model", "tokenizer_config.json"]
        if !names.contains(where: tokenizers.contains) { missing.append("tokenizer") }
        return missing
    }

    static func validate(_ dir: URL, name: String) throws {
        let missing = missing(in: dir)
        guard missing.isEmpty else { throw DownloadError.incompleteModel(name, missing: missing) }
    }
}
