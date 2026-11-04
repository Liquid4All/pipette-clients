import Foundation

// The crate's `pipette-mlx/src/models.rs`.

/// The crate's `REQUIRED_FILES`. Weight and tokenizer file names vary by repo, so naming
/// them would refuse snapshots that load fine.
private nonisolated let requiredFiles = ["config.json"]

nonisolated struct ResolvedMlxModel: Equatable, Sendable {
    let dir: String

    init(_ model: Model) throws {
        dir = try requireBoundModelDir(model, expected: .mlx, requiredFiles: requiredFiles)
    }
}

nonisolated enum MlxModels {

    /// As `mlx_lm`'s tokenizer wrapper collects them. Without the config's ids, a template
    /// ending on a second id (`<|im_end|>` beside `</s>`) would run every eval sample to the
    /// cap and score it as unfinished.
    static func eosTokenIds(bundle: String, tokenizerEos: Int?) -> Set<Int> {
        var ids = Set<Int>()
        if let tokenizerEos { ids.insert(tokenizerEos) }
        let root = URL(fileURLWithPath: bundle, isDirectory: true)
        for file in ["config.json", "generation_config.json"] {
            guard let data = try? Data(contentsOf: root.appendingPathComponent(file)),
                  let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
            else { continue }
            switch object["eos_token_id"] {
            case let id as Int: ids.insert(id)
            case let list as [Int]: ids.formUnion(list)
            default: break
            }
        }
        return ids
    }
}
