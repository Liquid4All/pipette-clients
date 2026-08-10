import Foundation

// Bound MLX model path projection from a prepared `RunRequest` — the Swift counterpart of
// `pipette-mlx/src/models.rs`. Install is `ensureModel`.

nonisolated enum MLXModels {
    /// Bound MLX model directory — the crate's `require_mlx_model_dir`. A directory, not a
    /// file: the bundle is `config.json` plus its safetensors shards.
    static func requireMlxModelDir(_ req: RunRequest) throws -> String {
        guard case let .mlx(m) = req.model.bound,
              case let .absoluteDir(dir) = m.source else {
            throw UnexpectedBoundModel(expected: "Mlx AbsoluteDir",
                                       got: req.model.declared.artifactName)
        }
        let directory = dir.value
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: directory, isDirectory: &isDirectory),
              isDirectory.boolValue
        else {
            throw BoundPathMissing(what: "MLX model directory", path: directory)
        }
        return directory
    }
}
