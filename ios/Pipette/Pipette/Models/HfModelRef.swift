import Foundation

/// A directory model (MLX or ONNX Runtime GenAI) to download: a repo, optionally narrowed
/// to one subtree. The on-disk location is not derived here: that is `ModelStorageKey`.
nonisolated struct HfModelRef: Hashable, Codable, Sendable, CustomStringConvertible {
    let repo: HfRepo
    /// `nil` → the model is at the repo root (the whole repo is one model).
    let subpath: RepoSubpath?

    /// `org/name` or `org/name/subpath`.
    var description: String { subpath.map { "\(repo)/\($0)" } ?? "\(repo)" }
    var key: String { description }

    var leaf: String { subpath?.leaf ?? repo.repoName.value }

    /// HubApi match globs. A subpath takes its whole subtree (`fnmatch` with flags 0, so
    /// `*` crosses `/`).
    ///
    /// At a repo root MLX names its files, because the repo may also carry a GGUF or
    /// PyTorch copy. ONNX takes everything: its external-data sidecars have arbitrary
    /// names, so any allowlist could drop one the model needs.
    func globs(for type: ModelType) -> [String] {
        if let subpath { return ["\(subpath.value)/*"] }
        switch type {
        case .onnxGenai: return ["*"]
        default: return Self.rootGlobs
        }
    }

    /// The pin, else `main`. `repo.revision` stays `nil` when unpinned because that is what
    /// keys the store entry.
    var hubRevision: String { repo.revision?.value ?? "main" }

    static func parse(repo repoSlug: String, subpath sub: String? = nil) throws -> HfModelRef {
        // A bare `RepoSubpath.init` reference can bind to the non-throwing
        // `init(validated:)` and skip validation.
        HfModelRef(repo: try HfRepo.parseOrgRepo(repoSlug), subpath: try sub.map { try RepoSubpath($0) })
    }

    func asMlx() -> Mlx {
        Mlx(source: .huggingFace(repo: repo, prefix: subpath))
    }

    /// `nil` for a model that is not a directory model with a Hugging Face coordinate.
    static func of(_ model: Model) -> HfModelRef? {
        switch model {
        case let .mlx(m): return m.ref
        case let .onnxGenai(m): return m.ref
        case .ggufText, .ggufVision, .ggufVisionSpeculative, .appleFoundationText: return nil
        }
    }

    /// MLX files at a repo root. `*.jinja` is the chat template eval needs.
    static let rootGlobs = [
        "*.safetensors", "*.safetensors.index.json", "config.json",
        "generation_config.json", "tokenizer.json", "tokenizer_config.json",
        "tokenizer.model", "special_tokens_map.json", "added_tokens.json",
        "*.jinja", "*.tiktoken", "vocab.json", "merges.txt",
    ]
}
