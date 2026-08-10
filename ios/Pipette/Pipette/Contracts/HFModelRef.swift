import Foundation

/// A single MLX model to download: a repo, optionally narrowed to one model
/// directory inside a multi-model repo (`subpath`). Every derivation the pipeline
/// needs — download key, display leaf, HubApi globs — lives here, so no caller
/// re-rolls string logic. The on-disk location is not one of them: that is the
/// spec's `ModelStorageKey` entry, shared by every model format.
nonisolated struct HFModelRef: Hashable, Codable, Sendable, CustomStringConvertible {
    let repo: HFRepo
    /// `nil` → the model is at the repo root (the whole repo is one model).
    let subpath: RepoSubpath?

    /// Canonical form + stable identity key: `org/name` or `org/name/subpath`.
    var description: String { subpath.map { "\(repo)/\($0)" } ?? "\(repo)" }
    var key: String { description }

    /// The model directory's leaf name on disk.
    var leaf: String { subpath?.leaf ?? repo.repoName.value }

    /// HubApi match globs. A subpath scopes its entire subtree (POSIX `fnmatch`,
    /// flags = 0, so `*` crosses `/`); a root repo pulls the MLX file set.
    var globs: [String] { subpath.map { ["\($0.value)/*"] } ?? Self.rootGlobs }

    static func parse(repo repoSlug: String, subpath sub: String? = nil) throws -> HFModelRef {
        // Explicit throwing init so validation runs — a bare `RepoSubpath.init`
        // reference can bind to the non-throwing `init(validated:)` and skip it.
        HFModelRef(repo: try HFRepo.parse(repoSlug), subpath: try sub.map { try RepoSubpath($0) })
    }

    /// The MLX model spec (identity) for this ref — the one place a ref is widened back
    /// into a `Mlx`. Generation flags are not involved: they belong to the cell.
    func asMlx() -> Mlx {
        Mlx(source: .huggingFace(repo: repo, prefix: subpath))
    }

    /// MLX model files at a repo root: weight shards + index + config + tokenizer +
    /// chat template (`*.jinja`, needed by eval's chat-templating).
    static let rootGlobs = [
        "*.safetensors", "*.safetensors.index.json", "config.json",
        "generation_config.json", "tokenizer.json", "tokenizer_config.json",
        "tokenizer.model", "special_tokens_map.json", "added_tokens.json",
        "*.jinja", "*.tiktoken", "vocab.json", "merges.txt",
    ]
}
