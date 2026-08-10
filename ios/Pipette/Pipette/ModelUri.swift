import Foundation

/// The compact `model=` URI, mirroring `crates/pipette-cli/src/model_uri.rs`.
///
/// The CLI's `--model` accepts a JSON `Model`, this URI, or a digest reference; `Display`
/// (`{repo}[:{path}]`) is **not** an input spelling there and is not one here either — it
/// is what logs and warehouse keys use.
///
/// Grammar, verbatim from the crate:
///
///     uri    ::= scheme "://" body            ; split on the FIRST "://"
///     scheme ::= "gguf-text" | "gguf-vision" | "mlx" | "torch" | "openvino"
///     body   ::= "" | pair ("&" pair)*        ; keys unordered, each at most once
///     pair   ::= key "=" value                ; first "=" splits; value may contain "="
///
/// iOS mirrors the repo-backed forms only. The `url=` spellings have no counterpart —
/// `Model` deliberately has no absolute-URL arm — and `torch`/`openvino` name engines this
/// app does not link. Both are refused by name rather than silently unmatched, so a cell
/// this device cannot run fails at parse instead of at load.
nonisolated enum ModelUri {
    /// Why a `model=` value is not a model. Mirrors the crate's `ModelUriError` for the
    /// cases iOS can reach, and adds the two the crate has no need for (an engine this
    /// build lacks, a digest this client cannot resolve).
    enum Failure: Error, Equatable {
        case notAUri
        case unknownScheme(String)
        case unsupportedScheme(String)
        case malformedPair(String)
        case duplicateKey(String)
        case unknownKey(String)
        case missingKey(String)
        case notRepresentable(String)

        var reason: String {
            switch self {
            case .notAUri:
                return "expected a `<scheme>://<key>=<value>` URI or a JSON object"
            case let .unknownScheme(scheme):
                return "unknown scheme `\(scheme)`"
            case let .unsupportedScheme(scheme):
                return "`\(scheme)` names an engine this build does not link"
            case let .malformedPair(pair):
                return "`\(pair)` is not a `key=value` pair"
            case let .duplicateKey(key):
                return "`\(key)` appears more than once"
            case let .unknownKey(key):
                return "no such key `\(key)`"
            case let .missingKey(key):
                return "missing `\(key)`"
            case let .notRepresentable(what):
                return "\(what) cannot be expressed on this client"
            }
        }
    }

    /// Every key the crate's vocabulary defines, so an unrecognized one is reported as
    /// unknown and a *recognized but unrepresentable* one (a URL source, a digest) gets
    /// its own refusal. Spelling matches `model_uri.rs`'s `KEY_*` constants.
    private static let knownKeys: Set<String> = [
        "repo", "path", "model", "mmproj", "prefix", "rev", "url",
        "sha256", "model_sha256", "mmproj_sha256",
    ]

    /// The compact URI this model reads back from — the crate's `model_to_uri`, and the
    /// form `models`/`benchmarks run` print so a listing feeds `--model` directly.
    ///
    /// `nil` where no URI can name the model on this client: a store-relative or bound
    /// arm (already-installed bytes, not an importable coordinate), a `url=` source, and
    /// Apple Foundation, which ships with the OS. A pinned `sha256` also returns `nil`
    /// rather than a URI that drops it — `parse` refuses digests, so emitting one would
    /// print something this client cannot read back.
    static func uri(for model: Model) -> String? {
        switch model {
        case let .ggufText(m):
            guard case let .huggingFace(repo, path, sha256) = m.source, sha256 == nil else {
                return nil
            }
            return body("gguf-text", repo, [("path", path.value)])
        case let .ggufVision(m):
            guard case let .huggingFace(repo, model, modelSha, mmproj, mmprojSha) = m.source,
                  modelSha == nil, mmprojSha == nil
            else { return nil }
            return body("gguf-vision", repo,
                        [("model", model.value), ("mmproj", mmproj.value)])
        case let .mlx(m):
            guard case let .huggingFace(repo, prefix) = m.source else { return nil }
            return body("mlx", repo, prefix.map { [("prefix", $0.value)] } ?? [])
        case .appleFoundationText:
            return nil
        }
    }

    /// `<scheme>://repo=<slug>&<pairs>[&rev=<revision>]`, with the keys in the order the
    /// crate writes them: the repo, the scheme's own keys in their declared order, then
    /// the revision. Ordered pairs rather than a dictionary, so `model` precedes `mmproj`
    /// as `gguf_vision_to_uri` writes them — one model renders one string, and the same
    /// string either client renders it.
    private static func body(_ scheme: String, _ repo: HFRepo,
                             _ pairs: [(String, String)]) -> String {
        var parts = ["repo=\(repo.description)"]
        parts += pairs.map { "\($0.0)=\($0.1)" }
        if let revision = repo.revision { parts.append("rev=\(revision.value)") }
        return "\(scheme)://" + parts.joined(separator: "&")
    }

    /// Parse a compact URI into a `Model`. Throws `Failure.notAUri` when the string has no
    /// `://` at all, which is the caller's signal to try the JSON spelling.
    static func parse(_ raw: String) throws -> Model {
        guard let split = raw.range(of: "://") else { throw Failure.notAUri }
        let scheme = String(raw[raw.startIndex..<split.lowerBound])
        let body = String(raw[split.upperBound...])

        // Scheme before keys, as the crate does: it selects the variant by scheme and then
        // validates that variant's keys. Checking keys first let `banana://url=x` complain
        // about the key and never mention the scheme.
        switch scheme {
        case "gguf-text", "gguf-vision", "mlx": break
        case "torch", "openvino": throw Failure.unsupportedScheme(scheme)
        default: throw Failure.unknownScheme(scheme)
        }

        var pairs: [String: String] = [:]
        for pair in body.split(separator: "&") where !pair.isEmpty {
            guard let eq = pair.firstIndex(of: "=") else {
                throw Failure.malformedPair(String(pair))
            }
            let key = String(pair[pair.startIndex..<eq])
            guard knownKeys.contains(key) else { throw Failure.unknownKey(key) }
            guard pairs[key] == nil else { throw Failure.duplicateKey(key) }
            pairs[key] = String(pair[pair.index(after: eq)...])
        }
        // A URL source and a digest are both real crate spellings with no iOS
        // counterpart; naming them beats reporting a missing `repo`.
        if pairs["url"] != nil { throw Failure.notRepresentable("a `url=` source") }
        for digest in ["sha256", "model_sha256", "mmproj_sha256"] where pairs[digest] != nil {
            throw Failure.notRepresentable("a `\(digest)=` digest")
        }

        func repo() throws -> HFRepo {
            guard let slug = pairs["repo"] else { throw Failure.missingKey("repo") }
            var repo = try HFRepo.parse(slug)
            if let rev = pairs["rev"] { repo.revision = try HFRevision(rev) }
            return repo
        }
        /// Refuse any key outside this scheme's vocabulary — a key that is real elsewhere
        /// (`path` on an `mlx://`) is as wrong as one that exists nowhere.
        func allowOnly(_ allowed: Set<String>) throws {
            if let extra = pairs.keys.first(where: { !allowed.contains($0) }) {
                throw Failure.unknownKey(extra)
            }
        }

        switch scheme {
        case "gguf-text":
            try allowOnly(["repo", "path", "rev"])
            guard let path = pairs["path"] else { throw Failure.missingKey("path") }
            return .ggufText(.init(source: .huggingFace(
                repo: try repo(), path: try RepoSubpath(path), sha256: nil)))
        case "gguf-vision":
            try allowOnly(["repo", "model", "mmproj", "rev"])
            guard let model = pairs["model"] else { throw Failure.missingKey("model") }
            guard let mmproj = pairs["mmproj"] else { throw Failure.missingKey("mmproj") }
            return .ggufVision(.init(source: .huggingFace(
                repo: try repo(), model: try RepoSubpath(model), modelSha256: nil,
                mmproj: try RepoSubpath(mmproj), mmprojSha256: nil)))
        case "mlx":
            try allowOnly(["repo", "prefix", "rev"])
            let subpath = try pairs["prefix"].map { try RepoSubpath($0) }
            return .mlx(.init(source: .huggingFace(repo: try repo(), prefix: subpath)))
        default:
            // Unreachable: the scheme was validated above. Kept as a total switch rather
            // than a fatal error so a scheme added to the guard cannot silently fall here.
            throw Failure.unknownScheme(scheme)
        }
    }
}
