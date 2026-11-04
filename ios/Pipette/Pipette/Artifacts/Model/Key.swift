import CryptoKit
import Foundation

/// A model's flat store directory name: its identity segments, each normalized to
/// `[A-Za-z0-9._-]` and joined with `__`. Must stay byte-identical to the crate's
/// `ModelStorageKey` (same segments, sanitize and length cap).
nonisolated struct ModelStorageKey: Hashable, Sendable, CustomStringConvertible, Codable {
    let value: String

    var description: String { value }

    /// Relative to the `models/` root, as the crate's `relative_dir`.
    var relativeDir: String { value }

    /// Relative to the `models/` root, as the crate's `blobs_dir`.
    var blobsDir: String { "\(value)/\(Entry.blobsDirName)" }

    /// A longer slug folds to `<head>_<hash8>`.
    private static let maxLen = 32
    private static let hashLen = 8

    /// `nil` where the crate's `of` returns `NotStorable`: a store-relative arm is already
    /// inside a store, and AFM ships with the OS.
    static func of(_ model: Model) -> ModelStorageKey? {
        let segments: [String]
        switch model {
        case let .ggufText(m):
            switch m.source {
            case let .huggingFace(repo, path, _): segments = repoSegments(repo) + [path.value]
            case let .url(url, _): segments = [url.value]
            case let .vaultUrl(url, _, _, _): segments = vaultSegments(url)
            case let .absoluteFile(path): segments = [path.value]
            case .relativeFile: return nil
            }
        case let .ggufVision(m):
            // Two VL models can share a repo and differ only in weights or projector.
            switch m.source {
            case let .huggingFace(repo, model, _, mmproj, _):
                segments = repoSegments(repo) + [model.value, mmproj.value]
            case let .url(model, _, mmproj, _): segments = [model.value, mmproj.value]
            case let .vaultUrl(model, _, mmproj, _, _, _):
                segments = vaultSegments(model) + [mmproj.resourceId.value]
            case let .absoluteFiles(model, mmproj): segments = [model.value, mmproj.value]
            case .relativeFiles: return nil
            }
        case let .ggufVisionSpeculative(m):
            // `spec` keeps a merged speculative model apart from the `gguf_vision` over
            // the same files (one entry, one model type). A separate drafter is a segment
            // because rows differing only by it measure different things.
            let draft: String?
            switch m.source {
            case let .huggingFace(_, _, _, _, _, path, _): draft = path?.value
            case let .url(_, _, _, _, url, _): draft = url?.value
            // One `vault_<id>` segment, as the crate spells it.
            case let .vaultUrl(_, _, _, _, path, _, _, _):
                draft = path.map { vaultSegments($0).joined(separator: "_") }
            case let .absoluteFiles(_, _, path): draft = path?.value
            case .relativeFiles: return nil
            }
            guard let vision = of(.ggufVision(GgufVision(source: m.source.vision))) else {
                return nil
            }
            segments = ["spec", vision.value] + (draft.map { [$0] } ?? [])
        case let .mlx(m):
            switch m.source {
            case let .huggingFace(repo, prefix):
                segments = repoSegments(repo) + (prefix.map { [$0.value] } ?? [])
            case let .vaultArchive(url, _, _, _): segments = vaultSegments(url)
            case let .absoluteDir(dir): segments = [dir.value]
            case .relativeDir: return nil
            }
        case let .onnxGenai(m):
            // An ONNX GenAI repo bundles several packages; the prefix tells them apart.
            switch m.source {
            case let .huggingFace(repo, prefix):
                segments = repoSegments(repo) + (prefix.map { [$0.value] } ?? [])
            case let .vaultArchive(url, _, _, _): segments = vaultSegments(url)
            case let .absoluteDir(dir): segments = [dir.value]
            case .relativeDir: return nil
            }
        case .appleFoundationText:
            return nil
        }
        return ModelStorageKey(value: bound(slugFrom(segments)))
    }

    /// The pin is a segment so two revisions of one repo never share an entry, where
    /// fetching the second would overwrite the first.
    private static func repoSegments(_ repo: HfRepo) -> [String] {
        [repo.org.value, repo.repoName.value] + (repo.revision.map { [$0.value] } ?? [])
    }

    /// The route around the id is the same for every resource, so only the id counts.
    private static func vaultSegments(_ path: VaultPath) -> [String] {
        ["vault", path.resourceId.value]
    }

    /// `__` marks a segment boundary, distinct from the single `_` sanitizing leaves.
    private static func slugFrom(_ segments: [String]) -> String {
        segments.map(sanitize).filter { !$0.isEmpty }.joined(separator: "__")
    }

    /// Each run of unsafe characters becomes one `_`; leading and trailing runs vanish.
    private static func sanitize(_ segment: String) -> String {
        segment.split(whereSeparator: { !isSafe($0) }).joined(separator: "_")
    }

    private static func isSafe(_ c: Character) -> Bool {
        (c.isASCII && (c.isLetter || c.isNumber)) || c == "." || c == "-" || c == "_"
    }

    /// The hash tail of the full slug keeps distinct long models distinct. The slug is
    /// ASCII, so a character-count prefix is byte-safe.
    private static func bound(_ slug: String) -> String {
        guard slug.count > maxLen else { return slug }
        let hash = SHA256.hash(data: Data(slug.utf8))
            .map { String(format: "%02x", $0) }.joined().prefix(hashLen)
        var head = String(slug.prefix(maxLen - hashLen - 1))
        while head.hasSuffix("_") { head.removeLast() }
        return "\(head)_\(hash)"
    }
}

// `Codable` is declared on the nonisolated type so the conformance stays nonisolated;
// the coding methods live here so the struct keeps its memberwise `init(value:)`, which
// an initializer in the body would suppress.
extension ModelStorageKey {
    /// A bare string on the wire, so the admission ledger stays legible.
    nonisolated init(from decoder: any Decoder) throws {
        value = try decoder.singleValueContainer().decode(String.self)
    }

    nonisolated func encode(to encoder: any Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(value)
    }
}
