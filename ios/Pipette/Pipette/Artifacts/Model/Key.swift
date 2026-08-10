import CryptoKit
import Foundation

/// Flat, filesystem-safe storage identity for a `Model` — the Swift mirror of the
/// crate's `ModelStorageKey` (`pipette-artifacts`). The store addresses a model by a
/// single directory name rather than a nested `<org>/<repo>/<file>` tree: the spec's
/// identity segments, each normalized to `[A-Za-z0-9._-]` and joined with `__`.
///
/// Same segments, same sanitize, same length cap as the crate, so a key computed on
/// either client for the same coordinate is byte-identical. Flags never enter it —
/// they are not on a `Model` at all, they belong to the cell.
nonisolated struct ModelStorageKey: Hashable, Sendable, CustomStringConvertible {
    let value: String

    var description: String { value }

    /// Maximum key length; a longer slug folds to `<head>_<hash8>`.
    private static let maxLength = 32
    /// Hex chars of SHA-256 kept as the disambiguating tail when folding.
    private static let hashLength = 8

    /// The key for a model spec, or `nil` for one the store cannot address — the crate's
    /// `of`, whose `NotStorable` error this returns as `nil`.
    ///
    /// A store-relative arm is `nil` because it is *already* inside a store and names no
    /// coordinate to key on. AFM is `nil` because it ships with the OS. Every other arm
    /// keys on what identifies it: the repo plus the file(s), the URL, or the host path.
    static func of(_ model: Model) -> ModelStorageKey? {
        let segments: [String]
        switch model {
        case let .ggufText(m):
            switch m.source {
            case let .huggingFace(repo, path, _): segments = repoSegments(repo) + [path.value]
            case let .url(url, _): segments = [url.value]
            case let .absoluteFile(path): segments = [path.value]
            case .relativeFile: return nil
            }
        case let .ggufVision(m):
            // Both files identify a VL instance: two can share a repo and differ only
            // in the weights or the projector.
            switch m.source {
            case let .huggingFace(repo, model, _, mmproj, _):
                segments = repoSegments(repo) + [model.value, mmproj.value]
            case let .url(model, _, mmproj, _): segments = [model.value, mmproj.value]
            case let .absoluteFiles(model, mmproj): segments = [model.value, mmproj.value]
            case .relativeFiles: return nil
            }
        case let .mlx(m):
            switch m.source {
            case let .huggingFace(repo, prefix):
                segments = repoSegments(repo) + (prefix.map { [$0.value] } ?? [])
            case let .absoluteDir(dir): segments = [dir.value]
            case .relativeDir: return nil
            }
        case .appleFoundationText:
            return nil
        }
        return ModelStorageKey(value: bound(slug(from: segments)))
    }

    /// `org`, `repo_name`, and the revision when pinned — the crate's `repo_segments`.
    /// The pin has to be a segment: without it two revisions of one repo share an entry
    /// directory, and fetching the second overwrites the first's weights.
    private static func repoSegments(_ repo: HFRepo) -> [String] {
        [repo.org.value, repo.repoName.value] + (repo.revision.map { [$0.value] } ?? [])
    }

    /// Normalize each segment and join with `__`. The wider `__` join reads as the
    /// segment boundary against the single `_` a sanitized run collapses to.
    private static func slug(from segments: [String]) -> String {
        segments.map(sanitize).filter { !$0.isEmpty }.joined(separator: "__")
    }

    /// Split on unsafe characters and rejoin the non-empty runs with `_`, collapsing
    /// every run of unsafe characters (leading and trailing ones included) to one `_`.
    private static func sanitize(_ segment: String) -> String {
        segment.split(whereSeparator: { !isSafe($0) }).joined(separator: "_")
    }

    private static func isSafe(_ c: Character) -> Bool {
        (c.isASCII && (c.isLetter || c.isNumber)) || c == "." || c == "-" || c == "_"
    }

    /// Cap the slug: a longer one keeps its head and gains an 8-hex SHA-256 tail of
    /// the full slug, so distinct long models stay distinct. The slug is ASCII, so
    /// taking the head by character count is byte-safe.
    private static func bound(_ slug: String) -> String {
        guard slug.count > maxLength else { return slug }
        let hash = SHA256.hash(data: Data(slug.utf8))
            .map { String(format: "%02x", $0) }.joined().prefix(hashLength)
        var head = String(slug.prefix(maxLength - hashLength - 1))
        while head.hasSuffix("_") { head.removeLast() }
        return "\(head)_\(hash)"
    }
}
