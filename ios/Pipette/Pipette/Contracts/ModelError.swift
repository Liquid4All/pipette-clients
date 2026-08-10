import Foundation

/// Model-coordinate construction/parse failures — the Swift analog of
/// `pipette_plan_types::Error` (the crate names it plainly `Error`; the `Model`
/// prefix is the namespacing the crate gets for free). Thrown when a repo slug,
/// org/name segment, subpath, GGUF filename, or model-type tag fails to validate —
/// none of which involve a download. Wraps the offending string (not an underlying
/// `Error`) so the enum stays `Equatable` for assertions.
nonisolated enum ModelError: Error, Equatable {
    case repoMissingSeparator(String)
    case invalidOrg(String)
    case invalidRepoName(String)
    case invalidSubpath(String)
    case invalidRelativePath(String)
    case invalidAbsolutePath(String)
    case invalidResourceUrl(String)
    case invalidRevision(String)
    case invalidSha256(String)
    case emptyValue(String)
    /// Deliberately payload-free, unlike its siblings: the rejected input *is* the
    /// token, and an error message is exactly the sort of place it must not surface.
    case invalidAuthToken
    case unknownModelType(String)
}

extension ModelError: LocalizedError {
    var errorDescription: String? {
        switch self {
        case let .repoMissingSeparator(s): "HF repo must be in the form `org/repo_name`: \(s)"
        case let .invalidOrg(s): "invalid HF org: \(s)"
        case let .invalidRepoName(s): "invalid HF repo name: \(s)"
        case let .invalidSubpath(s): "invalid repo subpath: \(s)"
        case let .invalidRelativePath(s): "path must be store-relative: \(s)"
        case let .invalidAbsolutePath(s): "path must be absolute: \(s)"
        case let .invalidResourceUrl(s): "not an http(s)/file URL: \(s)"
        case let .invalidRevision(s): "invalid HF revision: \(s)"
        case let .invalidSha256(s): "invalid sha256 digest: \(s)"
        case .emptyValue: "expected a non-empty value"
        // No interpolation: the rejected value is the token.
        case .invalidAuthToken: "the HF access token is empty"
        case let .unknownModelType(s): "unknown model type: \(s)"
        }
    }
}
