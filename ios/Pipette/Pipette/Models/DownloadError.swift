import Foundation

/// Failures of the model download / install pipeline (MLX directory pulls and
/// GGUF fetches): the download itself, and the validate → relocate → record steps
/// that follow. Shared across the subsystem's types (`DownloadCoordinator`,
/// `HubMLXModelDownloader`, `MLXModelLayout`, `MLXModelInstaller`), so it's a
/// top-level domain error rather than nested under one owner. Transport/IO wrap a
/// message (not the underlying `Error`) so the enum stays `Equatable` for assertions.
nonisolated enum DownloadError: Error, Equatable {
    case emptyMatch(HFModelRef)
    case incompleteModel(HFModelRef, missing: [String])
    case transport(String)
    case io(String)
    case cancelled
    /// A single artifact bigger than the whole storage quota. Refused before the
    /// fetch: fetching it would evict the entire store and still not fit.
    case exceedsQuota(neededBytes: Int64, quotaBytes: Int64)
}

extension DownloadError: LocalizedError {
    var errorDescription: String? {
        switch self {
        case let .emptyMatch(ref): "no files matched for \(ref)"
        case let .incompleteModel(ref, missing): "\(ref) is not a complete MLX model (missing: \(missing.joined(separator: ", ")))"
        case let .transport(m): "download failed: \(m)"
        case let .io(m): "file error: \(m)"
        case .cancelled: "download cancelled"
        // Points at the Settings limit, not at "free up space": the limit is a cap on
        // the store, not on free disk, so deleting other models cannot make room for
        // this one. Raising the limit can.
        case let .exceedsQuota(neededBytes, quotaBytes):
            "this model needs \(ByteFormat.fileSize(neededBytes)), more than the "
                + "\(ByteFormat.storageLimit(quotaBytes)) model storage limit. Raise "
                + "the limit in Settings to download it"
        }
    }
}
