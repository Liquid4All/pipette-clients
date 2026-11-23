import Foundation

/// Written when a GGUF transfer starts, so its row survives the system relaunching the
/// app.
struct DownloadRecord: Codable {
    let filename: String
    let urlString: String
    let repo: String?
    /// Keeps a restored row grouped under its model in the Models tab.
    var modelId: String?
    /// So a completion redelivered after relaunch still writes an exact manifest. Nil for
    /// sideloads and old records, whose provenance falls back to strings.
    var source: Model?
    /// See `DownloadCoordinator.Download.sha256`; persisted so a redelivered completion
    /// verifies the same digest.
    var sha256: Sha256?
    /// See `DownloadCoordinator.Download.resumeETag`; persisted so a resume after relaunch
    /// is checked the same way.
    var etag: String?
    /// See `DownloadCoordinator.Download.resumeProxy`.
    var proxy: String?
    /// Nil for records written under the original repo/filename key.
    var keyOverride: String?

    var key: String {
        keyOverride ?? LocalStorage.modelRelativePath(repo: repo, filename: filename)
    }

    /// The stored key of `modelId` is `familyId`, the key that records on disk use.
    enum CodingKeys: String, CodingKey {
        case filename, urlString, repo, source, sha256, etag, proxy, keyOverride
        case modelId = "familyId"
    }
}
