import Foundation

/// Written when a GGUF transfer starts, so its row survives the system relaunching the
/// app.
struct DownloadRecord: Codable {
    let filename: String
    let urlString: String
    let repo: String?
    /// Keeps a restored row grouped under its family in the Models tab.
    var familyId: String?
    /// So a completion redelivered after relaunch still writes an exact manifest. Nil for
    /// sideloads and old records, whose provenance falls back to strings.
    var source: Model?
    /// See `DownloadCoordinator.Download.sha256`; persisted so a redelivered completion
    /// verifies the same digest.
    var sha256: Sha256?
    /// See `DownloadCoordinator.Download.resumeETag`; persisted so a resume after relaunch
    /// is checked the same way.
    var etag: String?
}
