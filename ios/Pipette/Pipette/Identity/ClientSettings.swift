import Foundation

/// Client-wide settings stored in `metadata/settings.json` — the crate's
/// `ClientSettings` (`pipette-cli/src/identity/types.rs:45`), which keeps the same
/// file under `identity/`.
///
/// Neither registration nor a `UserDefaults` preference: these belong to the data
/// root, beside `registration.json`. Snake-cased keys match the CLI's file.
nonisolated struct ClientSettings: Codable, Hashable, Sendable {
    /// Cap on the disk the model store may occupy. See `docs/storage-quota.md`.
    var storageQuotaBytes: Int64

    enum CodingKeys: String, CodingKey {
        case storageQuotaBytes = "storage_quota_bytes"
    }
}
