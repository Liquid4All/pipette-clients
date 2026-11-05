import Foundation

/// Client-wide settings in `identity/settings.json`, the same file as the crate's
/// `ClientSettings` (`crates/pipette-cli/src/identity/settings.rs`). Not `UserDefaults`:
/// they belong to the data root, beside `registration.json`.
nonisolated struct ClientSettings: Codable, Hashable, Sendable {
    /// Cap on the disk the model store may occupy. See `docs/storage-quota.md`.
    var storageQuotaBytes: Int64

    enum CodingKeys: String, CodingKey {
        case storageQuotaBytes = "storage_quota_bytes"
    }
}

/// `identity/settings.json`: the crate's `ClientSettingsStore`.
nonisolated struct ClientSettingsStore: Sendable {
    private static let settingsName = "settings.json"

    /// The workspace `identity/` directory.
    let root: URL

    /// Nil when absent or unreadable, so a corrupt file degrades to the defaults. The crate
    /// errors instead; here the only setting is a quota with a safe default, and refusing to
    /// launch over it would be worse.
    func get() -> ClientSettings? {
        try? IdentityFiles.readJson(root, Self.settingsName)
    }

    func put(_ settings: ClientSettings) throws {
        try IdentityFiles.writeJson(root, Self.settingsName, settings)
    }

    func clear() throws {
        try IdentityFiles.remove(root, Self.settingsName)
    }
}
