import Foundation

// The quota half of `ClientSettings`. The file itself is the identity store's — as it
// is the CLI's, where `settings.json` sits in `identity/` and `resolve_storage_quota`
// reads it through the store rather than owning a path of its own.

extension FileStorage {
    var storageQuotaBytes: Int64 {
        identity.getSettings()?.storageQuotaBytes ?? Self.defaultQuotaBytes(volumeAt: dataRoot)
    }

    var defaultStorageQuotaBytes: Int64 {
        Self.defaultQuotaBytes(volumeAt: dataRoot)
    }

    func setStorageQuotaBytes(_ bytes: Int64) throws {
        var settings = identity.getSettings() ?? ClientSettings(storageQuotaBytes: bytes)
        settings.storageQuotaBytes = bytes
        try identity.putSettings(settings)
    }

    /// `min(16 GiB, 25% of the volume)`. Read from the data root's volume so a store
    /// rooted in a temp directory answers the same way the app's container does.
    /// `.volumeTotalCapacityKey` needs the URL to exist, hence `ensureDirectory`.
    static func defaultQuotaBytes(volumeAt url: URL) -> Int64 {
        let flatCap: Int64 = 16 << 30
        guard let capacity = (try? ensureDirectory(url)
            .resourceValues(forKeys: [.volumeTotalCapacityKey]))?.volumeTotalCapacity
        else { return flatCap }
        return min(flatCap, Int64(capacity) / 4)
    }
}
