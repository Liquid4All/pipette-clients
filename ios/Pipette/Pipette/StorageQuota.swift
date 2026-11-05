import Foundation

// The crate's `storage_quota.rs`: the quota half of `ClientSettings`, read through the
// settings store as the CLI does.

nonisolated extension PipetteWorkspace {
    /// The crate's `resolve_storage_quota`, less the argument rung iOS has no flag for.
    ///
    /// A non-positive configured quota is refused as the crate refuses it. The crate fails
    /// the command; this cannot fail and every download reads it, so it logs an error and
    /// falls back to the default, with source `default` so `storage status` does not credit
    /// the overruled file.
    var storageQuota: StorageQuota {
        guard let configured = settings.get()?.storageQuotaBytes else {
            return StorageQuota(bytes: defaultStorageQuotaBytes, source: .default)
        }
        guard configured > 0 else {
            AppLog.storage.error("identity/settings.json sets storage_quota_bytes to "
                + "\(configured); a zero quota can hold no artifact, so the default applies")
            return StorageQuota(bytes: defaultStorageQuotaBytes, source: .default)
        }
        return StorageQuota(bytes: configured, source: .settings)
    }

    var storageQuotaBytes: Int64 { storageQuota.bytes }

    var defaultStorageQuotaBytes: Int64 {
        Self.defaultQuotaBytes(volumeAt: root)
    }

    /// The crate's `parse_quota_bytes` rule: a quota that holds nothing is a typo, not a
    /// configuration.
    func setStorageQuotaBytes(_ bytes: Int64) throws {
        guard bytes > 0 else { throw StorageQuotaError.notPositive }
        var configured = settings.get() ?? ClientSettings(storageQuotaBytes: bytes)
        configured.storageQuotaBytes = bytes
        try settings.put(configured)
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

/// The crate's `QuotaSource` without `Arg`: iOS has no `--storage-quota` rung. `rawValue`
/// is the word `storage status` prints.
nonisolated enum QuotaSource: String, Sendable, Equatable {
    case settings
    case `default`

    /// Names the thing to go and edit.
    var label: String {
        switch self {
        case .settings: return "identity/settings.json"
        case .default: return "built-in default"
        }
    }
}

nonisolated struct StorageQuota: Sendable, Equatable {
    let bytes: Int64
    let source: QuotaSource
}

nonisolated enum StorageQuotaError: LocalizedError, Equatable {
    case notPositive

    var errorDescription: String? {
        switch self {
        case .notPositive: return "storage quota must be greater than zero"
        }
    }
}
