import Foundation

/// Presets rather than a byte field: a typo in a free-form count would strand every
/// installed model behind an unreachable cap.
nonisolated struct StorageLimitOption: Identifiable, Hashable, Sendable {
    let bytes: Int64
    /// Always offered, so a user who changed the limit can get back to it
    /// (`docs/storage-quota.md`, section 4).
    let isDefault: Bool

    var id: Int64 { bytes }

    var title: String {
        isDefault ? "Default (\(ByteFormat.storageLimit(bytes)))" : ByteFormat.storageLimit(bytes)
    }

    /// Starts at 8 GiB: a smaller cap can't hold one quantized model of the benchmarked
    /// size. A device whose computed default is below it still gets that default.
    static let ladderBytes: [Int64] = [8, 16, 32, 64, 128].map { Int64($0) << 30 }

    /// A preset equal to the default is dropped, or both rows would take the checkmark.
    /// Not filtered against capacity: the limit caps the store, not free disk.
    static func all(default defaultBytes: Int64) -> [StorageLimitOption] {
        [StorageLimitOption(bytes: defaultBytes, isDefault: true)]
            + ladderBytes
            .filter { $0 != defaultBytes }
            .map { StorageLimitOption(bytes: $0, isDefault: false) }
    }

    /// Lowering the limit evicts nothing, so an over-limit store must be disclosed. It
    /// promises eviction, not a fit: pinned entries survive `PipetteWorkspace.sweepToQuota`.
    /// Binary units, matching the limit it is the difference from.
    static func overLimitMessage(usedBytes: Int64, limitBytes: Int64) -> String? {
        guard usedBytes > limitBytes else { return nil }
        return "Over the limit by \(ByteFormat.storageLimit(usedBytes - limitBytes)). "
            + "The next download (or Free up space) removes least-recently-used "
            + "models to reclaim space."
    }
}
