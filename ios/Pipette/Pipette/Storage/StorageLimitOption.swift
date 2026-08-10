import Foundation

/// One entry in the Settings model-storage limit ladder, plus the over-limit wording
/// the storage card shows. Presets rather than a byte field: a typo in a free-form
/// byte count would strand every installed model behind an unreachable cap.
nonisolated struct StorageLimitOption: Identifiable, Hashable, Sendable {
    let bytes: Int64
    /// The capacity-derived built-in, which is always offered so a user who changed
    /// the limit can get back to it (`docs/storage-quota.md`, rule 4).
    let isDefault: Bool

    var id: Int64 { bytes }

    var title: String {
        isDefault ? "Default (\(ByteFormat.storageLimit(bytes)))" : ByteFormat.storageLimit(bytes)
    }

    /// Starts at 8 GiB: a smaller cap cannot hold a single quantized model of the size
    /// these benchmarks run, so offering it would only strand downloads. A device whose
    /// capacity puts the computed default below the ladder still gets that default.
    static let ladderBytes: [Int64] = [8, 16, 32, 64, 128].map { Int64($0) << 30 }

    /// The computed default leads, then the ladder. A preset equal to the default is
    /// dropped: two rows resolving to the same byte count would both take the picker's
    /// checkmark. Presets are not filtered against volume capacity — the limit caps the
    /// store, not free disk, so a pick above capacity just means "effectively uncapped
    /// here" and the sweep is what meets reality.
    static func all(default defaultBytes: Int64) -> [StorageLimitOption] {
        [StorageLimitOption(bytes: defaultBytes, isDefault: true)]
            + ladderBytes
            .filter { $0 != defaultBytes }
            .map { StorageLimitOption(bytes: $0, isDefault: false) }
    }

    /// Non-nil only while the store is over its limit; the card shows it verbatim.
    /// Lowering the limit evicts nothing on its own, so an over-limit store is a state
    /// the user can sit in — it has to be disclosed, never silent.
    ///
    /// It promises eviction, not a fit: the sweep warns and continues when only pinned
    /// entries are left (`FileStorage.sweepToQuota`), so a store whose every model is
    /// held by a running job stays over and the notice stays up.
    ///
    /// `storageLimit` for the overage, not `fileSize`: the card renders the limit in
    /// binary units, and an overage in decimal ones would not be the difference between
    /// the two numbers beside it.
    static func overLimitMessage(usedBytes: Int64, limitBytes: Int64) -> String? {
        guard usedBytes > limitBytes else { return nil }
        return "Over the limit by \(ByteFormat.storageLimit(usedBytes - limitBytes)). "
            + "The next download (or Free up space) removes least-recently-used "
            + "models to reclaim space."
    }
}
