import Foundation
import Testing

@testable import Pipette

/// The quota as `FileStorage` resolves it from `ClientSettings`. The file itself is the
/// identity store's (`IdentityStoreTests` covers the round trip); this suite is about the
/// default, the corrupt-file fallback, and the cap.
/// Each test injects its own temporary `FileStorage`, so the suite carries no shared
/// global and runs in parallel.
struct StorageSettingsTests {
    @Test func quotaRoundTripsThroughSettingsJson() throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }

        try storage.setStorageQuotaBytes(42 << 30)

        #expect(storage.storageQuotaBytes == 42 << 30)
        let json = try #require(try JSONSerialization.jsonObject(
            with: Data(contentsOf: storage.identity.root.appendingPathComponent("settings.json"))
        ) as? [String: Any])
        #expect(json["storage_quota_bytes"] as? Int == 42 << 30)
    }

    @Test func absentSettingsFallBackToTheDefault() throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }

        #expect(storage.identity.getSettings() == nil)
        #expect(storage.storageQuotaBytes == FileStorage.defaultQuotaBytes(volumeAt: storage.dataRoot))
    }

    @Test func settingsAndRegistrationCoexist() throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }

        try storage.identity.putRegistration(registrationData())
        try storage.setStorageQuotaBytes(7 << 30)

        #expect(storage.identity.getRegistration()?.clientId.value == "client-1")
        #expect(storage.storageQuotaBytes == 7 << 30)
    }

    /// A corrupt settings file degrades to the default rather than trapping — the
    /// quota is a preference, not a load-bearing record.
    @Test func corruptSettingsFallBackToTheDefault() throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }
        _ = FileStorage.ensureDirectory(storage.identity.root)
        try Data("not json".utf8)
            .write(to: storage.identity.root.appendingPathComponent("settings.json"))

        #expect(storage.identity.getSettings() == nil)
        #expect(storage.storageQuotaBytes == FileStorage.defaultQuotaBytes(volumeAt: storage.dataRoot))
    }

    /// `min(16 GiB, 25% of the volume)`: never above the flat cap, never zero.
    @Test func defaultQuotaCapsAtSixteenGiB() throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }
        let quota = FileStorage.defaultQuotaBytes(volumeAt: storage.dataRoot)

        #expect(quota > 0)
        #expect(quota <= 16 << 30)
    }

    /// The seam the Settings ladder reads, so a view never names the concrete store.
    @Test func defaultStorageQuotaBytesMatchesTheStaticDefault() throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }

        #expect(storage.defaultStorageQuotaBytes
            == FileStorage.defaultQuotaBytes(volumeAt: storage.dataRoot))
    }

    @Test func theLadderLeadsWithTheDefault() throws {
        let first = try #require(StorageLimitOption.all(default: 5 << 30).first)

        #expect(first.isDefault)
        #expect(first.bytes == 5 << 30)
        #expect(first.title.contains("Default"))
    }

    /// Two rows resolving to the same byte count would both take the picker's checkmark.
    @Test func aDefaultMatchingAPresetAppearsOnce() {
        let options = StorageLimitOption.all(default: 16 << 30)

        #expect(options.filter { $0.bytes == 16 << 30 }.count == 1)
        #expect(options.count == StorageLimitOption.ladderBytes.count)
    }

    @Test func aDefaultOutsideTheLadderKeepsEveryPreset() {
        let options = StorageLimitOption.all(default: 5 << 30)

        #expect(options.count == StorageLimitOption.ladderBytes.count + 1)
        for preset in StorageLimitOption.ladderBytes {
            #expect(options.contains { $0.bytes == preset })
        }
        let presets = options.dropFirst().map(\.bytes)
        #expect(presets == presets.sorted())
    }

    /// Rule 4 of `docs/storage-quota.md`: a user who changed the limit can get back to
    /// the capacity-derived default.
    @Test func choosingTheDefaultRestoresTheComputedLimit() throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }
        try storage.setStorageQuotaBytes(4 << 30)
        #expect(storage.storageQuotaBytes == 4 << 30)

        try storage.setStorageQuotaBytes(storage.defaultStorageQuotaBytes)

        #expect(storage.storageQuotaBytes == FileStorage.defaultQuotaBytes(volumeAt: storage.dataRoot))
    }

    @Test func aStoreWithinItsLimitIsNotDisclosedAsOver() {
        #expect(StorageLimitOption.overLimitMessage(usedBytes: 4, limitBytes: 8) == nil)
        #expect(StorageLimitOption.overLimitMessage(usedBytes: 8, limitBytes: 8) == nil)
    }

    @Test func anOverLimitStoreIsDisclosedWithItsOverage() throws {
        let message = try #require(StorageLimitOption.overLimitMessage(
            usedBytes: 12_000_000_000, limitBytes: 8_000_000_000))

        #expect(message.contains(ByteFormat.storageLimit(4_000_000_000)))
        #expect(message.contains("Free up space"))
    }

    /// The overage is stated in the same units as the card's two numbers, so the three
    /// agree. Mixed bases put "Over the limit by 20.1 MB" beside "17.2 GB / 16 GB".
    @Test func theOverageUsesTheSameUnitsAsTheLimit() throws {
        let limit = Int64(16) << 30
        let message = try #require(StorageLimitOption.overLimitMessage(
            usedBytes: limit + (Int64(2) << 30), limitBytes: limit))

        #expect(message.contains(ByteFormat.storageLimit(Int64(2) << 30)))
        #expect(!message.contains(ByteFormat.fileSize(Int64(2) << 30)))
    }

    /// The sweep warns and continues when everything left is pinned, so the notice must
    /// not promise a fit it cannot deliver.
    @Test func theNoticePromisesReclaimNotAFit() throws {
        let message = try #require(StorageLimitOption.overLimitMessage(
            usedBytes: 12, limitBytes: 8))

        #expect(!message.contains("until the store fits"))
    }
}
