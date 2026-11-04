import Foundation
import Testing

@testable import Pipette

/// The quota as `PipetteWorkspace` resolves it from `ClientSettings`. The file itself is the
/// settings store's (`RegistrationTests` covers the round trip); this suite is about the
/// default, the corrupt-file fallback, and the cap.
struct StorageQuotaTests {
    @Test func quotaRoundTripsThroughSettingsJson() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }

        try ws.setStorageQuotaBytes(42 << 30)

        #expect(ws.storageQuotaBytes == 42 << 30)
        let json = try #require(try JSONSerialization.jsonObject(
            with: Data(contentsOf: ws.registration.root.appendingPathComponent("settings.json"))
        ) as? [String: Any])
        #expect(json["storage_quota_bytes"] as? Int == 42 << 30)
    }

    @Test func absentSettingsFallBackToTheDefault() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }

        #expect(ws.settings.get() == nil)
        #expect(ws.storageQuotaBytes == PipetteWorkspace.defaultQuotaBytes(volumeAt: ws.root))
    }

    @Test func settingsAndRegistrationCoexist() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }

        try ws.registration.put(try registrationData())
        try ws.setStorageQuotaBytes(7 << 30)

        #expect(ws.registration.get()?.clientId.value == "client-1")
        #expect(ws.storageQuotaBytes == 7 << 30)
    }

    /// Rather than trapping: the quota is a preference, not a load-bearing record.
    @Test func corruptSettingsFallBackToTheDefault() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        _ = PipetteWorkspace.ensureDirectory(ws.registration.root)
        try Data("not json".utf8)
            .write(to: ws.registration.root.appendingPathComponent("settings.json"))

        #expect(ws.settings.get() == nil)
        #expect(ws.storageQuotaBytes == PipetteWorkspace.defaultQuotaBytes(volumeAt: ws.root))
    }

    /// `min(16 GiB, 25% of the volume)`: never above the flat cap, never zero.
    @Test func defaultQuotaCapsAtSixteenGiB() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let quota = PipetteWorkspace.defaultQuotaBytes(volumeAt: ws.root)

        #expect(quota > 0)
        #expect(quota <= 16 << 30)
    }

    /// The seam the Settings ladder reads, so a view never names the concrete store.
    @Test func defaultStorageQuotaBytesMatchesTheStaticDefault() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }

        #expect(ws.defaultStorageQuotaBytes
            == PipetteWorkspace.defaultQuotaBytes(volumeAt: ws.root))
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
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        try ws.setStorageQuotaBytes(4 << 30)
        #expect(ws.storageQuotaBytes == 4 << 30)

        try ws.setStorageQuotaBytes(ws.defaultStorageQuotaBytes)

        #expect(ws.storageQuotaBytes == PipetteWorkspace.defaultQuotaBytes(volumeAt: ws.root))
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

    /// A quota that holds nothing is a typo, not a configuration: the crate's
    /// `parse_quota_bytes` refuses zero, and so does the setter.
    @Test(arguments: [Int64(0), -1])
    func aQuotaThatIsNotPositiveIsRefused(bytes: Int64) throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        #expect(throws: StorageQuotaError.notPositive) { try ws.setStorageQuotaBytes(bytes) }
        #expect(ws.settings.get() == nil)
    }

    /// The crate fails the command; the property cannot fail, so the zero is refused by the
    /// default applying.
    @Test func aZeroQuotaInTheSettingsFileIsRefused() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        try ws.settings.put(ClientSettings(storageQuotaBytes: 0))
        #expect(ws.storageQuotaBytes == ws.defaultStorageQuotaBytes)
    }
}
