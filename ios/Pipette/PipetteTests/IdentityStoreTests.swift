import Foundation
import Testing

@testable import Pipette

/// The identity store's registration half and the `registration.json` wire form.
///
/// The key half (`getPrivateKey`/`deletePrivateKey`/`signingIdentity`) is not covered
/// here: a Simulator test host cannot store Keychain items, so every one of those would
/// assert against an unconditional nil. `HeadlessAuthResetTests` covers the reset path
/// through the injected `deleteKey` seam instead.
///
/// Each test injects its own temporary `FileStorage`, so the suite carries no shared
/// global and runs in parallel.
struct IdentityStoreTests {

    @Test func registrationRoundTripsThroughTheStore() throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }

        try storage.identity.putRegistration(registrationData())

        let loaded = try #require(storage.identity.getRegistration())
        #expect(loaded == registrationData())
        #expect(storage.identity.isRegistered)

        storage.identity.deleteRegistration()
        #expect(storage.identity.getRegistration() == nil)
        #expect(!storage.identity.isRegistered)
    }

    /// Signing out un-pins the device from the account without disturbing the identity it
    /// uploads under. The whole point of unlinking rather than clearing the registration:
    /// deleting it would take the `clientId` and the signing key too, orphaning every
    /// result already submitted.
    @Test func unlinkingClerkKeepsTheUploadIdentity() throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }

        try storage.identity.putRegistration(registrationData().withoutClerkLink())

        let loaded = try #require(storage.identity.getRegistration())
        #expect(loaded.clerkUserId == nil)
        #expect(loaded.clerkSessionId == nil)
        #expect(loaded.clerkPrimaryEmail == nil)
        // Dates the current link, so it cannot outlive it: `withClerkLink` preserves an
        // existing value, and the next account's link would inherit this one's timestamp.
        #expect(loaded.clerkLinkedAt == nil)
        // Everything the server issued survives. Asserted field by field on purpose:
        // `withoutClerkLink` re-lists the fields to keep, so a field added to
        // `IdentityRegistration` later is dropped silently unless a test names it.
        #expect(loaded.clientId == registrationData().clientId)
        #expect(loaded.serverUrl == registrationData().serverUrl)
        #expect(loaded.organization == "Example")
        #expect(loaded.registeredAt == "2026-05-28T16:41:00Z")
        #expect(loaded.status == registrationData().status)
        #expect(loaded.contactEmail == registrationData().contactEmail)
        #expect(storage.identity.isRegistered)
    }

    /// An unlinked record then re-links to whichever account signs in next, which is what
    /// keeps a second user off the mismatch screen.
    @Test func anUnlinkedRegistrationRelinksToADifferentAccount() throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }

        let relinked = registrationData()
            .withoutClerkLink()
            .withClerkLink(userId: "user_2", sessionId: "session_2", primaryEmail: "second@example.com")
        try storage.identity.putRegistration(relinked)

        let loaded = try #require(storage.identity.getRegistration())
        #expect(loaded.clerkUserId == "user_2")
        #expect(loaded.clerkPrimaryEmail == "second@example.com")
        // Freshly stamped rather than carried over from the first account's link.
        #expect(loaded.clerkLinkedAt != nil)
        #expect(loaded.clerkLinkedAt != "2026-05-28T16:42:00Z")
    }

    /// The file the CLI would read: `snake_case`, as `identity/registration.json` is
    /// there and as `settings.json` already is here.
    @Test func registrationWritesSnakeCaseKeys() throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }

        try storage.identity.putRegistration(registrationData())

        let data = try Data(contentsOf: storage.identity.root.appendingPathComponent("registration.json"))
        let json = try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any])
        #expect(json["client_id"] as? String == "client-1")
        #expect(json["server_url"] as? String == "https://collector.example.com")
        #expect(json["contact_email"] as? String == "user@example.com")
        #expect(json["registered_at"] as? String == "2026-05-28T16:41:00Z")
        #expect(json["clerk_user_id"] as? String == "user_1")
        // The old spelling is gone, not dual-written.
        #expect(json["clientId"] == nil)
        #expect(json["serverUrl"] == nil)
    }

    /// A device that registered before the rename keeps its registration. Without this
    /// the record reads as absent and the app drops the user back to setup — and
    /// re-registering mints a new keypair, orphaning every result already submitted.
    @Test func aCamelCaseRegistrationStillDecodes() throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }

        let legacy = """
        {
          "clientId": "client-legacy",
          "status": "approved",
          "serverUrl": "https://collector.example.com",
          "organization": "Example",
          "contactEmail": "user@example.com",
          "registeredAt": "2026-05-28T16:41:00Z",
          "clerkUserId": "user_1",
          "clerkLinkedAt": "2026-05-28T16:42:00Z"
        }
        """
        _ = FileStorage.ensureDirectory(storage.identity.root)
        try Data(legacy.utf8)
            .write(to: storage.identity.root.appendingPathComponent("registration.json"))

        let loaded = try #require(storage.identity.getRegistration())
        #expect(loaded.clientId.value == "client-legacy")
        #expect(loaded.serverUrl.value == "https://collector.example.com")
        #expect(loaded.contactEmail == "user@example.com")
        #expect(loaded.registeredAt == "2026-05-28T16:41:00Z")
        #expect(loaded.clerkUserId == "user_1")
        #expect(loaded.clerkSessionId == nil)
    }

    /// Clerk linkage after the rename rewrites the file in the new spelling, so a record
    /// never ends up half-and-half — and the per-key fallback means it would still read
    /// correctly if it did.
    @Test func aMixedSpellingRegistrationDecodesBothHalves() throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }

        let mixed = """
        {
          "clientId": "client-mixed",
          "status": "approved",
          "server_url": "https://collector.example.com",
          "organization": "Example",
          "contact_email": "user@example.com",
          "registeredAt": "2026-05-28T16:41:00Z"
        }
        """
        _ = FileStorage.ensureDirectory(storage.identity.root)
        try Data(mixed.utf8)
            .write(to: storage.identity.root.appendingPathComponent("registration.json"))

        let loaded = try #require(storage.identity.getRegistration())
        #expect(loaded.clientId.value == "client-mixed")
        #expect(loaded.serverUrl.value == "https://collector.example.com")
        #expect(loaded.contactEmail == "user@example.com")
        #expect(loaded.registeredAt == "2026-05-28T16:41:00Z")
    }

    /// A registration missing the identity fields is not a registration. Reported as
    /// "unregistered" rather than trapping, matching how an unreadable file is treated.
    @Test func aRegistrationWithoutIdentityFieldsReadsAsAbsent() throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }

        _ = FileStorage.ensureDirectory(storage.identity.root)
        try Data(#"{"status":"approved","organization":"Example"}"#.utf8)
            .write(to: storage.identity.root.appendingPathComponent("registration.json"))

        #expect(storage.identity.getRegistration() == nil)
    }

    /// Only the upload identity is required, as in the crate's record. The rest are
    /// convenience copies of server-owned state, so a record missing one degrades that
    /// value rather than reading as unregistered — which would push a registered device
    /// back through setup and mint a keypair that orphans its submitted results.
    @Test func aRegistrationMissingServerOwnedFieldsStillDecodes() throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }

        _ = FileStorage.ensureDirectory(storage.identity.root)
        try Data(#"{"client_id":"client-1","server_url":"https://c.example.com"}"#.utf8)
            .write(to: storage.identity.root.appendingPathComponent("registration.json"))

        let loaded = try #require(storage.identity.getRegistration())
        #expect(loaded.clientId.value == "client-1")
        #expect(loaded.status.isEmpty)
        #expect(loaded.organization.isEmpty)
        #expect(storage.identity.isRegistered)
    }

    /// `mgmtSession` names which half is missing — `auth me` reports that distinction,
    /// and a single nil would collapse the two causes into one message.
    @Test func mgmtSessionNamesTheMissingHalf() throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }

        #expect(throws: IdentityError.registrationMissing) { try storage.mgmtSession() }

        // With a registration but no signing key, the other half is named instead. The
        // temporary store supplies no key on any host, so this holds on a registered
        // device as much as on the Simulator.
        try storage.identity.putRegistration(registrationData())
        #expect(throws: IdentityError.privateKeyMissing) { try storage.mgmtSession() }
    }

    /// The other side of the same call, testable for the first time now that a store owns
    /// its key source — with the Keychain there was no way to put a key under a temporary
    /// store, so only the two failures were ever pinned.
    @Test func mgmtSessionCarriesBothHalves() throws {
        var storage = makeTemporaryStorage()
        storage.privateKeySource = { PrivateKeyHex("deadbeef") }
        defer { removeStorage(storage) }
        try storage.identity.putRegistration(registrationData())

        let session = try storage.mgmtSession()
        #expect(session.registration.clientId == registrationData().clientId)
        #expect(session.auth.clientId == registrationData().clientId)
        #expect(session.auth.privateKeyHex == PrivateKeyHex("deadbeef"))
    }

    @Test func settingsRoundTripThroughTheStore() throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }

        #expect(storage.identity.getSettings() == nil)
        try storage.identity.putSettings(ClientSettings(storageQuotaBytes: 9 << 30))
        #expect(storage.identity.getSettings()?.storageQuotaBytes == 9 << 30)
    }

    /// Registration and settings are separate files under one root, as they are in the
    /// crate's `identity/` — writing one must not disturb the other.
    @Test func registrationAndSettingsAreIndependentFiles() throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }

        try storage.identity.putRegistration(registrationData())
        try storage.identity.putSettings(ClientSettings(storageQuotaBytes: 3 << 30))

        storage.identity.deleteRegistration()
        #expect(storage.identity.getSettings()?.storageQuotaBytes == 3 << 30)
    }

    // MARK: - The metadata/ -> identity/ move

    /// A device that registered before the directory moved keeps its registration. This
    /// is the migration's whole job: losing the file here pushes a registered device
    /// back through setup, minting a keypair that orphans its submitted results.
    @Test func theMigrationCarriesRegistrationAndSettingsAcross() throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }
        let legacy = FileStorage.ensureDirectory(storage.legacyMetadataDir)
        try Coding.encoder.encode(registrationData())
            .write(to: legacy.appendingPathComponent("registration.json"))
        try Coding.encoder.encode(ClientSettings(storageQuotaBytes: 7 << 30))
            .write(to: legacy.appendingPathComponent("settings.json"))

        #expect(Set(storage.migrateIdentityDirectory()) == ["registration.json", "settings.json"])

        #expect(storage.identity.getRegistration() == registrationData())
        #expect(storage.identity.getSettings()?.storageQuotaBytes == 7 << 30)
        // The old directory goes once it is empty.
        #expect(!FileManager.default.fileExists(atPath: storage.legacyMetadataDir.path))
    }

    /// Idempotent: the second launch has nothing to move and must not disturb what the
    /// first one wrote.
    @Test func theMigrationIsIdempotent() throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }
        let legacy = FileStorage.ensureDirectory(storage.legacyMetadataDir)
        try Coding.encoder.encode(registrationData())
            .write(to: legacy.appendingPathComponent("registration.json"))

        #expect(storage.migrateIdentityDirectory() == ["registration.json"])
        #expect(storage.migrateIdentityDirectory().isEmpty)
        #expect(storage.identity.getRegistration() == registrationData())
    }

    /// A file already at the destination wins — it is the newer one, written by a build
    /// that had already moved. The stale copy is dropped rather than overwriting it.
    @Test func anAlreadyMovedFileIsNotOverwritten() throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }
        try storage.identity.putRegistration(registrationData())

        let legacy = FileStorage.ensureDirectory(storage.legacyMetadataDir)
        try Data(#"{"client_id":"stale","server_url":"https://old.example.com"}"#.utf8)
            .write(to: legacy.appendingPathComponent("registration.json"))

        #expect(storage.migrateIdentityDirectory().isEmpty)
        #expect(storage.identity.getRegistration()?.clientId.value == "client-1")
    }

    /// An unexpected file in `metadata/` is someone else's, so the directory stays.
    @Test func anUnknownLeftoverKeepsTheOldDirectory() throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }
        let legacy = FileStorage.ensureDirectory(storage.legacyMetadataDir)
        try Data("{}".utf8).write(to: legacy.appendingPathComponent("something-else.json"))

        storage.migrateIdentityDirectory()

        #expect(FileManager.default.fileExists(atPath: storage.legacyMetadataDir.path))
    }

    // MARK: - Surviving a reinstall

    /// Deleting the data root is what an uninstall does to the container; the Keychain half
    /// is what it cannot touch, so the identity comes back.
    @Test func aReinstallRecoversTheRegistrationFromTheMirror() throws {
        let mirror = FakeRegistrationMirror()
        var storage = makeTemporaryStorage()
        storage.registrationMirror = mirror.mirror
        defer { removeStorage(storage) }
        try storage.identity.putRegistration(registrationData())

        // The uninstall: the container goes, the Keychain stays.
        try FileManager.default.removeItem(at: storage.dataRoot)

        #expect(storage.identity.getRegistration() == registrationData())
        #expect(storage.identity.isRegistered)
    }

    /// The restore rewrites the file, so every other reader takes the ordinary path and the
    /// mirror is consulted once rather than on every call.
    @Test func aRestoredRegistrationIsWrittenBackToTheFile() throws {
        let mirror = FakeRegistrationMirror()
        var storage = makeTemporaryStorage()
        storage.registrationMirror = mirror.mirror
        defer { removeStorage(storage) }
        try storage.identity.putRegistration(registrationData())
        try FileManager.default.removeItem(at: storage.dataRoot)

        _ = storage.identity.getRegistration()

        let file = storage.identity.root.appendingPathComponent("registration.json")
        #expect(FileManager.default.fileExists(atPath: file.path))
        let reloaded = try Coding.decoder.decode(
            IdentityRegistration.self, from: Data(contentsOf: file))
        #expect(reloaded == registrationData())
    }

    /// "Forget this device" has to reach both copies. Leaving the mirror behind would let
    /// the next read restore exactly what `auth reset` just discarded.
    @Test func deletingTheRegistrationClearsTheMirrorToo() throws {
        let mirror = FakeRegistrationMirror()
        var storage = makeTemporaryStorage()
        storage.registrationMirror = mirror.mirror
        defer { removeStorage(storage) }
        try storage.identity.putRegistration(registrationData())

        storage.identity.deleteRegistration()

        #expect(!storage.identity.isRegistrationMirrored)
        #expect(storage.identity.getRegistration() == nil)
    }

    /// The level an operator actually invokes: `auth reset` goes through here, and a mirror
    /// left behind would let the next read restore the identity it just discarded.
    @Test func clearingRegistrationMaterialClearsTheMirror() throws {
        let mirror = FakeRegistrationMirror()
        var storage = makeTemporaryStorage()
        storage.registrationMirror = mirror.mirror
        defer { removeStorage(storage) }
        try storage.identity.putRegistration(registrationData())

        #expect(storage.identity.clearRegistrationMaterial(deleteKey: { true }))

        #expect(!storage.identity.isRegistrationMirrored)
        #expect(storage.identity.getRegistration() == nil)
    }

    /// A device registered before the mirror existed has a file and no copy. Without the
    /// backfill it would gain the protection only by re-registering, which for an
    /// already-registered device never happens.
    @Test func theBackfillMirrorsAnExistingRegistration() throws {
        let mirror = FakeRegistrationMirror()
        var storage = makeTemporaryStorage()
        defer { removeStorage(storage) }
        // Registered with no mirror at all, as an older build would have.
        try storage.identity.putRegistration(registrationData())
        storage.registrationMirror = mirror.mirror
        #expect(!storage.identity.isRegistrationMirrored)

        storage.identity.backfillRegistrationMirror()

        #expect(storage.identity.isRegistrationMirrored)
        try FileManager.default.removeItem(at: storage.dataRoot)
        #expect(storage.identity.getRegistration() == registrationData())
    }

    /// Idempotent, and it must not overwrite a mirror that is already current — the file is
    /// the authority, but a launch is not a write.
    @Test func theBackfillIsANoOpWhenAlreadyMirrored() throws {
        let mirror = FakeRegistrationMirror()
        var storage = makeTemporaryStorage()
        storage.registrationMirror = mirror.mirror
        defer { removeStorage(storage) }
        try storage.identity.putRegistration(registrationData())
        let writesAfterRegistration = mirror.saveCount

        storage.identity.backfillRegistrationMirror()
        storage.identity.backfillRegistrationMirror()

        #expect(mirror.saveCount == writesAfterRegistration)
    }

    /// An unregistered device has nothing to mirror, and must not be made to look like one
    /// that does.
    @Test func theBackfillDoesNothingWithoutARegistration() {
        let mirror = FakeRegistrationMirror()
        var storage = makeTemporaryStorage()
        storage.registrationMirror = mirror.mirror
        defer { removeStorage(storage) }

        storage.identity.backfillRegistrationMirror()

        #expect(!storage.identity.isRegistrationMirrored)
        #expect(storage.identity.getRegistration() == nil)
    }

    /// A failed mirror write must not fail the registration — the device is registered
    /// either way.
    @Test func aFailedMirrorWriteStillRegistersTheDevice() throws {
        var storage = makeTemporaryStorage()
        storage.registrationMirror = RegistrationMirror(
            load: { nil }, save: { _ in false }, clear: { false })
        defer { removeStorage(storage) }

        try storage.identity.putRegistration(registrationData())

        #expect(storage.identity.getRegistration() == registrationData())
        #expect(!storage.identity.isRegistrationMirrored)
    }
}

/// An in-memory stand-in for the Keychain, which a Simulator test host cannot use. Counts
/// writes so the backfill's idempotence is assertable rather than inferred.
private final class FakeRegistrationMirror: @unchecked Sendable {
    private let lock = NSLock()
    private var stored: Data?
    private var saves = 0

    var saveCount: Int {
        lock.withLock { saves }
    }

    var mirror: RegistrationMirror {
        RegistrationMirror(
            load: { [self] in lock.withLock { stored } },
            save: { [self] data in
                lock.withLock {
                    stored = data
                    saves += 1
                }
                return true
            },
            clear: { [self] in
                lock.withLock { stored = nil }
                return true
            }
        )
    }
}
