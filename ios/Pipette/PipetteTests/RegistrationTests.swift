import Foundation
import Testing

@testable import Pipette

/// The identity store: both halves of the identity, their two backings, and the
/// `registration.json` wire form. A Simulator test host cannot reach a Keychain item, so
/// tests that need one supply an in-memory mirror.
struct RegistrationTests {

    @Test func registrationRoundTripsAndClears() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }

        try ws.registration.put(try registrationData())

        let loaded = try #require(ws.registration.get())
        #expect(loaded == (try registrationData()))
        #expect(ws.registration.isRegistered)

        ws.registration.clear()
        #expect(ws.registration.get() == nil)
        #expect(!ws.registration.isRegistered)
    }

    /// Sign-out unlinks rather than clears: clearing would take the `clientId` and signing
    /// key too, orphaning every result already submitted.
    @Test func unlinkingClerkKeepsTheUploadIdentity() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }

        try ws.registration.put(try registrationData().withoutClerkLink())

        let loaded = try #require(ws.registration.get())
        #expect(loaded.clerkUserId == nil)
        #expect(loaded.clerkSessionId == nil)
        #expect(loaded.clerkPrimaryEmail == nil)
        // `withClerkLink` preserves an existing value, so a kept one would date the next
        // account's link.
        #expect(loaded.clerkLinkedAt == nil)
        // Field by field on purpose: `withoutClerkLink` re-lists the fields to keep, so a new
        // `IdentityRegistration` field is dropped silently unless a test names it.
        #expect(loaded.clientId == (try registrationData().clientId))
        #expect(loaded.serverUrl == (try registrationData().serverUrl))
        #expect(loaded.organization == "Example")
        #expect(loaded.registeredAt == "2026-05-28T16:41:00Z")
        #expect(loaded.status == (try registrationData().status))
        #expect(loaded.contactEmail == (try registrationData().contactEmail))
        #expect(ws.registration.isRegistered)
    }

    /// An unlinked record then re-links to whichever account signs in next, which is what
    /// keeps a second user off the mismatch screen.
    @Test func anUnlinkedRegistrationRelinksToADifferentAccount() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }

        let relinked = try registrationData()
            .withoutClerkLink()
            .withClerkLink(userId: "user_2", sessionId: "session_2", primaryEmail: "second@example.com")
        try ws.registration.put(relinked)

        let loaded = try #require(ws.registration.get())
        #expect(loaded.clerkUserId == "user_2")
        #expect(loaded.clerkPrimaryEmail == "second@example.com")
        #expect(loaded.clerkLinkedAt != nil)
        #expect(loaded.clerkLinkedAt != "2026-05-28T16:42:00Z")
    }

    /// The file the CLI would read: `snake_case`, as `identity/registration.json` is
    /// there and as `settings.json` already is here.
    @Test func registrationWritesSnakeCaseKeys() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }

        try ws.registration.put(try registrationData())

        let data = try Data(contentsOf: ws.registration.root.appendingPathComponent("registration.json"))
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

    /// Reading it as absent would send the device back through setup and mint a keypair
    /// that orphans its submitted results.
    @Test func aCamelCaseRegistrationStillDecodes() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }

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
        _ = PipetteWorkspace.ensureDirectory(ws.registration.root)
        try Data(legacy.utf8)
            .write(to: ws.registration.root.appendingPathComponent("registration.json"))

        let loaded = try #require(ws.registration.get())
        #expect(loaded.clientId.value == "client-legacy")
        #expect(loaded.serverUrl.value == "https://collector.example.com")
        #expect(loaded.contactEmail == "user@example.com")
        #expect(loaded.registeredAt == "2026-05-28T16:41:00Z")
        #expect(loaded.clerkUserId == "user_1")
        #expect(loaded.clerkSessionId == nil)
    }

    /// A rewrite uses the new spelling, so this should not arise; the per-key fallback reads
    /// it anyway.
    @Test func aMixedSpellingRegistrationDecodesBothHalves() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }

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
        _ = PipetteWorkspace.ensureDirectory(ws.registration.root)
        try Data(mixed.utf8)
            .write(to: ws.registration.root.appendingPathComponent("registration.json"))

        let loaded = try #require(ws.registration.get())
        #expect(loaded.clientId.value == "client-mixed")
        #expect(loaded.serverUrl.value == "https://collector.example.com")
        #expect(loaded.contactEmail == "user@example.com")
        #expect(loaded.registeredAt == "2026-05-28T16:41:00Z")
    }

    /// Reported as unregistered rather than trapping, like an unreadable file.
    @Test func aRegistrationWithoutIdentityFieldsReadsAsAbsent() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }

        _ = PipetteWorkspace.ensureDirectory(ws.registration.root)
        try Data(#"{"status":"approved","organization":"Example"}"#.utf8)
            .write(to: ws.registration.root.appendingPathComponent("registration.json"))

        #expect(ws.registration.get() == nil)
    }

    /// Only the upload identity is required, as in the crate's record: the rest are copies
    /// of server-owned state, and reading as unregistered would mint a keypair that orphans
    /// the device's submitted results.
    @Test func oldRegistrationWithDroppedFieldsStillDeserializes() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }

        _ = PipetteWorkspace.ensureDirectory(ws.registration.root)
        try Data(#"{"client_id":"client-1","server_url":"https://c.example.com"}"#.utf8)
            .write(to: ws.registration.root.appendingPathComponent("registration.json"))

        let loaded = try #require(ws.registration.get())
        #expect(loaded.clientId.value == "client-1")
        #expect(loaded.status.isEmpty)
        #expect(loaded.organization.isEmpty)
        #expect(ws.registration.isRegistered)
    }

    /// `auth me` reports which half is missing, so a single nil would collapse two causes.
    @Test func mgmtSessionNamesTheMissingHalf() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }

        #expect(throws: IdentityError.registrationMissing) { try ws.registration.mgmtSession() }

        // The temporary store supplies no key on any host, so this holds on a device too.
        try ws.registration.put(try registrationData())
        #expect(throws: IdentityError.privateKeyMissing) { try ws.registration.mgmtSession() }
    }

    @Test func mgmtSessionCarriesBothHalves() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        try ws.registration.putPrivateKey(try PrivateKey(testPrivateKeyHex))
        try ws.registration.put(try registrationData())

        let session = try ws.registration.mgmtSession()
        #expect(session.registration.clientId == (try registrationData().clientId))
        #expect(session.auth.clientId == (try registrationData().clientId))
        #expect(session.auth.privateKeyHex == (try PrivateKey(testPrivateKeyHex)))
    }

    @Test func settingsRoundTripThroughTheStore() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }

        #expect(ws.settings.get() == nil)
        try ws.settings.put(ClientSettings(storageQuotaBytes: 9 << 30))
        #expect(ws.settings.get()?.storageQuotaBytes == 9 << 30)
    }

    @Test func registrationAndSettingsAreIndependentFiles() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }

        try ws.registration.put(try registrationData())
        try ws.settings.put(ClientSettings(storageQuotaBytes: 3 << 30))

        ws.registration.clear()
        #expect(ws.settings.get()?.storageQuotaBytes == 3 << 30)
    }

    // MARK: - The metadata/ -> identity/ move

    /// Losing the file would push the device back through setup, minting a keypair that
    /// orphans its submitted results.
    @Test func theMigrationCarriesRegistrationAndSettingsAcross() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let legacy = PipetteWorkspace.ensureDirectory(ws.legacyMetadataDir)
        try Coding.encoder.encode(try registrationData())
            .write(to: legacy.appendingPathComponent("registration.json"))
        try Coding.encoder.encode(ClientSettings(storageQuotaBytes: 7 << 30))
            .write(to: legacy.appendingPathComponent("settings.json"))

        #expect(Set(ws.migrateIdentityDirectory()) == ["registration.json", "settings.json"])

        #expect(ws.registration.get() == (try registrationData()))
        #expect(ws.settings.get()?.storageQuotaBytes == 7 << 30)
        // The old directory goes once it is empty.
        #expect(!FileManager.default.fileExists(atPath: ws.legacyMetadataDir.path))
    }

    @Test func theMigrationIsIdempotent() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let legacy = PipetteWorkspace.ensureDirectory(ws.legacyMetadataDir)
        try Coding.encoder.encode(try registrationData())
            .write(to: legacy.appendingPathComponent("registration.json"))

        #expect(ws.migrateIdentityDirectory() == ["registration.json"])
        #expect(ws.migrateIdentityDirectory().isEmpty)
        #expect(ws.registration.get() == (try registrationData()))
    }

    /// The destination copy is the newer one, written by a build that had already moved.
    @Test func anAlreadyMovedFileIsNotOverwritten() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        try ws.registration.put(try registrationData())

        let legacy = PipetteWorkspace.ensureDirectory(ws.legacyMetadataDir)
        try Data(#"{"client_id":"stale","server_url":"https://old.example.com"}"#.utf8)
            .write(to: legacy.appendingPathComponent("registration.json"))

        #expect(ws.migrateIdentityDirectory().isEmpty)
        #expect(ws.registration.get()?.clientId.value == "client-1")
    }

    /// An unexpected file in `metadata/` is someone else's, so the directory stays.
    @Test func anUnknownLeftoverKeepsTheOldDirectory() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let legacy = PipetteWorkspace.ensureDirectory(ws.legacyMetadataDir)
        try Data("{}".utf8).write(to: legacy.appendingPathComponent("something-else.json"))

        ws.migrateIdentityDirectory()

        #expect(FileManager.default.fileExists(atPath: ws.legacyMetadataDir.path))
    }

    // MARK: - Surviving a reinstall

    @Test func aReinstallRecoversTheRegistrationFromTheMirror() throws {
        let mirror = FakeRegistrationMirror()
        var ws = makeTemporaryWorkspace()
        ws.registrationMirror = mirror.mirror
        defer { removeWorkspace(ws) }
        try ws.registration.put(try registrationData())

        // The uninstall: the container goes, the Keychain stays.
        try FileManager.default.removeItem(at: ws.root)

        #expect(ws.registration.get() == (try registrationData()))
        #expect(ws.registration.isRegistered)
    }

    /// The restore rewrites the file, so every other reader takes the ordinary path and the
    /// mirror is consulted once rather than on every call.
    @Test func aRestoredRegistrationIsWrittenBackToTheFile() throws {
        let mirror = FakeRegistrationMirror()
        var ws = makeTemporaryWorkspace()
        ws.registrationMirror = mirror.mirror
        defer { removeWorkspace(ws) }
        try ws.registration.put(try registrationData())
        try FileManager.default.removeItem(at: ws.root)

        _ = ws.registration.get()

        let file = ws.registration.root.appendingPathComponent("registration.json")
        #expect(FileManager.default.fileExists(atPath: file.path))
        let reloaded = try Coding.decoder.decode(
            IdentityRegistration.self, from: Data(contentsOf: file))
        #expect(reloaded == (try registrationData()))
    }

    /// "Forget this device" has to reach both copies. Leaving the mirror behind would let
    /// the next read restore exactly what `auth reset` just discarded.
    @Test func deletingTheRegistrationClearsTheMirrorToo() throws {
        let mirror = FakeRegistrationMirror()
        var ws = makeTemporaryWorkspace()
        ws.registrationMirror = mirror.mirror
        defer { removeWorkspace(ws) }
        try ws.registration.put(try registrationData())

        ws.registration.clear()

        #expect(!ws.registration.isRegistrationMirrored)
        #expect(ws.registration.get() == nil)
    }

    /// `auth reset` clears through here (`clearIdentity`).
    @Test func clearingRegistrationMaterialClearsTheMirror() throws {
        let mirror = FakeRegistrationMirror()
        var ws = makeTemporaryWorkspace()
        ws.registrationMirror = mirror.mirror
        defer { removeWorkspace(ws) }
        try ws.registration.put(try registrationData())

        #expect(ws.registration.clear())

        #expect(!ws.registration.isRegistrationMirrored)
        #expect(ws.registration.get() == nil)
    }

    /// A device registered before the mirror existed has a file and no copy, and would
    /// otherwise gain one only by re-registering.
    @Test func theBackfillMirrorsAnExistingRegistration() throws {
        let mirror = FakeRegistrationMirror()
        var ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        try ws.registration.put(try registrationData())
        ws.registrationMirror = mirror.mirror
        #expect(!ws.registration.isRegistrationMirrored)

        ws.registration.backfillRegistrationMirror()

        #expect(ws.registration.isRegistrationMirrored)
        try FileManager.default.removeItem(at: ws.root)
        #expect(ws.registration.get() == (try registrationData()))
    }

    /// The file is the authority, but a launch is not a write.
    @Test func theBackfillIsANoOpWhenAlreadyMirrored() throws {
        let mirror = FakeRegistrationMirror()
        var ws = makeTemporaryWorkspace()
        ws.registrationMirror = mirror.mirror
        defer { removeWorkspace(ws) }
        try ws.registration.put(try registrationData())
        let writesAfterRegistration = mirror.saveCount

        ws.registration.backfillRegistrationMirror()
        ws.registration.backfillRegistrationMirror()

        #expect(mirror.saveCount == writesAfterRegistration)
    }

    @Test func theBackfillDoesNothingWithoutARegistration() {
        let mirror = FakeRegistrationMirror()
        var ws = makeTemporaryWorkspace()
        ws.registrationMirror = mirror.mirror
        defer { removeWorkspace(ws) }

        ws.registration.backfillRegistrationMirror()

        #expect(!ws.registration.isRegistrationMirrored)
        #expect(ws.registration.get() == nil)
    }

    @Test func aFailedMirrorWriteStillRegistersTheDevice() throws {
        var ws = makeTemporaryWorkspace()
        ws.registrationMirror = RegistrationMirror(
            load: { nil }, save: { _ in false }, clear: { false })
        defer { removeWorkspace(ws) }

        try ws.registration.put(try registrationData())

        #expect(ws.registration.get() == (try registrationData()))
        #expect(!ws.registration.isRegistrationMirrored)
    }
    // MARK: - Signing key

    private func keyPath(_ ws: PipetteWorkspace) -> URL {
        ws.registration.root.appendingPathComponent("id_ed25519")
    }

    /// The file is the store's own copy: written under `root`, so a temporary store carries
    /// a whole identity and a device keeps one the Keychain cannot answer for.
    @Test func privateKeyRoundTripAndDelete() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }

        try ws.registration.putPrivateKey(try PrivateKey(testPrivateKeyHex))

        #expect(ws.registration.getPrivateKey() == (try PrivateKey(testPrivateKeyHex)))
        #expect(try String(contentsOf: keyPath(ws), encoding: .utf8) == testPrivateKeyHex)
    }

    /// `0600` even over a `0644` copy: the atomic write replaces the file, so the
    /// permissions are set again (the crate's `an_existing_loose_file_is_tightened_and_replaced`).
    /// The protection class is asserted on a device only: a Simulator applies no data protection.
    @Test func privateKeyIsCreatedOwnerOnly() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }

        try ws.registration.putPrivateKey(try PrivateKey(testPrivateKeyHex))
        try FileManager.default.setAttributes(
            [.posixPermissions: 0o644], ofItemAtPath: keyPath(ws).path)
        try ws.registration.putPrivateKey(try PrivateKey(testPrivateKeyHex))

        let attributes = try FileManager.default.attributesOfItem(atPath: keyPath(ws).path)
        #expect(attributes[.posixPermissions] as? Int == 0o600)
    }

    /// A value the curve refuses cannot sign; the crate's `get_private_key` reads it as
    /// absent too.
    @Test func aKeyFileThatCannotSignReadsAsAbsent() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        try FileManager.default.createDirectory(at: ws.registration.root,
                                                withIntermediateDirectories: true)
        try Data("not a key".utf8).write(to: keyPath(ws))

        #expect(ws.registration.getPrivateKey() == nil)
    }

    /// Unusable and absent both fall through to the mirror, so a key file a partial write
    /// left short is repaired from the Keychain copy.
    @Test func anUnusableKeyFileIsReplacedFromTheMirror() throws {
        let mirror = FakePrivateKeyMirror()
        var ws = makeTemporaryWorkspace()
        ws.privateKeyMirror = mirror.mirror
        defer { removeWorkspace(ws) }
        try FileManager.default.createDirectory(at: ws.registration.root,
                                                withIntermediateDirectories: true)
        try Data("not a key".utf8).write(to: keyPath(ws))
        _ = mirror.mirror.save(try PrivateKey(testPrivateKeyHex))

        #expect(ws.registration.getPrivateKey() == (try PrivateKey(testPrivateKeyHex)))

        #expect(try String(contentsOf: keyPath(ws), encoding: .utf8) == testPrivateKeyHex)
    }

    /// The reinstall path for the key: the read rewrites the file so later readers take the
    /// ordinary path.
    @Test func theKeyIsRestoredFromTheMirrorWhenTheFileIsGone() throws {
        let mirror = FakePrivateKeyMirror()
        var ws = makeTemporaryWorkspace()
        ws.privateKeyMirror = mirror.mirror
        defer { removeWorkspace(ws) }
        try ws.registration.putPrivateKey(try PrivateKey(testPrivateKeyHex))
        try FileManager.default.removeItem(at: ws.root)

        #expect(ws.registration.getPrivateKey() == (try PrivateKey(testPrivateKeyHex)))
        #expect(try String(contentsOf: keyPath(ws), encoding: .utf8) == testPrivateKeyHex)
    }

    /// "Forget this device" has to reach both copies of the key, or the next read restores
    /// exactly what `auth reset` discarded.
    @Test func deletingTheKeyClearsBothCopies() throws {
        let mirror = FakePrivateKeyMirror()
        var ws = makeTemporaryWorkspace()
        ws.privateKeyMirror = mirror.mirror
        defer { removeWorkspace(ws) }
        try ws.registration.putPrivateKey(try PrivateKey(testPrivateKeyHex))

        #expect(ws.registration.deletePrivateKey())

        #expect(ws.registration.getPrivateKey() == nil)
        #expect(mirror.mirror.load() == nil)
    }

    /// The state an older build leaves: the key in the Keychain and no file.
    @Test func aKeychainOnlyKeyGainsItsFileOnTheFirstRead() throws {
        let mirror = FakePrivateKeyMirror()
        var ws = makeTemporaryWorkspace()
        ws.privateKeyMirror = mirror.mirror
        defer { removeWorkspace(ws) }
        _ = mirror.mirror.save(try PrivateKey(testPrivateKeyHex))
        let writesBeforeRead = mirror.saveCount

        #expect(ws.registration.getPrivateKey() == (try PrivateKey(testPrivateKeyHex)))

        #expect(try String(contentsOf: keyPath(ws), encoding: .utf8) == testPrivateKeyHex)
        #expect(mirror.saveCount == writesBeforeRead, "a read restores the file, not the mirror")
    }

    /// A build re-signed under another team keeps both files but reaches neither Keychain
    /// copy. The launch re-mirrors the record, so it must re-mirror the key too, or the next
    /// uninstall restores a record the device cannot sign for.
    @Test func theBackfillMirrorsAnExistingPrivateKey() throws {
        let mirror = FakePrivateKeyMirror()
        var ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        try ws.registration.putPrivateKey(try PrivateKey(testPrivateKeyHex))
        ws.privateKeyMirror = mirror.mirror
        #expect(mirror.mirror.load() == nil)

        ws.registration.backfillPrivateKeyMirror()

        #expect(mirror.mirror.load() == (try PrivateKey(testPrivateKeyHex)))
        try FileManager.default.removeItem(at: ws.root)
        #expect(ws.registration.getPrivateKey() == (try PrivateKey(testPrivateKeyHex)))
    }

    /// The file is the authority, but a launch is not a write.
    @Test func thePrivateKeyBackfillIsANoOpWhenAlreadyMirrored() throws {
        let mirror = FakePrivateKeyMirror()
        var ws = makeTemporaryWorkspace()
        ws.privateKeyMirror = mirror.mirror
        defer { removeWorkspace(ws) }
        try ws.registration.putPrivateKey(try PrivateKey(testPrivateKeyHex))
        let writesBeforeBackfill = mirror.saveCount

        ws.registration.backfillPrivateKeyMirror()

        #expect(mirror.saveCount == writesBeforeBackfill)
    }

    /// Inventing an entry would make an unregistered device look registered.
    @Test func thePrivateKeyBackfillIsANoOpWithNoKey() {
        let mirror = FakePrivateKeyMirror()
        var ws = makeTemporaryWorkspace()
        ws.privateKeyMirror = mirror.mirror
        defer { removeWorkspace(ws) }

        ws.registration.backfillPrivateKeyMirror()

        #expect(mirror.mirror.load() == nil)
        #expect(mirror.saveCount == 0)
    }

    /// The Keychain copies cannot cross a backup, so a restored container would carry a
    /// record with no key or, with the key beside it, one `client_id` on two devices.
    @Test func theIdentityDirectoryIsExcludedFromBackup() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }

        try ws.registration.putPrivateKey(try PrivateKey(testPrivateKeyHex))
        try ws.registration.put(try registrationData())

        let values = try ws.registration.root.resourceValues(forKeys: [.isExcludedFromBackupKey])
        #expect(values.isExcludedFromBackup == true)
    }

    /// A device holding the record and no key to write reaches no writer, so only the launch
    /// pass keeps its `identity/` out of backups.
    @Test func theLaunchPassExcludesADirectoryNoWriterTouched() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        // Created the way the migration creates it, with no identity write behind it.
        try FileManager.default.createDirectory(
            at: ws.registration.root, withIntermediateDirectories: true)
        let before = try ws.registration.root.resourceValues(forKeys: [.isExcludedFromBackupKey])
        #expect(before.isExcludedFromBackup != true)

        ws.excludeIdentityFromBackup()

        let after = try ws.registration.root.resourceValues(forKeys: [.isExcludedFromBackupKey])
        #expect(after.isExcludedFromBackup == true)
    }

    /// It must not create the directory it protects: a device that has never registered has
    /// no identity, and an empty `identity/` would make `migrateIdentityDirectory()` the
    /// second writer of a path this store owns.
    @Test func theLaunchPassCreatesNoDirectory() {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }

        ws.excludeIdentityFromBackup()

        #expect(!FileManager.default.fileExists(atPath: ws.registration.root.path))
    }

    @Test func readingAKeyNeitherBackingHoldsWritesNothing() {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }

        #expect(ws.registration.getPrivateKey() == nil)
        #expect(!FileManager.default.fileExists(atPath: keyPath(ws).path))
    }

    /// Refusing would trade a working identity for a durable one.
    @Test func aFailedKeyMirrorWriteStillStoresTheKey() throws {
        var ws = makeTemporaryWorkspace()
        ws.privateKeyMirror = PrivateKeyMirror(
            load: { nil }, save: { _ in false }, clear: { false })
        defer { removeWorkspace(ws) }

        try ws.registration.putPrivateKey(try PrivateKey(testPrivateKeyHex))

        #expect(ws.registration.getPrivateKey() == (try PrivateKey(testPrivateKeyHex)))
    }

    /// The state that stranded a device: the record reads from its file while the key has
    /// none.
    @Test func aRecordWithNoKeyComposesNoIdentityAndStillBlocksRegistration() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        try ws.registration.put(try registrationData())

        #expect(ws.registration.isRegistered)
        #expect(ws.registration.getPrivateKey() == nil)
        #expect(ws.registration.signingIdentity() == nil)
        #expect(throws: IdentityError.self) { try ws.registration.ensureEmpty() }
    }

}

/// An in-memory stand-in for the Keychain's signing-key item, on the terms
/// `FakeRegistrationMirror` stands in for its record.
private nonisolated final class FakePrivateKeyMirror: @unchecked Sendable {
    private let lock = NSLock()
    private var stored: PrivateKey?
    private var saves = 0

    var saveCount: Int {
        lock.withLock { saves }
    }

    var mirror: PrivateKeyMirror {
        PrivateKeyMirror(
            load: { [self] in lock.withLock { stored } },
            save: { [self] key in
                lock.withLock {
                    stored = key
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
