import CryptoKit
import Foundation
import Testing

@testable import Pipette

/// The Keychain as the `auth` leaves see it, stubbed: a Simulator test host cannot use one.
/// `key` is the mirror copy; the store reads `identity/id_ed25519` first, so seeding only
/// this one models a device whose container lost the file.
private nonisolated final class KeychainStub: @unchecked Sendable {
    var key: String?
    var keySaves = 0
    var keyDeletes = 0
    var userToken: String?
    var modelTokens: [String: String] = [:]
    var tokenClears = 0

    init(key: String? = nil) { self.key = key }

    var privateKeyMirror: PrivateKeyMirror {
        PrivateKeyMirror(
            load: { [self] in key.flatMap { try? PrivateKey($0) } },
            save: { [self] in key = $0.value; keySaves += 1; return true },
            clear: { [self] in key = nil; keyDeletes += 1; return true }
        )
    }
}

/// What the `auth` leaves do to stored state: `pipette-cli`'s `commands/auth.rs` tests,
/// ported. `export`/`import` are the only path off the device for a signing key, so a wrong
/// restore costs the device its client id.
@MainActor struct AuthCommandsTests {

    // MARK: - Fixtures

    /// Any 32 bytes are a valid Ed25519 seed. The crate's test keys, so a token either
    /// suite builds reads the same.
    static let key = "aa11bb22cc33dd44ee55ff6600778899aa11bb22cc33dd44ee55ff6600778899"
    static let otherKey = "1122334455667788990011223344556677889900112233445566778899001122"
    static let server = "https://collector.example.test"

    private func registration(_ clientId: String, server: String = server) throws
        -> IdentityRegistration {
        IdentityRegistration(clientId: try ClientId(clientId), status: "approved",
                             serverUrl: try ServerUrl(server), organization: "Example",
                             contactEmail: "user@example.com",
                             registeredAt: "2026-05-28T16:41:00Z")
    }

    /// A device holding `clientId` signed by `key`, as the crate's `seed_identity` builds a
    /// workspace: both halves of the identity under the store's own root.
    private func seedIdentity(_ ws: PipetteWorkspace, _ clientId: String, key: String) throws {
        try ws.registration.putPrivateKey(try PrivateKey(key))
        try ws.registration.put(try registration(clientId))
    }

    /// A result still awaiting submission, as the crate's `add_pending` makes one: the
    /// directory is what `listIds` counts.
    private func addPending(_ ws: PipetteWorkspace, _ id: String) throws {
        let dir = try #require(ws.results.resultDir(.remotePending, CellId(id)))
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    }

    private func exists(_ url: URL) -> Bool {
        FileManager.default.fileExists(atPath: url.path)
    }

    /// The token `auth export` prints for `ws`, as the crate's `export_to_file` does.
    private func exportToken(_ ws: PipetteWorkspace, userToken: String? = nil,
                             modelTokens: [String: String] = [:]) throws -> String {
        try AuthCommands.exportBundle(
            session: try ws.registration.mgmtSession(),
            loadUserToken: { userToken.flatMap { try? AuthToken($0) } },
            loadModelTokens: { modelTokens }).encode()
    }

    /// `auth import` into a store mirroring to `keychain`, as the production store mirrors
    /// to the real Keychain.
    @discardableResult
    private func importToken(_ token: String, force: Bool = false,
                             into ws: inout PipetteWorkspace, keychain: KeychainStub,
                             modelTokensStore: Bool = true) -> Bool {
        ws.privateKeyMirror = keychain.privateKeyMirror
        return AuthCommands.import(
            bundle: token, force: force, ws: ws,
            saveUserToken: { keychain.userToken = $0.value; return true },
            saveModelToken: {
                guard modelTokensStore else { return false }
                keychain.modelTokens[$1] = $0.value
                return true
            },
            clearModelTokens: { keychain.tokenClears += 1; return 3 })
    }

    /// `auth reset` on a store mirroring to `keychain`, so the command's verdict on the
    /// credential is assertable without a Keychain.
    private func reset(_ ws: inout PipetteWorkspace, force: Bool, keychain: KeychainStub,
                       clearModelTokens: () -> Int = { 0 }) -> Bool {
        ws.privateKeyMirror = keychain.privateKeyMirror
        return AuthCommands.reset(force: force, ws: ws,
                                  clearModelTokens: clearModelTokens)
    }

    // MARK: - import

    @Test func exportThenImportMovesTheIdentityToAnotherDevice() throws {
        let source = makeTemporaryWorkspace()
        var target = makeTemporaryWorkspace()
        defer { removeWorkspace(source); removeWorkspace(target) }
        try seedIdentity(source, "ev1_abc", key: Self.key)
        let keychain = KeychainStub()

        #expect(importToken(try exportToken(source), into: &target, keychain: keychain))

        #expect(target.registration.get()?.clientId.value == "ev1_abc")
        #expect(keychain.key == Self.key)
        // And the composed session works, which is the point of the move.
        #expect(try target.registration.mgmtSession().auth.clientId.value == "ev1_abc")
    }

    /// The machine's own labels are not the identity's, so an import must not bring the
    /// exporting machine's over, nor touch the importing machine's.
    @Test func importLeavesTheImportingDeviceLabelsAlone() throws {
        let source = makeTemporaryWorkspace()
        var target = makeTemporaryWorkspace()
        defer { removeWorkspace(source); removeWorkspace(target) }
        try seedIdentity(source, "ev1_abc", key: Self.key)
        try source.device.put(DeviceLabels(deviceName: "source box",
                                                         deviceFormFactor: .server))
        try target.device.put(DeviceLabels(deviceName: "target box",
                                                         deviceFormFactor: .tablet))

        #expect(importToken(try exportToken(source), into: &target, keychain: KeychainStub()))

        let labels = try target.device.get()
        #expect(labels.deviceName == "target box")
        #expect(labels.deviceFormFactor == .tablet)
    }

    @Test func importRefusesADifferentIdentityWithoutForce() throws {
        let source = makeTemporaryWorkspace()
        var target = makeTemporaryWorkspace()
        defer { removeWorkspace(source); removeWorkspace(target) }
        try seedIdentity(source, "ev1_abc", key: Self.key)
        try seedIdentity(target, "ev1_held", key: Self.otherKey)
        try addPending(target, "r-1")
        let keychain = KeychainStub(key: Self.otherKey)

        #expect(!importToken(try exportToken(source, userToken: "hf_user"), into: &target,
                             keychain: keychain))

        // Refused means untouched: the key, the registration, the caches and the tokens.
        #expect(keychain.key == Self.otherKey)
        #expect(keychain.keySaves == 0)
        #expect(keychain.userToken == nil)
        #expect(target.registration.get()?.clientId.value == "ev1_held")
        #expect(target.results.listIds(.remotePending).count == 1)
    }

    /// A rerun after an import that failed partway finds the key that same bundle wrote,
    /// and must finish rather than refuse.
    @Test func importOfTheHeldIdentityNeedsNoForce() throws {
        let source = makeTemporaryWorkspace()
        var target = makeTemporaryWorkspace()
        defer { removeWorkspace(source); removeWorkspace(target) }
        try seedIdentity(source, "ev1_abc", key: Self.key)
        let token = try exportToken(source)
        // The state a partial import leaves: the key landed, the registration did not.
        // Upper case, so a comparison that read the casing as a different key would refuse.
        let keychain = KeychainStub(key: Self.key.uppercased())

        #expect(importToken(token, into: &target, keychain: keychain))
        #expect(target.registration.get()?.clientId.value == "ev1_abc")
        // And a second full rerun is still not a conflict.
        #expect(importToken(token, into: &target, keychain: keychain))
        #expect(keychain.tokenClears == 0, "nothing was replaced, so nothing is cleared")
    }

    @Test func forcedImportReplacesTheIdentityAndDropsItsCaches() throws {
        let source = makeTemporaryWorkspace()
        var target = makeTemporaryWorkspace()
        defer { removeWorkspace(source); removeWorkspace(target) }
        try seedIdentity(source, "ev1_abc", key: Self.key)
        try seedIdentity(target, "ev1_held", key: Self.otherKey)
        try addPending(target, "r-1")
        try target.benchmarks.cacheRemote(.maxMemoryUsage(MaxMemoryUsageBenchmark(
            benchmarkId: BenchmarkId(validated: "max_memory_usage_512"),
            parameterPrefillTokens: 512)))
        let keychain = KeychainStub(key: Self.otherKey)

        #expect(importToken(try exportToken(source, modelTokens: ["mlx://repo=o/r": "hf_m"]),
                            force: true, into: &target, keychain: keychain))

        #expect(target.registration.get()?.clientId.value == "ev1_abc")
        #expect(keychain.key == Self.key)
        // The outgoing client's unsubmitted results would otherwise be submitted under the
        // restored client's id on the next `sync`.
        #expect(!exists(target.results.locationDir(.remotePending)))
        #expect(target.benchmarks.list(.remote).isEmpty)
        // Its per-model tokens go too, before the bundle's own are written.
        #expect(keychain.tokenClears == 1)
        #expect(keychain.modelTokens == ["mlx://repo=o/r": "hf_m"])
    }

    @Test(arguments: ["not a bundle", "eyJ1bnJlbGF0ZWQiOiB0cnVlfQ"])
    func aBundleThisBuildCannotReadWritesNothing(token: String) {
        var target = makeTemporaryWorkspace()
        defer { removeWorkspace(target) }
        let keychain = KeychainStub()

        #expect(!importToken(token, into: &target, keychain: keychain))
        #expect(keychain.keySaves == 0)
        #expect(target.registration.get() == nil)
    }

    /// The decision a provisioning step keys on. Asserted on the disposition, as the crate
    /// asserts it: the `unchanged` arm returns before any write, so this is the same fact.
    @Test func aSecondImportOfTheSameBundleChangesNothing() throws {
        let source = makeTemporaryWorkspace()
        var target = makeTemporaryWorkspace()
        defer { removeWorkspace(source); removeWorkspace(target) }
        try seedIdentity(source, "ev1_abc", key: Self.key)
        let token = try exportToken(source)
        let keychain = KeychainStub()
        #expect(importToken(token, into: &target, keychain: keychain))

        let bundle = try IdentityBundle.decode(token)
        #expect(AuthCommands.ImportDisposition.of(
            heldKey: target.registration.getPrivateKey(), held: target.registration.get(),
            bundle: bundle) == .unchanged)
        // And the rerun itself writes no key.
        #expect(importToken(token, into: &target, keychain: keychain))
        #expect(keychain.keySaves == 1)
    }

    /// The line that decision reaches a caller as.
    @Test(arguments: [(true, "changed=yes"), (false, "changed=no")])
    func theSummaryReportsWhetherTheImportWrote(changed: Bool, expected: String) throws {
        let bundle = IdentityBundle(clientId: try ClientId("ev1_abc"),
                                    serverUrl: try ServerUrl(Self.server),
                                    privateKeyHex: try PrivateKey(Self.key))
        let summary = AuthCommands.importSummary(
            bundle: bundle, publicKey: try PublicKey(String(repeating: "ca7e", count: 16)),
            changed: changed)
        #expect(summary.hasSuffix(expected))
        #expect(!summary.contains(Self.key), "the signing key must not print")
    }

    /// The same client against a different server is a write to make, not an identity to
    /// replace, so it needs no `force=1`, and the device keeps what it knew about that
    /// client.
    @Test func theSameClientMovingServersNeedsNoForce() throws {
        let source = makeTemporaryWorkspace()
        var target = makeTemporaryWorkspace()
        defer { removeWorkspace(source); removeWorkspace(target) }
        try seedIdentity(source, "ev1_abc", key: Self.key)
        try target.registration.put(
            try registration("ev1_abc", server: "https://old-collector.example.test"))
        let keychain = KeychainStub(key: Self.key)

        #expect(importToken(try exportToken(source), into: &target, keychain: keychain))

        let restored = try #require(target.registration.get())
        #expect(restored.serverUrl.value == Self.server)
        #expect(restored.organization == "Example")
        #expect(keychain.tokenClears == 0)
    }

    /// A token that will not store means a gated download 401s later, so it fails the
    /// command rather than passing as a restore that only looks complete.
    @Test func importFailsWhenATokenIsNotStored() throws {
        let source = makeTemporaryWorkspace()
        var target = makeTemporaryWorkspace()
        defer { removeWorkspace(source); removeWorkspace(target) }
        try seedIdentity(source, "ev1_abc", key: Self.key)

        #expect(!importToken(try exportToken(source, modelTokens: ["mlx://repo=o/r": "hf_m"]),
                             into: &target, keychain: KeychainStub(), modelTokensStore: false))
        // The identity itself still landed: it is the half worth having.
        #expect(target.registration.get()?.clientId.value == "ev1_abc")
    }

    // MARK: - export

    /// A registration with no key cannot sign; a key with no registration cannot say who
    /// it signs for. Neither restores anything.
    @Test(arguments: [true, false])
    func exportRefusesHalfAnIdentity(withRegistration: Bool) throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        if withRegistration {
            try ws.registration.put(try registration("ev1_abc"))
        } else {
            try ws.registration.putPrivateKey(try PrivateKey(Self.key))
        }

        #expect(!AuthCommands.export(ws: ws, loadUserToken: { nil },
                                     loadModelTokens: { [:] }))
    }

    /// The model reference is what makes each token findable again on the device that
    /// restores it.
    @Test func theExportedBundleCarriesTheIdentityAndTheTokens() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        try seedIdentity(ws, "ev1_abc", key: Self.key)

        let bundle = AuthCommands.exportBundle(
            session: try ws.registration.mgmtSession(),
            loadUserToken: { try? AuthToken("hf_user") },
            loadModelTokens: { ["mlx://repo=org/b": "token_b", "mlx://repo=org/a": "token_a"] })

        #expect(bundle.clientId.value == "ev1_abc")
        #expect(bundle.serverUrl.value == Self.server)
        #expect(bundle.privateKeyHex.value == Self.key)
        #expect(bundle.userToken == "hf_user")
        #expect(bundle.modelTokens == ["mlx://repo=org/a": "token_a", "mlx://repo=org/b": "token_b"])
        #expect(bundle.tokenCount == 3)
    }

    @Test func exportSucceedsWithBothHalves() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        try seedIdentity(ws, "ev1_abc", key: Self.key)

        #expect(AuthCommands.export(ws: ws, loadUserToken: { nil },
                                    loadModelTokens: { [:] }))
    }

    // MARK: - me

    /// `local=1` answers from storage, so it works before the server has approved the
    /// client and without a request. The public key is the derived one, not the signing
    /// key this reads to derive it.
    @Test func meLocalReportsTheIdentityOnDisk() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        try seedIdentity(ws, "ev1_abc", key: Self.key)
        let raw = try #require(Data(hexEncoded: Self.key))
        let publicKey = try Curve25519.Signing.PrivateKey(rawRepresentation: raw)
            .publicKey.rawRepresentation.hexEncodedString()

        let line = try AuthCommands.localIdentity(ws: ws)

        #expect(line == "auth me client_id=ev1_abc server_url=\(Self.server) "
            + "public_key=\(publicKey)")
        #expect(!line.contains(Self.key), "the signing key must not print")
    }

    @Test(arguments: [false, true])
    func meLocalRefusesHalfAnIdentity(withRegistration: Bool) throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        if withRegistration {
            try ws.registration.put(try registration("ev1_abc"))
        }
        #expect(throws: withRegistration ? IdentityError.privateKeyMissing
                                         : IdentityError.registrationMissing) {
            try AuthCommands.localIdentity(ws: ws)
        }
    }

    @Test(arguments: [([String](), "(none)"),
                      (["team/mobile"], "team/mobile"),
                      (["team/mobile", "batch/2026-q3"], "batch/2026-q3,team/mobile")])
    func formatTagsCases(tags: [String], expected: String) {
        #expect(AuthCommands.formatTags(tags) == expected)
    }

    // MARK: - register

    /// The crate's `ensure_empty`: registering over an identity the device already holds
    /// is refused before a keypair is minted, so nothing it holds changes.
    @Test(arguments: [true, false])
    func registerRefusesOverAnExistingIdentity(withRegistration: Bool) async throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        if withRegistration {
            try ws.registration.put(try registration("ev1_held"))
        } else {
            try ws.registration.putPrivateKey(try PrivateKey(Self.key))
        }
        let input = AuthCommands.RegisterArgs(
            serverUrl: try ServerUrl(Self.server), organization: "Example",
            contactEmail: try ContactEmail("user@example.com"), clientDetails: "ci box")

        #expect(await AuthCommands.register(input, ws: ws) == false)
        #expect(ws.registration.get()?.clientId.value
            == (withRegistration ? "ev1_held" : nil))
    }

    // MARK: - reset

    @Test func resetRefusesUnsubmittedResultsWithoutForce() throws {
        var ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        try seedIdentity(ws, "ev1_abc", key: Self.key)
        try addPending(ws, "r-1")

        let keychain = KeychainStub(key: Self.key)
        #expect(!reset(&ws, force: false, keychain: keychain))
        // A refused reset deletes neither the pending results nor the credential.
        #expect(exists(ws.results.locationDir(.remotePending)))
        #expect(keychain.keyDeletes == 0)
        #expect(ws.registration.getPrivateKey() != nil)
        #expect(ws.registration.get() != nil)
    }

    @Test func resetForceClearsUnsubmittedResults() throws {
        var ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        try seedIdentity(ws, "ev1_abc", key: Self.key)
        try addPending(ws, "r-1")

        #expect(reset(&ws, force: true, keychain: KeychainStub(key: Self.key)))
        #expect(!exists(ws.results.locationDir(.remotePending)))
    }

    /// Every identity file goes, across all three stores, and so does the synced catalog.
    @Test func resetWithoutPendingProceedsWithoutForce() throws {
        var ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        try seedIdentity(ws, "ev1_abc", key: Self.key)
        try ws.device.put(DeviceLabels(deviceName: "lab box"))
        try ws.settings.put(ClientSettings(storageQuotaBytes: 4096))
        try ws.benchmarks.cacheRemote(.maxMemoryUsage(MaxMemoryUsageBenchmark(
            benchmarkId: BenchmarkId(validated: "max_memory_usage_512"),
            parameterPrefillTokens: 512)))
        try ws.benchmarks.putLocal(.maxMemoryUsage(MaxMemoryUsageBenchmark(
            benchmarkId: BenchmarkId(validated: "max_memory_usage_512"),
            parameterPrefillTokens: 512)))
        try ws.benchmarks.putRemoteIndex(Coding.decoder.decode(
            [PipetteMgmtClient.BenchmarkSummary].self,
            from: Data(#"[{"benchmark_id":"max_memory_usage_512","benchmark_type":"max_memory_usage"}]"#.utf8)))
        try ws.benchmarks.putSyncState(RemoteSyncState(
            pulledAt: "2026-01-01T00:00:00Z", benchmarkCount: 1, serverUrl: nil,
            benchmarksEtag: nil))
        try ws.benchmarks.writeOlderBuildBookkeeping(
            rows: [], server: "https://mgmt.test", listEtag: #""list-v1""#, benchmarkEtags: [:])

        let keychain = KeychainStub(key: Self.key)
        var tokenClears = 0
        #expect(reset(&ws, force: false, keychain: keychain,
                      clearModelTokens: { tokenClears += 1; return 2 }))

        #expect(ws.registration.get() == nil)
        #expect(ws.registration.getPrivateKey() == nil)
        #expect(keychain.keyDeletes == 1)
        #expect(tokenClears == 1)
        #expect(try ws.device.get() == .empty)
        #expect(ws.settings.get() == nil)
        #expect(ws.benchmarks.list(.remote).isEmpty)
        #expect(try ws.benchmarks.listRemoteIndex().isEmpty)
        #expect(try ws.benchmarks.getSyncState() == nil)
        #expect(!exists(ws.benchmarks.root.appendingPathComponent("remote")))
        // Local benchmarks are server-independent and kept.
        #expect(ws.benchmarks.list(.local).count == 1)
    }

    /// A device with nothing stored is already in the state the command produces, so it
    /// succeeds rather than reporting a failure the operator cannot act on.
    @Test func resetOnAnUnregisteredDeviceSucceeds() {
        var ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }

        #expect(reset(&ws, force: false, keychain: KeychainStub()))
        #expect(ws.registration.get() == nil)
    }

    /// Leaving the key behind is the outcome the command exists to prevent.
    @Test func aFailedKeyDeleteFailsTheReset() throws {
        var ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        try seedIdentity(ws, "ev1_abc", key: Self.key)

        ws.privateKeyMirror = PrivateKeyMirror(
            load: { nil }, save: { _ in true }, clear: { false })
        #expect(!AuthCommands.reset(force: true, ws: ws, clearModelTokens: { 0 }))
        // The record still goes: a half-cleared identity is reported, not silently kept.
        #expect(ws.registration.get() == nil)
    }
}
