import Foundation

/// The `auth` group's read and reset leaves — `auth me` and `auth reset`. Registration
/// itself stays in `HeadlessRunner.startRegister`, which owns the keypair generation.
///
/// Field names are the wire spellings (`reindex_pending`, not `reindexPending`) so a
/// console consumer greps the same token it would in a server response.
enum AuthCommands {
    /// `auth me`: the authenticated profile, live from `GET /clients/me`. The same fields
    /// the CLI prints, on one line — `tags` and `capabilities` are comma-joined rather
    /// than omitted when empty, so a parser sees the key either way.
    static func me(storage: Storage) async -> Bool {
        // `mgmtSession` distinguishes the two halves, so the log says which one is
        // missing rather than a flat "not registered".
        let session: MgmtSession
        do {
            session = try storage.mgmtSession()
        } catch {
            HeadlessRunner.log("auth me ERROR \(error.localizedDescription)")
            return false
        }
        do {
            let profile = try await ManagementClient.me(
                serverUrl: session.serverUrl,
                auth: session.auth)
            HeadlessRunner.log("auth me client_id=\(profile.clientId) "
                + "organization=\(profile.organization) "
                + "client_details=\(profile.clientDetails) "
                + "contact_email=\(profile.contactEmail) "
                + "status=\(profile.status) "
                + "tags=\(profile.tags.joined(separator: ",")) "
                + "reindex_pending=\(profile.reindexPending) "
                + "capabilities=\(profile.capabilities.joined(separator: ","))")
            return true
        } catch {
            HeadlessRunner.log("auth me ERROR \(error)")
            return false
        }
    }

    /// `auth reset`: forget this device's identity — the registration record *and* the
    /// signing key it is useless without.
    ///
    /// Both halves, deliberately: deleting only the record leaves the private key orphaned
    /// in the Keychain, which is not "forgotten" by any reading an operator would accept
    /// from a command whose purpose is discarding a credential.
    ///
    /// Gated on `force=1` because it is not undoable — re-registering mints a fresh
    /// keypair, so every result already submitted under the old key stops being
    /// attributable to this device. The CLI gates it the same way.
    /// `deleteKey` and `clearModelTokens` are injected so the outcome is assertable
    /// off-device: a Simulator test host cannot perform Keychain operations, and the
    /// command's result depends on whether the credentials actually went.
    static func reset(
        force: Bool,
        storage: Storage,
        deleteKey: () -> Bool = KeychainHelper.deletePrivateKey,
        clearModelTokens: () -> Int = KeychainHelper.deleteAllModelHfTokens
    ) -> Bool {
        guard force else {
            HeadlessRunner.log("auth reset ERROR refusing without force=1: this discards the "
                + "signing key, and re-registering mints a new one")
            return false
        }
        let registration = storage.identity.getRegistration()
        // The same teardown the Settings and sign-out paths run; a Keychain delete can
        // fail on its own, and the summary line reports that rather than implying the
        // credential is gone when it is still there.
        let keyCleared = storage.identity.clearRegistrationMaterial(deleteKey: deleteKey)
        // Every credential, not just the signing key: a plan's model tokens are stored on
        // this device too, and "forget this device's identity" that leaves them behind is
        // the same half-measure this command already shipped with once.
        let modelTokens = clearModelTokens()
        guard let registration else {
            HeadlessRunner.log("auth reset no registration; signing_key_cleared=\(keyCleared) "
                + "model_tokens_cleared=\(modelTokens)")
            return keyCleared
        }
        HeadlessRunner.log("auth reset cleared client_id=\(registration.clientId.value) "
            + "signing_key_cleared=\(keyCleared) model_tokens_cleared=\(modelTokens)")
        return keyCleared
    }
}
