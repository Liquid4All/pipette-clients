import Foundation

/// A registered identity and the server it is registered with — the crate's
/// `MgmtSession` (`pipette-cli/src/workspace.rs:28`), minted from the storage seam the
/// way the CLI mints it from the workspace.
///
/// Carries the whole registration where the crate carries only `server_url`, because
/// the two records differ: the CLI fetches organization / contact / status from
/// `auth me` on demand, while this one keeps them so the app can render them without a
/// round trip. `serverUrl` is the field callers actually reach for.
///
/// No `client` field: `ManagementClient` is a static namespace here, so there is no
/// per-server instance to bind — the session passes `serverUrl` to each call instead.
nonisolated struct MgmtSession: Sendable {
    let registration: IdentityRegistration
    let auth: AuthIdentity

    var serverUrl: ServerURL { registration.serverUrl }
}

extension Storage {
    /// The registered identity plus the server to talk to — the opening move of every
    /// call that reaches the management server, as the crate's `mgmt_session()` is.
    ///
    /// Throws rather than returning nil so a caller can say *which* half is missing;
    /// `auth reset` and `auth me` both report that distinction. Callers that only need
    /// "can this device talk to the server at all" use `try?`.
    ///
    /// Reads the registration once and composes the `AuthIdentity` here rather than
    /// calling ``IdentityStore/signingIdentity()``, which would re-read and re-decode
    /// the file. The crate can afford that second read — it mints a session once per
    /// command; the planner worker mints one per claim round.
    nonisolated func mgmtSession() throws -> MgmtSession {
        guard let registration = identity.getRegistration() else {
            throw IdentityError.registrationMissing
        }
        guard let privateKey = identity.getPrivateKey() else {
            throw IdentityError.privateKeyMissing
        }
        return MgmtSession(
            registration: registration,
            auth: AuthIdentity(clientId: registration.clientId, privateKeyHex: privateKey))
    }
}
