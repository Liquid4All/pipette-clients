import Foundation

/// A registered identity and the server it is registered with. No crate counterpart: the
/// CLI reads both off `ClientRegistrationStore` per call, while this keeps the whole
/// registration so the app can render it without a round trip.
nonisolated struct MgmtSession: Sendable {
    let registration: IdentityRegistration
    let auth: ClientIdentity

    var serverUrl: ServerUrl { registration.serverUrl }
}

nonisolated extension ClientRegistrationStore {
    /// Throws rather than returning nil so a caller can say which half is missing.
    ///
    /// Composes the `ClientIdentity` itself rather than calling ``signingIdentity()``, which
    /// would decode the file a second time; every dataset fetch, vault download and result
    /// upload mints one.
    func mgmtSession() throws -> MgmtSession {
        guard let registration = get() else {
            throw IdentityError.registrationMissing
        }
        guard let privateKey = getPrivateKey() else {
            throw IdentityError.privateKeyMissing
        }
        return MgmtSession(
            registration: registration,
            auth: ClientIdentity(clientId: registration.clientId, privateKeyHex: privateKey))
    }
}
