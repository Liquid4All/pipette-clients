import Foundation

/// What signs a request to the management server — the crate's `AuthIdentity`
/// (`pipette-mgmt-client/src/auth.rs:7`).
///
/// The two halves live in different stores (the client id in `registration.json`, the
/// key in the Keychain) and are useless apart, so they travel as one value that every
/// authenticated `ManagementClient` call takes.
///
/// Not `Codable`: `PrivateKeyHex` deliberately is not, so this can never be persisted
/// or logged.
nonisolated struct AuthIdentity: Hashable, Sendable {
    let clientId: ClientID
    let privateKeyHex: PrivateKeyHex
}
