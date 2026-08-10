import Foundation

// Strongly-typed wrappers around the three string identifiers that flow through
// the management-client / sync / submission paths. They are passed positionally
// through injected closures (`ListFetcher`, `BatchSubmitter`, …), where bare
// `String`s in a row make transposing `clientId` and `privateKey` a silent
// runtime failure instead of a compile error. Single-property value types are
// effectively zero-cost (inline layout, no heap/ARC on the wrapper).
//
// `ClientID` and `PrivateKeyHex` additionally travel as one ``AuthIdentity``,
// the crate's shape — these are its two halves.
//
// `ServerURL`/`ClientID` are `StringNewtype`s — `Codable`-transparent (they
// encode/decode as the bare string, like `EvalId`), so an `IdentityRegistration`
// spells them as plain strings on disk. `PrivateKeyHex` is deliberately neither
// `Codable` nor printable, so the signing key can't be serialized or logged.

/// The management server base URL.
nonisolated struct ServerURL: StringNewtype, Hashable {
    let value: String
    init(_ value: String) { self.value = value }
}

/// The server-assigned client id sent as `X-Client-Id` on signed requests.
nonisolated struct ClientID: StringNewtype, Hashable {
    let value: String
    init(_ value: String) { self.value = value }
}

/// The Curve25519 signing key as hex. Not `Codable` (never serialized) and
/// rendered redacted (never logged) — the only way to read the key is the
/// explicit `value`.
nonisolated struct PrivateKeyHex: Hashable {
    let value: String
    init(_ value: String) { self.value = value }
}

extension PrivateKeyHex: CustomStringConvertible, CustomDebugStringConvertible {
    var description: String { "PrivateKeyHex(redacted)" }
    var debugDescription: String { "PrivateKeyHex(redacted)" }
}
