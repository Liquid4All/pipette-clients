import Foundation

// The ids of the principals a plan can name: pipette-mgmt's
// `crates/pipette-plan-types/src/principal.rs`.

/// The crate's `ClientId`, assigned at registration.
///
/// Trimmed before validation: it rides in the `X-Client-Id` header and the signed payload
/// of every authenticated request, and a padded copy signs as a client the server does
/// not know.
nonisolated struct ClientId: ValidatedString {
    let value: String
    init(validated value: String) { self.value = value }
    static func validate(_ value: String) -> Bool { isIdCharset(value) }
    static func sanitize(_ value: String) -> String { rustTrim(value) }
    static let rejection = ModelError.invalidClientId
}

/// The crate's `VaultResourceId` (`vault_…`). The charset makes it safe as one segment of
/// a URL path or a store key.
nonisolated struct VaultResourceId: ValidatedString {
    let value: String
    init(validated value: String) { self.value = value }
    static func validate(_ value: String) -> Bool { isPrefixedId(value, prefix: "vault_") }
    static func sanitize(_ value: String) -> String { rustTrim(value) }
    static let rejection = ModelError.invalidVaultResourceId
}

/// The crate's `VaultPrincipalId` (`org_…` or `user_…`). Carried, never acted on: the
/// server decides who may fetch.
nonisolated struct VaultPrincipalId: ValidatedString {
    let value: String
    init(validated value: String) { self.value = value }
    static func validate(_ value: String) -> Bool {
        isPrefixedId(value, prefix: "org_") || isPrefixedId(value, prefix: "user_")
    }
    static func sanitize(_ value: String) -> String { rustTrim(value) }
    static let rejection = ModelError.invalidVaultPrincipalId
}

/// The crate's `check_prefixed_id`.
nonisolated private func isPrefixedId(_ value: String, prefix: String) -> Bool {
    isIdCharset(value) && value.hasPrefix(prefix)
}

/// The server's rule for an id.
nonisolated private func isIdCharset(_ value: String) -> Bool {
    value.wholeMatch(of: /[A-Za-z0-9_-]+/) != nil
}
