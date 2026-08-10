import CryptoKit
import Foundation
import Security

/// Manages private signing key, registration record, and token storage in the iOS Keychain.
nonisolated enum KeychainHelper {
    private static let service = "ai.liquid.liquid-pipette"
    private static let privateKeyAccount = "private_key_hex"
    private static let pendingPrivateKeyAccount = "pending_private_key_hex"
    private static let hfTokenAccount = "hf_token"
    private static let registrationAccount = "registration_json"

    /// Generate a signing keypair and stage the private key in the Keychain.
    /// Returns the public key hex string to send to the management server.
    static func generatePendingSigningKeyPair() -> String? {
        let privateKey = Curve25519.Signing.PrivateKey()
        let privateKeyHex = privateKey.rawRepresentation.hexEncodedString()
        guard saveString(privateKeyHex, account: pendingPrivateKeyAccount) else {
            return nil
        }
        return privateKey.publicKey.rawRepresentation.hexEncodedString()
    }

    /// Promote the staged private key after server registration succeeds.
    static func promotePendingPrivateKey() -> Bool {
        guard let pendingPrivateKey = loadString(account: pendingPrivateKeyAccount),
              savePrivateKey(pendingPrivateKey)
        else {
            return false
        }

        _ = deletePendingPrivateKey()
        return true
    }

    /// Delete any staged private key left by an interrupted registration.
    static func deletePendingPrivateKey() -> Bool {
        deleteItem(account: pendingPrivateKeyAccount)
    }

    /// Save the private signing key hex string to the Keychain.
    static func savePrivateKey(_ hex: String) -> Bool {
        saveString(hex, account: privateKeyAccount)
    }

    /// Load the private signing key from the Keychain. This is the loose→strict
    /// boundary: the stored hex string becomes a `PrivateKeyHex` here and stays
    /// strongly typed through the signing/submission paths.
    static func loadPrivateKey() -> PrivateKeyHex? {
        loadString(account: privateKeyAccount).map { PrivateKeyHex($0) }
    }

    /// Delete the private key from the Keychain.
    static func deletePrivateKey() -> Bool {
        deleteItem(account: privateKeyAccount)
    }

    /// The registration record, mirrored beside the signing key so both halves outlive an
    /// uninstall. Why it has to be both halves: ``RegistrationMirror``.
    static func saveRegistration(_ json: Data) -> Bool {
        saveData(json, account: registrationAccount)
    }

    static func loadRegistration() -> Data? {
        loadData(account: registrationAccount)
    }

    static func deleteRegistration() -> Bool {
        deleteItem(account: registrationAccount)
    }

    /// Save the Hugging Face access token to the Keychain.
    static func saveHfToken(_ token: AuthToken) -> Bool {
        saveString(token.value, account: hfTokenAccount)
    }

    /// Load the Hugging Face access token from the Keychain. Loose→strict boundary, as
    /// `loadPrivateKey` is for the signing key: the stored string becomes an `AuthToken`
    /// here, so every consumer downstream has a credential that is non-empty by type and
    /// redacts itself in logs. A stored value `AuthToken` rejects reads as absent — which
    /// is the honest answer, since an empty token authenticates nothing.
    static func loadHfToken() -> AuthToken? {
        loadString(account: hfTokenAccount).flatMap { try? AuthToken($0) }
    }

    /// Delete the Hugging Face access token from the Keychain.
    static func deleteHfToken() -> Bool {
        deleteItem(account: hfTokenAccount)
    }

    /// A plan-supplied token, keyed by the model's reference (the crate's per-source
    /// `reference()`). Keying by repo alone collided: two cells pulling different files
    /// out of one repo shared an entry, so the second claim's token replaced the first's.
    ///
    /// Internal rather than private so the derivation is assertable without a Keychain —
    /// a Simulator test host cannot store items. Why iOS persists a token at all, when
    /// the CLI stores none, is in `docs/pipette-ios/execution-alignment.md`.
    static func modelTokenAccount(_ model: Model) -> String? {
        let reference: String
        switch model {
        case let .ggufText(m): reference = m.source.reference
        // One repo, one credential: the main weights name the entry the projector shares.
        case let .ggufVision(m): reference = m.source.modelReference
        case let .mlx(m): reference = m.source.reference
        // Ships with the OS — nothing to fetch, so nothing to authenticate.
        case .appleFoundationText: return nil
        }
        return "\(hfTokenAccount):\(reference)"
    }

    static func saveHfToken(_ token: AuthToken, forModel model: Model) -> Bool {
        guard let account = modelTokenAccount(model) else { return false }
        return saveString(token.value, account: account)
    }

    /// Same loose→strict boundary as `loadHfToken()`.
    static func loadHfToken(forModel model: Model) -> AuthToken? {
        modelTokenAccount(model)
            .flatMap { loadString(account: $0) }
            .flatMap { try? AuthToken($0) }
    }

    /// Forget every stored model token. The CLI has no counterpart because it stores none
    /// — it re-reads `PIPETTE_HF_TOKEN` each invocation — so the whole lifecycle is iOS's
    /// to own, and `auth reset` is the point where "forget this device" has to include
    /// the credentials it accumulated.
    @discardableResult
    static func deleteAllModelHfTokens() -> Int {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecReturnAttributes as String: true,
            kSecMatchLimit as String: kSecMatchLimitAll,
        ]
        var result: CFTypeRef?
        guard SecItemCopyMatching(query as CFDictionary, &result) == errSecSuccess,
              let items = result as? [[String: Any]]
        else { return 0 }
        let prefix = "\(hfTokenAccount):"
        return items
            .compactMap { $0[kSecAttrAccount as String] as? String }
            .filter { $0.hasPrefix(prefix) }
            .filter { deleteItem(account: $0) }
            .count
    }

    // MARK: - Generic helpers

    private static func saveString(_ value: String, account: String) -> Bool {
        guard let data = value.data(using: .utf8) else { return false }
        return saveData(data, account: account)
    }

    private static func saveData(_ data: Data, account: String) -> Bool {
        let addQuery: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
            kSecValueData as String: data,
            kSecAttrAccessible as String: kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly,
        ]

        let addStatus = SecItemAdd(addQuery as CFDictionary, nil)
        if addStatus == errSecSuccess {
            return true
        }
        guard addStatus == errSecDuplicateItem else {
            return false
        }

        let updateQuery: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
        ]
        let update: [String: Any] = [
            kSecValueData as String: data,
        ]
        return SecItemUpdate(updateQuery as CFDictionary, update as CFDictionary) == errSecSuccess
    }

    private static func loadString(account: String) -> String? {
        loadData(account: account).flatMap { String(data: $0, encoding: .utf8) }
    }

    private static func loadData(account: String) -> Data? {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne,
        ]

        var result: AnyObject?
        let status = SecItemCopyMatching(query as CFDictionary, &result)

        guard status == errSecSuccess, let data = result as? Data else {
            return nil
        }
        return data
    }

    private static func deleteItem(account: String) -> Bool {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
        ]
        let status = SecItemDelete(query as CFDictionary)
        return status == errSecSuccess || status == errSecItemNotFound
    }
}

private extension Data {
    nonisolated func hexEncodedString() -> String {
        map { String(format: "%02x", $0) }.joined()
    }
}
