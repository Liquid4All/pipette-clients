import CryptoKit
import Foundation
import Security

/// Manages private signing key, registration record, and token storage in the iOS Keychain.
nonisolated enum KeychainHelper {
    private static let service = "ai.liquid.liquid-pipette"
    private static let privateKeyAccount = "private_key_hex"
    private static let pendingPrivateKeyAccount = "pending_private_key_hex"
    private static let hfTokenAccount = "hf_token"
    private static let modelTokenPrefix = "\(hfTokenAccount):"
    private static let registrationAccount = "registration_json"

    /// Stage a fresh private key in the Keychain and return the public key hex to register.
    static func generatePendingSigningKeyPair() -> String? {
        let privateKey = Curve25519.Signing.PrivateKey()
        let privateKeyHex = privateKey.rawRepresentation.hexEncodedString()
        guard saveString(privateKeyHex, account: pendingPrivateKeyAccount) else {
            return nil
        }
        return privateKey.publicKey.rawRepresentation.hexEncodedString()
    }

    /// Nil when none is staged or the staged value cannot sign.
    ///
    /// Reads rather than promotes: only ``ClientRegistrationStore`` writes both the file and
    /// the Keychain copy, so promoting here would leave the file behind.
    static func loadPendingPrivateKey() -> PrivateKey? {
        loadString(account: pendingPrivateKeyAccount).flatMap { try? PrivateKey($0) }
    }

    static func deletePendingPrivateKey() -> Bool {
        deleteItem(account: pendingPrivateKeyAccount)
    }

    static func savePrivateKey(_ hex: String) -> Bool {
        saveString(hex, account: privateKeyAccount)
    }

    /// A stored value that cannot sign reads as no key, which is what the caller acts on
    /// anyway: the device must re-register.
    static func loadPrivateKey() -> PrivateKey? {
        loadString(account: privateKeyAccount).flatMap { try? PrivateKey($0) }
    }

    static func deletePrivateKey() -> Bool {
        deleteItem(account: privateKeyAccount)
    }

    /// Mirrored beside the signing key so both halves outlive an uninstall; see
    /// ``RegistrationMirror``.
    static func saveRegistration(_ json: Data) -> Bool {
        saveData(json, account: registrationAccount)
    }

    static func loadRegistration() -> Data? {
        loadData(account: registrationAccount)
    }

    static func deleteRegistration() -> Bool {
        deleteItem(account: registrationAccount)
    }

    static func saveHfToken(_ token: AuthToken) -> Bool {
        saveString(token.value, account: hfTokenAccount)
    }

    /// A stored value `AuthToken` rejects reads as absent: an empty token authenticates
    /// nothing.
    static func loadHfToken() -> AuthToken? {
        loadString(account: hfTokenAccount).flatMap { try? AuthToken($0) }
    }

    static func deleteHfToken() -> Bool {
        deleteItem(account: hfTokenAccount)
    }

    /// The Keychain account a plan-supplied token lands in, keyed by the model's reference
    /// rather than its repo: two cells pulling different files from one repo would otherwise
    /// overwrite each other's token.
    ///
    /// Internal so tests can assert it: a Simulator test host cannot store Keychain items.
    /// Why iOS persists tokens at all: `docs/pipette-ios/execution-alignment.md`.
    static func modelTokenAccount(_ model: Model) -> String? {
        let reference: String
        switch model {
        case let .ggufText(m): reference = m.source.reference
        // One repo, one credential: the main weights name the entry the projector shares.
        case let .ggufVision(m): reference = m.source.weightsReference
        // Same rule, and it covers the drafter too: a separate `draft` gguf comes out of
        // the same repo as the weights, so it is the same credential.
        case let .ggufVisionSpeculative(m): reference = m.source.vision.weightsReference
        case let .mlx(m): reference = m.source.reference
        case let .onnxGenai(m): reference = m.source.reference
        // Ships with the OS: nothing to fetch.
        case .appleFoundationText: return nil
        }
        return modelTokenAccount(reference: reference)
    }

    /// Shared by the plan and import paths so the two cannot write different accounts.
    private static func modelTokenAccount(reference: String) -> String {
        "\(modelTokenPrefix)\(reference)"
    }

    static func saveHfToken(_ token: AuthToken, forModel model: Model) -> Bool {
        guard let account = modelTokenAccount(model) else { return false }
        return saveString(token.value, account: account)
    }

    /// For `auth import`. The account is built from the reference, so a bundle can only
    /// reach the per-model family of Keychain accounts.
    static func saveModelHfToken(_ token: AuthToken, reference: String) -> Bool {
        saveString(token.value, account: modelTokenAccount(reference: reference))
    }

    static func loadHfToken(forModel model: Model) -> AuthToken? {
        modelTokenAccount(model)
            .flatMap { loadString(account: $0) }
            .flatMap { try? AuthToken($0) }
    }

    /// A failed query reads as no stored tokens; both callers print a count the operator
    /// can check.
    private static func modelTokenAccounts() -> [String] {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecReturnAttributes as String: true,
            kSecMatchLimit as String: kSecMatchLimitAll,
        ]
        var result: CFTypeRef?
        guard SecItemCopyMatching(query as CFDictionary, &result) == errSecSuccess,
              let items = result as? [[String: Any]]
        else { return [] }
        return items
            .compactMap { $0[kSecAttrAccount as String] as? String }
            .filter { $0.hasPrefix(modelTokenPrefix) }
    }

    /// Keyed by model reference, so no caller handles a Keychain account. A token whose read
    /// fails is left out; `auth export` prints the count, so a short bundle is visible.
    static func loadAllModelHfTokens() -> [String: String] {
        modelTokenAccounts().reduce(into: [:]) { tokens, account in
            guard let value = loadString(account: account) else { return }
            tokens[String(account.dropFirst(modelTokenPrefix.count))] = value
        }
    }

    /// No CLI counterpart: the CLI stores no token and re-reads `PIPETTE_HF_TOKEN` each
    /// invocation.
    @discardableResult
    static func deleteAllModelHfTokens() -> Int {
        modelTokenAccounts().filter { deleteItem(account: $0) }.count
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

    /// Nil both for no item and for an item not readable yet: before the first unlock after
    /// boot, `kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly` items return
    /// `errSecInteractionNotAllowed`.
    ///
    /// Still safe for `auth import`, which refuses over a key it finds: in that window the
    /// write fails too (`SecItemAdd`, and the `SecItemUpdate` fallback cannot re-encrypt
    /// without the class key), so an unreadable key is never overwritten.
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
