import Foundation

// The identity token `auth export` writes and `auth import` reads, in the wire format of
// `crates/pipette-cli/src/identity/bundle.rs`, so a token either client prints restores on
// the other.

/// A restorable client identity: the id the server assigned, the server that assigned it,
/// and the key that signs for it.
///
/// Only what the importing machine cannot rebuild from anywhere else: no endpoint looks a
/// `client_id` up by key, and the key is the credential. Device labels and the storage quota
/// stay behind: the importing device probes itself and pushes that profile on its first
/// worker startup.
///
/// The key sits in the Keychain under `kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly`, so
/// neither iCloud Keychain nor a restore carries it to new hardware; this token is the only
/// way to move the identity.
///
/// iOS adds the Hugging Face tokens it stores (`user_token`, `model_tokens`), optional on the
/// wire. The crate's decoder ignores unknown fields, so a desktop import still works, and a
/// device holding no tokens emits exactly the crate's four fields.
nonisolated struct IdentityBundle: Equatable, Sendable {
    /// The only version this build writes or reads.
    static let currentVersion = 1

    let version: Int
    let clientId: ClientId
    /// Typed, so a bad URL is refused while decoding rather than written to
    /// `registration.json` and discovered on the restored client's first request.
    let serverUrl: ServerUrl
    let privateKeyHex: PrivateKey
    /// The operator's own Hugging Face token.
    let userToken: String?
    /// A plan-supplied token per model, keyed by the model's reference. Not by Keychain
    /// account: `KeychainHelper` derives the account, so an imported bundle cannot name an
    /// arbitrary storage slot.
    let modelTokens: [String: String]

    init(clientId: ClientId, serverUrl: ServerUrl, privateKeyHex: PrivateKey,
         userToken: String? = nil, modelTokens: [String: String] = [:]) {
        self.version = Self.currentVersion
        self.clientId = clientId
        self.serverUrl = serverUrl
        self.privateKeyHex = privateKeyHex
        self.userToken = userToken
        self.modelTokens = modelTokens
    }

    /// The count `auth export` and `auth import` print.
    var tokenCount: Int {
        (userToken == nil ? 0 : 1) + modelTokens.count
    }

    /// Derived, never persisted: what an operator can check against the server's record
    /// without disclosing anything.
    func publicKeyHex() throws -> PublicKey {
        try PipetteMgmtClient.Auth.publicKeyHexFromPrivate(privateKeyHex)
    }

    /// Compact JSON, base64url without padding: no `+`, `/` or `=`, so it survives a shell,
    /// a URL and a copy-paste as a single word.
    func encode() throws -> String {
        let record = BundleRecord(
            version: version, clientId: clientId.value, serverUrl: serverUrl.value,
            privateKeyHex: privateKeyHex.value, userToken: userToken,
            modelTokens: modelTokens.isEmpty ? nil : modelTokens)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        return Self.base64urlEncoded(try encoder.encode(record))
    }

    /// Validates every field, so `auth import` writes nothing until this returns. The
    /// version is read first, so a later format reports its version rather than the fields
    /// this build failed to find.
    static func decode(_ token: String) throws(IdentityBundleError) -> IdentityBundle {
        guard let json = base64urlDecoded(token.trimmingCharacters(in: .whitespacesAndNewlines))
        else { throw .notBase64url }
        let probe: VersionProbe
        do {
            probe = try JSONDecoder().decode(VersionProbe.self, from: json)
        } catch {
            throw .notAnIdentity("\(error)")
        }
        guard probe.version == currentVersion else {
            throw .version(found: probe.version, expected: currentVersion)
        }
        let wire: BundleWire
        do {
            wire = try JSONDecoder().decode(BundleWire.self, from: json)
        } catch {
            throw .notAnIdentity("\(error)")
        }
        guard let rawClientId = wire.clientId ?? wire.registration?.clientId,
              let rawServerUrl = wire.serverUrl ?? wire.registration?.serverUrl
        else { throw .notAnIdentity("the record names no `client_id` and `server_url`") }
        let clientId: ClientId
        do {
            clientId = try ClientId(rawClientId)
        } catch {
            throw .clientId(error.localizedDescription)
        }
        let serverUrl: ServerUrl
        do {
            serverUrl = try ServerUrl(rawServerUrl)
        } catch {
            throw .serverUrl(error.localizedDescription)
        }
        let privateKey: PrivateKey
        do {
            privateKey = try PrivateKey(wire.privateKeyHex)
        } catch {
            throw .privateKey(error.message)
        }
        return IdentityBundle(clientId: clientId, serverUrl: serverUrl,
                              privateKeyHex: privateKey, userToken: wire.userToken,
                              modelTokens: wire.modelTokens ?? [:])
    }

    private struct VersionProbe: Decodable {
        let version: Int
    }

    private struct BundleRecord: Encodable {
        let version: Int
        let clientId: String
        let serverUrl: String
        let privateKeyHex: String
        let userToken: String?
        let modelTokens: [String: String]?

        enum CodingKeys: String, CodingKey {
            case version
            case clientId = "client_id"
            case serverUrl = "server_url"
            case privateKeyHex = "private_key_hex"
            case userToken = "user_token"
            case modelTokens = "model_tokens"
        }
    }

    /// Plain strings, so a bad field is refused under its own name rather than as a JSON
    /// error at a line and column the operator never saw. Unknown fields are ignored, as the
    /// crate does.
    ///
    /// Also reads the version-1 shape that nests `client_id` and `server_url` under
    /// `registration`, which the Android client still writes. Delete once no client does.
    private struct BundleWire: Decodable {
        let version: Int
        let clientId: String?
        let serverUrl: String?
        let privateKeyHex: String
        let userToken: String?
        let modelTokens: [String: String]?
        let registration: LegacyRegistration?

        struct LegacyRegistration: Decodable {
            let clientId: String
            let serverUrl: String

            enum CodingKeys: String, CodingKey {
                case clientId = "client_id"
                case serverUrl = "server_url"
            }
        }

        enum CodingKeys: String, CodingKey {
            case version
            case clientId = "client_id"
            case serverUrl = "server_url"
            case privateKeyHex = "private_key_hex"
            case userToken = "user_token"
            case modelTokens = "model_tokens"
            case registration
        }
    }

    /// Internal so the tests build tokens with this rather than a copy that could drift.
    static func base64urlEncoded(_ data: Data) -> String {
        data.base64EncodedString()
            .replacingOccurrences(of: "+", with: "-")
            .replacingOccurrences(of: "/", with: "_")
            .replacingOccurrences(of: "=", with: "")
    }

    private static func base64urlDecoded(_ token: String) -> Data? {
        var standard = token
            .replacingOccurrences(of: "-", with: "+")
            .replacingOccurrences(of: "_", with: "/")
        standard += String(repeating: "=", count: (4 - standard.count % 4) % 4)
        return Data(base64Encoded: standard)
    }
}

/// The crate's `Error::Bundle*` cases, with its wording: `auth import` prints these to the
/// operator.
nonisolated enum IdentityBundleError: LocalizedError, Equatable {
    case notBase64url
    case notAnIdentity(String)
    case version(found: Int, expected: Int)
    case clientId(String)
    case serverUrl(String)
    case privateKey(String)

    var errorDescription: String? {
        switch self {
        case .notBase64url:
            "the bundle is not base64url; pass the whole token `auth export` printed"
        case let .notAnIdentity(reason):
            "the bundle decoded but is not an identity record: \(reason)"
        case let .version(found, expected):
            "bundle version \(found) is not readable by this build, which reads "
                + "version \(expected)"
        case let .clientId(reason):
            "the bundle's `client_id` names no client to restore: \(reason)"
        case let .serverUrl(reason):
            "the bundle's server URL is unusable: \(reason)"
        case let .privateKey(reason):
            "the bundle's signing key is unusable: \(reason)"
        }
    }
}
