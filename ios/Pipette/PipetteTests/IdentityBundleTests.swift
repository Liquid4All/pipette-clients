import CryptoKit
import Foundation
import Testing

@testable import Pipette

/// Ports of pipette-cli's `identity/bundle.rs` tests, plus the interop: a token either
/// client prints restores on the other.
struct IdentityBundleTests {

    /// Any 32 bytes are a valid Ed25519 seed; the crate's test key.
    static let key = "aa11bb22cc33dd44ee55ff6600778899aa11bb22cc33dd44ee55ff6600778899"
    static let server = "https://collector.example.test"

    /// What the crate's `IdentityBundle::encode` prints for `ev1_abc` at
    /// `https://collector.example.test` under `key`: serde_json's compact rendering of
    /// `{"version","client_id","server_url","private_key_hex"}` in field order, base64url
    /// without padding. A literal, so a change to either encoder that breaks interop
    /// fails here rather than on an operator's device.
    static let crateToken = "eyJ2ZXJzaW9uIjoxLCJjbGllbnRfaWQiOiJldjFfYWJjIiwic2VydmVyX3VybCI6Imh0"
        + "dHBzOi8vY29sbGVjdG9yLmV4YW1wbGUudGVzdCIsInByaXZhdGVfa2V5X2hleCI6ImFhMTFiYjIyY2Mz"
        + "M2RkNDRlZTU1ZmY2NjAwNzc4ODk5YWExMWJiMjJjYzMzZGQ0NGVlNTVmZjY2MDA3Nzg4OTkifQ"

    private func bundle(userToken: String? = nil,
                        modelTokens: [String: String] = [:]) throws -> IdentityBundle {
        IdentityBundle(clientId: try ClientId("ev1_abc"), serverUrl: try ServerUrl(Self.server),
                       privateKeyHex: try PrivateKey(Self.key), userToken: userToken,
                       modelTokens: modelTokens)
    }

    /// A token from raw field values, for the cases the typed struct cannot express.
    private func token(_ object: [String: Any]) throws -> String {
        IdentityBundle.base64urlEncoded(try JSONSerialization.data(withJSONObject: object))
    }

    private func token(clientId: String = "ev1_abc", serverUrl: String = server,
                       privateKeyHex: String = key) throws -> String {
        try token(["version": 1, "client_id": clientId, "server_url": serverUrl,
                   "private_key_hex": privateKeyHex])
    }

    /// The token's JSON, as an object, for asserting on the fields a build writes.
    private func fields(of token: String) throws -> [String: Any] {
        var standard = token.replacingOccurrences(of: "-", with: "+")
            .replacingOccurrences(of: "_", with: "/")
        standard += String(repeating: "=", count: (4 - standard.count % 4) % 4)
        let data = try #require(Data(base64Encoded: standard))
        return try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any])
    }

    /// Why `token` is refused, or nil when it decodes.
    private func refusal(_ token: String) -> IdentityBundleError? {
        do {
            _ = try IdentityBundle.decode(token)
            return nil
        } catch {
            return error
        }
    }

    // MARK: - Interop

    @Test func aTokenTheCratePrintedRestoresHere() throws {
        let decoded = try IdentityBundle.decode(Self.crateToken)
        #expect(decoded == (try bundle()))
        #expect(decoded.tokenCount == 0)
    }

    /// With no tokens to carry, this build writes exactly the crate's four fields, so the
    /// crate reads the token as it reads its own.
    @Test func aTokenThisBuildPrintsCarriesTheCratesFields() throws {
        let written = try fields(of: try bundle().encode())
        #expect(Set(written.keys) == ["version", "client_id", "server_url", "private_key_hex"])
        #expect(written["version"] as? Int == 1)
        #expect(written["client_id"] as? String == "ev1_abc")
        #expect(written["server_url"] as? String == Self.server)
        #expect(written["private_key_hex"] as? String == Self.key)
    }

    /// The Hugging Face tokens are this client's extra, and ride beside the crate's fields
    /// rather than wrapping them: the crate's decoder ignores fields it does not know.
    @Test func theTokensRideBesideTheCratesFields() throws {
        let written = try fields(of: try bundle(
            userToken: "hf_user", modelTokens: ["mlx://repo=org/repo": "hf_model"]).encode())
        #expect(written["client_id"] as? String == "ev1_abc")
        #expect(written["user_token"] as? String == "hf_user")
        #expect(written["model_tokens"] as? [String: String] == ["mlx://repo=org/repo": "hf_model"])
    }

    /// The nested version-1 shape the Android client writes (as older iOS builds did). Read
    /// so a token from either keeps restoring; the machine-scoped fields are ignored.
    @Test func aTokenInTheEarlierNestedShapeStillRestores() throws {
        let nested = try token([
            "version": 1, "private_key_hex": Self.key,
            "registration": ["client_id": "ev1_abc", "server_url": Self.server,
                             "status": "approved", "organization": "Example"],
            "device_labels": ["device_name": "lab-07"],
            "settings": ["storage_quota_bytes": 12_000_000_000],
            "user_token": "hf_user", "model_tokens": [String: String](),
        ])
        #expect(try IdentityBundle.decode(nested) == (try bundle(userToken: "hf_user")))
    }

    // MARK: - The token

    @Test func roundTripsThroughAToken() throws {
        let source = try bundle(userToken: "hf_user",
                                modelTokens: ["mlx://repo=org/repo": "hf_model"])
        let decoded = try IdentityBundle.decode(try source.encode())
        #expect(decoded == source)
        #expect(decoded.version == IdentityBundle.currentVersion)
        // The per-model token's key is a reference holding `/`, `:` and `=`, which both a JSON
        // key and the token shape have to survive.
        #expect(decoded.tokenCount == 2)
    }

    /// The token has to survive a shell, a URL and a copy-paste as one word.
    @Test func tokenIsOneWordOfBase64url() throws {
        let token = try bundle(modelTokens: ["mlx://repo=org/repo": "hf_model"]).encode()
        #expect(!token.isEmpty)
        #expect(token.allSatisfy { $0.isLetter || $0.isNumber || $0 == "-" || $0 == "_" })
    }

    /// Surrounding whitespace is what a copy-paste carries.
    @Test func decodeToleratesSurroundingWhitespace() throws {
        #expect(try IdentityBundle.decode("  \(Self.crateToken)\n").clientId.value == "ev1_abc")
    }

    @Test func publicKeyIsDerivedFromThePrivateOne() throws {
        let raw = try #require(Data(hexEncoded: Self.key))
        let expected = try Curve25519.Signing.PrivateKey(rawRepresentation: raw)
            .publicKey.rawRepresentation.hexEncodedString()
        #expect(try bundle().publicKeyHex().description == expected)
    }

    // MARK: - Refusals

    @Test func rejectsATokenThatIsNotBase64url() {
        #expect(refusal("not a token!") == .notBase64url)
    }

    @Test func rejectsBase64urlThatIsNotABundle() {
        guard case .notAnIdentity = refusal(
            IdentityBundle.base64urlEncoded(Data(#"{"unrelated": true}"#.utf8)))
        else { Issue.record("expected notAnIdentity"); return }
    }

    /// A later build's record reports its version, not a missing field.
    @Test func rejectsAVersionThisBuildCannotRead() throws {
        let later = try token(["version": 2, "client_id": "ev1_abc", "server_url": Self.server,
                               "private_key_hex": Self.key, "something_new": 7])
        #expect(refusal(later) == .version(found: 2, expected: 1))
    }

    /// A token is only worth restoring if the key in it can sign.
    @Test(arguments: [String(repeating: "zz", count: 32), "aabb",
                      // `UInt8(_:radix:)` reads a sign, so these decode to 32 bytes unless
                      // the codec checks for hex digits first.
                      String(repeating: "+a", count: 32), String(repeating: "-0", count: 32)])
    func rejectsAKeyThatCannotSign(key: String) throws {
        guard case .privateKey = refusal(try token(privateKeyHex: key)) else {
            Issue.record("expected privateKey for \(key)"); return
        }
    }

    /// `ClientId` trims before it checks, so a blank id is as refused as an absent one,
    /// and none rides into the `X-Client-Id` header of a restored request.
    @Test(arguments: ["", "   ", "ev1.a"])
    func rejectsAnUnusableClientId(id: String) throws {
        guard case let .clientId(reason) = refusal(try token(clientId: id)),
              reason.contains("client id must match") else {
            Issue.record("expected clientId for \(id)"); return
        }
    }

    /// A server URL the client could not reach never reaches `registration.json`, and is
    /// named as the field that is wrong rather than as a malformed record.
    @Test(arguments: ["", "   ", "collector.example.test", "ftp://collector.example.test",
                      "https://collector.example.test?tenant=a"])
    func rejectsAServerUrlTheClientCouldNotReach(url: String) throws {
        guard case .serverUrl = refusal(try token(serverUrl: url)) else {
            Issue.record("expected serverUrl for \(url)"); return
        }
    }

    /// Padding around a field would otherwise reach `registration.json` and every signed
    /// request. The server URL's own sanitizer also drops the trailing slash.
    @Test func decodeTrimsTheFieldsItRestores() throws {
        let padded = try token(clientId: "  ev1_abc  ", serverUrl: " \(Self.server)/\n",
                               privateKeyHex: "  \(Self.key)\n")
        #expect(try IdentityBundle.decode(padded) == (try bundle()))
    }
}
