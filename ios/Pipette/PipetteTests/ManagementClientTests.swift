import CryptoKit
import Foundation
import Testing

@testable import Pipette

@MainActor
struct ManagementClientTests {
    @Test func signedHeadersVerifyWithGeneratedPublicKey() throws {
        let privateKey = Curve25519.Signing.PrivateKey()
        let privateKeyHex = privateKey.rawRepresentation.testHexEncodedString()
        let publicKeyHex = privateKey.publicKey.rawRepresentation.testHexEncodedString()
        let timestamp = "2026-06-03T18:05:00Z"
        let nonce = "0f1e2d3c4b5a69788796a5b4c3d2e1f0"

        let headers = try ManagementClient.signedHeaders(
            auth: AuthIdentity(clientId: ClientID("client-1"), privateKeyHex: PrivateKeyHex(privateKeyHex)),
            method: "GET",
            pathAndQuery: "/clients/me",
            timestamp: timestamp,
            nonce: nonce
        )

        #expect(privateKeyHex.count == 64)
        #expect(publicKeyHex.count == 64)
        #expect(headers["X-Client-Id"] == "client-1")
        #expect(headers["X-Timestamp"] == timestamp)
        #expect(headers["X-Nonce"] == nonce)

        let publicKey = try Curve25519.Signing.PublicKey(
            rawRepresentation: Data(testHexEncoded: publicKeyHex)
        )
        let signature = try Data(testHexEncoded: #require(headers["X-Signature"]))
        let payload = "v1\nGET\n/clients/me\n\(timestamp)\nclient-1\n\(nonce)"
        #expect(publicKey.isValidSignature(signature, for: Data(payload.utf8)))
    }

    // The server rejects an empty or repeated nonce, and reads the payload as
    // newline-delimited fields. Hex satisfies all three: never empty, never
    // carrying a newline that could forge a field boundary, and fresh per call.
    @Test func generateNonceIsFreshHexOf32Chars() {
        let first = ManagementClient.generateNonce()

        #expect(first.count == 32)
        #expect(first.allSatisfy { $0.isHexDigit && !$0.isUppercase })
        #expect(first != ManagementClient.generateNonce())
    }

    // A signature is single-use, so two requests from one identity must not
    // reuse a nonce even when the method, target, and timestamp all match.
    @Test func eachSignedRequestCarriesAFreshNonce() throws {
        let privateKeyHex = Curve25519.Signing.PrivateKey().rawRepresentation.testHexEncodedString()
        let auth = AuthIdentity(clientId: ClientID("client-1"), privateKeyHex: PrivateKeyHex(privateKeyHex))
        let timestamp = "2026-06-03T18:05:00Z"

        let first = try ManagementClient.signedHeaders(
            auth: auth, method: "GET", pathAndQuery: "/clients/me", timestamp: timestamp
        )
        let second = try ManagementClient.signedHeaders(
            auth: auth, method: "GET", pathAndQuery: "/clients/me", timestamp: timestamp
        )

        #expect(first["X-Nonce"] != second["X-Nonce"])
        #expect(first["X-Signature"] != second["X-Signature"])
    }

    // The payload is a byte-for-byte wire contract with the server, which
    // rebuilds the same string from the request it received and verifies against
    // it. Field order, the `v1` tag, and the newline delimiters are all
    // load-bearing: get any of them wrong and every authenticated request 401s.
    @Test func signedPayloadIsSixNewlineSeparatedFields() {
        #expect(
            ManagementClient.signedPayload(
                method: "GET",
                pathAndQuery: "/clients/me?page=2",
                timestamp: "2026-03-10T12:00:00Z",
                clientId: "ev1_a3f8",
                nonce: "0f1e2d3c4b5a69788796a5b4c3d2e1f0"
            ) == "v1\nGET\n/clients/me?page=2\n2026-03-10T12:00:00Z\nev1_a3f8\n0f1e2d3c4b5a69788796a5b4c3d2e1f0"
        )
    }

    // The signature must cover the target the *server* sees, so a base URL
    // carrying a path prefix has to appear in the signed payload — signing the
    // bare endpoint path would 401 every request against such a deployment.
    @Test func endpointSignsThePathPrefixCarriedByTheBaseURL() throws {
        let endpoint = try ManagementClient.endpoint(
            serverUrl: ServerURL("https://mgmt.example.com/api/"),
            path: "/clients/me"
        )
        #expect(endpoint.url.absoluteString == "https://mgmt.example.com/api/clients/me")
        #expect(endpoint.pathAndQuery == "/api/clients/me")
    }

    // A nil pre-auth key must drop the field entirely (not emit `null`), so
    // keyless registration's wire body is unchanged; a present key rides the
    // request verbatim under the snake_case `preauth_key`.
    @Test func registerRequestOmitsPreauthKeyWhenNil() throws {
        let json = try encodedFields(preauthKey: nil)
        #expect(json["preauth_key"] == nil)
        #expect(json["public_key"] as? String == "pk")
        #expect(json["client_details"] as? String == "iPhone")
    }

    @Test func registerRequestIncludesPreauthKeyWhenSet() throws {
        let json = try encodedFields(preauthKey: "preauth_abc.secret")
        #expect(json["preauth_key"] as? String == "preauth_abc.secret")
    }

    // An empty profile must contribute no keys at all, so a caller with nothing
    // to report still sends the original five-field body.
    @Test func registerRequestOmitsDeviceProfileWhenEmpty() throws {
        let json = try encodedFields(preauthKey: nil)
        #expect(json["device_name"] == nil)
        #expect(json["capabilities"] == nil)
        #expect(json.count == 4)
    }

    // The profile encodes flat alongside the credentials rather than nested
    // under a `device` key — that flat object is the shape the server accepts.
    @Test func registerRequestFlattensDeviceProfile() throws {
        let json = try encodedFields(
            preauthKey: nil,
            device: DeviceProfileUpdate(
                deviceName: "iPhone 16 Pro",
                deviceOsName: "iOS",
                deviceOsVersion: "26.0",
                capabilities: ["runtime:llama_cpp"]
            )
        )
        #expect(json["device"] == nil)
        #expect(json["device_name"] as? String == "iPhone 16 Pro")
        #expect(json["device_os_name"] as? String == "iOS")
        #expect(json["capabilities"] as? [String] == ["runtime:llama_cpp"])
        // Still carries the credentials it is flattened into.
        #expect(json["public_key"] as? String == "pk")
    }

    // `client_details` is the one key both the registration fields and the
    // flattened profile can write, and the flattened half encodes last — so
    // without the encoder dropping it, the profile's value would silently win.
    // A keyed container reports no duplicate-key error, so this is the only thing
    // standing between a caller and a wrong `client_details` on the wire.
    @Test func registerRequestClientDetailsArgumentWinsOverTheProfileCopy() throws {
        var device = DeviceProfileUpdate()
        device.clientDetails = "from-the-profile"
        device.deviceName = "iPhone 16 Pro"
        let json = try encodedFields(preauthKey: nil, device: device)
        #expect(json["client_details"] as? String == "iPhone")
        // The rest of the profile still rides along.
        #expect(json["device_name"] as? String == "iPhone 16 Pro")
    }

    // The real registration payload, end to end: whatever `ProfileReporter`
    // produces must not displace the caller's `client_details`.
    @Test func registerRequestWithARealProfileKeepsTheCallersClientDetails() throws {
        let json = try encodedFields(preauthKey: nil, device: ProfileReporter.profile())
        #expect(json["client_details"] as? String == "iPhone")
    }

    private func encodedFields(
        preauthKey: String?,
        device: DeviceProfileUpdate = DeviceProfileUpdate()
    ) throws -> [String: Any] {
        let request = RegisterRequest(
            publicKey: "pk",
            organization: "LiquidAI",
            contactEmail: "user@example.com",
            clientDetails: "iPhone",
            preauthKey: preauthKey,
            device: device
        )
        let data = try JSONEncoder().encode(request)
        return try #require(
            JSONSerialization.jsonObject(with: data) as? [String: Any]
        )
    }

    @Test func signedHeadersRejectInvalidPrivateKeyHex() {
        do {
            _ = try ManagementClient.signedHeaders(
                auth: AuthIdentity(clientId: ClientID("client-1"), privateKeyHex: PrivateKeyHex("not-hex")),
                method: "GET",
                pathAndQuery: "/clients/me",
                timestamp: "2026-06-03T18:05:00Z"
            )
            Issue.record("expected invalidPrivateKeyHex to be thrown")
        } catch let error as ManagementClientError {
            guard case .invalidPrivateKeyHex = error else {
                Issue.record("expected invalidPrivateKeyHex, got \(error)")
                return
            }
        } catch {
            Issue.record("expected ManagementClientError, got \(error)")
        }
    }
}

private extension Data {
    func testHexEncodedString() -> String {
        map { String(format: "%02x", $0) }.joined()
    }

    init(testHexEncoded hex: String) throws {
        guard hex.count.isMultiple(of: 2) else {
            throw ManagementClientError.invalidPrivateKeyHex
        }

        var bytes: [UInt8] = []
        bytes.reserveCapacity(hex.count / 2)
        var index = hex.startIndex
        while index < hex.endIndex {
            let next = hex.index(index, offsetBy: 2)
            guard let byte = UInt8(hex[index..<next], radix: 16) else {
                throw ManagementClientError.invalidPrivateKeyHex
            }
            bytes.append(byte)
            index = next
        }
        self.init(bytes)
    }
}
