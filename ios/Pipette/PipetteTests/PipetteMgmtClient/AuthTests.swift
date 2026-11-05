import CryptoKit
import Foundation
import Testing

@testable import Pipette

/// Mirrors the crate's `auth.rs` tests.
@Suite
struct AuthTests {
    private func identity(_ clientId: String = "ev1_a3f8") throws -> (ClientIdentity, PublicKey) {
        let key = Curve25519.Signing.PrivateKey()
        let privateKey = try PrivateKey(key.rawRepresentation.hexEncodedString())
        let publicKey = try PublicKey(key.publicKey.rawRepresentation.hexEncodedString())
        return (ClientIdentity(clientId: try ClientId(clientId), privateKeyHex: privateKey), publicKey)
    }

    private func header(_ headers: [(name: String, value: String)], _ name: String) -> String? {
        headers.first { $0.name == name }?.value
    }

    /// The timestamp and nonce are read back off the headers, not supplied: the
    /// server verifies what the request carried, so a value that never reached a
    /// header would verify here and 401 in production.
    @Test func signedHeadersVerifyWithThePairedPublicKey() throws {
        let (auth, publicKeyHex) = try identity()

        let headers = try auth.signedHeaders(method: "GET", pathAndQuery: "/clients/me")

        #expect(header(headers, "X-Client-Id") == "ev1_a3f8")
        let timestamp = try #require(header(headers, "X-Timestamp"))
        let nonce = try #require(header(headers, "X-Nonce"))
        let payload = "v1\nGET\n/clients/me\n\(timestamp)\nev1_a3f8\n\(nonce)"

        let publicKeyBytes = try #require(Data(hexEncoded: publicKeyHex.value))
        let publicKey = try Curve25519.Signing.PublicKey(rawRepresentation: publicKeyBytes)
        let signatureHex = try #require(header(headers, "X-Signature"))
        let signature = try #require(Data(hexEncoded: signatureHex))
        #expect(publicKey.isValidSignature(signature, for: Data(payload.utf8)))
    }

    /// The public half is never stored, so this is how a caller recovers the
    /// value the server has on record.
    @Test func publicKeyIsDerivedFromThePrivateOne() throws {
        let (auth, publicKeyHex) = try identity()
        #expect(try PipetteMgmtClient.Auth.publicKeyHexFromPrivate(auth.privateKeyHex)
            == publicKeyHex)
    }

    /// The server rebuilds this string byte for byte: field order, the `v1` tag
    /// and the newlines are all load-bearing, or every signed request 401s.
    @Test func clientPayloadIsSixNewlineSeparatedFields() {
        #expect(
            PipetteMgmtClient.Auth.clientPayload(
                method: "GET",
                pathAndQuery: "/clients/me?page=2",
                timestamp: "2026-03-10T12:00:00Z",
                clientId: "ev1_a3f8",
                nonce: "0f1e2d3c4b5a69788796a5b4c3d2e1f0"
            ) == "v1\nGET\n/clients/me?page=2\n2026-03-10T12:00:00Z\nev1_a3f8\n0f1e2d3c4b5a69788796a5b4c3d2e1f0"
        )
    }

    /// `X-User-Key-Id` belongs to a user's `v2` signature; a client has to sign
    /// the `v1` scheme it is verified under.
    @Test func aClientSendsNoUserKeyIdHeader() throws {
        let (auth, _) = try identity()

        let headers = try auth.signedHeaders(method: "GET", pathAndQuery: "/clients/me")

        #expect(header(headers, "X-Client-Id") == "ev1_a3f8")
        #expect(header(headers, "X-User-Key-Id") == nil)
    }

    /// Fractional seconds, as the crate's `Rfc3339` renders `now_utc()`, and
    /// parseable as the server's own stamps are.
    @Test func aRequestIsStampedWithSubSecondPrecision() throws {
        let (auth, _) = try identity()

        let headers = try auth.signedHeaders(method: "GET", pathAndQuery: "/clients/me")

        let timestamp = try #require(header(headers, "X-Timestamp"))
        #expect(timestamp.contains("."), "\(timestamp)")
        #expect(timestamp.hasSuffix("Z"), "\(timestamp)")
        #expect((try? Timestamp(timestamp)) != nil, "\(timestamp)")
    }

    /// The server rejects an empty or repeated nonce, and a newline in it would
    /// forge a payload field boundary. Hex is never empty and never has one.
    @Test func generateNonceIsFreshHexOf32Chars() {
        let first = PipetteMgmtClient.Auth.generateNonce()

        #expect(first.count == 32)
        #expect(first.allSatisfy { $0.isHexDigit && !$0.isUppercase })
        #expect(first != PipetteMgmtClient.Auth.generateNonce())
    }

    /// The server's replay cache rejects a repeated nonce, even for an
    /// identical method and target.
    @Test func eachSignedRequestCarriesAFreshNonce() throws {
        let (auth, _) = try identity()

        let first = try auth.signedHeaders(method: "GET", pathAndQuery: "/clients/me")
        let second = try auth.signedHeaders(method: "GET", pathAndQuery: "/clients/me")

        #expect(header(first, "X-Nonce") != header(second, "X-Nonce"))
        #expect(header(first, "X-Signature") != header(second, "X-Signature"))
    }

    // MARK: - The timestamp

    /// A client stamping whole seconds would send a spelling no other client in
    /// the fleet does, over a field the signature covers.
    @Test func theStampIsRfc3339WithFractionalSeconds() throws {
        let stamp = PipetteMgmtClient.Auth.rfc3339Timestamp()

        #expect(
            stamp.wholeMatch(of: /\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z/) != nil,
            "\(stamp)")
        #expect(throws: Never.self) { try Timestamp(stamp) }
    }

    /// Checks the header, not just the helper: the signature covers the value
    /// the header carries.
    @Test func aSignedRequestStampsAFractionalSecondTimestamp() throws {
        let (auth, _) = try identity()
        let headers = try auth.signedHeaders(method: "GET", pathAndQuery: "/clients/me")

        let stamp = try #require(header(headers, "X-Timestamp"))
        #expect(
            stamp.wholeMatch(of: /\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z/) != nil,
            "\(stamp)")
    }

    /// The stamp comes from a `Sendable` format value; a shared
    /// `ISO8601DateFormatter` is what concurrent signing would break.
    @Test func concurrentStampsAreAllWellFormed() async throws {
        let stamps = await withTaskGroup(of: String.self, returning: [String].self) { group in
            for _ in 0..<64 {
                group.addTask { PipetteMgmtClient.Auth.rfc3339Timestamp() }
            }
            return await group.reduce(into: [String]()) { $0.append($1) }
        }

        #expect(stamps.count == 64)
        for stamp in stamps {
            #expect(throws: Never.self) { try Timestamp(stamp) }
        }
    }

    // MARK: - The signing key

    /// Named as a length error, as the crate's `signing_key` does. CryptoKit
    /// would report "the curve refused this key", which misleads.
    @Test func signingKeyRefusesASeedOfTheWrongLength() {
        do {
            _ = try PipetteMgmtClient.Auth.signingKey(seed: Data(repeating: 0, count: 16))
            Issue.record("expected a refusal")
        } catch {
            guard case .invalidPrivateKey(let reason) = error else {
                Issue.record("expected a private-key refusal, got \(error)")
                return
            }
            #expect(reason == .length(found: 16))
        }
    }

    @Test func signingKeyAcceptsATwoAndThirtyByteSeed() throws {
        let key = Curve25519.Signing.PrivateKey()
        let seed = key.rawRepresentation
        #expect(seed.count == privateKeyLengthBytes)
        #expect(try PipetteMgmtClient.Auth.signingKey(seed: seed).rawRepresentation == seed)
    }
}
