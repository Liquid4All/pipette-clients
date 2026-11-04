import CryptoKit
import Foundation
import Testing

@testable import Pipette

/// Mirrors the crate's `call.rs` tests.
@Suite
struct CallTests {
    private func header(_ request: PipetteMgmtClient.HttpRequest, _ name: String) -> String? {
        request.headers.first { $0.name == name }?.value
    }

    private func identity() throws -> (ClientIdentity, Curve25519.Signing.PrivateKey) {
        let key = Curve25519.Signing.PrivateKey()
        let auth = ClientIdentity(
            clientId: try ClientId("ev1_a3f8"),
            privateKeyHex: try PrivateKey(key.rawRepresentation.hexEncodedString())
        )
        return (auth, key)
    }

    private func dispatched(status: Int, body: String) -> PipetteMgmtClient.Call.Dispatched {
        PipetteMgmtClient.Call.Dispatched(
            response: PipetteMgmtClient.HttpResponse(
                status: status, headers: [], body: Data(body.utf8)),
            url: "u",
            method: "GET"
        )
    }

    /// The signature covers the target the server sees, so a base URL's path
    /// prefix must be signed too, or every request to such a deployment 401s.
    @Test func signedHeadersCoverTheMethodAndTheFullRequestTarget() throws {
        let (auth, key) = try identity()
        // Assembled as `dispatch` does, so `/api` comes from the server URL.
        let server = try ServerUrl("https://mgmt.example.com/api/")
        let url = "\(server.value)/clients/me"
        #expect(url == "https://mgmt.example.com/api/clients/me")
        #expect(try PipetteMgmtClient.Call.requestTarget(url) == "/api/clients/me")

        let request = try PipetteMgmtClient.Call.buildRequest(
            method: .patch, url: url, auth: auth, ifNoneMatch: nil, body: nil)

        #expect(header(request, "X-Client-Id") == "ev1_a3f8")
        let timestamp = try #require(header(request, "X-Timestamp"))
        // Read off the header: the server verifies the nonce the request
        // carried, so one that never reached the header would 401 in production.
        let nonce = try #require(header(request, "X-Nonce"))
        let payload = "v1\nPATCH\n/api/clients/me\n\(timestamp)\nev1_a3f8\n\(nonce)"
        let signatureHex = try #require(header(request, "X-Signature"))
        let signature = try #require(Data(hexEncoded: signatureHex))
        #expect(key.publicKey.isValidSignature(signature, for: Data(payload.utf8)))
    }

    /// The server's replay cache rejects a repeated nonce.
    @Test func eachSignedRequestCarriesAFreshNonce() throws {
        let (auth, _) = try identity()
        let url = "https://mgmt.example.com/clients/me"

        let first = try PipetteMgmtClient.Call.buildRequest(
            method: .get, url: url, auth: auth, ifNoneMatch: nil, body: nil)
        let second = try PipetteMgmtClient.Call.buildRequest(
            method: .get, url: url, auth: auth, ifNoneMatch: nil, body: nil)

        #expect(header(first, "X-Nonce") != header(second, "X-Nonce"))
    }

    @Test(arguments: [true, false])
    func contentTypeIsSentWithABodyAndOnlyWithOne(withBody: Bool) throws {
        let request = try PipetteMgmtClient.Call.buildRequest(
            method: .post, url: "https://mgmt.example.com/plans/claim", auth: nil,
            ifNoneMatch: nil, body: withBody ? Data("{}".utf8) : nil)
        #expect((header(request, "Content-Type") == "application/json") == withBody)
    }

    @Test func ifNoneMatchIsSentOnlyWhenAnEtagIsHeld() throws {
        let url = "https://mgmt.example.com/benchmarks"
        let conditional = try PipetteMgmtClient.Call.buildRequest(
            method: .get, url: url, auth: nil,
            ifNoneMatch: IfNoneMatch(try EntityTag("\"v1\"")), body: nil)
        let plain = try PipetteMgmtClient.Call.buildRequest(
            method: .get, url: url, auth: nil, ifNoneMatch: nil, body: nil)

        #expect(header(conditional, "If-None-Match") == "\"v1\"")
        #expect(header(plain, "If-None-Match") == nil)
    }

    /// `Range` and `If-Range` are not signed; the signature still verifies.
    @Test func aResumeSendsItsRangeOutsideTheSignature() throws {
        let (auth, key) = try identity()
        let resume = PipetteMgmtClient.ResumeFrom(offset: 100, ifRange: try EntityTag("\"pin\""))
        let request = try PipetteMgmtClient.Call.buildRequest(
            method: .get, url: "https://mgmt.example.com/vault/resources/vault_1/content",
            auth: auth, ifNoneMatch: nil, extra: resume.headers, body: nil)

        #expect(header(request, "Range") == "bytes=100-")
        #expect(header(request, "If-Range") == "\"pin\"")
        let timestamp = try #require(header(request, "X-Timestamp"))
        let nonce = try #require(header(request, "X-Nonce"))
        let payload =
            "v1\nGET\n/vault/resources/vault_1/content\n\(timestamp)\nev1_a3f8\n\(nonce)"
        let signature = try #require(header(request, "X-Signature").flatMap(Data.init(hexEncoded:)))
        #expect(key.publicKey.isValidSignature(signature, for: Data(payload.utf8)))
    }

    /// A caller with no registration has no key, so a signature here would
    /// re-impose the registration the catalog reads exist without.
    @Test func anUnauthenticatedReadSendsNoCredentials() throws {
        let request = try PipetteMgmtClient.Call.buildRequest(
            method: .get, url: "https://mgmt.example.com/benchmarks", auth: nil,
            ifNoneMatch: nil, body: nil)
        #expect(header(request, "X-Signature") == nil)
        #expect(header(request, "X-Client-Id") == nil)
    }

    @Test func requestTargetKeepsTheQueryString() throws {
        #expect(
            try PipetteMgmtClient.Call.requestTarget(
                "https://mgmt.example.com/clients/me?page=2") == "/clients/me?page=2")
    }

    /// A coding path alone points into JSON the operator never saw.
    @Test func aDecodeErrorQuotesTheBodyThatWouldNotParse() {
        let response = dispatched(
            status: 200, body: #"{"keys":[{"user_key_id":"userkey_9f2c"}]}"#)

        do {
            let _: WantedKeyId = try PipetteMgmtClient.Call.decodeSuccess(response)
            Issue.record("expected a decode error")
        } catch {
            let message = error.errorDescription ?? ""
            #expect(message.contains("key_id"), "\(message)")
            #expect(message.contains(#"; body: {"keys":[{"user_key_id":"userkey_9f2c"}]}"#),
                    "\(message)")
        }
    }

    @Test func aStatusErrorQuotesTheBodyBehindItsSeparator() {
        let error = PipetteMgmtClient.Call.statusError(dispatched(status: 409, body: "taken\n"))

        #expect(error.errorDescription == "request returned 409 for GET u: taken")
        #expect(PipetteMgmtClient.Call.statusError(dispatched(status: 503, body: ""))
            .errorDescription == "request returned 503 for GET u")
    }

    /// Counted in scalars, so a multi-byte one is never split.
    @Test func aLongBodyIsTruncatedAndMarked() {
        let limit = PipetteMgmtClient.Call.excerptMaxChars
        let excerpt = PipetteMgmtClient.Call.bodyExcerpt(
            Data(String(repeating: "é", count: limit + 10).utf8), separator: "; body: ")

        #expect(excerpt.hasSuffix("..."))
        #expect(excerpt.filter { $0 == "é" }.count == limit)
    }

    @Test func anEmptyBodyAddsNothingToTheMessage() {
        #expect(PipetteMgmtClient.Call.bodyExcerpt(Data(), separator: "; body: ") == "")
        #expect(PipetteMgmtClient.Call.bodyExcerpt(Data("   \n  ".utf8), separator: ": ") == "")
    }

    /// The planner worker retries on `isTransient`; a `4xx` is the server's final answer.
    @Test func httpStatus5xxIsTransient4xxIsNot() {
        let transient = PipetteMgmtClient.Error.httpStatus(
            method: "POST", url: "u", status: 503, body: "")
        let definitive = PipetteMgmtClient.Error.httpStatus(
            method: "POST", url: "u", status: 404, body: "")
        let offline = PipetteMgmtClient.Error.transport(
            method: "GET", url: "u", message: "reset")

        #expect(transient.isTransient)
        #expect(!definitive.isTransient)
        #expect(offline.isTransient)
        #expect(definitive.httpStatusCode == 404)
        #expect(offline.httpStatusCode == nil)
    }
}

/// At file scope and `nonisolated` so its `Decodable` conformance is usable
/// from the nonisolated pipeline.
private nonisolated struct WantedKeyId: Decodable {
    let keyId: String

    enum CodingKeys: String, CodingKey {
        case keyId = "key_id"
    }
}
