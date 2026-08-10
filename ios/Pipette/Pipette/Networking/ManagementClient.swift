import CryptoKit
import Foundation

enum ManagementClient {
    struct RegistrationResult {
        let clientId: ClientID
        let status: String
    }

    private static let userAgent = "pipette-ios"
    /// Empty JSON object body for POST/PUT endpoints that require a body.
    private static let emptyJSONObject = Data("{}".utf8)

    /// `POST /clients/register`. **Unauthenticated.**
    ///
    /// `device` carries the same profile + capability fields the server accepts
    /// on `PATCH /clients/me`; supplying them here establishes accurate matching
    /// input in one request, so a first-time client needs no follow-up patch
    /// before it is matchable (mgmt `client-integration.md` §2).
    ///
    /// Required rather than defaulted: registering with no matching input is the
    /// state this parameter exists to prevent, so there is no sensible default to
    /// fall back to. Pass an explicitly empty `DeviceProfileUpdate()` to opt out.
    static func register(
        serverUrl: ServerURL,
        organization: String,
        contactEmail: String,
        clientDetails: String,
        publicKeyHex: String,
        preauthKey: String? = nil,
        device: DeviceProfileUpdate
    ) async throws -> RegistrationResult {
        let body = RegisterRequest(
            publicKey: publicKeyHex,
            organization: organization,
            contactEmail: contactEmail,
            clientDetails: clientDetails,
            preauthKey: preauthKey,
            device: device
        )
        let response: RegisterResponse = try await request(
            serverUrl: serverUrl,
            path: "/clients/register",
            method: "POST",
            auth: nil,
            body: try Coding.encoder.encode(body)
        )
        return RegistrationResult(
            clientId: ClientID(response.clientId),
            status: response.status
        )
    }

    static func submitResult(
        serverUrl: ServerURL,
        auth: AuthIdentity,
        payloadJson: String
    ) async throws -> String {
        guard let body = payloadJson.data(using: .utf8) else {
            throw ManagementClientError.invalidPayloadEncoding
        }

        let responseData = try await requestData(
            serverUrl: serverUrl,
            path: "/benchmarks",
            method: "POST",
            auth: auth,
            body: body
        )
        guard let response = String(data: responseData, encoding: .utf8) else {
            throw ManagementClientError.invalidResponseEncoding
        }
        return response
    }

    // MARK: - Planner (claim / heartbeat / reclaim / profile)

    /// `POST /plans/claim` — lease the next eligible job, or `nil` on `204`.
    /// Throws on `403` (not approved) and other errors.
    static func claim(
        serverUrl: ServerURL,
        auth: AuthIdentity
    ) async throws -> ClaimedJob? {
        let response = try await requestResponse(
            serverUrl: serverUrl,
            path: "/plans/claim",
            method: "POST",
            auth: auth,
            body: emptyJSONObject
        )
        if response.statusCode == 204 { return nil }
        guard (200..<300).contains(response.statusCode) else {
            throw ManagementClientError.httpStatus(
                statusCode: response.statusCode,
                body: String(data: response.data, encoding: .utf8) ?? ""
            )
        }
        do {
            return try Coding.decoder.decode(ClaimedJob.self, from: response.data)
        } catch {
            throw ManagementClientError.decodeJson(error)
        }
    }

    /// `PUT /plans/{jobId}/heartbeat` — renew the lease. Empty body on success.
    static func heartbeat(
        serverUrl: ServerURL,
        auth: AuthIdentity,
        jobId: String
    ) async throws {
        try await requestEmpty(
            serverUrl: serverUrl,
            path: "/plans/\(jobId)/heartbeat",
            method: "PUT",
            auth: auth,
            body: emptyJSONObject
        )
    }

    /// `POST /plans/{jobId}/reclaim` — re-acquire a previously held lease.
    static func reclaim(
        serverUrl: ServerURL,
        auth: AuthIdentity,
        jobId: String
    ) async throws {
        try await requestEmpty(
            serverUrl: serverUrl,
            path: "/plans/\(jobId)/reclaim",
            method: "POST",
            auth: auth,
            body: emptyJSONObject
        )
    }

    /// `GET /clients/me` — authenticated client profile.
    static func me(
        serverUrl: ServerURL,
        auth: AuthIdentity
    ) async throws -> ClientProfile {
        try await request(
            serverUrl: serverUrl,
            path: "/clients/me",
            method: "GET",
            auth: auth,
            body: nil
        )
    }

    /// `PATCH /clients/me` — refresh device profile / capabilities.
    static func updateMe(
        serverUrl: ServerURL,
        auth: AuthIdentity,
        update: DeviceProfileUpdate
    ) async throws -> ClientProfile {
        try await request(
            serverUrl: serverUrl,
            path: "/clients/me",
            method: "PATCH",
            auth: auth,
            body: try Coding.encoder.encode(update)
        )
    }

    /// Submit a plan-attached failure (`message_type: "failure"`).
    static func submitFailure(
        serverUrl: ServerURL,
        auth: AuthIdentity,
        failure: FailureSubmission
    ) async throws {
        _ = try await requestData(
            serverUrl: serverUrl,
            path: "/benchmarks",
            method: "POST",
            auth: auth,
            body: try Coding.encoder.encode(failure)
        )
    }

    /// Submit an already-serialized plan-attached success payload (must carry `job_id`).
    static func submitPlanResult(
        serverUrl: ServerURL,
        auth: AuthIdentity,
        payloadJson: Data
    ) async throws {
        _ = try await requestData(
            serverUrl: serverUrl,
            path: "/benchmarks",
            method: "POST",
            auth: auth,
            body: payloadJson
        )
    }

    static func submitResultBatch(
        serverUrl: ServerURL,
        auth: AuthIdentity,
        payloadsJson: String
    ) async throws -> String {
        // Parse the pre-serialized payload array and rebuild the envelope with
        // JSONSerialization rather than string-concatenating into the JSON: a
        // concat silently produces malformed bodies if the blob is ever empty
        // or not the array the wire shape assumes. Wire shape is unchanged:
        // {"submissions": [...]}.
        guard let payloadsData = payloadsJson.data(using: .utf8),
              let submissions = try? JSONSerialization.jsonObject(with: payloadsData) as? [Any]
        else {
            throw ManagementClientError.invalidPayloadEncoding
        }
        let body = try JSONSerialization.data(withJSONObject: ["submissions": submissions])

        let responseData = try await requestData(
            serverUrl: serverUrl,
            path: "/benchmarks/batch",
            method: "POST",
            auth: auth,
            body: body
        )
        guard let response = String(data: responseData, encoding: .utf8) else {
            throw ManagementClientError.invalidResponseEncoding
        }
        return response
    }

    /// Outcome of a conditional `GET` (`If-None-Match`). `json` is the raw response
    /// body on `200`, `nil` on `304` (caller keeps its cached copy); `etag` is the
    /// server's content hash to echo back as `If-None-Match` next time.
    struct ConditionalGet {
        let json: Data?
        let etag: String?

        var notModified: Bool { json == nil }
    }

    /// Fetch the benchmark catalog (definitions only — no eval samples),
    /// ETag-conditional. The `GET /benchmarks` endpoints are public — no client id
    /// or signature — so this needs no registration. See `conditionalGet`.
    static func fetchBenchmarks(
        serverUrl: ServerURL,
        ifNoneMatch: String?
    ) async throws -> ConditionalGet {
        try await conditionalGet(serverUrl: serverUrl, path: "/benchmarks", ifNoneMatch: ifNoneMatch)
    }

    /// Fetch a single benchmark by id, ETag-conditional. Unlike the list, the
    /// per-id response includes the eval `samples`, so every benchmark must be
    /// fetched individually to be fully hydrated. Public — no auth. See
    /// `conditionalGet`.
    static func fetchBenchmark(
        serverUrl: ServerURL,
        benchmarkId: String,
        ifNoneMatch: String?
    ) async throws -> ConditionalGet {
        try await conditionalGet(
            serverUrl: serverUrl, path: "/benchmarks/\(benchmarkId)", ifNoneMatch: ifNoneMatch)
    }

    /// Unauthenticated `GET` that honors `If-None-Match`: sends the last-seen `etag`,
    /// and a `304` returns `notModified` (no body). Used for the public benchmark
    /// endpoints. (`requestData` can't be reused — it treats the `304` as a
    /// non-success status and throws.)
    private static func conditionalGet(
        serverUrl: ServerURL,
        path: String,
        ifNoneMatch: String?
    ) async throws -> ConditionalGet {
        let url = try endpoint(serverUrl: serverUrl, path: path).url
        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        request.setValue(userAgent, forHTTPHeaderField: "User-Agent")
        if let ifNoneMatch {
            request.setValue(ifNoneMatch, forHTTPHeaderField: "If-None-Match")
        }

        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw ManagementClientError.invalidResponse
        }
        let etag = http.value(forHTTPHeaderField: "ETag")
        if http.statusCode == 304 {
            return ConditionalGet(json: nil, etag: etag)
        }
        guard (200..<300).contains(http.statusCode) else {
            throw ManagementClientError.httpStatus(
                statusCode: http.statusCode,
                body: String(data: data, encoding: .utf8) ?? ""
            )
        }
        return ConditionalGet(json: data, etag: etag)
    }

    /// The signed management-auth headers for one request.
    ///
    /// `pathAndQuery` must be the request target the server receives — the base
    /// URL's own path prefix included, and the query string when there is one.
    /// The server verifies against its own `uri.path_and_query()`, so anything
    /// else fails to verify. `Endpoint` carries the right value.
    static func signedHeaders(
        auth: AuthIdentity,
        method: String,
        pathAndQuery: String,
        timestamp: String = Self.rfc3339Timestamp(),
        nonce: String = Self.generateNonce()
    ) throws -> [String: String] {
        let privateKeyData = try Data(hexEncoded: auth.privateKeyHex.value)
        let privateKey = try Curve25519.Signing.PrivateKey(rawRepresentation: privateKeyData)
        let payload = signedPayload(
            method: method,
            pathAndQuery: pathAndQuery,
            timestamp: timestamp,
            clientId: auth.clientId.value,
            nonce: nonce
        )
        let signature = try privateKey
            .signature(for: Data(payload.utf8))
            .hexEncodedString()
        return [
            "X-Client-Id": auth.clientId.value,
            "X-Timestamp": timestamp,
            "X-Nonce": nonce,
            "X-Signature": signature
        ]
    }

    /// The `v1` signed payload: six newline-separated fields — scheme tag,
    /// method, request target, timestamp, client id, nonce (mgmt
    /// `authentication.md` §2.1). Binding the method and target scopes a
    /// signature to that method and target; the nonce makes it single-use, so a
    /// captured signature cannot be replayed inside the freshness window. The
    /// request body is still not covered.
    static func signedPayload(
        method: String,
        pathAndQuery: String,
        timestamp: String,
        clientId: String,
        nonce: String
    ) -> String {
        "v1\n\(method)\n\(pathAndQuery)\n\(timestamp)\n\(clientId)\n\(nonce)"
    }

    /// A fresh per-request nonce: 16 CSPRNG bytes, lowercase hex.
    ///
    /// Hex rather than an arbitrary byte string on purpose. The nonce is a field
    /// in a newline-delimited payload, so a value carrying a newline could forge
    /// a field boundary and make two different requests hash to one payload; hex
    /// cannot. It also satisfies the server's non-empty and valid-UTF-8 rules by
    /// construction. 128 bits makes a collision across the fleet negligible, so
    /// the server's replay cache can reject a repeat without the client
    /// coordinating.
    ///
    /// `SystemRandomNumberGenerator`, which `random(in:)` uses, is documented as
    /// cryptographically secure on Apple platforms.
    static func generateNonce() -> String {
        Data((0..<16).map { _ in UInt8.random(in: .min ... .max) }).hexEncodedString()
    }

    private static func request<T: Decodable>(
        serverUrl: ServerURL,
        path: String,
        method: String,
        auth: AuthIdentity?,
        body: Data?
    ) async throws -> T {
        let data = try await requestData(
            serverUrl: serverUrl,
            path: path,
            method: method,
            auth: auth,
            body: body
        )
        do {
            return try Coding.decoder.decode(T.self, from: data)
        } catch {
            throw ManagementClientError.decodeJson(error)
        }
    }

    private struct HTTPResponse {
        let statusCode: Int
        let data: Data
    }

    private static func requestResponse(
        serverUrl: ServerURL,
        path: String,
        method: String,
        auth: AuthIdentity?,
        body: Data?
    ) async throws -> HTTPResponse {
        let endpoint = try endpoint(serverUrl: serverUrl, path: path)
        var request = URLRequest(url: endpoint.url)
        request.httpMethod = method
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        request.setValue(userAgent, forHTTPHeaderField: "User-Agent")

        if let body {
            request.httpBody = body
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        }

        if let auth {
            let headers = try signedHeaders(
                auth: auth,
                method: method,
                pathAndQuery: endpoint.pathAndQuery
            )
            for (name, value) in headers {
                request.setValue(value, forHTTPHeaderField: name)
            }
        }

        let (data, response) = try await URLSession.shared.data(
            for: request,
            delegate: Self.refuseRedirects
        )
        guard let http = response as? HTTPURLResponse else {
            throw ManagementClientError.invalidResponse
        }
        return HTTPResponse(statusCode: http.statusCode, data: data)
    }

    /// Stateless, so one instance serves every request.
    private static let refuseRedirects = RefuseRedirects()

    private static func requestEmpty(
        serverUrl: ServerURL,
        path: String,
        method: String,
        auth: AuthIdentity?,
        body: Data?
    ) async throws {
        let response = try await requestResponse(
            serverUrl: serverUrl,
            path: path,
            method: method,
            auth: auth,
            body: body
        )
        guard (200..<300).contains(response.statusCode) else {
            throw ManagementClientError.httpStatus(
                statusCode: response.statusCode,
                body: String(data: response.data, encoding: .utf8) ?? ""
            )
        }
    }

    private static func requestData(
        serverUrl: ServerURL,
        path: String,
        method: String,
        auth: AuthIdentity?,
        body: Data?
    ) async throws -> Data {
        let response = try await requestResponse(
            serverUrl: serverUrl,
            path: path,
            method: method,
            auth: auth,
            body: body
        )
        guard (200..<300).contains(response.statusCode) else {
            throw ManagementClientError.httpStatus(
                statusCode: response.statusCode,
                body: String(data: response.data, encoding: .utf8) ?? ""
            )
        }
        return response.data
    }

    /// A resolved endpoint: the URL to send, and the request target to sign.
    struct Endpoint {
        let url: URL
        /// The path the server receives — the base URL's own prefix included —
        /// plus the query string when there is one, percent-encoded exactly as
        /// it goes on the wire. This is what `signedHeaders` must cover.
        let pathAndQuery: String
    }

    static func endpoint(serverUrl: ServerURL, path: String) throws -> Endpoint {
        let trimmed = serverUrl.value.trimmingCharacters(in: .whitespacesAndNewlines)
        guard var components = URLComponents(string: trimmed),
              components.scheme != nil,
              components.host != nil
        else {
            throw ManagementClientError.invalidBaseURL(serverUrl.value)
        }

        let basePath = components.path.trimmingTrailingSlashes()
        components.path = basePath + path

        guard let url = components.url else {
            throw ManagementClientError.invalidBaseURL(serverUrl.value)
        }
        // Read back from the same components the URL was built from, so the
        // signed target and the sent target cannot drift.
        let encodedPath = components.percentEncodedPath
        let pathAndQuery = components.percentEncodedQuery.map { "\(encodedPath)?\($0)" } ?? encodedPath
        return Endpoint(url: url, pathAndQuery: pathAndQuery)
    }

    private static func rfc3339Timestamp() -> String {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime]
        formatter.timeZone = TimeZone(secondsFromGMT: 0)
        return formatter.string(from: Date())
    }
}

/// Task delegate that refuses redirects, so the `3xx` surfaces as the response
/// instead of being followed.
///
/// The `v1` signature covers the request target: following a redirect would
/// present a signature over the pre-redirect target and earn a `401`, and
/// `URLSession` forwards custom headers, `X-Signature` included, to the redirect
/// host.
///
/// Implements the completion-handler form rather than the `async` one: the
/// `async` variant makes SILGen crash emitting the method's ObjC thunk
/// (`emitObjCMethodThunk`) in a debug build.
private final class RefuseRedirects: NSObject, URLSessionTaskDelegate {
    func urlSession(
        _ session: URLSession,
        task: URLSessionTask,
        willPerformHTTPRedirection response: HTTPURLResponse,
        newRequest request: URLRequest,
        completionHandler: @escaping (URLRequest?) -> Void
    ) {
        completionHandler(nil)
    }
}

enum ManagementClientError: LocalizedError {
    case invalidBaseURL(String)
    case invalidPayloadEncoding
    case invalidPrivateKeyHex
    case invalidResponse
    case invalidResponseEncoding
    case httpStatus(statusCode: Int, body: String)
    case decodeJson(Error)

    var errorDescription: String? {
        switch self {
        case .invalidBaseURL(let value):
            return "Invalid management server URL: \(value)"
        case .invalidPayloadEncoding:
            return "Failed to encode result payload"
        case .invalidPrivateKeyHex:
            return "Invalid private key"
        case .invalidResponse:
            return "Invalid response from management server"
        case .invalidResponseEncoding:
            return "Management server returned non-UTF-8 response data"
        case .httpStatus(let statusCode, let body):
            if body.isEmpty {
                return "Management server returned HTTP \(statusCode)"
            }
            return "Management server returned HTTP \(statusCode): \(body)"
        case .decodeJson(let error):
            return "Failed to parse management server response: \(error.localizedDescription)"
        }
    }
}

// Internal (not private) so a test can pin the wire contract by encoding it
// directly; the app can only build one via `ManagementClient.register`.
struct RegisterRequest: Encodable {
    let publicKey: String
    let organization: String
    let contactEmail: String
    let clientDetails: String
    /// Optional pre-auth key that admits the client already `approved`. Omitted
    /// from the JSON when nil, so keyless registration's wire body carries no
    /// empty key. Transient — the secret is never persisted or logged.
    let preauthKey: String?
    /// Device profile + capabilities, encoded **flat** alongside the fields
    /// above rather than nested under a key of their own. Empty by default, in
    /// which case it contributes nothing and the wire body is byte-for-byte the
    /// original five-field shape.
    let device: DeviceProfileUpdate

    enum CodingKeys: String, CodingKey {
        case publicKey = "public_key"
        case organization
        case contactEmail = "contact_email"
        case clientDetails = "client_details"
        case preauthKey = "preauth_key"
    }

    /// Hand-written because Swift has no `#[serde(flatten)]`: passing `encoder`
    /// straight to `device` lets it write its own snake_case keys into *this*
    /// keyed container, producing the flat object the server expects. Anything
    /// added to `DeviceProfileUpdate` therefore reaches registration for free.
    ///
    /// `client_details` is the one key both halves can write, and the flattened
    /// half goes last — so its value would silently win. Registration's own
    /// argument owns that key, so the profile's copy is dropped here rather than
    /// at the call site: a keyed container has no duplicate-key error to catch,
    /// so getting this wrong would be invisible on the wire.
    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(publicKey, forKey: .publicKey)
        try container.encode(organization, forKey: .organization)
        try container.encode(contactEmail, forKey: .contactEmail)
        try container.encode(clientDetails, forKey: .clientDetails)
        try container.encodeIfPresent(preauthKey, forKey: .preauthKey)
        var device = device
        device.clientDetails = nil
        try device.encode(to: encoder)
    }
}

private struct RegisterResponse: Decodable {
    let clientId: String
    let status: String

    enum CodingKeys: String, CodingKey {
        case clientId = "client_id"
        case status
    }
}

private extension Data {
    init(hexEncoded hex: String) throws {
        let trimmed = hex.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.count.isMultiple(of: 2) else {
            throw ManagementClientError.invalidPrivateKeyHex
        }

        var bytes: [UInt8] = []
        bytes.reserveCapacity(trimmed.count / 2)

        var index = trimmed.startIndex
        while index < trimmed.endIndex {
            let next = trimmed.index(index, offsetBy: 2)
            guard let byte = UInt8(trimmed[index..<next], radix: 16) else {
                throw ManagementClientError.invalidPrivateKeyHex
            }
            bytes.append(byte)
            index = next
        }
        self.init(bytes)
    }

    func hexEncodedString() -> String {
        map { String(format: "%02x", $0) }.joined()
    }
}

private extension String {
    func trimmingTrailingSlashes() -> String {
        var result = self
        while result.last == "/" {
            result.removeLast()
        }
        return result
    }
}
