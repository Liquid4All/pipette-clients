import CryptoKit
import Foundation
import Testing

@testable import Pipette

/// Mirrors the crate's `client_api.rs` tests over a stubbed transport.
///
/// Serialized because the stub is a type-level handler.
@Suite(.serialized)
struct ClientApiTests {
    private func server() throws -> ServerUrl { try ServerUrl("https://mgmt.test") }

    private func auth() throws -> ClientIdentity {
        ClientIdentity(
            clientId: try ClientId("ev1_test"),
            privateKeyHex: try PrivateKey(String(repeating: "a3f8", count: 16))
        )
    }

    private func stub(
        _ reply: @escaping @Sendable (URLRequest) -> StubResponse
    ) -> (http: HttpClient, seen: StubRecorder) {
        let recorder = StubRecorder()
        StubURLProtocol.answer = { request in
            recorder.record(request)
            return reply(request)
        }
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [StubURLProtocol.self]
        return (HttpClient(userAgent: "pipette-test/0", configuration: configuration), recorder)
    }

    private func json(
        _ status: Int,
        _ body: String,
        headers: [String: String] = [:]
    ) -> @Sendable (URLRequest) -> StubResponse {
        { _ in StubResponse(status: status, body: Data(body.utf8), headers: headers) }
    }

    // MARK: - Claim

    @Test func claim200ReturnsJob() async throws {
        let body = """
            {"job_id":"job-1","benchmark_id":"prefill_throughput_256",
             "time_window":"PT5M","future_field":true}
            """
        let (http, seen) = stub(json(200, body))

        let job = try await PipetteMgmtClient.ClientApi.claim(
            server: try server(), http: http, auth: try auth())

        #expect(job?.jobId == "job-1")
        #expect(job?.timeWindow == "PT5M")
        #expect(seen.requests.first?.httpMethod == "POST")
    }

    /// `204` means no work, not a missing body.
    @Test func claim204ReturnsNil() async throws {
        let (http, _) = stub(json(204, ""))
        #expect(
            try await PipetteMgmtClient.ClientApi.claim(
                server: try server(), http: http, auth: try auth()) == nil)
    }

    /// The endpoint throws the client's own error type, so a caller needs no cast.
    @Test func claim403IsHttpStatus() async throws {
        let (http, _) = stub(json(403, "pending"))
        // Built outside the `do`: they throw their own error type, and a block throwing two
        // binds `error` as `any Error`.
        let (server, auth) = (try server(), try auth())

        do {
            _ = try await PipetteMgmtClient.ClientApi.claim(
                server: server, http: http, auth: auth)
            Issue.record("expected a failure")
        } catch {
            #expect(error.httpStatusCode == 403)
        }
    }

    /// As the crate's do: a `{}` would be a body the route never asked for. The paths also
    /// pin that a lease `job_id` travels raw, echoed to the server that minted it, where the
    /// catalog reads encode their id.
    @Test func thePlanVerbsSendNoBody() async throws {
        // Routed so the claim can answer `204` and need no body.
        let (http, seen) = RoutingURLProtocol.serving([
            "/plans/claim": StubResponse(status: 204, body: Data(), headers: [:]),
            "/plans/job-1/heartbeat": StubResponse(status: 200, body: Data(), headers: [:]),
            "/plans/job-1/reclaim": StubResponse(status: 200, body: Data(), headers: [:]),
        ])

        _ = try await PipetteMgmtClient.ClientApi.claim(
            server: try server(), http: http, auth: try auth())
        try await PipetteMgmtClient.ClientApi.heartbeat(
            server: try server(), http: http, auth: try auth(), jobId: "job-1")
        try await PipetteMgmtClient.ClientApi.reclaim(
            server: try server(), http: http, auth: try auth(), jobId: "job-1")

        #expect(seen.methodsAndPaths == [
            ("POST", "/plans/claim"),
            ("PUT", "/plans/job-1/heartbeat"),
            ("POST", "/plans/job-1/reclaim"),
        ].map { "\($0.0) \($0.1)" })
        #expect(seen.requests.allSatisfy { $0.httpBody == nil && $0.httpBodyStream == nil })
        #expect(seen.requests.allSatisfy { $0.value(forHTTPHeaderField: "Content-Type") == nil })
    }

    /// `409` means the job is someone else's now; the lease paths branch on it.
    @Test func reclaimUsesPostAndSurfaces409() async throws {
        let (http, seen) = stub(json(409, "taken"))
        let (server, auth) = (try server(), try auth())

        do {
            try await PipetteMgmtClient.ClientApi.reclaim(
                server: server, http: http, auth: auth, jobId: "job-1")
            Issue.record("expected a failure")
        } catch {
            #expect(error.httpStatusCode == 409)
        }
        #expect(seen.methodsAndPaths == ["POST /plans/job-1/reclaim"])
    }

    /// A followed redirect would 401 (the signature covers the original target) and leak
    /// `X-Signature` to the redirect host. The session's refusal to follow is pinned on
    /// ``RefuseRedirects`` in `HttpClientTests`: a `URLProtocol` stub cannot offer a
    /// redirect the session would act on. Surfacing the `3xx` keeps a redirecting deployment
    /// diagnosable.
    @Test func aRedirectSurfacesInsteadOfBeingFollowed() async throws {
        let (http, seen) = RoutingURLProtocol.serving([
            "/plans/claim": StubResponse(
                status: 308, body: Data(), headers: ["Location": "/plans/claim/"]),
            "/plans/claim/": StubResponse(status: 200, body: Data("{}".utf8), headers: [:]),
        ])

        let (server, auth) = (try server(), try auth())

        do {
            _ = try await PipetteMgmtClient.ClientApi.claim(
                server: server, http: http, auth: auth)
            Issue.record("expected a failure")
        } catch {
            #expect(error.httpStatusCode == 308)
        }
        #expect(seen.requests.count == 1, "the redirect target must never be reached")
    }

    // MARK: - Catalog

    /// A caller with no registration has no key, so a signature here would
    /// re-impose the registration the catalog reads exist without.
    @Test func theListReadSendsNoCredentials() async throws {
        let entry = """
            [{"benchmark_id":"p","benchmark_type":"prefill_throughput","parameter_prefill_tokens":512}]
            """
        let (http, seen) = stub(json(200, entry, headers: ["ETag": "\"v1\""]))

        let response = try await PipetteMgmtClient.ClientApi.listBenchmarksConditional(
            server: try server(), http: http, ifNoneMatch: nil)

        guard case .modified(let summaries, let etag) = response else {
            Issue.record("expected a body, got \(response)")
            return
        }
        #expect(summaries.map(\.benchmarkId) == ["p"])
        #expect(etag == (try EntityTag("\"v1\"")))
        let request = try #require(seen.requests.first)
        #expect(request.value(forHTTPHeaderField: "X-Signature") == nil)
        #expect(request.value(forHTTPHeaderField: "X-Client-Id") == nil)
        #expect(request.value(forHTTPHeaderField: "If-None-Match") == nil)
        // The stack names the build; no endpoint sets this.
        #expect(request.value(forHTTPHeaderField: "User-Agent") == "pipette-test/0")
    }

    @Test func theDetailReadWithoutAnIdentitySendsNoCredentials() async throws {
        let (http, seen) = stub(
            json(200, #"{"benchmark_id":"b1","benchmark_type":"prefill_throughput"}"#))

        let benchmark = try await PipetteMgmtClient.ClientApi.getBenchmark(
            server: try server(), http: http, auth: nil, benchmarkId: "b1")

        #expect(benchmark.benchmarkId == "b1")
        let request = try #require(seen.requests.first)
        #expect(request.url?.path == "/benchmarks/b1")
        #expect(request.value(forHTTPHeaderField: "X-Signature") == nil)
        #expect(request.value(forHTTPHeaderField: "X-Client-Id") == nil)
    }

    @Test func theDetailReadSignsWithAnIdentity() async throws {
        let (http, seen) = stub(
            json(200, #"{"benchmark_id":"b1","benchmark_type":"prefill_throughput"}"#))

        _ = try await PipetteMgmtClient.ClientApi.getBenchmark(
            server: try server(), http: http, auth: try auth(), benchmarkId: "b1")

        let request = try #require(seen.requests.first)
        #expect(request.url?.path == "/benchmarks/b1")
        #expect(request.value(forHTTPHeaderField: "X-Client-Id") == "ev1_test")
        #expect(request.value(forHTTPHeaderField: "X-Timestamp") != nil)
        #expect(request.value(forHTTPHeaderField: "X-Nonce") != nil)
        #expect(request.value(forHTTPHeaderField: "X-Signature") != nil)
    }

    /// A `304` has no body to decode.
    @Test func aConditionalDetailReadReportsNotModified() async throws {
        let (http, seen) = stub(json(304, "", headers: ["ETag": "\"v1\""]))

        let response = try await PipetteMgmtClient.ClientApi.getBenchmarkConditional(
            server: try server(), http: http, auth: nil, benchmarkId: "b1",
            ifNoneMatch: IfNoneMatch(try EntityTag("\"v1\"")))

        guard case .notModified(let etag) = response else {
            Issue.record("expected 304, got \(response)")
            return
        }
        #expect(etag == (try EntityTag("\"v1\"")))
        #expect(seen.requests.first?.value(forHTTPHeaderField: "If-None-Match") == "\"v1\"")
        #expect(seen.requests.first?.url?.path == "/benchmarks/b1")
    }

    /// Raw, an id carrying `?` would become a query and a different request.
    @Test func aBenchmarkIdTravelsAsOnePathSegment() async throws {
        let (http, seen) = stub(json(200, #"{"benchmark_id":"a?b","benchmark_type":"t"}"#))

        _ = try await PipetteMgmtClient.ClientApi.getBenchmark(
            server: try server(), http: http, auth: nil, benchmarkId: "a?b")

        let url = try #require(seen.requests.first?.url)
        #expect(url.query == nil)
        #expect(url.absoluteString.hasSuffix("/benchmarks/a%3Fb"))
    }

    // MARK: - Jobs

    /// The id is whatever the server minted; raw, one carrying `?` would become a query.
    @Test func theJobsReadEncodesTheIdItWasGiven() async throws {
        let (http, seen) = stub(json(200, #"{"status":"processed","metrics":null}"#))

        _ = try await PipetteMgmtClient.ClientApi.getJob(
            server: try server(), http: http, auth: try auth(), jobId: "a?b")

        let url = try #require(seen.requests.first?.url)
        #expect(url.query == nil)
        #expect(url.absoluteString.hasSuffix("/jobs/a%3Fb"))
    }

    // MARK: - Submissions

    @Test func submitResultPostsTheStoredPayloadAndReadsTheJobId() async throws {
        let (http, seen) = stub(json(202, #"{"job_id":"server-1"}"#))
        let payload = Data(#"{"benchmark_id":"b1"}"#.utf8)

        let response = try await PipetteMgmtClient.ClientApi.submitResult(
            server: try server(), http: http, auth: try auth(), payloadJson: payload)

        #expect(response.jobId == "server-1")
        let request = try #require(seen.requests.first)
        #expect(request.httpMethod == "POST")
        #expect(request.value(forHTTPHeaderField: "Content-Type") == "application/json")
        #expect(request.value(forHTTPHeaderField: "X-Signature") != nil)
    }

    /// Rebuilt from the stored payloads rather than concatenated, so an empty or
    /// non-object payload cannot produce a malformed body.
    @Test func submitResultBatchWrapsThePayloadsInTheEnvelope() async throws {
        let (http, seen) = stub(json(200, #"{"results":[{"index":0,"job_id":"server-1"}]}"#))

        let response = try await PipetteMgmtClient.ClientApi.submitResultBatch(
            server: try server(), http: http, auth: try auth(),
            payloads: [Data(#"{"benchmark_id":"b1"}"#.utf8)])

        #expect(response.results.map(\.jobId) == ["server-1"])
        let body = try #require(seen.bodies.first)
        let object = try #require(JSONSerialization.jsonObject(with: body) as? [String: Any])
        let submissions = try #require(object["submissions"] as? [[String: Any]])
        #expect(submissions.first?["benchmark_id"] as? String == "b1")
    }
}
