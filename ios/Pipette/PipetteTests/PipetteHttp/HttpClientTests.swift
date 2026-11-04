import Foundation
import Testing

@testable import Pipette

/// Ports of the `pipette-http` crate's tests that have an iOS counterpart.
///
/// Serialized because the stub answers through a type-level handler.
@Suite(.serialized)
struct HttpClientTests {
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

    // MARK: - The builder

    @Test func theBuilderDefaultsTheConnectTimeoutAndLeavesTheRequestOneOpen() {
        let plain = HttpClient(userAgent: "pipette-test/0")
        #expect(plain.client.configuration.timeoutIntervalForRequest
            == HttpClient.defaultConnectTimeout)
        // Nothing imposed: the crate's `timeout(None)`.
        #expect(plain.client.configuration.timeoutIntervalForResource
            == URLSessionConfiguration.default.timeoutIntervalForResource)
    }

    @Test func theBuilderAppliesAnOverallRequestTimeout() {
        let timed = HttpClient.builder(userAgent: "pipette-test/0").requestTimeout(5).build()
        #expect(timed.client.configuration.timeoutIntervalForResource == 5)
        #expect(HttpClient.withRequestTimeout(userAgent: "pipette-test/0", timeout: 5)
            .client.configuration.timeoutIntervalForResource == 5)
    }

    @Test func theBuilderKeepsAConnectTimeoutWhenTheRequestOneIsCleared() {
        let client = HttpClient.builder(userAgent: "pipette-test/0")
            .connectTimeout(5)
            .requestTimeout(9)
            .noRequestTimeout()
            .build()
        #expect(client.client.configuration.timeoutIntervalForRequest == 5)
        #expect(client.client.configuration.timeoutIntervalForResource
            == URLSessionConfiguration.default.timeoutIntervalForResource)
    }

    @Test func aBuilderIsAValueAndDoesNotLeakBetweenClients() {
        let base = HttpClient.builder(userAgent: "pipette-test/0")
        let timed = base.requestTimeout(5).build()
        let plain = base.build()
        #expect(timed.client.configuration.timeoutIntervalForResource == 5)
        #expect(plain.client.configuration.timeoutIntervalForResource
            == URLSessionConfiguration.default.timeoutIntervalForResource)
    }

    /// The header is on the session configuration, not on a request.
    @Test func bothSessionsNameTheBuild() async throws {
        let (http, seen) = stub { _ in
            StubResponse(status: 200, body: Data("{}".utf8), headers: [:])
        }
        _ = try await http.client.data(from: #require(URL(string: "https://mgmt.test/a")))
        _ = try await http.clientNoRedirects.data(
            from: #require(URL(string: "https://mgmt.test/b")))
        #expect(seen.requests.count == 2)
        #expect(seen.requests.allSatisfy {
            $0.value(forHTTPHeaderField: "User-Agent") == "pipette-test/0"
        })
    }

    // MARK: - Redirects

    /// Following a redirect on a signed request would forward `X-Signature` to the redirect
    /// host. Asserted on the delegate rather than through a stub: `URLSession` ignores a `3xx`
    /// a custom `URLProtocol` returns, so a stubbed redirect would pass whatever the policy.
    @Test func onlyTheRedirectRefusingSessionCarriesTheRefusingDelegate() {
        let http = HttpClient(userAgent: "pipette-test/0")
        #expect(http.client.delegate == nil)
        #expect(http.clientNoRedirects.delegate is RefuseRedirects)
    }

    @Test func refuseRedirectsAnswersWithNoNewRequest() async throws {
        let from = try #require(URL(string: "https://mgmt.test/from"))
        let to = try #require(URL(string: "https://mgmt.test/to"))
        let response = try #require(
            HTTPURLResponse(
                url: from, statusCode: 308, httpVersion: "HTTP/1.1",
                headerFields: ["Location": "/to"]))
        let session = URLSession(configuration: .ephemeral)
        defer { session.invalidateAndCancel() }

        let next: URLRequest? = await withCheckedContinuation { continuation in
            RefuseRedirects().urlSession(
                session,
                task: session.dataTask(with: from),
                willPerformHTTPRedirection: response,
                newRequest: URLRequest(url: to),
                completionHandler: { continuation.resume(returning: $0) })
        }

        #expect(next == nil)
    }

    // MARK: - The JSON helper

    @Test func jsonRequestDecodesASuccessAndAsksForJson() async throws {
        let (http, seen) = stub { _ in
            StubResponse(status: 200, body: Data(#"{"job_id":"server-1"}"#.utf8), headers: [:])
        }

        let response: SubmitResponse = try await http.jsonRequest(
            method: "GET", url: "https://mgmt.test/jobs/1", headers: nil, body: nil)

        #expect(response.jobId == "server-1")
        let request = try #require(seen.requests.first)
        #expect(request.value(forHTTPHeaderField: "Accept") == "application/json")
        #expect(request.httpMethod == "GET")
        #expect(request.value(forHTTPHeaderField: "Content-Type") == nil)
    }

    @Test func jsonRequestSendsTheBodyItsContentTypeAndTheCallersHeaders() async throws {
        let (http, seen) = stub { _ in
            StubResponse(status: 200, body: Data(#"{"job_id":"j"}"#.utf8), headers: [:])
        }

        let _: SubmitResponse = try await http.jsonRequest(
            method: "POST", url: "https://mgmt.test/benchmarks",
            headers: [(name: "X-Trace", value: "t1")], body: Data("{}".utf8))

        let request = try #require(seen.requests.first)
        #expect(request.value(forHTTPHeaderField: "Content-Type") == "application/json")
        #expect(request.value(forHTTPHeaderField: "X-Trace") == "t1")
        #expect(seen.bodies.first == Data("{}".utf8))
    }

    @Test func jsonRequestQuotesTheStatusTheUrlAndTheBody() async throws {
        let (http, _) = stub { _ in
            StubResponse(status: 503, body: Data("upstream down".utf8), headers: [:])
        }

        do {
            let _: SubmitResponse = try await http.jsonRequest(
                method: "GET", url: "https://mgmt.test/jobs/1", headers: nil, body: nil)
            Issue.record("expected a failure")
        } catch {
            let message = error.localizedDescription
            #expect(message.contains("503"), "\(message)")
            #expect(message.contains("https://mgmt.test/jobs/1"), "\(message)")
            #expect(message.contains("upstream down"), "\(message)")
        }
    }

    /// And names the URL, so the reader knows which response it was.
    @Test func jsonRequestReportsAnUnreadableBodyAsAParseFailure() async throws {
        let (http, _) = stub { _ in
            StubResponse(status: 200, body: Data(#"{"not_a_job":1}"#.utf8), headers: [:])
        }

        do {
            let _: SubmitResponse = try await http.jsonRequest(
                method: "GET", url: "https://mgmt.test/jobs/1", headers: nil, body: nil)
            Issue.record("expected a failure")
        } catch {
            guard case .json(let message) = error else {
                Issue.record("expected a parse failure, got \(error)")
                return
            }
            #expect(message.contains("https://mgmt.test/jobs/1"), "\(message)")
        }
    }
}
