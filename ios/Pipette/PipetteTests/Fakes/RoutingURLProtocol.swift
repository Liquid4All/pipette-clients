import Foundation

@testable import Pipette

/// Answers each request from a table keyed by path, for a test whose point is which URL was
/// reached. An unrouted path gets `404` with the path in the body, so a wrong URL fails
/// naming it.
///
/// A `3xx` is never followed: `URLSession` ignores a redirect a custom `URLProtocol`
/// returns, and `urlProtocol(_:wasRedirectedTo:redirectResponse:)` leaves the task hung.
/// Redirect policy is tested on ``RefuseRedirects`` directly, in `HttpClientTests`.
///
/// The table is type-level, because `URLSession` instantiates the protocol itself, so a
/// suite using this runs `.serialized`.
nonisolated final class RoutingURLProtocol: URLProtocol {
    private static let lock = NSLock()
    nonisolated(unsafe) private static var routes: [String: StubResponse] = [:]
    nonisolated(unsafe) private static var recorder = StubRecorder()

    /// Call once per test, before the first request.
    static func serving(
        _ routes: [String: StubResponse]
    ) -> (http: HttpClient, seen: StubRecorder) {
        let recorder = StubRecorder()
        lock.withLock {
            Self.routes = routes
            Self.recorder = recorder
        }
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [RoutingURLProtocol.self]
        return (HttpClient(userAgent: "pipette-test/0", configuration: configuration), recorder)
    }

    override static func canInit(with request: URLRequest) -> Bool { true }
    override static func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        let path = request.url?.path ?? ""
        let reply = Self.lock.withLock { () -> StubResponse in
            Self.recorder.record(request)
            return Self.routes[path]
                ?? StubResponse(status: 404, body: Data("no route for \(path)".utf8), headers: [:])
        }
        guard let url = request.url,
              let response = HTTPURLResponse(
                url: url, statusCode: reply.status, httpVersion: "HTTP/1.1",
                headerFields: reply.headers)
        else {
            client?.urlProtocol(self, didFailWithError: URLError(.badServerResponse))
            return
        }
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        if !reply.body.isEmpty {
            client?.urlProtocol(self, didLoad: reply.body)
        }
        client?.urlProtocolDidFinishLoading(self)
    }

    override func stopLoading() {}
}
