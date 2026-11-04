import Foundation
@testable import Pipette

/// ``StubURLProtocol``'s parallel-safe sibling: each ``serve(_:)`` mints a fresh host and
/// routes by it, so a route belongs to the one test that made it and suites need not run
/// `.serialized`.
nonisolated final class RoutedURLProtocol: URLProtocol {
    /// The `Int` is the request's 0-based position on this route, so a test can script
    /// "fail twice, then succeed".
    typealias Handler = @Sendable (URLRequest, Int) -> StubResponse

    private struct Route {
        let handler: Handler
        let recorder: StubRecorder
    }

    private static let lock = NSLock()
    nonisolated(unsafe) private static var routes: [String: Route] = [:]

    static func serve(
        _ reply: @escaping Handler
    ) throws -> (http: HttpClient, server: ServerUrl, seen: StubRecorder) {
        let host = "\(UUID().uuidString.lowercased()).test"
        let recorder = StubRecorder()
        lock.withLock { routes[host] = Route(handler: reply, recorder: recorder) }
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [RoutedURLProtocol.self]
        return (
            HttpClient(userAgent: "pipette-test/0", configuration: configuration),
            try ServerUrl("https://\(host)"),
            recorder
        )
    }

    override static func canInit(with request: URLRequest) -> Bool { true }
    override static func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        guard let url = request.url,
              let host = url.host,
              let route = Self.lock.withLock({ Self.routes[host] })
        else {
            client?.urlProtocol(self, didFailWithError: URLError(.cannotFindHost))
            return
        }
        route.recorder.record(request)
        let reply = route.handler(request, route.recorder.requests.count - 1)
        guard let response = HTTPURLResponse(
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

extension StubResponse {
    nonisolated static func reply(_ status: Int, _ body: String = "") -> StubResponse {
        StubResponse(status: status, body: Data(body.utf8), headers: [:])
    }

    /// A `200` carrying `body` under `etag`, as mgmt serves a benchmark.
    nonisolated static func reply(_ body: String, etag: String) -> StubResponse {
        StubResponse(status: 200, body: Data(body.utf8), headers: ["ETag": etag])
    }
}
