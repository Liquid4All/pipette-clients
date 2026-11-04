import Foundation

@testable import Pipette

/// Fails every request at once, as a device with no route to the Hub would. Stateless, so
/// it does not race the process-global state of `StubURLProtocol` or `ScriptedURLProtocol`.
nonisolated final class UnreachableURLProtocol: URLProtocol {
    override static func canInit(with request: URLRequest) -> Bool { true }
    override static func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func startLoading() {
        client?.urlProtocol(self, didFailWithError: URLError(.cannotConnectToHost))
    }
    override func stopLoading() {}

    static var httpClient: HttpClient {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [UnreachableURLProtocol.self]
        return HttpClient(userAgent: "pipette-test/0", configuration: configuration)
    }
}
