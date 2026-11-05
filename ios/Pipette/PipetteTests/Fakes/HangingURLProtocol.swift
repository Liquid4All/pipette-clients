import Foundation

/// Keeps every transfer in flight until the test disposes of it, so a test chooses when a
/// `.file` download changes state. `pause` can't do it: it refuses a `.queued` row, and an
/// MLX row cannot be paused at all.
nonisolated final class HangingURLProtocol: URLProtocol {
    override static func canInit(with request: URLRequest) -> Bool { true }
    override static func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func startLoading() {}
    override func stopLoading() {}

    /// Ephemeral so nothing touches a shared cache or cookie store.
    static var configuration: URLSessionConfiguration {
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [HangingURLProtocol.self]
        return config
    }
}
