import Foundation

// The management-client tests' stand-in for the crate's `httpmock` server.

nonisolated struct StubResponse {
    let status: Int
    let body: Data
    let headers: [String: String]
}

/// A lock rather than an actor, and `nonisolated` against the project's MainActor default:
/// the protocol answers on `URLSession`'s own queue, where there is nothing to await.
nonisolated final class StubRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var stored: [URLRequest] = []
    private var storedBodies: [Data] = []

    func record(_ request: URLRequest) {
        lock.lock()
        defer { lock.unlock() }
        stored.append(request)
        // `URLSession` hands the body over as a stream; drain it so it can be asserted on.
        if let body = request.httpBody ?? request.httpBodyStream.map(Self.drain) {
            storedBodies.append(body)
        }
    }

    var requests: [URLRequest] {
        lock.lock()
        defer { lock.unlock() }
        return stored
    }

    var bodies: [Data] {
        lock.lock()
        defer { lock.unlock() }
        return storedBodies
    }

    var methodsAndPaths: [String] {
        requests.map { "\($0.httpMethod ?? "?") \($0.url?.path ?? "?")" }
    }

    private static func drain(_ stream: InputStream) -> Data {
        stream.open()
        defer { stream.close() }
        var data = Data()
        var buffer = [UInt8](repeating: 0, count: 4096)
        while stream.hasBytesAvailable {
            let read = stream.read(&buffer, maxLength: buffer.count)
            guard read > 0 else { break }
            data.append(buffer, count: read)
        }
        return data
    }
}

/// The handler is type-level, because `URLSession` instantiates the protocol itself, so a
/// suite using this runs `.serialized`.
nonisolated final class StubURLProtocol: URLProtocol {
    nonisolated(unsafe) static var answer: (@Sendable (URLRequest) -> StubResponse)?

    override static func canInit(with request: URLRequest) -> Bool { true }
    override static func canonicalRequest(for request: URLRequest) -> URLRequest {
        request
    }

    override func startLoading() {
        guard let answer = Self.answer,
              let url = request.url,
              let reply = Optional(answer(request)),
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
