import Foundation

/// Answers each request from a scripted list, so a test can drive a `.file` transfer to
/// completion, to a terminal failure, or through retries and resumes without a network.
/// A suite that needs only a hang should use the stateless `HangingURLProtocol`.
///
/// A `200` with an ETag that breaks part-way gives resume data, and the resume asks for the
/// rest with `Range` and `If-Range`.
///
/// The script and the request log are process-global, as `URLProtocol` registration is,
/// so a suite using this runs `.serialized`.
nonisolated final class ScriptedURLProtocol: URLProtocol {
    enum Answer: Sendable {
        /// `200` carrying these bytes, which a download task writes to the temp file it
        /// hands the installer.
        case body(Data)
        /// A transport failure before any answer. `.badServerResponse` is permanent; a
        /// connectivity code retries or, offline, pauses the row.
        case failure(URLError.Code)
        /// This status with no body, carrying `headers` (a `Retry-After`, say).
        case status(Int, headers: [String: String] = [:])
        /// `200` for the whole file, or `206` from the offset a `Range` asks for. Ignores
        /// `If-Range`, as the Hub's CDN does. With `cutAfter`, the body stops after that many
        /// bytes and then the connection is reported lost.
        case file(Data, etag: String?, cutAfter: Int? = nil)
        /// `200` with the whole of `data` and `etag`, whatever the `Range` asks. An origin
        /// that obeys `If-Range` answers a resume this way after the file changes. With
        /// `cutAfter`, the body stops as it does for `file`.
        case wholeFile(Data, etag: String, cutAfter: Int? = nil)
        /// `200` with `etag` and the first `sent` bytes of `data`, and then nothing: a
        /// transfer that has moved some bytes and stays open until the test stops it.
        case stalledFile(Data, etag: String, sent: Int)
        /// No answer. The transfer stays open until the test cancels it.
        case hang
    }

    static let cutDelay: TimeInterval = 0.3

    private var cut: Timer?

    private static let lock = NSLock()
    nonisolated(unsafe) private static var answers: [Answer] = [.failure(.badServerResponse)]
    nonisolated(unsafe) private static var requests: [URLRequest] = []
    nonisolated(unsafe) private static var loadedTasks: [URLSessionTask] = []

    /// Clears the request log. Call before the transfer starts.
    static func serve(_ answer: Answer) {
        serve([answer])
    }

    /// The last answer repeats. Clears the request log. Call before the transfer starts.
    static func serve(_ answers: [Answer]) {
        lock.withLock {
            Self.answers = answers.isEmpty ? [.failure(.badServerResponse)] : answers
            Self.requests = []
            Self.loadedTasks = []
        }
    }

    /// Resume attempts included: the difference between recovering once and retrying forever.
    static var requestCount: Int {
        lock.withLock { requests.count }
    }

    /// Nil for a request from zero.
    static var ranges: [String?] {
        lock.withLock { requests.map { $0.value(forHTTPHeaderField: "Range") } }
    }

    /// So a test can call a delegate callback for a given request's task.
    static var tasks: [URLSessionTask] {
        lock.withLock { loadedTasks }
    }

    override static func canInit(with request: URLRequest) -> Bool { true }
    override static func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        let answer = Self.lock.withLock { () -> Answer in
            Self.requests.append(request)
            if let task { Self.loadedTasks.append(task) }
            return Self.answers.count > 1 ? Self.answers.removeFirst() : Self.answers[0]
        }
        // A scripted failure needs no URL: a resume from a placeholder blob arrives here
        // with none, and meets the failure the test scripted.
        if case let .failure(code) = answer {
            client?.urlProtocol(self, didFailWithError: URLError(code))
            return
        }
        guard let url = request.url else {
            client?.urlProtocol(self, didFailWithError: URLError(.badURL))
            return
        }
        switch answer {
        case .hang:
            break
        case let .body(data):
            respond(url: url, status: 200, headers: [:], body: data)
        case .failure:
            break
        case let .status(status, headers):
            respond(url: url, status: status, headers: headers, body: Data())
        case let .stalledFile(data, etag, sent):
            let headers = ["ETag": etag, "Content-Length": "\(data.count)",
                           "Accept-Ranges": "bytes"]
            guard let response = HTTPURLResponse(url: url, statusCode: 200,
                                                 httpVersion: "HTTP/1.1", headerFields: headers)
            else {
                client?.urlProtocol(self, didFailWithError: URLError(.badURL))
                return
            }
            client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
            client?.urlProtocol(self, didLoad: data.prefix(sent))
        case let .wholeFile(data, etag, cutAfter):
            respond(url: url, status: 200,
                    headers: ["ETag": etag, "Content-Length": "\(data.count)"], body: data,
                    cutAfter: cutAfter)
        case let .file(data, etag, cutAfter):
            let offset = request.value(forHTTPHeaderField: "Range")
                .flatMap { Int($0.dropFirst("bytes=".count).prefix { $0 != "-" }) } ?? 0
            var headers = [
                "Content-Length": "\(data.count - offset)",
                "Accept-Ranges": "bytes",
                "Last-Modified": "Wed, 21 Oct 2015 07:28:00 GMT",
            ]
            if let etag { headers["ETag"] = etag }
            if offset > 0 { headers["Content-Range"] = "bytes \(offset)-\(data.count - 1)/\(data.count)" }
            respond(url: url, status: offset > 0 ? 206 : 200, headers: headers,
                    body: data.subdata(in: offset..<data.count), cutAfter: cutAfter)
        }
    }

    private func respond(url: URL, status: Int, headers: [String: String], body: Data,
                         cutAfter: Int? = nil) {
        guard let response = HTTPURLResponse(url: url, statusCode: status,
                                             httpVersion: "HTTP/1.1", headerFields: headers)
        else {
            client?.urlProtocol(self, didFailWithError: URLError(.badURL))
            return
        }
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        guard let cutAfter else {
            client?.urlProtocol(self, didLoad: body)
            client?.urlProtocolDidFinishLoading(self)
            return
        }
        client?.urlProtocol(self, didLoad: body.prefix(cutAfter))
        // Delayed: URLSession counts the bytes on its own schedule, and a failure that
        // arrives first leaves resume data recording zero bytes, so the resume asks `bytes=0-`.
        let cut = Timer(timeInterval: Self.cutDelay, repeats: false) { [weak self] _ in
            guard let self else { return }
            self.client?.urlProtocol(self, didFailWithError: URLError(.networkConnectionLost))
        }
        RunLoop.current.add(cut, forMode: .common)
        self.cut = cut
    }

    override func stopLoading() {
        cut?.invalidate()
    }

    /// Ephemeral so nothing touches a shared cache or cookie store.
    static var configuration: URLSessionConfiguration {
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [ScriptedURLProtocol.self]
        return config
    }
}
