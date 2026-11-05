import Foundation

// The request pipeline every endpoint call runs through: the crate's `call.rs`.
// Signing, JSON coding, conditional requests and status handling live here once, so
// ``PipetteMgmtClient/ClientApi`` holds one function per endpoint. Stateless: the
// ``ServerUrl`` and ``HttpClient`` are passed into every call.

nonisolated extension PipetteMgmtClient {
    enum ConditionalResponse<T> {
        case modified(value: T, etag: EntityTag?)
        case notModified(etag: EntityTag?)

        var etag: EntityTag? {
            switch self {
            case .modified(_, let etag), .notModified(let etag): return etag
            }
        }
    }

    /// If the object no longer has the `ifRange` `ETag`, the server sends it whole.
    struct ResumeFrom {
        let offset: UInt64
        let ifRange: EntityTag

        var headers: [(name: String, value: String)] {
            [(name: "Range", value: "bytes=\(offset)-"), (name: "If-Range", value: ifRange.value)]
        }
    }

    enum Call {
        /// One exchange, with what it takes to describe it in an error.
        struct Dispatched {
            let response: HttpResponse
            let url: String
            let method: String
        }

        static func sendJson<T: Decodable, B: Encodable>(
            server: ServerUrl,
            http: HttpClient,
            method: HttpMethod,
            path: String,
            auth: (any SignsRequest)?,
            body: B
        ) async throws(PipetteMgmtClient.Error) -> T {
            let encoded: Data
            do {
                encoded = try Coding.encoder.encode(body)
            } catch {
                throw .encodeJson(error)
            }
            return try await sendJson(
                server: server, http: http, method: method, path: path, auth: auth,
                bodyJson: encoded)
        }

        /// For a body that is already JSON: payloads read off disk go out as is,
        /// since a round trip through `Encodable` could only alter them.
        static func sendJson<T: Decodable>(
            server: ServerUrl,
            http: HttpClient,
            method: HttpMethod,
            path: String,
            auth: (any SignsRequest)?,
            bodyJson: Data
        ) async throws(PipetteMgmtClient.Error) -> T {
            try decodeSuccess(
                await dispatch(
                    server: server, http: http, method: method, path: path, auth: auth,
                    body: bodyJson))
        }

        static func getJson<T: Decodable>(
            server: ServerUrl,
            http: HttpClient,
            path: String,
            auth: (any SignsRequest)?
        ) async throws(PipetteMgmtClient.Error) -> T {
            try decodeSuccess(
                await dispatch(
                    server: server, http: http, method: .get, path: path, auth: auth, body: nil))
        }

        static func getJsonConditional<T: Decodable>(
            server: ServerUrl,
            http: HttpClient,
            path: String,
            auth: (any SignsRequest)?,
            ifNoneMatch: IfNoneMatch?
        ) async throws(PipetteMgmtClient.Error) -> ConditionalResponse<T> {
            let dispatched = try await dispatch(
                server: server, http: http, method: .get, path: path, auth: auth,
                ifNoneMatch: ifNoneMatch, body: nil)
            let etag = dispatched.response.header("etag").flatMap(EntityTag.fromHeaderValue)
            if dispatched.response.status == 304 {
                return .notModified(etag: etag)
            }
            return .modified(value: try decodeSuccess(dispatched), etag: etag)
        }

        /// The response unread, for a caller that must see the status first
        /// (`POST /plans/claim` tells a `200` from a `204`).
        static func send(
            server: ServerUrl,
            http: HttpClient,
            method: HttpMethod,
            path: String,
            auth: (any SignsRequest)?
        ) async throws(PipetteMgmtClient.Error) -> Dispatched {
            try await dispatch(
                server: server, http: http, method: method, path: path, auth: auth, body: nil)
        }

        /// A `GET` whose body goes to `sink` as it arrives, for an artifact of many
        /// gigabytes. `extra` headers (such as `Range`) are outside the signature.
        ///
        /// A non-2xx body is an error message, read into memory and thrown as
        /// `.httpStatus`. If the sink refuses or stops the transfer, the call throws
        /// `.transport` and the sink holds the reason.
        static func getStream(
            server: ServerUrl,
            http: HttpClient,
            path: String,
            auth: (any SignsRequest)?,
            extra: [(name: String, value: String)] = [],
            into sink: any ByteSink
        ) async throws(PipetteMgmtClient.Error) -> HttpResponse {
            let url = "\(server.value)\(path)"
            let method = HttpMethod.get.asStr
            let request = try buildRequest(
                method: .get, url: url, auth: auth, ifNoneMatch: nil, extra: extra, body: nil)
            let response: HttpResponse
            switch await StreamingExchange(sink: sink)
                .run(urlRequest(request), on: http.clientNoRedirects) {
            case .success(let streamed):
                response = streamed
            case .failure(let error):
                throw .transport(method: method, url: url, message: error.localizedDescription)
            }
            let dispatched = Dispatched(response: response, url: url, method: method)
            guard (200..<300).contains(response.status) else { throw statusError(dispatched) }
            return response
        }

        private static func dispatch(
            server: ServerUrl,
            http: HttpClient,
            method: HttpMethod,
            path: String,
            auth: (any SignsRequest)?,
            ifNoneMatch: IfNoneMatch? = nil,
            body: Data?
        ) async throws(PipetteMgmtClient.Error) -> Dispatched {
            let url = "\(server.value)\(path)"
            let request = try buildRequest(
                method: method, url: url, auth: auth, ifNoneMatch: ifNoneMatch, body: body)
            let response = try await execute(
                request, on: http, method: method.asStr, url: url)
            return Dispatched(response: response, url: url, method: method.asStr)
        }

        /// Internal so a test can assert the headers without a server. The
        /// signature does not cover `extra`.
        static func buildRequest(
            method: HttpMethod,
            url: String,
            auth: (any SignsRequest)?,
            ifNoneMatch: IfNoneMatch?,
            extra: [(name: String, value: String)] = [],
            body: Data?
        ) throws(PipetteMgmtClient.Error) -> HttpRequest {
            // `User-Agent` comes from the session on ``HttpClient``.
            var headers: [(name: String, value: String)] = [
                (name: "Accept", value: "application/json"),
            ]
            if let auth {
                headers += try auth.signedHeaders(
                    method: method.asStr, pathAndQuery: try requestTarget(url))
            }
            if let ifNoneMatch {
                headers.append((name: "If-None-Match", value: ifNoneMatch.headerValue))
            }
            headers += extra
            if body != nil {
                headers.append((name: "Content-Type", value: "application/json"))
            }
            guard let parsed = URL(string: url) else {
                throw .invalidUrl(url: url, source: "not a URL")
            }
            return HttpRequest(method: method, url: parsed, headers: headers, body: body)
        }

        static func decodeSuccess<T: Decodable>(
            _ dispatched: Dispatched
        ) throws(PipetteMgmtClient.Error) -> T {
            guard (200..<300).contains(dispatched.response.status) else {
                throw statusError(dispatched)
            }
            do {
                return try Coding.decoder.decode(T.self, from: dispatched.response.body)
            } catch {
                throw .decodeJson(
                    method: dispatched.method, url: dispatched.url,
                    body: bodyExcerpt(dispatched.response.body, separator: "; body: "),
                    source: error)
            }
        }

        /// In Unicode scalars (the crate's `char`s), so the cut lands on a scalar
        /// boundary.
        static let excerptMaxChars = 512

        /// Empty for an empty body, so a message never ends in a bare separator.
        static func bodyExcerpt(_ body: Data, separator: String) -> String {
            // Replaces invalid UTF-8, so a binary body still yields a message.
            let text = String(decoding: body, as: UTF8.self)
                .trimmingCharacters(in: .whitespacesAndNewlines)
            guard !text.isEmpty else { return "" }
            var excerpt = ""
            excerpt.unicodeScalars.append(contentsOf: text.unicodeScalars.prefix(excerptMaxChars))
            let ellipsis = text.unicodeScalars.count > excerptMaxChars ? "..." : ""
            return "\(separator)\(excerpt)\(ellipsis)"
        }

        static func sendEmpty(
            server: ServerUrl,
            http: HttpClient,
            method: HttpMethod,
            path: String,
            auth: (any SignsRequest)?
        ) async throws(PipetteMgmtClient.Error) {
            let dispatched = try await send(
                server: server, http: http, method: method, path: path, auth: auth)
            guard (200..<300).contains(dispatched.response.status) else {
                throw statusError(dispatched)
            }
        }

        static func statusError(_ dispatched: Dispatched) -> PipetteMgmtClient.Error {
            .httpStatus(
                method: dispatched.method,
                url: dispatched.url,
                status: dispatched.response.status,
                body: bodyExcerpt(dispatched.response.body, separator: ": ")
            )
        }

        /// Parsed out of the assembled URL rather than concatenated, so it carries
        /// any base-URL path prefix and the parser's percent-encoding: the server
        /// verifies against what it receives.
        static func requestTarget(_ url: String) throws(PipetteMgmtClient.Error) -> String {
            guard let components = URLComponents(string: url) else {
                throw .invalidUrl(url: url, source: "not a URL")
            }
            let path = components.percentEncodedPath
            return components.percentEncodedQuery.map { "\(path)?\($0)" } ?? path
        }

        /// On the session that refuses redirects: the signature covers the target,
        /// so a redirecting deployment surfaces its `3xx` here instead of a `401`.
        private static func execute(
            _ request: HttpRequest,
            on http: HttpClient,
            method: String,
            url: String
        ) async throws(PipetteMgmtClient.Error) -> HttpResponse {
            let data: Data
            let response: URLResponse
            do {
                (data, response) = try await http.clientNoRedirects.data(for: urlRequest(request))
            } catch {
                throw .transport(method: method, url: url, message: error.localizedDescription)
            }
            guard let httpResponse = response as? HTTPURLResponse else {
                throw .transport(method: method, url: url, message: "response was not HTTP")
            }
            return HttpResponse(
                status: httpResponse.statusCode, headers: headerPairs(httpResponse), body: data)
        }

        fileprivate static func urlRequest(_ request: HttpRequest) -> URLRequest {
            var urlRequest = URLRequest(url: request.url)
            urlRequest.httpMethod = request.method.asStr
            for header in request.headers {
                urlRequest.setValue(header.value, forHTTPHeaderField: header.name)
            }
            urlRequest.httpBody = request.body
            return urlRequest
        }

        fileprivate static func headerPairs(_ response: HTTPURLResponse) -> [(name: String, value: String)] {
            response.allHeaderFields.compactMap { name, value -> (String, String)? in
                guard let name = name as? String, let value = value as? String else {
                    return nil
                }
                return (name, value)
            }
        }
    }
}

/// Where a streamed 2xx body goes. `begin` gets the status and headers before the first
/// chunk; either method returns `false` to stop. The sink keeps its own reason, as a
/// refused response or a local write failure is not the management client's to name.
nonisolated protocol ByteSink: AnyObject, Sendable {
    /// `head` has an empty body.
    func begin(_ head: PipetteMgmtClient.HttpResponse) -> Bool
    func write(_ chunk: Data) -> Bool
}

/// One streamed exchange: a 2xx the sink accepts goes to the sink, anything else into a
/// capped buffer.
///
/// `URLSession` asks the task's delegate before the session's, so the exchange refuses a
/// redirect itself, for the reason ``HttpClient/clientNoRedirects`` gives.
///
/// `@unchecked Sendable`: `URLSession` calls a task's delegate on the session's serial
/// delegate queue, and a lock guards the handoff between `run` and that queue.
nonisolated private final class StreamingExchange: NSObject, URLSessionDataDelegate,
    @unchecked Sendable {
    /// An error body is only a log message, and an artifact route can send a huge one.
    private static let errorBodyLimit = 64 * 1024

    typealias Outcome = Result<PipetteMgmtClient.HttpResponse, any Error>

    private let sink: any ByteSink
    private var response: HTTPURLResponse?
    private var errorBody = Data()
    /// A cancel can complete the task before `run` holds its continuation, so whichever
    /// side arrives second resumes it.
    private let lock = NSLock()
    private var continuation: CheckedContinuation<Outcome, Never>?
    private var outcome: Outcome?

    init(sink: any ByteSink) {
        self.sink = sink
    }

    func run(_ request: URLRequest, on session: URLSession) async -> Outcome {
        if Task.isCancelled { return .failure(CancellationError()) }
        let task = session.dataTask(with: request)
        task.delegate = self
        return await withTaskCancellationHandler {
            await withCheckedContinuation { continuation in
                let early = lock.withLock {
                    if outcome == nil { self.continuation = continuation }
                    return outcome
                }
                if let early {
                    continuation.resume(returning: early)
                } else {
                    task.resume()
                }
            }
        } onCancel: {
            task.cancel()
        }
    }

    func urlSession(
        _ session: URLSession,
        dataTask: URLSessionDataTask,
        didReceive response: URLResponse,
        completionHandler: @escaping (URLSession.ResponseDisposition) -> Void
    ) {
        let http = response as? HTTPURLResponse
        self.response = http
        // A refusal cancels the task, which completes with a cancel error.
        if let http, (200..<300).contains(http.statusCode),
           !sink.begin(PipetteMgmtClient.HttpResponse(
               status: http.statusCode, headers: PipetteMgmtClient.Call.headerPairs(http),
               body: Data())) {
            completionHandler(.cancel)
            return
        }
        completionHandler(.allow)
    }

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
        guard let response, (200..<300).contains(response.statusCode) else {
            errorBody.append(data.prefix(max(0, Self.errorBodyLimit - errorBody.count)))
            return
        }
        if !sink.write(data) { dataTask.cancel() }
    }

    func urlSession(
        _ session: URLSession,
        task: URLSessionTask,
        willPerformHTTPRedirection response: HTTPURLResponse,
        newRequest request: URLRequest,
        completionHandler: @escaping (URLRequest?) -> Void
    ) {
        completionHandler(nil)
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: (any Error)?) {
        let result: Outcome
        if let error {
            result = .failure(error)
        } else if let response {
            result = .success(PipetteMgmtClient.HttpResponse(
                status: response.statusCode,
                headers: PipetteMgmtClient.Call.headerPairs(response),
                body: errorBody))
        } else {
            result = .failure(URLError(.badServerResponse))
        }
        let waiting = lock.withLock {
            defer { continuation = nil }
            if continuation == nil { outcome = result }
            return continuation
        }
        waiting?.resume(returning: result)
    }
}
