import Foundation

/// Workspace HTTPS client policy and the process-shared ``HttpClient``: the
/// `pipette-http` crate's `lib.rs`. Build once at the application edge and pass it
/// down; no endpoint defaults one in.
///
/// Not mirrored from the crate, as nothing here could set them: the rustls setup
/// (`tls_client_config`, `TlsMode`, `preconfigured_tls`), since `URLSession` owns
/// TLS; `DEFAULT_POOL_IDLE_TIMEOUT`, since `URLSessionConfiguration` has no such
/// knob; and the client-build errors, since `URLSession(configuration:)` cannot fail.
///
/// `@unchecked Sendable`: it holds a `URLSession`, which is safe to use from any
/// thread but is not yet marked `Sendable` on every SDK this builds against.
nonisolated struct HttpClient: @unchecked Sendable {
    /// The crate's `pipette_http::Error`, nested so it keeps that name without
    /// shadowing `Swift.Error` for the app. The crate's `Status` folds into `request`.
    enum Error: LocalizedError {
        /// A transport failure, or a non-2xx with its status and body quoted so a
        /// caller can surface the server's own diagnostics.
        case request(String)

        case json(String)

        var errorDescription: String? {
            switch self {
            case .request(let message): return "HTTP request failed: \(message)"
            case .json(let message): return "failed to parse JSON response: \(message)"
            }
        }
    }

    /// The crate's `DEFAULT_CONNECT_TIMEOUT`, applied as `timeoutIntervalForRequest`:
    /// reqwest times the TCP connect, `URLSession` the wait for the next byte. Both
    /// bound a host that accepts nothing; neither bounds the whole transfer.
    static let defaultConnectTimeout: TimeInterval = 30

    /// Set on the session's configuration, not per call, so no caller can send
    /// one of its own.
    let userAgent: String

    /// Follows redirects: a HuggingFace resolve URL redirects to its CDN, and a
    /// fetch is sized by the blob behind it.
    let client: URLSession

    /// Surfaces a `3xx` as the response. For requests whose signature covers the
    /// target: a redirect would present a signature over the old target (a `401`),
    /// and `URLSession` forwards custom headers, `X-Signature` too, to the new host.
    let clientNoRedirects: URLSession

    static func builder(userAgent: String) -> HttpClientBuilder {
        HttpClientBuilder(userAgent: userAgent)
    }

    /// The default connect timeout and no overall request timeout.
    init(userAgent: String) {
        self = Self.builder(userAgent: userAgent).build()
    }

    static func withRequestTimeout(userAgent: String, timeout: TimeInterval) -> HttpClient {
        builder(userAgent: userAgent).requestTimeout(timeout).build()
    }

    /// Takes the configuration rather than defaulting one, so a test can answer
    /// with a stub `URLProtocol` on an ephemeral configuration.
    init(userAgent: String, configuration: URLSessionConfiguration) {
        self.userAgent = userAgent
        // Copied: `URLSessionConfiguration` is a class, so two clients built from
        // one configuration would otherwise overwrite each other's `User-Agent`.
        let configuration = (configuration.copy() as? URLSessionConfiguration) ?? configuration
        var headers = configuration.httpAdditionalHeaders ?? [:]
        headers["User-Agent"] = userAgent
        configuration.httpAdditionalHeaders = headers
        client = URLSession(configuration: configuration)
        clientNoRedirects = URLSession(
            configuration: configuration, delegate: RefuseRedirects(), delegateQueue: nil)
    }

    /// Never invalidated: it lives as long as the process.
    static let shared = HttpClient(userAgent: "pipette-ios")

    /// One-shot JSON request on ``client``, which follows redirects. A caller whose
    /// signature covers the target wants the ``PipetteMgmtClient/Call`` pipeline.
    /// `body` is already-encoded JSON.
    func jsonRequest<T: Decodable>(
        method: String,
        url: String,
        headers: [(name: String, value: String)]?,
        body: Data?
    ) async throws(Error) -> T {
        guard let parsed = URL(string: url) else {
            throw .request("not a URL (\(url))")
        }
        var request = URLRequest(url: parsed)
        request.httpMethod = method
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        for header in headers ?? [] {
            request.setValue(header.value, forHTTPHeaderField: header.name)
        }
        if let body {
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            request.httpBody = body
        }

        let data: Data
        let response: URLResponse
        do {
            (data, response) = try await client.data(for: request)
        } catch {
            throw .request("\(error.localizedDescription) (\(url))")
        }
        guard let httpResponse = response as? HTTPURLResponse else {
            throw .request("response was not HTTP (\(url))")
        }
        guard (200..<300).contains(httpResponse.statusCode) else {
            throw .request(
                "\(httpResponse.statusCode) (\(url)): \(String(decoding: data, as: UTF8.self))")
        }
        do {
            return try Coding.decoder.decode(T.self, from: data)
        } catch {
            throw .json("\(error) (\(url))")
        }
    }
}

/// ``build()`` does not fail where the crate's returns a `Result`:
/// `URLSession(configuration:)` cannot.
nonisolated struct HttpClientBuilder: Sendable {
    private let userAgent: String
    private var connectTimeout: TimeInterval = HttpClient.defaultConnectTimeout
    private var requestTimeout: TimeInterval?

    init(userAgent: String) {
        self.userAgent = userAgent
    }

    func connectTimeout(_ timeout: TimeInterval) -> Self {
        var copy = self
        copy.connectTimeout = timeout
        return copy
    }

    /// Overall request/body timeout. Default is none (downloads, long streams).
    func requestTimeout(_ timeout: TimeInterval) -> Self {
        var copy = self
        copy.requestTimeout = timeout
        return copy
    }

    func noRequestTimeout() -> Self {
        var copy = self
        copy.requestTimeout = nil
        return copy
    }

    func build() -> HttpClient {
        let configuration = URLSessionConfiguration.default
        configuration.timeoutIntervalForRequest = connectTimeout
        if let requestTimeout {
            configuration.timeoutIntervalForResource = requestTimeout
        }
        return HttpClient(userAgent: userAgent, configuration: configuration)
    }
}

/// The completion-handler form, not the `async` one: the `async` variant crashes
/// SILGen emitting its ObjC thunk (`emitObjCMethodThunk`) in a debug build.
///
/// Internal so a test can pin it directly: `URLSession` ignores a `3xx` from a
/// custom `URLProtocol`, and `urlProtocol(_:wasRedirectedTo:redirectResponse:)`
/// hangs the task, so a stub cannot drive a real redirect.
final class RefuseRedirects: NSObject, URLSessionTaskDelegate {
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
