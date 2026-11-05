import Foundation

// Wire types for management HTTP: the crate's `transport.rs`. Its `TransportError`
// is not mirrored (`Error.transport` carries its message).

nonisolated extension PipetteMgmtClient {
    enum HttpMethod: String {
        case get = "GET"
        case post = "POST"
        case put = "PUT"
        case patch = "PATCH"
        case delete = "DELETE"

        var asStr: String { rawValue }
    }

    struct HttpRequest {
        let method: HttpMethod
        let url: URL
        /// Pairs rather than a dictionary, to keep the order the scheme names
        /// the signed headers in.
        let headers: [(name: String, value: String)]
        let body: Data?
    }

    struct HttpResponse {
        let status: Int
        let headers: [(name: String, value: String)]
        let body: Data

        /// Case-insensitive: `URLSession` preserves the server's casing.
        func header(_ name: String) -> String? {
            headers.first { $0.name.caseInsensitiveCompare(name) == .orderedSame }?.value
        }

        var contentRange: ContentRange? {
            header("Content-Range").flatMap(ContentRange.init(headerValue:))
        }

        /// Quotes included; `nil` if weak.
        var strongEtag: String? {
            header("ETag").flatMap(strongEtagValue)
        }
    }
}
