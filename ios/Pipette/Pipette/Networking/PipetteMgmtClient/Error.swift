import Foundation

// Why a management call failed: the crate's `error.rs`. Nested so it keeps the
// crate's name without shadowing `Swift.Error` for the app.
//
// A `do` that can also throw something else (building a ``ServerUrl``, say) catches
// `any Error`, so build the values first if the `catch` wants this type.

nonisolated extension PipetteMgmtClient {
    /// A case that names a request carries the method and URL, and one with a
    /// body quotes it: a log line has no request beside it.
    ///
    /// Not mirrored: the crate's timestamp and entropy errors (infallible here),
    /// `BuildClient` (`URLSession(configuration:)` cannot fail), and the user-only
    /// `UnusableKeyId`, `Signer` and `DotSegment`.
    enum Error: LocalizedError {
        case invalidUrl(url: String, source: String)

        /// The message names the shape, never the key.
        case invalidPrivateKey(PrivateKeyError)

        /// Only from key derivation, over an already validated private key: a
        /// failure is an internal disagreement, not bad caller input.
        case invalidPublicKey(PublicKeyError)

        case encodeJson(any Swift.Error)

        case transport(method: String, url: String, message: String)

        /// `body` carries its own separator
        /// (``PipetteMgmtClient/Call/bodyExcerpt(_:separator:)``), so an empty
        /// response adds nothing to the message.
        case httpStatus(method: String, url: String, status: Int, body: String)

        /// `body` is quoted because `JSONDecoder`'s coding path names a position in
        /// JSON the operator never saw.
        case decodeJson(method: String, url: String, body: String, source: any Swift.Error)

        var httpStatusCode: Int? {
            guard case .httpStatus(_, _, let status, _) = self else { return nil }
            return status
        }

        /// A transport failure or a `5xx`, which the worker and vault reads retry.
        var isTransient: Bool {
            switch self {
            case .transport: return true
            case .httpStatus(_, _, let status, _): return status >= 500
            default: return false
            }
        }

        var errorDescription: String? {
            switch self {
            case .invalidUrl(let url, let source):
                return "invalid request URL `\(url)`: \(source)"
            case .invalidPrivateKey(let reason):
                return reason.message
            case .invalidPublicKey(let reason):
                return reason.errorDescription
            case .encodeJson(let source):
                return "failed to serialize request body: \(source)"
            case .transport(let method, let url, let message):
                return "request failed for \(method) \(url): \(message)"
            case .httpStatus(let method, let url, let status, let body):
                return "request returned \(status) for \(method) \(url)\(body)"
            case .decodeJson(let method, let url, let body, let source):
                return "failed to parse JSON response for \(method) \(url): \(source)\(body)"
            }
        }
    }
}
