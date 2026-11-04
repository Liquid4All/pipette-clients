import Foundation
import Hub
import HuggingFace

extension Disposition {
    /// Whether a `HubApi` failure is transient, for the retries of a directory pull.
    ///
    /// - A file GET throws `HTTPClientError.responseError` with the whole response, so
    ///   the status and `Retry-After` are both available.
    /// - A HEAD or listing refusal is a `Hub.HubClientError`: the status, not the response.
    /// - A listing turns a `URLError` into `.downloadError`, keeping only the description,
    ///   and the file GET reports a `429` that way too. So `.downloadError` is transient,
    ///   and a permanent transport error on a listing (a refused certificate) uses all the
    ///   attempts.
    nonisolated init(ofHub error: any Error) {
        switch error {
        case let error as URLError:
            self.init(of: error.code)
        case let HTTPClientError.responseError(response, _):
            self.init(status: response.statusCode, retryAfter: retryAfter(of: response))
        case let error as Hub.HubClientError:
            switch error {
            case let .httpStatusCode(status):
                self.init(status: status, retryAfter: nil)
            case let .networkError(error):
                self.init(of: error.code)
            case .downloadError:
                self = .transient(retryAfter: nil)
            case .authorizationRequired, .parse, .jsonSerialization, .unexpectedError,
                 .fileNotFound, .resourceNotFound, .configurationMissing, .fileSystemError,
                 .parseError:
                self = .permanent
            }
        default:
            self = .permanent
        }
    }
}
