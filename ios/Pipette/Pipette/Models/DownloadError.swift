import Foundation

/// Failures of the model download and install pipeline. Transport and IO carry a message,
/// not the underlying `Error`, so the enum stays `Equatable` for assertions.
nonisolated enum DownloadError: Error, Equatable {
    case emptyMatch(HfModelRef)
    /// The model, by its `Model.description`, and the pieces its directory lacks.
    case incompleteModel(String, missing: [String])
    case transport(String)
    case io(String)
    case cancelled
}

extension DownloadError: LocalizedError {
    var errorDescription: String? {
        switch self {
        case let .emptyMatch(ref): "no files matched for \(ref)"
        case let .incompleteModel(name, missing): "\(name) is not a complete MLX model (missing: \(missing.joined(separator: ", ")))"
        case let .transport(m): "download failed: \(m)"
        case let .io(m): "file error: \(m)"
        case .cancelled: "download cancelled"
        }
    }
}

// MARK: - Error display

/// The user-facing message that replaces the system message for a `URLError.code`, if
/// any.
nonisolated func downloadURLErrorMessage(_ code: URLError.Code) -> String? {
    switch code {
    case .notConnectedToInternet, .networkConnectionLost:
        "No internet connection. Check your network and try again."
    case .timedOut:
        "The download timed out. Check your connection and try again."
    case .cannotFindHost, .cannotConnectToHost, .dnsLookupFailed, .secureConnectionFailed:
        "Couldn't reach the download server. Please try again in a moment."
    case .userAuthenticationRequired:
        "This model is gated. Add a Hugging Face access token in Settings, then try again."
    default:
        nil
    }
}

/// Whether a URLSession failure means the device is offline: the download then keeps its
/// resume data and waits for the network without using an attempt.
nonisolated func isOfflineError(_ error: Error) -> Bool {
    guard let urlError = error as? URLError else { return false }
    return isOffline(urlError.code)
}

/// A short sentence for the "Download Error" alert.
///
/// An HTTP failure also names the status and URL, as the crate's `ModelFetchError::Http`
/// does: on a plan-dispatched run, which coordinate was refused is the whole diagnosis.
func humanizedDownloadError(_ error: Error) -> String {
    if let urlError = error as? URLError {
        return downloadURLErrorMessage(urlError.code) ?? urlError.localizedDescription
    }

    let ns = error as NSError
    if ns.domain == "Pipette" {
        let sentence: String? = switch ns.code {
        case 401, 403:
            "Access to this model was denied. It may be gated. Add a Hugging Face token in Settings and try again."
        case 404:
            "This model file couldn't be found on the server. It may have moved or been removed."
        case 500...599:
            "The download server ran into a problem. Please try again in a moment."
        default:
            nil
        }
        if let sentence {
            guard let url = ns.userInfo[NSURLErrorFailingURLErrorKey] as? URL else {
                return sentence
            }
            return "\(sentence) (HTTP \(ns.code) for \(url.absoluteString))"
        }
    }

    return error.localizedDescription
}
