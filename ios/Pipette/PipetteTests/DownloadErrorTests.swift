import Testing
import Foundation
@testable import Pipette

private struct OfflineErrorCase: Sendable, CustomTestStringConvertible {
    let code: URLError.Code
    let offline: Bool
    var testDescription: String { "\(code)" }
}

/// Only the offline codes pause a download until the network returns (PIP-385). The other
/// connectivity codes retry on the backoff schedule instead (PIP-702).
@Suite struct DownloadErrorTests {
    @Test(arguments: [
        OfflineErrorCase(code: .notConnectedToInternet, offline: true),
        OfflineErrorCase(code: .dataNotAllowed, offline: true),
        OfflineErrorCase(code: .internationalRoamingOff, offline: true),
        OfflineErrorCase(code: .networkConnectionLost, offline: false),
        OfflineErrorCase(code: .timedOut, offline: false),
        OfflineErrorCase(code: .cannotConnectToHost, offline: false),
        OfflineErrorCase(code: .cannotFindHost, offline: false),
        OfflineErrorCase(code: .dnsLookupFailed, offline: false),
        OfflineErrorCase(code: .secureConnectionFailed, offline: false),
        OfflineErrorCase(code: .badURL, offline: false),
        OfflineErrorCase(code: .unsupportedURL, offline: false),
        OfflineErrorCase(code: .userAuthenticationRequired, offline: false),
        OfflineErrorCase(code: .cancelled, offline: false),
    ])
    fileprivate func classifiesURLErrors(_ testCase: OfflineErrorCase) {
        #expect(isOfflineError(URLError(testCase.code)) == testCase.offline)
    }

    @Test func nonURLErrorIsNotOffline() {
        #expect(isOfflineError(NSError(domain: "Pipette", code: 404)) == false)
    }

    /// An HTTP failure names the status and the URL, as the crate's `ModelFetchError::Http`
    /// does: a dead revision pin and a mistyped filename are both 404, and the sentence
    /// alone cannot tell a plan run which one it hit.
    @Test func anHttpFailureNamesTheStatusAndTheUrl() {
        let url = URL(string:
            "https://huggingface.co/org/repo-GGUF/resolve/deadbeef/model-Q4_K_M.gguf")!

        let message = humanizedDownloadError(NSError(
            domain: "Pipette", code: 404,
            userInfo: [NSURLErrorFailingURLErrorKey: url]))

        #expect(message.contains("couldn't be found on the server"))
        #expect(message.contains("HTTP 404"))
        #expect(message.contains(url.absoluteString))
    }

    /// With no URL the plain sentence stands alone rather than trailing an empty
    /// parenthetical.
    @Test func anHttpFailureWithoutAUrlKeepsThePlainSentence() {
        let message = humanizedDownloadError(NSError(domain: "Pipette", code: 403))

        #expect(message.hasSuffix("try again."))
        #expect(!message.contains("HTTP 403"))
    }
}
