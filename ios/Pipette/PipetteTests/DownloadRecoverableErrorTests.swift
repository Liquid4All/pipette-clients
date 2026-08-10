import Testing
import Foundation
@testable import Pipette

/// One row for the recoverable-vs-terminal error table.
private struct RecoverableErrorCase: Sendable, CustomTestStringConvertible {
    let code: URLError.Code
    let recoverable: Bool
    var testDescription: String { "\(code)" }
}

/// PIP-385: only connectivity errors keep a download resumable (so it recovers
/// on reconnect); terminal errors still drop it.
@Suite struct DownloadRecoverableErrorTests {
    @Test(arguments: [
        RecoverableErrorCase(code: .notConnectedToInternet, recoverable: true),
        RecoverableErrorCase(code: .networkConnectionLost, recoverable: true),
        RecoverableErrorCase(code: .timedOut, recoverable: true),
        RecoverableErrorCase(code: .cannotConnectToHost, recoverable: true),
        RecoverableErrorCase(code: .cannotFindHost, recoverable: true),
        RecoverableErrorCase(code: .dnsLookupFailed, recoverable: true),
        RecoverableErrorCase(code: .dataNotAllowed, recoverable: true),
        RecoverableErrorCase(code: .internationalRoamingOff, recoverable: true),
        RecoverableErrorCase(code: .badURL, recoverable: false),
        RecoverableErrorCase(code: .unsupportedURL, recoverable: false),
        RecoverableErrorCase(code: .userAuthenticationRequired, recoverable: false),
        RecoverableErrorCase(code: .cancelled, recoverable: false),
    ])
    fileprivate func classifiesURLErrors(_ testCase: RecoverableErrorCase) {
        #expect(isRecoverableNetworkError(URLError(testCase.code)) == testCase.recoverable)
    }

    @Test func nonURLErrorIsNotRecoverable() {
        #expect(isRecoverableNetworkError(NSError(domain: "Pipette", code: 404)) == false)
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

    /// Nothing to name when the error carries no URL — the plain sentence stands alone
    /// rather than trailing an empty parenthetical.
    @Test func anHttpFailureWithoutAUrlKeepsThePlainSentence() {
        let message = humanizedDownloadError(NSError(domain: "Pipette", code: 403))

        #expect(message.hasSuffix("try again."))
        #expect(!message.contains("HTTP 403"))
    }

    /// The limit caps the store, not free disk, so the remedy is raising it in Settings
    /// — deleting other models cannot make room for an oversize one.
    @Test func exceedsQuotaPointsAtTheSettingsLimit() {
        let message = DownloadError.exceedsQuota(neededBytes: 9_000_000, quotaBytes: 1_000_000)
            .localizedDescription

        #expect(message.contains("Settings"))
        #expect(message.range(of: "free up space", options: .caseInsensitive) == nil)
    }
}
