import Foundation
import Testing

@testable import Pipette

/// Which credential a HuggingFace fetch authenticates with, and which hosts see one at
/// all. The claim's token wins over the Keychain: a plan that ships a credential is
/// naming the one that can read *that* repo, and the CLI injects the plan's token rather
/// than consulting a local store.
///
/// Every case here supplies a claim token, so no assertion depends on what the Keychain
/// happens to hold — `claimToken` short-circuits before it is read.
struct HfAuthHeaderTests {
    private func header(url: String, claimToken: AuthToken?) throws -> String? {
        var request = URLRequest(url: try #require(URL(string: url)))
        attachHfAuth(&request, claimToken: claimToken)
        return request.value(forHTTPHeaderField: "Authorization")
    }

    @Test func aClaimTokenAuthenticatesTheFetch() throws {
        let token = try AuthToken("hf_fromtheclaim")
        #expect(try header(url: "https://huggingface.co/org/repo/resolve/main/m.gguf",
                           claimToken: token) == "Bearer hf_fromtheclaim")
    }

    /// The tier is what the log line names, so a claim's token has to report itself as one
    /// — `source=claim` in a run's output is the evidence that the plan's credential got
    /// through, and it is worthless if every tier renders the same.
    @Test func aClaimTokenReportsItselfAsTheClaimTier() throws {
        let resolved = resolveHfToken(claimToken: try AuthToken("hf_fromtheclaim"), model: nil)
        #expect(resolved?.source == .claim)
        #expect(resolved?.source.rawValue == "claim")
    }

    /// The host guard is what keeps a credential from reaching a third party, and it
    /// applies to a claim's token exactly as it did to the stored one.
    @Test(arguments: ["https://example.com/org/repo/resolve/main/m.gguf",
                      "https://huggingface.co.evil.test/org/repo/m.gguf"])
    func aNonHuggingFaceHostGetsNoCredential(url: String) throws {
        #expect(try header(url: url, claimToken: try AuthToken("hf_fromtheclaim")) == nil)
    }

    /// A CDN redirect target is a different host, so it is covered by the same guard.
    @Test func theSubdomainFormIsStillHuggingFace() throws {
        #expect(try header(url: "https://cdn-lfs.huggingface.co/repos/x/m.gguf",
                           claimToken: try AuthToken("hf_fromtheclaim")) == "Bearer hf_fromtheclaim")
    }
}
