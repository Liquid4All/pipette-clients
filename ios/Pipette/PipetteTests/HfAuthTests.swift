import Foundation
import Testing

@testable import Pipette

/// Which credential a HuggingFace fetch authenticates with, and which hosts see one at
/// all. The claim's token wins over the Keychain: a plan that ships a credential names
/// the one that can read *that* repo.
///
/// Every case supplies a claim token, which short-circuits before the Keychain is read,
/// so no assertion depends on what the Keychain holds.
struct HfAuthTests {
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

    /// `source=claim` in a run's log is the evidence the plan's credential got through,
    /// worthless if every tier renders the same.
    @Test func aClaimTokenReportsItselfAsTheClaimTier() throws {
        let resolved = resolveHfToken(claimToken: try AuthToken("hf_fromtheclaim"), model: nil)
        #expect(resolved?.source == .claim)
        #expect(resolved?.source.rawValue == "claim")
    }

    /// The host guard keeps a credential from reaching a third party, and applies to a
    /// claim's token as to a stored one.
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
