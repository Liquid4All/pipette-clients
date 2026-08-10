import Foundation
import Testing

@testable import Pipette

/// A plan's `auth_token` is decoded, used, and kept in the Keychain — never written into
/// anything a `Model` is persisted in.
///
/// The encoder is the load-bearing half: `SubmissionRef.ModelRef` currently keeps tokens
/// out of submitted descriptors by hand, and that hand-written encoder is scheduled for
/// deletion. Once `Model.encode` is the only encoder, this is what stops a credential
/// reaching the warehouse.
struct ModelTokenCustodyTests {
    private func tokenedModel() throws -> Model {
        var repo = try HFRepo.parse("LiquidAI/Gated-GGUF")
        repo.revision = try HFRevision("v1.2.3")
        repo.authToken = try AuthToken("hf_secret")
        return .ggufText(.init(source: .huggingFace(repo: repo, path: try RepoSubpath("m-Q4_K_M.gguf"), sha256: nil)))
    }

    @Test func encodingAModelNeverWritesTheToken() throws {
        let encoded = try JSONEncoder().encode(tokenedModel())
        let json = try #require(try JSONSerialization.jsonObject(with: encoded) as? [String: Any])
        #expect(json["auth_token"] == nil)
        #expect(!String(decoding: encoded, as: UTF8.self).contains("hf_secret"))
        // The revision is identity and feeds the storage key, so it must survive.
        #expect(json["revision"] as? String == "v1.2.3")
    }

    /// Decode still accepts one: that is how a claim delivers a credential in the first
    /// place. Strip on the way out only.
    @Test func decodingAModelStillAcceptsAToken() throws {
        let json = #"""
        {"type":"gguf_text","source":"huggingface","org":"o","repo_name":"r",
         "path":"m.gguf","auth_token":"hf_secret"}
        """#
        let model = try JSONDecoder().decode(Model.self, from: Data(json.utf8))
        #expect(model.repo?.authToken?.value == "hf_secret")
    }

    /// A round trip is where the credential is lost, and that is the point: what comes
    /// back off disk addresses the same weights without carrying the secret.
    @Test func aRoundTripKeepsIdentityAndDropsTheCredential() throws {
        let original = try tokenedModel()
        let restored = try JSONDecoder().decode(Model.self, from: JSONEncoder().encode(original))
        #expect(restored.repo?.authToken == nil)
        #expect(restored == original)  // identity ignores the token
        #expect(restored.repo?.reference == "LiquidAI/Gated-GGUF@v1.2.3")
    }

    /// The credential must never reach the warehouse. Asserted over the serialized bytes
    /// of every model kind rather than a key name, so it still holds once `ModelRef` is
    /// deleted and `Model.encode` is the only encoder left.
    @Test func noModelKindSubmitsItsToken() throws {
        var repo = try HFRepo.parse("LiquidAI/Gated")
        repo.revision = try HFRevision("v1.2.3")
        repo.authToken = try AuthToken("hf_secret")
        let models: [Model] = [
            .ggufText(.init(source: .huggingFace(repo: repo, path: try RepoSubpath("m-Q4_K_M.gguf"), sha256: nil))),
            .ggufVision(.init(source: .huggingFace(repo: repo, model: try RepoSubpath("m.gguf"), modelSha256: nil, mmproj: try RepoSubpath("mmproj.gguf"), mmprojSha256: nil))),
            .mlx(.init(source: .huggingFace(repo: repo, prefix: try RepoSubpath("4bit")))),
        ]
        for model in models {
            let descriptor = try SubmissionRef.model(model)
            #expect(!descriptor.contains("hf_secret"), "\(ModelType.of(model)) leaked its token")
            #expect(!descriptor.contains("auth_token"))
        }
    }

    /// Per-model custody. Two files out of one repo are two definitions and may have been
    /// granted different credentials; keying by repo made them one entry, so the second
    /// claim's token silently replaced the first's.
    ///
    /// Asserts the derivation rather than a round trip — a Simulator test host cannot
    /// store Keychain items, so a round trip would test the harness, not the keying.
    @Test func twoModelsInOneRepoKeySeparately() throws {
        let repo = try HFRepo.parse("LiquidAI/Gated-GGUF")
        let q4 = Model.ggufText(.init(source: .huggingFace(repo: repo, path: try RepoSubpath("m-Q4_K_M.gguf"), sha256: nil)))
        let q8 = Model.ggufText(.init(source: .huggingFace(repo: repo, path: try RepoSubpath("m-Q8_0.gguf"), sha256: nil)))
        let account = try #require(KeychainHelper.modelTokenAccount(q4))

        #expect(account != KeychainHelper.modelTokenAccount(q8))
        #expect(account.hasSuffix("LiquidAI/Gated-GGUF:m-Q4_K_M.gguf"))
    }

    /// The revision is part of the reference, so a pinned definition keys separately from
    /// the same file on the default branch.
    @Test func aPinnedModelKeysSeparatelyFromAnUnpinnedOne() throws {
        var repo = try HFRepo.parse("LiquidAI/Gated-GGUF")
        let filename = try RepoSubpath("m-Q4_K_M.gguf")
        let unpinned = KeychainHelper.modelTokenAccount(.ggufText(.init(source: .huggingFace(repo: repo, path: filename, sha256: nil))))
        repo.revision = try HFRevision("v1.2.3")
        let pinned = KeychainHelper.modelTokenAccount(.ggufText(.init(source: .huggingFace(repo: repo, path: filename, sha256: nil))))

        #expect(pinned != unpinned)
        #expect(pinned?.hasSuffix("LiquidAI/Gated-GGUF@v1.2.3:m-Q4_K_M.gguf") == true)
    }

    /// The stored token is looked up from the persisted copy, which has been stripped. If
    /// stripping moved the account, the whole stored tier would be write-only.
    @Test func strippingTheTokenDoesNotMoveTheAccount() throws {
        let model = try tokenedModel()
        #expect(KeychainHelper.modelTokenAccount(model)
            == KeychainHelper.modelTokenAccount(model.withoutAuthToken))
    }

    /// AFM ships with the OS: nothing is fetched, so there is no credential to key.
    @Test func appleFoundationHasNoTokenAccount() {
        #expect(KeychainHelper.modelTokenAccount(.appleFoundationText) == nil)
    }

    /// `auth reset` exists to discard credentials, so it has to discard these too — the
    /// CLI has no equivalent because it stores no tokens at all.
    @MainActor
    @Test func resetClearsEveryStoredModelToken() throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }
        var cleared = 0

        #expect(AuthCommands.reset(force: true, storage: storage,
                                   deleteKey: { true },
                                   clearModelTokens: { cleared += 1; return 2 }))
        #expect(cleared == 1)
    }

    /// A refusal must not discard anything — neither half of the credential.
    @MainActor
    @Test func aRefusedResetClearsNothing() throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }
        var touched = 0

        #expect(!AuthCommands.reset(force: false, storage: storage,
                                    deleteKey: { touched += 1; return true },
                                    clearModelTokens: { touched += 1; return 0 }))
        #expect(touched == 0)
    }
}
