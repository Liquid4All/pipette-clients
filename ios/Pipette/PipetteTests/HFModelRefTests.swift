import Foundation
import Testing

@testable import Pipette

/// The validated coordinate types: canonical parsing (round-trip), exact rejection
/// variants, and the derivations (`key`/`leaf`/`globs`) every downstream step relies
/// on. The on-disk location is `ModelStorageKey`'s, not this type's.
struct HFModelRefTests {

    // MARK: - Parse + derive

    @Test func parsesRootStructuredRepo() throws {
        let ref = try HFModelRef.parse(repo: "LiquidAI/LFM2.5-350M-MLX-4bit")
        #expect(ref.repo.org.value == "LiquidAI")
        #expect(ref.repo.repoName.value == "LFM2.5-350M-MLX-4bit")
        #expect(ref.subpath == nil)
        #expect(ref.description == "LiquidAI/LFM2.5-350M-MLX-4bit")
        #expect(ref.key == "LiquidAI/LFM2.5-350M-MLX-4bit")
        #expect(ref.leaf == "LFM2.5-350M-MLX-4bit")
        #expect(ref.globs == HFModelRef.rootGlobs)
    }

    @Test func parsesMultiModelRepoWithSubpath() throws {
        let ref = try HFModelRef.parse(repo: "org/multi", subpath: "variant-4bit")
        #expect(ref.key == "org/multi/variant-4bit")
        #expect(ref.leaf == "variant-4bit")
        #expect(ref.globs == ["variant-4bit/*"])
    }

    @Test func nestedSubpathLeafIsFinalComponent() throws {
        let ref = try HFModelRef.parse(repo: "org/multi", subpath: "sub/model-b")
        #expect(ref.key == "org/multi/sub/model-b")
        #expect(ref.leaf == "model-b")
        #expect(ref.globs == ["sub/model-b/*"])
    }

    @Test func rootGlobsCoverConfigWeightsTokenizerAndChatTemplate() {
        for needed in ["config.json", "*.safetensors", "tokenizer.json", "*.jinja"] {
            #expect(HFModelRef.rootGlobs.contains(needed), "rootGlobs missing \(needed)")
        }
    }

    // MARK: - Rejection (exact variant)

    @Test func rejectsRepoMissingSeparator() {
        #expect(throws: ModelError.repoMissingSeparator("just-a-name")) {
            try HFModelRef.parse(repo: "just-a-name")
        }
    }

    @Test func rejectsMoreThanTwoSegments() {
        #expect(throws: ModelError.repoMissingSeparator("a/b/c")) {
            try HFModelRef.parse(repo: "a/b/c")
        }
    }

    @Test func rejectsInvalidOrg() {
        #expect(throws: ModelError.invalidOrg(".bad")) {
            try HFModelRef.parse(repo: ".bad/repo")
        }
    }

    @Test func rejectsEmptyRepoName() {
        #expect(throws: ModelError.invalidRepoName("")) {
            try HFModelRef.parse(repo: "good/")
        }
    }

    @Test func rejectsInvalidSubpath() {
        #expect(throws: ModelError.invalidSubpath("/leading")) {
            try HFModelRef.parse(repo: "org/repo", subpath: "/leading")
        }
    }

    /// A subpath is server-authored and ends up joined onto a cache path that is later
    /// removed, so a dot segment must not survive validation — the crate rejects these
    /// by name for the same reason.
    @Test(arguments: ["..", "../escape", "dir/../seg", "./seg", "dir/."])
    func rejectsADotSegmentSubpath(_ subpath: String) {
        #expect(throws: ModelError.self) {
            try HFModelRef.parse(repo: "org/repo", subpath: subpath)
        }
    }

}
