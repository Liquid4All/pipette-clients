import Foundation
import Testing

@testable import Pipette

/// `HfModelRef` parsing, exact rejection variants, and the `key`/`leaf`/`globs`
/// derivations. The on-disk location is `ModelStorageKey`'s.
struct HfModelRefTests {

    // MARK: - Parse + derive

    @Test func parsesRootStructuredRepo() throws {
        let ref = try HfModelRef.parse(repo: "LiquidAI/LFM2.5-350M-MLX-4bit")
        #expect(ref.repo.org.value == "LiquidAI")
        #expect(ref.repo.repoName.value == "LFM2.5-350M-MLX-4bit")
        #expect(ref.subpath == nil)
        #expect(ref.description == "LiquidAI/LFM2.5-350M-MLX-4bit")
        #expect(ref.key == "LiquidAI/LFM2.5-350M-MLX-4bit")
        #expect(ref.leaf == "LFM2.5-350M-MLX-4bit")
        #expect(ref.globs(for: .mlx) == HfModelRef.rootGlobs)
    }

    @Test func parsesMultiModelRepoWithSubpath() throws {
        let ref = try HfModelRef.parse(repo: "org/multi", subpath: "variant-4bit")
        #expect(ref.key == "org/multi/variant-4bit")
        #expect(ref.leaf == "variant-4bit")
        #expect(ref.globs(for: .mlx) == ["variant-4bit/*"])
    }

    @Test func nestedSubpathLeafIsFinalComponent() throws {
        let ref = try HfModelRef.parse(repo: "org/multi", subpath: "sub/model-b")
        #expect(ref.key == "org/multi/sub/model-b")
        #expect(ref.leaf == "model-b")
        #expect(ref.globs(for: .mlx) == ["sub/model-b/*"])
    }

    @Test func aPrefixSelectsTheSameSubtreeForEitherFormat() throws {
        let ref = try HfModelRef.parse(repo: "org/multi", subpath: "cpu-int4")
        #expect(ref.globs(for: .mlx) == ref.globs(for: .onnxGenai))
    }

    /// At a repo root the two diverge, and the ONNX side has to take everything: its
    /// external-data sidecars are named after the graph that references them, so an
    /// allowlist would drop a file the model cannot load without.
    @Test func anOnnxRootTakesEverythingWhereMlxTakesAnAllowlist() throws {
        let ref = try HfModelRef.parse(repo: "org/onnx-repo")
        #expect(ref.globs(for: .onnxGenai) == ["*"])
        #expect(ref.globs(for: .mlx) == HfModelRef.rootGlobs)
    }

    /// The directory pull hands `hubRevision` to HubApi, so dropping the pin would run main's
    /// weights under a pinned claim.
    @Test func hubRevisionIsThePinElseMain() throws {
        var repo = try HfRepo.parseOrgRepo("LiquidAI/LFM2.5-350M-MLX-4bit")
        #expect(HfModelRef(repo: repo, subpath: nil).hubRevision == "main")
        repo.revision = try HfRevision("0123abcd")
        #expect(HfModelRef(repo: repo, subpath: nil).hubRevision == "0123abcd")
    }

    @Test func rootGlobsCoverConfigWeightsTokenizerAndChatTemplate() {
        for needed in ["config.json", "*.safetensors", "tokenizer.json", "*.jinja"] {
            #expect(HfModelRef.rootGlobs.contains(needed), "rootGlobs missing \(needed)")
        }
    }

    // MARK: - Rejection (exact variant)

    @Test func rejectsRepoMissingSeparator() {
        #expect(throws: ModelError.repoMissingSeparator("just-a-name")) {
            try HfModelRef.parse(repo: "just-a-name")
        }
    }

    /// The split is on the first `/`, as `parse_org_repo` splits: a third segment leaves
    /// `b/c` as the repo name, which its own type refuses. The separator is there, so
    /// reporting it missing would name the wrong half.
    @Test func rejectsMoreThanTwoSegments() {
        #expect(throws: ModelError.invalidRepoName("b/c")) {
            try HfModelRef.parse(repo: "a/b/c")
        }
    }

    @Test func rejectsInvalidOrg() {
        #expect(throws: ModelError.invalidOrg(".bad")) {
            try HfModelRef.parse(repo: ".bad/repo")
        }
    }

    @Test func rejectsEmptyRepoName() {
        #expect(throws: ModelError.invalidRepoName("")) {
            try HfModelRef.parse(repo: "good/")
        }
    }

    @Test func rejectsInvalidSubpath() {
        #expect(throws: ModelError.invalidSubpath("/leading")) {
            try HfModelRef.parse(repo: "org/repo", subpath: "/leading")
        }
    }

    /// A subpath is server-authored and joined onto a cache path that is later removed, so a
    /// dot segment must not pass validation (the crate rejects these by name too).
    @Test(arguments: ["..", "../escape", "dir/../seg", "./seg", "dir/."])
    func rejectsADotSegmentSubpath(_ subpath: String) {
        #expect(throws: ModelError.self) {
            try HfModelRef.parse(repo: "org/repo", subpath: subpath)
        }
    }

}
