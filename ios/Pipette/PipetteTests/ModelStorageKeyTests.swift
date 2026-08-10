import Foundation
import Testing

@testable import Pipette

/// `ModelStorageKey` — the flat entry name the model store addresses a model by.
/// Pure derivation, so the suite needs no storage and runs in parallel.
struct ModelStorageKeyTests {
    private func key(_ spec: Model) throws -> String {
        try #require(ModelStorageKey.of(spec)).value
    }

    @Test func ggufTextKeyIsOrgRepoFilename() throws {
        #expect(try key(ggufTextSpec("org/repo", "file.gguf")) == "org__repo__file.gguf")
    }

    @Test func visionKeyCarriesBothFiles() throws {
        #expect(try key(ggufVisionSpec("org/repo", "w.gguf", "mm.gguf")) == "org__repo__w.gguf__mm.gguf")
        // Two VL models in one repo differing only in the projector must not alias.
        #expect(try key(ggufVisionSpec("org/repo", "w.gguf", "mm-a.gguf"))
            != key(ggufVisionSpec("org/repo", "w.gguf", "mm-b.gguf")))
    }

    @Test func mlxKeyIsRepoPlusOptionalSubpath() throws {
        #expect(try key(mlxSpec("org/repo")) == "org__repo")
        #expect(try key(mlxSpec("org/repo", subpath: "4bit")) == "org__repo__4bit")
    }

    @Test func keyIsASingleFlatComponentAndDeterministic() throws {
        let spec = try ggufTextSpec("org/repo", "file.gguf")
        let first = try key(spec)
        #expect(!first.contains("/"))
        #expect(first == (try key(spec)))
    }

    /// The filename segment separates a repo's GGUF from an MLX build of the same
    /// coordinate, so they get their own entries rather than overwriting each other.
    @Test func ggufAndMlxOfTheSameRepoDoNotCollide() throws {
        #expect(try key(ggufTextSpec("org/repo", "file.gguf")) != key(mlxSpec("org/repo")))
    }

    /// The pin is a segment, as in the crate's `repo_segments`. Without it two revisions
    /// of one repo share an entry and the second fetch overwrites the first's weights —
    /// which `HFRepo.revision`'s own doc already claimed was impossible.
    @Test func aPinnedRevisionGetsItsOwnEntry() throws {
        var repo = try HFRepo.parse("org/repo")
        let path = try RepoSubpath("w.gguf")
        let unpinned = try key(.ggufText(GgufText(source: .huggingFace(
            repo: repo, path: path, sha256: nil))))
        repo.revision = try HFRevision("v1")
        let pinned = try key(.ggufText(GgufText(source: .huggingFace(
            repo: repo, path: path, sha256: nil))))

        #expect(unpinned == "org__repo__w.gguf")
        #expect(pinned == "org__repo__v1__w.gguf")
    }

    /// A subdirectory separator collapses into its segment, matching the crate's
    /// `sub/dir/w.gguf` -> `org__repo__sub_dir_w.gguf`.
    @Test func aSubdirectoryPathCollapsesIntoOneSegment() throws {
        #expect(try key(ggufTextSpec("org/repo", "sub/dir/w.gguf")) == "org__repo__sub_dir_w.gguf")
    }

    @Test func appleFoundationHasNoKey() {
        #expect(ModelStorageKey.of(.appleFoundationText) == nil)
    }

    /// A slug over the 32-char cap folds to `<head>_<hash8>`; two long models sharing
    /// a head stay distinct through the hash tail.
    @Test func longSlugsFoldAndStayDistinct() throws {
        let long = { (repo: String) in
            try self.key(ggufTextSpec("really-long-shared-organization/\(repo)", "weights-q4-k-m.gguf"))
        }
        let a = try long("repo-a")
        let b = try long("repo-b")
        #expect(a.count <= 32 && b.count <= 32)
        #expect(a.hasPrefix("really-long-shared-orga"))
        #expect(a != b)
    }

    /// Cross-client parity: the same coordinates must key identically on iOS and in
    /// `pipette-artifacts` (`ModelStorageKey::declared_model_keys`), so a key computed
    /// on either client names the same entry.
    @Test func matchesTheCrateDeclaredKeys() throws {
        #expect(try key(ggufTextSpec("meta/llama", "Q4.gguf")) == "meta__llama__Q4.gguf")
        #expect(try key(ggufVisionSpec("liquidai/vl", "q4.gguf", "mm.gguf")) == "liquidai__vl__q4.gguf__mm.gguf")
        #expect(try key(mlxSpec("meta/Llama")) == "meta__Llama")
        #expect(try key(mlxSpec("meta/Llama", subpath: "4bit")) == "meta__Llama__4bit")
    }
}
