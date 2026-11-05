import Foundation
import Testing

@testable import Pipette

/// `ModelStorageKey`: the flat entry name the model store addresses a model by. Keys must
/// match `crates/pipette-artifacts/src/model/key.rs` byte for byte.
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
    /// of one repo share an entry and the second fetch overwrites the first's weights.
    @Test func aPinnedRevisionGetsItsOwnEntry() throws {
        var repo = try HfRepo.parseOrgRepo("org/repo")
        let path = try RepoSubpath("w.gguf")
        let unpinned = try key(.ggufText(GgufText(source: .huggingFace(
            repo: repo, path: path, sha256: nil))))
        repo.revision = try HfRevision("v1")
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

    /// A slug over the 32-char cap folds to `<head>_<hash8>`; two long models sharing a
    /// head stay distinct through the hash tail.
    @Test func longSlugsAreFoldedAndStayDistinct() throws {
        let long = { (repo: String) in
            try self.key(ggufTextSpec("really-long-shared-organization/\(repo)", "weights-q4-k-m.gguf"))
        }
        let a = try long("repo-a")
        let b = try long("repo-b")
        #expect(a.count <= 32 && b.count <= 32)
        #expect(a.hasPrefix("really-long-shared-orga"))
        #expect(a != b)
    }

    /// The crate's `declared_model_keys` cases.
    @Test func declaredModelKeys() throws {
        #expect(try key(ggufTextSpec("meta/llama", "Q4.gguf")) == "meta__llama__Q4.gguf")
        #expect(try key(ggufVisionSpec("liquidai/vl", "q4.gguf", "mm.gguf")) == "liquidai__vl__q4.gguf__mm.gguf")
        #expect(try key(mlxSpec("meta/Llama")) == "meta__Llama")
        #expect(try key(mlxSpec("meta/Llama", subpath: "4bit")) == "meta__Llama__4bit")
    }

    /// The vault case too: `spec`, then the plain vision key as one segment, then a
    /// separate drafter. The keys are the crate's, byte for byte.
    @Test func aSpeculativeModelKeysApartFromItsPlainVlTwin() throws {
        let repo = try HfRepo.parseOrgRepo("liquidai/vl")
        func speculative(_ draft: String?) throws -> Model {
            .ggufVisionSpeculative(GgufVisionSpeculative(source: .huggingFace(
                repo: repo, model: try RepoSubpath("q4.gguf"), modelSha256: nil,
                mmproj: try RepoSubpath("mm.gguf"), mmprojSha256: nil,
                draft: try draft.map { try RepoSubpath($0) }, draftSha256: nil)))
        }
        let plain = try key(ggufVisionSpec("liquidai/vl", "q4.gguf", "mm.gguf"))
        let merged = try key(speculative(nil))

        #expect(merged == "spec__liquidai__vl__q4._83c51347")
        #expect(merged != plain, "one entry cannot hold both types")
        #expect(try key(speculative("draft.gguf")) != merged, "the drafter is in the identity")
        #expect(try key(vaultSpeculativeSpec("vault_a", "vault_b", draft: "vault_c"))
            == "spec__vault__vault_a__v_b465a129")
    }

    /// The crate's `vault_segments`: `vault`, then the resource id, and a vision pair adds
    /// only the projector's id. The route around each id adds nothing.
    @Test func aVaultModelKeysOnItsResourceIds() throws {
        #expect(try key(vaultTextSpec("vault_1")) == "vault__vault_1")
        #expect(try key(vaultVisionSpec("vault_a", "vault_b")) == "vault__vault_a__vault_b")
        #expect(try key(vaultArchiveSpec("vault_1")) == "vault__vault_1")
        // A long id folds as every other long key does.
        #expect(try key(vaultTextSpec("vault_0195c3e2-a4f3-7c1b-9d2e-8f0a1b2c3d4e")).count == 32)
    }

    /// A vault source's `custom_label` is no segment: labelled or not, one entry.
    @Test func aCustomLabelDoesNotMoveTheKey() throws {
        for (plain, labelled) in try vaultSpecPairs() {
            #expect(try key(labelled) == key(plain), "\(ModelType.of(plain))")
        }
    }

    /// The crate's `declared_model_keys` URL case: the whole URL is the one segment.
    @Test func aUrlKeysOnTheWholeUrl() throws {
        let url = Model.ggufText(GgufText(source: .url(
            url: try ResourceUrl("https://ex.com/m/w.gguf"), sha256: nil)))
        #expect(try key(url) == "https_ex.com_m_w.gguf")
    }

    /// A host path is warehouse identity, imported into the store.
    @Test func absoluteLocalModelsHavePathKeys() throws {
        #expect(try key(.ggufText(GgufText(source: .absoluteFile(
            path: try AbsolutePath("/on/disk/w.gguf"))))) == "on_disk_w.gguf")
        #expect(try key(.mlx(Mlx(source: .absoluteDir(
            dir: try AbsolutePath("/models/x"))))) == "models_x")
        #expect(try key(.ggufVision(GgufVision(source: .absoluteFiles(
            model: try AbsolutePath("/a/m.gguf"), mmproj: try AbsolutePath("/a/mm.gguf")))))
            == "a_m.gguf__a_mm.gguf")
    }

    /// The crate's `no_storage_key_for_non_declared`: a store-relative arm is already
    /// inside a store and names no coordinate to key on.
    @Test func storeRelativeArmsHaveNoKey() throws {
        #expect(ModelStorageKey.of(.ggufText(GgufText(source: .relativeFile(
            path: try RelativePath("entry/blobs/w.gguf"))))) == nil)
        #expect(ModelStorageKey.of(.ggufVision(GgufVision(source: .relativeFiles(
            model: try RelativePath("entry/blobs/m.gguf"),
            mmproj: try RelativePath("entry/blobs/mm.gguf"))))) == nil)
        #expect(ModelStorageKey.of(.mlx(Mlx(source: .relativeDir(
            dir: try RelativePath("entry/blobs"))))) == nil)
    }

    /// The crate's `relative_dir` / `blobs_dir`: the entry is the key itself, one flat
    /// component, and its payload directory sits directly under it.
    @Test func theEntryAndItsBlobsAreRelativeToTheModelsRoot() throws {
        let storageKey = try #require(ModelStorageKey.of(try ggufTextSpec("meta/llama", "Q4.gguf")))
        #expect(storageKey.relativeDir == "meta__llama__Q4.gguf")
        #expect(storageKey.blobsDir == "meta__llama__Q4.gguf/blobs")
    }
}
