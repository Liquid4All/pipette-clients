import Foundation
import Testing

@testable import Pipette

/// `Model.toStored` / `underRoot`, mirroring `pipette-artifacts/src/model/stored.rs`. As
/// upstream, the relative arms are the stored form.
@MainActor struct StoredTests {
    /// A manifest's `stored` is relative to the `models/` root, so it names the entry as
    /// well as the file.
    @Test func storedFieldIsToStoredUnderKeyBlobsDir() throws {
        let declared = try ggufTextSpec("meta/llama", "Q4.gguf")
        let key = try #require(ModelStorageKey.of(declared))

        #expect(key.blobsDir == "meta__llama__Q4.gguf/blobs")
        #expect(declared.toStored(base: key.blobsDir)
            == .ggufText(GgufText(source: .relativeFile(
                path: try RelativePath("meta__llama__Q4.gguf/blobs/Q4.gguf")))))
    }

    /// `to_stored` joins each path under `base` independently and keeps the repo's
    /// nesting. The vision arm is the one that matters: two files, two subdirectories.
    @Test func toStoredKeepsEachPathsNesting() throws {
        let vision = try ggufVisionSpec("org/vl-GGUF", "quants/vl-Q4_0.gguf", "mmproj/proj-f16.gguf")

        let stored = try #require(vision.toStored(base: "entry/blobs"))
        guard case let .ggufVision(m) = stored,
              case let .relativeFiles(model, mmproj) = m.source else {
            Issue.record("expected a relativeFiles vision arm"); return
        }
        #expect(model.value == "entry/blobs/quants/vl-Q4_0.gguf")
        #expect(mmproj.value == "entry/blobs/mmproj/proj-f16.gguf")
    }

    /// The crate's `vault_filename`: a vault file is stored as `<resource_id>.gguf`, since
    /// every vault path ends in `content`. Two files that would land on one path have no
    /// store form, as the crate refuses them.
    @Test func aVaultFileIsStoredUnderItsResourceId() throws {
        #expect(try vaultTextSpec("vault_1").toStored(base: "entry/blobs")
            == .ggufText(GgufText(source: .relativeFile(
                path: try RelativePath("entry/blobs/vault_1.gguf")))))
        #expect(try vaultVisionSpec("vault_a", "vault_b").toStored(base: "entry/blobs")
            == .ggufVision(GgufVision(source: .relativeFiles(
                model: try RelativePath("entry/blobs/vault_a.gguf"),
                mmproj: try RelativePath("entry/blobs/vault_b.gguf")))))
        #expect(try vaultVisionSpec("vault_1", "vault_1").toStored(base: "entry/blobs") == nil)
        #expect(try vaultSpeculativeSpec("vault_a", "vault_b", draft: "vault_c")
            .toStored(base: "entry/blobs")
            == .ggufVisionSpeculative(GgufVisionSpeculative(source: .relativeFiles(
                model: try RelativePath("entry/blobs/vault_a.gguf"),
                mmproj: try RelativePath("entry/blobs/vault_b.gguf"),
                draft: try RelativePath("entry/blobs/vault_c.gguf")))))
        #expect(try vaultSpeculativeSpec("vault_a", "vault_b", draft: "vault_a")
            .toStored(base: "entry/blobs") == nil)
    }

    /// Weights, projector, then a separate drafter if there is one.
    @Test func aVaultModelListsItsFilesInFetchOrder() throws {
        let files = try #require(ModelFetch.vaultFiles(
            vaultSpeculativeSpec("vault_a", "vault_b", draft: "vault_c")))
        #expect(files.map(\.path.resourceId.value) == ["vault_a", "vault_b", "vault_c"])
        let merged = try #require(ModelFetch.vaultFiles(
            vaultSpeculativeSpec("vault_a", "vault_b", draft: nil)))
        #expect(merged.map(\.path.resourceId.value) == ["vault_a", "vault_b"])
    }

    @Test func aCustomLabelMovesNeitherTheStoredFormNorTheFetch() throws {
        for (plain, labelled) in try vaultSpecPairs() {
            let stored = try #require(labelled.toStored(base: "entry/blobs"))
            #expect(stored == plain.toStored(base: "entry/blobs"))
            #expect(ModelFetch.vaultFiles(labelled) == ModelFetch.vaultFiles(plain))
        }
    }

    /// For the arms a phone stores: a directory model's `prefix` nests under `base`, a
    /// whole repo is `base`.
    @Test func storedFormIsRelativeToModelStorageRoot() throws {
        func dir(_ model: Model) -> String? {
            guard case let .mlx(m) = model, case let .relativeDir(dir) = m.source else { return nil }
            return dir.value
        }
        let text = try #require(try ggufTextSpec("org/a-GGUF", "a.gguf").toStored(base: "entry/blobs"))
        guard case let .ggufText(t) = text, case let .relativeFile(path) = t.source else {
            Issue.record("expected a relativeFile arm"); return
        }
        #expect(path.value == "entry/blobs/a.gguf")

        #expect(dir(try #require(try mlxSpec("org/repo").toStored(base: "entry/blobs"))) == "entry/blobs")
        #expect(dir(try #require(try mlxSpec("org/repo", subpath: "4bit").toStored(base: "entry/blobs")))
            == "entry/blobs/4bit")

        // AFM ships with the OS: nothing stored, nothing to bind.
        #expect(Model.appleFoundationText.toStored(base: "entry/blobs") == nil)
    }

    /// The models dir is prefixed onto the recorded paths, which is what makes them survive
    /// a container-UUID change. The bound arms are the `Absolute*` ones, as upstream.
    @Test func bindUnderJoinsModelsDirOntoStoredPaths() throws {
        let manifest = try manifestFixture(try ggufTextSpec("meta/llama", "Q4.gguf"))

        #expect(manifest.bindUnder(URL(fileURLWithPath: "/ws/models"))
            == .ggufText(GgufText(source: .absoluteFile(
                path: try AbsolutePath("/ws/models/meta__llama__Q4.gguf/blobs/Q4.gguf")))))

        let vision = try #require(
            try ggufVisionSpec("org/vl-GGUF", "quants/vl.gguf", "mmproj/proj.gguf")
                .toStored(base: "key/blobs"))
        let paths = try #require(vision.underRoot(URL(fileURLWithPath: "/ws/models"))?.boundPaths)
        #expect(paths.payload == "/ws/models/key/blobs/quants/vl.gguf")
        #expect(paths.mmproj == "/ws/models/key/blobs/mmproj/proj.gguf")
    }

    @Test func storedRoundTripsThroughTheManifestWireShape() throws {
        let specs = [try ggufTextSpec("org/a-GGUF", "a.gguf"),
                     try ggufVisionSpec("org/vl-GGUF", "w.gguf", "p.gguf"),
                     try mlxSpec("org/m-MLX-4bit"),
                     try mlxSpec("org/multi-mlx", subpath: "4bit")]
        for spec in specs {
            let stored = try #require(spec.toStored(base: "blobs"))
            let data = try JSONEncoder().encode(stored)
            #expect(try JSONDecoder().decode(Model.self, from: data) == stored)
        }
    }

    /// The wire tags are the crate's, so a manifest reads the same on either side.
    @Test func theWireTagsAreTheCratesSpellings() throws {
        let stored = try #require(
            try ggufVisionSpec("org/vl-GGUF", "w.gguf", "p.gguf").toStored(base: "blobs"))
        let json = try #require(JSONSerialization.jsonObject(
            with: try JSONEncoder().encode(stored)) as? [String: Any])

        #expect(json["type"] as? String == "gguf_vision")
        #expect(json["source"] as? String == "relative_files")
        #expect(json["model"] as? String == "blobs/w.gguf")
        #expect(json["mmproj"] as? String == "blobs/p.gguf")
        // A store form names no repo: those keys belong to the HuggingFace arm alone.
        #expect(json["org"] == nil)
    }
}
