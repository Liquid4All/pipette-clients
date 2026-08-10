import Foundation
import Testing

@testable import Pipette

/// `ModelArtifactStore` — the crate's `find` / `list` / `remove` / `ensure` and the
/// entry layout they share (`pipette-artifacts/src/model/store.rs`).
///
/// Each test builds its own temporary store, so the suite holds no shared global.
@MainActor struct ModelArtifactStoreTests {
    private func store(_ storage: FileStorage) -> ModelArtifactStore {
        ModelArtifactStore(modelsDir: storage.modelsDir)
    }

    @Test func findAndListSeeAnInstalledEntry() throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }
        let spec = try ggufTextSpec("org/a-GGUF", "a-Q4_0.gguf")
        try installEntry(storage, spec)

        #expect(store(storage).find(spec)?.declared == spec)
        #expect(store(storage).list().map(\.declared) == [spec])
    }

    /// An unreadable manifest is not an entry — the rule the quota accountant's garbage
    /// phase depends on, so `find` and `list` have to agree with it.
    @Test func aManifestlessDirectoryIsNotAnEntry() throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }
        try FileManager.default.createDirectory(
            at: storage.modelsDir.appendingPathComponent("junk", isDirectory: true),
            withIntermediateDirectories: true)

        #expect(store(storage).list().isEmpty)
    }

    @Test func removeDropsTheEntryWhole() throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }
        let spec = try ggufVisionSpec("org/vl-GGUF", "vl-Q4_0.gguf", "mmproj-f16.gguf")
        try installEntry(storage, spec)

        #expect(store(storage).remove(spec))
        #expect(store(storage).find(spec) == nil)
        // Nothing left to remove the second time.
        #expect(!store(storage).remove(spec))
    }

    /// A hit returns the bound model without fetching — the crate's `ensure` on a cache
    /// hit publishes nothing.
    @Test func ensureReturnsTheBoundModelWithoutFetchingOnAHit() async throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }
        let spec = try ggufTextSpec("org/a-GGUF", "a-Q4_0.gguf")
        try installEntry(storage, spec)
        var fetched = false

        let bound = try await store(storage).ensure(spec) { _ in fetched = true }

        #expect(!fetched, "a stored entry must not re-fetch")
        #expect(bound.boundPaths?.payload == store(storage).payloadPath(for: spec))
    }

    /// A miss fetches, and "ensured" means located afterwards: a fetch that lands nothing
    /// is an error by name, not a nil the caller has to interpret.
    @Test func ensureFetchesOnAMissAndRequiresTheEntryToResolve() async throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }
        let spec = try ggufTextSpec("org/a-GGUF", "a-Q4_0.gguf")
        var fetched = false

        await #expect(throws: ModelStoreError.self) {
            _ = try await store(storage).ensure(spec) { _ in fetched = true }
        }
        #expect(fetched, "a miss has to attempt the fetch")
    }

    /// A manifest beside missing weights is half-installed, so `ensure` treats it as a
    /// miss and fetches rather than handing back an entry that cannot run.
    @Test func ensureRefetchesAnEntryWhosePayloadNeverLanded() async throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }
        let spec = try ggufTextSpec("org/a-GGUF", "a-Q4_0.gguf")
        let entry = try installEntry(storage, spec)
        try FileManager.default.removeItem(
            at: entry.appendingPathComponent(Entry.blobsDirName, isDirectory: true))
        var fetched = false

        #expect(store(storage).find(spec) != nil, "the manifest is still readable")
        await #expect(throws: ModelStoreError.self) {
            _ = try await store(storage).ensure(spec) { _ in fetched = true }
        }
        #expect(fetched, "no weights means a miss, whatever the manifest says")
    }

    /// The regression this fix exists for. A repo that keeps its weights in one
    /// subdirectory and its projector in another puts the two in different places, and
    /// the old derivation — the projector filename *beside the weights* — pointed at a
    /// path nothing ever wrote. `to_stored` joins each independently (`stored.rs:71`).
    @Test func aNestedVisionRepoResolvesBothFilesWhereTheyWereWritten() throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }
        let spec = try ggufVisionSpec("org/vl-GGUF", "quants/vl-Q4_0.gguf", "mmproj/proj-f16.gguf")
        let entry = try installEntry(storage, spec)
        let blobs = entry.appendingPathComponent(Entry.blobsDirName, isDirectory: true)

        #expect(store(storage).payloadPath(for: spec) == blobs.appendingPathComponent("quants/vl-Q4_0.gguf").path)
        #expect(store(storage).mmprojPath(for: spec) == blobs.appendingPathComponent("mmproj/proj-f16.gguf").path)

        // The projector is the store's own answer, not a guess beside the weights.
        let mmprojPath = try #require(store(storage).mmprojPath(for: spec))
        #expect(!mmprojPath.contains("quants/mmproj"), "the projector was resolved beside the weights")
    }

    /// The flat case has to keep working: both files at the repo root resolve to
    /// `blobs/<name>`, which is what every shipped VL model looks like today.
    @Test func aFlatVisionRepoStillResolvesBothFiles() throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }
        let spec = try ggufVisionSpec("org/vl-GGUF", "vl-Q4_0.gguf", "mmproj-f16.gguf")
        let entry = try installEntry(storage, spec)
        let blobs = entry.appendingPathComponent(Entry.blobsDirName, isDirectory: true)

        #expect(store(storage).mmprojPath(for: spec) == blobs.appendingPathComponent("mmproj-f16.gguf").path)
    }

    /// The MLX bundle is `blobs/` itself — the installer moves the downloaded directory
    /// there whole. Pinned here because it is a deliberate divergence from the crate,
    /// which keeps `blobs/<prefix>`.
    @Test func anMlxBundleIsTheBlobsDirectoryItself() throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }
        let spec = try mlxSpec("org/m-MLX-4bit")
        let entry = try installEntry(storage, spec)

        #expect(store(storage).payloadPath(for: spec)
            == entry.appendingPathComponent(Entry.blobsDirName, isDirectory: true).path)
        #expect(store(storage).mmprojPath(for: spec) == nil, "only a vision model has a projector")
    }

    /// AFM ships with the OS: no key, so no entry and nothing to resolve.
    @Test func appleFoundationHasNoEntry() {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }

        #expect(store(storage).entryDir(for: .appleFoundationText) == nil)
        #expect(store(storage).payloadPath(for: .appleFoundationText) == nil)
    }

    /// `publishManifest` measures `blobs/` so a survey can total the store by reading
    /// manifests instead of walking payloads — the crate's `blobs_bytes`.
    @Test func publishRecordsTheBlobsSize() throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }
        let spec = try ggufTextSpec("org/a-GGUF", "a-Q4_0.gguf")
        try installEntry(storage, spec, payloadBytes: 4_096)

        store(storage).publishManifest(for: spec)

        let recorded = try #require(store(storage).find(spec)?.blobsBytes)
        #expect(recorded >= 4_096)
        // `blobs/` alone: the manifest carrying the number is not inside it.
        #expect(recorded < DiskUsage.bytes(at: try #require(store(storage).entryDir(for: spec))) + 4_096)
    }
}

/// The store form of a `Model` and the way back — `toStored` / `bindUnder`, mirroring
/// `pipette-artifacts/src/model/stored.rs`. One `Model` carries both halves, as upstream:
/// the relative arms *are* the stored form.
@MainActor struct StoredModelTests {
    /// `to_stored` joins each path under the entry independently and keeps the repo's
    /// nesting. The vision arm is the one that matters: two files, two subdirectories.
    @Test func toStoredKeepsEachPathsNesting() throws {
        let vision = try ggufVisionSpec("org/vl-GGUF", "quants/vl-Q4_0.gguf", "mmproj/proj-f16.gguf")

        let stored = try #require(vision.toStored(base: Entry.blobsDirName))
        guard case let .ggufVision(m) = stored,
              case let .relativeFiles(model, mmproj) = m.source else {
            Issue.record("expected a relativeFiles vision arm"); return
        }
        #expect(model.value == "blobs/quants/vl-Q4_0.gguf")
        #expect(mmproj.value == "blobs/mmproj/proj-f16.gguf")
    }

    @Test func toStoredMapsEachModelShape() throws {
        let text = try #require(try ggufTextSpec("org/a-GGUF", "a.gguf").toStored(base: "blobs"))
        guard case let .ggufText(t) = text, case let .relativeFile(path) = t.source else {
            Issue.record("expected a relativeFile arm"); return
        }
        #expect(path.value == "blobs/a.gguf")

        let mlx = try #require(try mlxSpec("org/m-MLX-4bit").toStored(base: "blobs"))
        guard case let .mlx(m) = mlx, case let .relativeDir(dir) = m.source else {
            Issue.record("expected a relativeDir arm"); return
        }
        #expect(dir.value == "blobs")

        // AFM ships with the OS: nothing stored, nothing to bind.
        #expect(Model.appleFoundationText.toStored(base: "blobs") == nil)
    }

    /// `bind_under` prefixes the recorded paths with the current root, which is what makes
    /// them survive a container-UUID change. The bound arms are the `Absolute*` ones, as
    /// upstream — `boundPaths` is what a caller wants out of them.
    @Test func bindUnderJoinsOntoTheCurrentEntry() throws {
        let stored = try #require(
            try ggufVisionSpec("org/vl-GGUF", "quants/vl.gguf", "mmproj/proj.gguf")
                .toStored(base: Entry.blobsDirName))
        let entry = URL(fileURLWithPath: "/somewhere/models/org__vl-GGUF")

        let bound = try #require(stored.bindUnder(entry))
        guard case let .ggufVision(m) = bound, case .absoluteFiles = m.source else {
            Issue.record("expected an absoluteFiles vision arm"); return
        }
        let paths = try #require(bound.boundPaths)
        #expect(paths.payload == "/somewhere/models/org__vl-GGUF/blobs/quants/vl.gguf")
        #expect(paths.mmproj == "/somewhere/models/org__vl-GGUF/blobs/mmproj/proj.gguf")
    }

    @Test func storedRoundTripsThroughTheManifestWireShape() throws {
        let specs = [try ggufTextSpec("org/a-GGUF", "a.gguf"),
                     try ggufVisionSpec("org/vl-GGUF", "w.gguf", "p.gguf"),
                     try mlxSpec("org/m-MLX-4bit")]
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

    /// An entry published before `stored` still resolves: the store falls back to
    /// deriving from `declared`, and the next publish backfills the field.
    @Test func anEntryWithoutStoredFallsBackToDerivationThenBackfills() throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }
        let spec = try ggufTextSpec("org/a-GGUF", "a-Q4_0.gguf")
        let entry = try installEntry(storage, spec)
        let store = ModelArtifactStore(modelsDir: storage.modelsDir)
        // Age the entry to what an older build wrote: everything but `stored`.
        var aged = try #require(ModelManifest.forInstalledEntry(atDir: entry))
        aged.stored = nil
        aged.writeQuietly(to: ModelManifest.manifestURL(inEntryDir: entry))

        #expect(store.find(spec)?.stored == nil)
        #expect(store.payloadPath(for: spec) != nil, "derivation still finds the weights")

        store.publishManifest(for: spec)

        #expect(store.find(spec)?.stored == spec.toStored(base: Entry.blobsDirName))
    }
}
