import Foundation
import Testing

@testable import Pipette

/// The crate's `find` / `list` / `remove` / `ensure` and the entry layout they share
/// (`crates/pipette-artifacts/src/model/store.rs`).
@MainActor struct ModelArtifactStoreTests {
    private static let longAgo = Date(timeIntervalSince1970: 1_600_000_000)

    private func store(_ ws: PipetteWorkspace) -> ModelArtifactStore {
        ModelArtifactStore(modelsDir: ws.modelsDir)
    }

    // MARK: - find / list / remove

    @Test func findAndListSeeAnInstalledEntry() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let spec = try ggufTextSpec("org/a-GGUF", "a-Q4_0.gguf")
        try installEntry(ws, spec)

        #expect(try store(ws).find(spec)?.declared == spec)
        #expect(try store(ws).list().map(\.declared) == [spec])
    }

    /// The quota accountant's garbage phase relies on a manifestless directory not being
    /// an entry, so `find` and `list` have to agree with it.
    @Test func aManifestlessDirectoryIsNotAnEntry() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        try FileManager.default.createDirectory(
            at: ws.modelsDir.appendingPathComponent("junk", isDirectory: true),
            withIntermediateDirectories: true)

        #expect(try store(ws).list().isEmpty)
    }

    @Test func listIsEmptyForAFreshStore() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("absent-\(UUID().uuidString)/models", isDirectory: true)

        #expect(try ModelArtifactStore(modelsDir: root).list().isEmpty)
    }

    /// Sorted by declared identity whatever order they were stored in, and a dot-directory
    /// is never an entry, even one holding a manifest.
    @Test func listReturnsStoredModelsSortedAndSkipsStaging() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let zephyr = try ggufTextSpec("meta/zephyr", "Q4.gguf")
        let llama = try ggufTextSpec("meta/llama", "Q4.gguf")
        try installEntry(ws, zephyr)
        let llamaEntry = try installEntry(ws, llama)
        let staging = ws.modelsDir.appendingPathComponent(".staging", isDirectory: true)
        try FileManager.default.createDirectory(
            at: staging.appendingPathComponent("orphan/blobs", isDirectory: true),
            withIntermediateDirectories: true)
        try FileManager.default.copyItem(at: ModelManifest.manifestURL(inEntryDir: llamaEntry),
                                         to: ModelManifest.manifestURL(inEntryDir: staging))

        #expect(try store(ws).list().map(\.declared) == [llama, zephyr])
    }

    @Test func removeDeletesAStoredEntryAndReportsAbsence() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let spec = try ggufVisionSpec("org/vl-GGUF", "vl-Q4_0.gguf", "mmproj-f16.gguf")
        try installEntry(ws, spec)

        #expect(store(ws).remove(spec))
        #expect(try store(ws).find(spec) == nil)
        #expect(!store(ws).remove(spec))
    }

    // MARK: - Reading policy

    /// A token found on disk is dropped on read, so an in-memory `declared` never carries a
    /// secret.
    @Test func ensureAndFindStripAuthTokens() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let spec = try ggufTextSpec("org/gated-GGUF", "g-Q4_0.gguf")
        let entry = try installEntry(ws, spec)
        try editManifest(entry) { json in
            var declared = json["declared"] as? [String: Any] ?? [:]
            declared["auth_token"] = "hf_secret"
            json["declared"] = declared
        }

        let found = try #require(try store(ws).find(spec))
        #expect(found.declared.repo?.authToken == nil)
    }

    /// A `stored` that is no longer the declared model's store form is refused rather than
    /// followed.
    @Test func aDriftedManifestIsCorrupt() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let spec = try ggufTextSpec("org/a-GGUF", "a-Q4_0.gguf")
        let key = try #require(ModelStorageKey.of(spec))
        let entry = try installEntry(ws, spec)
        try editManifest(entry) { json in
            var stored = json["stored"] as? [String: Any] ?? [:]
            stored["path"] = "\(key.blobsDir)/TAMPERED.gguf"
            json["stored"] = stored
        }

        let error = #expect(throws: ModelStoreError.self) { _ = try store(ws).find(spec) }
        #expect({ if case .corrupt = error { true } else { false } }(), "\(String(describing: error))")
    }

    @Test func unsupportedManifestVersionIsCorrupt() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let spec = try ggufTextSpec("org/a-GGUF", "a-Q4_0.gguf")
        try editManifest(try installEntry(ws, spec)) { $0["manifest_version"] = 99 }

        let error = #expect(throws: ModelStoreError.self) { _ = try store(ws).find(spec) }
        #expect({ if case .corrupt = error { true } else { false } }(), "\(String(describing: error))")
    }

    @Test func aBrokenManifestSurfacesAsParseError() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let spec = try ggufTextSpec("org/a-GGUF", "a-Q4_0.gguf")
        let entry = try installEntry(ws, spec)
        try Data("not json".utf8).write(to: ModelManifest.manifestURL(inEntryDir: entry))

        let found = #expect(throws: ModelStoreError.self) { _ = try store(ws).find(spec) }
        let listed = #expect(throws: ModelStoreError.self) { _ = try store(ws).list() }
        #expect({ if case .parse = found { true } else { false } }(), "\(String(describing: found))")
        #expect({ if case .parse = listed { true } else { false } }(), "\(String(describing: listed))")
    }

    /// A record from before `last_used_at` fails to decode rather than being bridged.
    @Test func aV1ManifestIsUnreadable() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let spec = try ggufTextSpec("org/a-GGUF", "a-Q4_0.gguf")
        try editManifest(try installEntry(ws, spec)) { json in
            json["manifest_version"] = 1
            json.removeValue(forKey: "last_used_at")
        }

        let error = #expect(throws: ModelStoreError.self) { _ = try store(ws).find(spec) }
        #expect({ if case .parse = error { true } else { false } }(), "\(String(describing: error))")
    }

    // MARK: - ensure

    @Test func ensureReturnsTheBoundModelWithoutFetchingOnAHit() async throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let spec = try ggufTextSpec("org/a-GGUF", "a-Q4_0.gguf")
        try installEntry(ws, spec)
        var fetched = false

        let bound = try await store(ws).ensure(spec) { _ in fetched = true }

        #expect(!fetched, "a stored entry must not re-fetch")
        #expect(bound.boundPaths?.payload == store(ws).payloadPath(for: spec))
    }

    /// "Ensured" means located afterwards: a fetch that lands nothing is an error by name,
    /// not a nil the caller has to interpret.
    @Test func ensureFetchesOnAMissAndRequiresTheEntryToResolve() async throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let spec = try ggufTextSpec("org/a-GGUF", "a-Q4_0.gguf")
        var fetched = false

        await #expect(throws: ModelStoreError.self) {
            _ = try await store(ws).ensure(spec) { _ in fetched = true }
        }
        #expect(fetched, "a miss has to attempt the fetch")
    }

    @Test func ensureRefetchesAnEntryWhosePayloadNeverLanded() async throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let spec = try ggufTextSpec("org/a-GGUF", "a-Q4_0.gguf")
        let entry = try installEntry(ws, spec)
        try FileManager.default.removeItem(
            at: entry.appendingPathComponent(Entry.blobsDirName, isDirectory: true))
        var fetched = false

        #expect(try store(ws).find(spec) != nil, "the manifest is still readable")
        await #expect(throws: ModelStoreError.self) {
            _ = try await store(ws).ensure(spec) { _ in fetched = true }
        }
        #expect(fetched, "no weights means a miss, whatever the manifest says")
    }

    // MARK: - Entries written before `stored` was measured from `models/`

    /// The shape a released build wrote: paths measured from the entry (`blobs/w.gguf`), not
    /// `models/`, and no `fetched_at`. `storedOverride` replaces the path outright, for the
    /// MLX subtree whose recorded form is not a prefix strip.
    private func writeLegacyManifest(
        _ ws: PipetteWorkspace, _ spec: Model, storedOverride: String? = nil
    ) throws {
        let key = try #require(ModelStorageKey.of(spec))
        let entry = ws.models.modelsDir.appendingPathComponent(key.relativeDir)
        let url = ModelManifest.manifestURL(inEntryDir: entry)
        var json = try #require(
            try JSONSerialization.jsonObject(with: try Data(contentsOf: url)) as? [String: Any])
        var stored = try #require(json["stored"] as? [String: Any])
        for field in ["path", "dir", "model", "mmproj"] {
            guard let recorded = stored[field] as? String else { continue }
            stored[field] = storedOverride
                ?? String(recorded.dropFirst("\(key.value)/".count))
        }
        json["stored"] = stored
        json.removeValue(forKey: "fetched_at")
        try JSONSerialization.data(withJSONObject: json).write(to: url)
    }

    /// The layout changed without a `manifest_version` bump, so released builds left entries
    /// in the old shape. Refusing them would fail the cell, and the re-download that would
    /// fix it is deduplicated away because the file is still there.
    @Test func anEntryWrittenBeforeTheModelsRelativeLayoutStillResolves() async throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let spec = try ggufTextSpec("org/a-GGUF", "a-Q4_0.gguf")
        try installEntry(ws, spec, lastUsedAt: Self.longAgo)
        try writeLegacyManifest(ws, spec)

        let bound = try await store(ws).ensure(spec) { _ in
            Issue.record("an entry already on disk must not re-download")
        }

        let payload = try #require(bound.boundPaths?.payload)
        #expect(FileManager.default.fileExists(atPath: payload),
                "the re-rooted path must name the bytes that are actually there")
        let reread = try #require(try store(ws).find(spec))
        #expect(reread.stored == spec.toStored(base: try #require(ModelStorageKey.of(spec)).blobsDir))
        // An absent `fetched_at` falls back to the file's `last_used_at`, not now, so the
        // entry keeps its place in the eviction order. Resolving it was a hit, so
        // `lastUsedAt` moved on.
        #expect(reread.fetchedAt == Self.longAgo)
        #expect(reread.lastUsedAt > Self.longAgo)
    }

    /// An MLX repo subtree recorded `blobs` with its `prefix` dropped, so re-deriving from
    /// `declared` would name `blobs/<prefix>`, where none of its files are.
    @Test func aLegacyMlxSubtreeKeepsTheDirectoryItRecorded() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let spec = try mlxSpec("org/multi-mlx", subpath: "4bit")
        try installEntry(ws, spec)
        try writeLegacyManifest(ws, spec, storedOverride: Entry.blobsDirName)

        let manifest = try #require(try store(ws).find(spec))

        let key = try #require(ModelStorageKey.of(spec))
        #expect(manifest.stored.storedRelativePaths == [key.blobsDir])
        #expect(manifest.stored != spec.toStored(base: key.blobsDir),
                "the derived form names blobs/<prefix>, which is not where these files are")
    }

    /// The re-rooting is for one known older shape, not a licence to accept any path a
    /// manifest happens to carry.
    @Test func aStoredFormUnderAnotherKeyIsStillCorrupt() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let spec = try ggufTextSpec("org/a-GGUF", "a-Q4_0.gguf")
        try installEntry(ws, spec)
        try writeLegacyManifest(ws, spec, storedOverride: "someone__else/blobs/a-Q4_0.gguf")

        #expect(throws: ModelStoreError.self) { try store(ws).find(spec) }
    }

    @Test func ensureSeedsLastUsedAtOnPublish() async throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let spec = try ggufTextSpec("org/a-GGUF", "a-Q4_0.gguf")
        let store = store(ws)

        _ = try await store.ensure(spec) { try landAndPublish(store, $0) }

        let manifest = try #require(try store.find(spec))
        #expect(manifest.lastUsedAt == manifest.fetchedAt)
    }

    @Test func ensureTouchesLastUsedAtOnACacheHit() async throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let spec = try ggufTextSpec("org/a-GGUF", "a-Q4_0.gguf")
        try installEntry(ws, spec, lastUsedAt: Self.longAgo, fetchedAt: Self.longAgo)
        var fetched = false

        _ = try await store(ws).ensure(spec) { _ in fetched = true }

        #expect(!fetched, "a hit does not re-fetch")
        let manifest = try #require(try store(ws).find(spec))
        #expect(manifest.fetchedAt == Self.longAgo)
        #expect(manifest.lastUsedAt > Self.longAgo)
    }

    @Test func findDoesNotTouchLastUsedAt() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let spec = try ggufTextSpec("org/a-GGUF", "a-Q4_0.gguf")
        try installEntry(ws, spec, lastUsedAt: Self.longAgo)

        _ = try store(ws).find(spec)

        #expect(try store(ws).find(spec)?.lastUsedAt == Self.longAgo)
    }

    @Test func aFailedTouchStillResolves() async throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let spec = try ggufTextSpec("org/a-GGUF", "a-Q4_0.gguf")
        let entry = try installEntry(ws, spec, lastUsedAt: Self.longAgo)
        let fm = FileManager.default
        try fm.setAttributes([.posixPermissions: 0o555], ofItemAtPath: entry.path)
        defer { try? fm.setAttributes([.posixPermissions: 0o755], ofItemAtPath: entry.path) }

        let bound = try await store(ws).ensure(spec) { _ in Issue.record("a hit must not fetch") }

        #expect(bound.boundPaths != nil)
        #expect(try store(ws).find(spec)?.lastUsedAt == Self.longAgo)
    }

    /// The store persists no credential, but the fetch it drives gets the caller's, or
    /// every gated download would go out unauthenticated.
    @Test func ensureHandsTheAuthTokenToTheFetch() async throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        var repo = try HfRepo.parseOrgRepo("org/gated-GGUF")
        repo.authToken = try AuthToken("hf_secret")
        let spec = Model.ggufText(GgufText(source: .huggingFace(
            repo: repo, path: try RepoSubpath("g-Q4_0.gguf"), sha256: nil)))
        let store = store(ws)
        var seen: String?

        _ = try await store.ensure(spec) { declared in
            seen = declared.repo?.authToken?.value
            try landAndPublish(store, declared)
        }

        #expect(seen == "hf_secret")
        let manifest = ModelManifest.manifestURL(inEntryDir: try #require(store.entryDir(for: spec)))
        #expect(!String(decoding: try Data(contentsOf: manifest), as: UTF8.self).contains("hf_secret"),
                "the store persists no credential")
    }

    // MARK: - Layout

    /// Weights and projector in different subdirectories resolve independently, as the
    /// crate's `to_stored` joins them (`model/stored.rs`), not the projector beside the weights.
    @Test func aNestedVisionRepoResolvesBothFilesWhereTheyWereWritten() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let spec = try ggufVisionSpec("org/vl-GGUF", "quants/vl-Q4_0.gguf", "mmproj/proj-f16.gguf")
        let entry = try installEntry(ws, spec)
        let blobs = entry.appendingPathComponent(Entry.blobsDirName, isDirectory: true)

        #expect(store(ws).payloadPath(for: spec) == blobs.appendingPathComponent("quants/vl-Q4_0.gguf").path)
        #expect(store(ws).mmprojPath(for: spec) == blobs.appendingPathComponent("mmproj/proj-f16.gguf").path)

        let mmprojPath = try #require(store(ws).mmprojPath(for: spec))
        #expect(!mmprojPath.contains("quants/mmproj"), "the projector was resolved beside the weights")
    }

    /// Every shipped VL model keeps both files at the repo root.
    @Test func aFlatVisionRepoStillResolvesBothFiles() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let spec = try ggufVisionSpec("org/vl-GGUF", "vl-Q4_0.gguf", "mmproj-f16.gguf")
        let entry = try installEntry(ws, spec)
        let blobs = entry.appendingPathComponent(Entry.blobsDirName, isDirectory: true)

        #expect(store(ws).mmprojPath(for: spec) == blobs.appendingPathComponent("mmproj-f16.gguf").path)
    }

    @Test func dirModelPrefixNestsUnderEntryBareIsTheEntry() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let bare = try mlxSpec("org/m-MLX-4bit")
        let bareEntry = try installEntry(ws, bare)
        let subtree = try mlxSpec("org/multi-mlx", subpath: "4bit")
        let subtreeEntry = try installEntry(ws, subtree)

        #expect(store(ws).payloadPath(for: bare)
            == bareEntry.appendingPathComponent(Entry.blobsDirName, isDirectory: true).path)
        #expect(store(ws).payloadPath(for: subtree)
            == subtreeEntry.appendingPathComponent("blobs/4bit", isDirectory: true).path)
        #expect(store(ws).mmprojPath(for: bare) == nil, "only a vision model has a projector")
    }

    /// AFM ships with the OS, so it has no storage key: the crate's `NotStorable`.
    @Test func appleFoundationHasNoEntry() {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }

        #expect(store(ws).entryDir(for: .appleFoundationText) == nil)
        #expect(store(ws).payloadPath(for: .appleFoundationText) == nil)
        #expect(throws: ModelStoreError.notStorable(Model.appleFoundationText.artifactName)) {
            _ = try store(ws).find(.appleFoundationText)
        }
    }

    /// The crate's `blobs_bytes`: a survey totals the store from manifests instead of
    /// walking payloads.
    @Test func publishRecordsTheBlobsSize() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let spec = try ggufTextSpec("org/a-GGUF", "a-Q4_0.gguf")
        try installEntry(ws, spec, payloadBytes: 4_096)

        store(ws).publishManifest(for: spec)

        let recorded = try #require(try store(ws).find(spec)?.blobsBytes)
        #expect(recorded >= 4_096)
        // `blobs/` alone: the manifest is not inside it.
        #expect(recorded < entrySizeBytes(at: try #require(store(ws).entryDir(for: spec))) + 4_096)
    }

    // MARK: - Type collision

    /// `ModelStorageKey` omits the model type, so two directory models of one `org/repo`
    /// with no `prefix` share an entry. Handing back the other type's manifest would fail
    /// at load as an engine complaining about a format it was never given.
    @Test func aSecondModelTypeOnOneRepoIsRefusedByName() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let mlx = try mlxSpec("org/shared-repo")
        let onnx = try onnxGenaiSpec("org/shared-repo")
        #expect(ModelStorageKey.of(mlx) == ModelStorageKey.of(onnx))
        try installEntry(ws, mlx, payloadBytes: 64)

        do {
            _ = try ws.models.find(onnx)
            Issue.record("a model of another type in this entry must be refused")
        } catch let error as ModelStoreError {
            guard case let .typeCollision(_, stored, requested) = error else {
                Issue.record("expected a typeCollision, got \(error)"); return
            }
            #expect(stored == .mlx)
            #expect(requested == .onnxGenai)
        }
        // The guard refuses the mismatch, not the entry.
        #expect(try ws.models.find(mlx) != nil)
    }

    /// A prefix is the remedy the type-collision refusal names.
    @Test func aPrefixGivesEachDirectoryModelItsOwnEntry() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let mlx = try mlxSpec("org/shared-repo", subpath: "mlx-4bit")
        let onnx = try onnxGenaiSpec("org/shared-repo", subpath: "cpu-int4")
        #expect(ModelStorageKey.of(mlx) != ModelStorageKey.of(onnx))

        try installEntry(ws, mlx, payloadBytes: 64)
        try installEntry(ws, onnx, payloadBytes: 64)

        #expect(try ws.models.find(mlx) != nil)
        #expect(try ws.models.find(onnx) != nil)
    }
}
