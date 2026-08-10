import Foundation
import Testing

@testable import Pipette

/// Discovery over the entry layout (`models/<key>/{manifest.json, blobs/…}`):
/// one entry per model, recognized only through its manifest and never by sniffing the
/// file tree. Each test injects its own temporary `FileStorage`, so the suite carries
/// no shared global and runs in parallel.
@MainActor struct ModelEntryLayoutTests {
    /// Discovery also injects the built-in AFM model on capable devices; assert over
    /// the file-backed rows only.
    private func fileBacked(_ storage: FileStorage) -> [DiscoveredModel] {
        storage.availableModels().filter {
            if case .appleFoundationText = $0.source { false } else { true }
        }
    }

    /// A repo-relative path may name a subdirectory, and the store nests it the way the
    /// crate's `to_stored` does. Widening `path` to `RepoSubpath` without creating the
    /// intermediate directory accepted the plan and then failed the move — worse than the
    /// parse-time refusal it replaced, because the download is already paid for.
    @Test func discoversAGgufEntryStoredUnderASubdirectory() throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }
        let spec = try ggufTextSpec("LiquidAI/LFM2.5-350M-GGUF", "quants/LFM2.5-350M-Q4_0.gguf")
        let entry = try installEntry(storage, spec, payloadBytes: 64_000)

        let model = try #require(fileBacked(storage).first)
        #expect(model.path
            == entry.appendingPathComponent("blobs/quants/LFM2.5-350M-Q4_0.gguf").path)
        // The leaf names the artifact; the key flattened the separator separately.
        #expect(model.name == "LFM2.5-350M-Q4_0.gguf")
    }

    @Test func discoversAGgufEntryWithItsManifestProvenance() throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }
        let spec = try ggufTextSpec("LiquidAI/LFM2.5-350M-GGUF", "LFM2.5-350M-Q4_0.gguf")
        let entry = try installEntry(storage, spec, payloadBytes: 64_000)

        let model = try #require(fileBacked(storage).first)
        #expect(model.hfRepo == "LiquidAI/LFM2.5-350M-GGUF")
        #expect(model.name == "LFM2.5-350M-Q4_0.gguf")
        #expect(model.path == entry.appendingPathComponent("blobs/LFM2.5-350M-Q4_0.gguf").path)
        #expect(model.familyId == "lfm2.5-350m")
        #expect(model.sizeBytes == DiskUsage.bytes(at: entry))
    }

    @Test func anMlxEntryPointsAtItsBlobsDirectory() throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }
        let spec = try mlxSpec("LiquidAI/LFM2.5-350M-MLX-4bit")
        let entry = try installEntry(storage, spec)

        let model = try #require(fileBacked(storage).first)
        guard case .mlx = model.source else {
            Issue.record("installed model should be .mlx, got \(model)")
            return
        }
        #expect(model.path == entry.appendingPathComponent("blobs").path)
        #expect(model.name == "LFM2.5-350M-MLX-4bit")
    }

    @Test func anMlxSubpathEntryIsNamedForItsSubpathLeaf() throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }
        try installEntry(storage, try mlxSpec("org/multi-mlx", subpath: "variant-4bit"))

        #expect(fileBacked(storage).first?.name == "variant-4bit")
    }

    /// A vision model is one entry: weights and projector share it, the size covers
    /// both, and the resolved projector path points back into the same `blobs/`.
    @Test func aVisionEntryIsOneModelCoveringBothFiles() throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }
        let spec = try ggufVisionSpec("org/vl-GGUF", "vl-Q4_0.gguf", "mmproj-vl-F16.gguf")
        let entry = try installEntry(storage, spec, payloadBytes: 64_000)

        let models = fileBacked(storage)
        #expect(models.count == 1)
        let model = try #require(models.first)
        #expect(model.sizeBytes == DiskUsage.bytes(at: entry))
        guard case .ggufVision = model.source else {
            Issue.record("expected a .ggufVision model, got \(model.source)")
            return
        }
        // The projector is not on the discovered row — the coordinate names it, and the
        // store binds it. Asserted where the value actually comes from.
        #expect(ModelArtifactStore.bound(inEntryDir: entry, declared: model.source)?
            .boundPaths?.mmproj == entry.appendingPathComponent("blobs/mmproj-vl-F16.gguf").path)
    }

    @Test func skipsAManifestLessEntryDirectory() throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }
        let orphan = storage.modelsDir.appendingPathComponent("org__repo__x.gguf/blobs", isDirectory: true)
        try FileManager.default.createDirectory(at: orphan, withIntermediateDirectories: true)
        try Data("gguf".utf8).write(to: orphan.appendingPathComponent("x.gguf"))

        #expect(fileBacked(storage).isEmpty)
    }

    /// A version bump strands the store on purpose: a v1 manifest is unreadable, so
    /// its entry is not a model (and the sweeper reclaims it as garbage).
    @Test func skipsAnEntryFromASupersededManifestVersion() throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }
        let spec = try ggufTextSpec("org/repo-GGUF", "x-Q4_0.gguf")
        let entry = try installEntry(storage, spec)
        let json = #"{"manifest_version":1,"declared":{"type":"gguf_text","source":"huggingface","org":"org","repo_name":"repo-GGUF","path":"x-Q4_0.gguf"}}"#
        try Data(json.utf8).write(to: ModelManifest.manifestURL(inEntryDir: entry))

        #expect(fileBacked(storage).isEmpty)
    }

    /// No migrator: the pre-entry bucket tree (`models/<org>/<repo>/x.gguf` plus its
    /// `.gguf.pipette-manifest.json` sidecar) is simply not a valid entry.
    @Test func theOldBucketTreeYieldsNoModels() throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }
        let bucket = storage.modelsDir.appendingPathComponent("mistralai/Some-GGUF", isDirectory: true)
        try FileManager.default.createDirectory(at: bucket, withIntermediateDirectories: true)
        try Data("gguf".utf8).write(to: bucket.appendingPathComponent("Some-Q5_K_M.gguf"))
        try Data("{}".utf8).write(
            to: bucket.appendingPathComponent("Some-Q5_K_M.gguf.pipette-manifest.json"))

        #expect(fileBacked(storage).isEmpty)
    }

    /// A manifest whose payload is gone (an interrupted install) is not a runnable
    /// model, so discovery skips it rather than offering a row that can't load.
    @Test func skipsAnEntryWhosePayloadIsMissing() throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }
        let spec = try ggufTextSpec("org/repo-GGUF", "x-Q4_0.gguf")
        let entry = try installEntry(storage, spec)
        try FileManager.default.removeItem(at: entry.appendingPathComponent("blobs/x-Q4_0.gguf"))

        #expect(fileBacked(storage).isEmpty)
    }

    @Test func deleteRemovesTheWholeEntryAndKeepsModelsDir() throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }
        let spec = try ggufVisionSpec("org/vl-GGUF", "vl-Q4_0.gguf", "mmproj-vl-F16.gguf")
        let entry = try installEntry(storage, spec)

        storage.deleteModel(try #require(fileBacked(storage).first))

        #expect(!FileManager.default.fileExists(atPath: entry.path))
        #expect(FileManager.default.fileExists(atPath: storage.modelsDir.path))
        #expect(fileBacked(storage).isEmpty)
    }

    @Test func entryDirectoriesAreExcludedFromBackup() throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }
        let entry = try installEntry(storage, try ggufTextSpec("org/repo-GGUF", "x-Q4_0.gguf"))

        _ = storage.availableModels()

        let values = try entry.resourceValues(forKeys: [.isExcludedFromBackupKey])
        #expect(values.isExcludedFromBackup == true)
    }

    /// The manifest, not the directory name, is the provenance: an entry sitting in a
    /// mis-named directory is still reported with the manifest's coordinate.
    @Test func reportsTheManifestCoordinateNotTheDirectoryName() throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }
        let spec = try mlxSpec("LiquidAI/LFM2.5-350M-MLX-4bit")
        let misnamed = storage.modelsDir.appendingPathComponent("not-the-key", isDirectory: true)
        let blobs = misnamed.appendingPathComponent("blobs", isDirectory: true)
        try FileManager.default.createDirectory(at: blobs, withIntermediateDirectories: true)
        for name in mlxBundleFiles { try Data("x".utf8).write(to: blobs.appendingPathComponent(name)) }
        ModelManifest(declared: spec).writeQuietly(to: ModelManifest.manifestURL(inEntryDir: misnamed))

        let model = try #require(fileBacked(storage).first)
        #expect(model.hfRepo == "LiquidAI/LFM2.5-350M-MLX-4bit")
    }
}
