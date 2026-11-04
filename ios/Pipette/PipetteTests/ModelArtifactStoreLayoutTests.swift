import Foundation
import Testing

@testable import Pipette

/// Discovery over the entry layout (`models/<key>/{manifest.json, blobs/…}`): one entry
/// per model, recognized only through its manifest, never by sniffing the file tree.
@MainActor struct ModelArtifactStoreLayoutTests {
    /// Discovery also injects the built-in AFM model on capable devices; assert over
    /// the file-backed rows only.
    private func fileBacked(_ ws: PipetteWorkspace) -> [DiscoveredModel] {
        ws.availableModels().filter {
            if case .appleFoundationText = $0.source { false } else { true }
        }
    }

    /// A repo-relative path may name a subdirectory, nested as the crate's `to_stored` does.
    @Test func discoversAGgufEntryStoredUnderASubdirectory() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let spec = try ggufTextSpec("LiquidAI/LFM2.5-350M-GGUF", "quants/LFM2.5-350M-Q4_0.gguf")
        let entry = try installEntry(ws, spec, payloadBytes: 64_000)

        let model = try #require(fileBacked(ws).first)
        #expect(model.path
            == entry.appendingPathComponent("blobs/quants/LFM2.5-350M-Q4_0.gguf").path)
        // The leaf names the artifact; the key flattened the separator separately.
        #expect(model.name == "LFM2.5-350M-Q4_0.gguf")
    }

    @Test func discoversAGgufEntryWithItsManifestProvenance() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let spec = try ggufTextSpec("LiquidAI/LFM2.5-350M-GGUF", "LFM2.5-350M-Q4_0.gguf")
        let entry = try installEntry(ws, spec, payloadBytes: 64_000)

        let model = try #require(fileBacked(ws).first)
        #expect(model.hfRepo == "LiquidAI/LFM2.5-350M-GGUF")
        #expect(model.name == "LFM2.5-350M-Q4_0.gguf")
        #expect(model.path == entry.appendingPathComponent("blobs/LFM2.5-350M-Q4_0.gguf").path)
        #expect(model.familyId == "lfm2.5-350m")
        #expect(model.sizeBytes == entrySizeBytes(at: entry))
    }

    @Test func anMlxEntryPointsAtItsBlobsDirectory() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let spec = try mlxSpec("LiquidAI/LFM2.5-350M-MLX-4bit")
        let entry = try installEntry(ws, spec)

        let model = try #require(fileBacked(ws).first)
        guard case .mlx = model.source else {
            Issue.record("installed model should be .mlx, got \(model)")
            return
        }
        #expect(model.path == entry.appendingPathComponent("blobs").path)
        #expect(model.name == "LFM2.5-350M-MLX-4bit")
    }

    @Test func anMlxSubpathEntryIsNamedForItsSubpathLeaf() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        try installEntry(ws, try mlxSpec("org/multi-mlx", subpath: "variant-4bit"))

        #expect(fileBacked(ws).first?.name == "variant-4bit")
    }

    /// A vision model is one entry: weights and projector share it, the size covers
    /// both, and the resolved projector path points back into the same `blobs/`.
    @Test func aVisionEntryIsOneModelCoveringBothFiles() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let spec = try ggufVisionSpec("org/vl-GGUF", "vl-Q4_0.gguf", "mmproj-vl-F16.gguf")
        let entry = try installEntry(ws, spec, payloadBytes: 64_000)

        let models = fileBacked(ws)
        #expect(models.count == 1)
        let model = try #require(models.first)
        #expect(model.sizeBytes == entrySizeBytes(at: entry))
        guard case .ggufVision = model.source else {
            Issue.record("expected a .ggufVision model, got \(model.source)")
            return
        }
        // The discovered row has no projector path: the store binds it from the coordinate.
        #expect(ws.models.bound(model.source)?
            .boundPaths?.mmproj == entry.appendingPathComponent("blobs/mmproj-vl-F16.gguf").path)
    }

    @Test func skipsAManifestLessEntryDirectory() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let orphan = ws.modelsDir.appendingPathComponent("org__repo__x.gguf/blobs", isDirectory: true)
        try FileManager.default.createDirectory(at: orphan, withIntermediateDirectories: true)
        try Data("gguf".utf8).write(to: orphan.appendingPathComponent("x.gguf"))

        #expect(fileBacked(ws).isEmpty)
    }

    /// A version bump strands the store on purpose: a v1 manifest is unreadable, so
    /// its entry is not a model (and the sweeper reclaims it as garbage).
    @Test func skipsAnEntryFromASupersededManifestVersion() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let spec = try ggufTextSpec("org/repo-GGUF", "x-Q4_0.gguf")
        let entry = try installEntry(ws, spec)
        let json = #"{"manifest_version":1,"declared":{"type":"gguf_text","source":"huggingface","org":"org","repo_name":"repo-GGUF","path":"x-Q4_0.gguf"}}"#
        try Data(json.utf8).write(to: ModelManifest.manifestURL(inEntryDir: entry))

        #expect(fileBacked(ws).isEmpty)
    }

    /// No migrator: the pre-entry bucket tree (`models/<org>/<repo>/x.gguf` plus its
    /// `.gguf.pipette-manifest.json` sidecar) is not an entry.
    @Test func theOldBucketTreeYieldsNoModels() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let bucket = ws.modelsDir.appendingPathComponent("mistralai/Some-GGUF", isDirectory: true)
        try FileManager.default.createDirectory(at: bucket, withIntermediateDirectories: true)
        try Data("gguf".utf8).write(to: bucket.appendingPathComponent("Some-Q5_K_M.gguf"))
        try Data("{}".utf8).write(
            to: bucket.appendingPathComponent("Some-Q5_K_M.gguf.pipette-manifest.json"))

        #expect(fileBacked(ws).isEmpty)
    }

    /// A manifest whose payload is gone (an interrupted install) is not a runnable
    /// model, so discovery skips it rather than offering a row that can't load.
    @Test func skipsAnEntryWhosePayloadIsMissing() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let spec = try ggufTextSpec("org/repo-GGUF", "x-Q4_0.gguf")
        let entry = try installEntry(ws, spec)
        try FileManager.default.removeItem(at: entry.appendingPathComponent("blobs/x-Q4_0.gguf"))

        #expect(fileBacked(ws).isEmpty)
    }

    @Test func deleteRemovesTheWholeEntryAndKeepsModelsDir() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let spec = try ggufVisionSpec("org/vl-GGUF", "vl-Q4_0.gguf", "mmproj-vl-F16.gguf")
        let entry = try installEntry(ws, spec)

        ws.deleteModel(try #require(fileBacked(ws).first))

        #expect(!FileManager.default.fileExists(atPath: entry.path))
        #expect(FileManager.default.fileExists(atPath: ws.modelsDir.path))
        #expect(fileBacked(ws).isEmpty)
    }

    @Test func entryDirectoriesAreExcludedFromBackup() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let entry = try installEntry(ws, try ggufTextSpec("org/repo-GGUF", "x-Q4_0.gguf"))

        _ = ws.availableModels()

        let values = try entry.resourceValues(forKeys: [.isExcludedFromBackupKey])
        #expect(values.isExcludedFromBackup == true)
    }

    /// The manifest's `stored`, not the directory holding it, says where the bytes are:
    /// it names the key's entry, as the crate's does, so a copy under another name binds
    /// nothing of its own and is not a runnable model.
    @Test func aManifestInAMisnamedDirectoryBindsNothing() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let spec = try mlxSpec("LiquidAI/LFM2.5-350M-MLX-4bit")
        let misnamed = ws.modelsDir.appendingPathComponent("not-the-key", isDirectory: true)
        let blobs = misnamed.appendingPathComponent("blobs", isDirectory: true)
        try FileManager.default.createDirectory(at: blobs, withIntermediateDirectories: true)
        for name in mlxBundleFiles { try Data("x".utf8).write(to: blobs.appendingPathComponent(name)) }
        try manifestFixture(spec).writeQuietly(to: ModelManifest.manifestURL(inEntryDir: misnamed))

        #expect(fileBacked(ws).isEmpty)
    }
}
