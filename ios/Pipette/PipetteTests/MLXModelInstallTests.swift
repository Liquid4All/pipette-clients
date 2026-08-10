import Foundation
import Testing

@testable import Pipette

/// The install gate (`MLXModelLayout`) and installer (`MLXModelInstaller`): a model
/// is relocated into its store entry + recorded only if it's a *complete* MLX
/// directory. Each test that installs injects its own temporary `FileStorage`, so the
/// suite carries no shared global and runs in parallel.
@MainActor struct MLXModelInstallTests {

    // MARK: - MLXModelLayout gate

    @Test func layoutAcceptsCompleteModel() throws {
        let dir = try makeModelDir(files: ["config.json", "model.safetensors", "tokenizer.json"])
        #expect(MLXModelLayout.missing(in: dir).isEmpty)
    }

    @Test func layoutReportsEachMissingPiece() throws {
        #expect(MLXModelLayout.missing(in: try makeModelDir(files: [])).sorted()
            == ["*.safetensors", "config.json", "tokenizer"].sorted())
        #expect(MLXModelLayout.missing(in: try makeModelDir(files: ["config.json", "tokenizer.json"]))
            == ["*.safetensors"])
        #expect(MLXModelLayout.missing(in: try makeModelDir(files: ["model.safetensors", "tokenizer.model"]))
            == ["config.json"])
        #expect(MLXModelLayout.missing(in: try makeModelDir(files: ["config.json", "model.safetensors"]))
            == ["tokenizer"])
    }

    // MARK: - MLXModelInstaller

    @Test func installsRootModelAndDiscoversIt() throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }
        let ref = try HFModelRef.parse(repo: "LiquidAI/LFM2.5-350M-MLX-4bit")
        let src = try makeModelDir(files: ["config.json", "model.safetensors", "tokenizer.json"])

        try MLXModelInstaller(ref: ref, storage: storage).install(from: src)

        let spec = Model.mlx(ref.asMlx())
        let dest = try #require(storage.modelStore.blobsDir(for: spec))
        #expect(FileManager.default.fileExists(atPath: dest.appendingPathComponent("config.json").path))
        let model = try #require(storage.availableModels().first { $0.hfRepo == ref.repo.description })
        guard case .mlx = model.source else {
            Issue.record("installed model should be .mlx, got \(model)")
            return
        }
        #expect(model.name == "LFM2.5-350M-MLX-4bit")
        #expect(model.familyId == "lfm2.5-350m")

        // Install drops a self-describing manifest at the entry root.
        let entry = try #require(storage.modelStore.entryDir(for: spec))
        let manifest = try #require(ModelManifest.forInstalledEntry(atDir: entry))
        #expect(manifest.declared == spec)
    }

    @Test func installsSubpathModelInItsOwnEntry() throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }
        let ref = try HFModelRef.parse(repo: "org/multi-mlx", subpath: "variant-4bit")
        let src = try makeModelDir(files: ["config.json", "model.safetensors", "tokenizer_config.json"])

        try MLXModelInstaller(ref: ref, storage: storage).install(from: src)

        let model = try #require(storage.availableModels().first { m in
            guard case .mlx = m.source else { return false }
            return m.hfRepo == ref.repo.description
        })
        #expect(model.name == "variant-4bit")
    }

    @Test func overwritesAnExistingInstall() throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }
        let ref = try HFModelRef.parse(repo: "org/repo")
        try MLXModelInstaller(ref: ref, storage: storage).install(
            from: makeModelDir(files: ["config.json", "a.safetensors", "tokenizer.json"]))
        // Second install with a different weight file replaces the first cleanly.
        try MLXModelInstaller(ref: ref, storage: storage).install(
            from: makeModelDir(files: ["config.json", "b.safetensors", "tokenizer.json"]))
        let dest = try #require(storage.modelStore.blobsDir(for: .mlx(ref.asMlx())))
        #expect(FileManager.default.fileExists(atPath: dest.appendingPathComponent("b.safetensors").path))
        #expect(!FileManager.default.fileExists(atPath: dest.appendingPathComponent("a.safetensors").path))
    }

    /// The core regression guard: an incomplete download is rejected, the bad
    /// download is dropped, and nothing is recorded/installed.
    @Test func rejectsIncompleteDownloadAndRecordsNothing() throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }
        let ref = try HFModelRef.parse(repo: "org/empty")
        let src = try makeModelDir(files: [])  // empty — the poisoned-cache symptom

        #expect(throws: DownloadError.incompleteModel(ref, missing: ["config.json", "*.safetensors", "tokenizer"])) {
            try MLXModelInstaller(ref: ref, storage: storage).install(from: src)
        }
        #expect(!FileManager.default.fileExists(atPath: src.path), "bad download should be dropped")
        let blobs = try #require(storage.modelStore.blobsDir(for: .mlx(ref.asMlx())))
        #expect(!FileManager.default.fileExists(atPath: blobs.path))
        #expect(storage.availableModels().first { $0.hfRepo == ref.repo.description } == nil)
    }

    // MARK: - Helpers

    private func makeModelDir(files: [String]) throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("mlx-src-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        for f in files { try Data("x".utf8).write(to: dir.appendingPathComponent(f)) }
        return dir
    }
}
