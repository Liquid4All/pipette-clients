import Foundation
import Testing

@testable import Pipette

/// The install gate (`MlxModelLayout`) and installer (`DirectoryModelInstaller`): a model is
/// moved into its store entry and recorded only if it is a complete MLX directory.
@MainActor struct DirectoryModelInstallerTests {

    // MARK: - MlxModelLayout gate

    @Test func layoutAcceptsCompleteModel() throws {
        let dir = try makeModelDir(files: ["config.json", "model.safetensors", "tokenizer.json"])
        #expect(MlxModelLayout.missing(in: dir).isEmpty)
    }

    @Test func layoutReportsEachMissingPiece() throws {
        #expect(MlxModelLayout.missing(in: try makeModelDir(files: [])).sorted()
            == ["*.safetensors", "config.json", "tokenizer"].sorted())
        #expect(MlxModelLayout.missing(in: try makeModelDir(files: ["config.json", "tokenizer.json"]))
            == ["*.safetensors"])
        #expect(MlxModelLayout.missing(in: try makeModelDir(files: ["model.safetensors", "tokenizer.model"]))
            == ["config.json"])
        #expect(MlxModelLayout.missing(in: try makeModelDir(files: ["config.json", "model.safetensors"]))
            == ["tokenizer"])
    }

    // MARK: - DirectoryModelInstaller

    @Test func installsRootModelAndDiscoversIt() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let ref = try HfModelRef.parse(repo: "LiquidAI/LFM2.5-350M-MLX-4bit")
        let src = try makeModelDir(files: ["config.json", "model.safetensors", "tokenizer.json"])

        try DirectoryModelInstaller(source: .mlx(ref.asMlx()), ws: ws).install(from: src)

        let spec = Model.mlx(ref.asMlx())
        let dest = try #require(ws.models.blobsDir(for: spec))
        #expect(FileManager.default.fileExists(atPath: dest.appendingPathComponent("config.json").path))
        let model = try #require(ws.availableModels().first { $0.hfRepo == ref.repo.description })
        guard case .mlx = model.source else {
            Issue.record("installed model should be .mlx, got \(model)")
            return
        }
        #expect(model.name == "LFM2.5-350M-MLX-4bit")
        #expect(model.familyId == "lfm2.5-350m")

        // Install drops a self-describing manifest at the entry root.
        #expect(try ws.models.find(spec)?.declared == spec)
    }

    /// A repo subtree keeps its prefix under `blobs/`, where the crate's `to_stored` puts
    /// it, so the manifest's `stored` and the bytes agree.
    @Test func installsSubpathModelUnderItsPrefix() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let ref = try HfModelRef.parse(repo: "org/multi-mlx", subpath: "variant-4bit")
        let src = try makeModelDir(files: ["config.json", "model.safetensors", "tokenizer_config.json"])

        try DirectoryModelInstaller(source: .mlx(ref.asMlx()), ws: ws).install(from: src)

        let blobs = try #require(ws.models.blobsDir(for: .mlx(ref.asMlx())))
        #expect(FileManager.default.fileExists(
            atPath: blobs.appendingPathComponent("variant-4bit/config.json").path))
        let model = try #require(ws.availableModels().first { m in
            guard case .mlx = m.source else { return false }
            return m.hfRepo == ref.repo.description
        })
        #expect(model.name == "variant-4bit")
        #expect(model.path == blobs.appendingPathComponent("variant-4bit", isDirectory: true).path)
    }

    @Test func overwritesAnExistingInstall() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let ref = try HfModelRef.parse(repo: "org/repo")
        try DirectoryModelInstaller(source: .mlx(ref.asMlx()), ws: ws).install(
            from: makeModelDir(files: ["config.json", "a.safetensors", "tokenizer.json"]))
        try DirectoryModelInstaller(source: .mlx(ref.asMlx()), ws: ws).install(
            from: makeModelDir(files: ["config.json", "b.safetensors", "tokenizer.json"]))
        let dest = try #require(ws.models.blobsDir(for: .mlx(ref.asMlx())))
        #expect(FileManager.default.fileExists(atPath: dest.appendingPathComponent("b.safetensors").path))
        #expect(!FileManager.default.fileExists(atPath: dest.appendingPathComponent("a.safetensors").path))
    }

    /// The bad download is dropped too.
    @Test func rejectsIncompleteDownloadAndRecordsNothing() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let ref = try HfModelRef.parse(repo: "org/empty")
        let src = try makeModelDir(files: [])  // empty: the poisoned-cache symptom

        #expect(throws: DownloadError.incompleteModel(
            Model.mlx(ref.asMlx()).description, missing: ["config.json", "*.safetensors", "tokenizer"])) {
            try DirectoryModelInstaller(source: .mlx(ref.asMlx()), ws: ws).install(from: src)
        }
        #expect(!FileManager.default.fileExists(atPath: src.path), "bad download should be dropped")
        let blobs = try #require(ws.models.blobsDir(for: .mlx(ref.asMlx())))
        #expect(!FileManager.default.fileExists(atPath: blobs.path))
        #expect(ws.availableModels().first { $0.hfRepo == ref.repo.description } == nil)
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
