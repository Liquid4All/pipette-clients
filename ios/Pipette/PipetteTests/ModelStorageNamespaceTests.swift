import Foundation
import Testing
@testable import Pipette

/// Tests for the download-key helper, the model store's roots, and stale-path
/// recovery against the entry layout.
///
/// Each test builds its own `FileStorage` (or uses the pure `LocalStorage`
/// helpers, which need no root), so the suite carries no shared global and runs
/// in parallel.
@MainActor struct ModelStorageNamespaceTests {
    private let tmp: URL

    init() throws {
        tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("pipette-models-test-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: tmp, withIntermediateDirectories: true)
    }

    /// Per-test teardown: Swift Testing makes a fresh instance per test but
    /// structs have no `deinit`, so each test invokes this from a `defer`.
    private func cleanup() {
        try? FileManager.default.removeItem(at: tmp)
    }

    /// A `FileStorage` rooted under this test's `tmp`.
    private func storage() -> FileStorage {
        FileStorage(
            dataRoot: tmp.appendingPathComponent("data", isDirectory: true),
            cacheRoot: tmp.appendingPathComponent("cache", isDirectory: true)
        )
    }

    // MARK: - Path helpers

    @Test func cleanupLegacyEdgeEvalsStorageRemovesOldRootsButKeepsPipetteRoot() throws {
        defer { cleanup() }

        let appSupportRoot = tmp.appendingPathComponent("ApplicationSupport", isDirectory: true)
        let cacheRoot = tmp.appendingPathComponent("Caches", isDirectory: true)
        let pipetteRoot = appSupportRoot.appendingPathComponent("Pipette", isDirectory: true)
        let legacyDataRoot = appSupportRoot.appendingPathComponent("EdgeEvals", isDirectory: true)
        let legacyCacheRoot = cacheRoot.appendingPathComponent("EdgeEvals", isDirectory: true)

        try FileManager.default.createDirectory(at: pipetteRoot, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: legacyDataRoot, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: legacyCacheRoot, withIntermediateDirectories: true)
        try Data("new".utf8).write(to: pipetteRoot.appendingPathComponent("keep.txt"))
        try Data("old".utf8).write(to: legacyDataRoot.appendingPathComponent("remove.txt"))
        try Data("old-cache".utf8).write(to: legacyCacheRoot.appendingPathComponent("remove.txt"))

        let storage = FileStorage(dataRoot: pipetteRoot, cacheRoot: cacheRoot)
        let removed = storage.cleanupLegacyEdgeEvalsStorage().map(\.standardizedFileURL.path)

        #expect(FileManager.default.fileExists(atPath: pipetteRoot.path))
        #expect(!(FileManager.default.fileExists(atPath: legacyDataRoot.path)))
        #expect(!(FileManager.default.fileExists(atPath: legacyCacheRoot.path)))
        #expect(Set(removed) == Set([
            legacyDataRoot.standardizedFileURL.path,
            legacyCacheRoot.standardizedFileURL.path
        ]))
    }

    /// The per-transfer download key, not a storage path: it namespaces by repo so two
    /// repos publishing the same GGUF filename get distinct rows.
    @Test func modelRelativePathNamespacesByRepoButNotForSideloads() {
        #expect(
            LocalStorage.modelRelativePath(repo: "mistralai/Ministral-3-3B-Instruct-2512-GGUF",
                                           filename: "Ministral-3-3B-Instruct-2512-Q5_K_M.gguf") ==
            "mistralai/Ministral-3-3B-Instruct-2512-GGUF/Ministral-3-3B-Instruct-2512-Q5_K_M.gguf"
        )
        #expect(
            LocalStorage.modelRelativePath(repo: "unsloth/Ministral-3-3B-Instruct-2512-GGUF",
                                           filename: "Ministral-3-3B-Instruct-2512-Q5_K_M.gguf") !=
            LocalStorage.modelRelativePath(repo: "mistralai/Ministral-3-3B-Instruct-2512-GGUF",
                                           filename: "Ministral-3-3B-Instruct-2512-Q5_K_M.gguf")
        )
        #expect(LocalStorage.modelRelativePath(repo: nil, filename: "x.gguf") == "x.gguf")
        #expect(LocalStorage.modelRelativePath(repo: "", filename: "x.gguf") == "x.gguf")
    }

    @Test func modelsDirIsExcludedFromBackup() throws {
        defer { cleanup() }

        let values = try storage().modelsDir.resourceValues(forKeys: [.isExcludedFromBackupKey])

        #expect(values.isExcludedFromBackup == true)
    }

    @Test func modelEntryDirIsExcludedFromBackup() throws {
        defer { cleanup() }

        let spec = try ggufTextSpec("org/repo", "Model-Q4_0.gguf")
        let entry = try #require(storage().modelStore.prepareEntryDir(for: spec))
        let values = try entry.resourceValues(forKeys: [.isExcludedFromBackupKey])

        #expect(values.isExcludedFromBackup == true)
    }

    @Test func availableModelsDerivesMetadataForManifestBackedModel() throws {
        defer { cleanup() }

        let storage = storage()
        let repo = "LiquidAI/LFM2.5-350M-GGUF"
        let filename = "LFM2.5-350M-Q4_0.gguf"
        // A model is discovered only via its entry manifest; every display field is
        // then derived from the manifest's typed `source` — no separate stored copy.
        try installEntry(storage, try ggufTextSpec(repo, filename))

        let model = try #require(storage.availableModels().first { $0.name == filename })
        #expect(model.hfRepo == repo)
        #expect(model.displayName == "LFM 2.5 350M")
        #expect(model.familyId == "lfm2.5-350m")
    }

    // MARK: - resolveModelPath recovery

    @discardableResult
    private func writeModel(at relativePath: String) throws -> URL {
        let url = tmp.appendingPathComponent(relativePath)
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data("gguf".utf8).write(to: url)
        return url
    }

    @Test func resolveModelPathPrefersStoredEntryTail() throws {
        defer { cleanup() }

        let dest = try writeModel(at: "mistralai__Some-GGUF__Some-Q5_K_M.gguf/blobs/Some-Q5_K_M.gguf")
        let stale = "/old/container/models/mistralai__Some-GGUF__Some-Q5_K_M.gguf/blobs/Some-Q5_K_M.gguf"

        #expect(LocalStorage.resolveModelPath(stale, modelsRoot: tmp) == dest.path)
    }

    @Test func resolveModelPathRefusesAmbiguousByNameMatch() throws {
        defer { cleanup() }

        try writeModel(at: "unsloth__Some-GGUF__Some-Q5_K_M.gguf/blobs/Some-Q5_K_M.gguf")
        try writeModel(at: "mistralai__Some-GGUF__Some-Q5_K_M.gguf/blobs/Some-Q5_K_M.gguf")
        let stale = "/old/models/Some-Q5_K_M.gguf"

        #expect(LocalStorage.resolveModelPath(stale, modelsRoot: tmp) == nil,
                "an ambiguous by-name match must not guess a repo")
    }

    @Test func resolveModelPathFindsUniqueByName() throws {
        defer { cleanup() }

        let dest = try writeModel(at: "unsloth__Some-GGUF__Some-Q5_K_M.gguf/blobs/Some-Q5_K_M.gguf")
        let stale = "/old/models/Some-Q5_K_M.gguf"

        #expect(LocalStorage.resolveModelPath(stale, modelsRoot: tmp) == dest.path)
    }
}
