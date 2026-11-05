import Foundation
import Testing
@testable import Pipette

/// The root-independent `LocalStorage` namespace: the per-transfer download key, the
/// metadata parsers every label and pill is derived from, and stale-path recovery
/// against the entry layout. None of it needs a data root, so no test builds a store.
struct LocalStorageTests {
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

    // MARK: - The metadata parsers

    @Test func quantParsingSupportsCommonGgufSuffixes() {
        #expect(LocalStorage.parseQuant(from: "LFM2.5-350M-Q4_0.gguf") == "Q4_0")
        #expect(LocalStorage.parseQuant(from: "Qwen-Test-Q4_K_M.GGUF") == "Q4_K_M")
        #expect(LocalStorage.parseQuant(from: "Model-BF16.gguf") == "BF16")
        #expect(LocalStorage.parseQuant(from: "Model-IQ3_XXS.gguf") == "IQ3_XXS")
        #expect(LocalStorage.parseQuant(from: "Model-NoQuant.gguf") == nil)
    }

    @Test func modelStemStripsGgufAndTrailingQuantOnly() {
        #expect(LocalStorage.modelStem(from: "LFM2.5-350M-Q4_0.gguf") == "LFM2.5-350M")
        #expect(LocalStorage.modelStem(from: "Model-F16.gguf") == "Model")
        #expect(LocalStorage.modelStem(from: "Model-Preview.gguf") == "Model-Preview")
    }

    @Test func paramSizePrefersRepoThenFilenameAndRejectsVersionTokens() {
        #expect(
            LocalStorage.parseParamSize(name: "Fallback-350M-Q4_0.gguf", hfRepo: "liquid/LFM2.5-1.5B-GGUF") ==
            "1.5B"
        )
        #expect(
            LocalStorage.parseParamSize(name: "Gemma-3n-350M-Q4_0.gguf", hfRepo: nil) ==
            "350M"
        )
        #expect(LocalStorage.parseParamSize(name: "LFM2.5-Q4_0.gguf", hfRepo: nil) == nil)
    }

    @Test func normalizedModelStemPairsMmprojAndBaseAcrossQuantVariants() {
        #expect(
            LocalStorage.normalizedModelStem("mmproj-Gemma-3n-Q4_K_M.gguf") ==
            "gemma-3n"
        )
        #expect(
            LocalStorage.normalizedModelStem("Gemma-3n-Q4_0.gguf") ==
            "gemma-3n"
        )
        #expect(
            LocalStorage.normalizedModelStem("Gemma-3n.gguf") ==
            "gemma-3n"
        )
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
