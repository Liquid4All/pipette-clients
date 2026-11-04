import Foundation
import Testing
@testable import Pipette

/// `PipetteWorkspace`: its roots, legacy-root cleanup, backup exclusions, identity store
/// and model scan.
@MainActor struct PipetteWorkspaceTests {
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

    private func ws() -> PipetteWorkspace {
        PipetteWorkspace(
            root: tmp.appendingPathComponent("data", isDirectory: true),
            cacheRoot: tmp.appendingPathComponent("cache", isDirectory: true)
        )
    }

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

        let ws = PipetteWorkspace(root: pipetteRoot, cacheRoot: cacheRoot)
        let removed = ws.cleanupLegacyEdgeEvalsStorage().map(\.standardizedFileURL.path)

        #expect(FileManager.default.fileExists(atPath: pipetteRoot.path))
        #expect(!(FileManager.default.fileExists(atPath: legacyDataRoot.path)))
        #expect(!(FileManager.default.fileExists(atPath: legacyCacheRoot.path)))
        #expect(Set(removed) == Set([
            legacyDataRoot.standardizedFileURL.path,
            legacyCacheRoot.standardizedFileURL.path
        ]))
    }

    @Test func modelsDirIsExcludedFromBackup() throws {
        defer { cleanup() }

        let values = try ws().modelsDir.resourceValues(forKeys: [.isExcludedFromBackupKey])

        #expect(values.isExcludedFromBackup == true)
    }

    @Test func modelEntryDirIsExcludedFromBackup() throws {
        defer { cleanup() }

        let spec = try ggufTextSpec("org/repo", "Model-Q4_0.gguf")
        let entry = try #require(ws().models.prepareEntryDir(for: spec))
        let values = try entry.resourceValues(forKeys: [.isExcludedFromBackupKey])

        #expect(values.isExcludedFromBackup == true)
    }

    @Test func availableModelsDerivesMetadataForManifestBackedModel() throws {
        defer { cleanup() }

        let ws = ws()
        let repo = "LiquidAI/LFM2.5-350M-GGUF"
        let filename = "LFM2.5-350M-Q4_0.gguf"
        try installEntry(ws, try ggufTextSpec(repo, filename))

        let model = try #require(ws.availableModels().first { $0.name == filename })
        #expect(model.hfRepo == repo)
        #expect(model.displayName == "LFM 2.5 350M")
        #expect(model.familyId == "lfm2.5-350m")
    }

    @Test func savedRegistrationLoadsFromJson() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }

        let registration = try makeRegistration(clientId: "client-swiftdata")

        try ws.registration.put(registration)

        let loaded = ws.registration.get()
        #expect(loaded?.clientId.value == "client-swiftdata")
        #expect(ws.registration.isRegistered)
    }

    /// The entry manifest is the only provenance record, and a model without its payload is
    /// not discovered.
    @Test func availableModelsDiscoversManifestBackedModelAndDerivesMetadata() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }

        let repo = "LiquidAI/LFM2.5-350M-GGUF"
        let filename = "LFM2.5-350M-Q4_0.gguf"
        let entry = try installEntry(ws, try ggufTextSpec(repo, filename))

        // `availableModels()` also injects the built-in AFM model when the device has
        // Apple Intelligence, so assert over the file-backed models only.
        func fileBacked() -> [DiscoveredModel] {
            ws.availableModels().filter { if case .appleFoundationText = $0.source { false } else { true } }
        }
        let models = fileBacked()
        #expect(models.count == 1)
        #expect(models[0].hfRepo == repo)
        #expect(models[0].displayName == "LFM 2.5 350M")
        #expect(models[0].familyId == "lfm2.5-350m")

        try FileManager.default.removeItem(at: entry.appendingPathComponent("blobs/\(filename)"))
        #expect(fileBacked().isEmpty)
    }

    @Test func availableModelsDerivesStableMetadataAcrossRescans() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }

        let repo = "LiquidAI/LFM2.5-350M-GGUF"
        let filename = "LFM2.5-350M-Q4_0.gguf"
        try installEntry(ws, try ggufTextSpec(repo, filename))

        #expect(ws.availableModels().first?.familyId == "lfm2.5-350m")

        let models = ws.availableModels()
        #expect(models.first?.hfRepo == repo)
        #expect(models.first?.displayName == "LFM 2.5 350M")
        #expect(models.first?.familyId == "lfm2.5-350m")
    }

    // MARK: - RunContext

    /// Refused before any request. The route is checked before the registration, so an
    /// unregistered device still reports a missing route.
    @Test(arguments: [
        (nil, "no download URL"),
        ("/datasets/vl_spec_decode_corpus_test_gqa_k8", "no registration"),
    ] as [(String?, String)])
    func anUnfetchableDatasetIsRefused(url: String?, expected: String) async {
        defer { cleanup() }
        let sample = BenchmarkDefinitionTests.sampleDataset
        let dataset = Dataset(
            format: sample.format, adapter: sample.adapter, digest: sample.digest,
            bytes: sample.bytes, upstream: sample.upstream,
            url: url.map(NonEmptyString.init(validated:)))

        let error = await #expect(throws: (any Error).self) {
            try await ws().fetchDataset(dataset)
        }
        #expect(error?.localizedDescription.contains(expected) == true,
                "\(String(describing: error))")
    }

    private func makeRegistration(clientId: String) throws -> IdentityRegistration {
        IdentityRegistration(
            clientId: try ClientId(clientId),
            status: "registered",
            serverUrl: try ServerUrl("https://collector.example.com"),
            organization: "Liquid",
            contactEmail: "bench@example.com",
            registeredAt: "2026-06-05T18:00:00Z",
            clerkUserId: "user-1",
            clerkSessionId: "session-1",
            clerkPrimaryEmail: "bench@example.com",
            clerkLinkedAt: "2026-06-05T18:01:00Z"
        )
    }
}
