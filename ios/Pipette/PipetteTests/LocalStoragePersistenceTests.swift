import Foundation
import Testing
@testable import Pipette

/// Each test injects its own temporary `FileStorage`, so the suite carries no
/// shared global and runs in parallel.
@MainActor struct LocalStoragePersistenceTests {
    @Test func savedJobManifestLoadsFromJobFiles() throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }

        let manifest = makeManifest(jobId: "job-swiftdata")

        storage.saveJobManifest(manifest)

        let loaded = storage.loadJobManifest(jobId: manifest.jobId)
        #expect(loaded?.jobId == manifest.jobId)
        #expect(loaded?.cells.map(\.cellId) == ["cell-a", "cell-b"])
        #expect(loaded?.cells[1].serverJobId == "server-b")
    }

    @Test func savedRegistrationLoadsFromJson() throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }

        let registration = makeRegistration(clientId: "client-swiftdata")

        try storage.identity.putRegistration(registration)

        let loaded = storage.identity.getRegistration()
        #expect(loaded?.clientId.value == "client-swiftdata")
        #expect(storage.identity.isRegistered)
    }

    /// Models are self-describing: the entry manifest is the sole provenance record,
    /// so the disk scan recognizes a model and derives all its metadata from the
    /// manifest's spec — no separate persisted copy. Removing the payload removes it.
    @Test func availableModelsDiscoversManifestBackedModelAndDerivesMetadata() throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }

        let repo = "LiquidAI/LFM2.5-350M-GGUF"
        let filename = "LFM2.5-350M-Q4_0.gguf"
        let entry = try installEntry(storage, try ggufTextSpec(repo, filename))

        // `availableModels()` also injects the built-in AFM model when the device has
        // Apple Intelligence, so assert over the file-backed models only.
        func fileBacked() -> [DiscoveredModel] {
            storage.availableModels().filter { if case .appleFoundationText = $0.source { false } else { true } }
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
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }

        let repo = "LiquidAI/LFM2.5-350M-GGUF"
        let filename = "LFM2.5-350M-Q4_0.gguf"
        try installEntry(storage, try ggufTextSpec(repo, filename))

        #expect(storage.availableModels().first?.familyId == "lfm2.5-350m")

        // A second scan re-derives the same metadata from the on-disk manifest.
        let models = storage.availableModels()
        #expect(models.first?.hfRepo == repo)
        #expect(models.first?.displayName == "LFM 2.5 350M")
        #expect(models.first?.familyId == "lfm2.5-350m")
    }

    private func makeManifest(jobId: JobId) -> JobManifest {
        JobManifest(
            jobId: jobId,
            createdAt: "2026-06-05T18:00:00Z",
            nGpuLayers: 99,
            contextSize: 4096,
            cells: [
                JobCell(
                    cellId: "cell-a",
                    benchmarkId: "prefill",
                    benchmarkType: .prefillThroughput,
                    runStatus: .completed,
                    serverJobId: nil,
                    errorMessage: nil,
                    source: ggufTextFixture("test/A-GGUF", "a.gguf")
                ),
                JobCell(
                    cellId: "cell-b",
                    benchmarkId: "decode",
                    benchmarkType: .decodeThroughput,
                    runStatus: .completed,
                    serverJobId: "server-b",
                    errorMessage: nil,
                    source: ggufTextFixture("test/B-GGUF", "b.gguf")
                )
            ],
            status: .completed,
            contributeResults: true,
            title: "Persistence job"
        )
    }

    private func makeRegistration(clientId: String) -> IdentityRegistration {
        IdentityRegistration(
            clientId: ClientID(clientId),
            status: "registered",
            serverUrl: ServerURL("https://collector.example.com"),
            organization: "Liquid",
            contactEmail: "bench@example.com",
            registeredAt: "2026-06-05T18:00:00Z",
            clerkUserId: "user-1",
            clerkSessionId: "session-1",
            clerkPrimaryEmail: "bench@example.com",
            clerkLinkedAt: "2026-06-05T18:01:00Z"
        )
    }

    /// Deleting a job deletes the results its cells produced.
    ///
    /// They no longer live under `jobs/<jobId>/` — they sit in `results/<location>/<cellId>/`,
    /// as the crate files them — so removing the job tree alone would leave them
    /// unreferenced: invisible to every listing, and still counted against the storage
    /// quota, with nothing to report the leak.
    @Test func deletingAJobDeletesItsResults() throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }
        let manifest = makeManifest(jobId: "job-delete")
        storage.saveJobManifest(manifest)
        for cell in manifest.cells {
            try storage.results.saveResult(
                .remotePending, cell.cellId,
                payload: Data("{}".utf8), extras: Data("{}".utf8))
        }
        #expect(manifest.cells.allSatisfy { storage.results.state(of: $0.cellId) != nil })

        storage.deleteJob(jobId: "job-delete")

        #expect(storage.loadJobManifest(jobId: "job-delete") == nil)
        for cell in manifest.cells {
            #expect(storage.results.state(of: cell.cellId) == nil, "\(cell.cellId.value) leaked")
        }
    }

    /// A local result goes with its job too — the cascade walks every cell, not only the
    /// ones that were eligible to be submitted.
    @Test func deletingAJobDeletesLocalResultsAsWell() throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }
        let manifest = makeManifest(jobId: "job-local")
        storage.saveJobManifest(manifest)
        let cellId = manifest.cells[0].cellId
        try storage.results.saveResult(
            .local, cellId, payload: Data("{}".utf8), extras: Data("{}".utf8))

        storage.deleteJob(jobId: "job-local")

        #expect(storage.results.state(of: cellId) == nil)
    }
}
