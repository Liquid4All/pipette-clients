import Foundation
import Testing

@testable import Pipette

/// `ResultsStore.unsubmittedResultCount`, the number quoted to the user in two places: the
/// job detail's "Submit N Results" button and the Settings sign-out warning, which says how
/// many results the reset is about to destroy permanently. A count that drifts high turns
/// that warning into a false alarm; one that drifts low deletes work the user was never told
/// about.
struct UnsubmittedResultCountTests {

    @Test func countsOnlyCellsWhoseResultCouldStillGoUp() throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }

        // Already synced: the manifest carries a server job id.
        var synced = cell(id: "cell-1", benchmarkId: "bench-1")
        synced.serverJobId = "server-1"
        try writePayload(storage: storage, cellId: "cell-1", benchmarkId: "bench-1")

        // Genuinely pending: completed, no server id, payload waiting in remotePending.
        let pending = cell(id: "cell-2", benchmarkId: "bench-2")
        try writePayload(storage: storage, cellId: "cell-2", benchmarkId: "bench-2")

        // Parked in the `local/` results directory rather than `remotePending`, so it is not queued
        // for upload and `submittableDir` returns nil for it. Note this is the result's *location*,
        // unrelated to the catalog half the next case covers.
        let notQueued = cell(id: "cell-3", benchmarkId: "bench-3")
        try writePayload(storage: storage, cellId: "cell-3", benchmarkId: "bench-3", at: .local)

        // Queued like `pending`, and excluded only by `isSubmittable`: a cell whose benchmark came
        // from the generated catalog half is never submitted. The one case that pins that clause,
        // since every other cell here inherits the fixture's nil `benchmarkSource` and reads
        // as `.remote`.
        var fromLocalCatalog = cell(id: "cell-4", benchmarkId: "bench-4")
        fromLocalCatalog.benchmarkSource = .local
        try writePayload(storage: storage, cellId: "cell-4", benchmarkId: "bench-4")

        // Completed with nothing on disk: there is no payload to lose.
        let payloadless = cell(id: "cell-5", benchmarkId: "bench-5")

        // Never ran, so nothing to submit even though a stray payload exists.
        var pendingButNotRun = cell(id: "cell-6", benchmarkId: "bench-6")
        pendingButNotRun.runStatus = .pending
        try writePayload(storage: storage, cellId: "cell-6", benchmarkId: "bench-6")

        let manifest = manifest(
            cells: [synced, pending, notQueued, fromLocalCatalog, payloadless, pendingButNotRun])

        #expect(storage.results.unsubmittedResultCount(manifest) == 1)
    }

    /// The deletion warning counts what the reset destroys, which is a strictly wider set:
    /// `resetDeviceData()` removes the whole `results/` tree, so a `local/` result the submit
    /// button rightly ignores is still work the user is about to lose.
    @Test func theDeletableCountIncludesResultsThatCouldNeverBeSubmitted() throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }

        let localOnly = cell(id: "cell-1", benchmarkId: "bench-1")
        try writePayload(storage: storage, cellId: "cell-1", benchmarkId: "bench-1", at: .local)
        let manifest = manifest(cells: [localOnly])

        // Nothing to submit...
        #expect(storage.results.unsubmittedResultCount(manifest) == 0)
        // ...but the reset deletes it, so the warning must account for it.
        #expect(storage.results.deletableResultCount(manifest) == 1)
        #expect(storage.results.deletableResultCount(across: [manifest]) == 1)
    }

    /// An already-uploaded result is not "lost" by the reset in any sense the user cares
    /// about: the server has it.
    @Test func theDeletableCountSkipsResultsAlreadyUploaded() throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }

        var synced = cell(id: "cell-1", benchmarkId: "bench-1")
        synced.serverJobId = "server-1"
        try writePayload(storage: storage, cellId: "cell-1", benchmarkId: "bench-1")

        #expect(storage.results.deletableResultCount(manifest(cells: [synced])) == 0)
    }

    @Test func theDeviceWideTotalSumsEveryJob() throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }

        for id in ["cell-1", "cell-2", "cell-3"] {
            try writePayload(storage: storage, cellId: CellId(id), benchmarkId: "bench-\(id)")
        }
        let first = manifest(jobId: "job-1", cells: [
            cell(id: "cell-1", benchmarkId: "bench-cell-1"),
            cell(id: "cell-2", benchmarkId: "bench-cell-2")
        ])
        let second = manifest(jobId: "job-2", cells: [cell(id: "cell-3", benchmarkId: "bench-cell-3")])

        #expect(storage.results.deletableResultCount(across: [first, second]) == 3)
        // No jobs means no warning sentence, which is what the dialog keys off.
        #expect(storage.results.deletableResultCount(across: []) == 0)
    }

    private func manifest(jobId: JobId = "job-1", cells: [JobCell]) -> JobManifest {
        JobManifest(
            jobId: jobId,
            createdAt: "2026-06-10T10:00:00Z",
            nGpuLayers: 99,
            contextSize: 4096,
            cells: cells,
            status: .completed,
            contributeResults: true
        )
    }
}
