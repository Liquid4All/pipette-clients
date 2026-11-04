import Foundation
import Testing

@testable import Pipette

/// `JobStore` reads disk on first use, keeps a summary per job for the list, and holds only the
/// last few full manifests job pages read.
struct JobStoreTests {

    /// The farm launch: throwaway stores that save and apply but never show a job screen.
    @Test func theListIsReadOnFirstUseNotBefore() {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let existing = manifest("job-1", createdAt: "2026-10-01T10:00:00Z")
        ws.saveJobManifest(existing)
        let store = JobStore(ws: ws)

        store.save(manifest("job-2", createdAt: "2026-10-02T10:00:00Z"))
        store.apply(existing)
        store.reload()
        // Written behind the store: a store that had read the list would not see it.
        ws.saveJobManifest(manifest("job-3", createdAt: "2026-10-03T10:00:00Z"))

        #expect(store.jobs.map(\.jobId) == ["job-3", "job-2", "job-1"])
    }

    @Test func applyReplacesOrInsertsNewestFirst() {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        ws.saveJobManifest(manifest("t3", createdAt: "2026-10-03T10:00:00Z"))
        ws.saveJobManifest(manifest("t1", createdAt: "2026-10-01T10:00:00Z"))
        let store = JobStore(ws: ws)
        _ = store.jobs

        store.apply(manifest("t2", createdAt: "2026-10-02T10:00:00Z"))
        #expect(store.jobs.map(\.jobId) == ["t3", "t2", "t1"])

        store.apply(manifest("t1", createdAt: "2026-10-01T10:00:00Z", status: .paused))
        #expect(store.jobs.map(\.jobId) == ["t3", "t2", "t1"])
        #expect(store.jobs.last?.status == .paused)
    }

    @Test func aJobPageSeesAppliesAndDeletes() {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        var job = manifest("job-1", createdAt: "2026-10-01T10:00:00Z", status: .running)
        ws.saveJobManifest(job)
        let store = JobStore(ws: ws)
        #expect(store.job(id: "job-1")?.status == .running)

        job.status = .completed
        ws.saveJobManifest(job)
        store.apply(job)
        #expect(store.job(id: "job-1")?.status == .completed)

        _ = store.jobs
        store.delete(jobId: "job-1")
        #expect(store.job(id: "job-1") == nil)
        #expect(store.jobs.isEmpty)
    }

    @Test func onlyTheLastFewOpenJobsStayInMemory() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let ids = (0...JobStore.openJobLimit).map { JobId("job-\($0)") }
        for id in ids { ws.saveJobManifest(manifest(id, createdAt: "2026-10-01T10:00:00Z")) }
        let store = JobStore(ws: ws)
        for id in ids { _ = store.job(id: id) }

        // Written behind the store: only a job no longer in memory is read again.
        for id in ids { ws.saveJobManifest(manifest(id, createdAt: "2026-10-01T10:00:00Z", status: .paused)) }

        #expect(store.job(id: try #require(ids.last))?.status == .completed)
        #expect(store.job(id: try #require(ids.first))?.status == .paused)
    }

    @Test func reloadCatchesUpWithWritesBehindTheStore() {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        ws.saveJobManifest(manifest("job-1", createdAt: "2026-10-01T10:00:00Z"))
        let store = JobStore(ws: ws)
        _ = store.jobs
        _ = store.job(id: "job-1")

        ws.saveJobManifest(manifest("job-1", createdAt: "2026-10-01T10:00:00Z", status: .paused))
        ws.saveJobManifest(manifest("job-2", createdAt: "2026-10-02T10:00:00Z"))
        store.reload()

        #expect(store.jobs.map(\.jobId) == ["job-2", "job-1"])
        #expect(store.job(id: "job-1")?.status == .paused)
    }

    @Test func aSummaryCarriesWhatAJobRowShows() throws {
        var failed = cell(id: CellId("c2"), benchmarkId: "prefill_throughput_512", source: try mlxSpec("org/m-MLX-4bit"))
        failed.runStatus = .failed
        let job = JobManifest(
            jobId: "job-1",
            createdAt: "2026-10-01T10:00:00Z",
            nGpuLayers: 99,
            contextSize: 4096,
            cells: [cell(id: CellId("c1"), benchmarkId: "decode_throughput_512_100"), failed],
            status: .completed
        )

        let summary = JobSummary(job)

        #expect(summary.totalCells == 2)
        #expect(summary.completedCells == 1)
        #expect(summary.runtimeKinds == [.llamaCpp, .mlx])
        #expect(summary.modelNames == job.modelNames)
        #expect(summary.benchmarkIds == job.benchmarkIds)
        #expect(summary.displayTitle == job.displayTitle)
    }

    // MARK: - Fixtures

    private func manifest(
        _ jobId: JobId, createdAt: String, status: JobStatus = .completed
    ) -> JobManifest {
        JobManifest(
            jobId: jobId,
            createdAt: createdAt,
            nGpuLayers: 99,
            contextSize: 4096,
            cells: [cell(id: CellId("\(jobId.value)-cell"), benchmarkId: "prefill_throughput_512")],
            status: status
        )
    }
}
