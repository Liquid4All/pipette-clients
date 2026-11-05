import Foundation
import Testing

@testable import Pipette

/// `ResultsStore.unsubmittedResultCount`, the number quoted to the user in two places: the
/// job detail's "Submit N Results" button and the Settings sign-out warning, which says how
/// many results the reset is about to destroy permanently. A count that drifts high turns
/// that warning into a false alarm; one that drifts low deletes work the user was never told
/// about.
struct ResultsStoreTests {

    @Test func countsOnlyCellsWhoseResultCouldStillGoUp() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }

        // Already synced: the manifest carries a server job id.
        var synced = cell(id: "cell-1", benchmarkId: "bench-1")
        synced.serverJobId = "server-1"
        try writePayload(ws: ws, cellId: "cell-1", benchmarkId: "bench-1")

        // Genuinely pending: completed, no server id, payload waiting in remotePending.
        let pending = cell(id: "cell-2", benchmarkId: "bench-2")
        try writePayload(ws: ws, cellId: "cell-2", benchmarkId: "bench-2")

        // Filed under `local/`, so `submittableDir` returns nil. This is the result's
        // *location*, unrelated to the catalog half the next case covers.
        let notQueued = cell(id: "cell-3", benchmarkId: "bench-3")
        try writePayload(ws: ws, cellId: "cell-3", benchmarkId: "bench-3", at: .local)

        // Excluded only by `isSubmittable`: a local-catalog benchmark is never submitted. The
        // one case that pins that clause; every other cell here reads as `.remote`.
        var fromLocalCatalog = cell(id: "cell-4", benchmarkId: "bench-4")
        fromLocalCatalog.benchmarkSource = .local
        try writePayload(ws: ws, cellId: "cell-4", benchmarkId: "bench-4")

        // Completed with nothing on disk: there is no payload to lose.
        let payloadless = cell(id: "cell-5", benchmarkId: "bench-5")

        // Never ran, so nothing to submit even though a stray payload exists.
        var pendingButNotRun = cell(id: "cell-6", benchmarkId: "bench-6")
        pendingButNotRun.runStatus = .pending
        try writePayload(ws: ws, cellId: "cell-6", benchmarkId: "bench-6")

        let manifest = manifest(
            cells: [synced, pending, notQueued, fromLocalCatalog, payloadless, pendingButNotRun])

        #expect(ws.results.unsubmittedResultCount(manifest) == 1)
    }

    /// The deletion warning counts what the reset destroys, which is a strictly wider set:
    /// `resetDeviceData()` removes the whole `results/` tree, so a `local/` result the submit
    /// button rightly ignores is still work the user is about to lose.
    @Test func theDeletableCountIncludesResultsThatCouldNeverBeSubmitted() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }

        let localOnly = cell(id: "cell-1", benchmarkId: "bench-1")
        try writePayload(ws: ws, cellId: "cell-1", benchmarkId: "bench-1", at: .local)
        let manifest = manifest(cells: [localOnly])

        // Nothing to submit...
        #expect(ws.results.unsubmittedResultCount(manifest) == 0)
        // ...but the reset deletes it, so the warning must account for it.
        #expect(ws.results.deletableResultCount(manifest) == 1)
        #expect(ws.results.deletableResultCount(across: [manifest]) == 1)
    }

    /// An already-uploaded result is not "lost" by the reset in any sense the user cares
    /// about: the server has it.
    @Test func theDeletableCountSkipsResultsAlreadyUploaded() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }

        var synced = cell(id: "cell-1", benchmarkId: "bench-1")
        synced.serverJobId = "server-1"
        try writePayload(ws: ws, cellId: "cell-1", benchmarkId: "bench-1")

        #expect(ws.results.deletableResultCount(manifest(cells: [synced])) == 0)
    }

    @Test func theDeviceWideTotalSumsEveryJob() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }

        for id in ["cell-1", "cell-2", "cell-3"] {
            try writePayload(ws: ws, cellId: CellId(id), benchmarkId: "bench-\(id)")
        }
        let first = manifest(jobId: "job-1", cells: [
            cell(id: "cell-1", benchmarkId: "bench-cell-1"),
            cell(id: "cell-2", benchmarkId: "bench-cell-2")
        ])
        let second = manifest(jobId: "job-2", cells: [cell(id: "cell-3", benchmarkId: "bench-cell-3")])

        #expect(ws.results.deletableResultCount(across: [first, second]) == 3)
        // No jobs means no warning sentence, which is what the dialog keys off.
        #expect(ws.results.deletableResultCount(across: []) == 0)
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

    // MARK: - Scores

    /// The crate's `has_valid_metrics` and `corrupt_metrics_read_as_missing`: a later
    /// refresh can replace a corrupt score.
    @Test func onlyAParseableMetricsFileIsAValidScore() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        try writePayload(ws: ws, cellId: "cell-1", benchmarkId: "eval_smoke", at: .remoteSynced)
        #expect(!ws.results.hasValidMetrics(.remoteSynced, "cell-1"))

        let metrics = try #require(ws.results.metricsPath(.remoteSynced, "cell-1"))
        try Data("{not json".utf8).write(to: metrics)
        #expect(!ws.results.hasValidMetrics(.remoteSynced, "cell-1"))
        #expect(ws.results.state(of: "cell-1") == .submitted)

        try ws.results.saveMetrics(
            .remoteSynced, "cell-1",
            BenchmarkScoredResult(scoredAt: "2026-09-01T00:00:00Z",
                                  metrics: [BenchmarkJobMetric(metric: "accuracy", value: 1, unit: "ratio")]))
        #expect(ws.results.hasValidMetrics(.remoteSynced, "cell-1"))
        #expect(ws.results.state(of: "cell-1") == .scored)
    }

    // MARK: - Layout

    /// Ids are the result directories at a location, sorted; a stray file beside them is
    /// not a result, and a location never written lists empty.
    @Test func listIdsReturnsSortedDirectoriesOnly() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        #expect(ws.results.listIds(.local).isEmpty)

        for id in ["b", "a", "c"] {
            try writePayload(ws: ws, cellId: CellId(id), benchmarkId: "x", at: .local)
        }
        try Data("{}".utf8).write(
            to: ws.results.locationDir(.local).appendingPathComponent("stray.json"))

        #expect(ws.results.listIds(.local) == ["a", "b", "c"])
    }

    /// `saveResult` writes both halves, so a payload never lands without the extras that
    /// say what ran.
    @Test func saveResultWritesPayloadAndExtras() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }

        try ws.results.saveResult(
            .local, "cell-1",
            payload: Data(#"{"benchmark_id":"x"}"#.utf8),
            extras: Data(#"{"command":["llama-bench"]}"#.utf8))

        #expect(try ws.results.loadPayload(.local, "cell-1")["benchmark_id"] as? String == "x")
        let extras = try #require(ws.results.extrasPath(.local, "cell-1"))
        #expect(try Data(contentsOf: extras) == Data(#"{"command":["llama-bench"]}"#.utf8))
    }

    /// `metrics.json` is written as serde writes the crate's `BenchmarkScoredResult`: both
    /// fields present, `null` when absent.
    @Test func aSavedScoreHasTheCratesShape() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        try writePayload(ws: ws, cellId: "cell-1", benchmarkId: "eval_smoke", at: .remoteSynced)
        try ws.results.saveMetrics(
            .remoteSynced, "cell-1", BenchmarkScoredResult(scoredAt: nil, metrics: nil))

        let path = try #require(ws.results.metricsPath(.remoteSynced, "cell-1"))
        let object = try #require(JSONSerialization.jsonObject(with: Data(contentsOf: path))
            as? [String: Any])
        #expect(object.keys.sorted() == ["metrics", "scored_at"])
        #expect(object["scored_at"] is NSNull)
    }
}
