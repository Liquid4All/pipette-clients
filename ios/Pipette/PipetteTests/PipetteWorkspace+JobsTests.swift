import Foundation
import Testing

@testable import Pipette

/// The launch sweep that clears jobs whose manifest no longer decodes, such as one carrying
/// the pre-rename `hf_gguf_text` / `hf_mlx` model types: such a job lists nowhere, runs
/// nowhere, and logs a decode error on every sweep.
struct PipetteWorkspaceJobsTests {

    @Test func anUndecodableJobIsDiscardedWithItsResults() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let cellId: CellId = "cell-legacy"
        try writePayload(
            ws: ws, cellId: cellId, benchmarkId: "prefill_throughput_512", at: .remoteSynced)
        try writeLegacyManifest(ws: ws, jobId: "job-legacy", cellId: cellId)

        // The precondition the sweep exists for: the job is already unreadable.
        #expect(ws.loadJobManifest(jobId: "job-legacy") == nil)

        #expect(ws.discardUndecodableJobs() == ["job-legacy"])

        #expect(!FileManager.default.fileExists(atPath: ws.jobDir(jobId: "job-legacy").path))
        // The results too: they live under `results/`, keyed by cell, so removing the job
        // tree alone would leave them counted against the quota and visible to nothing.
        #expect(ws.results.location(of: cellId) == nil)
    }

    /// An older build reads a newer build's manifest as undecodable, and the newer build
    /// still submits the pending result.
    @Test func anUndecodableJobWithAPendingResultIsKept() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let cellId: CellId = "cell-newer"
        try writePayload(ws: ws, cellId: cellId, benchmarkId: "prefill_throughput_512")
        try writeLegacyManifest(ws: ws, jobId: "job-newer", cellId: cellId)
        #expect(ws.loadJobManifest(jobId: "job-newer") == nil)

        #expect(ws.discardUndecodableJobs().isEmpty)

        #expect(FileManager.default.fileExists(
            atPath: ws.jobDir(jobId: "job-newer").appendingPathComponent("manifest.json").path))
        #expect(ws.results.location(of: cellId) == .remotePending)
    }

    /// Touching a job it can still read would be data loss, not cleanup.
    @Test func adecodableJobSurvivesTheSweep() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let cellId: CellId = "cell-ok"
        try writePayload(ws: ws, cellId: cellId, benchmarkId: "prefill_throughput_512")
        ws.saveJobManifest(manifest(jobId: "job-ok", cellId: cellId))

        #expect(ws.discardUndecodableJobs().isEmpty)

        #expect(ws.loadJobManifest(jobId: "job-ok") != nil)
        #expect(ws.results.location(of: cellId) != nil)
    }

    /// A job directory with no manifest at all is not a decode failure. Deleting it here
    /// would race a job that is mid-creation, whose manifest is written after the directory.
    @Test func aJobDirectoryWithoutAManifestIsLeftAlone() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let dir = PipetteWorkspace.ensureDirectory(ws.jobDir(jobId: "job-empty"))

        #expect(ws.discardUndecodableJobs().isEmpty)

        #expect(FileManager.default.fileExists(atPath: dir.path))
    }

    @Test func theSweepIsIdempotent() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        try writeLegacyManifest(ws: ws, jobId: "job-legacy", cellId: "cell-legacy")

        #expect(ws.discardUndecodableJobs() == ["job-legacy"])
        #expect(ws.discardUndecodableJobs().isEmpty)
    }

    /// A mixed directory is the real upgrade case: a device accumulates jobs across builds.
    @Test func theSweepDiscardsOnlyTheUnreadableJob() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        ws.saveJobManifest(manifest(jobId: "job-ok", cellId: "cell-ok"))
        try writeLegacyManifest(ws: ws, jobId: "job-legacy", cellId: "cell-legacy")

        #expect(ws.discardUndecodableJobs() == ["job-legacy"])

        #expect(ws.loadAllJobManifests().map(\.jobId) == ["job-ok"])
    }

    /// A manifest the peek cannot read goes on to the full decode, which skips it as every
    /// loader does.
    @Test func loadJobManifestsDecodesOnlyWhatThePeekSelects() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let jobs: [(JobId, JobStatus, String)] = [
            ("job-done", .completed, "2026-10-01T10:00:00Z"),
            ("job-paused", .paused, "2026-10-02T10:00:00Z"),
            ("job-running", .running, "2026-10-03T10:00:00Z"),
        ]
        for (jobId, status, createdAt) in jobs {
            ws.saveJobManifest(JobManifest(
                jobId: jobId, createdAt: createdAt, nGpuLayers: 99, contextSize: 4096,
                cells: [cell(id: CellId("\(jobId.value)-cell"), benchmarkId: "prefill_throughput_512")],
                status: status))
        }
        let corrupt = PipetteWorkspace.ensureDirectory(ws.jobDir(jobId: "job-corrupt"))
        try Data("{".utf8).write(to: corrupt.appendingPathComponent("manifest.json"))

        var peeked: [JobId] = []
        let selected = ws.loadJobManifests { jobId, peek in
            peeked.append(jobId)
            return peek.status != .completed
        }

        #expect(selected.map(\.jobId) == ["job-running", "job-paused"])
        #expect(Set(peeked) == ["job-done", "job-paused", "job-running"])
        #expect(ws.loadAllJobManifests().count == 3)
    }

    @Test func filterJobManifestsKeepsOnlyWhatItSelectsNewestFirst() {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let jobs: [(JobId, JobStatus, String)] = [
            ("job-done", .completed, "2026-10-01T10:00:00Z"),
            ("job-running", .running, "2026-10-02T10:00:00Z"),
            ("job-paused", .paused, "2026-10-03T10:00:00Z"),
        ]
        for (jobId, status, createdAt) in jobs {
            ws.saveJobManifest(JobManifest(
                jobId: jobId, createdAt: createdAt, nGpuLayers: 99, contextSize: 4096,
                cells: [cell(id: CellId("\(jobId.value)-cell"), benchmarkId: "prefill_throughput_512")],
                status: status))
        }

        #expect(ws.filterJobManifests { $0.status != .running }.map(\.jobId) == ["job-paused", "job-done"])
    }

    // MARK: - Fixtures

    private func manifest(jobId: JobId, cellId: CellId) -> JobManifest {
        JobManifest(
            jobId: jobId,
            createdAt: "2026-06-09T10:00:00Z",
            nGpuLayers: 99,
            contextSize: 4096,
            cells: [cell(id: cellId, benchmarkId: "prefill_throughput_512")],
            status: .completed
        )
    }

    /// Write a real manifest, then rewrite its model type to the pre-rename spelling. Built
    /// from the encoder rather than a hand-authored blob so only the one field under test
    /// differs from what this build writes today.
    private func writeLegacyManifest(
        ws: PipetteWorkspace, jobId: JobId, cellId: CellId
    ) throws {
        ws.saveJobManifest(manifest(jobId: jobId, cellId: cellId))
        let path = ws.jobDir(jobId: jobId).appendingPathComponent("manifest.json")
        let legacy = try String(contentsOf: path, encoding: .utf8)
            .replacingOccurrences(of: "\"gguf_text\"", with: "\"hf_gguf_text\"")
        try legacy.write(to: path, atomically: true, encoding: .utf8)
    }

    // MARK: - Manifests and the delete cascade

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

    @Test func savedJobManifestLoadsFromJobFiles() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }

        let manifest = makeManifest(jobId: "job-swiftdata")

        ws.saveJobManifest(manifest)

        let loaded = ws.loadJobManifest(jobId: manifest.jobId)
        #expect(loaded?.jobId == manifest.jobId)
        #expect(loaded?.cells.map(\.cellId) == ["cell-a", "cell-b"])
        #expect(loaded?.cells[1].serverJobId == "server-b")
    }

    /// Results sit in `results/<location>/<cellId>/`, as the crate files them, so removing the
    /// job tree alone would leave them invisible to every listing and still counted against
    /// the storage quota.
    @Test func deletingAJobDeletesItsResults() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let manifest = makeManifest(jobId: "job-delete")
        ws.saveJobManifest(manifest)
        for cell in manifest.cells {
            try ws.results.saveResult(
                .remotePending, cell.cellId,
                payload: Data("{}".utf8), extras: Data("{}".utf8))
        }
        #expect(manifest.cells.allSatisfy { ws.results.state(of: $0.cellId) != nil })

        ws.deleteJob(jobId: "job-delete")

        #expect(ws.loadJobManifest(jobId: "job-delete") == nil)
        for cell in manifest.cells {
            #expect(ws.results.state(of: cell.cellId) == nil, "\(cell.cellId.value) leaked")
        }
    }

    /// The cascade walks every cell, not only the ones eligible to be submitted.
    @Test func deletingAJobDeletesLocalResultsAsWell() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let manifest = makeManifest(jobId: "job-local")
        ws.saveJobManifest(manifest)
        let cellId = manifest.cells[0].cellId
        try ws.results.saveResult(
            .local, cellId, payload: Data("{}".utf8), extras: Data("{}".utf8))

        ws.deleteJob(jobId: "job-local")

        #expect(ws.results.state(of: cellId) == nil)
    }

    /// No results left means the proxy deleted them or the run wrote none. A user's job and a
    /// claimed job still holding a result stay.
    @Test(arguments: [(nil, [], true), ("job-claimed", [], false), ("job-claimed", ["cell-b"], true)]
          as [(String?, [String], Bool)])
    func recoveryDropsAClaimedJobWithNoResultsLeft(
        claim: String?, seeded: [String], kept: Bool
    ) throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        var manifest = makeManifest(jobId: "job-1")
        manifest.claimJobId = claim
        ws.saveJobManifest(manifest)
        for cellId in seeded {
            try ws.results.saveResult(
                .local, CellId(cellId), payload: Data("{}".utf8), extras: Data("{}".utf8))
        }

        ws.recoverInterruptedJobs()

        #expect((ws.loadJobManifest(jobId: "job-1") != nil) == kept)
        #expect(seeded.allSatisfy { ws.results.location(of: CellId($0)) != nil })
    }
}
