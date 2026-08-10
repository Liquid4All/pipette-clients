import Foundation
import Testing

@testable import Pipette

/// The launch sweep that clears jobs whose manifest no longer decodes.
///
/// The case that motivated it: manifests written before the model-type rename carry
/// `hf_gguf_text` / `hf_mlx`, which this build has no case for, so the job lists nowhere,
/// runs nowhere, and logs a decode error on every sweep.
///
/// Each test injects its own temporary `FileStorage`, so the suite carries no shared global
/// and runs in parallel.
struct UndecodableJobSweepTests {

    @Test func anUndecodableJobIsDiscardedWithItsResults() throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }
        let cellId: CellId = "cell-legacy"
        try writePayload(storage: storage, cellId: cellId, benchmarkId: "prefill_throughput_512")
        try writeLegacyManifest(storage: storage, jobId: "job-legacy", cellId: cellId)

        // The precondition the sweep exists for: the job is already unreadable.
        #expect(storage.loadJobManifest(jobId: "job-legacy") == nil)

        #expect(storage.discardUndecodableJobs() == ["job-legacy"])

        #expect(!FileManager.default.fileExists(atPath: storage.jobDir(jobId: "job-legacy").path))
        // The results too: they live under `results/`, keyed by cell, so removing the job
        // tree alone would leave them counted against the quota and visible to nothing.
        #expect(storage.results.location(of: cellId) == nil)
    }

    /// The sweep must not touch a job it can still read — that would be data loss, not
    /// cleanup.
    @Test func adecodableJobSurvivesTheSweep() throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }
        let cellId: CellId = "cell-ok"
        try writePayload(storage: storage, cellId: cellId, benchmarkId: "prefill_throughput_512")
        storage.saveJobManifest(manifest(jobId: "job-ok", cellId: cellId))

        #expect(storage.discardUndecodableJobs().isEmpty)

        #expect(storage.loadJobManifest(jobId: "job-ok") != nil)
        #expect(storage.results.location(of: cellId) != nil)
    }

    /// A job directory with no manifest at all is not a decode failure. Deleting it here
    /// would race a job that is mid-creation, whose manifest is written after the directory.
    @Test func aJobDirectoryWithoutAManifestIsLeftAlone() throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }
        let dir = FileStorage.ensureDirectory(storage.jobDir(jobId: "job-empty"))

        #expect(storage.discardUndecodableJobs().isEmpty)

        #expect(FileManager.default.fileExists(atPath: dir.path))
    }

    /// Idempotent: the second launch has nothing left to discard.
    @Test func theSweepIsIdempotent() throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }
        try writeLegacyManifest(storage: storage, jobId: "job-legacy", cellId: "cell-legacy")

        #expect(storage.discardUndecodableJobs() == ["job-legacy"])
        #expect(storage.discardUndecodableJobs().isEmpty)
    }

    /// Only the unreadable job goes. A mixed directory is the real upgrade case — a device
    /// accumulates jobs across builds.
    @Test func theSweepDiscardsOnlyTheUnreadableJob() throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }
        storage.saveJobManifest(manifest(jobId: "job-ok", cellId: "cell-ok"))
        try writeLegacyManifest(storage: storage, jobId: "job-legacy", cellId: "cell-legacy")

        #expect(storage.discardUndecodableJobs() == ["job-legacy"])

        #expect(storage.loadAllJobManifests().map(\.jobId) == ["job-ok"])
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
        storage: FileStorage, jobId: JobId, cellId: CellId
    ) throws {
        storage.saveJobManifest(manifest(jobId: jobId, cellId: cellId))
        let path = storage.jobDir(jobId: jobId).appendingPathComponent("manifest.json")
        let legacy = try String(contentsOf: path, encoding: .utf8)
            .replacingOccurrences(of: "\"gguf_text\"", with: "\"hf_gguf_text\"")
        try legacy.write(to: path, atomically: true, encoding: .utf8)
    }
}
