import Testing
import Foundation
@testable import Pipette

/// Background auto-pause: leaving the foreground trips the run's CancelFlag
/// with a `.background` reason, and the end-of-run bookkeeping records why
/// the job paused so the UI can explain a pause the user didn't ask for.
@MainActor
struct AutoPauseTests {
    @Test func cancelFlagRecordsReasonAndFirstCancelWins() {
        let flag = CancelFlag()
        #expect(flag.reason == nil)

        flag.cancel(reason: .background)
        #expect(flag.isCancelled)
        #expect(flag.reason == .background)

        // A follow-up user tap can't relabel the auto-pause.
        flag.cancel()
        #expect(flag.reason == .background)
    }

    @Test func cancelFlagDefaultsToUserReason() {
        let flag = CancelFlag()
        flag.cancel()
        #expect(flag.reason == .user)
    }

    @Test func jobRunnerForwardsCancelReasonToTheActiveFlag() {
        let runner = JobRunner()
        let flag = CancelFlag()
        #expect(runner.start(jobId: "job-1", flag: flag))

        runner.cancel(reason: .background)
        #expect(flag.reason == .background)
    }

    @Test func finalizeRunEndOnBackgroundCancelPausesWithReason() {
        var manifest = manifest(cells: [
            cell(id: "done", status: .completed),
            cell(id: "queued", status: .pending)
        ])

        manifest.finalizeRunEnd(cancelled: true, cancelReason: .background)

        #expect(manifest.status == .paused)
        #expect(manifest.cells.map(\.runStatus) == [.completed, .cancelled])
        #expect(manifest.pausedReason?.contains("Auto-paused") == true)
    }

    @Test func finalizeRunEndOnUserCancelPausesWithoutReason() {
        var manifest = manifest(cells: [cell(id: "queued", status: .pending)])

        manifest.finalizeRunEnd(cancelled: true, cancelReason: .user)

        #expect(manifest.status == .paused)
        #expect(manifest.pausedReason == nil)
    }

    @Test func finalizeRunEndCompletionClearsStaleReason() {
        var manifest = manifest(cells: [cell(id: "done", status: .completed)])
        manifest.pausedReason = "stale"

        manifest.finalizeRunEnd(cancelled: false, cancelReason: nil)

        #expect(manifest.status == .completed)
        #expect(manifest.pausedReason == nil)

        // Cancelled with nothing left to resume also completes, reason-free.
        var drained = self.manifest(cells: [cell(id: "done", status: .completed)])
        drained.pausedReason = "stale"
        drained.finalizeRunEnd(cancelled: true, cancelReason: .background)
        #expect(drained.status == .completed)
        #expect(drained.pausedReason == nil)
    }

    @Test func pausedReasonSurvivesRoundTripAndOldManifestsDecode() throws {
        var manifest = manifest(cells: [cell(id: "queued", status: .cancelled)])
        manifest.pausedReason = "Auto-paused"

        let data = try JSONEncoder().encode(manifest)
        let decoded = try JSONDecoder().decode(JobManifest.self, from: data)
        #expect(decoded.pausedReason == "Auto-paused")

        var json = try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any])
        json.removeValue(forKey: "pausedReason")
        let legacy = try JSONDecoder().decode(
            JobManifest.self,
            from: JSONSerialization.data(withJSONObject: json)
        )
        #expect(legacy.pausedReason == nil)
    }

    private func manifest(cells: [JobCell]) -> JobManifest {
        JobManifest(
            jobId: "job-1",
            createdAt: "2026-06-09T10:00:00Z",
            nGpuLayers: 99,
            contextSize: 4096,
            cells: cells,
            status: .running
        )
    }

    private func cell(id: CellId, status: CellRunStatus) -> JobCell {
        JobCell(
            cellId: id,
            benchmarkId: "decode",
            benchmarkType: .decodeThroughput,
            runStatus: status,
            serverJobId: nil,
            errorMessage: nil,
            source: ggufTextFixture("test/Qwen-GGUF", "Qwen-Q4_0.gguf")
        )
    }
}
