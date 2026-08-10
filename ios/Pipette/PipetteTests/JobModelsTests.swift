import Testing
import Foundation
@testable import Pipette

@Suite @MainActor struct JobModelsTests {
    @Test func jobManifestDerivedCountsAndUniqueNames() {
        let manifest = JobManifest(
            jobId: "job-1",
            createdAt: "2026-05-28T16:41:00Z",
            nGpuLayers: 99,
            contextSize: 4096,
            cells: [
                cell(id: "a", benchmarkId: "decode", modelName: "Qwen", status: .completed, serverJobId: "server-a"),
                cell(id: "b", benchmarkId: "prefill", modelName: "Gemma", status: .failed),
                cell(id: "c", benchmarkId: "decode", modelName: "Qwen", status: .cancelled)
            ],
            status: .completed
        )

        #expect(manifest.totalCells == 3)
        #expect(manifest.completedCells == 1)
        #expect(manifest.failedCells == 1)
        #expect(manifest.cancelledCells == 1)
        #expect(manifest.submittedCells == 1)
        #expect(manifest.modelNames == ["test/Gemma-GGUF", "test/Qwen-GGUF"])
        #expect(manifest.benchmarkIds == ["decode", "prefill"])
    }

    @Test func recoverInterruptedRunningManifestMarksLiveAndPendingCellsResumable() {
        var manifest = JobManifest(
            jobId: "job-1",
            createdAt: "2026-05-28T16:41:00Z",
            nGpuLayers: 99,
            contextSize: 4096,
            cells: [
                cell(id: "done", benchmarkId: "decode", modelName: "Qwen", status: .completed),
                cell(id: "active", benchmarkId: "prefill", modelName: "Qwen", status: .running),
                cell(id: "queued", benchmarkId: "memory", modelName: "Qwen", status: .pending),
                cell(id: "failed", benchmarkId: "e2e", modelName: "Qwen", status: .failed)
            ],
            status: .running
        )

        // Pulled out of `#expect` — its expansion captures `manifest` immutably,
        // so the mutating call can't run inside the macro.
        let recovered = manifest.recoverInterruptedRunState()
        #expect(recovered)
        #expect(manifest.status == .paused)
        #expect(manifest.cells.map(\.runStatus) == [.completed, .cancelled, .cancelled, .failed])
        #expect(manifest.cancelledCells == 2)
    }

    @Test func recoverInterruptedRunningManifestCompletesWhenNoResumableWorkRemains() {
        var manifest = JobManifest(
            jobId: "job-1",
            createdAt: "2026-05-28T16:41:00Z",
            nGpuLayers: 99,
            contextSize: 4096,
            cells: [
                cell(id: "done", benchmarkId: "decode", modelName: "Qwen", status: .completed),
                cell(id: "failed", benchmarkId: "prefill", modelName: "Qwen", status: .failed)
            ],
            status: .running
        )

        let recovered = manifest.recoverInterruptedRunState()
        #expect(recovered)
        #expect(manifest.status == .completed)
        #expect(manifest.cells.map(\.runStatus) == [.completed, .failed])
    }

    @Test func recoverInterruptedRunStateLeavesTerminalManifestUnchanged() {
        var manifest = JobManifest(
            jobId: "job-1",
            createdAt: "2026-05-28T16:41:00Z",
            nGpuLayers: 99,
            contextSize: 4096,
            cells: [cell(id: "done", benchmarkId: "decode", modelName: "Qwen", status: .completed)],
            status: .completed
        )

        let recovered = manifest.recoverInterruptedRunState()
        #expect(!recovered)
        #expect(manifest.status == .completed)
        #expect(manifest.cells.map(\.runStatus) == [.completed])
    }

    @Test func displayTitleUsesExplicitNonBlankTitle() {
        let manifest = manifest(title: " Nightly run ")

        #expect(manifest.displayTitle == " Nightly run ")
    }

    @Test func displayTitleFallsBackToDateModelAndBenchmarkSummary() {
        let manifest = JobManifest(
            jobId: "job-1",
            createdAt: "2026-05-28T16:41:00Z",
            nGpuLayers: 99,
            contextSize: 4096,
            cells: [
                cell(id: "a", benchmarkId: "decode", modelName: "Qwen", status: .completed),
                cell(id: "b", benchmarkId: "prefill", modelName: "Gemma", status: .completed)
            ],
            status: .completed,
            title: "   "
        )

        #expect(manifest.displayTitle == "2026-05-28 · 2 models · 2 benchmarks")
    }

    @Test func displayTitleFallsBackToCreatedAtPrefixWhenDateIsInvalid() {
        let manifest = JobManifest(
            jobId: "job-1",
            createdAt: "not-a-date-value",
            nGpuLayers: 99,
            contextSize: 4096,
            cells: [cell(id: "a", benchmarkId: "decode", modelName: "Qwen", status: .completed)],
            status: .completed
        )

        #expect(manifest.displayTitle == "not-a-date · 1 model · 1 benchmark")
    }

    @Test func formatJobErrorAugmentsOutOfMemoryWithContextSizeAndDetail() {
        let message = formatJobError(RuntimeError.outOfMemory("allocation failed"), contextSize: 8192)
        #expect(message.contains("context size 8192"))
        #expect(message.contains("allocation failed"))
    }

    @Test func formatJobErrorPassesNonOutOfMemoryThroughAsLocalizedDescription() {
        struct PlainError: LocalizedError { var errorDescription: String? { "plain message" } }
        // Non-OOM errors surface their localizedDescription verbatim, without the
        // smaller-quant/context hint the OOM branch adds.
        #expect(formatJobError(PlainError(), contextSize: 2048) == "plain message")
    }

    @Test func jobRunnerResetsObservableRunStateOnStartAndFinish() {
        let runner = JobRunner()
        let flag = CancelFlag()

        runner.currentCellLabel = "old"
        runner.currentProgressText = "old progress"
        runner.currentCellFraction = 0.75
        #expect(runner.start(jobId: "job-123", flag: flag))

        #expect(runner.runningJobId == "job-123")
        #expect(runner.isRunning)
        #expect(runner.currentCellLabel == "")
        #expect(runner.currentProgressText == "")
        #expect(runner.currentCellFraction == 0)
        #expect(runner.startedAt != nil)

        runner.cancel()
        #expect(flag.isCancelled)

        runner.finish()
        #expect(runner.runningJobId == nil)
        #expect(runner.cancelFlag == nil)
        #expect(!runner.isRunning)
        #expect(runner.startedAt == nil)
    }

    @Test func jobRunnerRejectsConcurrentStarts() {
        let runner = JobRunner()
        let first = CancelFlag()
        let second = CancelFlag()

        #expect(runner.start(jobId: "job-1", flag: first))
        #expect(!runner.start(jobId: "job-2", flag: second))

        #expect(runner.runningJobId == "job-1")
        runner.cancel()
        #expect(first.isCancelled)
        #expect(!second.isCancelled)
    }

    @Test func jobRunnerFinishWithMismatchedJobDoesNotClearActiveRun() {
        let runner = JobRunner()
        #expect(runner.start(jobId: "job-1", flag: CancelFlag()))

        runner.finish(jobId: "job-2")
        #expect(runner.runningJobId == "job-1")

        runner.finish(jobId: "job-1")
        #expect(runner.runningJobId == nil)
    }

    @Test func anchorETAUsesRunRelativeProgress() {
        let runner = JobRunner()
        // Resumed job: 90 cells done before this run, 10 left to run.
        runner.start(jobId: "job-123", flag: CancelFlag(), completedAtStart: 90, totalToRun: 10)
        let started = runner.startedAt!

        // 1 of 10 this-run cells done after 60s → 9 more minutes, even though
        // the manifest is 91% complete overall. Anchoring off whole-job progress
        // here was the bug: it produced "1s left" on resume.
        runner.anchorETA(completedCells: 91, now: started.addingTimeInterval(60))
        #expect(runner.estimatedTimeLeft(jobId: "job-123", now: started.addingTimeInterval(60)) == "9 min left")

        // Within-cell fraction counts toward this-run progress: half of the
        // first cell after 30s → 5% of 10 cells → 570s left, rounds to 10 min.
        runner.currentCellFraction = 0.5
        runner.anchorETA(completedCells: 90, now: started.addingTimeInterval(30))
        #expect(runner.estimatedTimeLeft(jobId: "job-123", now: started.addingTimeInterval(30)) == "10 min left")
    }

    @Test func estimatedTimeLeftReturnsNilWithoutUsableRunState() {
        let runner = JobRunner()
        let now = Date()

        // No run in flight.
        #expect(runner.estimatedTimeLeft(jobId: "job-123", now: now) == nil)

        runner.start(jobId: "job-123", flag: CancelFlag(), completedAtStart: 0, totalToRun: 10)
        let later = runner.startedAt!.addingTimeInterval(60)

        // Running, but no projection anchored yet.
        #expect(runner.estimatedTimeLeft(jobId: "job-123", now: later) == nil)

        runner.anchorETA(completedCells: 2, now: later)
        // Different job than the one running.
        #expect(runner.estimatedTimeLeft(jobId: "other-job", now: later) == nil)

        // finish() clears the projection so a stale estimate can't leak out.
        runner.finish()
        #expect(runner.estimatedTimeLeft(jobId: "job-123", now: later) == nil)
    }

    @Test func estimatedTimeLeftCountsDownFromAnchorBetweenProgressUpdates() {
        let runner = JobRunner()
        runner.start(jobId: "job-123", flag: CancelFlag(), completedAtStart: 0, totalToRun: 10)
        let started = runner.startedAt!

        // 2 of 10 cells done after 60s → 300s projected total → finish at +300s.
        runner.anchorETA(completedCells: 2, now: started.addingTimeInterval(60))

        // With the projection anchored, the label counts DOWN as `now` advances
        // even though no further progress is reported — the fix for a static /
        // upward-creeping "Ns left".
        #expect(runner.estimatedTimeLeft(jobId: "job-123", now: started.addingTimeInterval(60)) == "4 min left")
        #expect(runner.estimatedTimeLeft(jobId: "job-123", now: started.addingTimeInterval(120)) == "3 min left")
        #expect(runner.estimatedTimeLeft(jobId: "job-123", now: started.addingTimeInterval(280)) == "20s left")
        // Clamps at zero once the projection is reached, rather than going negative.
        #expect(runner.estimatedTimeLeft(jobId: "job-123", now: started.addingTimeInterval(600)) == "0s left")

        // finish() drops the projection so it can't leak into the next run.
        runner.finish()
        #expect(runner.projectedFinish == nil)
    }

    @Test func anchorETAKeepsPriorProjectionUntilProgressIsMeaningful() {
        let runner = JobRunner()
        runner.start(jobId: "job-123", flag: CancelFlag(), completedAtStart: 0, totalToRun: 10)
        let started = runner.startedAt!

        // Too little this-run progress to project from — no projection yet.
        runner.anchorETA(completedCells: 0, now: started.addingTimeInterval(5))
        #expect(runner.projectedFinish == nil)

        // Once a cell completes, a projection lands and a later low-progress
        // anchor doesn't wipe it — the ETA keeps ticking down.
        runner.anchorETA(completedCells: 2, now: started.addingTimeInterval(60))
        let projected = runner.projectedFinish
        #expect(projected != nil)
        runner.anchorETA(completedCells: 0, now: started.addingTimeInterval(65))
        #expect(runner.projectedFinish == projected)
    }

    @Test func resultSubmissionFeatureGateAllowsRegisteredNonLiquidEmail() {
        let registration = registrationData(contactEmail: "user@example.com")

        #expect(ResultSubmissionFeatureGate.canSubmitResults(registration: registration))
    }

    @Test func resultSubmissionFeatureGateRequiresRegistration() {
        #expect(!ResultSubmissionFeatureGate.canSubmitResults(registration: nil))
    }

    private func manifest(title: String? = nil) -> JobManifest {
        JobManifest(
            jobId: "job-1",
            createdAt: "2026-05-28T16:41:00Z",
            nGpuLayers: 99,
            contextSize: 4096,
            cells: [cell(id: "a", benchmarkId: "decode", modelName: "Qwen", status: .completed)],
            status: .completed,
            title: title
        )
    }

    private func cell(
        id: CellId,
        benchmarkId: String,
        modelName: String,
        status: CellRunStatus,
        serverJobId: String? = nil
    ) -> JobCell {
        JobCell(
            cellId: id,
            benchmarkId: benchmarkId,
            benchmarkType: BenchmarkType(rawValue: benchmarkId),
            runStatus: status,
            serverJobId: serverJobId,
            errorMessage: nil,
            source: ggufTextFixture("test/\(modelName)-GGUF", "\(modelName)-Q4_0.gguf")
        )
    }

    private func registrationData(contactEmail: String) -> IdentityRegistration {
        IdentityRegistration(
            clientId: ClientID("client-1"),
            status: "approved",
            serverUrl: ServerURL("https://collector.example.com"),
            organization: "Example",
            contactEmail: contactEmail,
            registeredAt: "2026-05-28T16:41:00Z",
            clerkUserId: "user_1",
            clerkSessionId: "session_1",
            clerkPrimaryEmail: contactEmail,
            clerkLinkedAt: "2026-05-28T16:42:00Z"
        )
    }
}
