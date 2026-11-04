import Foundation

/// The one job-start sequence: mint a `JobId`, reserve the runner (the double-start
/// guard), persist the manifest as `.running`, and hand it to `JobExecutor`. Shared by
/// `NewJobView`, the headless CLI and deep links, so a job is identical whichever started it.
@MainActor
enum JobLauncher {
    /// Returns the minted `JobId`, or `nil` when another job is already running.
    ///
    /// `onFinish` runs once on the main actor with the final persisted manifest, and never
    /// when this returns `nil`. `claimedAt` is when this device received `claimJobId`, and
    /// `claimedBenchmarks` the definitions that claim resolved through the server.
    static func launch(
        cells: [JobCell],
        nGpuLayers: Int,
        contextSize: Int,
        prefillBatch: Int?,
        threads: Int? = nil,
        contributeResults: Bool,
        claimJobId: String? = nil,
        claimedAt: ContinuousClock.Instant? = nil,
        claimedBenchmarks: [BenchmarkDefinition] = [],
        jobRunner: JobRunner,
        jobStore: JobStore,
        ws: PipetteWorkspace,
        onFinish: @escaping @MainActor (JobManifest) -> Void = { _ in }
    ) -> JobId? {
        let jobId = JobId(UUID().uuidString)
        let flag = CancelFlag()
        guard jobRunner.start(
            jobId: jobId,
            flag: flag,
            completedAtStart: 0,
            totalToRun: cells.count
        ) else {
            return nil
        }

        let manifest = JobManifest(
            jobId: jobId,
            createdAt: JobDateFormat.iso8601.string(from: Date()),
            nGpuLayers: nGpuLayers,
            contextSize: contextSize,
            prefillBatch: prefillBatch,
            threads: threads,
            cells: cells,
            status: .running,
            contributeResults: contributeResults,
            claimJobId: claimJobId
        )
        jobStore.save(manifest)

        JobExecutor.run(
            manifest: manifest, jobRunner: jobRunner, jobStore: jobStore,
            flag: flag, ws: ws, source: .new, claimedAt: claimedAt,
            claimedBenchmarks: claimedBenchmarks, onFinish: onFinish
        )
        return jobId
    }

    /// Re-run a saved job: every cell in `target` goes back to `.pending` with its error
    /// cleared. Returns `nil` when another job is already running. Shared by the CLI
    /// (`job run`) and the deep-link router.
    static func rerun(
        _ manifest: JobManifest,
        resetting target: CellRunStatus,
        jobRunner: JobRunner,
        jobStore: JobStore,
        ws: PipetteWorkspace,
        onFinish: @escaping @MainActor (JobManifest) -> Void = { _ in }
    ) -> JobId? {
        let flag = CancelFlag()
        guard jobRunner.start(jobId: manifest.jobId, flag: flag) else { return nil }
        var manifest = manifest
        for i in manifest.cells.indices where manifest.cells[i].runStatus == target {
            manifest.cells[i].runStatus = .pending
            manifest.cells[i].errorMessage = nil
        }
        manifest.status = .running
        manifest.pausedReason = nil
        jobStore.save(manifest)
        JobExecutor.run(
            manifest: manifest, jobRunner: jobRunner, jobStore: jobStore,
            flag: flag, ws: ws,
            // Derived from `target`, so `job run --scope cancelled` and the deep-link resume
            // report the same intent as the identical action from JobDetailView.
            source: .resettingCells(withStatus: target), onFinish: onFinish
        )
        return manifest.jobId
    }
}
