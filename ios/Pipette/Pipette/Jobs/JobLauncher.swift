import Foundation

/// The one job-start sequence: mint a `JobId`, reserve the runner (the
/// double-start guard), persist the manifest as `.running`, and hand it to
/// `JobExecutor`. Used by both `NewJobView`'s start action and the headless
/// CLI's bench command, so a CLI-created job is identical to a UI-created one.
@MainActor
enum JobLauncher {
    /// Returns the minted `JobId`, or `nil` when blocked because another job
    /// is already running — the caller decides how to surface that.
    ///
    /// `onFinish` is forwarded to `JobExecutor.run` and invoked exactly once
    /// on the main actor with the final persisted manifest (never called when
    /// launch returns `nil`).
    static func launch(
        cells: [JobCell],
        nGpuLayers: Int,
        contextSize: Int,
        prefillBatch: Int?,
        threads: Int? = nil,
        contributeResults: Bool,
        readiness: ReadinessPolicy = .init(),
        jobRunner: JobRunner,
        jobStore: JobStore,
        storage: Storage,
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
            contributeResults: contributeResults
        )
        jobStore.save(manifest)

        JobExecutor.run(
            manifest: manifest, jobRunner: jobRunner, jobStore: jobStore,
            flag: flag, storage: storage, source: .new, readiness: readiness,
            onFinish: onFinish
        )
        return jobId
    }

    /// Re-run a saved job's terminal cells: reserve the runner, flip every cell
    /// currently in `target` back to `.pending` (clearing its error), mark the
    /// job `.running`, persist, and hand off to `JobExecutor`. Returns the job id,
    /// or `nil` when another job is already running. The one resume/retry sequence
    /// shared by the CLI (`job run`) and the deep-link router; the caller loads the
    /// manifest and surfaces the missing-job / busy cases in its own idiom.
    static func rerun(
        _ manifest: JobManifest,
        resetting target: CellRunStatus,
        jobRunner: JobRunner,
        jobStore: JobStore,
        storage: Storage,
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
            flag: flag, storage: storage,
            // Derived, not hard-coded: this same function backs `job run --scope cancelled` and the
            // deep-link resume, and reporting those as `rerun` would file them under a different
            // intent than the identical action taken from JobDetailView.
            source: .resettingCells(withStatus: target), onFinish: onFinish
        )
        return manifest.jobId
    }
}
