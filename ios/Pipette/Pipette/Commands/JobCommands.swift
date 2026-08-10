import Foundation

/// Jobs-tab / job-detail CLI verbs (`jobs`, `job rm|run|export|submit`): thin
/// handlers over the same `JobStore` / `JobLauncher` / uploader paths the UI
/// binds. Dispatched from `HeadlessRunner.startIfRequested`.
enum JobCommands {
    /// `jobs`: list every job manifest with the same status/counts the jobs
    /// list shows. The free-form `title` goes last so the fixed fields stay
    /// machine-parseable.
    static func list(storage: Storage) async {
        let manifests = storage.loadAllJobManifests()
        HeadlessRunner.log("jobs count=\(manifests.count)")
        for m in manifests {
            HeadlessRunner.log("job id=\(m.jobId.value) status=\(m.status.rawValue) "
                + "completed=\(m.completedCells) failed=\(m.failedCells) "
                + "cancelled=\(m.cancelledCells) total=\(m.totalCells) "
                + "submitted=\(m.submittedCells) title=\(m.displayTitle)")
        }
    }

    /// `job rm`: delete the job and all its results — the UI's delete
    /// confirmation, via the same `JobStore.delete`.
    static func remove(jobId: JobId, storage: Storage) async -> Bool {
        guard let manifest = storage.loadJobManifest(jobId: jobId) else {
            HeadlessRunner.log("job rm ERROR no job with id=\(jobId.value)")
            return false
        }
        await MainActor.run { JobStore(storage: storage).delete(jobId: jobId) }
        HeadlessRunner.log("job rm deleted id=\(jobId.value) title=\(manifest.displayTitle)")
        return true
    }

    /// `job run`: flip the scoped terminal cells back to `.pending` and re-run
    /// the job — exactly `JobDetailView.runCells(scope:)` for the failed
    /// (retry) and cancelled (resume) scopes, which reset `runStatus` and
    /// `errorMessage` only; submission records are kept, matching the UI (only
    /// the UI's per-cell re-run wipes them). Exits by the job's final status.
    static func run(jobId: JobId, scope: HeadlessCommand.RunScope, storage: Storage) async -> Bool {
        let final: JobManifest? = await withCheckedContinuation { cont in
            Task { @MainActor in
                guard let manifest = storage.loadJobManifest(jobId: jobId) else {
                    HeadlessRunner.log("job run ERROR no job with id=\(jobId.value)")
                    cont.resume(returning: nil)
                    return
                }
                let target = scope.resetTarget
                let flipped = manifest.cells.filter { $0.runStatus == target }.count
                let launched = JobLauncher.rerun(
                    manifest, resetting: target, jobRunner: JobRunner(),
                    jobStore: JobStore(storage: storage), storage: storage,
                    onFinish: { cont.resume(returning: $0) })
                guard launched != nil else {
                    HeadlessRunner.log("job run ERROR another job is running")
                    cont.resume(returning: nil)
                    return
                }
                HeadlessRunner.log("job run start id=\(jobId.value) scope=\(scope.rawValue) cells=\(flipped)")
            }
        }
        guard let final else { return false }
        HeadlessRunner.log("job run finished status=\(final.status.rawValue) completed=\(final.completedCells) "
            + "failed=\(final.failedCells) cancelled=\(final.cancelledCells)")
        return final.status == .completed
    }

    /// `job export`: print the same CSV the export button saves, raw to stdout
    /// (between marker lines) so a `--console` capture can extract it — the
    /// export control minus the file picker.
    static func export(jobId: JobId, storage: Storage) async -> Bool {
        guard let manifest = storage.loadJobManifest(jobId: jobId) else {
            HeadlessRunner.log("job export ERROR no job with id=\(jobId.value)")
            return false
        }
        let csv = await MainActor.run { CompletedResultsCSVExporter.csv(for: manifest, storage: storage) }
        HeadlessRunner.log("job export id=\(jobId.value) BEGIN CSV")
        print(csv)
        HeadlessRunner.log("END CSV")
        return true
    }

    /// `job submit`: upload the job's unsubmitted completed results with the
    /// stored registration — exactly `JobDetailView.submitAll()`: the same
    /// registration + Keychain-key guard, then a drain through the shared
    /// uploader so it serializes with any launch/foreground drain.
    static func submit(jobId: JobId, storage: Storage) async -> Bool {
        guard storage.loadJobManifest(jobId: jobId) != nil else {
            HeadlessRunner.log("job submit ERROR no job with id=\(jobId.value)")
            return false
        }
        let registered = await MainActor.run {
            storage.identity.signingIdentity() != nil
        }
        guard registered else {
            HeadlessRunner.log("job submit ERROR Result submission requires registration.")
            return false
        }
        let outcome = await ResultUploader.shared.drainJob(jobId: jobId)
        HeadlessRunner.log("job submit submitted=\(outcome.submitted) errors=\(outcome.errors.count)")
        for error in outcome.errors { HeadlessRunner.log("job submit ERROR \(error)") }
        return outcome.errors.isEmpty
    }
}
