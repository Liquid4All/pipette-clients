import Foundation

/// `jobs` and `job rm|run|export|submit`: thin handlers over the `JobStore`, `JobLauncher`
/// and uploader paths the UI binds.
enum JobCommands {
    /// The free-form `title` goes last so the fixed fields stay machine-parseable.
    static func list(ws: PipetteWorkspace) async {
        let manifests = ws.loadAllJobManifests()
        HeadlessRunner.log("jobs count=\(manifests.count)")
        for m in manifests {
            HeadlessRunner.log("job id=\(m.jobId.value) status=\(m.status.rawValue) "
                + "completed=\(m.completedCells) failed=\(m.failedCells) "
                + "cancelled=\(m.cancelledCells) total=\(m.totalCells) "
                + "submitted=\(m.submittedCells) title=\(m.displayTitle)")
        }
    }

    /// `job rm`: delete the job and all its results, as the UI's delete does.
    static func remove(jobId: JobId, ws: PipetteWorkspace) async -> Bool {
        guard let manifest = ws.loadJobManifest(jobId: jobId) else {
            HeadlessRunner.log("job rm ERROR no job with id=\(jobId.value)")
            return false
        }
        await MainActor.run { JobStore(ws: ws).delete(jobId: jobId) }
        HeadlessRunner.log("job rm deleted id=\(jobId.value) title=\(manifest.displayTitle)")
        return true
    }

    /// `job run`: re-run the failed (retry) or cancelled (resume) cells, as
    /// `JobDetailView.runCells(scope:)` does. Submission records are kept, as in the UI.
    static func run(jobId: JobId, scope: HeadlessCommand.RunScope, ws: PipetteWorkspace) async -> Bool {
        let final: JobManifest? = await withCheckedContinuation { cont in
            Task { @MainActor in
                guard let manifest = ws.loadJobManifest(jobId: jobId) else {
                    HeadlessRunner.log("job run ERROR no job with id=\(jobId.value)")
                    cont.resume(returning: nil)
                    return
                }
                let target = scope.resetTarget
                let flipped = manifest.cells.filter { $0.runStatus == target }.count
                let launched = JobLauncher.rerun(
                    manifest, resetting: target, jobRunner: JobRunner(),
                    jobStore: JobStore(ws: ws), ws: ws,
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

    /// `job export`: the CSV the export button saves. The CSV alone goes to stdout and every
    /// `[HEADLESS]` line (markers, refusal, `BENCH_DONE`) to stderr, so
    /// `job export ... > out.csv` is the CSV and nothing else.
    static func export(jobId: JobId, ws: PipetteWorkspace) async -> Bool {
        guard let manifest = ws.loadJobManifest(jobId: jobId) else {
            HeadlessRunner.logDiagnostic("job export ERROR no job with id=\(jobId.value)")
            return false
        }
        let csv = await MainActor.run { CompletedResultsCSVExporter.csv(for: manifest, ws: ws) }
        HeadlessRunner.logDiagnostic("job export id=\(jobId.value) BEGIN CSV")
        HeadlessRunner.emitPayload(csv)
        HeadlessRunner.logDiagnostic("job export id=\(jobId.value) END CSV")
        return true
    }

    /// `job submit`, as `JobDetailView.submitAll()`: the drain goes through the shared
    /// uploader so it serializes with any launch or foreground drain.
    static func submit(jobId: JobId, ws: PipetteWorkspace) async -> Bool {
        guard ws.loadJobManifest(jobId: jobId) != nil else {
            HeadlessRunner.log("job submit ERROR no job with id=\(jobId.value)")
            return false
        }
        let registered = await MainActor.run {
            ws.registration.signingIdentity() != nil
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
