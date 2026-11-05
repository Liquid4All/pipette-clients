import Foundation

/// Single source of truth for job manifests in the UI. Views render `jobs` and `job(id:)`
/// directly rather than keeping their own copies, so a mutation through the store reaches
/// every screen.
///
/// Memory stays small however many jobs the device has: a `JobSummary` per job and only the
/// last few full manifests `job(id:)` read, each loaded on first use, so a headless launch,
/// which shows no job screen, never reads the job history.
///
/// UI mutations go through `save`/`delete`. Writers that persist through the workspace
/// themselves (`JobExecutor`, result submission) call `apply` on the main actor afterwards.
@MainActor
@Observable
final class JobStore {
    /// More than the job screens that can show at once (a job page and its Pocket Mode).
    static let openJobLimit = 4

    private let ws: PipetteWorkspace
    // The reads register on this, not on the caches they fill while a view renders.
    private var generation = 0
    @ObservationIgnored private var summaries: [JobSummary]?
    @ObservationIgnored private var openJobs: [JobManifest] = []

    init(ws: PipetteWorkspace) {
        self.ws = ws
    }

    /// Newest first, as `loadJobSummaries` sorts them.
    var jobs: [JobSummary] {
        _ = generation
        if let summaries { return summaries }
        let loaded = ws.loadJobSummaries()
        summaries = loaded
        return loaded
    }

    func job(id: JobId) -> JobManifest? {
        _ = generation
        if let open = openJobs.first(where: { $0.jobId == id }) { return open }
        guard let manifest = ws.loadJobManifest(jobId: id) else { return nil }
        openJobs.append(manifest)
        if openJobs.count > Self.openJobLimit { openJobs.removeFirst() }
        return manifest
    }

    /// Drop what is in memory, so the next read catches up with writes made behind the store.
    func reload() {
        summaries = nil
        openJobs = []
        generation += 1
    }

    func save(_ manifest: JobManifest) {
        ws.saveJobManifest(manifest)
        apply(manifest)
    }

    /// Sync one already-persisted manifest into memory without re-reading every job from disk.
    func apply(_ manifest: JobManifest) {
        if let index = openJobs.firstIndex(where: { $0.jobId == manifest.jobId }) {
            openJobs[index] = manifest
        }
        if var list = summaries {
            let summary = JobSummary(manifest)
            if let index = list.firstIndex(where: { $0.jobId == manifest.jobId }) {
                list[index] = summary
            } else {
                // Unknown job: insert it where the newest-first load would have put it.
                list.insert(summary, at: list.firstIndex { $0.createdAt < manifest.createdAt } ?? list.endIndex)
            }
            summaries = list
        }
        generation += 1
    }

    func delete(jobId: JobId) {
        ws.deleteJob(jobId: jobId)
        openJobs.removeAll { $0.jobId == jobId }
        summaries?.removeAll { $0.jobId == jobId }
        generation += 1
    }
}
