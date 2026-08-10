import Foundation

/// Single source of truth for job manifests in the UI.
///
/// Views render `jobs` (or `job(id:)`) directly instead of keeping their own
/// `@State` copies loaded from disk, so any mutation that goes through the
/// store updates every screen immediately — there is no per-view copy to go
/// stale and no invalidation signal to forget.
///
/// All UI-initiated mutations (create, rename, delete, status changes) must go
/// through `save`/`delete`. Writers that persist via `LocalStorage` themselves
/// (`JobExecutor` from its background task, result submission) call `apply` on
/// the main actor afterwards to bring the store back in sync.
@MainActor
@Observable
final class JobStore {
    /// All job manifests, newest first (the `LocalStorage` sort order).
    private(set) var jobs: [JobManifest] = []
    private let storage: Storage

    init(storage: Storage) {
        self.storage = storage
    }

    func job(id: JobId) -> JobManifest? {
        jobs.first { $0.jobId == id }
    }

    /// Re-read every manifest from disk. The store starts empty and is loaded
    /// by the root view's `onAppear` (not in `init`) so launch-time storage
    /// migration and interrupted-job recovery run first.
    func reload() {
        jobs = storage.loadAllJobManifests()
    }

    func save(_ manifest: JobManifest) {
        storage.saveJobManifest(manifest)
        apply(manifest)
    }

    /// Sync one already-persisted manifest into the in-memory list without
    /// re-reading every job from disk.
    func apply(_ manifest: JobManifest) {
        if let index = jobs.firstIndex(where: { $0.jobId == manifest.jobId }) {
            jobs[index] = manifest
        } else {
            // Unknown job: reload so it lands in the canonical sort position.
            reload()
        }
    }

    func delete(jobId: JobId) {
        storage.deleteJob(jobId: jobId)
        jobs.removeAll { $0.jobId == jobId }
    }
}
