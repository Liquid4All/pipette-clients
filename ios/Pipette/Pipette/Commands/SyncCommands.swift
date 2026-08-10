import Foundation

/// `sync`: pull the benchmark catalog, then submit what is pending — the phases
/// `pipette sync` runs, in its order, over the services the UI already binds.
///
/// One `[HEADLESS] sync …` line per phase, so a console reader sees the same shape
/// whether or not a phase found anything to do.
///
/// The CLI has a third phase, refreshing scores for already-submitted results. iOS has no
/// score refresh yet, so this reports `scores=not-refreshed` rather than printing nothing:
/// a missing line reads as "nothing needed refreshing", which is a different claim.
enum SyncCommands {
    /// Returns `true` when every phase completed.
    ///
    /// A failed catalog pull ends the command before submission, as the CLI's `?` on
    /// `pull_remote_benchmarks` does. Per-job submission errors are reported and do not
    /// fail it, matching `pipette sync`, which prints them and returns `Ok`.
    static func run(jobId: String?, storage: Storage) async -> Bool {
        guard let registration = storage.identity.getRegistration() else {
            // Both phases talk to the collector, so there is nothing to attempt.
            HeadlessRunner.log("sync ERROR this device has no registration: "
                + "run `headlessrun auth register org=… email=…` first")
            return false
        }
        let serverUrl = registration.serverUrl

        do {
            let pulled = try await BenchmarkSyncCoordinator.shared.sync(
                serverUrl: serverUrl, storage: storage)
            HeadlessRunner.log("sync pulled benchmarks=\(pulled)")
        } catch {
            HeadlessRunner.log("sync ERROR benchmark pull failed: \(error)")
            return false
        }

        // Narrowed to one job when asked. The CLI narrows by *result* id; iOS addresses
        // whole jobs, which is the unit its store and its submission record share.
        let outcomes: [JobId: JobDrainOutcome]
        if let jobId {
            let id = JobId(jobId)
            outcomes = [id: await ResultUploader.shared.drainJob(jobId: id)]
        } else {
            outcomes = await ResultUploader.shared.drainAll()
        }
        let submitted = outcomes.values.reduce(0) { $0 + $1.submitted }
        let errors = outcomes.values.flatMap(\.errors)
        HeadlessRunner.log("sync submitted results=\(submitted) jobs=\(outcomes.count) "
            + "errors=\(errors.count)")
        for message in errors { HeadlessRunner.log("sync error \(message)") }

        HeadlessRunner.log("sync scores=not-refreshed")
        return true
    }
}
