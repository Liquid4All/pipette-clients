import Foundation

/// Outcome of draining one job's unsubmitted results.
struct JobDrainOutcome: nonisolated Sendable {
    let submitted: Int
    let errors: [String]
}

/// Aggregate outcome of a collector-change resend sweep across every job.
struct ResendOutcome: nonisolated Sendable {
    let submitted: Int
    let errors: [String]
}

/// Serialized, idempotent uploader that re-drives stranded benchmark result
/// submissions. A network error during submission leaves cells marked
/// `.failed` with their payloads still on disk; nothing about the run loop
/// retries them, so without this the results stay stranded until the user
/// happens to tap Submit on that job's page.
///
/// Drains run on coarse triggers: app launch, app foreground (scenePhase
/// `.active`), job end (`JobExecutor`), and the manual Submit tap
/// (`JobDetailView`). These reach the uploader from different isolation
/// contexts (the main actor and a background executor), so it is an `actor`:
/// actor isolation serializes the `queueTail` read-modify-write that chains the
/// passes. Each drain awaits the previous one — so two triggers firing together
/// can never double-submit the same cell.
actor ResultUploader {
    static let shared = ResultUploader(storage: FileStorage.production)

    /// Throws rather than answering nil, so the reason reaches the caller: a missing
    /// registration and a signing key the Keychain no longer holds are different
    /// problems with different remedies, and `IdentityError` already says which.
    typealias CredentialsLoader = () throws -> (registration: IdentityRegistration, auth: AuthIdentity)
    typealias Sleeper = (TimeInterval) async -> Void

    private let submitResultBatch: ResultSubmissionService.BatchSubmitter?
    private let credentials: CredentialsLoader
    private let retryDelays: [TimeInterval]
    private let sleep: Sleeper
    private let storage: Storage
    /// Tail of the work queue: every drain or re-upload awaits the task before
    /// it, so passes never interleave and each pass sees the manifests the
    /// previous one saved. Actor isolation makes the read-modify-write in
    /// `enqueue` atomic — it is the synchronous prefix before the only `await`.
    private var queueTail: Task<Void, Never>?

    init(
        submitResultBatch: ResultSubmissionService.BatchSubmitter? = nil,
        credentials: CredentialsLoader? = nil,
        retryDelays: [TimeInterval] = [2, 5],
        storage: Storage,
        sleep: @escaping Sleeper = { seconds in
            try? await Task.sleep(nanoseconds: UInt64(seconds * 1_000_000_000))
        }
    ) {
        self.submitResultBatch = submitResultBatch
        self.credentials = credentials ?? {
            let session = try storage.mgmtSession()
            return (session.registration, session.auth)
        }
        self.retryDelays = retryDelays
        self.storage = storage
        self.sleep = sleep
    }

    /// Submit every stranded result across all jobs. Used by the launch and
    /// foreground triggers, so it only touches jobs where submission was
    /// already intended (see `hasStrandedResults`) and never a job that is
    /// currently running — the executor submits that one itself at job end.
    @discardableResult
    func drainAll() async -> [JobId: JobDrainOutcome] {
        await enqueue { await self.performDrain(forcedJobId: nil) }
    }

    /// Submit one job's unsubmitted results regardless of its contribute
    /// opt-in — the user (or the job-end auto-submit) asked for this job
    /// explicitly. Callers reload the manifest afterwards; the drain saves
    /// every accepted `serverJobId` through `LocalStorage`.
    func drainJob(jobId: JobId) async -> JobDrainOutcome {
        let outcomes = await enqueue { await self.performDrain(forcedJobId: jobId) }
        // No entry means the drain never saw the job — a manifest that is not on disk,
        // not that the device lacks credentials (which the pass reports per job).
        return outcomes[jobId]
            ?? JobDrainOutcome(submitted: 0, errors: ["no job manifest for \(jobId.value)"])
    }

    /// Migrate results onto the currently configured collector: re-send every
    /// completed result whose `benchmarkId` is in `benchmarkIds` (the collector's
    /// current catalog) and that was previously submitted to a *different*
    /// collector. Results already on the current collector are left untouched.
    /// Serializes behind the same queue as drains, so a resend and a drain can't
    /// submit the same cell at once.
    func resendForCollectorChange(benchmarkIds: Set<String>) async -> ResendOutcome {
        await enqueue { await self.performResend(benchmarkIds: benchmarkIds) }
    }

    /// A job needs a drain when it has completed cells whose payloads sit on
    /// disk without a server ack, and submitting them was already intended:
    /// either the job opted into auto-contribution, or a previous attempt
    /// left a submission record (failed mid-flight, or accepted but never
    /// adopted into the manifest). Jobs that never opted in stay manual-only.
    func hasStrandedResults(_ manifest: JobManifest) -> Bool {
        // `isUnsubmitted` carries the `isSubmittable` clause for the same reason the submit sweep
        // checks it: a result from the generated catalog half never goes up, so treating it as
        // stranded wakes this uploader on every launch and foreground to do nothing.
        let stranded = manifest.cells.filter { storage.results.isUnsubmitted($0) }
        guard !stranded.isEmpty else { return false }
        if manifest.contributeResults == true { return true }
        return stranded.contains {
            storage.results.loadSubmission($0.cellId) != nil
        }
    }

    /// Chain `work` onto the queue tail so passes never interleave: it awaits
    /// the previous task before running, and becomes the tail the next call
    /// waits on. The read-modify-write of `queueTail` is the synchronous prefix
    /// before the only `await`, so actor isolation makes it atomic.
    private func enqueue<T: Sendable>(_ work: @escaping () async -> T) async -> T {
        let previous = queueTail
        let task = Task { () -> T in
            _ = await previous?.value
            return await work()
        }
        queueTail = Task { _ = await task.value }
        return await task.value
    }

    private func performDrain(forcedJobId: JobId?) async -> [JobId: JobDrainOutcome] {
        let manifests: [JobManifest]
        if let forcedJobId {
            manifests = storage.loadJobManifest(jobId: forcedJobId).map { [$0] } ?? []
        } else {
            manifests = storage.loadAllJobManifests()
        }

        let loaded = Result { try credentials() }
        var outcomes: [JobId: JobDrainOutcome] = [:]
        for manifest in manifests {
            if forcedJobId == nil {
                guard manifest.status != .running, hasStrandedResults(manifest) else { continue }
            }
            let registration: IdentityRegistration
            let auth: AuthIdentity
            switch loaded {
            case let .success(loaded):
                (registration, auth) = loaded
            case let .failure(error):
                if forcedJobId != nil {
                    outcomes[manifest.jobId] = JobDrainOutcome(
                        submitted: 0,
                        errors: [error.localizedDescription]
                    )
                }
                continue
            }

            let outcome = await submitWithRetry(
                manifest: manifest,
                registration: registration,
                auth: auth
            )
            outcomes[manifest.jobId] = JobDrainOutcome(
                submitted: outcome.submitted,
                errors: outcome.errors
            )
            if !outcome.errors.isEmpty {
                AppLog.resultUploader.error("\(manifest.jobId): \(outcome.errors.joined(separator: "; "))")
            }
            // Only when this pass actually did something. The launch/foreground drain routinely
            // finds nothing pending, and capturing those no-op sweeps would swamp the event with
            // rows describing no user-visible action. Error TEXT is never sent: those strings hold
            // server messages that embed benchmark ids and the server URL, only whether any occurred.
            if outcome.submitted > 0 || !outcome.errors.isEmpty {
                Analytics.capture(AnalyticsEvents.resultsSubmitted, [
                    AnalyticsEvents.jobId: manifest.jobId.value,
                    AnalyticsEvents.submittedCount: outcome.submitted,
                    AnalyticsEvents.ok: outcome.errors.isEmpty,
                ])
                // The only capture that can happen mid-job: `JobExecutor` drains per-cell between
                // cells (PIP-358). Flush here so the queue is empty again before the next cell's
                // measurement, rather than leaving a row for the SDK's periodic timer to POST while
                // a cell is being timed. We are already off the measurement path (the upload above
                // just did network I/O on this same thread), so the flush costs nothing extra.
                Analytics.flush()
            }
        }
        return outcomes
    }

    private func performResend(benchmarkIds: Set<String>) async -> ResendOutcome {
        guard !benchmarkIds.isEmpty else { return ResendOutcome(submitted: 0, errors: []) }
        let registration: IdentityRegistration
        let auth: AuthIdentity
        do {
            (registration, auth) = try credentials()
        } catch {
            return ResendOutcome(submitted: 0, errors: [error.localizedDescription])
        }

        var submitted = 0
        var errors: [String] = []
        for manifest in storage.loadAllJobManifests() {
            let outcome = await ResultSubmissionService.submit(
                manifest: manifest,
                registration: registration,
                auth: auth,
                resubmitForCollectorChange: true,
                allowedBenchmarkIds: benchmarkIds,
                submitResultBatch: submitResultBatch,
                storage: storage
            )
            submitted += outcome.submitted
            errors.append(contentsOf: outcome.errors)
            if !outcome.errors.isEmpty {
                AppLog.resultUploader.error("\(manifest.jobId) collector-change resend: \(outcome.errors.joined(separator: "; "))")
            }
        }
        return ResendOutcome(submitted: submitted, errors: errors)
    }

    /// One submit pass plus modest backed-off retries, but only while the
    /// failure was in transit — per-result server rejections won't change on
    /// a retry and are left for the next coarse trigger or manual tap.
    private func submitWithRetry(
        manifest: JobManifest,
        registration: IdentityRegistration,
        auth: AuthIdentity
    ) async -> ResultSubmissionOutcome {
        var outcome = await ResultSubmissionService.submit(
            manifest: manifest,
            registration: registration,
            auth: auth,
            submitResultBatch: submitResultBatch,
            storage: storage
        )
        for delay in retryDelays {
            guard outcome.hadTransportFailure else { break }
            await sleep(delay)
            outcome = await ResultSubmissionService.submit(
                manifest: outcome.manifest,
                registration: registration,
                auth: auth,
                submitResultBatch: submitResultBatch,
                storage: storage
            )
        }
        return outcome
    }
}
