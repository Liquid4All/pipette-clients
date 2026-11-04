import Foundation

struct JobDrainOutcome: nonisolated Sendable {
    let submitted: Int
    let errors: [String]
}

struct ResendOutcome: nonisolated Sendable {
    let submitted: Int
    let errors: [String]
}

/// Re-drives result submissions a network error left stranded on disk; nothing else
/// retries them. Triggers (launch, foreground, `JobExecutor`, the Submit tap, commands)
/// arrive from different isolation contexts, so this is an actor and each pass awaits the
/// previous one: two triggers firing together never double-submit a cell.
actor ResultUploader {
    static let shared = ResultUploader(ws: PipetteWorkspace.production)

    /// Throws rather than answering nil: a missing registration and a signing key the
    /// Keychain no longer holds need different remedies, and `IdentityError` says which.
    typealias CredentialsLoader = () throws -> (registration: IdentityRegistration, auth: ClientIdentity)
    typealias Sleeper = (TimeInterval) async -> Void

    private let submitResultBatch: ResultSubmissionService.BatchSubmitter?
    private let credentials: CredentialsLoader
    private let retryDelays: [TimeInterval]
    private let sleep: Sleeper
    private let ws: PipetteWorkspace
    /// Each pass awaits this, so passes never interleave and each sees the manifests the
    /// previous one saved.
    private var queueTail: Task<Void, Never>?

    init(
        submitResultBatch: ResultSubmissionService.BatchSubmitter? = nil,
        credentials: CredentialsLoader? = nil,
        retryDelays: [TimeInterval] = [2, 5],
        ws: PipetteWorkspace,
        sleep: @escaping Sleeper = { seconds in
            try? await Task.sleep(nanoseconds: UInt64(seconds * 1_000_000_000))
        }
    ) {
        self.submitResultBatch = submitResultBatch
        self.credentials = credentials ?? {
            let session = try ws.registration.mgmtSession()
            return (session.registration, session.auth)
        }
        self.retryDelays = retryDelays
        self.ws = ws
        self.sleep = sleep
    }

    /// Only touches jobs where submission was already intended (`hasStrandedResults`), never
    /// a running job: the executor submits that one itself.
    @discardableResult
    func drainAll() async -> [JobId: JobDrainOutcome] {
        await enqueue { await self.performDrain(forcedJobId: nil) }
    }

    /// Ignores the job's contribute opt-in: the caller asked for this job explicitly.
    /// The drain saves accepted `serverJobId`s to disk, so callers reload the manifest.
    func drainJob(jobId: JobId) async -> JobDrainOutcome {
        let outcomes = await enqueue { await self.performDrain(forcedJobId: jobId) }
        // No entry means no manifest on disk; missing credentials are reported per job.
        return outcomes[jobId]
            ?? JobDrainOutcome(submitted: 0, errors: ["no job manifest for \(jobId.value)"])
    }

    /// Re-sends results submitted to a different collector whose `benchmarkId` is in the
    /// current collector's catalog (`benchmarkIds`).
    func resendForCollectorChange(benchmarkIds: Set<String>) async -> ResendOutcome {
        await enqueue { await self.performResend(benchmarkIds: benchmarkIds) }
    }

    /// Unacked results whose submission was already intended: the job opted into
    /// contribution, or a previous attempt left a submission record. Never a claimed job:
    /// its claimer submits, and a late upload would echo a lease that may be gone.
    func hasStrandedResults(_ manifest: JobManifest) -> Bool {
        guard manifest.claimJobId == nil else { return false }
        let stranded = manifest.cells.filter { ws.results.isUnsubmitted($0) }
        guard !stranded.isEmpty else { return false }
        if manifest.contributeResults == true { return true }
        return stranded.contains {
            ws.results.loadSubmission($0.cellId) != nil
        }
    }

    /// The `queueTail` read-modify-write is the synchronous prefix before the only `await`,
    /// so actor isolation makes it atomic.
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
            manifests = ws.loadJobManifest(jobId: forcedJobId).map { [$0] } ?? []
        } else {
            manifests = ws.filterJobManifests { $0.status != .running && hasStrandedResults($0) }
        }

        let loaded = Result { try credentials() }
        var outcomes: [JobId: JobDrainOutcome] = [:]
        for manifest in manifests {
            let registration: IdentityRegistration
            let auth: ClientIdentity
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
            // Skip no-op sweeps, which would swamp the event. Error text is never sent: it
            // embeds benchmark ids and the server URL.
            if outcome.submitted > 0 || !outcome.errors.isEmpty {
                Analytics.capture(AnalyticsEvents.resultsSubmitted, [
                    AnalyticsEvents.jobId: manifest.jobId.value,
                    AnalyticsEvents.submittedCount: outcome.submitted,
                    AnalyticsEvents.ok: outcome.errors.isEmpty,
                ])
                // `JobExecutor` drains between cells, so flush now rather than let the SDK's
                // timer POST while the next cell is being timed.
                Analytics.flush()
            }
        }
        return outcomes
    }

    private func performResend(benchmarkIds: Set<String>) async -> ResendOutcome {
        guard !benchmarkIds.isEmpty else { return ResendOutcome(submitted: 0, errors: []) }
        let registration: IdentityRegistration
        let auth: ClientIdentity
        do {
            (registration, auth) = try credentials()
        } catch {
            return ResendOutcome(submitted: 0, errors: [error.localizedDescription])
        }

        var submitted = 0
        var errors: [String] = []
        for manifest in ws.loadAllJobManifests() {
            let outcome = await ResultSubmissionService.submit(
                manifest: manifest,
                registration: registration,
                auth: auth,
                resubmitForCollectorChange: true,
                allowedBenchmarkIds: benchmarkIds,
                submitResultBatch: submitResultBatch,
                ws: ws
            )
            submitted += outcome.submitted
            errors.append(contentsOf: outcome.errors)
            if !outcome.errors.isEmpty {
                AppLog.resultUploader.error("\(manifest.jobId) collector-change resend: \(outcome.errors.joined(separator: "; "))")
            }
        }
        return ResendOutcome(submitted: submitted, errors: errors)
    }

    /// Retries only transport failures: a server rejection won't change on a retry.
    private func submitWithRetry(
        manifest: JobManifest,
        registration: IdentityRegistration,
        auth: ClientIdentity
    ) async -> ResultSubmissionOutcome {
        var outcome = await ResultSubmissionService.submit(
            manifest: manifest,
            registration: registration,
            auth: auth,
            submitResultBatch: submitResultBatch,
            ws: ws
        )
        for delay in retryDelays {
            guard outcome.hadTransportFailure else { break }
            await sleep(delay)
            outcome = await ResultSubmissionService.submit(
                manifest: outcome.manifest,
                registration: registration,
                auth: auth,
                submitResultBatch: submitResultBatch,
                ws: ws
            )
        }
        return outcome
    }
}
