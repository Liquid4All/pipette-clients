import Foundation
import UIKit

// The claim loop, as `pipette-cli`'s `commands/worker.rs`.

/// The claim loop's knobs, the crate's `WorkerArgs` field for field and default for default.
///
/// A type rather than `setEnabled` parameters because the loop reads them at four points
/// and a session must answer with the same values at all four. The Settings toggle uses
/// the defaults, so it and a bare `headlessrun worker` are the same session.
nonisolated struct WorkerArgs: Equatable, Sendable {
    /// Base idle wait after a 204 claim.
    var idleSecs: TimeInterval = defaultIdleWait
    /// Max jitter added to the idle wait, so a fleet woken together does not claim
    /// together.
    var idleJitterSecs: TimeInterval = defaultIdleJitter
    /// Heartbeat period while a job runs. `nil` takes half the claim's `time_window`.
    var heartbeatSecs: TimeInterval?
    /// Stop after this many jobs finish; `0` runs until the process is killed.
    var maxJobs: UInt = 0
    /// Skip the startup profile PATCH. Debugging only: a server whose profile is stale will
    /// match this device against work it cannot run.
    var skipProfileRefresh: Bool = false

    /// Rolled per sleep, not per session: a single roll would give each device a fixed
    /// offset and keep the thundering herd the jitter exists to break up.
    func idleWaitWithJitter() -> TimeInterval {
        Pipette.idleWaitWithJitter(base: idleSecs, jitter: idleJitterSecs)
    }
}

/// The planner claim loop behind the Settings "Planner worker" toggle and
/// `headlessrun worker` (the mgmt repo's `docs/client-integration.md`). The worker runs
/// exactly the cell the claim names and invents no runtime.
///
/// ## Limits
/// - Desktop runtime descriptors fail non-retriably.
/// - Missing weights are fetched under the lease, so a claim can hold one for the length
///   of a multi-gigabyte download. A failed fetch is retriable unless the vault refusal
///   is permanent.
/// - No mid-run cancel on lease loss; submit is skipped after the run. A lease lost
///   mid-download discards the transfer and the cell.
/// - A submit that exhausts its backoff drops the outcome, as the crate's does. The
///   result stays on disk, filed as synced, so nothing re-sends it.
/// - Backgrounding cancels an in-flight cell and reports a retriable failure.
@MainActor
@Observable
final class PlannerWorker {
    static let shared = PlannerWorker()

    /// Prose for the Settings row; never key behaviour on it.
    private(set) var statusText: String = "Off"

    /// Why the loop is no longer running, or `nil` while it is. `headlessrun worker` keys
    /// its exit code on this, not on `statusText`, so a copy-edit is not a behaviour change.
    private(set) var sessionOutcome: SessionOutcome?

    enum SessionOutcome: Equatable, Sendable {
        /// `--max-jobs` reached.
        case completed(jobs: UInt)
        /// Stopped by the Settings toggle, the scene leaving the foreground, or
        /// process teardown.
        case stopped
        /// The server refused the claim, or the startup profile is not `approved`. The
        /// crate exits non-zero on both.
        case notApproved
        /// The controllers were not bound, or there is no registration to sign with.
        case notStarted
        /// An error the loop does not retry: a fatal claim error, or a failed startup
        /// profile refresh.
        case failed(String)

        /// Drives the headless exit code.
        var isSuccess: Bool {
            switch self {
            case .completed, .stopped: return true
            case .notApproved, .notStarted, .failed: return false
            }
        }
    }

    private var loopTask: Task<Void, Never>?

    /// Whether a session is up in this process, including one draining an in-flight claim
    /// after a stop.
    var isRunning: Bool { loopTask != nil }

    /// Whether the worker is switched on in this process: what the Settings toggle shows.
    /// Nothing saves it, so every launch starts with the worker off.
    var isEnabled: Bool { enabled }

    /// When false, the loop exits after the current claim finishes, or at once from an
    /// idle sleep.
    private var enabled = false
    private var stopReason: PlannerStopReason = .user

    /// Bound when the loop starts and used for stop/cancel, so Settings, headless or
    /// scenePhase cannot pass a different `JobRunner` than the one executing the claim.
    private var sessionStorage: PipetteWorkspace?
    private var sessionRunner: JobRunner?
    private var sessionStore: JobStore?
    /// Bound for the same reason: the Settings toggle and `headlessrun worker` start a
    /// session with different knobs.
    private var sessionOptions = WorkerArgs()
    /// Bumped once per session, so a task can tell whether it still owns the state it
    /// is about to write.
    ///
    /// Stopping nils `loopTask` synchronously, but the cancelled task resumes later. If a
    /// new session started in between, that task's teardown would clear the new session's
    /// controllers. Toggling Settings off and on, or a quick background cycle, reaches it.
    private var sessionGeneration: UInt64 = 0

    private enum PlannerStopReason {
        case user
        case background
    }

    /// Set by the heartbeat on lease loss, read at run completion; it does not cancel the
    /// job. Internal for the tests that pin the lease-loss contract.
    final class LeaseLostFlag: @unchecked Sendable {
        private let lock = NSLock()
        private var lost = false
        var isLost: Bool {
            lock.lock(); defer { lock.unlock() }
            return lost
        }
        func mark() {
            lock.lock(); lost = true; lock.unlock()
        }
    }

    /// A run that produced a payload. The eval checkpoint is kept until the server has the
    /// result. Internal so `PlannerWorkerTests` can hand one to ``submitFinishedRun``.
    struct CompletedRun {
        let manifest: JobManifest
        let cellId: CellId
        let payload: [String: Any]
        let evalCheckpoint: EvalRunDigest?
    }

    private init() {}

    /// Callers must pass the app-wide `JobRunner` / `JobStore` so planner work and local UI
    /// jobs share one busy gate. Stop always cancels the session runner bound at start.
    ///
    /// With a claimed job in flight, stopping cancels the runner but leaves the loop task
    /// alive to submit a retriable failure (or skip submit on lease loss). Idle sleeps are
    /// cancelled immediately.
    func setEnabled(
        _ on: Bool,
        ws: PipetteWorkspace,
        jobRunner: JobRunner,
        jobStore: JobStore,
        background: Bool = false,
        options: WorkerArgs = WorkerArgs()
    ) {
        if on {
            stopReason = .user
            guard ws.registration.isRegistered else {
                enabled = false
                statusText = "Needs registration"
                sessionOutcome = .notStarted
                AppLog.jobRun.error("planner worker: not registered")
                return
            }
            enabled = true
            // A running session keeps its knobs: the loop copied `sessionOptions` once.
            guard loopTask == nil else {
                if options != sessionOptions {
                    AppLog.jobRun.info(
                        "planner worker: a session is already running; its knobs stand")
                }
                return
            }
            sessionStorage = ws
            sessionRunner = jobRunner
            sessionStore = jobStore
            sessionOptions = options
            statusText = "Starting…"
            // So a watcher cannot read the previous session's ending as this one's.
            sessionOutcome = nil
            sessionGeneration &+= 1
            let generation = sessionGeneration
            loopTask = Task { [weak self] in
                await self?.runUntilStopped(generation: generation)
            }
            return
        }

        // Stop through the session runner, which owns any in-flight claim.
        stopReason = background ? .background : .user
        enabled = false
        let runner = sessionRunner ?? jobRunner
        if runner.isRunning {
            runner.cancel(reason: background ? .background : .user)
            statusText = background ? "Stopping (background)…" : "Stopping…"
            AppLog.jobRun.info(
                "planner worker: stop requested (\(background ? "background" : "user")); draining in-flight job"
            )
            // Do not cancel `loopTask`: that would abort the failure submit.
        } else {
            loopTask?.cancel()
            loopTask = nil
            clearSession()
            statusText = "Off"
            sessionOutcome = .stopped
            AppLog.jobRun.info("planner worker: stopped")
        }
    }

    private func clearSession() {
        sessionStorage = nil
        sessionRunner = nil
        sessionStore = nil
        sessionOptions = WorkerArgs()
    }

    // MARK: - Loop

    private func runUntilStopped(generation: UInt64) async {
        /// A stale task writes nothing.
        func owns() -> Bool { generation == sessionGeneration }
        func finish(_ outcome: SessionOutcome) {
            guard owns() else { return }
            sessionOutcome = outcome
        }
        // Above the guard, so its early return also clears `loopTask`; a stale one would
        // stop `setEnabled` from ever starting the loop again.
        defer {
            if owns() {
                loopTask = nil
                clearSession()
                if !enabled || Task.isCancelled {
                    statusText = "Off"
                }
            }
        }
        guard let ws = sessionStorage,
              let jobRunner = sessionRunner,
              let jobStore = sessionStore
        else {
            statusText = "Off"
            finish(.notStarted)
            return
        }
        // Read once, so every call this session makes signs as the client it started as.
        guard let session = try? ws.registration.mgmtSession() else {
            statusText = "Needs registration"
            enabled = false
            finish(.notStarted)
            return
        }

        let options = sessionOptions
        AppLog.jobRun.info(
            "planner worker: starting (client_id=\(session.auth.clientId), "
                + "server=\(session.serverUrl.value), idle=\(Int(options.idleSecs))s"
                + "+0..\(Int(options.idleJitterSecs))s jitter, heartbeat="
                + "\(options.heartbeatSecs.map { "\(Int($0))s" } ?? "half time_window"), "
                + "max_jobs=\(options.maxJobs == 0 ? "unlimited" : String(options.maxJobs)))"
        )

        if options.skipProfileRefresh {
            AppLog.jobRun.warning(
                "planner worker: skipping startup profile refresh; matching may be stale "
                    + "until the next PATCH")
        } else {
            statusText = "Refreshing profile…"
            let profile: ClientProfile?
            do {
                profile = try await refreshProfileAtStartup(
                    registration: ws.registration, deviceStore: ws.device,
                    capabilities: installedRuntimeCapabilities() + installedRuntimeBuildFlags())
            } catch {
                if Task.isCancelled {
                    finish(.stopped)
                    return
                }
                // Not retried: a client that cannot report its profile would be matched
                // against a stale one.
                statusText = "Profile refresh failed"
                AppLog.jobRun.error(
                    "planner worker: profile refresh failed: \(error.localizedDescription)")
                enabled = false
                finish(.failed("profile refresh failed: \(error.localizedDescription)"))
                return
            }
            guard let profile else {
                statusText = "Needs registration"
                enabled = false
                finish(.notStarted)
                return
            }
            if profile.status != .approved {
                stopNotApproved(
                    "planner worker: client not approved (status=\(profile.status)); an "
                        + "operator must run `pipette-mgmt clients approve "
                        + "\(profile.clientId)`, then restart this client")
                finish(.notApproved)
                return
            }
        }

        var idleRounds = 0
        // Every job `runClaimedJob` returns from counts (a refused claim, a lost lease, a
        // dropped submit): each was leased and answered. A `fatal` submit throws and does not.
        var completedJobs: UInt = 0
        AppLog.jobRun.info("planner worker: entering claim loop")
        while !Task.isCancelled && enabled {
            statusText = idleRounds == 0 ? "Claiming…" : "Idle (round \(idleRounds))"
            AppLog.jobRun.debug(
                "planner worker: polling claim "
                    + "(completed=\(completedJobs), idle_rounds=\(idleRounds))"
            )
            let poll = await Self.claimWithRetry(
                server: session.serverUrl, auth: session.auth
            ) { [weak self] wait in
                self?.statusText = "Claim error, retry \(Int(wait))s"
                await Self.sleepInterruptible(wait)
            }
            switch poll {
            case .job(let job):
                // The job's `device_time_ms` counts from here.
                let claimedAt = ContinuousClock.now
                idleRounds = 0
                statusText = "Running \(job.benchmarkId)"
                logClaimedJob(job)
                do {
                    try await runClaimedJob(
                        job,
                        claimedAt: claimedAt,
                        session: session,
                        ws: ws,
                        jobRunner: jobRunner,
                        jobStore: jobStore,
                        options: options
                    )
                } catch {
                    AppLog.jobRun.error(
                        "planner worker: job \(job.jobId) ended with error "
                            + "(continuing claim loop): \(error.localizedDescription)")
                    continue
                }
                completedJobs += 1
                AppLog.jobRun.info(
                    "planner worker: job \(job.jobId) done (session completed=\(completedJobs)"
                        + "\(options.maxJobs > 0 ? ", max_jobs=\(options.maxJobs)" : ""))")
                if options.maxJobs > 0, completedJobs >= options.maxJobs {
                    AppLog.jobRun.info(
                        "planner worker: reached max-jobs=\(options.maxJobs); leaving claim loop")
                    enabled = false
                    finish(.completed(jobs: completedJobs))
                    return
                }
            case .idle:
                idleRounds += 1
                let wait = options.idleWaitWithJitter()
                statusText = "No work, sleep \(Int(wait))s"
                AppLog.jobRun.info(
                    "planner worker: no work available (204); sleeping \(Int(wait))s "
                        + "(base=\(Int(options.idleSecs))s, "
                        + "jitter=0..\(Int(options.idleJitterSecs))s, idle_round=\(idleRounds))"
                )
                do {
                    try await Task.sleep(for: .seconds(wait))
                } catch {
                    finish(.stopped)
                    return
                }
            case .refused(let error):
                // The server's own text: the refusals look alike but need different fixes.
                stopNotApproved(
                    "planner worker: claim returned 403: client \(session.auth.clientId) "
                        + "refused (\(error.localizedDescription)); stopping")
                finish(.notApproved)
                return
            case .transient(let error):
                // `claimWithRetry` hands a transient back only when the session is stopping;
                // anything else is a defensive arm.
                if Task.isCancelled { break }
                statusText = "Claim failed"
                AppLog.jobRun.error(
                    "planner worker: unexpected unhandled claim transient: "
                        + "\(error.localizedDescription)")
                enabled = false
                finish(.failed("unexpected claim transient: \(error.localizedDescription)"))
                return
            case .fatal(let error):
                statusText = "Claim failed"
                AppLog.jobRun.error(
                    "planner worker: fatal claim error: \(error.localizedDescription)")
                enabled = false
                finish(.failed("fatal claim error: \(error.localizedDescription)"))
                return
            }
        }
        finish(.stopped)
    }

    private func stopNotApproved(_ message: String) {
        statusText = "Not approved"
        AppLog.jobRun.error("\(message)")
        enabled = false
    }

    private func logClaimedJob(_ job: ClaimedJob) {
        AppLog.jobRun.info(
            "planner worker: claimed \(job.jobId) (benchmark=\(job.benchmarkId), "
                + "time_window=\(job.timeWindow), "
                + "expires_at=\(job.expiresAt?.value ?? "-"))"
        )
        // Token stripped. Debug-level: a cell is large and most runs never need it.
        AppLog.jobRun.debug(
            "planner worker: \(job.jobId) spec=\(job.spec?.redactedDescription ?? "<absent>")"
        )
    }

    /// A second, doubling to a minute.
    static let backoffFirst: TimeInterval = 1
    static let backoffCap: TimeInterval = 60

    /// Retries transients forever: a worker is a daemon, and a server down now will be up
    /// later. The only way out of a streak is the session stopping: a cancelled task hands
    /// the transient back. `sleep` is injected so tests need not wait.
    static func claimWithRetry(
        server: ServerUrl,
        http: HttpClient = .shared,
        auth: ClientIdentity,
        sleep: (TimeInterval) async -> Void
    ) async -> ClaimPoll {
        var attempt = 0
        while true {
            attempt += 1
            AppLog.jobRun.debug("planner worker: claim attempt \(attempt)")
            let poll = await pollClaim(server: server, http: http, auth: auth)
            guard case .transient(let error) = poll, !Task.isCancelled else { return poll }
            let backoff = exponentialBackoff(
                attempt: attempt, first: backoffFirst, cap: backoffCap)
            AppLog.jobRun.warning(
                "planner worker: claim attempt \(attempt) transient failure "
                    + "(\(error.localizedDescription)); retrying in \(Int(backoff))s"
            )
            await sleep(backoff)
        }
    }

    /// A backoff wait that a stop cuts short rather than turns into an error.
    static func sleepInterruptible(_ seconds: TimeInterval) async {
        try? await Task.sleep(for: .seconds(seconds))
    }

    // MARK: - One job

    /// Heartbeat, run, submit. Throws only when a submit is refused outright. Every refusal is
    /// a failure submission, so the server learns of each job taken; a lost lease submits nothing.
    private func runClaimedJob(
        _ job: ClaimedJob,
        claimedAt: ContinuousClock.Instant,
        session: MgmtSession,
        ws: PipetteWorkspace,
        jobRunner: JobRunner,
        jobStore: JobStore,
        options: WorkerArgs
    ) async throws {
        let protocolHeartbeat = resolveHeartbeatInterval(timeWindow: job.timeWindow)
        let hbEvery = resolveHeartbeatInterval(
            timeWindow: job.timeWindow, overrideSecs: options.heartbeatSecs)
        if options.heartbeatSecs != nil, hbEvery > protocolHeartbeat {
            AppLog.jobRun.warning(
                "planner worker: \(job.jobId): heartbeat-secs=\(Int(hbEvery))s is longer than "
                    + "half time_window (\(job.timeWindow) = \(Int(protocolHeartbeat))s); lease "
                    + "may expire between heartbeats"
            )
        }
        AppLog.jobRun.info(
            "planner worker: \(job.jobId): starting run (heartbeat every \(hbEvery)s"
                + "\(options.heartbeatSecs != nil ? ", from heartbeat-secs" : ", half time_window")"
                + "; time_window=\(job.timeWindow))"
        )
        let leaseLost = LeaseLostFlag()
        let heartbeat = Self.spawnHeartbeat(
            jobId: job.jobId, every: hbEvery, session: session, leaseLost: leaseLost)
        defer { heartbeat.cancel() }

        let runResult = await executeJob(
            job, claimedAt: claimedAt, session: session, ws: ws, jobRunner: jobRunner,
            jobStore: jobStore)
        // The clock stops before the submit. A success's `device_time_ms` is stamped earlier
        // by the executor, so this span also counts the filing.
        let elapsed = ContinuousClock.now - claimedAt

        heartbeat.cancel()
        _ = await heartbeat.value

        try await submitFinishedRun(
            job,
            runResult: runResult,
            leaseLost: leaseLost.isLost,
            elapsed: elapsed,
            session: session,
            ws: ws,
            jobStore: jobStore
        )
    }

    /// Split from ``runClaimedJob`` so the rule that a run whose lease was lost reports
    /// nothing at all is testable without executing a benchmark.
    func submitFinishedRun(
        _ job: ClaimedJob,
        runResult: Result<CompletedRun, any Error>,
        leaseLost: Bool,
        elapsed: Duration,
        session: MgmtSession,
        http: HttpClient = .shared,
        ws: PipetteWorkspace,
        jobStore: JobStore
    ) async throws {
        if leaseLost {
            AppLog.jobRun.warning(
                "planner worker: \(job.jobId): lease was lost during the run "
                    + "(\(Self.secondsText(elapsed))s elapsed); skipping submit "
                    + "(result would be rejected)"
            )
            statusText = "Lease lost, skipped submit"
            return
        }

        switch runResult {
        case .success(let run):
            AppLog.jobRun.info(
                "planner worker: \(job.jobId): run succeeded in \(Self.secondsText(elapsed))s")
            try await submitSuccess(
                job, run: run, session: session, http: http, ws: ws, jobStore: jobStore)
        case .failure(let error):
            let retriable = classifyRunError(error)
            let reason = formatFailureReason(Self.failureDetail(error))
            AppLog.jobRun.warning(
                "planner worker: \(job.jobId): run failed after \(Self.secondsText(elapsed))s "
                    + "(retriable=\(retriable)): \(reason)"
            )
            try await Self.submitFailure(
                job,
                reason: reason,
                retriable: retriable,
                deviceTime: elapsed,
                server: session.serverUrl,
                http: http,
                auth: session.auth,
                clientVersion: Bundle.main.appVersionDisplayString,
                sleep: Self.sleepInterruptible
            )
            statusText = retriable ? "Reported retriable failure" : "Reported terminal failure"
        }
    }

    /// Every failure comes back as `.failure`, so the caller reports it against the lease
    /// in one place.
    private func executeJob(
        _ job: ClaimedJob,
        claimedAt: ContinuousClock.Instant,
        session: MgmtSession,
        ws: PipetteWorkspace,
        jobRunner: JobRunner,
        jobStore: JobStore
    ) async -> Result<CompletedRun, any Error> {
        do {
            // The only place the raw `spec` is read.
            let claimed = try runSpecFromClaim(job)
            // Take custody before anything is written down: `Model.encode` strips the
            // token, so this is the last point it exists outside the Keychain.
            stashClaimCredential(claimed.model)
            // Coherence and flag legality only: missing weights are fetched by the run
            // (`RunCell.prepare`), not a precondition.
            let spec = try ClientRunSpec.validated(claimed)
            AppLog.jobRun.info(
                "planner worker: \(job.jobId): cell model=`\(spec.model.description)`, "
                    + "runtime=`\(RuntimeType.of(spec.runtime).rawValue)`")
            logIfLlamaPinDiffers(spec.runtime)

            let benchmark = try await Self.ensureClaimBenchmarkCached(
                store: ws.benchmarks, server: session.serverUrl, auth: session.auth,
                benchmarkId: job.benchmarkId)
            AppLog.jobRun.info(
                "planner worker: \(job.jobId): benchmark ready (id=\(job.benchmarkId), "
                    + "type=\(benchmark.benchmarkType.rawValue))")

            // A stop during the server read in `ensureClaimBenchmarkCached` above cancels this
            // task while the runner is idle; without this check the whole cell, download
            // included, would run in a session already switched off. Retriable, so the job is
            // requeued.
            if Task.isCancelled {
                throw WorkerResolveError.deviceUnavailable("worker stopped before the run began")
            }
            // Same JobRunner as UI jobs, for mutual exclusion.
            guard !jobRunner.isRunning else {
                throw WorkerResolveError.deviceUnavailable("device busy with a local job")
            }

            let manifest = makeManifest(job: job, spec: spec, benchmark: benchmark)
            jobStore.save(manifest)
            ws.saveJobManifest(manifest)

            let flag = CancelFlag()
            guard jobRunner.start(
                jobId: manifest.jobId,
                flag: flag,
                completedAtStart: 0,
                totalToRun: manifest.cells.count
            ) else {
                throw WorkerResolveError.deviceUnavailable("could not start job runner")
            }

            let kept = KeptEvalCheckpoint()
            let keptFetch = KeptFetchFailure()
            await withCheckedContinuation { (cont: CheckedContinuation<Void, Never>) in
                JobExecutor.run(
                    manifest: manifest,
                    jobRunner: jobRunner,
                    jobStore: jobStore,
                    flag: flag,
                    ws: ws,
                    source: .planner,
                    autoSubmit: false,
                    claimedAt: claimedAt,
                    claimedBenchmarks: [benchmark],
                    keepEvalCheckpoint: kept,
                    keepFetchFailure: keptFetch
                ) { _ in
                    cont.resume()
                }
            }

            let finished = ws.loadJobManifest(jobId: manifest.jobId)
            let cell = finished?.cells.first
            // Cancelled, but the lease is still held: fail retriably so the job is requeued.
            if flag.isCancelled
                || cell?.runStatus == .cancelled
                || finished?.status == .paused
                || finished?.status == .cancelled {
                let why: String =
                    switch flag.reason {
                    case .background: "app backgrounded mid-run"
                    case .user, nil: "run cancelled"
                    }
                throw WorkerResolveError.deviceUnavailable(why)
            }

            guard let finished, let cell,
                  cell.runStatus == .completed,
                  let payloadURL = ws.results.submittableDir(cell.cellId)?
                    .appendingPathComponent("payload.json"),
                  let payloadData = try? Data(contentsOf: payloadURL),
                  let payload = try? JSONSerialization.jsonObject(with: payloadData)
                    as? [String: Any]
            else {
                // Typed, so `classifyRunError` can retire a job no device can fetch.
                if let fetchFailure = keptFetch.error { throw fetchFailure }
                throw WorkerResolveError.deviceUnavailable(
                    cell?.errorMessage ?? "run produced no payload")
            }
            return .success(CompletedRun(manifest: finished, cellId: cell.cellId, payload: payload,
                                         evalCheckpoint: kept.digest))
        } catch {
            return .failure(error)
        }
    }

    /// The executor runs what this returns, not the synced list's row, which can be older.
    ///
    /// Signed with the claiming identity, so a benchmark private to its organization is not
    /// a `404` that retires the claim.
    ///
    /// Only an id the server does not know, or a body this build cannot read, is terminal.
    /// A fetch that fails in transit says nothing about the benchmark: a cached copy runs
    /// unchecked, else the claim fails retriably.
    static func ensureClaimBenchmarkCached(
        store: BenchmarkStore,
        server: ServerUrl,
        http: HttpClient = .shared,
        auth: ClientIdentity,
        benchmarkId: String
    ) async throws -> BenchmarkDefinition {
        do {
            return try await ensureRemoteBenchmarkCached(
                store: store, server: server, http: http, auth: auth, benchmarkId: benchmarkId)
        } catch let unknown as UnknownRemoteBenchmarkError {
            throw WorkerResolveError.unknownBenchmark(unknown.message)
        }
    }

    // MARK: - Heartbeat

    /// The wait after a transient heartbeat: the lease still has half a window of slack,
    /// so a tick lost in transit is retried soon rather than waited out.
    static let retryWait: TimeInterval = 2

    /// Cancelling the returned task stops it. Static so a test drives it against a stub
    /// server without a worker session.
    static func spawnHeartbeat(
        jobId: String,
        every: TimeInterval,
        session: MgmtSession,
        http: HttpClient = .shared,
        leaseLost: LeaseLostFlag
    ) -> Task<Void, Never> {
        Task {
            // First tick after `every`: the claim just granted a fresh window.
            AppLog.jobRun.info(
                "planner worker: \(jobId): heartbeat loop started (first tick in \(Int(every))s)")
            var wait = every
            var ticks = 0
            while !Task.isCancelled {
                do {
                    try await Task.sleep(for: .seconds(wait))
                } catch {
                    return
                }
                if Task.isCancelled { return }
                AppLog.jobRun.debug("planner worker: \(jobId): sending heartbeat (tick \(ticks + 1))")
                switch await keepaliveLease(
                    server: session.serverUrl, http: http, auth: session.auth, jobId: jobId
                ) {
                case .ok:
                    ticks += 1
                    wait = every
                    AppLog.jobRun.info(
                        "planner worker: \(jobId): heartbeat ok (tick=\(ticks), next in \(Int(every))s)"
                    )
                case .abort(let error):
                    leaseLost.mark()
                    AppLog.jobRun.error(
                        "planner worker: \(jobId): heartbeat aborted lease "
                            + "(\(error.localizedDescription)); flagging lease_lost — the run "
                            + "continues but submit is skipped"
                    )
                    return
                case .retry(let error):
                    wait = Self.retryWait
                    AppLog.jobRun.warning(
                        "planner worker: \(jobId): heartbeat transient "
                            + "(\(error.localizedDescription)); retrying in \(Int(wait))s"
                    )
                }
            }
        }
    }

    // MARK: - Failure classification

    /// Why a valid claim could not become a local run. Retriable or terminal is decided in
    /// one place, ``classifyRunError(_:)``.
    nonisolated enum WorkerResolveError: Error, LocalizedError {
        /// The server does not know the benchmark (even one cached here), or its body is not
        /// a definition this build can read.
        case unknownBenchmark(String)
        /// The device is busy, the runner would not start, or the run produced nothing:
        /// all transient and local.
        case deviceUnavailable(String)

        var message: String {
            switch self {
            case .unknownBenchmark(let m), .deviceUnavailable(let m):
                return m
            }
        }

        var errorDescription: String? { message }
    }

    private static func failureDetail(_ error: Error) -> String {
        (error as? WorkerResolveError)?.message ?? error.localizedDescription
    }

    /// A soft pin: the run continues, because the result's descriptor is regenerated from
    /// local build constants. A claim may name the tag or the commit at any abbreviation.
    private func logIfLlamaPinDiffers(_ runtime: Runtime) {
        guard case let .llamacppIosPipette(source, _, _) = runtime else { return }
        let version = source.repositoryVersion.value
        let build = LlamaPins.current.build
        let local = build.commit
        guard !version.isEmpty, !local.isEmpty, build.tag != version,
              !version.hasPrefix(local), !local.hasPrefix(version) else { return }
        AppLog.jobRun.info(
            "planner worker: claim llama version=\(version) local=\(local) (continuing)"
        )
    }

    // MARK: - Manifest

    /// `Model`'s encoding excludes the token so it cannot reach the manifest or the
    /// downloads cache, so the Keychain is where a relaunch can still find it.
    private func stashClaimCredential(_ model: Model) {
        guard let token = model.repo?.authToken else { return }
        _ = KeychainHelper.saveHfToken(token, forModel: model)
    }

    /// Knobs come from the claim's `ClientRunSpec`, not UI defaults. Internal for
    /// `PlannerWorkerTests`; reads no instance state.
    func makeManifest(
        job: ClaimedJob,
        spec: ClientRunSpec,
        benchmark: BenchmarkDefinition
    ) -> JobManifest {
        // The cell travels whole, so a knob added to `ClientRunSpec` reaches the run
        // untouched. Its in-memory `auth_token` is what `DownloadCoordinator` prefers.

        // The manifest's load settings are a display copy, read out of the cell so the two
        // cannot disagree, defaulted as the New Job screen would.
        let flags = spec.runtimeFlags
        let nGpuLayers = flags?.numberGpuLayers.map(Int.init) ?? 99
        let contextSize = flags?.ctxSize.map { max(1, Int($0)) }
            ?? Int(LlamaRuntimeFlags.ctxSize(for: benchmark))
        let prefillBatch = flags?.nUbatch.map { max(1, Int($0)) }
        let threads = flags?.threads.map { max(1, Int($0)) }

        let cell = JobCell(
            cellId: CellId(UUID().uuidString),
            benchmarkId: job.benchmarkId,
            benchmarkType: benchmark.benchmarkType,
            runStatus: .pending,
            serverJobId: nil,
            errorMessage: nil,
            spec: spec
        )

        AppLog.jobRun.info(
            "planner worker: materialize job ngl=\(nGpuLayers) ctx=\(contextSize) "
                + "batch=\(String(describing: prefillBatch)) "
                + "threads=\(String(describing: threads)) "
                + "runtime=\(RuntimeType.of(spec.runtime).rawValue)"
        )

        return JobManifest(
            jobId: JobId("plan-\(job.jobId)"),
            createdAt: JobDateFormat.iso8601.string(from: Date()),
            schemaVersion: JobManifestSchema.currentVersion,
            nGpuLayers: nGpuLayers,
            contextSize: contextSize,
            prefillBatch: prefillBatch,
            threads: threads,
            cells: [cell],
            status: .running,
            contributeResults: false, // plan path submits itself
            title: "Planner \u{b7} \(job.benchmarkId)",
            pausedReason: nil
        )
    }

    // MARK: - Submit

    /// A refused submit leaves the eval checkpoint for the job's next run, and an
    /// interrupted one leaves it beside the pending result.
    private func submitSuccess(
        _ job: ClaimedJob,
        run: CompletedRun,
        session: MgmtSession,
        http: HttpClient = .shared,
        ws: PipetteWorkspace,
        jobStore: JobStore
    ) async throws {
        AppLog.jobRun.info(
            "planner worker: \(job.jobId): posting the success body to the server (device_time_ms="
                + "\(run.payload["device_time_ms"].map { "\($0)" } ?? "none"))")
        let body = try JSONSerialization.data(
            withJSONObject: attachClaimToSuccessPayload(run.payload, job: job))
        let disposition = try await Self.postCompletion(
            server: session.serverUrl,
            http: http,
            auth: session.auth,
            jobId: job.jobId,
            payload: body,
            sleep: Self.sleepInterruptible
        )
        let serverJobId: String
        switch disposition {
        case .accepted(let accepted):
            serverJobId = accepted
            statusText = "Submitted \(job.jobId)"
        case .dropped:
            serverJobId = job.jobId
            statusText = "Submit dropped (superseded)"
        case .interrupted:
            // Nothing was decided: no record and no `serverJobId`, so the result stays in
            // `remote/pending` rather than filed as submitted. No sweep sends it (it would
            // lack `job_id`); the lease running out answers the claim.
            statusText = "Submit interrupted (result kept pending)"
            return
        case .retryTransient, .fatal:
            // `postCompletion` throws on `fatal`, so this is exhaustion.
            serverJobId = job.jobId
            statusText = "Submit gave up (result dropped)"
        }
        Self.storeCompletedResult(
            results: ws.results,
            jobId: job.jobId,
            cellId: run.cellId,
            serverJobId: serverJobId,
            collector: session.serverUrl
        )
        RunCell.finalizeEvalCheckpoint(ws: ws, digest: run.evalCheckpoint)
        // The sweep skips a cell with a `serverJobId` before reading anything else.
        var manifest = run.manifest
        if let index = manifest.cells.firstIndex(where: { $0.cellId == run.cellId }) {
            manifest.cells[index].serverJobId = serverJobId
            ws.saveJobManifest(manifest)
            jobStore.save(manifest)
        }
    }

    /// File the result as synced so no sweep posts it again through the generic endpoint.
    ///
    /// After the submit, whatever it answered short of `fatal`: filing first would call it
    /// submitted too early, and leaving a dropped submit pending would let the next sweep
    /// post it again. Best-effort: the job is already settled server-side.
    static func storeCompletedResult(
        results: ResultsStore,
        jobId: String,
        cellId: CellId,
        serverJobId: String,
        collector: ServerUrl
    ) {
        do {
            try results.saveSubmission(
                .submitted(serverJobId: serverJobId, collector: collector), cellId)
            try results.move(cellId, to: .remoteSynced)
            AppLog.jobRun.info(
                "planner worker: \(jobId): result stored under \(results.locationDir(.remoteSynced).path)")
        } catch {
            AppLog.jobRun.warning(
                "planner worker: \(jobId): submitted but not stored locally: "
                    + "\(error.localizedDescription)")
        }
    }

    /// Through the same retrying POST a success takes.
    @discardableResult
    static func submitFailure(
        _ job: ClaimedJob,
        reason: String,
        retriable: Bool,
        deviceTime: Duration,
        server: ServerUrl,
        http: HttpClient = .shared,
        auth: ClientIdentity,
        clientVersion: String,
        sleep: (TimeInterval) async -> Void
    ) async throws -> SubmitDisposition {
        let body = failureFromClaim(
            job, reason: reason, retriable: retriable, clientVersion: clientVersion,
            deviceTime: deviceTime)
        AppLog.jobRun.info(
            "planner worker: \(job.jobId): posting the failure body to the server "
                + "(retriable=\(retriable), "
                + "device_time_ms=\(body.deviceTimeMs.map(String.init) ?? "none"))")
        return try await postCompletion(
            server: server,
            http: http,
            auth: auth,
            jobId: job.jobId,
            payload: try Coding.encoder.encode(body),
            sleep: sleep
        )
    }

    /// Throws on `fatal`: a dropped submit and an exhausted backoff are answers the loop
    /// logs and moves past, a definitive refusal is not.
    @discardableResult
    static func postCompletion(
        server: ServerUrl,
        http: HttpClient = .shared,
        auth: ClientIdentity,
        jobId: String,
        payload: Data,
        sleep: (TimeInterval) async -> Void
    ) async throws -> SubmitDisposition {
        AppLog.jobRun.info("planner worker: \(jobId): POST /benchmarks (plan-attached)")
        let disposition = await submitPlanResultWithBackoff(
            server: server, http: http, auth: auth, jobId: jobId, payload: payload
        ) { wait in
            AppLog.jobRun.warning(
                "planner worker: \(jobId): submit transient; backing off \(Int(wait))s before retry")
            await sleep(wait)
        }
        switch disposition {
        case .accepted(let accepted):
            AppLog.jobRun.info(
                "planner worker: \(jobId): submit accepted (server job_id=\(accepted))")
        case .dropped:
            AppLog.jobRun.warning(
                "planner worker: \(jobId): submit dropped (409 superseded, or reclaim after "
                    + "404 failed)")
        case .retryTransient:
            AppLog.jobRun.warning(
                "planner worker: \(jobId): gave up submitting after repeated transient "
                    + "failures; result dropped")
        case .interrupted:
            AppLog.jobRun.warning(
                "planner worker: \(jobId): submit interrupted by a stop; leaving the result "
                    + "pending for the lease to run out")
        case .fatal(let error):
            AppLog.jobRun.error(
                "planner worker: \(jobId): submit fatal error: \(error.localizedDescription)")
            throw error
        }
        return disposition
    }

    /// The crate's `{:.1}s`.
    private static func secondsText(_ d: Duration) -> String {
        let seconds = Double(d.components.seconds) + Double(d.components.attoseconds) / 1e18
        return String(format: "%.1f", seconds)
    }
}
