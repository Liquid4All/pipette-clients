import Foundation
import UIKit

/// Long-running planner claim loop for the iOS app (client-integration.md).
///
/// Enabled via Settings → "Planner worker". When on, the app claims jobs from
/// the management server, runs them against locally installed models that
/// match the claim's `model_descriptor`, heartbeats while running, and submits
/// success/failure with the claim echo.
///
/// ## How a claim becomes a run
/// The job is the plan-types cell: `runtime_descriptor` names the iOS `Runtime`
/// variant (`llamacpp_ios_pipette` / `mlx_ios_pipette` / `apple_foundation`);
/// `model_descriptor` is the `Model`; opaque `runtime_flags` / `model_flags`
/// supply load/generation knobs (`ClientRunSpec.validated`). The worker does **not**
/// invent a runtime — it only materializes what the claim already specified.
///
/// ## v1 limits
/// - Desktop runtime descriptors fail non-retriably.
/// - Missing local model weights → retriable failure.
/// - No mid-run cancel on lease loss; submit is skipped after the run.
/// - Claim `runtime_flags` are plan-types iOS cells and/or opaque knobs
///   (`ClientRunSpec.validated`); doomloop is not wired; eval samples are local-catalog only.
/// - Backgrounding cancels an in-flight cell and reports a retriable failure.
@MainActor
@Observable
final class PlannerWorker {
    static let shared = PlannerWorker()

    /// Human-readable status for Settings / debugging. `@Observable` so the
    /// Settings row updates as the loop progresses.
    private(set) var statusText: String = "Off"

    private var loopTask: Task<Void, Never>?
    /// When false, the loop exits after the current claim finishes (or wakes
    /// from idle sleep via task cancel if nothing is running).
    private var enabled = false
    /// Why the last stop was requested — shapes the retriable failure reason.
    private var stopReason: PlannerStopReason = .user

    /// Controllers for the active loop session. Bound when the loop starts and
    /// used for stop/cancel so Settings / headless / scenePhase cannot pass a
    /// different `JobRunner` than the one executing the claim.
    private var sessionStorage: Storage?
    private var sessionRunner: JobRunner?
    private var sessionStore: JobStore?

    private enum PlannerStopReason {
        case user
        case background
    }

    /// Lease-lost side channel for heartbeat → run completion (not job cancel).
    private final class LeaseLostFlag: @unchecked Sendable {
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

    private init() {}

    /// Start or stop the claim loop to match the Settings toggle / lifecycle.
    ///
    /// Callers must pass the **app-wide** `JobRunner` / `JobStore` so planner
    /// work and local UI jobs share one busy gate. Stop always cancels the
    /// session runner bound at start, not whatever instance the caller passes.
    ///
    /// Stopping is **graceful** when a claimed job is in flight: the runner is
    /// cancelled and the loop task is left alive long enough to submit a
    /// retriable failure (or skip submit on lease loss). Idle sleeps are
    /// cancelled immediately so the toggle feels snappy.
    func setEnabled(
        _ on: Bool,
        storage: Storage,
        jobRunner: JobRunner,
        jobStore: JobStore,
        background: Bool = false
    ) {
        if on {
            stopReason = .user
            guard storage.identity.isRegistered else {
                enabled = false
                LocalStorage.plannerWorkerEnabled = false
                statusText = "Needs registration"
                AppLog.jobRun.error("planner worker: not registered")
                return
            }
            enabled = true
            guard loopTask == nil else { return }
            sessionStorage = storage
            sessionRunner = jobRunner
            sessionStore = jobStore
            statusText = "Starting…"
            loopTask = Task { [weak self] in
                await self?.runLoop()
            }
            return
        }

        // Stop path — always the session runner that owns any in-flight claim.
        stopReason = background ? .background : .user
        enabled = false
        let runner = sessionRunner ?? jobRunner
        if runner.isRunning {
            runner.cancel(reason: background ? .background : .user)
            statusText = background ? "Stopping (background)…" : "Stopping…"
            AppLog.jobRun.info(
                "planner worker: stop requested (\(background ? "background" : "user")); draining in-flight job"
            )
            // Do **not** cancel `loopTask` here — that would abort failure submit.
        } else {
            loopTask?.cancel()
            loopTask = nil
            clearSession()
            statusText = "Off"
            AppLog.jobRun.info("planner worker: stopped")
        }
    }

    private func clearSession() {
        sessionStorage = nil
        sessionRunner = nil
        sessionStore = nil
    }

    // MARK: - Loop

    private func runLoop() async {
        guard let storage = sessionStorage,
              let jobRunner = sessionRunner,
              let jobStore = sessionStore
        else {
            statusText = "Off"
            return
        }
        AppLog.jobRun.info("planner worker: entering claim loop")
        defer {
            loopTask = nil
            clearSession()
            if !enabled || Task.isCancelled {
                statusText = "Off"
            }
        }

        do {
            statusText = "Refreshing profile…"
            try await refreshProfile(storage: storage)
        } catch {
            statusText = "Profile refresh failed"
            AppLog.jobRun.error("planner worker: profile refresh failed: \(error.localizedDescription)")
            // Keep looping — claim will 403 if not approved; profile can retry later.
        }

        var idleRounds = 0
        while !Task.isCancelled && enabled {
            guard let session = try? storage.mgmtSession() else {
                statusText = "Needs registration"
                try? await Task.sleep(for: .seconds(30))
                continue
            }

            statusText = idleRounds == 0 ? "Claiming…" : "Idle (round \(idleRounds))"
            do {
                let job = try await ManagementClient.claim(
                    serverUrl: session.serverUrl,
                    auth: session.auth
                )
                if let job {
                    idleRounds = 0
                    statusText = "Running \(job.benchmarkId)"
                    AppLog.jobRun.info(
                        "planner worker: claimed \(job.jobId) benchmark=\(job.benchmarkId) window=\(job.timeWindow)"
                    )
                    // The payload an operator needs when a cell is refused, with
                    // any plan-supplied token stripped. Debug-level for the same
                    // reason the Rust worker logs it there: a cell is large and
                    // most runs never need it.
                    AppLog.jobRun.debug(
                        "planner worker: \(job.jobId) spec=\(job.spec?.redactedDescription ?? "<absent>")"
                    )
                    await runClaimedJob(
                        job,
                        registration: session.registration,
                        auth: session.auth,
                        storage: storage,
                        jobRunner: jobRunner,
                        jobStore: jobStore
                    )
                } else {
                    idleRounds += 1
                    let wait = idleWaitSeconds()
                    statusText = "No work, sleep \(Int(wait))s"
                    AppLog.jobRun.info("planner worker: no work (204); sleep \(wait)s")
                    try await Task.sleep(for: .seconds(wait))
                }
            } catch let error as ManagementClientError {
                if case .httpStatus(let code, _) = error, code == 403 {
                    statusText = "Not approved"
                    AppLog.jobRun.error(
                        "planner worker: client not approved (403) — disable worker or approve device"
                    )
                    enabled = false
                    LocalStorage.plannerWorkerEnabled = false
                    return
                }
                if isTransient(error) {
                    let backoff = min(60.0, pow(2.0, Double(min(idleRounds, 5))))
                    statusText = "Claim error, retry \(Int(backoff))s"
                    AppLog.jobRun.error("planner worker: claim transient: \(error.localizedDescription)")
                    try? await Task.sleep(for: .seconds(backoff))
                    idleRounds += 1
                } else {
                    statusText = "Claim failed"
                    AppLog.jobRun.error("planner worker: claim fatal: \(error.localizedDescription)")
                    try? await Task.sleep(for: .seconds(30))
                }
            } catch {
                if Task.isCancelled { return }
                statusText = "Claim error"
                AppLog.jobRun.error("planner worker: claim error: \(error.localizedDescription)")
                try? await Task.sleep(for: .seconds(15))
            }
        }
    }

    // MARK: - One job

    private func runClaimedJob(
        _ job: ClaimedJob,
        registration: IdentityRegistration,
        auth: AuthIdentity,
        storage: Storage,
        jobRunner: JobRunner,
        jobStore: JobStore
    ) async {
        let leaseLost = LeaseLostFlag()
        let hbEvery = Iso8601Duration.heartbeatInterval(timeWindow: job.timeWindow)
        let heartbeat = Task {
            await heartbeatLoop(
                jobId: job.jobId,
                every: hbEvery,
                registration: registration,
                auth: auth,
                leaseLost: leaseLost
            )
        }
        defer { heartbeat.cancel() }

        let resolution: RunResolution
        do {
            // Type the payload once, here, and pass the typed cell down — the
            // only place the raw `spec` is read (`run_spec_from_claim`).
            let spec = try ClientRunSpec.runSpec(from: job)
            // Take custody before anything is written down: `Model.encode` strips the
            // token, so this is the last point it exists outside the Keychain.
            stashClaimCredential(spec.model)
            resolution = try resolveLocalRun(spec: spec, storage: storage)
        } catch {
            await submitFailure(
                job,
                error,
                registration: registration,
                auth: auth
            )
            return
        }

        // Same JobRunner as UI jobs — mutual exclusion.
        if jobRunner.isRunning {
            await submitFailure(
                job,
                WorkerResolveError.deviceUnavailable("device busy with a local job"),
                registration: registration,
                auth: auth
            )
            return
        }

        let manifest = makeManifest(job: job, resolution: resolution)
        guard !manifest.cells.isEmpty else {
            await submitFailure(
                job,
                WorkerResolveError.unknownBenchmark(
                    "benchmark \(job.benchmarkId) is not a known definition on this device"
                ),
                registration: registration,
                auth: auth
            )
            return
        }
        jobStore.save(manifest)
        storage.saveJobManifest(manifest)

        let flag = CancelFlag()
        guard jobRunner.start(
            jobId: manifest.jobId,
            flag: flag,
            completedAtStart: 0,
            totalToRun: manifest.cells.count
        ) else {
            await submitFailure(
                job,
                WorkerResolveError.deviceUnavailable("could not start job runner"),
                registration: registration,
                auth: auth
            )
            return
        }

        await withCheckedContinuation { (cont: CheckedContinuation<Void, Never>) in
            JobExecutor.run(
                manifest: manifest,
                jobRunner: jobRunner,
                jobStore: jobStore,
                flag: flag,
                storage: storage,
                source: .planner,
                autoSubmit: false
            ) { _ in
                cont.resume()
            }
        }

        heartbeat.cancel()
        _ = await heartbeat.result

        if leaseLost.isLost {
            AppLog.jobRun.error(
                "planner worker: lease lost during \(job.jobId); skipping submit"
            )
            statusText = "Lease lost, skipped submit"
            return
        }

        let finished = storage.loadJobManifest(jobId: manifest.jobId)
        let cell = finished?.cells.first
        let cancelled =
            flag.isCancelled
            || cell?.runStatus == .cancelled
            || finished?.status == .paused
            || finished?.status == .cancelled

        // Background / user stop / runner cancel: still hold the lease → tell
        // the server this client failed retriably so the job returns to avail/.
        if cancelled {
            let why: String =
                switch flag.reason {
                case .background: "app backgrounded mid-run"
                case .user: "run cancelled"
                case nil: "run cancelled"
                }
            await submitFailure(
                job,
                WorkerResolveError.deviceUnavailable(why),
                registration: registration,
                auth: auth
            )
            return
        }

        // Load the cell payload and attach claim identity for plan submit.
        guard let cell,
              cell.runStatus == .completed,
              let payloadURL = storage.results.submittableDir(cell.cellId)?
                .appendingPathComponent("payload.json"),
              let payloadData = try? Data(contentsOf: payloadURL),
              var payloadObj = try? JSONSerialization.jsonObject(with: payloadData) as? [String: Any]
        else {
            let errMsg = cell?.errorMessage ?? "run produced no payload"
            await submitFailure(
                job,
                WorkerResolveError.deviceUnavailable(errMsg),
                registration: registration,
                auth: auth
            )
            return
        }

        attachClaimEcho(&payloadObj, job: job)
        do {
            let body = try JSONSerialization.data(withJSONObject: payloadObj)
            let outcome = try await submitWithRetry(
                jobId: job.jobId,
                body: body,
                registration: registration,
                auth: auth
            )
            switch outcome {
            case .accepted:
                statusText = "Submitted \(job.jobId)"
                AppLog.jobRun.info("planner worker: submitted \(job.jobId)")
            case .dropped:
                statusText = "Submit dropped (superseded)"
                AppLog.jobRun.info("planner worker: submit dropped for \(job.jobId)")
            }
        } catch {
            statusText = "Submit failed (will rely on lease expiry)"
            AppLog.jobRun.error(
                "planner worker: submit failed for \(job.jobId): \(error.localizedDescription)"
            )
        }
    }

    // MARK: - Heartbeat

    private func heartbeatLoop(
        jobId: String,
        every: TimeInterval,
        registration: IdentityRegistration,
        auth: AuthIdentity,
        leaseLost: LeaseLostFlag
    ) async {
        // First tick after `every` — claim just granted a fresh window.
        while !Task.isCancelled {
            do {
                try await Task.sleep(for: .seconds(every))
            } catch {
                return
            }
            if Task.isCancelled { return }
            do {
                try await ManagementClient.heartbeat(
                    serverUrl: registration.serverUrl,
                    auth: auth,
                    jobId: jobId
                )
                AppLog.jobRun.info("planner worker: heartbeat ok \(jobId)")
            } catch let error as ManagementClientError {
                if case .httpStatus(let code, _) = error {
                    if code == 409 {
                        leaseLost.mark()
                        AppLog.jobRun.error("planner worker: heartbeat 409 for \(jobId)")
                        return
                    }
                    if code == 404 {
                        do {
                            try await ManagementClient.reclaim(
                                serverUrl: registration.serverUrl,
                                auth: auth,
                                jobId: jobId
                            )
                            AppLog.jobRun.info("planner worker: reclaimed \(jobId)")
                            continue
                        } catch {
                            leaseLost.mark()
                            AppLog.jobRun.error(
                                "planner worker: reclaim failed for \(jobId): \(error.localizedDescription)"
                            )
                            return
                        }
                    }
                }
                AppLog.jobRun.error(
                    "planner worker: heartbeat error \(jobId): \(error.localizedDescription)"
                )
                try? await Task.sleep(for: .seconds(2))
            } catch {
                if Task.isCancelled { return }
                AppLog.jobRun.error(
                    "planner worker: heartbeat error \(jobId): \(error.localizedDescription)"
                )
                try? await Task.sleep(for: .seconds(2))
            }
        }
    }

    // MARK: - Profile

    /// Report the profile via `ProfileReporter`, then hold the loop until the
    /// client is actually clear to claim.
    ///
    /// The extra work over a plain refresh is the `reindex_pending` gate. This
    /// worker holds a lease and claims jobs, so a voided standing means every
    /// plan operation is refused until `queue-maintenance` re-evaluates the
    /// client — entering the claim loop before the gate lifts would just burn
    /// retries. Callers that only report and never claim (the launch refresh)
    /// have nothing in flight and skip this entirely.
    private func refreshProfile(storage: Storage) async throws {
        // `refresh` returning nil means no registration — nothing to report and
        // nothing to wait on. It owns that check, so this doesn't duplicate it.
        guard var profile = try await ProfileReporter.refresh(storage: storage) else { return }

        // Wait out reindex gate (at most ~5 minutes). Credentials are loaded here
        // rather than up front because polling is the only thing that needs them.
        if profile.reindexPending, let session = try? storage.mgmtSession() {
            var waited = 0
            while profile.reindexPending && waited < 300 && !Task.isCancelled {
                try await Task.sleep(for: .seconds(5))
                waited += 5
                profile = try await ManagementClient.me(
                    serverUrl: session.serverUrl,
                    auth: session.auth
                )
            }
        }
        if profile.status != "approved" {
            throw ManagementClientError.httpStatus(
                statusCode: 403,
                body: "client status=\(profile.status)"
            )
        }
    }

    // MARK: - Resolve claim → plan-types run + local model

    private struct RunResolution {
        /// The validated cell the claim carried — the crate's `ClientRunSpec`, checked
        /// the way `validate_spec_flags` + `is_compatible` check one.
        let spec: ClientRunSpec
        /// Local weights (or AFM) that satisfy `config` + `model_descriptor`.
        let model: DiscoveredModel
    }

    /// Why a claim could not be turned into a local run. Carries no disposition:
    /// retriable-vs-terminal is decided in one place, by
    /// [`PlannerWorker.retriable(_:)`].
    enum WorkerResolveError: Error {
        /// A model the cell names that no local model matches. Specific to this
        /// device — another may already have the weights.
        case noLocalModel(String)
        /// The benchmark id names no definition this build knows.
        case unknownBenchmark(String)
        /// The device is busy, the runner would not start, or the run produced
        /// nothing — all transient and local.
        case deviceUnavailable(String)

        var message: String {
            switch self {
            case .noLocalModel(let m), .unknownBenchmark(let m), .deviceUnavailable(let m):
                return m
            }
        }
    }

    /// Retriable vs terminal for a failure submission — the single decision
    /// point, mirroring the CLI's `classify_run_error`.
    ///
    /// Defaults to retriable, and the terminal list stays narrow on purpose:
    /// `client-integration.md` §6 is explicit that a wrongly-non-retriable job
    /// is discarded for the whole fleet on one device's say-so, where a
    /// wrongly-retriable one is bounded by its `expires_at`.
    static func retriable(_ error: Error) -> Bool {
        // A claim that could not be read is permanent by construction — no need
        // to recognize it by its wording.
        if let refusal = error as? UnrunnableClaim {
            return refusal.retriable
        }
        if case .unknownBenchmark = error as? WorkerResolveError {
            // The catalog is compiled in, so no client on this build reads it —
            // matching the CLI's permanent 404 from `GET /benchmarks/{id}`.
            return false
        }
        return true
    }

    private func resolveLocalRun(spec claimed: ClientRunSpec, storage: Storage) throws -> RunResolution {
        let spec = try ClientRunSpec.validated(claimed)

        // Soft pin: claim llama build may differ from this binary — still run.
        if case let .llamacppIosPipette(source, _, _) = spec.runtime {
            let version = source.repositoryVersion.value
            // A claim may name either level this build advertises — the tag or
            // the commit — and the commit at any abbreviation length.
            let build = LlamaCppBuildInfo.build
            let local = build.commit
            let matchesTag = build.tag == version
            if !version.isEmpty,
               !local.isEmpty,
               !matchesTag,
               !version.hasPrefix(local),
               !local.hasPrefix(version) {
                AppLog.jobRun.info(
                    "planner worker: claim llama version=\(version) local=\(local) (continuing)"
                )
            }
        }

        if case .appleFoundation = spec.runtime {
            return RunResolution(spec: spec, model: .appleFoundation)
        }

        // The cell's coherence was settled by `ClientRunSpec.validated`; what
        // remains is whether *this* device holds the weights, which is a
        // different question with a different disposition.
        let coord = ModelCoordinate(spec.model)
        let models = storage.loadDiscoveredModels()
        guard let match = models.first(where: { Self.matchesModel($0, coord: coord) }) else {
            let artifact = coord.path ?? coord.weights ?? coord.prefix ?? "?"
            throw WorkerResolveError.noLocalModel(
                "no local model matching \(coord.org)/\(coord.repo) \(artifact) "
                    + "for \(RuntimeType.of(spec.runtime).rawValue)"
            )
        }
        return RunResolution(spec: spec, model: match)
    }

    /// The repo coordinates a [`Model`] addresses, flattened for matching against the
    /// local inventory.
    ///
    /// `type` is kept alongside the coordinates because the fields alone do not identify
    /// the kind: an `mlx` model addresses a repo plus an optional subdirectory exactly as
    /// a `torch` one does on the Rust side, so matching pairs on the type first rather
    /// than inferring the kind from which fields are populated.
    struct ModelCoordinate {
        let type: ModelType
        let org: String
        let repo: String
        let path: String? // gguf_text
        let weights: String? // gguf_vision
        let mmproj: String? // gguf_vision
        let prefix: String? // mlx subdirectory

        init(_ model: Model) {
            type = ModelType.of(model)
            // Only a HuggingFace arm carries coordinates to match against the inventory.
            // Every other arm is refused earlier — a store form names already-installed
            // bytes and a `url` arm has no iOS fetch — so the empty coordinate here is
            // unreachable, not a silent match-anything.
            let axes: (org: String, repo: String, path: String?, weights: String?,
                       mmproj: String?, prefix: String?) = {
                switch model {
                case let .ggufText(m):
                    guard case let .huggingFace(hf, filePath, _) = m.source else { break }
                    return (hf.org.value, hf.repoName.value, filePath.value, nil, nil, nil)
                case let .ggufVision(m):
                    guard case let .huggingFace(hf, weights, _, projector, _) = m.source else { break }
                    return (hf.org.value, hf.repoName.value, nil, weights.value, projector.value, nil)
                case let .mlx(m):
                    guard case let .huggingFace(hf, subdir) = m.source else { break }
                    return (hf.org.value, hf.repoName.value, nil, nil, nil, subdir?.value)
                case .appleFoundationText:
                    break  // ships with the OS: nothing to match, nothing to fetch
                }
                return ("", "", nil, nil, nil, nil)
            }()
            (org, repo) = (axes.org, axes.repo)
            (path, weights, mmproj, prefix) = (axes.path, axes.weights, axes.mmproj, axes.prefix)
        }
    }

    /// Whether a model already on the device serves the cell's coordinate.
    ///
    /// Pairs the cell's model type against the discovered kind explicitly. The
    /// pairing is what keeps the match honest: a `torch` cell addresses a repo
    /// and subdirectory exactly as an `mlx` one does, so matching on the
    /// coordinate fields alone would let a torch cell claim a local MLX model.
    /// `ClientRunSpec.validated` refuses that cell before this runs, but this
    /// function should not depend on a guarantee made somewhere else.
    static func matchesModel(_ discovered: DiscoveredModel, coord: ModelCoordinate) -> Bool {
        switch (coord.type, discovered.source) {
        case (.appleFoundationText, .appleFoundationText):
            return true
        case (.ggufText, .ggufText(let m)):
            guard case let .huggingFace(repo, path, _) = m.source else { return false }
            return sameRepo(repo, coord) && coord.path == path.value
        case (.ggufVision, .ggufVision(let m)):
            // An unnamed projector accepts whichever one the repo shipped: the
            // plan may omit `mmproj`, and a VL repo carries a single projector.
            guard case let .huggingFace(repo, weights, _, projector, _) = m.source else { return false }
            let projectorOk = coord.mmproj.map { $0 == projector.value } ?? true
            return sameRepo(repo, coord) && coord.weights == weights.value && projectorOk
        case (.mlx, .mlx(let m)):
            guard case let .huggingFace(repo, prefix) = m.source else { return false }
            return sameRepo(repo, coord) && coord.prefix == prefix?.value
        default:
            return false
        }
    }

    private static func sameRepo(_ repo: HFRepo, _ coord: ModelCoordinate) -> Bool {
        repo.org.value == coord.org && repo.repoName.value == coord.repo
    }

    // MARK: - Manifest / submit helpers

    /// Build a local `JobManifest` from the claim's plan-types config — engine
    /// knobs and model flags come from the claim's `ClientRunSpec`, not UI defaults.
    /// Move a claim's `auth_token` into the Keychain, keyed by the repo it authenticates.
    ///
    /// Nothing else persists it: the token is excluded from `Model`'s encoding precisely
    /// so it cannot reach the batch manifest or the downloads cache, which means the
    /// credential has to be kept somewhere a relaunch can still find it.
    private func stashClaimCredential(_ model: Model) {
        guard let token = model.repo?.authToken else { return }
        _ = KeychainHelper.saveHfToken(token, forModel: model)
    }

    private func makeManifest(job: ClaimedJob, resolution: RunResolution) -> JobManifest {
        // The cell the claim asked for, with the model it matched on this device: the claim
        // names a coordinate, `resolveLocalRun` answered which local artifact satisfies it,
        // and the run has to load that one. Every other part of the cell is the claim's.
        let claimed = resolution.spec
        let spec = ClientRunSpec(
            benchmark: claimed.benchmark, model: resolution.model.source,
            runtime: claimed.runtime, runtimeFlags: claimed.runtimeFlags,
            modelFlags: claimed.modelFlags)

        // The manifest's load settings are the job's display copy — the cell carries the
        // entry that actually reaches the load. Read out of it so the two cannot disagree,
        // defaulted to what the New Job screen would have offered.
        let flags = spec.runtimeFlags
        let nGpuLayers = flags?.numberGpuLayers.map(Int.init) ?? 99
        let contextSize = flags?.ctxSize.map { max(1, Int($0)) }
            ?? BenchmarkDefinition(parsingId: job.benchmarkId)
                .map { Int(LlamaRuntimeFlags.contextSize(for: $0)) } ?? 0
        let prefillBatch = flags?.nUbatch.map { max(1, Int($0)) }
        let threads = flags?.threads.map { max(1, Int($0)) }

        let cells: [JobCell] = {
            guard let def = BenchmarkDefinition(parsingId: job.benchmarkId) else { return [] }
            return [
                JobCell(
                    cellId: CellId(UUID().uuidString),
                    benchmarkId: job.benchmarkId,
                    benchmarkType: def.type,
                    runStatus: .pending,
                    serverJobId: nil,
                    errorMessage: nil,
                    spec: spec
                ),
            ]
        }()

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
            cells: cells,
            status: .running,
            contributeResults: false, // plan path submits itself
            title: "Planner \u{b7} \(job.benchmarkId)",
            pausedReason: nil
        )
    }

    /// Bind the submission to the lease it came from.
    ///
    /// Only `job_id`. The payload's descriptors come from the run, built from
    /// the same cell the claim carried, so they already are the echo the
    /// server's claim-binding check looks for — and the server holds the job
    /// body for everything else it wants to compare.
    private func attachClaimEcho(_ payload: inout [String: Any], job: ClaimedJob) {
        payload["job_id"] = job.jobId
    }

    /// Report a failure for this lease. The single funnel: every failure path
    /// hands its error here and [`retriable(_:)`] decides the disposition, so
    /// that policy lives in one place rather than at each throw site.
    private func submitFailure(
        _ job: ClaimedJob,
        _ error: Error,
        registration: IdentityRegistration,
        auth: AuthIdentity
    ) async {
        let detail = (error as? WorkerResolveError)?.message ?? error.localizedDescription
        let reason = "[\(isoNow())] \(detail)"
        let retriable = Self.retriable(error)
        let body = FailureSubmission.fromClaim(job, reason: reason, retriable: retriable)
        do {
            try await ManagementClient.submitFailure(
                serverUrl: registration.serverUrl,
                auth: auth,
                failure: body
            )
            statusText = retriable ? "Reported retriable failure" : "Reported terminal failure"
            AppLog.jobRun.info(
                "planner worker: failure for \(job.jobId) retriable=\(retriable): \(reason)"
            )
        } catch {
            statusText = "Failure submit error"
            AppLog.jobRun.error(
                "planner worker: could not submit failure for \(job.jobId): \(error.localizedDescription)"
            )
        }
    }

    private enum SubmitOutcome: Sendable {
        case accepted
        /// 409 or unreclaimable 404 — result must not be treated as success.
        case dropped
    }

    private enum SubmitError: Error, LocalizedError {
        case exhaustedRetries(jobId: String)

        var errorDescription: String? {
            switch self {
            case .exhaustedRetries(let jobId):
                return "gave up submitting \(jobId) after repeated transient failures"
            }
        }
    }

    /// Plan-attached submit with backoff. Throws on exhaustion or hard errors;
    /// returns `.dropped` when the server says the lease is gone (not success).
    private func submitWithRetry(
        jobId: String,
        body: Data,
        registration: IdentityRegistration,
        auth: AuthIdentity
    ) async throws -> SubmitOutcome {
        var backoff: TimeInterval = 1
        for _ in 0..<16 {
            do {
                try await ManagementClient.submitPlanResult(
                    serverUrl: registration.serverUrl,
                    auth: auth,
                    payloadJson: body
                )
                return .accepted
            } catch let error as ManagementClientError {
                if case .httpStatus(let code, _) = error {
                    if code == 409 {
                        AppLog.jobRun.error("planner worker: submit 409 for \(jobId) — dropped")
                        return .dropped
                    }
                    if code == 404 {
                        do {
                            try await ManagementClient.reclaim(
                                serverUrl: registration.serverUrl,
                                auth: auth,
                                jobId: jobId
                            )
                            continue
                        } catch {
                            AppLog.jobRun.error(
                                "planner worker: reclaim after submit 404 failed for \(jobId)"
                            )
                            return .dropped
                        }
                    }
                }
                if isTransient(error) {
                    try await Task.sleep(for: .seconds(backoff))
                    backoff = min(30, backoff * 2)
                    continue
                }
                throw error
            }
        }
        AppLog.jobRun.error("planner worker: gave up submitting \(jobId)")
        throw SubmitError.exhaustedRetries(jobId: jobId)
    }

    private func isTransient(_ error: ManagementClientError) -> Bool {
        // 5xx only — 4xx are definitive. Transport `URLError`s are not
        // `ManagementClientError` and propagate to the caller separately.
        if case .httpStatus(let code, _) = error { return code >= 500 }
        return false
    }

    private func idleWaitSeconds() -> TimeInterval {
        // 5 min + 0..60 s jitter (client-integration §4).
        300 + TimeInterval(Int.random(in: 0...60))
    }

    private func isoNow() -> String {
        JobDateFormat.iso8601.string(from: Date())
    }
}

// MARK: - Model store helper

private extension Storage {
    /// Best-effort scan of installed models for claim matching.
    @MainActor
    func loadDiscoveredModels() -> [DiscoveredModel] {
        availableModels()
    }
}
