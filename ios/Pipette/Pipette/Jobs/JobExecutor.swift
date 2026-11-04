import Foundation
import SwiftUI
import UIKit

/// The single benchmark run loop, shared by every job start: `JobLauncher`, the resume /
/// retry / re-run flow (JobDetailView) and the planner worker. Run-path changes (recovery,
/// gating, pause behavior) belong here.
@MainActor
enum JobExecutor {
    /// "<benchmark> <params> · <model>" label for the running cell. The token summary
    /// tells same-type benchmarks apart; the model shows its repo tail only.
    static func liveCellLabel(_ cell: JobCell, params: [String: Any]?) -> String {
        let name: String
        if let type = cell.benchmarkType, type != .eval {
            let base = BenchmarkCatalog.displayName(for: type.rawValue)
            if let summary = benchmarkParamSummary(type: type, params: params ?? [:]) {
                name = "\(base) · \(summary)"
            } else {
                name = base
            }
        } else {
            // Eval (and unknown/legacy) benchmarks are identified by their id — the
            // dataset name — not by token params, so show that verbatim.
            name = cell.benchmarkId
        }
        let model = cell.modelName.split(separator: "/").last.map(String.init) ?? cell.modelName
        return "\(name) · \(model)"
    }

    /// Compact token summary distinguishing same-type benchmarks, e.g.
    /// `"512→100 tok"` (prefill→decode) or `"512 tok"`. Nil when no sizing params
    /// are present.
    private static func benchmarkParamSummary(type: BenchmarkType, params: [String: Any]) -> String? {
        func intVal(_ key: String) -> Int? {
            if let n = params[key] as? Int { return n }
            if let n = params[key] as? NSNumber { return n.intValue }
            if let s = params[key] as? String, let n = Int(s) { return n }
            return nil
        }
        let prefill = intVal("parameter_prefill_tokens")
        let decode = intVal("parameter_decode_tokens")
        switch type {
        case .prefillThroughput, .maxMemoryUsage:
            return prefill.map { "\($0) tok" }
        case .decodeThroughput, .endToEndLatency:
            if let prefill, let decode { return "\(prefill)→\(decode) tok" }
            return (decode ?? prefill).map { "\($0) tok" }
        case .vlThroughput, .vlMaxMemory:
            guard let w = intVal("parameter_image_width"), let h = intVal("parameter_image_height") else { return nil }
            let text = intVal("parameter_text_tokens").map { " · \($0) tok" } ?? ""
            return "\(w)×\(h)px\(text)"
        case .asrThroughput, .asrMaxMemory:
            guard let ms = intVal("parameter_audio_duration_ms") else { return nil }
            let text = intVal("parameter_text_tokens").map { " · \($0) tok" } ?? ""
            return "\(ms) ms audio\(text)"
        case .vlSpecDecode:
            // The corpus is the workload, and it is a pin rather than a size. What
            // distinguishes two cells on one corpus is the draft length, which is what the
            // catalog id's own suffix names (`…_k8`).
            return intVal("parameter_draft_length").map { "k\($0)" }
        case .eval:
            return nil
        }
    }

    /// Whether a completed cell's result should be auto-submitted now, for both the
    /// per-cell upload and the job-end sweep. An offline run skips rather than burning the
    /// uploader's retries.
    nonisolated static func shouldAutoSubmit(
        _ manifest: JobManifest,
        autoSubmit: Bool,
        online: Bool,
        registration: IdentityRegistration?
    ) -> Bool {
        autoSubmit
            && manifest.contributeResults == true
            && online
            && ResultSubmissionFeatureGate.canSubmitResults(registration: registration)
    }

    /// The benchmark each of `ids` runs. Ids that resolve nowhere drop out, and their cells
    /// fail as a missing benchmark. In order:
    ///
    /// 1. `claimed`: a claim revalidated or fetched it, so it is newer than anything on disk.
    /// 2. The synced detail, for a corpus-pinned type: the server stamps `dataset.url` only on
    ///    `GET /benchmarks/{id}`, so the list row names the corpus but not where to get it.
    /// 3. The synced list's row, else the local half. The list keeps `parameter_*` fields this
    ///    build may not have classified, which the live label reads.
    /// 4. A synced detail the list does not name: a claim caches a fetched benchmark there.
    /// 5. A definition reconstructed from a structured id.
    static func pendingBenchmarks(
        ids: Set<String>, store: BenchmarkStore, claimed: [BenchmarkDefinition]
    ) -> [BenchmarkItem] {
        let claimedById = Dictionary(
            claimed.map { ($0.benchmarkId.value, $0) }, uniquingKeysWith: { _, last in last })
        let catalogById = Dictionary(
            BenchmarkCatalog.all(store: store).map { ($0.benchmarkId, $0) },
            uniquingKeysWith: { first, _ in first }
        )
        let detail: (String) -> BenchmarkItem? = { id in
            (try? BenchmarkId(id)).flatMap { try? store.get(.remote($0)) }
                .map { BenchmarkItem(parsed: $0) }
        }
        return ids.compactMap { id in
            claimedById[id].map { BenchmarkItem(parsed: $0) }
                ?? (BenchmarkType.fromId(id)?.needsCorpus == true ? detail(id) : nil)
                ?? catalogById[id]
                ?? detail(id)
                ?? BenchmarkCatalog.item(forId: id, in: catalogById)
        }
    }

    /// Execute every `.pending` cell of `manifest` on a detached task, then auto-submit if
    /// the finished job is marked for contribution. The caller has already reserved the
    /// runner via `jobRunner.start(jobId:flag:)` and persisted `manifest` as `.running`.
    ///
    /// `onFinish` runs on the main actor with the final persisted manifest, **exactly
    /// once**: per-cell errors are caught in the loop. Callers awaiting it through
    /// `withCheckedContinuation` (headless, the planner worker) hang if an early
    /// `return`/`throw` is added inside the task before the finalize step.
    ///
    /// `keepEvalCheckpoint`: for a caller that still owes the server the saved result; it
    /// receives the cell's eval checkpoint rather than the executor finalizing it. It holds
    /// one digest, so pass it only with a one-cell manifest.
    ///
    /// `keepFetchFailure`: for a caller that reports the failure to the server; it receives
    /// the first cell's typed vault failure, which `errorMessage` holds only as text.
    ///
    /// `claimedAt`: when this device received the claim that started the job. Each result
    /// records the time since it as `device_time_ms` (none when nil). Every cell counts from
    /// the same instant, so pass it only with a one-cell manifest.
    ///
    /// `claimedBenchmarks`: the definitions a claim resolved through the server. Each runs as
    /// given, ahead of the synced list, which can be older.
    static func run(
        manifest: JobManifest,
        jobRunner: JobRunner,
        jobStore: JobStore,
        flag: CancelFlag,
        ws: PipetteWorkspace,
        source: RunSource,
        autoSubmit: Bool = true,
        claimedAt: ContinuousClock.Instant? = nil,
        claimedBenchmarks: [BenchmarkDefinition] = [],
        keepEvalCheckpoint: KeptEvalCheckpoint? = nil,
        keepFetchFailure: KeptFetchFailure? = nil,
        onFinish: @escaping @MainActor (JobManifest) -> Void = { _ in }
    ) {
        // Scoped to the cells THIS run executes, so a resume of a 90%-done job doesn't
        // report as a full run. Matches Android's run-relative `totalInRun` /
        // `cellsCompletedInRun`.
        let pendingCells = manifest.cells.filter { $0.runStatus == .pending }
        let cellIdsThisRun = Set(pendingCells.map(\.cellId))
        let runStartedAt = Date()

        // Capture-then-flush before the run starts, so the SDK's periodic flush has nothing
        // to send while a cell is timed; the network call overlaps model load instead. See
        // `AnalyticsEvents`.
        Analytics.capture(AnalyticsEvents.jobStarted, [
            AnalyticsEvents.jobId: manifest.jobId.value,
            AnalyticsEvents.cellCount: cellIdsThisRun.count,
            AnalyticsEvents.source: source.rawValue,
        ])
        Analytics.flush()
        let pendingBenchmarks = Self.pendingBenchmarks(
            ids: Set(pendingCells.map { $0.benchmarkId }), store: ws.benchmarks,
            claimed: claimedBenchmarks)
        let benchmarkJsonMap: [String: [String: Any]] = pendingBenchmarks
            .reduce(into: [:]) { $0[$1.benchmarkId] = $1.rawJson }
        // Typed benchmark definitions fed to `RunCell.dispatchRun` for every cell. The raw
        // `benchmarkJsonMap` above is kept for the live cell label and the existence
        // guard, both of which read parameters this build may not have classified.
        let definitionMap: [String: BenchmarkDefinition] = pendingBenchmarks
            .reduce(into: [:]) { if let def = $1.definition { $0[$1.benchmarkId] = def } }

        let jobId = manifest.jobId
        let ngl = manifest.nGpuLayers
        let gpuLayers = UInt32(ngl)
        let ubatch = UInt32(manifest.effectivePrefillBatch)   // llama n_ubatch / MLX prefill chunk
        // Absent means the engine picks: `LlamaRuntimeFlags.forRun` fills it from the
        // device and reports what it used, so the record still names a number.
        let threads = manifest.threads.map(UInt32.init)

        jobRunner.setRunBaselines(
            completedAtStart: manifest.completedCells,
            totalToRun: manifest.cells.filter { $0.runStatus == .pending }.count
        )
        UIApplication.shared.isIdleTimerDisabled = true

        // Class-reference locals so the detached task doesn't capture a view's
        // environment storage, which may be torn down mid-run.
        let runner = jobRunner
        let store = jobStore
        let initialManifest = manifest

        Task.detached {
            var updatedManifest = initialManifest

            // Model -> formatted reason from its first failed load, so later cells for it
            // show the real cause. Keyed by model, not path: every undownloaded model has
            // an empty path.
            var failedModels: [Model: String] = [:]

            // A cell authored since the spec existed carries its own load settings; an older
            // one carries the model alone, so the job's settings stand in.
            func cellSpec(_ cell: JobCell, definition: BenchmarkDefinition) -> ClientRunSpec {
                guard cell.spec.runtimeFlags == nil else { return cell.spec }
                // Only the load settings stand in. The readiness gate the cell named is not
                // among them, and rebuilding the spec here would drop it from the record
                // while the gate below still applied it.
                let settings = ClientRunSpec.authored(
                    benchmarkId: cell.benchmarkId, benchmarkType: definition.benchmarkType,
                    model: cell.source, numberGpuLayers: gpuLayers, nUbatch: ubatch,
                    threads: threads)
                return cell.spec.replacingRuntimeFlags(settings.runtimeFlags)
            }

            // The context a cell asks for, read from the engine's own sizing rather than
            // recomputed; the VL and ASR kinds replace theirs with the projector's count at
            // load. A cell whose benchmark this build cannot classify is
            // refused before it loads, so the value only has to order it stably.
            func cellContextSize(_ cell: JobCell) -> Int {
                definitionMap[cell.benchmarkId]
                    .map { Int(LlamaRuntimeFlags.ctxSize(for: $0)) } ?? 0
            }

            // Persist and sync the store in one call, re-adopting the fields the job page may
            // edit mid-run.
            func saveRunManifest(_ manifest: inout JobManifest) async {
                if let latest = ws.loadJobManifest(jobId: manifest.jobId) {
                    manifest.contributeResults = ResultSubmissionFeatureGate.canSubmitResults(registration: ws.registration.get())
                        && (latest.contributeResults == true)
                    manifest.title = latest.title
                    // `serverJobId` is the uploader's, which acks straight to disk. Re-adopt
                    // its acks so this write never clobbers a submission just recorded; a
                    // running cell has no ack yet, so `runStatus` still wins.
                    let acked = Dictionary(
                        latest.cells.compactMap { cell in cell.serverJobId.map { (cell.cellId, $0) } },
                        uniquingKeysWith: { first, _ in first }
                    )
                    for i in manifest.cells.indices {
                        if let serverJobId = acked[manifest.cells[i].cellId] {
                            manifest.cells[i].serverJobId = serverJobId
                        }
                    }
                }
                ws.saveJobManifest(manifest)
                let snapshot = manifest
                await MainActor.run { store.apply(snapshot) }
            }

            // Group pending cells by (model, ctx), whatever order the manifest was stored
            // in. The key is the declared `Model`, stable whether or not the weights are on
            // disk yet.
            let executionOrder: [Int] = updatedManifest.cells.indices
                .filter { updatedManifest.cells[$0].runStatus == .pending }
                .sorted {
                    (updatedManifest.cells[$0].source.description, cellContextSize(updatedManifest.cells[$0]))
                        < (updatedManifest.cells[$1].source.description, cellContextSize(updatedManifest.cells[$1]))
                }

            for index in executionOrder {
                if flag.isCancelled { break }

                let cell = updatedManifest.cells[index]

                if let reason = failedModels[cell.source] {
                    updatedManifest.cells[index].runStatus = .failed
                    updatedManifest.cells[index].errorMessage = reason
                    await saveRunManifest(&updatedManifest)
                    continue
                }

                // Stable for the whole cell, so cells that emit no intra-cell samples still
                // re-anchor the ETA at each boundary.
                let completedBeforeCell = updatedManifest.completedCells
                await MainActor.run {
                    runner.currentCellLabel = liveCellLabel(cell, params: benchmarkJsonMap[cell.benchmarkId])
                    runner.currentProgressText = ""
                    runner.currentCellFraction = 0
                    runner.readinessStatus = nil
                    runner.readinessStartedAt = nil
                    runner.anchorETA(completedCells: completedBeforeCell)
                }

                // If the process dies in this iteration (a jetsam kill never reaches the
                // catch below), the sentinel tells the next launch which cell was running.
                let cellStartedAt = JobDateFormat.iso8601.string(from: Date())
                ws.saveActiveCellSentinel(
                    ActiveCellSentinel(cellId: cell.cellId, startedAt: cellStartedAt),
                    jobId: jobId
                )
                defer { ws.clearActiveCellSentinel(jobId: jobId) }

                updatedManifest.cells[index].runStatus = .running
                // With the `.running` save, so crash recovery keeps it.
                updatedManifest.cells[index].measuredAfterOtherWork = MemoryFreshness.record(
                    for: cell.benchmarkType, ranEngineWork: ProcessMemory.hasRunEngineWork)
                await saveRunManifest(&updatedManifest)

                guard benchmarkJsonMap[cell.benchmarkId] != nil else {
                    // Logged so a server-assigned job with an unrecognized benchmark isn't
                    // failed silently.
                    let msg = "Benchmark definition not found"
                    AppLog.jobRun.error("\(cell.benchmarkId) / \(cell.modelName): \(msg)")
                    updatedManifest.cells[index].runStatus = .failed
                    updatedManifest.cells[index].errorMessage = msg
                    await saveRunManifest(&updatedManifest)
                    continue
                }

                // What an out-of-memory message names: nothing for the cells whose window
                // only exists inside the load.
                let cellCtx: Int? = switch definitionMap[cell.benchmarkId]?.benchmarkType {
                case .vlThroughput?, .vlMaxMemory?, .asrThroughput?, .asrMaxMemory?: nil
                default: cellContextSize(cell)
                }

                do {
                    guard let definition = definitionMap[cell.benchmarkId] else {
                        throw NSError(domain: "Pipette", code: -1, userInfo: [
                            NSLocalizedDescriptionKey:
                                "no typed benchmark definition for \(cell.benchmarkId)",
                        ])
                    }

                    let coordinator = await MainActor.run { DownloadCoordinator.shared }
                    let run: FinishedRun
                    do {
                        // The steps and their order are `RunCell.runCell`'s; this closure set
                        // is the presentation: the wait's status line, the download percentage,
                        // the jetsam sentinel, and the intra-cell progress text.
                        run = try await RunCell.runCell(
                            spec: cellSpec(cell, definition: definition),
                            benchmark: definition,
                            ws: ws,
                            coordinator: coordinator,
                            readiness: { policy in
                                BenchmarkReadiness(
                                    cancelFlag: flag,
                                    readiness: policy
                                ) { status in
                                    // `info`, not `debug`: this is the only account of a wait
                                    // that can run for minutes, and a headless run drops
                                    // `debug` without `-v`.
                                    AppLog.jobRun.info("\(status)")
                                    Task { @MainActor in
                                        if status.isWaiting() {
                                            // Stamp the start once, on entering cooling, so
                                            // the TimelineView anchor stays fixed across polls.
                                            if runner.readinessStatus?.isWaiting() != true {
                                                runner.readinessStartedAt = Date()
                                            }
                                            runner.readinessStatus = status
                                        } else {
                                            runner.readinessStatus = nil
                                            runner.readinessStartedAt = nil
                                        }
                                    }
                                }
                            },
                            cancelled: { flag.isCancelled },
                            // A re-fetch can take minutes; without this the run looks hung.
                            fetchProgress: { fetch in
                                Task { @MainActor in
                                    runner.currentProgressText =
                                        "Downloading \(cell.modelName): "
                                        + "\(Int(fetch.fraction * 100))%"
                                }
                            },
                            jobId: jobId,
                            preflight: { _, memory in
                                // Re-stamp the sentinel with the model's weight against the
                                // remaining budget, so a jetsam kill can be explained.
                                if let memory {
                                    ws.saveActiveCellSentinel(
                                        ActiveCellSentinel(
                                            cellId: cell.cellId,
                                            startedAt: cellStartedAt,
                                            availableBytes: memory.availableBytes,
                                            modelBytes: memory.modelBytes
                                        ),
                                        jobId: jobId
                                    )
                                }
                                let memoryWarning = memory.flatMap {
                                    MemoryGate.warning(
                                        modelName: cell.modelName,
                                        modelBytes: $0.modelBytes,
                                        availableBytes: $0.availableBytes
                                    )
                                }
                                if let memoryWarning {
                                    AppLog.jobRun.warning("\(cell.benchmarkId) / \(cell.modelName): \(memoryWarning)")
                                }
                                // The model name is already on the cell-label row, so this
                                // states the phase alone.
                                await MainActor.run {
                                    runner.currentProgressText = memoryWarning ?? "Loading…"
                                }
                            },
                            // An eval sample advances the progress bar; a throughput attempt
                            // is a status line only, so that bar advances at cell boundaries.
                            progress: { event in
                                Task { @MainActor in
                                    switch event {
                                    case .sample(let completed, let total):
                                        if total > 0 {
                                            runner.currentCellFraction = min(1, Double(completed) / Double(total))
                                            runner.anchorETA(completedCells: completedBeforeCell)
                                        }
                                        runner.currentProgressText = "Sample \(completed)/\(total)"
                                    case .attempt(let completed, let total):
                                        // A measured repetition; the runner averages several.
                                        runner.currentProgressText = "Measurement \(completed)/\(total)"
                                    case .corpusDownload:
                                        runner.currentProgressText = "Downloading the corpus…"
                                    }
                                }
                            })
                    } catch let failure as ModelLoadFailure {
                        // Only a failure to load the weights belongs to the model: every
                        // later cell for it would hit the same wall. A per-cell failure (a
                        // rejected rep, a cancelled wait, a body this build cannot read)
                        // leaves the rest of the job runnable.
                        failedModels[cell.source] = formatJobError(failure, contextSize: cellCtx)
                        throw failure
                    }

                    // The result is ready: a claimed job's clock stops here, before the
                    // result is filed or submitted.
                    let payload = timedFromClaim(
                        run.payload, claimedAt: claimedAt, now: ContinuousClock.now)
                    // Filed by the half its benchmark came from, as the crate files it.
                    try recordResult(
                        results: ws.results, payload: payload, extras: run.extras,
                        location: BenchmarkResultLocation(cell.benchmarkSource ?? .remote),
                        resultId: cell.cellId)
                    // Saved on disk, so the checkpoint has done its job, unless the caller
                    // still owes the server the result.
                    if let keepEvalCheckpoint {
                        await MainActor.run { keepEvalCheckpoint.digest = run.evalCheckpoint }
                    } else {
                        RunCell.finalizeEvalCheckpoint(ws: ws, digest: run.evalCheckpoint)
                    }

                    AppLog.jobRun.info(
                        "result \(cell.benchmarkId) / \(cell.modelName): \(run.payload.result)")
                    updatedManifest.cells[index].runStatus = .completed
                    await saveRunManifest(&updatedManifest)

                    // Upload now, before the next cell, so a crash or jetsam kill mid-job
                    // doesn't strand every completed result until job end. `drainJob` is
                    // serialized and idempotent; the ack lands on disk and the next
                    // `saveRunManifest` re-adopts it.
                    if Self.shouldAutoSubmit(updatedManifest, autoSubmit: autoSubmit,
                                             online: NetworkReachability.shared.isConnected,
                                             registration: ws.registration.get()) {
                        let outcome = await ResultUploader.shared.drainJob(jobId: jobId)
                        if !outcome.errors.isEmpty {
                            AppLog.jobRun.error("per-cell submit \(cell.benchmarkId) / \(cell.modelName): \(outcome.errors.joined(separator: "; "))")
                        }
                    }

                    // No cooldown here: the readiness gate waits on the SoC die temperature
                    // before each timing cell's model load and before each measured rep.
                } catch {
                    if flag.isCancelled {
                        updatedManifest.cells[index].runStatus = .cancelled
                    } else {
                        let formatted = formatJobError(error, contextSize: cellCtx)
                        AppLog.jobRun.error("\(cell.benchmarkId) / \(cell.modelName) failed: \(formatted)")
                        updatedManifest.cells[index].runStatus = .failed
                        updatedManifest.cells[index].errorMessage = formatted
                        if let keepFetchFailure {
                            await MainActor.run { keepFetchFailure.keep(error) }
                        }
                    }
                    await saveRunManifest(&updatedManifest)
                    if flag.isCancelled { break }
                }
            }

            updatedManifest.finalizeRunEnd(cancelled: flag.isCancelled, cancelReason: flag.reason)
            await saveRunManifest(&updatedManifest)

            // Final sweep: the same gate as the per-cell upload, plus a completed-status
            // check. The planner worker passes autoSubmit=false, so it never runs there.
            if updatedManifest.status == .completed,
               Self.shouldAutoSubmit(updatedManifest, autoSubmit: autoSubmit,
                                     online: NetworkReachability.shared.isConnected,
                                     registration: ws.registration.get()) {
                await MainActor.run {
                    runner.currentProgressText = "Submitting results..."
                }

                // Through the shared uploader so this serializes with a launch/foreground
                // drain. Most cells were submitted per-cell; this catches transient failures.
                let outcome = await ResultUploader.shared.drainJob(jobId: jobId)
                if !outcome.errors.isEmpty {
                    AppLog.jobRun.error("contribution failed: \(outcome.errors.joined(separator: "; "))")
                }
            }

            // The disk manifest is authoritative: every serverJobId ack was written there.
            let finalManifest = ws.loadJobManifest(jobId: jobId) ?? updatedManifest
            // The outcome comes from the persisted manifest, not a thrown error: the loop
            // always writes a terminal status, so a non-terminal one means the run itself
            // died. Mirrors Android's `jobOutcome`.
            Analytics.capture(AnalyticsEvents.jobCompleted, [
                AnalyticsEvents.jobId: jobId.value,
                AnalyticsEvents.outcome: analyticsOutcome(cancelled: flag.isCancelled, status: finalManifest.status),
                // Run-scoped, to stay comparable with the `job_started` count above.
                AnalyticsEvents.cellsCompleted: finalManifest.cells
                    .filter { cellIdsThisRun.contains($0.cellId) && $0.runStatus == .completed }
                    .count,
                AnalyticsEvents.cellCount: cellIdsThisRun.count,
                // Wall-clock for the whole run (load and cooldown included), matching
                // Android's `System.currentTimeMillis()` delta.
                AnalyticsEvents.durationMs: Int(Date().timeIntervalSince(runStartedAt) * 1000),
            ])
            await MainActor.run {
                store.apply(finalManifest)
                runner.finish(jobId: jobId)
                UIApplication.shared.isIdleTimerDisabled = false
                onFinish(finalManifest)
            }
        }
    }
}

/// A saved cell's eval checkpoint, left by `JobExecutor.run` for its caller to finalize.
@MainActor
final class KeptEvalCheckpoint {
    var digest: EvalRunDigest?
}

/// The first cell's vault failure, left by `JobExecutor.run` in its type, so the worker
/// can tell a refusal every device would meet from one worth another claim.
@MainActor
final class KeptFetchFailure {
    private(set) var error: VaultFetchError?

    /// Keep `error` if it is the first vault failure of the run. A later failure, or an
    /// error of another type, leaves the kept failure in place.
    func keep(_ error: any Error) {
        guard self.error == nil, let vault = error as? VaultFetchError else { return }
        self.error = vault
    }
}
