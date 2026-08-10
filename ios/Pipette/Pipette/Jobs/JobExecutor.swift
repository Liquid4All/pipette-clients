import Foundation
import SwiftUI
import UIKit

/// The single benchmark run loop, shared by the "start new job" flow
/// (NewJobView) and the resume / retry / re-run flow (JobDetailView). The two
/// flows previously carried near-identical copies of this loop that had
/// already drifted in small ways; run-path changes (recovery, gating, pause
/// behavior) belong here so they land in both flows at once.
@MainActor
enum JobExecutor {
    /// Human-readable "<benchmark> <params> · <model>" label for the cell
    /// currently running, shown live in the progress views. Uses the benchmark's
    /// catalog display name plus a compact token summary — so several benchmarks
    /// of the same type (e.g. two decode runs at different token counts) are
    /// distinguishable — and the model's repo tail rather than the full org/repo.
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
        case .vlThroughput:
            guard let w = intVal("parameter_image_width"), let h = intVal("parameter_image_height") else { return nil }
            let text = intVal("parameter_text_tokens").map { " · \($0) tok" } ?? ""
            return "\(w)×\(h)px\(text)"
        case .eval:
            return nil
        }
    }

    /// Whether a completed cell's result should be auto-submitted now — the shared
    /// gate for both the per-cell upload and the job-end sweep. True only when the
    /// run opted into auto-submit, the job opted into contributing, the network is
    /// reachable (so an offline run skips rather than burning the uploader's retries),
    /// and a registration exists to submit under. Pure, so it's unit-testable without
    /// standing up the run loop.
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

    /// Execute every `.pending` cell of `manifest` on a detached background
    /// task, then auto-submit if the finished job is marked for contribution.
    ///
    /// The caller has already reserved the runner via
    /// `jobRunner.start(jobId:flag:)` (the double-start guard) and persisted
    /// `manifest` with the cells to execute set to `.pending` and
    /// `status == .running`.
    ///
    /// `onFinish` runs on the main actor after the loop and any auto-submit
    /// pass, with the final persisted manifest. It is **always invoked exactly
    /// once** — per-cell errors are caught in the loop and never escape, so the
    /// detached task always reaches the finalize step. Headless relies on this
    /// to resume its `withCheckedContinuation`; any early `return`/`throw` added
    /// inside the task before that step would silently hang every headless run.
    static func run(
        manifest: JobManifest,
        jobRunner: JobRunner,
        jobStore: JobStore,
        flag: CancelFlag,
        storage: Storage,
        source: RunSource,
        autoSubmit: Bool = true,
        readiness readinessOverrides: ReadinessPolicy = .init(),
        onFinish: @escaping @MainActor (JobManifest) -> Void = { _ in }
    ) {
        // The cells THIS run will execute. Both analytics events are scoped to these, never to the
        // whole manifest: a resume of a 90%-done job runs a handful of cells, and reporting the
        // job's full cell count at the end would make every resume look like a full run. Matches
        // Android's run-relative `totalInRun` / `cellsCompletedInRun`.
        let pendingCells = manifest.cells.filter { $0.runStatus == .pending }
        let cellIdsThisRun = Set(pendingCells.map(\.cellId))
        let runStartedAt = Date()

        // Capture-then-flush BEFORE the detached run task starts, so the analytics queue is empty
        // entering the measurement window and the SDK's periodic flush timer has nothing to send
        // while a cell is being timed. The flush's network call overlaps model load (seconds), not
        // a timed measurement. The loop's only capture is `results_submitted` from the per-cell
        // drain below, which flushes itself for the same reason. See `AnalyticsEvents`.
        Analytics.capture(AnalyticsEvents.jobStarted, [
            AnalyticsEvents.jobId: manifest.jobId.value,
            AnalyticsEvents.cellCount: cellIdsThisRun.count,
            AnalyticsEvents.source: source.rawValue,
        ])
        Analytics.flush()
        let neededIds = Set(pendingCells.map { $0.benchmarkId })
        // Resolve every needed cell's benchmark catalog-independently via
        // `BenchmarkCatalog.item(forId:in:)`: the synced-catalog entry when listed,
        // else a `BenchmarkItem` reconstructed from the structured id. The parsed
        // item carries the same shape downstream, so context sizing and the
        // existence guard work uniformly. Ids that neither resolve nor parse drop
        // out here and fail per-cell exactly as a missing benchmark.
        let catalogById = Dictionary(
            BenchmarkCatalog.all(store: storage.benchmarks).map { ($0.benchmarkId, $0) },
            uniquingKeysWith: { first, _ in first }
        )
        let pendingBenchmarks: [BenchmarkItem] = neededIds.compactMap {
            BenchmarkCatalog.item(forId: $0, in: catalogById)
        }
        let benchmarkJsonMap: [String: [String: Any]] = pendingBenchmarks
            .reduce(into: [:]) { $0[$1.benchmarkId] = $1.rawJson }
        // Typed benchmark definitions fed to `RunCell.dispatch` for every cell. The raw
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

            // path → formatted failure reason from the first failed load.
            // Reused so subsequent cells in the group show the real cause.
            // Keyed by the model, not its on-disk path: a cell for a model that has not
            // been downloaded yet carries an empty path, and every such cell would share
            // one bucket — so the first failure would mark every undownloaded model
            // failed. The coordinate is the identity that actually distinguishes them.
            var failedModels: [Model: String] = [:]

            // The cell to run. One authored since the spec exists carries its own — from a
            // claim or from New Job. One written before it carries the model alone, so the
            // job's settings stand in: they are where a pre-spec manifest recorded them.
            func cellSpec(_ cell: JobCell, definition: BenchmarkDefinition) -> ClientRunSpec {
                guard cell.spec.runtimeFlags == nil else { return cell.spec }
                // Only the load settings stand in. The readiness gate the cell named is not
                // among them, and rebuilding the spec here would drop it from the record
                // while the gate below still applied it.
                let settings = ClientRunSpec.authored(
                    benchmarkId: cell.benchmarkId, benchmarkType: definition.type,
                    model: cell.source, numberGpuLayers: gpuLayers, nUbatch: ubatch,
                    threads: threads)
                return cell.spec.replacingRuntimeFlags(settings.runtimeFlags)
            }

            // The context a cell will load with, read from the engine's own sizing rather
            // than recomputed. A cell whose benchmark this build cannot classify is refused
            // before it loads, so the value only has to order it stably.
            func cellContextSize(_ cell: JobCell) -> Int {
                definitionMap[cell.benchmarkId]
                    .map { Int(LlamaRuntimeFlags.contextSize(for: $0)) } ?? 0
            }

            // Persist `manifest` (re-adopting the user-editable fields the job
            // page may have changed mid-run) and sync the in-memory store —
            // one call so no save site can forget the store half.
            func saveRunManifest(_ manifest: inout JobManifest) async {
                if let latest = storage.loadJobManifest(jobId: manifest.jobId) {
                    manifest.contributeResults = ResultSubmissionFeatureGate.canSubmitResults(registration: storage.identity.getRegistration())
                        && (latest.contributeResults == true)
                    manifest.title = latest.title
                    // `serverJobId` is owned by the uploader, which acks straight to
                    // disk (it also serves launch/foreground/manual triggers that hold
                    // no in-memory manifest). Re-adopt those acks so this write — which
                    // carries the executor-owned `runStatus` — never clobbers a
                    // submission the uploader just recorded. A cell the loop is actively
                    // running has no ack yet, so `runStatus` still wins.
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
                storage.saveJobManifest(manifest)
                let snapshot = manifest
                await MainActor.run { store.apply(snapshot) }
            }

            // Sort pending cells by (model, ctx) so the load-once-per-model
            // optimization holds even for manifests persisted in
            // benchmark-major order, and so we only reload when the context
            // window changes within a model group. The compare key is the
            // coordinate — the cell's declared `Model`, stable whether or not the
            // weights are on disk yet, so cells for one model stay adjacent and the
            // engine loads it once.
            let executionOrder: [Int] = updatedManifest.cells.indices
                .filter { updatedManifest.cells[$0].runStatus == .pending }
                .sorted {
                    (updatedManifest.cells[$0].source.reference, cellContextSize(updatedManifest.cells[$0]))
                        < (updatedManifest.cells[$1].source.reference, cellContextSize(updatedManifest.cells[$1]))
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

                // Completed count entering this cell — stable for its whole
                // duration, so the ETA anchor can reuse it as cells that emit no
                // intra-cell samples (e.g. throughput) still re-project each
                // boundary and count down in between.
                let completedBeforeCell = updatedManifest.completedCells
                await MainActor.run {
                    runner.currentCellLabel = liveCellLabel(cell, params: benchmarkJsonMap[cell.benchmarkId])
                    runner.currentProgressText = ""
                    runner.currentCellFraction = 0
                    runner.readinessStatus = nil
                    runner.readinessStartedAt = nil
                    runner.anchorETA(completedCells: completedBeforeCell)
                }

                // Crash forensics: if the process dies anywhere in this
                // iteration (a jetsam OOM kill never reaches the catch
                // below), the sentinel left on disk tells the next launch
                // exactly which cell was running. Cleared when the iteration
                // ends with the cell's terminal status already saved.
                let cellStartedAt = JobDateFormat.iso8601.string(from: Date())
                storage.saveActiveCellSentinel(
                    ActiveCellSentinel(cellId: cell.cellId, startedAt: cellStartedAt),
                    jobId: jobId
                )
                defer { storage.clearActiveCellSentinel(jobId: jobId) }

                updatedManifest.cells[index].runStatus = .running
                await saveRunManifest(&updatedManifest)

                guard benchmarkJsonMap[cell.benchmarkId] != nil else {
                    // Neither the synced catalog nor `BenchmarkDefinition(parsingId:)`
                    // could resolve this id — log it (the sibling guards below do)
                    // so a server-assigned job with an unrecognized benchmark isn't
                    // failed silently.
                    let msg = "Benchmark definition not found"
                    AppLog.jobRun.error("\(cell.benchmarkId) / \(cell.modelName): \(msg)")
                    updatedManifest.cells[index].runStatus = .failed
                    updatedManifest.cells[index].errorMessage = msg
                    await saveRunManifest(&updatedManifest)
                    continue
                }

                let cellCtx = cellContextSize(cell)

                do {
                    // Cool the device before each measured rep — the gate before
                    // a cell's first measured rep also serves as the between-cell
                    // cooldown (throughput benchmarks only; ignored by the
                    // engine for eval / max_memory / vl).
                    // The cell's own gate, over the job's: a plan authors `readiness` per
                    // cell because waiving the temperature criterion changes what the
                    // result means, and the job-level value is a headless-run default.
                    // Resolved, not read raw: `resolve` is what checks the entry belongs
                    // to this cell, and the request records the resolved value — honouring
                    // an unresolved one would hold the run under a policy the result denies.
                    let gate = (try? cell.spec.benchmarkFlags?.resolve())?.readiness?
                        .resolved(over: readinessOverrides) ?? readinessOverrides
                    let readiness = BenchmarkReadiness(
                        cancelFlag: flag,
                        maxSeconds: gate.maxSeconds,
                        skipThermal: gate.skipThermal
                    ) { status in
                        // Keep the CLI-style line in the log (raw temp, -1 included,
                        // for debugging); surface the structured status to the UI.
                        // `info`, not `debug`: this is the only account of a wait that can
                        // run for minutes, and a headless run drops `debug` without `-v`.
                        AppLog.jobRun.info("\(status)")
                        Task { @MainActor in
                            if status.isWaiting {
                                // Stamp the start once, at the transition into cooling,
                                // so the TimelineView anchor stays fixed across polls.
                                if runner.readinessStatus?.isWaiting != true {
                                    runner.readinessStartedAt = Date()
                                }
                                runner.readinessStatus = status
                            } else {
                                runner.readinessStatus = nil
                                runner.readinessStartedAt = nil
                            }
                        }
                    }

                    guard let definition = definitionMap[cell.benchmarkId] else {
                        throw NSError(domain: "Pipette", code: -1, userInfo: [
                            NSLocalizedDescriptionKey:
                                "no typed benchmark definition for \(cell.benchmarkId)",
                        ])
                    }

                    let coordinator = await MainActor.run { DownloadCoordinator.shared }
                    let spec = cellSpec(cell, definition: definition)
                    let request = try await RunCell.prepare(
                        spec: spec, benchmark: definition,
                        storage: storage, coordinator: coordinator,
                        // A re-fetch can take minutes; without this the run looks hung.
                        progress: { fetch in
                            Task { @MainActor in
                                runner.currentProgressText =
                                    "Downloading \(cell.modelName): "
                                    + "\(Int(fetch.fraction * 100))%"
                            }
                        })
                    let model = request.model

                    // Memory pre-flight: measure the model against the remaining jetsam
                    // budget, but never block the load — barely-fitting runs are
                    // legitimate benchmarks, and the sentinel turns a kill into an
                    // explained failure. Re-stamp the sentinel with the numbers so that
                    // explanation can say how tight the fit was, and surface a warning
                    // while the model loads. AFM has no bound path, so it weighs nothing.
                    let memory = model.bound.boundPaths
                        .flatMap { MemoryGate.snapshot(modelPath: $0.payload) }
                    if let memory {
                        storage.saveActiveCellSentinel(
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

                    // Reload fresh for every benchmark — isolated, comparable
                    // measurements that reset memory between cells. Each runtime
                    // owns its model handle for the duration of `.run(...)` and
                    // unloads it before returning.
                    await MainActor.run {
                        // The model name is already shown on the cell-label row, so
                        // the loading line just states the phase — no duplicate name.
                        runner.currentProgressText = memoryWarning ?? "Loading…"
                    }

                    let response: RunResponse
                    do {
                        // Presentation policy for intra-cell progress lives here — the
                        // runtimes report *what* advanced; we decide *how* to show it.
                        // Faithful to the historical behavior: an eval sample advances
                        // the progress bar (eval reported per-sample); a throughput
                        // attempt is a status line only, so the bar advances at cell
                        // boundaries via the completed-cells counter. To get an
                        // intra-cell throughput bar, also set `currentCellFraction` in
                        // the `.attempt` arm.
                        response = try await RunCell.dispatch(
                            request,
                            storage: storage,
                            // Throughput/latency wait for a cool baseline before each
                            // measured rep; eval instead runs flat-out and only stops on
                            // cancel (see `cancellationGate`).
                            readiness: { readiness.waitUntilReady() },
                            isCancelled: { flag.isCancelled },
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
                                        // A measured repetition of a throughput/latency
                                        // benchmark (the runner averages several).
                                        runner.currentProgressText = "Measurement \(completed)/\(total)"
                                    }
                                }
                            })
                    } catch {
                        failedModels[cell.source] = formatJobError(error, contextSize: cellCtx)
                        throw error
                    }

                    try await MainActor.run {
                        try PayloadBuilder.writeLocal(
                            request: request,
                            response: response,
                            cellId: cell.cellId,
                            source: cell.benchmarkSource ?? .remote,
                            storage: storage
                        )
                    }

                    AppLog.jobRun.info(
                        "result \(cell.benchmarkId) / \(cell.modelName): \(response.resultData)")
                    updatedManifest.cells[index].runStatus = .completed
                    await saveRunManifest(&updatedManifest)

                    // Upload this result now, before the next cell — so a crash,
                    // jetsam kill, or interruption mid-job doesn't strand every
                    // completed cell's data until job end (PIP-358). `drainJob` is
                    // serialized and idempotent, so no double-submit; the ack lands
                    // on disk and the next `saveRunManifest` re-adopts it.
                    if Self.shouldAutoSubmit(updatedManifest, autoSubmit: autoSubmit,
                                             online: NetworkReachability.shared.isConnected,
                                             registration: storage.identity.getRegistration()) {
                        let outcome = await ResultUploader.shared.drainJob(jobId: jobId)
                        if !outcome.errors.isEmpty {
                            AppLog.jobRun.error("per-cell submit \(cell.benchmarkId) / \(cell.modelName): \(outcome.errors.joined(separator: "; "))")
                        }
                    }

                    // Between-cell cooldown is handled inside the benchmark
                    // runner: every throughput cell gates on the SoC die
                    // temperature before its warm-up (cell entry) and before
                    // each measured rep, via the same `BenchmarkReadiness`.
                } catch {
                    if flag.isCancelled {
                        updatedManifest.cells[index].runStatus = .cancelled
                    } else {
                        let formatted = formatJobError(error, contextSize: cellCtx)
                        AppLog.jobRun.error("\(cell.benchmarkId) / \(cell.modelName) failed: \(formatted)")
                        updatedManifest.cells[index].runStatus = .failed
                        updatedManifest.cells[index].errorMessage = formatted
                    }
                    await saveRunManifest(&updatedManifest)
                    if flag.isCancelled { break }
                }
            }

            // If the run stopped early, mark remaining pending cells as
            // cancelled and put the job in the .paused state so it can be
            // resumed; an auto-pause (app backgrounded) records why.
            updatedManifest.finalizeRunEnd(cancelled: flag.isCancelled, cancelReason: flag.reason)
            await saveRunManifest(&updatedManifest)

            // Final sweep after the job finishes: same `shouldAutoSubmit` gate as
            // the per-cell upload, plus a completed-status check. Headless passes
            // autoSubmit=false, so it never runs here.
            if updatedManifest.status == .completed,
               Self.shouldAutoSubmit(updatedManifest, autoSubmit: autoSubmit,
                                     online: NetworkReachability.shared.isConnected,
                                     registration: storage.identity.getRegistration()) {
                await MainActor.run {
                    runner.currentProgressText = "Submitting results..."
                }

                // Route through the shared uploader so this job-end submit
                // serializes with any launch/foreground drain firing at the
                // same time. This is a final sweep — most cells were already
                // submitted per-cell during the loop; it catches any that hit a
                // transient error then.
                let outcome = await ResultUploader.shared.drainJob(jobId: jobId)
                if !outcome.errors.isEmpty {
                    AppLog.jobRun.error("contribution failed: \(outcome.errors.joined(separator: "; "))")
                }
            }

            // Hand the UI the final on-disk state. Every serverJobId was recorded
            // to disk (the uploader's acks, folded into the in-memory manifest by
            // `saveRunManifest`'s re-adopt), so the disk manifest is authoritative
            // — no need to thread the acks back through `updatedManifest`.
            let finalManifest = storage.loadJobManifest(jobId: jobId) ?? updatedManifest
            // The outcome comes from the persisted manifest rather than from a thrown error: the
            // cell loop absorbs per-cell failures and always writes a terminal status, so a
            // manifest left non-terminal is the "the run itself died" case. Mirrors Android's
            // `jobOutcome`.
            Analytics.capture(AnalyticsEvents.jobCompleted, [
                AnalyticsEvents.jobId: jobId.value,
                AnalyticsEvents.outcome: analyticsOutcome(cancelled: flag.isCancelled, status: finalManifest.status),
                // Run-scoped, to stay comparable with the `job_started` count above.
                AnalyticsEvents.cellsCompleted: finalManifest.cells
                    .filter { cellIdsThisRun.contains($0.cellId) && $0.runStatus == .completed }
                    .count,
                AnalyticsEvents.cellCount: cellIdsThisRun.count,
                // Wall-clock for the whole run, matching Android's `System.currentTimeMillis()`
                // delta. Includes model load and between-cell cooldown, so it is run duration, not
                // summed measurement time.
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
