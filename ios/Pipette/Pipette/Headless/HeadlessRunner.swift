import Foundation
import MLXLMCommon

/// The headless process's exit codes, mirroring the CLI's split.
nonisolated enum HeadlessExit {
    static let ok: Int32 = 0
    static let failure: Int32 = 1
    static let usage: Int32 = 2

    /// Ends the process on the main actor, so no main-thread callback runs while `exit`
    /// destroys C++ statics (a Sentry display-link callback once locked a destroyed mutex).
    @MainActor static func terminate(_ code: Int32) -> Never {
        exit(code)
    }

    /// Ends the process at once from any thread, with no exit handlers or C++ static
    /// destructors to race the threads still running: for an engine call that cannot be
    /// stopped and may hold the main actor `terminate` waits on.
    static func abandon(_ code: Int32) -> Never {
        fflush(stdout)
        _exit(code)
    }

    /// `terminate` once in-flight downloads have saved their partials for the next launch.
    /// Every exit after work starts goes through here, except `abandon`.
    @MainActor static func finish(_ code: Int32) async -> Never {
        finishing = true
        await DownloadCoordinator.shared.keepPartialsForNextLaunch()
        terminate(code)
    }

    /// Set once an exit is underway, so a SIGTERM during it neither prints a second
    /// `BENCH_DONE` nor replaces the exit code.
    @MainActor private static var finishing = false

    /// Kept for the life of the process: a released source stops delivering the signal.
    @MainActor private static var sigtermSource: DispatchSourceSignal?

    /// End on SIGTERM through `finish`. `pipette-ios-proxy` sends it at its run deadline and
    /// force-terminates 10 s later, which bounds the time to save the partials.
    @MainActor static func finishOnSigterm() {
        signal(SIGTERM, SIG_IGN)
        let source = DispatchSource.makeSignalSource(signal: SIGTERM, queue: .main)
        source.setEventHandler {
            Task { @MainActor in
                guard !finishing else { return }
                HeadlessRunner.benchDone(failure, "terminated by SIGTERM")
                await finish(failure)
            }
        }
        source.resume()
        sigtermSource = source
    }
}

/// Headless CLI front-end over the same controllers the SwiftUI views bind, launched via
/// `devicectl device process launch --console`. Each launch runs one command, opens with
/// `[HEADLESS] hello pid=<pid>` on stderr, prints its other `[HEADLESS] …` lines to stdout
/// (diagnostics go to stderr, `logDiagnostic`), and ends with the `BENCH_DONE` sentinel on
/// every terminal path, early refusals included: a console consumer keyed on it hangs
/// otherwise.
///
/// `AppLog` lines reach `--console` on stderr as well (`debug` only with `-v`). The unified
/// log has them too:
///   log stream --predicate 'subsystem == "ai.liquid.liquid-pipette"' --level info
///
/// The grammar is `pipette`'s with `key=value` for `--key value` (both parse), declared in
/// `HeadlessTree`. The verbs, their parameters and where they differ from `pipette` are
/// in docs/pipette-ios/usage.md.
///
/// Example:
///   xcrun devicectl device process launch --device <UDID> --console \
///     ai.liquid.liquid-pipette headlessrun benchmarks run \
///     benchmark=prefill_throughput_512 runtime=mlx \
///     model=mlx://repo=LiquidAI/LFM2.5-350M-MLX-4bit sync=1
///   xcrun devicectl device process launch --device <UDID> --console \
///     ai.liquid.liquid-pipette headlessrun runtime=mlx batch=512 \
///     metrics=prefill,decode,maxmem offsets=256,512,1024,2048,4096
enum HeadlessRunner {
    /// The run's terminal line: `BENCH_DONE <exit-code> [detail]`.
    ///
    /// `devicectl` does not pass the app's exit code through, so this line is the only
    /// channel for it. `pipette-ctl`'s `scan_sentinel` reads a bare `BENCH_DONE` as status
    /// 0, which would record a refusal as a completed cell. It reads only the first token
    /// after the sentinel, so the rest is free text.
    nonisolated static func benchDone(_ code: Int32, _ detail: String? = nil,
                                     onStderr: Bool = false) {
        let line = benchDoneLine(code, detail)
        if onStderr { logDiagnostic(line) } else { log(line) }
    }

    /// `benchDone` for a handler that reports success as a `Bool`.
    nonisolated static func benchDone(ok: Bool, onStderr: Bool = false) {
        benchDone(ok ? HeadlessExit.ok : HeadlessExit.failure, onStderr: onStderr)
    }

    nonisolated static func benchDoneLine(_ code: Int32, _ detail: String? = nil) -> String {
        detail.map { "BENCH_DONE \(code) \($0)" } ?? "BENCH_DONE \(code)"
    }

    nonisolated static func benchDoneLine(ok: Bool) -> String {
        benchDoneLine(ok ? HeadlessExit.ok : HeadlessExit.failure)
    }

    /// Every headless launch's first line, on stderr to keep stdout clean: `pipette-ios-proxy`
    /// relaunches a launch silent past `first_line_timeout_secs` (Single App Mode's reopen race).
    nonisolated static func helloLine(pid: Int32) -> String {
        "hello pid=\(pid)"
    }

    /// A `claim=` run is one cell, so a failed cell fails it, its error flattened to one
    /// line for `pipette-ios-proxy` to file. Otherwise the job's status decides.
    nonisolated static func sweepDone(status: JobStatus, cellFailure: String?,
                                      claimed: Bool) -> (code: Int32, detail: String) {
        if claimed, let cellFailure {
            let error = cellFailure.split(whereSeparator: \.isNewline).joined(separator: " ")
            return (HeadlessExit.failure, "failed: \(error)")
        }
        return (status == .completed ? HeadlessExit.ok : HeadlessExit.failure, "\(status)")
    }

    /// Whether this process runs one headless command and exits; `PipetteApp` skips its
    /// startup client work then.
    nonisolated static let isOneShot = isOneShotLaunch(CommandLine.arguments)

    /// The llama.cpp build a `runtime=` descriptor named, loaded after the gates and before
    /// the run measures anything. A build that does not load ends the run.
    private static func load(_ pin: LlamaPin?) {
        guard let pin else { return }
        do {
            try LlamaPins.load(pin)
        } catch {
            log("ERROR runtime= \(error.localizedDescription)")
            benchDone(HeadlessExit.failure)
            HeadlessExit.terminate(HeadlessExit.failure)
        }
    }

    /// ``isOneShot`` for `args`: every command but the resident `worker`. A refused
    /// invocation counts: it exits before anything else.
    nonisolated static func isOneShotLaunch(_ args: [String]) -> Bool {
        switch HeadlessCommand.parse(args) {
        case .success(.worker), .failure(.notHeadless): return false
        case .success, .failure: return true
        }
    }

    /// `jobRunner` / `jobStore` must be the app-wide instances so planner worker and UI
    /// jobs share one busy gate.
    static func startIfRequested(ws: PipetteWorkspace, jobRunner: JobRunner, jobStore: JobStore) {
        let args = CommandLine.arguments
        guard args.contains("headlessrun") else { return }
        // A `claim=` run receives its job here, in the launch arguments: the result's
        // `device_time_ms` counts from this instant.
        let launchedAt = ContinuousClock.now
        // A pipe is block-buffered by default, which hides output over `--console` until
        // the buffer fills or the process exits.
        setvbuf(stdout, nil, _IOLBF, 0)
        // Keep above `HeadlessCommand.parse` and every gate: a refusal must print it too.
        logDiagnostic(helloLine(pid: getpid()))

        let command: HeadlessCommand
        switch HeadlessCommand.parse(args) {
        case let .success(parsed):
            command = parsed
        case let .failure(error):
            let tokens = args.drop(while: { $0 != "headlessrun" }).dropFirst()
            log("ERROR \(error.message): \(tokens.joined(separator: " "))")
            benchDone(HeadlessExit.usage)
            // A distinct code lets a consumer retry a flaky run but not a malformed one.
            HeadlessExit.terminate(HeadlessExit.usage)
        }

        // Submission is the default, and `JobExecutor` silently ANDs it with the
        // registration gate, so an unregistered run would measure every cell and discard
        // every result.
        if command.submitsResults,
           !ResultSubmissionFeatureGate.canSubmitResults(registration: ws.registration.get()) {
            log("ERROR submit is on (the default) but this device has no registration: "
                + "run `headlessrun auth register organization=… contact-email=… "
                + "client-details=…` first, or pass submit=0 to record locally")
            benchDone(HeadlessExit.failure)
            HeadlessExit.terminate(HeadlessExit.failure)
        }

        HeadlessExit.finishOnSigterm()

        /// The process must exit so `devicectl … --console` returns; the app otherwise
        /// stays resident.
        func dispatch(_ handler: @escaping () async -> Bool) {
            dispatch(sentinelOnStderr: false, handler)
        }

        /// `sentinelOnStderr` is for commands whose stdout is a payload (`results show … |
        /// jq`). Everything else keeps it on stdout, which is all `pipette-ctl` scans.
        func dispatch(sentinelOnStderr: Bool, _ handler: @escaping () async -> Bool) {
            Task.detached(priority: .userInitiated) {
                let ok = await handler()
                benchDone(ok: ok, onStderr: sentinelOnStderr)
                await HeadlessExit.finish(ok ? HeadlessExit.ok : HeadlessExit.failure)
            }
        }

        switch command {
        case let .authRegister(input):
            dispatch { await AuthCommands.register(input, ws: ws) }

        case let .diagMemseq(models, batch, prefill):
            dispatch {
                await runMemSeq(modelNames: models, batch: batch, prefill: prefill, ws: ws)
                return true
            }

        case let .modelsList(format):
            dispatch { await ModelsCommands.list(format: format, ws: ws); return true }

        case let .modelsRm(name, repo):
            dispatch { await ModelsCommands.remove(name: name, repo: repo, ws: ws) }

        case let .runtimesList(format):
            dispatch { RuntimesCommands.list(format: format) }

        case let .benchmarksList(benchmarkType):
            dispatch { BenchmarksCommands.list(benchmarkType: benchmarkType, ws: ws); return true }

        case let .benchmarksShow(reference):
            dispatch { BenchmarksCommands.show(reference, ws: ws) }

        case .benchmarksInitLocal:
            dispatch { BenchmarksCommands.initLocal(ws: ws) }

        case let .authMe(local):
            dispatch { await AuthCommands.me(local: local, ws: ws) }

        case let .authSetDevice(deviceName, deviceFormFactor, clientDetails):
            dispatch {
                await AuthCommands.setDevice(deviceName: deviceName, deviceFormFactor: deviceFormFactor,
                                             clientDetails: clientDetails, ws: ws)
            }

        case let .authReset(force):
            dispatch { AuthCommands.reset(force: force, ws: ws) }

        case .authExport:
            dispatch(sentinelOnStderr: true) { AuthCommands.export(ws: ws) }

        case let .authImport(bundle, force):
            dispatch { AuthCommands.import(bundle: bundle, force: force, ws: ws) }

        case let .diagProbe(kind):
            dispatch {
                switch kind {
                case .enforcement: await AFMRuntime.enforcementProbe { log("diag probe \($0)") }
                case .cap: await AFMRuntime.capRespectProbe { log("diag probe \($0)") }
                case .generation: await AFMRuntime.generationTokenProbe { log("diag probe \($0)") }
                case .token: await AFMRuntime.tokenProbe { log("diag probe \($0)") }
                }
                return true
            }

        case let .resultsList(benchmark, benchmarkType, state, limit):
            dispatch {
                ResultsCommands.list(benchmark: benchmark, benchmarkType: benchmarkType, state: state,
                                     limit: limit, ws: ws)
                return true
            }

        case let .resultsShow(resultId):
            dispatch(sentinelOnStderr: true) {
                ResultsCommands.show(resultId: resultId, ws: ws)
            }

        case let .resultsDelete(resultId):
            dispatch { ResultsCommands.delete(resultId: resultId, ws: ws) }

        case let .sync(jobId, serverUrl):
            dispatch {
                await SyncCommands.run(jobId: jobId, serverUrl: serverUrl, ws: ws)
            }

        case .version:
            dispatch { runVersion(); return true }

        case .storageStatus:
            dispatch { await StorageCommands.status(ws: ws); return true }

        case let .storageGc(dryRun):
            dispatch { await StorageCommands.gc(dryRun: dryRun, ws: ws) }

        case .listJobs:
            dispatch { await JobCommands.list(ws: ws); return true }

        case let .removeJob(id):
            dispatch { await JobCommands.remove(jobId: JobId(id), ws: ws) }

        case let .runJob(id, scope):
            dispatch { await JobCommands.run(jobId: JobId(id), scope: scope, ws: ws) }

        case let .exportJob(id):
            dispatch(sentinelOnStderr: true) {
                await JobCommands.export(jobId: JobId(id), ws: ws)
            }

        case let .submitJob(id):
            dispatch { await JobCommands.submit(jobId: JobId(id), ws: ws) }

        case .status:
            dispatch { await runStatus(ws: ws); return true }

        case let .worker(options):
            // Does not exit: the session ends at `max-jobs=`, a 403, the app leaving the
            // foreground, or a kill.
            Task.detached(priority: .userInitiated) {
                await runPlannerWorkerSession(
                    options: options,
                    ws: ws,
                    jobRunner: jobRunner,
                    jobStore: jobStore
                )
            }

        case let .bareBench(runtime, batch, metrics, offsets, benchmarks, model, submit, llamaPin):
            load(llamaPin)
            let benchmarkIds = BenchmarksCommands.resolveBenchmarkIds(metrics: metrics, offsets: offsets,
                                                   explicit: benchmarks)
            Task.detached(priority: .userInitiated) {
                // `runSweep` prints its own BENCH_DONE line, so it bypasses `dispatch`.
                let ok = await BenchmarksCommands.runSweep(runtime: runtime, batch: batch,
                                   benchmarkIds: benchmarkIds,
                                   runCoherence: metrics.contains("coherence"),
                                   runCalibrate: metrics.contains("calibrate"),
                                   runPromptSeed: metrics.contains("promptseed"),
                                   offsets: offsets, modelMatch: model, submit: submit, ws: ws)
                await HeadlessExit.finish(ok ? HeadlessExit.ok : HeadlessExit.failure)
            }

        case let .modelsPull(selector):
            dispatch {
                guard let model = await BenchmarksCommands.resolve(selector, ws: ws) else { return false }
                return await ModelsCommands.pull(model, ws: ws)
            }

        case let .modelsDelete(selector):
            dispatch {
                guard let model = await BenchmarksCommands.resolve(selector, ws: ws) else { return false }
                return await MainActor.run { ModelsCommands.delete(model, ws: ws) }
            }

        case let .benchmarksRun(benchmark, model, runtime, batch, runtimeFlags, sync,
                                benchmarkFlags, llamaPin):
            load(llamaPin)
            Task.detached(priority: .userInitiated) {
                let ok = await BenchmarksCommands.run(spec: model, modelMatch: nil, quant: nil,
                                        runtime: runtime, batch: batch,
                                        runtimeFlags: runtimeFlags,
                                        benchmarkFlags: benchmarkFlags,
                                        benchmarks: [benchmark.description],
                                        metrics: [], offsets: [], submit: sync,
                                        ws: ws, http: .shared)
                await HeadlessExit.finish(ok ? HeadlessExit.ok : HeadlessExit.failure)
            }

        case let .bench(spec, model, quant, runtime, batch, runtimeFlags, benchmarks,
                        metrics, offsets, submit, benchmarkFlags, claim, llamaPin):
            load(llamaPin)
            Task.detached(priority: .userInitiated) {
                // `run` prints its own BENCH_DONE line, so it bypasses `dispatch`.
                let ok = await BenchmarksCommands.run(spec: spec, modelMatch: model, quant: quant, runtime: runtime,
                                        batch: batch, runtimeFlags: runtimeFlags,
                                        benchmarkFlags: benchmarkFlags,
                                        benchmarks: benchmarks, metrics: metrics,
                                        offsets: offsets, submit: submit, claim: claim,
                                        claimedAt: claim.map { _ in launchedAt }, ws: ws,
                                        http: .shared)
                await HeadlessExit.finish(ok ? HeadlessExit.ok : HeadlessExit.failure)
            }

        case let .afm(metrics, benchmarks, offsets, submit, claim, benchmarkFlags):
            // A probe token in `metrics` runs a diagnostic that records nothing and exits 0;
            // otherwise the benchmarks run through the normal job path.
            let benchmarkIds = BenchmarksCommands.resolveBenchmarkIds(metrics: metrics, offsets: offsets,
                                                   explicit: benchmarks)
            Task.detached(priority: .userInitiated) {
                setvbuf(stdout, nil, _IONBF, 0)
                let ok: Bool
                if metrics.contains("enforceprobe") {
                    await AFMRuntime.enforcementProbe { log("afm \($0)") }; benchDone(HeadlessExit.ok); ok = true
                } else if metrics.contains("capprobe") {
                    await AFMRuntime.capRespectProbe { log("afm \($0)") }; benchDone(HeadlessExit.ok); ok = true
                } else if metrics.contains("genprobe") {
                    await AFMRuntime.generationTokenProbe { log("afm \($0)") }; benchDone(HeadlessExit.ok); ok = true
                } else if metrics.contains("tokprobe") {
                    await AFMRuntime.tokenProbe { log("afm \($0)") }; benchDone(HeadlessExit.ok); ok = true
                } else {
                    ok = await BenchmarksCommands.runAFM(benchmarkIds: benchmarkIds, submit: submit,
                                                         claim: claim,
                                                         claimedAt: claim.map { _ in launchedAt },
                                                         benchmarkFlags: benchmarkFlags,
                                                         ws: ws, http: .shared)
                }
                await HeadlessExit.finish(ok ? HeadlessExit.ok : HeadlessExit.failure)
            }
        }
    }

    /// `print`, not `os.Logger`: these lines are a CLI contract read over `--console`.
    /// One `[HEADLESS] <group> <key=value>…` line per record, with a `<group> count=N`
    /// header before a list; derive the count from the rows, never maintain it beside them.
    nonisolated static func log(_ s: String) { print("[HEADLESS] \(s)") }

    /// Read from the arguments rather than set, so it is already true for anything that
    /// logs before `startIfRequested` runs.
    nonisolated static let isHeadless = CommandLine.arguments.contains("headlessrun")

    /// Also mirror `debug`. A global switch, so `HeadlessCommand.parse` drops the token
    /// before the grammar sees it.
    nonisolated static let isVerbose = CommandLine.arguments.contains("-v")
        || CommandLine.arguments.contains("--verbose")

    /// A `[HEADLESS]` line on stderr: the launch's hello line, payload commands' diagnostics
    /// and headless `AppLog` lines.
    /// Not the default because `pipette-ctl`'s `run_streaming_scanning` scans only stdout
    /// for `BENCH_DONE`.
    nonisolated static func logDiagnostic(_ s: String) {
        FileHandle.standardError.write(Data(diagnosticLine(s).utf8))
    }

    /// The bytes `logDiagnostic` writes: `pipette-ios-proxy` keys on the prefix (`APP_LINE`).
    nonisolated static func diagnosticLine(_ s: String) -> String {
        "[HEADLESS] \(s)\n"
    }

    /// Verbatim on stdout, so `results show … | jq` works: no prefix, and a newline only
    /// when the payload lacks one.
    nonisolated static func emitPayload(_ s: String) {
        FileHandle.standardOutput.write(Data((s.hasSuffix("\n") ? s : s + "\n").utf8))
    }

    nonisolated static func memSeqBenchmark(prefill: NonNegativeCount) -> BenchmarkDefinition {
        .maxMemoryUsage(MaxMemoryUsageBenchmark(
            benchmarkId: BenchmarkId(validated: "max_memory_usage_\(prefill)"),
            parameterPrefillTokens: prefill))
    }

    /// Authored as a job cell is, so each runtime gets its shipped flags: `batch` is
    /// llama's `n_ubatch` or MLX's prefill chunk.
    nonisolated static func memSeqSpec(model: Model, batch: UInt32,
                                       benchmark: BenchmarkDefinition) -> ClientRunSpec {
        ClientRunSpec.authored(
            benchmarkId: benchmark.benchmarkId.value,
            benchmarkType: benchmark.benchmarkType, model: model, nUbatch: batch)
    }

    /// `max_memory_usage` for each named model in one process, to expose memory carried
    /// from one cell into the next, through the same `RunCell.dispatchRun` `JobExecutor` uses.
    private static func runMemSeq(modelNames: [String], batch: UInt32, prefill: UInt32,
                                  ws: PipetteWorkspace) async {
        let available = await MainActor.run { ws.availableModels() }
        log("memseq start models=\(modelNames) batch=\(batch) prefill=\(prefill)")
        guard !modelNames.isEmpty else {
            log("memseq ERROR no models= given; have: "
                + available.map { "\($0.name)[\($0.engineLabel)]" }.joined(separator: ", "))
            return
        }
        guard let prefillTokens = try? NonNegativeCount(Int(prefill)) else {
            log("memseq ERROR prefill=\(prefill) is past \(Int32.max)")
            return
        }
        let benchmark = memSeqBenchmark(prefill: prefillTokens)
        for name in modelNames {
            guard let m = available.first(where: { $0.name == name }) else {
                log("memseq SKIP \(name): not on device"); continue
            }
            do {
                // Measured before this cell's settle gate: high here with a clean REPORTED
                // below is the gate working.
                let carriedIn = Double(ProcessMemory.physFootprintBytes()) / 1_048_576
                log(String(format: "memseq PRE %@ carriedIn=%.0fMB", name, carriedIn))
                let request = try await RunCell.prepare(
                    spec: memSeqSpec(model: m.source, batch: batch, benchmark: benchmark),
                    benchmark: benchmark,
                    ws: ws, coordinator: DownloadCoordinator.shared)
                let (r, _) = try await RunCell.dispatchRun(request, ws: ws,
                                                        readiness: ResolvedReadiness(),
                                                        waitUntilReady: { _ in .ready })
                if case let .maxMemoryUsage(host, _, _) = r.resultData {
                    let ran = r.runtimeFlags.map(RuntimeFlagRef.init)
                    log(String(format: "memseq REPORTED %@ [%@] host=%.0fMB",
                               name, m.engineLabel, Double(host) / 1_048_576)
                        + (ran?.ctxSize.map { " ctx=\($0)" } ?? "")
                        + (ran?.nUbatch.map { " n_ubatch=\($0)" } ?? ""))
                }
            } catch {
                log("memseq ERROR \(name): \(error)")
            }
        }
    }

    // MARK: - Status

    /// `worker` and `settings run`: the Settings toggle's claim loop, kept resident.
    ///
    /// The `worker start` line is the readiness signal: `BENCH_DONE` arrives only when the
    /// session ends, which `max-jobs=` makes reachable.
    private static func runPlannerWorkerSession(
        options: WorkerArgs,
        ws: PipetteWorkspace,
        jobRunner: JobRunner,
        jobStore: JobStore
    ) async {
        let started = await MainActor.run { () -> Bool in
            PlannerWorker.shared.setEnabled(
                true,
                ws: ws,
                jobRunner: jobRunner,
                jobStore: jobStore,
                options: options
            )
            // `setEnabled` starts nothing when not registered.
            guard PlannerWorker.shared.isRunning else {
                log("worker ERROR this device has no registration: "
                    + "run `headlessrun auth register organization=… contact-email=… "
                    + "client-details=…` first")
                return false
            }
            let text = PlannerWorker.shared.statusText
            // The resolved knobs, so a console capture shows how the worker was paced.
            log("worker start status=\(text) "
                + "idle_secs=\(Int(options.idleSecs)) "
                + "idle_jitter_secs=\(Int(options.idleJitterSecs)) "
                + "heartbeat_secs=\(options.heartbeatSecs.map { String(Int($0)) } ?? "half_time_window") "
                + "max_jobs=\(options.maxJobs == 0 ? "unlimited" : String(options.maxJobs)) "
                + "skip_profile_refresh=\(options.skipProfileRefresh ? "yes" : "no")")
            return true
        }
        guard started else {
            benchDone(HeadlessExit.failure)
            HeadlessExit.terminate(HeadlessExit.failure)
        }
        // The exit code comes from the session outcome, not from the loop stopping: a 403
        // stops it too, and a fleet script must tell "ran its jobs" from "not approved".
        while !Task.isCancelled {
            try? await Task.sleep(for: .seconds(30))
            let (text, outcome) = await MainActor.run {
                (PlannerWorker.shared.statusText, PlannerWorker.shared.sessionOutcome)
            }
            log("worker status=\(text)")
            guard let outcome else { continue }
            log("worker finished \(outcomeFields(outcome)) status=\(text)")
            let code = outcome.isSuccess ? HeadlessExit.ok : HeadlessExit.failure
            benchDone(code)
            await HeadlessExit.finish(code)
        }
    }

    private static func outcomeFields(_ outcome: PlannerWorker.SessionOutcome) -> String {
        switch outcome {
        case let .completed(jobs): return "outcome=completed jobs=\(jobs)"
        case .stopped: return "outcome=stopped"
        case .notApproved: return "outcome=not_approved"
        case .notStarted: return "outcome=not_started"
        // The reason is prose for the log line above, not a value for this one.
        case .failed: return "outcome=failed"
        }
    }

    /// `client_version` (what every submission sends) and the build flags that change what a
    /// measurement means: a run gated by SoC die temperature is not comparable to one gated
    /// by `thermalState`, nor one after a reclaim-ring flush to one without.
    private static func runVersion() {
        log("version client_version=\(Bundle.main.appVersionDisplayString) "
            + "private_thermal=\(BuildFlavor.hasPrivateThermal ? "on" : "off") "
            + "private_memory=\(BuildFlavor.hasPrivateMemory ? "on" : "off") "
            + "thermal=\(BuildFlavor.thermalDescription) "
            + "memory=\(BuildFlavor.memoryDescription)")
    }

    private static func runStatus(ws: PipetteWorkspace) async {
        if let reg = ws.registration.get() {
            // `mirrored=`: whether the registration record survives a reinstall (the key
            // is always in the Keychain).
            log("status registered client_id=\(reg.clientId.value) "
                + "server_url=\(reg.serverUrl.value) "
                + "mirrored=\(ws.registration.isRegistrationMirrored ? "yes" : "no")")
        } else {
            log("status unregistered")
        }
        let modelCount = await MainActor.run { ws.availableModels().count }
        log("status models=\(modelCount)")
        let jobs = ws.loadAllJobManifests()
        let byStatus = Dictionary(grouping: jobs, by: { $0.status.rawValue })
            .map { "\($0.key)=\($0.value.count)" }.sorted().joined(separator: " ")
        log("status jobs total=\(jobs.count)\(byStatus.isEmpty ? "" : " " + byStatus)")
    }
}
