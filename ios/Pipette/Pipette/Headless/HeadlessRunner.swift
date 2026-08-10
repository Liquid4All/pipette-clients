import Foundation
import MLXLMCommon

/// The headless process's exit codes, mirroring the CLI's split: 0 succeeded, 1 the work
/// failed, 2 the invocation was malformed. Named so a call site cannot write `exit(2)`
/// and leave the reader to guess which 2 it is.
nonisolated enum HeadlessExit {
    static let ok: Int32 = 0
    static let failure: Int32 = 1
    static let usage: Int32 = 2
}

/// Headless CLI front-end over the same controllers the SwiftUI views bind, so
/// every UI-exposed control is scriptable via `devicectl process launch`
/// (install → open) with no UI taps. One-shot process model: each launch runs
/// one command to completion, prints its `[HEADLESS] …` lines to stdout
/// (captured by `--console`), ends with the `BENCH_DONE` sentinel, and exits
/// `HeadlessExit.ok` / `.failure` / `.usage`. Every terminal path emits the sentinel,
/// including the early refusals — a console consumer keyed on it hangs otherwise.
///
/// Note: only this runner's own `[HEADLESS] …` / result lines go to stdout. The
/// rest of the app's diagnostics (e.g. `AppLog.mlx` `[MLXMEM] …` memory figures)
/// are `os.Logger`, which the unified log captures but `--console` does not — to
/// see those during a headless run, stream them in parallel:
///   log stream --predicate 'subsystem == "ai.liquid.liquid-pipette"' --level info
///
/// Verbs (pass after the bundle id; the `headlessrun` keyword enables it —
/// argv parsing lives in `HeadlessCommand.parse`):
///
///   `submit` defaults to 1 (contribute) for catalog-resolved runs; pass
///   `submit=0` to record locally only. A `spec=` run is opt-in instead
///   (default 0) — the catalog never validated that model. A submitting run on
///   an unregistered device is refused before the first cell.
///
///   Benchmarks — `bench` verb (resolve a named model + runtime, download it if
///   it isn't on device, then run through the same `JobLauncher`/`JobExecutor`
///   path as a UI-started job):
///     headlessrun bench model=<name/repo substring> runtime=llama|mlx [quant=<token>]
///             (benchmarks=<id,...> | metrics=<...> offsets=<...>) [batch=512] [submit=0|1]
///             // friendly catalog selection; resolves one CatalogEntry (none/ambiguous is an error)
///     headlessrun bench spec='<json>' benchmarks=<id,...>
///             // full Model definition decoded by Model's own Codable; the engine is
///             // derived from the decoded format, so runtime/quant/model are ignored.
///             // e.g. spec='{"type":"hf_mlx","org":"mlx-community","repo_name":"Qwen3.5-0.8B-4bit"}'
///     headlessrun bench runtime=afm (benchmarks=<id,...> | metrics=<...> offsets=<...>) [submit=0|1]
///             // Apple's on-device foundation model — no model, no download; runs
///             // through the normal record/submit job path like the other runtimes.
///
///   Benchmarks (the bare form — New Job's start control; runs through
///   `JobLauncher`/`JobExecutor` exactly like a UI-started job over whatever
///   matching model is already on device — does NOT download):
///     headlessrun runtime=mlx|llama [batch=512]
///             [metrics=prefill,decode,maxmem] [offsets=256,512,1024,2048,4096]
///             [benchmarks=<id,id,...>]   // explicit catalog ids; overrides metrics×offsets
///             [model=<name-substring>] [submit=0|1]
///
///   Models (the Models tab):
///     headlessrun models                                   // list discovered models
///     headlessrun models rm name=<exact> | repo=<org/name> // delete downloaded model(s)
///     headlessrun mlxget repo=<org/name> [prefix=<sub>]    // download an MLX model dir
///     headlessrun ggufget repo=<org/name-GGUF> quant=<Q4_0> | file=<name.gguf>
///
///   Jobs (the Jobs tab / job detail page):
///     headlessrun jobs                                     // list job manifests
///     headlessrun job run id=<jobId> [scope=failed|cancelled]
///                              // retry failed / resume cancelled cells (default: cancelled)
///     headlessrun job export id=<jobId>                    // print the results CSV
///     headlessrun job submit id=<jobId>                    // upload unsubmitted completed results
///     headlessrun job rm id=<jobId>                        // delete the job and its results
///
///   Inspect (read-only; nothing runs):
///     headlessrun runtimes                                 // compiled-in engines + build ids
///     headlessrun benchmarks [type=<benchmark_type>]       // the synced catalog
///     headlessrun benchmarks show benchmark=<id>           // one benchmark, or non-zero exit
///     headlessrun storage status                           // quota, usage, model count
///
///   Sync (pull the catalog, then submit what is pending):
///     headlessrun sync [job=<jobId>]
///
///   Device:
///     headlessrun version                                  // app version + build flags (thermal)
///     headlessrun status                                   // registration, model & job counts, worker
///     headlessrun settings                                 // print Settings prefs (worker, …)
///     headlessrun settings set worker=on|off               // planner claim-loop preference only
///     headlessrun settings run                             // worker=on + start claim loop (stays up)
///     headlessrun worker                                   // the crate's spelling for `settings run`
///     headlessrun register org=<name> email=<addr> [server=<url>]
///             // org/email are required — they are recorded with the device identity
///             // and registering is not reversible. server defaults to production.
///
///   Diagnostics:
///     headlessrun diag memseq models=<name>,<name>,... [batch=512]
///     headlessrun diag probe kind=token|generation|cap|enforcement
///     headlessrun memseq models=<name>,<name>,... [batch=512]   // former spelling
///     headlessrun runtime=mlx|llama metrics=coherence|calibrate|promptseed
///
/// `metrics=promptseed` is a self-check (not a benchmark): it confirms
/// `PromptSeed.buildPromptText` lands on exactly each target token count under the
/// model's own tokenizer, logging `promptseed <runtime> target=N got=M OK|MISMATCH`.
///
/// Example:
///   xcrun devicectl device process launch --device <UDID> --console \
///     ai.liquid.liquid-pipette headlessrun runtime=mlx batch=512 \
///     metrics=prefill,decode,maxmem offsets=256,512,1024,2048,4096
///   xcrun devicectl device process launch --device <UDID> --console \
///     ai.liquid.liquid-pipette headlessrun bench model=Qwen3.5-0.8B runtime=mlx \
///     benchmarks=prefill_throughput_512   // downloads the model first if missing
///
/// The benchmarks to run are resolved (`resolveBenchmarkIds`) from either an
/// explicit `benchmarks=` list of catalog ids, or the cartesian product of
/// `metrics` × `offsets`. New benchmark kinds are exposed to the CLI by adding a
/// single entry to `metricBuilders` — the one place that maps a CLI metric token
/// to its catalog benchmark id.
enum HeadlessRunner {
    /// The run's terminal line: `BENCH_DONE <exit-code> [detail]`.
    ///
    /// Numeric because that is what `pipette-plan` scans for (`scan_sentinel`): a bare
    /// `BENCH_DONE` reads there as status 0, so a refusal — a malformed argument, a submit
    /// with no registration, a model that would not resolve — was recorded as a completed
    /// cell, never retried, and indistinguishable from one that measured. `devicectl` does
    /// not pass the app's exit code through, so this line is the only channel that carries
    /// it. Anything after the code is free text: the scanner reads the first token.
    nonisolated static func benchDone(_ code: Int32, _ detail: String? = nil,
                                     onStderr: Bool = false) {
        let line = benchDoneLine(code, detail)
        if onStderr { logDiagnostic(line) } else { log(line) }
    }

    /// `benchDone` for a handler that reports success as a `Bool`.
    nonisolated static func benchDone(ok: Bool, onStderr: Bool = false) {
        benchDone(ok ? HeadlessExit.ok : HeadlessExit.failure, onStderr: onStderr)
    }

    /// The line itself, so a test can assert the contract without capturing stdout.
    nonisolated static func benchDoneLine(_ code: Int32, _ detail: String? = nil) -> String {
        detail.map { "BENCH_DONE \(code) \($0)" } ?? "BENCH_DONE \(code)"
    }

    nonisolated static func benchDoneLine(ok: Bool) -> String {
        benchDoneLine(ok ? HeadlessExit.ok : HeadlessExit.failure)
    }

    /// Call from `PipetteApp.init`. No-ops unless `headlessrun` is in the launch args.
    /// `jobRunner` / `jobStore` must be the app-wide instances so planner worker
    /// and UI jobs share one busy gate.
    static func startIfRequested(storage: Storage, jobRunner: JobRunner, jobStore: JobStore) {
        let args = CommandLine.arguments
        guard args.contains("headlessrun") else { return }
        // Line-buffer stdout so each log line flushes immediately over the
        // `devicectl … --console` pipe (default is block buffering to a pipe,
        // which hides all output until the buffer fills or the process exits).
        setvbuf(stdout, nil, _IOLBF, 0)

        let command: HeadlessCommand
        switch HeadlessCommand.parse(args) {
        case let .success(parsed):
            command = parsed
        case let .failure(error):
            let tokens = args.drop(while: { $0 != "headlessrun" }).dropFirst()
            log("ERROR \(error.message): \(tokens.joined(separator: " "))")
            benchDone(HeadlessExit.usage)
            // A usage error is not a failed run. Exiting 2 lets a console consumer retry
            // a flaky benchmark and fix a malformed invocation, rather than treating both
            // as "the run failed".
            exit(HeadlessExit.usage)
        }

        // Submission is the default, and `JobExecutor` silently ANDs it with the
        // registration gate — so on an unregistered device an unchecked run would
        // measure every cell and then discard every result. Refuse before the
        // first cell instead of after the last.
        if command.submitsResults,
           !ResultSubmissionFeatureGate.canSubmitResults(registration: storage.identity.getRegistration()) {
            log("ERROR submit is on (the default) but this device has no registration: "
                + "run `headlessrun register` first, or pass submit=0 to record locally")
            benchDone(HeadlessExit.failure)
            exit(HeadlessExit.failure)
        }

        /// Run one async handler to completion and self-terminate with its
        /// exit code. The headless process must exit so `devicectl … --console`
        /// returns and stdout flushes; the app otherwise stays resident.
        func dispatch(_ handler: @escaping () async -> Bool) {
            dispatch(sentinelOnStderr: false, handler)
        }

        /// `sentinelOnStderr` is for the commands whose stdout carries a payload: the
        /// sentinel is a control line, and `results show … | jq` must not be handed one.
        /// Everything else keeps it on stdout, because `pipette-plan` scans stdout alone
        /// for it and would otherwise never see a run finish.
        func dispatch(sentinelOnStderr: Bool, _ handler: @escaping () async -> Bool) {
            Task.detached(priority: .userInitiated) {
                let ok = await handler()
                benchDone(ok: ok, onStderr: sentinelOnStderr)
                exit(ok ? HeadlessExit.ok : HeadlessExit.failure)
            }
        }

        switch command {
        case let .register(server, org, email, preauth, clientDetails, deviceName):
            startRegister(server: server, org: org, email: email, preauth: preauth,
                          clientDetails: clientDetails, deviceName: deviceName,
                          storage: storage)

        case let .memSeq(models, batch):
            dispatch { await runMemSeq(modelNames: models, batch: batch, storage: storage); return true }

        case let .listModels(format):
            dispatch { await ModelCommands.list(format: format, storage: storage); return true }

        case let .removeModel(name, repo):
            dispatch { await ModelCommands.remove(name: name, repo: repo, storage: storage) }

        case .listRuntimes:
            dispatch { RuntimeCommands.list(); return true }

        case let .listBenchmarks(type):
            dispatch { BenchmarkCommands.list(type: type, storage: storage); return true }

        case let .showBenchmark(id):
            dispatch { BenchmarkCommands.show(id: id, storage: storage) }

        case .initLocalBenchmarks:
            dispatch { BenchmarkCommands.initLocal(storage: storage) }

        case .authMe:
            dispatch { await AuthCommands.me(storage: storage) }

        case let .authReset(force):
            dispatch { AuthCommands.reset(force: force, storage: storage) }

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

        case let .listResults(benchmark, type, state, limit):
            dispatch {
                ResultCommands.list(benchmark: benchmark, type: type, state: state,
                                               limit: limit, storage: storage)
                return true
            }

        case let .showResult(id):
            dispatch(sentinelOnStderr: true) {
                ResultCommands.show(id: id, storage: storage)
            }

        case let .deleteResult(id):
            dispatch { ResultCommands.delete(id: id, storage: storage) }

        case let .sync(jobId):
            dispatch { await SyncCommands.run(jobId: jobId, storage: storage) }

        case .version:
            dispatch { runVersion(); return true }

        case .storageStatus:
            dispatch { await StorageCommands.status(storage: storage); return true }

        case let .storageGc(dryRun):
            dispatch { await StorageCommands.gc(dryRun: dryRun, storage: storage) }

        case .listJobs:
            dispatch { await JobCommands.list(storage: storage); return true }

        case let .removeJob(id):
            dispatch { await JobCommands.remove(jobId: JobId(id), storage: storage) }

        case let .runJob(id, scope):
            dispatch { await JobCommands.run(jobId: JobId(id), scope: scope, storage: storage) }

        case let .exportJob(id):
            dispatch { await JobCommands.export(jobId: JobId(id), storage: storage) }

        case let .submitJob(id):
            dispatch { await JobCommands.submit(jobId: JobId(id), storage: storage) }

        case .status:
            dispatch { await runStatus(storage: storage); return true }

        case let .settings(op):
            switch op {
            case .show, .setWorker:
                // Preference get/set only — process still exits (one-shot).
                dispatch {
                    switch op {
                    case .show:
                        log("settings worker=\(LocalStorage.plannerWorkerEnabled ? "on" : "off")")
                    case let .setWorker(on):
                        LocalStorage.plannerWorkerEnabled = on
                        log("settings worker=\(on ? "on" : "off")")
                    case .run:
                        break
                    }
                    return true
                }
            case .run:
                // Enable + start PlannerWorker; do not exit — device stays a worker
                // until the process is killed (devicectl stop / app swipe-away).
                Task.detached(priority: .userInitiated) {
                    await runPlannerWorkerSession(
                        storage: storage,
                        jobRunner: jobRunner,
                        jobStore: jobStore
                    )
                }
            }

        case let .bareBench(runtime, batch, nGpuLayers, threads, metrics, offsets, benchmarks,
                            model, submit):
            let benchmarkIds = resolveBenchmarkIds(metrics: metrics, offsets: offsets,
                                                   explicit: benchmarks)
            Task.detached(priority: .userInitiated) {
                // `run` owns its own BENCH_DONE line (it appends the final job
                // status), so it is not routed through `dispatch`.
                let ok = await run(runtime: runtime, batch: batch, nGpuLayers: nGpuLayers,
                                   threads: threads, benchmarkIds: benchmarkIds,
                                   runCoherence: metrics.contains("coherence"),
                                   runCalibrate: metrics.contains("calibrate"),
                                   runPromptSeed: metrics.contains("promptseed"),
                                   offsets: offsets, modelMatch: model, submit: submit, storage: storage)
                exit(ok ? HeadlessExit.ok : HeadlessExit.failure)
            }

        case let .pullModel(selector):
            dispatch {
                guard let model = await resolve(selector, storage: storage) else { return false }
                return await ModelCommands.pull(model, storage: storage)
            }

        case let .deleteModel(selector):
            dispatch {
                guard let model = await resolve(selector, storage: storage) else { return false }
                return await MainActor.run { ModelCommands.delete(model, storage: storage) }
            }

        case let .benchmarksRun(benchmark, model, runtime, batch, nGpuLayers, threads, sync,
                                readiness):
            Task.detached(priority: .userInitiated) {
                // The named model is authoritative and the benchmark is one id, so the
                // catalog match and the metrics x offsets expansion are both bypassed.
                let ok = await runBench(spec: model, modelMatch: nil, quant: nil,
                                        runtime: runtime, batch: batch,
                                        nGpuLayers: nGpuLayers, threads: threads,
                                        benchmarks: [benchmark],
                                        metrics: [], offsets: [], submit: sync,
                                        readiness: readiness, storage: storage)
                exit(ok ? HeadlessExit.ok : HeadlessExit.failure)
            }

        case let .bench(spec, model, quant, runtime, batch, nGpuLayers, threads, benchmarks,
                        metrics, offsets, submit):
            Task.detached(priority: .userInitiated) {
                // `runBench` delegates to `run`/`runAFM`, which own their own
                // BENCH_DONE line, so it is not routed through `dispatch`.
                let ok = await runBench(spec: spec, modelMatch: model, quant: quant, runtime: runtime,
                                        batch: batch, nGpuLayers: nGpuLayers, threads: threads,
                                        benchmarks: benchmarks, metrics: metrics,
                                        offsets: offsets, submit: submit, storage: storage)
                exit(ok ? HeadlessExit.ok : HeadlessExit.failure)
            }

        case let .afm(metrics, benchmarks, offsets, submit):
            // Apple's on-device foundation model. A `metrics` probe token runs a direct
            // diagnostic (no pass/fail → exit 0, nothing recorded); otherwise the
            // resolved catalog benchmarks run through the normal job path so results
            // persist and submission (on by default) auto-submits. Only the latter
            // needs a registration, so a probe run is never rejected for lacking one.
            let benchmarkIds = resolveBenchmarkIds(metrics: metrics, offsets: offsets,
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
                    ok = await runAFM(benchmarkIds: benchmarkIds, submit: submit, storage: storage)
                }
                exit(ok ? HeadlessExit.ok : HeadlessExit.failure)
            }
        }
    }

    /// Headless device registration. Generates the signing keypair on-device
    /// (private key stays in the Keychain so submissions can be signed),
    /// registers the public key with the collector, and persists the result.
    ///
    /// `server` defaults to the production collector; pass a full https URL to
    /// override. `org` and `email` are required by the parser — they are recorded with
    /// the device identity and cannot be corrected afterwards without re-registering,
    /// which rotates the signing key.
    private static func startRegister(server: String?, org: String, email: String,
                                      preauth: String?, clientDetails: String?,
                                      deviceName: String?, storage: Storage) {
        // A non-empty `server=` that isn't an http(s) URL is almost certainly a
        // typo; silently falling back to production would register against the
        // wrong collector, so reject it instead.
        let serverArg = server ?? ""
        let serverUrl: String
        if serverArg.isEmpty {
            serverUrl = CollectorEndpoint.productionURL
        } else if serverArg.lowercased().hasPrefix("http") {
            serverUrl = serverArg
        } else {
            log("REGISTER ERROR invalid server=\(serverArg) (must be an http(s) URL)")
            // Emit the sentinel even here: a console consumer waits for `BENCH_DONE`, so
            // an early exit without one hangs it until the launch times out.
            benchDone(HeadlessExit.usage)
            exit(HeadlessExit.usage)
        }
        // Re-registering mints a fresh keypair and overwrites the existing
        // identity; warn so a stray re-run doesn't silently rotate the key.
        if let existing = storage.identity.getRegistration() {
            log("register WARN device already registered clientId=\(existing.clientId.value); re-registering rotates the signing key")
        }
        Task.detached(priority: .userInitiated) {
            // The key itself is never logged — it is a credential, and one that the
            // server spends on this call.
            log("register start server=\(serverUrl) org=\(org) email=\(email) "
                + "preauth=\(preauth == nil ? "no" : "yes")")
            do {
                // Labels are written before the call, as the crate keeps them independent of
                // it: a register that fails still leaves the device named.
                if let deviceName {
                    var labels = storage.identity.getDeviceLabels()
                    labels.deviceName = deviceName
                    try? storage.identity.putDeviceLabels(labels)
                }
                let r = try await RegistrationService.register(
                    serverUrl: ServerURL(serverUrl), organization: org, contactEmail: email,
                    preauthKey: preauth, clientDetails: clientDetails, storage: storage)
                log("REGISTERED clientId=\(r.clientId.value) status=\(r.status) server=\(serverUrl)")
                benchDone(HeadlessExit.ok)
                exit(HeadlessExit.ok)
            } catch {
                log("REGISTER ERROR \(error)")
                benchDone(HeadlessExit.failure)
                exit(HeadlessExit.failure)
            }
        }
    }

    /// Run the requested catalog benchmarks against the Apple foundation model through
    /// the *same* `JobLauncher`/`JobExecutor` path a UI-started (or `bench`) job uses:
    /// build one `JobCell` per benchmark with `source: .appleFoundation` (bare — no path,
    /// no repo), `modelName: AFMRuntime.submissionModelName`, and launch. The executor
    /// records each result's payload to disk and — when `submit=1` sets the job's
    /// contribute flag — auto-submits it, exactly like the file-backed runtimes. AFM
    /// logs its own `[AFM] …` breadcrumbs. Returns `true` only if the job completed —
    /// drives the headless exit code so a scripted caller can key off `$?`.
    private static func runAFM(benchmarkIds: [String], submit: Bool, storage: Storage) async -> Bool {
        setvbuf(stdout, nil, _IONBF, 0)   // unbuffered → per-token progress shows live
        log("afm availability=\(AFMRuntime.availabilityText()) benchmarks=\(benchmarkIds) submit=\(submit)")

        // Build the AFM job cells. `source` is the bare `.appleFoundation`, which the
        // executor switches on to skip path resolution and the memory gate — AFM has no
        // weights on disk. Each cell's type comes from the parsed benchmark definition,
        // matching the other bench paths.
        let cells = JobCell.pending(benchmarkIds: benchmarkIds, for: .appleFoundation) {
            log("skip \($0): unrecognized benchmark id")
        }
        guard !cells.isEmpty else {
            log("afm ERROR no benchmarks resolved from \(benchmarkIds)")
            benchDone(HeadlessExit.failure); return false
        }

        // Launch through the same `JobLauncher` sequence the New Job screen and the
        // other bench paths use — including the executor's end-of-run auto-submit,
        // which only fires when `contributeResults` is set (i.e. `submit=1`). AFM has
        // no load knobs, so the ngl / ctx / batch params are inert placeholders.
        let final: JobManifest? = await withCheckedContinuation { cont in
            Task { @MainActor in
                let launched = JobLauncher.launch(
                    cells: cells, nGpuLayers: 99, contextSize: 4096, prefillBatch: 512,
                    contributeResults: submit, jobRunner: JobRunner(), jobStore: JobStore(storage: storage), storage: storage,
                    onFinish: { cont.resume(returning: $0) })
                if launched == nil { cont.resume(returning: nil) }
            }
        }
        guard let final else {
            log("afm ERROR job launch blocked: another job is running")
            benchDone(HeadlessExit.failure); return false
        }
        await reportResults(final, submitRequested: submit, storage: storage)
        benchDone(final.status == .completed ? HeadlessExit.ok : HeadlessExit.failure,
                  "\(final.status)")
        return final.status == .completed
    }

    // MARK: - Bench (the verb: resolve → download-if-missing → run)

    /// `bench`: resolve a model from a `spec=<json>` definition or the friendly
    /// catalog selectors, download it if it isn't on device, then run the resolved
    /// benchmarks through the same `run` (Engine/JobLauncher) path the bare form uses.
    /// `runtime=afm` skips resolution/download and benchmarks Apple's on-device
    /// foundation model. Returns `true` on success, `false` on any failure (drives
    /// the headless exit code); the delegated `run`/`runAFM` own the BENCH_DONE line,
    /// so the early-failure paths here print it themselves.
    /// The model a selector names: the one it carries, or the installed model its digest
    /// prefix matches.
    private static func resolve(_ selector: HeadlessCommand.ModelSelector,
                                storage: Storage) async -> Model? {
        switch selector {
        case let .model(model): return model
        case let .digest(prefix):
            return await MainActor.run { modelMatching(digest: prefix, storage: storage) }
        }
    }

    /// The one installed model whose descriptor digest starts with `prefix`.
    ///
    /// Ambiguity is an error rather than a pick, as upstream: two entries sharing a prefix
    /// are two different artifacts, and guessing would bench the wrong one.
    private static func modelMatching(digest prefix: String, storage: Storage) -> Model? {
        let installed = storage.modelStore.list().map(\.declared)
        let matched = installed.filter {
            ((try? Descriptor.digest($0.withoutAuthToken)) ?? "").hasPrefix(prefix)
        }
        switch matched.count {
        case 1: return matched[0]
        case 0:
            log("bench ERROR no installed model has a descriptor digest starting `\(prefix)` "
                + "(`headlessrun models` shows the installed set)")
            return nil
        default:
            let names = matched.map(\.artifactName).joined(separator: ", ")
            log("bench ERROR digest `\(prefix)` is ambiguous across \(matched.count) models "
                + "(\(names)); use more characters")
            return nil
        }
    }

    private static func runBench(spec: HeadlessCommand.ModelSelector?, modelMatch: String?,
                                 quant: String?,
                                 runtime: RuntimeType, batch: Int,
                                 nGpuLayers: UInt32?, threads: UInt32?,
                                 benchmarks: [String], metrics: [String], offsets: [Int],
                                 submit: Bool, readiness: ReadinessPolicy = .init(),
                                 storage: Storage) async -> Bool {
        let benchmarkIds = resolveBenchmarkIds(metrics: metrics, offsets: offsets, explicit: benchmarks)
        guard !benchmarkIds.isEmpty else {
            log("bench ERROR no benchmarks resolved from metrics=\(metrics) offsets=\(offsets) benchmarks=\(benchmarks)")
            benchDone(HeadlessExit.failure); return false
        }

        // AFM: no model to resolve or download — run the on-device foundation model
        // through the same recording job path, honoring `submit`.
        if runtime == .appleFoundation { return await runAFM(benchmarkIds: benchmarkIds, submit: submit, storage: storage) }

        // Resolve the exact `Model` to bench: an explicit `spec=` wins; otherwise pick
        // the single catalog row matching runtime + model substring + quant.
        let target: Model
        if let spec {
            switch spec {
            case let .model(model):
                target = model
            case let .digest(prefix):
                guard let matched = modelMatching(digest: prefix, storage: storage) else {
                    benchDone(HeadlessExit.failure); return false
                }
                target = matched
            }
        } else {
            // AFM returned above, so `runtime` here is `.mlx` or `.llamacppIosPipette`; match a
            // catalog row's format to the requested engine.
            let match = modelMatch ?? ""
            let candidates = CatalogEntry.catalog.filter { entry in
                let nameMatches = match.isEmpty
                    || entry.name.localizedCaseInsensitiveContains(match)
                    || entry.repoIdentifier.localizedCaseInsensitiveContains(match)
                let quantMatches = (quant ?? "").isEmpty
                    || entry.quant?.caseInsensitiveCompare(quant!) == .orderedSame
                return runtime.matches(entry.source) && nameMatches && quantMatches
            }
            let runtimeTag = runtime.rawValue
            guard !candidates.isEmpty else {
                let available = CatalogEntry.catalog
                    .filter { runtime.matches($0.source) }
                    .map { "\($0.name)[\($0.quant ?? "-")]" }.joined(separator: ", ")
                log("bench ERROR no catalog model matches model=\(match) quant=\(quant ?? "-"); "
                    + "available for \(runtimeTag): \(available)")
                benchDone(HeadlessExit.failure); return false
            }
            guard candidates.count == 1 else {
                let names = candidates.map { "\($0.name)[\($0.quant ?? "-")]" }.joined(separator: ", ")
                log("bench ERROR ambiguous (\(candidates.count)): \(names); narrow with quant=")
                benchDone(HeadlessExit.failure); return false
            }
            target = candidates[0].source
        }

        // The engine to run is derived from the resolved model's format, so a `spec=`
        // is authoritative even if `runtime=` disagreed.
        let targetIsMLX: Bool = { if case .mlx = target { return true }; return false }()

        // AFM returned above, so a resolved download target always has a repo.
        let repoLabel = target.repo?.description ?? "?"

        // Find-or-fetch through the one store entry point: a hit returns without a
        // transfer, a miss downloads and must resolve afterwards. Formats differ inside
        // `ensureModel`, so `bench` no longer carries a per-format download coordinate.
        log("bench ensuring \(repoLabel)")
        let transfer = DownloadProgressLog(label: "bench", what: repoLabel)
        do {
            _ = try await ensureModel(target, storage: storage, coordinator: .shared,
                                      progress: transfer.report)
            transfer.finish()
        } catch {
            log("bench FAILED \(repoLabel): \(error.localizedDescription)")
            benchDone(HeadlessExit.failure); return false
        }

        // The discovered file whose source is exactly this `Model`, read on the
        // MainActor. Explicit return type so the nested `.first(where:)` isn't
        // mistaken for `MainActor.run`'s own closure argument.
        func discover() async -> DiscoveredModel? {
            await MainActor.run { () -> DiscoveredModel? in
                storage.availableModels().first(where: { $0.source == target })
            }
        }

        // The store resolved it, so a discovery miss here is the two disagreeing — worth
        // its own line rather than folding into the ensure failure above.
        guard let file = await discover() else {
            log("bench NOT-DISCOVERED \(repoLabel)")
            benchDone(HeadlessExit.failure); return false
        }
        // The engine is the resolved target's format (a `spec=` is authoritative even
        // if `runtime=` disagreed); AFM returned earlier, so it's mlx or llama.cpp.
        return await run(runtime: targetIsMLX ? .mlxIosPipette : .llamacppIosPipette, batch: batch,
                         nGpuLayers: nGpuLayers, threads: threads,
                         benchmarkIds: benchmarkIds,
                         runCoherence: false, runCalibrate: false, runPromptSeed: false,
                         offsets: offsets, resolved: file, modelMatch: nil,
                         submit: submit, readiness: readiness, storage: storage)
    }

    // Deliberately `print`, not `os.Logger`: these `[HEADLESS] …` lines are a CLI
    // contract captured over `devicectl … --console` stdout, not diagnostic logging.
    // `nonisolated` because handlers log from detached tasks and `print` is
    // thread-safe (stdout is line-buffered by `startIfRequested`).
    /// One `[HEADLESS] <group> <key=value>…` line per record, with a `<group> count=N`
    /// header before a list — more machine-readable than the CLI's tables, and the reason
    /// the count must be derived from the rows rather than maintained beside them.
    nonisolated static func log(_ s: String) { print("[HEADLESS] \(s)") }

    /// Whether this process is a CLI run. Read rather than set, so it is already
    /// true for anything logging before `startIfRequested` gets to run; `static let`
    /// makes the one-time evaluation thread-safe. `AppLog` uses it to decide whether
    /// a diagnostic has a terminal to reach.
    nonisolated static let isHeadless = CommandLine.arguments.contains("headlessrun")

    /// `-v` / `--verbose`: also mirror `debug`. A global switch rather than a verb's
    /// parameter, so `HeadlessCommand.parse` drops the token before the grammar sees it.
    nonisolated static let isVerbose = CommandLine.arguments.contains("-v")
        || CommandLine.arguments.contains("--verbose")

    /// A `[HEADLESS]` line on **stderr**, for a command whose stdout carries a payload.
    ///
    /// `results show` and every headless `AppLog` line use it. The rest stay on stdout because
    /// `pipette-plan`'s `run_streaming_scanning` scans stdout alone for `BENCH_DONE`;
    /// moving the sentinel wholesale would leave the iOS transport unable to tell a
    /// finished run from a hung one.
    nonisolated static func logDiagnostic(_ s: String) {
        FileHandle.standardError.write(Data("[HEADLESS] \(s)\n".utf8))
    }

    /// A stored file, verbatim on stdout, so `results show … | jq` works. No prefix and no
    /// trailing newline of our own — the payload is the payload.
    nonisolated static func emitPayload(_ s: String) {
        FileHandle.standardOutput.write(Data((s.hasSuffix("\n") ? s : s + "\n").utf8))
    }

    /// Baked LFM2.5-8B token-id fixtures for the `coherence` probe. Fixed token ids
    /// (not text) feed the engine identical input with no tokenizer in the loop, so
    /// its greedy output is directly comparable to the Python `mlx_lm` reference
    /// values (from the routing-corrected model).
    private enum ParityProbe {
        /// Arbitrary in-vocab probe sequence for last-token top-5 parity.
        static let tokens: [Int32] = [124894, 100, 2000, 345, 6789, 42, 999, 17,
                                      2560, 88, 12345, 5, 678, 9012, 34, 567]
        /// Reference last-token top-5 (order-insensitive set) for `tokens`.
        static let expectedTop5: [Int] = [96, 207, 81, 34, 342]
        /// A real chat-templated prompt (token ids).
        static let realPrompt: [Int32] = [124894, 124899, 5922, 207, 2992, 355, 278, 5205,
                                          302, 3980, 39, 41774, 296, 734, 1858, 22, 124900,
                                          207, 124899, 63514, 207]
        /// Reference greedy continuation of `realPrompt` (token ids).
        static let realGreedy: [Int] = [124901, 207, 597, 4695, 20589, 34, 496, 2992, 355,
                                        278, 5205, 302, 3980, 39, 41774, 296, 734, 1858,
                                        2426, 8, 2083, 1946, 6119, 415]
    }

    /// Maps a CLI `metrics=` token → a builder for the catalog benchmark id at a
    /// given offset. The single extension point for exposing more benchmarks over
    /// the CLI: add an entry here (and the matching catalog id) to make it runnable.
    private static let metricBuilders: [String: (Int) -> String] = [
        "prefill": { "prefill_throughput_\($0)" },
        "decode": { "decode_throughput_\($0)_100" },
        "maxmem": { "max_memory_usage_\($0)" },
        "e2e": { "end_to_end_latency_\($0)_256" },   // catalog fixes e2e decode at 256
    ]

    /// Resolve the catalog benchmark ids to run: an explicit `benchmarks=` list
    /// takes precedence; otherwise the cartesian product of `metrics` × `offsets`
    /// via `metricBuilders` (unknown metric tokens are ignored). Internal so the
    /// deep-link path (`DeepLinkRouter`) resolves ids through the same
    /// `metricBuilders` table — one source of truth for the metric→id mapping.
    static func resolveBenchmarkIds(metrics: [String], offsets: [Int], explicit: [String]) -> [String] {
        if !explicit.isEmpty { return explicit }
        var ids: [String] = []
        for off in offsets {
            for met in metrics {
                guard let build = metricBuilders[met] else { continue }
                ids.append(build(off))
            }
        }
        return ids
    }

    /// Run `max_memory_usage` for each named model, in order, in this one process —
    /// the contamination surface. Each cell goes through the shipping `RunCell.dispatch`
    /// (same path `JobExecutor` uses), which logs `enter/floor/peak`. The reported
    /// host bytes are echoed per model so the `--console` transcript is self-contained.
    private static func runMemSeq(modelNames: [String], batch: Int, storage: Storage) async {
        let available = await MainActor.run { storage.availableModels() }
        log("memseq start models=\(modelNames) batch=\(batch)")
        guard !modelNames.isEmpty else {
            log("memseq ERROR no models= given; have: "
                + available.map { "\($0.name)[\($0.engineLabel)]" }.joined(separator: ", "))
            return
        }
        for name in modelNames {
            guard let m = available.first(where: { $0.name == name }) else {
                log("memseq SKIP \(name): not on device"); continue
            }
            do {
                // Footprint carried in from the prior cell, before this cell's own
                // settle gate runs — the contamination indicator. High here + a
                // clean REPORTED below is the gate working.
                let carriedIn = Double(ProcessMemory.physFootprintBytes()) / 1_048_576
                log(String(format: "memseq PRE %@ carriedIn=%.0fMB", name, carriedIn))
                // Fixed 512-token prefill + 1 decode; 2048 ctx is comfortably above that.
                let def = BenchmarkDefinition.maxMemoryUsage(
                    benchmarkId: "max_memory_usage_512", prefillTokens: 512)
                let runtime = Runtime.thisBuild(for: m.source)
                let request = try await RunCell.prepare(
                    spec: ClientRunSpec(
                        benchmark: def.benchmarkId, model: m.source, runtime: runtime,
                        runtimeFlags: RuntimeFlagRef(
                            benchmarkType: def.type, runtimeType: RuntimeType.of(runtime),
                            modelType: ModelType.of(m.source),
                            numberGpuLayers: 99, ctxSize: 2048, nUbatch: UInt32(batch))),
                    benchmark: def,
                    storage: storage, coordinator: DownloadCoordinator.shared)
                let r = try await RunCell.dispatch(request, storage: storage, readiness: { .ready })
                if case let .maxMemoryUsage(host, _, _) = r.resultData {
                    log(String(format: "memseq REPORTED %@ [%@] host=%.0fMB",
                               name, m.engineLabel, Double(host) / 1_048_576))
                }
            } catch {
                log("memseq ERROR \(name): \(error)")
            }
        }
    }

    /// Print how each completed cell was filed, and — when the run asked to submit —
    /// whether that happened, as the CLI's `print_record_done` does. Reads the manifest
    /// the executor persisted, so it reports what is on disk rather than what was intended.
    private static func reportResults(_ manifest: JobManifest, submitRequested: Bool,
                                      storage: Storage) async {
        @Sendable func filings(_ manifest: JobManifest) async -> [ResultReport] {
            await MainActor.run {
                manifest.cells
                    .filter { $0.runStatus == .completed }
                    .map { ResultReport(cell: $0, store: storage.results) }
            }
        }
        var reports = await filings(manifest)
        // The same terms `shouldAutoSubmit` gates on, so the line names the one that
        // actually stopped the upload.
        let blocker = await MainActor.run {
            ResultReporter.submitBlocker(
                registered: storage.identity.getRegistration() != nil,
                online: NetworkReachability.shared.isConnected)
        }
        // Submit here rather than leaning on the executor's own pass, as the crate's
        // command layer does: `record_and_maybe_submit_run` is called by `benchmarks run`,
        // not by `run_cell`, which is what lets it report the outcome. The drain is
        // serialized and idempotent, so the executor having already run one is a no-op.
        var errors: [String] = []
        if submitRequested, blocker == nil, reports.contains(where: { $0.location == .remotePending }) {
            errors = await ResultUploader.shared.drainJob(jobId: manifest.jobId).errors
            // Re-read: the acks land on disk, so a cell that just synced must not still
            // report as pending.
            if let acked = await MainActor.run(body: { storage.loadJobManifest(jobId: manifest.jobId) }) {
                reports = await filings(acked)
            }
        }
        for line in ResultReporter.lines(reports: reports, submitRequested: submitRequested,
                                         blocker: blocker, errors: errors) {
            log(line)
        }
    }

    // MARK: - Status (a thin handler over the shipping controllers)

    /// `settings run`: turn the planner worker on and keep the process alive so
    /// the claim loop can run (same path as Settings → Planner worker on).
    private static func runPlannerWorkerSession(
        storage: Storage,
        jobRunner: JobRunner,
        jobStore: JobStore
    ) async {
        let started = await MainActor.run { () -> Bool in
            LocalStorage.plannerWorkerEnabled = true
            jobStore.reload()
            PlannerWorker.shared.setEnabled(
                true,
                storage: storage,
                jobRunner: jobRunner,
                jobStore: jobStore
            )
            let text = PlannerWorker.shared.statusText
            // setEnabled clears the preference when not registered.
            if text == "Needs registration" || !LocalStorage.plannerWorkerEnabled {
                log("settings run ERROR not registered")
                return false
            }
            log("settings worker=on status=\(text)")
            return true
        }
        guard started else {
            benchDone(HeadlessExit.failure)
            exit(HeadlessExit.failure)
        }
        // Stay resident: periodic status lines for `--console`; no BENCH_DONE
        // until the OS kills the process.
        while !Task.isCancelled {
            try? await Task.sleep(for: .seconds(30))
            let text = await MainActor.run { PlannerWorker.shared.statusText }
            log("settings worker status=\(text)")
        }
    }

    /// `version`: the app version and the build flags that change what a measurement
    /// means — the crate's `pipette --version`, which prints its own build stamp.
    ///
    /// `thermal=` is the load-bearing half: a run gated by a real SoC die temperature is
    /// not comparable to one gated by `thermalState`, so confirming which build is on a
    /// device is a prerequisite to trusting its numbers.
    private static func runVersion() {
        log("version app=\(Bundle.main.appVersionDisplayString) "
            + "privateThermal=\(BuildFlavor.hasPrivateThermal ? "on" : "off") "
            + "thermal=\(BuildFlavor.thermalDescription)")
    }

    /// `status`: registration state, model count, and job counts by status.
    private static func runStatus(storage: Storage) async {
        if let reg = storage.identity.getRegistration() {
            // `mirrored=` answers "would this identity come back after a reinstall" — the
            // record is the half that used to be lost, the key having always been in the
            // Keychain.
            log("status registered clientId=\(reg.clientId.value) server=\(reg.serverUrl.value) "
                + "mirrored=\(storage.identity.isRegistrationMirrored ? "yes" : "no")")
        } else {
            log("status unregistered")
        }
        let modelCount = await MainActor.run { storage.availableModels().count }
        log("status models=\(modelCount)")
        let jobs = storage.loadAllJobManifests()
        let byStatus = Dictionary(grouping: jobs, by: { $0.status.rawValue })
            .map { "\($0.key)=\($0.value.count)" }.sorted().joined(separator: " ")
        log("status jobs total=\(jobs.count)\(byStatus.isEmpty ? "" : " " + byStatus)")
        log("status worker=\(LocalStorage.plannerWorkerEnabled ? "on" : "off")")
    }

    // MARK: - Bench (the bare form)

    /// Returns `true` on success, `false` on any failure — drives the headless
    /// process exit code.
    private static func run(runtime: RuntimeType, batch: Int,
                            nGpuLayers: UInt32?, threads: UInt32? = nil,
                            benchmarkIds: [String],
                            runCoherence: Bool, runCalibrate: Bool, runPromptSeed: Bool,
                            offsets: [Int], resolved: DiscoveredModel? = nil,
                            modelMatch: String?, submit: Bool,
                            readiness: ReadinessPolicy = .init(),
                            storage: Storage) async -> Bool {
        let engineTag = runtime.engineLabel
        let models = await MainActor.run { storage.availableModels() }
        // The entry, not a name to re-match: a pinned revision and an unpinned copy of one
        // repo share a filename, so re-matching ran the wrong weights and recorded the
        // wrong coordinate.
        let model = resolved ?? models.first { m in
            // Pair the requested engine with a model of the matching format;
            // `run` is only called for `.mlx`/`.llamacppIosPipette`, so AFM never matches here.
            return runtime.matches(m.source)
                && (modelMatch == nil || m.name.localizedCaseInsensitiveContains(modelMatch!))
        }
        guard let model else {
            log("ERROR no \(engineTag) model found; have: "
                + models.map { "\($0.name)[\($0.engineLabel)]" }.joined(separator: ", "))
            benchDone(HeadlessExit.failure); return false
        }
        // The crate's `is_compatible` gate, which every `benchmarks run` passes through:
        // the search above enforces it, but a handed-over entry has not been checked, and
        // an engine loading another format is a construction bug either way.
        guard runtime.matches(model.source) else {
            log("ERROR \(model.name) is a \(model.engineLabel) model, not \(engineTag)")
            benchDone(HeadlessExit.failure); return false
        }
        let ctx = UInt32((offsets.max() ?? 4096) + 300)
        log("start runtime=\(engineTag) model=\(model.name) batch=\(batch) "
            + "benchmarks=\(benchmarkIds) submit=\(submit)")

        // Prompt-seed self-check (`metrics=promptseed`): ask the runtime to build a
        // prompt for each target under THIS model's own tokenizer and report the
        // count it actually produced; the run only logs OK/MISMATCH. All prompt
        // construction lives in the runtime (`*.promptSeedCounts`), not here.
        if runPromptSeed {
            let targets = [100, 256, 512, 1024, 2048]
            log("promptseed runtime=\(engineTag) model=\(model.name) targets=\(targets)")
            var allOK = false
            do {
                let results = runtime == .mlxIosPipette
                    ? try await MLXRuntime.promptSeedCounts(modelPath: model.path, targets: targets)
                    : try LlamaBenchmark.promptSeedCounts(modelPath: model.path, nGpuLayers: 99,
                                                          contextSize: ctx, nUbatch: UInt32(batch), targets: targets)
                allOK = true
                for r in results {
                    let ok = r.got == r.target
                    allOK = allOK && ok
                    log("promptseed \(engineTag) target=\(r.target) got=\(r.got) \(ok ? "OK" : "MISMATCH")")
                }
                log("promptseed \(allOK ? "ALL OK" : "HAD MISMATCHES")")
            } catch { log("promptseed ERROR \(error)") }
            benchDone(ok: allOK); return allOK
        }

        // Calibrate the public IMU thermometer (`IMUThermometer`) against soc_temp,
        // so the readiness gate can use the IMU estimate on builds without the
        // private sensor. Requires the PIPETTE_PRIVATE_THERMAL build (for the
        // ground-truth soc_temp) and a STATIONARY device. Persists the per-device
        // model to UserDefaults. `metrics=calibrate`.
        if runCalibrate {
            guard runtime == .mlxIosPipette else { log("calibrate requires runtime=mlx"); benchDone(HeadlessExit.failure); return false }
            if socTemp() <= 0 {
                log("calibrate ERROR: soc_temp unavailable; build with PIPETTE_PRIVATE_THERMAL=1")
                benchDone(HeadlessExit.failure); return false
            }
            let mlxModel: any LanguageModel
            do { mlxModel = try await MLXRuntime.loadModel(path: model.path) } catch {
                log("ERROR load \(error)"); benchDone(HeadlessExit.failure); return false
            }
            defer { MLXRuntime.releaseModel() }
            // Two heat/cool cycles at different peaks → (IMU, soc_temp) pairs covering
            // a range of temps reached at different times (decouples the time confound).
            var samples = [(imu: [Double], temp: Double)]()
            for bursts in [8, 14] {
                for _ in 0 ..< bursts { _ = MLXRuntime.prefillBurst(mlxModel, tokens: 4096, prefillChunk: batch) }
                for _ in 0 ..< 30 {
                    let imu = IMUThermometer.averagedIMU()
                    let t = socTemp()
                    if t > 0 { samples.append((imu, t)) }
                    log(String(format: "calibrate sample n=%d soc_temp=%.2f", samples.count, t))
                    if t > 0, t <= imuThresholdC { break }
                    try? await Task.sleep(for: .seconds(4))
                }
            }
            guard let rmse = IMUThermometer.calibrate(samples) else {
                log("calibrate FAILED (n=\(samples.count), need >=7)"); benchDone(HeadlessExit.failure); return false
            }
            log(String(format: "calibrate DONE n=%d trainRMSE=%.2fC, persisted per-device model",
                       samples.count, rmse))

            // LAG TEST: the IMU senses the IMU *chip's* temperature, which lags the SoC
            // die under load (die spikes fast, IMU catches up slowly). Heat in bursts
            // from cool and log soc_temp vs imuEst — lag>0 means the IMU reads cooler
            // than the die (the dangerous case: gate could proceed while die is hot).
            log("calibrate: LAG TEST, heating ramp (soc_temp vs imuEst)")
            for step in 0 ..< 12 {
                _ = MLXRuntime.prefillBurst(mlxModel, tokens: 4096, prefillChunk: batch)
                let soc = socTemp()
                let est = IMUThermometer.estimate() ?? -1
                log(String(format: "lag heat step=%d soc_temp=%.2f imuEst=%.2f lag=%+.2f", step, soc, est, soc - est))
            }
            log("calibrate: LAG TEST, settle (idle, does imuEst catch up?)")
            for step in 0 ..< 16 {
                let soc = socTemp()
                let est = IMUThermometer.estimate() ?? -1
                log(String(format: "lag settle step=%d soc_temp=%.2f imuEst=%.2f lag=%+.2f", step, soc, est, soc - est))
                try? await Task.sleep(for: .seconds(4))
            }
            benchDone(HeadlessExit.ok); return true
        }

        // Coherence + top-token comparison (same baked prompt ids → directly
        // comparable greedy continuations across runtimes; no tokenizer needed).
        if runCoherence {
            do {
                if runtime == .mlxIosPipette {
                    let probe = try await MLXRuntime.coherence(modelPath: model.path,
                        promptIds: ParityProbe.tokens, gen: 0, prefillChunk: batch)
                    let cont = try await MLXRuntime.coherence(modelPath: model.path,
                        promptIds: ParityProbe.realPrompt, gen: 24, prefillChunk: batch)
                    var pfx = 0
                    for (a, b) in zip(cont.ids, ParityProbe.realGreedy) { if a == b { pfx += 1 } else { break } }
                    log("coherence mlx probe_top5=\(probe.top5) ref=\(ParityProbe.expectedTop5) "
                        + "setmatch=\(Set(probe.top5) == Set(ParityProbe.expectedTop5))")
                    log("coherence mlx greedy=\(cont.ids)")
                    log("coherence mlx ref_greedy=\(ParityProbe.realGreedy) prefix_match=\(pfx)/\(ParityProbe.realGreedy.count)")
                } else {
                    // Native llama.cpp ops — the same ones the benchmark path uses.
                    let m = try LlamaCpp.load(path: model.path, nGpuLayers: 99,
                                              contextSize: ctx, nUbatch: UInt32(batch),
                                              threads: LlamaCpp.defaultThreads,
                                              swaFull: LlamaCpp.defaultSwaFull)
                    defer { m.free() }
                    try LlamaCpp.prefill(m, ParityProbe.realPrompt)
                    let text = try LlamaCpp.decodeGreedyText(m, maxTokens: 24)
                    let ids = (try? LlamaCpp.tokenize(m, text, addSpecial: false)) ?? []
                    log("coherence llama text=\(text.replacingOccurrences(of: "\n", with: " "))")
                    log("coherence llama greedy_ids=\(ids)")
                }
            } catch { log("ERROR coherence \(error)") }
        }

        // Run the requested benchmarks through the *same* path the UI uses:
        // build a one-model job manifest and execute it via `JobExecutor`, which
        // logs each cell's result to the console (`[JobRun] result …`). Nothing is
        // submitted unless `submit=1` (headless defaults to off).
        // Derive each cell's type from the structured id rather than the synced
        // catalog — the CLI's benchmark ids (`metricBuilders`) are self-describing,
        // and JobExecutor resolves the full definition + context size at run time
        // via the same `BenchmarkDefinition(parsingId:)` fallback.
        let cells = JobCell.pending(benchmarkIds: benchmarkIds, for: model) {
            log("skip \($0): unrecognized benchmark id")
        }
        // No cells is success only for a coherence-only run (the diagnostic above
        // already ran); an empty benchmark request that resolved to nothing is a
        // failure.
        guard !cells.isEmpty else { benchDone(ok: runCoherence); return runCoherence }

        // Launch through the same `JobLauncher` sequence the New Job screen
        // uses, so a CLI-created job is identical to a UI-created one
        // (including the executor's end-of-run auto-submit, which only fires
        // when `contributeResults` is set — i.e. `submit=1`).
        let final: JobManifest? = await withCheckedContinuation { cont in
            Task { @MainActor in
                let launched = JobLauncher.launch(
                    cells: cells, nGpuLayers: Int(nGpuLayers ?? HeadlessCommand.defaultGpuLayers),
                    contextSize: Int(ctx), prefillBatch: batch, threads: threads.map(Int.init),
                    contributeResults: submit, readiness: readiness, jobRunner: JobRunner(), jobStore: JobStore(storage: storage), storage: storage,
                    onFinish: { cont.resume(returning: $0) })
                if launched == nil { cont.resume(returning: nil) }
            }
        }
        guard let final else {
            // Can't happen with this process's fresh JobRunner, but keep the
            // blocked path explicit rather than force-unwrapping it away.
            log("ERROR job launch blocked: another job is running")
            benchDone(HeadlessExit.failure)
            return false
        }
        await reportResults(final, submitRequested: submit, storage: storage)
        benchDone(final.status == .completed ? HeadlessExit.ok : HeadlessExit.failure,
                  "\(final.status)")
        return final.status == .completed
    }
}
