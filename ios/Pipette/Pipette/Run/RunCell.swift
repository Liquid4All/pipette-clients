import Foundation

// One cell, from a coordinate to what it measured: the Swift counterpart of
// `crates/pipette-cli/src/run.rs`. Not `Run.swift`: plan-types' `run.rs` already holds that
// basename, and one Swift module cannot have two.
//
// Unlike upstream's `prepare`: no runtime to ensure (this binary is the runtime) and no GPU
// preflight. A memory preflight takes its place, since the jetsam budget is a phone's
// failure mode. `require_desktop_runtime` has no counterpart: `Runtime` cannot decode a
// desktop runtime, so `runSpecFromClaim(_:)` refuses one before a cell exists.

/// What one finished cell hands back: the crate's `(BenchmarkSubmissionPayload,
/// BenchmarkResultExtras, Option<EvalRunDigest>)`.
nonisolated struct FinishedRun: Sendable {
    let payload: BenchmarkSubmissionPayload
    let extras: BenchmarkResultExtras
    let evalCheckpoint: EvalRunDigest?
}

nonisolated enum RunCell {
    /// The crate's `run_cell`. Recording is the caller's (`recordResult`).
    ///
    /// `readiness` builds the wait from the resolved policy: the policy is a plan decision,
    /// while how a phone waits (the cancel flag, the UI status) is the caller's.
    /// `preflight` gets the request and the model's weight against the jetsam budget, so the
    /// caller can stamp its crash sentinel first. An eval also returns its checkpoint digest:
    /// the caller retires it with `finalizeEvalCheckpoint` once the result is saved or accepted.
    /// `abandon` is `dispatchRun`'s.
    static func runCell(
        spec: ClientRunSpec,
        benchmark: BenchmarkDefinition,
        ws: PipetteWorkspace,
        coordinator: DownloadCoordinator,
        readiness: (ResolvedReadiness) -> ReadinessCallback,
        cancelled: @escaping () -> Bool = { false },
        fetchProgress: ProgressSink? = nil,
        jobId: JobId? = nil,
        preflight: (RunRequest, MemoryGate.Snapshot?) async -> Void = { _, _ in },
        abandon: (@Sendable (AttemptTimeoutError) -> Never)? = nil,
        progress: @escaping (BenchmarkProgress) -> Void = { _ in }
    ) async throws -> FinishedRun {
        // Ahead of prepare: its ensure can download weights for minutes, and a stall or a
        // failure in there is only attributable if the cell is already named.
        AppLog.jobRun.info("run: \(spec)")
        // Also before prepare, so a cell that refuses on battery says so before the download.
        try await refuseOnBattery(benchmark)
        let request = try await prepare(spec: spec, benchmark: benchmark, ws: ws,
                                        coordinator: coordinator, progress: fetchProgress,
                                        jobId: jobId)

        // Never blocking: a barely fitting run is a legitimate benchmark, and the caller's
        // sentinel turns a kill into an explained failure. Apple Foundation has no bound path.
        let memory = request.model.bound.boundPaths
            .flatMap { MemoryGate.snapshot(modelPath: $0.payload) }
        await preflight(request, memory)

        // Resolved here, not inside the gate: the same value has to drive the wait and the
        // record, or the result describes a run that did not happen.
        let policy = resolveReadiness(maxWait: readinessMaxWait(request),
                                      thermal: readinessThermalGate(request))
        let wait = readiness(policy)

        let (response, telemetry) = try await dispatchRun(
            request, ws: ws, readiness: policy,
            waitUntilReady: { wait.waitUntilReady(observing: $0) },
            cancelled: cancelled, abandon: abandon, progress: progress)
        let record = try await finishedRunPayload(
            deviceStore: ws.device, req: request, outcome: response, telemetry: telemetry)
        // The identity every engine opened its checkpoint under: this request.
        let evalCheckpoint: EvalRunDigest? = if case .eval = request.benchmark {
            try runRequestDigest(request)
        } else {
            nil
        }
        return FinishedRun(payload: record.payload, extras: record.extras,
                           evalCheckpoint: evalCheckpoint)
    }

    /// The crate's `finalize_eval_checkpoint`. Until the result is saved or accepted, the
    /// checkpoint is what a lost result is rebuilt from. Best-effort: one left behind is
    /// resumed, whole, by the next run of the cell.
    static func finalizeEvalCheckpoint(ws: PipetteWorkspace, digest: EvalRunDigest?) {
        guard let digest else { return }
        do {
            try ws.evalCompletions.finalize(digest)
        } catch {
            AppLog.jobRun.warning(
                "eval checkpoint not finalized digest=\(digest.value) error=\(error.localizedDescription)")
        }
    }

    /// Only `vl_spec_decode` is checked: a ratio whose arms ran under different power states is
    /// no measurement, and the server records the state without rejecting.
    static func refuseOnBattery(_ benchmark: BenchmarkDefinition) async throws {
        guard benchmark.benchmarkType == .vlSpecDecode else { return }
        let power = await detectPowerState()
        switch powerVerdict(power.powerState) {
        case .stable:
            return
        case .moving:
            throw RuntimeError.readiness(
                "`\(benchmark.benchmarkId)` measures two arms against each other, so it "
                    + "refuses to start while the power state can move under it "
                    + "(device_power_state=\(power.powerState.map(\.rawValue) ?? "nil"), "
                    + "battery \(power.batteryLevel.map { "\($0)" } ?? "nil")%). Run it on AC "
                    + "with the battery near full.")
        case .unknown:
            AppLog.jobRun.warning(
                "`\(benchmark.benchmarkId)` could not read this device's power state; it must "
                    + "be on AC with the battery near full, or the two arms are not comparable")
        }
    }

    enum PowerVerdict: Equatable {
        /// On external power and not charging: the one state that holds still.
        case stable
        /// On battery, or charging: a fast-charging battery throttles the GPU even on AC.
        case moving
        /// Undetectable: refusing would make the cell unrunnable wherever the state is unread.
        case unknown
    }

    static func powerVerdict(_ state: DevicePowerState?) -> PowerVerdict {
        switch state {
        case .pluggedInNotCharging: .stable
        case .charging, .notCharging: .moving
        case nil: .unknown
        }
    }

    /// The crate's `prepare`: ensure the model, then name every axis. Takes the spec whole so
    /// there is no per-knob argument list to forget a knob from. `progress` reports the
    /// ensure's transfer, which can run for minutes; without it a download looks like a hang.
    static func prepare(
        spec: ClientRunSpec,
        benchmark: BenchmarkDefinition,
        ws: PipetteWorkspace,
        coordinator: DownloadCoordinator,
        progress: ProgressSink? = nil,
        jobId: JobId? = nil
    ) async throws -> RunRequest {
        // The payload's id and the eval digest come from the body, so a mismatch would
        // record a cell that never ran.
        guard spec.benchmark == benchmark.benchmarkId else {
            throw RunCellError.benchmarkMismatch(spec: spec.benchmark.value,
                                                 body: benchmark.benchmarkId.value)
        }
        // The cell the flag groups must name, from the *resolved* body: the claim path
        // checked them against the type it could guess from the id, which a body is free
        // to contradict. Upstream validates twice for the same reason.
        let cell: FlagAxes = (benchmark.benchmarkType, RuntimeType.of(spec.runtime),
                              ModelType.of(spec.model))
        let declared = spec.model
        // Apple Foundation ships with the OS: nothing to ensure and no path to bind.
        let model: DeclaredBound<Model>
        if case .appleFoundationText = declared {
            model = .alreadyBound(declared)
        } else {
            let bound = try await ensureModel(declared, ws: ws,
                                              coordinator: coordinator, jobId: jobId,
                                              progress: progress)
            guard bound.boundPaths != nil else {
                throw ModelStoreError.unresolvedAfterFetch(declared.artifactName)
            }
            model = DeclaredBound(declared: declared, bound: bound)
        }

        // A llama.cpp cell loads its build before anything is measured, so the load's time
        // and pages stay out of the cell's windows. The build is the one that a headless
        // `runtime=` named (already loaded), or main if the run named none.
        if case .llamacppIosPipette = Runtime.thisBuild(for: declared) {
            try LlamaPins.load(LlamaPins.current)
        }
        // The build this process runs, even when `spec.runtime` names another version: the
        // descriptor a result carries has to be what actually ran. `ClientRunSpec.validated`
        // checks the claim's runtime kind and private APIs against this build, not its version.
        let runtime = Runtime.thisBuild(for: declared)
        return RunRequest(
            runtime: .alreadyBound(runtime),
            model: model,
            runtimeFlags: try resolved(spec.runtimeFlags, cell: cell),
            // The claim path refuses every model-flag knob, so this is `nil` or a variant
            // with nothing set.
            modelFlags: try resolved(spec.modelFlags, cell: cell),
            benchmarkFlags: try resolved(spec.benchmarkFlags, cell: cell),
            benchmark: benchmark)
    }

    /// Upstream's `validate_spec_flags` then the `TryFrom` to a variant. Both refuse rather
    /// than drop: otherwise a group naming another cell resolves to that cell's variant, and
    /// a triple with no variant runs on engine defaults reported as what the cell asked for.
    private static func resolved(_ ref: RuntimeFlagRef?, cell: FlagAxes) throws -> RuntimeFlags? {
        guard let ref else { return nil }
        try check((ref.benchmarkType, ref.runtimeType, ref.modelType),
                  group: "runtime_flags", cell: cell)
        return try ref.resolve()
    }

    private static func resolved(_ ref: BenchmarkFlagRef?, cell: FlagAxes) throws -> BenchmarkFlags? {
        guard let ref else { return nil }
        try check((ref.benchmarkType, ref.runtimeType, ref.modelType),
                  group: "benchmark_flags", cell: cell)
        return try ref.resolve()
    }

    /// Skips the runtime axis: a `ModelFlagRef` carries none, as the crate's
    /// `matches(bt, model)` reads none.
    private static func resolved(_ ref: ModelFlagRef?, cell: FlagAxes) throws -> ModelFlags? {
        guard let ref else { return nil }
        try check((ref.benchmarkType, cell.runtime, ref.modelType),
                  group: "model_flags", cell: cell)
        return try ref.resolve()
    }

    private static func check(_ carried: FlagAxes, group: String, cell: FlagAxes) throws {
        guard carried == cell else {
            throw RunCellError.flagsAreNotForThisCell(group: group, carried: carried, cell: cell)
        }
    }

    /// The crate's `dispatch_run`. Routes on the bound runtime only; each engine checks the
    /// model it was handed, so a mismatched pair fails in the engine that would load it.
    ///
    /// The rep series is owned here, not by the caller, so a caller can't silently record no
    /// readings. `readiness` arrives already resolved: resolving twice could straddle a
    /// change, and the gate and the record must agree. `ws` carries the eval store rather
    /// than an open session: only the eval executor knows whether there are samples to resume.
    ///
    /// No engine here can be stopped mid-call, so an attempt past its limit fails the cell at
    /// the next boundary (a gate, a rep, a peak-memory attempt) or once the engine returns.
    /// `abandon` is for a launch that cannot wait for that: the watchdog's thread calls it at
    /// the overrun, and it ends the process.
    static func dispatchRun(
        _ request: RunRequest,
        ws: PipetteWorkspace,
        readiness: ResolvedReadiness,
        waitUntilReady: @escaping (_ observe: @escaping (ReadinessStatus) -> Void)
            -> ReadinessOutcome,
        cancelled: @escaping () -> Bool = { false },
        abandon: (@Sendable (AttemptTimeoutError) -> Never)? = nil,
        progress: @escaping (BenchmarkProgress) -> Void = { _ in }
    ) async throws -> (response: RunResponse, telemetry: RepTelemetry) {
        // Inference must run off the main actor (JobExecutor's `Task.detached` plus this
        // layer being `nonisolated`); JobRunner's single-running-job guard serializes runs.
        // `Thread.isMainThread` is banned in async contexts, hence `pthread_main_np`.
        assert(pthread_main_np() == 0, "RunCell.dispatchRun must not execute on the main thread")
        // iOS-only, not in the crate's `dispatch_run`: a later memory cell asks whether this
        // process ran anything before it. On exit, so this cell's own bracket reads clean.
        defer { ProcessMemory.markEngineWork() }

        // After `prepare`'s downloads, so the first attempt starts with the model on disk.
        let attemptClock = Attempt.limit(request).map { AttemptClock(limit: $0) }
        let gateWaitRecorder = GateWaitRecorder()
        let readinessGate: ReadinessGate = {
            try attemptClock?.check()
            // A readiness wait is no part of an attempt; the gate has its own limit.
            attemptClock?.hold()
            defer { attemptClock?.resume() }
            gateWaitRecorder.begin()
            try waitUntilReady(gateWaitRecorder.observe).get()
            gateWaitRecorder.pass()
        }
        // Sampled outside the timed window: the sensor read is one IPC round trip per
        // sensor.
        let series = ThermalSeries<RepSample>()
        // The corpus traversal is one rep and reports its own samples.
        let reps = RepProgress(total: request.benchmark.benchmarkType == .vlSpecDecode
                                   ? nil : BenchmarkMeasurement.reps,
                               progress: progress)
        // Sampled inside the device readings, so their IPC stays out of the clock window.
        let clocks = RepClocks(marker: request.benchmark.benchmarkId.value)
        // Each rep, and the work after it, is an attempt of its own.
        let observer = MeasurementObserver(onBeforeRepStart: {
                                               try series.start(RepSample(
                                                   device: detectDeviceReadings(),
                                                   gate: gateWaitRecorder.take()))
                                               try attemptClock?.beginRep()
                                               clocks.repStarting()
                                           },
                                           onAfterRepEnd: {
                                               clocks.repEnded()
                                               try attemptClock?.beginAfterRep()
                                               try series.finish(RepSample(
                                                   device: detectDeviceReadings(),
                                                   gate: nil))
                                               reps.finished()
                                           },
                                           onAttemptStart: { try attemptClock?.begin($0) },
                                           onHold: { attemptClock?.hold() },
                                           onResume: { attemptClock?.resume() })

        // The whole request, so a new field is logged too; `AuthToken` redacts itself.
        AppLog.jobRun.info("dispatch: \(request)")
        let watchdog = attemptClock.map { AttemptWatchdog(watching: $0, abandon: abandon) }
        var response: RunResponse
        do {
            switch request.runtime.bound {
            case .llamacppIosPipette:
                response = try await LlamaExecute.run(
                    request, runContext: ws, readinessGate: readinessGate, observer: observer,
                    isCancelled: cancelled, progress: progress)
            case .mlxIosPipette:
                response = try await MlxExecute.run(
                    request, runContext: ws, readinessGate: readinessGate, observer: observer,
                    isCancelled: cancelled, progress: progress)
            case .onnxGenaiIosPipette:
                response = try await OnnxGenaiExecute.run(
                    request, readinessGate: readinessGate, observer: observer)
            case .appleFoundation:
                response = try await AFMRuntime.run(
                    request, runContext: ws, readinessGate: readinessGate, observer: observer,
                    isCancelled: cancelled, progress: progress)
            }
        } catch {
            try watchdog?.finish(after: error)
            throw error
        }
        try watchdog?.finish(after: nil)
        // A series that cannot be paired fails the run here, as the crate's `take` does.
        let telemetry = RepTelemetry.fromReps(try series.take())
        response.benchmarkFlags = try resolvedFlags(request, readiness: readiness)
        return (response, telemetry)
    }

    /// The deadline this cell authored, or `nil` to defer to the environment, then the
    /// default.
    static func readinessMaxWait(_ request: RunRequest) -> Duration? {
        request.benchmarkFlags?.readiness?.maxWaitSecs.map(Duration.fromSecs)
    }

    /// What this cell says about the thermal criterion, keeping "waive it", "require it"
    /// and "didn't say" apart as the crate does: `unset` defers to
    /// `PIPETTE_READINESS_SKIP_THERMAL`.
    static func readinessThermalGate(_ request: RunRequest) -> ThermalGate {
        switch request.benchmarkFlags?.readiness?.skipThermal {
        case true?: .skip
        case false?: .enforce
        case nil: .unset
        }
    }

    /// The crate's `resolved_flags`: the plan's entry with the readiness block filled in from
    /// what actually applied, since the authored block may leave it to platform defaults.
    /// `nil` rather than `{}` when nothing resolved: the server NULLs an empty object anyway.
    static func resolvedFlags(_ request: RunRequest,
                              readiness: ResolvedReadiness) throws -> BenchmarkFlags? {
        var ref = try request.benchmarkFlagsRef()
        // Which cells gate is also encoded in whether their flag variant carries a
        // readiness field: two spellings of one fact.
        if request.benchmark.benchmarkType.gatesOnReadiness {
            ref.readiness = ReadinessOverrides(maxWaitSecs: readiness.maxWaitSecs(),
                                               skipThermal: readiness.skipThermal())
        }
        // The authored limit rides on `ref`; a defaulted one is not reported.
        do {
            let flags = try ref.resolve()
            return reportsNothing(flags) ? nil : flags
        } catch BenchmarkFlagsError.noSuchCombination {
            // A cell the flag schema does not model (eval, which has no iOS variant) has
            // nothing to report: not a failure.
            return nil
        }
    }

    /// The crate's `reports_nothing`.
    static func reportsNothing(_ flags: BenchmarkFlags) -> Bool {
        (try? flags.submissionValue()) == "{}"
    }
}

/// The checks `prepare` makes before anything is fetched or loaded.
nonisolated enum RunCellError: Error, Equatable, LocalizedError {
    case benchmarkMismatch(spec: String, body: String)
    case flagsAreNotForThisCell(group: String, carried: FlagAxes, cell: FlagAxes)

    var errorDescription: String? {
        switch self {
        case let .benchmarkMismatch(spec, body):
            "spec names benchmark `\(spec)` but the resolved body is `\(body)`"
        case let .flagsAreNotForThisCell(group, carried, cell):
            "\(group) for \(Self.render(carried)) do not match resolved cell \(Self.render(cell))"
        }
    }

    /// Hand-written: an enum carrying tuples gets no synthesized `==`.
    static func == (lhs: RunCellError, rhs: RunCellError) -> Bool {
        switch (lhs, rhs) {
        case let (.benchmarkMismatch(leftSpec, leftBody), .benchmarkMismatch(rightSpec, rightBody)):
            leftSpec == rightSpec && leftBody == rightBody
        case let (.flagsAreNotForThisCell(leftGroup, leftCarried, leftCell),
                  .flagsAreNotForThisCell(rightGroup, rightCarried, rightCell)):
            leftGroup == rightGroup && leftCarried == rightCarried && leftCell == rightCell
        default:
            false
        }
    }

    private static func render(_ axes: FlagAxes) -> String {
        "\(axes.benchmark.rawValue) \u{d7} \(axes.runtime.rawValue) \u{d7} \(axes.model.rawValue)"
    }
}

/// Each measured rep as `.attempt` progress, the screen's "Measurement n/5"; the CLI only
/// logs it. `@unchecked Sendable`: only the entry's sequential rep loop calls it.
nonisolated final class RepProgress: @unchecked Sendable {
    private let total: Int?
    private let progress: (BenchmarkProgress) -> Void
    private var completed = 0

    init(total: Int?, progress: @escaping (BenchmarkProgress) -> Void) {
        self.total = total
        self.progress = progress
    }

    func finished() {
        guard let total else { return }
        completed += 1
        progress(.attempt(completed: completed, total: total))
    }
}
