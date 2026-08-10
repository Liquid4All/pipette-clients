import Foundation

// One cell, from a coordinate to what it measured — the Swift counterpart of
// `crates/pipette-cli/src/run.rs`. Not `Run.swift`: plan-types' `run.rs` already holds that
// basename, and one Swift module cannot have two.
//
// Steps of upstream's `prepare` with no counterpart here: no runtime to ensure (this binary
// *is* the runtime, so `Runtime.thisBuild` supplies both halves), no GPU preflight (no
// driver that can break independently of the client), and no spec-names-this-body check (a
// caller resolves the body from the cell's own id, so there is no second spelling to
// disagree). `require_desktop_runtime` inverts: `ClientRunSpec.validated` refuses a claim naming a
// desktop runtime before a cell exists.
//
// No `run_cell` either — the memory gate and the load line run between `prepare` and
// `dispatch`, and the record is `PayloadBuilder`'s to write into the results store.

nonisolated enum RunCell {
    /// The request for one cell: ensure the model, then name every axis.
    ///
    /// `progress` reports the ensure's transfer, which can run for minutes on a model the
    /// quota sweep reclaimed — without it a caller cannot tell a download from a hang.
    /// Takes the cell whole — the crate's `prepare(spec, benchmark, …)`, which copies
    /// `spec.model` and `spec.runtime_flags` into the request rather than rebuilding them
    /// from parts. There is no per-knob argument list here to forget a knob from.
    ///
    /// `benchmark` arrives beside the spec, not inside it, for the reason upstream gives:
    /// the spec names a benchmark id, and resolving it against a catalog is the caller's
    /// business.
    static func prepare(
        spec: ClientRunSpec,
        benchmark: BenchmarkDefinition,
        storage: Storage,
        coordinator: DownloadCoordinator,
        progress: ((FetchProgress) -> Void)? = nil
    ) async throws -> RunRequest {
        let declared = spec.model
        // Apple Foundation ships with the OS: nothing to ensure, and the two halves are
        // equal because there is no path to bind — the same reason `alreadyBound` exists
        // for a runtime the store holds but cannot relocate.
        let model: DeclaredBound<Model>
        if case .appleFoundationText = declared {
            model = .alreadyBound(declared)
        } else {
            // `ensureModel` already answers with the bound form — the declared spec with
            // its source rewritten to the `absolute*` arms — so the pair is just the two
            // values, not a third type built from them.
            let bound = try await ensureModel(declared, storage: storage,
                                              coordinator: coordinator, progress: progress)
            guard bound.boundPaths != nil else {
                throw ModelStoreError.unresolvedAfterFetch(declared.artifactName)
            }
            model = DeclaredBound(declared: declared, bound: bound)
        }

        // Both halves are this build, even when `spec.runtime` names something else: a
        // claim may pin a llama version this binary is not, and the descriptor a result
        // carries has to be what actually compiled. The claim's declaration is checked
        // (and logged when it differs) where the claim is read, not silently adopted here.
        let runtime = Runtime.thisBuild(for: declared)
        return RunRequest(
            runtime: .alreadyBound(runtime),
            model: model,
            // A knob the cell does not carry is a construction error, and swallowing it
            // would run on engine defaults and report those as what the cell asked for —
            // the failure this whole path exists to prevent. `noSuchCombination` is the
            // other answer: AFM, and a vision model outside `vl_throughput`, genuinely have
            // no variant, and the engine refuses those again when it derives its own.
            runtimeFlags: try spec.runtimeFlags.flatMap { ref -> RuntimeFlags? in
                do {
                    return try ref.resolve()
                } catch RuntimeFlagResolveError.noSuchCombination {
                    return nil
                }
            },
            benchmarkFlags: try spec.benchmarkFlags.flatMap { ref -> BenchmarkFlags? in
                do {
                    return try ref.resolve()
                } catch RuntimeFlagResolveError.noSuchCombination {
                    return nil
                }
            },
            benchmark: benchmark)
    }

    /// Run one prepared request — the crate's `dispatch_run`, whole: bind the thermal
    /// series, route to an engine, attach what was measured.
    ///
    /// Routing is on the **bound runtime**, as upstream states its rule: "the bound runtime
    /// `prepare` produces is the only thing `dispatch_run` matches on". The model is not
    /// consulted. Each engine checks the model it was handed through its own `require*`
    /// (`LlamaModels.requireGgufText`, `MLXModels.requireMlxModelDir`,
    /// `AFMModels.requireAppleFoundation`), which is where upstream puts that check too —
    /// so a mismatched pair fails by name, in the engine that would have loaded it.
    ///
    /// The loaded resource is assembled inside the engine's own scope (`withInference` /
    /// `MLXRuntime.withFreshModel`) and never escapes; nothing here owns a model.
    ///
    /// The thermal series is owned and attached here, not by the caller: the engine reports
    /// each rep through the opaque `RepObserver` and never sees the probe, so owning it here
    /// is what keeps a caller from silently recording no readings.
    ///
    /// `readiness` arrives already resolved, as upstream takes it: resolving inside the
    /// gate would let the wait and the record describe different policies.
    ///
    /// `storage` is here for the eval checkpoint alone, and the *store* travels down rather
    /// than an open session: upstream hands each engine `&ws.eval_completions()` and lets
    /// the eval executor open it, because only that executor knows whether the benchmark
    /// has samples to resume at all.
    static func dispatch(
        _ request: RunRequest,
        storage: Storage,
        readiness: @escaping () -> ReadinessOutcome,
        isCancelled: @escaping () -> Bool = { false },
        progress: @escaping (BenchmarkProgress) -> Void = { _ in }
    ) async throws -> RunResponse {
        // Inference must run off the main actor: it relies on JobExecutor's `Task.detached`
        // plus this whole layer being `nonisolated`, and on JobRunner's single-running-job
        // guard for non-concurrency (the actor isolation that used to serialize runs is
        // gone). Assert the off-main half so a future re-isolation is caught in dev/CI
        // rather than silently freezing the UI during a run again. (`Thread.isMainThread`
        // is banned in async contexts, so use the C `pthread_main_np`.)
        assert(pthread_main_np() == 0, "RunCell.dispatch must not execute on the main thread")

        let series = ThermalSeries()
        let observer = RepObserver(started: { series.start() }, finished: { series.finish() })
        let evalCompletions = storage.evalCompletions

        // Each engine module overlays its own flags (`LlamaRuntimeFlags`, `MLXRuntimeFlags`)
        // and that one value is both loaded with and reported — so a result records what
        // ran rather than the unset a cell carried. A triple plan-types defines no variant
        // for is refused there, before anything loads.
        func respond(_ result: BenchmarkResult, _ ran: RuntimeFlags?,
                     engineLog: String = "") -> RunResponse {
            var out = RunResponse(resultData: result)
            out.runtimeFlags = ran
            out.benchmarkFlags = request.benchmarkFlags
            out.stderr = engineLog
            return out
        }

        var response: RunResponse
        switch request.runtime.bound {
        case .llamacppIosPipette:
            let llamaFlags = try LlamaRuntimeFlags.forRun(request)
            response = respond(try await LlamaBenchmark.run(
                request, flags: llamaFlags, evalCompletions: evalCompletions,
                gate: { try readinessGate(readiness) }, observer: observer,
                isCancelled: isCancelled, progress: progress), llamaFlags,
                engineLog: LlamaCpp.capturedLoadLog)
        case .mlxIosPipette:
            let mlxFlags = try MLXRuntimeFlags.forRun(request)
            response = respond(try await MLXRuntime.run(
                request, flags: mlxFlags, evalCompletions: evalCompletions,
                readiness: readiness, observer: observer,
                isCancelled: isCancelled, progress: progress), mlxFlags)
        case .appleFoundation:
            // Apple Foundation has no flags variant upstream either, so there is nothing
            // to overlay and nothing to report.
            response = respond(try await AFMRuntime.run(
                request, evalCompletions: evalCompletions, readiness: readiness,
                observer: observer,
                isCancelled: isCancelled, progress: progress), nil)
        }
        response.thermal = series.runThermal
        return response
    }
}
