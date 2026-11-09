import Foundation

/// `ONE_AT_A_TIME`, as the MLX and ONNX GenAI crates name it: this engine holds one KV cache.
private nonisolated let oneAtATime = "the in-process llama.cpp engine decodes one sequence"

/// The crate's `pipette-llamacpp/src/execute/mod.rs`, one file per benchmark module. Where the
/// crate shells out to `llama-bench` and `llama-server`, this calls their C API in process;
/// each benchmark runs on a loaded `Inference`, which tests hand-build.
nonisolated enum LlamaExecute {
    static func run(_ request: RunRequest, runContext: any RunContext,
                    readinessGate: ReadinessGate, observer: MeasurementObserver,
                    isCancelled: @escaping () -> Bool = { false },
                    progress: @escaping (BenchmarkProgress) -> Void = { _ in }
    ) async throws -> RunResponse {
        let definition = request.benchmark
        // One overlay, both loaded with and reported, so a result records what ran. The
        // vision cells replace it once their projector has sized the context.
        var flags = try LlamaRuntimeFlags.forRun(request)
        let load = LoadSettings(flags, definition)

        let resultData: BenchmarkResultData
        // A traversal answers its own response, corpus observations included.
        var traversed: RunResponse?
        var clientExtras: AsrClientExtras?
        switch definition {
        case let .prefillThroughput(benchmark):
            try BenchmarkMeasurement.requireOneInFlight(definition, engine: oneAtATime)
            let model = try ResolvedGgufTextModel(request.model.bound)
            // Before the load, which would warm the device the gate waited to cool.
            try readinessGate()
            resultData = try await load.withInference(model) {
                try await prefillThroughput(benchmark, $0, readinessGate: readinessGate,
                                            observer: observer)
            }
        case let .decodeThroughput(benchmark):
            try BenchmarkMeasurement.requireOneInFlight(definition, engine: oneAtATime)
            let model = try ResolvedGgufTextModel(request.model.bound)
            try readinessGate()
            resultData = try await load.withInference(model) {
                try await decodeThroughput(benchmark, $0, readinessGate: readinessGate,
                                           observer: observer)
            }
        case let .endToEndLatency(benchmark):
            try BenchmarkMeasurement.requireOneInFlight(definition, engine: oneAtATime)
            let model = try ResolvedGgufTextModel(request.model.bound)
            try readinessGate()
            resultData = try await load.withInference(model) {
                try await endToEndLatency(benchmark, $0, readinessGate: readinessGate,
                                          observer: observer)
            }
        case let .maxMemoryUsage(benchmark):
            // Ungated: a peak is thermal-invariant.
            resultData = try await maxMemoryUsage(
                benchmark, load: load, model: try ResolvedGgufTextModel(request.model.bound),
                observer: observer)
        case let .eval(benchmark):
            // Before the load: a body without rows has nothing to run.
            let cell = try EvalCell(benchmark)
            let model = try ResolvedGgufTextModel(request.model.bound)
            let checkpoint = try runContext.evalCompletions.open(request)
            resultData = try await load.withInference(model) {
                try await eval(benchmark, cell: cell, $0, checkpoint: checkpoint,
                               isCancelled: isCancelled, progress: progress)
            }
        case let .vlThroughput(benchmark):
            let model = try ResolvedGgufVisionModel(request.model.bound)
            // Before the weights: an unrunnable frame count should not cost a model load.
            _ = try LlamaMtmd.CtxInputs(benchmark)
            try readinessGate()
            (resultData, flags) = try await withCalibratedVision(
                request, model, flags: flags, width: benchmark.parameterImageWidth.value.magnitude,
                height: benchmark.parameterImageHeight.value.magnitude, observer: observer
            ) {
                try await vlThroughput(benchmark, $0, readinessGate: readinessGate,
                                       observer: observer)
            }
        case let .vlMaxMemory(benchmark):
            let model = try ResolvedGgufVisionModel(request.model.bound)
            _ = try LlamaMtmd.CtxInputs(benchmark)
            // Ungated: a peak is thermal-invariant.
            (resultData, flags) = try await vlMaxMemory(benchmark, request, flags: flags,
                                                        model: model, observer: observer)
        case let .vlSpecDecode(benchmark):
            // Before the corpus download, which a batched cell would waste.
            try BenchmarkMeasurement.requireOneInFlight(definition, engine: oneAtATime)
            progress(.corpusDownload)
            let lease = try await VlSpecDecodeCorpus.download(
                runContext, benchmarkId: benchmark.benchmarkId.value, dataset: benchmark.dataset)
            // Released explicitly: ARC may free a lease after its last use, which would delete
            // the corpus mid-traversal.
            defer {
                do {
                    try lease.release()
                } catch {
                    AppLog.corpus.warning("failed to delete the corpus: \(error)")
                }
            }
            // `forRun` resolved the flag against the model, so what loads is the condition
            // the row records.
            let model = try LoadedModel.resolve(request.model.bound,
                                                enabled: flags.speculativeEnabled == true)
            let traversal = try await vlSpecDecode(
                benchmark, load: load, model: model, corpus: lease.corpus,
                readinessGate: readinessGate, observer: observer, isCancelled: isCancelled,
                progress: progress)
            resultData = traversal.resultData
            traversed = traversal
        case let .asrThroughput(benchmark):
            let model = try ResolvedGgufVisionModel(request.model.bound)
            // Every refusal before the weights: the recipe is read off the projector's header.
            let cell = try AsrCell(benchmark, recipe: AsrRecipe.forMmproj(model.mmproj))
            try readinessGate()
            (resultData, flags, clientExtras) = try await asrThroughput(
                benchmark, request, cell: cell, model: model, flags: flags,
                readinessGate: readinessGate, observer: observer)
        case let .asrMaxMemory(benchmark):
            let model = try ResolvedGgufVisionModel(request.model.bound)
            let cell = try AsrCell(benchmark, recipe: AsrRecipe.forMmproj(model.mmproj))
            // Ungated: a peak is thermal-invariant.
            (resultData, flags, clientExtras) = try await asrMaxMemory(
                benchmark, request, cell: cell, model: model, flags: flags, observer: observer)
        }
        var response = traversed
            ?? RunResponse(resultData: resultData, runtimeOutput: RuntimeOutput())
        response.runtimeOutput = RuntimeOutput(runtimeStderr: LlamaCpp.capturedLoadLog)
        response.runtimeFlags = LlamaRuntimeFlags.reported(request, ran: flags)
        response.clientExtras = try clientExtras?.json()
        return response
    }

    /// The flags `forRun` resolved. The overlay fills every knob, so each `??` is a fail-open,
    /// not a live branch.
    struct LoadSettings {
        let numberGpuLayers: UInt32
        let ctxSize: UInt32
        let nUbatch: UInt32
        let threads: UInt32
        let swaFull: Bool
        let loadMode: LlamacppLoadMode

        init(_ flags: RuntimeFlags, _ definition: BenchmarkDefinition) {
            numberGpuLayers = flags.numberGpuLayers ?? LlamaCpp.defaultNumberGpuLayers
            ctxSize = flags.ctxSize ?? LlamaRuntimeFlags.ctxSize(for: definition)
            nUbatch = flags.nUbatch ?? LlamaCpp.defaultNUbatch
            threads = flags.threads ?? LlamaCpp.defaultThreads
            swaFull = flags.swaFull ?? LlamaCpp.defaultSwaFull
            loadMode = flags.loadMode ?? LlamaCpp.defaultLoadMode
        }

        func withInference<T>(_ model: ResolvedGgufTextModel,
                              _ body: (Inference) async throws -> T) async throws -> T {
            try await Pipette.withInference(
                path: model.path, nGpuLayers: numberGpuLayers, contextSize: ctxSize,
                nUbatch: nUbatch, threads: threads, swaFull: swaFull, loadMode: loadMode, body)
        }
    }

    /// Run `body` with the context sized from the projector's count for one frame, as the
    /// crate sizes it from its calibration helper; a pinned `ctx_size` is kept. Answers the
    /// flags the cell ran with.
    static func withCalibratedVision<T>(
        _ request: RunRequest, _ model: ResolvedGgufVisionModel, flags: RuntimeFlags,
        width: UInt32, height: UInt32, observer: MeasurementObserver,
        _ body: (VisionInference) async throws -> T
    ) async throws -> (T, RuntimeFlags) {
        var ran = flags
        let load = LoadSettings(flags, request.benchmark)
        let result = try await Pipette.withVisionInference(
            path: model.backbone, mmprojPath: model.mmproj, nGpuLayers: load.numberGpuLayers,
            threads: load.threads, swaFull: load.swaFull, loadMode: load.loadMode,
            window: { mtmd in
                // Calibration is not an attempt (`attempt_timeout_seconds`).
                let perImage = try observer.outside {
                    try LlamaMtmd.perImageTokens(mtmd, width: width, height: height)
                }
                ran = try LlamaRuntimeFlags.forRun(request, perImageTokens: UInt32(perImage))
                // Both from the counted flags: the estimate's micro-batch is clamped to the
                // estimate's window, which the count can exceed.
                let counted = LoadSettings(ran, request.benchmark)
                return (counted.ctxSize, counted.nUbatch)
            }, body)
        return (result, ran)
    }

    // MARK: - Prompt-seed self-check (diagnostics)

    /// The counts `PromptSeed.buildPromptText` reaches under this model's real tokenizer,
    /// for `metrics=promptseed`; the unit tests only cover a synthetic counter.
    static func promptSeedCounts(modelPath: String, nGpuLayers: UInt32, contextSize: UInt32,
                                 nUbatch: UInt32, targets: [Int]) async throws -> [(target: Int, got: Int)] {
        let model = try LlamaCpp.load(path: modelPath, nGpuLayers: nGpuLayers,
                                      contextSize: contextSize, nUbatch: nUbatch,
                                      threads: LlamaCpp.defaultThreads,
                                      swaFull: LlamaCpp.defaultSwaFull,
                                      loadMode: LlamaCpp.defaultLoadMode)
        defer { model.free() }
        var counts: [(target: Int, got: Int)] = []
        for target in targets {
            let text = try await PromptSeed.buildPromptText(targetTokens: target) {
                try LlamaCpp.tokenize(model, $0, addSpecial: true).count
            }
            counts.append((target, try LlamaCpp.tokenize(model, text, addSpecial: true).count))
        }
        return counts
    }

    // MARK: - Prompt seeding

    /// `PromptSeed.promptSeedText` tiled and truncated to exactly `target` tokens.
    static func seedTokens(_ inf: Inference, target: Int, addSpecial: Bool) throws -> [Int32] {
        guard target > 0 else { return [] }
        let unit = try inf.tokenize(PromptSeed.promptSeedText, false)
        guard !unit.isEmpty else { throw RuntimeError.engine("seed corpus tokenized to zero tokens") }
        // Merges across the tiling boundary can yield fewer than repeats x unit tokens.
        var repeats = max(1, (target + unit.count - 1) / unit.count)
        var ids = try inf.tokenize(String(repeating: PromptSeed.promptSeedText, count: repeats), addSpecial)
        var grows = 0
        while ids.count < target && grows < 8 {
            repeats *= 2
            ids = try inf.tokenize(String(repeating: PromptSeed.promptSeedText, count: repeats), addSpecial)
            grows += 1
        }
        // A degenerate tokenizer can still fall short after the bounded growth.
        guard ids.count >= target else {
            throw RuntimeError.engine("seed produced \(ids.count) tokens, needed \(target)")
        }
        return Array(ids.prefix(target))
    }
}
