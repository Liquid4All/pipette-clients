import Foundation

/// Native-Swift llama.cpp benchmark methodology — stateless functions over an
/// `Inference` witness, the structural twin of `MLXRuntime`'s kernels. No stored
/// engine, no JSON: the loaded model is assembled inside the execution scope
/// (`withInference`) and driven through the shared `BenchmarkMeasurement` core.
///
/// Methodology mostly mirrors the Rust harness (`native/benchmarks.rs`): the readiness
/// gate runs before every measured rep; decode is timed alone
/// (prefill excluded) via the ignore-EOG fixed-count primitive; `max_memory_usage`
/// samples the peak across the model load. e2e is the deliberate exception — it
/// times tokenize + prefill + decode (tokenization in-window), matching the
/// llama.cpp CLI's "true" end-to-end latency (which posts a string to /completion)
/// rather than the in-process native kernel, which times prefill + decode only.
///
/// The warm-up is the deliberate departure: it runs the cell's *own* shape, one untimed
/// rep, where the crate warms a fixed 32/50. Metal compiles pipelines per specialization,
/// and a 32-token prompt takes the matrix-vector path while a 1024-token prefill takes the
/// batched matrix-matrix one — so the fixed shape left kernels to compile inside measured
/// rep 1. Matches the MLX path, which has always warmed with `body`.
nonisolated enum LlamaBenchmark {
    /// Measured reps after the warm-up (`REPS`), transcribed from
    /// `pipette_ops::measurement`. Not bridged: the Swift engine reaches Rust only through
    /// the UniFFI seam, which carries schema and prompt seed, not constants — so this is a
    /// copy, and changing one means changing both.
    static let measurementRuns = 5

    // MARK: - Entry

    /// Assemble the model fresh, run one benchmark, return the typed result. The
    /// throughput/eval kernels assemble once (`withInference`) and dispatch through
    /// `runOn`; `max_memory_usage` brackets the assembly with the peak sampler.
    /// `gate` throws the llama error on a non-ready readiness outcome.
    static func run(_ request: RunRequest, flags: RuntimeFlags,
                    evalCompletions: EvalCompletionsStore,
                    gate: () throws -> Void,
                    observer: RepObserver,
                    isCancelled: () -> Bool = { false },
                    progress: (BenchmarkProgress) -> Void = { _ in }) async throws -> BenchmarkResult {
        let definition = request.benchmark
        // The bound weights, checked to be on disk before anything is loaded or measured.
        // `requireGgufText`, not a whichever-arm helper: upstream calls it from every text
        // execute module and `require_gguf_vision` from `vl_throughput` alone, so a vision
        // model on a timing benchmark is refused here exactly as its flags are.
        let path = try LlamaModels.requireGgufText(request)
        // The flags `Engine` resolved for this cell — the plan values with this engine's
        // defaults overlaid — so what a result says it loaded with is what this loads with.
        //
        // The overlay fills every one of them, so each `??` is the fail-open a tool applies
        // for an absent flag rather than a live branch: `bench::args_for` omits it and lets
        // llama-bench default it. A knob the overlay stops filling therefore loads on the
        // engine default instead of failing the run.
        let nGpuLayers = flags.numberGpuLayers ?? LlamaCpp.defaultNumberGpuLayers
        let contextSize = flags.ctxSize ?? LlamaRuntimeFlags.contextSize(for: definition)
        let nUbatch = flags.nUbatch ?? LlamaCpp.defaultNUbatch
        let threads = flags.threads ?? LlamaCpp.defaultThreads
        let swaFull = flags.swaFull ?? LlamaCpp.defaultSwaFull
        switch definition {
        case .maxMemoryUsage(_, let prefill):
            return try await maxMemory(path: path, nGpuLayers: nGpuLayers,
                                       contextSize: contextSize, nUbatch: nUbatch,
                                       threads: threads, swaFull: swaFull,
                                       prefillTokens: Int(prefill))

        case .vlThroughput:
            throw RuntimeError.unsupported(definition.benchmarkId)

        case .prefillThroughput, .decodeThroughput, .endToEndLatency, .eval:
            return try await withInference(path: path, nGpuLayers: nGpuLayers,
                                           contextSize: contextSize, nUbatch: nUbatch,
                                           threads: threads, swaFull: swaFull) {
                try await runOn(definition, $0, gate: gate, observer: observer,
                                openCheckpoint: { try evalCompletions.open(request: request) },
                                isCancelled: isCancelled, progress: progress)
            }
        }
    }

    /// Dispatch a benchmark on an already-assembled `Inference` — the engine-agnostic,
    /// model-free core (unit-testable with a fake witness, no load). `max_memory_usage`
    /// (needs to bracket the load) and `vl_throughput` (unsupported) are owned by `run`.
    /// `progress` reports semantic intra-cell events (`.attempt` per rep, `.sample`
    /// per eval row); the UI layer decides how to present each.
    ///
    /// `openCheckpoint` is a closure rather than the store itself so the session is minted
    /// only on the arm that has samples to resume — a throughput cell would otherwise leave
    /// an empty checkpoint behind — and so this core stays callable from a test with a fake
    /// witness and no `RunRequest` to digest.
    static func runOn(_ definition: BenchmarkDefinition, _ inf: Inference,
                      gate: () throws -> Void,
                      observer: RepObserver,
                      openCheckpoint: () throws -> EvalCompletionSession? = { nil },
                      isCancelled: () -> Bool = { false },
                      progress: (BenchmarkProgress) -> Void = { _ in }) async throws -> BenchmarkResult {
        switch definition {
        case .eval(_, _, _, let maxTokens, let mcqChoices, let samples):
            return try evalOn(inf, samples: samples ?? [], maxTokens: Int(maxTokens),
                              mcqChoices: mcqChoices, checkpoint: try openCheckpoint(),
                              isCancelled: isCancelled, progress: progress)
        case .prefillThroughput, .decodeThroughput, .endToEndLatency:
            return try await throughput(definition, inf, gate: gate, observer: observer,
                                        progress: progress)
        case .maxMemoryUsage, .vlThroughput:
            throw RuntimeError.unsupported(definition.benchmarkId)
        }
    }

    // MARK: - Throughput kernels (assemble once, rep)

    private static func throughput(_ definition: BenchmarkDefinition, _ inf: Inference,
                                   gate: () throws -> Void,
                                   observer: RepObserver,
                                   progress: (BenchmarkProgress) -> Void) async throws -> BenchmarkResult {
        switch definition {
        case .prefillThroughput(_, let prefill):
            let toks = try seedTokens(inf, target: Int(prefill), addSpecial: true)
            let rep = {
                inf.resetContext(); inf.resetSampler()
                return try BenchmarkMeasurement.timed { try inf.prefill(toks) }.milliseconds
            }
            let (mean, sd) = try await BenchmarkMeasurement.measure(
                label: definition.benchmarkType, runs: measurementRuns,
                warmup: { _ = try rep() }, gate: gate,
                observer: observer, onProgress: progress, body: rep)
            return .prefillThroughput(timeMs: mean, stddev: sd)

        case .decodeThroughput(_, let prefill, let decode):
            let toks = try seedTokens(inf, target: Int(prefill), addSpecial: true)
            let rep = {
                inf.resetContext(); inf.resetSampler()
                try inf.prefill(toks)                                  // untimed
                return try BenchmarkMeasurement.timed { try inf.decodeIgnoringEoG(Int(decode)) }.milliseconds
            }
            let (mean, sd) = try await BenchmarkMeasurement.measure(
                label: definition.benchmarkType, runs: measurementRuns,
                warmup: { _ = try rep() }, gate: gate,
                observer: observer, onProgress: progress, body: rep)
            return .decodeThroughput(timeMs: mean, stddev: sd)

        case .endToEndLatency(_, let prefill, let decode):
            // True end-to-end latency: tokenization is INSIDE the timed window, to
            // match the llama.cpp CLI (which posts a *string* to /completion, so the
            // server tokenizes in-band). The prompt text is built once, untimed — as
            // the CLI builds it via build_prompt_text outside its measured loop — to
            // tokenize to *exactly* `prefill` tokens, so the timed tokenize covers
            // the same token count the CLI's does. Each rep times tokenize + prefill
            // + decode; no truncation, the prompt is already exact.
            let promptText = try PromptSeed.buildPromptText(target: Int(prefill)) {
                try inf.tokenize($0, true).count
            }
            let rep = {
                inf.resetContext(); inf.resetSampler()
                return try BenchmarkMeasurement.timed {
                    try inf.prefill(inf.tokenize(promptText, true))
                    try inf.decodeIgnoringEoG(Int(decode))
                }.milliseconds
            }
            let (mean, sd) = try await BenchmarkMeasurement.measure(
                label: definition.benchmarkType, runs: measurementRuns,
                warmup: { _ = try rep() }, gate: gate,
                observer: observer, onProgress: progress, body: rep)
            return .endToEndLatency(timeMs: mean, stddev: sd)

        default:
            throw RuntimeError.unsupported(definition.benchmarkId)
        }
    }

    // MARK: - max_memory_usage (sample the process footprint across the assembly)

    private static func maxMemory(path: String, nGpuLayers: UInt32,
                                  contextSize: UInt32, nUbatch: UInt32, threads: UInt32,
                                  swaFull: Bool,
                                  prefillTokens: Int) async throws -> BenchmarkResult {
        try await ProcessMemory.maxMemoryBracket(label: "llama") {
            try await withInference(path: path, nGpuLayers: nGpuLayers, contextSize: contextSize,
                                    nUbatch: nUbatch, threads: threads, swaFull: swaFull) {
                try driveMaxMemory($0, prefillTokens: prefillTokens)
            }
        }
    }

    /// The work `max_memory_usage` does on the assembled model (single prefill +
    /// one decode to touch the KV path). Separated from `maxMemory` so it's
    /// exercisable with a fake `Inference`, while the peak sampling — which is
    /// inherently about the real load — is integration-tested on device.
    static func driveMaxMemory(_ inf: Inference, prefillTokens: Int) throws {
        let toks = try seedTokens(inf, target: prefillTokens, addSpecial: true)
        inf.resetContext(); inf.resetSampler()
        try inf.prefill(toks)
        try inf.decodeIgnoringEoG(1)
    }

    // MARK: - eval (per-sample, no rep loop)

    private static func evalOn(_ inf: Inference, samples: [EvalSample], maxTokens: Int,
                               mcqChoices: [String]?, checkpoint: EvalCompletionSession?,
                               isCancelled: () -> Bool,
                               progress: (BenchmarkProgress) -> Void) throws -> BenchmarkResult {
        // MCQ constrains to a single choice token (Rust `effective_max = 1`).
        let effectiveMax = mcqChoices != nil ? 1 : maxTokens
        let completions = try evalSamples(
            samples,
            resuming: checkpoint,
            // Cancel checkpoint + per-sample KV reset. Runs outside evalSamples' catch,
            // so a cancel aborts the run instead of becoming a failed sample.
            beforeSample: { _ in try cancellationGate(isCancelled); inf.resetContext(); inf.resetSampler() },
            progress: progress
        ) { sample in
            try inf.chatCompletion(messagesJSONString(sample.messages), effectiveMax, mcqChoices)
        }
        // Drops the file when every sample passed, keeps the failures otherwise. The return
        // is discarded, unlike upstream: our per-sample append is best-effort, so a
        // completion whose write failed is in `completions` but not in the session.
        checkpoint?.finalize()
        return .eval(completions: completions)
    }

    // MARK: - Prompt-seed self-check (diagnostics)

    /// For each `target`, build a prompt with `PromptSeed.buildPromptText` under
    /// THIS model's GGUF tokenizer and return the count it actually produced — so
    /// `metrics=promptseed` can verify exact sizing on a real (lumpy) tokenizer
    /// rather than the synthetic-counter unit tests. Loads the model for its
    /// tokenizer; runs no benchmark.
    static func promptSeedCounts(modelPath: String, nGpuLayers: UInt32, contextSize: UInt32,
                                 nUbatch: UInt32, targets: [Int]) throws -> [(target: Int, got: Int)] {
        let model = try LlamaCpp.load(path: modelPath, nGpuLayers: nGpuLayers,
                                      contextSize: contextSize, nUbatch: nUbatch,
                                      threads: LlamaCpp.defaultThreads,
                                      swaFull: LlamaCpp.defaultSwaFull)
        defer { model.free() }
        return try targets.map { target in
            let text = try PromptSeed.buildPromptText(target: target) {
                try LlamaCpp.tokenize(model, $0, addSpecial: true).count
            }
            return (target, try LlamaCpp.tokenize(model, text, addSpecial: true).count)
        }
    }

    // MARK: - Prompt seeding

    /// Tile `PromptSeed.corpus` and tokenize to exactly `target` tokens (truncating the
    /// tail). Real text content; only the count is parameterized.
    private static func seedTokens(_ inf: Inference, target: Int, addSpecial: Bool) throws -> [Int32] {
        guard target > 0 else { return [] }
        let unit = try inf.tokenize(PromptSeed.corpus, false)
        // target > 0 here (target 0 returned above). An empty unit can't tile to a
        // positive target, so route it to the same undersized-seed failure rather
        // than letting an empty seed escape.
        guard !unit.isEmpty else { throw RuntimeError.engine("seed corpus tokenized to zero tokens") }
        // Tile the corpus to at least `target` tokens then truncate to exactly
        // `target`. Tokenizer merges across the tiling boundary can yield fewer
        // tokens than repeats × unit, so grow until the count is reached (bounded).
        var repeats = max(1, (target + unit.count - 1) / unit.count)
        var ids = try inf.tokenize(String(repeating: PromptSeed.corpus, count: repeats), addSpecial)
        var grows = 0
        while ids.count < target && grows < 8 {
            repeats *= 2
            ids = try inf.tokenize(String(repeating: PromptSeed.corpus, count: repeats), addSpecial)
            grows += 1
        }
        // The grow loop is bounded, so a degenerate tokenizer could still fall short
        // of `target`. Truncating silently would emit a wrong-shape cell with no
        // error — fail loudly instead.
        guard ids.count >= target else {
            throw RuntimeError.engine("seed produced \(ids.count) tokens, needed \(target)")
        }
        return Array(ids.prefix(target))
    }

    private static func messagesJSONString(_ messages: [[String: String]]) throws -> String {
        let data = try JSONSerialization.data(withJSONObject: messages, options: [])
        return String(decoding: data, as: UTF8.self)
    }
}
