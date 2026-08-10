import Foundation
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import MLXNN
import Tokenizers

/// In-process **MLX** benchmark runtime — the Swift counterpart to the Rust/llama
/// `run_benchmark` FFI. It executes the *same* `BenchmarkCatalog` definitions on
/// any `mlx-swift-lm` model and emits the **same result-JSON shape** the
/// llama path produces (`prefill_time_ms` / `decode_time_ms` / `max_ram_bytes`),
/// so the downstream payload → CSV → submission pipeline is identical for both
/// runtimes. No Rust/FFI is involved.
///
/// Measurement discipline mirrors the Rust path: 1 warm-up + 5 measured reps with a
/// cooldown gate before each measured rep; throughput benches report mean ± stddev
/// in milliseconds. Throughput/memory prompts are synthetic in-vocab token ids
/// (those benches only care about token *count*, so no tokenizer is loaded); e2e is
/// the exception — it tokenizes a real exact-length prompt in-window to match the
/// CLI's true latency.
///
/// Prefill feeds the KV/conv cache in `prefillChunk`-token windows with `asyncEval`
/// so the activation peak is bounded and the GPU pipeline stays full (the fix
/// validated in v0; the model's own `prepare()` does the same for production decode).
///
/// The implementation is split across this folder by layer: this file owns the
/// runtime identity, model lifecycle, observability, and the single `run` entry;
/// `MLXGenerate` the MLX-graph generation/timing primitives; `MLXBenchmark` the
/// definition → result methodology; `MLXProbes` the calibration / coherence tools
/// — the structural twin of the `Llama/` split (`LlamaCpp` / `LlamaBenchmark`).
nonisolated enum MLXRuntime {
    /// Matches the Rust `PERF_NUM_MEASUREMENT_RUNS` (+ 1 warm-up) so MLX and llama
    /// numbers are averaged over the same number of reps.
    static let measurementRuns = 5

    /// Chunk size when a cell names none — MLX has no other load setting.
    static let defaultPrefillChunk: UInt32 = 512

    /// Composite `runtime_version` submitted for MLX results: each component of
    /// the MLX stack labeled with its own version so a result traces
    /// unambiguously to each, even though they pin differently (semver vs.
    /// commit). The values come from `MLXBuildInfo`, generated from
    /// `Package.resolved` at build time by `ios/gen-mlx-build-info.sh` (the same
    /// build-time-codegen pattern as `LlamaCppBuildInfo`), so the
    /// reported versions track the SwiftPM pins automatically.
    ///
    /// - `mlx-swift`: the core tensor library (Metal compute / numerics).
    /// - `mlx-swift-lm`: the MLXLLM model/inference code (routing, sampling,
    ///   generation — e.g. LFM2-MoE); revision-pinned, reported as a 9-char short
    ///   hash (same length as the llama.cpp commit) until a release tag carries
    ///   the fix.
    /// - `swift-transformers`: the tokenizer (text → token ids); tokenization
    ///   changes the prompt encoding and therefore the output.
    static let submissionRuntimeVersion =
        "mlx-swift=\(MLXBuildInfo.mlxSwiftVersion) "
        + "mlx-swift-lm=\(MLXBuildInfo.mlxSwiftLMRevision) "
        + "swift-transformers=\(MLXBuildInfo.swiftTransformersVersion)"

    // MARK: - Milestone logging

    // Phase breadcrumbs (load / benchmark / eval) with the jetsam-relevant memory
    // figures at each. Logged at `.info`, not `.debug` — these are permanent
    // diagnostics for tracking the on-device memory ramp, so they stay visible in
    // Console.app / `log stream` (and in Sentry breadcrumbs) without turning on
    // debug output.

    /// Process physical footprint (the jetsam-relevant number) in MB. Wraps the
    /// shared `ProcessMemory.physFootprintBytes` — the same counter `max_memory_usage`
    /// now reports for both runtimes.
    static func footprintMB() -> Double {
        Double(ProcessMemory.physFootprintBytes()) / 1_048_576
    }

    static func logMem(_ label: String) {
        let mb = 1_048_576.0
        let fp = footprintMB()
        let active = Double(MLX.Memory.activeMemory) / mb
        let peak = Double(MLX.Memory.peakMemory) / mb
        AppLog.mlx.info(String(
            format: "%@ footprint=%.0fMB mlxActive=%.0fMB mlxPeak=%.0fMB",
            label, fp, active, peak))
    }

    // MARK: - Load / unload

    /// Load a model directory into GPU memory (weights eval'd resident) and
    /// return it. The caller owns the model for as long as it needs it; release
    /// the weight buffers afterward with `releaseModel()` — or use
    /// `withFreshModel`, which does it for you.
    static func loadModel(path: String) async throws -> any LanguageModel {
        let dir = URL(fileURLWithPath: path)
        do {
            logMem("load.start \(dir.lastPathComponent)")
            let data = try Data(contentsOf: dir.appendingPathComponent("config.json"))
            let base = try JSONDecoder().decode(BaseConfiguration.self, from: data)
            // Build whatever model `config.json` declares (`model_type`) via the
            // mlx-swift-lm type registry, so the runtime benchmarks any supported
            // model generically — not just LFM2-MoE. `createModel` is actor-isolated,
            // hence the `await` (and why `loadModel` is `async`).
            let m = try await LLMTypeRegistry.shared.createModel(
                configuration: data, modelType: base.modelType)
            // `perLayerQuantization` supersedes the deprecated `quantization`;
            // `loadWeights` prefers it and ignores `quantization` when it's present,
            // so we pass only the modern one.
            try loadWeights(
                modelDirectory: dir, model: m,
                perLayerQuantization: base.perLayerQuantization
            )
            logMem("load.afterWeights")
            eval(m)
            logMem("load.afterEval")
            return m
        } catch {
            throw RuntimeError.engine("MLX model load failed: \(error)")
        }
    }

    /// Release dropped weight buffers back to the OS. Dropping a model returns
    /// its buffers to MLX's GPU buffer cache, NOT to the OS; without this the
    /// next load stacks a second ~5 GB copy on top of the cached one and the app
    /// is jetsam-killed.
    static func releaseModel() {
        MLX.Memory.clearCache()
    }

    /// Load a model, hand it to `body`, then `releaseModel()` — the load → use →
    /// release lifecycle in one place so callers can't leak the weights.
    static func withFreshModel<T>(path: String, _ body: (any LanguageModel) async throws -> T) async throws -> T {
        let model = try await loadModel(path: path)
        defer { releaseModel() }
        return try await body(model)
    }

    // MARK: - Run

    /// Single MLX execution entry: fresh-load the model once, then run one
    /// benchmark on it, dispatching on type. Both the UI (`JobExecutor`) and
    /// headless (`HeadlessRunner`) paths call this, so the type → primitive mapping
    /// lives in one place rather than drifting per caller. The fresh load is owned
    /// here — every benchmark is measured from a clean load — so the primitives
    /// take an already-loaded model and aren't each named for "fresh". `eval` and
    /// e2e load a tokenizer up front (hence `async`); the throughput/memory
    /// primitives are sync.
    static func run(_ request: RunRequest, flags: RuntimeFlags,
                    evalCompletions: EvalCompletionsStore,
                    readiness: @escaping () -> ReadinessOutcome,
                    observer: RepObserver,
                    isCancelled: @escaping () -> Bool = { false },
                    progress: @escaping (BenchmarkProgress) -> Void = { _ in }
    ) async throws -> BenchmarkResult {
        // The bound bundle, checked to be a directory — the crate's `require_mlx_model_dir`,
        // called by the engine rather than by dispatch, as every `pipette-mlx` executor
        // calls it. This is also what refuses a model MLX cannot load.
        let modelPath = try MLXModels.requireMlxModelDir(request)
        let definition = request.benchmark
        // The prefill chunk `RunCell.dispatch` resolved for this cell, so what a result says it ran
        // with is what this runs with. `MLXRuntimeFlags.forRun` fills it; the `??` is the
        // fail-open for an absent flag, as on the llama side.
        let prefillChunk = Int(flags.nUbatch ?? defaultPrefillChunk)
        // max_memory_usage brackets the *whole* fresh-load + drive with the process
        // footprint sampler (see `maxMemory`), so it owns its own load rather than
        // routing through `withFreshModel` → `runOnModel` like the timing benches.
        if case .maxMemoryUsage(_, let prefill) = definition {
            return try await maxMemory(modelPath: modelPath, prefillChunk: prefillChunk,
                                       prefillTokens: Int(prefill))
        }
        if case .eval(_, _, _, let maxTokens, _, let samples) = definition {
            // Eval needs a tokenizer (chat template + de/encode); load it before the
            // model so generation stays pure token space.
            let tokenizer = try await #huggingFaceTokenizerLoader().load(
                from: URL(fileURLWithPath: modelPath))
            // Opened on this arm alone — only an eval has samples to resume. The llama
            // path defers the same open into a closure because its `runOn` is a test seam.
            let checkpoint = try evalCompletions.open(request: request)
            return try await withFreshModel(path: modelPath) {
                try Self.evalCompletions(
                    $0, samples: samples ?? [], maxTokens: Int(maxTokens),
                    tokenizer: tokenizer, prefillChunk: prefillChunk, checkpoint: checkpoint,
                    isCancelled: isCancelled, progress: progress)
            }
        }
        // e2e times tokenization in-window (CLI parity), so it also needs the
        // tokenizer up front; the throughput/memory benches don't (synthetic ids).
        var tokenizer: (any MLXLMCommon.Tokenizer)?
        if case .endToEndLatency = definition {
            tokenizer = try await #huggingFaceTokenizerLoader().load(from: URL(fileURLWithPath: modelPath))
        }
        return try await withFreshModel(path: modelPath) {
            try await runOnModel($0, definition, prefillChunk: prefillChunk, readiness: readiness,
                                 observer: observer, progress: progress, tokenizer: tokenizer)
        }
    }
}
