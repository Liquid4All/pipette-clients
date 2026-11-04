import Foundation

// pipette-mgmt's `crates/pipette-plan-types/src/cell/mod.rs`: `ClientRunSpec`,
// the work payload a claim carries, and `is_compatible`.
//
// Unlike the flag groups, `Model` and `Runtime` tolerate unknown keys: the crate's use
// `#[serde(flatten)]`, which serde cannot combine with `deny_unknown_fields`.

/// The cell a claim asks for, or one the app authors from its own settings.
///
/// `benchmark` is duplicated on the wire (the server resolves the catalog from the
/// envelope, the cell runs from the payload), so `runSpecFromClaim(_:)` checks they agree
/// before anything reads either.
///
/// The flag groups are held in their flat wire form (`…FlagRef`), where the crate holds
/// the resolved variants: the claim path reports each group's refusal by name, which needs
/// the group decoded before it is resolved.
nonisolated struct ClientRunSpec: Codable, Sendable {
    let benchmark: BenchmarkId
    let model: Model
    let runtime: Runtime
    let runtimeFlags: RuntimeFlagRef?
    let modelFlags: ModelFlagRef?
    let benchmarkFlags: BenchmarkFlagRef?

    enum CodingKeys: String, CodingKey {
        case benchmark, model, runtime
        case runtimeFlags = "runtime_flags"
        case modelFlags = "model_flags"
        case benchmarkFlags = "benchmark_flags"
    }

    init(benchmark: BenchmarkId, model: Model, runtime: Runtime,
         runtimeFlags: RuntimeFlagRef? = nil, modelFlags: ModelFlagRef? = nil,
         benchmarkFlags: BenchmarkFlagRef? = nil) {
        self.benchmark = benchmark
        self.model = model
        self.runtime = runtime
        self.runtimeFlags = runtimeFlags
        self.modelFlags = modelFlags
        self.benchmarkFlags = benchmarkFlags
    }

    /// A cell authored on this device from an id it already holds as a string, such as a
    /// job cell's `benchmarkId`. A claim never reaches this; its id is validated at decode.
    ///
    /// Takes a `String` because `JobCell.benchmarkId` is persisted and an older manifest can
    /// hold `remote/prefill_throughput_512`, which `BenchmarkId` refuses; decoding it as one
    /// would lose the whole saved job. Asserts rather than throws: an authored id has
    /// passed `SourcedBenchmarkId.init(parsing:)` or comes from the server-minted catalog.
    init(benchmark: String, model: Model, runtime: Runtime,
         runtimeFlags: RuntimeFlagRef? = nil, modelFlags: ModelFlagRef? = nil,
         benchmarkFlags: BenchmarkFlagRef? = nil) {
        assert(BenchmarkId.validate(benchmark), "`\(benchmark)` is not a benchmark id")
        self.init(benchmark: BenchmarkId(validated: benchmark), model: model, runtime: runtime,
                  runtimeFlags: runtimeFlags, modelFlags: modelFlags,
                  benchmarkFlags: benchmarkFlags)
    }

    /// Every group's key is written, `null` when unset, as the crate serializes it, so a
    /// spec this client writes is the shape a Rust client writes. Decode accepts the key
    /// absent or `null`.
    func encode(to encoder: any Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(benchmark, forKey: .benchmark)
        try c.encode(model, forKey: .model)
        try c.encode(runtime, forKey: .runtime)
        try c.encode(runtimeFlags, forKey: .runtimeFlags)
        try c.encode(modelFlags, forKey: .modelFlags)
        try c.encode(benchmarkFlags, forKey: .benchmarkFlags)
    }
}

nonisolated extension ClientRunSpec {
    /// Rebuilding the spec instead would mean restating every group, which is how one
    /// goes missing.
    func replacingRuntimeFlags(_ ref: RuntimeFlagRef?) -> ClientRunSpec {
        ClientRunSpec(benchmark: benchmark, model: model, runtime: runtime,
                      runtimeFlags: ref, modelFlags: modelFlags,
                      benchmarkFlags: benchmarkFlags)
    }

    /// A cell authored on this device from the app's load settings, as pipette-cli's
    /// `into_client_run_spec` authors one for a manual `benchmarks run`.
    ///
    /// Knobs are set only where the cell's variant declares them: an MLX cell carries the
    /// prefill chunk alone, and a llama setting would be refused at `resolve`, taking the
    /// chunk down with it. A `nil` `benchmarkType` (an id this build cannot classify) has
    /// no axis to key an entry on, so the cell carries no flags.
    static func authored(
        benchmarkId: String, benchmarkType: BenchmarkType?, model: Model,
        numberGpuLayers: UInt32? = nil, ctxSize: UInt32? = nil,
        nUbatch: UInt32? = nil, threads: UInt32? = nil
    ) -> ClientRunSpec {
        let runtime = Runtime.thisBuild(for: model)
        guard let benchmarkType else {
            return ClientRunSpec(benchmark: benchmarkId, model: model, runtime: runtime)
        }
        // No `RuntimeFlags` variant exists for either runtime, and `RunCell.prepare`
        // refuses an entry with no variant rather than dropping it.
        switch runtime {
        case .appleFoundation, .onnxGenaiIosPipette:
            return ClientRunSpec(benchmark: benchmarkId, model: model, runtime: runtime)
        case .llamacppIosPipette, .mlxIosPipette:
            break
        }
        var ref = RuntimeFlagRef(benchmarkType: benchmarkType,
                                 runtimeType: RuntimeType.of(runtime),
                                 modelType: ModelType.of(model))
        if case .llamacppIosPipette = runtime {
            ref.numberGpuLayers = numberGpuLayers
            ref.ctxSize = ctxSize
            ref.threads = threads
        }
        ref.nUbatch = nUbatch
        return ClientRunSpec(benchmark: benchmarkId, model: model, runtime: runtime,
                             runtimeFlags: ref)
    }
}

/// Whether `runtime` can load `model`'s format: the crate's `is_compatible`, without the
/// desktop runtimes' rows (no iOS `Runtime` can reach them).
///
/// **A missing row is silent.** The `default` fails closed, so a new `Model` case compiles
/// and then refuses every cell that names it. `ModelType.of` and `RuntimeKind.init` are
/// exhaustive; this table has to be remembered.
nonisolated func isCompatible(_ model: Model, _ runtime: Runtime) -> Bool {
    switch (model, runtime) {
    case (.ggufText, .llamacppIosPipette): true
    case (.ggufVision, .llamacppIosPipette): true
    // The engine links llama.cpp's speculative loop (`ios/native/pipette_vlspec.h`).
    // Whether this model's drafter arrangement can be attached is decided at load.
    case (.ggufVisionSpeculative, .llamacppIosPipette): true
    case (.mlx, .mlxIosPipette): true
    case (.onnxGenai, .onnxGenaiIosPipette): true
    case (.appleFoundationText, .appleFoundation): true
    default: false
    }
}
