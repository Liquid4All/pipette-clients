import Foundation

// The Swift mirror of plan-types `ClientRunSpec` — the work payload a claim
// carries (`crates/pipette-plan-types/src/plan.rs`).
//
// This is the single point where the claim's opaque `spec` becomes typed, as
// `run_spec_from_claim` is on the Rust side. Everything downstream reads the
// typed value; nothing reaches back into the raw JSON.
//
// Unlike the flag groups, `Model` and `Runtime` are *not* `deny_unknown_fields`
// in plan-types — they use `#[serde(flatten)]`, which serde forbids combining
// with it — so unknown keys are tolerated here too rather than tightened past
// what the Rust side does.

/// A GitHub source checkout: plan-types `SourceRepository`, flattened into the
/// runtime that carries it. `repository_url` defaults; `repository_version` is
/// required and non-empty.
/// The cell this claim asks for, ready to configure a run.
///
/// `benchmark` is duplicated on the wire — the server resolves the catalog from
/// the envelope, the cell runs from the payload — so [`ClientRunSpec.from`]
/// agrees them before anything else reads either.
/// Authored as well as decoded, which is what lets one type describe a cell whatever its
/// origin: a claim decodes one, and the app builds one from its own settings — the crate's
/// `into_client_run_spec`, which the CLI's manual `benchmarks run` uses to reach the same
/// `run_cell` the claim worker reaches.
nonisolated struct ClientRunSpec: Codable, Sendable {
    let benchmark: String
    let model: Model
    let runtime: Runtime
    let runtimeFlags: RuntimeFlagRef?
    let modelFlags: ModelFlagRef?
    /// The readiness gate this cell runs under. iOS declares the timing cells' variants
    /// and no others: `eval` and `vl` carry server settings an in-process engine has
    /// nothing to apply, so a group naming one of those is still refused.
    let benchmarkFlags: BenchmarkFlagRef?

    enum CodingKeys: String, CodingKey {
        case benchmark, model, runtime
        case runtimeFlags = "runtime_flags"
        case modelFlags = "model_flags"
        case benchmarkFlags = "benchmark_flags"
    }

    init(benchmark: String, model: Model, runtime: Runtime,
         runtimeFlags: RuntimeFlagRef? = nil, modelFlags: ModelFlagRef? = nil,
         benchmarkFlags: BenchmarkFlagRef? = nil) {
        self.benchmark = benchmark
        self.model = model
        self.runtime = runtime
        self.runtimeFlags = runtimeFlags
        self.modelFlags = modelFlags
        self.benchmarkFlags = benchmarkFlags
    }

    func encode(to encoder: any Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(benchmark, forKey: .benchmark)
        try c.encode(model, forKey: .model)
        try c.encode(runtime, forKey: .runtime)
        try c.encodeIfPresent(runtimeFlags, forKey: .runtimeFlags)
        try c.encodeIfPresent(modelFlags, forKey: .modelFlags)
        try c.encodeIfPresent(benchmarkFlags, forKey: .benchmarkFlags)
    }
}

nonisolated extension ClientRunSpec {
    /// This cell with its load settings replaced — everything else it named travels
    /// unchanged. A caller that rebuilt the spec instead would have to restate every group,
    /// which is how one goes missing.
    func replacingRuntimeFlags(_ ref: RuntimeFlagRef?) -> ClientRunSpec {
        ClientRunSpec(benchmark: benchmark, model: model, runtime: runtime,
                      runtimeFlags: ref, modelFlags: modelFlags,
                      benchmarkFlags: benchmarkFlags)
    }

    /// A cell authored on this device from the app's load settings — the crate's
    /// `into_client_run_spec`, which is how its manual `benchmarks run` reaches the same
    /// execution path a claim reaches.
    ///
    /// The runtime is derived: this client can only be what it compiled as, so there is one
    /// answer to fill it with. Knobs are offered only where the cell's variant declares
    /// them — an MLX cell carries the prefill chunk alone, and handing it a llama setting
    /// would be refused at `resolve`, taking the chunk down with it.
    ///
    /// The context window is not among them. A cell that does not name one is answered by
    /// the engine from its benchmark (`LlamaRuntimeFlags.contextSize`), as `for_server`
    /// answers it upstream — so the app states its settings and leaves the sizing where the
    /// reference client keeps it.
    ///
    /// `benchmarkType` is optional because a benchmark id this build cannot classify has no
    /// axis to key an entry on; such a cell carries none, exactly as one naming no flags
    /// does.
    static func authored(
        benchmarkId: String, benchmarkType: BenchmarkType?, model: Model,
        numberGpuLayers: UInt32? = nil, ctxSize: UInt32? = nil,
        nUbatch: UInt32? = nil, threads: UInt32? = nil
    ) -> ClientRunSpec {
        let runtime = Runtime.thisBuild(for: model)
        guard let benchmarkType else {
            return ClientRunSpec(benchmark: benchmarkId, model: model, runtime: runtime)
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
