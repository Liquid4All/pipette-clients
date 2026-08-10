import Foundation

// One cell as this client runs it: what an engine is handed. Mirrors
// `pipette-plan-types/src/run.rs`.

/// Plan identity vs host-bound form of one artifact — the crate's `DeclaredBound<T>`.
///
/// The two halves are the same value in two forms. For something the host holds but
/// cannot relocate — Apple Foundation, whose weights ship with the OS — they are equal,
/// which `alreadyBound` names rather than restating.
nonisolated struct DeclaredBound<T> {
    /// Plan coordinate: what descriptors, the storage key and the record are taken from.
    let declared: T
    /// Launch form on this host. Meaningless off it, so it never reaches the wire.
    let bound: T

    /// The two halves as the same value — nothing relocated it.
    static func alreadyBound(_ value: T) -> DeclaredBound<T> {
        DeclaredBound(declared: value, bound: value)
    }
}

extension DeclaredBound: Equatable where T: Equatable {}
extension DeclaredBound: Sendable where T: Sendable {}

/// What an engine is handed for one cell — the crate's `RunRequest`.
///
/// Both axes carry the pair, as upstream does. `declared` is the plan coordinate the
/// record and every descriptor are taken from; `bound` is the launch form on this host —
/// for a model, the same `Model` with its source rewritten to the `absolute*` arms that
/// `bindUnder` produces, which is what an engine's `require*` matches on.
///
/// The runtime needs the pair because a plan declares one build and this binary is
/// another. The model needs it because the plan names a repo and the engine needs a path.
///
/// Not ported: `benchmark_flags`, which no iOS cell can carry (a group naming one is
/// refused at parse), and `model_flags`, which nothing on this side produces yet — the
/// cell drops `enable_thinking` before a run is built, so a field here would be
/// permanently `nil`.
nonisolated struct RunRequest: Sendable {
    let runtime: DeclaredBound<Runtime>
    let model: DeclaredBound<Model>
    /// Authored form, as upstream: an engine reads the values a cell set and supplies its
    /// own where the cell left them unset.
    let runtimeFlags: RuntimeFlags?
    /// The readiness gate this cell asks for. `nil` on a cell that names none, and on the
    /// cells that carry no variant at all (`eval`, `vl`, `max_memory_usage`).
    let benchmarkFlags: BenchmarkFlags?
    let benchmark: BenchmarkDefinition
}

extension RunRequest {
    /// The cell's flags in flat form, or an all-unset ref when the run carries none — the
    /// crate's `runtime_flags_ref`. An engine starts here, overlays the values its
    /// execution supplied, and converts back for `RunResponse.runtimeFlags`, so the flags
    /// a result reports are routed and validated like an authored entry rather than echoed
    /// from the request.
    ///
    /// The axis check is a backstop, as upstream's is: `ClientRunSpec.validated` refuses an entry
    /// naming another cell, and `RunCell.prepare` derives the flags from the cell they
    /// belong to.
    nonisolated func runtimeFlagsRef() throws -> RuntimeFlagRef {
        let axes: FlagAxes = (benchmark.type, RuntimeType.of(runtime.declared),
                              ModelType.of(model.declared))
        guard let runtimeFlags else {
            return RuntimeFlagRef(benchmarkType: axes.benchmark, runtimeType: axes.runtime,
                                  modelType: axes.model)
        }
        guard runtimeFlags.axes == axes else {
            throw RuntimeFlagsAxisMismatch(carried: runtimeFlags.axes, cell: axes)
        }
        return RuntimeFlagRef(runtimeFlags)
    }
}

/// What one run answers with — the crate's `RunResponse`. The caller pairs it with the
/// `RunRequest` it handed over, which is where the submitted descriptors come from.
///
/// Not ported: `command` and `executable`, which name a shelled-out invocation a phone
/// never makes (they already go unfilled for desktop MLX).
///
/// `stdout` is absent for the same reason: an in-process engine writes none. `stderr` is
/// here because one is produced — llama.cpp routes its log through the callback
/// `LlamaCpp` installs.
nonisolated struct RunResponse: Sendable {
    let resultData: BenchmarkResult
    /// The gate this run was held by — reported, not echoed: a cell that waived the
    /// temperature criterion is not comparable to one that didn't, and the record is the
    /// only place that difference survives.
    var benchmarkFlags: BenchmarkFlags?
    /// Filled by the caller from the marks the engine made through its thermal gate, as
    /// upstream does: the probe and the series belong to the caller, not the engine.
    var thermal: RunThermal
    /// The request's flags as the run resolved them: the cell's values with the engine's
    /// own where it left a knob unset, so a reader reports what the run loaded with rather
    /// than defaulting again. `nil` for a cell whose axes name no flags variant — Apple
    /// Foundation, which has none upstream either.
    var runtimeFlags: RuntimeFlags?
    /// What the engine wrote while this run loaded its model. Empty for a runtime that
    /// routes no log through a callback this build installs.
    var stderr: String

    /// The metrics half, provenance empty — the crate's `RunResponse::new`.
    init(resultData: BenchmarkResult) {
        self.resultData = resultData
        self.thermal = RunThermal()
        self.runtimeFlags = nil
        self.stderr = ""
    }
}
