import Foundation

// What an engine is handed for one cell and what it answers: pipette-mgmt's
// `crates/pipette-plan-types/src/run.rs`.

/// The crate's `DeclaredBound<T>`: plan identity and host-bound form of one artifact.
nonisolated struct DeclaredBound<T> {
    /// Plan coordinate: what descriptors, the storage key and the record are taken from.
    let declared: T
    /// Launch form on this host. Meaningless off it, so it never reaches the wire.
    let bound: T

    /// For something the host cannot relocate: Apple Foundation, and this build's runtime.
    static func alreadyBound(_ value: T) -> DeclaredBound<T> {
        DeclaredBound(declared: value, bound: value)
    }
}

extension DeclaredBound: Equatable where T: Equatable {}
extension DeclaredBound: Sendable where T: Sendable {}

/// The crate's `RunRequest`.
///
/// The runtime is a pair as in the crate, but here both halves are this build
/// (`RunCell.prepare`): the descriptor a result carries must name what compiled. The model
/// is a pair because the plan names a repo and the engine needs a path (the `absolute*` arms
/// `bindUnder` produces, which an engine's `require*` matches on).
nonisolated struct RunRequest: Sendable {
    let runtime: DeclaredBound<Runtime>
    let model: DeclaredBound<Model>
    /// Authored form: an engine supplies its own values where the cell left them unset.
    let runtimeFlags: RuntimeFlags?
    /// Carried so the record reports it. No engine here reads one: the claim path refuses
    /// `enable_thinking` and `reasoning_effort`, so a run carries none or a variant with
    /// nothing set.
    var modelFlags: ModelFlags?
    /// `nil` on a cell that names none, and on the kinds with no variant (`eval` and the
    /// memory kinds).
    let benchmarkFlags: BenchmarkFlags?
    let benchmark: BenchmarkDefinition
}

extension RunRequest {
    /// From the declared halves, as the crate reads them.
    nonisolated var cellAxes: FlagAxes {
        (benchmark.benchmarkType, RuntimeType.of(runtime.declared), ModelType.of(model.declared))
    }

    /// The crate's `runtime_flags_ref`: an all-unset ref when the run carries none. An
    /// engine overlays the values it supplied and converts back for
    /// `RunResponse.runtimeFlags`, so a result's flags are validated, not echoed. The axis
    /// check is a backstop behind `ClientRunSpec.validated`.
    nonisolated func runtimeFlagsRef() throws -> RuntimeFlagRef {
        let axes = cellAxes
        guard let runtimeFlags else {
            return RuntimeFlagRef(benchmarkType: axes.benchmark, runtimeType: axes.runtime,
                                  modelType: axes.model)
        }
        guard runtimeFlags.matches(benchmark: benchmark.benchmarkType,
                                   runtime: runtime.declared, model: model.declared) else {
            throw RunRequestError.runtimeFlagsForAnotherCell(flags: runtimeFlags, cell: axes)
        }
        return RuntimeFlagRef(runtimeFlags)
    }

    /// The crate's `benchmark_flags_ref`, used as `runtimeFlagsRef` is with the resolved
    /// readiness block.
    nonisolated func benchmarkFlagsRef() throws -> BenchmarkFlagRef {
        let axes = cellAxes
        guard let benchmarkFlags else {
            return BenchmarkFlagRef(benchmarkType: axes.benchmark, runtimeType: axes.runtime,
                                    modelType: axes.model)
        }
        guard benchmarkFlags.matches(benchmark: benchmark.benchmarkType,
                                     runtime: runtime.declared, model: model.declared) else {
            throw RunRequestError.benchmarkFlagsForAnotherCell(flags: benchmarkFlags, cell: axes)
        }
        return BenchmarkFlagRef(benchmarkFlags)
    }
}

/// The crate's `RunRequestError`. `ClientRunSpec.validated` refuses an entry naming
/// another cell, so reaching either case is a construction bug.
nonisolated enum RunRequestError: Error, Equatable, LocalizedError {
    case runtimeFlagsForAnotherCell(flags: RuntimeFlags, cell: FlagAxes)
    case benchmarkFlagsForAnotherCell(flags: BenchmarkFlags, cell: FlagAxes)

    /// Hand-written: a case holding a tuple gets no synthesized `==`.
    static func == (lhs: Self, rhs: Self) -> Bool {
        switch (lhs, rhs) {
        case let (.runtimeFlagsForAnotherCell(lf, lc), .runtimeFlagsForAnotherCell(rf, rc)):
            lf == rf && lc == rc
        case let (.benchmarkFlagsForAnotherCell(lf, lc), .benchmarkFlagsForAnotherCell(rf, rc)):
            lf == rf && lc == rc
        default:
            false
        }
    }

    var errorDescription: String? {
        switch self {
        case let .runtimeFlagsForAnotherCell(flags, cell):
            "runtime flags for \(render(flags.axes)) are not for this \(render(cell)) cell"
        case let .benchmarkFlagsForAnotherCell(flags, cell):
            "benchmark flags for \(render(flags.axes)) are not for this \(render(cell)) cell"
        }
    }
}

private nonisolated func render(_ axes: FlagAxes) -> String {
    "\(axes.benchmark.rawValue) \u{d7} \(axes.runtime.rawValue) \u{d7} \(axes.model.rawValue)"
}

/// The crate's `RunResponse`, without `command` and `executable` (a phone shells out to
/// nothing) and `memory` (the engines here sample memory for `max_memory_usage` alone).
/// `client_extras` is the serialized object, not the crate's map: only ASR sets it.
nonisolated struct RunResponse: Sendable {
    let resultData: BenchmarkResultData
    /// `runtimeStdout` stays `nil` in-process; `runtimeStderr` is llama.cpp's load log from
    /// the callback `LlamaCpp` installs, `nil` for other engines.
    var runtimeOutput: RuntimeOutput
    /// A cell that waived the thermal criterion is not comparable to one that did not, and
    /// the record is the only place that difference survives.
    var benchmarkFlags: BenchmarkFlags?
    /// The cell's values with the engine's own where it left a knob unset, so a reader
    /// reports what the run loaded with. `nil` for a cell with no flags variant.
    var runtimeFlags: RuntimeFlags?
    /// sha256 of the corpus tar, submitted as `observation_dataset_digest` whether or not
    /// it matched the catalog's pin.
    var datasetDigest: String?
    /// sha256 over the traversal's replies, in order: `observation_output_digest`.
    var outputDigest: String?
    /// Never submitted: kept in the run's extras.
    var turns: [TurnObservation]
    /// A JSON object in a string, submitted as `client_extras`; `nil` sends none.
    var clientExtras: String?

    /// The crate's `RunResponse::new`.
    init(resultData: BenchmarkResultData, runtimeOutput: RuntimeOutput) {
        self.resultData = resultData
        self.runtimeOutput = runtimeOutput
        self.runtimeFlags = nil
        self.datasetDigest = nil
        self.outputDigest = nil
        self.turns = []
        self.clientExtras = nil
    }
}

/// The crate's `TurnObservation`, kept beside the run record and never sent. The reply's
/// sha256 says which turns matched between two rows without holding either's text.
nonisolated struct TurnObservation: Sendable, Equatable, Codable {
    var sampleId: String
    var turn: UInt32
    var textSha256: String
    var generatedTokens: UInt64
    /// Zero on a row with no drafter.
    var draftedTokens: UInt64
    var acceptedTokens: UInt64
    /// Decode-time target forwards, prefill excluded. Not written when zero.
    var countedForwards: UInt64
    /// Always written: `false` is a reply truncated at `max_tokens`, not an absent reading.
    var endedOnEos: Bool
    var promptMs: Double
    var predictedMs: Double
    var ttftMs: Double
    var decodeMs: Double
    var e2eMs: Double

    enum CodingKeys: String, CodingKey {
        case sampleId = "sample_id"
        case turn
        case textSha256 = "text_sha256"
        case generatedTokens = "generated_tokens"
        case draftedTokens = "drafted_tokens"
        case acceptedTokens = "accepted_tokens"
        case countedForwards = "counted_forwards"
        case verifySteps = "verify_steps"
        case endedOnEos = "ended_on_eos"
        case promptMs = "prompt_ms"
        case predictedMs = "predicted_ms"
        case ttftMs = "ttft_ms"
        case decodeMs = "decode_ms"
        case e2eMs = "e2e_ms"
    }
}

// In an extension, so the memberwise initializer is still synthesized.
nonisolated extension TurnObservation {
    /// `counted_forwards` and `ended_on_eos` default, as in the crate, so older records
    /// still read; `verify_steps` is `counted_forwards`' older name.
    init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        sampleId = try c.decode(String.self, forKey: .sampleId)
        turn = try c.decode(UInt32.self, forKey: .turn)
        textSha256 = try c.decode(String.self, forKey: .textSha256)
        generatedTokens = try c.decode(UInt64.self, forKey: .generatedTokens)
        draftedTokens = try c.decode(UInt64.self, forKey: .draftedTokens)
        acceptedTokens = try c.decode(UInt64.self, forKey: .acceptedTokens)
        countedForwards = try c.decodeIfPresent(UInt64.self, forKey: .countedForwards)
            ?? c.decodeIfPresent(UInt64.self, forKey: .verifySteps) ?? 0
        endedOnEos = try c.decodeIfPresent(Bool.self, forKey: .endedOnEos) ?? false
        promptMs = try c.decode(Double.self, forKey: .promptMs)
        predictedMs = try c.decode(Double.self, forKey: .predictedMs)
        ttftMs = try c.decode(Double.self, forKey: .ttftMs)
        decodeMs = try c.decode(Double.self, forKey: .decodeMs)
        e2eMs = try c.decode(Double.self, forKey: .e2eMs)
    }

    func encode(to encoder: any Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(sampleId, forKey: .sampleId)
        try c.encode(turn, forKey: .turn)
        try c.encode(textSha256, forKey: .textSha256)
        try c.encode(generatedTokens, forKey: .generatedTokens)
        try c.encode(draftedTokens, forKey: .draftedTokens)
        try c.encode(acceptedTokens, forKey: .acceptedTokens)
        if countedForwards > 0 { try c.encode(countedForwards, forKey: .countedForwards) }
        try c.encode(endedOnEos, forKey: .endedOnEos)
        try c.encode(promptMs, forKey: .promptMs)
        try c.encode(predictedMs, forKey: .predictedMs)
        try c.encode(ttftMs, forKey: .ttftMs)
        try c.encode(decodeMs, forKey: .decodeMs)
        try c.encode(e2eMs, forKey: .e2eMs)
    }
}
