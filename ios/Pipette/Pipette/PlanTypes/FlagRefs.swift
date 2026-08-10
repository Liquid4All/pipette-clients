import Foundation

// Swift mirrors of the plan-types flag wire forms, and the resolution their
// `TryFrom` impls perform (`crates/pipette-plan-types/src/runtime_flags.rs`,
// `model.rs`).
//
// Modelled as types rather than as name lists beside a dictionary: the fields a
// group may carry are the struct's properties, and the knobs a cell accepts are
// the ones its resolved variant reads. Neither can drift from the decoder the
// way a parallel `Set<String>` can, and `Decodable` supplies the value typing
// (`Option<u32>` rejects a quoted digit, a float, a negative) that the Rust side
// gets from serde.
//
// This is still a hand-maintained mirror: the app links no Rust, so nothing here
// fails to compile when plan-types gains a knob.
//
// It mirrors a *subset* of the knobs. The axes themselves live beside the types they
// discriminate — `RuntimeType` in `Runtime.swift`, `ModelType` in `Model.swift`, as
// `runtime.rs` and `model.rs` declare them — and are complete, so an unrecognized
// spelling fails to decode here exactly as it does there. The knob set is only what an
// iOS variant can materialize. A server-family knob is not
// declared, so it is refused as an unknown field rather than as a knob the cell
// does not carry: the type states what this client can honour, and there is no
// list of knobs-we-accept-but-ignore to drift out of date.

/// A field the wire form does not declare. `RuntimeFlagRef` and `ModelFlagRef`
/// are both `deny_unknown_fields`, which `Decodable` does not do on its own.
///
/// Carries the spec key the group sat under, since it is thrown from inside a
/// nested decode and the caller reports refusals per group.
nonisolated struct UnknownFlagField: Error {
    let group: String?
    let name: String
}

/// A `runtime-flags=` payload that is not a JSON object of knobs — an array (the wire
/// before the axes were derived) or a scalar.
nonisolated struct RuntimeFlagsNotAnObject: Error {}

/// Reject any key the `CodingKeys` do not name.
private nonisolated func rejectUnknownFields<
    Key: CodingKey & CaseIterable & RawRepresentable
>(
    _ decoder: Decoder,
    keys: Key.Type
) throws where Key.RawValue == String {
    let container = try decoder.container(keyedBy: AnyFlagKey.self)
    let known = Set(Key.allCases.map(\.rawValue))
    for key in container.allKeys where !known.contains(key.stringValue) {
        throw UnknownFlagField(
            group: decoder.codingPath.first?.stringValue,
            name: key.stringValue
        )
    }
}

private nonisolated struct AnyFlagKey: CodingKey {
    let stringValue: String
    let intValue: Int? = nil
    init?(stringValue: String) { self.stringValue = stringValue }
    init?(intValue _: Int) { nil }
}

// MARK: - Runtime flags

/// The `(benchmark, runtime, model)` triple a flags variant is keyed on — the tuple the
/// crate's `RuntimeFlags::axes()` returns. A tuple, not a named type: plan-types has none.
typealias FlagAxes = (benchmark: BenchmarkType, runtime: RuntimeType, model: ModelType)

/// Flat wire form of `RuntimeFlags` — the Swift mirror of `RuntimeFlagRef`.
/// The three axes are required; every knob is optional.
///
/// Read on the way in (a claim's `runtime_flags`) and written on the way out: an engine
/// starts from the ref its request resolves to, overlays the values it supplied, and
/// converts back, so the flags a result reports route and validate like an authored entry.
nonisolated struct RuntimeFlagRef: Codable, Equatable, Sendable {
    let benchmarkType: BenchmarkType
    let runtimeType: RuntimeType
    let modelType: ModelType

    // Only the knobs an iOS variant can materialize — see the file header.
    //
    // `var`, unlike the axes: an engine's overlay assigns the values its execution supplied
    // before converting back, which is how the crate's flag modules are written.
    var numberGpuLayers: UInt32?
    var ctxSize: UInt32?
    var nUbatch: UInt32?
    /// llama's `n_threads`/`n_threads_batch`. In-process on iOS, so unlike `mmap` or
    /// `flash_attention` it is a load-path setting rather than an argv one.
    var threads: UInt32?
    /// llama's sliding-window cache policy. `true` allocates KV for the whole context on
    /// the windowed layers, `false` for the window alone — so on a SWA model this moves
    /// the memory a cell reports without moving `ctxSize`.
    var swaFull: Bool?

    enum CodingKeys: String, CodingKey, CaseIterable {
        case benchmarkType = "benchmark_type"
        case runtimeType = "runtime_type"
        case modelType = "model_type"
        case numberGpuLayers = "number_gpu_layers"
        case ctxSize = "ctx_size"
        case nUbatch = "n_ubatch"
        case threads
        case swaFull = "swa_full"
    }

    init(from decoder: Decoder) throws {
        try rejectUnknownFields(decoder, keys: CodingKeys.self)
        let c = try decoder.container(keyedBy: CodingKeys.self)
        benchmarkType = try c.decode(BenchmarkType.self, forKey: .benchmarkType)
        runtimeType = try c.decode(RuntimeType.self, forKey: .runtimeType)
        modelType = try c.decode(ModelType.self, forKey: .modelType)
        numberGpuLayers = try c.decodeIfPresent(UInt32.self, forKey: .numberGpuLayers)
        ctxSize = try c.decodeIfPresent(UInt32.self, forKey: .ctxSize)
        nUbatch = try c.decodeIfPresent(UInt32.self, forKey: .nUbatch)
        threads = try c.decodeIfPresent(UInt32.self, forKey: .threads)
        swaFull = try c.decodeIfPresent(Bool.self, forKey: .swaFull)
    }

    /// The knobs alone, against the cell the caller already resolved — the crate's
    /// `RuntimeFlags::from_cell_json`.
    ///
    /// This is the form the plan runner sends: a client parses `--benchmark`, `--runtime`
    /// and `--model` before it reads any flags, so restating the cell here would be a
    /// second source of truth that could only agree or contradict. An axis key is refused
    /// rather than checked, for the same reason — it is not the caller's to supply.
    ///
    /// The axes are injected into the payload and the whole thing handed to this type's
    /// own decoder, as the crate injects them into a `serde_json::Value`. That keeps one
    /// field list: a knob added below is understood here with no second place to update,
    /// where a parallel knobs-only struct would silently start refusing it.
    static func knobs(from json: Data, axes: FlagAxes) throws -> RuntimeFlagRef {
        guard var fields = try JSONSerialization.jsonObject(with: json) as? [String: Any] else {
            // An array — the wire before the axes were derived — or a scalar.
            throw RuntimeFlagsNotAnObject()
        }
        for key in [CodingKeys.benchmarkType, .runtimeType, .modelType]
        where fields[key.rawValue] != nil {
            throw UnknownFlagField(group: nil, name: key.rawValue)
        }
        fields[CodingKeys.benchmarkType.rawValue] = axes.benchmark.rawValue
        fields[CodingKeys.runtimeType.rawValue] = axes.runtime.rawValue
        fields[CodingKeys.modelType.rawValue] = axes.model.rawValue
        return try JSONDecoder().decode(
            RuntimeFlagRef.self, from: try JSONSerialization.data(withJSONObject: fields))
    }

    /// A cell's axes with no knob set — the crate's `RuntimeFlagRef::new`, the form a run
    /// carrying no authored flags starts from.
    init(benchmarkType: BenchmarkType, runtimeType: RuntimeType, modelType: ModelType,
         numberGpuLayers: UInt32? = nil, ctxSize: UInt32? = nil, nUbatch: UInt32? = nil,
         threads: UInt32? = nil, swaFull: Bool? = nil) {
        self.benchmarkType = benchmarkType
        self.runtimeType = runtimeType
        self.modelType = modelType
        self.numberGpuLayers = numberGpuLayers
        self.ctxSize = ctxSize
        self.nUbatch = nUbatch
        self.threads = threads
        self.swaFull = swaFull
    }

    /// Flatten resolved flags back onto their cell — the crate's `From<RuntimeFlags>`.
    /// The variant carries its own axes, so none are passed in.
    init(_ flags: RuntimeFlags) {
        let axes = flags.axes
        self.init(benchmarkType: axes.benchmark, runtimeType: axes.runtime,
                  modelType: axes.model, numberGpuLayers: flags.numberGpuLayers,
                  ctxSize: flags.ctxSize, nUbatch: flags.nUbatch, threads: flags.threads,
                  swaFull: flags.swaFull)
    }

    /// The wire form is what a claim sends *and* what a submission value is stripped from,
    /// so it encodes as well as decodes.
    func encode(to encoder: any Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        // The axis enums are `Decodable` only — they read a claim, they do not author one —
        // so the wire spelling is written from `rawValue`.
        try c.encode(benchmarkType.rawValue, forKey: .benchmarkType)
        try c.encode(runtimeType.rawValue, forKey: .runtimeType)
        try c.encode(modelType.rawValue, forKey: .modelType)
        try c.encodeIfPresent(numberGpuLayers, forKey: .numberGpuLayers)
        try c.encodeIfPresent(ctxSize, forKey: .ctxSize)
        try c.encodeIfPresent(nUbatch, forKey: .nUbatch)
        try c.encodeIfPresent(threads, forKey: .threads)
        try c.encodeIfPresent(swaFull, forKey: .swaFull)
    }

    /// The keys `submissionValue` removes — the crate's `AXIS_KEYS`.
    static let axisKeys = [CodingKeys.benchmarkType, .runtimeType, .modelType].map(\.rawValue)
}

/// The resolved `RuntimeFlags` variant for an iOS cell, carrying exactly the
/// knobs plan-types declares on it.
///
/// One case per plan-types variant, named for it: the eleven the crate defines for
/// `llamacpp_ios_pipette` and `mlx_ios_pipette`. Keyed on all three axes, as upstream keys
/// them — which is what lets a variant answer `axes` and therefore what makes an axis
/// check, a flatten, and a submission value complete rather than family-deep.
nonisolated enum RuntimeFlags: Equatable, Sendable {
    // `llamacpp_ios_pipette` × gguf_text: the throughput ladder plus eval.
    case prefillLlamacppIosPipetteGgufText(
        numberGpuLayers: UInt32?, ctxSize: UInt32?, nUbatch: UInt32?, threads: UInt32?,
        swaFull: Bool?)
    case decodeLlamacppIosPipetteGgufText(
        numberGpuLayers: UInt32?, ctxSize: UInt32?, nUbatch: UInt32?, threads: UInt32?,
        swaFull: Bool?)
    case maxMemoryLlamacppIosPipetteGgufText(
        numberGpuLayers: UInt32?, ctxSize: UInt32?, nUbatch: UInt32?, threads: UInt32?,
        swaFull: Bool?)
    case endToEndLlamacppIosPipetteGgufText(
        numberGpuLayers: UInt32?, ctxSize: UInt32?, nUbatch: UInt32?, threads: UInt32?,
        swaFull: Bool?)
    case evalLlamacppIosPipetteGgufText(
        numberGpuLayers: UInt32?, ctxSize: UInt32?, nUbatch: UInt32?, threads: UInt32?,
        swaFull: Bool?)
    /// The one vision variant plan-types declares — `vl_throughput` × gguf_vision.
    case vlLlamacppIosPipetteGgufVision(
        numberGpuLayers: UInt32?, ctxSize: UInt32?, nUbatch: UInt32?, threads: UInt32?,
        swaFull: Bool?)
    // `mlx_ios_pipette` × mlx: prefill chunk only, no other load setting.
    case prefillMlxIosPipetteMlx(nUbatch: UInt32?)
    case decodeMlxIosPipetteMlx(nUbatch: UInt32?)
    case maxMemoryMlxIosPipetteMlx(nUbatch: UInt32?)
    case endToEndMlxIosPipetteMlx(nUbatch: UInt32?)
    case evalMlxIosPipetteMlx(nUbatch: UInt32?)

    /// The `(benchmark, runtime, model)` triple this variant belongs to — the crate's
    /// `RuntimeFlags::axes()`.
    var axes: FlagAxes {
        switch self {
        case .prefillLlamacppIosPipetteGgufText: (.prefillThroughput, .llamacppIosPipette, .ggufText)
        case .decodeLlamacppIosPipetteGgufText: (.decodeThroughput, .llamacppIosPipette, .ggufText)
        case .maxMemoryLlamacppIosPipetteGgufText: (.maxMemoryUsage, .llamacppIosPipette, .ggufText)
        case .endToEndLlamacppIosPipetteGgufText: (.endToEndLatency, .llamacppIosPipette, .ggufText)
        case .evalLlamacppIosPipetteGgufText: (.eval, .llamacppIosPipette, .ggufText)
        case .vlLlamacppIosPipetteGgufVision: (.vlThroughput, .llamacppIosPipette, .ggufVision)
        case .prefillMlxIosPipetteMlx: (.prefillThroughput, .mlxIosPipette, .mlx)
        case .decodeMlxIosPipetteMlx: (.decodeThroughput, .mlxIosPipette, .mlx)
        case .maxMemoryMlxIosPipetteMlx: (.maxMemoryUsage, .mlxIosPipette, .mlx)
        case .endToEndMlxIosPipetteMlx: (.endToEndLatency, .mlxIosPipette, .mlx)
        case .evalMlxIosPipetteMlx: (.eval, .mlxIosPipette, .mlx)
        }
    }

    /// The prefill chunk this cell asks for — llama's `n_ubatch`, MLX's chunk size. `nil`
    /// leaves the engine's own default in place, which is what an absent `Option` means
    /// on the Rust side.
    ///
    /// The knob reads below are what `bench::args_for` gets from `RuntimeFlagRef::from`:
    /// one reader over every variant, so a caller never matches per variant to read a knob.
    var nUbatch: UInt32? {
        switch self {
        case let .prefillLlamacppIosPipetteGgufText(_, _, value, _, _),
             let .decodeLlamacppIosPipetteGgufText(_, _, value, _, _),
             let .maxMemoryLlamacppIosPipetteGgufText(_, _, value, _, _),
             let .endToEndLlamacppIosPipetteGgufText(_, _, value, _, _),
             let .evalLlamacppIosPipetteGgufText(_, _, value, _, _),
             let .vlLlamacppIosPipetteGgufVision(_, _, value, _, _),
             let .prefillMlxIosPipetteMlx(value),
             let .decodeMlxIosPipetteMlx(value),
             let .maxMemoryMlxIosPipetteMlx(value),
             let .endToEndMlxIosPipetteMlx(value),
             let .evalMlxIosPipetteMlx(value):
            value
        }
    }

    /// The context window this cell asks for. MLX sizes its own, so it reports `nil`.
    var ctxSize: UInt32? {
        switch self {
        case let .prefillLlamacppIosPipetteGgufText(_, value, _, _, _),
             let .decodeLlamacppIosPipetteGgufText(_, value, _, _, _),
             let .maxMemoryLlamacppIosPipetteGgufText(_, value, _, _, _),
             let .endToEndLlamacppIosPipetteGgufText(_, value, _, _, _),
             let .evalLlamacppIosPipetteGgufText(_, value, _, _, _),
             let .vlLlamacppIosPipetteGgufVision(_, value, _, _, _):
            value
        case .prefillMlxIosPipetteMlx,
             .decodeMlxIosPipetteMlx,
             .maxMemoryMlxIosPipetteMlx,
             .endToEndMlxIosPipetteMlx,
             .evalMlxIosPipetteMlx:
            nil
        }
    }

    /// How many layers to offload. MLX has no such setting, so it reports `nil` rather
    /// than a number that would look authored.
    var numberGpuLayers: UInt32? {
        switch self {
        case let .prefillLlamacppIosPipetteGgufText(value, _, _, _, _),
             let .decodeLlamacppIosPipetteGgufText(value, _, _, _, _),
             let .maxMemoryLlamacppIosPipetteGgufText(value, _, _, _, _),
             let .endToEndLlamacppIosPipetteGgufText(value, _, _, _, _),
             let .evalLlamacppIosPipetteGgufText(value, _, _, _, _),
             let .vlLlamacppIosPipetteGgufVision(value, _, _, _, _):
            value
        case .prefillMlxIosPipetteMlx,
             .decodeMlxIosPipetteMlx,
             .maxMemoryMlxIosPipetteMlx,
             .endToEndMlxIosPipetteMlx,
             .evalMlxIosPipetteMlx:
            nil
        }
    }

    /// The CPU thread count this cell asks for — llama's `n_threads`. MLX schedules its
    /// own, so it reports `nil` rather than a number that would look authored.
    var threads: UInt32? {
        switch self {
        case let .prefillLlamacppIosPipetteGgufText(_, _, _, value, _),
             let .decodeLlamacppIosPipetteGgufText(_, _, _, value, _),
             let .maxMemoryLlamacppIosPipetteGgufText(_, _, _, value, _),
             let .endToEndLlamacppIosPipetteGgufText(_, _, _, value, _),
             let .evalLlamacppIosPipetteGgufText(_, _, _, value, _),
             let .vlLlamacppIosPipetteGgufVision(_, _, _, value, _):
            value
        case .prefillMlxIosPipetteMlx,
             .decodeMlxIosPipetteMlx,
             .maxMemoryMlxIosPipetteMlx,
             .endToEndMlxIosPipetteMlx,
             .evalMlxIosPipetteMlx:
            nil
        }
    }

    /// The sliding-window cache policy this cell asks for — llama's `swa_full`. MLX has no
    /// such setting, so it reports `nil` rather than a value that would look authored.
    var swaFull: Bool? {
        switch self {
        case let .prefillLlamacppIosPipetteGgufText(_, _, _, _, value),
             let .decodeLlamacppIosPipetteGgufText(_, _, _, _, value),
             let .maxMemoryLlamacppIosPipetteGgufText(_, _, _, _, value),
             let .endToEndLlamacppIosPipetteGgufText(_, _, _, _, value),
             let .evalLlamacppIosPipetteGgufText(_, _, _, _, value),
             let .vlLlamacppIosPipetteGgufVision(_, _, _, _, value):
            value
        case .prefillMlxIosPipetteMlx,
             .decodeMlxIosPipetteMlx,
             .maxMemoryMlxIosPipetteMlx,
             .endToEndMlxIosPipetteMlx,
             .evalMlxIosPipetteMlx:
            nil
        }
    }

    /// The knobs alone as JSON, axes removed — the crate's `submission_value`, which
    /// serializes the flat wire form and then deletes `AXIS_KEYS`. This is what a result
    /// carries as `runtime_flags`.
    ///
    /// Falls closed on an empty object if the encoded form is somehow not one, as upstream
    /// does, rather than dropping the record.
    func submissionValue() throws -> String {
        let encoded = try JSONEncoder().encode(RuntimeFlagRef(self))
        guard var object = try JSONSerialization.jsonObject(with: encoded) as? [String: Any]
        else { return "{}" }
        RuntimeFlagRef.axisKeys.forEach { object.removeValue(forKey: $0) }
        return String(
            decoding: try JSONSerialization.data(withJSONObject: object), as: UTF8.self)
    }
}

/// A knob set that the resolved cell does not carry (`KnobNotAllowed`), or a
/// triple with no variant at all (`NoSuchCombination`).
/// `Equatable` so a test can assert *which* knob was refused, not merely that
/// something was.
///
/// Each case carries the axes it is about and renders them, as the crate's
/// `RuntimeFlagError` does — a refusal reaching a log or a cell error names the cell rather
/// than leaving each call site to rebuild the message.
///
/// The axes are flat associated values, matching `NoSuchCombination { benchmark, runtime,
/// model }`: upstream groups them only as the tuple `RuntimeFlags::axes()` returns, and has
/// no type for them.
nonisolated enum RuntimeFlagResolveError: Error, Equatable, LocalizedError {
    case knobNotAllowed(
        knob: String, benchmarkType: BenchmarkType, runtimeType: RuntimeType,
        modelType: ModelType)
    case noSuchCombination(
        benchmarkType: BenchmarkType, runtimeType: RuntimeType, modelType: ModelType)

    var errorDescription: String? {
        switch self {
        case let .knobNotAllowed(knob, benchmark, runtime, model):
            "knob `\(knob)` is not accepted by \(Self.axes(benchmark, runtime, model))"
        case let .noSuchCombination(benchmark, runtime, model):
            "no runtime flags defined for \(Self.axes(benchmark, runtime, model))"
        }
    }

    private static func axes(_ benchmark: BenchmarkType, _ runtime: RuntimeType,
                             _ model: ModelType) -> String {
        "\(benchmark.rawValue) \u{d7} \(runtime.rawValue) \u{d7} \(model.rawValue)"
    }
}

/// A request whose carried flags belong to a different runtime family than the cell's — a
/// backstop against a request built by hand, as the crate's `runtime_flags_ref` bail is.
/// `ClientRunSpec.validated` refuses an entry naming another cell, and the spec carries
/// the flags from the runtime it binds, so reaching this is a construction bug.
///
/// Separate from `RuntimeFlagResolveError`: routing a ref and building a request fail for
/// different reasons, and folding this in would put an unreachable arm in every switch
/// over that one.
nonisolated struct RuntimeFlagsAxisMismatch: Error, Equatable, LocalizedError {
    let carried: FlagAxes
    let cell: FlagAxes

    /// Hand-written: a struct holding tuples gets no synthesized `==`.
    static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.carried == rhs.carried && lhs.cell == rhs.cell
    }

    var errorDescription: String? {
        "runtime flags for \(Self.render(carried)) are not for this \(Self.render(cell)) cell"
    }

    private static func render(_ axes: FlagAxes) -> String {
        "\(axes.benchmark.rawValue) \u{d7} \(axes.runtime.rawValue) \u{d7} \(axes.model.rawValue)"
    }
}

extension RuntimeFlagRef {
    /// Route the set knobs into the one variant this triple names, rejecting
    /// the rest — the `TryFrom<RuntimeFlagRef>` arms, iOS subset.
    nonisolated func resolve() throws -> RuntimeFlags {
        switch (benchmarkType, runtimeType, modelType) {
        // Every declared knob is permitted on a llama variant, so there is nothing to
        // refuse — the struct's field set is the permitted set.
        case (.prefillThroughput, .llamacppIosPipette, .ggufText):
            return .prefillLlamacppIosPipetteGgufText(
                numberGpuLayers: numberGpuLayers, ctxSize: ctxSize, nUbatch: nUbatch,
                threads: threads, swaFull: swaFull)
        case (.decodeThroughput, .llamacppIosPipette, .ggufText):
            return .decodeLlamacppIosPipetteGgufText(
                numberGpuLayers: numberGpuLayers, ctxSize: ctxSize, nUbatch: nUbatch,
                threads: threads, swaFull: swaFull)
        case (.maxMemoryUsage, .llamacppIosPipette, .ggufText):
            return .maxMemoryLlamacppIosPipetteGgufText(
                numberGpuLayers: numberGpuLayers, ctxSize: ctxSize, nUbatch: nUbatch,
                threads: threads, swaFull: swaFull)
        case (.endToEndLatency, .llamacppIosPipette, .ggufText):
            return .endToEndLlamacppIosPipetteGgufText(
                numberGpuLayers: numberGpuLayers, ctxSize: ctxSize, nUbatch: nUbatch,
                threads: threads, swaFull: swaFull)
        case (.eval, .llamacppIosPipette, .ggufText):
            return .evalLlamacppIosPipetteGgufText(
                numberGpuLayers: numberGpuLayers, ctxSize: ctxSize, nUbatch: nUbatch,
                threads: threads, swaFull: swaFull)
        // gguf_vision has the one variant upstream declares, and it is `vl_throughput`
        // alone: a vision model on a timing benchmark is not a cell plan-types describes.
        case (.vlThroughput, .llamacppIosPipette, .ggufVision):
            return .vlLlamacppIosPipetteGgufVision(
                numberGpuLayers: numberGpuLayers, ctxSize: ctxSize, nUbatch: nUbatch,
                threads: threads, swaFull: swaFull)
        case (.prefillThroughput, .mlxIosPipette, .mlx):
            try denyLlamaKnobs()
            return .prefillMlxIosPipetteMlx(nUbatch: nUbatch)
        case (.decodeThroughput, .mlxIosPipette, .mlx):
            try denyLlamaKnobs()
            return .decodeMlxIosPipetteMlx(nUbatch: nUbatch)
        case (.maxMemoryUsage, .mlxIosPipette, .mlx):
            try denyLlamaKnobs()
            return .maxMemoryMlxIosPipetteMlx(nUbatch: nUbatch)
        case (.endToEndLatency, .mlxIosPipette, .mlx):
            try denyLlamaKnobs()
            return .endToEndMlxIosPipetteMlx(nUbatch: nUbatch)
        case (.eval, .mlxIosPipette, .mlx):
            try denyLlamaKnobs()
            return .evalMlxIosPipetteMlx(nUbatch: nUbatch)
        default:
            // Includes every `apple_foundation` cell: plan-types defines no
            // `RuntimeFlags` variant for it on any benchmark.
            throw RuntimeFlagResolveError.noSuchCombination(
                benchmarkType: benchmarkType, runtimeType: runtimeType, modelType: modelType)
        }
    }

    /// An MLX variant carries the prefill chunk alone, so every other llama setting is
    /// refused by name — the crate's `deny_llama_knobs_except_ubatch`. `threads` included:
    /// MLX schedules its own work and has nothing to apply it to.
    private nonisolated func denyLlamaKnobs() throws {
        if numberGpuLayers != nil { throw refusing(knob: "number_gpu_layers") }
        if ctxSize != nil { throw refusing(knob: "ctx_size") }
        if threads != nil { throw refusing(knob: "threads") }
        if swaFull != nil { throw refusing(knob: "swa_full") }
    }

    /// This cell refusing `knob`, so each throw site stays one line.
    private nonisolated func refusing(knob: String) -> RuntimeFlagResolveError {
        .knobNotAllowed(knob: knob, benchmarkType: benchmarkType, runtimeType: runtimeType,
                        modelType: modelType)
    }
}

// MARK: - Benchmark flags

/// Host-readiness gate settings for a cell — the crate's `ReadinessOverrides`, authored as
/// a nested table (`readiness = { max_wait_secs = 1800 }`).
///
/// Optional fields, unlike the resolved `ReadinessPolicy` the gate runs with: a block whose
/// fields are unset describes a request, not what happened, so the two are different types.
nonisolated struct ReadinessOverrides: Codable, Equatable, Sendable {
    /// Override the gate's deadline, in whole seconds. Unset keeps the built-in.
    var maxWaitSecs: UInt64?
    /// Waive the *temperature* criterion, keeping the thermal-state one. This changes what
    /// "ready" means, so a cell run with it is not comparable to a gated one — which is why
    /// a plan sets it per cell rather than a device setting waiving it fleet-wide.
    var skipThermal: Bool?

    enum CodingKeys: String, CodingKey, CaseIterable {
        case maxWaitSecs = "max_wait_secs"
        case skipThermal = "skip_thermal"
    }

    init(maxWaitSecs: UInt64? = nil, skipThermal: Bool? = nil) {
        self.maxWaitSecs = maxWaitSecs
        self.skipThermal = skipThermal
    }

    init(from decoder: Decoder) throws {
        try rejectUnknownFields(decoder, keys: CodingKeys.self)
        let c = try decoder.container(keyedBy: CodingKeys.self)
        maxWaitSecs = try c.decodeIfPresent(UInt64.self, forKey: .maxWaitSecs)
        skipThermal = try c.decodeIfPresent(Bool.self, forKey: .skipThermal)
    }

    /// This block over `base` — the gate's own defaults, or a job-level override. Unset
    /// fields keep what `base` had, so a plan that names only a deadline still gets the
    /// device's thermal criterion.
    func resolved(over base: ReadinessPolicy) -> ReadinessPolicy {
        ReadinessPolicy(
            maxSeconds: maxWaitSecs.map(Double.init) ?? base.maxSeconds,
            skipThermal: skipThermal ?? base.skipThermal)
    }
}

/// Flat wire form of `BenchmarkFlags` — the Swift mirror of the crate's `BenchmarkFlagRef`.
///
/// `http_timeout_seconds` and `doomloop` are not declared: they belong to cells that talk to
/// a server, and an in-process engine has neither to drive. Undeclared rather than accepted
/// and ignored, so a body authored for another client is refused by name.
nonisolated struct BenchmarkFlagRef: Codable, Equatable, Sendable {
    let benchmarkType: BenchmarkType
    let runtimeType: RuntimeType
    let modelType: ModelType
    var readiness: ReadinessOverrides?

    enum CodingKeys: String, CodingKey, CaseIterable {
        case benchmarkType = "benchmark_type"
        case runtimeType = "runtime_type"
        case modelType = "model_type"
        case readiness
    }

    init(from decoder: Decoder) throws {
        try rejectUnknownFields(decoder, keys: CodingKeys.self)
        let c = try decoder.container(keyedBy: CodingKeys.self)
        benchmarkType = try c.decode(BenchmarkType.self, forKey: .benchmarkType)
        runtimeType = try c.decode(RuntimeType.self, forKey: .runtimeType)
        modelType = try c.decode(ModelType.self, forKey: .modelType)
        readiness = try c.decodeIfPresent(ReadinessOverrides.self, forKey: .readiness)
    }

    init(benchmarkType: BenchmarkType, runtimeType: RuntimeType, modelType: ModelType,
         readiness: ReadinessOverrides? = nil) {
        self.benchmarkType = benchmarkType
        self.runtimeType = runtimeType
        self.modelType = modelType
        self.readiness = readiness
    }

    init(_ flags: BenchmarkFlags) {
        let axes = flags.axes
        self.init(benchmarkType: axes.benchmark, runtimeType: axes.runtime,
                  modelType: axes.model, readiness: flags.readiness)
    }

    func encode(to encoder: any Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(benchmarkType.rawValue, forKey: .benchmarkType)
        try c.encode(runtimeType.rawValue, forKey: .runtimeType)
        try c.encode(modelType.rawValue, forKey: .modelType)
        try c.encodeIfPresent(readiness, forKey: .readiness)
    }

    /// The variant this triple names — the crate's `TryFrom<BenchmarkFlagRef>`. Only the
    /// timing cells this build runs have one: `eval` carries server settings an in-process
    /// engine cannot apply, and `max_memory_usage` does not gate at all.
    ///
    /// `vl_throughput` gates upstream and `BenchmarkType.gatesOnReadiness` says so, but no
    /// iOS engine runs it — so it has no variant here, and the two facts differ for that
    /// reason rather than by oversight.
    nonisolated func resolve() throws -> BenchmarkFlags {
        switch (benchmarkType, runtimeType, modelType) {
        case (.prefillThroughput, .llamacppIosPipette, .ggufText):
            return .prefillLlamacppIosPipetteGgufText(readiness: readiness)
        case (.decodeThroughput, .llamacppIosPipette, .ggufText):
            return .decodeLlamacppIosPipetteGgufText(readiness: readiness)
        case (.endToEndLatency, .llamacppIosPipette, .ggufText):
            return .endToEndLlamacppIosPipetteGgufText(readiness: readiness)
        case (.prefillThroughput, .mlxIosPipette, .mlx):
            return .prefillMlxIosPipetteMlx(readiness: readiness)
        case (.decodeThroughput, .mlxIosPipette, .mlx):
            return .decodeMlxIosPipetteMlx(readiness: readiness)
        case (.endToEndLatency, .mlxIosPipette, .mlx):
            return .endToEndMlxIosPipetteMlx(readiness: readiness)
        default:
            throw RuntimeFlagResolveError.noSuchCombination(
                benchmarkType: benchmarkType, runtimeType: runtimeType, modelType: modelType)
        }
    }
}

/// The resolved `BenchmarkFlags` variant for an iOS cell — the timing cells alone, each
/// carrying the readiness gate and nothing else.
nonisolated enum BenchmarkFlags: Equatable, Sendable {
    case prefillLlamacppIosPipetteGgufText(readiness: ReadinessOverrides?)
    case decodeLlamacppIosPipetteGgufText(readiness: ReadinessOverrides?)
    case endToEndLlamacppIosPipetteGgufText(readiness: ReadinessOverrides?)
    case prefillMlxIosPipetteMlx(readiness: ReadinessOverrides?)
    case decodeMlxIosPipetteMlx(readiness: ReadinessOverrides?)
    case endToEndMlxIosPipetteMlx(readiness: ReadinessOverrides?)

    var axes: FlagAxes {
        switch self {
        case .prefillLlamacppIosPipetteGgufText: (.prefillThroughput, .llamacppIosPipette, .ggufText)
        case .decodeLlamacppIosPipetteGgufText: (.decodeThroughput, .llamacppIosPipette, .ggufText)
        case .endToEndLlamacppIosPipetteGgufText: (.endToEndLatency, .llamacppIosPipette, .ggufText)
        case .prefillMlxIosPipetteMlx: (.prefillThroughput, .mlxIosPipette, .mlx)
        case .decodeMlxIosPipetteMlx: (.decodeThroughput, .mlxIosPipette, .mlx)
        case .endToEndMlxIosPipetteMlx: (.endToEndLatency, .mlxIosPipette, .mlx)
        }
    }

    var readiness: ReadinessOverrides? {
        switch self {
        case let .prefillLlamacppIosPipetteGgufText(value),
             let .decodeLlamacppIosPipetteGgufText(value),
             let .endToEndLlamacppIosPipetteGgufText(value),
             let .prefillMlxIosPipetteMlx(value),
             let .decodeMlxIosPipetteMlx(value),
             let .endToEndMlxIosPipetteMlx(value):
            value
        }
    }

    /// The knobs alone as JSON, axes removed — the crate's `submission_value`. This is what
    /// a result carries as `benchmark_flags`, and it is how a gated run is told apart from
    /// one that waived the temperature criterion.
    func submissionValue() throws -> String {
        let encoded = try JSONEncoder().encode(BenchmarkFlagRef(self))
        guard var object = try JSONSerialization.jsonObject(with: encoded) as? [String: Any]
        else { return "{}" }
        RuntimeFlagRef.axisKeys.forEach { object.removeValue(forKey: $0) }
        return String(
            decoding: try JSONSerialization.data(withJSONObject: object), as: UTF8.self)
    }
}

// MARK: - Model flags

/// Flat wire form of `ModelFlags`, mirroring the crate's `ModelFlagRef`. No runtime
/// axis: the pair that selects a variant is `(benchmark, model)`.
nonisolated struct ModelFlagRef: Codable, Equatable, Sendable {
    let benchmarkType: BenchmarkType
    let modelType: ModelType
    let enableThinking: Bool?

    enum CodingKeys: String, CodingKey, CaseIterable {
        case benchmarkType = "benchmark_type"
        case modelType = "model_type"
        case enableThinking = "enable_thinking"
    }

    init(from decoder: Decoder) throws {
        try rejectUnknownFields(decoder, keys: CodingKeys.self)
        let c = try decoder.container(keyedBy: CodingKeys.self)
        benchmarkType = try c.decode(BenchmarkType.self, forKey: .benchmarkType)
        modelType = try c.decode(ModelType.self, forKey: .modelType)
        enableThinking = try c.decodeIfPresent(Bool.self, forKey: .enableThinking)
    }

    /// Written as it arrived. The axis enums read a claim rather than authoring one, so
    /// their wire spelling comes from `rawValue` — as `RuntimeFlagRef` writes its own.
    func encode(to encoder: any Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(benchmarkType.rawValue, forKey: .benchmarkType)
        try c.encode(modelType.rawValue, forKey: .modelType)
        try c.encodeIfPresent(enableThinking, forKey: .enableThinking)
    }

    /// The typed flags this cell carries — the crate's `TryFrom<ModelFlagRef>`, whose
    /// arms exist for `eval` alone: a timing cell has no generation to shape, and no
    /// variant covers `apple_foundation_text`.
    func resolve() throws -> ModelFlags {
        switch (benchmarkType, modelType) {
        case (.eval, .ggufText): return .evalGgufText(enableThinking: enableThinking)
        case (.eval, .ggufVision): return .evalGgufVision(enableThinking: enableThinking)
        case (.eval, .mlx): return .evalMlx(enableThinking: enableThinking)
        case (.eval, .torch): return .evalTorch(enableThinking: enableThinking)
        default:
            throw RuntimeFlagResolveError.noSuchCombination(
                benchmarkType: benchmarkType, runtimeType: .llamacppIosPipette,
                modelType: modelType)
        }
    }
}
