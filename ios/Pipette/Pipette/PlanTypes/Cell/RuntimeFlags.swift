import Foundation

// The runtime-flag wire forms and their resolution: pipette-mgmt's
// `crates/pipette-plan-types/src/cell/runtime_flags.rs`. Shared decode machinery is in
// `FlagDecoding.swift`.
//
// A hand-maintained mirror (the app links no Rust), so nothing here fails to compile when
// plan-types gains a knob. It declares only the knobs an iOS variant can materialize; a
// server-family knob is refused as an unknown field. The axes (`RuntimeType`, `ModelType`)
// are complete, so an unrecognized spelling fails to decode as it does in the crate.

/// The tuple the crate's `RuntimeFlags::axes()` returns.
typealias FlagAxes = (benchmark: BenchmarkType, runtime: RuntimeType, model: ModelType)

/// The crate's `RuntimeFlagRef`, the flat wire form of `RuntimeFlags`.
///
/// Read from a claim and written into a result: an engine overlays the values it supplied
/// and converts back, so the flags a result reports route and validate like an authored
/// entry.
nonisolated struct RuntimeFlagRef: Codable, Equatable, Sendable {
    let benchmarkType: BenchmarkType
    let runtimeType: RuntimeType
    let modelType: ModelType

    // `var`, unlike the axes, for the engine's overlay.
    var numberGpuLayers: UInt32?
    var ctxSize: UInt32?
    var nUbatch: UInt32?
    /// llama's `n_threads`/`n_threads_batch`.
    var threads: UInt32?
    /// llama's sliding-window cache policy. `true` allocates KV for the whole context on
    /// the windowed layers, `false` for the window alone, so on a SWA model this moves the
    /// memory a cell reports without moving `ctxSize`.
    var swaFull: Bool?
    /// llama's `load_mode`. The `mmap` boolean it replaces upstream is refused.
    var loadMode: LlamacppLoadMode?
    /// Whether the model's drafter runs, on `vl_spec_decode` alone. Unset follows the
    /// model: on for a spec-decoding model, off for a regular one. `true` on a model with
    /// no drafter is refused.
    var speculativeEnabled: Bool?

    enum CodingKeys: String, CodingKey, CaseIterable {
        case benchmarkType = "benchmark_type"
        case runtimeType = "runtime_type"
        case modelType = "model_type"
        case numberGpuLayers = "number_gpu_layers"
        case ctxSize = "ctx_size"
        case nUbatch = "n_ubatch"
        case threads
        case swaFull = "swa_full"
        case loadMode = "load_mode"
        case speculativeEnabled = "speculative_enabled"
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
        loadMode = try c.decodeIfPresent(LlamacppLoadMode.self, forKey: .loadMode)
        speculativeEnabled = try c.decodeIfPresent(Bool.self, forKey: .speculativeEnabled)
    }

    /// The crate's `RuntimeFlagRef::new`.
    init(benchmarkType: BenchmarkType, runtimeType: RuntimeType, modelType: ModelType,
         numberGpuLayers: UInt32? = nil, ctxSize: UInt32? = nil, nUbatch: UInt32? = nil,
         threads: UInt32? = nil, swaFull: Bool? = nil,
         loadMode: LlamacppLoadMode? = nil, speculativeEnabled: Bool? = nil) {
        self.benchmarkType = benchmarkType
        self.runtimeType = runtimeType
        self.modelType = modelType
        self.numberGpuLayers = numberGpuLayers
        self.ctxSize = ctxSize
        self.nUbatch = nUbatch
        self.threads = threads
        self.swaFull = swaFull
        self.loadMode = loadMode
        self.speculativeEnabled = speculativeEnabled
    }

    /// The crate's `From<RuntimeFlags>`.
    init(_ flags: RuntimeFlags) {
        let axes = flags.axes
        self.init(benchmarkType: axes.benchmark, runtimeType: axes.runtime,
                  modelType: axes.model, numberGpuLayers: flags.numberGpuLayers,
                  ctxSize: flags.ctxSize, nUbatch: flags.nUbatch, threads: flags.threads,
                  swaFull: flags.swaFull, loadMode: flags.loadMode,
                  speculativeEnabled: flags.speculativeEnabled)
    }

    func encode(to encoder: any Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(benchmarkType.rawValue, forKey: .benchmarkType)
        try c.encode(runtimeType.rawValue, forKey: .runtimeType)
        try c.encode(modelType.rawValue, forKey: .modelType)
        try c.encodeIfPresent(numberGpuLayers, forKey: .numberGpuLayers)
        try c.encodeIfPresent(ctxSize, forKey: .ctxSize)
        try c.encodeIfPresent(nUbatch, forKey: .nUbatch)
        try c.encodeIfPresent(threads, forKey: .threads)
        try c.encodeIfPresent(swaFull, forKey: .swaFull)
        try c.encodeIfPresent(loadMode, forKey: .loadMode)
        try c.encodeIfPresent(speculativeEnabled, forKey: .speculativeEnabled)
    }

    /// The keys `submissionValue` removes — the crate's `AXIS_KEYS`.
    static let axisKeys = [CodingKeys.runtimeType, .modelType, .benchmarkType].map(\.rawValue)
}

/// The crate's `RuntimeFlags` variants for `llamacpp_ios_pipette` and `mlx_ios_pipette`,
/// each carrying exactly the knobs the crate declares on it.
nonisolated enum RuntimeFlags: Equatable, Sendable {
    case prefillThroughputLlamacppIosPipetteGgufText(
        numberGpuLayers: UInt32?, ctxSize: UInt32?, nUbatch: UInt32?, threads: UInt32?,
        swaFull: Bool?, loadMode: LlamacppLoadMode?)
    case decodeThroughputLlamacppIosPipetteGgufText(
        numberGpuLayers: UInt32?, ctxSize: UInt32?, nUbatch: UInt32?, threads: UInt32?,
        swaFull: Bool?, loadMode: LlamacppLoadMode?)
    case maxMemoryUsageLlamacppIosPipetteGgufText(
        numberGpuLayers: UInt32?, ctxSize: UInt32?, nUbatch: UInt32?, threads: UInt32?,
        swaFull: Bool?, loadMode: LlamacppLoadMode?)
    case endToEndLatencyLlamacppIosPipetteGgufText(
        numberGpuLayers: UInt32?, ctxSize: UInt32?, nUbatch: UInt32?, threads: UInt32?,
        swaFull: Bool?, loadMode: LlamacppLoadMode?)
    case evalLlamacppIosPipetteGgufText(
        numberGpuLayers: UInt32?, ctxSize: UInt32?, nUbatch: UInt32?, threads: UInt32?,
        swaFull: Bool?, loadMode: LlamacppLoadMode?)
    case vlThroughputLlamacppIosPipetteGgufVision(
        numberGpuLayers: UInt32?, ctxSize: UInt32?, nUbatch: UInt32?, threads: UInt32?,
        swaFull: Bool?, loadMode: LlamacppLoadMode?)
    case vlMaxMemoryLlamacppIosPipetteGgufVision(
        numberGpuLayers: UInt32?, ctxSize: UInt32?, nUbatch: UInt32?, threads: UInt32?,
        swaFull: Bool?, loadMode: LlamacppLoadMode?)
    /// `speculativeEnabled` may only be `false` here: the model has no drafter.
    case vlSpecDecodeLlamacppIosPipetteGgufVision(
        numberGpuLayers: UInt32?, ctxSize: UInt32?, nUbatch: UInt32?, threads: UInt32?,
        swaFull: Bool?, loadMode: LlamacppLoadMode?, speculativeEnabled: Bool?)
    /// `speculativeEnabled` unset means on. The drafter takes no load knobs of its own: it
    /// gets the target's `numberGpuLayers` and `threads`, and the live context's window and
    /// batch sizes, since a cell that offloaded the target and not the drafter would be
    /// measuring the transfer.
    case vlSpecDecodeLlamacppIosPipetteGgufVisionSpeculative(
        numberGpuLayers: UInt32?, ctxSize: UInt32?, nUbatch: UInt32?, threads: UInt32?,
        swaFull: Bool?, loadMode: LlamacppLoadMode?, speculativeEnabled: Bool?)
    case asrThroughputLlamacppIosPipetteGgufVision(
        numberGpuLayers: UInt32?, ctxSize: UInt32?, nUbatch: UInt32?, threads: UInt32?,
        swaFull: Bool?, loadMode: LlamacppLoadMode?)
    case asrMaxMemoryLlamacppIosPipetteGgufVision(
        numberGpuLayers: UInt32?, ctxSize: UInt32?, nUbatch: UInt32?, threads: UInt32?,
        swaFull: Bool?, loadMode: LlamacppLoadMode?)
    case prefillThroughputMlxIosPipetteMlx(nUbatch: UInt32?)
    case decodeThroughputMlxIosPipetteMlx(nUbatch: UInt32?)
    case maxMemoryUsageMlxIosPipetteMlx(nUbatch: UInt32?)
    case endToEndLatencyMlxIosPipetteMlx(nUbatch: UInt32?)
    case evalMlxIosPipetteMlx(nUbatch: UInt32?)

    /// The crate's `RuntimeFlags::axes()`.
    var axes: FlagAxes {
        switch self {
        case .prefillThroughputLlamacppIosPipetteGgufText: (.prefillThroughput, .llamacppIosPipette, .ggufText)
        case .decodeThroughputLlamacppIosPipetteGgufText: (.decodeThroughput, .llamacppIosPipette, .ggufText)
        case .maxMemoryUsageLlamacppIosPipetteGgufText: (.maxMemoryUsage, .llamacppIosPipette, .ggufText)
        case .endToEndLatencyLlamacppIosPipetteGgufText: (.endToEndLatency, .llamacppIosPipette, .ggufText)
        case .evalLlamacppIosPipetteGgufText: (.eval, .llamacppIosPipette, .ggufText)
        case .vlThroughputLlamacppIosPipetteGgufVision: (.vlThroughput, .llamacppIosPipette, .ggufVision)
        case .vlMaxMemoryLlamacppIosPipetteGgufVision:
            (.vlMaxMemory, .llamacppIosPipette, .ggufVision)
        case .vlSpecDecodeLlamacppIosPipetteGgufVision:
            (.vlSpecDecode, .llamacppIosPipette, .ggufVision)
        case .vlSpecDecodeLlamacppIosPipetteGgufVisionSpeculative:
            (.vlSpecDecode, .llamacppIosPipette, .ggufVisionSpeculative)
        case .asrThroughputLlamacppIosPipetteGgufVision:
            (.asrThroughput, .llamacppIosPipette, .ggufVision)
        case .asrMaxMemoryLlamacppIosPipetteGgufVision:
            (.asrMaxMemory, .llamacppIosPipette, .ggufVision)
        case .prefillThroughputMlxIosPipetteMlx: (.prefillThroughput, .mlxIosPipette, .mlx)
        case .decodeThroughputMlxIosPipetteMlx: (.decodeThroughput, .mlxIosPipette, .mlx)
        case .maxMemoryUsageMlxIosPipetteMlx: (.maxMemoryUsage, .mlxIosPipette, .mlx)
        case .endToEndLatencyMlxIosPipetteMlx: (.endToEndLatency, .mlxIosPipette, .mlx)
        case .evalMlxIosPipetteMlx: (.eval, .mlxIosPipette, .mlx)
        }
    }

    /// The crate's `matches`.
    func matches(benchmark: BenchmarkType, runtime: Runtime, model: Model) -> Bool {
        axes == (benchmark, RuntimeType.of(runtime), ModelType.of(model))
    }

    /// The prefill chunk: llama's `n_ubatch`, MLX's chunk size. `nil` leaves the engine's
    /// default. On the knob readers below, a runtime without the setting reports `nil`
    /// rather than a value that would look authored.
    var nUbatch: UInt32? {
        switch self {
        case let .prefillThroughputLlamacppIosPipetteGgufText(_, _, value, _, _, _),
             let .decodeThroughputLlamacppIosPipetteGgufText(_, _, value, _, _, _),
             let .maxMemoryUsageLlamacppIosPipetteGgufText(_, _, value, _, _, _),
             let .endToEndLatencyLlamacppIosPipetteGgufText(_, _, value, _, _, _),
             let .evalLlamacppIosPipetteGgufText(_, _, value, _, _, _),
             let .vlThroughputLlamacppIosPipetteGgufVision(_, _, value, _, _, _),
             let .vlMaxMemoryLlamacppIosPipetteGgufVision(_, _, value, _, _, _),
             let .vlSpecDecodeLlamacppIosPipetteGgufVision(_, _, value, _, _, _, _),
             let .vlSpecDecodeLlamacppIosPipetteGgufVisionSpeculative(_, _, value, _, _, _, _),
             let .asrThroughputLlamacppIosPipetteGgufVision(_, _, value, _, _, _),
             let .asrMaxMemoryLlamacppIosPipetteGgufVision(_, _, value, _, _, _),
             let .prefillThroughputMlxIosPipetteMlx(value),
             let .decodeThroughputMlxIosPipetteMlx(value),
             let .maxMemoryUsageMlxIosPipetteMlx(value),
             let .endToEndLatencyMlxIosPipetteMlx(value),
             let .evalMlxIosPipetteMlx(value):
            value
        }
    }

    var ctxSize: UInt32? {
        switch self {
        case let .prefillThroughputLlamacppIosPipetteGgufText(_, value, _, _, _, _),
             let .decodeThroughputLlamacppIosPipetteGgufText(_, value, _, _, _, _),
             let .maxMemoryUsageLlamacppIosPipetteGgufText(_, value, _, _, _, _),
             let .endToEndLatencyLlamacppIosPipetteGgufText(_, value, _, _, _, _),
             let .evalLlamacppIosPipetteGgufText(_, value, _, _, _, _),
             let .vlThroughputLlamacppIosPipetteGgufVision(_, value, _, _, _, _),
             let .vlMaxMemoryLlamacppIosPipetteGgufVision(_, value, _, _, _, _),
             let .vlSpecDecodeLlamacppIosPipetteGgufVision(_, value, _, _, _, _, _),
             let .vlSpecDecodeLlamacppIosPipetteGgufVisionSpeculative(_, value, _, _, _, _, _),
             let .asrThroughputLlamacppIosPipetteGgufVision(_, value, _, _, _, _),
             let .asrMaxMemoryLlamacppIosPipetteGgufVision(_, value, _, _, _, _):
            value
        case .prefillThroughputMlxIosPipetteMlx,
             .decodeThroughputMlxIosPipetteMlx,
             .maxMemoryUsageMlxIosPipetteMlx,
             .endToEndLatencyMlxIosPipetteMlx,
             .evalMlxIosPipetteMlx:
            nil
        }
    }

    var numberGpuLayers: UInt32? {
        switch self {
        case let .prefillThroughputLlamacppIosPipetteGgufText(value, _, _, _, _, _),
             let .decodeThroughputLlamacppIosPipetteGgufText(value, _, _, _, _, _),
             let .maxMemoryUsageLlamacppIosPipetteGgufText(value, _, _, _, _, _),
             let .endToEndLatencyLlamacppIosPipetteGgufText(value, _, _, _, _, _),
             let .evalLlamacppIosPipetteGgufText(value, _, _, _, _, _),
             let .vlThroughputLlamacppIosPipetteGgufVision(value, _, _, _, _, _),
             let .vlMaxMemoryLlamacppIosPipetteGgufVision(value, _, _, _, _, _),
             let .vlSpecDecodeLlamacppIosPipetteGgufVision(value, _, _, _, _, _, _),
             let .vlSpecDecodeLlamacppIosPipetteGgufVisionSpeculative(value, _, _, _, _, _, _),
             let .asrThroughputLlamacppIosPipetteGgufVision(value, _, _, _, _, _),
             let .asrMaxMemoryLlamacppIosPipetteGgufVision(value, _, _, _, _, _):
            value
        case .prefillThroughputMlxIosPipetteMlx,
             .decodeThroughputMlxIosPipetteMlx,
             .maxMemoryUsageMlxIosPipetteMlx,
             .endToEndLatencyMlxIosPipetteMlx,
             .evalMlxIosPipetteMlx:
            nil
        }
    }

    var threads: UInt32? {
        switch self {
        case let .prefillThroughputLlamacppIosPipetteGgufText(_, _, _, value, _, _),
             let .decodeThroughputLlamacppIosPipetteGgufText(_, _, _, value, _, _),
             let .maxMemoryUsageLlamacppIosPipetteGgufText(_, _, _, value, _, _),
             let .endToEndLatencyLlamacppIosPipetteGgufText(_, _, _, value, _, _),
             let .evalLlamacppIosPipetteGgufText(_, _, _, value, _, _),
             let .vlThroughputLlamacppIosPipetteGgufVision(_, _, _, value, _, _),
             let .vlMaxMemoryLlamacppIosPipetteGgufVision(_, _, _, value, _, _),
             let .vlSpecDecodeLlamacppIosPipetteGgufVision(_, _, _, value, _, _, _),
             let .vlSpecDecodeLlamacppIosPipetteGgufVisionSpeculative(_, _, _, value, _, _, _),
             let .asrThroughputLlamacppIosPipetteGgufVision(_, _, _, value, _, _),
             let .asrMaxMemoryLlamacppIosPipetteGgufVision(_, _, _, value, _, _):
            value
        case .prefillThroughputMlxIosPipetteMlx,
             .decodeThroughputMlxIosPipetteMlx,
             .maxMemoryUsageMlxIosPipetteMlx,
             .endToEndLatencyMlxIosPipetteMlx,
             .evalMlxIosPipetteMlx:
            nil
        }
    }

    var swaFull: Bool? {
        switch self {
        case let .prefillThroughputLlamacppIosPipetteGgufText(_, _, _, _, value, _),
             let .decodeThroughputLlamacppIosPipetteGgufText(_, _, _, _, value, _),
             let .maxMemoryUsageLlamacppIosPipetteGgufText(_, _, _, _, value, _),
             let .endToEndLatencyLlamacppIosPipetteGgufText(_, _, _, _, value, _),
             let .evalLlamacppIosPipetteGgufText(_, _, _, _, value, _),
             let .vlThroughputLlamacppIosPipetteGgufVision(_, _, _, _, value, _),
             let .vlMaxMemoryLlamacppIosPipetteGgufVision(_, _, _, _, value, _),
             let .vlSpecDecodeLlamacppIosPipetteGgufVision(_, _, _, _, value, _, _),
             let .vlSpecDecodeLlamacppIosPipetteGgufVisionSpeculative(_, _, _, _, value, _, _),
             let .asrThroughputLlamacppIosPipetteGgufVision(_, _, _, _, value, _),
             let .asrMaxMemoryLlamacppIosPipetteGgufVision(_, _, _, _, value, _):
            value
        case .prefillThroughputMlxIosPipetteMlx,
             .decodeThroughputMlxIosPipetteMlx,
             .maxMemoryUsageMlxIosPipetteMlx,
             .endToEndLatencyMlxIosPipetteMlx,
             .evalMlxIosPipetteMlx:
            nil
        }
    }

    var loadMode: LlamacppLoadMode? {
        switch self {
        case let .prefillThroughputLlamacppIosPipetteGgufText(_, _, _, _, _, value),
             let .decodeThroughputLlamacppIosPipetteGgufText(_, _, _, _, _, value),
             let .maxMemoryUsageLlamacppIosPipetteGgufText(_, _, _, _, _, value),
             let .endToEndLatencyLlamacppIosPipetteGgufText(_, _, _, _, _, value),
             let .evalLlamacppIosPipetteGgufText(_, _, _, _, _, value),
             let .vlThroughputLlamacppIosPipetteGgufVision(_, _, _, _, _, value),
             let .vlMaxMemoryLlamacppIosPipetteGgufVision(_, _, _, _, _, value),
             let .vlSpecDecodeLlamacppIosPipetteGgufVision(_, _, _, _, _, value, _),
             let .vlSpecDecodeLlamacppIosPipetteGgufVisionSpeculative(_, _, _, _, _, value, _),
             let .asrThroughputLlamacppIosPipetteGgufVision(_, _, _, _, _, value),
             let .asrMaxMemoryLlamacppIosPipetteGgufVision(_, _, _, _, _, value):
            value
        case .prefillThroughputMlxIosPipetteMlx,
             .decodeThroughputMlxIosPipetteMlx,
             .maxMemoryUsageMlxIosPipetteMlx,
             .endToEndLatencyMlxIosPipetteMlx,
             .evalMlxIosPipetteMlx:
            nil
        }
    }

    /// The knobs alone as JSON, axes removed: the crate's `submission_value`, what a result
    /// carries as `runtime_flags`.
    ///
    /// `.sortedKeys` makes this the same string the CLI submits: serde_json's `Map` is a
    /// `BTreeMap` (no `preserve_order`), while Foundation orders by a per-process hash seed.
    func submissionValue() throws -> String {
        let encoded = try JSONEncoder().encode(RuntimeFlagRef(self))
        guard var object = try JSONSerialization.jsonObject(with: encoded) as? [String: Any]
        else { return "{}" }
        RuntimeFlagRef.axisKeys.forEach { object.removeValue(forKey: $0) }
        return String(
            decoding: try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys]),
            as: UTF8.self)
    }

    /// Parse a `runtime-flags=` JSON object of knobs against the cell the caller already
    /// resolved: the crate's `RuntimeFlags::from_cell_json`.
    ///
    /// An axis key is refused rather than checked: the cell is already known, so a second
    /// source could only agree or be wrong. An empty object is `nil` (the engine's
    /// defaults). Decode errors propagate unwrapped, where the crate wraps them in
    /// `RuntimeFlagsError::Json`.
    static func fromCellJson(
        _ json: Data, runtimeType: RuntimeType, modelType: ModelType,
        benchmarkType: BenchmarkType
    ) throws -> RuntimeFlags? {
        guard var fields = try JSONSerialization.jsonObject(
            with: json, options: [.fragmentsAllowed]) as? [String: Any]
        else { throw RuntimeFlagsError.notAnObject }
        if fields.isEmpty { return nil }
        let derived: [(RuntimeFlagRef.CodingKeys, String)] = [
            (.runtimeType, runtimeType.rawValue), (.modelType, modelType.rawValue),
            (.benchmarkType, benchmarkType.rawValue),
        ]
        for (key, value) in derived {
            guard fields[key.rawValue] == nil else {
                throw RuntimeFlagsError.axisNotAccepted(key: key.rawValue)
            }
            fields[key.rawValue] = value
        }
        let ref = try JSONDecoder().decode(
            RuntimeFlagRef.self, from: try JSONSerialization.data(withJSONObject: fields))
        return try ref.resolve()
    }
}

nonisolated extension RuntimeFlags {
    /// The crate's `RuntimeFlags::speculative_enabled`.
    var speculativeEnabled: Bool? {
        switch self {
        case let .vlSpecDecodeLlamacppIosPipetteGgufVision(_, _, _, _, _, _, value),
             let .vlSpecDecodeLlamacppIosPipetteGgufVisionSpeculative(_, _, _, _, _, _, value):
            value
        default:
            nil
        }
    }
}

/// Whether a corpus run's drafter is on: the crate's `resolve_speculation`, one rule for
/// every engine so a row's mode cannot depend on which client measured it.
nonisolated func resolveSpeculation(_ model: ModelType, _ enabled: Bool?)
    throws(RuntimeFlagsError) -> Bool {
    let hasDrafter = switch model {
    case .ggufVisionSpeculative, .mlxSpeculative, .torchSpeculative: true
    case .ggufText, .ggufVision, .mlx, .coreAi, .torch, .openvino, .onnxGenai,
         .appleFoundationText: false
    }
    switch enabled {
    case nil: return hasDrafter
    case true? where !hasDrafter: throw .speculationWithoutDrafter(model: model)
    case let value?: return value
    }
}

/// The crate's `RuntimeFlagsError`, iOS subset: no iOS variant declares `raw` or
/// `parallel`, and its `Json` arm is left to the decoder. Each cell case carries its axes
/// so the message names the cell.
nonisolated enum RuntimeFlagsError: Error, Equatable, LocalizedError {
    case noSuchCombination(benchmark: BenchmarkType, runtime: RuntimeType, model: ModelType)
    case knobNotAllowed(
        knob: String, benchmark: BenchmarkType, runtime: RuntimeType, model: ModelType)
    case speculationWithoutDrafter(model: ModelType)
    /// An array (the wire before the axes were derived) or a scalar.
    case notAnObject
    case axisNotAccepted(key: String)

    var errorDescription: String? {
        switch self {
        case let .speculationWithoutDrafter(model):
            "speculative_enabled = true, but a \(model.rawValue) model has no drafter; use "
                + "a speculative model type, or leave the flag unset"
        case .notAnObject:
            "runtime flags must be a JSON object of knobs, e.g. {\"threads\":4} "
                + "(the one-element array is no longer accepted)"
        case let .axisNotAccepted(key):
            "runtime flags must not carry `\(key)`: the cell comes from the benchmark, "
                + "runtime and model, which are parsed first"
        case let .noSuchCombination(benchmark, runtime, model):
            "no runtime flags defined for \(Self.axes(benchmark, runtime, model))"
        case let .knobNotAllowed(knob, benchmark, runtime, model):
            "knob `\(knob)` is not accepted by \(Self.axes(benchmark, runtime, model))"
        }
    }

    private static func axes(_ benchmark: BenchmarkType, _ runtime: RuntimeType,
                             _ model: ModelType) -> String {
        "\(benchmark.rawValue) \u{d7} \(runtime.rawValue) \u{d7} \(model.rawValue)"
    }
}

extension RuntimeFlagRef {
    /// The crate's `TryFrom<RuntimeFlagRef>` arms, iOS subset, in the crate's order.
    nonisolated func resolve() throws -> RuntimeFlags {
        // Before routing, where the crate checks it: ahead of an MLX arm's refusals and of
        // "no variant".
        if runtimeType != .llamacppIosPipette { try denySwaFull() }
        if benchmarkType == .vlSpecDecode {
            _ = try resolveSpeculation(modelType, speculativeEnabled)
        } else if speculativeEnabled != nil {
            throw refusing(knob: "speculative_enabled")
        }

        switch (benchmarkType, runtimeType, modelType) {
        // Every declared knob is permitted on a llama variant.
        case (.prefillThroughput, .llamacppIosPipette, .ggufText):
            return .prefillThroughputLlamacppIosPipetteGgufText(
                numberGpuLayers: numberGpuLayers, ctxSize: ctxSize, nUbatch: nUbatch,
                threads: threads, swaFull: swaFull, loadMode: loadMode)
        case (.decodeThroughput, .llamacppIosPipette, .ggufText):
            return .decodeThroughputLlamacppIosPipetteGgufText(
                numberGpuLayers: numberGpuLayers, ctxSize: ctxSize, nUbatch: nUbatch,
                threads: threads, swaFull: swaFull, loadMode: loadMode)
        case (.maxMemoryUsage, .llamacppIosPipette, .ggufText):
            return .maxMemoryUsageLlamacppIosPipetteGgufText(
                numberGpuLayers: numberGpuLayers, ctxSize: ctxSize, nUbatch: nUbatch,
                threads: threads, swaFull: swaFull, loadMode: loadMode)
        case (.endToEndLatency, .llamacppIosPipette, .ggufText):
            return .endToEndLatencyLlamacppIosPipetteGgufText(
                numberGpuLayers: numberGpuLayers, ctxSize: ctxSize, nUbatch: nUbatch,
                threads: threads, swaFull: swaFull, loadMode: loadMode)
        case (.eval, .llamacppIosPipette, .ggufText):
            return .evalLlamacppIosPipetteGgufText(
                numberGpuLayers: numberGpuLayers, ctxSize: ctxSize, nUbatch: nUbatch,
                threads: threads, swaFull: swaFull, loadMode: loadMode)
        case (.vlThroughput, .llamacppIosPipette, .ggufVision):
            return .vlThroughputLlamacppIosPipetteGgufVision(
                numberGpuLayers: numberGpuLayers, ctxSize: ctxSize, nUbatch: nUbatch,
                threads: threads, swaFull: swaFull, loadMode: loadMode)
        case (.vlMaxMemory, .llamacppIosPipette, .ggufVision):
            return .vlMaxMemoryLlamacppIosPipetteGgufVision(
                numberGpuLayers: numberGpuLayers, ctxSize: ctxSize, nUbatch: nUbatch,
                threads: threads, swaFull: swaFull, loadMode: loadMode)
        case (.vlSpecDecode, .llamacppIosPipette, .ggufVision):
            return .vlSpecDecodeLlamacppIosPipetteGgufVision(
                numberGpuLayers: numberGpuLayers, ctxSize: ctxSize, nUbatch: nUbatch,
                threads: threads, swaFull: swaFull, loadMode: loadMode,
                speculativeEnabled: speculativeEnabled)
        case (.vlSpecDecode, .llamacppIosPipette, .ggufVisionSpeculative):
            return .vlSpecDecodeLlamacppIosPipetteGgufVisionSpeculative(
                numberGpuLayers: numberGpuLayers, ctxSize: ctxSize, nUbatch: nUbatch,
                threads: threads, swaFull: swaFull, loadMode: loadMode,
                speculativeEnabled: speculativeEnabled)
        case (.asrThroughput, .llamacppIosPipette, .ggufVision):
            return .asrThroughputLlamacppIosPipetteGgufVision(
                numberGpuLayers: numberGpuLayers, ctxSize: ctxSize, nUbatch: nUbatch,
                threads: threads, swaFull: swaFull, loadMode: loadMode)
        case (.asrMaxMemory, .llamacppIosPipette, .ggufVision):
            return .asrMaxMemoryLlamacppIosPipetteGgufVision(
                numberGpuLayers: numberGpuLayers, ctxSize: ctxSize, nUbatch: nUbatch,
                threads: threads, swaFull: swaFull, loadMode: loadMode)
        case (.prefillThroughput, .mlxIosPipette, .mlx):
            try denyLlamaKnobsExceptUbatch()
            try denyServerKnobs()
            return .prefillThroughputMlxIosPipetteMlx(nUbatch: nUbatch)
        case (.decodeThroughput, .mlxIosPipette, .mlx):
            try denyLlamaKnobsExceptUbatch()
            try denyServerKnobs()
            return .decodeThroughputMlxIosPipetteMlx(nUbatch: nUbatch)
        case (.maxMemoryUsage, .mlxIosPipette, .mlx):
            try denyLlamaKnobsExceptUbatch()
            try denyServerKnobs()
            return .maxMemoryUsageMlxIosPipetteMlx(nUbatch: nUbatch)
        case (.endToEndLatency, .mlxIosPipette, .mlx):
            try denyLlamaKnobsExceptUbatch()
            try denyServerKnobs()
            return .endToEndLatencyMlxIosPipetteMlx(nUbatch: nUbatch)
        case (.eval, .mlxIosPipette, .mlx):
            try denyLlamaKnobsExceptUbatch()
            try denyServerKnobs()
            return .evalMlxIosPipetteMlx(nUbatch: nUbatch)
        default:
            throw RuntimeFlagsError.noSuchCombination(
                benchmark: benchmarkType, runtime: runtimeType, model: modelType)
        }
    }

    /// The crate's `deny_swa_full`.
    private nonisolated func denySwaFull() throws {
        if swaFull != nil { throw refusing(knob: "swa_full") }
    }

    /// The crate's `deny_llama_knobs_except_ubatch`, in its order.
    private nonisolated func denyLlamaKnobsExceptUbatch() throws {
        if threads != nil { throw refusing(knob: "threads") }
        if numberGpuLayers != nil { throw refusing(knob: "number_gpu_layers") }
        if loadMode != nil { throw refusing(knob: "load_mode") }
    }

    /// The crate's `deny_server_knobs`; `ctx_size` is the one such field an iOS ref declares.
    private nonisolated func denyServerKnobs() throws {
        if ctxSize != nil { throw refusing(knob: "ctx_size") }
    }

    private nonisolated func refusing(knob: String) -> RuntimeFlagsError {
        .knobNotAllowed(knob: knob, benchmark: benchmarkType, runtime: runtimeType,
                        model: modelType)
    }
}
