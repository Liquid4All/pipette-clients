import Foundation

// The benchmark-flag wire forms and their resolution — pipette-mgmt's
// `crates/pipette-plan-types/src/cell/benchmark_flags.rs`. Shared decode machinery is in
// `FlagDecoding.swift`.

/// The crate's `ReadinessOverrides`, authored as `readiness = { max_wait_secs = 1800 }`.
/// Optional fields, unlike the `ResolvedReadiness` the gate runs with: this is a request,
/// not what happened.
nonisolated struct ReadinessOverrides: Codable, Equatable, Sendable {
    /// The gate's deadline in whole seconds. Unset keeps the built-in.
    var maxWaitSecs: UInt64?
    /// Waive the thermal criterion (OS thermal state and die temperature); the CPU-load
    /// check still applies. A cell run with it is not comparable to a gated one, which is
    /// why a plan sets it per cell rather than a device setting waiving it fleet-wide.
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
}

/// The crate's `BenchmarkFlagRef`, the flat wire form of `BenchmarkFlags`.
///
/// `http_timeout_seconds` and `bench_timeout_seconds` bound a server or a watched
/// subprocess, which an in-process engine lacks, so they are undeclared and a body
/// authored for another client is refused by name. The obsolete `doomloop` is absorbed
/// unread, as in the crate.
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
        /// Obsolete; declared so `rejectUnknownFields` lets it through. Never read or written.
        case doomloopObsolete = "doomloop"
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

    /// The keys `submissionValue` removes — the crate's `AXIS_KEYS` for this family.
    static let axisKeys = [CodingKeys.runtimeType, .modelType, .benchmarkType].map(\.rawValue)

    /// The crate's `TryFrom<BenchmarkFlagRef>`. Only the timing and corpus cells this build
    /// runs have a variant: `eval` carries server settings an in-process engine cannot
    /// apply, and the memory kinds do not gate at all.
    nonisolated func resolve() throws(BenchmarkFlagsError) -> BenchmarkFlags {
        switch (benchmarkType, runtimeType, modelType) {
        case (.vlSpecDecode, .llamacppIosPipette, .ggufVision):
            return .vlSpecDecodeLlamacppIosPipetteGgufVision(readiness: readiness)
        case (.vlSpecDecode, .llamacppIosPipette, .ggufVisionSpeculative):
            return .vlSpecDecodeLlamacppIosPipetteGgufVisionSpeculative(readiness: readiness)
        case (.prefillThroughput, .llamacppIosPipette, .ggufText):
            return .prefillThroughputLlamacppIosPipetteGgufText(readiness: readiness)
        case (.decodeThroughput, .llamacppIosPipette, .ggufText):
            return .decodeThroughputLlamacppIosPipetteGgufText(readiness: readiness)
        case (.endToEndLatency, .llamacppIosPipette, .ggufText):
            return .endToEndLatencyLlamacppIosPipetteGgufText(readiness: readiness)
        case (.vlThroughput, .llamacppIosPipette, .ggufVision):
            return .vlThroughputLlamacppIosPipetteGgufVision(readiness: readiness)
        case (.asrThroughput, .llamacppIosPipette, .ggufVision):
            return .asrThroughputLlamacppIosPipetteGgufVision(readiness: readiness)
        case (.prefillThroughput, .mlxIosPipette, .mlx):
            return .prefillThroughputMlxIosPipetteMlx(readiness: readiness)
        case (.decodeThroughput, .mlxIosPipette, .mlx):
            return .decodeThroughputMlxIosPipetteMlx(readiness: readiness)
        case (.endToEndLatency, .mlxIosPipette, .mlx):
            return .endToEndLatencyMlxIosPipetteMlx(readiness: readiness)
        case (.prefillThroughput, .onnxGenaiIosPipette, .onnxGenai):
            return .prefillThroughputOnnxGenaiIosPipetteOnnxGenai(readiness: readiness)
        case (.decodeThroughput, .onnxGenaiIosPipette, .onnxGenai):
            return .decodeThroughputOnnxGenaiIosPipetteOnnxGenai(readiness: readiness)
        case (.endToEndLatency, .onnxGenaiIosPipette, .onnxGenai):
            return .endToEndLatencyOnnxGenaiIosPipetteOnnxGenai(readiness: readiness)
        default:
            throw .noSuchCombination(
                benchmark: benchmarkType, runtime: runtimeType, model: modelType)
        }
    }
}

/// Resolving a `BenchmarkFlagRef` into `BenchmarkFlags` failed. iOS subset:
/// `KnobNotAllowed` and `KnobOutOfRange` refuse nothing here: `readiness` is the only knob
/// an iOS ref declares, and every cell with a variant carries it.
nonisolated enum BenchmarkFlagsError: Error, Equatable, LocalizedError {
    case noSuchCombination(benchmark: BenchmarkType, runtime: RuntimeType, model: ModelType)

    var errorDescription: String? {
        switch self {
        case let .noSuchCombination(benchmark, runtime, model):
            "no benchmark flags defined for \(model.rawValue) running \(benchmark.rawValue) "
                + "on \(runtime.rawValue)"
        }
    }
}

/// The resolved `BenchmarkFlags` variant for an iOS cell — the timing cells and the
/// corpus cells, each carrying the readiness gate and nothing else.
nonisolated enum BenchmarkFlags: Equatable, Sendable {
    case vlSpecDecodeLlamacppIosPipetteGgufVision(readiness: ReadinessOverrides?)
    case vlSpecDecodeLlamacppIosPipetteGgufVisionSpeculative(readiness: ReadinessOverrides?)
    case prefillThroughputLlamacppIosPipetteGgufText(readiness: ReadinessOverrides?)
    case decodeThroughputLlamacppIosPipetteGgufText(readiness: ReadinessOverrides?)
    case endToEndLatencyLlamacppIosPipetteGgufText(readiness: ReadinessOverrides?)
    case vlThroughputLlamacppIosPipetteGgufVision(readiness: ReadinessOverrides?)
    case asrThroughputLlamacppIosPipetteGgufVision(readiness: ReadinessOverrides?)
    case prefillThroughputMlxIosPipetteMlx(readiness: ReadinessOverrides?)
    case decodeThroughputMlxIosPipetteMlx(readiness: ReadinessOverrides?)
    case endToEndLatencyMlxIosPipetteMlx(readiness: ReadinessOverrides?)
    case prefillThroughputOnnxGenaiIosPipetteOnnxGenai(readiness: ReadinessOverrides?)
    case decodeThroughputOnnxGenaiIosPipetteOnnxGenai(readiness: ReadinessOverrides?)
    case endToEndLatencyOnnxGenaiIosPipetteOnnxGenai(readiness: ReadinessOverrides?)

    var axes: FlagAxes {
        switch self {
        case .vlSpecDecodeLlamacppIosPipetteGgufVision:
            (.vlSpecDecode, .llamacppIosPipette, .ggufVision)
        case .vlSpecDecodeLlamacppIosPipetteGgufVisionSpeculative:
            (.vlSpecDecode, .llamacppIosPipette, .ggufVisionSpeculative)
        case .prefillThroughputLlamacppIosPipetteGgufText: (.prefillThroughput, .llamacppIosPipette, .ggufText)
        case .decodeThroughputLlamacppIosPipetteGgufText: (.decodeThroughput, .llamacppIosPipette, .ggufText)
        case .endToEndLatencyLlamacppIosPipetteGgufText: (.endToEndLatency, .llamacppIosPipette, .ggufText)
        case .vlThroughputLlamacppIosPipetteGgufVision: (.vlThroughput, .llamacppIosPipette, .ggufVision)
        case .asrThroughputLlamacppIosPipetteGgufVision:
            (.asrThroughput, .llamacppIosPipette, .ggufVision)
        case .prefillThroughputMlxIosPipetteMlx: (.prefillThroughput, .mlxIosPipette, .mlx)
        case .decodeThroughputMlxIosPipetteMlx: (.decodeThroughput, .mlxIosPipette, .mlx)
        case .endToEndLatencyMlxIosPipetteMlx: (.endToEndLatency, .mlxIosPipette, .mlx)
        case .prefillThroughputOnnxGenaiIosPipetteOnnxGenai:
            (.prefillThroughput, .onnxGenaiIosPipette, .onnxGenai)
        case .decodeThroughputOnnxGenaiIosPipetteOnnxGenai:
            (.decodeThroughput, .onnxGenaiIosPipette, .onnxGenai)
        case .endToEndLatencyOnnxGenaiIosPipetteOnnxGenai:
            (.endToEndLatency, .onnxGenaiIosPipette, .onnxGenai)
        }
    }

    /// The crate's `matches`.
    func matches(benchmark: BenchmarkType, runtime: Runtime, model: Model) -> Bool {
        axes == (benchmark, RuntimeType.of(runtime), ModelType.of(model))
    }

    var readiness: ReadinessOverrides? {
        switch self {
        case let .vlSpecDecodeLlamacppIosPipetteGgufVision(value),
             let .vlSpecDecodeLlamacppIosPipetteGgufVisionSpeculative(value),
             let .prefillThroughputLlamacppIosPipetteGgufText(value),
             let .decodeThroughputLlamacppIosPipetteGgufText(value),
             let .endToEndLatencyLlamacppIosPipetteGgufText(value),
             let .vlThroughputLlamacppIosPipetteGgufVision(value),
             let .asrThroughputLlamacppIosPipetteGgufVision(value),
             let .prefillThroughputMlxIosPipetteMlx(value),
             let .decodeThroughputMlxIosPipetteMlx(value),
             let .endToEndLatencyMlxIosPipetteMlx(value),
             let .prefillThroughputOnnxGenaiIosPipetteOnnxGenai(value),
             let .decodeThroughputOnnxGenaiIosPipetteOnnxGenai(value),
             let .endToEndLatencyOnnxGenaiIosPipetteOnnxGenai(value):
            value
        }
    }

    /// The knobs alone as JSON, axes removed: the crate's `submission_value`, what a result
    /// carries as `benchmark_flags`. It is how a gated run is told apart from one that
    /// waived the thermal criterion. `.sortedKeys` as in `RuntimeFlags.submissionValue`.
    func submissionValue() throws -> String {
        let encoded = try JSONEncoder().encode(BenchmarkFlagRef(self))
        guard var object = try JSONSerialization.jsonObject(with: encoded) as? [String: Any]
        else { return "{}" }
        BenchmarkFlagRef.axisKeys.forEach { object.removeValue(forKey: $0) }
        return String(
            decoding: try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys]),
            as: UTF8.self)
    }
}
