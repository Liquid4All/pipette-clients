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
/// `http_timeout_seconds` bounds a request to a server, which an in-process engine lacks,
/// so it is undeclared and a body authored for another client is refused by name. The
/// obsolete `bench_timeout_seconds` and `doomloop` are absorbed unread, as in the crate.
nonisolated struct BenchmarkFlagRef: Codable, Equatable, Sendable {
    let benchmarkType: BenchmarkType
    let runtimeType: RuntimeType
    let modelType: ModelType
    /// The crate's `PositiveSeconds`: whole seconds, at least one.
    let attemptTimeoutSeconds: UInt64?
    var readiness: ReadinessOverrides?

    enum CodingKeys: String, CodingKey, CaseIterable {
        case benchmarkType = "benchmark_type"
        case runtimeType = "runtime_type"
        case modelType = "model_type"
        case attemptTimeoutSeconds = "attempt_timeout_seconds"
        case readiness
        /// Obsolete; declared so `rejectUnknownFields` lets them through. Never read or written.
        case benchTimeoutObsolete = "bench_timeout_seconds"
        case doomloopObsolete = "doomloop"
    }

    init(from decoder: Decoder) throws {
        try rejectUnknownFields(decoder, keys: CodingKeys.self)
        let c = try decoder.container(keyedBy: CodingKeys.self)
        benchmarkType = try c.decode(BenchmarkType.self, forKey: .benchmarkType)
        runtimeType = try c.decode(RuntimeType.self, forKey: .runtimeType)
        modelType = try c.decode(ModelType.self, forKey: .modelType)
        attemptTimeoutSeconds = try c.decodeIfPresent(UInt64.self, forKey: .attemptTimeoutSeconds)
        // Zero would stop every attempt as it starts.
        guard attemptTimeoutSeconds != 0 else {
            throw DecodingError.dataCorruptedError(
                forKey: .attemptTimeoutSeconds, in: c,
                debugDescription: "attempt_timeout_seconds must be at least 1")
        }
        readiness = try c.decodeIfPresent(ReadinessOverrides.self, forKey: .readiness)
    }

    init(benchmarkType: BenchmarkType, runtimeType: RuntimeType, modelType: ModelType,
         attemptTimeoutSeconds: UInt64? = nil, readiness: ReadinessOverrides? = nil) {
        self.benchmarkType = benchmarkType
        self.runtimeType = runtimeType
        self.modelType = modelType
        self.attemptTimeoutSeconds = attemptTimeoutSeconds
        self.readiness = readiness
    }

    init(_ flags: BenchmarkFlags) {
        let axes = flags.axes
        self.init(benchmarkType: axes.benchmark, runtimeType: axes.runtime,
                  modelType: axes.model, attemptTimeoutSeconds: flags.attemptTimeout,
                  readiness: flags.readiness)
    }

    func encode(to encoder: any Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(benchmarkType.rawValue, forKey: .benchmarkType)
        try c.encode(runtimeType.rawValue, forKey: .runtimeType)
        try c.encode(modelType.rawValue, forKey: .modelType)
        try c.encodeIfPresent(attemptTimeoutSeconds, forKey: .attemptTimeoutSeconds)
        try c.encodeIfPresent(readiness, forKey: .readiness)
    }

    /// The keys `submissionValue` removes — the crate's `AXIS_KEYS` for this family.
    static let axisKeys = [CodingKeys.runtimeType, .modelType, .benchmarkType].map(\.rawValue)

    /// The crate's `TryFrom<BenchmarkFlagRef>`, iOS subset. `eval` has no variant: it
    /// carries server settings an in-process engine cannot apply.
    nonisolated func resolve() throws(BenchmarkFlagsError) -> BenchmarkFlags {
        switch (benchmarkType, runtimeType, modelType) {
        // The iOS corpus cells: the gate, and nothing that bounds a process this engine
        // does not spawn.
        case (.vlSpecDecode, .llamacppIosPipette, .ggufVision):
            try deny(attemptTimeoutSeconds != nil, "attempt_timeout_seconds")
            return .vlSpecDecodeLlamacppIosPipetteGgufVision(readiness: readiness)
        case (.vlSpecDecode, .llamacppIosPipette, .ggufVisionSpeculative):
            try deny(attemptTimeoutSeconds != nil, "attempt_timeout_seconds")
            return .vlSpecDecodeLlamacppIosPipetteGgufVisionSpeculative(readiness: readiness)
        case (.prefillThroughput, .llamacppIosPipette, .ggufText):
            return .prefillThroughputLlamacppIosPipetteGgufText(
                attemptTimeoutSeconds: attemptTimeoutSeconds, readiness: readiness)
        case (.decodeThroughput, .llamacppIosPipette, .ggufText):
            return .decodeThroughputLlamacppIosPipetteGgufText(
                attemptTimeoutSeconds: attemptTimeoutSeconds, readiness: readiness)
        case (.endToEndLatency, .llamacppIosPipette, .ggufText):
            return .endToEndLatencyLlamacppIosPipetteGgufText(
                attemptTimeoutSeconds: attemptTimeoutSeconds, readiness: readiness)
        case (.vlThroughput, .llamacppIosPipette, .ggufVision):
            return .vlThroughputLlamacppIosPipetteGgufVision(
                attemptTimeoutSeconds: attemptTimeoutSeconds, readiness: readiness)
        case (.asrThroughput, .llamacppIosPipette, .ggufVision):
            return .asrThroughputLlamacppIosPipetteGgufVision(
                attemptTimeoutSeconds: attemptTimeoutSeconds, readiness: readiness)
        case (.prefillThroughput, .mlxIosPipette, .mlx):
            return .prefillThroughputMlxIosPipetteMlx(
                attemptTimeoutSeconds: attemptTimeoutSeconds, readiness: readiness)
        case (.decodeThroughput, .mlxIosPipette, .mlx):
            return .decodeThroughputMlxIosPipetteMlx(
                attemptTimeoutSeconds: attemptTimeoutSeconds, readiness: readiness)
        case (.endToEndLatency, .mlxIosPipette, .mlx):
            return .endToEndLatencyMlxIosPipetteMlx(
                attemptTimeoutSeconds: attemptTimeoutSeconds, readiness: readiness)
        case (.prefillThroughput, .onnxGenaiIosPipette, .onnxGenai):
            return .prefillThroughputOnnxGenaiIosPipetteOnnxGenai(
                attemptTimeoutSeconds: attemptTimeoutSeconds, readiness: readiness)
        case (.decodeThroughput, .onnxGenaiIosPipette, .onnxGenai):
            return .decodeThroughputOnnxGenaiIosPipetteOnnxGenai(
                attemptTimeoutSeconds: attemptTimeoutSeconds, readiness: readiness)
        case (.endToEndLatency, .onnxGenaiIosPipette, .onnxGenai):
            return .endToEndLatencyOnnxGenaiIosPipetteOnnxGenai(
                attemptTimeoutSeconds: attemptTimeoutSeconds, readiness: readiness)
        // The memory cells take the attempt limit alone, since they do not gate.
        case (.maxMemoryUsage, .llamacppIosPipette, .ggufText):
            try deny(readiness != nil, "readiness")
            return .maxMemoryUsageLlamacppIosPipetteGgufText(
                attemptTimeoutSeconds: attemptTimeoutSeconds)
        case (.maxMemoryUsage, .mlxIosPipette, .mlx):
            try deny(readiness != nil, "readiness")
            return .maxMemoryUsageMlxIosPipetteMlx(attemptTimeoutSeconds: attemptTimeoutSeconds)
        case (.maxMemoryUsage, .onnxGenaiIosPipette, .onnxGenai):
            try deny(readiness != nil, "readiness")
            return .maxMemoryUsageOnnxGenaiIosPipetteOnnxGenai(
                attemptTimeoutSeconds: attemptTimeoutSeconds)
        case (.vlMaxMemory, .llamacppIosPipette, .ggufVision):
            try deny(readiness != nil, "readiness")
            return .vlMaxMemoryLlamacppIosPipetteGgufVision(
                attemptTimeoutSeconds: attemptTimeoutSeconds)
        case (.asrMaxMemory, .llamacppIosPipette, .ggufVision):
            try deny(readiness != nil, "readiness")
            return .asrMaxMemoryLlamacppIosPipetteGgufVision(
                attemptTimeoutSeconds: attemptTimeoutSeconds)
        case (.decodeThroughput, .appleFoundation, .appleFoundationText):
            return .decodeThroughputAppleFoundationAppleFoundationText(
                attemptTimeoutSeconds: attemptTimeoutSeconds, readiness: readiness)
        case (.endToEndLatency, .appleFoundation, .appleFoundationText):
            return .endToEndLatencyAppleFoundationAppleFoundationText(
                attemptTimeoutSeconds: attemptTimeoutSeconds, readiness: readiness)
        default:
            throw .noSuchCombination(
                benchmark: benchmarkType, runtime: runtimeType, model: modelType)
        }
    }

    /// The crate's `deny`: a knob set on a cell whose variant has no field for it.
    private nonisolated func deny(_ set: Bool, _ knob: String) throws(BenchmarkFlagsError) {
        guard set else { return }
        throw .knobNotAllowed(knob: knob, benchmark: benchmarkType, runtime: runtimeType,
                              model: modelType)
    }
}

/// Resolving a `BenchmarkFlagRef` into `BenchmarkFlags` failed.
nonisolated enum BenchmarkFlagsError: Error, Equatable, LocalizedError {
    case noSuchCombination(benchmark: BenchmarkType, runtime: RuntimeType, model: ModelType)
    /// A knob set on a cell whose variant has no field for it.
    case knobNotAllowed(
        knob: String, benchmark: BenchmarkType, runtime: RuntimeType, model: ModelType)

    var errorDescription: String? {
        switch self {
        case let .noSuchCombination(benchmark, runtime, model):
            "no benchmark flags defined for \(model.rawValue) running \(benchmark.rawValue) "
                + "on \(runtime.rawValue)"
        case let .knobNotAllowed(knob, benchmark, runtime, model):
            "\(knob) is not a valid benchmark flag for \(model.rawValue) running "
                + "\(benchmark.rawValue) on \(runtime.rawValue)"
        }
    }
}

/// The resolved `BenchmarkFlags` variant for an iOS cell. The timing cells carry the attempt
/// limit and the readiness gate, the memory cells the limit alone, the corpus cells the
/// gate alone. An unset limit keeps the default (`Attempt.limit`).
nonisolated enum BenchmarkFlags: Equatable, Sendable {
    case vlSpecDecodeLlamacppIosPipetteGgufVision(readiness: ReadinessOverrides?)
    case vlSpecDecodeLlamacppIosPipetteGgufVisionSpeculative(readiness: ReadinessOverrides?)
    case prefillThroughputLlamacppIosPipetteGgufText(
        attemptTimeoutSeconds: UInt64? = nil, readiness: ReadinessOverrides?)
    case decodeThroughputLlamacppIosPipetteGgufText(
        attemptTimeoutSeconds: UInt64? = nil, readiness: ReadinessOverrides?)
    case endToEndLatencyLlamacppIosPipetteGgufText(
        attemptTimeoutSeconds: UInt64? = nil, readiness: ReadinessOverrides?)
    case vlThroughputLlamacppIosPipetteGgufVision(
        attemptTimeoutSeconds: UInt64? = nil, readiness: ReadinessOverrides?)
    case asrThroughputLlamacppIosPipetteGgufVision(
        attemptTimeoutSeconds: UInt64? = nil, readiness: ReadinessOverrides?)
    case prefillThroughputMlxIosPipetteMlx(
        attemptTimeoutSeconds: UInt64? = nil, readiness: ReadinessOverrides?)
    case decodeThroughputMlxIosPipetteMlx(
        attemptTimeoutSeconds: UInt64? = nil, readiness: ReadinessOverrides?)
    case endToEndLatencyMlxIosPipetteMlx(
        attemptTimeoutSeconds: UInt64? = nil, readiness: ReadinessOverrides?)
    case prefillThroughputOnnxGenaiIosPipetteOnnxGenai(
        attemptTimeoutSeconds: UInt64? = nil, readiness: ReadinessOverrides?)
    case decodeThroughputOnnxGenaiIosPipetteOnnxGenai(
        attemptTimeoutSeconds: UInt64? = nil, readiness: ReadinessOverrides?)
    case endToEndLatencyOnnxGenaiIosPipetteOnnxGenai(
        attemptTimeoutSeconds: UInt64? = nil, readiness: ReadinessOverrides?)
    case maxMemoryUsageLlamacppIosPipetteGgufText(attemptTimeoutSeconds: UInt64?)
    case maxMemoryUsageMlxIosPipetteMlx(attemptTimeoutSeconds: UInt64?)
    case maxMemoryUsageOnnxGenaiIosPipetteOnnxGenai(attemptTimeoutSeconds: UInt64?)
    case vlMaxMemoryLlamacppIosPipetteGgufVision(attemptTimeoutSeconds: UInt64?)
    case asrMaxMemoryLlamacppIosPipetteGgufVision(attemptTimeoutSeconds: UInt64?)
    case decodeThroughputAppleFoundationAppleFoundationText(
        attemptTimeoutSeconds: UInt64? = nil, readiness: ReadinessOverrides?)
    case endToEndLatencyAppleFoundationAppleFoundationText(
        attemptTimeoutSeconds: UInt64? = nil, readiness: ReadinessOverrides?)

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
        case .maxMemoryUsageLlamacppIosPipetteGgufText: (.maxMemoryUsage, .llamacppIosPipette, .ggufText)
        case .maxMemoryUsageMlxIosPipetteMlx: (.maxMemoryUsage, .mlxIosPipette, .mlx)
        case .maxMemoryUsageOnnxGenaiIosPipetteOnnxGenai:
            (.maxMemoryUsage, .onnxGenaiIosPipette, .onnxGenai)
        case .vlMaxMemoryLlamacppIosPipetteGgufVision: (.vlMaxMemory, .llamacppIosPipette, .ggufVision)
        case .asrMaxMemoryLlamacppIosPipetteGgufVision:
            (.asrMaxMemory, .llamacppIosPipette, .ggufVision)
        case .decodeThroughputAppleFoundationAppleFoundationText:
            (.decodeThroughput, .appleFoundation, .appleFoundationText)
        case .endToEndLatencyAppleFoundationAppleFoundationText:
            (.endToEndLatency, .appleFoundation, .appleFoundationText)
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
             let .prefillThroughputLlamacppIosPipetteGgufText(_, value),
             let .decodeThroughputLlamacppIosPipetteGgufText(_, value),
             let .endToEndLatencyLlamacppIosPipetteGgufText(_, value),
             let .vlThroughputLlamacppIosPipetteGgufVision(_, value),
             let .asrThroughputLlamacppIosPipetteGgufVision(_, value),
             let .prefillThroughputMlxIosPipetteMlx(_, value),
             let .decodeThroughputMlxIosPipetteMlx(_, value),
             let .endToEndLatencyMlxIosPipetteMlx(_, value),
             let .prefillThroughputOnnxGenaiIosPipetteOnnxGenai(_, value),
             let .decodeThroughputOnnxGenaiIosPipetteOnnxGenai(_, value),
             let .endToEndLatencyOnnxGenaiIosPipetteOnnxGenai(_, value),
             let .decodeThroughputAppleFoundationAppleFoundationText(_, value),
             let .endToEndLatencyAppleFoundationAppleFoundationText(_, value):
            value
        case .maxMemoryUsageLlamacppIosPipetteGgufText, .maxMemoryUsageMlxIosPipetteMlx,
             .maxMemoryUsageOnnxGenaiIosPipetteOnnxGenai, .vlMaxMemoryLlamacppIosPipetteGgufVision,
             .asrMaxMemoryLlamacppIosPipetteGgufVision:
            nil
        }
    }

    /// The crate's `attempt_timeout`: the limit on one attempt, in whole seconds. `nil`
    /// keeps the default.
    var attemptTimeout: UInt64? {
        switch self {
        case let .prefillThroughputLlamacppIosPipetteGgufText(value, _),
             let .decodeThroughputLlamacppIosPipetteGgufText(value, _),
             let .endToEndLatencyLlamacppIosPipetteGgufText(value, _),
             let .vlThroughputLlamacppIosPipetteGgufVision(value, _),
             let .asrThroughputLlamacppIosPipetteGgufVision(value, _),
             let .prefillThroughputMlxIosPipetteMlx(value, _),
             let .decodeThroughputMlxIosPipetteMlx(value, _),
             let .endToEndLatencyMlxIosPipetteMlx(value, _),
             let .prefillThroughputOnnxGenaiIosPipetteOnnxGenai(value, _),
             let .decodeThroughputOnnxGenaiIosPipetteOnnxGenai(value, _),
             let .endToEndLatencyOnnxGenaiIosPipetteOnnxGenai(value, _),
             let .decodeThroughputAppleFoundationAppleFoundationText(value, _),
             let .endToEndLatencyAppleFoundationAppleFoundationText(value, _),
             let .maxMemoryUsageLlamacppIosPipetteGgufText(value),
             let .maxMemoryUsageMlxIosPipetteMlx(value),
             let .maxMemoryUsageOnnxGenaiIosPipetteOnnxGenai(value),
             let .vlMaxMemoryLlamacppIosPipetteGgufVision(value),
             let .asrMaxMemoryLlamacppIosPipetteGgufVision(value):
            value
        case .vlSpecDecodeLlamacppIosPipetteGgufVision,
             .vlSpecDecodeLlamacppIosPipetteGgufVisionSpeculative:
            nil
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
