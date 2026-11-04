import Foundation

// pipette-mgmt's `crates/pipette-plan-types/src/cell/model_flags.rs`.

nonisolated enum ModelFlagsError: Error, Equatable, LocalizedError {
    case noSuchCombination(benchmark: BenchmarkType, model: ModelType)
    case knobNotAllowed(knob: String, benchmark: BenchmarkType, model: ModelType)

    var errorDescription: String? {
        switch self {
        case let .noSuchCombination(benchmark, model):
            "no model flags defined for \(model.rawValue) on \(benchmark.rawValue)"
        case let .knobNotAllowed(knob, benchmark, model):
            "\(knob) is not a valid model flag for \(model.rawValue) on \(benchmark.rawValue)"
        }
    }
}

/// The crate's `ReasoningEffort`: the `reasoning_effort` a chat template reads (gpt-oss
/// renders it as `Reasoning: <effort>`). Closed to the three levels both desktop eval
/// runtimes pass through.
nonisolated enum ReasoningEffort: String, Codable, Hashable, Sendable, CaseIterable {
    case low
    case medium
    case high
}

/// The crate's `ModelFlags`, one variant per `(benchmark, model)` cell that carries
/// generation flags. They shape only the chat-templated eval path, so every variant is
/// `eval…`. They attach to the cell, never to `Model`, which stays identity-only.
///
/// `evalTorch` is carried though no torch engine exists on a phone, so a claim naming it
/// is refused by the engine check rather than by a gap in this enum. No engine here
/// applies `enableThinking` or `reasoningEffort` either (see `ClientRunSpec.validated`).
nonisolated enum ModelFlags: Hashable, Sendable, CaseIterable {
    case evalGgufText(enableThinking: Bool?, reasoningEffort: ReasoningEffort? = nil)
    case evalGgufVision(enableThinking: Bool?)
    case evalMlx(enableThinking: Bool?)
    case evalTorch(enableThinking: Bool?, reasoningEffort: ReasoningEffort? = nil)

    /// One of each variant, for the sweeps that must cover every one.
    static let allCases: [ModelFlags] = [
        .evalGgufText(enableThinking: nil), .evalGgufVision(enableThinking: nil),
        .evalMlx(enableThinking: nil), .evalTorch(enableThinking: nil),
    ]

    /// The crate's `axes()`.
    var axes: (benchmark: BenchmarkType, model: ModelType) {
        switch self {
        case .evalGgufText: (.eval, .ggufText)
        case .evalGgufVision: (.eval, .ggufVision)
        case .evalMlx: (.eval, .mlx)
        case .evalTorch: (.eval, .torch)
        }
    }

    /// The crate's `matches`.
    func matches(benchmark: BenchmarkType, model: Model) -> Bool {
        axes.benchmark == benchmark && axes.model == ModelType.of(model)
    }

    /// `nil` is not "no thinking": the kwarg is left out and the engine's default applies.
    var enableThinking: Bool? {
        switch self {
        case let .evalGgufText(v, _), let .evalGgufVision(v), let .evalMlx(v),
             let .evalTorch(v, _): v
        }
    }

    var reasoningEffort: ReasoningEffort? {
        switch self {
        case let .evalGgufText(_, effort), let .evalTorch(_, effort): effort
        case .evalGgufVision, .evalMlx: nil
        }
    }

    /// The crate's `canonical_string`. Its spelling is load-bearing for row hashing.
    var canonicalString: String? {
        let knobs = [
            enableThinking.map { "enable_thinking=\($0)" },
            reasoningEffort.map { "reasoning_effort=\($0.rawValue)" },
        ].compactMap { $0 }
        return knobs.isEmpty ? nil : knobs.joined(separator: ",")
    }

    /// The crate's `submission_string`. `nil` off evals: a flag on a throughput row would
    /// split warehouse joins on a value that had no effect.
    func submissionString(_ benchmarkType: BenchmarkType) -> String? {
        switch benchmarkType {
        case .eval:
            canonicalString
        case .prefillThroughput, .decodeThroughput, .endToEndLatency, .maxMemoryUsage,
             .vlThroughput, .vlMaxMemory, .vlSpecDecode, .asrThroughput, .asrMaxMemory:
            nil
        }
    }
}

/// The crate's `ModelFlagRef`, the flat wire form of `ModelFlags`. No runtime axis.
nonisolated struct ModelFlagRef: Codable, Equatable, Sendable {
    let benchmarkType: BenchmarkType
    let modelType: ModelType
    let enableThinking: Bool?
    let reasoningEffort: ReasoningEffort?

    enum CodingKeys: String, CodingKey, CaseIterable {
        case modelType = "model_type"
        case benchmarkType = "benchmark_type"
        case enableThinking = "enable_thinking"
        case reasoningEffort = "reasoning_effort"
    }

    init(benchmarkType: BenchmarkType, modelType: ModelType, enableThinking: Bool? = nil,
         reasoningEffort: ReasoningEffort? = nil) {
        self.benchmarkType = benchmarkType
        self.modelType = modelType
        self.enableThinking = enableThinking
        self.reasoningEffort = reasoningEffort
    }

    /// The crate's `From<ModelFlags>`.
    init(_ flags: ModelFlags) {
        self.init(benchmarkType: flags.axes.benchmark, modelType: flags.axes.model,
                  enableThinking: flags.enableThinking, reasoningEffort: flags.reasoningEffort)
    }

    init(from decoder: Decoder) throws {
        try rejectUnknownFields(decoder, keys: CodingKeys.self)
        let c = try decoder.container(keyedBy: CodingKeys.self)
        benchmarkType = try c.decode(BenchmarkType.self, forKey: .benchmarkType)
        modelType = try c.decode(ModelType.self, forKey: .modelType)
        enableThinking = try c.decodeIfPresent(Bool.self, forKey: .enableThinking)
        reasoningEffort = try c.decodeIfPresent(ReasoningEffort.self, forKey: .reasoningEffort)
    }

    func encode(to encoder: any Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(modelType.rawValue, forKey: .modelType)
        try c.encode(benchmarkType.rawValue, forKey: .benchmarkType)
        try c.encodeIfPresent(enableThinking, forKey: .enableThinking)
        try c.encodeIfPresent(reasoningEffort, forKey: .reasoningEffort)
    }

    /// The crate's `TryFrom<ModelFlagRef>`. A `reasoning_effort` on a variant with no
    /// field for it is refused by name.
    func resolve() throws(ModelFlagsError) -> ModelFlags {
        switch (benchmarkType, modelType) {
        case (.eval, .ggufText):
            return .evalGgufText(enableThinking: enableThinking, reasoningEffort: reasoningEffort)
        case (.eval, .ggufVision):
            try denyReasoningEffort()
            return .evalGgufVision(enableThinking: enableThinking)
        case (.eval, .mlx):
            try denyReasoningEffort()
            return .evalMlx(enableThinking: enableThinking)
        case (.eval, .torch):
            return .evalTorch(enableThinking: enableThinking, reasoningEffort: reasoningEffort)
        default:
            throw .noSuchCombination(benchmark: benchmarkType, model: modelType)
        }
    }

    /// The crate's `deny_reasoning_effort`.
    private func denyReasoningEffort() throws(ModelFlagsError) {
        if reasoningEffort != nil {
            throw .knobNotAllowed(knob: CodingKeys.reasoningEffort.rawValue,
                                  benchmark: benchmarkType, model: modelType)
        }
    }
}
