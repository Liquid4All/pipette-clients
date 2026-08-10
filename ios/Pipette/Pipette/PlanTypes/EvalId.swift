import Foundation

/// The evals the scoring backend can score (mirrors the Rust `KnownEvalId` /
/// `pipette-scores`' `EvalId`). `math_500` is spelled explicitly — it wouldn't
/// derive from the `math500` case name (digit-adjacency).
nonisolated enum KnownEvalId: String, CaseIterable, Equatable {
    case ifbench
    case ifstruct
    case gpqaDiamond = "gpqa_diamond"
    case math500 = "math_500"

    /// Sampling temperature for this eval's `/completion` requests. Every
    /// supported eval is generative pass@k, so all sample at `0.6`; the switch
    /// is exhaustive so a new case forces a decision.
    var samplingTemperature: Double {
        switch self {
        case .ifbench, .ifstruct, .gpqaDiamond, .math500: return 0.6
        }
    }
}

/// An eval id parsed from the loose `parameter_eval_id` (mirrors the Rust
/// `EvalId`). Total: a known id is `.known`, anything else is preserved verbatim
/// as `.unknown`, so an eval the client doesn't know yet still round-trips
/// losslessly. Wire form is a plain string, not a tagged object.
nonisolated enum EvalId: Equatable {
    case known(KnownEvalId)
    case unknown(String)

    init(_ raw: String) {
        self = KnownEvalId(rawValue: raw).map(EvalId.known) ?? .unknown(raw)
    }

    /// The wire spelling: a known eval's canonical id, or the preserved string.
    var rawValue: String {
        switch self {
        case .known(let known): return known.rawValue
        case .unknown(let raw): return raw
        }
    }

    /// Client-side sampling temperature for this eval's `/completion` requests
    /// (see `KnownEvalId.samplingTemperature`); an unknown eval is greedy.
    var samplingTemperature: Double {
        switch self {
        case .known(let known): return known.samplingTemperature
        case .unknown: return 0.0
        }
    }
}

extension EvalId: Codable {
    init(from decoder: any Decoder) throws {
        let raw = try decoder.singleValueContainer().decode(String.self)
        self = EvalId(raw)
    }

    func encode(to encoder: any Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(rawValue)
    }
}
