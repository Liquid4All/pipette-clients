import Foundation

// The eval a benchmark names — pipette-mgmt's
// `crates/pipette-plan-types/src/benchmark/eval_id.rs`.

/// The crate's `KnownEvalId`: the evals `pipette-scores` can score. Raw values must match
/// its `EvalId` exactly.
nonisolated enum KnownEvalId: String, CaseIterable, Equatable {
    case ifbench
    case ifstruct
    case gpqaDiamond = "gpqa_diamond"
    case math500 = "math_500"

    /// Pass@k evals sample at `0.6` so their repeated draws differ. No `default:`, so a
    /// new case forces a decision.
    var samplingTemperature: Double {
        switch self {
        case .ifbench, .ifstruct, .gpqaDiamond, .math500: return 0.6
        }
    }
}

/// The crate's `EvalId`. Total: an id this client does not know is kept verbatim as
/// `.unknown`, so it still round-trips. A plain string on the wire.
nonisolated enum EvalId: Equatable {
    case known(KnownEvalId)
    case unknown(String)

    init(_ raw: String) {
        self = KnownEvalId(rawValue: raw).map(EvalId.known) ?? .unknown(raw)
    }

    var rawValue: String {
        switch self {
        case .known(let known): return known.rawValue
        case .unknown(let raw): return raw
        }
    }

    /// An unknown eval is greedy.
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
