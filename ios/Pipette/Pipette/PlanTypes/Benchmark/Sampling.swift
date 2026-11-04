import Foundation

// The sampling an entry may pin — pipette-mgmt's
// `crates/pipette-plan-types/src/benchmark/sampling.rs`.

/// A bounded sampling knob that validates at construction and on decode, as the crate's
/// `nutype`s do, so an out-of-range pin fails where the entry is read.
protocol SamplingKnob: Codable, Hashable, Sendable {
    associatedtype Raw: Codable & Hashable & Sendable & Comparable
    nonisolated var value: Raw { get }
    nonisolated init(validated value: Raw)
    nonisolated static var bounds: ClosedRange<Raw> { get }
    /// The entry's field, for the refusal.
    nonisolated static var field: String { get }
}

extension SamplingKnob {
    /// A `NaN` fails `contains` like any value outside the bounds.
    nonisolated init(_ value: Raw) throws {
        guard Self.bounds.contains(value) else {
            throw SamplingError.outOfRange(field: Self.field, value: "\(value)",
                                           bounds: "\(Self.bounds)")
        }
        self.init(validated: value)
    }

    nonisolated init(from decoder: any Decoder) throws {
        try self.init(try decoder.singleValueContainer().decode(Raw.self))
    }

    nonisolated func encode(to encoder: any Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(value)
    }
}

nonisolated enum SamplingError: Error, Equatable, LocalizedError {
    case outOfRange(field: String, value: String, bounds: String)

    var errorDescription: String? {
        switch self {
        case let .outOfRange(field, value, bounds): "\(field) \(value) is outside \(bounds)"
        }
    }
}

/// Sampling temperature, `0.0` (greedy) to `2.0`, the most the OpenAI-compatible servers
/// accept.
nonisolated struct SamplingTemperature: SamplingKnob {
    let value: Double
    /// A `-0.0` passes the `>= 0.0` bound but prints differently from `0.0`. This initializer
    /// and the ones of `TopP` and `MinP` store it as `0.0`, as the crate's `positive_zero` does.
    init(validated value: Double) { self.value = value == 0 ? 0 : value }
    static let bounds = 0.0 ... 2.0
    static let field = "parameter_temperature"
    static let greedy = SamplingTemperature(validated: 0)
}

/// Nucleus-sampling mass.
nonisolated struct TopP: SamplingKnob {
    let value: Double
    init(validated value: Double) { self.value = value == 0 ? 0 : value }
    static let bounds = 0.0 ... 1.0
    static let field = "parameter_top_p"
}

/// Minimum-probability cutoff relative to the most likely token; `0` disables it.
nonisolated struct MinP: SamplingKnob {
    let value: Double
    init(validated value: Double) { self.value = value == 0 ? 0 : value }
    static let bounds = 0.0 ... 1.0
    static let field = "parameter_min_p"
}

/// Tokens the top-k cut keeps; `0` disables it, which turns off llama.cpp's default of 40.
nonisolated struct TopK: SamplingKnob {
    let value: Int32
    init(validated value: Int32) { self.value = value }
    static let bounds: ClosedRange<Int32> = 0 ... .max
    static let field = "parameter_top_k"
}

/// The base seed an entry pins. Capped at `UInt32.max`, llama.cpp's seed width.
nonisolated struct SamplingSeed: SamplingKnob {
    let value: Int64
    init(validated value: Int64) { self.value = value }
    static let bounds: ClosedRange<Int64> = 0 ... Int64(UInt32.max)
    static let field = "parameter_seed"
}
