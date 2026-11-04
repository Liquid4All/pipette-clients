import Foundation
import Testing

@testable import Pipette

/// `KnownEvalId` / `EvalId`, mirroring pipette-mgmt's
/// `crates/pipette-plan-types/src/benchmark/eval_id.rs`.
struct EvalIdTests {
    /// The digit trap: `math_500` is spelled explicitly, not derived from the case name.
    @Test func knownEvalIdSpellingMatchesScoringBackend() {
        #expect(KnownEvalId.ifbench.rawValue == "ifbench")
        #expect(KnownEvalId.ifstruct.rawValue == "ifstruct")
        #expect(KnownEvalId.gpqaDiamond.rawValue == "gpqa_diamond")
        #expect(KnownEvalId.math500.rawValue == "math_500")
    }

    @Test func knownEvalIdRoundTripsEveryVariant() {
        for known in KnownEvalId.allCases {
            #expect(EvalId(known.rawValue) == .known(known))
        }
    }

    /// `known_eval_id_unrecognized_is_none`: calibration-only ids (not scoreable) and
    /// anything unknown are outside the known set.
    @Test func calibrationAndUnknownIdsAreNotKnown() {
        #expect(KnownEvalId(rawValue: "mmlu_pro") == nil)
        #expect(KnownEvalId(rawValue: "totally_unknown_eval") == nil)
    }

    @Test func evalIdParsesKnownAndPreservesUnknown() {
        #expect(EvalId("ifbench") == .known(.ifbench))
        #expect(EvalId("math_500") == .known(.math500))
        #expect(EvalId("some_future_eval") == .unknown("some_future_eval"))
    }

    /// `eval_id_round_trips_through_string_losslessly`, through a container since an id is
    /// always a field on the wire.
    @Test func evalIdRoundTripsThroughAStringLosslessly() throws {
        struct Wrapper: Codable, Equatable { let evalId: EvalId }
        for id in [EvalId.known(.math500), EvalId.unknown("some_future_eval")] {
            let data = try JSONEncoder().encode(Wrapper(evalId: id))
            #expect(try JSONDecoder().decode(Wrapper.self, from: data).evalId == id)
        }
        let json = String(decoding: try JSONEncoder().encode(Wrapper(evalId: .known(.math500))), as: UTF8.self)
        #expect(json.contains("\"math_500\""))
    }

    @Test func anUnknownEvalIsGreedy() {
        #expect(EvalId.unknown("anything").samplingTemperature == 0.0)
        #expect(EvalId.known(.ifbench).samplingTemperature == 0.6)
    }

    /// `every_eval_id_rule_is_a_valid_temperature`: `BenchmarkParams` declares greedy
    /// sampling for a rule out of range, so every rule must be in range.
    @Test(arguments: KnownEvalId.allCases)
    func everyEvalIdRuleIsAValidTemperature(eval: KnownEvalId) {
        #expect(throws: Never.self) { try SamplingTemperature(eval.samplingTemperature) }
    }
}
