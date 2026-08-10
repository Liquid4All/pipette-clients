import XCTest

@testable import Pipette

/// Mirrors the Rust `KnownEvalId`/`EvalId` tests in `crates/pipette-plan-types/src/benchmark/mod.rs`:
/// the known set parses, spellings match the scoring backend, unknown ids are
/// preserved verbatim, and the `Codable` round-trip through a string is lossless.
final class EvalIdTests: XCTestCase {
    func testKnownEvalIdSpellingMatchesScoringBackend() {
        // Wire spellings must match pipette-scores' EvalId exactly. The digit
        // trap: math_500 is spelled explicitly, not derived from the case name.
        XCTAssertEqual(KnownEvalId.ifbench.rawValue, "ifbench")
        XCTAssertEqual(KnownEvalId.ifstruct.rawValue, "ifstruct")
        XCTAssertEqual(KnownEvalId.gpqaDiamond.rawValue, "gpqa_diamond")
        XCTAssertEqual(KnownEvalId.math500.rawValue, "math_500")
    }

    func testKnownEvalIdRoundTripsEveryVariant() {
        for known in KnownEvalId.allCases {
            XCTAssertEqual(EvalId(known.rawValue), .known(known))
        }
    }

    func testEvalIdParsesKnownAndPreservesUnknown() {
        XCTAssertEqual(EvalId("ifbench"), .known(.ifbench))
        XCTAssertEqual(EvalId("math_500"), .known(.math500))
        // An id the client doesn't recognize is kept verbatim, not dropped.
        XCTAssertEqual(EvalId("some_future_eval"), .unknown("some_future_eval"))
    }

    func testEvalIdRoundTripsThroughStringLosslessly() throws {
        // EvalId is always a field on the wire (never a top-level fragment), so
        // exercise it through a container. Known -> canonical wire spelling,
        // unknown -> preserved exactly; both survive encode -> decode unchanged.
        struct Wrapper: Codable, Equatable { let evalId: EvalId }
        for id in [EvalId.known(.math500), EvalId.unknown("some_future_eval")] {
            let data = try JSONEncoder().encode(Wrapper(evalId: id))
            XCTAssertEqual(try JSONDecoder().decode(Wrapper.self, from: data).evalId, id)
        }
        let json = String(decoding: try JSONEncoder().encode(Wrapper(evalId: .known(.math500))), as: UTF8.self)
        XCTAssertTrue(json.contains("\"math_500\""), json)
    }

    func testEvalIdUnknownIsGreedy() {
        XCTAssertEqual(EvalId.unknown("anything").samplingTemperature, 0.0)
        XCTAssertEqual(EvalId.known(.ifbench).samplingTemperature, 0.6)
    }

    func testBenchmarkItemSurfacesTypedEvalId() {
        let eval = BenchmarkItem(
            benchmarkId: "eval_ifbench", benchmarkType: "eval",
            sampleCount: nil, rawJson: ["parameter_eval_id": "ifbench"], definition: nil)
        XCTAssertEqual(eval.evalId, .known(.ifbench))

        // An eval id the client doesn't recognize is surfaced verbatim.
        let unknown = BenchmarkItem(
            benchmarkId: "eval_future", benchmarkType: "eval",
            sampleCount: nil, rawJson: ["parameter_eval_id": "some_future_eval"], definition: nil)
        XCTAssertEqual(unknown.evalId, .unknown("some_future_eval"))

        // Non-eval benchmarks carry no eval id.
        let prefill = BenchmarkItem(
            benchmarkId: "prefill_throughput_512", benchmarkType: "prefill_throughput",
            sampleCount: nil, rawJson: ["parameter_prefill_tokens": 512], definition: nil)
        XCTAssertNil(prefill.evalId)
    }
}
