import XCTest

@testable import Pipette

/// `BenchmarkDefinition` must decode the server/wire shape — the same flat
/// `benchmark_type` + `parameter_*` JSON the Rust `BenchmarkDefinition` parses —
/// into the typed variant, preserve an unknown eval id, and reject an unknown type.
final class BenchmarkDefinitionTests: XCTestCase {
    private func decode(_ json: String) throws -> BenchmarkDefinition {
        try JSONDecoder().decode(BenchmarkDefinition.self, from: Data(json.utf8))
    }

    func testDecodesThroughputAndMemoryVariants() throws {
        XCTAssertEqual(
            try decode(#"{"benchmark_id":"p","benchmark_type":"prefill_throughput","parameter_prefill_tokens":512}"#),
            .prefillThroughput(benchmarkId: "p", prefillTokens: 512))
        XCTAssertEqual(
            try decode(#"{"benchmark_id":"d","benchmark_type":"decode_throughput","parameter_prefill_tokens":256,"parameter_decode_tokens":100}"#),
            .decodeThroughput(benchmarkId: "d", prefillTokens: 256, decodeTokens: 100))
        XCTAssertEqual(
            try decode(#"{"benchmark_id":"m","benchmark_type":"max_memory_usage","parameter_prefill_tokens":1024}"#),
            .maxMemoryUsage(benchmarkId: "m", prefillTokens: 1024))
    }

    func testDecodesEvalWithTypedIdAndSamples() throws {
        let json = #"""
            {"benchmark_id":"eval_ifbench","benchmark_type":"eval","parameter_eval_id":"ifbench",
             "parameter_dataset_name":"release_v1_0","parameter_max_tokens":256,
             "samples":[{"id":"q1","messages":[{"role":"user","content":"hi"}]}]}
            """#
        guard case let .eval(id, evalId, dataset, maxTokens, _, samples) = try decode(json) else {
            return XCTFail("expected .eval")
        }
        XCTAssertEqual(id, "eval_ifbench")
        XCTAssertEqual(evalId, .known(.ifbench))
        XCTAssertEqual(dataset, "release_v1_0")
        XCTAssertEqual(maxTokens, 256)
        XCTAssertEqual(samples?.first, EvalSample(id: "q1", messages: [["role": "user", "content": "hi"]]))
    }

    func testEvalPreservesUnknownEvalId() throws {
        guard
            case let .eval(_, evalId, _, _, _, _) = try decode(
                #"{"benchmark_id":"e","benchmark_type":"eval","parameter_eval_id":"future_eval","parameter_dataset_name":"d","parameter_max_tokens":8}"#)
        else { return XCTFail("expected .eval") }
        XCTAssertEqual(evalId, .unknown("future_eval"))
    }

    func testRejectsUnknownBenchmarkType() {
        XCTAssertThrowsError(try decode(#"{"benchmark_id":"x","benchmark_type":"mystery_metric"}"#))
    }
}
