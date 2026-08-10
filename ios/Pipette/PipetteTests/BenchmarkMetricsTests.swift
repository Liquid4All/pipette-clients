import Testing
import Foundation
@testable import Pipette

struct BenchmarkMetricsTests {
    private struct Case {
        let label: String
        let type: BenchmarkType
        let payload: [String: Any]
        let params: [String: Any]
        let expected: (name: String, unit: String, value: Double, higherIsBetter: Bool)?
    }

    @Test func computeCoversEveryBenchmarkType() throws {
        let cases: [Case] = [
            Case(label: "prefill throughput", type: .prefillThroughput,
                 payload: ["prefill_time_ms": 250], params: ["parameter_prefill_tokens": 512],
                 expected: ("Prefill throughput", "tok/s", 2048, true)),
            Case(label: "decode throughput from string ms", type: .decodeThroughput,
                 payload: ["decode_time_ms": "500"], params: ["parameter_decode_tokens": 100],
                 expected: ("Decode throughput", "tok/s", 200, true)),
            Case(label: "end-to-end latency", type: .endToEndLatency,
                 payload: ["total_time_ms": 99.4], params: [:],
                 expected: ("E2E latency", "ms", 99.4, false)),
            Case(label: "max memory usage", type: .maxMemoryUsage,
                 payload: ["max_ram_bytes": 1_073_741_824], params: [:],
                 expected: ("Max memory", "bytes", 1_073_741_824, false)),
            Case(label: "vl throughput with payload prompt tokens", type: .vlThroughput,
                 payload: ["prompt_ms": 250, "predicted_ms": 750, "prompt_tokens": 72],
                 params: ["parameter_decode_tokens": 128],
                 expected: ("VL throughput", "tok/s", 200, true)),
            Case(label: "vl throughput falls back to catalog text tokens", type: .vlThroughput,
                 payload: ["prompt_ms": 250, "predicted_ms": 750],
                 params: ["parameter_text_tokens": 72, "parameter_decode_tokens": 128],
                 expected: ("VL throughput", "tok/s", 200, true)),
            // Reconciled semantics: a missing token count yields zero throughput,
            // not the cell detail view's former raw-milliseconds fallback.
            Case(label: "prefill with absent token count is zero throughput", type: .prefillThroughput,
                 payload: ["prefill_time_ms": 250], params: [:],
                 expected: ("Prefill throughput", "tok/s", 0, true)),
            Case(label: "eval has no single-cell metric", type: .eval,
                 payload: ["value": 1], params: [:], expected: nil),
            Case(label: "prefill with zero time", type: .prefillThroughput,
                 payload: ["prefill_time_ms": 0], params: ["parameter_prefill_tokens": 512], expected: nil),
            Case(label: "decode with zero time", type: .decodeThroughput,
                 payload: ["decode_time_ms": 0], params: ["parameter_decode_tokens": 100], expected: nil),
            Case(label: "latency without total time", type: .endToEndLatency,
                 payload: [:], params: [:], expected: nil),
            Case(label: "vl with zero elapsed time", type: .vlThroughput,
                 payload: ["prompt_ms": 0, "predicted_ms": 0], params: [:], expected: nil)
        ]

        for testCase in cases {
            let metric = try BenchmarkMetrics.compute(
                payload: testCase.payload, params: testCase.params, type: testCase.type
            )
            guard let expected = testCase.expected else {
                #expect(metric == nil, "\(testCase.label) should yield no metric")
                continue
            }
            let actual = try #require(metric, "\(testCase.label) should yield a metric")
            #expect(actual.name == expected.name, "\(testCase.label) name")
            #expect(actual.unit == expected.unit, "\(testCase.label) unit")
            #expect(abs(actual.numericValue - expected.value) <= 0.001, "\(testCase.label) value")
            #expect(actual.higherIsBetter == expected.higherIsBetter, "\(testCase.label) direction")
        }
    }

    @Test func computeThrowsWhenRequiredFieldIsPresentButNotNumeric() {
        #expect(throws: PayloadError.self) {
            try BenchmarkMetrics.compute(
                payload: ["decode_time_ms": "not-a-number"],
                params: ["parameter_decode_tokens": 100],
                type: .decodeThroughput
            )
        }
        #expect(throws: PayloadError.self) {
            try BenchmarkMetrics.compute(
                payload: ["max_ram_bytes": ["nested": 1]],
                params: [:],
                type: .maxMemoryUsage
            )
        }
    }

    @Test func scalarStringRendersLocaleStableValues() {
        #expect(PayloadScalars.string(1.6) == "1.6")
        #expect(PayloadScalars.string(2048.0) == "2048")
        #expect(PayloadScalars.string(true) == "true")
        #expect(PayloadScalars.string("q4_km") == "q4_km")
        #expect(PayloadScalars.string(NSNull()) == nil)
        #expect(PayloadScalars.string(["nested": 1]) == nil)
    }

    @Test func readObjectDistinguishesAbsentFromCorrupt() throws {
        let dir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("BenchmarkMetricsTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        // Absent file -> nil, not an error.
        let absent = dir.appendingPathComponent("absent.json")
        #expect(try ResultPayload.readObject(at: absent) == nil)

        // Valid JSON object -> parsed dictionary.
        let valid = dir.appendingPathComponent("valid.json")
        try Data(#"{"decode_time_ms":500}"#.utf8).write(to: valid)
        let parsed = try #require(try ResultPayload.readObject(at: valid))
        #expect(PayloadScalars.string(parsed["decode_time_ms"]) == "500")

        // Present but corrupt -> throws.
        let corrupt = dir.appendingPathComponent("corrupt.json")
        try Data("{not json".utf8).write(to: corrupt)
        #expect(throws: PayloadError.self) { try ResultPayload.readObject(at: corrupt) }

        // Valid JSON that isn't an object -> throws.
        let array = dir.appendingPathComponent("array.json")
        try Data("[1,2,3]".utf8).write(to: array)
        #expect(throws: PayloadError.self) { try ResultPayload.readObject(at: array) }
    }
}
