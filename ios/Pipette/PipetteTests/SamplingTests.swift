import Foundation
import Testing

@testable import Pipette

/// The sampling an entry may pin (mgmt's `pipette-plan-types` `benchmark/sampling.rs` and
/// `benchmark/mod.rs`), and the `Sampling` an eval cell resolves from it.
struct SamplingTests {
    private static let evalWire: [String: Any] = [
        "benchmark_id": "eval_math_500_2026.06.1",
        "benchmark_type": "eval",
        "parameter_eval_id": "math_500",
        "parameter_dataset_name": "2026.06.1",
        "parameter_max_tokens": 8192,
    ]

    private func decode(_ wire: [String: Any]) throws -> BenchmarkDefinition {
        try JSONDecoder().decode(BenchmarkDefinition.self,
                                 from: JSONSerialization.data(withJSONObject: wire))
    }

    private func eval(_ pins: [String: Any] = [:]) throws -> EvalBenchmark {
        try decode(Self.evalWire.merging(pins) { $1 }).asEval()
    }

    /// The eval body with `key` set to `json`, a JSON value spelled as text.
    private func evalWire(_ key: String, _ json: String) throws -> [String: Any] {
        let value = try JSONSerialization.jsonObject(with: Data(json.utf8),
                                                     options: .fragmentsAllowed)
        return Self.evalWire.merging([key: value]) { $1 }
    }

    private func encoded(_ definition: BenchmarkDefinition) throws -> [String: Any] {
        try #require(try JSONSerialization.jsonObject(
            with: JSONEncoder().encode(definition)) as? [String: Any])
    }

    @Test func evalSamplingIsAbsentUntilPinned() throws {
        let unpinned = try eval()
        #expect(unpinned.parameterTemperature == nil && unpinned.parameterTopP == nil
            && unpinned.parameterTopK == nil && unpinned.parameterMinP == nil
            && unpinned.parameterSeed == nil)
        let pinned = try eval([
            "parameter_temperature": 1.0, "parameter_top_p": 0.95, "parameter_top_k": 64,
            "parameter_min_p": 0.05, "parameter_seed": 7,
        ])
        #expect(pinned.parameterTemperature == (try SamplingTemperature(1.0)))
        #expect(pinned.parameterTopP == (try TopP(0.95)))
        #expect(pinned.parameterTopK == (try TopK(64)))
        #expect(pinned.parameterMinP == (try MinP(0.05)))
        #expect(pinned.parameterSeed == (try SamplingSeed(7)))
    }

    /// `eval_sampling_out_of_range_is_refused`.
    @Test(arguments: [
        ("parameter_temperature", "2.1"), ("parameter_top_p", "1.5"), ("parameter_min_p", "1.5"),
        ("parameter_top_k", "-1"), ("parameter_seed", "4294967296"),
    ])
    func evalSamplingOutOfRangeIsRefused(key: String, json: String) throws {
        let wire = try evalWire(key, json)
        #expect(throws: (any Error).self, "\(key)=\(json)") { try decode(wire) }
    }

    /// `eval_sampling_edges_land_as_values`.
    @Test(arguments: [
        ("parameter_temperature", "0.0"), ("parameter_temperature", "2.0"),
        ("parameter_top_p", "1.0"), ("parameter_min_p", "0.0"), ("parameter_top_k", "0"),
        ("parameter_seed", "0"),
    ])
    func evalSamplingEdgesLandAsValues(key: String, json: String) throws {
        let wire = try evalWire(key, json)
        #expect((try encoded(try decode(wire))[key] as? NSNumber) == (wire[key] as? NSNumber),
                "\(key)")
    }

    /// `negative_zero_reads_as_zero`.
    @Test func negativeZeroReadsAsZero() throws {
        try expectPositiveZero(SamplingTemperature.self)
        try expectPositiveZero(TopP.self)
        try expectPositiveZero(MinP.self)
    }

    /// `init(_:)` and the decoder both store a `-0.0` as `0.0`, which encodes as `0`.
    private func expectPositiveZero<K: SamplingKnob>(_: K.Type) throws where K.Raw == Double {
        let legs = [("init", try K(-0.0)),
                    ("decode", try JSONDecoder().decode(K.self, from: Data("-0.0".utf8)))]
        for (leg, knob) in legs {
            #expect(knob.value == 0 && knob.value.sign == .plus, "\(K.field) \(leg) \(knob.value)")
            #expect(String(decoding: try JSONEncoder().encode(knob), as: UTF8.self) == "0",
                    "\(K.field) \(leg)")
        }
    }

    /// A pinned `0.0` is greedy, not an absence.
    @Test(arguments: [(0.0 as Double?, 0.0), (1.0, 1.0), (nil, 0.6)])
    func aPinnedTemperatureWinsOverTheEvalIdPolicy(pinned: Double?, expected: Double) throws {
        let pins: [String: Any] = pinned.map { ["parameter_temperature": $0] } ?? [:]
        let parsed = try decode(Self.evalWire.merging(pins) { $1 })
        #expect(try parsed.evalTemperature().value == expected)
    }

    /// `a_samples_seed_offsets_by_its_repeat_index`.
    @Test(arguments: [
        ("row-3#0", UInt32(7)), ("row-3#2", 9), ("row-3", 7), ("row-3#draft", 7), ("row-3#-5", 7),
    ])
    func aSamplesSeedOffsetsByItsRepeatIndex(sampleId: String, expected: UInt32) throws {
        #expect(try eval(["parameter_seed": 7]).sampleSeed(sampleId) == expected)
    }

    @Test func aDerivedSeedWrapsInsideTheCap() throws {
        let benchmark = try eval(["parameter_seed": 4_294_967_295 as Int64])
        #expect(benchmark.sampleSeed("row-1#0") == 4_294_967_295)
        #expect(benchmark.sampleSeed("row-1#1") == 0)
        #expect(benchmark.sampleSeed("row-1#2") == 1)
        #expect(try eval().sampleSeed("row-1#1") == nil)
    }

    @Test func unpinnedSamplingIsElidedFromTheWire() throws {
        let wire = try encoded(try decode(Self.evalWire))
        for key in ["parameter_temperature", "parameter_top_p", "parameter_top_k",
                    "parameter_min_p", "parameter_seed"] {
            #expect(wire[key] == nil, "\(key) must be absent")
        }
    }

    /// The value is always written.
    @Test(arguments: [(nil as Double?, 0.0), (0.0, 0.0), (0.8, 0.8), (2.0, 2.0)])
    func vlSpecDecodeTemperatureDefaultsToGreedy(pinned: Double?, expected: Double) throws {
        var wire = try encoded(.vlSpecDecode(VlSpecDecodeBenchmark(
            benchmarkId: BenchmarkId(validated: "c"),
            parameterDraftLength: 8,
            parameterMaxTokens: 2048,
            dataset: BenchmarkDefinitionTests.sampleDataset)))
        wire["parameter_temperature"] = pinned
        let parsed = try decode(wire)
        #expect(try parsed.asVlSpecDecode().parameterTemperature.value == expected)
        #expect(try encoded(parsed)["parameter_temperature"] as? Double == expected)
    }

    /// A `null` temperature is not an absent one: the crate's field is a plain value.
    @Test func aNullVlSpecDecodeTemperatureIsRefused() throws {
        var wire = try encoded(.vlSpecDecode(VlSpecDecodeBenchmark(
            benchmarkId: BenchmarkId(validated: "c"),
            parameterDraftLength: 8,
            parameterMaxTokens: 2048,
            dataset: BenchmarkDefinitionTests.sampleDataset)))
        wire["parameter_temperature"] = NSNull()
        #expect(throws: DecodingError.self) { try decode(wire) }
    }
}
