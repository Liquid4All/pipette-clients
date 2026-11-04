import Testing

@testable import Pipette

/// `BenchmarkType` as mgmt plan-types' `benchmark/mod.rs` defines it: the closed set, its
/// two parses, and `fromId`, the one id-to-type rule this client has.
struct BenchmarkTypeTests {
    /// The crate's variant list in the crate's order: `StandardBenchmarks` emits in it, and
    /// a type missing here is a spelling that fails to decode on this side alone.
    @Test func declaresTheCratesVariantsInItsOrder() {
        #expect(BenchmarkType.allCases.map(\.rawValue) == [
            "prefill_throughput", "decode_throughput", "end_to_end_latency",
            "max_memory_usage", "eval", "vl_throughput", "vl_max_memory", "vl_spec_decode",
            "asr_throughput", "asr_max_memory",
        ])
    }

    /// `from_str_round_trips_every_variant`.
    @Test func parsingRoundTripsEveryVariant() throws {
        for type in BenchmarkType.allCases {
            #expect(try BenchmarkType(parsing: type.rawValue) == type)
        }
    }

    /// `from_str_accepts_snake_and_kebab`: the kebab spelling the `--type` filter
    /// advertises parses too, and an unknown one is refused by name.
    @Test func parsingAcceptsSnakeAndKebab() throws {
        #expect(try BenchmarkType(parsing: "prefill_throughput") == .prefillThroughput)
        #expect(try BenchmarkType(parsing: "prefill-throughput") == .prefillThroughput)
        #expect(throws: BenchmarkTypeError("mystery_metric")) {
            try BenchmarkType(parsing: "mystery_metric")
        }
    }

    /// `from_id_tolerates_unknown_as_none`: a type the client does not implement is
    /// tolerated, not an error, so a listing can carry it.
    @Test func fromIdToleratesUnknownAsNil() {
        #expect(BenchmarkType.fromId("prefill_throughput_256") == .prefillThroughput)
        #expect(BenchmarkType.fromId("remote/eval_mmlu") == .eval)
        #expect(BenchmarkType.fromId("mystery_metric_1") == nil)
    }

    /// `from_id_maps_every_benchmark_family`: `vl_max_memory` must not be swallowed by
    /// the `vl_throughput` arm, and a catalog side is stripped before matching.
    @Test(arguments: [
        ("prefill_throughput_256", .prefillThroughput),
        ("decode_throughput_512_100", .decodeThroughput),
        ("end_to_end_latency_8192_256", .endToEndLatency),
        ("max_memory_usage_4096", .maxMemoryUsage),
        ("eval_ifbench_2026.06.1", .eval),
        ("vl_max_memory_256x256_text0_img1", .vlMaxMemory),
        ("vl_throughput_512x512_text0_decode64_img1", .vlThroughput),
        ("local/vl_max_memory_512x512_text1024_img1", .vlMaxMemory),
        ("vl_spec_decode_mmspec_test_gqa_k8", .vlSpecDecode),
        ("remote/vl_spec_decode_mmspec_test_gqa_k8", .vlSpecDecode),
        ("asr_throughput_15000ms_text1024_decode256", .asrThroughput),
        ("remote/asr_max_memory_15000ms_text0_decode256", .asrMaxMemory),
        // No iOS engine: unknown here rather than misfiled under the ASR timing kinds.
        ("asr_eval_fleurs_nl_nl", nil),
        ("bogus_id", nil),
    ] as [(String, BenchmarkType?)])
    func fromIdMapsEveryBenchmarkFamily(id: String, expected: BenchmarkType?) {
        #expect(BenchmarkType.fromId(id) == expected)
    }

    /// A word that merely starts with the letters is not one.
    @Test func evalIsTheBareTypeOrAnUnderscoredId() {
        #expect(BenchmarkType.fromId("eval") == .eval)
        #expect(BenchmarkType.fromId("eval_smoke") == .eval)
        #expect(BenchmarkType.fromId("evaluation") == nil)
    }

    /// The memory kinds do not gate and load fresh; the vision kinds need a projector.
    @Test func theVisionMemoryKindIsAMemoryKindAndAVisionKind() {
        #expect(!BenchmarkType.vlMaxMemory.gatesOnReadiness)
        #expect(BenchmarkType.vlMaxMemory.requiresFreshLoad)
        #expect(BenchmarkType.vlMaxMemory.isVisionLanguage)
        #expect(BenchmarkType.vlThroughput.gatesOnReadiness)
        #expect(!BenchmarkType.vlThroughput.requiresFreshLoad)
    }

    @Test func theAsrKindsAreAudioKinds() {
        #expect(BenchmarkType.asrThroughput.gatesOnReadiness)
        #expect(!BenchmarkType.asrThroughput.requiresFreshLoad)
        #expect(!BenchmarkType.asrMaxMemory.gatesOnReadiness)
        #expect(BenchmarkType.asrMaxMemory.requiresFreshLoad)
        for kind in [BenchmarkType.asrThroughput, .asrMaxMemory] {
            #expect(kind.isAudio)
            #expect(kind.needsProjector)
            #expect(!kind.isVisionLanguage)
            #expect(!kind.needsCorpus)
        }
        #expect(BenchmarkType.vlThroughput.needsProjector)
        #expect(!BenchmarkType.vlThroughput.isAudio)
        #expect(!BenchmarkType.decodeThroughput.needsProjector)
    }

    /// The corpus benchmark gates, loads fresh, needs a projector and a corpus. Whether its
    /// drafter is on is the run's `speculative_enabled`, not the kind's.
    @Test func theCorpusKindGatesLoadsFreshAndNeedsACorpus() {
        let kind = BenchmarkType.vlSpecDecode
        #expect(kind.gatesOnReadiness)
        #expect(kind.requiresFreshLoad)
        #expect(kind.isVisionLanguage)
        #expect(kind.needsCorpus)
    }

    // MARK: - Completeness against the crate
    //
    // The axis must spell every variant plan-types does, so an unrecognized value fails to
    // decode here exactly where it fails there, and a *recognized* one this client cannot
    // run is refused by variant instead of dying as an unreadable payload.

    @Test(arguments: ["prefill_throughput", "decode_throughput", "end_to_end_latency",
                      "max_memory_usage", "eval", "vl_throughput", "vl_max_memory",
                      "vl_spec_decode", "asr_throughput", "asr_max_memory"])
    func benchmarkTypeSpellsEveryCrateVariant(raw: String) {
        #expect(BenchmarkType(rawValue: raw) != nil)
    }

    /// A spelling neither side knows still fails, which is what makes the above a
    /// completeness claim rather than a permissive decoder.
    @Test func anInventedBenchmarkTypeStillFails() {
        #expect(BenchmarkType(rawValue: "mystery_metric") == nil)
    }
}
