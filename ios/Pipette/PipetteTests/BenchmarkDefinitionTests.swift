import Foundation
import Testing

@testable import Pipette

/// The flat `benchmark_type` + `parameter_*` document the mgmt crate's `BenchmarkDefinition`
/// parses (pipette-plan-types `benchmark/mod.rs`), decoded into the typed variant and back.
struct BenchmarkDefinitionTests {
    private func decode(_ json: String) throws -> BenchmarkDefinition {
        try JSONDecoder().decode(BenchmarkDefinition.self, from: Data(json.utf8))
    }

    private func object(_ definition: BenchmarkDefinition) throws -> [String: Any] {
        try #require(try JSONSerialization.jsonObject(
            with: JSONEncoder().encode(definition)) as? [String: Any])
    }

    @Test func decodesThroughputAndMemoryVariants() throws {
        #expect(try decode(#"{"benchmark_id":"p","benchmark_type":"prefill_throughput","parameter_prefill_tokens":512}"#)
            == .prefillThroughput(PrefillThroughputBenchmark(
                benchmarkId: BenchmarkId(validated: "p"), parameterPrefillTokens: 512)))
        #expect(try decode(#"{"benchmark_id":"d","benchmark_type":"decode_throughput","parameter_prefill_tokens":256,"parameter_decode_tokens":100}"#)
            == .decodeThroughput(DecodeThroughputBenchmark(
                benchmarkId: BenchmarkId(validated: "d"), parameterPrefillTokens: 256,
                parameterDecodeTokens: 100)))
        #expect(try decode(#"{"benchmark_id":"m","benchmark_type":"max_memory_usage","parameter_prefill_tokens":1024}"#)
            == .maxMemoryUsage(MaxMemoryUsageBenchmark(
                benchmarkId: BenchmarkId(validated: "m"), parameterPrefillTokens: 1024)))
    }

    @Test func decodesEvalWithTypedIdAndSamples() throws {
        let json = #"""
            {"benchmark_id":"eval_ifbench","benchmark_type":"eval","parameter_eval_id":"ifbench",
             "parameter_dataset_name":"release_v1_0","parameter_max_tokens":256,
             "samples":[{"id":"q1","messages":[{"role":"user","content":"hi"}]}]}
            """#
        let eval = try decode(json).asEval()
        #expect(eval.benchmarkId.value == "eval_ifbench")
        #expect(eval.parameterEvalId == .known(.ifbench))
        #expect(eval.parameterDatasetName == "release_v1_0")
        #expect(eval.parameterMaxTokens == 256)
        #expect(eval.samples?.first
            == EvalSample(id: "q1", messages: [["role": "user", "content": "hi"]]))
    }

    @Test func evalPreservesAnUnknownEvalId() throws {
        let eval = try decode(
            #"{"benchmark_id":"e","benchmark_type":"eval","parameter_eval_id":"future_eval","parameter_dataset_name":"d","parameter_max_tokens":8}"#)
            .asEval()
        #expect(eval.parameterEvalId == .unknown("future_eval"))
    }

    @Test func anUnknownBenchmarkTypeStillThrows() {
        #expect(throws: (any Error).self) {
            try decode(#"{"benchmark_id":"x","benchmark_type":"mystery_metric"}"#)
        }
    }

    /// Reading a missing batch size as anything but one would file a different measurement
    /// under the entry's id.
    @Test func aTimingBodyWithoutABatchSizeMeansOneInFlight() throws {
        #expect(try decode(#"{"benchmark_id":"p","benchmark_type":"prefill_throughput","parameter_prefill_tokens":512}"#)
            .batchSize == .one)
        #expect(try decode(#"{"benchmark_id":"d","benchmark_type":"decode_throughput","parameter_prefill_tokens":256,"parameter_decode_tokens":100}"#)
            .batchSize == .one)
        #expect(try decode(#"{"benchmark_id":"e","benchmark_type":"end_to_end_latency","parameter_prefill_tokens":512,"parameter_decode_tokens":256}"#)
            .batchSize == .one)
        #expect(try decode(#"{"benchmark_id":"e_bs4","benchmark_type":"end_to_end_latency","parameter_prefill_tokens":512,"parameter_decode_tokens":256,"parameter_batch_size":4}"#)
            == .endToEndLatency(EndToEndLatencyBenchmark(
                benchmarkId: BenchmarkId(validated: "e_bs4"),
                parameterPrefillTokens: 512,
                parameterDecodeTokens: 256,
                parameterBatchSize: BatchSize(validated: 4))))
        // Kinds with no batch answer nil rather than a guessed one.
        #expect(try decode(#"{"benchmark_id":"m","benchmark_type":"max_memory_usage","parameter_prefill_tokens":1024}"#)
            .batchSize == nil)
    }

    /// An accessor for another kind names both.
    @Test func asEvalErrorsOnPrefillDefinition() throws {
        let definition = BenchmarkDefinition.prefillThroughput(PrefillThroughputBenchmark(
            benchmarkId: BenchmarkId(validated: "prefill_throughput_512"),
            parameterPrefillTokens: 512))
        let error = #expect(throws: BenchmarkDefinitionError.self) { try definition.asEval() }
        #expect(error?.errorDescription
            == "expected eval benchmark, got `prefill_throughput_512` (prefill_throughput)")
    }

    /// The column is Int32, so a batch past it is refused at the wire.
    @Test(arguments: [("2147483647", true), ("2147483648", false)])
    func aBatchSizeFitsItsWarehouseColumn(json: String, ok: Bool) {
        #expect(((try? JSONDecoder().decode(BatchSize.self, from: Data(json.utf8))) != nil) == ok)
    }

    @Test func aZeroBatchSizeIsRefusedAtTheWire() {
        #expect(throws: DecodingError.self) {
            try decode(#"{"benchmark_id":"end_to_end_latency_512_256_bs0","benchmark_type":"end_to_end_latency","parameter_prefill_tokens":512,"parameter_decode_tokens":256,"parameter_batch_size":0}"#)
        }
    }

    @Test func everyKindRoundTrips() throws {
        let definitions: [BenchmarkDefinition] = [
            .prefillThroughput(PrefillThroughputBenchmark(
                benchmarkId: BenchmarkId(validated: "prefill_throughput_512"),
                parameterPrefillTokens: 512)),
            .prefillThroughput(PrefillThroughputBenchmark(
                benchmarkId: BenchmarkId(validated: "prefill_throughput_512_bs4"),
                parameterPrefillTokens: 512,
                parameterBatchSize: BatchSize(validated: 4))),
            .decodeThroughput(DecodeThroughputBenchmark(
                benchmarkId: BenchmarkId(validated: "decode_throughput_512_100"),
                parameterPrefillTokens: 512,
                parameterDecodeTokens: 100)),
            .decodeThroughput(DecodeThroughputBenchmark(
                benchmarkId: BenchmarkId(validated: "decode_throughput_512_100_bs2"),
                parameterPrefillTokens: 512,
                parameterDecodeTokens: 100,
                parameterBatchSize: BatchSize(validated: 2))),
            .endToEndLatency(EndToEndLatencyBenchmark(
                benchmarkId: BenchmarkId(validated: "end_to_end_latency_512_256"),
                parameterPrefillTokens: 512,
                parameterDecodeTokens: 256)),
            .endToEndLatency(EndToEndLatencyBenchmark(
                benchmarkId: BenchmarkId(validated: "end_to_end_latency_512_256_bs4"),
                parameterPrefillTokens: 512,
                parameterDecodeTokens: 256,
                parameterBatchSize: BatchSize(validated: 4))),
            .maxMemoryUsage(MaxMemoryUsageBenchmark(
                benchmarkId: BenchmarkId(validated: "max_memory_usage_512"),
                parameterPrefillTokens: 512)),
            .eval(EvalBenchmark(
                benchmarkId: BenchmarkId(validated: "eval_mmlu_pro_edge_2026.03.1"),
                parameterEvalId: EvalId("mmlu_pro"),
                parameterDatasetName: "edge_2026.03.1",
                parameterMaxTokens: 256,
                parameterMcqChoices: ["A", "B"],
                samples: [EvalSample(id: "s", messages: [["role": "user", "content": "x"]])])),
            .vlThroughput(VlThroughputBenchmark(
                benchmarkId: BenchmarkId(validated: "vl_throughput_224x224_8_64"),
                parameterImageWidth: 224,
                parameterImageHeight: 224,
                parameterTextTokens: 8,
                parameterDecodeTokens: 64,
                parameterNumImages: 1)),
            .vlMaxMemory(VlMaxMemoryBenchmark(
                benchmarkId: BenchmarkId(validated: "vl_max_memory_224x224_t0_f1"),
                parameterImageWidth: 224,
                parameterImageHeight: 224,
                parameterTextTokens: 0,
                parameterNumImages: 1)),
            .vlMaxMemory(VlMaxMemoryBenchmark(
                benchmarkId: BenchmarkId(validated: "vl_max_memory_256x256_text1024_decode256_img5"),
                parameterImageWidth: 256,
                parameterImageHeight: 256,
                parameterTextTokens: 1024,
                parameterDecodeTokens: 256,
                parameterNumImages: 5)),
            .eval(EvalBenchmark(
                benchmarkId: BenchmarkId(validated: "eval_math_500_2026.06.1_t1"),
                parameterEvalId: EvalId("math_500"),
                parameterDatasetName: "2026.06.1",
                parameterMaxTokens: 8192,
                parameterMcqChoices: nil,
                parameterTemperature: try SamplingTemperature(1),
                parameterTopP: try TopP(0.95),
                parameterTopK: try TopK(64),
                parameterMinP: try MinP(0.05),
                parameterSeed: try SamplingSeed(7),
                samples: nil)),
            .vlSpecDecode(VlSpecDecodeBenchmark(
                benchmarkId: BenchmarkId(validated: "vl_spec_decode_mmspec_test_gqa_k8_bs8"),
                parameterDraftLength: 8,
                parameterMaxTokens: 2048,
                parameterBatchSize: BatchSize(validated: 8),
                dataset: Self.sampleDataset)),
            .vlSpecDecode(VlSpecDecodeBenchmark(
                benchmarkId: BenchmarkId(validated: "vl_spec_decode_mmspec_test_gqa_k8_t0.8"),
                parameterDraftLength: 8,
                parameterMaxTokens: 2048,
                parameterTemperature: try SamplingTemperature(0.8),
                dataset: Self.sampleDataset)),
            .asrThroughput(Self.asrThroughput("asr_throughput_15000ms_text1024_decode256")),
            .asrMaxMemory(Self.asrMaxMemory("asr_max_memory_15000ms_text0_decode256")),
        ]
        for definition in definitions {
            let restored = try JSONDecoder().decode(
                BenchmarkDefinition.self, from: JSONEncoder().encode(definition))
            #expect(restored == definition, "\(definition.benchmarkId)")
        }
    }

    @Test func serializesToAFlatDocument() throws {
        let json = try object(.eval(EvalBenchmark(
            benchmarkId: BenchmarkId(validated: "eval_mmlu_pro_edge_2026.03.1"),
            parameterEvalId: EvalId("mmlu_pro"),
            parameterDatasetName: "edge_2026.03.1",
            parameterMaxTokens: 256,
            parameterMcqChoices: ["A", "B"],
            samples: nil)))

        #expect(json["benchmark_id"] as? String == "eval_mmlu_pro_edge_2026.03.1")
        #expect(json["benchmark_type"] as? String == "eval")
        #expect(json["parameter_eval_id"] as? String == "mmlu_pro")
        #expect(json["parameter_dataset_name"] as? String == "edge_2026.03.1")
        #expect(json["parameter_max_tokens"] as? Int == 256)
        #expect(json["parameter_mcq_choices"] as? [String] == ["A", "B"])
        // Absent, not null, as `skip_serializing_if` leaves it.
        #expect(!json.keys.contains("samples"))
    }

    @Test func evalTemperatureNonEvalIsGreedy() throws {
        let prefill = BenchmarkDefinition.prefillThroughput(PrefillThroughputBenchmark(
            benchmarkId: BenchmarkId(validated: "prefill_throughput_512"),
            parameterPrefillTokens: 512))
        #expect(try prefill.evalTemperature() == .greedy)
    }

    /// Matches the crate's `default_num_images`.
    @Test func aVisionThroughputBodyWithoutAnImageCountMeansOne() throws {
        let definition = try decode(#"{"benchmark_id":"v","benchmark_type":"vl_throughput","parameter_image_width":224,"parameter_image_height":224,"parameter_text_tokens":8,"parameter_decode_tokens":64}"#)

        #expect(definition == .vlThroughput(VlThroughputBenchmark(
            benchmarkId: BenchmarkId(validated: "v"),
            parameterImageWidth: 224,
            parameterImageHeight: 224,
            parameterTextTokens: 8,
            parameterDecodeTokens: 64,
            parameterNumImages: 1)))
    }

    /// A missing decode count stays none: the one-token cell.
    @Test func aVisionMemoryBodyDefaultsItsTextAndImageCount() throws {
        let definition = try decode(#"{"benchmark_id":"m","benchmark_type":"vl_max_memory","parameter_image_width":256,"parameter_image_height":256}"#)

        #expect(definition == .vlMaxMemory(VlMaxMemoryBenchmark(
            benchmarkId: BenchmarkId(validated: "m"),
            parameterImageWidth: 256,
            parameterImageHeight: 256,
            parameterTextTokens: 0,
            parameterNumImages: 1)))
        #expect(definition.benchmarkType == .vlMaxMemory)
    }

    /// The clip and decode count are required.
    @Test func anAsrBodyDefaultsItsTextAndWritesItBack() throws {
        let definition = try decode(#"{"benchmark_id":"a","benchmark_type":"asr_max_memory","parameter_audio_duration_ms":15000,"parameter_audio_sample_rate_hz":16000,"parameter_audio_channels":1,"parameter_decode_tokens":256}"#)

        #expect(definition == .asrMaxMemory(Self.asrMaxMemory("a")))
        #expect(try object(definition)["parameter_text_tokens"] as? Int == 0)
        #expect(definition.parameterFields["parameter_audio_duration_ms"] as? Int == 15000)
        #expect(throws: DecodingError.self) {
            try decode(#"{"benchmark_id":"a","benchmark_type":"asr_throughput","parameter_audio_duration_ms":15000,"parameter_audio_sample_rate_hz":16000,"parameter_audio_channels":1}"#)
        }
    }

    @Test func anAsrBodyOutOfBoundsIsRefused() {
        #expect(throws: (any Error).self) {
            try decode(#"{"benchmark_id":"a","benchmark_type":"asr_throughput","parameter_audio_duration_ms":15000,"parameter_audio_sample_rate_hz":16000,"parameter_audio_channels":3,"parameter_decode_tokens":256}"#)
        }
        #expect(throws: (any Error).self) {
            try decode(#"{"benchmark_id":"a","benchmark_type":"asr_throughput","parameter_audio_duration_ms":15000,"parameter_audio_sample_rate_hz":16000,"parameter_audio_channels":1,"parameter_decode_tokens":1}"#)
        }
    }

    static func asrThroughput(_ id: String) -> AsrThroughputBenchmark {
        AsrThroughputBenchmark(
            benchmarkId: BenchmarkId(validated: id),
            parameterAudioDurationMs: AudioDurationMillis(validated: 15000),
            parameterAudioSampleRateHz: AudioSampleRate(validated: 16000),
            parameterAudioChannels: AudioChannels(validated: 1),
            parameterTextTokens: 1024,
            parameterDecodeTokens: AsrDecodeTokens(validated: 256))
    }

    static func asrMaxMemory(_ id: String) -> AsrMaxMemoryBenchmark {
        AsrMaxMemoryBenchmark(
            benchmarkId: BenchmarkId(validated: id),
            parameterAudioDurationMs: AudioDurationMillis(validated: 15000),
            parameterAudioSampleRateHz: AudioSampleRate(validated: 16000),
            parameterAudioChannels: AudioChannels(validated: 1),
            parameterDecodeTokens: AsrDecodeTokens(validated: 256))
    }

    /// The defaulted fields are written back, as the crate serializes them, so a local
    /// definition this client writes is the crate's document.
    @Test func defaultedFieldsAreWrittenBack() throws {
        let throughput = try object(.vlThroughput(VlThroughputBenchmark(
            benchmarkId: BenchmarkId(validated: "v"),
            parameterImageWidth: 224,
            parameterImageHeight: 224,
            parameterTextTokens: 8,
            parameterDecodeTokens: 64)))
        #expect(throughput["parameter_num_images"] as? Int == 1)

        let memory = try object(.vlMaxMemory(VlMaxMemoryBenchmark(
            benchmarkId: BenchmarkId(validated: "m"), parameterImageWidth: 224,
            parameterImageHeight: 224)))
        #expect(memory["parameter_text_tokens"] as? Int == 0)
        #expect(memory["parameter_num_images"] as? Int == 1)
        #expect(memory["parameter_decode_tokens"] == nil)

        let corpus = try object(.vlSpecDecode(VlSpecDecodeBenchmark(
            benchmarkId: BenchmarkId(validated: "c"),
            parameterDraftLength: 8,
            parameterMaxTokens: 2048,
            dataset: Self.sampleDataset)))
        #expect(corpus["parameter_batch_size"] as? Int == 1)

        // The timing cells write the batch size back for the same reason, and
        // `max_memory_usage`, which carries none, writes no key at all.
        let prefill = try object(.prefillThroughput(PrefillThroughputBenchmark(
            benchmarkId: BenchmarkId(validated: "p"), parameterPrefillTokens: 512)))
        #expect(prefill["parameter_batch_size"] as? Int == 1)
        let latency = try object(.endToEndLatency(EndToEndLatencyBenchmark(
            benchmarkId: BenchmarkId(validated: "e"),
            parameterPrefillTokens: 512,
            parameterDecodeTokens: 256,
            parameterBatchSize: BatchSize(validated: 4))))
        #expect(latency["parameter_batch_size"] as? Int == 4)
        let peak = try object(.maxMemoryUsage(MaxMemoryUsageBenchmark(
            benchmarkId: BenchmarkId(validated: "m"), parameterPrefillTokens: 512)))
        #expect(peak["parameter_batch_size"] == nil)
    }

    @Test func aCorpusBodyWithoutABatchSizeMeansOneInFlight() throws {
        let definition = try decode(#"{"benchmark_id":"vl_spec_decode_mmspec_test_gqa_k8","benchmark_type":"vl_spec_decode","parameter_draft_length":8,"parameter_max_tokens":2048,"dataset":{"format":"vl-spec-decode-corpus/1","adapter":"mmspec@1","digest":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","bytes":15955456,"subset":"GQA","upstream":{"source":"git","url":"github.com/killthefullmoon/MMSpec","revision":"290486861eabd746075ca776adb66175636c1fc0","subpath":"dataset/MMSpec/test","identity":"2923603292c7dd463ad01a119f57965817778fc3","digest":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","license":"Apache-2.0"},"url":"/datasets/vl_spec_decode_mmspec_test_gqa_k8"}}"#)

        #expect(definition == .vlSpecDecode(VlSpecDecodeBenchmark(
            benchmarkId: BenchmarkId(validated: "vl_spec_decode_mmspec_test_gqa_k8"),
            parameterDraftLength: 8,
            parameterMaxTokens: 2048,
            parameterBatchSize: BatchSize(validated: 1),
            dataset: Self.sampleDataset)))
        #expect(definition.parameterFields["parameter_batch_size"] as? Int == 1)
    }

    @Test func benchmarkTypeIsTheDecodedTag() throws {
        for type in BenchmarkType.allCases {
            let definition: BenchmarkDefinition = switch type {
            case .prefillThroughput: .prefillThroughput(PrefillThroughputBenchmark(
                benchmarkId: BenchmarkId(validated: "x"), parameterPrefillTokens: 1))
            case .decodeThroughput: .decodeThroughput(DecodeThroughputBenchmark(
                benchmarkId: BenchmarkId(validated: "x"), parameterPrefillTokens: 1,
                parameterDecodeTokens: 1))
            case .endToEndLatency: .endToEndLatency(EndToEndLatencyBenchmark(
                benchmarkId: BenchmarkId(validated: "x"), parameterPrefillTokens: 1,
                parameterDecodeTokens: 1))
            case .maxMemoryUsage: .maxMemoryUsage(MaxMemoryUsageBenchmark(
                benchmarkId: BenchmarkId(validated: "x"), parameterPrefillTokens: 1))
            case .eval: .eval(EvalBenchmark(
                benchmarkId: BenchmarkId(validated: "x"),
                parameterEvalId: EvalId("x"),
                parameterDatasetName: "d",
                parameterMaxTokens: 1,
                parameterMcqChoices: nil,
                samples: nil))
            case .vlThroughput: .vlThroughput(VlThroughputBenchmark(
                benchmarkId: BenchmarkId(validated: "x"),
                parameterImageWidth: 1,
                parameterImageHeight: 1,
                parameterTextTokens: 1,
                parameterDecodeTokens: 1))
            case .vlMaxMemory: .vlMaxMemory(VlMaxMemoryBenchmark(
                benchmarkId: BenchmarkId(validated: "x"), parameterImageWidth: 1,
                parameterImageHeight: 1))
            case .vlSpecDecode: .vlSpecDecode(VlSpecDecodeBenchmark(
                benchmarkId: BenchmarkId(validated: "x"),
                parameterDraftLength: 8,
                parameterMaxTokens: 256,
                dataset: Self.sampleDataset))
            case .asrThroughput: .asrThroughput(Self.asrThroughput("x"))
            case .asrMaxMemory: .asrMaxMemory(Self.asrMaxMemory("x"))
            }
            #expect(definition.benchmarkType == type)
            #expect(try object(definition)["benchmark_type"] as? String == type.rawValue)
        }
    }

    /// The corpus pin as `dataset.rs`'s own round-trip test spells it. A fixture rather
    /// than a builder: the fields are provenance, and a test that generated them would not
    /// notice one going missing from the wire.
    static let sampleDataset = Dataset(
        format: .vlSpecDecodeCorpusV1,
        adapter: .mmspecV1,
        digest: Sha256(validated: String(repeating: "a", count: 64)),
        bytes: 15_955_456,
        subset: "GQA",
        upstream: .git(
            url: DatasetRepositoryUrl(validated: "github.com/killthefullmoon/MMSpec"),
            revision: GitOid(validated: "290486861eabd746075ca776adb66175636c1fc0"),
            subpath: DatasetPath(validated: "dataset/MMSpec/test"),
            identity: GitOid(validated: "2923603292c7dd463ad01a119f57965817778fc3"),
            digest: Sha256(validated: String(repeating: "b", count: 64)),
            license: "Apache-2.0"),
        url: NonEmptyString(validated: "/datasets/vl_spec_decode_mmspec_test_gqa_k8"))

    /// The id parse reads the type through `BenchmarkType.fromId`, so it strips a catalog
    /// side the same way and names the bare id.
    @Test func parsingAnIdStripsTheCatalogSide() {
        #expect(BenchmarkDefinition(parsingId: "local/prefill_throughput_512")
            == .prefillThroughput(PrefillThroughputBenchmark(
                benchmarkId: BenchmarkId(validated: "prefill_throughput_512"),
                parameterPrefillTokens: 512)))
        #expect(BenchmarkDefinition(parsingId: "vl_max_memory_224") == nil)
    }
}
