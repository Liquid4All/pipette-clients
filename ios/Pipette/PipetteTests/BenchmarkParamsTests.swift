import Foundation
import Testing

@testable import Pipette

/// `BenchmarkParams` (pipette-mgmt's `pipette-plan-types/src/benchmark/params.rs`)
/// and the `benchmark_type` + `benchmark_params` pair a submission declares with it.
struct BenchmarkParamsTests {
    private func object(_ value: some Encodable) throws -> [String: Any] {
        try #require(try JSONSerialization.jsonObject(
            with: JSONEncoder().encode(value)) as? [String: Any])
    }

    /// `a_definition_maps_to_its_params`: every value distinct, so a field read from its
    /// neighbor fails here. An eval that pins no temperature takes its eval id's default.
    @Test(arguments: [
        (BenchmarkDefinition.prefillThroughput(PrefillThroughputBenchmark(
            benchmarkId: BenchmarkId(validated: "b"), parameterPrefillTokens: 512,
            parameterBatchSize: 4)),
         BenchmarkParams.prefillThroughput(prefillTokens: 512, batchSize: 4)),
        (.decodeThroughput(DecodeThroughputBenchmark(
            benchmarkId: BenchmarkId(validated: "b"), parameterPrefillTokens: 512,
            parameterDecodeTokens: 256, parameterBatchSize: 8)),
         .decodeThroughput(prefillTokens: 512, decodeTokens: 256, batchSize: 8)),
        (.endToEndLatency(EndToEndLatencyBenchmark(
            benchmarkId: BenchmarkId(validated: "b"), parameterPrefillTokens: 512,
            parameterDecodeTokens: 128, parameterBatchSize: 2)),
         .endToEndLatency(prefillTokens: 512, decodeTokens: 128, batchSize: 2)),
        (.maxMemoryUsage(MaxMemoryUsageBenchmark(
            benchmarkId: BenchmarkId(validated: "b"), parameterPrefillTokens: 1024)),
         .maxMemoryUsage(prefillTokens: 1024)),
        (.eval(EvalBenchmark(
            benchmarkId: BenchmarkId(validated: "b"),
            parameterEvalId: EvalId("mmlu_pro"),
            parameterDatasetName: "edge_2026.03.1",
            parameterMaxTokens: 256,
            parameterMcqChoices: nil,
            samples: nil)),
         .eval(evalId: EvalId("mmlu_pro"), datasetName: "edge_2026.03.1", maxTokens: 256,
               maxContext: nil, mcqChoices: nil, temperature: .greedy, topP: nil, topK: nil,
               minP: nil, seed: nil)),
        (.eval(EvalBenchmark(
            benchmarkId: BenchmarkId(validated: "b"),
            parameterEvalId: EvalId("gpqa_diamond"),
            parameterDatasetName: "2026.06.1",
            parameterMaxTokens: 8192,
            samples: nil)),
         .eval(evalId: EvalId("gpqa_diamond"), datasetName: "2026.06.1", maxTokens: 8192,
               maxContext: nil, mcqChoices: nil, temperature: SamplingTemperature(validated: 0.6),
               topP: nil, topK: nil, minP: nil, seed: nil)),
        (.eval(EvalBenchmark(
            benchmarkId: BenchmarkId(validated: "b"),
            parameterEvalId: EvalId("mmlu_pro"),
            parameterDatasetName: "edge_2026.03.1",
            parameterMaxTokens: 256,
            parameterMcqChoices: ["A", "B"],
            parameterTemperature: SamplingTemperature(validated: 1.0),
            parameterTopP: TopP(validated: 0.95),
            parameterTopK: TopK(validated: 64),
            parameterMinP: MinP(validated: 0.05),
            parameterSeed: SamplingSeed(validated: 7),
            samples: nil)),
         .eval(evalId: EvalId("mmlu_pro"), datasetName: "edge_2026.03.1", maxTokens: 256,
               maxContext: nil, mcqChoices: ["A", "B"],
               temperature: SamplingTemperature(validated: 1.0), topP: TopP(validated: 0.95),
               topK: TopK(validated: 64), minP: MinP(validated: 0.05),
               seed: SamplingSeed(validated: 7))),
        (.vlThroughput(VlThroughputBenchmark(
            benchmarkId: BenchmarkId(validated: "b"),
            parameterImageWidth: 224,
            parameterImageHeight: 336,
            parameterTextTokens: 8,
            parameterDecodeTokens: 64,
            parameterNumImages: 3)),
         .vlThroughput(imageWidth: 224, imageHeight: 336, textTokens: 8, decodeTokens: 64,
                       numImages: 3)),
        (.vlMaxMemory(VlMaxMemoryBenchmark(
            benchmarkId: BenchmarkId(validated: "b"), parameterImageWidth: 224,
            parameterImageHeight: 336)),
         .vlMaxMemory(imageWidth: 224, imageHeight: 336, textTokens: 0, decodeTokens: nil,
                      numImages: 1)),
        (.vlMaxMemory(VlMaxMemoryBenchmark(
            benchmarkId: BenchmarkId(validated: "b"), parameterImageWidth: 224,
            parameterImageHeight: 336, parameterTextTokens: 8, parameterDecodeTokens: 256,
            parameterNumImages: 3)),
         .vlMaxMemory(imageWidth: 224, imageHeight: 336, textTokens: 8, decodeTokens: 256,
                      numImages: 3)),
        (.vlSpecDecode(VlSpecDecodeBenchmark(
            benchmarkId: BenchmarkId(validated: "b"),
            parameterDraftLength: 7,
            parameterMaxTokens: 2048,
            parameterBatchSize: BatchSize(validated: 8),
            parameterTemperature: SamplingTemperature(validated: 0.7),
            dataset: BenchmarkDefinitionTests.sampleDataset)),
         .vlSpecDecode(draftLength: 7, maxTokens: 2048, batchSize: 8,
                       temperature: SamplingTemperature(validated: 0.7))),
        (.asrThroughput(BenchmarkDefinitionTests.asrThroughput("b")),
         .asrThroughput(audioDurationMs: AudioDurationMillis(validated: 15000),
                        audioSampleRateHz: AudioSampleRate(validated: 16000),
                        audioChannels: AudioChannels(validated: 1), textTokens: 1024,
                        decodeTokens: AsrDecodeTokens(validated: 256))),
        (.asrMaxMemory(BenchmarkDefinitionTests.asrMaxMemory("b")),
         .asrMaxMemory(audioDurationMs: AudioDurationMillis(validated: 15000),
                       audioSampleRateHz: AudioSampleRate(validated: 16000),
                       audioChannels: AudioChannels(validated: 1), textTokens: 0,
                       decodeTokens: AsrDecodeTokens(validated: 256))),
    ])
    func aDefinitionMapsToItsParams(definition: BenchmarkDefinition,
                                    expected: BenchmarkParams) throws {
        let params = BenchmarkParams(definition)
        #expect(params == expected)
        #expect(params.benchmarkType == definition.benchmarkType)
    }

    /// `a_prompt_may_be_empty_but_a_decode_may_not`: the definition holds the params'
    /// types, so a count the params could not hold is refused where the definition is read.
    @Test(arguments: [
        (UInt32(0), UInt32(32), true),
        (256, 0, false),
        (256, .max, false),
    ])
    func aPromptMayBeEmptyButADecodeMayNot(prefill: UInt32, decode: UInt32, parses: Bool) {
        let json = """
            {"benchmark_id": "decode_throughput_\(prefill)_\(decode)",
             "benchmark_type": "decode_throughput",
             "parameter_prefill_tokens": \(prefill), "parameter_decode_tokens": \(decode)}
            """
        let definition = try? JSONDecoder().decode(BenchmarkDefinition.self, from: Data(json.utf8))
        #expect((definition != nil) == parses)
    }

    /// The crate leaves `None` out, so an entry without a decode count keeps the params it
    /// had.
    @Test(arguments: [(nil, false), (256, true)] as [(PositiveCount?, Bool)])
    func aVisionMemoryCellDeclaresItsDecodeCountOnlyWhenSet(decode: PositiveCount?,
                                                            declared: Bool) throws {
        let params = try object(BenchmarkParams.vlMaxMemory(
            imageWidth: 256, imageHeight: 256, textTokens: 1024, decodeTokens: decode,
            numImages: 5))
        #expect((params["decode_tokens"] as? Int == 256) == declared)
        #expect(params.count == (declared ? 5 : 4))
    }

    /// An unset text count as `0`, as the crate writes it.
    @Test func anAsrCellDeclaresEveryField() throws {
        let params = try object(BenchmarkParams(.asrMaxMemory(
            BenchmarkDefinitionTests.asrMaxMemory("b"))))
        #expect(params as NSDictionary == [
            "audio_duration_ms": 15000, "audio_sample_rate_hz": 16000, "audio_channels": 1,
            "text_tokens": 0, "decode_tokens": 256,
        ] as NSDictionary)
    }

    /// The payload carries the tag as its own `benchmark_type` beside the untagged
    /// params, the crate's `declared_benchmark_params`; with none declared, neither key.
    @Test func aSubmissionDeclaresTheTypeBesideTheUntaggedParams() throws {
        func payload(_ params: BenchmarkParams?) -> BenchmarkSubmissionPayload {
            BenchmarkSubmissionPayload(
                benchmarkId: "prefill_throughput_512", benchmarkParams: params,
                device: deviceInfoFixture(), power: powerStateFixture(), modelFlags: nil,
                modelDescriptor: SubmissionFixtures.mlxModelDescriptor, runtimeFlags: nil,
                benchmarkFlags: nil, runtimeDescriptor: SubmissionFixtures.mlxRuntimeDescriptor,
                runtimeCpuVariant: nil, clientVersion: "1.2 (34)",
                submittedAt: "2026-06-24T00:00:00Z",
                result: .prefillThroughput(prefillTimeMs: 12, prefillTimeMsStddev: nil))
        }
        let declared = try object(payload(.prefillThroughput(prefillTokens: 512, batchSize: 1)))
        #expect(declared["benchmark_type"] as? String == "prefill_throughput")
        #expect(declared["benchmark_params"] as? NSDictionary
            == ["prefill_tokens": 512, "batch_size": 1] as NSDictionary)

        let undeclared = try object(payload(nil))
        #expect(undeclared["benchmark_type"] == nil)
        #expect(undeclared["benchmark_params"] == nil)
    }
}
