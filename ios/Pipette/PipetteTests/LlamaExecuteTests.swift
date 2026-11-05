import Testing
import Foundation
@testable import Pipette

/// The per-benchmark `LlamaExecute` functions against a fake `Inference`: no model and no
/// `llama` linked. The peak sampler is `ProcessMemoryTests`'.
@Suite struct LlamaExecuteTests {

    final class FakeInference {
        enum Boom: Error { case sample }
        var tokenizeCount = 0
        var prefillCount = 0
        var decodeCount = 0
        var lastDecodeCount = 0
        var generateCount = 0
        var lastGenerateCount = 0
        var generateEnds: [GenerationEnd] = []
        /// Stands in for an engine that decoded a different workload than the cell requested.
        var decodeReturns: Int?
        var generateReturns: Int?
        var chatCount = 0
        var lastChatMaxTokens = 0
        var lastChatMcq: [String]?
        var lastChatTemperature: Double = -1
        var lastChatSampling: Sampling?
        var lastChatSeeds: [UInt32?] = []
        var chatResult = "ok"
        var failChatMarker: String?
        var failChatError: any Error = Boom.sample

        func make() -> Inference {
            Inference(
                tokenize: { text, addSpecial in
                    self.tokenizeCount += 1
                    let n = text.count + (addSpecial ? 1 : 0)
                    return (0 ..< n).map { Int32($0 % 1000) }
                },
                resetContext: {},
                resetSampler: {},
                prefill: { _ in self.prefillCount += 1 },
                decode: { count in
                    self.decodeCount += 1; self.lastDecodeCount = count
                    return self.decodeReturns ?? count
                },
                generate: { count, end in
                    self.generateCount += 1; self.lastGenerateCount = count
                    self.generateEnds.append(end)
                    return LlamaGeneration(tokens: self.generateReturns ?? count, lastSampledAt: .now)
                },
                chatCompletion: { json, maxTokens, mcq, sampling, seed in
                    self.chatCount += 1; self.lastChatMaxTokens = maxTokens; self.lastChatMcq = mcq
                    self.lastChatTemperature = sampling.temperature.value
                    self.lastChatSampling = sampling
                    self.lastChatSeeds.append(seed)
                    if let marker = self.failChatMarker, json.contains(marker) { throw self.failChatError }
                    return EvalGeneration(text: self.chatResult, stopReason: .unknown, stopDetail: nil, completionTokens: nil)
                })
        }
    }

    private let ready: () throws -> Void = {}

    private let prefill512 = PrefillThroughputBenchmark(
        benchmarkId: BenchmarkId(validated: "p_512"), parameterPrefillTokens: 512)
    private let decode512x100 = DecodeThroughputBenchmark(
        benchmarkId: BenchmarkId(validated: "d_512_100"), parameterPrefillTokens: 512,
        parameterDecodeTokens: 100)
    private let endToEnd512x256 = EndToEndLatencyBenchmark(
        benchmarkId: BenchmarkId(validated: "e_512_256"), parameterPrefillTokens: 512,
        parameterDecodeTokens: 256)

    private func runEval(_ benchmark: EvalBenchmark, _ inf: Inference,
                         progress: @escaping (BenchmarkProgress) -> Void) async throws
        -> BenchmarkResultData {
        try await LlamaExecute.eval(
            benchmark, cell: try EvalCell(benchmark), inf, checkpoint: try scratchCheckpoint(),
            isCancelled: { false }, progress: progress)
    }

    /// The opener lets a test read back what a run kept.
    private func threeSampleEval(root: URL) -> (EvalBenchmark, () throws -> EvalCompletionSession) {
        let digest = EvalRunDigest(value: String(repeating: "1", count: 64))
        let meta = EvalCompletionMeta(benchmarkId: "eval", runtime: "test", model: "test")
        let benchmark = EvalBenchmark(
            benchmarkId: BenchmarkId(validated: "eval"),
            parameterEvalId: .known(.gpqaDiamond),
            parameterDatasetName: "gpqa",
            parameterMaxTokens: 8,
            samples: ["q1", "q2", "q3"].map {
                EvalSample(id: $0, messages: [["role": "user", "content": $0]])
            })
        return (benchmark, { try EvalCompletionSession(root: root, digest: digest, meta: meta) })
    }

    private let noCorpus = NoCorpus(root: FileManager.default.temporaryDirectory
        .appendingPathComponent("unused-\(UUID().uuidString)", isDirectory: true))

    // MARK: - Throughput dispatch (1 warmup + 5 reps)

    @Test func prefillRunsWarmupPlusReps() async throws {
        let fake = FakeInference()
        let result = try await LlamaExecute.prefillThroughput(
            prefill512, fake.make(), readinessGate: ready, observer: noObserver())
        guard case .prefillThroughput = result else { Issue.record("got \(result)"); return }
        #expect(fake.prefillCount == 6)   // 1 warmup + 5 reps
        #expect(fake.decodeCount == 0)
    }

    @Test func decodeTimesFixedCountWithWarmup() async throws {
        let fake = FakeInference()
        let result = try await LlamaExecute.decodeThroughput(
            decode512x100, fake.make(), readinessGate: ready, observer: noObserver())
        guard case .decodeThroughput = result else { Issue.record("got \(result)"); return }
        #expect(fake.prefillCount == 6)
        #expect(fake.decodeCount == 6)
        #expect(fake.lastDecodeCount == 100)
    }

    @Test func endToEndRunsPrefillAndDecode() async throws {
        let fake = FakeInference()
        let result = try await LlamaExecute.endToEndLatency(
            endToEnd512x256, fake.make(), readinessGate: ready, observer: noObserver())
        guard case .endToEndLatency = result else { Issue.record("got \(result)"); return }
        #expect(fake.prefillCount == 6)
        // e2e samples with EOG masked, as the crate's server is asked to; feeding tokens
        // without sampling is the decode cell's primitive.
        #expect(fake.generateCount == 6)
        #expect(fake.lastGenerateCount == 256)
        #expect(fake.decodeCount == 0)
        // e2e tokenizes inside each measured rep, as the crate times it.
        #expect(fake.tokenizeCount >= 5)
    }

    /// The crate's server stops a request once it has sampled the last token, so every
    /// attempt, warm-up included, stops there rather than decoding it.
    @Test func endToEndStopsAfterTheLastSample() async throws {
        let fake = FakeInference()
        _ = try await LlamaExecute.endToEndLatency(
            endToEnd512x256, fake.make(), readinessGate: ready, observer: noObserver())
        #expect(fake.generateEnds == Array(repeating: .afterLastSample, count: 6))
    }

    /// A wrong-sized generation measured a different workload: the crate's `expect_tokens`.
    /// The warm-up is checked too, which is where this one fails.
    @Test func aRepThatDecodesTheWrongCountFailsTheRun() async {
        let fake = FakeInference()
        fake.decodeReturns = 99

        await #expect(throws: DecodeMeasurementError.mismatch(
            field: "decode_tokens", actual: 99, expected: 100)) {
            try await LlamaExecute.decodeThroughput(
                decode512x100, fake.make(), readinessGate: ready, observer: noObserver())
        }
        #expect(fake.decodeCount == 1, "the warm-up is where a wrong-sized workload is caught")
    }

    @Test func aRepThatGeneratesTheWrongCountFailsTheRun() async {
        let fake = FakeInference()
        fake.generateReturns = 3

        await #expect(throws: EndToEndMeasurementError.mismatch(
            field: "decode_tokens", actual: 3, expected: 256)) {
            try await LlamaExecute.endToEndLatency(
                endToEnd512x256, fake.make(), readinessGate: ready, observer: noObserver())
        }
    }

    // MARK: - max_memory_usage on-model work

    @Test func maxMemoryUsageDrivesOnePrefillAndOneDecode() throws {
        let fake = FakeInference()
        try LlamaExecute.maxMemoryUsage(fake.make(), prefillTokens: 512)
        #expect(fake.prefillCount == 1)     // no warmup, no rep loop
        #expect(fake.decodeCount == 1)
        #expect(fake.lastDecodeCount == 1)
    }

    // MARK: - eval (per-sample, no rep loop)

    @Test func evalCompletesEachSample() async throws {
        let fake = FakeInference()
        fake.chatResult = "answer"
        let samples = [
            EvalSample(id: "q1", messages: [["role": "user", "content": "hi"]]),
            EvalSample(id: "q2", messages: [["role": "user", "content": "there"]]),
        ]
        let result = try await runEval(EvalBenchmark(
                benchmarkId: BenchmarkId(validated: "eval"),
                parameterEvalId: .known(.gpqaDiamond),
                parameterDatasetName: "gpqa",
                parameterMaxTokens: 256,
                parameterMcqChoices: nil,
                samples: samples),
            fake.make(), progress: { _ in })
        guard case .eval(let completions) = result else { Issue.record("got \(result)"); return }
        #expect(completions.count == 2)
        #expect(completions.map(\.id) == ["q1", "q2"])
        #expect(completions.allSatisfy { if case .completed(_, "answer", _, _, _) = $0 { true } else { false } })
        #expect(fake.lastChatMaxTokens == 256)
    }

    @Test func evalSampleFailureBecomesFailedCompletion() async throws {
        let fake = FakeInference()
        fake.failChatMarker = "BOOM"
        let samples = [
            EvalSample(id: "q1", messages: [["role": "user", "content": "fine"]]),
            EvalSample(id: "q2", messages: [["role": "user", "content": "BOOM"]]),
        ]
        let result = try await runEval(EvalBenchmark(
                benchmarkId: BenchmarkId(validated: "eval"),
                parameterEvalId: .known(.gpqaDiamond),
                parameterDatasetName: "gpqa",
                parameterMaxTokens: 256,
                parameterMcqChoices: nil,
                samples: samples),
            fake.make(), progress: { _ in })
        guard case .eval(let completions) = result else { Issue.record("got \(result)"); return }
        guard case .completed = completions[0] else { Issue.record("q1 should have completed"); return }
        guard case .failed(_, let reason) = completions[1] else { Issue.record("q2 should have failed"); return }
        #expect(!reason.isEmpty)
    }

    /// When `q2` fails as the stop lands, the failure is the stop's, so it leaves no row.
    @Test(arguments: [(nil, ["q1", "q2"]), ("q2", ["q1"])] as [(String?, [String])])
    func aStopEndsTheEvalAndKeepsWhatCameBefore(failing: String?, kept: [String]) async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("evals-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let (benchmark, checkpoint) = threeSampleEval(root: root)
        let fake = FakeInference()
        fake.failChatMarker = failing

        await #expect(throws: RuntimeError.cancelled) {
            _ = try await LlamaExecute.eval(
                benchmark, cell: try EvalCell(benchmark), fake.make(), checkpoint: try checkpoint(),
                isCancelled: { fake.chatCount >= 2 }, progress: { _ in })
        }

        #expect(fake.chatCount == 2)
        let resumed = try checkpoint()
        defer { resumed.close() }
        #expect(resumed.doneIds == kept)
    }

    @Test func aFailedBackendEndsTheEvalAndKeepsWhatCameBefore() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("evals-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let (benchmark, checkpoint) = threeSampleEval(root: root)
        let fake = FakeInference()
        fake.failChatMarker = "q2"
        fake.failChatError = RuntimeError.backend("decode decode rc=-3")

        await #expect(throws: RuntimeError.backend("decode decode rc=-3")) {
            _ = try await LlamaExecute.eval(
                benchmark, cell: try EvalCell(benchmark), fake.make(), checkpoint: try checkpoint(),
                isCancelled: { false }, progress: { _ in })
        }

        #expect(fake.chatCount == 2)
        let resumed = try checkpoint()
        defer { resumed.close() }
        #expect(resumed.doneIds == ["q1"])
    }

    /// A clean result with no completions would score as a cell that answered nothing, so it
    /// fails as the crate's `eval benchmark missing samples` does, before the model resolves.
    @Test func anEvalBodyWithNoSamplesIsRefused() async throws {
        let dir = try makeFixtureDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let declared = try ggufTextSpec("org/a-GGUF", "a-Q4_0.gguf")
        let request = RunRequest(
            runtime: .alreadyBound(Runtime.thisBuild(for: declared)),
            model: DeclaredBound(declared: declared,
                                 bound: try boundModel(declared, "/nowhere/a.gguf", nil)),
            runtimeFlags: nil, benchmarkFlags: nil,
            benchmark: .eval(EvalBenchmark(
                benchmarkId: BenchmarkId(validated: "eval"),
                parameterEvalId: .known(.gpqaDiamond),
                parameterDatasetName: "gpqa",
                parameterMaxTokens: 256,
                samples: nil)))

        await #expect(throws: EvalMeasurementError.missingSamples) {
            try await LlamaExecute.run(request, runContext: NoCorpus(root: dir),
                                         readinessGate: ready, observer: noObserver())
        }
    }

    /// A client-side policy keyed on the eval id, as the crate keys it: the generative evals
    /// draw at 0.6, and an id this build does not know stays greedy.
    @Test func theEvalIdSetsTheSamplingTemperature() async throws {
        let samples = [EvalSample(id: "q1", messages: [["role": "user", "content": "hi"]])]

        let sampled = FakeInference()
        _ = try await runEval(EvalBenchmark(
                benchmarkId: BenchmarkId(validated: "gpqa_diamond"),
                parameterEvalId: .known(.gpqaDiamond),
                parameterDatasetName: "gpqa",
                parameterMaxTokens: 8,
                parameterMcqChoices: nil,
                samples: samples),
            sampled.make(), progress: { _ in })
        #expect(sampled.lastChatTemperature == 0.6)

        let greedy = FakeInference()
        _ = try await runEval(EvalBenchmark(
                benchmarkId: BenchmarkId(validated: "eval_smoke"),
                parameterEvalId: EvalId("eval_smoke"),
                parameterDatasetName: "local",
                parameterMaxTokens: 8,
                parameterMcqChoices: nil,
                samples: samples),
            greedy.make(), progress: { _ in })
        #expect(greedy.lastChatTemperature == 0)
    }

    /// The temperature overrides the eval id's policy, and each sample's seed is offset by
    /// its `#k` repeat index.
    @Test func theEntrysPinnedSamplingReachesTheEngine() async throws {
        let samples = ["q1#0", "q1#2"].map {
            EvalSample(id: $0, messages: [["role": "user", "content": "hi"]])
        }
        let fake = FakeInference()
        _ = try await runEval(EvalBenchmark(
                benchmarkId: BenchmarkId(validated: "gpqa_diamond_t1"),
                parameterEvalId: .known(.gpqaDiamond),
                parameterDatasetName: "gpqa",
                parameterMaxTokens: 8,
                parameterMcqChoices: nil,
                parameterTemperature: try SamplingTemperature(1.0),
                parameterTopP: try TopP(0.95),
                parameterTopK: try TopK(64),
                parameterMinP: try MinP(0.05),
                parameterSeed: try SamplingSeed(7),
                samples: samples),
            fake.make(), progress: { _ in })

        let sampling = try #require(fake.lastChatSampling)
        #expect(sampling.temperature.value == 1.0)
        #expect(sampling.topP?.value == 0.95)
        #expect(sampling.topK?.value == 64)
        #expect(sampling.minP?.value == 0.05)
        #expect(fake.lastChatSeeds == [7, 9])
    }

    @Test func evalMcqConstrainsToSingleToken() async throws {
        let fake = FakeInference()
        let samples = [EvalSample(id: "q1", messages: [["role": "user", "content": "pick"]])]
        _ = try await runEval(EvalBenchmark(
                benchmarkId: BenchmarkId(validated: "eval"),
                parameterEvalId: .known(.gpqaDiamond),
                parameterDatasetName: "gpqa",
                parameterMaxTokens: 256,
                parameterMcqChoices: ["A", "B", "C", "D"],
                samples: samples),
            fake.make(), progress: { _ in })
        #expect(fake.lastChatMaxTokens == 1)
        #expect(fake.lastChatMcq == ["A", "B", "C", "D"])
    }

    /// The crate filters an empty choice list away: constraining to no choices would answer
    /// nothing.
    @Test func anEmptyMcqListIsAFreeFormCell() async throws {
        let fake = FakeInference()
        let samples = [EvalSample(id: "q1", messages: [["role": "user", "content": "pick"]])]

        _ = try await runEval(EvalBenchmark(
                benchmarkId: BenchmarkId(validated: "eval"),
                parameterEvalId: .known(.gpqaDiamond),
                parameterDatasetName: "gpqa",
                parameterMaxTokens: 256,
                parameterMcqChoices: [],
                samples: samples),
            fake.make(), progress: { _ in })

        #expect(fake.lastChatMcq == nil)
        #expect(fake.lastChatMaxTokens == 256, "free-form keeps the cell's token budget")
    }

    // MARK: - Dispatch

    /// As every timing execute module gates before it starts its server: a cell that waited
    /// for a cool device and then loaded for twenty seconds has warmed it up again. The gate
    /// refuses, so its error, not a load failure on the non-model file, proves the order.
    @Test func theGateClearsBeforeTheModelLoads() async throws {
        let dir = try makeFixtureDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let declared = try ggufTextSpec("org/a-GGUF", "a-Q4_0.gguf")
        let request = RunRequest(
            runtime: .alreadyBound(Runtime.thisBuild(for: declared)),
            model: DeclaredBound(
                declared: declared,
                bound: try boundModel(declared, try writeFixtureFile(dir, "not-a-model.gguf"), nil)),
            runtimeFlags: nil, benchmarkFlags: nil,
            benchmark: .prefillThroughput(PrefillThroughputBenchmark(
                benchmarkId: BenchmarkId(validated: "prefill_throughput_512"),
                parameterPrefillTokens: 512)))

        await #expect(throws: RuntimeError.readiness("hot")) {
            try await LlamaExecute.run(
                request, runContext: NoCorpus(root: dir),
                readinessGate: { throw RuntimeError.readiness("hot") }, observer: noObserver())
        }
    }

    /// With nothing on disk the cell stops at the vision pair, naming the file it wanted.
    @Test func aVisionCellProjectsTheVisionPair() async throws {
        let vision = try ggufVisionSpec("org/vl-GGUF", "vl-Q4_0.gguf", "mmproj-f16.gguf")
        let request = RunRequest(
            runtime: .alreadyBound(Runtime.thisBuild(for: vision)),
            model: DeclaredBound(declared: vision,
                                 bound: try boundModel(vision, "/nowhere/vl.gguf",
                                                       "/nowhere/mmproj.gguf")),
            runtimeFlags: nil, benchmarkFlags: nil,
            benchmark: .vlThroughput(VlThroughputBenchmark(
                benchmarkId: BenchmarkId(validated: "vl_throughput_512_512_32_64"),
                parameterImageWidth: 512,
                parameterImageHeight: 512,
                parameterTextTokens: 32,
                parameterDecodeTokens: 64)))

        await #expect(throws: BoundPathMissing(what: "GGUF vision model",
                                               path: "/nowhere/vl.gguf")) {
            try await LlamaExecute.run(request, runContext: noCorpus, readinessGate: ready,
                                         observer: noObserver())
        }
    }

    /// The model is not on disk, so the context's refusal being the cell's error proves the
    /// order.
    @Test func aCorpusCellAsksTheContextForItsDatasetFirst() async throws {
        let vision = try ggufVisionSpec("org/vl-GGUF", "vl-Q4_0.gguf", "mmproj-f16.gguf")
        let pinned = BenchmarkDefinitionTests.sampleDataset
        let request = RunRequest(
            runtime: .alreadyBound(Runtime.thisBuild(for: vision)),
            model: DeclaredBound(declared: vision,
                                 bound: try boundModel(vision, "/nowhere/vl.gguf",
                                                       "/nowhere/mmproj.gguf")),
            runtimeFlags: nil, benchmarkFlags: nil,
            benchmark: .vlSpecDecode(VlSpecDecodeBenchmark(
                benchmarkId: BenchmarkId(validated: "vl_spec_decode_corpus_download_test"),
                parameterDraftLength: 8,
                parameterMaxTokens: 64,
                dataset: pinned)))
        let runContext = RefusingContext()

        await #expect(throws: RuntimeError.engine("no corpus server in this test")) {
            try await LlamaExecute.run(request, runContext: runContext, readinessGate: ready,
                                         observer: noObserver())
        }
        #expect(runContext.datasets == [pinned])
    }

    @Test func aBatchedCorpusCellIsRefusedBeforeItsDownload() async throws {
        let vision = try ggufVisionSpec("org/vl-GGUF", "vl-Q4_0.gguf", "mmproj-f16.gguf")
        let request = RunRequest(
            runtime: .alreadyBound(Runtime.thisBuild(for: vision)),
            model: DeclaredBound(declared: vision,
                                 bound: try boundModel(vision, "/nowhere/vl.gguf",
                                                       "/nowhere/mmproj.gguf")),
            runtimeFlags: nil, benchmarkFlags: nil,
            benchmark: .vlSpecDecode(VlSpecDecodeBenchmark(
                benchmarkId: BenchmarkId(validated: "vl_spec_decode_corpus_bs8"),
                parameterDraftLength: 8,
                parameterMaxTokens: 64,
                parameterBatchSize: 8,
                dataset: BenchmarkDefinitionTests.sampleDataset)))
        let runContext = RefusingContext()

        let error = await #expect(throws: RuntimeError.self) {
            try await LlamaExecute.run(request, runContext: runContext, readinessGate: ready,
                                         observer: noObserver())
        }
        #expect("\(try #require(error))".contains("8 requests in flight"))
        #expect(runContext.datasets.isEmpty)
    }

    /// `dir` holds no weights.
    private func textCell(_ dir: URL, _ benchmark: BenchmarkDefinition,
                          runtime: RuntimeType = .llamacppIosPipette,
                          flags: RuntimeFlags? = nil) throws -> RunRequest {
        let declared = try ggufTextSpec("org/a-GGUF", "m.gguf")
        return RunRequest(
            runtime: .alreadyBound(try #require(Runtime.thisBuild(for: runtime))),
            model: DeclaredBound(declared: declared, bound: try boundModel(
                declared, dir.appendingPathComponent("m.gguf").path, nil)),
            runtimeFlags: flags, benchmarkFlags: nil, benchmark: benchmark)
    }

    /// As the onnx-genai and openvino crates'
    /// `a_batched_entry_is_refused_before_the_runtime_binds`; the model is the first thing a
    /// cell here resolves. An unbatched entry gets as far as the missing weights.
    @Test(arguments: [
        (BenchmarkDefinition.prefillThroughput(PrefillThroughputBenchmark(
            benchmarkId: BenchmarkId(validated: "prefill_throughput_512_bs4"),
            parameterPrefillTokens: 512, parameterBatchSize: 4)), "prefill_throughput_512_bs4"),
        (.decodeThroughput(DecodeThroughputBenchmark(
            benchmarkId: BenchmarkId(validated: "decode_throughput_512_128_bs4"),
            parameterPrefillTokens: 512, parameterDecodeTokens: 128, parameterBatchSize: 4)),
         "decode_throughput_512_128_bs4"),
        (.prefillThroughput(PrefillThroughputBenchmark(
            benchmarkId: BenchmarkId(validated: "prefill_throughput_512_bs4"),
            parameterPrefillTokens: 512, parameterBatchSize: 4)), "decodes one sequence"),
        (.prefillThroughput(PrefillThroughputBenchmark(
            benchmarkId: BenchmarkId(validated: "prefill_throughput_512"),
            parameterPrefillTokens: 512)), "GGUF text model"),
    ] as [(BenchmarkDefinition, String)])
    func aBatchedEntryIsRefusedBeforeTheModelIsResolved(_ benchmark: BenchmarkDefinition,
                                                        _ expected: String) async throws {
        let dir = try makeFixtureDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let request = try textCell(dir, benchmark)
        let error = await #expect(throws: (any Error).self) {
            try await LlamaExecute.run(request, runContext: noCorpus, readinessGate: ready,
                                         observer: noObserver())
        }
        let message = error?.localizedDescription ?? ""
        #expect(message.contains(expected), "\(message)")
    }

    @Test func aRuntimeOtherThanLlamacppIsRefused() async throws {
        let dir = try makeFixtureDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let request = try textCell(dir, .prefillThroughput(prefill512), runtime: .appleFoundation)
        let error = await #expect(throws: RuntimeFlagsError.self) {
            try await LlamaExecute.run(request, runContext: noCorpus, readinessGate: ready,
                                         observer: noObserver())
        }
        let message = error?.localizedDescription ?? ""
        #expect(message.contains("apple_foundation"), "\(message)")
    }

    @Test func flagsForAnotherCellAreRefused() async throws {
        let dir = try makeFixtureDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let request = try textCell(
            dir, .prefillThroughput(prefill512), flags: .evalLlamacppIosPipetteGgufText(
                numberGpuLayers: nil, ctxSize: nil, nUbatch: nil, threads: nil, swaFull: nil,
                loadMode: nil))
        await #expect(throws: RunRequestError.runtimeFlagsForAnotherCell(
            flags: .evalLlamacppIosPipetteGgufText(
                numberGpuLayers: nil, ctxSize: nil, nUbatch: nil, threads: nil, swaFull: nil,
                loadMode: nil),
            cell: (.prefillThroughput, .llamacppIosPipette, .ggufText))) {
            try await LlamaExecute.run(request, runContext: noCorpus, readinessGate: ready,
                                         observer: noObserver())
        }
    }

    private func visionModel(_ dir: URL, speculative: Bool) throws -> Model {
        let model = try writeFixtureFile(dir, "model.gguf")
        let mmproj = try writeFixtureFile(dir, "mmproj.gguf")
        guard speculative else {
            return try boundModel(
                try ggufVisionSpec("org/vl-GGUF", "model.gguf", "mmproj.gguf"), model, mmproj)
        }
        return try boundModel(
            try ggufVisionSpeculativeSpec("org/vl-GGUF", "model.gguf", "mmproj.gguf",
                                          draft: "draft.gguf"),
            model, mmproj, try writeFixtureFile(dir, "draft.gguf"))
    }

    /// Off, a speculative model loads as a plain one does; on is refused for a plain model,
    /// which has no drafter.
    @Test(arguments: [
        (true, true, VlSpecDecode.Speculation.on),
        (true, false, .off),
        (false, false, .off),
        (false, true, nil),
    ] as [(Bool, Bool, VlSpecDecode.Speculation?)])
    func onlySpeculationOnLoadsTheDrafter(_ speculative: Bool, _ enabled: Bool,
                                          _ expected: VlSpecDecode.Speculation?) throws {
        let dir = try makeFixtureDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let model = try visionModel(dir, speculative: speculative)
        guard let expected else {
            #expect(throws: UnexpectedBoundModel.self) {
                try LlamaExecute.LoadedModel.resolve(model, enabled: enabled)
            }
            return
        }
        let loaded = try LlamaExecute.LoadedModel.resolve(model, enabled: enabled)
        #expect(loaded.speculation == expected)
        if case let .speculativeOn(resolved) = loaded {
            #expect(resolved.draft == dir.appendingPathComponent("draft.gguf").path)
        }
    }

    // MARK: - Gating & unsupported types

    @Test func cancelledGateThrowsBeforeReps() async {
        let fake = FakeInference()
        await #expect(throws: RuntimeError.cancelled) {
            try await LlamaExecute.prefillThroughput(
                prefill512, fake.make(), readinessGate: { throw RuntimeError.cancelled }, observer: noObserver())
        }
        // The warm-up is ungated; the gate fires before the first measured rep.
        #expect(fake.prefillCount == 1)
    }

    @Test func timedOutGateThrowsReadiness() async {
        let fake = FakeInference()
        await #expect(throws: RuntimeError.readiness("hot")) {
            try await LlamaExecute.prefillThroughput(
                prefill512, fake.make(), readinessGate: { throw RuntimeError.readiness("hot") }, observer: noObserver())
        }
    }

    // MARK: - Semantic progress events

    @Test func evalReportsPerSampleProgress() async throws {
        let fake = FakeInference()
        var events: [BenchmarkProgress] = []
        let samples = (1 ... 3).map { EvalSample(id: "q\($0)", messages: [["role": "user", "content": "hi"]]) }
        _ = try await runEval(EvalBenchmark(
                benchmarkId: BenchmarkId(validated: "eval"),
                parameterEvalId: .known(.gpqaDiamond),
                parameterDatasetName: "gpqa",
                parameterMaxTokens: 8,
                parameterMcqChoices: nil,
                samples: samples),
            fake.make(), progress: { events.append($0) })
        #expect(events == [.sample(completed: 1, total: 3),
                           .sample(completed: 2, total: 3),
                           .sample(completed: 3, total: 3)])
    }

    // MARK: - Prompt-seed undersized guard

    @Test func seedThrowsWhenTokenizerCannotReachTarget() async {
        // The bounded grow loop can never reach `target`, so seeding must fail rather than
        // emit a short prompt.
        let inf = Inference(
            tokenize: { _, _ in [Int32](repeating: 7, count: 4) },
            resetContext: {}, resetSampler: {}, prefill: { _ in },
            decode: { $0 }, generate: { count, _ in LlamaGeneration(tokens: count, lastSampledAt: .now) },
            chatCompletion: { _, _, _, _, _ in EvalGeneration(text: "", stopReason: .unknown, stopDetail: nil, completionTokens: nil) })
        await #expect(throws: RuntimeError.self) {
            try await LlamaExecute.prefillThroughput(
                prefill512, inf, readinessGate: ready, observer: noObserver())
        }
    }

    @Test func seedThrowsWhenCorpusTokenizesToEmpty() async {
        // An empty unit can't tile to any positive target.
        let inf = Inference(
            tokenize: { _, _ in [] },
            resetContext: {}, resetSampler: {}, prefill: { _ in },
            decode: { $0 }, generate: { count, _ in LlamaGeneration(tokens: count, lastSampledAt: .now) },
            chatCompletion: { _, _, _, _, _ in EvalGeneration(text: "", stopReason: .unknown, stopDetail: nil, completionTokens: nil) })
        await #expect(throws: RuntimeError.self) {
            try await LlamaExecute.prefillThroughput(
                prefill512, inf, readinessGate: ready, observer: noObserver())
        }
    }
}

private nonisolated final class RefusingContext: RunContext, @unchecked Sendable {
    var datasets: [Dataset] = []

    func fetchDataset(_ dataset: Dataset) async throws -> Data {
        datasets.append(dataset)
        throw RuntimeError.engine("no corpus server in this test")
    }

    var datasetsDir: URL { FileManager.default.temporaryDirectory }

    /// Never opened: the cell stops at its corpus.
    let evalCompletions = EvalCompletionsStore(root: FileManager.default.temporaryDirectory
        .appendingPathComponent("unused-\(UUID().uuidString)", isDirectory: true))
}
