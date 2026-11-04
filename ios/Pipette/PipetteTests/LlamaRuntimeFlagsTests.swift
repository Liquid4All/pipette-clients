import Foundation
import Testing

import llama
@testable import Pipette

/// The llama.cpp engine's overlay onto a cell's flags, the counterpart of the crate's
/// `runtimes::get_load_mode`. A result reports the settings the run used: a cell that
/// named none reads back the engine's own, not the unset it carried.
struct LlamaRuntimeFlagsTests {
    /// The runtime comes from `thisBuild`, as both in-app constructions derive it. Flags are
    /// keyed on the declared axes, so the bound half is immaterial and the pair is the spec
    /// twice.
    private func request(_ model: Model, flags: RuntimeFlags?) -> RunRequest {
        RunRequest(
            runtime: .alreadyBound(Runtime.thisBuild(for: model)),
            model: .alreadyBound(model),
            runtimeFlags: flags,
            benchmarkFlags: nil,
            benchmark: .decodeThroughput(DecodeThroughputBenchmark(
                benchmarkId: BenchmarkId(validated: "decode_throughput_512_100"),
                parameterPrefillTokens: 512,
                parameterDecodeTokens: 100)))
    }

    private func ggufModel() throws -> Model {
        try ggufTextSpec("org/a-GGUF", "a-Q4_0.gguf")
    }

    private func visionModel() throws -> Model {
        try ggufVisionSpec("org/vl-GGUF", "vl-Q4_0.gguf", "mmproj-f16.gguf")
    }

    private func decodeFlags(
        numberGpuLayers: UInt32? = nil, ctxSize: UInt32? = nil, nUbatch: UInt32? = nil,
        threads: UInt32? = nil, swaFull: Bool? = nil, loadMode: LlamacppLoadMode? = nil
    ) -> RuntimeFlags {
        .decodeThroughputLlamacppIosPipetteGgufText(
            numberGpuLayers: numberGpuLayers, ctxSize: ctxSize, nUbatch: nUbatch,
            threads: threads, swaFull: swaFull, loadMode: loadMode)
    }

    /// `swa_full` is removed by name because a JSON bool casts to `Int`; `load_mode` is a
    /// word and drops out on its own.
    private func numericKnobs(_ json: String) throws -> [String: Int] {
        let object = try JSONSerialization.jsonObject(with: Data(json.utf8))
        let fields = try #require(object as? [String: Any])
        var knobs = fields.compactMapValues { $0 as? Int }
        knobs.removeValue(forKey: "swa_full")
        return knobs
    }

    /// The record is the whole picture of what ran, not just what the engine changed.
    @Test func theOverlayKeepsTheCellsValues() throws {
        let req = request(try ggufModel(), flags: decodeFlags(numberGpuLayers: 12))

        #expect(try LlamaRuntimeFlags.forRun(req) == decodeFlags(
            numberGpuLayers: 12, ctxSize: 612,   // 512 prefill + 100 decode
            nUbatch: LlamaCpp.defaultNUbatch, threads: LlamaCpp.defaultThreads,
            swaFull: LlamaCpp.defaultSwaFull, loadMode: LlamaCpp.defaultLoadMode))
    }

    @Test func theOverlayFillsAnEmptyCell() throws {
        #expect(try LlamaRuntimeFlags.forRun(request(try ggufModel(), flags: nil)) == decodeFlags(
            numberGpuLayers: LlamaCpp.defaultNumberGpuLayers, ctxSize: 612,
            nUbatch: LlamaCpp.defaultNUbatch, threads: LlamaCpp.defaultThreads,
            swaFull: LlamaCpp.defaultSwaFull, loadMode: LlamaCpp.defaultLoadMode))
    }

    @Test func aCellsLoadModeSurvivesTheOverlay() throws {
        #expect(try LlamaRuntimeFlags.forRun(request(try ggufModel(), flags: decodeFlags(loadMode: .mmap)))
            .loadMode == .mmap)
        #expect(try LlamaRuntimeFlags.forRun(request(try ggufModel(), flags: nil)).loadMode
                == LlamaCpp.defaultLoadMode)
        #expect(LlamaCpp.defaultLoadMode == LlamacppLoadMode.read,
                "weights in anonymous RAM unless a cell says otherwise, as the CLI benchmarks load them")
    }

    /// The setting exists to be swept, so a run that silently kept the device default would
    /// answer another question.
    @Test func aCellsThreadCountSurvivesTheOverlay() throws {
        let resolved = try LlamaRuntimeFlags.forRun(request(try ggufModel(), flags: decodeFlags(threads: 2)))

        #expect(resolved.threads == 2)
        #expect(try resolved.submissionValue().contains("\"threads\":2"))
    }

    @Test func theLoadAndTheRecordReadTheSameKnobs() throws {
        let flags = try LlamaRuntimeFlags.forRun(
            request(try ggufModel(), flags: decodeFlags(numberGpuLayers: 7, ctxSize: 512, nUbatch: 64)))

        #expect((flags.numberGpuLayers, flags.ctxSize, flags.nUbatch) == (7, 512, 64))
        // `threads` is absent from the cell, so the engine's own value is reported.
        #expect(try numericKnobs(flags.submissionValue())
            == ["number_gpu_layers": 7, "ctx_size": 512, "n_ubatch": 64,
                "threads": Int(LlamaCpp.defaultThreads)])
        #expect(try flags.submissionValue().contains("\"swa_full\":false"))
        #expect(try flags.submissionValue().contains("\"load_mode\":\"none\""))
    }

    /// `LlamaCpp.initContext` clamps the micro-batch to the context, so reporting 512 on a
    /// small context would name a prefill chunk the run never used.
    @Test func theReportedMicroBatchIsClampedToTheContext() throws {
        let clamped = try LlamaRuntimeFlags.forRun(
            request(try ggufModel(), flags: decodeFlags(ctxSize: 100, nUbatch: 512)))
        #expect(clamped.nUbatch == 100)
        #expect(clamped.ctxSize == 100)

        // `decode_throughput_512_100` derives 612, which fits the default.
        #expect(try LlamaRuntimeFlags.forRun(request(try ggufModel(), flags: nil)).nUbatch
                == LlamaCpp.defaultNUbatch)

        // An authored zero is llama's 512, which `initContext` applies for one.
        #expect(try LlamaRuntimeFlags.forRun(request(try ggufModel(), flags: decodeFlags(nUbatch: 0)))
            .nUbatch == LlamaCpp.defaultNUbatch)
    }

    /// The engine would keep llama's own default for a zero context, so running it would
    /// submit a context and a micro-batch the load never used.
    @Test func aZeroContextIsRefusedByTheEngine() throws {
        let failure = #expect(throws: KnobValueNotUsable.self) {
            _ = try LlamaRuntimeFlags.forRun(
                request(try ggufModel(), flags: decodeFlags(ctxSize: 0, nUbatch: 512)))
        }
        #expect(failure?.knob == "ctx_size")
        #expect(failure?.value == "0")
    }

    @Test func everyLoadModeSpellsItsPlanWord() throws {
        #expect(LlamacppLoadMode.allCases.map(\.rawValue)
                == ["none", "mmap", "mlock", "mmap_mlock", "direct_io"])
    }

    /// This is what the engine sets on `llama_model_params`, so a swapped pair would load one
    /// mode and report another with every other test still green.
    @Test func everyLoadModeMapsToItsLlamaDiscriminant() throws {
        #expect(LlamacppLoadMode.read.llamaValue == LLAMA_LOAD_MODE_NONE)
        #expect(LlamacppLoadMode.mmap.llamaValue == LLAMA_LOAD_MODE_MMAP)
        #expect(LlamacppLoadMode.mlock.llamaValue == LLAMA_LOAD_MODE_MLOCK)
        #expect(LlamacppLoadMode.mmapMlock.llamaValue == LLAMA_LOAD_MODE_MMAP_MLOCK)
        #expect(LlamacppLoadMode.directIo.llamaValue == LLAMA_LOAD_MODE_DIRECT_IO)
    }

    /// Off by default, matching what `llama-bench` and `llama-server` pin.
    @Test func theSlidingWindowCacheIsOffUnlessTheCellAsksForIt() throws {
        let defaulted = try LlamaRuntimeFlags.forRun(request(try ggufModel(), flags: nil))
        #expect(defaulted.swaFull == false)
        #expect(try defaulted.submissionValue().contains("\"swa_full\":false"))

        let asked = try LlamaRuntimeFlags.forRun(request(try ggufModel(), flags: decodeFlags(swaFull: true)))
        #expect(asked.swaFull == true)
        #expect(try asked.submissionValue().contains("\"swa_full\":true"))
    }

    /// Sized exactly as `pipette-llamacpp` sizes it. Asserted per benchmark because a silent
    /// change moves every peak-memory reading.
    @Test(arguments: [
        (BenchmarkDefinition.prefillThroughput(PrefillThroughputBenchmark(
            benchmarkId: BenchmarkId(validated: "p"), parameterPrefillTokens: 512)), UInt32(512)),
        (.maxMemoryUsage(MaxMemoryUsageBenchmark(
            benchmarkId: BenchmarkId(validated: "m"), parameterPrefillTokens: 512)), UInt32(513)),
        (.decodeThroughput(DecodeThroughputBenchmark(
            benchmarkId: BenchmarkId(validated: "d"), parameterPrefillTokens: 512,
            parameterDecodeTokens: 100)), UInt32(612)),
        (.endToEndLatency(EndToEndLatencyBenchmark(
            benchmarkId: BenchmarkId(validated: "e"), parameterPrefillTokens: 256,
            parameterDecodeTokens: 64)), UInt32(320)),
        (.eval(EvalBenchmark(
            benchmarkId: BenchmarkId(validated: "v"),
            parameterEvalId: EvalId("ifbench"),
            parameterDatasetName: "ifbench",
            parameterMaxTokens: 256,
            parameterMcqChoices: nil,
            samples: nil)), UInt32(8448)),
        // 224/14 × 224/14 = 256 image tokens + 32 text + 64 decode + 64 + 8 margins = 424,
        // under `vl_throughput`'s own 8192 floor.
        (.vlThroughput(VlThroughputBenchmark(
            benchmarkId: BenchmarkId(validated: "vl"),
            parameterImageWidth: 224,
            parameterImageHeight: 224,
            parameterTextTokens: 32,
            parameterDecodeTokens: 64)), UInt32(8192)),
        // No floor, as in the crate: padding would inflate the KV cache it reports.
        // 256 + its one decode token + 64 + 8.
        (.vlMaxMemory(VlMaxMemoryBenchmark(
            benchmarkId: BenchmarkId(validated: "vm"), parameterImageWidth: 224,
            parameterImageHeight: 224)), UInt32(329)),
        // 256 + 256 decode + 64 + 8.
        (.vlMaxMemory(VlMaxMemoryBenchmark(
            benchmarkId: BenchmarkId(validated: "vm_decode256"), parameterImageWidth: 224,
            parameterImageHeight: 224, parameterDecodeTokens: 256)), UInt32(584)),
        // The crate's 8192 per slot, whatever the generation cap: the phone runs one slot.
        (.vlSpecDecode(VlSpecDecodeBenchmark(
            benchmarkId: BenchmarkId(validated: "vl_spec_decode_mmspec_test_gqa_k8"),
            parameterDraftLength: 8,
            parameterMaxTokens: 2048,
            dataset: BenchmarkDefinitionTests.sampleDataset)), UInt32(8192)),
    ])
    func theEngineSizesTheContextFromTheBenchmark(
        _ benchmark: BenchmarkDefinition, _ expected: UInt32
    ) {
        #expect(LlamaRuntimeFlags.ctxSize(for: benchmark) == expected)
    }

    /// With a calibrated per-frame count (the crate's `CtxInputs::ctx_size`) neither kind is
    /// floored: the 8192 floor is for an estimate that can come up short.
    @Test func aCountedFrameSizesTheVisionContext() {
        let throughput = BenchmarkDefinition.vlThroughput(VlThroughputBenchmark(
            benchmarkId: BenchmarkId(validated: "vl_throughput_512x512_text0_decode64_img1"),
            parameterImageWidth: 512, parameterImageHeight: 512, parameterTextTokens: 0,
            parameterDecodeTokens: 64))
        let memory = BenchmarkDefinition.vlMaxMemory(VlMaxMemoryBenchmark(
            benchmarkId: BenchmarkId(validated: "vl_max_memory_256x256_text1024_img5"),
            parameterImageWidth: 256, parameterImageHeight: 256, parameterTextTokens: 1024,
            parameterNumImages: 5))

        // 256 + 64 decode + 64 + 8, against max(8192, 1296 + 64 + 64 + 8) estimated.
        #expect(LlamaRuntimeFlags.ctxSize(for: throughput, perImageTokens: 256) == 392)
        #expect(LlamaRuntimeFlags.ctxSize(for: throughput) == 8192)
        // 64 × 5 + 1024 + 1 + 64 + 8 × 5, against 324 × 5 + ... estimated.
        #expect(LlamaRuntimeFlags.ctxSize(for: memory, perImageTokens: 64) == 1449)
        #expect(LlamaRuntimeFlags.ctxSize(for: memory) == 2749)
    }

    /// The micro-batch is clamped as `LlamaCpp.initContext` clamps it, below the estimate or
    /// above it.
    @Test func aCountedFrameReachesTheRecordedFlags() throws {
        let benchmark = BenchmarkDefinition.vlMaxMemory(VlMaxMemoryBenchmark(
            benchmarkId: BenchmarkId(validated: "vl_max_memory_256x256_text0_img1"),
            parameterImageWidth: 256, parameterImageHeight: 256))
        func req(_ flags: RuntimeFlags?) throws -> RunRequest {
            let model = try visionModel()
            return RunRequest(runtime: .alreadyBound(Runtime.thisBuild(for: model)),
                              model: .alreadyBound(model), runtimeFlags: flags,
                              benchmarkFlags: nil, benchmark: benchmark)
        }

        // 64 + 1 + 64 + 8.
        let counted = try LlamaRuntimeFlags.forRun(try req(nil), perImageTokens: 64)
        #expect(counted.ctxSize == 137)
        #expect(counted.nUbatch == 137)

        // The estimate clamps the default 512 to its 324 + 1 + 64 + 8; a count above it
        // lifts the clamp: 576 + 1 + 64 + 8.
        #expect(try LlamaRuntimeFlags.forRun(try req(nil)).nUbatch == 397)
        let wide = try LlamaRuntimeFlags.forRun(try req(nil), perImageTokens: 576)
        #expect(wide.ctxSize == 649)
        #expect(wide.nUbatch == 512)

        let pinned = try LlamaRuntimeFlags.forRun(
            try req(.vlMaxMemoryLlamacppIosPipetteGgufVision(
                numberGpuLayers: nil, ctxSize: 4096, nUbatch: nil, threads: nil, swaFull: nil,
                loadMode: nil)),
            perImageTokens: 64)
        #expect(pinned.ctxSize == 4096)
    }

    @Test func anAuthoredContextSurvivesTheEngineDerivation() throws {
        #expect(try LlamaRuntimeFlags.forRun(request(try ggufModel(), flags: decodeFlags(ctxSize: 333)))
            .ctxSize == 333)
    }

    // MARK: - Eval, which the crate overlays and reports differently

    private func evalRequest(flags: RuntimeFlags? = nil) throws -> RunRequest {
        let model = try ggufModel()
        return RunRequest(
            runtime: .alreadyBound(Runtime.thisBuild(for: model)),
            model: .alreadyBound(model),
            runtimeFlags: flags,
            benchmarkFlags: nil,
            benchmark: .eval(EvalBenchmark(
                benchmarkId: BenchmarkId(validated: "ifbench"),
                parameterEvalId: EvalId("ifbench"),
                parameterDatasetName: "ifbench",
                parameterMaxTokens: 256,
                parameterMcqChoices: nil,
                samples: nil)))
    }

    /// The crate's eval arm leaves llama.cpp's mmap default alone, because a run that long
    /// amortizes the page faults.
    @Test func anEvalCellLoadsMappedWhileTheOthersPin() throws {
        #expect(try LlamaRuntimeFlags.forRun(try evalRequest()).loadMode == .mmap)
        #expect(try LlamaRuntimeFlags.forRun(request(try ggufModel(), flags: nil)).loadMode
                == LlamacppLoadMode.read)
    }

    @Test func anAuthoredLoadModeSurvivesOnEval() throws {
        let req = try evalRequest(flags: .evalLlamacppIosPipetteGgufText(
            numberGpuLayers: nil, ctxSize: nil, nUbatch: nil, threads: nil, swaFull: nil,
            loadMode: LlamacppLoadMode.read))

        #expect(try LlamaRuntimeFlags.forRun(req).loadMode == LlamacppLoadMode.read)
    }

    /// The crate reports `req.runtime_flags` unmodified for eval: what the overlay adds is
    /// the engine's own argv, not a decision the cell made.
    @Test func anEvalCellReportsTheFlagsItCarried() throws {
        let carried = RuntimeFlags.evalLlamacppIosPipetteGgufText(
            numberGpuLayers: nil, ctxSize: 4096, nUbatch: nil, threads: nil, swaFull: nil,
            loadMode: nil)
        let authored = try evalRequest(flags: carried)
        #expect(LlamaRuntimeFlags.reported(authored, ran: try LlamaRuntimeFlags.forRun(authored))
                == carried)

        let bare = try evalRequest()
        #expect(LlamaRuntimeFlags.reported(bare, ran: try LlamaRuntimeFlags.forRun(bare)) == nil)
    }

    /// For a timing cell the overlay is the load.
    @Test func aTimingCellReportsTheResolvedFlags() throws {
        let req = request(try ggufModel(), flags: nil)
        let ran = try LlamaRuntimeFlags.forRun(req)

        #expect(LlamaRuntimeFlags.reported(req, ran: ran) == ran)
    }

    /// plan-types defines no gguf-vision flags for a text timing benchmark.
    /// `ClientRunSpec.validated` refuses the cell only when the claim carries a flag
    /// group, so this guards a flagless claim or a hand-authored `spec=`, which
    /// would otherwise report base-weight timings under a vision descriptor.
    @Test func aVisionCellOnATimingBenchmarkIsRefused() throws {
        let req = request(try visionModel(), flags: nil)

        #expect(throws: RuntimeFlagsError.noSuchCombination(
            benchmark: .decodeThroughput, runtime: .llamacppIosPipette, model: .ggufVision)) {
            _ = try LlamaRuntimeFlags.forRun(req)
        }
    }
}
