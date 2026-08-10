import Foundation
import Testing

@testable import Pipette

/// `RunRequest.runtimeFlagsRef` and the trip back — the crate's
/// `runtime_flags_ref` → engine overlay → `RunResponse::runtime_flags`.
///
/// The property under test is that a result reports the settings the run *used*: a cell
/// that named none must read back the engine's own, not the unset it carried.
struct RuntimeFlagsRoundTripTests {
    /// The runtime comes from `thisBuild`, as both in-app constructions derive it — which
    /// is also why an axis mismatch cannot arise there.
    /// Flags are keyed on the *declared* axes, so the bound half is immaterial here and
    /// the pair is the spec twice.
    private func request(_ model: Model, flags: RuntimeFlags?) -> RunRequest {
        RunRequest(
            runtime: .alreadyBound(Runtime.thisBuild(for: model)),
            model: .alreadyBound(model),
            runtimeFlags: flags,
            benchmarkFlags: nil,
            benchmark: .decodeThroughput(benchmarkId: "decode_throughput_512_100",
                                         prefillTokens: 512, decodeTokens: 100))
    }

    private func ggufModel() throws -> Model {
        try ggufTextSpec("org/a-GGUF", "a-Q4_0.gguf")
    }

    /// A `runtime_flags` string parsed back into its knobs.
    /// The numeric knobs of a submission value. `swa_full` is dropped rather than
    /// compared here: a JSON boolean bridges to `1` through this cast, so asserting it as a
    /// number would pass whatever the value was. Its own assertion reads the raw JSON.
    private func numericKnobs(_ json: String) throws -> [String: Int] {
        let object = try JSONSerialization.jsonObject(with: Data(json.utf8))
        var knobs = try #require(object as? [String: Int])
        knobs.removeValue(forKey: "swa_full")
        return knobs
    }

    private func mlxModel() throws -> Model { try mlxSpec("org/m-MLX-4bit") }

    private func visionModel() throws -> Model {
        try ggufVisionSpec("org/vl-GGUF", "vl-Q4_0.gguf", "mmproj-f16.gguf")
    }

    /// A run carrying no flags starts from the cell's axes with every knob unset — the
    /// crate's `RuntimeFlagRef::new`.
    @Test func aRequestWithoutFlagsResolvesToTheCellsAxesUnset() throws {
        let ref = try request(try ggufModel(), flags: nil).runtimeFlagsRef()

        #expect(ref.benchmarkType == .decodeThroughput)
        #expect(ref.runtimeType == .llamacppIosPipette)
        #expect(ref.modelType == .ggufText)
        #expect(ref.numberGpuLayers == nil)
        #expect(ref.ctxSize == nil)
        #expect(ref.nUbatch == nil)
    }

    /// Authored values survive the flattening, so an engine overlays onto what the cell
    /// asked for rather than replacing it.
    @Test func authoredFlagsFlattenOntoTheAxes() throws {
        let ref = try request(
            try ggufModel(),
            flags: .decodeLlamacppIosPipetteGgufText(
                numberGpuLayers: 12, ctxSize: 2048, nUbatch: nil, threads: nil, swaFull: nil)).runtimeFlagsRef()

        #expect(ref.numberGpuLayers == 12)
        #expect(ref.ctxSize == 2048)
        #expect(ref.nUbatch == nil, "an unset knob stays unset until an engine overlays it")
    }

    /// Flags belonging to another runtime family are refused. The backstop upstream keeps
    /// too: nothing in the app builds this, since both constructions derive the flags from
    /// the runtime they bind.
    @Test func flagsForAnotherRuntimeAreRefused() throws {
        let mismatched = request(try ggufModel(), flags: .decodeMlxIosPipetteMlx(nUbatch: 256))

        #expect(throws: RuntimeFlagsAxisMismatch(
            carried: (.decodeThroughput, .mlxIosPipette, .mlx),
            cell: (.decodeThroughput, .llamacppIosPipette, .ggufText))) {
            _ = try mismatched.runtimeFlagsRef()
        }
    }

    /// Same runtime family, another benchmark: refused. A family-only comparison let this
    /// through, and it is the reason a variant carries all three axes.
    @Test func flagsForAnotherBenchmarkAreRefused() throws {
        let req = request(try ggufModel(), flags: .evalLlamacppIosPipetteGgufText(
            numberGpuLayers: nil, ctxSize: nil, nUbatch: nil, threads: nil, swaFull: nil))

        #expect(throws: RuntimeFlagsAxisMismatch(
            carried: (.eval, .llamacppIosPipette, .ggufText),
            cell: (.decodeThroughput, .llamacppIosPipette, .ggufText))) {
            _ = try req.runtimeFlagsRef()
        }
    }

    /// Same for the model axis: gguf-text flags on a gguf-vision cell.
    @Test func flagsForAnotherModelAreRefused() throws {
        let req = request(try visionModel(), flags: .decodeLlamacppIosPipetteGgufText(
            numberGpuLayers: nil, ctxSize: nil, nUbatch: nil, threads: nil, swaFull: nil))

        #expect(throws: RuntimeFlagsAxisMismatch(
            carried: (.decodeThroughput, .llamacppIosPipette, .ggufText),
            cell: (.decodeThroughput, .llamacppIosPipette, .ggufVision))) {
            _ = try req.runtimeFlagsRef()
        }
    }

    /// The llama overlay keeps what the cell set and fills the rest — the crate's
    /// `for_bench_keeps_the_cells_values`. The record is the whole picture of what ran, not
    /// just what the engine changed.
    @Test func theLlamaOverlayKeepsTheCellsValues() throws {
        let req = request(try ggufModel(),
                          flags: .decodeLlamacppIosPipetteGgufText(
                numberGpuLayers: 12, ctxSize: nil, nUbatch: nil, threads: nil, swaFull: nil))

        #expect(try LlamaRuntimeFlags.forRun(req) == .decodeLlamacppIosPipetteGgufText(
            numberGpuLayers: 12, ctxSize: 612,   // 512 prefill + 100 decode
            nUbatch: LlamaCpp.defaultNUbatch, threads: LlamaCpp.defaultThreads,
            swaFull: LlamaCpp.defaultSwaFull))
    }

    /// A cell that named nothing reads back every default, so the load and the record are
    /// the same values.
    @Test func theLlamaOverlayFillsAnEmptyCell() throws {
        let flags = try LlamaRuntimeFlags.forRun(request(try ggufModel(), flags: nil))

        #expect(flags == .decodeLlamacppIosPipetteGgufText(
            numberGpuLayers: LlamaCpp.defaultNumberGpuLayers,
            ctxSize: 612, nUbatch: LlamaCpp.defaultNUbatch,
            threads: LlamaCpp.defaultThreads, swaFull: LlamaCpp.defaultSwaFull))
    }

    /// A cell that names `threads` runs with it and reports it — the setting exists to be
    /// swept, so a run that silently kept the device default would answer another question.
    @Test func aCellsThreadCountSurvivesTheOverlay() throws {
        let req = request(try ggufModel(),
                          flags: .decodeLlamacppIosPipetteGgufText(
                              numberGpuLayers: nil, ctxSize: nil, nUbatch: nil, threads: 2, swaFull: nil))

        let resolved = try LlamaRuntimeFlags.forRun(req)

        #expect(resolved.threads == 2)
        #expect(try resolved.submissionValue().contains("\"threads\":2"))
    }

    /// The load and the submission read the same value, so neither can describe a setting
    /// the other did not use: the engine reads these knobs and `SubmissionRef` renders them.
    @Test func theLoadAndTheRecordReadTheSameKnobs() throws {
        let flags = try LlamaRuntimeFlags.forRun(
            request(try ggufModel(), flags: .decodeLlamacppIosPipetteGgufText(
                numberGpuLayers: 7, ctxSize: 512, nUbatch: 64, threads: nil, swaFull: nil)))

        #expect((flags.numberGpuLayers, flags.ctxSize, flags.nUbatch) == (7, 512, 64))
        // Compared as an object, not as a string: nothing sorts the keys, so the field
        // order is the encoder's business. The crate asserts the same way, against a
        // `serde_json::Value`.
        // `threads` is absent from the cell, so the engine's own value is reported —
        // the record is what ran, not what was authored.
        #expect(try numericKnobs(flags.submissionValue())
            == ["number_gpu_layers": 7, "ctx_size": 512, "n_ubatch": 64,
                "threads": Int(LlamaCpp.defaultThreads)])
        #expect(try flags.submissionValue().contains("\"swa_full\":false"))
    }

    /// The sliding-window cache is off unless a cell asks for it, matching what
    /// `llama-bench` and `llama-server` pin — and the value is reported either way, so a
    /// result taken under the full-size cache is told apart from one taken without it.
    /// Both halves are pinned: flipping the default silently would re-baseline every
    /// peak-memory figure on a SWA model.
    @Test func theSlidingWindowCacheIsOffUnlessTheCellAsksForIt() throws {
        let defaulted = try LlamaRuntimeFlags.forRun(request(try ggufModel(), flags: nil))
        #expect(defaulted.swaFull == false)
        #expect(try defaulted.submissionValue().contains("\"swa_full\":false"))

        let asked = try LlamaRuntimeFlags.forRun(
            request(try ggufModel(), flags: .decodeLlamacppIosPipetteGgufText(
                numberGpuLayers: nil, ctxSize: nil, nUbatch: nil, threads: nil, swaFull: true)))
        #expect(asked.swaFull == true)
        #expect(try asked.submissionValue().contains("\"swa_full\":true"))
    }

    /// The context a cell that named none runs with, sized from its benchmark exactly as
    /// `pipette-llamacpp` sizes it — `llama-bench` fits its window to the workload, and the
    /// server cells derive `default_ctx_size` the same way. Asserted per benchmark because
    /// the arithmetic moved here from the job layer, and a silent change to it moves every
    /// peak-memory reading.
    @Test(arguments: [
        (BenchmarkDefinition.prefillThroughput(benchmarkId: "p", prefillTokens: 512), UInt32(512)),
        (.maxMemoryUsage(benchmarkId: "m", prefillTokens: 512), UInt32(513)),
        (.decodeThroughput(benchmarkId: "d", prefillTokens: 512, decodeTokens: 100), UInt32(612)),
        (.endToEndLatency(benchmarkId: "e", prefillTokens: 256, decodeTokens: 64), UInt32(320)),
        (.eval(benchmarkId: "v", evalId: EvalId("ifbench"), datasetName: "ifbench",
               maxTokens: 256, mcqChoices: nil, samples: nil), UInt32(8448)),
        // 224/14 × 224/14 = 256 image tokens + text + decode, under the 8192 floor.
        (.vlThroughput(benchmarkId: "vl", imageWidth: 224, imageHeight: 224,
                       textTokens: 32, decodeTokens: 64), UInt32(8192)),
    ])
    func theEngineSizesTheContextFromTheBenchmark(
        _ benchmark: BenchmarkDefinition, _ expected: UInt32
    ) {
        #expect(LlamaRuntimeFlags.contextSize(for: benchmark) == expected)
    }

    /// A cell that names a context keeps it: the derivation answers only what was unset.
    @Test func anAuthoredContextSurvivesTheEngineDerivation() throws {
        let req = request(try ggufModel(), flags: .decodeLlamacppIosPipetteGgufText(
            numberGpuLayers: nil, ctxSize: 333, nUbatch: nil, threads: nil, swaFull: nil))

        #expect(try LlamaRuntimeFlags.forRun(req).ctxSize == 333)
    }

    /// MLX's one setting behaves the same way, and the two llama knobs stay absent — an
    /// MLX variant carries neither, so reporting one would look authored.
    @Test func theMlxOverlayReportsOnlyThePrefillChunk() throws {
        let ran = try MLXRuntimeFlags.forRun(request(try mlxModel(), flags: nil))

        #expect(ran == .decodeMlxIosPipetteMlx(nUbatch: MLXRuntime.defaultPrefillChunk))
        #expect(ran.numberGpuLayers == nil)
        #expect(ran.ctxSize == nil)
    }

    /// An authored chunk survives the overlay, as it does on the llama side.
    @Test func theMlxOverlayKeepsAnAuthoredChunk() throws {
        let ran = try MLXRuntimeFlags.forRun(request(try mlxModel(), flags: .decodeMlxIosPipetteMlx(nUbatch: 128)))

        #expect(ran.nUbatch == 128)
    }

    /// plan-types defines gguf-vision flags for `vl_throughput` alone, so a vision model on
    /// a timing benchmark is not a cell it describes — and the refusal names the triple, as
    /// the crate error does. `ClientRunSpec.validated` refuses the same cell at claim time; this is
    /// the guard for a hand-authored `spec=` or deep link, which would otherwise report
    /// base-weight timings under a vision descriptor.
    @Test func aVisionCellOnATimingBenchmarkIsRefused() throws {
        let req = request(try visionModel(), flags: nil)

        let failure = #expect(throws: RuntimeFlagResolveError.self) {
            _ = try LlamaRuntimeFlags.forRun(req)
        }
        guard case let .noSuchCombination(benchmark, _, model) = try #require(failure) else {
            Issue.record("expected noSuchCombination"); return
        }
        #expect(model == .ggufVision)
        #expect(benchmark == .decodeThroughput)
        #expect(failure?.localizedDescription.contains("gguf_vision") == true)
    }

    /// Apple Foundation has no flags variant upstream either, so its ref resolves to
    /// nothing rather than to an empty set of knobs.
    @Test func appleFoundationResolvesToNoVariant() throws {
        let ref = try request(.appleFoundationText, flags: nil).runtimeFlagsRef()

        #expect(ref.runtimeType == .appleFoundation)
        #expect(throws: RuntimeFlagResolveError.self) { _ = try ref.resolve() }
    }

    /// A ref round-trips through a variant unchanged, which is what lets an engine hand
    /// back the same authored values it was given.
    @Test func aRefRoundTripsThroughItsVariant() throws {
        let ref = RuntimeFlagRef(benchmarkType: .eval, runtimeType: .llamacppIosPipette,
                                 modelType: .ggufText, numberGpuLayers: 1, ctxSize: 2, nUbatch: 3)

        // No axes passed back in: the variant carries them, as the crate's
        // `From<RuntimeFlags>` reads them off it.
        #expect(RuntimeFlagRef(try ref.resolve()) == ref)
    }
}
