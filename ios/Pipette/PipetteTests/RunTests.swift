import Foundation
import Testing

@testable import Pipette

/// `RunRequest` as mgmt plan-types' `run.rs` defines it: what an engine is handed for one
/// cell, and the flat forms of its flag groups an engine starts from and converts back.
struct RunTests {
    private func request(
        _ model: Model, flags: RuntimeFlags? = nil, benchmarkFlags: BenchmarkFlags? = nil,
        modelFlags: ModelFlags? = nil,
        benchmark: BenchmarkDefinition = .decodeThroughput(DecodeThroughputBenchmark(
            benchmarkId: BenchmarkId(validated: "decode_throughput_512_100"),
            parameterPrefillTokens: 512,
            parameterDecodeTokens: 100))
    ) -> RunRequest {
        RunRequest(
            runtime: .alreadyBound(Runtime.thisBuild(for: model)),
            model: .alreadyBound(model),
            runtimeFlags: flags,
            modelFlags: modelFlags,
            benchmarkFlags: benchmarkFlags,
            benchmark: benchmark)
    }

    private func ggufModel() throws -> Model { try ggufTextSpec("org/a-GGUF", "a-Q4_0.gguf") }

    private func visionModel() throws -> Model {
        try ggufVisionSpec("org/vl-GGUF", "vl-Q4_0.gguf", "mmproj-f16.gguf")
    }

    private func decodeFlags(numberGpuLayers: UInt32? = nil, ctxSize: UInt32? = nil) -> RuntimeFlags {
        .decodeThroughputLlamacppIosPipetteGgufText(
            numberGpuLayers: numberGpuLayers, ctxSize: ctxSize, nUbatch: nil, threads: nil,
            swaFull: nil, loadMode: nil)
    }

    // MARK: - runtimeFlagsRef

    /// The crate's `RuntimeFlagRef::new`.
    @Test func aRequestWithoutFlagsResolvesToTheCellsAxesUnset() throws {
        let ref = try request(try ggufModel()).runtimeFlagsRef()

        #expect(ref == RuntimeFlagRef(benchmarkType: .decodeThroughput,
                                      runtimeType: .llamacppIosPipette, modelType: .ggufText))
    }

    /// Authored values survive the flattening, so an engine overlays onto what the cell
    /// asked for rather than replacing it.
    @Test func authoredFlagsFlattenOntoTheAxes() throws {
        let ref = try request(try ggufModel(), flags: decodeFlags(numberGpuLayers: 12, ctxSize: 2048))
            .runtimeFlagsRef()

        #expect(ref.numberGpuLayers == 12)
        #expect(ref.ctxSize == 2048)
        #expect(ref.nUbatch == nil, "an unset knob stays unset until an engine overlays it")
    }

    /// Refused on any axis (runtime, benchmark or model), the backstop the crate's
    /// `runtime_flags_ref` keeps. A family-only comparison would let the benchmark case
    /// through, which is why a variant carries all three axes.
    @Test func flagsForAnotherCellAreRefused() throws {
        let gguf = try ggufModel()
        #expect(throws: RunRequestError.runtimeFlagsForAnotherCell(
            flags: .decodeThroughputMlxIosPipetteMlx(nUbatch: 256),
            cell: (.decodeThroughput, .llamacppIosPipette, .ggufText))) {
            _ = try request(gguf, flags: .decodeThroughputMlxIosPipetteMlx(nUbatch: 256)).runtimeFlagsRef()
        }
        #expect(throws: RunRequestError.runtimeFlagsForAnotherCell(
            flags: .evalLlamacppIosPipetteGgufText(
                numberGpuLayers: nil, ctxSize: nil, nUbatch: nil, threads: nil, swaFull: nil,
                loadMode: nil),
            cell: (.decodeThroughput, .llamacppIosPipette, .ggufText))) {
            _ = try request(gguf, flags: .evalLlamacppIosPipetteGgufText(
                numberGpuLayers: nil, ctxSize: nil, nUbatch: nil, threads: nil, swaFull: nil,
                loadMode: nil)).runtimeFlagsRef()
        }
        #expect(throws: RunRequestError.runtimeFlagsForAnotherCell(
            flags: decodeFlags(),
            cell: (.decodeThroughput, .llamacppIosPipette, .ggufVision))) {
            _ = try request(try visionModel(), flags: decodeFlags()).runtimeFlagsRef()
        }
    }

    /// Apple Foundation has no flags variant upstream either, so its ref resolves to
    /// nothing rather than to an empty set of knobs.
    @Test func appleFoundationResolvesToNoVariant() throws {
        let ref = try request(.appleFoundationText).runtimeFlagsRef()

        #expect(ref.runtimeType == .appleFoundation)
        #expect(throws: RuntimeFlagsError.self) { _ = try ref.resolve() }
    }

    /// A ref round-trips through a variant unchanged, which is what lets an engine hand
    /// back the same authored values it was given.
    @Test func aRefRoundTripsThroughItsVariant() throws {
        let ref = RuntimeFlagRef(benchmarkType: .eval, runtimeType: .llamacppIosPipette,
                                 modelType: .ggufText, numberGpuLayers: 1, ctxSize: 2, nUbatch: 3)

        #expect(RuntimeFlagRef(try ref.resolve()) == ref)
    }

    // MARK: - benchmarkFlagsRef

    /// No gate named: the cell's axes with no readiness block, as the crate's
    /// `benchmark_flags_ref` builds it.
    @Test func aRequestWithoutBenchmarkFlagsResolvesToTheCellsAxesUnset() throws {
        let ref = try request(try ggufModel()).benchmarkFlagsRef()

        #expect(ref == BenchmarkFlagRef(benchmarkType: .decodeThroughput,
                                        runtimeType: .llamacppIosPipette, modelType: .ggufText))
    }

    @Test func aCarriedGateFlattensOntoItsCell() throws {
        let gate = ReadinessOverrides(maxWaitSecs: 1800, skipThermal: true)
        let ref = try request(try ggufModel(),
                              benchmarkFlags: .decodeThroughputLlamacppIosPipetteGgufText(readiness: gate))
            .benchmarkFlagsRef()

        #expect(ref.readiness == gate)
        #expect(ref.benchmarkType == .decodeThroughput)
    }

    @Test func benchmarkFlagsForAnotherCellAreRefused() throws {
        #expect(throws: RunRequestError.benchmarkFlagsForAnotherCell(
            flags: .decodeThroughputMlxIosPipetteMlx(readiness: nil),
            cell: (.decodeThroughput, .llamacppIosPipette, .ggufText))) {
            _ = try request(try ggufModel(),
                            benchmarkFlags: .decodeThroughputMlxIosPipetteMlx(readiness: nil))
                .benchmarkFlagsRef()
        }
    }

    // MARK: - modelFlags

    /// The cell's generation settings ride on the request, and a request built without
    /// them carries none.
    @Test func modelFlagsRideOnTheRequest() throws {
        let gguf = try ggufModel()
        #expect(request(gguf).modelFlags == nil)
        let eval = BenchmarkDefinition.eval(EvalBenchmark(
            benchmarkId: BenchmarkId(validated: "eval_x"),
            parameterEvalId: EvalId("x"),
            parameterDatasetName: "d",
            parameterMaxTokens: 8,
            parameterMcqChoices: nil,
            samples: nil))
        let carried = request(gguf, modelFlags: .evalGgufText(enableThinking: nil), benchmark: eval)
        #expect(carried.modelFlags == .evalGgufText(enableThinking: nil))
        #expect(carried.modelFlags?.matches(benchmark: carried.benchmark.benchmarkType,
                                            model: carried.model.declared) == true)
    }

    // MARK: - Redaction

    /// `debug_redacts_the_model_auth_token`: a request rendered for a log must not publish
    /// the token its model carries, on either half, whatever the model kind.
    @Test func renderingARequestRedactsTheModelsToken() throws {
        let token = "hf_tokenthatmustnotescape"
        var repo = try HfRepo.parseOrgRepo("liquid-ai/gated")
        repo.authToken = try AuthToken(token)
        let models: [Model] = [
            .ggufText(GgufText(source: .huggingFace(repo: repo, path: try RepoSubpath("model.gguf"), sha256: nil))),
            .ggufVision(GgufVision(source: .huggingFace(
                repo: repo, model: try RepoSubpath("model.gguf"), modelSha256: nil,
                mmproj: try RepoSubpath("mmproj.gguf"), mmprojSha256: nil))),
            .mlx(Mlx(source: .huggingFace(repo: repo, prefix: nil))),
        ]
        for model in models {
            let req = request(model)
            #expect(!String(describing: req).contains(token), "\(ModelType.of(model))")
            #expect(!String(reflecting: req).contains(token), "\(ModelType.of(model))")
        }
    }

    /// As the crate reads it: zero forwards are not written, `verify_steps` is the older
    /// name, and a record from before either field reads with both defaulted.
    @Test func anObservationReadsTheCratesOlderRecords() throws {
        let observation = TurnObservation(
            sampleId: "s1", turn: 0, textSha256: VlSpecDecode.textSha256("x"),
            generatedTokens: 1, draftedTokens: 0, acceptedTokens: 0, countedForwards: 0,
            endedOnEos: false, promptMs: 0, predictedMs: 1, ttftMs: 0, decodeMs: 0, e2eMs: 1)
        var json = try #require(try JSONSerialization.jsonObject(
            with: JSONEncoder().encode(observation)) as? [String: Any])
        #expect(json["counted_forwards"] == nil)
        #expect(json["ended_on_eos"] as? Bool == false)

        json["verify_steps"] = 5
        #expect(try JSONDecoder().decode(
            TurnObservation.self, from: JSONSerialization.data(withJSONObject: json))
            .countedForwards == 5)

        json["verify_steps"] = nil
        json["ended_on_eos"] = nil
        let old = try JSONDecoder().decode(
            TurnObservation.self, from: JSONSerialization.data(withJSONObject: json))
        #expect(old.countedForwards == 0)
        #expect(!old.endedOnEos)
    }
}
