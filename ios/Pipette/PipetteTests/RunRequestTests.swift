import Foundation
import Testing

@testable import Pipette

/// `RunRequest.runtimeFlagsRef`: plan-types' `runtime_flags_ref`, where an engine's overlay
/// starts and a request carrying another cell's flags is refused.
struct RunRequestTests {
    /// The runtime comes from `thisBuild`, as both in-app constructions derive it. Flags are
    /// keyed on the *declared* axes, so the bound half is immaterial here and the pair is
    /// the spec twice.
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

    /// A run carrying no flags starts from the cell's axes with every knob unset: the
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
            flags: .decodeThroughputLlamacppIosPipetteGgufText(
                numberGpuLayers: 12, ctxSize: 2048, nUbatch: nil, threads: nil, swaFull: nil, loadMode: nil)).runtimeFlagsRef()

        #expect(ref.numberGpuLayers == 12)
        #expect(ref.ctxSize == 2048)
        #expect(ref.nUbatch == nil, "an unset knob stays unset until an engine overlays it")
    }

    /// Flags belonging to another runtime family are refused. The backstop upstream keeps
    /// too: nothing in the app builds this, since both constructions derive the flags from
    /// the runtime they bind.
    @Test func flagsForAnotherRuntimeAreRefused() throws {
        let mismatched = request(try ggufModel(), flags: .decodeThroughputMlxIosPipetteMlx(nUbatch: 256))

        #expect(throws: RunRequestError.runtimeFlagsForAnotherCell(
            flags: .decodeThroughputMlxIosPipetteMlx(nUbatch: 256),
            cell: (.decodeThroughput, .llamacppIosPipette, .ggufText))) {
            _ = try mismatched.runtimeFlagsRef()
        }
    }

    /// A family-only comparison would let this through, which is why a variant carries all
    /// three axes.
    @Test func flagsForAnotherBenchmarkAreRefused() throws {
        let req = request(try ggufModel(), flags: .evalLlamacppIosPipetteGgufText(
            numberGpuLayers: nil, ctxSize: nil, nUbatch: nil, threads: nil, swaFull: nil, loadMode: nil))

        #expect(throws: RunRequestError.runtimeFlagsForAnotherCell(
            flags: .evalLlamacppIosPipetteGgufText(
                numberGpuLayers: nil, ctxSize: nil, nUbatch: nil, threads: nil, swaFull: nil,
                loadMode: nil),
            cell: (.decodeThroughput, .llamacppIosPipette, .ggufText))) {
            _ = try req.runtimeFlagsRef()
        }
    }

    /// Same for the model axis: gguf-text flags on a gguf-vision cell.
    @Test func flagsForAnotherModelAreRefused() throws {
        let req = request(try visionModel(), flags: .decodeThroughputLlamacppIosPipetteGgufText(
            numberGpuLayers: nil, ctxSize: nil, nUbatch: nil, threads: nil, swaFull: nil, loadMode: nil))

        #expect(throws: RunRequestError.runtimeFlagsForAnotherCell(
            flags: .decodeThroughputLlamacppIosPipetteGgufText(
                numberGpuLayers: nil, ctxSize: nil, nUbatch: nil, threads: nil, swaFull: nil,
                loadMode: nil),
            cell: (.decodeThroughput, .llamacppIosPipette, .ggufVision))) {
            _ = try req.runtimeFlagsRef()
        }
    }

    /// Apple Foundation has no flags variant upstream either, so its ref resolves to
    /// nothing rather than to an empty set of knobs.
    @Test func appleFoundationResolvesToNoVariant() throws {
        let ref = try request(.appleFoundationText, flags: nil).runtimeFlagsRef()

        #expect(ref.runtimeType == .appleFoundation)
        #expect(throws: RuntimeFlagsError.self) { _ = try ref.resolve() }
    }

    // MARK: - benchmarkFlagsRef

    /// A run carrying no gate starts from the cell's axes with the block unset, which is
    /// what the caller fills in from the policy that actually applied.
    @Test func aRequestWithoutBenchmarkFlagsResolvesToTheCellsAxes() throws {
        let ref = try request(try ggufModel(), flags: nil).benchmarkFlagsRef()

        #expect(ref.benchmarkType == .decodeThroughput)
        #expect(ref.runtimeType == .llamacppIosPipette)
        #expect(ref.modelType == .ggufText)
        #expect(ref.readiness == nil)
    }

    /// An authored gate survives the flattening, so the caller resolves onto what the cell
    /// asked for rather than replacing it.
    @Test func anAuthoredGateFlattensOntoTheAxes() throws {
        let model = try ggufModel()
        let req = RunRequest(
            runtime: .alreadyBound(Runtime.thisBuild(for: model)),
            model: .alreadyBound(model),
            runtimeFlags: nil,
            benchmarkFlags: .decodeThroughputLlamacppIosPipetteGgufText(
                readiness: ReadinessOverrides(maxWaitSecs: 1800, skipThermal: nil)),
            benchmark: .decodeThroughput(DecodeThroughputBenchmark(
                benchmarkId: BenchmarkId(validated: "decode_throughput_512_100"),
                parameterPrefillTokens: 512,
                parameterDecodeTokens: 100)))

        #expect(try req.benchmarkFlagsRef().readiness?.maxWaitSecs == 1800)
    }

    /// A gate belonging to another cell is refused, as the runtime group's is: the backstop
    /// upstream keeps for a request built by hand.
    @Test func benchmarkFlagsForAnotherCellAreRefused() throws {
        let model = try ggufModel()
        let mismatched = RunRequest(
            runtime: .alreadyBound(Runtime.thisBuild(for: model)),
            model: .alreadyBound(model),
            runtimeFlags: nil,
            // A prefill cell's gate, on a decode cell.
            benchmarkFlags: .prefillThroughputLlamacppIosPipetteGgufText(readiness: nil),
            benchmark: .decodeThroughput(DecodeThroughputBenchmark(
                benchmarkId: BenchmarkId(validated: "decode_throughput_512_100"),
                parameterPrefillTokens: 512,
                parameterDecodeTokens: 100)))

        #expect(throws: RunRequestError.benchmarkFlagsForAnotherCell(
            flags: .prefillThroughputLlamacppIosPipetteGgufText(readiness: nil),
            cell: (.decodeThroughput, .llamacppIosPipette, .ggufText))) {
            _ = try mismatched.benchmarkFlagsRef()
        }
    }

    /// `RunCell.dispatchRun` logs the whole request, so a token in its description would land
    /// in every run's log (plan-types' `debug_redacts_the_model_auth_token`). No `<redacted>`
    /// marker to look for: `HfRepo` renders as its coordinate, with no place for the token.
    @Test func theDispatchLogPublishesNoAuthToken() throws {
        let token = "hf_tokenthatmustnotescape"
        var repo = try HfRepo.parseOrgRepo("liquid-ai/gated")
        repo.authToken = try AuthToken(token)
        let gated = Model.ggufText(GgufText(source: .huggingFace(
            repo: repo, path: try RepoSubpath("model.gguf"), sha256: nil)))

        let dumped = "\(request(gated, flags: nil))"

        #expect(!dumped.contains(token), "the dispatch log would publish the token")
        #expect(dumped.contains("liquid-ai/gated"), "the line must still name the model")
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
