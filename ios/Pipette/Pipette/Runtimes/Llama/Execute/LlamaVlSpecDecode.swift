import Foundation

// The crate's `pipette-llamacpp/src/execute/vl_spec_decode.rs`; the traversal is
// `VlSpecDecodeCorpusTurn` (`Benchmarks/Turn.swift`).

nonisolated extension LlamaExecute {
    /// Context per slot when the cell pins none: twice MMSpec `test`'s worst turn (4,110
    /// tokens, images and history included). The phone runs one slot, so it is the window.
    static let defaultCtxSizePerSlot: UInt32 = 8192

    /// The model a corpus run loads, resolved with its speculation: one case per plan model
    /// kind and speculation it can run with.
    enum LoadedModel {
        /// A `gguf_vision` model: its backbone and projector, never a drafter.
        case vision(ResolvedGgufVisionModel)
        /// A `gguf_vision_speculative` model with speculation off: the backbone and projector
        /// alone; a merged drafter stays in the file, resident and unused.
        case speculativeOff(ResolvedGgufVisionSpeculativeModel)
        case speculativeOn(ResolvedGgufVisionSpeculativeModel)

        /// Which files load depends on both, so this matches the plan's model kind itself.
        static func resolve(_ model: Model, enabled: Bool) throws -> LoadedModel {
            switch (enabled, model) {
            case (true, _):
                return .speculativeOn(try ResolvedGgufVisionSpeculativeModel(model))
            case (false, .ggufVisionSpeculative):
                return .speculativeOff(try ResolvedGgufVisionSpeculativeModel(model))
            case (false, _):
                return .vision(try ResolvedGgufVisionModel(model))
            }
        }

        var speculation: VlSpecDecode.Speculation {
            switch self {
            case .vision, .speculativeOff: .off
            case .speculativeOn: .on
            }
        }

        /// The backbone and projector every case loads.
        var vision: ResolvedGgufVisionModel {
            switch self {
            case let .vision(vision): vision
            case let .speculativeOff(model), let .speculativeOn(model): model.vision
            }
        }
    }

    static func vlSpecDecode(
        _ benchmark: VlSpecDecodeBenchmark,
        load: LoadSettings,
        model: LoadedModel,
        corpus: VlSpecDecodeCorpus,
        readinessGate: ReadinessGate,
        observer: MeasurementObserver,
        isCancelled: () -> Bool,
        progress: (BenchmarkProgress) -> Void
    ) async throws -> RunResponse {
        let vision = model.vision
        // On: the drafter, beside the weights or merged into them.
        let drafter: DrafterArrangement = switch model {
        case let .speculativeOn(model): model.draft.map { .separate(path: $0) } ?? .merged
        case .vision, .speculativeOff: .none
        }
        // Gated inside the traversal, after the load, which would spend a wait taken here.
        let (data, traversal) = try await VlSpecDecodeCorpusTurn.run(
            speculation: model.speculation, corpus: corpus,
            draftLength: benchmark.parameterDraftLength.value.magnitude,
            maxTokens: benchmark.parameterMaxTokens.value.magnitude,
            readinessGate: readinessGate, observer: observer, isCancelled: isCancelled, progress: progress
        ) {
            try openVisionChat(
                path: vision.backbone, mmprojPath: vision.mmproj, drafter: drafter,
                draftLength: benchmark.parameterDraftLength.value.magnitude,
                maxTokens: Int(benchmark.parameterMaxTokens.value),
                temperature: benchmark.parameterTemperature,
                nGpuLayers: load.numberGpuLayers, contextSize: load.ctxSize,
                nUbatch: load.nUbatch, threads: load.threads, swaFull: load.swaFull,
                loadMode: load.loadMode)
        }
        var response = RunResponse(resultData: data, runtimeOutput: RuntimeOutput())
        response.datasetDigest = corpus.digest.value
        response.outputDigest = traversal.outputDigest
        response.turns = traversal.observations
        return response
    }
}
