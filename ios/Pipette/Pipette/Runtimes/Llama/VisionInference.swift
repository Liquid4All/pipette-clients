import Foundation

/// An explicit `free()` because the chunks own the preprocessed image tensors, the cell's
/// largest allocation at high frame counts, and a rep must release them before the next.
/// A test leaves `handle` nil and supplies the counts.
nonisolated final class VisionChunks {
    /// Text, image and template scaffolding: the result's `prompt_tokens`.
    let promptTokens: Int
    let imageTokens: Int
    /// A clip's share, as `imageTokens` is a frame's.
    let audioTokens: Int
    let handle: OpaquePointer?
    private let release: () -> Void
    private var released = false

    init(promptTokens: Int, imageTokens: Int, audioTokens: Int = 0, handle: OpaquePointer? = nil,
         release: @escaping () -> Void = {}) {
        self.promptTokens = promptTokens
        self.imageTokens = imageTokens
        self.audioTokens = audioTokens
        self.handle = handle
        self.release = release
    }

    func free() {
        guard !released else { return }
        released = true
        release()
    }

    deinit { free() }
}

/// The `Inference` seam for cells that go through a projector. Separate so text engines
/// don't carry vision fields they cannot fill.
nonisolated struct VisionInference {
    var mediaMarker: String
    var tokenize: (_ text: String, _ addSpecial: Bool) throws -> [Int32]
    var applyChatTemplate: (_ userContent: String) throws -> String
    var resetContext: () -> Void
    var resetSampler: () -> Void
    /// Inside the whole request's window but outside prefill and decode, as a desktop
    /// server preprocesses within the request.
    var tokenizeChunks: (_ prompt: String, _ width: UInt32, _ height: UInt32,
                         _ images: Int) throws -> VisionChunks
    /// The `prompt_ms` window.
    var evalChunks: (_ chunks: VisionChunks) throws -> Void
    /// `vl_throughput`'s `predicted_ms` window.
    var generate: (_ count: Int, _ end: GenerationEnd) throws -> LlamaGeneration
}

extension VisionInference {
    nonisolated static func live(_ m: LlamaModel, _ mtmd: MtmdContext) -> VisionInference {
        VisionInference(
            mediaMarker: LlamaMtmd.defaultMediaMarker,
            tokenize: { try LlamaCpp.tokenize(m, $0, addSpecial: $1) },
            applyChatTemplate: { try LlamaCpp.applyChatTemplate(m, user: $0) },
            resetContext: { LlamaCpp.resetContext(m) },
            resetSampler: { LlamaCpp.resetSampler(m) },
            tokenizeChunks: {
                try LlamaMtmd.tokenizeGray(mtmd, text: $0, width: $1, height: $2, images: $3)
            },
            evalChunks: { try LlamaMtmd.evalChunks(mtmd, m, $0) },
            generate: { try LlamaCpp.generate(m, count: $0, end: $1) })
    }
}

/// `withInference` for the vision path.
nonisolated func withVisionInference<T>(path: String, mmprojPath: String, nGpuLayers: UInt32,
                                        threads: UInt32, swaFull: Bool,
                                        loadMode: LlamacppLoadMode,
                                        window: (MtmdContext) throws
                                            -> (contextSize: UInt32, nUbatch: UInt32),
                                        _ body: (VisionInference) async throws -> T) async throws -> T {
    try await withMtmd(path: path, mmprojPath: mmprojPath, nGpuLayers: nGpuLayers,
                       threads: threads, swaFull: swaFull, loadMode: loadMode, media: .vision,
                       window: window) { try await body(.live($0, $1)) }
}

/// `withInference` for the multimodal path. The projector loads before the context so
/// `window` can size the context (and its clamped micro-batch) from one frame's or one
/// clip's count. mtmd holds the model, so it must be freed first.
nonisolated func withMtmd<T>(path: String, mmprojPath: String, nGpuLayers: UInt32,
                             threads: UInt32, swaFull: Bool, loadMode: LlamacppLoadMode,
                             media: MtmdMedia,
                             window: (MtmdContext) throws -> (contextSize: UInt32, nUbatch: UInt32),
                             _ body: (LlamaModel, MtmdContext) async throws -> T) async throws -> T {
    let weights = try LlamaCpp.loadWeights(path: path, nGpuLayers: nGpuLayers,
                                           loadMode: loadMode)
    // On the accelerator regardless of `nGpuLayers`, as `llama-server`'s `mmproj_use_gpu`
    // defaults on independently of `-ngl` (and `llamacpp_ios_pipette` has no
    // `--no-mmproj-offload`). The simulator has no Metal backend.
    #if targetEnvironment(simulator)
    let useGpu = false
    #else
    let useGpu = true
    #endif
    let mtmd: MtmdContext
    do {
        mtmd = try LlamaMtmd.load(model: weights, mmprojPath: mmprojPath, useGpu: useGpu,
                                  threads: threads, media: media)
    } catch {
        LlamaCpp.freeWeights(weights)
        throw error
    }
    let m: LlamaModel
    do {
        let (contextSize, nUbatch) = try window(mtmd)
        m = try LlamaCpp.initContext(weights, contextSize: contextSize, nUbatch: nUbatch,
                                     threads: threads, swaFull: swaFull)
    } catch {
        mtmd.free()
        LlamaCpp.freeWeights(weights)
        throw error
    }
    defer {
        mtmd.free()
        m.free()
    }
    return try await body(m, mtmd)
}
