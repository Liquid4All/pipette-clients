import Foundation
import llama

/// Derived from the model spec, since an iOS cell has no argv to carry `--spec-type`.
nonisolated enum DrafterArrangement {
    case none
    case separate(path: String)
    /// Inside the target weights, which must carry a multi-token-prediction head
    /// (`pipette_vlspec_attach`).
    case merged

    var raw: Int32 {
        switch self {
        case .none: Int32(PIPETTE_VLSPEC_DRAFTER_NONE.rawValue)
        case .separate: Int32(PIPETTE_VLSPEC_DRAFTER_SEPARATE.rawValue)
        case .merged: Int32(PIPETTE_VLSPEC_DRAFTER_MERGED.rawValue)
        }
    }

    var path: String? {
        if case let .separate(path) = self { return path }
        return nil
    }
}

/// One arm: model, projector and drafter. Freed explicitly because the next arm's
/// gigabytes cannot load until this one is gone.
nonisolated final class LlamaVisionChat {
    private let model: LlamaModel
    private let mtmd: MtmdContext
    private var spec: OpaquePointer?
    private let maxTokens: Int
    private var messages: [(role: String, content: String)] = []
    /// In marker order: each turn re-renders the whole conversation, earlier markers too.
    private var images: [String] = []
    /// Allocated once per arm: a reallocation in a timed window would join the number.
    private var tokenBuffer: [Int32]

    private init(model: LlamaModel, mtmd: MtmdContext, spec: OpaquePointer, maxTokens: Int) {
        self.model = model
        self.mtmd = mtmd
        self.spec = spec
        self.maxTokens = maxTokens
        self.tokenBuffer = [Int32](repeating: 0, count: max(1, maxTokens))
    }

    var hasDrafter: Bool {
        guard let spec else { return false }
        return pipette_vlspec_has_drafter(spec)
    }

    /// A merged drafter the vendored llama.cpp cannot reach is refused with the engine's
    /// reason rather than measured as an unspeculated run.
    static func attach(model: LlamaModel, mtmd: MtmdContext, drafter: DrafterArrangement,
                       draftLength: UInt32, maxTokens: Int, temperature: SamplingTemperature,
                       nGpuLayers: UInt32, threads: UInt32) throws -> LlamaVisionChat {
        var err = [CChar](repeating: 0, count: 1024)
        let spec = try withOptionalCString(drafter.path) { draft -> OpaquePointer in
            var params = pipette_vlspec_params(
                drafter: drafter.raw,
                draft_path: draft,
                n_draft_max: Int32(draftLength),
                n_gpu_layers: Int32(nGpuLayers),
                n_threads: Int32(threads),
                temperature: Float(temperature.value))
            guard let spec = pipette_vlspec_attach(model.model, model.ctx, &params,
                                                   &err, err.count) else {
                throw RuntimeError.engine(String(cString: err))
            }
            return spec
        }
        return LlamaVisionChat(model: model, mtmd: mtmd, spec: spec, maxTokens: maxTokens)
    }

    func free() {
        guard let spec else { return }
        self.spec = nil
        pipette_vlspec_free(spec)
    }

    deinit { free() }

    var chat: VisionChat {
        VisionChat(begin: { try self.begin(system: $0) },
                   turn: { try self.turn(text: $0, images: $1, maxTokens: $2) },
                   hasDrafter: hasDrafter)
    }

    private func begin(system: String) throws {
        guard spec != nil else { throw RuntimeError.engine("the vision chat arm is closed") }
        messages = system.isEmpty ? [] : [(role: "system", content: system)]
        images.removeAll(keepingCapacity: true)
    }

    private func turn(text: String, images turnImages: [String],
                      maxTokens: Int) throws -> TurnRecord {
        guard let spec else { throw RuntimeError.engine("the vision chat arm is closed") }
        messages.append((role: "user", content: text))
        images.append(contentsOf: turnImages)

        // No prompt cache, as the CLI's `cache_prompt: false`: every turn prefills the whole
        // conversation, or TTFT would read lower than the reference.
        LlamaCpp.resetContext(model)
        pipette_vlspec_begin(spec)

        // Re-rendered whole, as the reference server does: a template may place separators
        // differently once a conversation has several turns. Both arms render the same way,
        // so the ratio doesn't depend on which renderer (legacy or Jinja) ran.
        let prompt = try LlamaCpp.renderChatTemplate(model, messages)

        let clock = ContinuousClock()
        let promptStarted = clock.now
        let chunks = try LlamaMtmd.tokenizeFiles(mtmd, text: prompt.text, paths: images,
                                                 addSpecial: prompt.viaJinja)
        defer { chunks.free() }
        let nPast = try LlamaMtmd.evalChunks(mtmd, model, chunks, from: 0)
        let promptMs = durationMs(clock.now - promptStarted)

        // The speculator's prompt; its positions are the context's, hence `nPast` separately.
        let promptText = LlamaMtmd.textTokens(chunks)

        var stats = pipette_vlspec_turn_stats()
        var written = 0
        var err = [CChar](repeating: 0, count: 1024)
        let status = promptText.withUnsafeBufferPointer { prompt in
            pipette_vlspec_generate(spec, prompt.baseAddress, prompt.count,
                                    nPast, Int32(maxTokens),
                                    &tokenBuffer, tokenBuffer.count, &written,
                                    &stats, &err, err.count)
        }
        guard status == 0 else { throw RuntimeError.engine(String(cString: err)) }
        guard stats.stop != Int32(PIPETTE_VLSPEC_STOP_CONTEXT.rawValue) else {
            throw RuntimeError.engine(
                "a turn filled the context window, so the conversation it belongs to was "
                    + "not measured; size the cell's ctx_size for the corpus's longest "
                    + "conversation")
        }

        let generated = Array(tokenBuffer.prefix(written))
        let reply = LlamaCpp.detokenize(model, tokens: generated)
        // Back as text, as the CLI sends it in the next request.
        messages.append((role: "assistant", content: reply))

        return TurnRecord(
            promptMs: promptMs,
            predictedMs: stats.predicted_ms,
            // The reference stamps TTFT around an HTTP request; in process there is no
            // transport, so this is the prefill plus the first sample.
            ttftMs: promptMs + stats.first_token_ms,
            decodeMs: max(0, stats.predicted_ms - stats.first_token_ms),
            // Stamped by the traversal around the whole call.
            e2eMs: 0,
            generatedTokens: UInt64(max(0, stats.n_generated)),
            draftedTokens: UInt64(max(0, stats.n_drafted)),
            acceptedTokens: UInt64(max(0, stats.n_accepted)),
            targetForwards: UInt64(max(0, stats.n_target_forwards)),
            endedOnEos: stats.stop == Int32(PIPETTE_VLSPEC_STOP_EOG.rawValue),
            endedOnLength: stats.stop == Int32(PIPETTE_VLSPEC_STOP_LIMIT.rawValue),
            text: reply)
    }
}

/// `withVisionInference` for the paired cell. Teardown must reverse the load order: the
/// drafter borrows the target's context and the projector borrows its model.
nonisolated func openVisionChat(path: String, mmprojPath: String, drafter: DrafterArrangement,
                                draftLength: UInt32, maxTokens: Int, temperature: SamplingTemperature,
                                nGpuLayers: UInt32, contextSize: UInt32, nUbatch: UInt32,
                                threads: UInt32, swaFull: Bool,
                                loadMode: LlamacppLoadMode) throws
    -> (chat: VisionChat, close: () -> Void) {
    let model = try LlamaCpp.load(path: path, nGpuLayers: nGpuLayers,
                                  contextSize: contextSize, nUbatch: nUbatch, threads: threads,
                                  swaFull: swaFull, loadMode: loadMode)
    // On the accelerator regardless of `nGpuLayers`, as in `withVisionInference`.
    #if targetEnvironment(simulator)
    let useGpu = false
    #else
    let useGpu = true
    #endif
    let mtmd: MtmdContext
    do {
        mtmd = try LlamaMtmd.load(model: model.model, mmprojPath: mmprojPath, useGpu: useGpu,
                                  threads: threads)
    } catch {
        model.free()
        throw error
    }
    let arm: LlamaVisionChat
    do {
        arm = try LlamaVisionChat.attach(model: model, mtmd: mtmd, drafter: drafter,
                                         draftLength: draftLength, maxTokens: maxTokens,
                                         temperature: temperature,
                                         nGpuLayers: nGpuLayers, threads: threads)
    } catch {
        mtmd.free()
        model.free()
        throw error
    }
    return (arm.chat, {
        arm.free()
        mtmd.free()
        model.free()
    })
}

/// A `nil` path reaches C as `NULL`; an empty string would read as a named drafter.
private nonisolated func withOptionalCString<T>(
    _ value: String?, _ body: (UnsafePointer<CChar>?) throws -> T
) throws -> T {
    guard let value else { return try body(nil) }
    return try value.withCString { try body($0) }
}
