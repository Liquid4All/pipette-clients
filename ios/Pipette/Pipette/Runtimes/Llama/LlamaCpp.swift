import Foundation
import llama

// MARK: - llama.cpp log capture (for load-failure classification)

// As the Android shim's `g_last_load_error`: every log line still goes to stderr, and during
// a load it is also kept so a failure can be classified. The C callback can't capture state,
// so the buffer is file-scope, and locked because ggml logs from worker and Metal threads.
nonisolated private let llamaLogLock = NSLock()
nonisolated(unsafe) private var llamaLogBuffer = ""
nonisolated(unsafe) private var llamaCapturing = false

nonisolated private func llamaCaptureLog(_ level: ggml_log_level, _ text: UnsafePointer<CChar>?, _ user: UnsafeMutableRawPointer?) {
    guard let text else { return }
    let s = String(cString: text)
    fputs(s, stderr)
    llamaLogLock.lock()
    if llamaCapturing { llamaLogBuffer += s }
    llamaLogLock.unlock()
}

/// `pipette-android`'s `is_out_of_memory_error` fragments plus `pipette-llamacpp`'s eval
/// `ALLOCATION_ASSERT`: a null tensor data pointer is how a model that does not fit reports
/// itself first.
nonisolated func llamaLogIsOOM(_ msg: String) -> Bool {
    ["posix_memalign failed", "failed to allocate buffer", "out of memory",
     "tensor not allocated", "MTLCompiler", "Compiler failed to build request"]
        .contains { msg.contains($0) }
}

/// The crate's `LOAD_FAILED_ON_FILE`: llama.cpp's words for a model file it could not read.
nonisolated func llamaLogIsUnreadableFile(_ msg: String) -> Bool {
    ["failed to open", "failed to read magic", "corrupted or incomplete"]
        .contains { msg.contains($0) }
}

// MARK: - Loaded model handle

/// Freed explicitly with `free()` rather than left to ARC timing, since a cell reloads a
/// multi-GB model; the order (sampler, context, model) is the Android shim's.
nonisolated final class LlamaModel {
    let model: OpaquePointer
    let ctx: OpaquePointer
    let sampler: UnsafeMutablePointer<llama_sampler>
    let vocab: OpaquePointer
    /// What `generate` masks so a cell decodes the count it asked for. Found once at load,
    /// as `llama-server` does for `ignore_eos`: a vocabulary scan in a timed window would
    /// become part of the number.
    let eogBias: [llama_logit_bias]
    private var freed = false

    init(model: OpaquePointer, ctx: OpaquePointer,
         sampler: UnsafeMutablePointer<llama_sampler>, vocab: OpaquePointer,
         eogBias: [llama_logit_bias]) {
        self.model = model
        self.ctx = ctx
        self.sampler = sampler
        self.vocab = vocab
        self.eogBias = eogBias
    }

    func free() {
        guard !freed else { return }
        freed = true
        llama_sampler_free(sampler)
        llama_free(ctx)
        llama_model_free(model)
    }

    deinit { free() }
}

// MARK: - Stateless ops

/// Here because `PlanTypes` imports no engine headers.
nonisolated extension LlamacppLoadMode {
    /// Internal so a test pins it, as the Kotlin twin (`cValue`) is: a swapped case is
    /// otherwise invisible.
    var llamaValue: llama_load_mode {
        switch self {
        case .read: LLAMA_LOAD_MODE_NONE
        case .mmap: LLAMA_LOAD_MODE_MMAP
        case .mlock: LLAMA_LOAD_MODE_MLOCK
        case .mmapMlock: LLAMA_LOAD_MODE_MMAP_MLOCK
        case .directIo: LLAMA_LOAD_MODE_DIRECT_IO
        }
    }
}

/// Stateless llama.cpp C API calls over a `LlamaModel`. Parameters follow the stock tools
/// (`llama-bench`, `llama-server`) so on-device numbers track theirs.
nonisolated enum LlamaCpp {
    /// Process-global, run once. mtmd keeps its own log sink, so it gets the capture
    /// callback too, or a projector OOM would be classified as a generic engine error.
    private static let backendInit: Void = {
        llama_log_set(llamaCaptureLog, nil)
        mtmd_helper_log_set(llamaCaptureLog, nil)
        llama_backend_init()
    }()

    /// Defaults for load settings a cell left unset, overlaid only by
    /// `LlamaRuntimeFlags.forRun`, so what loads is what the response reports.
    static let defaultNumberGpuLayers: UInt32 = 99
    static let defaultNUbatch: UInt32 = 512
    /// Off, as the stock tools pin it, so a windowed layer allocates KV for its window only.
    /// llama's own default (on) put iOS SWA peaks above the CLI's: ~120 MiB on Gemma 4 E4B
    /// at ctx 4096, ~36 MiB on E2B.
    static let defaultSwaFull = false

    /// Anonymous memory, as the desktop CLI loads, not llama.cpp's mmap default: a mapped
    /// load lets a model that only fits file-backed produce rows, and accounts
    /// `max_memory_usage` differently from every other client.
    static let defaultLoadMode: LlamacppLoadMode = .read

    /// The performance-core count, as `common.cpp`'s `common_cpu_get_num_physical_cores` derives
    /// the stock tools' default on Apple silicon.
    static var defaultThreads: UInt32 {
        threadCount(performanceCores: performanceCoreCount(),
                    activeProcessors: ProcessInfo.processInfo.activeProcessorCount)
    }

    /// Split out to test off-device. The fallback (from the llama.swiftui reference) covers
    /// the simulator, which reports no performance level.
    static func threadCount(performanceCores: Int?, activeProcessors: Int) -> UInt32 {
        if let performanceCores, performanceCores > 0 { return UInt32(performanceCores) }
        return UInt32(max(1, min(8, activeProcessors - 2)))
    }

    private static func performanceCoreCount() -> Int? {
        var cores: Int32 = 0
        var size = MemoryLayout<Int32>.size
        guard sysctlbyname("hw.perflevel0.physicalcpu", &cores, &size, nil, 0) == 0 else {
            return nil
        }
        return Int(cores)
    }

    /// `1` is not out of memory: the cache filled, and llama.h's fix is a larger context,
    /// the opposite of what `formatJobError` suggests for OOM. Worded as
    /// `pipette-android`'s `decode_error_message`.
    ///
    /// Below `-1` the compute failed, and ggml-metal fails every later decode until the
    /// model is reloaded, hence `.backend`.
    static func decodeFailure(_ rc: Int32, phase: String) -> RuntimeError {
        switch rc {
        case 1:
            .engine("\(phase) decode failed: KV cache is full. The benchmark's "
                + "prompt+decode exceeds the allocated context size. (llama_decode rc=1)")
        case ..<(-1):
            .backend("\(phase) decode rc=\(rc)")
        default:
            .engine("\(phase) decode rc=\(rc)")
        }
    }

    // MARK: Lifecycle

    static func load(path: String, nGpuLayers: UInt32, contextSize: UInt32, nUbatch: UInt32,
                     threads: UInt32, swaFull: Bool,
                     loadMode: LlamacppLoadMode) throws -> LlamaModel {
        let model = try loadWeights(path: path, nGpuLayers: nGpuLayers, loadMode: loadMode)
        do {
            return try initContext(model, contextSize: contextSize, nUbatch: nUbatch,
                                   threads: threads, swaFull: swaFull)
        } catch {
            freeWeights(model)
            throw error
        }
    }

    /// `load`'s first half. The caller owns the weights until `initContext` wraps them in a
    /// `LlamaModel`.
    static func loadWeights(path: String, nGpuLayers: UInt32,
                            loadMode: LlamacppLoadMode) throws -> OpaquePointer {
        beginLoadCapture()
        defer { endLoadCapture() }

        #if targetEnvironment(simulator)
        let gpuLayers: Int32 = 0   // no Metal backend in the simulator slice
        #else
        let gpuLayers = Int32(nGpuLayers)
        #endif

        guard let model = pipette_llama_load_model(path, gpuLayers, loadMode.llamaValue) else {
            throw loadFailure("model load failed: \(path)")
        }
        return model
    }

    /// Free weights no `LlamaModel` holds yet.
    static func freeWeights(_ model: OpaquePointer) {
        llama_model_free(model)
    }

    /// `load`'s second half: a context, sampler and vocabulary over loaded weights. The
    /// returned `LlamaModel` owns `model`; on a throw the caller still does.
    static func initContext(_ model: OpaquePointer, contextSize: UInt32, nUbatch: UInt32,
                            threads: UInt32, swaFull: Bool) throws -> LlamaModel {
        // Continues the weights' capture, so a failure here is classified from the whole
        // load's log.
        beginLoadCapture(reset: false)
        defer { endLoadCapture() }

        // `llama_context_params` differs between the builds, so the build fills it
        // (`pipette_llama_table.cpp`).
        guard let ctx = pipette_llama_init_context(model, contextSize, nUbatch, Int32(threads),
                                                   swaFull) else {
            throw loadFailure("context init failed")
        }
        guard let sampler = llama_sampler_init_greedy() else {
            llama_free(ctx)
            throw loadFailure("sampler init failed")
        }
        guard let vocab = llama_model_get_vocab(model) else {
            llama_sampler_free(sampler); llama_free(ctx)
            throw loadFailure("no vocab")
        }
        return LlamaModel(model: model, ctx: ctx, sampler: sampler, vocab: vocab,
                          eogBias: eogBias(vocab))
    }

    /// The last load's engine log, which a result carries as the crate carries a
    /// subprocess's stderr.
    static var capturedLoadLog: String {
        llamaLogLock.lock()
        defer { llamaLogLock.unlock() }
        return llamaLogBuffer
    }

    /// Pair with `endLoadCapture()` on every path out. A load that continues an assembly
    /// (the projector after its model) passes `reset: false` so the log holds both halves.
    static func beginLoadCapture(reset: Bool = true) {
        _ = backendInit
        llamaLogLock.lock()
        if reset { llamaLogBuffer = "" }
        llamaCapturing = true
        llamaLogLock.unlock()
    }

    static func endLoadCapture() {
        llamaLogLock.lock(); llamaCapturing = false; llamaLogLock.unlock()
    }

    /// A `ModelLoadFailure`, which `JobExecutor` memoizes per model so the job's later
    /// cells for it don't retry the load.
    static func loadFailure(_ what: String) -> ModelLoadFailure {
        llamaLogLock.lock(); let log = llamaLogBuffer; llamaLogLock.unlock()
        let tail = log.isEmpty ? "" : ": " + String(log.suffix(400)).trimmingCharacters(in: .whitespacesAndNewlines)
        let detail = what + tail
        // The file words count only in `detail`, the text pipette-ios-proxy reads them in.
        let cause: RuntimeError = llamaLogIsOOM(log) ? .outOfMemory(detail)
            : llamaLogIsUnreadableFile(detail) ? .unreadableModel(detail) : .engine(detail)
        return ModelLoadFailure(cause: cause)
    }

    // MARK: Ops

    static func tokenize(_ m: LlamaModel, _ text: String, addSpecial: Bool) throws -> [Int32] {
        let byteLen = Int32(text.utf8.count)
        var capacity = Int(byteLen) + (addSpecial ? 1 : 0) + 8
        var tokens = [llama_token](repeating: 0, count: capacity)
        var n = text.withCString { cstr in
            llama_tokenize(m.vocab, cstr, byteLen, &tokens, Int32(capacity), addSpecial, true)
        }
        if n < 0 {                                   // buffer too small: -n is the needed size
            capacity = Int(-n)
            tokens = [llama_token](repeating: 0, count: capacity)
            n = text.withCString { cstr in
                llama_tokenize(m.vocab, cstr, byteLen, &tokens, Int32(capacity), addSpecial, true)
            }
            guard n >= 0 else { throw RuntimeError.engine("tokenize failed") }
        }
        return Array(tokens.prefix(Int(n)))
    }

    static func resetContext(_ m: LlamaModel) {
        llama_memory_clear(llama_get_memory(m.ctx), false)
    }

    static func resetSampler(_ m: LlamaModel) {
        llama_sampler_reset(m.sampler)
    }

    static func prefill(_ m: LlamaModel, _ tokens: [Int32]) throws {
        guard !tokens.isEmpty else { return }
        let nBatch = Int(llama_n_batch(m.ctx))
        var toks = tokens
        try toks.withUnsafeMutableBufferPointer { buf in
            var offset = 0
            while offset < buf.count {
                let chunk = min(buf.count - offset, nBatch)
                // Continues from the current KV position, so chunks append.
                let batch = llama_batch_get_one(buf.baseAddress! + offset, Int32(chunk))
                let rc = llama_decode(m.ctx, batch)
                if rc != 0 { throw decodeFailure(rc, phase: "prefill") }
                offset += chunk
            }
        }
        llama_synchronize(m.ctx)
    }

    /// `llama-bench`'s `test_gen`: a BOS (where the vocabulary adds one), then random ids,
    /// with no sampler in the loop, whose per-token cost is not part of a decode's.
    @discardableResult
    static func decode(_ m: LlamaModel, count: Int) throws -> Int {
        let vocabSize = llama_vocab_n_tokens(m.vocab)
        var token: llama_token = llama_vocab_get_add_bos(m.vocab)
            ? llama_vocab_bos(m.vocab)
            : llama_token.random(in: 0 ..< max(1, vocabSize))
        var decoded = 0
        for _ in 0 ..< max(0, count) {
            var next = token
            let rc = llama_decode(m.ctx, llama_batch_get_one(&next, 1))
            if rc != 0 { throw decodeFailure(rc, phase: "decode") }
            // Per step, as `test_gen` synchronizes: the wait is part of what it times.
            llama_synchronize(m.ctx)
            token = llama_token.random(in: 0 ..< max(1, vocabSize))
            decoded += 1
        }
        return decoded
    }

    /// The crate's `ignore_eos` at temperature 0. Masked rather than skipped after sampling,
    /// so the text never continues past an end-of-turn the model meant.
    static func generate(_ m: LlamaModel, count: Int, end: GenerationEnd) throws
        -> LlamaGeneration {
        guard let chain = llama_sampler_chain_init(llama_sampler_chain_default_params()) else {
            throw RuntimeError.engine("sampler chain init failed")
        }
        defer { llama_sampler_free(chain) }
        m.eogBias.withUnsafeBufferPointer { eog in
            guard let base = eog.baseAddress, !eog.isEmpty else { return }
            llama_sampler_chain_add(chain, llama_sampler_init_logit_bias(
                llama_vocab_n_tokens(m.vocab), Int32(eog.count), base))
        }
        llama_sampler_chain_add(chain, llama_sampler_init_greedy())
        let generation = try generationSteps(count: count, end: end, sample: {
            llama_sampler_sample(chain, m.ctx, -1)
        }, decode: { token in
            var next = token
            let rc = llama_decode(m.ctx, llama_batch_get_one(&next, 1))
            if rc != 0 { throw decodeFailure(rc, phase: "generate") }
        })
        // Waits out the last decode under `.afterLastDecode`; after a last sample the logits
        // read already synced.
        llama_synchronize(m.ctx)
        return generation
    }

    static func generationSteps(count: Int, end: GenerationEnd,
                                sample: () -> llama_token,
                                decode: (llama_token) throws -> Void) throws
        -> LlamaGeneration {
        guard count > 0 else {
            throw RuntimeError.engine("a generation of \(count) tokens measures nothing")
        }
        var firstSampledAt: ContinuousClock.Instant?
        try (1 ..< count).forEach { _ in
            let token = sample()
            if firstSampledAt == nil { firstSampledAt = .now }
            try decode(token)
        }
        let last = sample()
        let lastSampledAt = ContinuousClock.now
        if end == .afterLastDecode { try decode(last) }
        return LlamaGeneration(tokens: count, lastSampledAt: lastSampledAt,
                               firstSampledAt: firstSampledAt ?? lastSampledAt)
    }

    private static func eogBias(_ vocab: OpaquePointer) -> [llama_logit_bias] {
        (0 ..< llama_vocab_n_tokens(vocab))
            .filter { llama_vocab_is_eog(vocab, $0) }
            .map { llama_logit_bias(token: $0, bias: -.infinity) }
    }

    static func chatCompletion(_ m: LlamaModel, messagesJSON: String, maxTokens: Int,
                               mcqChoices: [String]?, sampling: Sampling,
                               seed: UInt32?) throws -> EvalGeneration {
        let prompt = try applyChatTemplate(m, messagesJSON: messagesJSON)
        let promptTokens = try tokenize(m, prompt, addSpecial: true)
        try prefill(m, promptTokens)

        // The crate's MCQ arm: one token under a grammar admitting only the choices.
        if let mcqChoices {
            let sampler = try evalSampler(m, sampling: sampling, seed: seed,
                                          grammar: mcqGrammar(mcqChoices))
            defer { llama_sampler_free(sampler) }
            let token = llama_sampler_sample(sampler, m.ctx, -1)
            llama_sampler_accept(sampler, token)
            let text = piece(m, token: token, special: false).trimmingCharacters(in: .whitespacesAndNewlines)
            llama_synchronize(m.ctx)
            // Stricter than the crate, where llama-server enforces the grammar: here the
            // sampler chain is ours, so this is the only place a grammar failure shows.
            guard mcqChoices.contains(text) else {
                throw RuntimeError.engine(
                    "mcq grammar produced `\(text)`, which is not one of \(mcqChoices)")
            }
            return EvalGeneration(text: text, stopReason: .unknown,
                                  stopDetail: "mcq arm (n_predict:1, grammar-constrained)",
                                  completionTokens: nil)
        }

        let sampler = try evalSampler(m, sampling: sampling, seed: seed, grammar: nil)
        defer { llama_sampler_free(sampler) }
        var output = ""
        var generated = 0
        var hitEog = false
        for _ in 0 ..< max(0, maxTokens) {
            let token = llama_sampler_sample(sampler, m.ctx, -1)
            llama_sampler_accept(sampler, token)
            if llama_vocab_is_eog(m.vocab, token) { hitEog = true; break }
            output += piece(m, token: token, special: false)
            generated += 1
            var t = token
            let rc = llama_decode(m.ctx, llama_batch_get_one(&t, 1))
            if rc != 0 { throw decodeFailure(rc, phase: "decode") }
        }
        llama_synchronize(m.ctx)
        return EvalGeneration(text: output,
                              stopReason: evalStopReason(stoppedOnEnd: hitEog),
                              stopDetail: nil,
                              completionTokens: UInt64(generated))
    }

    nonisolated enum SamplerStage: Equatable {
        case topK(Int32)
        case topP(Float, minKeep: Int)
        case minP(Float, minKeep: Int)
        case temperature(Float)
        case dist(seed: UInt32)
        case greedy
    }

    /// The chain `llama-server` builds for the same `/completion` request: greedy at
    /// temperature 0, else its `common_params_sampling` defaults with each pin in place.
    /// `EngineSamplingTests` holds these defaults to the vendored header.
    nonisolated static func evalChain(_ sampling: Sampling, seed: UInt32?) -> [SamplerStage] {
        guard drawsSample(atTemperature: sampling.temperature.value) else { return [.greedy] }
        return [
            .topK(sampling.topK?.value ?? 40),
            .topP(Float(sampling.topP?.value ?? 0.95), minKeep: 0),
            .minP(Float(sampling.minP?.value ?? 0.05), minKeep: 0),
            .temperature(Float(sampling.temperature.value)),
            .dist(seed: seed ?? LLAMA_DEFAULT_SEED),
        ]
    }

    /// The caller frees it. A grammar goes first, so the rest only sees tokens it admits.
    private static func evalSampler(_ m: LlamaModel, sampling: Sampling, seed: UInt32?,
                                    grammar: String?) throws -> UnsafeMutablePointer<llama_sampler> {
        guard let chain = llama_sampler_chain_init(llama_sampler_chain_default_params()) else {
            throw RuntimeError.engine("sampler chain init failed")
        }
        if let grammar {
            guard let constrained = llama_sampler_init_grammar(m.vocab, grammar, "root") else {
                llama_sampler_free(chain)
                throw RuntimeError.engine("grammar sampler init failed: \(grammar)")
            }
            llama_sampler_chain_add(chain, constrained)
        }
        for stage in evalChain(sampling, seed: seed) {
            let sampler = switch stage {
            case .topK(let k): llama_sampler_init_top_k(k)
            case let .topP(p, minKeep): llama_sampler_init_top_p(p, minKeep)
            case let .minP(p, minKeep): llama_sampler_init_min_p(p, minKeep)
            case .temperature(let t): llama_sampler_init_temp(t)
            case .dist(let seed): llama_sampler_init_dist(seed)
            case .greedy: llama_sampler_init_greedy()
            }
            llama_sampler_chain_add(chain, sampler)
        }
        return chain
    }

    /// The crate's `mcq_grammar`. Choices are quoted as Rust's `{:?}` writes them, so a
    /// quote or backslash in one can't end the rule early.
    static func mcqGrammar(_ choices: [String]) -> String {
        "root ::= " + choices.map(quoted).joined(separator: " | ")
    }

    private static func quoted(_ choice: String) -> String {
        var out = "\""
        for scalar in choice.unicodeScalars {
            switch scalar {
            case "\"": out += "\\\""
            case "\\": out += "\\\\"
            case "\n": out += "\\n"
            case "\r": out += "\\r"
            case "\t": out += "\\t"
            default: out.unicodeScalars.append(scalar)
            }
        }
        return out + "\""
    }

    /// For the headless coherence diagnostic.
    static func decodeGreedyText(_ m: LlamaModel, maxTokens: Int) throws -> String {
        var output = ""
        for _ in 0 ..< max(0, maxTokens) {
            let token = llama_sampler_sample(m.sampler, m.ctx, -1)
            llama_sampler_accept(m.sampler, token)
            if llama_vocab_is_eog(m.vocab, token) { break }
            output += piece(m, token: token)
            var t = token
            let rc = llama_decode(m.ctx, llama_batch_get_one(&t, 1))
            if rc != 0 { throw decodeFailure(rc, phase: "decode") }
        }
        llama_synchronize(m.ctx)
        return output
    }

    // MARK: Helpers

    /// A model with no template is refused: an eval scored under a guessed format would be
    /// a different eval. The caller tokenizes with `add_special` on, which suits either
    /// renderer, since the Jinja fallback strips the template's BOS as `llama-server` does.
    private static func applyChatTemplate(_ m: LlamaModel, messagesJSON: String) throws -> String {
        guard let tmpl = llama_model_chat_template(m.model, nil) else {
            throw RuntimeError.engine("model has no chat template")
        }
        let data = Data(messagesJSON.utf8)
        guard let raw = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]] else {
            throw RuntimeError.engine("malformed chat messages")
        }
        return try renderChatTemplate(tmpl, model: m.model, raw.map {
            (role: ($0["role"] as? String) ?? "", content: ($0["content"] as? String) ?? "")
        }).text
    }

    /// The vision cell's single-message request. A model without a template renders as
    /// chatml, as the server falls back (`common_chat_templates_init`), rather than being
    /// refused as on the eval path: a timing cell only needs the server's format.
    static func applyChatTemplate(_ m: LlamaModel, user content: String) throws -> String {
        try renderChatTemplate(m, [(role: "user", content: content)]).text
    }

    nonisolated struct ChatPrompt: Equatable {
        let text: String
        /// The Jinja path strips the template's leading BOS, as `llama-server` does, so a
        /// caller that tokenizes the legacy render without specials tokenizes this with them.
        let viaJinja: Bool
    }

    static func renderChatTemplate(_ m: LlamaModel,
                                   _ messages: [(role: String, content: String)]) throws -> ChatPrompt {
        try renderChatTemplate(llama_model_chat_template(m.model, nil), model: m.model, messages)
    }

    /// A `nil` template is chatml. The legacy C call renders first and Jinja only when it
    /// returns -1 (unknown template, e.g. Gemma-4's `<|turn>`): moving every template to
    /// Jinja would change the measured prefill of every vision cell that already renders.
    /// A `nil` `model` renders BOS/EOS empty, for tests.
    static func renderChatTemplate(_ tmpl: UnsafePointer<CChar>?, model: OpaquePointer?,
                                   _ messages: [(role: String, content: String)]) throws -> ChatPrompt {
        // Keep the C strings alive across both renders.
        var allocated: [UnsafeMutablePointer<CChar>] = []
        defer { allocated.forEach { free($0) } }
        let chat: [llama_chat_message] = try messages.map { message in
            guard let role = strdup(message.role), let content = strdup(message.content) else {
                throw RuntimeError.engine("chat template message allocation failed")
            }
            allocated.append(role); allocated.append(content)
            return llama_chat_message(role: role, content: content)
        }

        if let text = try legacyChatTemplate(tmpl, chat) {
            return ChatPrompt(text: text, viaJinja: false)
        }
        // As `llama-server` with no `--reasoning` or `--chat-template-kwargs`.
        let text = try jinjaChatTemplate(tmpl, model: model, messages, enableThinking: true,
                                         kwargs: [])
        return ChatPrompt(text: text, viaJinja: true)
    }

    /// `llama_chat_apply_template`, or `nil` when it returns -1.
    private static func legacyChatTemplate(_ tmpl: UnsafePointer<CChar>?,
                                           _ chat: [llama_chat_message]) throws -> String? {
        var capacity = 8192
        var buf = [CChar](repeating: 0, count: capacity)
        var n = llama_chat_apply_template(tmpl, chat, chat.count, true, &buf, Int32(capacity))
        if n > Int32(capacity) {                      // didn't fit: grow to the returned size
            capacity = Int(n)
            buf = [CChar](repeating: 0, count: capacity)
            n = llama_chat_apply_template(tmpl, chat, chat.count, true, &buf, Int32(capacity))
        }
        if n == -1 { return nil }
        guard n >= 0 else { throw RuntimeError.engine("chat template apply failed") }
        return String(decoding: buf.prefix(Int(n)).map { UInt8(bitPattern: $0) }, as: UTF8.self)
    }

    /// `kwargs` values are JSON text, as the server holds `--chat-template-kwargs`.
    static func jinjaChatTemplate(_ tmpl: UnsafePointer<CChar>?, model: OpaquePointer?,
                                  _ messages: [(role: String, content: String)],
                                  enableThinking: Bool,
                                  kwargs: [(key: String, json: String)]) throws -> String {
        var allocated: [UnsafeMutablePointer<CChar>] = []
        defer { allocated.forEach { free($0) } }
        func cStrings(_ strings: [String]) throws -> [UnsafePointer<CChar>?] {
            try strings.map {
                guard let copy = strdup($0) else {
                    throw RuntimeError.engine("chat template message allocation failed")
                }
                allocated.append(copy)
                return UnsafePointer(copy)
            }
        }
        let roles = try cStrings(messages.map(\.role))
        let contents = try cStrings(messages.map(\.content))
        let keys = try cStrings(kwargs.map(\.key))
        let values = try cStrings(kwargs.map(\.json))
        var err = [CChar](repeating: 0, count: 1024)
        var capacity = 8192
        // Twice at most: a template whose length changes between renders is refused.
        for _ in 0 ..< 2 {
            var buf = [CChar](repeating: 0, count: capacity)
            let n = pipette_chat_apply_jinja(model, tmpl, roles, contents, roles.count, true,
                                             enableThinking, keys, values, keys.count,
                                             &buf, Int32(capacity), &err, err.count)
            guard n >= 0 else {
                throw RuntimeError.engine("chat template apply failed: \(String(cString: err))")
            }
            if Int(n) <= capacity {
                return String(decoding: buf.prefix(Int(n)).map { UInt8(bitPattern: $0) }, as: UTF8.self)
            }
            capacity = Int(n)
        }
        throw RuntimeError.engine("chat template apply failed: the prompt's length changed between renders")
    }

    /// Control tokens are kept: the caller compares two arms' replies for identity, and
    /// dropping one could make different generations compare equal.
    static func detokenize(_ m: LlamaModel, tokens: [llama_token]) -> String {
        tokens.map { piece(m, token: $0) }.joined()
    }

    /// The eval path passes `special: false`, as `llama-server` renders a completion, so a
    /// template artefact is never scored as an answer. North-Mini-Code's preserved markers
    /// drop here too (gap 15 in `docs/client-parity.md`).
    private static func piece(_ m: LlamaModel, token: llama_token,
                              special: Bool = true) -> String {
        var capacity: Int32 = 16
        var buf = [CChar](repeating: 0, count: Int(capacity))
        var n = llama_token_to_piece(m.vocab, token, &buf, capacity, 0, special)
        if n < 0 {
            capacity = -n
            buf = [CChar](repeating: 0, count: Int(capacity))
            n = llama_token_to_piece(m.vocab, token, &buf, capacity, 0, special)
        }
        guard n > 0 else { return "" }
        return String(decoding: buf.prefix(Int(n)).map { UInt8(bitPattern: $0) }, as: UTF8.self)
    }
}
