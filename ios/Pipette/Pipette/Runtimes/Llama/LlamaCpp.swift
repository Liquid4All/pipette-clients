import Foundation
import llama

// Failures surface as the shared `RuntimeError` (RuntimeSupport.swift).

// MARK: - llama.cpp log capture (for load-failure classification)

// Mirrors the Rust shim's `g_last_load_error`: a `llama_log_set` callback that
// forwards every llama.cpp/ggml log line to stderr (preserving the normal
// console output, including the OOM `GGML_LOG_ERROR` from patch 001) and, while
// `llamaCapturing` is set (during a model load), accumulates it so the failure
// reason can be classified. The callback is a global C function — it can't
// capture state — so the buffer/flag are file-scope and lock-guarded (ggml
// fires logs from worker/Metal-compiler threads).
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

/// OOM fragments emitted by llama.cpp / the Metal compiler — mirrors the Rust
/// shim's `is_out_of_memory_error` list.
nonisolated private func llamaLogIsOOM(_ msg: String) -> Bool {
    ["posix_memalign failed", "failed to allocate buffer", "out of memory",
     "MTLCompiler", "Compiler failed to build request"].contains { msg.contains($0) }
}

// MARK: - Loaded model handle

/// Owns one freshly-loaded llama.cpp model + context + greedy sampler. Freed
/// deterministically via `free()` (and on `deinit`), in the same order as the
/// Rust `ee_llama_model_free`: sampler → context → model. Reference type so it
/// holds the native resources with a deterministic lifetime — benchmarks reload
/// fresh per cell and must not lean on ARC timing for a 5 GB model. The handle is
/// assembled inside the benchmark scope (`withInference`) and never escapes it.
nonisolated final class LlamaModel {
    let model: OpaquePointer
    let ctx: OpaquePointer
    let sampler: UnsafeMutablePointer<llama_sampler>
    let vocab: OpaquePointer
    private var freed = false

    init(model: OpaquePointer, ctx: OpaquePointer,
         sampler: UnsafeMutablePointer<llama_sampler>, vocab: OpaquePointer) {
        self.model = model
        self.ctx = ctx
        self.sampler = sampler
        self.vocab = vocab
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

/// The in-process llama.cpp operations the native benchmark path needs, as
/// stateless functions over a `LlamaModel` handle — the Swift replacement for the
/// Rust `llama::*` wrapper + `ee_*` C shim, calling the C API directly (`import
/// llama`, compiled from our vendored sources). Engine parameters follow
/// `native/llama_shim.cpp` so on-device numbers track the Rust path: no mmap,
/// greedy sampler, `n_batch`-chunked decode, BOS via `add_special`, KV
/// clear between reps. (Thread count follows the llama.swiftui reference — capped
/// to performance cores — rather than the shim's all-cores default; see `load`.)
nonisolated enum LlamaCpp {
    /// llama.cpp backend init is process-global and must run once. Also installs
    /// the log-capture callback so load failures can be classified (OOM vs other).
    private static let backendInit: Void = {
        llama_log_set(llamaCaptureLog, nil)
        llama_backend_init()
    }()

    /// Submission `runtime_name` for llama.cpp results — the `pipette-plan-types`
    /// What this engine supplies where a cell left a load setting unset. Overlaid in one
    /// place — `LlamaRuntimeFlags.forRun` — whose result is both what the engine loads with
    /// and what the response reports, so a submitted `runtime_flags` cannot describe a run
    /// that did not happen.
    static let defaultNumberGpuLayers: UInt32 = 99
    static let defaultNUbatch: UInt32 = 512
    /// Off, as the stock tools pin it: `llama-bench` sets `swa_full = false` explicitly and
    /// `llama-server` leaves it off, so a windowed layer allocates KV for its window rather
    /// than for the whole context. llama's own default is the opposite, and inheriting it
    /// put every iOS peak-memory figure on a SWA model above the CLI's for a reason nothing
    /// recorded — ~120 MiB on Gemma 4 E4B at ctx 4096, ~36 MiB on E2B.
    ///
    /// A cell that wants the full-size cache asks for it (`swa_full = true`), and the value
    /// is reported either way, so the two are told apart in the record rather than inferred
    /// from the client version.
    static let defaultSwaFull = false

    /// Cap to the performance-core count: Apple SoCs top out at ~6–8 P-cores, and
    /// scheduling ggml work onto E-cores hurts throughput. `-2` leaves headroom for the
    /// OS/UI; `8` is the ceiling no current iPhone P-core count exceeds. Follows the
    /// llama.swiftui reference rather than the shim's all-cores default.
    ///
    /// Derived rather than fixed, so it is a property of the device — which is why a cell
    /// that leaves `threads` unset still reports the number it ran with.
    static var defaultThreads: UInt32 {
        UInt32(max(1, min(8, ProcessInfo.processInfo.activeProcessorCount - 2)))
    }

    // MARK: Lifecycle

    static func load(path: String, nGpuLayers: UInt32, contextSize: UInt32, nUbatch: UInt32,
                     threads: UInt32, swaFull: Bool) throws -> LlamaModel {
        _ = backendInit
        // Capture this load's llama.cpp/ggml log so a failure can be classified.
        llamaLogLock.lock(); llamaLogBuffer = ""; llamaCapturing = true; llamaLogLock.unlock()
        defer { llamaLogLock.lock(); llamaCapturing = false; llamaLogLock.unlock() }

        var mparams = llama_model_default_params()
        mparams.n_gpu_layers = Int32(nGpuLayers)
        // Match the shim/llama-bench contract: no mmap, so a model that only fits
        // file-backed fails instead of producing misleading rows.
        mparams.load_mode = LLAMA_LOAD_MODE_NONE
        #if targetEnvironment(simulator)
        mparams.n_gpu_layers = 0   // no Metal backend in the simulator slice
        #endif

        guard let model = llama_model_load_from_file(path, mparams) else {
            throw loadFailure("model load failed: \(path)")
        }

        var cparams = llama_context_default_params()
        if contextSize > 0 { cparams.n_ctx = contextSize }
        let ub: UInt32 = nUbatch > 0 ? nUbatch : 512
        cparams.n_ubatch = min(ub, cparams.n_ctx)
        cparams.n_batch = min(max(2048, ub), cparams.n_ctx)
        cparams.n_seq_max = 1
        cparams.no_perf = true
        // Stated by the cell, never inherited: the library defaults this on, the stock CLI
        // tools pin it off, and on a SWA model the difference is KV for the whole context
        // against KV for the window — a memory result that would otherwise be unaccountable.
        cparams.swa_full = swaFull
        // Both, as `llama-bench` sets them: the cell names one thread count, and a prefill
        // measured with a different one than decode would not be the cell it asked for.
        cparams.n_threads = Int32(threads)
        cparams.n_threads_batch = Int32(threads)

        guard let ctx = llama_init_from_model(model, cparams) else {
            llama_model_free(model)
            throw loadFailure("context init failed")
        }
        guard let sampler = llama_sampler_init_greedy() else {
            llama_free(ctx); llama_model_free(model)
            throw loadFailure("sampler init failed")
        }
        guard let vocab = llama_model_get_vocab(model) else {
            llama_sampler_free(sampler); llama_free(ctx); llama_model_free(model)
            throw loadFailure("no vocab")
        }
        return LlamaModel(model: model, ctx: ctx, sampler: sampler, vocab: vocab)
    }

    /// What llama.cpp and ggml wrote while loading the model last — the engine output a
    /// result carries, as the crate's extras carry a subprocess's. Reset at the start of
    /// each load, so a reader gets that load's lines and not the whole session's.
    static var capturedLoadLog: String {
        llamaLogLock.lock()
        defer { llamaLogLock.unlock() }
        return llamaLogBuffer
    }

    /// Build the error for a load-stage failure, classifying OOM vs generic from
    /// the captured log (mirrors the Rust shim's load-error handling).
    private static func loadFailure(_ what: String) -> RuntimeError {
        llamaLogLock.lock(); let log = llamaLogBuffer; llamaLogLock.unlock()
        let tail = log.isEmpty ? "" : ": " + String(log.suffix(400)).trimmingCharacters(in: .whitespacesAndNewlines)
        let detail = what + tail
        return llamaLogIsOOM(log) ? .outOfMemory(detail) : .engine(detail)
    }

    // MARK: Ops

    static func tokenize(_ m: LlamaModel, _ text: String, addSpecial: Bool) throws -> [Int32] {
        let byteLen = Int32(text.utf8.count)
        var capacity = Int(byteLen) + (addSpecial ? 1 : 0) + 8
        var tokens = [llama_token](repeating: 0, count: capacity)
        var n = text.withCString { cstr in
            llama_tokenize(m.vocab, cstr, byteLen, &tokens, Int32(capacity), addSpecial, true)
        }
        if n < 0 {                                   // buffer too small → -n is the needed size
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
                // llama_batch_get_one continues from the current KV position, so
                // consecutive chunks append naturally (matches the shim).
                let batch = llama_batch_get_one(buf.baseAddress! + offset, Int32(chunk))
                let rc = llama_decode(m.ctx, batch)
                if rc != 0 { throw RuntimeError.engine("prefill decode rc=\(rc)") }
                offset += chunk
            }
        }
        llama_synchronize(m.ctx)
    }

    static func decodeIgnoringEoG(_ m: LlamaModel, count: Int) throws {
        for _ in 0 ..< max(0, count) {
            let token = llama_sampler_sample(m.sampler, m.ctx, -1)
            llama_sampler_accept(m.sampler, token)
            var t = token
            let rc = llama_decode(m.ctx, llama_batch_get_one(&t, 1))
            if rc != 0 { throw RuntimeError.engine("decode rc=\(rc)") }
        }
        llama_synchronize(m.ctx)
    }

    static func chatCompletion(_ m: LlamaModel, messagesJSON: String,
                               maxTokens: Int, mcqChoices: [String]?) throws -> EvalGeneration {
        let prompt = try applyChatTemplate(m, messagesJSON: messagesJSON)
        let promptTokens = try tokenize(m, prompt, addSpecial: true)
        try prefill(m, promptTokens)

        // MCQ: greedily sample one token; return it if it matches a choice,
        // else fall back to the first choice (mirrors the Android path). The
        // `n_predict:1` arm is out of scope for stop_reason, so leave it unset.
        if let mcqChoices {
            let token = llama_sampler_sample(m.sampler, m.ctx, -1)
            llama_sampler_accept(m.sampler, token)
            let generated = piece(m, token: token).trimmingCharacters(in: .whitespacesAndNewlines)
            llama_synchronize(m.ctx)
            let text = mcqChoices.contains(generated) ? generated : (mcqChoices.first ?? "")
            // A single grammar-constrained token has no meaningful eos/truncated
            // classification — label it `unknown` and say why.
            return EvalGeneration(text: text, stopReason: .unknown,
                                  stopDetail: "mcq arm (n_predict:1, grammar-constrained)",
                                  completionTokens: nil)
        }

        // Free-form: greedy decode up to maxTokens, stopping at EOS/EOG.
        // Breaking on EOG ⇒ the model stopped naturally (`eos`); running the
        // full loop without an EOG ⇒ we hit the output-token cap (`truncated`).
        var output = ""
        var generated = 0
        var hitEog = false
        for _ in 0 ..< max(0, maxTokens) {
            let token = llama_sampler_sample(m.sampler, m.ctx, -1)
            llama_sampler_accept(m.sampler, token)
            if llama_vocab_is_eog(m.vocab, token) { hitEog = true; break }
            output += piece(m, token: token)
            generated += 1
            var t = token
            let rc = llama_decode(m.ctx, llama_batch_get_one(&t, 1))
            // Fail the sample on a decode error rather than returning a truncated
            // "success" — the eval loop turns this into a failed completion.
            if rc != 0 { throw RuntimeError.engine("decode rc=\(rc)") }
        }
        llama_synchronize(m.ctx)
        return EvalGeneration(text: output,
                              stopReason: hitEog ? .eos : .truncated,
                              stopDetail: nil,
                              completionTokens: generated)
    }

    /// Greedily decode up to `maxTokens` from the current context, returning the
    /// detokenized text (stops at the first EOG). Convenience used by the headless
    /// coherence diagnostic; mirrors the former Rust `decode_greedy` text path.
    static func decodeGreedyText(_ m: LlamaModel, maxTokens: Int) throws -> String {
        var output = ""
        for _ in 0 ..< max(0, maxTokens) {
            let token = llama_sampler_sample(m.sampler, m.ctx, -1)
            llama_sampler_accept(m.sampler, token)
            if llama_vocab_is_eog(m.vocab, token) { break }
            output += piece(m, token: token)
            var t = token
            let rc = llama_decode(m.ctx, llama_batch_get_one(&t, 1))
            if rc != 0 { throw RuntimeError.engine("decode rc=\(rc)") }
        }
        llama_synchronize(m.ctx)
        return output
    }

    // MARK: Helpers

    /// Apply the model's chat template to a JSON `[{role, content}]` array,
    /// returning the formatted prompt string (mirrors `ee_llama_apply_chat_template`).
    private static func applyChatTemplate(_ m: LlamaModel, messagesJSON: String) throws -> String {
        guard let tmpl = llama_model_chat_template(m.model, nil) else {
            throw RuntimeError.engine("model has no chat template")
        }
        let data = Data(messagesJSON.utf8)
        guard let raw = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]] else {
            throw RuntimeError.engine("malformed chat messages")
        }
        // Keep the C strings alive across the call.
        var allocated: [UnsafeMutablePointer<CChar>] = []
        defer { allocated.forEach { free($0) } }
        let chat: [llama_chat_message] = raw.map { item in
            let role = strdup((item["role"] as? String) ?? "")
            let content = strdup((item["content"] as? String) ?? "")
            allocated.append(role!); allocated.append(content!)
            return llama_chat_message(role: role, content: content)
        }

        var capacity = 8192
        var buf = [CChar](repeating: 0, count: capacity)
        var n = llama_chat_apply_template(tmpl, chat, chat.count, true, &buf, Int32(capacity))
        if n > Int32(capacity) {                      // didn't fit → grow to the returned size
            capacity = Int(n)
            buf = [CChar](repeating: 0, count: capacity)
            n = llama_chat_apply_template(tmpl, chat, chat.count, true, &buf, Int32(capacity))
        }
        guard n >= 0 else { throw RuntimeError.engine("chat template apply failed") }
        return String(decoding: buf.prefix(Int(n)).map { UInt8(bitPattern: $0) }, as: UTF8.self)
    }

    /// Detokenize one token to its text piece (two-pass buffer grow).
    private static func piece(_ m: LlamaModel, token: llama_token) -> String {
        var capacity: Int32 = 16
        var buf = [CChar](repeating: 0, count: Int(capacity))
        var n = llama_token_to_piece(m.vocab, token, &buf, capacity, 0, true)
        if n < 0 {
            capacity = -n
            buf = [CChar](repeating: 0, count: Int(capacity))
            n = llama_token_to_piece(m.vocab, token, &buf, capacity, 0, true)
        }
        guard n > 0 else { return "" }
        return String(decoding: buf.prefix(Int(n)).map { UInt8(bitPattern: $0) }, as: UTF8.self)
    }
}
