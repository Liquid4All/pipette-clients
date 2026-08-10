import Foundation
import MLX
import MLXLMCommon

/// MLX-graph generation/timing primitives over an already-loaded model — the
/// engine layer beneath `MLXBenchmark`, working in pure token space (no
/// `BenchmarkResult`, no readiness, no tokenizer). The structural counterpart to
/// `LlamaCpp`'s stateless ops.
///
/// Prefill feeds the cache in `chunk`-token windows with `asyncEval` so the
/// activation peak stays bounded and the GPU pipeline stays full; decode keeps the
/// running token on-device and `asyncEval`s each step (see `generate`).
extension MLXRuntime {
    /// Chunked+async prefill of `tokens` synthetic ids → elapsed milliseconds.
    nonisolated static func prefillMillis(_ model: any LanguageModel, tokens: Int, chunk: Int) -> Double {
        let toks = MLXArray(PromptSeed.syntheticTokenIds(tokens)).reshaped(1, -1)
        let clock = ContinuousClock()
        let start = clock.now
        eval(lastLogits(model, toks, chunk: chunk))
        return (clock.now - start).milliseconds
    }

    /// Prefill `prefill` synthetic ids into a fresh cache (untimed, chunked+async),
    /// then greedily decode `decode` tokens; returns the decode-loop milliseconds
    /// (prefill excluded — matches the Rust `decode_throughput` timing).
    @discardableResult
    nonisolated static func generate(_ model: any LanguageModel, prefill: Int, decode: Int, chunk: Int) -> Double {
        let cache = model.newCache(parameters: nil)
        let p = MLXArray(PromptSeed.syntheticTokenIds(prefill)).reshaped(1, -1)
        let n = p.dim(1)
        let step = chunk > 0 ? chunk : n
        var logits = MLXArray()
        var i = 0
        while i < n {
            let end = min(i + step, n)
            logits = model(p[0 ..< 1, i ..< end], cache: cache)
            if end < n { asyncEval(logits[0, -1]) } else { eval(logits[0, -1]) }
            i = end
        }
        // Pipelined greedy decode (mirrors MLXLMCommon's TokenIterator): keep the
        // token on-device as an MLXArray and `asyncEval` each step so the CPU builds
        // the next step's graph while the GPU computes the current one — instead of a
        // blocking `.item()` CPU↔GPU sync per token, which serializes the loop and is
        // what made decode slow. A host-array rebuild per token is avoided too.
        var y = argMax(logits[0, -1], axis: -1).reshaped(1, 1).asType(.int32)
        eval(y)                       // materialize the seed (prefill result, untimed)
        let clock = ContinuousClock()
        let start = clock.now
        for _ in 0 ..< max(0, decode) {
            logits = model(y, cache: cache)
            y = argMax(logits[0, -1], axis: -1).reshaped(1, 1).asType(.int32)
            asyncEval(y)
        }
        eval(y)                       // single sync so all decode work is counted
        return (clock.now - start).milliseconds
    }

    /// Full-request latency: tokenize `promptText`, chunked-prefill it, then greedy
    /// decode `decode` tokens — timed as a single span (tokenize → last token).
    /// Tokenization is INSIDE the window to match the Rust `end_to_end_latency`
    /// `total_time_ms`, which posts a string and tokenizes server-side in-band.
    /// `promptText` is built once up front to tokenize to exactly the cell's prefill
    /// count, so unlike the throughput primitives this path uses real text + the
    /// tokenizer rather than synthetic ids.
    nonisolated static func endToEndMillis(_ model: any LanguageModel, tokenizer: any MLXLMCommon.Tokenizer,
                               promptText: String, decode: Int, chunk: Int) -> Double {
        let cache = model.newCache(parameters: nil)
        let clock = ContinuousClock()
        let start = clock.now  // before tokenize — the span covers tokenize → last token
        let p = MLXArray(tokenizer.encode(text: promptText).map { Int32($0) }).reshaped(1, -1)
        let n = p.dim(1)
        let step = chunk > 0 ? chunk : n
        var logits = MLXArray()
        var i = 0
        while i < n {
            let end = min(i + step, n)
            logits = model(p[0 ..< 1, i ..< end], cache: cache)
            if end < n { asyncEval(logits[0, -1]) } else { eval(logits[0, -1]) }
            i = end
        }
        var y = argMax(logits[0, -1], axis: -1).reshaped(1, 1).asType(.int32)
        asyncEval(y)
        for _ in 0 ..< max(0, decode) {
            logits = model(y, cache: cache)
            y = argMax(logits[0, -1], axis: -1).reshaped(1, 1).asType(.int32)
            asyncEval(y)
        }
        eval(y)  // single sync so all tokenize + prefill + decode work is counted
        return (clock.now - start).milliseconds
    }

    /// Chunked prefill (fresh cache); returns the last token's logits `[vocab]`.
    /// Interior chunks are scheduled with `asyncEval` (non-blocking) so activations
    /// don't accumulate across the sequence; the final chunk is eval'd to materialize
    /// the returned logits.
    nonisolated static func lastLogits(_ model: any LanguageModel, _ toks: MLXArray, chunk: Int) -> MLXArray {
        let cache = model.newCache(parameters: nil)
        let n = toks.dim(1)
        let step = chunk > 0 ? chunk : n
        var last = MLXArray()
        var i = 0
        while i < n {
            let end = min(i + step, n)
            last = model(toks[0 ..< 1, i ..< end], cache: cache)[0, -1]
            if end < n { asyncEval(last) } else { eval(last) }
            i = end
        }
        return last
    }

    /// Greedy token generation: chunked prefill of `promptIds`, then argmax one
    /// token at a time until `eosId` or `maxTokens`. Pure token space — encoding
    /// the prompt and decoding the result to text is the caller's job. Unlike the
    /// throughput loops it reads each id back to the host (to test EOS and collect
    /// the output), trading speed for the exact output ids eval needs.
    nonisolated static func generateGreedy(_ model: any LanguageModel, promptIds: [Int],
                               maxTokens: Int, prefillChunkSize: Int, eosId: Int?)
        -> (ids: [Int], stoppedOnEos: Bool) {
        let cache = model.newCache(parameters: nil)
        let prompt = MLXArray(promptIds.map { Int32($0) }).reshaped(1, -1)
        let n = prompt.dim(1)
        let step = prefillChunkSize > 0 ? prefillChunkSize : n
        var logits = MLXArray()
        var i = 0
        while i < n {
            let end = min(i + step, n)
            logits = model(prompt[0 ..< 1, i ..< end], cache: cache)
            eval(logits[0, -1])
            i = end
        }
        var ids: [Int] = []
        // Authoritative stop signal: `true` iff we broke on the EOS token,
        // `false` iff the `maxTokens` cap bound generation first.
        var stoppedOnEos = false
        var y = argMax(logits[0, -1], axis: -1).reshaped(1, 1).asType(.int32)
        eval(y)
        for _ in 0 ..< max(0, maxTokens) {
            let tid = Int(y.item(Int32.self))
            if let eosId, tid == eosId { stoppedOnEos = true; break }
            ids.append(tid)
            logits = model(y, cache: cache)
            y = argMax(logits[0, -1], axis: -1).reshaped(1, 1).asType(.int32)
            eval(y)
        }
        return (ids, stoppedOnEos)
    }
}
