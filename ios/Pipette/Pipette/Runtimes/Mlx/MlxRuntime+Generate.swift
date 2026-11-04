import Foundation
import MLX
import MLXLMCommon

/// Generation primitives beneath `MlxExecute`, in pure token space; the harness does the
/// timing.
extension MlxRuntime {
    /// Returns the last position's logits `[vocab]`, evaluated. Interior chunks are
    /// `asyncEval`ed so activations don't accumulate across the sequence.
    nonisolated static func prefill(_ model: any LanguageModel, _ tokens: MLXArray,
                                    cache: [any KVCache], chunk: Int) -> MLXArray {
        let n = tokens.dim(1)
        let step = chunk > 0 ? chunk : n
        var last = MLXArray()
        var i = 0
        while i < n {
            let end = min(i + step, n)
            last = model(tokens[0 ..< 1, i ..< end], cache: cache)[0, -1]
            if end < n { asyncEval(last) } else { eval(last) }
            i = end
        }
        return last
    }

    /// The windows `mlx_lm`'s `generate_step` fills the cache with before its first sampled
    /// forward: the first `count - 1` tokens in `chunk`-token windows (one window at
    /// `chunk <= 0`). The last token is held back for that forward.
    nonisolated static func cacheFillRanges(count: Int, chunk: Int) -> [Range<Int>] {
        let fill = count - 1
        guard fill > 0 else { return [] }
        let step = chunk > 0 ? chunk : fill
        return stride(from: 0, to: fill, by: step).map { $0 ..< min($0 + step, fill) }
    }

    /// `prefill` in `mlx_lm`'s shape: each `cacheFillRanges` window only fills `cache` and
    /// MLX's buffer cache is cleared after it; the held-back last token alone yields the
    /// logits `[vocab]`, evaluated. Windows are the cell's `chunk`, not the server's 2048.
    /// Only `max_memory_usage` runs it: moving the timing cells needs its own timing run.
    nonisolated static func prefillAsMlxLm(_ model: any LanguageModel, _ tokens: MLXArray,
                                           cache: [any KVCache], chunk: Int) -> MLXArray {
        let n = tokens.dim(1)
        guard n > 0 else { return prefill(model, tokens, cache: cache, chunk: chunk) }
        for window in cacheFillRanges(count: n, chunk: chunk) {
            _ = model(tokens[0 ..< 1, window], cache: cache)
            // Synchronous, so the window's activations are free for `clearCache` to return.
            eval(cache)
            MLX.Memory.clearCache()
        }
        let last = model(tokens[0 ..< 1, (n - 1) ..< n], cache: cache)[0, -1]
        eval(last)
        return last
    }

    /// Not evaluated: the caller decides whether that sync is timed.
    nonisolated static func greedyToken(_ logits: MLXArray) -> MLXArray {
        argMax(logits, axis: -1).reshaped(1, 1).asType(.int32)
    }

    /// As MLXLMCommon's TokenIterator: the token stays on-device and each step is
    /// `asyncEval`ed, so the CPU builds the next graph while the GPU computes. The final
    /// `eval` puts all decode work inside whoever times this.
    @discardableResult
    nonisolated static func decodeGreedy(_ model: any LanguageModel, from seed: MLXArray,
                                         cache: [any KVCache], count: Int) -> Int {
        var y = seed
        let steps = max(0, count)
        for _ in 0 ..< steps {
            y = greedyToken(model(y, cache: cache)[0, -1])
            asyncEval(y)
        }
        eval(y)
        return steps
    }

    /// For `max_memory_usage`, which measures what the workload provokes, not its time.
    nonisolated static func generate(_ model: any LanguageModel, prefill: Int, decode: Int,
                                    chunk: Int) throws {
        let cache = try model.newCache(parameters: nil)
        let ids = MLXArray(PromptSeed.syntheticTokenIds(prefill)).reshaped(1, -1)
        let seed = greedyToken(prefillAsMlxLm(model, ids, cache: cache, chunk: chunk))
        eval(seed)
        decodeGreedy(model, from: seed, cache: cache, count: decode)
    }

    /// What `mlx_lm.server` reads off the same request: an unpinned cut is its default,
    /// which disables it.
    nonisolated static func evalParameters(_ sampling: Sampling, seed: UInt32?)
        -> GenerateParameters {
        GenerateParameters(temperature: Float(sampling.temperature.value),
                           topP: Float(sampling.topP?.value ?? 1),
                           topK: Int(sampling.topK?.value ?? 0),
                           minP: Float(sampling.minP?.value ?? 0), seed: seed.map(UInt64.init))
    }

    /// One per sample: a seeded draw walks one random state.
    nonisolated static func evalSampler(_ sampling: Sampling, seed: UInt32?) -> any LogitSampler {
        evalParameters(sampling, seed: seed).sampler()
    }

    /// The samplers index `[batch, vocab]`: top-k's `[0..., k...]` traps on 1-D.
    nonisolated static func sampledToken(_ logits: MLXArray, with sampler: any LogitSampler)
        -> MLXArray {
        sampler.sample(logits: logits.reshaped(1, -1)).reshaped(1, 1).asType(.int32)
    }

    /// The eval path. Unlike the throughput loops it reads each id back to the host, trading
    /// speed for the exact ids eval needs.
    nonisolated static func generate(_ model: any LanguageModel, promptIds: [Int],
                                     maxTokens: Int, prefillChunkSize: Int,
                                     eosIds: Set<Int>, sampling: Sampling, seed: UInt32?)
        throws -> (ids: [Int], stoppedOnEos: Bool) {
        let sampler = evalSampler(sampling, seed: seed)
        let cache = try model.newCache(parameters: nil)
        let prompt = MLXArray(promptIds.map { Int32($0) }).reshaped(1, -1)
        var logits = prefill(model, prompt, cache: cache, chunk: prefillChunkSize)
        var ids: [Int] = []
        var stoppedOnEos = false
        var y = sampledToken(logits, with: sampler)
        eval(y)
        for _ in 0 ..< max(0, maxTokens) {
            let tid = Int(y.item(Int32.self))
            if eosIds.contains(tid) { stoppedOnEos = true; break }
            ids.append(tid)
            logits = model(y, cache: cache)[0, -1]
            y = sampledToken(logits, with: sampler)
            eval(y)
        }
        return (ids, stoppedOnEos)
    }
}
