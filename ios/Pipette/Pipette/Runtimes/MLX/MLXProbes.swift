import Foundation
import MLX
import MLXHuggingFace
import MLXLMCommon
import Tokenizers

/// MLX device probes that live alongside the benchmark path but aren't part of it:
/// a GPU heat burst for IMU-thermometer calibration, a coherence check that
/// returns concrete output ids for parity comparison vs llama / the reference, and
/// the prompt-seed exact-sizing check. Kept separate from `MLXBenchmark` because
/// none produces a `BenchmarkResult`.
extension MLXRuntime {
    /// For each `target`, build a prompt with `PromptSeed.buildPromptText` under
    /// this model's HF tokenizer and return the count it actually produced — the
    /// MLX peer of `LlamaBenchmark.promptSeedCounts`, for verifying exact prompt
    /// sizing (`metrics=promptseed`). Loads only the tokenizer (not the weights).
    nonisolated static func promptSeedCounts(modelPath: String, targets: [Int]) async throws -> [(target: Int, got: Int)] {
        let tokenizer = try await #huggingFaceTokenizerLoader().load(from: URL(fileURLWithPath: modelPath))
        return targets.map { target in
            let text = PromptSeed.buildPromptText(target: target) { tokenizer.encode(text: $0).count }
            return (target, tokenizer.encode(text: text).count)
        }
    }

    /// One chunked prefill of `tokens` synthetic ids — a GPU heat burst used by the
    /// IMU-thermometer calibration (`metrics=calibrate`). Returns elapsed ms, -1 if unloaded.
    nonisolated static func prefillBurst(_ model: any LanguageModel, tokens: Int, prefillChunk: Int) -> Double {
        defer { releaseModel() }
        return prefillMillis(model, tokens: tokens, chunk: prefillChunk)
    }

    /// Greedy-continue `promptIds` for `gen` tokens (fresh load). Returns the
    /// last-token top-5 of the *prompt* (for top-token parity) and the generated
    /// token ids (for coherence comparison vs llama / the reference). Uses `.item`
    /// per token (this isn't a throughput path — we need the ids).
    nonisolated static func coherence(modelPath: String, promptIds: [Int32], gen: Int,
                          prefillChunk: Int = 512) async throws -> (top5: [Int], ids: [Int]) {
        try await withFreshModel(path: modelPath) { model in
            let cache = model.newCache(parameters: nil)
            var logits = model(MLXArray(promptIds).reshaped(1, -1), cache: cache)
            eval(logits[0, -1])
            let top5 = topK(logits[0, -1], 5)
            var ids: [Int] = []
            var y = argMax(logits[0, -1], axis: -1).reshaped(1, 1).asType(.int32)
            for _ in 0 ..< gen {
                ids.append(y.item(Int.self))
                logits = model(y, cache: cache)
                y = argMax(logits[0, -1], axis: -1).reshaped(1, 1).asType(.int32)
            }
            return (top5, ids)
        }
    }

    /// Indices of the top-`k` logits (descending), as a plain `[Int]`.
    nonisolated static func topK(_ v1d: MLXArray, _ k: Int) -> [Int] {
        let idx = argPartition(-v1d, kth: k - 1, axis: -1)[..<k]
        let order = argSort(-takeAlong(v1d, idx, axis: -1), axis: -1)
        let sorted = takeAlong(idx, order, axis: -1)
        eval(sorted)
        return sorted.asArray(Int32.self).map(Int.init)
    }
}
