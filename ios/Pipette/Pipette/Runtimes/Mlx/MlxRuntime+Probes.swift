import Foundation
import MLX
import MLXHuggingFace
import MLXLMCommon
import Tokenizers

/// Device probes outside the benchmark path: none produces a `BenchmarkResultData`.
extension MlxRuntime {
    /// The MLX peer of `LlamaExecute.promptSeedCounts`; loads only the tokenizer.
    nonisolated static func promptSeedCounts(modelPath: String, targets: [Int]) async throws -> [(target: Int, got: Int)] {
        let tokenizer = try await #huggingFaceTokenizerLoader().load(from: URL(fileURLWithPath: modelPath))
        var counts: [(target: Int, got: Int)] = []
        for target in targets {
            let text = try await PromptSeed.buildPromptText(targetTokens: target) {
                tokenizer.encode(text: $0).count
            }
            counts.append((target, tokenizer.encode(text: text).count))
        }
        return counts
    }

    /// A GPU heat burst for the IMU-thermometer calibration (`metrics=calibrate`).
    nonisolated static func prefillBurst(_ model: any LanguageModel, tokens: Int,
                                         prefillChunk: Int) throws -> Double {
        defer { releaseModel() }
        let ids = MLXArray(PromptSeed.syntheticTokenIds(tokens)).reshaped(1, -1)
        let cache = try model.newCache(parameters: nil)
        let clock = ContinuousClock()
        let start = clock.now
        _ = prefill(model, ids, cache: cache, chunk: prefillChunk)
        return durationMs(clock.now - start)
    }

    /// The prompt's last-token top-5 and the greedy continuation's ids, for parity against
    /// llama and the reference.
    nonisolated static func coherence(modelPath: String, promptIds: [Int32], gen: Int,
                                      prefillChunk: Int = Int(defaultPrefillChunk)) async throws -> (top5: [Int], ids: [Int]) {
        try await withFreshModel(path: modelPath) { model in
            let cache = try model.newCache(parameters: nil)
            let last = prefill(model, MLXArray(promptIds).reshaped(1, -1), cache: cache,
                               chunk: prefillChunk)
            let top5 = topK(last, 5)
            var ids: [Int] = []
            var y = greedyToken(last)
            for _ in 0 ..< gen {
                ids.append(y.item(Int.self))
                y = greedyToken(model(y, cache: cache)[0, -1])
            }
            return (top5, ids)
        }
    }

    nonisolated static func topK(_ v1d: MLXArray, _ k: Int) -> [Int] {
        let idx = argPartition(-v1d, kth: k - 1, axis: -1)[..<k]
        let order = argSort(-takeAlong(v1d, idx, axis: -1), axis: -1)
        let sorted = takeAlong(idx, order, axis: -1)
        eval(sorted)
        return sorted.asArray(Int32.self).map(Int.init)
    }
}
