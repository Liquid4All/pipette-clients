import Testing
import Foundation
@testable import Pipette

/// Guards that the shared prompt-seed corpus is bundled and loadable. The corpus
/// is copied from `crates/pipette-ops/src/prompt_seed.txt` into the app bundle by
/// `ios/build-llama.sh`; if that copy or the resource bundling breaks,
/// `PromptSeed.corpus` would be missing and the llama benchmark prompts would be
/// wrong — caught here rather than on-device.
@Suite struct PromptSeedTests {
    @Test func corpusIsBundledAndSubstantial() {
        // The fleet corpus is ~24 KB of prose (Rust asserts > 20 000 chars for
        // 4096-token coverage); the old placeholder pangram was a few hundred.
        #expect(PromptSeed.corpus.count > 20_000)
    }

    @Test func syntheticTokenIdsAreInVocabAndExactLength() {
        let ids = PromptSeed.syntheticTokenIds(2500)
        #expect(ids.count == 2500)
        #expect(ids.allSatisfy { $0 >= 1 && $0 <= 1000 })
    }

    // MARK: - buildPromptText creates prompts of the needed size (llama + MLX e2e)

    // The benchmark prefill sizes we actually request. `buildPromptText` must land
    // on each EXACTLY, given a sane tokenizer, so the e2e prompt is the right
    // length. The real LFM2.5 tokenizers are exercised on-device
    // (`headlessrun metrics=promptseed`); these cover the algorithm in CI against
    // representative tokenizer shapes (mirrors the Rust `prompt_seed` tests).

    /// A variable 3–5 chars/token "subword" stand-in — lumpy enough to drive the
    /// bisection + tail-grow the way a real BPE tokenizer does, deterministically.
    static func subwordCount(_ s: String) -> Int {
        var tokens = 0, run = 0
        for ch in s {
            run += 1
            let stride = 3 + Int((ch.asciiValue ?? 97) % 3)   // 3, 4, or 5 chars/token
            if run >= stride { tokens += 1; run = 0 }
        }
        return tokens + (run > 0 ? 1 : 0)
    }

    @Test(arguments: [100, 256, 512, 1024, 2048])
    func buildPromptTextHitsNeededSizeOneToOne(_ target: Int) {
        // count == character length.
        let text = PromptSeed.buildPromptText(target: target) { $0.count }
        #expect(text.count == target)
    }

    @Test(arguments: [100, 256, 512, 1024, 2048])
    func buildPromptTextHitsNeededSizeLumpy(_ target: Int) {
        // A flat ≈3 chars/token tokenizer.
        let text = PromptSeed.buildPromptText(target: target) { $0.count / 3 }
        #expect(text.count / 3 == target)
    }

    @Test(arguments: [100, 256, 512, 1024, 2048])
    func buildPromptTextHitsNeededSizeSubword(_ target: Int) {
        // A lumpy variable-length tokenizer (the hardest of the three to hit).
        let text = PromptSeed.buildPromptText(target: target) { Self.subwordCount($0) }
        #expect(Self.subwordCount(text) == target)
    }

    @Test func buildPromptTextEmptyForZeroTarget() {
        #expect(PromptSeed.buildPromptText(target: 0) { $0.count }.isEmpty)
    }
}
