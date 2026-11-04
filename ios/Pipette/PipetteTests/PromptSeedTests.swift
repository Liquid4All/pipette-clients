import Testing
import Foundation
@testable import Pipette

/// `PromptSeed`, mirroring `pipette-ops`'s `prompt_seed.rs`. The builder cases port the
/// crate's own; real tokenizers are exercised on device with `metrics=promptseed`.
@Suite struct PromptSeedTests {
    /// `ios/build-llama.sh` copies `crates/pipette-ops/src/prompt_seed.txt` into the bundle;
    /// the crate asserts the same floor for 4096-token coverage.
    @Test func seedTextIsSubstantialNaturalLanguage() {
        #expect(PromptSeed.promptSeedText.utf8.count > 20_000)
    }

    @Test func syntheticTokenIdsAreInVocabAndExactLength() {
        let ids = PromptSeed.syntheticTokenIds(2500)
        #expect(ids.count == 2500)
        #expect(ids.allSatisfy { $0 >= 1 && $0 <= 1000 })
    }

    // MARK: - The crate's builder cases

    @Test func targetZeroReturnsEmptyWithoutTokenize() async throws {
        var calls = 0
        let text = try await PromptSeed.buildPromptTextFrom("seed", targetTokens: 0) {
            calls += 1
            return $0.utf8.count
        }
        #expect(text.isEmpty)
        #expect(calls == 0)
    }

    @Test(arguments: [1, 5, 17, 128, 1024, 4096, 8192])
    func hitsTargetOnByteTokenizerAcrossSizes(_ target: Int) async throws {
        var calls = 0
        let text = try await PromptSeed.buildPromptTextFrom(
            "abcdefghijklmnopqrstuvwxyz", targetTokens: target
        ) {
            calls += 1
            return $0.utf8.count
        }
        #expect(text.utf8.count == target)
        #expect(calls < 32, "byte tokenizer should converge in <32 calls, got \(calls)")
    }

    @Test(arguments: [1, 5, 7, 10, 100])
    func hitsTargetOnLumpyTokenizer(_ target: Int) async throws {
        var calls = 0
        let text = try await PromptSeed.buildPromptTextFrom("abcdefghi", targetTokens: target) {
            calls += 1
            return $0.utf8.count / 3
        }
        #expect(text.utf8.count / 3 == target)
        #expect(calls < PromptSeed.tokenizeBudget)
    }

    /// A count that reaches the target early, drops, and overshoots later: the small-target
    /// prefix scan finds the early hit that a bisection over the whole pool would step past.
    @Test func findsTargetWhenTokenCountDropsAfterAnEarlyExactPrefix() async throws {
        let text = try await PromptSeed.buildPromptTextFrom("abcdefghijkl", targetTokens: 2) {
            switch $0.utf8.count {
            case 0: 0
            case 1: 1
            case 2: 2
            case 3 ... 47: 1
            default: 3
            }
        }
        #expect(text.utf8.count == 2)
    }

    @Test func errorsOnEmptySeed() async {
        var calls = 0
        await #expect(throws: PromptSeedError.emptySeed) {
            _ = try await PromptSeed.buildPromptTextFrom("", targetTokens: 10) {
                calls += 1
                return $0.utf8.count
            }
        }
        #expect(calls == 0)
    }

    @Test func publicEntryUsesEmbeddedSeed() async throws {
        let text = try await PromptSeed.buildPromptText(targetTokens: 50) { $0.utf8.count }
        #expect(text.utf8.count == 50)
        #expect(PromptSeed.promptSeedText.contains(String(text.prefix(20))))
    }

    // MARK: - The refusals the crate bails with

    /// A tokenizer the pool can never reach spends the budget and fails rather than
    /// returning a prompt of the wrong size.
    @Test func aTokenizerThatNeverReachesTheTargetExhaustsTheBudget() async {
        var calls = 0
        await #expect(throws: PromptSeedError.budgetExceeded) {
            _ = try await PromptSeed.buildPromptTextFrom("abc", targetTokens: 5) { _ in
                calls += 1
                return 1
            }
        }
        #expect(calls == PromptSeed.tokenizeBudget)
    }

    /// Every extension overshoots from the best prefix below the target, so the tail cannot
    /// land on it: refused by name, with both counts.
    @Test func aTailThatCannotAdvanceIsRefused() async {
        await #expect(throws: PromptSeedError.tailGrowStuck(current: 39, target: 40)) {
            _ = try await PromptSeed.buildPromptTextFrom("abcdefghij", targetTokens: 40) {
                switch $0.utf8.count {
                case 0: 0
                case 1 ..< 80: 39
                default: 41
                }
            }
        }
    }

    @Test func aSeedThatTokenizesToNothingIsRefused() async {
        await #expect(throws: PromptSeedError.zeroTokenSeed) {
            _ = try await PromptSeed.buildPromptTextFrom("abc", targetTokens: 5) { _ in 0 }
        }
    }

    // MARK: - The benchmark sizes, over the real seed

    /// A variable 3–5 chars/token stand-in, lumpy enough to drive the bisection and the
    /// tail the way a real BPE tokenizer does, deterministically.
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
    func hitsEveryBenchmarkSizeOneToOne(_ target: Int) async throws {
        let text = try await PromptSeed.buildPromptText(targetTokens: target) { $0.count }
        #expect(text.count == target)
    }

    @Test(arguments: [100, 256, 512, 1024, 2048])
    func hitsEveryBenchmarkSizeThreeCharsPerToken(_ target: Int) async throws {
        let text = try await PromptSeed.buildPromptText(targetTokens: target) { $0.count / 3 }
        #expect(text.count / 3 == target)
    }

    @Test(arguments: [100, 256, 512, 1024, 2048])
    func hitsEveryBenchmarkSizeSubword(_ target: Int) async throws {
        let text = try await PromptSeed.buildPromptText(targetTokens: target) {
            Self.subwordCount($0)
        }
        #expect(Self.subwordCount(text) == target)
    }
}
