import Foundation

/// The shared natural-language passage every benchmark client builds synthetic prompts of
/// an exact token count from: the crate's `pipette-ops/src/prompt_seed.rs`.
///
/// Each runtime feeds a tokenizer callback into `buildPromptText`, which returns a string
/// that tokenizes to exactly the target. The string is sent to the engine as a prompt, so
/// tokenization stays inside the timed window for end-to-end latency.
nonisolated enum PromptSeed {
    /// The crate's `PROMPT_SEED_TEXT`, the same file the Rust clients embed, copied into the
    /// app bundle by `ios/build-llama.sh` so iOS numbers stay comparable to the fleet.
    static let promptSeedText: String = {
        guard let url = Bundle.main.url(forResource: "prompt_seed", withExtension: "txt"),
              let text = try? String(contentsOf: url, encoding: .utf8)
        else {
            preconditionFailure(
                "prompt_seed.txt missing from the app bundle; ios/build-llama.sh copies it "
                    + "from crates/pipette-ops/src/ into Generated/PromptSeed/")
        }
        return text
    }()

    /// Cap on `count` callback invocations per build: the crate's `TOKENIZE_BUDGET`. The
    /// expected shape is one calibration call, a handful of pool-extension calls,
    /// logarithmic bisection, and a short tail-growth pass.
    static let tokenizeBudget = 64
    /// The crate's `PREFIX_SCAN_WINDOW_CHARS`.
    static let prefixScanWindowChars = 32

    /// Candidate suffixes for tail growth: the crate's `TAIL_EXTENSIONS`. Each candidate is
    /// re-tokenized; the builder only accepts one that moves the count toward the target.
    static let tailExtensions = [" ", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ",", "."]

    /// `n` in-vocab synthetic ids (cycling 1...1000) for MLX's throughput prompts, which
    /// need a token *count* and no tokenizer. No crate counterpart: the crate's MLX server
    /// repeats the seed passage's tokens (`repeat_token_sequence`). Aligning the content is
    /// deferred (`docs/pipette-ios/execution-alignment.md`, "Deferred").
    static func syntheticTokenIds(_ n: Int) -> [Int32] {
        (0 ..< max(1, n)).map { Int32(($0 % 1000) + 1) }
    }

    /// Build a string that tokenizes to exactly `targetTokens` under the caller's `count`
    /// callback: the crate's `build_prompt_text`. The caller's tokenize settings must match
    /// the inference path's settings for string prompts.
    ///
    /// `async` because Apple Foundation counts tokens asynchronously; llama and MLX pass
    /// synchronous counters, which run without suspending.
    static func buildPromptText(targetTokens: Int,
                                count: (String) async throws -> Int) async throws -> String {
        try await buildPromptTextFrom(promptSeedText, targetTokens: targetTokens, count: count)
    }

    /// `buildPromptText` over an arbitrary seed: the crate's `build_prompt_text_from`,
    /// ported step for step. Positions are UTF-8 byte offsets clamped to character
    /// boundaries, as the crate's `&str` slicing is, so `charsPerToken` and the scan
    /// window count what the crate's count.
    static func buildPromptTextFrom(_ seed: String, targetTokens: Int,
                                    count: (String) async throws -> Int) async throws -> String {
        guard targetTokens > 0 else { return "" }
        let seedBytes = Array(seed.utf8)
        guard !seedBytes.isEmpty else { throw PromptSeedError.emptySeed }
        let target = targetTokens
        let budget = Budget()

        let seedCount = try await budget.count(count, seed)
        guard seedCount > 0 else { throw PromptSeedError.zeroTokenSeed }
        let charsPerToken = Double(seedBytes.count) / Double(seedCount)

        let initialChars = Int(Double(target) * charsPerToken * 1.20)
        let repeats = initialChars / seedBytes.count + 2
        var pool = String(repeating: seed, count: repeats)

        var fullCount = try await budget.count(count, pool)
        while fullCount < target {
            pool += seed
            fullCount = try await budget.count(count, pool)
        }
        if fullCount == target { return pool }
        let bytes = Array(pool.utf8)

        if target <= prefixScanWindowChars,
           case .done(let text) = try await scanPrefixWindow(bytes, start: 0, target: target,
                                                             budget: budget, count: count) {
            return text
        }

        var lo = 0
        var hi = bytes.count
        while hi - lo > 1 {
            let mid = clampCharBoundary(bytes, lo + (hi - lo) / 2)
            if mid == lo || mid == hi { break }
            let countAtMid = try await budget.count(count, prefix(bytes, mid))
            if countAtMid < target {
                lo = mid
            } else if countAtMid == target {
                return prefix(bytes, mid)
            } else {
                hi = mid
            }
        }

        var text: String
        var current: Int
        switch try await scanPrefixWindow(bytes, start: lo, target: target,
                                          budget: budget, count: count) {
        case .done(let done): return done
        case .under(let candidate): (text, current) = (candidate.text, candidate.count)
        }

        while current < target {
            switch try await tailStep(text, current: current, target: target,
                                      budget: budget, count: count) {
            case .done(let candidate)?:
                return candidate
            case .progress(let candidate, let candidateCount)?:
                text = candidate
                current = candidateCount
            case nil:
                throw PromptSeedError.tailGrowStuck(current: current, target: target)
            }
        }
        throw PromptSeedError.bisectionInvariantBroke(current: current, target: target)
    }

    // MARK: - The crate's private steps

    private nonisolated struct PrefixCandidate {
        let text: String
        let count: Int
    }

    private enum PrefixScan {
        case done(String)
        case under(PrefixCandidate)
    }

    /// The crate's `scan_prefix_window`: from `start`, up to `prefixScanWindowChars`
    /// characters forward, take an exact hit or keep the longest-counting prefix below
    /// the target.
    private static func scanPrefixWindow(_ pool: [UInt8], start: Int, target: Int,
                                         budget: Budget,
                                         count: (String) async throws -> Int) async throws -> PrefixScan {
        var pos = min(start, pool.count)
        var best: PrefixCandidate?
        var steps = 0
        while true {
            let text = prefix(pool, pos)
            let tokenCount = try await budget.count(count, text)
            if tokenCount == target { return .done(text) }
            if tokenCount < target {
                let shouldReplace = best.map {
                    tokenCount > $0.count
                        || (tokenCount == $0.count && text.utf8.count > $0.text.utf8.count)
                } ?? true
                if shouldReplace { best = PrefixCandidate(text: text, count: tokenCount) }
            }
            if steps >= prefixScanWindowChars || pos >= pool.count { break }
            pos = nextCharBoundary(pool, pos)
            steps += 1
        }
        guard let best else { throw PromptSeedError.noPrefixBelowTarget }
        return .under(best)
    }

    private enum TailStep {
        case done(String)
        case progress(text: String, count: Int)
    }

    /// The crate's `tail_step`: the first extension that lands on the target or advances
    /// toward it without overshooting, or `nil` when none does.
    private static func tailStep(_ text: String, current: Int, target: Int, budget: Budget,
                                 count: (String) async throws -> Int) async throws -> TailStep? {
        for ext in tailExtensions {
            let candidate = text + ext
            let candidateCount = try await budget.count(count, candidate)
            if candidateCount == target { return .done(candidate) }
            if candidateCount > current && candidateCount < target {
                return .progress(text: candidate, count: candidateCount)
            }
        }
        return nil
    }

    /// The crate's `Budget`: every `count` call goes through it, and the 65th fails.
    private final class Budget {
        private var remaining = PromptSeed.tokenizeBudget

        func count(_ count: (String) async throws -> Int, _ text: String) async throws -> Int {
            guard remaining > 0 else { throw PromptSeedError.budgetExceeded }
            remaining -= 1
            return try await count(text)
        }
    }

    private static func isCharBoundary(_ bytes: [UInt8], _ pos: Int) -> Bool {
        pos == 0 || pos >= bytes.count || bytes[pos] & 0xC0 != 0x80
    }

    /// The crate's `clamp_char_boundary`.
    private static func clampCharBoundary(_ bytes: [UInt8], _ pos: Int) -> Int {
        guard pos < bytes.count else { return bytes.count }
        var pos = pos
        while pos > 0 && !isCharBoundary(bytes, pos) { pos -= 1 }
        return pos
    }

    /// The crate's `next_char_boundary`.
    private static func nextCharBoundary(_ bytes: [UInt8], _ pos: Int) -> Int {
        var next = min(pos + 1, bytes.count)
        while next < bytes.count && !isCharBoundary(bytes, next) { next += 1 }
        return next
    }

    /// The first `length` bytes as text; `length` is always a character boundary.
    private static func prefix(_ bytes: [UInt8], _ length: Int) -> String {
        String(decoding: bytes[0 ..< length], as: UTF8.self)
    }
}

/// Why a prompt could not be built to its target: the crate's `build_prompt_text` bails,
/// with their messages. A prompt of the wrong length describes another workload, so the
/// cell fails rather than measuring it.
nonisolated enum PromptSeedError: Error, Equatable, LocalizedError {
    case emptySeed
    case zeroTokenSeed
    case budgetExceeded
    case noPrefixBelowTarget
    case tailGrowStuck(current: Int, target: Int)
    case bisectionInvariantBroke(current: Int, target: Int)

    var errorDescription: String? {
        switch self {
        case .emptySeed:
            "seed text is empty"
        case .zeroTokenSeed:
            "seed text tokenizes to zero tokens"
        case .budgetExceeded:
            "prompt builder exceeded tokenize budget of \(PromptSeed.tokenizeBudget) calls"
        case .noPrefixBelowTarget:
            "no prefix below target near bisection boundary"
        case let .tailGrowStuck(current, target):
            "tail-grow stuck at \(current) tokens, target \(target): "
                + "no extension advanced without overshooting"
        case let .bisectionInvariantBroke(current, target):
            "builder ended at \(current) tokens, target was \(target): bisection invariant broke"
        }
    }
}
