import Foundation

/// The single home for benchmark throughput/memory prompt material, so each
/// runtime draws on one source instead of defining its own.
///
/// The accessors aren't interchangeable — they differ by *content sensitivity*.
/// llama.cpp's per-token cost depends on the actual tokens, so it tokenizes real
/// text (`corpus`); MLX's cost depends only on sequence *length*, so it uses
/// `syntheticTokenIds` and never loads a tokenizer in the throughput path.
nonisolated enum PromptSeed {
    /// Real natural-language text for the content-sensitive path (llama.cpp),
    /// which tiles + tokenizes it to the target count. It's the *same* corpus the
    /// Rust CLIs and Android use — `crates/pipette-ops/src/prompt_seed.txt`,
    /// embedded there via `include_str!` — copied verbatim into the app bundle at
    /// build time by `ios/build-llama.sh`, so iOS llama numbers stay comparable to
    /// the fleet. Loaded lazily from the bundle on first use.
    static let corpus: String = {
        guard let url = Bundle.main.url(forResource: "prompt_seed", withExtension: "txt"),
              let text = try? String(contentsOf: url, encoding: .utf8)
        else {
            preconditionFailure(
                "prompt_seed.txt missing from the app bundle; ios/build-llama.sh copies it "
                    + "from crates/pipette-ops/src/ into Generated/PromptSeed/")
        }
        return text
    }()

    /// `n` in-vocab synthetic ids (cycling 1...1000) for content-independent
    /// runtimes (MLX) — token *count* is all they need, so no tokenizer is loaded.
    static func syntheticTokenIds(_ n: Int) -> [Int32] {
        (0 ..< max(1, n)).map { Int32(($0 % 1000) + 1) }
    }

    /// Build prompt TEXT from `corpus` that tokenizes to **exactly** `target`
    /// tokens under the caller's `count` tokenizer — the Swift peer of
    /// `pipette_ops::prompt_seed::build_prompt_text`. Both runtimes use it for end-to-end
    /// latency, where tokenization is timed in-window, so the prompt must be a
    /// real string of an exact length (each engine passes its own tokenizer's
    /// `count`). Tile a pool past `target`, bisect the character length to a
    /// prefix at/under `target`, then append single characters to land precisely
    /// on `target` (tokenizers aren't monotonic in length near the boundary).
    static func buildPromptText(target: Int, count: (String) throws -> Int) rethrows -> String {
        guard target > 0 else { return "" }
        let chars = Array(corpus)   // [Character]; the corpus is ASCII
        func tokens(_ slice: ArraySlice<Character>) throws -> Int { try count(String(slice)) }

        // 1. Tile to a pool that tokenizes to at least `target` (estimate, then grow).
        //    Size the estimate to ~target tokens (+30% headroom) so the bisection
        //    below doesn't re-tokenize a needlessly long string — the grow loop is
        //    what actually guarantees sufficiency.
        let seedCount = try tokens(chars[...])
        guard seedCount > 0 else { return "" }
        let charsPerToken = Double(chars.count) / Double(seedCount)
        var pool = chars
        let minChars = Int(Double(target) * charsPerToken * 1.3)
        while pool.count < minChars { pool.append(contentsOf: chars) }
        while try tokens(pool[...]) < target { pool.append(contentsOf: chars) }

        // 2. Bisect the character length for a prefix at/under `target`.
        var lo = 0, hi = pool.count
        while hi - lo > 1 {
            let mid = lo + (hi - lo) / 2
            let c = try tokens(pool[0 ..< mid])
            if c == target { return String(pool[0 ..< mid]) }
            if c < target { lo = mid } else { hi = mid }
        }

        // 3. Scan a small window past the boundary for an exact hit; else keep the
        //    best prefix under `target` to grow the tail from.
        var bestEnd = lo, bestCount = try tokens(pool[0 ..< lo])
        var end = lo
        while end <= pool.count && end - lo <= 32 {
            let c = try tokens(pool[0 ..< end])
            if c == target { return String(pool[0 ..< end]) }
            if c < target && c >= bestCount { bestEnd = end; bestCount = c }
            end += 1
        }

        // 4. Tail-grow: append single characters until exactly `target`.
        var text = String(pool[0 ..< bestEnd])
        var current = bestCount
        let extensions = [" ", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ",", "."]
        var steps = 0
        while current < target && steps < 64 {
            steps += 1
            var advanced = false
            for ext in extensions {
                let candidate = text + ext
                let c = try count(candidate)
                if c == target { return candidate }
                if c > current && c < target { text = candidate; current = c; advanced = true; break }
            }
            if !advanced { break }
        }
        return text   // best effort if the tokenizer can't land exactly (rare for prose)
    }
}
