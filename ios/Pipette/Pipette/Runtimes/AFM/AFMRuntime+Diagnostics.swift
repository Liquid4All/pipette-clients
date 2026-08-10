import Foundation
import FoundationModels

/// Dev-only diagnostics for the Apple foundation model — NOT part of the engine's
/// benchmark API. These probes established the on-device tokenizer's behavior
/// (the fixed template/BOS wrapper, exact generated-token counts, and the cap
/// semantics) and are kept so the findings can be re-verified when the OS updates.
/// Driven from the headless runner via `metrics=tokprobe|genprobe|capprobe|enforceprobe`.
extension AFMRuntime {
    /// Probe `tokenCount` on short inputs to expose the fixed wrapper (template / BOS /
    /// special tokens) vs. content tokens. `tokenCount("") > 0` ⇒ the wrapper is counted.
    nonisolated static func tokenProbe(progress: @escaping (String) -> Void) async {
        progress("availability=\(availabilityText())")
        guard #available(iOS 26.4, *) else { progress("tokenCount(for:) needs iOS 26.4+"); return }
        let model = SystemLanguageModel.default
        let inputs = ["", "Hi", "Hi there", "Hi Hi", "Hi Hi Hi",
                      "the", "the the", "Hello, world!", "antidisestablishmentarianism"]
        for s in inputs {
            let c = (try? await model.tokenCount(for: s)).map(String.init) ?? "error"
            progress("tokenCount(\"\(s)\") = \(c)")
        }
    }

    /// Generate with a small `maximumResponseTokens` cap, capture the output TEXT, then
    /// re-tokenize it — checks whether "tokens generated" (the cap) equals "tokens in
    /// the rendered output" (detokenize→retokenize is not always identity).
    nonisolated static func generationTokenProbe(progress: @escaping (String) -> Void) async {
        progress("availability=\(availabilityText())")
        guard case .available = availability() else {
            progress("model not available"); return
        }
        let wrapper = await tokenCount(of: "") ?? 0
        progress("template wrapper = \(wrapper) tokens")
        for cap in [1, 5, 10, 20] {
            do {
                let session = LanguageModelSession()
                let opts = GenerationOptions(sampling: .greedy, maximumResponseTokens: cap)
                var text = ""
                for try await partial in session.streamResponse(to: "Write a long story about a dragon.", options: opts) {
                    text = partial.content
                }
                let raw = await tokenCount(of: text) ?? -1
                let content = raw >= 0 ? raw - wrapper : -1
                let words = text.split(whereSeparator: { $0 == " " || $0 == "\n" }).count
                progress("cap=\(cap) → words=\(words) raw=\(raw) content=\(content) (chars=\(text.count))")
                progress("cap=\(cap) text=\"\(text.replacingOccurrences(of: "\n", with: "⏎"))\"")
            } catch {
                progress("cap=\(cap) ERROR \(error)")
            }
        }
    }

    /// Request a large cap (200) with prompts that naturally want a SHORT answer — does
    /// the model stop at EOS, or pad to the cap? If it stops short, `maximumResponseTokens`
    /// is a ceiling, so decode-throughput prompts must elicit open-ended output.
    nonisolated static func capRespectProbe(progress: @escaping (String) -> Void) async {
        progress("availability=\(availabilityText())")
        guard case .available = availability() else {
            progress("model not available"); return
        }
        let wrapper = await tokenCount(of: "") ?? 0
        let cap = 200
        let prompts = [
            "What is 2 + 2? Reply with only the number.",
            "What color is the sky? Answer in one word.",
            "Say hello.",
        ]
        for p in prompts {
            do {
                let session = LanguageModelSession()
                let opts = GenerationOptions(sampling: .greedy, maximumResponseTokens: cap)
                var text = ""
                for try await partial in session.streamResponse(to: p, options: opts) { text = partial.content }
                let content = (await tokenCount(of: text)).map { $0 - wrapper } ?? -1
                progress("cap=\(cap) prompt=\"\(p)\" → content_tokens=\(content) (\(content < cap ? "stopped early ✓" : "hit cap"))")
                progress("  text=\"\(text.replacingOccurrences(of: "\n", with: "⏎"))\"")
            } catch {
                progress("cap=\(cap) prompt=\"\(p)\" ERROR \(error)")
            }
        }
    }

    /// EXPERIMENT (PIP AFM): does guided generation (constrained decoding) tank
    /// decode throughput vs. free generation? Decodes a fixed `cap`-token workload
    /// both ways and reports tokens/sec + the slowdown factor.
    ///
    /// RESULT (iPhone 17 Pro, iOS 26.x, thermal-gated ≤36°C, both orderings): guided
    /// is within run-to-run noise of free (~0.96–0.97×, i.e. no penalty), so guided
    /// generation is the preferred way to *enforce* a fixed token count for AFM
    /// decode/e2e. Full write-up: docs/pipette-ios/afm-token-enforcement.md.
    ///   - free:   plain `streamResponse` + a non-terminating counting prompt; the
    ///             `maximumResponseTokens` cap binds → ~`cap` free-decoded tokens.
    ///   - guided: `streamResponse(generating:)` of a `[String]` forced with
    ///             `@Guide(.minimumCount(...))` (>> cap), so the cap truncates the
    ///             array mid-stream → ~`cap` tokens decoded under a per-token logit
    ///             mask + JSON structure. Same token budget, so the wall-time ratio
    ///             is the constrained-decoding tax. Real device + Apple Intelligence.
    nonisolated static func enforcementProbe(progress: @escaping (String) -> Void) async {
        progress("availability=\(availabilityText())")
        guard case .available = availability() else {
            progress("model not available"); return
        }
        let cap = 100
        let reps = 5
        let opts = GenerationOptions(sampling: .greedy, maximumResponseTokens: cap)
        let freePrompt = AFMRuntime.countingSeed
        let guidedPrompt = "List as many short common English words as you can, one per element."

        // Thermal readiness gate (PIPETTE_PRIVATE_THERMAL build → real SoC die temp):
        // block before every measured rep until the device has cooled below the
        // threshold, so heat can't confound the guided-vs-free comparison.
        let readiness = BenchmarkReadiness(cancelFlag: CancelFlag()) { progress("\($0)") }
        func cooled(_ label: String) -> Bool {
            switch readiness.waitUntilReady() {
            case .ready: return true
            case .cancelled: progress("\(label): cancelled"); return false
            case .timedOut(let o): progress("\(label): readiness timed out (\(o)), skipping"); return false
            }
        }
        func mean(_ a: [Double]) -> Double { a.isEmpty ? 0 : a.reduce(0, +) / Double(a.count) }

        // Untimed warm-up (spins up the model out-of-process).
        _ = try? await freeDecodeMs(prompt: freePrompt,
                                    opts: GenerationOptions(sampling: .greedy, maximumResponseTokens: 1))

        // GUIDED first (guided → free) to invert the previous run's ordering; with the
        // cooldown gate each rep starts from the same thermal baseline regardless.
        var guidedMs: [Double] = []
        for i in 0 ..< reps {
            guard cooled("guided r\(i + 1)") else { continue }
            do {
                let (ms, capped) = try await guidedDecodeMs(prompt: guidedPrompt, opts: opts)
                guidedMs.append(ms)
                progress(String(format: "guided r%d: decode=%.1fms tps≈%.1f (%@)",
                                i + 1, ms, ms > 0 ? Double(cap) / (ms / 1000) : 0,
                                capped ? "cap bound ✓" : "completed before cap ⚠︎"))
            } catch { progress("guided r\(i + 1) ERROR \(error)") }
        }

        var freeMs: [Double] = []
        for i in 0 ..< reps {
            guard cooled("free r\(i + 1)") else { continue }
            do {
                let (ms, tokens) = try await freeDecodeMs(prompt: freePrompt, opts: opts)
                freeMs.append(ms)
                progress(String(format: "free r%d: decode=%.1fms tokens≈%d tps≈%.1f",
                                i + 1, ms, tokens, ms > 0 ? Double(tokens) / (ms / 1000) : 0))
            } catch { progress("free r\(i + 1) ERROR \(error)") }
        }

        let fm = mean(freeMs), gm = mean(guidedMs)
        progress(String(format: "SUMMARY cap=%d reps=%d guided=%.1fms (%.1f tps) free=%.1fms (%.1f tps) slowdown=%.2fx",
                        cap, reps, gm, gm > 0 ? Double(cap) / (gm / 1000) : 0,
                        fm, fm > 0 ? Double(cap) / (fm / 1000) : 0, fm > 0 ? gm / fm : 0))
    }

    /// Free streamed generation; returns decode wall-time (first→last token) and the
    /// re-tokenized output-token count.
    nonisolated private static func freeDecodeMs(prompt: String, opts: GenerationOptions) async throws -> (ms: Double, tokens: Int) {
        let session = LanguageModelSession()
        let clock = ContinuousClock()
        var firstAt: ContinuousClock.Instant?
        var last = ""
        for try await partial in session.streamResponse(to: prompt, options: opts) {
            if firstAt == nil { firstAt = clock.now }
            last = partial.content
        }
        let end = clock.now
        let wrapper = await tokenCount(of: "") ?? 0
        let tokens = (await tokenCount(of: last)).map { max(0, $0 - wrapper) }
            ?? last.split(whereSeparator: { $0 == " " || $0 == "\n" }).count
        return ((end - (firstAt ?? end)).milliseconds, tokens)
    }

    /// Guided streamed generation of a `.minimumCount`-forced `[String]`; returns
    /// decode wall-time (first→last snapshot) and whether the cap truncated it.
    ///
    /// A throw *after* tokens have streamed is the expected cap truncation (the
    /// `>>cap minimumCount` can't be satisfied within the budget) → a valid measured
    /// rep. A throw *before* the first token is a real failure (availability, context,
    /// etc.), not a cap event — it is re-thrown so the caller drops the rep and logs
    /// the error, rather than averaging a bogus 0 ms sample or a false `cap bound ✓`.
    nonisolated private static func guidedDecodeMs(prompt: String, opts: GenerationOptions) async throws -> (ms: Double, capped: Bool) {
        let session = LanguageModelSession()
        let clock = ContinuousClock()
        var firstAt: ContinuousClock.Instant?
        var end: ContinuousClock.Instant?
        var capped = false
        do {
            for try await _ in session.streamResponse(to: prompt, generating: TokenBurst.self, options: opts) {
                if firstAt == nil { firstAt = clock.now }
                end = clock.now
            }
        } catch {
            if firstAt == nil { throw error }   // failed before any token → not a cap event
            capped = true
            end = clock.now
        }
        let e = end ?? clock.now
        return ((e - (firstAt ?? e)).milliseconds, capped)
    }
}

/// Guided-generation target for `enforcementProbe`: an array forced far longer than
/// the token cap, so constrained decoding keeps emitting elements until the cap
/// truncates it — the vehicle for measuring the per-token constrained-decoding tax.
@Generable
private struct TokenBurst {
    @Guide(.minimumCount(500))
    var items: [String]
}
