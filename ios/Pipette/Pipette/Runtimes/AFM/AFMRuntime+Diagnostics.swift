import Foundation
import FoundationModels

/// Dev-only probes that established the tokenizer wrapper, token counts and cap
/// semantics `AFMRuntime` relies on; kept to re-verify them when the OS updates.
/// Run headless with `metrics=tokprobe|genprobe|capprobe|enforceprobe`.
extension AFMRuntime {
    /// `tokenCount("") > 0` means the template wrapper is counted.
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

    /// Whether the cap equals the re-tokenized output length (detokenize then retokenize
    /// is not always identity).
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
                let opts = AFMRuntime.greedyOptions(maxTokens: cap)
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

    /// Whether short-answer prompts stop at EOS under a large cap; if so,
    /// `maximumResponseTokens` is only a ceiling.
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
                let opts = AFMRuntime.greedyOptions(maxTokens: cap)
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

    /// Decode-time cost of guided (constrained) generation against free generation over
    /// the same `cap`-token budget. Measured on iPhone 17 Pro, iOS 26.x, thermal-gated
    /// at 36°C or below: guided is within noise of free (~0.96 to 0.97x). The shipping
    /// path still uses free generation. Write-up: docs/pipette-ios/afm-token-enforcement.md.
    nonisolated static func enforcementProbe(progress: @escaping (String) -> Void) async {
        progress("availability=\(availabilityText())")
        guard case .available = availability() else {
            progress("model not available"); return
        }
        let cap = 100
        let reps = BenchmarkMeasurement.reps
        let opts = AFMRuntime.greedyOptions(maxTokens: cap)
        let freePrompt = AFMRuntime.countingSeed
        let guidedPrompt = "List as many short common English words as you can, one per element."

        // Cool down before every rep so heat can't confound the comparison.
        let readiness = BenchmarkReadiness(cancelFlag: CancelFlag()) { progress("\($0)") }
        func cooled(_ label: String) -> Bool {
            switch readiness.waitUntilReady() {
            case .ready: return true
            case .cancelled: progress("\(label): cancelled"); return false
            case .timedOut(_, let o): progress("\(label): readiness timed out (\(o)), skipping"); return false
            }
        }
        func mean(_ a: [Double]) -> Double { a.isEmpty ? 0 : a.reduce(0, +) / Double(a.count) }

        // Untimed: spins up the out-of-process model.
        _ = try? await freeDecodeMs(prompt: freePrompt,
                                    opts: AFMRuntime.greedyOptions(maxTokens: 1))

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
        return (durationMs(end - (firstAt ?? end)), tokens)
    }

    /// A throw after tokens have streamed is the cap truncating the forced-length array,
    /// a valid rep. A throw before the first token is a real failure and is rethrown.
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
            if firstAt == nil { throw error }
            capped = true
            end = clock.now
        }
        let e = end ?? clock.now
        return (durationMs(e - (firstAt ?? e)), capped)
    }

    /// Forced far longer than the cap, so the cap truncates it mid-stream.
    @Generable
    fileprivate struct TokenBurst {
        @Guide(.minimumCount(500))
        var items: [String]
    }
}
