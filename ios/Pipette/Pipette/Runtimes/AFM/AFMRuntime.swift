import Foundation
import FoundationModels

/// Normalized on-device foundation-model availability — the single shape every AFM
/// read (UI gating, logging, the `run` guard, and the dev diagnostics) derives from,
/// so the `SystemLanguageModel.default.availability` switch lives in exactly one place.
enum AFMAvailability: Equatable {
    case available
    case unavailable(reason: String)
}

/// PROTOTYPE: benchmark Apple's on-device foundation model (the FoundationModels
/// framework) as a third runtime alongside llama.cpp and MLX. It plugs into the
/// shared harness exactly like the others: `RunCell.dispatch` routes the
/// `(.appleFoundation, .afm)` pair here, it consumes the typed `BenchmarkDefinition`,
/// averages over `BenchmarkMeasurement.measure` (1 warm-up + gated reps + population
/// stats), and returns a typed `BenchmarkResult` whose **fields are identical** to
/// the llama/MLX paths (`decode_time_ms`, `total_time_ms`, …). Console breadcrumbs
/// go through `print` (`[AFM] …`), the same observability pattern MLX uses.
///
/// **Parity policy — same metric or raise an error (no approximations).** The
/// framework is high-level: no token-id input, no prefill/decode/KV access, no
/// in-process memory, runs out-of-process (ANE). So it can only reproduce the
/// harness benchmarks where it measures the *same* quantity the same way; otherwise
/// it throws rather than emit a stand-in number:
///   - `decode_throughput`  → ✅ `decode_time_ms` = streamed wall-time (first→last
///                              token). Guarded: the rep **errors** unless the model
///                              actually reaches the requested token cap (so the
///                              workload matches llama/MLX, not a short early-stop).
///   - `end_to_end_latency` → ✅ `total_time_ms` = full request wall-time (TTFT +
///                              decode), same cap guard.
///   - `prefill_throughput` → ❌ ERROR: the streaming API exposes only TTFT, which
///                              fuses prefill + first-token + scheduling; the prefill
///                              phase can't be isolated, so there's no honest
///                              `prefill_time_ms`.
///   - `max_memory_usage`   → ❌ ERROR: AFM runs out-of-process (system service /
///                              ANE); the in-process footprint probe can't see it.
///   - `vl_throughput`      → ❌ ERROR: text-only model.
///   - `eval`               → ✅ one chat completion per dataset sample; the server
///                              scores the text. No token forcing (maxTokens is a
///                              ceiling) and no gate (not a timing measurement). See
///                              `evalCompletions` for the chat-message→session mapping.
///
/// **Forcing the token count without a throughput penalty.** `maximumResponseTokens`
/// is only a ceiling and there's no min-tokens / ignore-EOS knob; guided generation
/// (`@Generable`/`@Guide(.count)`) *can* force a length via constrained decoding but
/// taxes every token (logit mask + JSON structure), which would corrupt a throughput
/// measurement. The lever that IS free: `maximumResponseTokens` is enforced exactly
/// in token space, so a generation that *reaches* the cap produced exactly N tokens.
/// AFM therefore uses **free** generation driven by a non-terminating counting prompt
/// (see `sizedElicitingPrompt`) so a greedy EOS never fires before the cap — the cap
/// binds and exactly `decode` tokens are produced. The shipping SDK (iOS 26.5) has no
/// generated-token count API (`Usage`/`session.usage` are documented but unshipped),
/// so the rep confirms the cap bound by re-tokenizing the output (a near-exact proxy
/// for the clean counting output) and errors if it fell short — a number is only
/// reported over a workload that matches llama/MLX. Sizing + the proxy count need
/// `tokenCount` (iOS 26.4+), so the timing benchmarks require 26.4 and error below;
/// the FoundationModels types themselves import unconditionally (deployment target 26.0).
nonisolated enum AFMRuntime {
    static let measurementRuns = 5

    /// Submission `model_name` for AFM results — the counterpart to a downloaded
    /// model's repo-derived name. AFM has no HF coordinate, so its identity is this
    /// fixed slug (the honest "no repo, but a stable name"). The `foundation-text`
    /// leaf marks the text model, leaving room for a future vision variant.
    nonisolated static let submissionModelName = "apple/foundation-text"

    /// Non-terminating counting seed, begun mid-sequence so a greedy instruction model
    /// just keeps emitting `11, 12, …` with no natural EOS — the strongest way to make
    /// generation run to the `maximumResponseTokens` cap. Shared by the production
    /// decode prompt (`sizedElicitingPrompt`) and the `enforcementProbe` so both drive
    /// the cap identically (the probe validated free decode with exactly this text).
    static let countingSeed = "Continue this sequence, one number per line, and never stop counting: "
        + "1, 2, 3, 4, 5, 6, 7, 8, 9, 10,"

    /// A single streamed generation's timings + the generated content-token count.
    private struct TimedRun {
        let ttftMs: Double
        let decodeMs: Double
        let outTokens: Int
    }

    /// The one place the system availability is read + normalized to `AFMAvailability`.
    /// Every AFM availability signal (UI gating, logging, the `run` guard, diagnostics)
    /// routes through here, so the `SystemLanguageModel.default.availability` switch —
    /// including the `@unknown default` mapping — is never re-derived elsewhere.
    /// `SystemLanguageModel.default.availability` is a synchronous, non-isolated read,
    /// so no actor hop is needed.
    static func availability() -> AFMAvailability {
        switch SystemLanguageModel.default.availability {
        case .available: return .available
        case .unavailable(let reason): return .unavailable(reason: "\(reason)")
        @unknown default: return .unavailable(reason: "unknown")
        }
    }

    /// Whether the on-device foundation model is usable right now — the signal the UI
    /// uses to decide if AFM should appear as a selectable built-in model. True iff the
    /// system reports `.available` (device supports Apple Intelligence, model
    /// downloaded, not disabled).
    static var isAvailable: Bool {
        if case .available = availability() { return true }
        return false
    }

    /// One-line model-availability description for logging / gating.
    static func availabilityText() -> String {
        switch availability() {
        case .available: return "available"
        case .unavailable(let reason): return "unavailable(\(reason))"
        }
    }

    /// Run one typed benchmark def against the Apple foundation model and return the
    /// typed `BenchmarkResult` (same fields the llama/MLX runtimes emit). `readiness`
    /// runs before each measured rep; a non-`.ready` outcome cancels or fails the
    /// cell. Benchmarks AFM can't reproduce with the same metric raise an error (see
    /// the type doc) rather than fabricating a number.
    static func run(_ request: RunRequest,
                    evalCompletions: EvalCompletionsStore,
                    readiness: @escaping () -> ReadinessOutcome,
                    observer: RepObserver,
                    isCancelled: @escaping () -> Bool = { false },
                    progress: @escaping (BenchmarkProgress) -> Void = { _ in }
    ) async throws -> BenchmarkResult {
        try AFMModels.requireAppleFoundation(request)
        let definition = request.benchmark
        switch availability() {
        case .available: break
        case .unavailable(let reason): throw RuntimeError.engine("Apple foundation model unavailable: \(reason)")
        }

        // Capability check first, so the set AFM accepts here can't drift from the set
        // the UI / headless paths offer via `isBenchmarkSupported`. The per-case doc
        // comments above (why prefill / max-memory / VL are out) explain the policy this
        // guard enforces; only the supported cases fall through to the implementations.
        guard isBenchmarkSupported(definition.type, on: .afm) else {
            log("\(definition.type.rawValue) unsupported on AFM (no prefill / max-memory / VL)")
            throw RuntimeError.unsupported(definition.benchmarkId)
        }

        switch definition {
        case .prefillThroughput, .maxMemoryUsage, .vlThroughput:
            // Unreachable: the guard above rejects these before the switch. Kept exhaustive
            // over `BenchmarkDefinition` with the capability guard as the authority.
            throw RuntimeError.unsupported(definition.benchmarkId)
        case .eval(_, _, _, let maxTokens, _, let samples):
            return try await Self.evalCompletions(
                samples: samples ?? [], maxTokens: Int(maxTokens),
                checkpoint: try evalCompletions.open(request: request),
                isCancelled: isCancelled, progress: progress)

        case .decodeThroughput(_, let prefill, let decode):
            let (wrapper, prompt, tokens) = try await preparedGeneration(prefill: Int(prefill))
            let cap = Int(decode)
            log("decode: prompt \(tokens) content tokens (target \(prefill)); decode cap \(cap)")
            let (mean, sd) = try await BenchmarkMeasurement.measure(
                label: definition.benchmarkType, runs: measurementRuns,
                warmup: { _ = try? await timeStream(prompt: prompt, maxTokens: 1, wrapper: wrapper, label: "decode warmup") },
                gate: { try readinessGate(readiness) },
                observer: observer,
                onProgress: progress,
                body: {
                    let r = try await timeStream(prompt: prompt, maxTokens: cap, wrapper: wrapper, label: "decode")
                    try requireCapReached(r, cap: cap, label: "decode")
                    return r.decodeMs
                })
            return .decodeThroughput(timeMs: mean, stddev: sd)

        case .endToEndLatency(_, let prefill, let decode):
            let (wrapper, prompt, tokens) = try await preparedGeneration(prefill: Int(prefill))
            let cap = Int(decode)
            log("e2e: prompt \(tokens) content tokens (target \(prefill)); decode cap \(cap)")
            let (mean, sd) = try await BenchmarkMeasurement.measure(
                label: definition.benchmarkType, runs: measurementRuns,
                warmup: { _ = try? await timeStream(prompt: prompt, maxTokens: 1, wrapper: wrapper, label: "e2e warmup") },
                gate: { try readinessGate(readiness) },
                observer: observer,
                onProgress: progress,
                body: {
                    let r = try await timeStream(prompt: prompt, maxTokens: cap, wrapper: wrapper, label: "e2e")
                    try requireCapReached(r, cap: cap, label: "e2e")
                    return r.ttftMs + r.decodeMs
                })
            return .endToEndLatency(timeMs: mean, stddev: sd)
        }
    }

    // MARK: - Eval

    /// `eval`: complete every dataset sample and return the `{id, completion}` shape
    /// the unified payload builder ingests (parity with the llama/MLX eval paths; the
    /// server scores the text). Unlike the throughput benches, eval needs **no token
    /// forcing** — `maxTokens` is a ceiling and we want the model's natural answer —
    /// and no thermal gate (it isn't a timing measurement), matching MLX's eval path.
    /// A sample that throws becomes a `.failed` completion rather than aborting the
    /// run; `.sample` progress is reported per row. Availability is already checked in
    /// `run` before dispatch.
    ///
    /// **Chat-message → session mapping.** Each `EvalSample.messages` entry is a
    /// `{role, content}` pair. `system` turns are joined into the session
    /// `instructions`; the remaining turns become the prompt. A lone user turn is used
    /// verbatim (the common case for this suite — GPQA / MATH / IFBench are
    /// single-turn); multiple turns are folded into one role-labeled prompt, because
    /// the shipping SDK only accepts prior turns via a `Transcript` and few-shot
    /// assistant turns are not used here. Greedy sampling for determinism.
    static func evalCompletions(samples: [EvalSample], maxTokens: Int,
                                checkpoint: EvalCompletionSession?,
                                isCancelled: @escaping () -> Bool,
                                progress: @escaping (BenchmarkProgress) -> Void) async throws -> BenchmarkResult {
        let opts = GenerationOptions(sampling: .greedy, maximumResponseTokens: max(1, maxTokens))
        log("eval: \(samples.count) samples, maxTokens=\(maxTokens)")
        var completions: [BenchmarkEvalCompletion] = []
        completions.reserveCapacity(samples.count)
        let cap = max(1, maxTokens)
        // Template overhead to subtract so the count reflects generated content
        // tokens. `nil` on iOS < 26.4 (no on-device token counting).
        let wrapper = await tokenCount(of: "") ?? 0
        for (index, sample) in samples.enumerated() {
            defer { progress(.sample(completed: index + 1, total: samples.count)) }
            // Reuse rather than re-run what an earlier attempt completed — the contract
            // `evalSamples` applies on the llama and MLX paths, restated here because this
            // loop is hand-rolled: `complete` is async and `evalSamples`' closure is not.
            if let prior = checkpoint?.completion(for: sample.id) {
                completions.append(prior)
                continue
            }
            // Before the do/catch, so a cancel aborts the run instead of becoming a
            // failed sample. (Eval has no thermal gate — see `cancellationGate`.)
            try cancellationGate(isCancelled)
            let completion: BenchmarkEvalCompletion
            do {
                let text = try await complete(sample, opts: opts)
                // AFM exposes no stop signal, so classify from its own token
                // count vs the cap: `>= cap` ⇒ hit the limit (`truncated`), fewer
                // ⇒ natural stop (`eos`). Indeterminate without token counting.
                // Heuristic, not authoritative: re-tokenization drift near the cap
                // can misclassify a boundary sample (no engine signal to confirm,
                // and no ε tolerance here — cf. the PIP-266 backfill).
                let content = (await tokenCount(of: text)).map { max(0, $0 - wrapper) }
                let stopReason: BenchmarkEvalCompletionStopReason = content.map { $0 >= cap ? .truncated : .eos } ?? .unknown
                // Record why when we couldn't classify (no on-device token count).
                let stopDetail = content == nil
                    ? "AFM: no on-device token counting (iOS < 26.4)" : nil
                completion = .completed(id: sample.id, text: text, stopReason: stopReason,
                                       stopDetail: stopDetail, completionTokens: content)
            } catch {
                completion = .failed(id: sample.id, reason: "\(error)")
            }
            completions.append(completion)
            // Durable before the next sample starts, so a jetsam kill costs one sample
            // rather than the run. Best-effort for the reason `evalSamples` gives.
            try? checkpoint?.append(completion)
        }
        // See `LlamaBenchmark.evalOn` for why the session's own view is not reported.
        checkpoint?.finalize()
        return .eval(completions: completions)
    }

    /// Map one sample's chat messages to an AFM session + prompt and greedy-decode the
    /// completion (see `evalCompletions` for the mapping rationale).
    private static func complete(_ sample: EvalSample, opts: GenerationOptions) async throws -> String {
        let (instructions, prompt) = splitMessages(sample.messages)
        let session = instructions.isEmpty ? LanguageModelSession()
                                           : LanguageModelSession(instructions: instructions)
        return try await session.respond(to: prompt, options: opts).content
    }

    /// Split a sample's `{role, content}` messages into the AFM session `instructions`
    /// (all `system` turns, joined) and the `prompt` (a lone user turn verbatim; else
    /// the remaining turns folded into one role-labeled block). Pure — unit-tested in
    /// `AFMRuntimeTests` — so the mapping rationale in `evalCompletions` stays honest.
    static func splitMessages(_ messages: [[String: String]]) -> (instructions: String, prompt: String) {
        func role(_ m: [String: String]) -> String { m["role"] ?? "user" }
        func content(_ m: [String: String]) -> String { m["content"] ?? "" }
        let instructions = messages.filter { role($0) == "system" }.map(content).joined(separator: "\n\n")
        let turns = messages.filter { role($0) != "system" }
        let prompt = turns.count == 1
            ? content(turns[0])
            : turns.map { "\(role($0)): \(content($0))" }.joined(separator: "\n")
        return (instructions, prompt)
    }

    // MARK: - Generation prep + verification

    /// Up-front setup shared by the timing benchmarks: require exact token counting
    /// (iOS 26.4+ — without it we can neither size the prompt to match the harness
    /// nor verify the decode workload), measure the fixed template wrapper, and build
    /// a prompt sized to `prefill` content tokens that *elicits a long continuation*
    /// so the decode cap (not an early EOS) is what bounds generation.
    private static func preparedGeneration(prefill: Int) async throws -> (wrapper: Int, prompt: String, tokens: Int) {
        guard let wrapper = await tokenCount(of: "") else {
            throw RuntimeError.engine("AFM timing benchmarks require on-device token counting "
                + "(SystemLanguageModel.tokenCount, iOS 26.4+) to size the prompt and gauge the output length")
        }
        let (prompt, tokens) = await sizedElicitingPrompt(prefillTokens: prefill, wrapper: wrapper)
        return (wrapper, prompt, tokens)
    }

    /// Fail the rep unless the model generated ~`cap` tokens. `maximumResponseTokens`
    /// is enforced exactly in token space, so a run that reaches the cap generated
    /// exactly `cap` tokens (that's the "forcing"); a greedy EOS before then yields a
    /// short workload we refuse rather than time against llama/MLX. The shipping SDK
    /// gives no generated-token count, so `outTokens` is the re-tokenized output — a
    /// near-exact proxy for the clean counting output — hence a small tolerance rather
    /// than strict equality; a genuine early-stop lands far below the cap.
    /// Minimum re-tokenized output length that counts as "reached the cap": the cap
    /// minus ~5% (floored at 2) to absorb retokenization drift on the clean counting
    /// output. Pure — unit-tested in `AFMRuntimeTests`.
    static func capFloor(_ cap: Int) -> Int { cap - max(2, cap / 20) }

    private static func requireCapReached(_ r: TimedRun, cap: Int, label: String) throws {
        guard r.outTokens >= capFloor(cap) else {
            throw RuntimeError.engine("AFM \(label): output re-tokenizes to ~\(r.outTokens) tokens, well short of the "
                + "\(cap)-token cap (model stopped early). Workload shorter than llama/MLX; refusing a mismatched measurement")
        }
    }

    /// Fresh session per call (isolation — avoids transcript/context carryover across
    /// reps). **Free** generation (no guided/constrained decoding — that adds a
    /// per-token logit-mask + JSON-structure tax that would corrupt a throughput
    /// measurement); the prompt is what drives length. Streams the response for the
    /// TTFT/decode split, timed on a **monotonic** `ContinuousClock` (wall-clock
    /// `Date` can step under NTP / manual time changes mid-stream), matching
    /// `BenchmarkMeasurement.timed`'s discipline.
    ///
    /// The shipping SDK (iOS 26.5) exposes no generated-token count — `Usage` /
    /// `session.usage` are documented but unshipped — so the returned `outTokens` is
    /// the output text re-tokenized via `tokenCount` (minus the fixed template
    /// `wrapper`). That's a proxy: detokenize→retokenize isn't identity, but for the
    /// digit/newline counting output it's near-exact, and it's only used to tell a
    /// genuine early-stop (far short of the cap) from a cap-bound run.
    private static func timeStream(prompt: String, maxTokens: Int, wrapper: Int, label: String)
        async throws -> TimedRun {
        let session = LanguageModelSession()
        let opts = GenerationOptions(sampling: .greedy, maximumResponseTokens: max(1, maxTokens))
        let clock = ContinuousClock()
        let t0 = clock.now
        var firstAt: ContinuousClock.Instant?
        var lastContent = ""
        var prevWords = 0
        let stream = session.streamResponse(to: prompt, options: opts)
        for try await partial in stream {
            if firstAt == nil {
                let f = clock.now; firstAt = f
                log(String(format: "%@: first token @ %.0fms", label, (f - t0).milliseconds))
            }
            lastContent = partial.content
            // Per-snapshot delta ≈ tokens just produced (the framework batches a few
            // tokens per snapshot — finest progress the streaming API exposes).
            let words = lastContent.split(whereSeparator: { $0 == " " || $0 == "\n" }).count
            let dt = (clock.now - (firstAt ?? t0)).milliseconds / 1000
            let tps = dt > 0 ? Double(words) / dt : 0
            log(String(format: "%@: tok≈%d (+%d) t=%.1fs tps≈%.1f", label, words, words - prevWords, dt, tps))
            prevWords = words
        }
        let end = clock.now
        let first = firstAt ?? end
        // Re-tokenize the output (minus the template wrapper) as the generated-token
        // proxy; word count is the last-ditch fallback if tokenCount is unavailable.
        let outTokens = (await tokenCount(of: lastContent)).map { max(0, $0 - wrapper) } ?? prevWords
        return TimedRun(ttftMs: (first - t0).milliseconds, decodeMs: (end - first).milliseconds, outTokens: outTokens)
    }

    /// Exact number of tokens the on-device model consumes for `text`, via
    /// `SystemLanguageModel.tokenCount(for:)`. The count always includes the model's
    /// fixed template/BOS wrapper (empirically 8 tokens, present even for `""`) — there
    /// is no option to exclude it. To get a text's own tokens, subtract
    /// `tokenCount(of: "")`. `tokenCount(for:)` is iOS 26.4+, so returns nil below that
    /// (deployment target is 26.0) — callers treat nil as "can't measure".
    static func tokenCount(of text: String) async -> Int? {
        guard #available(iOS 26.4, *) else { return nil }
        return try? await SystemLanguageModel.default.tokenCount(for: text)
    }

    /// Build a prompt of ~`prefillTokens` content tokens (model tokens minus the
    /// `wrapper`) that ends with a **counting seed** — the strongest reliably
    /// non-terminating task for a greedy instruction model: begun mid-sequence
    /// (`… 9, 10,`), greedy continuation just keeps emitting `11, 12, …` with no
    /// natural EOS, so the `maximumResponseTokens` cap (not an early stop) bounds
    /// generation and the `requireCapReached` guard passes on a healthy run. This is
    /// deliberately *free* generation, not a guided/`@Guide(.count)` schema:
    /// constrained decoding would force the length but tax every token with a
    /// logit-mask + JSON-structure overhead, corrupting the throughput we're
    /// measuring. Filler is grown word-by-word and the *combined* text (filler +
    /// seed) is checked against the target, so the seed is included in the size
    /// (parity with the harness, which sizes the whole prompt). Returns the text and
    /// its content-token count.
    private static func sizedElicitingPrompt(prefillTokens: Int, wrapper: Int) async -> (text: String, tokens: Int) {
        let instruction = Self.countingSeed
        let pool = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
                    "and", "then", "runs", "across", "green", "field", "under", "sky"]
        func compose(_ filler: [String]) -> String {
            filler.isEmpty ? instruction : filler.joined(separator: " ") + " " + instruction
        }
        var filler: [String] = []
        var i = 0
        while filler.count < max(1, prefillTokens) * 2 + 64 {
            let text = compose(filler)
            if let total = await tokenCount(of: text) {
                let c = max(0, total - wrapper)
                if c >= prefillTokens { return (text, c) }
            }
            filler.append(pool[i % pool.count]); i += 1
        }
        let text = compose(filler)
        let content = (await tokenCount(of: text)).map { max(0, $0 - wrapper) } ?? filler.count
        return (text, content)
    }

    /// Console breadcrumb — the only window into a headless on-device run, kept as a
    /// permanent diagnostic (the MLX runtime logs `[MLXMEM] …` the same way).
    static func log(_ s: String) { print("[AFM] \(s)") }
}
