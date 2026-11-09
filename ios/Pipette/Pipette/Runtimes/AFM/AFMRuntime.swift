import Foundation
import FoundationModels

enum AFMAvailability: Equatable {
    case available
    case unavailable(reason: String)
}

/// Apple's on-device foundation model (FoundationModels) as a benchmark runtime, emitting
/// the same result fields as llama/MLX.
///
/// Same metric or an error, never an approximation. The framework has no token-id input,
/// no prefill/decode split and runs out-of-process, so prefill, max-memory and VL
/// benchmarks throw: TTFT fuses prefill with the first token and scheduling, and the
/// in-process footprint probe cannot see the system service.
///
/// `maximumResponseTokens` is only a ceiling, and guided generation would force a length
/// but tax every token. So decode runs free generation on a non-terminating counting
/// prompt; the cap is enforced exactly in token space, so a run that reaches it produced
/// exactly N tokens. The SDK (iOS 26.5) ships no generated-token count, so the rep
/// re-tokenizes the output to confirm the cap bound and errors if it fell short. That and
/// prompt sizing need `tokenCount` (iOS 26.4+), so the timing benchmarks error below 26.4.
nonisolated enum AFMRuntime {
    /// AFM has no HF repo, so its submission identity is this fixed slug; `foundation-text`
    /// leaves room for a vision variant.
    nonisolated static let submissionModelName = "apple/foundation-text"

    /// Begun mid-sequence so a greedy model keeps emitting `11, 12, …` with no natural EOS
    /// and generation runs to the cap. `enforcementProbe` validated free decode with exactly
    /// this text.
    static let countingSeed = "Continue this sequence, one number per line, and never stop counting: "
        + "1, 2, 3, 4, 5, 6, 7, 8, 9, 10,"

    /// Times are from the request's start.
    private struct TimedRun {
        let ttftMs: Double
        let lastTokenMs: Double
        let totalMs: Double
        let outTokens: Int
    }

    /// The only place `SystemLanguageModel.default.availability` is read; every AFM
    /// availability check goes through here.
    static func availability() -> AFMAvailability {
        switch SystemLanguageModel.default.availability {
        case .available: return .available
        case .unavailable(let reason): return .unavailable(reason: "\(reason)")
        @unknown default: return .unavailable(reason: "unknown")
        }
    }

    static var isAvailable: Bool {
        if case .available = availability() { return true }
        return false
    }

    static func availabilityText() -> String {
        switch availability() {
        case .available: return "available"
        case .unavailable(let reason): return "unavailable(\(reason))"
        }
    }

    static func run(_ request: RunRequest,
                    runContext: any RunContext,
                    readinessGate: ReadinessGate,
                    observer: MeasurementObserver,
                    isCancelled: @escaping () -> Bool = { false },
                    progress: @escaping (BenchmarkProgress) -> Void = { _ in }
    ) async throws -> RunResponse {
        try AFMModels.requireAppleFoundation(request.model.bound)
        let definition = request.benchmark
        switch availability() {
        case .available: break
        case .unavailable(let reason): throw RuntimeError.readiness("Apple foundation model unavailable: \(reason)")
        }

        // Before the capability check and the session: the system model serves this
        // process one request at a time, so a batched entry is refused by name rather than
        // measured serially and filed under the batched benchmark id.
        try BenchmarkMeasurement.requireOneInFlight(
            definition, engine: "the system model answers one request at a time")

        // The same check the UI and headless paths offer from, so the two sets can't drift.
        guard isBenchmarkSupported(definition.benchmarkType, on: .afm) else {
            log("\(definition.benchmarkType.rawValue) unsupported on AFM (no prefill / max-memory / VL / ASR)")
            throw RuntimeError.unsupported(definition.benchmarkId.value)
        }

        let resultData: BenchmarkResultData
        switch definition {
        case .prefillThroughput, .maxMemoryUsage, .vlThroughput, .vlMaxMemory,
             .vlSpecDecode, .asrThroughput, .asrMaxMemory:
            // Unreachable: the guard above rejects these.
            throw RuntimeError.unsupported(definition.benchmarkId.value)
        case let .eval(benchmark):
            let cell = try EvalCell(benchmark)
            // Before the checkpoint opens: a pin this engine cannot apply fails the cell.
            try Self.requireHonoured(cell.sampling)
            resultData = try await Self.eval(
                benchmark, cell: cell,
                checkpoint: try runContext.evalCompletions.open(request),
                isCancelled: isCancelled, progress: progress)
        case let .decodeThroughput(benchmark):
            resultData = try await decodeThroughput(benchmark, readinessGate: readinessGate,
                                                    observer: observer)
        case let .endToEndLatency(benchmark):
            resultData = try await endToEndLatency(benchmark, readinessGate: readinessGate,
                                                   observer: observer)
        }
        return RunResponse(resultData: resultData, runtimeOutput: RuntimeOutput())
    }

    // MARK: - decode_throughput

    private static func decodeThroughput(_ benchmark: DecodeThroughputBenchmark,
                                         readinessGate: ReadinessGate,
                                         observer: MeasurementObserver) async throws -> BenchmarkResultData {
        // Up front: the reading would refuse it only after the prompt sizing and the warm-up.
        guard benchmark.parameterDecodeTokens.value >= 2 else {
            throw RuntimeError.engine("parameter_decode_tokens must be at least 2: the first "
                + "token is the prefill's, so a decode rate starts at the second; got "
                + "\(benchmark.parameterDecodeTokens)")
        }
        let prefillTokens = Int(benchmark.parameterPrefillTokens.value)
        let (wrapper, prompt, tokens) = try await preparedGeneration(prefill: prefillTokens)
        let cap = Int(benchmark.parameterDecodeTokens.value)
        log("decode: prompt \(tokens) content tokens (target \(prefillTokens)); decode cap \(cap)")
        // One token rather than the cell's shape: nothing on this side compiles per shape,
        // and generation is the scarce resource on the system model. Then `inline`: each
        // attempt opens a fresh session.
        _ = try await timeStream(prompt: prompt, maxTokens: 1, wrapper: wrapper, label: "decode warmup")
        return try await BenchmarkMeasurement.decode(
            benchmark, reps: BenchmarkMeasurement.reps, readinessGate: readinessGate,
            observer: observer, warmup: .inline, ctx: ()
        ) { _ in
            let run = try await timeStream(prompt: prompt, maxTokens: cap, wrapper: wrapper,
                                           label: "decode")
            // Past the floor, the cap is taken to have bound the run, and the SDK enforces it
            // exactly, so the reading counts `cap`.
            try requireCapReached(run, cap: cap, label: "decode")
            return .perPrompt([try DecodeMeasurementPerPromptReading.fromStream(
                prefillTokens: try PositiveCount(tokens), decodeTokens: try PositiveCount(cap),
                ttftMs: run.ttftMs, lastTokenMs: run.lastTokenMs)])
        }
    }

    // MARK: - end_to_end_latency

    private static func endToEndLatency(_ benchmark: EndToEndLatencyBenchmark,
                                        readinessGate: ReadinessGate,
                                        observer: MeasurementObserver) async throws -> BenchmarkResultData {
        let prefillTokens = Int(benchmark.parameterPrefillTokens.value)
        let (wrapper, prompt, tokens) = try await preparedGeneration(prefill: prefillTokens)
        let cap = Int(benchmark.parameterDecodeTokens.value)
        log("e2e: prompt \(tokens) content tokens (target \(prefillTokens)); decode cap \(cap)")
        _ = try await timeStream(prompt: prompt, maxTokens: 1, wrapper: wrapper, label: "e2e warmup")
        return try await BenchmarkMeasurement.endToEnd(
            benchmark, reps: BenchmarkMeasurement.reps, readinessGate: readinessGate,
            observer: observer, warmup: .inline, ctx: ()
        ) { _ in
            let run = try await timeStream(prompt: prompt, maxTokens: cap, wrapper: wrapper,
                                           label: "e2e")
            try requireCapReached(run, cap: cap, label: "e2e")
            return .perPrompt([try EndToEndMeasurementPerPromptReading.fromRequest(
                prefillTokens: try PositiveCount(tokens), decodeTokens: try PositiveCount(cap),
                elapsedMs: run.totalMs)])
        }
    }

    // MARK: - Eval

    /// One sample at a time, since the system model answers one request at a time.
    static func eval(_ benchmark: EvalBenchmark, cell: EvalCell,
                     checkpoint: EvalCompletionSession,
                     isCancelled: @escaping () -> Bool,
                     progress: @escaping (BenchmarkProgress) -> Void) async throws -> BenchmarkResultData {
        let maxTokens = Int(cell.maxTokens)
        log("eval: \(benchmark.samples?.count ?? 0) samples, maxTokens=\(maxTokens), "
            + "temperature=\(cell.sampling.temperature.value)")
        let cap = max(1, maxTokens)
        // Template overhead, subtracted so counts are content tokens.
        let wrapper = await tokenCount(of: "") ?? 0
        return try await BenchmarkMeasurement.eval(
            benchmark, inFlight: .parallel(1), checkpoint: checkpoint, ctx: (),
            progress: progress
        ) { _, samples, record in
            for (at, sample) in samples.enumerated() {
                // A stop ends the cell rather than failing the sample.
                try cancellationGate(isCancelled)
                let reading: EvalMeasurementReading
                do {
                    let opts = evalOptions(cell.sampling, seed: cell.sampling.seedFor(sample.id),
                                           maxTokens: maxTokens)
                    let text = try await complete(sample.messages, opts: opts)
                    reading = await generation(text, cap: cap, wrapper: wrapper).reading
                } catch let stopped as CancellationError {
                    throw stopped
                } catch {
                    // A stop can fail the sample in flight, so a failure with one pending is
                    // the stop's.
                    try cancellationGate(isCancelled)
                    reading = .failed(reason: "\(error)")
                }
                try record(at, reading)
            }
        }
    }

    /// AFM exposes no stop signal, so the stop is inferred from the re-tokenized count
    /// against the cap. Heuristic: drift near the cap can misclassify a boundary sample
    /// (PIP-266).
    private static func generation(_ text: String, cap: Int, wrapper: Int) async -> EvalGeneration {
        let content = (await tokenCount(of: text)).map { max(0, $0 - wrapper) }
        let stopReason: BenchmarkEvalCompletionStopReason = content.map { $0 >= cap ? .truncated : .eos } ?? .unknown
        let stopDetail = content == nil ? "AFM: no on-device token counting (iOS < 26.4)" : nil
        return EvalGeneration(text: text, stopReason: stopReason, stopDetail: stopDetail,
                              completionTokens: content.map(UInt64.init))
    }

    /// Refuse a pin the framework cannot apply rather than draw without it: it takes one
    /// cut, top-k or a probability threshold, and has no min-p. Greedy ignores every cut.
    static func requireHonoured(_ sampling: Sampling) throws {
        guard drawsSample(atTemperature: sampling.temperature.value) else { return }
        if let minP = sampling.minP?.value, minP > 0 {
            throw RuntimeError.engine("Apple Foundation has no min-p cut; parameter_min_p "
                + "is \(minP)")
        }
        if let topK = sampling.topK?.value, topK > 0, let topP = sampling.topP?.value, topP < 1 {
            throw RuntimeError.engine("Apple Foundation takes one cut, top-k or top-p; the "
                + "entry pins top-k \(topK) and top-p \(topP)")
        }
    }

    /// An entry that pins nothing keeps the framework's own sampler; one that pins a cut
    /// off, or only a seed, draws from the whole distribution.
    static func evalOptions(_ sampling: Sampling, seed: UInt32?, maxTokens: Int)
        -> GenerationOptions {
        let cap = max(1, maxTokens)
        guard drawsSample(atTemperature: sampling.temperature.value) else {
            return greedyOptions(maxTokens: cap)
        }
        let seed = seed.map(UInt64.init)
        let mode: GenerationOptions.SamplingMode?
        if let topK = sampling.topK?.value, topK > 0 {
            mode = .random(top: Int(topK), seed: seed)
        } else if let topP = sampling.topP?.value, topP < 1 {
            mode = .random(probabilityThreshold: topP, seed: seed)
        } else if seed != nil || sampling.topK != nil || sampling.topP != nil
                    || sampling.minP != nil {
            mode = .random(probabilityThreshold: 1, seed: seed)
        } else {
            mode = nil
        }
        #if compiler(>=6.4)
        return GenerationOptions(samplingMode: mode, temperature: sampling.temperature.value,
                                 maximumResponseTokens: cap)
        #else
        return GenerationOptions(sampling: mode, temperature: sampling.temperature.value,
                                 maximumResponseTokens: cap)
        #endif
    }

    /// Xcode Cloud's newer SDK deprecates `sampling:` for `samplingMode:` (an error under
    /// warnings-as-errors), while CI's Xcode 26.4 (Swift 6.3) only has `sampling:`.
    /// Assumes the newer SDK ships Swift 6.4 or later.
    static func greedyOptions(maxTokens: Int) -> GenerationOptions {
        #if compiler(>=6.4)
        return GenerationOptions(samplingMode: .greedy, maximumResponseTokens: maxTokens)
        #else
        return GenerationOptions(sampling: .greedy, maximumResponseTokens: maxTokens)
        #endif
    }

    private static func complete(_ messages: [[String: String]],
                                 opts: GenerationOptions) async throws -> String {
        let (instructions, prompt) = splitMessages(messages)
        let session = instructions.isEmpty ? LanguageModelSession()
                                           : LanguageModelSession(instructions: instructions)
        return try await session.respond(to: prompt, options: opts).content
    }

    /// Several turns fold into one role-labeled prompt because the SDK takes prior turns
    /// only through a `Transcript`.
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

    private static func preparedGeneration(prefill: Int) async throws -> (wrapper: Int, prompt: String, tokens: Int) {
        guard let wrapper = await tokenCount(of: "") else {
            throw RuntimeError.engine("AFM timing benchmarks require on-device token counting "
                + "(SystemLanguageModel.tokenCount, iOS 26.4+) to size the prompt and gauge the output length")
        }
        let (prompt, tokens) = try await sizedElicitingPrompt(prefillTokens: prefill,
                                                              wrapper: wrapper)
        return (wrapper, prompt, tokens)
    }

    /// The re-tokenized length that counts as reaching the cap. The ~5% slack absorbs
    /// re-tokenization drift; a genuine early stop lands far below it.
    static func capFloor(_ cap: Int) -> Int { cap - max(2, cap / 20) }

    private static func requireCapReached(_ r: TimedRun, cap: Int, label: String) throws {
        guard r.outTokens >= capFloor(cap) else {
            throw RuntimeError.engine("AFM \(label): output re-tokenizes to ~\(r.outTokens) tokens, well short of the "
                + "\(cap)-token cap (model stopped early). Workload shorter than llama/MLX; refusing a mismatched measurement")
        }
    }

    /// A fresh session per call so no transcript carries over between reps. Timed on
    /// `ContinuousClock` because wall-clock `Date` can step mid-stream.
    ///
    /// `outTokens` is the output re-tokenized minus the wrapper, since the SDK (iOS 26.5)
    /// ships no generated-token count (`Usage` / `session.usage` are documented but
    /// unshipped). Detokenize then retokenize isn't identity, but for the counting output
    /// it is near-exact, and it only tells an early stop from a cap-bound run.
    private static func timeStream(prompt: String, maxTokens: Int, wrapper: Int, label: String)
        async throws -> TimedRun {
        let session = LanguageModelSession()
        let opts = greedyOptions(maxTokens: max(1, maxTokens))
        let clock = ContinuousClock()
        let t0 = clock.now
        var firstAt: ContinuousClock.Instant?
        var lastAt: ContinuousClock.Instant?
        var lastContent = ""
        var prevWords = 0
        let stream = session.streamResponse(to: prompt, options: opts)
        for try await partial in stream {
            let at = clock.now
            if firstAt == nil {
                firstAt = at
                log(String(format: "%@: first token @ %.0fms", label, durationMs(at - t0)))
            }
            lastAt = at
            lastContent = partial.content
            // Word count approximates tokens; the framework batches a few tokens per snapshot.
            let words = lastContent.split(whereSeparator: { $0 == " " || $0 == "\n" }).count
            let dt = durationMs(clock.now - (firstAt ?? t0)) / 1000
            let tps = dt > 0 ? Double(words) / dt : 0
            log(String(format: "%@: tok≈%d (+%d) t=%.1fs tps≈%.1f", label, words, words - prevWords, dt, tps))
            prevWords = words
        }
        let end = clock.now
        let first = firstAt ?? end
        let outTokens = (await tokenCount(of: lastContent)).map { max(0, $0 - wrapper) } ?? prevWords
        return TimedRun(ttftMs: durationMs(first - t0), lastTokenMs: durationMs((lastAt ?? end) - t0),
                        totalMs: durationMs(end - t0), outTokens: outTokens)
    }

    /// The count always includes the fixed template/BOS wrapper (empirically 8 tokens, even
    /// for `""`), so a text's own tokens are this minus `tokenCount(of: "")`. Nil below
    /// iOS 26.4.
    static func tokenCount(of text: String) async -> Int? {
        guard #available(iOS 26.4, *) else { return nil }
        return try? await SystemLanguageModel.default.tokenCount(for: text)
    }

    /// The shared corpus prose the llama and MLX cells prefill, followed by `countingSeed`,
    /// sized to `prefillTokens` content tokens under this model's counter. The seed counts
    /// toward the size, as the harness sizes the whole prompt.
    private static func sizedElicitingPrompt(prefillTokens: Int,
                                             wrapper: Int) async throws -> (text: String, tokens: Int) {
        let seed = Self.countingSeed
        func compose(_ body: String) -> String {
            body.isEmpty ? seed : body + " " + seed
        }
        func contentTokens(_ text: String) async throws -> Int {
            guard let total = await tokenCount(of: text) else {
                throw RuntimeError.engine("AFM stopped answering token counts mid-sizing")
            }
            return max(0, total - wrapper)
        }
        let body = try await PromptSeed.buildPromptText(targetTokens: prefillTokens) {
            try await contentTokens(compose($0))
        }
        let text = compose(body)
        return (text, try await contentTokens(text))
    }

    /// Not `print`: stdout is a parsed contract (`pipette-ctl` scans it for `BENCH_DONE`).
    static func log(_ s: String) { AppLog.afm.info(s) }
}
