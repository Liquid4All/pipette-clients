import Foundation

// The crate's `pipette-llamacpp/src/execute/asr_throughput.rs`, which posts a WAV to
// `llama-server`; this drives mtmd directly, as `LlamaVlThroughput.swift` does for a frame.
// `total_time_ms` starts before the clip's decode and mel inside `mtmd_tokenize`, which the
// server runs in the request's handler. The CLI's post-rep probes are tokenizations here: a
// request's `prompt_n` is its prompt's `mtmd_tokenize` count.

/// An ASR cell's workload, its refusals checked before anything loads.
nonisolated struct AsrCell: Equatable, Sendable {
    let recipe: AsrRecipe
    let durationMs: UInt32
    let sampleRateHz: UInt32
    let channels: UInt16
    let textTokens: UInt32
    let decodeTokens: UInt32
    /// `asr_beyond_model_audio_limit`.
    let beyondLimit: Bool

    init(recipe: AsrRecipe, durationMs: UInt32, sampleRateHz: UInt32, channels: UInt16,
         textTokens: UInt32, decodeTokens: UInt32) throws {
        try recipe.checkDuration(durationMs, sampleRateHz: sampleRateHz)
        self.recipe = recipe
        self.durationMs = durationMs
        self.sampleRateHz = sampleRateHz
        self.channels = channels
        self.textTokens = textTokens
        self.decodeTokens = decodeTokens
        beyondLimit = recipe.beyondDocumentedLimit(durationMs)
    }

    init(_ benchmark: AsrThroughputBenchmark, recipe: AsrRecipe) throws {
        try self.init(recipe: recipe, durationMs: benchmark.parameterAudioDurationMs.value.magnitude,
                      sampleRateHz: benchmark.parameterAudioSampleRateHz.value.magnitude,
                      channels: benchmark.parameterAudioChannels.value,
                      textTokens: benchmark.parameterTextTokens.value.magnitude,
                      decodeTokens: benchmark.parameterDecodeTokens.value.magnitude)
    }

    init(_ benchmark: AsrMaxMemoryBenchmark, recipe: AsrRecipe) throws {
        try self.init(recipe: recipe, durationMs: benchmark.parameterAudioDurationMs.value.magnitude,
                      sampleRateHz: benchmark.parameterAudioSampleRateHz.value.magnitude,
                      channels: benchmark.parameterAudioChannels.value,
                      textTokens: benchmark.parameterTextTokens.value.magnitude,
                      decodeTokens: benchmark.parameterDecodeTokens.value.magnitude)
    }
}

nonisolated extension LlamaExecute {
    static func asrThroughput(
        _ benchmark: AsrThroughputBenchmark, _ request: RunRequest, cell: AsrCell,
        model: ResolvedGgufVisionModel, flags: RuntimeFlags,
        readinessGate: ReadinessGate, observer: MeasurementObserver
    ) async throws -> (BenchmarkResultData, RuntimeFlags, AsrClientExtras) {
        let pinned = request.runtimeFlags?.ctxSize != nil
        let ((resultData, source), ran) = try await withAsrClip(cell) { clip in
            // Counted even behind a pinned context, as the CLI calibrates, so it can be confirmed.
            try await withCalibratedAudio(request, model, cell, flags: flags, clipPath: clip,
                                          calibrate: true, observer: observer) { aud, calibrated in
                try await asrThroughput(benchmark, cell, aud, clipPath: clip,
                                        calibrated: calibrated, pinned: pinned,
                                        readinessGate: readinessGate, observer: observer)
            }
        }
        return (resultData, ran,
                AsrClientExtras(contextSizing: AsrContextSizing(pinned: pinned, calibrated: true),
                                audioTokensSource: source, beyondModelAudioLimit: cell.beyondLimit))
    }

    /// Apart from the load so tests run it on a hand-built `AudioInference`.
    static func asrThroughput(
        _ benchmark: AsrThroughputBenchmark, _ cell: AsrCell, _ aud: AudioInference,
        clipPath: String, calibrated: Int?, pinned: Bool,
        readinessGate: ReadinessGate, observer: MeasurementObserver
    ) async throws -> (BenchmarkResultData, AsrAudioTokensSource) {
        let userText = try await asrUserText(aud, cell)
        let userContent = cell.recipe.userContent(userText, marker: aud.mediaMarker)
        let decodeTokens = Int(cell.decodeTokens)
        let clock = ContinuousClock()
        var source: AsrAudioTokensSource?
        let resultData = try await BenchmarkMeasurement.asrThroughput(
            benchmark, reps: BenchmarkMeasurement.reps, readinessGate: readinessGate,
            observer: observer, warmup: .once, ctx: aud,
            attempt: { aud in
                aud.resetContext()
                aud.resetSampler()
                // Timed as llama-server times a request: rendered and tokenized in the handler,
                // `prompt_ms` to the first sample, `predicted_ms` the N-1 steps after it scaled
                // to N as the crate does, the request over at the last sample.
                let requestStarted = clock.now
                let chunks = try aud.tokenizeClip(try aud.render(userContent), clipPath)
                defer { chunks.free() }
                try requireAudioChunk(chunks, calibrated: calibrated)
                let promptStarted = clock.now
                try aud.evalChunks(chunks)
                let generation = try aud.generate(decodeTokens, .afterLastSample)
                guard let firstSampledAt = generation.firstSampledAt, generation.tokens > 1 else {
                    throw RuntimeError.engine("decoded \(generation.tokens) tokens: no step "
                        + "after the first to time")
                }
                let steps = Double(generation.tokens)
                return try VlThroughputMeasurementReading.fromMillis(
                    promptTokens: try PositiveCount(chunks.promptTokens), imageTokens: nil,
                    decodeTokens: try PositiveCount(generation.tokens),
                    promptMs: durationMs(firstSampledAt - promptStarted),
                    predictedMs: durationMs(generation.lastSampledAt - firstSampledAt)
                        / (steps - 1) * steps,
                    totalTimeMs: durationMs(generation.lastSampledAt - requestStarted))
            },
            settleAudioTokens: { promptTokens in
                let (tokens, from) = try settleAudioTokens(
                    aud, cell, userText: userText, promptTokens: Int(promptTokens.value),
                    calibrated: calibrated, pinned: pinned)
                source = from
                return tokens
            })
        guard let source else { throw RuntimeError.engine("the run settled no audio token count") }
        return (resultData, source)
    }

    // MARK: - Shared with `asr_max_memory`

    /// Written once, outside every window, where mtmd's own reader opens it.
    static func withAsrClip<T>(_ cell: AsrCell, _ body: (String) async throws -> T) async throws -> T {
        let url = FileManager.default.temporaryDirectory
            .appending(path: "asr-\(UUID().uuidString).wav")
        // Registered before the write, so a failed write leaves nothing behind either.
        defer {
            do {
                try FileManager.default.removeItem(at: url)
            } catch CocoaError.fileNoSuchFile {
                // The write failed before creating it.
            } catch {
                AppLog.jobRun.warning("failed to delete the clip \(url.path): \(error)")
            }
        }
        try WavTone.tone(durationMs: cell.durationMs, sampleRateHz: cell.sampleRateHz,
                         channels: cell.channels)
            .write(to: url, options: .atomic)
        return try await body(url.path)
    }

    /// Sizes the context from the projector's count for the clip, as the crate's calibration
    /// helper does; a pinned `ctx_size` is kept.
    static func withCalibratedAudio<T>(
        _ request: RunRequest, _ model: ResolvedGgufVisionModel, _ cell: AsrCell,
        flags: RuntimeFlags, clipPath: String, calibrate: Bool, observer: MeasurementObserver,
        _ body: (AudioInference, Int?) async throws -> T
    ) async throws -> (T, RuntimeFlags) {
        var ran = flags
        var calibrated: Int?
        let load = LoadSettings(flags, request.benchmark)
        let result = try await withMtmd(
            path: model.backbone, mmprojPath: model.mmproj, nGpuLayers: load.numberGpuLayers,
            threads: load.threads, swaFull: load.swaFull, loadMode: load.loadMode, media: .audio,
            window: { mtmd in
                guard calibrate else { return (load.ctxSize, load.nUbatch) }
                // Calibration is not an attempt (`attempt_timeout_seconds`).
                let tokens = try observer.outside {
                    try LlamaMtmd.clipAudioTokens(mtmd, clipPath: clipPath)
                }
                guard tokens > 0 else {
                    throw RuntimeError.engine("the clip tokenized to no audio chunk")
                }
                calibrated = tokens
                ran = try LlamaRuntimeFlags.forRun(request, perImageTokens: UInt32(tokens),
                                                   recipeTextBytes: cell.recipe.textBytes)
                let counted = LoadSettings(ran, request.benchmark)
                return (counted.ctxSize, counted.nUbatch)
            }, { m, mtmd in try await body(.live(m, mtmd, cell.recipe), calibrated) })
        return (result, ran)
    }

    /// The crate's `build_text` and `user_text`. Untimed, once per cell.
    static func asrUserText(_ aud: AudioInference, _ cell: AsrCell) async throws -> String {
        cell.recipe.userText(try await asrText(aud, cell.recipe, textTokens: cell.textTokens))
    }

    static func asrPrompt(_ aud: AudioInference, _ cell: AsrCell) async throws
        -> (userText: String, prompt: String) {
        let userText = try await asrUserText(aud, cell)
        return (userText, try aud.render(cell.recipe.userContent(userText, marker: aud.mediaMarker)))
    }

    /// The whole user text part counted, less the bare ask, with no specials and untrimmed,
    /// as the crate counts it through `/tokenize`.
    static func asrText(_ aud: AudioInference, _ recipe: AsrRecipe,
                        textTokens: UInt32) async throws -> String {
        guard textTokens > 0 else { return "" }
        let ask = recipe.userText("")
        let without = ask.isEmpty ? 0 : try aud.tokenize(ask, false).count
        return try await PromptSeed.buildPromptText(targetTokens: Int(textTokens)) {
            max(0, try aud.tokenize(recipe.userText($0), false).count - without)
        }
    }

    /// The crate's `asr::audio_tokens` then `confirm_served_text`. An unconfirmed count fails
    /// the cell unless the context is pinned, when the measured difference is reported.
    static func settleAudioTokens(
        _ aud: AudioInference, _ cell: AsrCell, userText: String, promptTokens: Int,
        calibrated: Int?, pinned: Bool
    ) throws -> (PositiveCount, AsrAudioTokensSource) {
        let recipe = cell.recipe
        let textOnly = try aud.promptTokens(aud.render(recipe.userContent(userText, marker: nil)))
        guard promptTokens > textOnly else {
            throw RuntimeError.engine("the prompt took \(promptTokens) tokens with its clip and "
                + "\(textOnly) without: the clip added none")
        }
        let measured = promptTokens - textOnly
        let tokens: Int
        let source: AsrAudioTokensSource
        if let calibrated, calibrated + Int(recipe.audioMarkers) == measured {
            (tokens, source) = (calibrated, .calibrated)
        } else if let calibrated {
            let unconfirmed = AsrRefusalError.countUnconfirmed(
                calibrated: calibrated, markers: recipe.audioMarkers, measured: measured)
            guard pinned else { throw unconfirmed }
            AppLog.jobRun.warning("\(unconfirmed.localizedDescription); reporting the measured "
                + "audio tokens instead")
            (tokens, source) = (measured, .measured)
        } else {
            (tokens, source) = (measured, .measured)
        }
        AppLog.jobRun.info("token counts prompt_tokens=\(promptTokens) "
            + "text_only_prompt_tokens=\(textOnly) audio_tokens=\(tokens) "
            + "audio_tokens_source=\(source.rawValue)")
        if cell.textTokens > 0 {
            let withoutText = try aud.promptTokens(
                aud.render(recipe.userContent(recipe.userText(""), marker: nil)))
            guard textOnly - withoutText == Int(cell.textTokens) else {
                throw AsrRefusalError.textUnconfirmed(textTokens: cell.textTokens,
                                                      withText: textOnly, withoutText: withoutText)
            }
        }
        return (try PositiveCount(tokens), source)
    }

    /// A zero means the marker never expanded, so the prompt was plain text; a count other
    /// than the one that sized the context is not the clip the cell counted.
    static func requireAudioChunk(_ chunks: VisionChunks, calibrated: Int?) throws {
        guard chunks.audioTokens > 0 else {
            throw RuntimeError.engine(
                "asr prompt tokenized to \(chunks.promptTokens) tokens with no audio chunk "
                    + "among them; the media marker did not expand")
        }
        if let calibrated, chunks.audioTokens != calibrated {
            throw RuntimeError.engine(
                "the clip tokenized to \(chunks.audioTokens) audio tokens, \(calibrated) when "
                    + "the context was sized")
        }
    }
}
