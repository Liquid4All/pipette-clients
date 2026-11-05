import Foundation
import Testing

@testable import Pipette

/// Against a fake `AudioInference`; the live path is tested on device.
@Suite struct AsrThroughputTests {

    /// One token per character, so a confirmed count is the arithmetic of the strings.
    final class FakeAudio {
        enum Boom: Error { case engine }

        var recipe: AsrRecipe = .qwen25Omni
        var audioTokens = 375
        /// The begin/end tokens mtmd adds around the clip.
        var markers = 2
        /// Added to the audio count on each call after the first.
        var audioTokenDrift = 0
        /// Added to the text-free probe's count.
        var textFreeDrift = 0
        var evalDuration: TimeInterval = 0.004
        /// Up to the first sample, which ends `prompt_ms`.
        var firstSampleDuration: TimeInterval = 0
        var generateDuration: TimeInterval = 0.001
        /// Inside the whole request only.
        var tokenizeClipDuration: TimeInterval = 0
        var renderDuration: TimeInterval = 0
        var clipRenders = 0
        var failEvalOnCall: Int?

        var tokenizeClipCount = 0
        var evalCount = 0
        var generateEnds: [GenerationEnd] = []
        var freedChunks = 0
        var prompts: [String] = []
        var clipPaths: [String] = []
        var probes: [String] = []
        var addSpecials: Set<Bool> = []

        func make() -> AudioInference {
            AudioInference(
                mediaMarker: "<__media__>",
                tokenize: { text, addSpecial in
                    self.addSpecials.insert(addSpecial)
                    return (0 ..< (text.count + (addSpecial ? 1 : 0))).map { Int32($0 % 1000) }
                },
                // The system turn too, so a kernel that dropped it cannot match.
                render: { content in
                    if content.contains("<__media__>") {
                        self.clipRenders += 1
                        Thread.sleep(forTimeInterval: self.renderDuration)
                    }
                    return "<s>\(self.recipe.system ?? "")<u>\(content)</u>"
                },
                resetContext: {},
                resetSampler: {},
                tokenizeClip: { prompt, path in
                    let call = self.tokenizeClipCount
                    self.tokenizeClipCount += 1
                    self.prompts.append(prompt)
                    self.clipPaths.append(path)
                    Thread.sleep(forTimeInterval: self.tokenizeClipDuration)
                    let audio = self.audioTokens + call * self.audioTokenDrift
                    return VisionChunks(
                        promptTokens: prompt.count - "<__media__>".count + audio + self.markers,
                        imageTokens: 0, audioTokens: audio,
                        release: { self.freedChunks += 1 })
                },
                promptTokens: { prompt in
                    self.probes.append(prompt)
                    return prompt.count + (self.probes.count == 2 ? self.textFreeDrift : 0)
                },
                evalChunks: { _ in
                    let call = self.evalCount
                    self.evalCount += 1
                    if call == self.failEvalOnCall { throw Boom.engine }
                    Thread.sleep(forTimeInterval: self.evalDuration)
                },
                generate: { count, end in
                    self.generateEnds.append(end)
                    Thread.sleep(forTimeInterval: self.firstSampleDuration)
                    let first = ContinuousClock.now
                    Thread.sleep(forTimeInterval: self.generateDuration)
                    return LlamaGeneration(tokens: count, lastSampledAt: .now, firstSampledAt: first)
                })
        }
    }

    static func benchmark(text: Int32 = 0, decode: Int32 = 16) throws -> AsrThroughputBenchmark {
        AsrThroughputBenchmark(
            benchmarkId: BenchmarkId(validated: "asr_throughput_2500ms_text\(text)_decode\(decode)"),
            parameterAudioDurationMs: try AudioDurationMillis(2_500),
            parameterAudioSampleRateHz: try AudioSampleRate(16_000),
            parameterAudioChannels: try AudioChannels(1),
            parameterTextTokens: try NonNegativeCount(Int(text)),
            parameterDecodeTokens: try AsrDecodeTokens(decode))
    }

    private func run(_ fake: FakeAudio, text: Int32 = 0, decode: Int32 = 16,
                     calibrated: Int? = 375, pinned: Bool = false,
                     gate: ReadinessGate = {}) async throws -> (BenchmarkResultData, AsrAudioTokensSource) {
        let benchmark = try Self.benchmark(text: text, decode: decode)
        return try await LlamaExecute.asrThroughput(
            benchmark, AsrCell(benchmark, recipe: fake.recipe), fake.make(),
            clipPath: "/tmp/clip.wav", calibrated: calibrated, pinned: pinned,
            readinessGate: gate, observer: noObserver())
    }

    // MARK: - Shape of the run

    @Test func runsOneWarmupThenFiveMeasuredReps() async throws {
        let fake = FakeAudio()
        let (result, source) = try await run(fake)
        guard case let .asrThroughput(promptTokens, audioTokens, _, promptStddev?, _, _, _, _)
            = result else {
            Issue.record("got \(result)"); return
        }
        #expect(fake.evalCount == 6)       // 1 warm-up + 5 measured
        #expect(fake.tokenizeClipCount == 6)
        #expect(fake.freedChunks == 6)
        #expect(fake.clipPaths == Array(repeating: "/tmp/clip.wav", count: 6))
        #expect(audioTokens == 375)
        #expect(promptTokens == UInt32(fake.prompts[0].count - "<__media__>".count + 377))
        #expect(promptStddev >= 0)
        #expect(source == .calibrated)
        #expect(fake.generateEnds == Array(repeating: .afterLastSample, count: 6))
        #expect(fake.clipRenders == 6)
    }

    @Test func aFailedRepReleasesItsChunks() async {
        let fake = FakeAudio()
        fake.failEvalOnCall = 3
        await #expect(throws: FakeAudio.Boom.engine) { _ = try await run(fake) }
        #expect(fake.freedChunks == fake.tokenizeClipCount)
    }

    @Test func theGateFiresBeforeTheFirstMeasuredRep() async {
        let fake = FakeAudio()
        await #expect(throws: RuntimeError.cancelled) {
            _ = try await run(fake, gate: { throw RuntimeError.cancelled })
        }
        #expect(fake.evalCount == 1)
    }

    // MARK: - Prompt construction

    /// No newline around the marker.
    @Test(arguments: [
        (AsrRecipe.qwen25Omni, "<s>You are a speech recognition model.<u><__media__>"
            + "Transcribe the English audio into text without any punctuation marks.</u>"),
        (.lfm2Audio, "<s>Perform ASR.<u><__media__></u>"),
        (.gemma4, "<s><u>\(AsrRecipe.gemma4.instruction)<__media__></u>"),
    ] as [(AsrRecipe, String)])
    func rendersTheRecipesTurns(_ recipe: AsrRecipe, _ expected: String) async throws {
        let fake = FakeAudio()
        fake.recipe = recipe
        fake.markers = Int(recipe.audioMarkers)
        _ = try await run(fake)
        #expect(Set(fake.prompts) == [expected])
    }

    /// No specials, as the crate counts through `/tokenize`; the text goes before the ask.
    @Test func sizesTheTextAsTheCrate() async throws {
        let fake = FakeAudio()
        _ = try await run(fake, text: 12)
        #expect(fake.addSpecials == [false])
        let prompt = try #require(fake.prompts.first)
        let ask = AsrRecipe.qwen25Omni.instruction
        let user = try #require(prompt.components(separatedBy: "<__media__>").last?
            .dropLast("</u>".count))
        #expect(user.hasSuffix("\n" + ask))
        #expect(user.count - ask.count == 12)
    }

    // MARK: - Timed windows

    @Test func reportsThePrefillAndDecodeWindowsSeparately() async throws {
        let fake = FakeAudio()
        fake.evalDuration = 0.008
        let (result, _) = try await run(fake)
        guard case let .asrThroughput(_, _, promptMs, _, predictedMs, _, _, _) = result else {
            Issue.record("got \(result)"); return
        }
        #expect(promptMs >= 8)
        #expect(predictedMs < promptMs)
    }

    /// As the server's handler runs the clip's decode and mel.
    @Test func theClipsPreprocessingIsInTheTotalOnly() async throws {
        let fake = FakeAudio()
        fake.tokenizeClipDuration = 0.02
        let (result, _) = try await run(fake)
        guard case let .asrThroughput(_, _, promptMs, _, predictedMs, _, totalTimeMs?, _) = result
        else {
            Issue.record("got \(result)"); return
        }
        #expect(totalTimeMs - promptMs - predictedMs >= 15)
    }

    /// As llama-server renders each request in its handler.
    @Test func theRenderIsInTheTotalOnly() async throws {
        let fake = FakeAudio()
        fake.renderDuration = 0.02
        let (result, _) = try await run(fake)
        guard case let .asrThroughput(_, _, promptMs, _, predictedMs, _, totalTimeMs?, _) = result
        else {
            Issue.record("got \(result)"); return
        }
        #expect(totalTimeMs - promptMs - predictedMs >= 15)
    }

    /// The server's windows: `prompt_ms` to the first sample, then N-1 steps scaled to N.
    @Test func timesAsTheServerDoes() async throws {
        let fake = FakeAudio()
        fake.evalDuration = 0
        fake.firstSampleDuration = 0.02
        fake.generateDuration = 0.03
        let (result, _) = try await run(fake, decode: 4)
        guard case let .asrThroughput(_, _, promptMs, _, predictedMs, _, _, _) = result else {
            Issue.record("got \(result)"); return
        }
        #expect(promptMs >= 20)
        #expect(predictedMs >= 40)    // 30 ms over 3 steps, scaled to 4
    }

    // MARK: - The count

    /// With-clip less clip-free must be the counted audio plus the recipe's markers.
    @Test func anUnconfirmedCountFailsTheCell() async {
        let fake = FakeAudio()
        fake.markers = 3
        await #expect(throws: AsrRefusalError.countUnconfirmed(
            calibrated: 375, markers: 2, measured: 378)) { _ = try await run(fake) }
    }

    /// The count sized nothing, so the measured difference is reported, markers included.
    @Test func aPinnedContextReportsTheMeasuredCount() async throws {
        let fake = FakeAudio()
        fake.markers = 3
        let (result, source) = try await run(fake, pinned: true)
        guard case let .asrThroughput(_, audioTokens, _, _, _, _, _, _) = result else {
            Issue.record("got \(result)"); return
        }
        #expect(audioTokens == 378)
        #expect(source == .measured)
    }

    /// The clip every rep reads must be the one that sized the context.
    @Test func aDriftingAudioCountFailsTheCell() async {
        let fake = FakeAudio()
        fake.audioTokenDrift = 1
        await #expect(throws: RuntimeError.engine(
            "the clip tokenized to 376 audio tokens, 375 when the context was sized")) {
            _ = try await run(fake)
        }
    }

    @Test func aClipWithNoAudioChunkFailsTheCell() async {
        let fake = FakeAudio()
        fake.audioTokens = 0
        await #expect(throws: RuntimeError.self) { _ = try await run(fake, calibrated: nil) }
        #expect(fake.evalCount == 0)   // refused on the warm-up, before its prefill
    }

    /// Otherwise the row would not be the text the entry names.
    @Test func anUnconfirmedTextIsRefused() async throws {
        let fake = FakeAudio()
        fake.recipe = .lfm2Audio
        fake.markers = 0
        fake.textFreeDrift = 1
        await #expect(throws: AsrRefusalError.self) { _ = try await run(fake, text: 12) }
        // Skipped without text.
        _ = try await run(fake, text: 0)
    }

    /// Both clip-free: with the cell's text, then the bare ask.
    @Test func probesTheTextOnlyAndTextFreePrompts() async throws {
        let fake = FakeAudio()
        fake.recipe = .lfm2Audio
        fake.markers = 0
        _ = try await run(fake, text: 12)
        #expect(fake.probes.count == 2)
        #expect(fake.probes[1] == "<s>Perform ASR.<u></u>")
        #expect(fake.probes[0] == fake.prompts[0].replacingOccurrences(of: "<__media__>", with: ""))
    }

    @Test func theClipIsTheToneAndIsDeletedAfterAThrow() async throws {
        let cell = try AsrCell(Self.benchmark(), recipe: .qwen25Omni)
        var seen: String?
        await #expect(throws: FakeAudio.Boom.engine) {
            try await LlamaExecute.withAsrClip(cell) { path in
                seen = path
                #expect(try Data(contentsOf: URL(filePath: path)) == WavTone.tone(
                    durationMs: cell.durationMs, sampleRateHz: cell.sampleRateHz,
                    channels: cell.channels))
                throw FakeAudio.Boom.engine
            }
        }
        let path = try #require(seen)
        #expect(!FileManager.default.fileExists(atPath: path))
    }

    /// The crate entry's refusal: every template adds tokens of its own.
    @Test func audioAndTextFillingThePromptAreRefused() async throws {
        let benchmark = try Self.benchmark(text: 10)
        await #expect(throws: AsrThroughputMeasurementError.audioTokensFillPrompt(
            audioTokens: 95, textTokens: 10, promptTokens: 105)) {
            _ = try await BenchmarkMeasurement.asrThroughput(
                benchmark, reps: 2, readinessGate: {}, observer: noObserver(), warmup: .once,
                ctx: (), attempt: { _ in
                    try VlThroughputMeasurementReading.fromMillis(
                        promptTokens: 105, imageTokens: nil, decodeTokens: 16, promptMs: 1,
                        predictedMs: 1, totalTimeMs: 2)
                }, settleAudioTokens: { _ in 95 })
        }
    }
}
