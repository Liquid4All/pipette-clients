import Testing
import Foundation
@testable import Pipette

/// `LlamaExecute.vlThroughput` against a fake `VisionInference`: no model, projector or
/// `mtmd` load. The live path with a real projector (`LlamaMtmd`) is tested on device.
@Suite struct VlThroughputTests {

    /// The two engine calls sleep for *different* durations so the tests can tell the two
    /// reported windows apart: with no-op closures a kernel that timed one window around
    /// both calls, or swapped them, would pass.
    final class FakeVision {
        enum Boom: Error { case engine }

        var evalDuration: TimeInterval = 0.004
        var generateDuration: TimeInterval = 0.001
        /// Building and tokenizing the frames, inside the whole request only.
        var tokenizeChunksDuration: TimeInterval = 0
        /// Spent after the last sample, as decoding the last token is.
        var lastDecodeDuration: TimeInterval = 0

        var tokenizeChunksCount = 0
        var evalCount = 0
        var generateCount = 0
        var freedChunks = 0
        var prompts: [String] = []
        var lastWidth: UInt32 = 0
        var lastHeight: UInt32 = 0
        var lastImages = 0
        var lastGenerateCount = 0
        var generateEnds: [GenerationEnd] = []
        var lastAddSpecial: Bool?

        var promptTokens = 1_300
        var imageTokens = 1_296
        /// Added to `promptTokens` on each call after the first, to stand in for an engine
        /// whose preprocessing is not reproducible across reps.
        var promptTokenDrift = 0
        /// Return a count other than the one asked for, for the short-generation case.
        var generateReturns: Int?
        /// Throw from `evalChunks` on this call index (0-based, warm-up included).
        var failEvalOnCall: Int?

        func make() -> VisionInference {
            VisionInference(
                mediaMarker: "<__media__>",
                tokenize: { text, addSpecial in
                    self.lastAddSpecial = addSpecial
                    // One id per character, which is enough for `PromptSeed` to size a
                    // prompt against: the builder only needs a monotone counter.
                    return (0 ..< (text.count + (addSpecial ? 1 : 0))).map { Int32($0 % 1000) }
                },
                // A visibly-wrapped template, so a kernel that skipped it or put the
                // content anywhere but inside the user turn cannot match the expectations.
                applyChatTemplate: { "<|user|>\($0)<|assistant|>" },
                resetContext: {},
                resetSampler: {},
                tokenizeChunks: { prompt, width, height, images in
                    let call = self.tokenizeChunksCount
                    self.tokenizeChunksCount += 1
                    self.prompts.append(prompt)
                    self.lastWidth = width; self.lastHeight = height; self.lastImages = images
                    Thread.sleep(forTimeInterval: self.tokenizeChunksDuration)
                    return VisionChunks(
                        promptTokens: self.promptTokens + call * self.promptTokenDrift,
                        imageTokens: self.imageTokens,
                        release: { self.freedChunks += 1 })
                },
                evalChunks: { _ in
                    let call = self.evalCount
                    self.evalCount += 1
                    if call == self.failEvalOnCall { throw Boom.engine }
                    Thread.sleep(forTimeInterval: self.evalDuration)
                },
                generate: { count, end in
                    self.generateCount += 1; self.lastGenerateCount = count
                    self.generateEnds.append(end)
                    Thread.sleep(forTimeInterval: self.generateDuration)
                    let lastSampledAt = ContinuousClock.now
                    Thread.sleep(forTimeInterval: self.lastDecodeDuration)
                    return LlamaGeneration(tokens: self.generateReturns ?? count,
                                           lastSampledAt: lastSampledAt)
                })
        }
    }

    private func run(_ vis: VisionInference, width: PositiveCount = 224,
                     height: PositiveCount = 224, text: NonNegativeCount = 8,
                     decode: PositiveCount = 16, images: NonNegativeCount = 1,
                     gate: ReadinessGate = {}) async throws -> BenchmarkResultData {
        try await LlamaExecute.vlThroughput(
            VlThroughputBenchmark(
                benchmarkId: BenchmarkId(validated: "vl_throughput_\(width)x\(height)"),
                parameterImageWidth: width, parameterImageHeight: height,
                parameterTextTokens: text, parameterDecodeTokens: decode,
                parameterNumImages: images),
            vis, readinessGate: gate, observer: noObserver())
    }

    // MARK: - Shape of the run

    @Test func runsOneWarmupThenFiveMeasuredReps() async throws {
        let fake = FakeVision()
        let result = try await run(fake.make())
        guard case let .vlThroughput(promptTokens, imageTokens, _, _, _, _, _, _) = result else {
            Issue.record("got \(result)"); return
        }
        #expect(fake.evalCount == 6)       // 1 warm-up + 5 measured
        #expect(fake.generateCount == 6)
        #expect(promptTokens == 1_300)
        #expect(imageTokens == 1_296)
    }

    /// The chunks own the preprocessed image tensors, the cell's largest allocation at high
    /// frame counts, so each rep releases its own before the next builds another.
    @Test func everyRepReleasesItsChunks() async throws {
        let fake = FakeVision()
        _ = try await run(fake.make())
        #expect(fake.tokenizeChunksCount == 6)
        #expect(fake.freedChunks == 6)
    }

    @Test func aFailedRepReleasesItsChunks() async {
        let fake = FakeVision()
        fake.failEvalOnCall = 3          // warm-up + two reps in
        await #expect(throws: FakeVision.Boom.engine) { _ = try await run(fake.make()) }
        #expect(fake.freedChunks == fake.tokenizeChunksCount)
    }

    @Test func forwardsTheCellsShapeToTheEngine() async throws {
        let fake = FakeVision()
        _ = try await run(fake.make(), width: 512, height: 384, text: 8, decode: 64, images: 3)
        #expect(fake.lastWidth == 512)
        #expect(fake.lastHeight == 384)
        #expect(fake.lastImages == 3)
        #expect(fake.lastGenerateCount == 64)
    }

    // MARK: - Prompt construction

    /// The desktop runner's request: one user turn holding the text part, then one marker
    /// per frame with nothing between, the join the server applies to a text part followed
    /// by image parts. mtmd merges adjacent frames, so a video-style encoder can compress
    /// across them.
    @Test(arguments: [NonNegativeCount(1), 2, 8])
    func templatesTheTextThenOneMarkerPerFrame(_ images: NonNegativeCount) async throws {
        let fake = FakeVision()
        _ = try await run(fake.make(), text: 12, images: images)
        let prompt = try #require(fake.prompts.first)
        let markers = String(repeating: "<__media__>", count: Int(images.value))
        #expect(prompt.hasPrefix("<|user|>"))
        #expect(prompt.hasSuffix(markers + "<|assistant|>"))
        let text = prompt.dropFirst("<|user|>".count).dropLast((markers + "<|assistant|>").count)
        #expect(!text.contains("<__media__>"))
        #expect(!text.isEmpty)
        // The counts are read off one rep, so every rep must send the identical prompt.
        #expect(Set(fake.prompts).count == 1)
    }

    /// A zero text budget sends the markers alone inside the template, as a `t0` cell
    /// means it: the image's prefill cost is the thing being isolated, under the same
    /// scaffolding the desktop's text-less chat request carries.
    @Test func zeroTextTokensSendsTheMarkersAlone() async throws {
        let fake = FakeVision()
        _ = try await run(fake.make(), text: 0)
        #expect(fake.prompts.first == "<|user|><__media__><|assistant|>")
    }

    /// The marker is the projector's own, never a literal this side restates: a prompt
    /// carrying the wrong one tokenizes to pure text, and the cell would measure a text
    /// prefill under a vision benchmark's name.
    @Test func usesTheProjectorsOwnMarker() async throws {
        let fake = FakeVision()
        var vis = fake.make()
        vis.mediaMarker = "<<media>>"
        _ = try await run(vis, text: 0)
        #expect(fake.prompts.first == "<|user|><<media>><|assistant|>")
    }

    /// The text budget counts specials, as the desktop's `/tokenize` call does
    /// (`add_special: true`): both paths then prefill the same number of text-side tokens
    /// for the same `parameter_text_tokens`.
    @Test func sizesTheTextCountingSpecials() async throws {
        let fake = FakeVision()
        _ = try await run(fake.make(), text: 12)
        #expect(fake.lastAddSpecial == true)
    }

    // MARK: - The two timed windows

    /// `prompt_ms` (vision encode + prefill) and `predicted_ms` (generation) are two
    /// windows, not one split after the fact.
    @Test func reportsThePrefillAndDecodeWindowsSeparately() async throws {
        let fake = FakeVision()
        fake.evalDuration = 0.008
        fake.generateDuration = 0.001
        let result = try await run(fake.make())
        guard case let .vlThroughput(_, _, promptMs, _, predictedMs, _, _, _) = result else {
            Issue.record("got \(result)"); return
        }
        #expect(promptMs >= 8)
        #expect(predictedMs >= 1)
        // No upper bound: a scheduling stall on a loaded runner would fail it for nothing
        // the kernel did. The ordering alone rules out a single or swapped window.
        #expect(predictedMs < promptMs)
    }

    /// `total_time_ms` runs from tokenizing the frames (in neither phase) to the last sample
    /// (before its decode, which `predicted_ms` includes). 20 ms in either moves it off the
    /// phases' sum; 15 ms leaves 5 ms for stalls.
    @Test(arguments: [
        TotalWindowCase(tokenizing: 0.02, lastDecode: 0, sign: 1),
        TotalWindowCase(tokenizing: 0, lastDecode: 0.02, sign: -1),
    ])
    func timesOneWindowFromTokenizingToTheLastSample(_ window: TotalWindowCase) async throws {
        let fake = FakeVision()
        fake.tokenizeChunksDuration = window.tokenizing
        fake.lastDecodeDuration = window.lastDecode
        let result = try await run(fake.make())
        guard case let .vlThroughput(_, _, promptMs, _, predictedMs, _, totalTimeMs?, _) = result
        else {
            Issue.record("got \(result)"); return
        }
        #expect(window.sign * (totalTimeMs - promptMs - predictedMs) >= 15)
    }

    /// As the benchmark defines `predicted_ms`: the last token's decode included.
    @Test func theDecodeWindowDecodesEveryGeneratedToken() async throws {
        let fake = FakeVision()
        _ = try await run(fake.make())
        #expect(fake.generateEnds == Array(repeating: .afterLastDecode, count: 6))
    }

    // MARK: - Per-rep validation

    /// A generation that came out the wrong length describes a different workload, so the
    /// run fails rather than reporting it.
    @Test func aShortGenerationFailsTheCell() async {
        let fake = FakeVision()
        fake.generateReturns = 15
        await #expect(throws: VlThroughputMeasurementError.mismatch(
            field: "decode_tokens", actual: 15, expected: 16)) {
            _ = try await run(fake.make())
        }
        // The warm-up is checked too, so nothing is measured at all.
        #expect(fake.evalCount == 1)
    }

    /// No image chunk in the prompt means the markers never expanded: the rep prefilled
    /// plain text and would have been reported as a vision measurement.
    @Test func aPromptWithNoImageChunkFailsTheCell() async {
        let fake = FakeVision()
        fake.imageTokens = 0
        await #expect(throws: RuntimeError.engine(
            "vl_throughput prompt tokenized to 1300 tokens with no image chunk among them; "
                + "the media marker did not expand")) {
            _ = try await run(fake.make())
        }
        #expect(fake.evalCount == 1)   // refused on the warm-up
    }

    /// Every rep must repeat the warm-up's prompt: the same frames preprocessed into another
    /// token count are not one workload.
    @Test func driftingPromptTokensFailTheCell() async {
        let fake = FakeVision()
        fake.promptTokenDrift = 1
        await #expect(throws: VlThroughputMeasurementError.promptVaried(
            first: 1_300, actual: 1_301)) {
            _ = try await run(fake.make())
        }
        #expect(fake.evalCount == 2)   // the warm-up, then the rep that drifted from it
    }

    /// The frame count is bounded, as the crate bounds it: every frame is a preprocessed
    /// image tensor held until the rep ends, so a nonsensical plan value is refused by
    /// name before anything is built.
    @Test func anAbsurdFrameCountIsRefusedBeforeAnythingRuns() async {
        let fake = FakeVision()
        let over = LlamaMtmd.maxNumImages + 1
        await #expect(throws: RuntimeError.engine(
            "parameter_num_images \(over) exceeds the supported maximum "
                + "\(LlamaMtmd.maxNumImages)")) {
            _ = try await run(fake.make(), images: NonNegativeCount(validated: Int32(over)))
        }
        #expect(fake.tokenizeChunksCount == 0)
    }

    // MARK: - Gating

    /// The warm-up runs ungated and the gate fires before the first measured rep, as it
    /// does for every other timing cell.
    @Test func theGateFiresBeforeTheFirstMeasuredRep() async {
        let fake = FakeVision()
        await #expect(throws: RuntimeError.cancelled) {
            _ = try await run(fake.make(), gate: { throw RuntimeError.cancelled })
        }
        #expect(fake.evalCount == 1)
    }
}

/// Seconds added before the prefill or to the last decode, and which way that moves
/// `total_time_ms` off the two phases' sum.
struct TotalWindowCase: Sendable {
    let tokenizing: TimeInterval
    let lastDecode: TimeInterval
    let sign: Double
}
