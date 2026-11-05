import Foundation
import Testing

@testable import Pipette

/// On `AsrThroughputTests`'s fake `AudioInference`; the peak itself is `ProcessMemory`'s.
@Suite struct AsrMaxMemoryTests {
    private typealias FakeAudio = AsrThroughputTests.FakeAudio

    private func run(_ fake: FakeAudio, text: Int32 = 0, decode: Int32 = 256,
                     calibrated: Int? = 375) async throws {
        let benchmark = AsrMaxMemoryBenchmark(
            benchmarkId: BenchmarkId(validated: "asr_max_memory_2500ms_text\(text)_decode\(decode)"),
            parameterAudioDurationMs: try AudioDurationMillis(2_500),
            parameterAudioSampleRateHz: try AudioSampleRate(16_000),
            parameterAudioChannels: try AudioChannels(1),
            parameterTextTokens: try NonNegativeCount(Int(text)),
            parameterDecodeTokens: try AsrDecodeTokens(decode))
        try await LlamaExecute.asrMaxMemory(AsrCell(benchmark, recipe: fake.recipe), fake.make(),
                                            clipPath: "/tmp/clip.wav", calibrated: calibrated)
    }

    /// No warm-up or repeat, as the crate's measured server.
    @Test func runsOneClipRequestThenItsChecks() async throws {
        let fake = FakeAudio()
        try await run(fake, text: 12)
        #expect(fake.tokenizeClipCount == 1)
        #expect(fake.evalCount == 1)
        #expect(fake.generateEnds == [.afterLastSample])
        #expect(fake.freedChunks == 1)
        #expect(fake.probes.count == 2)
        #expect(fake.probes.allSatisfy { !$0.contains("<__media__>") })
    }

    @Test func sendsAsrThroughputsMessage() async throws {
        let fake = FakeAudio()
        fake.recipe = .lfm2Audio
        fake.markers = 0
        try await run(fake)
        #expect(fake.prompts == ["<s>Perform ASR.<u><__media__></u>"])
    }

    /// The peak would carry KV for audio the clip lacks.
    @Test func anUnconfirmedCountFailsTheCell() async {
        let fake = FakeAudio()
        fake.markers = 3
        await #expect(throws: AsrRefusalError.countUnconfirmed(
            calibrated: 375, markers: 2, measured: 378)) { try await run(fake) }
    }

    @Test func aPinnedContextChecksNoCount() async throws {
        let fake = FakeAudio()
        fake.markers = 3
        try await run(fake, calibrated: nil)
        #expect(fake.evalCount == 1)
    }

    @Test func aClipWithNoAudioChunkFailsBeforeThePrefill() async {
        let fake = FakeAudio()
        fake.audioTokens = 0
        await #expect(throws: RuntimeError.self) { try await run(fake, calibrated: nil) }
        #expect(fake.evalCount == 0)
        #expect(fake.freedChunks == 1)
    }
}
