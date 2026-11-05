import Foundation
import Testing

@testable import Pipette

/// `WavTone` against the crate's `wav.rs` tests.
@Suite struct WavToneTests {
    private func u16(_ wav: Data, _ at: Int) -> UInt16 {
        UInt16(wav[at]) | UInt16(wav[at + 1]) << 8
    }

    private func u32(_ wav: Data, _ at: Int) -> UInt32 {
        (0 ..< 4).reduce(0) { $0 | UInt32(wav[at + $1]) << (8 * $1) }
    }

    private func samples(_ wav: Data) -> [Int16] {
        stride(from: WavTone.headerBytes, to: wav.count, by: 2).map {
            Int16(bitPattern: u16(wav, $0))
        }
    }

    /// At the windows' edges an off-by-one sample would move a token count.
    @Test(arguments: [(UInt32(1), 16), (2_500, 40_000), (9_400, 150_400), (29_900, 478_400),
                      (30_000, 480_000), (30_100, 481_600)])
    func aClipHoldsExactlyItsSamples(_ durationMs: UInt32, _ samples: Int) {
        let wav = WavTone.tone(durationMs: durationMs, sampleRateHz: 16_000, channels: 1)
        #expect(wav.count == 44 + samples * 2)
        #expect(WavTone.samplesIn(durationMs: durationMs, sampleRateHz: 16_000) == UInt64(samples))
    }

    /// A fraction of a sample short at 44.1 kHz, as the crate floors it.
    @Test func theFrameCountIsFloored() {
        #expect(WavTone.samplesIn(durationMs: 1, sampleRateHz: 44_100) == 44)
        #expect(WavTone.samplesIn(durationMs: 2_500, sampleRateHz: 44_100) == 110_250)
    }

    @Test func theHeaderDescribes16BitMonoPcmAt16Khz() {
        let wav = WavTone.tone(durationMs: 1_000, sampleRateHz: 16_000, channels: 1)
        #expect(Array(wav[0 ..< 4]) == Array("RIFF".utf8))
        #expect(Array(wav[8 ..< 16]) == Array("WAVEfmt ".utf8))
        #expect(Int(u32(wav, 4)) == wav.count - 8)
        #expect(u32(wav, 16) == 16, "fmt chunk size")
        #expect(u16(wav, 20) == 1, "PCM")
        #expect(u16(wav, 22) == 1, "mono")
        #expect(u32(wav, 24) == 16_000)
        #expect(u32(wav, 28) == 32_000, "byte rate")
        #expect(u16(wav, 32) == 2, "block align")
        #expect(u16(wav, 34) == 16, "bits per sample")
        #expect(Array(wav[36 ..< 40]) == Array("data".utf8))
        #expect(Int(u32(wav, 40)) == wav.count - 44)
    }

    /// Every channel carries every frame too.
    @Test(arguments: [(UInt32(8_000), UInt16(1), 8_000), (44_100, 2, 88_200), (48_000, 2, 96_000)])
    func theHeaderCarriesTheEntryFormat(_ rate: UInt32, _ channels: UInt16, _ samples: Int) {
        let wav = WavTone.tone(durationMs: 1_000, sampleRateHz: rate, channels: channels)
        #expect(u16(wav, 22) == channels)
        #expect(u32(wav, 24) == rate)
        #expect(u32(wav, 28) == rate * 2 * UInt32(channels))
        #expect(u16(wav, 32) == 2 * channels, "block align")
        #expect(wav.count == 44 + samples * 2)
        if channels == 2 {
            let pcm = self.samples(wav)
            #expect(stride(from: 0, to: pcm.count, by: 2).allSatisfy { pcm[$0] == pcm[$0 + 1] })
        }
    }

    /// Audio, not silence: an encoder given zeros would be given something else.
    @Test func theClipIsAToneInsideTheRange() {
        let pcm = samples(WavTone.tone(durationMs: 100, sampleRateHz: 16_000, channels: 1))
        let peak = pcm.map(\.magnitude).max() ?? 0
        #expect((3_000 ... 3_400).contains(peak), "peak \(peak)")
        #expect(pcm.contains { $0 < 0 } && pcm.contains { $0 > 0 })
    }

    /// As the crate's `as i16`: samples 5 and 19 of the 16 kHz tone are 2491.62… and -461.69….
    @Test func samplesTruncateTowardZero() {
        let pcm = samples(WavTone.tone(durationMs: 2, sampleRateHz: 16_000, channels: 1))
        #expect(pcm[0] == 0)
        #expect(pcm[5] == 2491)
        #expect(pcm[19] == -461)
    }
}
