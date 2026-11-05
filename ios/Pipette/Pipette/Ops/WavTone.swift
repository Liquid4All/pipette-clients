import Foundation

// The crate's `pipette-ops/src/wav.rs`, byte for byte: the clip an ASR cell sends. The audio
// token count depends only on the sample count, so every client draws the same quiet tone.

nonisolated enum WavTone {
    /// What every audio encoder mtmd serves takes (`mtmd_get_audio_sample_rate`).
    static let sampleRate: UInt32 = 16_000

    static let headerBytes = 44
    private static let bytesPerSample: UInt16 = 2
    /// A tone well inside the range, so it is audio and not silence or clipping.
    private static let toneHz = 440.0
    private static let toneAmplitude = 0.1

    /// Samples per channel, rounded down: a fraction of a sample short at 44.1 kHz.
    static func samplesIn(durationMs: UInt32, sampleRateHz: UInt32) -> UInt64 {
        UInt64(durationMs) * UInt64(sampleRateHz) / 1000
    }

    /// The plan's types bound the clip to ten minutes at 48 kHz stereo, under 120 MB, so every
    /// size fits the header's `UInt32`s.
    static func tone(durationMs: UInt32, sampleRateHz: UInt32, channels: UInt16) -> Data {
        let frames = Int(samplesIn(durationMs: durationMs, sampleRateHz: sampleRateHz))
        let blockAlign = bytesPerSample * channels
        let dataLen = UInt32(frames * Int(blockAlign))
        var wav = Data(capacity: headerBytes + Int(dataLen))
        func le<T: FixedWidthInteger>(_ value: T) {
            withUnsafeBytes(of: value.littleEndian) { wav.append(contentsOf: $0) }
        }
        wav.append(contentsOf: Array("RIFF".utf8))
        le(36 + dataLen)
        wav.append(contentsOf: Array("WAVEfmt ".utf8))
        le(UInt32(16))                                  // fmt chunk size
        le(UInt16(1))                                   // PCM
        le(channels)
        le(sampleRateHz)
        le(sampleRateHz * UInt32(blockAlign))           // byte rate
        le(blockAlign)
        le(UInt16(16))                                  // bits per sample
        wav.append(contentsOf: Array("data".utf8))
        le(dataLen)
        let step = 2 * Double.pi * toneHz / Double(sampleRateHz)
        var samples: [Int16] = []
        samples.reserveCapacity(frames * Int(channels))
        for i in 0 ..< frames {
            // `Int16(_:)` truncates toward zero, as the crate's `as i16` does.
            let sample = Int16(toneAmplitude * Double(Int16.max) * sin(step * Double(i)))
            for _ in 0 ..< channels { samples.append(sample.littleEndian) }
        }
        samples.withUnsafeBytes { wav.append(contentsOf: $0) }
        return wav
    }
}
