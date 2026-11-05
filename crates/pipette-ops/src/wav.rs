//! The clip an ASR cell sends. How many audio tokens a clip becomes depends on
//! how many samples it holds and never on what they are, so every engine draws
//! the same quiet tone, in the format the entry names.

use pipette_plan_types::{AudioChannels, AudioDurationMillis, AudioSampleRate};

/// Every audio encoder llama.cpp's multimodal layer serves takes 16 kHz mono
/// (`mtmd_get_audio_sample_rate`), and the server resamples anything else, so
/// the token count of a clip is set by its duration at this rate.
pub const SAMPLE_RATE: u32 = 16_000;

const HEADER_BYTES: usize = 44;
const BYTES_PER_SAMPLE: u16 = 2;
/// A tone well inside the range, so it is audio and not silence or clipping.
const TONE_HZ: f64 = 440.0;
const TONE_AMPLITUDE: f64 = 0.1;

#[derive(Debug, thiserror::Error)]
pub enum ToneError {
    /// Out of reach of a plan: [`AudioDurationMillis`] bounds a clip to ten
    /// minutes, under 120 MB at 48 kHz stereo.
    #[error("{duration_ms} ms of audio does not fit a WAV file on this host")]
    TooLarge { duration_ms: u32 },
}

/// How a clip is encoded: 16-bit PCM at a rate and channel count. Built only
/// from the plan's validated types (or [`ClipFormat::ENCODER`]), so every format
/// is one an entry can name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClipFormat {
    sample_rate_hz: u32,
    channels: u16,
}

impl ClipFormat {
    /// What the encoders take, so a clip in it reaches them unresampled.
    pub const ENCODER: Self = Self {
        sample_rate_hz: SAMPLE_RATE,
        channels: 1,
    };

    /// The format an ASR entry names. The types bound the rate to 8 to 48 kHz,
    /// so it is positive.
    pub fn of(sample_rate_hz: AudioSampleRate, channels: AudioChannels) -> Self {
        Self {
            sample_rate_hz: sample_rate_hz.into_inner().unsigned_abs(),
            channels: channels.into_inner(),
        }
    }

    pub fn sample_rate_hz(self) -> u32 {
        self.sample_rate_hz
    }

    pub fn channels(self) -> u16 {
        self.channels
    }
}

/// The samples per channel in `duration` of audio at `sample_rate_hz`,
/// rounded down: exact wherever a millisecond is a whole number of samples (as
/// at [`SAMPLE_RATE`]), a fraction of a sample short at 44.1 kHz. The clip a cell
/// sends and the one its count is calibrated on are both this length.
pub fn samples_in(duration: AudioDurationMillis, sample_rate_hz: u32) -> u64 {
    u64::from(duration.as_u32()) * u64::from(sample_rate_hz) / 1000
}

/// A WAV file in `format` of `duration` of a tone, the same on every
/// channel. The plan's type bounds the duration, so the clip is never empty.
pub fn tone(duration: AudioDurationMillis, format: ClipFormat) -> Result<Vec<u8>, ToneError> {
    let duration_ms = duration.as_u32();
    let ClipFormat {
        sample_rate_hz,
        channels,
    } = format;
    let frames = usize::try_from(samples_in(duration, sample_rate_hz))
        .map_err(|_| ToneError::TooLarge { duration_ms })?;
    let block_align = BYTES_PER_SAMPLE * channels;
    let data_bytes = frames * usize::from(block_align);
    let mut wav = Vec::with_capacity(HEADER_BYTES + data_bytes);
    let data_len = u32::try_from(data_bytes).map_err(|_| ToneError::TooLarge { duration_ms })?;
    wav.extend(b"RIFF");
    wav.extend((36 + data_len).to_le_bytes());
    wav.extend(b"WAVEfmt ");
    wav.extend(16_u32.to_le_bytes()); // fmt chunk size
    wav.extend(1_u16.to_le_bytes()); // PCM
    wav.extend(channels.to_le_bytes());
    wav.extend(sample_rate_hz.to_le_bytes());
    wav.extend((sample_rate_hz * u32::from(block_align)).to_le_bytes()); // byte rate
    wav.extend(block_align.to_le_bytes());
    wav.extend(16_u16.to_le_bytes()); // bits per sample
    wav.extend(b"data");
    wav.extend(data_len.to_le_bytes());
    let step = std::f64::consts::TAU * TONE_HZ / f64::from(sample_rate_hz);
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss,
        reason = "the index stays under 30 M, exact in f64, and the product is inside i16"
    )]
    for i in 0..frames {
        let sample = (TONE_AMPLITUDE * f64::from(i16::MAX) * (step * i as f64).sin()) as i16;
        for _ in 0..channels {
            wav.extend(sample.to_le_bytes());
        }
    }
    Ok(wav)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ms(duration_ms: i32) -> anyhow::Result<AudioDurationMillis> {
        Ok(AudioDurationMillis::try_new(duration_ms)?)
    }

    /// A clip is exactly as long as asked, at every length the suite uses and at
    /// the windows' edges, where an off-by-one sample would move a token count.
    #[rstest::rstest]
    #[case(1, 16)]
    #[case(2_500, 40_000)]
    #[case(9_400, 150_400)]
    #[case(29_900, 478_400)]
    #[case(30_000, 480_000)]
    #[case(30_100, 481_600)]
    #[case(600_000, 9_600_000)]
    fn a_clip_holds_exactly_its_samples(
        #[case] duration_ms: u32,
        #[case] samples: usize,
    ) -> anyhow::Result<()> {
        let duration = AudioDurationMillis::try_new(i32::try_from(duration_ms)?)?;
        let wav = tone(duration, ClipFormat::ENCODER)?;
        assert_eq!(wav.len(), 44 + samples * 2);
        assert_eq!(samples_in(duration, SAMPLE_RATE), samples as u64);
        Ok(())
    }

    #[test]
    fn the_header_describes_16_bit_mono_pcm_at_16_khz() -> anyhow::Result<()> {
        let wav = tone(ms(1_000)?, ClipFormat::ENCODER)?;
        let u16_at = |at: usize| u16::from_le_bytes([wav[at], wav[at + 1]]);
        let u32_at =
            |at: usize| u32::from_le_bytes([wav[at], wav[at + 1], wav[at + 2], wav[at + 3]]);
        assert_eq!(&wav[..4], b"RIFF");
        assert_eq!(&wav[8..16], b"WAVEfmt ");
        assert_eq!(u32_at(4) as usize, wav.len() - 8);
        assert_eq!(u16_at(20), 1, "PCM");
        assert_eq!(u16_at(22), 1, "mono");
        assert_eq!(u32_at(24), 16_000);
        assert_eq!(u32_at(28), 32_000, "byte rate");
        assert_eq!(u16_at(34), 16, "bits per sample");
        assert_eq!(&wav[36..40], b"data");
        assert_eq!(u32_at(40) as usize, wav.len() - 44);
        Ok(())
    }

    /// The entry's format reaches the header, and every channel carries every
    /// frame.
    #[rstest::rstest]
    #[case(8_000, 1, 8_000)]
    #[case(44_100, 2, 88_200)]
    #[case(48_000, 2, 96_000)]
    fn the_header_carries_the_entry_format(
        #[case] sample_rate_hz: u32,
        #[case] channels: u16,
        #[case] samples: usize,
    ) -> anyhow::Result<()> {
        let format = ClipFormat {
            sample_rate_hz,
            channels,
        };
        let wav = tone(ms(1_000)?, format)?;
        let u16_at = |at: usize| u16::from_le_bytes([wav[at], wav[at + 1]]);
        let u32_at =
            |at: usize| u32::from_le_bytes([wav[at], wav[at + 1], wav[at + 2], wav[at + 3]]);
        assert_eq!(u16_at(22), channels);
        assert_eq!(u32_at(24), sample_rate_hz);
        assert_eq!(u32_at(28), sample_rate_hz * 2 * u32::from(channels));
        assert_eq!(u16_at(32), 2 * channels, "block align");
        assert_eq!(wav.len(), 44 + samples * 2);
        if channels == 2 {
            assert!(
                wav[44..]
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .all(|f| f[..2] == f[2..])
            );
        }
        Ok(())
    }

    /// Audio, not silence: an encoder given zeros would be given something else.
    #[test]
    fn the_clip_is_a_tone_inside_the_range() -> anyhow::Result<()> {
        let wav = tone(ms(100)?, ClipFormat::ENCODER)?;
        let samples: Vec<i16> = wav[44..]
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&b| i16::from_le_bytes(b))
            .collect();
        let peak = samples.iter().map(|s| s.unsigned_abs()).max().unwrap_or(0);
        assert!((3_000..=3_400).contains(&peak), "peak {peak}");
        assert!(samples.iter().any(|&s| s < 0) && samples.iter().any(|&s| s > 0));
        Ok(())
    }
}
