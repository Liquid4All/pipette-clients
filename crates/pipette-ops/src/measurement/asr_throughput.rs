//! ASR throughput: the times to prefill an audio clip, to decode after it and
//! to answer the whole request, reduced over the reps
//! (`docs/methodology/asr-throughput.md`).
//!
//! The audio counterpart of [`super::vl_throughput`], but how many tokens a
//! clip becomes is the model's business (the encoder's frame rate, its 30 s
//! windows, its pooling), so the engine counts them and this entry holds the
//! count below what the server prefilled.

use std::num::NonZeroUsize;

use anyhow::Context;

use pipette_plan_types::{
    AsrThroughputBenchmark, BenchmarkResultData, PositiveCount, PositiveMillis,
};

use super::vl_throughput::{
    ThroughputReading, VlThroughputMeasurementError, count, millis, require_decode_steps,
    throughput,
};
use super::{AttemptError, MeasurementObserver, MeasurementWarmup, Stats};
use crate::generation::{Clock, Generation};
use crate::readiness::ReadinessGate;

/// An audio count an ASR throughput cell cannot use; a reading it cannot use
/// is a [`VlThroughputMeasurementError`].
#[derive(Debug, thiserror::Error)]
pub enum AsrThroughputMeasurementError {
    #[error(
        "engine counted audio_tokens {audio_tokens} of prompt_tokens {prompt_tokens} beside \
         {text_tokens} text tokens: the prompt holds the template too, so the audio and the text \
         are less than all of it"
    )]
    AudioTokensFillPrompt {
        audio_tokens: PositiveCount,
        text_tokens: u32,
        prompt_tokens: PositiveCount,
    },
}

/// What one attempt ran, as the engine reports it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AsrThroughputMeasurementReading {
    /// The whole prompt: audio, instruction and template tokens. The same on
    /// every rep.
    pub prompt_tokens: PositiveCount,
    /// Tokens generated, checked against `parameter_decode_tokens`.
    pub decode_tokens: PositiveCount,
    /// Time to process all `prompt_tokens`, the audio encode included.
    pub prompt_ms: PositiveMillis,
    /// Time to generate all `decode_tokens`.
    pub predicted_ms: PositiveMillis,
    /// The whole request on the client's clock, measured rather than
    /// `prompt_ms + predicted_ms`
    /// (`docs/methodology/asr-throughput.md#end-to-end-latency`).
    pub total_time_ms: PositiveMillis,
}

impl AsrThroughputMeasurementReading {
    /// A reading from a generation that ran to its limit, timed on `clock` and
    /// scaled to every decode step as
    /// `VlThroughputMeasurementReading::from_generation` is.
    ///
    /// [`Clock::Client`] needs a streamed generation, whose first and last
    /// token arrive apart; an engine that sends one request and reads one reply
    /// (llama.cpp's cell) has all three stamps at one instant, no decode time,
    /// and times on [`Clock::Server`].
    pub fn from_generation(generation: &Generation, clock: Clock) -> anyhow::Result<Self> {
        let (prompt_ms, after_first_ms) = generation.phases_ms(clock)?;
        let (prompt_tokens, decode_tokens) = generation.counts_at_limit()?.positive()?;
        require_decode_steps(decode_tokens)?;
        let tokens = f64::from(decode_tokens.into_inner());
        Ok(Self::from_millis(
            prompt_tokens,
            decode_tokens,
            prompt_ms,
            after_first_ms / (tokens - 1.0) * tokens,
            generation.end_ms,
        )?)
    }

    /// From a decode time already covering all `decode_tokens` steps.
    pub fn from_millis(
        prompt_tokens: PositiveCount,
        decode_tokens: PositiveCount,
        prompt_ms: f64,
        predicted_ms: f64,
        total_time_ms: f64,
    ) -> Result<Self, VlThroughputMeasurementError> {
        Ok(Self {
            prompt_tokens,
            decode_tokens,
            prompt_ms: millis("prompt_ms", prompt_ms)?,
            predicted_ms: millis("predicted_ms", predicted_ms)?,
            total_time_ms: millis("total_time_ms", total_time_ms)?,
        })
    }
}

impl ThroughputReading for AsrThroughputMeasurementReading {
    fn prompt_tokens(&self) -> PositiveCount {
        self.prompt_tokens
    }

    fn decode_tokens(&self) -> PositiveCount {
        self.decode_tokens
    }

    fn times(&self) -> [PositiveMillis; 3] {
        [self.prompt_ms, self.predicted_ms, self.total_time_ms]
    }

    fn log(&self, label: &str) {
        tracing::info!(
            cell = label,
            prompt_ms = f64::from(self.prompt_ms.into_inner()),
            predicted_ms = f64::from(self.predicted_ms.into_inner()),
            total_time_ms = f64::from(self.total_time_ms.into_inner()),
            prompt_tokens = self.prompt_tokens.into_inner(),
            decode_tokens = self.decode_tokens.into_inner(),
            "asr throughput reading"
        );
    }

    /// The audio count is settled after the reps, not read off each one.
    fn check(&self, _first: Option<&Self>) -> Result<(), VlThroughputMeasurementError> {
        Ok(())
    }
}

/// Measure an ASR throughput cell on the reps [`super::vl_throughput`] runs:
/// `reps` attempts, each one engine call, checked against the entry and the
/// first attempt's prompt, and each time reduced to a mean and stddev. Each rep
/// waits on `readiness_gate` first.
///
/// `settle_audio_tokens` is called once, after the reps, with the prompt's
/// token count, and returns how many of them are the audio: a calibrated count
/// or a measured subtraction, the engine's to choose. The entry refuses one
/// that, with the cell's text, is not below the prompt, since every chat
/// template adds tokens of its own. After the reps, so the probe a subtraction
/// needs is not a request a rep follows.
///
/// `ctx` is lent to every attempt, the warm-up's included, for what the engine
/// records while they run: the server it sends to, or the processes it ran.
/// Lend a reference to read it back afterwards, a failed run included.
#[expect(
    clippy::too_many_arguments,
    reason = "the cell, the seams every timing entry takes, and the engine's two callbacks"
)]
pub fn asr_throughput<C>(
    benchmark: &AsrThroughputBenchmark,
    reps: NonZeroUsize,
    readiness_gate: ReadinessGate<'_>,
    observer: MeasurementObserver<'_>,
    warmup: MeasurementWarmup,
    ctx: C,
    attempt: impl FnMut(&mut C) -> Result<AsrThroughputMeasurementReading, AttemptError>,
    settle_audio_tokens: impl FnOnce(PositiveCount) -> anyhow::Result<PositiveCount>,
) -> anyhow::Result<BenchmarkResultData> {
    // Names the cell in every log line and error.
    let marker = benchmark.benchmark_id.to_string();
    let (first, [prompt, predicted, total]) = throughput(
        &marker,
        benchmark
            .parameter_decode_tokens
            .into_inner()
            .unsigned_abs(),
        reps,
        readiness_gate,
        observer,
        warmup,
        ctx,
        attempt,
    )?;

    let audio_tokens = settle_audio_tokens(first.prompt_tokens)
        .with_context(|| format!("{marker}: the audio token count"))?;
    let text_tokens = benchmark.parameter_text_tokens.into_inner().unsigned_abs();
    if u64::from(audio_tokens.into_inner().unsigned_abs()) + u64::from(text_tokens)
        >= u64::from(first.prompt_tokens.into_inner().unsigned_abs())
    {
        return Err(AsrThroughputMeasurementError::AudioTokensFillPrompt {
            audio_tokens,
            text_tokens,
            prompt_tokens: first.prompt_tokens,
        })
        .context(format!("{marker}: the audio token count"));
    }
    tracing::info!(
        cell = marker,
        prompt_tokens = first.prompt_tokens.into_inner(),
        audio_tokens = audio_tokens.into_inner(),
        "asr token counts"
    );

    // One rep has no spread to report.
    let stddev = |stats: Stats| (reps.get() > 1).then_some(stats.stddev_ms);

    Ok(BenchmarkResultData::AsrThroughput {
        asr_prompt_tokens: count(first.prompt_tokens),
        asr_audio_tokens: count(audio_tokens),
        asr_prompt_ms: prompt.mean_ms,
        asr_prompt_ms_stddev: stddev(prompt),
        asr_predicted_ms: predicted.mean_ms,
        asr_predicted_ms_stddev: stddev(predicted),
        total_time_ms: Some(total.mean_ms),
        total_time_ms_stddev: stddev(total),
    })
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use pipette_plan_types::{
        AsrDecodeTokens, AudioChannels, AudioDurationMillis, AudioSampleRate, BenchmarkId,
        NonNegativeCount,
    };

    use super::*;
    use crate::generation::{Counts, Finish, ServerClock};

    const MARKER: &str = "asr_throughput_10000ms_text0_decode64";

    /// A reading whose whole request took as long as its two phases.
    fn reading(
        prompt_tokens: i32,
        decode_tokens: i32,
        prompt_ms: f64,
        predicted_ms: f64,
    ) -> anyhow::Result<AsrThroughputMeasurementReading> {
        timed_reading(
            prompt_tokens,
            decode_tokens,
            prompt_ms,
            predicted_ms,
            prompt_ms + predicted_ms,
        )
    }

    fn timed_reading(
        prompt_tokens: i32,
        decode_tokens: i32,
        prompt_ms: f64,
        predicted_ms: f64,
        total_time_ms: f64,
    ) -> anyhow::Result<AsrThroughputMeasurementReading> {
        Ok(AsrThroughputMeasurementReading::from_millis(
            PositiveCount::try_new(prompt_tokens)?,
            PositiveCount::try_new(decode_tokens)?,
            prompt_ms,
            predicted_ms,
            total_time_ms,
        )?)
    }

    fn body() -> anyhow::Result<AsrThroughputBenchmark> {
        body_with_text(0)
    }

    fn body_with_text(text_tokens: i32) -> anyhow::Result<AsrThroughputBenchmark> {
        Ok(AsrThroughputBenchmark {
            benchmark_id: BenchmarkId::try_new(MARKER)?,
            parameter_audio_duration_ms: AudioDurationMillis::try_new(10_000)?,
            parameter_audio_sample_rate_hz: AudioSampleRate::try_new(16_000)?,
            parameter_audio_channels: AudioChannels::try_new(1)?,
            parameter_text_tokens: NonNegativeCount::try_new(text_tokens)?,
            parameter_decode_tokens: AsrDecodeTokens::try_new(64)?,
        })
    }

    fn no_observer() -> MeasurementObserver<'static> {
        MeasurementObserver {
            on_before_rep_start: &|| Ok(()),
            on_after_rep_end: &|| Ok(()),
        }
    }

    /// The entry behind an open gate, with no observer and no warm-up, its
    /// audio count settled to `audio_tokens`.
    fn unobserved(
        benchmark: &AsrThroughputBenchmark,
        reps: NonZeroUsize,
        audio_tokens: i32,
        mut attempt: impl FnMut() -> anyhow::Result<AsrThroughputMeasurementReading>,
    ) -> anyhow::Result<BenchmarkResultData> {
        asr_throughput(
            benchmark,
            reps,
            &|| Ok(()),
            no_observer(),
            MeasurementWarmup::Inline,
            (),
            |()| Ok(attempt()?),
            |_| Ok(PositiveCount::try_new(audio_tokens)?),
        )
    }

    /// The engine's times and counts are taken as given; the audio count is the
    /// one the settle step returned. One rep has no spread to report.
    #[rstest]
    #[case::as_given(reading(300, 64, 120.0, 40.0), 256, 5)]
    #[case::all_but_one(reading(300, 64, 120.0, 40.0), 299, 5)]
    #[case::one_rep(reading(300, 64, 120.0, 40.0), 256, 1)]
    fn a_reading_becomes_the_result(
        #[case] reading: anyhow::Result<AsrThroughputMeasurementReading>,
        #[case] audio_tokens: i32,
        #[case] reps: usize,
    ) -> anyhow::Result<()> {
        let reading = reading?;
        let result_data = unobserved(
            &body()?,
            NonZeroUsize::try_from(reps)?,
            audio_tokens,
            || Ok(reading),
        )?;
        let stddev = (reps > 1).then_some(0.0);
        assert_eq!(
            result_data,
            BenchmarkResultData::AsrThroughput {
                asr_prompt_tokens: 300,
                asr_audio_tokens: u32::try_from(audio_tokens)?,
                asr_prompt_ms: 120.0,
                asr_prompt_ms_stddev: stddev,
                asr_predicted_ms: 40.0,
                asr_predicted_ms_stddev: stddev,
                total_time_ms: Some(160.0),
                total_time_ms_stddev: stddev,
            }
        );
        Ok(())
    }

    /// A reading off the decode length the entry asked for, or an audio count
    /// that fills the prompt it is part of, is refused.
    #[rstest]
    #[case::stopped_early(reading(300, 63, 120.0, 40.0), 256, |e: &anyhow::Error| matches!(e.downcast_ref(), Some(VlThroughputMeasurementError::Mismatch { field: "decode_tokens", .. })))]
    #[case::audio_is_the_prompt(reading(300, 64, 120.0, 40.0), 300, |e: &anyhow::Error| matches!(e.downcast_ref(), Some(AsrThroughputMeasurementError::AudioTokensFillPrompt { .. })))]
    #[case::audio_exceeds_prompt(reading(300, 64, 120.0, 40.0), 301, |e: &anyhow::Error| matches!(e.downcast_ref(), Some(AsrThroughputMeasurementError::AudioTokensFillPrompt { .. })))]
    fn a_reading_off_the_entry_is_refused(
        #[case] reading: anyhow::Result<AsrThroughputMeasurementReading>,
        #[case] audio_tokens: i32,
        #[case] is_expected: fn(&anyhow::Error) -> bool,
    ) -> anyhow::Result<()> {
        let reading = reading?;
        let err = unobserved(&body()?, NonZeroUsize::try_from(5)?, audio_tokens, || {
            Ok(reading)
        })
        .err()
        .context("a refused reading was measured")?;
        assert!(is_expected(&err), "{err:#}");
        Ok(())
    }

    /// An audio count is kept only while it and the cell's text leave the
    /// template at least one token of the prompt.
    #[rstest]
    #[case::room_for_the_template(1024, 1300, 256, true)]
    #[case::one_token_of_template(1024, 1281, 256, true)]
    #[case::audio_and_text_are_the_prompt(1024, 1280, 256, false)]
    #[case::text_alone_is_the_prompt(1024, 1024, 1, false)]
    fn the_audio_and_the_text_are_within_the_prompt(
        #[case] text_tokens: i32,
        #[case] prompt_tokens: i32,
        #[case] audio_tokens: i32,
        #[case] kept: bool,
    ) -> anyhow::Result<()> {
        let reading = reading(prompt_tokens, 64, 120.0, 40.0)?;
        let got = unobserved(
            &body_with_text(text_tokens)?,
            NonZeroUsize::try_from(5)?,
            audio_tokens,
            || Ok(reading),
        );
        match got {
            Ok(_) => assert!(
                kept,
                "{text_tokens} + {audio_tokens} of {prompt_tokens} kept"
            ),
            Err(err) => assert!(
                !kept
                    && matches!(
                        err.downcast_ref::<AsrThroughputMeasurementError>(),
                        Some(AsrThroughputMeasurementError::AudioTokensFillPrompt { .. })
                    ),
                "{err:#}"
            ),
        }
        Ok(())
    }

    /// A generation of `decode_tokens` that ran to its limit; the client
    /// stamps span 63 ms, the server's own decode `decode_ms`.
    fn generation(decode_tokens: u32, decode_ms: f64) -> Generation {
        Generation {
            first_token_ms: 120.0,
            last_token_ms: 183.0,
            end_ms: 183.0,
            finish: Some(Finish::Limit),
            counts: Some(Counts {
                prompt_tokens: 300,
                completion_tokens: decode_tokens,
            }),
            server_clock: Some(ServerClock {
                prefill_ms: 100.0,
                decode_ms,
            }),
        }
    }

    /// Both clocks start the decode at the first token, so 64 tokens time 63
    /// steps: the client's 63 ms becomes 64 and the server's 126 ms becomes
    /// 128. The prefill is taken as given, and the whole request is the
    /// client's 183 ms on either clock, not the sum of the two.
    #[rstest]
    #[case::client(Clock::Client, 120.0, 64.0)]
    #[case::server(Clock::Server, 100.0, 128.0)]
    fn a_generation_is_scaled_to_every_decode_step(
        #[case] clock: Clock,
        #[case] prompt_ms: f64,
        #[case] predicted_ms: f64,
    ) -> anyhow::Result<()> {
        let got = AsrThroughputMeasurementReading::from_generation(&generation(64, 126.0), clock)?;
        assert_eq!(got, timed_reading(300, 64, prompt_ms, predicted_ms, 183.0)?);
        Ok(())
    }

    /// A generation that reports a single token has no decode step to time,
    /// whatever the cell asked for; two is the least that can be scaled.
    #[rstest]
    #[case::one(1, false)]
    #[case::two(2, true)]
    fn a_reported_single_token_decode_is_refused(
        #[case] tokens: u32,
        #[case] accepted: bool,
    ) -> anyhow::Result<()> {
        let got = AsrThroughputMeasurementReading::from_generation(
            &generation(tokens, 1.0),
            Clock::Server,
        );
        match (got, accepted) {
            (Ok(_), true) => Ok(()),
            (Err(err), false) => {
                anyhow::ensure!(
                    matches!(
                        err.downcast_ref::<VlThroughputMeasurementError>(),
                        Some(VlThroughputMeasurementError::TooFewTokens { tokens: got })
                            if got.into_inner().unsigned_abs() == tokens
                    ),
                    "expected too few tokens, got {err:#}"
                );
                Ok(())
            }
            (got, accepted) => anyhow::bail!("accepted={accepted}, got {got:?}"),
        }
    }

    /// The settle step sees the prompt the reps ran, once, after them.
    #[test]
    fn the_audio_count_is_settled_once_against_the_prompt() -> anyhow::Result<()> {
        let attempts = std::cell::Cell::new(0);
        let seen = std::cell::RefCell::new(Vec::new());
        asr_throughput(
            &body()?,
            NonZeroUsize::try_from(3)?,
            &|| Ok(()),
            no_observer(),
            MeasurementWarmup::Inline,
            (),
            |()| {
                attempts.set(attempts.get() + 1);
                Ok(reading(300, 64, 120.0, 40.0)?)
            },
            |prompt_tokens| {
                seen.borrow_mut()
                    .push((attempts.get(), prompt_tokens.into_inner()));
                Ok(PositiveCount::try_new(256)?)
            },
        )?;
        assert_eq!(*seen.borrow(), vec![(3, 300)]);
        Ok(())
    }

    /// A failed settle step (a probe that could not run) is the cell's error,
    /// named by the marker.
    #[test]
    fn a_failed_audio_count_fails_the_cell() -> anyhow::Result<()> {
        let err = asr_throughput(
            &body()?,
            NonZeroUsize::try_from(2)?,
            &|| Ok(()),
            no_observer(),
            MeasurementWarmup::Inline,
            (),
            |()| Ok(reading(300, 64, 120.0, 40.0)?),
            |_| anyhow::bail!("the text-only probe failed"),
        )
        .err()
        .context("a cell with no audio count was measured")?;
        let text = format!("{err:#}");
        assert!(
            text.starts_with(&format!("{MARKER}: the audio token count")),
            "{text}"
        );
        assert!(text.contains("the text-only probe failed"), "{text}");
        Ok(())
    }
}
