//! VL throughput: the times to prefill an image-plus-text prompt, to decode
//! after it and to answer the whole request, reduced over the reps
//! (`docs/methodology/vl-throughput.md`).

use std::num::NonZeroUsize;

use anyhow::Context;

use pipette_plan_types::{
    BenchmarkResultData, PositiveCount, PositiveMillis, PositiveMillisError, VlThroughputBenchmark,
};

use super::{AttemptError, MeasurementObserver, MeasurementWarmup, Stats};
use crate::generation::{Clock, Generation};
use crate::readiness::ReadinessGate;

/// A reading a VL or ASR throughput cell cannot use.
#[derive(Debug, thiserror::Error)]
pub enum VlThroughputMeasurementError {
    #[error("engine returned {field} {actual}, expected {expected}")]
    Mismatch {
        field: &'static str,
        actual: PositiveCount,
        expected: u32,
    },

    #[error(
        "engine returned prompt_tokens {actual}, {first} on the first attempt: prompt caching \
         must be off"
    )]
    PromptVaried {
        first: PositiveCount,
        actual: PositiveCount,
    },

    #[error(
        "engine returned image_tokens {}, {} on the first attempt",
        or_none(*.actual),
        or_none(*.first)
    )]
    ImageTokensVaried {
        first: Option<PositiveCount>,
        actual: Option<PositiveCount>,
    },

    #[error("engine returned image_tokens {image_tokens} of prompt_tokens {prompt_tokens}")]
    ImageTokensExceedPrompt {
        image_tokens: PositiveCount,
        prompt_tokens: PositiveCount,
    },

    #[error("a decode time needs at least 2 tokens, got {tokens}")]
    TooFewTokens { tokens: PositiveCount },

    #[error("invalid {field}: {ms}")]
    InvalidTime {
        field: &'static str,
        ms: f64,
        #[source]
        source: PositiveMillisError,
    },
}

/// What one attempt ran, as the engine reports it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VlThroughputMeasurementReading {
    /// The whole prompt: image, text and template tokens. The same on every rep.
    pub prompt_tokens: PositiveCount,
    /// The image-only part of `prompt_tokens`, where the engine can tell
    /// (`docs/methodology/vl-image-tokens.md`). The same on every rep, so an
    /// engine that cannot vouch for it (a stale calibration row, a failed probe)
    /// reports `None` on each.
    pub image_tokens: Option<PositiveCount>,
    /// Tokens generated, checked against `parameter_decode_tokens`.
    pub decode_tokens: PositiveCount,
    /// Time to process all `prompt_tokens`, the image encode included.
    pub prompt_ms: PositiveMillis,
    /// Time to generate all `decode_tokens`.
    pub predicted_ms: PositiveMillis,
    /// The whole request on the client's clock, measured rather than
    /// `prompt_ms + predicted_ms`
    /// (`docs/methodology/vl-throughput.md#end-to-end-latency`).
    pub total_time_ms: PositiveMillis,
}

impl VlThroughputMeasurementReading {
    /// A reading from a generation that ran to its limit, its prefill and
    /// decode timed on `clock`. Either clock starts the decode at the first
    /// token, which comes from the prefill, so it covers `decode_tokens - 1`
    /// steps; it is scaled to all of them, which the in-process clients time.
    /// The whole request is the client's, from the send to the generation's
    /// end, on either clock.
    pub fn from_generation(
        generation: &Generation,
        image_tokens: Option<PositiveCount>,
        clock: Clock,
    ) -> anyhow::Result<Self> {
        let (prompt_ms, after_first_ms) = generation.phases_ms(clock)?;
        let (prompt_tokens, decode_tokens) = generation.counts_at_limit()?.positive()?;
        require_decode_steps(decode_tokens)?;
        let tokens = f64::from(decode_tokens.into_inner());
        Ok(Self::from_millis(
            prompt_tokens,
            image_tokens,
            decode_tokens,
            prompt_ms,
            after_first_ms / (tokens - 1.0) * tokens,
            generation.end_ms,
        )?)
    }

    /// From a decode time already covering all `decode_tokens` steps.
    pub fn from_millis(
        prompt_tokens: PositiveCount,
        image_tokens: Option<PositiveCount>,
        decode_tokens: PositiveCount,
        prompt_ms: f64,
        predicted_ms: f64,
        total_time_ms: f64,
    ) -> Result<Self, VlThroughputMeasurementError> {
        Ok(Self {
            prompt_tokens,
            image_tokens,
            decode_tokens,
            prompt_ms: millis("prompt_ms", prompt_ms)?,
            predicted_ms: millis("predicted_ms", predicted_ms)?,
            total_time_ms: millis("total_time_ms", total_time_ms)?,
        })
    }
}

/// Refuses fewer than 2 decode tokens: the first is the prefill's, so a
/// decode time needs a second. An engine checks the cell's count before its
/// model loads, and each reading checks the count its generation reports.
pub fn require_decode_steps(tokens: PositiveCount) -> Result<(), VlThroughputMeasurementError> {
    if tokens.into_inner() < 2 {
        return Err(VlThroughputMeasurementError::TooFewTokens { tokens });
    }
    Ok(())
}

/// Measure a VL throughput cell: `reps` attempts, each one engine call, checked
/// against the entry and the first attempt's prompt, and each time reduced to
/// a mean and stddev.
/// Each rep waits on `readiness_gate` first.
///
/// `ctx` is lent to every attempt, the warm-up's included, for what the engine
/// records while they run: the server it sends to, or the processes it ran.
/// Lend a reference to read it back afterwards, a failed run included.
pub fn vl_throughput<C>(
    benchmark: &VlThroughputBenchmark,
    reps: NonZeroUsize,
    readiness_gate: ReadinessGate<'_>,
    observer: MeasurementObserver<'_>,
    warmup: MeasurementWarmup,
    ctx: C,
    attempt: impl FnMut(&mut C) -> Result<VlThroughputMeasurementReading, AttemptError>,
) -> anyhow::Result<BenchmarkResultData> {
    // Names the cell in every log line and error. TODO: add the job id.
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
    // One rep has no spread to report.
    let stddev = |stats: Stats| (reps.get() > 1).then_some(stats.stddev_ms);

    Ok(BenchmarkResultData::VlThroughput {
        prompt_tokens: count(first.prompt_tokens),
        image_tokens: first.image_tokens.map(count),
        prompt_ms: prompt.mean_ms,
        prompt_ms_stddev: stddev(prompt),
        predicted_ms: predicted.mean_ms,
        predicted_ms_stddev: stddev(predicted),
        total_time_ms: Some(total.mean_ms),
        total_time_ms_stddev: stddev(total),
    })
}

/// What [`throughput`] reads off an attempt, VL's or ASR's.
pub(super) trait ThroughputReading: Copy {
    fn prompt_tokens(&self) -> PositiveCount;
    fn decode_tokens(&self) -> PositiveCount;
    /// `prompt_ms`, `predicted_ms` and `total_time_ms`.
    fn times(&self) -> [PositiveMillis; 3];
    /// Logged before any check, so a rejected reading is in the log too.
    fn log(&self, label: &str);
    /// Checks of the reading's own, after the decode length and the prompt.
    fn check(&self, first: Option<&Self>) -> Result<(), VlThroughputMeasurementError>;
}

impl ThroughputReading for VlThroughputMeasurementReading {
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
            image_tokens = self.image_tokens.map(PositiveCount::into_inner),
            decode_tokens = self.decode_tokens.into_inner(),
            "vl throughput reading"
        );
    }

    fn check(&self, first: Option<&Self>) -> Result<(), VlThroughputMeasurementError> {
        if let Some(first) = first
            && self.image_tokens != first.image_tokens
        {
            return Err(VlThroughputMeasurementError::ImageTokensVaried {
                first: first.image_tokens,
                actual: self.image_tokens,
            });
        }
        if let Some(image_tokens) = self.image_tokens.filter(|&t| t > self.prompt_tokens) {
            return Err(VlThroughputMeasurementError::ImageTokensExceedPrompt {
                image_tokens,
                prompt_tokens: self.prompt_tokens,
            });
        }
        Ok(())
    }
}

/// The reps every throughput cell runs: an optional warm-up, then `reps`
/// attempts, each checked against `decode_tokens` and the first attempt's
/// prompt. Returns the first rep's reading and the three times' reductions.
#[expect(
    clippy::too_many_arguments,
    reason = "the cell's marker and decode length, then the seams every timing entry takes"
)]
pub(super) fn throughput<R: ThroughputReading, C>(
    marker: &str,
    decode_tokens: u32,
    reps: NonZeroUsize,
    readiness_gate: ReadinessGate<'_>,
    observer: MeasurementObserver<'_>,
    warmup: MeasurementWarmup,
    mut ctx: C,
    mut attempt: impl FnMut(&mut C) -> Result<R, AttemptError>,
) -> anyhow::Result<(R, [Stats; 3])> {
    // The prompt every rep must repeat is the first the server saw: the
    // warm-up's where there is one, so a cache it primed shows from rep 1.
    let warm = match warmup {
        MeasurementWarmup::Once => {
            let label = format!("{marker} warm-up");
            Some(warm_up(&label, &mut ctx, &mut attempt, decode_tokens).context(label)?)
        }
        MeasurementWarmup::Inline => None,
    };
    let mut measure_rep = |rep: usize, first: Option<&R>| {
        let label = format!("{marker} rep {rep}/{reps}");
        run_rep(
            &label,
            readiness_gate,
            observer,
            &mut ctx,
            &mut attempt,
            decode_tokens,
            first,
        )
        .context(label)
    };
    let first = measure_rep(1, warm.as_ref())?;
    let reference = warm.unwrap_or(first);
    let readings = std::iter::once(Ok(first))
        .chain((2..=reps.get()).map(|rep| measure_rep(rep, Some(&reference))))
        .collect::<anyhow::Result<Vec<_>>>()?;
    let times = |pick: fn([PositiveMillis; 3]) -> PositiveMillis| {
        readings.iter().map(move |t| pick(t.times()))
    };
    Ok((
        first,
        [
            reduce(marker, "prompt_ms", times(|[t, _, _]| t)),
            reduce(marker, "predicted_ms", times(|[_, t, _]| t)),
            reduce(marker, "total_time_ms", times(|[_, _, t]| t)),
        ],
    ))
}

/// Ungated and unobserved: the warm-up is not a rep.
fn warm_up<R: ThroughputReading, C>(
    label: &str,
    ctx: &mut C,
    attempt: &mut impl FnMut(&mut C) -> Result<R, AttemptError>,
    decode_tokens: u32,
) -> anyhow::Result<R> {
    tracing::info!(cell = label, "running");
    let reading = attempt(ctx).map_err(AttemptError::into_anyhow)?;
    sample(label, &reading, decode_tokens, None)?;
    Ok(reading)
}

fn run_rep<R: ThroughputReading, C>(
    label: &str,
    readiness_gate: ReadinessGate<'_>,
    observer: MeasurementObserver<'_>,
    ctx: &mut C,
    attempt: &mut impl FnMut(&mut C) -> Result<R, AttemptError>,
    decode_tokens: u32,
    first: Option<&R>,
) -> anyhow::Result<R> {
    readiness_gate()?;

    (observer.on_before_rep_start)()?;

    tracing::info!(cell = label, "running");
    let reading = attempt(ctx).map_err(AttemptError::into_anyhow)?;

    (observer.on_after_rep_end)()?;

    sample(label, &reading, decode_tokens, first)?;
    Ok(reading)
}

fn sample<R: ThroughputReading>(
    label: &str,
    reading: &R,
    decode_tokens: u32,
    first: Option<&R>,
) -> anyhow::Result<()> {
    reading.log(label);
    check(reading, decode_tokens, first)?;

    Ok(())
}

/// The workload the benchmark asked for, on the prompt the first attempt ran;
/// the times are valid by their type.
fn check<R: ThroughputReading>(
    reading: &R,
    decode_tokens: u32,
    first: Option<&R>,
) -> Result<(), VlThroughputMeasurementError> {
    expect("decode_tokens", reading.decode_tokens(), decode_tokens)?;
    // Before the reading's own checks: a prompt cut short by a cache can fall
    // below the image count, and the cache is the cause to name.
    if let Some(first) = first
        && reading.prompt_tokens() != first.prompt_tokens()
    {
        return Err(VlThroughputMeasurementError::PromptVaried {
            first: first.prompt_tokens(),
            actual: reading.prompt_tokens(),
        });
    }
    reading.check(first)
}

/// Widened to `i64` so a signed count and an unsigned parameter compare.
fn expect(
    field: &'static str,
    actual: PositiveCount,
    expected: u32,
) -> Result<(), VlThroughputMeasurementError> {
    if i64::from(actual.into_inner()) == i64::from(expected) {
        Ok(())
    } else {
        Err(VlThroughputMeasurementError::Mismatch {
            field,
            actual,
            expected,
        })
    }
}

fn reduce(marker: &str, metric: &str, times: impl Iterator<Item = PositiveMillis>) -> Stats {
    let samples = times
        .map(|ms| f64::from(ms.into_inner()))
        .collect::<Vec<_>>();
    let stats = super::mean_stddev(samples.iter().copied());
    super::log_reduction(marker, Some(metric), &samples, stats);
    stats
}

/// A count on the wire. `PositiveCount` is never negative, so this is exact.
pub(super) fn count(tokens: PositiveCount) -> u32 {
    tokens.into_inner().unsigned_abs()
}

fn or_none(tokens: Option<PositiveCount>) -> String {
    tokens.map_or_else(|| "none".to_owned(), |tokens| tokens.to_string())
}

/// Narrowed to the `f32` an attempt reports its time in.
#[expect(
    clippy::cast_possible_truncation,
    reason = "an f32 keeps ~7 significant digits, finer than any of the three \
              times' noise, and an overflow becomes inf, which the type refuses"
)]
pub(super) fn millis(
    field: &'static str,
    ms: f64,
) -> Result<PositiveMillis, VlThroughputMeasurementError> {
    PositiveMillis::try_new(ms as f32).map_err(|source| VlThroughputMeasurementError::InvalidTime {
        field,
        ms,
        source,
    })
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use pipette_plan_types::{BenchmarkId, NonNegativeCount};

    use super::*;
    use crate::generation::{Counts, Finish, ServerClock};

    const MARKER: &str = "vl_throughput_512x512_text0_decode64_img1";

    /// A reading whose whole request took as long as its two phases.
    fn reading(
        prompt_tokens: i32,
        image_tokens: Option<i32>,
        decode_tokens: i32,
        prompt_ms: f64,
        predicted_ms: f64,
    ) -> anyhow::Result<VlThroughputMeasurementReading> {
        timed_reading(
            prompt_tokens,
            image_tokens,
            decode_tokens,
            prompt_ms,
            predicted_ms,
            prompt_ms + predicted_ms,
        )
    }

    fn timed_reading(
        prompt_tokens: i32,
        image_tokens: Option<i32>,
        decode_tokens: i32,
        prompt_ms: f64,
        predicted_ms: f64,
        total_time_ms: f64,
    ) -> anyhow::Result<VlThroughputMeasurementReading> {
        Ok(VlThroughputMeasurementReading::from_millis(
            PositiveCount::try_new(prompt_tokens)?,
            image_tokens.map(PositiveCount::try_new).transpose()?,
            PositiveCount::try_new(decode_tokens)?,
            prompt_ms,
            predicted_ms,
            total_time_ms,
        )?)
    }

    /// A generation of `decode_tokens` that ran to its limit; the client
    /// stamps span 63 ms and the reply ends 7 ms after the last token, the
    /// server's own decode `decode_ms`.
    fn generation(decode_tokens: u32, decode_ms: f64) -> Generation {
        Generation {
            first_token_ms: 120.0,
            last_token_ms: 183.0,
            end_ms: 190.0,
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

    fn body() -> anyhow::Result<VlThroughputBenchmark> {
        Ok(VlThroughputBenchmark {
            benchmark_id: BenchmarkId::try_new(MARKER)?,
            parameter_image_width: PositiveCount::try_new(512)?,
            parameter_image_height: PositiveCount::try_new(512)?,
            parameter_text_tokens: NonNegativeCount::try_new(0)?,
            parameter_decode_tokens: PositiveCount::try_new(64)?,
            parameter_num_images: NonNegativeCount::try_new(1)?,
        })
    }

    fn no_observer() -> MeasurementObserver<'static> {
        MeasurementObserver {
            on_before_rep_start: &|| Ok(()),
            on_after_rep_end: &|| Ok(()),
        }
    }

    /// The entry behind an open gate, with no observer and no warm-up.
    fn unobserved(
        benchmark: &VlThroughputBenchmark,
        reps: NonZeroUsize,
        mut attempt: impl FnMut() -> anyhow::Result<VlThroughputMeasurementReading>,
    ) -> anyhow::Result<BenchmarkResultData> {
        vl_throughput(
            benchmark,
            reps,
            &|| Ok(()),
            no_observer(),
            MeasurementWarmup::Inline,
            (),
            |()| Ok(attempt()?),
        )
    }

    /// Any time zero, non-finite or past `f32` is no reading, named by field.
    #[rstest]
    #[case::zero_prompt(0.0, 40.0, 160.0, "prompt_ms")]
    #[case::nan_prompt(f64::NAN, 40.0, 160.0, "prompt_ms")]
    #[case::negative_predicted(300.0, -1.0, 160.0, "predicted_ms")]
    #[case::infinite_predicted(300.0, f64::INFINITY, 160.0, "predicted_ms")]
    #[case::past_f32(1e300, 40.0, 160.0, "prompt_ms")]
    #[case::zero_total(300.0, 40.0, 0.0, "total_time_ms")]
    #[case::infinite_total(300.0, 40.0, f64::INFINITY, "total_time_ms")]
    fn a_time_without_a_rate_is_refused(
        #[case] prompt_ms: f64,
        #[case] predicted_ms: f64,
        #[case] total_time_ms: f64,
        #[case] field: &str,
    ) -> anyhow::Result<()> {
        let got = VlThroughputMeasurementReading::from_millis(
            PositiveCount::try_new(300)?,
            None,
            PositiveCount::try_new(64)?,
            prompt_ms,
            predicted_ms,
            total_time_ms,
        );
        assert!(
            matches!(&got, Err(VlThroughputMeasurementError::InvalidTime { field: f, .. }) if *f == field),
            "{got:?}"
        );
        Ok(())
    }

    /// A 64-token decode timed from its first token, on either clock, spans
    /// 63 steps: 63 ms becomes 64. The prefill is taken as given, and the
    /// whole request is the client's 190 ms to the reply's end on either
    /// clock: not the last token's 183 ms, nor the sum of the two.
    #[rstest]
    #[case::client(Clock::Client, 120.0, 64.0)]
    #[case::server(Clock::Server, 100.0, 128.0)]
    fn a_generation_is_scaled_to_every_decode_step(
        #[case] clock: Clock,
        #[case] prompt_ms: f64,
        #[case] predicted_ms: f64,
    ) -> anyhow::Result<()> {
        let got =
            VlThroughputMeasurementReading::from_generation(&generation(64, 126.0), None, clock)?;
        assert_eq!(
            got,
            timed_reading(300, None, 64, prompt_ms, predicted_ms, 190.0)?
        );
        Ok(())
    }

    /// One token has no decode step after the prefill's to time, whether the
    /// cell asks for it or a generation reports it.
    #[rstest]
    #[case::one(1, Err(()))]
    #[case::two(2, Ok(()))]
    fn a_single_token_decode_is_refused(
        #[case] tokens: i32,
        #[case] expected: Result<(), ()>,
    ) -> anyhow::Result<()> {
        let from_cell =
            require_decode_steps(PositiveCount::try_new(tokens)?).map_err(anyhow::Error::from);
        let from_reading = VlThroughputMeasurementReading::from_generation(
            &generation(tokens.unsigned_abs(), 1.0),
            None,
            Clock::Server,
        )
        .map(|_| ());
        [from_cell, from_reading]
            .into_iter()
            .try_for_each(|got| match (got, expected) {
                (Ok(()), Ok(())) => Ok(()),
                (Err(err), Err(())) => match err.downcast_ref::<VlThroughputMeasurementError>() {
                    Some(VlThroughputMeasurementError::TooFewTokens { tokens: got }) => {
                        anyhow::ensure!(
                            got.into_inner() == tokens,
                            "refused {got}, asked {tokens}"
                        );
                        Ok(())
                    }
                    _ => anyhow::bail!("expected too few tokens, got {err:#}"),
                },
                (got, expected) => anyhow::bail!("expected {expected:?}, got {got:?}"),
            })
    }

    /// The mean and the stddev are both scaled: decodes of 63 and 126 ms
    /// after the first token reduce as 64 and 128 would.
    #[test]
    fn the_mean_and_stddev_are_both_scaled() -> anyhow::Result<()> {
        let mut after_first = [63.0, 126.0].into_iter();
        let result_data = unobserved(&body()?, NonZeroUsize::try_from(2)?, || {
            VlThroughputMeasurementReading::from_generation(
                &generation(64, after_first.next().context("attempted past the reps")?),
                None,
                Clock::Server,
            )
        })?;
        let BenchmarkResultData::VlThroughput {
            predicted_ms,
            predicted_ms_stddev,
            ..
        } = result_data
        else {
            anyhow::bail!("not a VL throughput result: {result_data:?}");
        };
        assert!((predicted_ms - 96.0).abs() < 1e-9, "{predicted_ms}");
        let stddev = predicted_ms_stddev.context("two reps report a spread")?;
        assert!((stddev - 32.0 * 2.0_f64.sqrt()).abs() < 1e-9, "{stddev}");
        Ok(())
    }

    /// The engine's times and counts are taken as given.
    #[rstest]
    #[case::as_given(reading(300, Some(256), 64, 120.0, 40.0), Some(256))]
    #[case::image_unmeasured(reading(300, None, 64, 120.0, 40.0), None)]
    #[case::image_is_the_prompt(reading(300, Some(300), 64, 120.0, 40.0), Some(300))]
    fn a_reading_becomes_the_result(
        #[case] reading: anyhow::Result<VlThroughputMeasurementReading>,
        #[case] image_tokens: Option<u32>,
    ) -> anyhow::Result<()> {
        let reading = reading?;
        let result_data = unobserved(&body()?, NonZeroUsize::try_from(5)?, || Ok(reading))?;
        assert_eq!(
            result_data,
            BenchmarkResultData::VlThroughput {
                prompt_tokens: 300,
                image_tokens,
                prompt_ms: 120.0,
                prompt_ms_stddev: Some(0.0),
                predicted_ms: 40.0,
                predicted_ms_stddev: Some(0.0),
                total_time_ms: Some(160.0),
                total_time_ms_stddev: Some(0.0),
            }
        );
        Ok(())
    }

    /// A reading off the decode length the entry asked for, or with an image
    /// larger than its prompt, is refused.
    #[rstest]
    #[case::stopped_early(reading(300, Some(256), 63, 120.0, 40.0), |e: &VlThroughputMeasurementError| matches!(e, VlThroughputMeasurementError::Mismatch { field: "decode_tokens", .. }))]
    #[case::image_exceeds_prompt(reading(300, Some(301), 64, 120.0, 40.0), |e: &VlThroughputMeasurementError| matches!(e, VlThroughputMeasurementError::ImageTokensExceedPrompt { .. }))]
    fn a_reading_off_the_entry_is_refused(
        #[case] reading: anyhow::Result<VlThroughputMeasurementReading>,
        #[case] is_expected: fn(&VlThroughputMeasurementError) -> bool,
    ) -> anyhow::Result<()> {
        let reading = reading?;
        let err = unobserved(&body()?, NonZeroUsize::try_from(5)?, || Ok(reading))
            .err()
            .context("a refused reading was measured")?;
        assert!(
            err.downcast_ref::<VlThroughputMeasurementError>()
                .is_some_and(is_expected),
            "{err:#}"
        );
        Ok(())
    }

    /// Prefills of 1, 2, … ms, decodes of 2, 4, … ms and totals of 4, 8, … ms,
    /// past the two phases' sum, reduce to their means and sample stddevs; one
    /// rep has no spread to report.
    #[rstest]
    #[case::five(5, 3.0, Some(2.5_f64.sqrt()))]
    #[case::one(1, 1.0, None)]
    fn the_reps_reduce_to_mean_and_stddev(
        #[case] reps: usize,
        #[case] mean: f64,
        #[case] stddev: Option<f64>,
    ) -> anyhow::Result<()> {
        let mut ms = 0.0;
        let result_data = unobserved(&body()?, NonZeroUsize::try_from(reps)?, || {
            ms += 1.0;
            timed_reading(300, Some(256), 64, ms, 2.0 * ms, 4.0 * ms)
        })?;
        let BenchmarkResultData::VlThroughput {
            prompt_ms,
            prompt_ms_stddev,
            predicted_ms,
            predicted_ms_stddev,
            total_time_ms,
            total_time_ms_stddev,
            ..
        } = result_data
        else {
            anyhow::bail!("not a VL throughput result: {result_data:?}");
        };
        assert!((prompt_ms - mean).abs() < 1e-9, "{prompt_ms}");
        assert!((predicted_ms - 2.0 * mean).abs() < 1e-9, "{predicted_ms}");
        let total_time_ms = total_time_ms.context("every rep timed the request")?;
        assert!((total_time_ms - 4.0 * mean).abs() < 1e-9, "{total_time_ms}");
        match (
            prompt_ms_stddev,
            predicted_ms_stddev,
            total_time_ms_stddev,
            stddev,
        ) {
            (Some(prompt), Some(predicted), Some(total), Some(want)) => {
                assert!((prompt - want).abs() < 1e-9, "{prompt}");
                assert!((predicted - 2.0 * want).abs() < 1e-9, "{predicted}");
                assert!((total - 4.0 * want).abs() < 1e-9, "{total}");
            }
            (None, None, None, None) => {}
            (prompt, predicted, total, want) => {
                anyhow::bail!("stddev {prompt:?}, {predicted:?} and {total:?}, expected {want:?}")
            }
        }
        Ok(())
    }

    /// A rep whose prompt or image count differs from the first attempt's stops
    /// the cell at that rep. With a warm-up that attempt is the warm-up, so a
    /// prompt it left cached fails rep 1; without one, rep 2.
    #[rstest]
    #[case::cached_prompt(Some(256), reading(280, Some(256), 64, 120.0, 40.0), |e: &VlThroughputMeasurementError| matches!(e, VlThroughputMeasurementError::PromptVaried { .. }))]
    #[case::cached_image(Some(256), reading(10, Some(256), 64, 120.0, 40.0), |e: &VlThroughputMeasurementError| matches!(e, VlThroughputMeasurementError::PromptVaried { .. }))]
    #[case::image_count(Some(256), reading(300, Some(255), 64, 120.0, 40.0), |e: &VlThroughputMeasurementError| matches!(e, VlThroughputMeasurementError::ImageTokensVaried { .. }))]
    #[case::image_dropped(Some(256), reading(300, None, 64, 120.0, 40.0), |e: &VlThroughputMeasurementError| matches!(e, VlThroughputMeasurementError::ImageTokensVaried { .. }))]
    #[case::image_added(None, reading(300, Some(256), 64, 120.0, 40.0), |e: &VlThroughputMeasurementError| matches!(e, VlThroughputMeasurementError::ImageTokensVaried { .. }))]
    fn a_prompt_that_changes_after_the_first_attempt_is_refused(
        #[case] first_image_tokens: Option<i32>,
        #[case] second: anyhow::Result<VlThroughputMeasurementReading>,
        #[case] is_expected: fn(&VlThroughputMeasurementError) -> bool,
        #[values(MeasurementWarmup::Once, MeasurementWarmup::Inline)] warmup: MeasurementWarmup,
    ) -> anyhow::Result<()> {
        let readings = [reading(300, first_image_tokens, 64, 120.0, 40.0)?, second?];
        let mut attempts = readings.iter().copied();
        let err = vl_throughput(
            &body()?,
            NonZeroUsize::try_from(5)?,
            &|| Ok(()),
            no_observer(),
            warmup,
            (),
            |()| Ok(attempts.next().context("attempted past the refused rep")?),
        )
        .err()
        .context("a changed prompt was measured")?;
        let rep = match warmup {
            MeasurementWarmup::Once => 1,
            MeasurementWarmup::Inline => 2,
        };
        assert!(
            format!("{err:#}").starts_with(&format!("{MARKER} rep {rep}/5: ")),
            "{err:#}"
        );
        assert!(
            err.downcast_ref::<VlThroughputMeasurementError>()
                .is_some_and(is_expected),
            "{err:#}"
        );
        Ok(())
    }

    /// A `Once` warm-up is one more attempt, and its reading is not in the
    /// result; an `Inline` one adds none.
    #[rstest]
    #[case::once(MeasurementWarmup::Once, &[1000.0, 120.0, 120.0])]
    #[case::inline(MeasurementWarmup::Inline, &[120.0, 120.0])]
    fn only_the_reps_are_reduced(
        #[case] warmup: MeasurementWarmup,
        #[case] times: &[f64],
    ) -> anyhow::Result<()> {
        let mut times = times.iter();
        let result_data = vl_throughput(
            &body()?,
            NonZeroUsize::try_from(2)?,
            &|| Ok(()),
            no_observer(),
            warmup,
            (),
            |()| {
                let prompt_ms = *times.next().context("attempted past the reps")?;
                Ok(reading(300, Some(256), 64, prompt_ms, 40.0)?)
            },
        )?;
        assert_eq!(times.next(), None, "an attempt was not made");
        let BenchmarkResultData::VlThroughput { prompt_ms, .. } = result_data else {
            anyhow::bail!("not a VL throughput result: {result_data:?}");
        };
        assert!((prompt_ms - 120.0).abs() < 1e-9, "{prompt_ms}");
        Ok(())
    }

    /// A `Once` warm-up, checked, before the first gate, and none with
    /// `Inline`; then gate, start, attempt, end and the check per rep. A failing
    /// step abandons the cell there, and its error is the cell's, named by the
    /// marker and by the warm-up or the rep.
    #[rstest]
    #[case::none(MeasurementWarmup::Once, None, &["warm-up", "gate", "start", "attempt", "end", "gate", "start", "attempt", "end"])]
    #[case::inline(MeasurementWarmup::Inline, None, &["gate", "start", "attempt", "end", "gate", "start", "attempt", "end"])]
    #[case::warm_up(MeasurementWarmup::Once, Some("warm-up"), &["warm-up"])]
    #[case::warm_up_check(MeasurementWarmup::Once, Some("warm-up check"), &["warm-up"])]
    #[case::gate(MeasurementWarmup::Once, Some("gate"), &["warm-up", "gate"])]
    #[case::start(MeasurementWarmup::Once, Some("start"), &["warm-up", "gate", "start"])]
    #[case::attempt(MeasurementWarmup::Once, Some("attempt"), &["warm-up", "gate", "start", "attempt"])]
    #[case::end(MeasurementWarmup::Once, Some("end"), &["warm-up", "gate", "start", "attempt", "end"])]
    #[case::check(MeasurementWarmup::Once, Some("check"), &["warm-up", "gate", "start", "attempt", "end"])]
    fn the_steps_run_in_order_until_one_fails(
        #[case] warmup: MeasurementWarmup,
        #[case] failing: Option<&str>,
        #[case] expected: &[&str],
    ) -> anyhow::Result<()> {
        let events = std::cell::RefCell::new(Vec::new());
        let step = |event: &'static str| {
            events.borrow_mut().push(event);
            anyhow::ensure!(Some(event) != failing, "{event} failed");
            Ok(())
        };
        let observer = MeasurementObserver {
            on_before_rep_start: &|| step("start"),
            on_after_rep_end: &|| step("end"),
        };
        // Nothing marks the warm-up to the attempt; it is the first call.
        let mut warming = warmup == MeasurementWarmup::Once;
        let got = vl_throughput(
            &body()?,
            NonZeroUsize::try_from(2)?,
            &|| step("gate"),
            observer,
            warmup,
            (),
            |()| {
                let (event, refused) = if std::mem::take(&mut warming) {
                    ("warm-up", "warm-up check")
                } else {
                    ("attempt", "check")
                };
                step(event)?;
                let decode_tokens = if failing == Some(refused) { 63 } else { 64 };
                Ok(reading(300, Some(256), decode_tokens, 120.0, 40.0)?)
            },
        );
        if let (Err(err), Some(step)) = (&got, failing) {
            let prefix = if step.starts_with("warm-up") {
                format!("{MARKER} warm-up: ")
            } else {
                format!("{MARKER} rep 1/2: ")
            };
            assert!(format!("{err:#}").starts_with(&prefix), "{err:#}");
        }
        match (got, failing) {
            (Ok(_), None) => {}
            (Err(err), Some("check" | "warm-up check")) => assert!(
                matches!(
                    err.downcast_ref::<VlThroughputMeasurementError>(),
                    Some(VlThroughputMeasurementError::Mismatch {
                        field: "decode_tokens",
                        ..
                    })
                ),
                "{err:#}"
            ),
            (Err(err), Some(failing)) => assert!(
                format!("{err:#}").contains(&format!("{failing} failed")),
                "{err:#}"
            ),
            (got, failing) => anyhow::bail!("failing {failing:?}, got {got:?}"),
        }
        assert_eq!(*events.borrow(), expected);
        Ok(())
    }

    /// The context is lent to the warm-up and to each attempt in turn, and
    /// what they recorded is still the caller's after one fails, the warm-up
    /// or a rep; the attempt's error keeps its type.
    #[rstest]
    #[case::warm_up(1)]
    #[case::rep(3)]
    fn the_context_outlives_a_failed_attempt(#[case] fails_on: usize) -> anyhow::Result<()> {
        let mut recorded = Vec::new();
        let got = vl_throughput(
            &body()?,
            NonZeroUsize::try_from(5)?,
            &|| Ok(()),
            no_observer(),
            MeasurementWarmup::Once,
            &mut recorded,
            |recorded| {
                recorded.push(format!("attempt {}", recorded.len()));
                if recorded.len() == fails_on {
                    let cause = std::io::Error::other("the server went away");
                    return Err(anyhow::Error::from(cause).into());
                }
                Ok(reading(300, Some(256), 64, 120.0, 40.0)?)
            },
        );
        let err = got.err().context("an attempt should fail")?;
        assert!(
            format!("{err:#}").contains("the server went away"),
            "{err:#}"
        );
        assert!(
            err.chain().any(|cause| cause.is::<std::io::Error>()),
            "{err:#}"
        );
        assert_eq!(
            recorded,
            (0..fails_on)
                .map(|n| format!("attempt {n}"))
                .collect::<Vec<_>>()
        );
        Ok(())
    }
}
