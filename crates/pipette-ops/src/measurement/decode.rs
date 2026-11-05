//! Decode throughput: the time to generate the requested tokens after the
//! prompt, reduced over the reps.

use std::num::NonZeroUsize;

use anyhow::Context;

use pipette_plan_types::{
    BatchSize, BenchmarkResultData, DecodeThroughputBenchmark, PositiveCount, PositiveMillis,
    PositiveMillisError,
};

use super::{AttemptError, MeasurementObserver, MeasurementWarmup};
use crate::generation::Generation;
use crate::readiness::ReadinessGate;

/// A batch or a reading a decode cell cannot use.
#[derive(Debug, thiserror::Error)]
pub enum DecodeMeasurementError {
    #[error("engine answered {actual} prompts, expected a batch of {expected}")]
    PromptCount { actual: usize, expected: BatchSize },

    #[error("engine returned {field} {actual}, expected {expected}")]
    Mismatch {
        field: &'static str,
        actual: u32,
        expected: u32,
    },

    #[error("engine returned {field} {actual} over {prompts} prompts, expected {each} each")]
    TotalsMismatch {
        field: &'static str,
        actual: PositiveCount,
        prompts: BatchSize,
        each: u32,
    },

    #[error("a decode rate needs at least 2 tokens, got {tokens}")]
    TooFewTokens { tokens: PositiveCount },

    #[error("invalid ttft_ms: {ttft_ms}")]
    InvalidTtft { ttft_ms: f64 },

    #[error("invalid decode interval: last token at {last_token_ms} ms, TTFT {ttft_ms} ms")]
    InvalidInterval { ttft_ms: f64, last_token_ms: f64 },

    #[error("invalid decode_time_ms: {ms}")]
    InvalidTime {
        ms: f64,
        #[source]
        source: PositiveMillisError,
    },
}

/// The shape every engine converts a batched reply to.
#[derive(Debug, Clone, PartialEq)]
pub enum DecodeMeasurementReading {
    PerPrompt(Vec<DecodeMeasurementPerPromptReading>),
    Totals(DecodeMeasurementTotalsReading),
}

/// Each count is checked against the parameter of the same name.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DecodeMeasurementPerPromptReading {
    /// Context tokens before the first decode step: none when llama-bench
    /// generates from an empty cache (`--n-depth 0`).
    pub prefill_tokens: u32,
    /// Tokens generated, the first one included.
    pub decode_tokens: PositiveCount,
    /// Time to generate all `decode_tokens`.
    pub decode_time_ms: PositiveMillis,
}

impl DecodeMeasurementPerPromptReading {
    /// A reading from a streamed generation that ran to its limit, timed from
    /// its first token to its last.
    pub fn from_generation(generation: &Generation) -> anyhow::Result<Self> {
        let (prefill_tokens, decode_tokens) = generation.counts_at_limit()?.positive()?;
        Ok(Self::from_stream(
            prefill_tokens,
            decode_tokens,
            generation.first_token_ms,
            generation.last_token_ms,
        )?)
    }

    /// A reading from a stream's client-clock stamps. The first token comes
    /// from the prefill, so the interval after it covers `decode_tokens - 1`
    /// steps; it is scaled to the full count, which is what llama-bench times.
    pub fn from_stream(
        prefill_tokens: PositiveCount,
        decode_tokens: PositiveCount,
        ttft_ms: f64,
        last_token_ms: f64,
    ) -> Result<Self, DecodeMeasurementError> {
        let tokens = decode_tokens.into_inner();
        if tokens < 2 {
            return Err(DecodeMeasurementError::TooFewTokens {
                tokens: decode_tokens,
            });
        }
        if !ttft_ms.is_finite() || ttft_ms <= 0.0 {
            return Err(DecodeMeasurementError::InvalidTtft { ttft_ms });
        }
        let interval_ms = last_token_ms - ttft_ms;
        if interval_ms <= 0.0 || !interval_ms.is_finite() {
            return Err(DecodeMeasurementError::InvalidInterval {
                ttft_ms,
                last_token_ms,
            });
        }
        Self::from_millis(
            prefill_tokens.into_inner().unsigned_abs(),
            decode_tokens,
            interval_ms / f64::from(tokens - 1) * f64::from(tokens),
        )
    }

    /// A reading from a decode the engine timed itself (a llama-bench
    /// `n_prompt = 0` row), over all `decode_tokens`.
    pub fn from_millis(
        prefill_tokens: u32,
        decode_tokens: PositiveCount,
        decode_time_ms: f64,
    ) -> Result<Self, DecodeMeasurementError> {
        Ok(Self {
            prefill_tokens,
            decode_tokens,
            decode_time_ms: millis(decode_time_ms)?,
        })
    }
}

/// Counts summed over the streams and one time they all share, which any batch
/// reply gives exactly. The prompts are identical and generation is capped, so
/// sums of `parameter_batch_size` times each parameter mean each stream ran
/// them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DecodeMeasurementTotalsReading {
    pub prefill_tokens: PositiveCount,
    pub decode_tokens: PositiveCount,
    pub decode_time_ms: PositiveMillis,
}

/// Measure a decode cell: `reps` attempts, each one engine call, checked
/// against the entry and reduced to a mean and stddev.
/// Each rep waits on `readiness_gate` first.
///
/// `ctx` is lent to every attempt, the warm-up's included, for what the engine
/// records while they run: the server it sends to, or the processes it ran.
/// Lend a reference to read it back afterwards, a failed run included.
///
/// An attempt runs the `parameter_batch_size` prompts together; a rep reads
/// their mean time, or the totals' shared one.
pub fn decode<C>(
    benchmark: &DecodeThroughputBenchmark,
    reps: NonZeroUsize,
    readiness_gate: ReadinessGate<'_>,
    observer: MeasurementObserver<'_>,
    warmup: MeasurementWarmup,
    mut ctx: C,
    mut attempt: impl FnMut(&mut C) -> Result<DecodeMeasurementReading, AttemptError>,
) -> anyhow::Result<BenchmarkResultData> {
    // Names the cell in every log line and error. TODO: add the job id.
    let marker = benchmark.benchmark_id.to_string();

    if warmup == MeasurementWarmup::Once {
        let label = format!("{marker} warm-up");
        warm_up(&label, &mut ctx, &mut attempt, benchmark).context(label)?;
    }
    let times = (1..=reps.get())
        .map(|rep| {
            let label = format!("{marker} rep {rep}/{reps}");
            run_rep(
                &label,
                readiness_gate,
                observer,
                &mut ctx,
                &mut attempt,
                benchmark,
            )
            .context(label)
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    let stats = super::mean_stddev(times.iter().copied());
    super::log_reduction(&marker, None, &times, stats);

    Ok(BenchmarkResultData::DecodeThroughput {
        decode_time_ms: stats.mean_ms,
        // One rep has no spread to report.
        decode_time_ms_stddev: (reps.get() > 1).then_some(stats.stddev_ms),
    })
}

/// Ungated and unobserved: the warm-up is not a rep.
fn warm_up<C>(
    label: &str,
    ctx: &mut C,
    attempt: &mut impl FnMut(&mut C) -> Result<DecodeMeasurementReading, AttemptError>,
    benchmark: &DecodeThroughputBenchmark,
) -> anyhow::Result<()> {
    tracing::info!(cell = label, "running");
    let reading = attempt(ctx).map_err(AttemptError::into_anyhow)?;
    sample(label, &reading, benchmark)?;
    Ok(())
}

fn run_rep<C>(
    label: &str,
    readiness_gate: ReadinessGate<'_>,
    observer: MeasurementObserver<'_>,
    ctx: &mut C,
    attempt: &mut impl FnMut(&mut C) -> Result<DecodeMeasurementReading, AttemptError>,
    benchmark: &DecodeThroughputBenchmark,
) -> anyhow::Result<f64> {
    readiness_gate()?;

    (observer.on_before_rep_start)()?;

    tracing::info!(cell = label, "running");
    let reading = attempt(ctx).map_err(AttemptError::into_anyhow)?;

    (observer.on_after_rep_end)()?;

    sample(label, &reading, benchmark)
}

/// Each shape is logged before its checks, so a rejected reading is in the log
/// too.
fn sample(
    label: &str,
    reading: &DecodeMeasurementReading,
    benchmark: &DecodeThroughputBenchmark,
) -> anyhow::Result<f64> {
    match reading {
        DecodeMeasurementReading::PerPrompt(readings) => per_prompt(label, readings, benchmark),
        DecodeMeasurementReading::Totals(reading) => totals(label, reading, benchmark),
    }
}

fn per_prompt(
    label: &str,
    readings: &[DecodeMeasurementPerPromptReading],
    benchmark: &DecodeThroughputBenchmark,
) -> anyhow::Result<f64> {
    let count = readings.len();
    let times = readings
        .iter()
        .enumerate()
        .map(|(index, reading)| {
            let ms = f64::from(reading.decode_time_ms.into_inner());
            tracing::info!(
                cell = label,
                prompt = index + 1,
                prompts = count,
                decode_ms = ms,
                prefill_tokens = reading.prefill_tokens,
                decode_tokens = reading.decode_tokens.into_inner(),
                "decode reading"
            );
            ms
        })
        .collect::<Vec<_>>();
    answered(count, benchmark)?;
    readings
        .iter()
        .enumerate()
        .try_for_each(|(index, reading)| {
            check(reading, benchmark).map_err(|err| in_prompt(err.into(), index, count))
        })?;

    let ms = super::per_request_mean(&times)?;
    if count > 1 {
        tracing::info!(
            cell = label,
            mean_ms = ms,
            prompts = count,
            "per-prompt mean"
        );
    }
    Ok(ms)
}

fn totals(
    label: &str,
    reading: &DecodeMeasurementTotalsReading,
    benchmark: &DecodeThroughputBenchmark,
) -> anyhow::Result<f64> {
    let ms = f64::from(reading.decode_time_ms.into_inner());
    let batch_size = benchmark.parameter_batch_size;
    tracing::info!(
        cell = label,
        decode_ms = ms,
        prefill_tokens = reading.prefill_tokens.into_inner(),
        decode_tokens = reading.decode_tokens.into_inner(),
        prompts = batch_size.as_u32(),
        "decode totals reading"
    );
    expect_totals(
        "prefill_tokens",
        reading.prefill_tokens,
        batch_size,
        benchmark
            .parameter_prefill_tokens
            .into_inner()
            .unsigned_abs(),
    )?;
    expect_totals(
        "decode_tokens",
        reading.decode_tokens,
        batch_size,
        benchmark
            .parameter_decode_tokens
            .into_inner()
            .unsigned_abs(),
    )?;
    Ok(ms)
}

fn answered(
    prompts: usize,
    benchmark: &DecodeThroughputBenchmark,
) -> Result<(), DecodeMeasurementError> {
    let expected = benchmark.parameter_batch_size;
    if u32::try_from(prompts).ok() == Some(expected.as_u32()) {
        Ok(())
    } else {
        Err(DecodeMeasurementError::PromptCount {
            actual: prompts,
            expected,
        })
    }
}

fn in_prompt(err: anyhow::Error, index: usize, count: usize) -> anyhow::Error {
    if count == 1 {
        err
    } else {
        err.context(format!("prompt {}/{count}", index + 1))
    }
}

/// The workload the benchmark asked for; the time is valid by its type.
fn check(
    reading: &DecodeMeasurementPerPromptReading,
    benchmark: &DecodeThroughputBenchmark,
) -> Result<(), DecodeMeasurementError> {
    expect(
        "prefill_tokens",
        reading.prefill_tokens,
        benchmark
            .parameter_prefill_tokens
            .into_inner()
            .unsigned_abs(),
    )?;
    expect(
        "decode_tokens",
        reading.decode_tokens.into_inner().unsigned_abs(),
        benchmark
            .parameter_decode_tokens
            .into_inner()
            .unsigned_abs(),
    )
}

fn expect(field: &'static str, actual: u32, expected: u32) -> Result<(), DecodeMeasurementError> {
    if actual == expected {
        Ok(())
    } else {
        Err(DecodeMeasurementError::Mismatch {
            field,
            actual,
            expected,
        })
    }
}

/// In `i128`, so the product cannot overflow.
fn expect_totals(
    field: &'static str,
    actual: PositiveCount,
    prompts: BatchSize,
    each: u32,
) -> Result<(), DecodeMeasurementError> {
    if i128::from(actual.into_inner()) == i128::from(prompts.as_u32()) * i128::from(each) {
        Ok(())
    } else {
        Err(DecodeMeasurementError::TotalsMismatch {
            field,
            actual,
            prompts,
            each,
        })
    }
}

/// Narrowed to the `f32` an attempt reports its time in.
#[expect(
    clippy::cast_possible_truncation,
    reason = "an f32 keeps ~7 significant digits, finer than a decode time's \
              noise, and an overflow becomes inf, which the type refuses"
)]
fn millis(ms: f64) -> Result<PositiveMillis, DecodeMeasurementError> {
    PositiveMillis::try_new(ms as f32)
        .map_err(|source| DecodeMeasurementError::InvalidTime { ms, source })
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use pipette_plan_types::{BenchmarkId, NonNegativeCount};

    use super::*;

    fn reading(
        prefill_tokens: u32,
        decode_tokens: i32,
        decode_time_ms: f64,
    ) -> anyhow::Result<DecodeMeasurementReading> {
        Ok(DecodeMeasurementReading::PerPrompt(vec![prompt(
            prefill_tokens,
            decode_tokens,
            decode_time_ms,
        )?]))
    }

    fn body(batch_size: u32) -> anyhow::Result<DecodeThroughputBenchmark> {
        Ok(DecodeThroughputBenchmark {
            benchmark_id: BenchmarkId::try_new("decode_throughput_512_10")?,
            parameter_prefill_tokens: NonNegativeCount::try_new(512)?,
            parameter_decode_tokens: PositiveCount::try_new(10)?,
            parameter_batch_size: BatchSize::try_new(batch_size)?,
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
        benchmark: &DecodeThroughputBenchmark,
        reps: NonZeroUsize,
        mut attempt: impl FnMut() -> anyhow::Result<DecodeMeasurementReading>,
    ) -> anyhow::Result<BenchmarkResultData> {
        decode(
            benchmark,
            reps,
            &|| Ok(()),
            no_observer(),
            MeasurementWarmup::Inline,
            (),
            |()| Ok(attempt()?),
        )
    }

    fn prompt(
        prefill_tokens: u32,
        decode_tokens: i32,
        decode_time_ms: f64,
    ) -> anyhow::Result<DecodeMeasurementPerPromptReading> {
        Ok(DecodeMeasurementPerPromptReading::from_millis(
            prefill_tokens,
            PositiveCount::try_new(decode_tokens)?,
            decode_time_ms,
        )?)
    }

    fn prompts(decode_tokens: &[i32]) -> anyhow::Result<DecodeMeasurementReading> {
        Ok(DecodeMeasurementReading::PerPrompt(
            decode_tokens
                .iter()
                .map(|&tokens| prompt(512, tokens, 40.0))
                .collect::<anyhow::Result<_>>()?,
        ))
    }

    fn totals(
        prefill_tokens: i32,
        decode_tokens: i32,
        decode_time_ms: f32,
    ) -> anyhow::Result<DecodeMeasurementReading> {
        Ok(DecodeMeasurementReading::Totals(
            DecodeMeasurementTotalsReading {
                prefill_tokens: PositiveCount::try_new(prefill_tokens)?,
                decode_tokens: PositiveCount::try_new(decode_tokens)?,
                decode_time_ms: PositiveMillis::try_new(decode_time_ms)?,
            },
        ))
    }

    fn mean_and_stddev(result_data: BenchmarkResultData) -> anyhow::Result<(f64, Option<f64>)> {
        let BenchmarkResultData::DecodeThroughput {
            decode_time_ms,
            decode_time_ms_stddev,
        } = result_data
        else {
            anyhow::bail!("not a decode result: {result_data:?}");
        };
        Ok((decode_time_ms, decode_time_ms_stddev))
    }

    fn from_stream(
        decode_tokens: i32,
        ttft_ms: f64,
        last_token_ms: f64,
    ) -> anyhow::Result<Result<DecodeMeasurementPerPromptReading, DecodeMeasurementError>> {
        Ok(DecodeMeasurementPerPromptReading::from_stream(
            PositiveCount::try_new(512)?,
            PositiveCount::try_new(decode_tokens)?,
            ttft_ms,
            last_token_ms,
        ))
    }

    /// A time the result could not divide by, or past `f32`, is no reading.
    #[rstest]
    #[case::zero(0.0)]
    #[case::negative(-1.0)]
    #[case::nan(f64::NAN)]
    #[case::infinite(f64::INFINITY)]
    #[case::past_f32(1e300)]
    fn a_time_without_a_rate_is_refused(#[case] ms: f64) -> anyhow::Result<()> {
        let got =
            DecodeMeasurementPerPromptReading::from_millis(512, PositiveCount::try_new(10)?, ms);
        assert!(
            matches!(&got, Err(DecodeMeasurementError::InvalidTime { .. })),
            "{got:?}"
        );
        Ok(())
    }

    /// A stream's 100 ms interval over the 9 steps after its first token is
    /// 100/9*10 over the full count.
    #[test]
    fn a_stream_becomes_the_decode_time() -> anyhow::Result<()> {
        let reading = from_stream(10, 100.0, 200.0)??;
        assert_eq!(reading.prefill_tokens, 512);
        assert_eq!(reading.decode_tokens.into_inner(), 10);
        let ms = f64::from(reading.decode_time_ms.into_inner());
        assert!((ms - 100.0 / 9.0 * 10.0).abs() < 1e-4, "{ms}");
        Ok(())
    }

    /// One token has no step to rate; no TTFT or a non-positive interval is no
    /// reading.
    #[rstest]
    #[case::single_token(1, 100.0, 100.0, |e: &DecodeMeasurementError| matches!(e, DecodeMeasurementError::TooFewTokens { .. }))]
    #[case::no_ttft(10, 0.0, 100.0, |e: &DecodeMeasurementError| matches!(e, DecodeMeasurementError::InvalidTtft { .. }))]
    #[case::nan_ttft(10, f64::NAN, 100.0, |e: &DecodeMeasurementError| matches!(e, DecodeMeasurementError::InvalidTtft { .. }))]
    #[case::past_f32(10, 1.0, 1e300, |e: &DecodeMeasurementError| matches!(e, DecodeMeasurementError::InvalidTime { .. }))]
    #[case::zero_interval(10, 200.0, 200.0, |e: &DecodeMeasurementError| matches!(e, DecodeMeasurementError::InvalidInterval { .. }))]
    #[case::negative_interval(10, 300.0, 200.0, |e: &DecodeMeasurementError| matches!(e, DecodeMeasurementError::InvalidInterval { .. }))]
    fn a_stream_without_a_rate_is_refused(
        #[case] decode_tokens: i32,
        #[case] ttft_ms: f64,
        #[case] last_token_ms: f64,
        #[case] is_expected: fn(&DecodeMeasurementError) -> bool,
    ) -> anyhow::Result<()> {
        let got = from_stream(decode_tokens, ttft_ms, last_token_ms)?;
        assert!(matches!(&got, Err(e) if is_expected(e)), "{got:?}");
        Ok(())
    }

    /// The engine's time is taken as given, per prompt or in totals, on the
    /// batch and workload the entry asked for; a batch's rejected stream is
    /// named.
    #[rstest]
    #[case::as_given(1, reading(512, 10, 40.0), Ok(40.0))]
    #[case::stopped_early(
        1,
        reading(512, 9, 40.0),
        Err("engine returned decode_tokens 9, expected 10")
    )]
    #[case::wrong_depth(
        1,
        reading(256, 10, 40.0),
        Err("engine returned prefill_tokens 256, expected 512")
    )]
    #[case::no_depth(
        1,
        reading(0, 10, 40.0),
        Err("engine returned prefill_tokens 0, expected 512")
    )]
    #[case::stopped_early_in_a_batch(3, prompts(&[10, 9, 10]), Err("prompt 2/3: engine returned decode_tokens 9, expected 10"))]
    #[case::no_stream(1, prompts(&[]), Err("engine answered 0 prompts, expected a batch of 1"))]
    #[case::extra_stream(1, prompts(&[10, 10]), Err("engine answered 2 prompts, expected a batch of 1"))]
    #[case::short_batch(4, prompts(&[10; 3]), Err("engine answered 3 prompts, expected a batch of 4"))]
    #[case::long_batch(4, prompts(&[10; 5]), Err("engine answered 5 prompts, expected a batch of 4"))]
    #[case::summed(4, totals(2048, 40, 40.0), Ok(40.0))]
    #[case::summed_serial(1, totals(512, 10, 40.0), Ok(40.0))]
    #[case::a_stream_stopped_early(
        4,
        totals(2048, 39, 40.0),
        Err("engine returned decode_tokens 39 over 4 prompts, expected 10 each")
    )]
    #[case::summed_other_depth(
        4,
        totals(1024, 40, 40.0),
        Err("engine returned prefill_tokens 1024 over 4 prompts, expected 512 each")
    )]
    #[case::first_stream_only(
        4,
        totals(512, 10, 40.0),
        Err("engine returned prefill_tokens 512 over 4 prompts, expected 512 each")
    )]
    fn a_reading_becomes_the_decode_time(
        #[case] batch_size: u32,
        #[case] reading: anyhow::Result<DecodeMeasurementReading>,
        #[case] expected: Result<f64, &str>,
    ) -> anyhow::Result<()> {
        let reading = reading?;
        match (
            unobserved(&body(batch_size)?, NonZeroUsize::try_from(5)?, || {
                Ok(reading.clone())
            }),
            expected,
        ) {
            (Ok(result_data), Ok(ms)) => {
                let (decode_time_ms, _) = mean_and_stddev(result_data)?;
                assert!((decode_time_ms - ms).abs() < 1e-9, "{decode_time_ms}");
            }
            (Err(err), Err(message)) => {
                assert_eq!(
                    format!("{err:#}"),
                    format!("decode_throughput_512_10 rep 1/5: {message}")
                );
                assert!(
                    err.downcast_ref::<DecodeMeasurementError>().is_some(),
                    "{err:#}"
                );
            }
            (got, expected) => anyhow::bail!("expected {expected:?}, got {got:?}"),
        }
        Ok(())
    }

    /// A benchmark that asks for no context (llama-bench `--n-depth 0`) takes a
    /// reading of none.
    #[test]
    fn a_decode_without_a_prompt_is_measured() -> anyhow::Result<()> {
        let benchmark = DecodeThroughputBenchmark {
            benchmark_id: BenchmarkId::try_new("decode_throughput_0_32")?,
            parameter_prefill_tokens: NonNegativeCount::try_new(0)?,
            parameter_decode_tokens: PositiveCount::try_new(32)?,
            parameter_batch_size: BatchSize::one(),
        };
        let result_data = unobserved(&benchmark, NonZeroUsize::try_from(5)?, || {
            reading(0, 32, 40.0)
        })?;
        let (decode_time_ms, _) = mean_and_stddev(result_data)?;
        assert!((decode_time_ms - 40.0).abs() < 1e-9, "{decode_time_ms}");
        Ok(())
    }

    /// Readings of 1, 2, … ms reduce to their mean and sample stddev; one rep
    /// has no spread to report.
    #[rstest]
    #[case::five(5, 3.0, Some(2.5_f64.sqrt()))]
    #[case::one(1, 1.0, None)]
    fn the_reps_reduce_to_mean_and_stddev(
        #[case] reps: usize,
        #[case] mean: f64,
        #[case] stddev: Option<f64>,
    ) -> anyhow::Result<()> {
        let mut ms = 0.0;
        let result_data = unobserved(&body(1)?, NonZeroUsize::try_from(reps)?, || {
            ms += 1.0;
            reading(512, 10, ms)
        })?;
        let (decode_time_ms, decode_time_ms_stddev) = mean_and_stddev(result_data)?;
        assert!((decode_time_ms - mean).abs() < 1e-9, "{decode_time_ms}");
        match (decode_time_ms_stddev, stddev) {
            (Some(got), Some(want)) => assert!((got - want).abs() < 1e-9, "{got}"),
            (None, None) => {}
            (got, want) => anyhow::bail!("stddev {got:?}, expected {want:?}"),
        }
        Ok(())
    }

    /// A `Once` warm-up is one more attempt, a whole batch of it, and its
    /// readings are not in the result; an `Inline` one adds none.
    #[rstest]
    #[case::once(MeasurementWarmup::Once, 1, &[1000.0, 40.0, 40.0])]
    #[case::inline(MeasurementWarmup::Inline, 1, &[40.0, 40.0])]
    #[case::once_batched(MeasurementWarmup::Once, 3, &[1000.0, 40.0, 40.0])]
    fn only_the_reps_are_reduced(
        #[case] warmup: MeasurementWarmup,
        #[case] batch_size: u32,
        #[case] times: &[f64],
    ) -> anyhow::Result<()> {
        let mut times = times.iter();
        let prompts = usize::try_from(batch_size)?;
        let result_data = decode(
            &body(batch_size)?,
            NonZeroUsize::try_from(2)?,
            &|| Ok(()),
            no_observer(),
            warmup,
            (),
            |()| {
                let ms = *times.next().context("attempted past the reps")?;
                let answer = prompt(512, 10, ms)?;
                Ok(DecodeMeasurementReading::PerPrompt(vec![answer; prompts]))
            },
        )?;
        assert_eq!(times.next(), None, "an attempt was not made");
        let (decode_time_ms, _) = mean_and_stddev(result_data)?;
        assert!((decode_time_ms - 40.0).abs() < 1e-9, "{decode_time_ms}");
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
        let got = decode(
            &body(1)?,
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
                let decode_tokens = if failing == Some(refused) { 9 } else { 10 };
                Ok(reading(512, decode_tokens, 40.0)?)
            },
        );
        if let (Err(err), Some(step)) = (&got, failing) {
            let prefix = if step.starts_with("warm-up") {
                "decode_throughput_512_10 warm-up: "
            } else {
                "decode_throughput_512_10 rep 1/2: "
            };
            assert!(format!("{err:#}").starts_with(prefix), "{err:#}");
        }
        match (got, failing) {
            (Ok(_), None) => {}
            (Err(err), Some("check" | "warm-up check")) => assert!(
                matches!(
                    err.downcast_ref::<DecodeMeasurementError>(),
                    Some(DecodeMeasurementError::Mismatch {
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
        let got = decode(
            &body(1)?,
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
                Ok(reading(512, 10, 40.0)?)
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

    /// Rep `r` reads `10r + 0..4`, so it averages to `10r + 1.5`; the stddev
    /// is across reps only.
    #[test]
    fn a_batch_reduces_to_its_per_request_mean() -> anyhow::Result<()> {
        let mut rep = 0.0;
        let result_data = unobserved(&body(4)?, NonZeroUsize::try_from(5)?, || {
            rep += 10.0;
            Ok(DecodeMeasurementReading::PerPrompt(
                [0.0, 1.0, 2.0, 3.0]
                    .into_iter()
                    .map(|ms| prompt(512, 10, rep + ms))
                    .collect::<anyhow::Result<_>>()?,
            ))
        })?;
        let (decode_time_ms, decode_time_ms_stddev) = mean_and_stddev(result_data)?;
        assert!((decode_time_ms - 31.5).abs() < 1e-9, "{decode_time_ms}");
        let stddev = decode_time_ms_stddev.context("five reps have a spread")?;
        assert!((stddev - 250.0_f64.sqrt()).abs() < 1e-9, "{stddev}");
        Ok(())
    }
}
