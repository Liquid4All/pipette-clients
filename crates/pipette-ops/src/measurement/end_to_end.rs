//! End-to-end latency: the time a caller waits for one whole completion,
//! reduced over the reps.

use std::num::NonZeroUsize;

use anyhow::Context;

use pipette_plan_types::{
    BatchSize, BenchmarkResultData, EndToEndLatencyBenchmark, PositiveCount, PositiveMillis,
    PositiveMillisError,
};

use super::{AttemptError, MeasurementObserver, MeasurementWarmup};
use crate::generation::Generation;
use crate::readiness::ReadinessGate;

/// A batch or a reading an end-to-end cell cannot use.
#[derive(Debug, thiserror::Error)]
pub enum EndToEndMeasurementError {
    #[error("engine answered {actual} prompts, expected a batch of {expected}")]
    PromptCount { actual: usize, expected: BatchSize },

    #[error("engine returned {field} {actual}, expected {expected}")]
    Mismatch {
        field: &'static str,
        actual: PositiveCount,
        expected: u32,
    },

    #[error("engine returned {field} {actual} over {prompts} prompts, expected {each} each")]
    TotalsMismatch {
        field: &'static str,
        actual: PositiveCount,
        prompts: BatchSize,
        each: u32,
    },

    #[error("invalid total_time_ms: {ms}")]
    InvalidTime {
        ms: f64,
        #[source]
        source: PositiveMillisError,
    },
}

/// The shape every engine converts a batched reply to.
#[derive(Debug, Clone, PartialEq)]
pub enum EndToEndMeasurementReading {
    PerPrompt(Vec<EndToEndMeasurementPerPromptReading>),
    Totals(EndToEndMeasurementTotalsReading),
}

/// Each count is checked against the parameter of the same name.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EndToEndMeasurementPerPromptReading {
    /// Prompt tokens, as the engine's tokenizer counted the text.
    pub prefill_tokens: PositiveCount,
    pub decode_tokens: PositiveCount,
    /// The request's send to this prompt's whole answer.
    pub total_time_ms: PositiveMillis,
}

impl EndToEndMeasurementPerPromptReading {
    /// A reading from a generation that ran to its limit, timed from the send
    /// to its end.
    pub fn from_generation(generation: &Generation) -> anyhow::Result<Self> {
        let (prefill_tokens, decode_tokens) = generation.counts_at_limit()?.positive()?;
        Ok(Self::from_request(
            prefill_tokens,
            decode_tokens,
            generation.end_ms,
        )?)
    }

    /// On the client's clock, from the request's send to this prompt's answer.
    pub fn from_request(
        prefill_tokens: PositiveCount,
        decode_tokens: PositiveCount,
        elapsed_ms: f64,
    ) -> Result<Self, EndToEndMeasurementError> {
        Ok(Self {
            prefill_tokens,
            decode_tokens,
            total_time_ms: millis(elapsed_ms)?,
        })
    }
}

/// Counts summed over the prompts and the request's time, which answered them
/// all. The prompts are identical and generation is capped, so sums of
/// `parameter_batch_size` times each parameter mean each prompt ran them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EndToEndMeasurementTotalsReading {
    pub prefill_tokens: PositiveCount,
    pub decode_tokens: PositiveCount,
    pub total_time_ms: PositiveMillis,
}

/// Measure an end-to-end cell: `reps` attempts, each one engine call, checked
/// against the entry and reduced to a mean and stddev.
/// Each rep waits on `readiness_gate` first.
///
/// `ctx` is lent to every attempt, the warm-up's included, for what the engine
/// records while they run: the server it sends to, or the processes it ran.
/// Lend a reference to read it back afterwards, a failed run included.
///
/// An attempt runs the `parameter_batch_size` prompts together; a rep reads
/// their mean time, or the totals' shared one.
pub fn end_to_end<C>(
    benchmark: &EndToEndLatencyBenchmark,
    reps: NonZeroUsize,
    readiness_gate: ReadinessGate<'_>,
    observer: MeasurementObserver<'_>,
    warmup: MeasurementWarmup,
    mut ctx: C,
    mut attempt: impl FnMut(&mut C) -> Result<EndToEndMeasurementReading, AttemptError>,
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

    Ok(BenchmarkResultData::EndToEndLatency {
        total_time_ms: stats.mean_ms,
        // One rep has no spread to report.
        total_time_ms_stddev: (reps.get() > 1).then_some(stats.stddev_ms),
    })
}

/// Ungated and unobserved: the warm-up is not a rep.
fn warm_up<C>(
    label: &str,
    ctx: &mut C,
    attempt: &mut impl FnMut(&mut C) -> Result<EndToEndMeasurementReading, AttemptError>,
    benchmark: &EndToEndLatencyBenchmark,
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
    attempt: &mut impl FnMut(&mut C) -> Result<EndToEndMeasurementReading, AttemptError>,
    benchmark: &EndToEndLatencyBenchmark,
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
    reading: &EndToEndMeasurementReading,
    benchmark: &EndToEndLatencyBenchmark,
) -> anyhow::Result<f64> {
    match reading {
        EndToEndMeasurementReading::PerPrompt(readings) => per_prompt(label, readings, benchmark),
        EndToEndMeasurementReading::Totals(reading) => totals(label, reading, benchmark),
    }
}

fn per_prompt(
    label: &str,
    readings: &[EndToEndMeasurementPerPromptReading],
    benchmark: &EndToEndLatencyBenchmark,
) -> anyhow::Result<f64> {
    let count = readings.len();
    let times = readings
        .iter()
        .enumerate()
        .map(|(index, reading)| {
            let ms = f64::from(reading.total_time_ms.into_inner());
            tracing::info!(
                cell = label,
                prompt = index + 1,
                prompts = count,
                total_ms = ms,
                prefill_tokens = reading.prefill_tokens.into_inner(),
                decode_tokens = reading.decode_tokens.into_inner(),
                "end-to-end reading"
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
    reading: &EndToEndMeasurementTotalsReading,
    benchmark: &EndToEndLatencyBenchmark,
) -> anyhow::Result<f64> {
    let ms = f64::from(reading.total_time_ms.into_inner());
    let batch_size = benchmark.parameter_batch_size;
    tracing::info!(
        cell = label,
        total_ms = ms,
        prefill_tokens = reading.prefill_tokens.into_inner(),
        decode_tokens = reading.decode_tokens.into_inner(),
        prompts = batch_size.as_u32(),
        "end-to-end totals reading"
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
    benchmark: &EndToEndLatencyBenchmark,
) -> Result<(), EndToEndMeasurementError> {
    let expected = benchmark.parameter_batch_size;
    if u32::try_from(prompts).ok() == Some(expected.as_u32()) {
        Ok(())
    } else {
        Err(EndToEndMeasurementError::PromptCount {
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
    reading: &EndToEndMeasurementPerPromptReading,
    benchmark: &EndToEndLatencyBenchmark,
) -> Result<(), EndToEndMeasurementError> {
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
        reading.decode_tokens,
        benchmark
            .parameter_decode_tokens
            .into_inner()
            .unsigned_abs(),
    )
}

/// Widened to `i64` so a signed count and an unsigned parameter compare.
fn expect(
    field: &'static str,
    actual: PositiveCount,
    expected: u32,
) -> Result<(), EndToEndMeasurementError> {
    if i64::from(actual.into_inner()) == i64::from(expected) {
        Ok(())
    } else {
        Err(EndToEndMeasurementError::Mismatch {
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
) -> Result<(), EndToEndMeasurementError> {
    if i128::from(actual.into_inner()) == i128::from(prompts.as_u32()) * i128::from(each) {
        Ok(())
    } else {
        Err(EndToEndMeasurementError::TotalsMismatch {
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
    reason = "an f32 keeps ~7 significant digits, finer than a request's \
              noise, and an overflow becomes inf, which the type refuses"
)]
fn millis(ms: f64) -> Result<PositiveMillis, EndToEndMeasurementError> {
    PositiveMillis::try_new(ms as f32)
        .map_err(|source| EndToEndMeasurementError::InvalidTime { ms, source })
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use pipette_plan_types::{BenchmarkId, NonNegativeCount};

    use super::*;

    fn reading(
        prefill_tokens: i32,
        decode_tokens: i32,
        total_time_ms: f64,
    ) -> anyhow::Result<EndToEndMeasurementReading> {
        Ok(EndToEndMeasurementReading::PerPrompt(vec![prompt(
            prefill_tokens,
            decode_tokens,
            total_time_ms,
        )?]))
    }

    fn prompt(
        prefill_tokens: i32,
        decode_tokens: i32,
        total_time_ms: f64,
    ) -> anyhow::Result<EndToEndMeasurementPerPromptReading> {
        Ok(EndToEndMeasurementPerPromptReading::from_request(
            PositiveCount::try_new(prefill_tokens)?,
            PositiveCount::try_new(decode_tokens)?,
            total_time_ms,
        )?)
    }

    fn prompts(decode_tokens: &[i32]) -> anyhow::Result<EndToEndMeasurementReading> {
        Ok(EndToEndMeasurementReading::PerPrompt(
            decode_tokens
                .iter()
                .map(|&tokens| prompt(512, tokens, 900.0))
                .collect::<anyhow::Result<_>>()?,
        ))
    }

    fn totals(
        prefill_tokens: i32,
        decode_tokens: i32,
        total_time_ms: f32,
    ) -> anyhow::Result<EndToEndMeasurementReading> {
        Ok(EndToEndMeasurementReading::Totals(
            EndToEndMeasurementTotalsReading {
                prefill_tokens: PositiveCount::try_new(prefill_tokens)?,
                decode_tokens: PositiveCount::try_new(decode_tokens)?,
                total_time_ms: PositiveMillis::try_new(total_time_ms)?,
            },
        ))
    }

    fn mean_and_stddev(result_data: BenchmarkResultData) -> anyhow::Result<(f64, Option<f64>)> {
        let BenchmarkResultData::EndToEndLatency {
            total_time_ms,
            total_time_ms_stddev,
        } = result_data
        else {
            anyhow::bail!("not an end-to-end result: {result_data:?}");
        };
        Ok((total_time_ms, total_time_ms_stddev))
    }

    fn body(batch_size: u32) -> anyhow::Result<EndToEndLatencyBenchmark> {
        Ok(EndToEndLatencyBenchmark {
            benchmark_id: BenchmarkId::try_new("end_to_end_latency_512_256")?,
            parameter_prefill_tokens: NonNegativeCount::try_new(512)?,
            parameter_decode_tokens: PositiveCount::try_new(256)?,
            parameter_batch_size: BatchSize::try_new(batch_size)?,
        })
    }

    fn no_observer() -> MeasurementObserver<'static> {
        MeasurementObserver::QUIET
    }

    /// The entry behind an open gate, with no observer and no warm-up.
    fn unobserved(
        benchmark: &EndToEndLatencyBenchmark,
        reps: NonZeroUsize,
        mut attempt: impl FnMut() -> anyhow::Result<EndToEndMeasurementReading>,
    ) -> anyhow::Result<BenchmarkResultData> {
        end_to_end(
            benchmark,
            reps,
            &|| Ok(()),
            no_observer(),
            MeasurementWarmup::Inline,
            (),
            |()| Ok(attempt()?),
        )
    }

    /// A time the result could not use, or past `f32`, is no reading.
    #[rstest]
    #[case::zero(0.0)]
    #[case::negative(-1.0)]
    #[case::nan(f64::NAN)]
    #[case::infinite(f64::INFINITY)]
    #[case::past_f32(1e300)]
    fn a_request_without_a_valid_time_is_refused(#[case] ms: f64) -> anyhow::Result<()> {
        let got = EndToEndMeasurementPerPromptReading::from_request(
            PositiveCount::try_new(512)?,
            PositiveCount::try_new(256)?,
            ms,
        );
        assert!(
            matches!(&got, Err(EndToEndMeasurementError::InvalidTime { .. })),
            "{got:?}"
        );
        Ok(())
    }

    /// The engine's time is taken as given, per prompt or in totals, on the
    /// batch and workload the entry asked for; a batch's rejected prompt is
    /// named.
    #[rstest]
    #[case::as_given(1, reading(512, 256, 900.0), Ok(900.0))]
    #[case::stopped_early(
        1,
        reading(512, 255, 900.0),
        Err("engine returned decode_tokens 255, expected 256")
    )]
    #[case::wrong_prompt(
        1,
        reading(511, 256, 900.0),
        Err("engine returned prefill_tokens 511, expected 512")
    )]
    #[case::stopped_early_in_a_batch(3, prompts(&[256, 255, 256]), Err("prompt 2/3: engine returned decode_tokens 255, expected 256"))]
    #[case::no_prompt(1, prompts(&[]), Err("engine answered 0 prompts, expected a batch of 1"))]
    #[case::extra_prompt(1, prompts(&[256, 256]), Err("engine answered 2 prompts, expected a batch of 1"))]
    #[case::short_batch(4, prompts(&[256; 3]), Err("engine answered 3 prompts, expected a batch of 4"))]
    #[case::long_batch(4, prompts(&[256; 5]), Err("engine answered 5 prompts, expected a batch of 4"))]
    #[case::summed(4, totals(2048, 1024, 900.0), Ok(900.0))]
    #[case::summed_serial(1, totals(512, 256, 900.0), Ok(900.0))]
    #[case::a_prompt_stopped_early(
        4,
        totals(2048, 1023, 900.0),
        Err("engine returned decode_tokens 1023 over 4 prompts, expected 256 each")
    )]
    #[case::summed_other_prompts(
        4,
        totals(2044, 1024, 900.0),
        Err("engine returned prefill_tokens 2044 over 4 prompts, expected 512 each")
    )]
    // llama-server's `/v1/completions` reports the first prompt's usage only.
    #[case::first_prompt_only(
        4,
        totals(512, 256, 900.0),
        Err("engine returned prefill_tokens 512 over 4 prompts, expected 512 each")
    )]
    fn a_reading_becomes_the_total_time(
        #[case] batch_size: u32,
        #[case] reading: anyhow::Result<EndToEndMeasurementReading>,
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
                let (total_time_ms, _) = mean_and_stddev(result_data)?;
                assert!((total_time_ms - ms).abs() < 1e-9, "{total_time_ms}");
            }
            (Err(err), Err(message)) => {
                assert_eq!(
                    format!("{err:#}"),
                    format!("end_to_end_latency_512_256 rep 1/5: {message}")
                );
                assert!(
                    err.downcast_ref::<EndToEndMeasurementError>().is_some(),
                    "{err:#}"
                );
            }
            (got, expected) => anyhow::bail!("expected {expected:?}, got {got:?}"),
        }
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
            reading(512, 256, ms)
        })?;
        let (total_time_ms, total_time_ms_stddev) = mean_and_stddev(result_data)?;
        assert!((total_time_ms - mean).abs() < 1e-9, "{total_time_ms}");
        match (total_time_ms_stddev, stddev) {
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
        let result_data = end_to_end(
            &body(batch_size)?,
            NonZeroUsize::try_from(2)?,
            &|| Ok(()),
            no_observer(),
            warmup,
            (),
            |()| {
                let ms = *times.next().context("attempted past the reps")?;
                let answer = prompt(512, 256, ms)?;
                Ok(EndToEndMeasurementReading::PerPrompt(vec![answer; prompts]))
            },
        )?;
        assert_eq!(times.next(), None, "an attempt was not made");
        let (total_time_ms, _) = mean_and_stddev(result_data)?;
        assert!((total_time_ms - 40.0).abs() < 1e-9, "{total_time_ms}");
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
            ..MeasurementObserver::QUIET
        };
        // Nothing marks the warm-up to the attempt; it is the first call.
        let mut warming = warmup == MeasurementWarmup::Once;
        let got = end_to_end(
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
                let decode_tokens = if failing == Some(refused) { 255 } else { 256 };
                Ok(reading(512, decode_tokens, 900.0)?)
            },
        );
        if let (Err(err), Some(step)) = (&got, failing) {
            let prefix = if step.starts_with("warm-up") {
                "end_to_end_latency_512_256 warm-up: "
            } else {
                "end_to_end_latency_512_256 rep 1/2: "
            };
            assert!(format!("{err:#}").starts_with(prefix), "{err:#}");
        }
        match (got, failing) {
            (Ok(_), None) => {}
            (Err(err), Some("check" | "warm-up check")) => assert!(
                matches!(
                    err.downcast_ref::<EndToEndMeasurementError>(),
                    Some(EndToEndMeasurementError::Mismatch {
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
        let got = end_to_end(
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
                Ok(reading(512, 256, 900.0)?)
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
            Ok(EndToEndMeasurementReading::PerPrompt(
                [0.0, 1.0, 2.0, 3.0]
                    .into_iter()
                    .map(|ms| prompt(512, 256, rep + ms))
                    .collect::<anyhow::Result<_>>()?,
            ))
        })?;
        let (total_time_ms, total_time_ms_stddev) = mean_and_stddev(result_data)?;
        assert!((total_time_ms - 31.5).abs() < 1e-9, "{total_time_ms}");
        let stddev = total_time_ms_stddev.context("five reps have a spread")?;
        assert!((stddev - 250.0_f64.sqrt()).abs() < 1e-9, "{stddev}");
        Ok(())
    }
}
