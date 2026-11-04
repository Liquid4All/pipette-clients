//! One timing cell's measurement: its repetitions and what they reduce to
//! ([`run`]). The methodology (rep count, readiness wait, observer bracket,
//! reduction) lives here; the engines supply only the work.
//!
//! An engine's untimed warm-up runs at the cell's own prefill and decode
//! counts, never a lighter stand-in: engines compile kernels per tensor shape,
//! so a smaller warm-up leaves that cost inside the first measured rep.
//!
//! The device conditions each rep is captured under are in
//! `docs/methodology/device-conditions.md`.

use std::num::NonZeroUsize;
use std::time::{Duration, Instant};

use pipette_plan_types::{BatchSize, BenchmarkDefinition, PositiveSeconds, RunRequest};

use crate::readiness::ReadinessGate;

mod asr_throughput;
mod decode;
mod end_to_end;
mod eval;
mod max_memory;
mod prefill;
mod vl_max_memory;
mod vl_spec_decode;
mod vl_throughput;

pub use self::{
    asr_throughput::{
        AsrThroughputMeasurementError, AsrThroughputMeasurementReading, asr_throughput,
    },
    decode::{
        DecodeMeasurementError, DecodeMeasurementPerPromptReading, DecodeMeasurementReading,
        DecodeMeasurementTotalsReading, decode,
    },
    end_to_end::{
        EndToEndMeasurementError, EndToEndMeasurementPerPromptReading, EndToEndMeasurementReading,
        EndToEndMeasurementTotalsReading, end_to_end,
    },
    eval::{
        EvalMeasurementDataset, EvalMeasurementError, EvalMeasurementInFlight,
        EvalMeasurementReading, EvalMeasurementRecord, EvalMeasurementSample, EvalMessages,
        EvalRows, chunked_pending, eval, eval_complete, eval_complete_chunked,
    },
    max_memory::{MaxMemoryMeasurementError, MaxMemoryMeasurementReading, max_memory},
    prefill::{
        PrefillMeasurementError, PrefillMeasurementPerPromptReading, PrefillMeasurementReading,
        PrefillMeasurementTotalsReading, prefill,
    },
    vl_max_memory::{
        VlMaxMemoryMeasurementError, VlMaxMemoryMeasurementReading, decoded_in_full, vl_max_memory,
    },
    vl_spec_decode::{
        VlSpecDecodeMeasurementAttemptFailure, VlSpecDecodeMeasurementError,
        VlSpecDecodeMeasurementReading, VlSpecDecodeMeasurementRow, VlSpecDecodeMeasurementSample,
        VlSpecDecodeMeasurementSpeculation, VlSpecDecodeMeasurementTurn, vl_spec_decode,
    },
    vl_throughput::{
        VlThroughputMeasurementError, VlThroughputMeasurementReading, require_decode_steps,
        vl_throughput,
    },
};

/// Repetitions every timing cell measures. Shared, because cells that ran
/// different counts would not be comparable.
pub const REPS: usize = 5;

/// Called around each measured rep, in field order; for spec decode the rep is
/// the whole traversal. An `Err` from either abandons the cell; a cell that
/// measures no reps calls neither.
#[derive(Clone, Copy)]
pub struct MeasurementObserver<'a> {
    /// After the gate; the CLI takes a thermal reading.
    pub on_before_rep_start: &'a dyn Fn() -> anyhow::Result<()>,
    /// After the rep's timed work; the CLI takes a thermal reading.
    pub on_after_rep_end: &'a dyn Fn() -> anyhow::Result<()>,
}

/// Where a timing entry's warm-up happens.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeasurementWarmup {
    /// One untimed, checked attempt before the reps, its reading not reduced,
    /// so a first-use cost (a server's first request, a cold kernel build)
    /// stays out of rep 1.
    Once,
    /// Each attempt is a fresh process that warms itself at the cell's shape
    /// (a `llama-bench` prefill, `llama-batched-bench`).
    Inline,
}

/// Whether a peak-memory entry runs its attempt once before the measured one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryMeasurementWarmup {
    /// The measured attempt is the cell's first.
    Cold,
    /// One checked attempt first, its reading dropped, for an engine that
    /// compiles into a cache that outlives its process (a GPU driver's kernel
    /// cache, OpenVINO's cache dir), so the measured attempt loads what the
    /// warm-up compiled. A peak is a high-water mark, so each attempt must
    /// launch its own process, the warm-up's stopped before the measured one
    /// starts. An engine whose process outlives the attempt
    /// (torch-oai, whose probe would fail on a second call) passes
    /// [`Self::Cold`].
    Once,
}

/// Why an engine's attempt failed.
#[derive(Debug, thiserror::Error)]
pub enum AttemptError {
    #[error(transparent)]
    Original(#[from] anyhow::Error),
}

impl AttemptError {
    /// The carried error as raised. `?` would wrap it, and the transparent
    /// wrapper hides its type from a chain walk such as the worker's
    /// `was_cancelled`.
    fn into_anyhow(self) -> anyhow::Error {
        match self {
            Self::Original(err) => err,
        }
    }
}

/// A positive, finite reading, or an error naming the metric. Zero, negative,
/// `NaN` and infinity all mean the engine reported nothing usable, and each
/// would otherwise reduce to a mean that looks like a measurement.
pub fn positive_finite(metric: &str, value: f64) -> anyhow::Result<f64> {
    if !value.is_finite() || value <= 0.0 {
        anyhow::bail!("invalid {metric}: {value}");
    }
    Ok(value)
}

/// The requests the cell keeps in flight, for an engine that can run more
/// than one: `parameter_batch_size` as a positive count, and the reference
/// procedure on a kind that carries none.
pub fn batch_size(benchmark: &BenchmarkDefinition) -> anyhow::Result<NonZeroUsize> {
    Ok(benchmark
        .batch_size()
        .unwrap_or_else(BatchSize::one)
        .in_flight()?)
}

/// Refuse a batched entry on an engine that runs one request at a time, before
/// anything loads. Running it serially would file the reference measurement
/// under the batched benchmark id.
///
/// `engine` is the sentence fragment that says why ("the OpenVINO server
/// generates one sequence at a time").
pub fn require_one_in_flight(benchmark: &BenchmarkDefinition, engine: &str) -> anyhow::Result<()> {
    let batch_size = batch_size(benchmark)?;
    anyhow::ensure!(
        batch_size.get() == 1,
        "{} keeps {batch_size} requests in flight, and {engine}; a batched {} entry \
         needs a server with that many slots: vLLM or SGLang, or llama-server for \
         end_to_end_latency",
        benchmark.benchmark_id(),
        benchmark.benchmark_type(),
    );
    Ok(())
}

/// Run `work` once per slot with all of them in flight, and return the results
/// in slot order. At batch size 1 it runs on the calling thread.
///
/// Each request is timed from its own submission to its own last token, so a
/// rep's value is a per-request reading under load, not the batch's wall
/// clock (which grows with the batch and describes the host).
///
/// A failed slot fails the rep: a mean over the slots that succeeded would be
/// at a concurrency the row does not name. Siblings are not cancelled but run
/// to their end or their HTTP timeout; the error surfaces once every slot has
/// retired.
pub fn in_flight<T: Send>(
    batch_size: NonZeroUsize,
    work: impl Fn(usize) -> anyhow::Result<T> + Sync,
) -> anyhow::Result<Vec<T>> {
    if batch_size.get() == 1 {
        return Ok(vec![work(0)?]);
    }
    let work = &work;
    let outcomes = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..batch_size.get())
            .map(|slot| scope.spawn(move || work(slot)))
            .collect();
        handles
            .into_iter()
            .map(|handle| {
                // A panic in a dependency (this crate denies its own) is
                // reported as an error rather than resumed.
                handle
                    .join()
                    .unwrap_or_else(|_| Err(anyhow::anyhow!("a request slot panicked")))
            })
            .collect::<Vec<_>>()
    });
    outcomes.into_iter().collect()
}

/// The rep's value from its slots' readings: their mean. The batch is one rep,
/// so `stddev` still describes rep-to-rep spread, not spread within a batch.
pub fn per_request_mean(samples: &[f64]) -> anyhow::Result<f64> {
    let count = u32::try_from(samples.len()).unwrap_or(u32::MAX);
    anyhow::ensure!(count > 0, "a rep produced no readings to average");
    Ok(samples.iter().sum::<f64>() / f64::from(count))
}

/// Default HTTP timeout when `benchmark_flags.http_timeout` is unset, for
/// every server-driven cell.
pub const DEFAULT_HTTP_TIMEOUT_SECS: u64 = 120;

/// A cell's HTTP timeout: a pinned `http_timeout_seconds` passes through
/// unscaled, while the default grows with the requests in flight, since a
/// request waits its turn in the scheduler on top of its own work.
///
/// The default is the caller's because it depends on the workload: a text
/// cell's two minutes would cut a corpus turn that decodes a vision prompt one
/// token at a time.
///
/// `slots` is never zero (callers pass `1` or a `BatchSize`, which refuses
/// zero): zero would scale the default to `Duration::ZERO` and time every
/// request out at once.
pub fn http_timeout_with_default(
    pinned: Option<PositiveSeconds>,
    slots: u32,
    default_secs: u64,
) -> Duration {
    match pinned {
        Some(pinned) => Duration::from_secs(pinned.into_inner()),
        None => Duration::from_secs(default_secs).saturating_mul(slots),
    }
}

/// [`http_timeout_with_default`] over a request, at [`DEFAULT_HTTP_TIMEOUT_SECS`].
pub fn http_timeout(req: &RunRequest, slots: u32) -> Duration {
    http_timeout_for(
        req.benchmark_flags.as_ref().and_then(|f| f.http_timeout()),
        slots,
    )
}

/// [`http_timeout`] for a pinned `http_timeout`.
pub fn http_timeout_for(pinned: Option<PositiveSeconds>, slots: u32) -> Duration {
    http_timeout_with_default(pinned, slots, DEFAULT_HTTP_TIMEOUT_SECS)
}

/// Default bound on silence between SSE events when the plan pins no
/// `http_timeout`. Generous because a long prefill on a loaded host can go
/// quiet for minutes before the first token.
pub const DEFAULT_STREAM_IDLE_TIMEOUT_SECS: u64 = 1800;

/// The `http_timeout` flag read as a bound on the gap between SSE events
/// rather than on the whole request (see [`crate::sse::data_lines`]). Unscaled
/// by slots: concurrency makes a stream slower to finish, not slower between
/// its events.
///
/// A pin is a floor here, unlike in [`http_timeout_with_default`]: it was
/// chosen as a whole-request budget, and a stream's longest gap is its
/// prefill, so a short pin would fail runs a slow first token alone explains.
pub fn stream_idle_timeout(req: &RunRequest) -> Duration {
    stream_idle_timeout_for(req.benchmark_flags.as_ref().and_then(|f| f.http_timeout()))
}

/// [`stream_idle_timeout`] for a pinned `http_timeout`.
pub fn stream_idle_timeout_for(pinned: Option<PositiveSeconds>) -> Duration {
    Duration::from_secs(
        pinned
            .map_or(0, PositiveSeconds::into_inner)
            .max(DEFAULT_STREAM_IDLE_TIMEOUT_SECS),
    )
}

/// At most the last `n` lines of `text`, rejoined with `\n`.
pub fn tail_lines(text: &str, n: usize) -> String {
    let mut tail: Vec<&str> = text.lines().rev().take(n).collect();
    tail.reverse();
    tail.join("\n")
}

/// Reject a rep whose token count is not what the cell asked for. What this
/// catches (a truncating static shape, a template the engine added, an early
/// stop) is silent otherwise.
pub fn expect_tokens(metric: &str, actual: u32, expected: u32) -> anyhow::Result<()> {
    if actual != expected {
        anyhow::bail!("engine returned {metric} {actual}, expected {expected}");
    }
    Ok(())
}

/// One repetition's result and the wall clock of its work alone (the readiness
/// wait is untimed). Cells whose sample comes from the response itself (a
/// server-reported tokens/sec, a `llama-bench` row) ignore `elapsed`.
#[derive(Debug, Clone, PartialEq)]
pub struct Rep<T> {
    pub value: T,
    pub elapsed: Duration,
}

impl<T> Rep<T> {
    /// The rep's wall clock in milliseconds, as the result schema stores it.
    /// Lossy, so the field stays a `Duration`.
    pub fn elapsed_ms(&self) -> f64 {
        duration_ms(self.elapsed)
    }
}

/// Run [`REPS`] repetitions of one cell, in a fixed order:
///
/// 1. wait on readiness,
/// 2. report the rep's start (the entry-condition reading),
/// 3. `prepare` the rep, untimed,
/// 4. `work`, timed,
/// 5. report the rep's end,
/// 6. `sample` the rep: check it and read off the metric.
///
/// The ordering is a capture contract: callers supply only steps 3, 4 and 6,
/// so they cannot put anything between the others. `prepare` (a KV reset, a
/// re-prefill) runs after the start is reported, so it cannot move what that
/// reading describes. `sample` runs before the next rep's gate, so a rejected
/// response abandons the run at the rep that produced it.
///
/// Any failure abandons the cell: a mean over a subset of reps is not the
/// measurement the result claims. `label` names the cell in the log; each rep
/// is logged as it is sampled, so a cell that fails on rep 4 still leaves
/// three readings behind.
pub fn run<T>(
    label: &str,
    gate: ReadinessGate,
    observer: MeasurementObserver<'_>,
    mut prepare: impl FnMut(usize) -> anyhow::Result<()>,
    mut work: impl FnMut(usize) -> anyhow::Result<T>,
    mut sample: impl FnMut(usize, &Rep<T>) -> anyhow::Result<f64>,
) -> anyhow::Result<Measurement<T>> {
    let reps = (0..REPS)
        .map(|idx| {
            gate()?;
            (observer.on_before_rep_start)()?;
            tracing::info!(cell = label, rep = idx + 1, reps = REPS, "measurement run");
            prepare(idx)?;

            let started = Instant::now();
            let value = work(idx)?;
            let elapsed = started.elapsed();
            (observer.on_after_rep_end)()?;

            let rep = Rep { value, elapsed };
            let sample = sample(idx, &rep)?;
            tracing::info!(
                cell = label,
                rep = idx + 1,
                reps = REPS,
                sample_ms = sample,
                "measurement run sampled"
            );
            Ok((rep, sample))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    let samples = reps.iter().map(|(_, sample)| *sample).collect::<Vec<_>>();
    let stats = mean_stddev(samples.iter().copied());
    log_reduction(label, None, &samples, stats);
    Ok(Measurement {
        label: label.to_string(),
        reps: reps.into_iter().map(|(rep, _)| rep).collect(),
        stats,
    })
}

/// A completed run: its repetitions, and the reduction of the metric it
/// sampled. A cell picks *which* metric it measures, never *how* it is
/// reduced.
#[derive(Debug)]
pub struct Measurement<T> {
    /// The cell name [`run`] logged under, kept so a second metric reduced off
    /// the same reps reports under the same cell.
    label: String,
    reps: Vec<Rep<T>>,
    stats: Stats,
}

/// A metric's mean and spread across a run, in milliseconds: the unit every
/// sample a cell reports must already be in.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Stats {
    pub mean_ms: f64,
    pub stddev_ms: f64,
}

impl<T> Measurement<T> {
    /// The reduction of the metric [`run`] sampled.
    pub fn stats(&self) -> Stats {
        self.stats
    }

    /// The reduction of a second metric, `name`, off the same reps, logged
    /// under the same cell.
    pub fn metric(&self, name: &str, sample: impl Fn(&Rep<T>) -> f64) -> Stats {
        let samples = self.reps.iter().map(sample).collect::<Vec<_>>();
        let stats = mean_stddev(samples.iter().copied());
        log_reduction(&self.label, Some(name), &samples, stats);
        stats
    }

    pub fn iter(&self) -> std::slice::Iter<'_, Rep<T>> {
        self.reps.iter()
    }

    /// The first rep's result, for a per-run constant such as the prompt-token
    /// count every rep shares.
    pub fn first(&self) -> Option<&T> {
        self.reps.first().map(|rep| &rep.value)
    }
}

impl<T> IntoIterator for Measurement<T> {
    type Item = Rep<T>;
    type IntoIter = std::vec::IntoIter<Rep<T>>;

    fn into_iter(self) -> Self::IntoIter {
        self.reps.into_iter()
    }
}

/// Sample mean and standard deviation (Bessel-corrected, n−1). Empty is
/// `(0.0, 0.0)`; a single sample is `(mean, 0.0)`.
pub fn mean_stddev(samples: impl IntoIterator<Item = f64>) -> Stats {
    let samples = samples.into_iter().collect::<Vec<_>>();
    let n = samples.len() as f64;
    if n == 0.0 {
        return Stats {
            mean_ms: 0.0,
            stddev_ms: 0.0,
        };
    }
    let mean_ms = samples.iter().sum::<f64>() / n;
    if n <= 1.0 {
        return Stats {
            mean_ms,
            stddev_ms: 0.0,
        };
    }
    let variance = samples.iter().map(|x| (x - mean_ms).powi(2)).sum::<f64>() / (n - 1.0);
    Stats {
        mean_ms,
        stddev_ms: variance.sqrt(),
    }
}

/// Logs a metric's samples and their reduction in one event: a mean alone
/// hides its outliers. `metric` names a second metric reduced off the same
/// reps.
fn log_reduction(cell: &str, metric: Option<&str>, samples: &[f64], stats: Stats) {
    tracing::info!(
        cell,
        metric = metric.map(tracing::field::display),
        samples_ms = ?samples,
        mean_ms = stats.mean_ms,
        stddev_ms = stats.stddev_ms,
        "reduced samples"
    );
}

/// A duration in milliseconds, as the result schema stores it.
fn duration_ms(duration: Duration) -> f64 {
    duration.as_nanos() as f64 / 1_000_000.0
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use anyhow::Context;
    use rstest::rstest;

    use pipette_plan_types::{
        BatchSize, BenchmarkId, DecodeThroughputBenchmark, EndToEndLatencyBenchmark,
        MaxMemoryUsageBenchmark, NonNegativeCount, PositiveCount, PrefillThroughputBenchmark,
    };

    use super::*;

    #[rstest]
    #[case::pinned_ignores_the_default(Some(30), 8, 120, 30)]
    #[case::pinned_ignores_the_slots(Some(30), 1, 600, 30)]
    #[case::unpinned_scales_with_slots(None, 8, 120, 960)]
    #[case::unpinned_takes_the_callers_default(None, 1, 600, 600)]
    fn a_pin_passes_through_and_only_the_default_scales(
        #[case] pinned: Option<u64>,
        #[case] slots: u32,
        #[case] default_secs: u64,
        #[case] expected_secs: u64,
    ) -> anyhow::Result<()> {
        let pinned = pinned.map(PositiveSeconds::try_new).transpose()?;
        assert_eq!(
            http_timeout_with_default(pinned, slots, default_secs),
            Duration::from_secs(expected_secs)
        );
        Ok(())
    }

    #[rstest]
    #[case::unset(None, DEFAULT_STREAM_IDLE_TIMEOUT_SECS)]
    #[case::shorter_pin_is_floored(Some(600), DEFAULT_STREAM_IDLE_TIMEOUT_SECS)]
    #[case::longer_pin_wins(Some(3600), 3600)]
    fn a_pinned_budget_only_lengthens_the_idle_bound(
        #[case] pinned: Option<u64>,
        #[case] expected_secs: u64,
    ) -> anyhow::Result<()> {
        let pinned = pinned.map(PositiveSeconds::try_new).transpose()?;
        assert_eq!(
            stream_idle_timeout_for(pinned),
            Duration::from_secs(expected_secs)
        );
        Ok(())
    }

    #[derive(Debug, Clone, Copy)]
    enum Kind {
        Prefill,
        Decode,
        EndToEnd,
        MaxMemory,
    }

    fn definition(kind: Kind, batch_size: BatchSize) -> anyhow::Result<BenchmarkDefinition> {
        Ok(match kind {
            Kind::Prefill => BenchmarkDefinition::PrefillThroughput(PrefillThroughputBenchmark {
                benchmark_id: BenchmarkId::try_new("prefill_throughput_512_bs2".to_owned())?,
                parameter_prefill_tokens: NonNegativeCount::try_new(512)?,
                parameter_batch_size: batch_size,
            }),
            Kind::Decode => BenchmarkDefinition::DecodeThroughput(DecodeThroughputBenchmark {
                benchmark_id: BenchmarkId::try_new("decode_throughput_512_256_bs2".to_owned())?,
                parameter_prefill_tokens: NonNegativeCount::try_new(512)?,
                parameter_decode_tokens: PositiveCount::try_new(256)?,
                parameter_batch_size: batch_size,
            }),
            Kind::EndToEnd => BenchmarkDefinition::EndToEndLatency(EndToEndLatencyBenchmark {
                benchmark_id: BenchmarkId::try_new("end_to_end_latency_512_256_bs4".to_owned())?,
                parameter_prefill_tokens: NonNegativeCount::try_new(512)?,
                parameter_decode_tokens: PositiveCount::try_new(256)?,
                parameter_batch_size: batch_size,
            }),
            Kind::MaxMemory => BenchmarkDefinition::MaxMemoryUsage(MaxMemoryUsageBenchmark {
                benchmark_id: BenchmarkId::try_new("max_memory_usage_512".to_owned())?,
                parameter_prefill_tokens: NonNegativeCount::try_new(512)?,
            }),
        })
    }

    #[rstest]
    #[case::reference(Kind::EndToEnd, 1, None)]
    #[case::batched(Kind::EndToEnd, 4, Some("end_to_end_latency_512_256_bs4"))]
    #[case::prefill_batched(Kind::Prefill, 2, Some("2 requests in flight"))]
    #[case::decode_batched(Kind::Decode, 2, Some("decode_throughput"))]
    // A kind with no batch parameter is the reference procedure, so it passes.
    #[case::kind_without_a_batch(Kind::MaxMemory, 4, None)]
    fn one_in_flight_is_required_by_name(
        #[case] kind: Kind,
        #[case] batch_size: u32,
        #[case] expected: Option<&str>,
    ) -> anyhow::Result<()> {
        let benchmark = definition(kind, BatchSize::try_new(batch_size)?)?;
        let result = require_one_in_flight(&benchmark, "this engine runs one sequence");
        match expected {
            None => result?,
            Some(substr) => {
                let err = result.err().context("expected a refusal")?;
                assert!(format!("{err:#}").contains(substr), "got {err:#}");
            }
        }
        Ok(())
    }

    #[rstest]
    #[case::reference(1)]
    #[case::pair(2)]
    #[case::four(4)]
    fn in_flight_keeps_every_slot_in_flight(#[case] slots: usize) -> anyhow::Result<()> {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let batch_size = NonZeroUsize::new(slots).context("slots")?;
        let live = AtomicUsize::new(0);
        let peak = AtomicUsize::new(0);
        let out = in_flight(batch_size, |slot| {
            let now = live.fetch_add(1, Ordering::SeqCst) + 1;
            peak.fetch_max(now, Ordering::SeqCst);
            // Held until every sibling arrives, so the peak reads the fan-out
            // rather than scheduling luck; the deadline bounds a broken one.
            let deadline = Instant::now() + Duration::from_secs(5);
            while live.load(Ordering::SeqCst) < slots && Instant::now() < deadline {
                std::thread::sleep(Duration::from_millis(1));
            }
            live.fetch_sub(1, Ordering::SeqCst);
            Ok(slot)
        })?;
        assert_eq!(out, (0..slots).collect::<Vec<_>>(), "in slot order");
        assert_eq!(peak.load(Ordering::SeqCst), slots);
        Ok(())
    }

    #[test]
    fn a_failed_slot_fails_the_batch() -> anyhow::Result<()> {
        let batch_size = NonZeroUsize::new(4).context("slots")?;
        let err = in_flight(batch_size, |slot| {
            if slot == 2 {
                anyhow::bail!("slot {slot} refused");
            }
            Ok(slot)
        })
        .err()
        .context("expected the batch to fail")?;
        assert!(format!("{err:#}").contains("slot 2 refused"), "got {err:#}");
        Ok(())
    }

    #[rstest]
    #[case::single(&[12.5], Some(12.5))]
    #[case::batched(&[10.0, 20.0, 30.0, 40.0], Some(25.0))]
    #[case::empty(&[], None)]
    fn the_rep_value_is_the_per_request_mean(
        #[case] samples: &[f64],
        #[case] expected: Option<f64>,
    ) -> anyhow::Result<()> {
        match (per_request_mean(samples), expected) {
            (Ok(mean), Some(want)) => assert!((mean - want).abs() < 1e-9, "got {mean}"),
            (Err(_), None) => {}
            (got, want) => anyhow::bail!("expected {want:?}, got {got:?}"),
        }
        Ok(())
    }

    /// What a run reported, in order, so a test can assert the interleaving.
    #[derive(Debug, PartialEq)]
    enum Event {
        Gate,
        Started,
        Prepared(usize),
        Work(usize),
        Finished,
        Sampled(usize),
    }

    /// Runs `work` and `sample` with every hook appending to one log.
    fn record<T>(
        work: impl FnMut(usize) -> anyhow::Result<T>,
        sample: impl FnMut(usize, &Rep<T>) -> anyhow::Result<f64>,
    ) -> (anyhow::Result<Measurement<T>>, Vec<Event>) {
        let log = RefCell::new(Vec::new());
        // Scoped so the hooks, which borrow `log`, are gone before it is read.
        let result = {
            let gate = || {
                log.borrow_mut().push(Event::Gate);
                Ok(())
            };
            let observer = MeasurementObserver {
                on_before_rep_start: &|| {
                    log.borrow_mut().push(Event::Started);
                    Ok(())
                },
                on_after_rep_end: &|| {
                    log.borrow_mut().push(Event::Finished);
                    Ok(())
                },
            };
            let (mut work, mut sample) = (work, sample);
            run(
                "test",
                &gate,
                observer,
                |idx| {
                    log.borrow_mut().push(Event::Prepared(idx));
                    Ok(())
                },
                |idx| {
                    log.borrow_mut().push(Event::Work(idx));
                    work(idx)
                },
                |idx, rep| {
                    log.borrow_mut().push(Event::Sampled(idx));
                    sample(idx, rep)
                },
            )
        };
        (result, log.into_inner())
    }

    #[test]
    fn each_rep_is_gated_then_bracketed_then_sampled() -> anyhow::Result<()> {
        let (result, events) = record(Ok, |idx, _| Ok(idx as f64));

        assert_eq!(result?.iter().count(), REPS);
        assert_eq!(
            events,
            (0..REPS)
                .flat_map(|idx| [
                    Event::Gate,
                    Event::Started,
                    Event::Prepared(idx),
                    Event::Work(idx),
                    Event::Finished,
                    Event::Sampled(idx)
                ])
                .collect::<Vec<_>>()
        );
        Ok(())
    }

    /// The failed rep reports no end, and no later rep starts.
    #[test]
    fn a_failed_rep_stops_the_run() {
        let (result, events) = record(
            |idx| {
                if idx == 1 {
                    anyhow::bail!("rep failed")
                }
                Ok(idx)
            },
            |idx, _| Ok(idx as f64),
        );

        assert!(result.is_err());
        assert_eq!(
            events,
            vec![
                Event::Gate,
                Event::Started,
                Event::Prepared(0),
                Event::Work(0),
                Event::Finished,
                Event::Sampled(0),
                Event::Gate,
                Event::Started,
                Event::Prepared(1),
                Event::Work(1),
            ]
        );
    }

    #[test]
    fn a_rejected_sample_stops_the_run() {
        let (result, events) = record(Ok, |idx, _| {
            if idx == 1 {
                anyhow::bail!("invalid sample")
            }
            Ok(idx as f64)
        });

        assert!(result.is_err());
        assert_eq!(
            events.iter().filter(|e| **e == Event::Started).count(),
            2,
            "reps after the rejected one should not run: {events:?}"
        );
        assert_eq!(events.last(), Some(&Event::Sampled(1)));
    }

    #[test]
    fn the_readiness_wait_is_not_timed() -> anyhow::Result<()> {
        let wait = Duration::from_millis(25);
        let gate = || {
            std::thread::sleep(wait);
            Ok(())
        };
        let observer = MeasurementObserver {
            on_before_rep_start: &|| Ok(()),
            on_after_rep_end: &|| Ok(()),
        };

        let measured = run(
            "test",
            &gate,
            observer,
            |_| Ok(()),
            |_| Ok(()),
            |_, rep| Ok(rep.elapsed_ms()),
        )?;

        assert!(
            measured.iter().all(|rep| rep.elapsed < wait),
            "a rep's elapsed time included the wait"
        );
        Ok(())
    }

    #[test]
    fn reports_the_wall_clock_in_milliseconds() {
        let rep = Rep {
            value: (),
            elapsed: Duration::from_micros(1_234_500),
        };

        assert_eq!(rep.elapsed_ms(), 1234.5);
    }

    /// The sampled metric is reduced, not the wall clock.
    #[test]
    fn reduces_the_sampled_metric() -> anyhow::Result<()> {
        let (result, _) = record(Ok, |_, _| Ok(42.0));

        assert_eq!(
            result?.stats(),
            Stats {
                mean_ms: 42.0,
                stddev_ms: 0.0
            }
        );
        Ok(())
    }

    #[test]
    fn reduces_a_second_metric_off_the_same_reps() -> anyhow::Result<()> {
        let (result, _) = record(|idx| Ok(idx as f64), |_, _| Ok(1.0));
        let measured = result?;

        assert_eq!(measured.stats().mean_ms, 1.0);
        assert_eq!(
            measured.metric("second", |rep| rep.value),
            Stats {
                // 0..4: mean 2, sample stddev sqrt(10/4).
                mean_ms: 2.0,
                stddev_ms: 2.5_f64.sqrt(),
            }
        );
        Ok(())
    }

    /// Captures what the subscriber writes.
    #[derive(Clone, Default)]
    struct Written(std::sync::Arc<std::sync::Mutex<Vec<u8>>>);

    impl std::io::Write for Written {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.0
                .lock()
                .map_err(|e| std::io::Error::other(e.to_string()))?
                .extend_from_slice(bytes);
            Ok(bytes.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    #[rstest]
    #[case::samples(
        vec![2.0, 4.0, 4.0, 4.0, 5.0],
        "[2.0, 4.0, 4.0, 4.0, 5.0]",
        3.8,
        1.2_f64.sqrt()
    )]
    #[case::empty(vec![], "[]", 0.0, 0.0)]
    fn the_reduction_reports_the_samples_behind_the_mean(
        #[case] samples: Vec<f64>,
        #[case] expected_samples: &str,
        #[case] expected_mean: f64,
        #[case] expected_stddev: f64,
    ) -> anyhow::Result<()> {
        let written = Written::default();
        let writer = written.clone();
        let subscriber = tracing_subscriber::fmt()
            .json()
            .flatten_event(true)
            .with_writer(move || writer.clone())
            .finish();

        tracing::subscriber::with_default(subscriber, || {
            log_reduction(
                "prefill_throughput",
                None,
                &samples,
                mean_stddev(samples.clone()),
            );
        });

        let bytes = written
            .0
            .lock()
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .clone();
        let event: serde_json::Value = serde_json::from_slice(&bytes)?;
        assert_eq!(event["message"], "reduced samples");
        assert_eq!(event["cell"], "prefill_throughput");
        assert_eq!(event["samples_ms"], expected_samples);
        assert_eq!(event["mean_ms"], expected_mean);
        let stddev = event["stddev_ms"]
            .as_f64()
            .ok_or_else(|| anyhow::anyhow!("no stddev_ms in {event}"))?;
        assert!((stddev - expected_stddev).abs() < 1e-9, "got {event}");
        assert!(event.get("metric").is_none(), "got {event}");
        Ok(())
    }

    #[rstest]
    #[case::empty(vec![], 0.0, 0.0)]
    #[case::single(vec![42.0], 42.0, 0.0)]
    #[case::known_spread(vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0], 5.0, 2.138)]
    fn reduces_samples(#[case] samples: Vec<f64>, #[case] mean: f64, #[case] stddev: f64) {
        let stats = mean_stddev(samples);

        assert!(
            (stats.mean_ms - mean).abs() < 1e-9,
            "mean was {}",
            stats.mean_ms
        );
        assert!(
            (stats.stddev_ms - stddev).abs() < 0.01,
            "stddev was {}",
            stats.stddev_ms
        );
    }

    /// The last lines, because a server that dies during load says why on the
    /// way out.
    #[test]
    fn a_quoted_stderr_tail_is_bounded_and_ends_at_the_error() {
        let stderr = (0..10_000).fold(String::new(), |mut acc, i| {
            acc.push_str(&format!("load_tensors: layer {i}\n"));
            acc
        }) + "ggml_metal_graph_compute: command buffer 0 failed";

        let quoted = tail_lines(&stderr, 40);

        assert_eq!(quoted.lines().count(), 40);
        assert!(quoted.ends_with("command buffer 0 failed"), "{quoted}");
    }

    #[rstest]
    #[case("one line", "one line")]
    #[case("a\nb\nc", "a\nb\nc")]
    #[case("", "")]
    fn a_short_stderr_tail_is_passed_through(#[case] input: &str, #[case] expected: &str) {
        assert_eq!(tail_lines(input, 40), expected);
    }

    #[rstest]
    #[case::exact(512, None)]
    #[case::short(511, Some("engine returned prompt_tokens 511, expected 512"))]
    #[case::over(513, Some("engine returned prompt_tokens 513, expected 512"))]
    fn a_wrong_token_count_fails_naming_both_counts(
        #[case] actual: u32,
        #[case] error: Option<&str>,
    ) {
        let got = expect_tokens("prompt_tokens", actual, 512)
            .err()
            .map(|err| format!("{err:#}"));
        assert_eq!(got.as_deref(), error);
    }
}
