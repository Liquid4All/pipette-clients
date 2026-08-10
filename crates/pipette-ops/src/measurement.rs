//! One timing cell's measurement: its repetitions and what they reduce to
//! ([`run`]).
//!
//! Every timing benchmark measures the same way: a fixed number of reps, each
//! waiting on readiness, each bracketed by the caller's [`RepObserver`], each
//! contributing one sample that reduces to a mean and a standard deviation.
//! That's the methodology, not any one engine's business, so it lives here and
//! the engines supply only the work.
//!
//! Every sample is logged as it is taken, and again alongside the mean it
//! reduces to, so the numbers behind a reported average are always readable
//! without re-running the cell. That belongs here for the same reason the
//! reduction does: a client that showed its samples and one that showed only
//! the mean would not be reporting the same measurement.
//!
//! The untimed warm-up that precedes those reps is each engine's to issue, but
//! its shape is fixed: the cell's own prefill and decode counts, never a lighter
//! stand-in. Engines select and compile kernels per tensor shape, so a warm-up
//! smaller than the workload leaves that cost to be paid inside the first
//! measured rep, where it lands in the mean and inflates the spread.
//!
//! See `pipette-mgmt` `docs/methodology/thermal-telemetry.md` for the capture
//! contract this enforces.

use std::time::{Duration, Instant};

use crate::readiness::{ReadinessGate, RepObserver};

/// Repetitions every timing cell measures. One number because the result
/// schema treats the per-rep arrays of different cells as one methodology —
/// a cell that ran a different count would not be comparable.
pub const REPS: usize = 5;

/// A positive, finite reading, or an error naming the metric.
///
/// The check every engine's reducer needs before a number is allowed to become
/// a result: zero, negative, `NaN` and infinity all mean the engine reported
/// nothing usable, and each would otherwise reduce to a mean that looks like a
/// measurement. Shared so a rejected reading reads identically whichever backend
/// produced it.
pub fn positive_finite(metric: &str, value: f64) -> anyhow::Result<f64> {
    if !value.is_finite() || value <= 0.0 {
        anyhow::bail!("invalid {metric}: {value}");
    }
    Ok(value)
}

/// Reject a rep whose token count is not what the cell asked for.
///
/// A prompt or a generation that came out the wrong size means the number
/// describes a different workload, so the run fails rather than reporting it —
/// the failure mode this catches (a truncating static shape, a template the
/// engine added, an early stop) is silent otherwise.
pub fn expect_tokens(metric: &str, actual: u32, expected: u32) -> anyhow::Result<()> {
    if actual != expected {
        anyhow::bail!("engine returned {metric} {actual}, expected {expected}");
    }
    Ok(())
}

/// One repetition's result and the wall clock the harness measured around it.
///
/// `elapsed` covers the work and nothing else: the readiness wait happens
/// before the rep is timed. Cells whose sample comes out of the response
/// itself — a server-reported tokens/sec, a `llama-bench` row — ignore it.
#[derive(Debug, Clone, PartialEq)]
pub struct Rep<T> {
    pub value: T,
    pub elapsed: Duration,
}

impl<T> Rep<T> {
    /// The rep's wall clock as the result schema stores it. Lossy, so it lives
    /// here rather than in the field: the harness measures a duration, and
    /// milliseconds are what a `BenchmarkResultData` happens to carry.
    pub fn elapsed_ms(&self) -> f64 {
        duration_ms(self.elapsed)
    }
}

/// Run [`REPS`] repetitions of one cell, in a fixed order:
///
/// 1. wait on readiness,
/// 2. report the rep's start — the entry-condition reading is taken here,
/// 3. `prepare` the rep, untimed,
/// 4. `work`, timed,
/// 5. report the rep's end,
/// 6. `sample` the rep: check it and read off the metric.
///
/// The caller supplies only steps 3, 4 and 6; the count, the wait, the two
/// reports and the reduction are the harness's. That ordering is a capture
/// contract rather than a convention, and it holds here because there is
/// nowhere for a caller to put anything between the steps.
///
/// The three-way split is what keeps them honest. `prepare` is setup a rep
/// needs standing before the clock — a KV reset, a re-prefill — and it runs
/// *after* the start is reported, so however expensive it is it cannot move
/// what that reading describes: the device as the gate cleared, which is the
/// only definition under which two runtimes' readings mean the same thing.
/// `work` is the timed region and nothing else, so the reports bracket the
/// measurement alone. `sample` runs untimed and *before the next rep's gate*,
/// so a response the cell rejects abandons the run at the rep that produced it
/// instead of after another four minutes of reps.
///
/// A failure from any of them abandons the cell: the remaining reps do not run
/// and the error propagates, because a mean over a subset of the methodology
/// is not the measurement the result claims to be.
///
/// `label` names the cell in the log: each rep's sample is logged as step 6
/// takes it, then the whole series is logged with what it reduced to. A rep is
/// reported the moment it is sampled rather than at the end, so a cell that
/// fails on rep 4 still leaves the three readings it did collect in the log.
pub fn run<T>(
    label: &str,
    gate: ReadinessGate,
    observer: &RepObserver,
    mut prepare: impl FnMut(usize) -> anyhow::Result<()>,
    mut work: impl FnMut(usize) -> anyhow::Result<T>,
    mut sample: impl FnMut(usize, &Rep<T>) -> anyhow::Result<f64>,
) -> anyhow::Result<Measurement<T>> {
    let reps = (0..REPS)
        .map(|idx| {
            gate()?;
            observer.rep_started()?;
            log::info!("{label}: measurement run {}/{}", idx + 1, REPS);
            prepare(idx)?;

            let started = Instant::now();
            let value = work(idx)?;
            let elapsed = started.elapsed();
            observer.rep_finished()?;

            let rep = Rep { value, elapsed };
            let sample = sample(idx, &rep)?;
            log::info!(
                "{label}: measurement run {}/{} sampled {sample:.3} ms",
                idx + 1,
                REPS
            );
            Ok((rep, sample))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    let samples = reps.iter().map(|(_, sample)| *sample).collect::<Vec<_>>();
    let stats = mean_stddev(samples.iter().copied());
    log::info!("{}", reduction_line(label, &samples, stats));
    Ok(Measurement {
        label: label.to_string(),
        reps: reps.into_iter().map(|(rep, _)| rep).collect(),
        stats,
    })
}

/// A completed run: its repetitions, and the reduction of the metric it
/// sampled.
///
/// The reduction is the harness's rather than something a cell computes from
/// the reps, because the statistic is as much of the methodology as the rep
/// count is: a cell picks *which* metric it measures, never *how* it is
/// reduced.
#[derive(Debug)]
pub struct Measurement<T> {
    /// The cell name [`run`] logged under, kept so a second metric reduced off
    /// the same reps reports under the same cell.
    label: String,
    reps: Vec<Rep<T>>,
    stats: Stats,
}

/// A metric's central value and spread across a run, in milliseconds — which
/// is the unit every sample a cell reports must already be in.
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

    /// The reduction of a *second* metric off the same run, for a cell that
    /// reports more than one number from one set of reps.
    ///
    /// `name` is the metric, which [`run`] never has to name because the cell's
    /// headline metric is the cell. Its samples are logged the same way, so a
    /// second reported number is no less traceable than the first.
    pub fn metric(&self, name: &str, sample: impl Fn(&Rep<T>) -> f64) -> Stats {
        let samples = self.reps.iter().map(sample).collect::<Vec<_>>();
        let stats = mean_stddev(samples.iter().copied());
        log::info!(
            "{}",
            reduction_line(&format!("{} {name}", self.label), &samples, stats)
        );
        stats
    }

    pub fn iter(&self) -> std::slice::Iter<'_, Rep<T>> {
        self.reps.iter()
    }

    /// The first rep's result — what a cell reads a per-run constant from,
    /// such as the prompt-token count every rep shares.
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
/// `(0.0, 0.0)`; a single sample has no spread, so `(mean, 0.0)`.
///
/// Crate-private: a caller reduces through [`Measurement`], which is what keeps
/// the statistic uniform across every benchmark and client. The spread needs a
/// second pass over the samples, hence the collect.
fn mean_stddev(samples: impl IntoIterator<Item = f64>) -> Stats {
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

/// The samples a metric collected and what they reduced to, on one line.
///
/// Both halves together, because either alone is the thing that gets
/// misread: a mean with no samples hides the outlier that produced it, and
/// samples with no mean leave the reader arithmetic to do against the number
/// that was actually submitted.
fn reduction_line(label: &str, samples: &[f64], stats: Stats) -> String {
    let values = samples
        .iter()
        .map(|sample| format!("{sample:.3}"))
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "{label}: samples [{values}] ms -> mean {:.3} ms stddev {:.3} ms",
        stats.mean_ms, stats.stddev_ms
    )
}

/// A duration as the result schema stores it. One conversion so a rep's wall
/// clock means the same thing on every path.
fn duration_ms(duration: Duration) -> f64 {
    duration.as_nanos() as f64 / 1_000_000.0
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use rstest::rstest;

    use super::*;

    /// Records what a run reported, in the order it was reported, so a test
    /// can assert the interleaving rather than just the counts.
    #[derive(Debug, PartialEq)]
    enum Event {
        Gate,
        Started,
        Prepared(usize),
        Work(usize),
        Finished,
        Sampled(usize),
    }

    /// Drives `work` and `sample` through a harness whose gate and observer
    /// append to one log, and returns that log alongside the result.
    fn record<T>(
        work: impl FnMut(usize) -> anyhow::Result<T>,
        sample: impl FnMut(usize, &Rep<T>) -> anyhow::Result<f64>,
    ) -> (anyhow::Result<Measurement<T>>, Vec<Event>) {
        let log = RefCell::new(Vec::new());
        // Scoped so the hooks — which borrow `log` for as long as they live —
        // are gone before the recording is read back.
        let result = {
            let gate = || {
                log.borrow_mut().push(Event::Gate);
                Ok(())
            };
            let observer = RepObserver::new(
                || {
                    log.borrow_mut().push(Event::Started);
                    Ok(())
                },
                || {
                    log.borrow_mut().push(Event::Finished);
                    Ok(())
                },
            );
            let (mut work, mut sample) = (work, sample);
            run(
                "test",
                &gate,
                &observer,
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

    /// The capture contract: wait, report the start, run, report the end —
    /// once per rep, with the sample read after the bracket closes.
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

    /// A failed rep abandons the cell rather than reporting an end it never
    /// reached or measuring the reps after it.
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

    /// A rep the cell rejects stops the run at that rep — the reps after it
    /// are never even started, so a misbehaving server is not measured five
    /// times before anyone says so.
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

    /// The gate's wait is not the rep's, so it stays out of `elapsed`.
    #[test]
    fn the_readiness_wait_is_not_timed() -> anyhow::Result<()> {
        let wait = Duration::from_millis(25);
        let gate = || {
            std::thread::sleep(wait);
            Ok(())
        };
        let observer = RepObserver::new(|| Ok(()), || Ok(()));

        let measured = run(
            "test",
            &gate,
            &observer,
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

    /// The sampled metric is what gets reduced, not the wall clock.
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

    /// A second metric reduces off the same reps, and reports under the cell
    /// [`run`] was given.
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

    /// What a reader is shown: every sample, in the order it was collected,
    /// next to the mean it produced.
    #[test]
    fn reports_the_samples_behind_the_mean() {
        let samples = [2.0, 4.0, 4.0, 4.0, 5.0];

        let line = reduction_line("prefill_throughput", &samples, mean_stddev(samples));

        assert_eq!(
            line,
            "prefill_throughput: samples [2.000, 4.000, 4.000, 4.000, 5.000] ms \
             -> mean 3.800 ms stddev 1.095 ms"
        );
    }

    /// A cell that collected nothing must not read as one that measured zero.
    #[test]
    fn reports_an_empty_series_as_empty() {
        let line = reduction_line("decode_throughput", &[], mean_stddev([]));

        assert!(line.contains("samples [] ms"), "{line}");
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
}
