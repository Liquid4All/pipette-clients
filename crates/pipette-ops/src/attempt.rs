//! What every client shares about `attempt_timeout_seconds`; each runs its own
//! clock through [`MeasurementObserver`](crate::measurement::MeasurementObserver).

use std::time::Duration;

use pipette_plan_types::{BenchmarkFlags, BenchmarkType, RunRequest};

/// The limit on a cell whose flags set none, on the kinds that take a default.
pub const DEFAULT_LIMIT: Duration = Duration::from_secs(300);

/// What a client's clock names the first attempt: the model load counts
/// against it.
pub const LOAD_AND_WARM_UP: &str = "the model load and warm-up";

/// The cell's limit: its flags' `attempt_timeout_seconds`, else the default for
/// its kind. `None` leaves it unbounded.
pub fn limit(req: &RunRequest) -> Option<Duration> {
    req.benchmark_flags
        .as_ref()
        .and_then(BenchmarkFlags::attempt_timeout)
        .map(|secs| Duration::from_secs(secs.into_inner()))
        .or_else(|| default_limit(req.benchmark.benchmark_type()))
}

/// Exhaustive, so a new kind has to say whether it gets one.
pub fn default_limit(benchmark: BenchmarkType) -> Option<Duration> {
    match benchmark {
        BenchmarkType::PrefillThroughput
        | BenchmarkType::DecodeThroughput
        | BenchmarkType::EndToEndLatency
        | BenchmarkType::MaxMemoryUsage
        | BenchmarkType::VlThroughput
        | BenchmarkType::VlMaxMemory
        | BenchmarkType::AsrThroughput
        | BenchmarkType::AsrMaxMemory => Some(DEFAULT_LIMIT),
        // A corpus or an eval sets its own length, so a fixed default would cut
        // healthy runs.
        BenchmarkType::VlSpecDecode
        | BenchmarkType::Eval
        | BenchmarkType::VlmEval
        | BenchmarkType::AsrEval => None,
    }
}

/// An attempt that ran past the limit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Overrun {
    pub what: String,
    pub limit: Duration,
    pub elapsed: Duration,
}

/// The failure a run past its limit reports, final for every client.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub struct AttemptTimeoutError {
    pub overrun: Overrun,
    /// What the engine failed with once its children were stopped.
    pub engine_error: Option<String>,
}

impl std::fmt::Display for AttemptTimeoutError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let Overrun {
            what,
            limit,
            elapsed,
        } = &self.overrun;
        write!(
            f,
            "{what} ran {:.1}s, past attempt_timeout_seconds ({}s), and was stopped",
            elapsed.as_secs_f64(),
            limit.as_secs()
        )?;
        if let Some(engine) = &self.engine_error {
            write!(f, "; the engine then reported: {engine}")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_failure_names_the_attempt_and_the_limit() -> anyhow::Result<()> {
        let failure = AttemptTimeoutError {
            overrun: Overrun {
                what: "rep 3 of 5".into(),
                limit: Duration::from_secs(300),
                elapsed: Duration::from_millis(300_400),
            },
            engine_error: Some("llama-bench was killed".into()),
        };
        let text = failure.to_string();
        anyhow::ensure!(
            text == "rep 3 of 5 ran 300.4s, past attempt_timeout_seconds (300s), and was \
                     stopped; the engine then reported: llama-bench was killed",
            "{text}"
        );
        Ok(())
    }

    #[test]
    fn only_the_timing_and_memory_kinds_default() -> anyhow::Result<()> {
        anyhow::ensure!(default_limit(BenchmarkType::DecodeThroughput) == Some(DEFAULT_LIMIT));
        anyhow::ensure!(default_limit(BenchmarkType::VlMaxMemory) == Some(DEFAULT_LIMIT));
        anyhow::ensure!(default_limit(BenchmarkType::VlSpecDecode).is_none());
        anyhow::ensure!(default_limit(BenchmarkType::Eval).is_none());
        Ok(())
    }
}
