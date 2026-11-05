//! The per-rep thermal readings a run collects ([`ThermalSeries`]).
//!
//! Readings are stored as pairs because two independent `before`/`after`
//! vectors drift the moment one rep contributes to one and not the other.
//! Array position is the iteration index, so alignment is the contract.

use anyhow::Context;

use pipette_plan_types::ThermalReading;

/// Collects each repetition's two readings, paired.
///
/// A rep whose work fails abandons the run, so in a run that completes every
/// start has its finish; an unmatched sequence is a caller bug and is refused
/// rather than silently misaligned. Generic so a caller can pair anything it
/// samples at a rep's two ends.
#[derive(Debug)]
pub struct ThermalSeries<R = ThermalReading> {
    finished_reps: Vec<(R, R)>,
    open_rep_start: Option<R>,
}

impl<R> Default for ThermalSeries<R> {
    fn default() -> Self {
        Self {
            finished_reps: Vec::new(),
            open_rep_start: None,
        }
    }
}

impl<R> ThermalSeries<R> {
    pub fn start(&mut self, reading: R) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.open_rep_start.is_none(),
            "a repetition was started while the previous one was still open"
        );
        self.open_rep_start = Some(reading);
        Ok(())
    }

    pub fn finish(&mut self, reading: R) -> anyhow::Result<()> {
        let start = self
            .open_rep_start
            .take()
            .context("a repetition ended that was never started")?;
        self.finished_reps.push((start, reading));
        Ok(())
    }

    /// The collected pairs, leaving the collector empty. Drains in place
    /// because the hooks that fill it borrow it for as long as they live.
    pub fn take(&mut self) -> anyhow::Result<Vec<(R, R)>> {
        anyhow::ensure!(
            self.open_rep_start.is_none(),
            "the run finished with a repetition still open"
        );
        Ok(std::mem::take(&mut self.finished_reps))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_second_start_without_an_end_is_refused() -> anyhow::Result<()> {
        let mut series = ThermalSeries::default();
        series.start(ThermalReading::default())?;

        assert!(series.start(ThermalReading::default()).is_err());
        Ok(())
    }

    #[test]
    fn an_end_without_a_start_is_refused() {
        let mut series = ThermalSeries::default();

        assert!(series.finish(ThermalReading::default()).is_err());
    }

    #[test]
    fn matched_reports_pair_up() -> anyhow::Result<()> {
        let mut series = ThermalSeries::default();
        (0..3).try_for_each(|_| {
            series.start(ThermalReading::default())?;
            series.finish(ThermalReading::default())
        })?;

        assert_eq!(series.take()?.len(), 3);
        Ok(())
    }

    #[test]
    fn a_run_that_ends_mid_repetition_is_refused() -> anyhow::Result<()> {
        let mut series = ThermalSeries::default();
        series.start(ThermalReading::default())?;
        series.finish(ThermalReading::default())?;
        series.start(ThermalReading::default())?;

        assert!(series.take().is_err());
        Ok(())
    }
}
