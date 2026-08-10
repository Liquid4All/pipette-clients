//! The per-rep thermal readings a run collects ([`ThermalSeries`]).
//!
//! A measurement reports the two ends of each repetition; something has to
//! hold the readings. That belongs beside the reports rather than in each
//! caller, because the pairing is the part that is easy to get wrong: two
//! independent `before`/`after` vectors drift the moment one rep contributes
//! to one and not the other, and every later reading lands on the wrong
//! iteration with nothing to say so.
//!
//! See `pipette-mgmt` `docs/methodology/thermal-telemetry.md`: array position
//! *is* the iteration index, so alignment is the contract.

use anyhow::Context;

use pipette_plan_types::thermal::ThermalReading;

/// Collects each repetition's two readings, paired.
///
/// [`Self::start`] opens a repetition and [`Self::finish`] closes it, so a
/// rep's two readings can only be stored together. A rep whose work fails
/// never reaches its end — the run is abandoned — so in a run that completes,
/// every report is matched. A sequence that isn't is a caller misreporting its
/// reps, and is refused rather than silently misaligned.
#[derive(Debug, Default)]
pub struct ThermalSeries {
    /// The `(start, end)` readings of each repetition that has finished, in
    /// rep order — the series the result carries.
    finished_reps: Vec<(ThermalReading, ThermalReading)>,
    /// The start reading of the repetition currently in flight, held until its
    /// end arrives. `None` between repetitions.
    open_rep_start: Option<ThermalReading>,
}

impl ThermalSeries {
    pub fn start(&mut self, reading: ThermalReading) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.open_rep_start.is_none(),
            "a repetition was started while the previous one was still open"
        );
        self.open_rep_start = Some(reading);
        Ok(())
    }

    pub fn finish(&mut self, reading: ThermalReading) -> anyhow::Result<()> {
        let start = self
            .open_rep_start
            .take()
            .context("a repetition ended that was never started")?;
        self.finished_reps.push((start, reading));
        Ok(())
    }

    /// The collected pairs, leaving the collector empty. Drains in place
    /// because the hooks that fill it borrow it for as long as they live.
    ///
    /// A run that ends with a repetition still open reported a start and never
    /// its end — the same misreporting the two above refuse, but the only
    /// shape they cannot see, since nothing follows it.
    pub fn take(&mut self) -> anyhow::Result<Vec<(ThermalReading, ThermalReading)>> {
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

    /// Opening a second repetition without closing the first would pair the
    /// wrong two readings, so it fails the run instead.
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

    /// A start with no end is invisible to the two reports — nothing follows
    /// it — so the drain is where that run has to fail.
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
