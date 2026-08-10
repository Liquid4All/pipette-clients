//! Shared Linux readiness probes and the poll loop.
//!
//! Every Linux board reads the kernel thermal zones and `/proc/loadavg`
//! and drives the same wait loop; only the per-board "ready" criteria and
//! summary string differ (see [`super::generic`] / [`super::pi5`]).

use std::fs;
use std::path::Path;
use std::time::{Duration, Instant};

use crate::ReadinessError;

/// Highest acceptable normalized 1-minute load. 0.30 means the box is
/// less than ~30% busy on average across all cores; benchmarks landing
/// at that point should see room.
pub(super) const CPU_LOAD_THRESHOLD: f64 = 0.30;

/// One probe iteration's verdict plus its human-readable summary.
pub(super) struct Probe {
    pub ready: bool,
    pub summary: String,
}

/// Poll `probe` every [`crate::SENSOR_POLL_INTERVAL`] until it reports ready or the
/// deadline passes. A timeout is surfaced as an error (not silently
/// ignored) so the caller fails the cell rather than record numbers taken
/// under non-steady conditions.
pub(super) fn wait_loop(
    max_wait: Duration,
    mut probe: impl FnMut() -> Result<Probe, ReadinessError>,
) -> Result<(), ReadinessError> {
    let deadline = Instant::now() + max_wait;
    loop {
        let Probe { ready, summary } = probe()?;
        if ready {
            log::info!("readiness: {summary} → proceeding");
            return Ok(());
        }
        if Instant::now() >= deadline {
            log::info!("readiness: {summary} → timed out after {max_wait:?}");
            return Err(ReadinessError::TimedOut {
                max_wait,
                observed: summary,
            });
        }
        log::info!(
            "readiness: {summary} → waiting {:?}",
            crate::SENSOR_POLL_INTERVAL
        );
        std::thread::sleep(crate::SENSOR_POLL_INTERVAL);
    }
}

/// Hottest reading across `/sys/class/thermal/thermal_zone*`, in °C.
pub(super) fn read_hottest_zone_celsius() -> Result<i32, ReadinessError> {
    let dir = fs::read_dir("/sys/class/thermal")?;
    dir.filter_map(Result::ok)
        .filter(|entry| {
            entry
                .file_name()
                .to_str()
                .is_some_and(|name| name.starts_with("thermal_zone"))
        })
        .filter_map(|entry| read_zone_celsius(&entry.path().join("temp")))
        .max()
        .ok_or(ReadinessError::Unavailable("no readable thermal zones"))
}

fn read_zone_celsius(temp_path: &Path) -> Option<i32> {
    let raw = fs::read_to_string(temp_path).ok()?;
    let millicelsius: i32 = raw.trim().parse().ok()?;
    Some(millicelsius / 1000)
}

/// Normalized 1-minute load average (loadavg / core count).
pub(super) fn read_cpu_load() -> Result<f64, ReadinessError> {
    let raw = fs::read_to_string("/proc/loadavg")?;
    let loadavg = parse_loadavg_first_field(&raw)?;
    let cores = std::thread::available_parallelism()
        .map(|n| n.get() as f64)
        .unwrap_or(1.0);
    Ok(loadavg / cores)
}

fn parse_loadavg_first_field(raw: &str) -> Result<f64, ReadinessError> {
    let first = raw
        .split_whitespace()
        .next()
        .ok_or(ReadinessError::Unavailable("empty /proc/loadavg"))?;
    first.parse().map_err(|_| ReadinessError::ParseFailure {
        raw: first.to_string(),
    })
}

/// Shared `cpu=load:NN%(verdict)` summary fragment.
pub(super) fn cpu_part(cpu_load: f64) -> String {
    let cpu_pct = (cpu_load * 100.0).round() as u32;
    let verdict = if cpu_load < CPU_LOAD_THRESHOLD {
        "ok"
    } else {
        "busy"
    };
    format!("cpu=load:{cpu_pct}%({verdict})")
}

#[cfg(test)]
mod tests {
    use anyhow::Context;
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case("0.10 0.20 0.30 1/123 4567", 0.10)]
    #[case("3.50 4.10 4.30 8/200 9999\n", 3.50)]
    fn loadavg_takes_first_field(
        #[case] raw: &str,
        #[case] want: f64,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let got = parse_loadavg_first_field(raw)?;
        assert!((got - want).abs() < 1e-9, "got {got}, want {want}");
        Ok(())
    }

    #[test]
    fn loadavg_empty_is_unavailable() -> anyhow::Result<()> {
        let err = parse_loadavg_first_field("")
            .err()
            .context("expected empty /proc/loadavg to be Unavailable")?;
        assert!(
            matches!(err, ReadinessError::Unavailable("empty /proc/loadavg")),
            "got {err:?}",
        );
        Ok(())
    }

    #[test]
    fn loadavg_garbage_is_parse_failure() -> anyhow::Result<()> {
        let err = parse_loadavg_first_field("notanumber 0.1 0.2")
            .err()
            .context("expected garbage loadavg to be a ParseFailure")?;
        assert!(
            matches!(err, ReadinessError::ParseFailure { ref raw } if raw == "notanumber"),
            "got {err:?}",
        );
        Ok(())
    }
}
