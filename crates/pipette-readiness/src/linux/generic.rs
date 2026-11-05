//! Readiness for any Linux host without board-specific throttling rules:
//! hottest kernel thermal zone plus CPU load, no vendor tools.

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use super::common::{self, CpuLoad};
use crate::wait::{Probe, wait_loop};
use crate::{
    CpuPercent, ReadinessCallbacks, ReadinessConfiguration, ReadinessError, ReadinessPlatform,
    ThermalGate,
};

/// Set once the sensorless-host warning is written. The fact is fixed for the
/// process and the gate polls every few seconds, so a warning per poll would
/// fill a fleet log with one unchanging line.
static SENSORLESS_WARNED: AtomicBool = AtomicBool::new(false);

/// "Cool" zone temp threshold. 70 °C leaves headroom over a 30–50 °C idle
/// baseline while staying well below typical throttle thresholds
/// (95–100 °C on x86, ~110 °C on ARM SoCs).
const THERMAL_THRESHOLD_C: i32 = 70;

/// What the gate compares against: its own numbers, or the ones the host's
/// `[readiness.linux]` table sets.
#[derive(Debug, Clone, Copy)]
struct Bounds {
    hottest_below_c: i32,
    /// The host set `hottest_below_c` itself, so a host with no thermal zone
    /// fails the wait instead of gating on load alone.
    hottest_set: bool,
    cpu_load_below: f64,
}

impl Bounds {
    fn new(configuration: Option<&ReadinessConfiguration>) -> Self {
        let config = configuration.and_then(ReadinessConfiguration::linux);
        let hottest = config.and_then(|c| c.hottest_below_c);
        let load = config.and_then(|c| c.cpu_load_below_pct);
        Self {
            hottest_below_c: hottest.map_or(THERMAL_THRESHOLD_C, |c| c.get().cast_signed()),
            hottest_set: hottest.is_some(),
            cpu_load_below: load.map_or(common::CPU_LOAD_THRESHOLD, CpuPercent::ratio),
        }
    }

    /// An absent reading passes: under a waived gate that is the waiver, under
    /// an enforced one the sensorless host of [`thermal_reading`]. Load still
    /// applies.
    fn temp_ok(self, hottest_c: Option<i32>, thermal: ThermalGate) -> bool {
        thermal == ThermalGate::Skip || hottest_c.is_none_or(|c| c < self.hottest_below_c)
    }

    fn ready(self, hottest_c: Option<i32>, cpu_load: f64, thermal: ThermalGate) -> bool {
        self.temp_ok(hottest_c, thermal) && cpu_load < self.cpu_load_below
    }
}

pub(super) fn wait_until_ready(
    max_wait: Duration,
    thermal: ThermalGate,
    configuration: Option<&ReadinessConfiguration>,
    callbacks: &ReadinessCallbacks<'_>,
) -> Result<(), ReadinessError> {
    let bounds = Bounds::new(configuration);
    wait_loop(ReadinessPlatform::Linux, max_wait, callbacks, || {
        let hottest = thermal_reading(
            common::read_hottest_zone_celsius(),
            thermal,
            bounds.hottest_set,
        )?;
        let cpu = common::read_cpu_load()?;
        Ok(Probe {
            ready: bounds.ready(hottest, cpu.ratio(), thermal),
            summary: format_summary(hottest, cpu, thermal, bounds),
        })
    })
}

/// The temperature the gate acts on, or `None` when it has none.
///
/// The zones are read even when the gate is skipped so the log records the
/// temperature; a skipped gate therefore survives any probe failure.
///
/// A host with no `thermal_zone*` at all passes an enforced gate on load alone,
/// unless its table sets `hottest_below_c` (`bound_set`), which is then an
/// error. Rented VMs often expose none (Lambda H100 boxes: 26 cooling devices,
/// zero zones), and enforcing would fail every cell before the first sample.
/// Zones that exist and do not read still fail an enforced gate: an unknown
/// temperature is not an absent one.
fn thermal_reading(
    probe: Result<i32, ReadinessError>,
    thermal: ThermalGate,
    bound_set: bool,
) -> Result<Option<i32>, ReadinessError> {
    let err = match probe {
        Ok(hottest) => return Ok(Some(hottest)),
        Err(err) => err,
    };
    let sensorless = matches!(err, ReadinessError::Unavailable(common::NO_THERMAL_ZONES));
    // `resolve_readiness` decides `Unset` before a platform sees it, and it
    // means "enforce unless waived", so it reads as enforcing.
    let enforced = thermal != ThermalGate::Skip;
    if enforced && !sensorless {
        return Err(err);
    }
    if enforced && bound_set {
        return Err(ReadinessError::BoundWithoutThermalZones);
    }
    if enforced {
        if !SENSORLESS_WARNED.swap(true, Ordering::Relaxed) {
            tracing::warn!(
                error = %err,
                "host exposes no thermal zones; the thermal criterion cannot be evaluated \
                 here, so every wait in this process gates on load alone"
            );
        }
    } else if matches!(
        err,
        ReadinessError::Unavailable(common::NO_READABLE_THERMAL_ZONES)
    ) {
        tracing::warn!(
            error = %err,
            "thermal zones are present and none read; the thermal gate is skipped for this \
             wait, so continuing without it"
        );
    } else {
        tracing::debug!(
            error = %err,
            "thermal zones unreadable; the thermal gate is skipped for this wait, so \
             continuing without it"
        );
    }
    Ok(None)
}

/// `hottest_c` is `None` when unreadable, printed apart from a reading: a run
/// that never learned its temperature is not the same evidence as a cool die.
/// An enforced gate reaches `None` only on a sensorless host, so that pair
/// prints `[NO SENSORS]`.
fn format_summary(
    hottest_c: Option<i32>,
    cpu: CpuLoad,
    thermal: ThermalGate,
    bounds: Bounds,
) -> String {
    let thermal_field = match hottest_c {
        Some(c) if c < bounds.hottest_below_c => format!("hottest:{c}°C(ok)"),
        Some(c) => format!("hottest:{c}°C(hot)"),
        None => "hottest:unreadable".to_string(),
    };
    let note = match (hottest_c, thermal) {
        (_, ThermalGate::Skip) => "[SKIPPED]",
        (None, ThermalGate::Enforce | ThermalGate::Unset) => "[NO SENSORS]",
        (Some(_), ThermalGate::Enforce | ThermalGate::Unset) => "",
    };
    format!(
        "thermal={thermal_field}{note} {}",
        common::cpu_part(cpu, bounds.cpu_load_below)
    )
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;
    use crate::configuration::table;

    fn sensorless() -> Result<i32, ReadinessError> {
        Err(ReadinessError::Unavailable(common::NO_THERMAL_ZONES))
    }

    fn unreadable() -> Result<i32, ReadinessError> {
        Err(ReadinessError::Unavailable(
            common::NO_READABLE_THERMAL_ZONES,
        ))
    }

    fn no_sysfs() -> Result<i32, ReadinessError> {
        Err(std::io::Error::from(std::io::ErrorKind::NotFound).into())
    }

    /// A waived gate outlives every probe failure; an enforced one only a
    /// sensorless host without `hottest_below_c`, where the criterion does not
    /// exist.
    #[rstest]
    #[case::sensorless_and_waived(sensorless(), ThermalGate::Skip, None, Ok(None))]
    #[case::sensorless_and_enforced(sensorless(), ThermalGate::Enforce, None, Ok(None))]
    #[case::sensorless_with_an_empty_table(sensorless(), ThermalGate::Enforce, Some(""), Ok(None))]
    #[case::sensorless_with_only_a_load_key(
        sensorless(),
        ThermalGate::Enforce,
        Some("cpu_load_below_pct = 50"),
        Ok(None)
    )]
    #[case::sensorless_with_a_bound(
        sensorless(),
        ThermalGate::Enforce,
        Some("hottest_below_c = 60"),
        Err("this host has no thermal zones")
    )]
    #[case::sensorless_with_a_waived_bound(
        sensorless(),
        ThermalGate::Skip,
        Some("hottest_below_c = 60"),
        Ok(None)
    )]
    #[case::flaky_sensor_and_waived(unreadable(), ThermalGate::Skip, None, Ok(None))]
    #[case::flaky_sensor_and_enforced(
        unreadable(),
        ThermalGate::Enforce,
        None,
        Err("none readable")
    )]
    #[case::flaky_sensor_with_a_bound(
        unreadable(),
        ThermalGate::Enforce,
        Some("hottest_below_c = 60"),
        Err("none readable")
    )]
    #[case::no_sysfs_and_waived(no_sysfs(), ThermalGate::Skip, None, Ok(None))]
    #[case::no_sysfs_and_enforced(no_sysfs(), ThermalGate::Enforce, None, Err("i/o failure"))]
    #[case::readable_and_waived(Ok(40), ThermalGate::Skip, None, Ok(Some(40)))]
    #[case::readable_and_enforced(Ok(40), ThermalGate::Enforce, None, Ok(Some(40)))]
    #[case::readable_with_a_bound(
        Ok(40),
        ThermalGate::Enforce,
        Some("hottest_below_c = 60"),
        Ok(Some(40))
    )]
    fn a_gate_outlives_a_probe_it_could_never_satisfy(
        #[case] probe: Result<i32, ReadinessError>,
        #[case] thermal: ThermalGate,
        #[case] keys: Option<&str>,
        #[case] expected: Result<Option<i32>, &str>,
    ) -> anyhow::Result<()> {
        let config = keys
            .map(|keys| table(ReadinessPlatform::Linux, keys))
            .transpose()?;
        let bound_set = Bounds::new(config.as_ref()).hottest_set;
        match (thermal_reading(probe, thermal, bound_set), expected) {
            (Ok(got), Ok(want)) => assert_eq!(got, want),
            (Err(got), Err(want)) => assert!(got.to_string().contains(want), "{got}"),
            (got, want) => anyhow::bail!("got {got:?}, want {want:?}"),
        }
        Ok(())
    }

    /// Each key replaces its own default alone, under the same strict
    /// comparison.
    #[rstest]
    // No table, an empty table and the restated defaults gate alike.
    #[case::own_numbers(None, ThermalGate::Enforce, Some(69), 0.29, true)]
    #[case::own_temp_holds(None, ThermalGate::Enforce, Some(70), 0.10, false)]
    #[case::own_load_holds(None, ThermalGate::Enforce, Some(40), 0.30, false)]
    #[case::empty_table(Some(""), ThermalGate::Enforce, Some(70), 0.10, false)]
    #[case::restated_temp(
        Some("hottest_below_c = 70\ncpu_load_below_pct = 30"),
        ThermalGate::Enforce,
        Some(70),
        0.10,
        false
    )]
    #[case::restated_load(
        Some("hottest_below_c = 70\ncpu_load_below_pct = 30"),
        ThermalGate::Enforce,
        Some(40),
        0.30,
        false
    )]
    #[case::temp_raised(
        Some("hottest_below_c = 80"),
        ThermalGate::Enforce,
        Some(75),
        0.10,
        true
    )]
    #[case::temp_raised_is_strict(
        Some("hottest_below_c = 80"),
        ThermalGate::Enforce,
        Some(80),
        0.10,
        false
    )]
    #[case::temp_lowered(
        Some("hottest_below_c = 60"),
        ThermalGate::Enforce,
        Some(65),
        0.10,
        false
    )]
    #[case::temp_key_leaves_load(
        Some("hottest_below_c = 80"),
        ThermalGate::Enforce,
        Some(40),
        0.40,
        false
    )]
    #[case::load_raised(
        Some("cpu_load_below_pct = 50"),
        ThermalGate::Enforce,
        Some(40),
        0.40,
        true
    )]
    #[case::load_raised_is_strict(
        Some("cpu_load_below_pct = 50"),
        ThermalGate::Enforce,
        Some(40),
        0.50,
        false
    )]
    #[case::load_lowered(
        Some("cpu_load_below_pct = 5"),
        ThermalGate::Enforce,
        Some(40),
        0.10,
        false
    )]
    #[case::load_key_leaves_temp(
        Some("cpu_load_below_pct = 50"),
        ThermalGate::Enforce,
        Some(75),
        0.10,
        false
    )]
    // Skipping thermal skips the temperature bound, set or not; load stays.
    #[case::skip_ignores_temp(
        Some("hottest_below_c = 60"),
        ThermalGate::Skip,
        Some(95),
        0.10,
        true
    )]
    #[case::skip_keeps_load(
        Some("cpu_load_below_pct = 50"),
        ThermalGate::Skip,
        Some(95),
        0.60,
        false
    )]
    fn the_hosts_table_moves_the_gate(
        #[case] keys: Option<&str>,
        #[case] thermal: ThermalGate,
        #[case] hottest_c: Option<i32>,
        #[case] cpu_load: f64,
        #[case] want_ready: bool,
    ) -> Result<(), toml::de::Error> {
        let config = keys
            .map(|keys| table(ReadinessPlatform::Linux, keys))
            .transpose()?;
        let bounds = Bounds::new(config.as_ref());
        assert_eq!(bounds.ready(hottest_c, cpu_load, thermal), want_ready);
        Ok(())
    }

    #[rstest]
    #[case::cool_and_enforced(Some(40), ThermalGate::Enforce, true)]
    #[case::hot_and_enforced(Some(95), ThermalGate::Enforce, false)]
    #[case::hot_but_waived(Some(95), ThermalGate::Skip, true)]
    #[case::sensorless_and_enforced(None, ThermalGate::Enforce, true)]
    #[case::sensorless_and_waived(None, ThermalGate::Skip, true)]
    fn a_missing_reading_passes_the_temperature_criterion(
        #[case] hottest_c: Option<i32>,
        #[case] thermal: ThermalGate,
        #[case] expected: bool,
    ) {
        assert_eq!(Bounds::new(None).temp_ok(hottest_c, thermal), expected);
    }

    /// A fleet log must tell a gated, a waived and a sensorless wait apart. A
    /// table's numbers decide the verdicts.
    #[rstest]
    #[case::cool(None, Some(40), 0.1, ThermalGate::Enforce, "hottest:40°C(ok)")]
    #[case::hot(None, Some(95), 0.1, ThermalGate::Enforce, "hottest:95°C(hot)")]
    #[case::cool_but_waived(None, Some(40), 0.1, ThermalGate::Skip, "hottest:40°C(ok)[SKIPPED]")]
    #[case::no_sensor_waived(None, None, 0.1, ThermalGate::Skip, "hottest:unreadable[SKIPPED]")]
    #[case::no_sensor_enforced(
        None,
        None,
        0.1,
        ThermalGate::Enforce,
        "hottest:unreadable[NO SENSORS]"
    )]
    #[case::under_the_table(
        Some("hottest_below_c = 80\ncpu_load_below_pct = 50"),
        Some(75),
        0.40,
        ThermalGate::Enforce,
        "thermal=hottest:75°C(ok) cpu=load:40%(ok)"
    )]
    #[case::at_the_table(
        Some("hottest_below_c = 80\ncpu_load_below_pct = 50"),
        Some(80),
        0.50,
        ThermalGate::Enforce,
        "thermal=hottest:80°C(hot) cpu=load:50%(busy)"
    )]
    fn summary_states_the_thermal_evidence(
        #[case] keys: Option<&str>,
        #[case] hottest_c: Option<i32>,
        #[case] cpu_load: f64,
        #[case] thermal: ThermalGate,
        #[case] expected: &str,
    ) -> Result<(), toml::de::Error> {
        let config = keys
            .map(|keys| table(ReadinessPlatform::Linux, keys))
            .transpose()?;
        let summary = format_summary(
            hottest_c,
            CpuLoad::LoadAverage(cpu_load),
            thermal,
            Bounds::new(config.as_ref()),
        );
        assert!(
            summary.contains(expected),
            "summary {summary:?} should hold {expected:?}"
        );
        Ok(())
    }

    /// The table's load bound judges a busy share as it does a load average.
    #[rstest]
    #[case::unconfined(None, CpuLoad::LoadAverage(0.02), "cpu=load:2%(ok)")]
    #[case::confined(None, CpuLoad::BusyShare(0.02), "cpu=busy:2%(ok)")]
    #[case::confined_under_the_table(
        Some("cpu_load_below_pct = 50"),
        CpuLoad::BusyShare(0.40),
        "cpu=busy:40%(ok)"
    )]
    #[case::confined_at_the_table(
        Some("cpu_load_below_pct = 50"),
        CpuLoad::BusyShare(0.50),
        "cpu=busy:50%(busy)"
    )]
    fn summary_names_the_cpu_reading(
        #[case] keys: Option<&str>,
        #[case] cpu: CpuLoad,
        #[case] expected: &str,
    ) -> Result<(), toml::de::Error> {
        let config = keys
            .map(|keys| table(ReadinessPlatform::Linux, keys))
            .transpose()?;
        let bounds = Bounds::new(config.as_ref());
        let summary = format_summary(Some(40), cpu, ThermalGate::Enforce, bounds);
        assert_eq!(summary, format!("thermal=hottest:40°C(ok) {expected}"));
        Ok(())
    }
}
