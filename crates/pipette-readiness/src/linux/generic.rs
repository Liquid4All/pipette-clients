//! Generic Linux readiness — hottest kernel thermal zone + 1-minute load
//! average, no vendor tools. Used for any Linux host without board-specific
//! throttling rules.

use std::time::Duration;

use super::common;
use crate::{ReadinessError, ThermalGate};

/// "Cool" zone temp threshold. 70 °C leaves headroom over a 30–50 °C idle
/// baseline while staying well below typical throttle thresholds
/// (95–100 °C on x86, ~110 °C on ARM SoCs).
const THERMAL_THRESHOLD_C: i32 = 70;

pub(super) fn wait_until_ready(
    max_wait: Duration,
    thermal: ThermalGate,
) -> Result<(), ReadinessError> {
    common::wait_loop(max_wait, || {
        let hottest = common::read_hottest_zone_celsius()?;
        let cpu_load = common::read_cpu_load()?;
        let temp_ok = thermal == ThermalGate::Skip || hottest < THERMAL_THRESHOLD_C;
        let ready = temp_ok && cpu_load < common::CPU_LOAD_THRESHOLD;
        Ok(common::Probe {
            ready,
            summary: format_summary(hottest, cpu_load),
        })
    })
}

fn format_summary(hottest_c: i32, cpu_load: f64) -> String {
    let verdict = if hottest_c < THERMAL_THRESHOLD_C {
        "ok"
    } else {
        "hot"
    };
    format!(
        "linux thermal=hottest:{hottest_c}°C({verdict}) {}",
        common::cpu_part(cpu_load)
    )
}
