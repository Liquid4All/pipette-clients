//! Fallback for platforms we don't have readiness probes for —
//! everything other than Android, Linux, macOS, and Windows. The wait
//! is a true no-op: returns `Ok(())` immediately. Operators running
//! on these platforms get the throughput-class benchmarks back-to-back;
//! if that produces unstable numbers, the right fix is to add probes
//! for that target rather than to special-case it at the call sites.

use std::time::Duration;

use super::{ReadinessError, ThermalGate};

/// No deadline meaningful on a no-probe platform — kept so the
/// per-OS `probe::DEFAULT_MAX_WAIT` lookup is uniform across targets.
pub(super) const DEFAULT_MAX_WAIT: Duration = Duration::ZERO;

pub(super) fn wait_until_ready(
    _max_wait: Duration,
    _thermal: ThermalGate,
) -> Result<(), ReadinessError> {
    Ok(())
}
