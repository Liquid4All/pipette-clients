//! Fallback for targets without readiness probes: the wait returns at once.
//! If that yields unstable numbers, add probes for the target rather than
//! special-casing it at the call sites.

use std::time::Duration;

use super::{
    ReadinessCallbacks, ReadinessError, ReadinessPhase, ReadinessPlatform, ReadinessStatus,
    ThermalGate,
};

/// Meaningless here; exists so `platform_default_max_wait` is uniform.
pub(super) const DEFAULT_MAX_WAIT: Duration = Duration::ZERO;

pub(super) fn wait_until_ready(
    max_wait: Duration,
    _thermal: ThermalGate,
    callbacks: &ReadinessCallbacks<'_>,
) -> Result<(), ReadinessError> {
    // Reported anyway: a renderer expects a terminal frame per wait, and this
    // is the only record that the gate ran and checked nothing.
    callbacks.report(&ReadinessStatus {
        phase: ReadinessPhase::Proceeding,
        platform: ReadinessPlatform::Other,
        observed: "no readiness probes on this platform".to_string(),
        elapsed: Duration::ZERO,
        max_wait,
    });
    Ok(())
}
