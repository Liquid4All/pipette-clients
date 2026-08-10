//! Linux readiness, split by board type.
//!
//! `cfg(target_os = "linux")` selects this module; here we branch on the
//! runtime [`Board`]. A Raspberry Pi 5 ([`pi5`]) follows the SoC
//! firmware's throttle rules; every other Linux host takes the generic
//! thermal-zone + load gate ([`generic`]) and never shells out to
//! `vcgencmd`. Shared probes and the poll loop live in [`common`]. Add a
//! board: a `Board` variant + a sibling module + one match arm here —
//! the existing types stay untouched.

mod board;
mod common;
mod generic;
mod pi5;

use std::time::Duration;

use self::board::Board;
use crate::{ReadinessError, ThermalGate};

/// Default deadline for the wait: the shared cross-platform value (see
/// [`crate::DEFAULT_MAX_WAIT`]). On a hot or busy host the wait may need minutes
/// to clear, and 5 minutes is enough for most cases without blocking an
/// interactive operator forever. Same number as before, now stated once.
pub(super) const DEFAULT_MAX_WAIT: Duration = crate::DEFAULT_MAX_WAIT;

pub(super) fn wait_until_ready(
    max_wait: Duration,
    thermal: ThermalGate,
) -> Result<(), ReadinessError> {
    match Board::current() {
        Board::RaspberryPi5 => pi5::wait_until_ready(max_wait, thermal),
        Board::Other => generic::wait_until_ready(max_wait, thermal),
    }
}
