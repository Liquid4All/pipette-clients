//! Linux readiness, branching on the runtime [`Board`]: a Raspberry Pi 5
//! ([`pi5`]) follows the firmware's throttle bits, a Jetson Orin ([`orin`])
//! adds GPU load, kernel throttling and over-current to the thermal gate, and
//! every other host takes the thermal-zone and load gate ([`generic`]). Shared
//! probes live in [`common`].

mod board;
mod common;
mod generic;
mod orin;
mod pi5;

use std::time::Duration;

use self::board::Board;
use crate::{
    ReadinessCallbacks, ReadinessConfiguration, ReadinessError, ReadinessPlatform, ThermalGate,
};

pub(super) const DEFAULT_MAX_WAIT: Duration = crate::DEFAULT_MAX_WAIT;

/// The gate [`wait_until_ready`] runs on this board.
pub(super) fn platform() -> ReadinessPlatform {
    Board::current().platform()
}

pub(super) fn wait_until_ready(
    max_wait: Duration,
    thermal: ThermalGate,
    configuration: Option<&ReadinessConfiguration>,
    callbacks: &ReadinessCallbacks<'_>,
) -> Result<(), ReadinessError> {
    match Board::current() {
        Board::RaspberryPi5 => pi5::wait_until_ready(max_wait, thermal, configuration, callbacks),
        Board::JetsonOrin => orin::wait_until_ready(max_wait, thermal, configuration, callbacks),
        Board::Other => generic::wait_until_ready(max_wait, thermal, configuration, callbacks),
    }
}
