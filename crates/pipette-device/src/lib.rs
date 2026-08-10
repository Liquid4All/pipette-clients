//! What host is this, and what state is it in?
//!
//! Two questions with one set of per-platform probes behind them:
//!
//! - **Identity** — [`DeviceInfo`](pipette_plan_types::device::DeviceInfo), detected once per run by
//!   [`detect_device_info`] and carried on every submission.
//! - **State** — the power state and thermal readings
//!   ([`ThermalReading`](pipette_plan_types::thermal::ThermalReading),
//!   [`RunThermal`](pipette_plan_types::thermal::RunThermal),
//!   [`ThermalTelemetry`](pipette_plan_types::thermal::ThermalTelemetry)) sampled
//!   around a timed run.
//!
//! The leaf vocabulary the readings are spelled in (`AppleThermalState`,
//! `LinuxThermalZone`, `AndroidTemperatureSensor`, `DevicePowerState`) lives
//! in `pipette-plan-types`; this crate owns the probes and the composites
//! over them.
//!
//! The API is flat: consumers reference `pipette_device::detect_thermal`,
//! `pipette_plan_types::device::DeviceInfo` etc. without seeing the submodules.

#[cfg(target_os = "macos")]
mod die_temp;
mod probe;

#[cfg(target_os = "macos")]
pub use die_temp::die_temp_max_c;
pub use probe::{detect_device_info, detect_power_state, detect_thermal};

#[cfg(target_os = "android")]
pub use probe::android_hal_sensors;
pub use probe::android_thermal_status_from_code;
