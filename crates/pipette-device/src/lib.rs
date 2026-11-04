//! What host is this, and what state is it in?
//!
//! Identity is [`DeviceInfo`](pipette_plan_types::DeviceInfo), detected once
//! per run by [`detect_device_info`]. State is the power state, read once per
//! run, and the thermal readings and each OS facility's own records
//! ([`DeviceReadings`]), sampled at each timed rep's two ends. The other
//! reading types live in
//! `pipette-plan-types`; this crate owns the probes.
//! [`MemoryPressure`] logs the host's memory pressure at a cell's start and
//! each rep's edges; no row carries it. On Linux and Android,
//! `drop_cached_files` drops a directory's page cache before a cell, so an
//! earlier model's file does not crowd the next model's load.

mod capabilities;
#[cfg(target_os = "macos")]
mod die_temp;
mod memory_pressure;
#[cfg(any(target_os = "linux", target_os = "android"))]
mod page_cache;
mod probe;
mod readings;

pub use capabilities::hardware_capabilities;
#[cfg(target_os = "macos")]
pub use die_temp::die_temp_max_c;
pub use memory_pressure::MemoryPressure;
#[cfg(any(target_os = "linux", target_os = "android"))]
pub use page_cache::{DroppedCache, drop_cached_files};
pub use probe::{CPU_ARCH, OS_FAMILY, detect_device_info, detect_power_state, detect_thermal};
#[cfg(target_os = "windows")]
pub use readings::gpu_temp_max_c;
pub use readings::{DeviceReadings, detect_device_readings};

#[cfg(target_os = "android")]
pub use probe::android::android_hal_sensors;
pub use probe::android_thermal_status_from_code;
#[cfg(target_os = "android")]
pub use probe::boot_reason;
