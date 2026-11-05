//! Each OS facility's own records, read at a timed rep's two ends for the
//! submission's `device_hardware_readings` and `device_os_throttling_readings`
//! (mgmt's `docs/design/device-telemetry.md`).
//!
//! Unlike [`ThermalReading`](pipette_plan_types::ThermalReading), nothing is
//! converted: each facility is a key holding the records it returned, in the
//! OS's names, units and codes, and one that failed or does not exist is absent.
//!
//! Only Windows reads anything yet; the other platforms keep their thermal
//! fields until they get a reader here.

use serde_json::{Map, Value};

// In every test build too, so the grouping is tested on every host.
#[cfg(any(target_os = "windows", test))]
mod windows;

#[cfg(target_os = "windows")]
pub use windows::gpu::gpu_temp_max_c;

/// One sample point's readings, split as the submission splits them.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DeviceReadings {
    /// What the device's sensors reported, by facility.
    pub hardware: Map<String, Value>,
    /// What the OS or firmware reported about its own throttling, by facility.
    pub os_throttling: Map<String, Value>,
}

/// This host's readings now; empty where the platform has no reader yet.
/// About a second on Windows, whose rate counters need a 1 s window, so call it
/// outside a timed window.
pub fn detect_device_readings() -> DeviceReadings {
    #[cfg(target_os = "windows")]
    {
        windows::read()
    }
    #[cfg(not(target_os = "windows"))]
    {
        DeviceReadings::default()
    }
}

#[cfg(all(test, not(target_os = "windows")))]
mod tests {
    use super::*;

    #[test]
    fn other_platforms_read_nothing_yet() {
        assert_eq!(detect_device_readings(), DeviceReadings::default());
    }
}
