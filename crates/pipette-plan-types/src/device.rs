//! What the host *is*: the identity fields every benchmark submission
//! carries, and the trimming newtype they are spelled in.
//!
//! Shape and wire contract only. `pipette_device::detect_device_info` produces
//! the values, and the mobile clients mirror these fields — iOS names them in
//! its payload builder, profile reporter and CSV exporter — which is why the
//! shape lives here rather than with the prober.

use nutype::nutype;
use serde::{Deserialize, Serialize};
use strum::{Display, EnumString};

/// A `String` that trims surrounding whitespace on every construction
/// or deserialization. Used for free-form metadata fields (device,
/// model, runtime identity) where some platform detectors (notably
/// Windows PowerShell `Win32_VideoController.Name`) emit padded values
/// that would otherwise reach the server as distinct rows from the
/// trimmed equivalent. Trim is intrinsic to the type — no caller needs
/// to remember to normalize before assigning.
#[nutype(
    sanitize(trim),
    default = "",
    derive(
        Debug,
        Clone,
        PartialEq,
        Eq,
        Hash,
        Default,
        Serialize,
        Deserialize,
        AsRef,
        Display,
        From,
    )
)]
pub struct TrimmedString(String);

/// Physical form factor of the device running the benchmark.
///
/// Serializes to lowercase (`"phone"`, `"laptop"`, etc.) to match the
/// management server API.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Display, EnumString)]
#[serde(rename_all = "lowercase")]
#[strum(serialize_all = "lowercase")]
pub enum DeviceFormFactor {
    Phone,
    Tablet,
    Laptop,
    Desktop,
    Server,
    Embedded,
}

impl Default for DeviceFormFactor {
    /// Returns `Desktop` as the fallback when deserializing old payloads that
    /// lack a `device_form_factor` field.  New payloads always set an explicit
    /// value via `pipette_device::detect_device_info`.
    fn default() -> Self {
        Self::Desktop
    }
}

/// Device metadata sent alongside every benchmark submission.
///
/// `device_name` and `device_form_factor` are user-provided (from
/// registration config).  The remaining required fields are auto-detected
/// by `pipette_device::detect_device_info`.
///
/// **Serde note**: every field carries `#[serde(default)]` so a
/// `payload.json` written before a field existed still deserializes. New
/// payloads always go through `detect_device_info` and populate all of them.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DeviceInfo {
    #[serde(default)]
    pub device_name: TrimmedString,
    #[serde(default)]
    pub device_form_factor: DeviceFormFactor,
    #[serde(default)]
    pub device_os_name: TrimmedString,
    #[serde(default)]
    pub device_os_version: TrimmedString,
    /// Precise OS build id, finer-grained than `device_os_version` (e.g. macOS
    /// `24F74`, Windows `26100.1234`, Linux kernel release). Optional — omitted
    /// from the wire payload when detection can't source it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device_os_build: Option<TrimmedString>,
    /// OS security-patch level where the platform exposes one (Android only for
    /// now). Optional; omitted otherwise.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device_os_security_patch: Option<TrimmedString>,
    #[serde(default)]
    pub device_chip_model: TrimmedString,
    #[serde(default)]
    pub device_ram_bytes: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device_gpu_model: Option<TrimmedString>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device_gpu_vram_bytes: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device_npu_model: Option<TrimmedString>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device_npu_vram_bytes: Option<u64>,
}

#[cfg(test)]
mod tests {
    use anyhow::Context;
    use rstest::rstest;

    use super::*;

    #[test]
    fn trimmed_string_trims_on_construction_and_deserialize() -> anyhow::Result<()> {
        // Construction trims.
        let padded = TrimmedString::new("AMD RYZEN AI MAX+ 395 w/ Radeon 8060S          ");
        assert_eq!(padded.as_ref(), "AMD RYZEN AI MAX+ 395 w/ Radeon 8060S");

        // Deserialization trims.
        let raw = serde_json::json!("  Apple M3 Max\n");
        let parsed: TrimmedString = serde_json::from_value(raw)?;
        assert_eq!(parsed.as_ref(), "Apple M3 Max");

        // From<&str> and From<String> both trim via the sanitizer.
        let from_str: TrimmedString = "  hello  ".into();
        assert_eq!(from_str.as_ref(), "hello");
        let from_string: TrimmedString = "  world  ".to_string().into();
        assert_eq!(from_string.as_ref(), "world");
        Ok(())
    }

    #[test]
    fn device_info_round_trip_trims_padded_values_via_deserialize() -> anyhow::Result<()> {
        // Padded values in the wire payload come back trimmed thanks to
        // TrimmedString's deserialize sanitizer.
        let value = serde_json::json!({
            "device_name": "   Test Device  ",
            "device_form_factor": "desktop",
            "device_os_name": "Windows",
            "device_os_version": "10.0.26100",
            "device_chip_model": "AMD Ryzen AI Max+ 395",
            "device_ram_bytes": 36_000_000_000u64,
            "device_gpu_model": "AMD RYZEN AI MAX+ 395 w/ Radeon 8060S          ",
        });
        let info: DeviceInfo = serde_json::from_value(value)?;
        assert_eq!(info.device_name.as_ref(), "Test Device");
        assert_eq!(
            info.device_gpu_model
                .as_ref()
                .context("device_gpu_model")?
                .as_ref(),
            "AMD RYZEN AI MAX+ 395 w/ Radeon 8060S"
        );
        Ok(())
    }

    #[rstest]
    #[case(DeviceFormFactor::Phone, "phone")]
    #[case(DeviceFormFactor::Tablet, "tablet")]
    #[case(DeviceFormFactor::Laptop, "laptop")]
    #[case(DeviceFormFactor::Desktop, "desktop")]
    #[case(DeviceFormFactor::Server, "server")]
    #[case(DeviceFormFactor::Embedded, "embedded")]
    fn device_form_factor_round_trip(
        #[case] variant: DeviceFormFactor,
        #[case] label: &str,
    ) -> anyhow::Result<()> {
        assert_eq!(variant.to_string(), label);
        assert_eq!(label.parse::<DeviceFormFactor>()?, variant);
        let json = serde_json::to_value(variant)?;
        assert_eq!(json, serde_json::Value::String(label.to_string()));
        assert_eq!(serde_json::from_value::<DeviceFormFactor>(json)?, variant);
        Ok(())
    }

    #[test]
    fn device_info_serializes_flat() -> anyhow::Result<()> {
        let info = DeviceInfo {
            device_name: "Test Device".into(),
            device_form_factor: DeviceFormFactor::Desktop,
            device_os_name: "macOS".into(),
            device_os_version: "15.0".into(),
            device_os_build: Some("24F74".into()),
            device_os_security_patch: None,
            device_chip_model: "Apple M3".into(),
            device_ram_bytes: 36_000_000_000,
            device_gpu_model: None,
            device_gpu_vram_bytes: None,
            device_npu_model: None,
            device_npu_vram_bytes: None,
        };
        let value = serde_json::to_value(&info)?;
        assert_eq!(value["device_name"], "Test Device");
        assert_eq!(value["device_form_factor"], "desktop");
        assert_eq!(value["device_ram_bytes"], 36_000_000_000u64);
        assert!(value.get("device_gpu_model").is_none());
        // Present build serializes; absent security-patch is omitted.
        assert_eq!(value["device_os_build"], "24F74");
        assert!(value.get("device_os_security_patch").is_none());
        Ok(())
    }
}
