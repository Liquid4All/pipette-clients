//! Local identity files: the upload registration, the device labels, and the
//! client-wide settings.

use serde::{Deserialize, Serialize};

use pipette_plan_types::device::{DeviceFormFactor, TrimmedString};

/// Local registration metadata stored in `registration.json`.
///
/// Holds only the upload identity (`client_id`, `server_url`). Fields like
/// `organization`, `contact_email`, `client_details`, and approval `status`
/// live server-side and are fetched on demand via `auth me`. Device labels
/// live separately in [`DeviceLabels`] (`identity/device.json`) — the
/// registration carries no device fields.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IdentityRegistration {
    pub client_id: String,
    pub server_url: String,
}

/// Device labels stored locally in `identity/device.json`.
///
/// Carries `device_name` and `device_form_factor` independently of the
/// upload identity. `benchmarks run` only needs labels; `sync` needs the
/// full registration (keypair + `client_id` + `server_url`). Splitting
/// them lets a fresh box run `init && benchmarks run` without forcing
/// `auth register` first.
///
/// Either field can be `None` — the auto-detect path in
/// [`pipette_device::detect_device_info`] fills missing values.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceLabels {
    #[serde(default)]
    pub device_name: Option<TrimmedString>,
    #[serde(default)]
    pub device_form_factor: Option<DeviceFormFactor>,
}

/// Client-wide settings stored in `identity/settings.json`.
///
/// The file is optional and an absent one means the built-in defaults. No serde
/// defaults on the fields: a present-but-partial file is an error rather than a
/// silently half-applied configuration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClientSettings {
    /// Cap on the disk the model and runtime artifact stores may occupy
    /// (`docs/storage-quota.md`).
    pub storage_quota_bytes: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn settings_without_every_field_do_not_deserialize() {
        // The no-defaults rule: a half-written settings.json must surface, not
        // quietly resolve to the built-in quota.
        assert!(serde_json::from_value::<ClientSettings>(serde_json::json!({})).is_err());
    }

    #[test]
    fn old_registration_with_dropped_fields_still_deserializes() -> anyhow::Result<()> {
        // A pre-split `registration.json` carries server-side fields the
        // client never reads (and possibly mirrored device labels); dropping
        // them client-side must not break reading an older file.
        let old_json = serde_json::json!({
            "client_id": "ev1_abc",
            "status": "approved",
            "server_url": "https://mgmt.example.com",
            "organization": "LiquidAI",
            "contact_email": "user@example.com",
            "client_details": "test machine",
            "device_name": "test box",
            "device_form_factor": "laptop",
            "registered_at": "2026-01-01T00:00:00Z"
        });
        let reg: IdentityRegistration = serde_json::from_value(old_json)?;
        assert_eq!(reg.client_id, "ev1_abc");
        assert_eq!(reg.server_url, "https://mgmt.example.com");
        Ok(())
    }
}
