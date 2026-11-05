//! Device labels (`identity/device.json`): what this machine calls itself.
//! Kept apart from the registration because a local `benchmarks run` needs the
//! labels and no identity at all.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use pipette_plan_types::{DeviceFormFactor, TrimmedString};

use super::files;
use super::registration::REGISTRATION_NAME;
use crate::error::{Error, Result};

const DEVICE_NAME: &str = "device.json";

/// Either field can be `None`: [`pipette_device::detect_device_info`] fills
/// missing values.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceLabels {
    #[serde(default)]
    pub device_name: Option<TrimmedString>,
    #[serde(default)]
    pub device_form_factor: Option<DeviceFormFactor>,
}

impl DeviceLabels {
    /// Empty or whitespace-only is no value. [`ClientDeviceStore::update`]
    /// takes that as "unset this label", `register` as "keep the prior one".
    pub(crate) fn supplied_name(value: String) -> Option<TrimmedString> {
        let trimmed: TrimmedString = value.into();
        (!trimmed.as_ref().is_empty()).then_some(trimmed)
    }
}

/// Labels that clients predating `device.json` kept on `registration.json`.
/// Read-only compat: those workspaces cannot re-register (`ensure_empty`
/// refuses), so without this their pinned labels would silently fall back to
/// auto-detect. Delete once none remain.
#[derive(Deserialize)]
struct LegacyRegistrationLabels {
    #[serde(default)]
    device_name: Option<TrimmedString>,
    #[serde(default)]
    device_form_factor: Option<DeviceFormFactor>,
}

/// Minted from the workspace with `ws.device()`.
#[derive(Debug, Clone)]
pub struct ClientDeviceStore {
    root: PathBuf,
}

impl ClientDeviceStore {
    /// `root` = workspace `identity/` directory.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// `device.json`, else [`LegacyRegistrationLabels`], else empty labels.
    pub fn get(&self) -> Result<DeviceLabels> {
        if let Some(labels) = files::read_json(&self.root, DEVICE_NAME)? {
            return Ok(labels);
        }
        let legacy: Option<LegacyRegistrationLabels> =
            files::read_json(&self.root, REGISTRATION_NAME)?;
        Ok(legacy
            .map(|legacy| DeviceLabels {
                device_name: legacy.device_name,
                device_form_factor: legacy.device_form_factor,
            })
            .unwrap_or_default())
    }

    pub fn put(&self, labels: &DeviceLabels) -> Result<()> {
        files::write_json(&self.root, DEVICE_NAME, labels)
    }

    /// Apply the labels named, keep the rest, and return what `device.json`
    /// now holds. A blank `device_name` unsets that label. Needs no prior
    /// `auth register`. Naming neither field is [`Error::SetDeviceEmpty`].
    pub fn update(
        &self,
        device_name: Option<String>,
        device_form_factor: Option<DeviceFormFactor>,
    ) -> Result<DeviceLabels> {
        if device_name.is_none() && device_form_factor.is_none() {
            return Err(Error::SetDeviceEmpty);
        }
        let mut labels = self.get()?;
        if let Some(name) = device_name {
            labels.device_name = DeviceLabels::supplied_name(name);
        }
        if let Some(form_factor) = device_form_factor {
            labels.device_form_factor = Some(form_factor);
        }
        self.put(&labels)?;
        Ok(labels)
    }

    /// Idempotent when absent.
    pub fn clear(&self) -> Result<()> {
        files::remove(&self.root, DEVICE_NAME)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store() -> anyhow::Result<(tempfile::TempDir, ClientDeviceStore)> {
        let tmp = tempfile::tempdir()?;
        let s = ClientDeviceStore::new(tmp.path().join("identity"));
        Ok((tmp, s))
    }

    #[test]
    fn labels_read_write_device_json_and_clear() -> anyhow::Result<()> {
        let (_tmp, s) = store()?;
        assert!(s.get()?.device_name.is_none());
        let labels = DeviceLabels {
            device_name: Some(TrimmedString::new("dev")),
            device_form_factor: Some(DeviceFormFactor::Laptop),
        };
        s.put(&labels)?;
        assert_eq!(s.get()?, labels);
        s.clear()?;
        assert_eq!(s.get()?, DeviceLabels::default());
        s.clear()?;
        Ok(())
    }

    #[test]
    fn update_writes_device_json_with_no_registration_in_sight() -> anyhow::Result<()> {
        let (_tmp, s) = store()?;
        let labels = s.update(Some("dev box".into()), Some(DeviceFormFactor::Desktop))?;
        assert_eq!(
            labels.device_name.as_ref().map(AsRef::as_ref),
            Some("dev box")
        );
        assert_eq!(labels.device_form_factor, Some(DeviceFormFactor::Desktop));
        assert_eq!(s.get()?, labels);
        Ok(())
    }

    #[rstest::rstest]
    #[case::name_only(Some("renamed"), None, Some("renamed"), Some(DeviceFormFactor::Laptop))]
    #[case::form_factor_only(
        None,
        Some(DeviceFormFactor::Server),
        Some("pinned"),
        Some(DeviceFormFactor::Server)
    )]
    #[case::blank_name_clears_the_name(Some("   "), None, None, Some(DeviceFormFactor::Laptop))]
    fn update_applies_only_the_labels_named(
        #[case] name: Option<&'static str>,
        #[case] form_factor: Option<DeviceFormFactor>,
        #[case] want_name: Option<&'static str>,
        #[case] want_form_factor: Option<DeviceFormFactor>,
    ) -> anyhow::Result<()> {
        let (_tmp, s) = store()?;
        s.put(&DeviceLabels {
            device_name: Some(TrimmedString::new("pinned")),
            device_form_factor: Some(DeviceFormFactor::Laptop),
        })?;

        let labels = s.update(name.map(str::to_string), form_factor)?;

        assert_eq!(labels.device_name.as_ref().map(AsRef::as_ref), want_name);
        assert_eq!(labels.device_form_factor, want_form_factor);
        assert_eq!(s.get()?, labels);
        Ok(())
    }

    #[test]
    fn update_naming_no_label_is_refused() -> anyhow::Result<()> {
        let (_tmp, s) = store()?;
        assert!(matches!(s.update(None, None), Err(Error::SetDeviceEmpty)));
        assert_eq!(s.get()?, DeviceLabels::default());
        Ok(())
    }

    #[test]
    fn labels_fall_back_to_a_pre_device_json_registration() -> anyhow::Result<()> {
        let (tmp, s) = store()?;
        std::fs::create_dir_all(tmp.path().join("identity"))?;
        std::fs::write(
            s.root.join(REGISTRATION_NAME),
            serde_json::to_vec(&serde_json::json!({
                "client_id": "c1",
                "server_url": "https://example.test",
                "device_name": "old box",
                "device_form_factor": "server",
            }))?,
        )?;
        let labels = s.get()?;
        assert_eq!(
            labels.device_name.as_ref().map(AsRef::as_ref),
            Some("old box")
        );
        assert_eq!(labels.device_form_factor, Some(DeviceFormFactor::Server));

        // device.json wins once written.
        s.put(&DeviceLabels {
            device_name: Some(TrimmedString::new("new box")),
            device_form_factor: None,
        })?;
        assert_eq!(
            s.get()?.device_name.as_ref().map(AsRef::as_ref),
            Some("new box")
        );
        Ok(())
    }

    #[test]
    fn labels_surface_parse_error_on_malformed_file() -> anyhow::Result<()> {
        let (_tmp, s) = store()?;
        s.put(&DeviceLabels::default())?;
        std::fs::write(s.root.join(DEVICE_NAME), b"{not json")?;
        let err = s
            .get()
            .err()
            .ok_or_else(|| anyhow::anyhow!("malformed device.json should error"))?;
        let Error::Ops(pipette_ops::Error::ParseJson { ref path, .. }) = err else {
            anyhow::bail!("expected ParseJson, got {err:?}");
        };
        assert!(
            path.ends_with(DEVICE_NAME),
            "expected parse error to name device.json, got {}",
            path.display()
        );
        Ok(())
    }
}
