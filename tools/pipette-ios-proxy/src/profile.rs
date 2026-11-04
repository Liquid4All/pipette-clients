//! The profile the server matches jobs against: the phone's own hardware fields
//! plus the capability flags of the builds in the pool.
//!
//! Every device field comes from the phone, never from the Mac. `devicectl device
//! info details` gives name, iOS version, product type and CPU type; chip and
//! RAM come only from a completed run's `payload.json` (kept in `crate::state`),
//! since `devicectl` reports neither.

use pipette_mgmt_client::types::DeviceProfileFields;

use crate::config::DeviceConfig;
use crate::device::DeviceDetails;
use crate::state::DeviceFacts;

const FORM_FACTOR: &str = "phone";

/// `device_os_name`, sent only beside `device_os_version`: the server refuses a
/// version without a name.
pub(crate) const OS_NAME: &str = "iOS";

/// `device_os_family`. Every phone that the proxy controls runs iOS, so the
/// family is the OS name. The server derives `os_family:` from it.
const OS_FAMILY: &str = OS_NAME;

const DETAILS_PLACEHOLDER_UDID: &str = "{udid}";
const DETAILS_PLACEHOLDER_PRODUCT_TYPE: &str = "{product_type}";
const DETAILS_PLACEHOLDER_HOST: &str = "{host}";

/// What `{host}` renders as when this Mac's name cannot be read.
const UNKNOWN_HOST: &str = "unknown-host";

/// `capabilities` REPLACES the stored set wholesale when present, so an empty set
/// is sent as `None`. Chip, RAM and the iOS version are absent until known: the
/// server derives `chip:`, `ram_bytes:` and `os_version:` from them, so a guess is
/// worse than a gap, and an absent field keeps the server's stored value (it has
/// no per-field clear). GPU/NPU stay absent: iOS exposes neither.
pub(crate) fn device_profile(
    entry: &DeviceConfig,
    details: &DeviceDetails,
    facts: Option<&DeviceFacts>,
    capabilities: &[String],
) -> DeviceProfileFields {
    let os_version = details
        .os_version
        .as_deref()
        .map(str::trim)
        .filter(|version| !version.is_empty());
    // devicectl's own spelling (`arm64e`); the server maps it to `arch:arm64`.
    let cpu_arch = details
        .cpu_type
        .as_deref()
        .map(str::trim)
        .filter(|arch| !arch.is_empty());
    DeviceProfileFields {
        device_name: device_name(entry, details, facts),
        device_form_factor: Some(FORM_FACTOR.to_string()),
        device_os_name: os_version.map(|_| OS_NAME.to_string()),
        device_os_version: os_version.map(str::to_owned),
        device_cpu_arch: cpu_arch.map(str::to_owned),
        device_os_family: Some(OS_FAMILY.to_string()),
        device_chip_model: facts.and_then(|f| f.chip_model.clone()),
        device_ram_bytes: facts.and_then(|f| f.ram_bytes),
        device_gpu_model: None,
        device_gpu_vram_bytes: None,
        device_npu_model: None,
        device_npu_vram_bytes: None,
        capabilities: if capabilities.is_empty() {
            None
        } else {
            Some(capabilities.to_vec())
        },
    }
}

/// Operator override, then the phone's own name, then the last run's, then the
/// marketing name, then the product type. Never `None`.
fn device_name(
    entry: &DeviceConfig,
    details: &DeviceDetails,
    facts: Option<&DeviceFacts>,
) -> Option<String> {
    entry
        .device_name
        .clone()
        .or_else(|| details.device_name.clone())
        .or_else(|| facts.and_then(|f| f.name.clone()))
        .or_else(|| details.marketing_name.clone())
        .or_else(|| Some(details.product_type.clone()))
}

/// Render `[server] client_details`, substituting `{udid}`, `{product_type}` and
/// `{host}`.
pub(crate) fn render_client_details(template: &str, details: &DeviceDetails, host: &str) -> String {
    template
        .replace(DETAILS_PLACEHOLDER_UDID, &details.identifier)
        .replace(DETAILS_PLACEHOLDER_PRODUCT_TYPE, &details.product_type)
        .replace(DETAILS_PLACEHOLDER_HOST, host)
}

/// This Mac's name, for `{host}`; `unknown-host` rather than an error, since a
/// cosmetic field must not stop a registration.
pub(crate) fn host_name() -> String {
    let uname = rustix::system::uname();
    match uname.nodename().to_str() {
        Ok(name) if !name.is_empty() => name.to_owned(),
        _ => UNKNOWN_HOST.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Context;

    use pipette_mgmt_client::types::PrivateKey;
    use pipette_plan_types::ClientId;

    use super::*;
    use crate::config::ClientCredentials;

    fn details() -> DeviceDetails {
        DeviceDetails {
            identifier: "1ACFAA4F-A9FD-5840-B495-072D33D2A5F5".to_string(),
            hardware_udid: Some("00008150-0012743C1100C01C".to_string()),
            product_type: "iPhone16,1".to_string(),
            marketing_name: Some("iPhone 15 Pro".to_string()),
            os_version: Some("26.5.1".to_string()),
            os_build: Some("23F81".to_string()),
            cpu_type: Some("arm64e".to_string()),
            device_name: Some("Lab 15 pro".to_string()),
            tunnel_state: Some("connected".to_string()),
            developer_mode: Some("enabled".to_string()),
        }
    }

    fn entry() -> anyhow::Result<DeviceConfig> {
        Ok(DeviceConfig {
            udid: Some("1ACFAA4F-A9FD-5840-B495-072D33D2A5F5".to_string()),
            device_name: None,
            bundle_id: "ai.liquid.liquid-pipette".to_string(),
            credentials: ClientCredentials {
                client_id: ClientId::try_new("ev1_9f2c1d3e4b5a6c7d")?,
                private_key: PrivateKey::try_new(
                    "1122334455667788990011223344556677889900112233445566778899001122",
                )?,
            },
        })
    }

    fn facts() -> DeviceFacts {
        DeviceFacts {
            name: Some("Lab 15 pro".to_string()),
            chip_model: Some("Apple A17 Pro".to_string()),
            ram_bytes: Some(8_589_934_592),
        }
    }

    #[test]
    fn the_profile_reports_what_the_phone_said() -> anyhow::Result<()> {
        let capabilities = vec!["runtime:mlx".to_string(), "runtime:llama_cpp".to_string()];
        let facts = facts();
        let profile = device_profile(&entry()?, &details(), Some(&facts), &capabilities);

        assert_eq!(profile.device_name.as_deref(), Some("Lab 15 pro"));
        assert_eq!(profile.device_form_factor.as_deref(), Some(FORM_FACTOR));
        assert_eq!(profile.device_os_name.as_deref(), Some(OS_NAME));
        assert_eq!(profile.device_os_version.as_deref(), Some("26.5.1"));
        assert_eq!(profile.device_cpu_arch.as_deref(), Some("arm64e"));
        assert_eq!(profile.device_os_family.as_deref(), Some("iOS"));
        assert_eq!(profile.device_chip_model.as_deref(), Some("Apple A17 Pro"));
        assert_eq!(profile.device_ram_bytes, Some(8_589_934_592));
        assert_eq!(
            profile.capabilities.as_deref(),
            Some(capabilities.as_slice())
        );
        Ok(())
    }

    #[test]
    fn a_phone_that_has_not_run_yet_still_produces_a_sendable_profile() -> anyhow::Result<()> {
        let profile = device_profile(&entry()?, &details(), None, &[]);

        assert_eq!(profile.device_chip_model, None);
        assert_eq!(profile.device_ram_bytes, None);
        assert_eq!(profile.device_cpu_arch.as_deref(), Some("arm64e"));
        assert_eq!(profile.device_os_family.as_deref(), Some("iOS"));
        // An empty list would erase the server's stored set.
        assert!(profile.capabilities.is_none());
        assert_eq!(profile.device_os_name.as_deref(), Some(OS_NAME));
        assert_eq!(profile.device_os_version.as_deref(), Some("26.5.1"));
        Ok(())
    }

    /// Trimmed; a blank version is sent as no version.
    #[rstest::rstest]
    #[case::read(Some("26.5.1"), Some("26.5.1"))]
    #[case::padded(Some(" 26.6\n"), Some("26.6"))]
    #[case::not_read(None, None)]
    #[case::blank(Some("  "), None)]
    fn the_ios_version_goes_with_its_name_or_not_at_all(
        #[case] read: Option<&str>,
        #[case] reported: Option<&str>,
    ) -> anyhow::Result<()> {
        let mut details = details();
        details.os_version = read.map(str::to_owned);
        let facts = facts();
        let profile = device_profile(&entry()?, &details, Some(&facts), &[]);

        assert_eq!(profile.device_os_version.as_deref(), reported);
        assert_eq!(profile.device_os_name.as_deref(), reported.map(|_| OS_NAME));
        Ok(())
    }

    /// Trimmed; a blank or unread CPU type is sent as no architecture.
    #[rstest::rstest]
    #[case::read(Some("arm64e"), Some("arm64e"))]
    #[case::padded(Some(" arm64e\n"), Some("arm64e"))]
    #[case::not_read(None, None)]
    #[case::blank(Some("  "), None)]
    fn the_cpu_arch_is_the_phones_own_cpu_type(
        #[case] read: Option<&str>,
        #[case] reported: Option<&str>,
    ) -> anyhow::Result<()> {
        let mut details = details();
        details.cpu_type = read.map(str::to_owned);
        let profile = device_profile(&entry()?, &details, None, &[]);

        assert_eq!(profile.device_cpu_arch.as_deref(), reported);
        Ok(())
    }

    #[test]
    fn a_configured_label_overrides_the_phones_own_name() -> anyhow::Result<()> {
        let mut entry = entry()?;
        entry.device_name = Some("fleet-iphone-15pro-a".to_string());
        let facts = facts();
        let profile = device_profile(&entry, &details(), Some(&facts), &[]);
        assert_eq!(profile.device_name.as_deref(), Some("fleet-iphone-15pro-a"));
        Ok(())
    }

    #[test]
    fn the_name_falls_back_rather_than_going_missing() -> anyhow::Result<()> {
        let mut anonymous = details();
        anonymous.device_name = None;
        let profile = device_profile(&entry()?, &anonymous, None, &[]);
        assert_eq!(profile.device_name.as_deref(), Some("iPhone 15 Pro"));

        anonymous.marketing_name = None;
        let profile = device_profile(&entry()?, &anonymous, None, &[]);
        assert_eq!(profile.device_name.as_deref(), Some("iPhone16,1"));
        Ok(())
    }

    #[test]
    fn client_details_substitutes_every_placeholder() -> anyhow::Result<()> {
        let rendered = render_client_details(
            "pipette-ios-proxy relaying {product_type} ({udid}) from {host}",
            &details(),
            "lab-east-mac-2",
        );
        assert_eq!(
            rendered,
            "pipette-ios-proxy relaying iPhone16,1 (1ACFAA4F-A9FD-5840-B495-072D33D2A5F5) from \
             lab-east-mac-2"
        );
        Ok(())
    }

    #[test]
    fn host_name_is_never_empty() -> anyhow::Result<()> {
        let name = host_name();
        assert!(!name.is_empty());
        let _ = name
            .chars()
            .next()
            .context("a non-empty host name has a first character")?;
        Ok(())
    }
}
