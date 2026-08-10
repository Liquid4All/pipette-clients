//! Client identity store — keys, registration, device labels.
//!
//! Concrete FS handle over workspace `identity/`, same shape as
//! [`pipette_ops::EvalCompletionsStore`] / [`crate::benchmarks::BenchmarkStore`].
//! Call sites take `&IdentityStore` — no path-trait indirection.
//!
//! Layout (private):
//! ```text
//! identity/
//!   id_ed25519            # private key hex (0600)
//!   id_ed25519.pub        # public key hex
//!   registration.json     # client_id + server_url
//!   device.json           # device labels (local-only runs)
//!   settings.json         # client-wide settings (storage quota)
//! ```

use std::fs;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use pipette_mgmt_client::AuthIdentity;

use pipette_ops::fs::{read_json_typed, write_json_typed};
use pipette_plan_types::device::{DeviceFormFactor, TrimmedString};

use super::{ClientSettings, DeviceLabels, IdentityRegistration};
use crate::error::{Error, Result};

const PRIVATE_KEY_NAME: &str = "id_ed25519";
const PUBLIC_KEY_NAME: &str = "id_ed25519.pub";
const REGISTRATION_NAME: &str = "registration.json";
const DEVICE_NAME: &str = "device.json";
const SETTINGS_NAME: &str = "settings.json";

/// Labels as written by clients that predate `device.json`, which kept them on
/// `registration.json` instead. Read-only compat: those workspaces can't
/// re-register (`ensure_empty` refuses), so without this their pinned labels
/// would silently fall back to auto-detect. Delete once none remain.
#[derive(Deserialize)]
struct LegacyRegistrationLabels {
    #[serde(default)]
    device_name: Option<TrimmedString>,
    #[serde(default)]
    device_form_factor: Option<DeviceFormFactor>,
}

/// Capability handle for client identity (`root/` = workspace `identity/`).
///
/// Layout is private. Mint from the workspace when wired (`ws.identity()`).
#[derive(Debug, Clone)]
pub struct IdentityStore {
    root: PathBuf,
}

impl IdentityStore {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    fn path(&self, name: &str) -> PathBuf {
        self.root.join(name)
    }

    fn ensure_root(&self) -> Result<()> {
        fs::create_dir_all(&self.root).map_err(|source| pipette_ops::Error::Io {
            path: self.root.clone(),
            source,
        })?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&self.root, fs::Permissions::from_mode(0o700)).map_err(
                |source| pipette_ops::Error::Io {
                    path: self.root.clone(),
                    source,
                },
            )?;
        }
        Ok(())
    }

    /// Persist the Ed25519 private key (hex), created `0600` on unix.
    pub fn put_private_key(&self, private_key_hex: &str) -> Result<()> {
        self.ensure_root()?;
        let path = self.path(PRIVATE_KEY_NAME);
        write_private_key(&path, private_key_hex.as_bytes()).map_err(|source| {
            pipette_ops::Error::Io {
                path: path.clone(),
                source,
            }
        })?;
        Ok(())
    }

    pub fn get_private_key(&self) -> Result<Option<String>> {
        let path = self.path(PRIVATE_KEY_NAME);
        if !path.exists() {
            return Ok(None);
        }
        fs::read_to_string(&path)
            .map(Some)
            .map_err(|source| pipette_ops::Error::Io { path, source }.into())
    }

    pub fn delete_private_key(&self) -> Result<()> {
        self.delete_if_exists(PRIVATE_KEY_NAME)
    }

    pub fn put_public_key(&self, public_key_hex: &str) -> Result<()> {
        self.ensure_root()?;
        let path = self.path(PUBLIC_KEY_NAME);
        fs::write(&path, public_key_hex).map_err(|source| {
            pipette_ops::Error::Io {
                path: path.clone(),
                source,
            }
            .into()
        })
    }

    pub fn get_registration(&self) -> Result<Option<IdentityRegistration>> {
        let path = self.path(REGISTRATION_NAME);
        if !path.exists() {
            return Ok(None);
        }
        Ok(Some(read_json_typed(&path)?))
    }

    pub fn put_registration(&self, registration: &IdentityRegistration) -> Result<()> {
        self.ensure_root()?;
        write_json_typed(&self.path(REGISTRATION_NAME), registration).map_err(Into::into)
    }

    /// Load registration or [`Error::RegistrationMissing`].
    pub fn require_registration(&self) -> Result<IdentityRegistration> {
        let path = self.path(REGISTRATION_NAME);
        self.get_registration()
            .and_then(|opt| opt.ok_or(Error::RegistrationMissing(path)))
    }

    /// Remove public key + registration + private key; leave `device.json`.
    ///
    /// Used when `register` fails mid-persist so a retry is not blocked, while
    /// preserving labels from a prior `auth set-device`.
    pub fn clear_registration_material(&self) -> Result<()> {
        self.delete_private_key()?;
        self.delete_if_exists(PUBLIC_KEY_NAME)?;
        self.delete_if_exists(REGISTRATION_NAME)?;
        Ok(())
    }

    fn delete_if_exists(&self, name: &str) -> Result<()> {
        let path = self.path(name);
        if path.exists() {
            fs::remove_file(&path).map_err(|source| pipette_ops::Error::Io {
                path: path.clone(),
                source,
            })?;
        }
        Ok(())
    }

    /// Compose the signing identity for server requests: the registered
    /// `client_id` plus the Ed25519 private key. Errors with
    /// [`Error::RegistrationMissing`] / [`Error::PrivateKeyMissing`] when
    /// either half is absent.
    pub fn signing_identity(&self) -> Result<AuthIdentity> {
        let registration = self.require_registration()?;
        let private_key_hex = self.get_private_key()?.ok_or(Error::PrivateKeyMissing)?;
        Ok(AuthIdentity {
            client_id: registration.client_id,
            private_key_hex: private_key_hex.trim().to_string(),
        })
    }

    /// Read device labels from `device.json` — the single source of truth for
    /// labels, falling back to [`LegacyRegistrationLabels`] for workspaces
    /// written before that file existed. Neither present means empty
    /// [`DeviceLabels`] (auto-detect fills values at run time).
    pub fn get_device_labels(&self) -> Result<DeviceLabels> {
        let device_path = self.path(DEVICE_NAME);
        if device_path.exists() {
            return read_json_typed(&device_path).map_err(Into::into);
        }
        let registration_path = self.path(REGISTRATION_NAME);
        if registration_path.exists() {
            let legacy: LegacyRegistrationLabels = read_json_typed(&registration_path)?;
            return Ok(DeviceLabels {
                device_name: legacy.device_name,
                device_form_factor: legacy.device_form_factor,
            });
        }
        Ok(DeviceLabels::default())
    }

    pub fn put_device_labels(&self, labels: &DeviceLabels) -> Result<()> {
        self.ensure_root()?;
        write_json_typed(&self.path(DEVICE_NAME), labels).map_err(Into::into)
    }

    /// Read `settings.json`, or `None` when the file is absent — the caller
    /// falls back to the built-in defaults.
    pub fn get_settings(&self) -> Result<Option<ClientSettings>> {
        let path = self.path(SETTINGS_NAME);
        if !path.exists() {
            return Ok(None);
        }
        Ok(Some(read_json_typed(&path)?))
    }

    pub fn put_settings(&self, settings: &ClientSettings) -> Result<()> {
        self.ensure_root()?;
        write_json_typed(&self.path(SETTINGS_NAME), settings).map_err(Into::into)
    }

    /// Fail if a private key, public key, or registration already exists.
    ///
    /// `device.json` is intentionally ignored so `set-device` before `register`
    /// still works.
    pub fn ensure_empty(&self) -> Result<()> {
        if let Some(path) = [PRIVATE_KEY_NAME, PUBLIC_KEY_NAME, REGISTRATION_NAME]
            .into_iter()
            .map(|n| self.path(n))
            .find(|p| p.exists())
        {
            return Err(Error::IdentityExists(path.display().to_string()));
        }
        Ok(())
    }

    /// Remove keypair, registration, device labels, and settings (files only).
    ///
    /// Leaves the `identity/` directory. Idempotent for missing files.
    pub fn clear(&self) -> Result<()> {
        self.clear_registration_material()?;
        self.delete_if_exists(DEVICE_NAME)?;
        self.delete_if_exists(SETTINGS_NAME)
    }
}

/// Write the private key so it is never on disk world-readable, even briefly.
///
/// `fs::write` then `chmod` leaves the key readable at the umask default for the
/// window between the two. Here `.mode()` applies at creation, and the reassert
/// below covers a file that already existed at a looser mode (`.mode()` is
/// ignored then) — it runs on the open handle, so it is an `fchmod` with no path
/// to re-resolve. Both happen before `write_all`, so the bytes only ever land in
/// an already-restricted file.
///
/// The `0700` parent from `ensure_root` narrows the exposure but does not remove
/// it: modes are re-read per open, so anything already inside the directory can
/// still catch the window.
#[cfg(unix)]
fn write_private_key(path: &Path, contents: &[u8]) -> std::io::Result<()> {
    use std::io::Write;
    use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};

    let mut file = fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .mode(0o600)
        .open(path)?;
    file.set_permissions(fs::Permissions::from_mode(0o600))?;
    file.write_all(contents)
}

/// Windows has no mode to set at creation, so the key rests on the ACL the
/// profile directory grants — typically owner-only under the user profile, but
/// not asserted here. Tightening it needs a Win32 ACL call, which is why this is
/// still the plain write; see PIP-461.
#[cfg(not(unix))]
fn write_private_key(path: &Path, contents: &[u8]) -> std::io::Result<()> {
    fs::write(path, contents)
}

#[cfg(test)]
mod tests {
    use pipette_plan_types::device::{DeviceFormFactor, TrimmedString};

    use super::*;

    fn store() -> anyhow::Result<(tempfile::TempDir, IdentityStore)> {
        let tmp = tempfile::tempdir()?;
        let s = IdentityStore::new(tmp.path().join("identity"));
        Ok((tmp, s))
    }

    fn reg() -> IdentityRegistration {
        IdentityRegistration {
            client_id: "c1".into(),
            server_url: "https://example.test".into(),
        }
    }

    #[test]
    fn put_get_registration_and_clear() -> anyhow::Result<()> {
        let (_tmp, s) = store()?;
        assert!(s.get_registration()?.is_none());
        s.put_registration(&reg())?;
        let loaded = s
            .get_registration()?
            .ok_or_else(|| anyhow::anyhow!("registration present"))?;
        assert_eq!(loaded.client_id, "c1");
        s.clear()?;
        assert!(s.get_registration()?.is_none());
        s.clear()?;
        Ok(())
    }

    #[test]
    fn private_key_round_trip_and_delete() -> anyhow::Result<()> {
        let (_tmp, s) = store()?;
        assert!(s.get_private_key()?.is_none());
        s.put_private_key("deadbeef")?;
        assert_eq!(s.get_private_key()?.as_deref(), Some("deadbeef"));
        s.delete_private_key()?;
        assert!(s.get_private_key()?.is_none());
        Ok(())
    }

    #[cfg(unix)]
    fn assert_owner_only(path: &Path) -> anyhow::Result<()> {
        use std::os::unix::fs::PermissionsExt;

        let mode = fs::metadata(path)?.permissions().mode() & 0o777;
        assert_eq!(mode, 0o600, "expected 0600, got {mode:o}");
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn private_key_is_created_owner_only() -> anyhow::Result<()> {
        let (_tmp, s) = store()?;
        s.put_private_key("deadbeef")?;
        assert_owner_only(&s.path(PRIVATE_KEY_NAME))
    }

    #[cfg(unix)]
    #[test]
    fn private_key_tightens_a_file_that_was_already_loose() -> anyhow::Result<()> {
        use std::os::unix::fs::PermissionsExt;

        // A key written by an older client sits at the umask default. `.mode()`
        // applies only at creation, so without the reassert the rewrite would
        // leave it world-readable.
        let (_tmp, s) = store()?;
        s.put_private_key("deadbeef")?;
        let path = s.path(PRIVATE_KEY_NAME);
        fs::set_permissions(&path, fs::Permissions::from_mode(0o644))?;

        s.put_private_key("cafebabe")?;

        assert_owner_only(&path)?;
        assert_eq!(s.get_private_key()?.as_deref(), Some("cafebabe"));
        Ok(())
    }

    #[test]
    fn device_labels_read_writes_device_json() -> anyhow::Result<()> {
        let (_tmp, s) = store()?;
        // Absent file → empty labels.
        assert!(s.get_device_labels()?.device_name.is_none());
        let labels = DeviceLabels {
            device_name: Some(TrimmedString::new("dev")),
            device_form_factor: Some(DeviceFormFactor::Laptop),
        };
        s.put_device_labels(&labels)?;
        assert_eq!(s.get_device_labels()?, labels);
        Ok(())
    }

    #[test]
    fn device_labels_fall_back_to_a_pre_device_json_registration() -> anyhow::Result<()> {
        let (_tmp, s) = store()?;
        // A registration.json from before device.json existed.
        std::fs::create_dir_all(&s.root)?;
        std::fs::write(
            s.root.join("registration.json"),
            serde_json::to_vec(&serde_json::json!({
                "client_id": "c1",
                "server_url": "https://example.test",
                "device_name": "old box",
                "device_form_factor": "server",
            }))?,
        )?;
        let labels = s.get_device_labels()?;
        assert_eq!(
            labels.device_name.as_ref().map(AsRef::as_ref),
            Some("old box")
        );
        assert_eq!(labels.device_form_factor, Some(DeviceFormFactor::Server));

        // device.json wins once written.
        s.put_device_labels(&DeviceLabels {
            device_name: Some(TrimmedString::new("new box")),
            device_form_factor: None,
        })?;
        assert_eq!(
            s.get_device_labels()?
                .device_name
                .as_ref()
                .map(AsRef::as_ref),
            Some("new box")
        );
        Ok(())
    }

    #[test]
    fn settings_round_trip_and_clear() -> anyhow::Result<()> {
        let (_tmp, s) = store()?;
        assert!(s.get_settings()?.is_none());
        let settings = ClientSettings {
            storage_quota_bytes: 42 * 1024 * 1024 * 1024,
        };
        s.put_settings(&settings)?;
        assert_eq!(s.get_settings()?, Some(settings));
        s.clear()?;
        assert!(s.get_settings()?.is_none());
        Ok(())
    }

    #[test]
    fn malformed_settings_surface_a_parse_error() -> anyhow::Result<()> {
        let (_tmp, s) = store()?;
        s.put_settings(&ClientSettings {
            storage_quota_bytes: 1,
        })?;
        std::fs::write(s.root.join(SETTINGS_NAME), b"{not json")?;
        let err = s
            .get_settings()
            .err()
            .ok_or_else(|| anyhow::anyhow!("malformed settings.json should error"))?;
        let Error::Ops(pipette_ops::Error::ParseJson { ref path, .. }) = err else {
            anyhow::bail!("expected ParseJson, got {err:?}");
        };
        assert!(
            path.ends_with(SETTINGS_NAME),
            "expected parse error to name settings.json, got {}",
            path.display()
        );
        Ok(())
    }

    #[test]
    fn signing_identity_composes_registration_and_key() -> anyhow::Result<()> {
        let (_tmp, s) = store()?;
        s.put_registration(&reg())?;
        // No key yet → PrivateKeyMissing.
        assert!(matches!(
            s.signing_identity(),
            Err(Error::PrivateKeyMissing)
        ));
        s.put_private_key("deadbeef")?;
        let auth = s.signing_identity()?;
        assert_eq!(auth.client_id, "c1");
        assert_eq!(auth.private_key_hex, "deadbeef");
        Ok(())
    }

    #[test]
    fn signing_identity_errors_without_registration() -> anyhow::Result<()> {
        let (_tmp, s) = store()?;
        assert!(matches!(
            s.signing_identity(),
            Err(Error::RegistrationMissing(_))
        ));
        Ok(())
    }

    #[test]
    fn ensure_empty_rejects_existing_artifacts() -> anyhow::Result<()> {
        let (_tmp, s) = store()?;
        s.ensure_empty()?;

        s.put_private_key("aa")?;
        assert!(matches!(s.ensure_empty(), Err(Error::IdentityExists(_))));
        s.clear()?;

        s.put_public_key("pub")?;
        assert!(matches!(s.ensure_empty(), Err(Error::IdentityExists(_))));
        s.clear()?;

        s.put_registration(&reg())?;
        assert!(matches!(s.ensure_empty(), Err(Error::IdentityExists(_))));
        s.clear()?;

        // device-only is allowed
        s.put_device_labels(&DeviceLabels::default())?;
        s.ensure_empty()?;
        Ok(())
    }

    #[test]
    fn require_registration_reports_missing_file() -> anyhow::Result<()> {
        let (_tmp, s) = store()?;
        assert!(matches!(
            s.require_registration(),
            Err(Error::RegistrationMissing(_))
        ));
        Ok(())
    }

    #[test]
    fn device_labels_surface_parse_error_on_malformed_file() -> anyhow::Result<()> {
        let (_tmp, s) = store()?;
        s.put_device_labels(&DeviceLabels::default())?;
        let path = s.root.join(DEVICE_NAME);
        std::fs::write(&path, b"{not json")?;
        let err = s
            .get_device_labels()
            .err()
            .ok_or_else(|| anyhow::anyhow!("malformed device.json should error"))?;
        let Error::Ops(pipette_ops::Error::ParseJson { ref path, .. }) = err else {
            anyhow::bail!("expected ParseJson, got {err:?}");
        };
        assert!(
            path.ends_with("device.json"),
            "expected parse error to name device.json, got {}",
            path.display()
        );
        Ok(())
    }

    #[test]
    fn clear_registration_material_leaves_device_json() -> anyhow::Result<()> {
        let (_tmp, s) = store()?;
        s.put_private_key("k")?;
        s.put_public_key("p")?;
        s.put_registration(&reg())?;
        s.put_device_labels(&DeviceLabels {
            device_name: Some(TrimmedString::new("keep")),
            device_form_factor: None,
        })?;
        s.clear_registration_material()?;
        assert!(s.get_private_key()?.is_none());
        assert!(s.get_registration()?.is_none());
        assert_eq!(
            s.get_device_labels()?
                .device_name
                .as_ref()
                .map(AsRef::as_ref),
            Some("keep")
        );
        s.ensure_empty()?;
        Ok(())
    }
}
