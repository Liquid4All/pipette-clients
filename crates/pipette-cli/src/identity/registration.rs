//! `registration.json` and the keypair that signs as it: who this client is to
//! the management server.
//!
//! ```text
//! identity/
//!   id_ed25519            # private key hex (0600)
//!   id_ed25519.pub        # public key hex
//!   registration.json     # client_id + server_url
//! ```

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use pipette_mgmt_client::{ClientIdentity, PrivateKey, PublicKey, ServerUrl};
use pipette_plan_types::ClientId;

use super::files;
use crate::error::{Error, Result};

const PRIVATE_KEY_NAME: &str = "id_ed25519";
const PUBLIC_KEY_NAME: &str = "id_ed25519.pub";
/// `pub(super)` because [`ClientDeviceStore`](super::ClientDeviceStore) reads
/// this file too, for labels written before `device.json` existed.
pub(super) const REGISTRATION_NAME: &str = "registration.json";

/// `registration.json`. Everything else about the client (organization,
/// contact, status) is server-side and read with `auth me`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IdentityRegistration {
    pub client_id: ClientId,
    pub server_url: ServerUrl,
}

/// The same record with both fields unparsed, so commands that report or
/// delete an identity (`auth me --local`, `auth reset`, `auth import --force`)
/// still work on a file [`IdentityRegistration`] refuses.
#[derive(Debug, Clone, Deserialize)]
pub struct StoredRegistration {
    pub client_id: String,
    pub server_url: String,
}

/// Minted from the workspace with `ws.registration()`.
#[derive(Debug, Clone)]
pub struct ClientRegistrationStore {
    root: PathBuf,
}

impl ClientRegistrationStore {
    /// `root` = workspace `identity/` directory.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// Created `0600` on unix.
    pub fn put_private_key(&self, private_key: &PrivateKey) -> Result<()> {
        files::write_secret(
            &self.root,
            PRIVATE_KEY_NAME,
            private_key.as_ref().as_bytes(),
        )
    }

    /// Absent, blank and unusable all read as `None`: `secret_file::write`
    /// truncates before it writes, so a crash leaves a blank file, and no such
    /// file is worth making `auth import` demand `--force`. An unusable file is
    /// logged because its repair (`auth import`) differs from an absent one's
    /// (`auth register`).
    pub fn get_private_key(&self) -> Result<Option<PrivateKey>> {
        Ok(
            files::read_text(&self.root, PRIVATE_KEY_NAME)?.and_then(|stored| {
                PrivateKey::try_new(stored)
                    .inspect_err(|reason| {
                        tracing::warn!(
                            path = %files::path(&self.root, PRIVATE_KEY_NAME).display(),
                            error = %reason,
                            "ignoring unusable private key"
                        )
                    })
                    .ok()
            }),
        )
    }

    pub fn delete_private_key(&self) -> Result<()> {
        files::remove(&self.root, PRIVATE_KEY_NAME)
    }

    pub fn put_public_key(&self, public_key: &PublicKey) -> Result<()> {
        files::write_text(&self.root, PUBLIC_KEY_NAME, public_key.as_ref())
    }

    /// Absent means never written, which an import that stopped partway
    /// treats as work still to do.
    pub fn get_public_key(&self) -> Result<Option<PublicKey>> {
        Ok(
            files::read_text(&self.root, PUBLIC_KEY_NAME)?.and_then(|stored| {
                PublicKey::try_new(stored)
                    .inspect_err(|reason| {
                        tracing::warn!(
                            path = %files::path(&self.root, PUBLIC_KEY_NAME).display(),
                            error = %reason,
                            "ignoring unusable public key"
                        )
                    })
                    .ok()
            }),
        )
    }

    pub fn get(&self) -> Result<Option<IdentityRegistration>> {
        files::read_json(&self.root, REGISTRATION_NAME)
    }

    /// See [`StoredRegistration`].
    pub fn stored(&self) -> Result<Option<StoredRegistration>> {
        files::read_json(&self.root, REGISTRATION_NAME)
    }

    pub fn put(&self, registration: &IdentityRegistration) -> Result<()> {
        files::write_json(&self.root, REGISTRATION_NAME, registration)
    }

    pub fn require(&self) -> Result<IdentityRegistration> {
        let path = files::path(&self.root, REGISTRATION_NAME);
        self.get()
            .and_then(|opt| opt.ok_or(Error::RegistrationMissing(path)))
    }

    /// The lenient twin of [`Self::require`].
    pub fn require_stored(&self) -> Result<StoredRegistration> {
        let path = files::path(&self.root, REGISTRATION_NAME);
        self.stored()
            .and_then(|opt| opt.ok_or(Error::RegistrationMissing(path)))
    }

    /// Errors name whichever half is missing.
    pub fn require_identity(&self) -> Result<(IdentityRegistration, PrivateKey)> {
        let registration = self.require()?;
        let private_key = self.get_private_key()?.ok_or(Error::PrivateKeyMissing)?;
        Ok((registration, private_key))
    }

    pub fn server_url(&self) -> Result<ServerUrl> {
        Ok(self.require()?.server_url)
    }

    pub fn signing_identity(&self) -> Result<ClientIdentity> {
        let (registration, private_key_hex) = self.require_identity()?;
        Ok(ClientIdentity {
            client_id: registration.client_id,
            private_key_hex,
        })
    }

    /// Ignores `device.json` and `settings.json`, so `set-device` before
    /// `register` still works.
    pub fn ensure_empty(&self) -> Result<()> {
        if let Some(path) = [PRIVATE_KEY_NAME, PUBLIC_KEY_NAME, REGISTRATION_NAME]
            .into_iter()
            .map(|n| files::path(&self.root, n))
            .find(|p| p.exists())
        {
            return Err(Error::IdentityExists(path.display().to_string()));
        }
        Ok(())
    }

    /// Leaves `device.json` and `settings.json`, so a `register` that fails
    /// mid-persist neither blocks a retry nor loses pinned labels. `auth reset`
    /// clears the other two stores itself. Idempotent.
    pub fn clear(&self) -> Result<()> {
        self.delete_private_key()?;
        files::remove(&self.root, PUBLIC_KEY_NAME)?;
        files::remove(&self.root, REGISTRATION_NAME)
    }
}

#[cfg(test)]
mod tests {
    // Unix-only: only the mode assertions use these.
    #[cfg(unix)]
    use std::fs;
    #[cfg(unix)]
    use std::path::Path;

    use pipette_plan_types::TrimmedString;

    use super::*;
    use crate::identity::{ClientDeviceStore, ClientSettings, ClientSettingsStore, DeviceLabels};

    fn store() -> anyhow::Result<(tempfile::TempDir, ClientRegistrationStore)> {
        let tmp = tempfile::tempdir()?;
        let s = ClientRegistrationStore::new(identity_dir(&tmp));
        Ok((tmp, s))
    }

    fn identity_dir(tmp: &tempfile::TempDir) -> std::path::PathBuf {
        tmp.path().join("identity")
    }

    /// Any 32 bytes are a valid Ed25519 seed, so hex of the right length is a
    /// usable key without generating one.
    const KEY: &str = "aa11bb22cc33dd44ee55ff6600778899aa11bb22cc33dd44ee55ff6600778899";

    fn key(hex: &str) -> anyhow::Result<PrivateKey> {
        Ok(PrivateKey::try_new(hex)?)
    }

    fn reg() -> anyhow::Result<IdentityRegistration> {
        Ok(IdentityRegistration {
            client_id: ClientId::try_new("c1")?,
            server_url: ServerUrl::try_new("https://example.test")?,
        })
    }

    #[test]
    fn registration_round_trips_and_clears() -> anyhow::Result<()> {
        let (_tmp, s) = store()?;
        assert!(s.get()?.is_none());
        s.put(&reg()?)?;
        let loaded = s
            .get()?
            .ok_or_else(|| anyhow::anyhow!("registration present"))?;
        assert_eq!(loaded.client_id.as_ref(), "c1");
        s.clear()?;
        assert!(s.get()?.is_none());
        s.clear()?;
        Ok(())
    }

    #[test]
    fn private_key_round_trip_and_delete() -> anyhow::Result<()> {
        let (_tmp, s) = store()?;
        assert!(s.get_private_key()?.is_none());
        s.put_private_key(&key(KEY)?)?;
        assert_eq!(s.get_private_key()?, Some(key(KEY)?));
        s.delete_private_key()?;
        assert!(s.get_private_key()?.is_none());
        Ok(())
    }

    #[rstest::rstest]
    #[case::blank("   \n")]
    #[case::not_hex("not a key")]
    #[case::wrong_length("aabb")]
    fn a_key_file_that_cannot_sign_reads_as_absent(#[case] stored: &str) -> anyhow::Result<()> {
        let (_tmp, s) = store()?;
        s.put(&reg()?)?;
        std::fs::write(s.root.join(PRIVATE_KEY_NAME), stored)?;
        assert!(s.get_private_key()?.is_none());
        assert!(matches!(
            s.require_identity(),
            Err(Error::PrivateKeyMissing)
        ));
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
        s.put_private_key(&key(KEY)?)?;
        assert_owner_only(&s.root.join(PRIVATE_KEY_NAME))
    }

    #[cfg(unix)]
    #[test]
    fn private_key_tightens_a_file_that_was_already_loose() -> anyhow::Result<()> {
        use std::os::unix::fs::PermissionsExt;

        // Here, not beside `KEY`: at module scope it is dead code off unix.
        const OTHER_KEY: &str = "bb22cc33dd44ee55ff6600778899aa11bb22cc33dd44ee55ff6600778899aa11";

        // `.mode()` applies only at creation, so without the reassert a
        // rewrite would leave an older umask-default key world-readable.
        let (_tmp, s) = store()?;
        s.put_private_key(&key(KEY)?)?;
        let path = s.root.join(PRIVATE_KEY_NAME);
        fs::set_permissions(&path, fs::Permissions::from_mode(0o644))?;

        s.put_private_key(&key(OTHER_KEY)?)?;

        assert_owner_only(&path)?;
        assert_eq!(s.get_private_key()?, Some(key(OTHER_KEY)?));
        Ok(())
    }

    #[test]
    fn public_key_round_trips_and_clears() -> anyhow::Result<()> {
        let (_tmp, s) = store()?;
        assert!(s.get_public_key()?.is_none());
        let public = PublicKey::try_new("ca7e".repeat(16))?;
        s.put_public_key(&public)?;
        assert_eq!(s.get_public_key()?, Some(public));
        s.clear()?;
        assert!(s.get_public_key()?.is_none());
        Ok(())
    }

    #[test]
    fn signing_identity_composes_registration_and_key() -> anyhow::Result<()> {
        let (_tmp, s) = store()?;
        s.put(&reg()?)?;
        assert!(matches!(
            s.signing_identity(),
            Err(Error::PrivateKeyMissing)
        ));
        s.put_private_key(&key(KEY)?)?;
        let auth = s.signing_identity()?;
        assert_eq!(auth.client_id.as_ref(), "c1");
        assert_eq!(auth.private_key_hex, key(KEY)?);
        Ok(())
    }

    /// The path prefix matters: requests are signed over it, so dropping it
    /// would send every request to a target the server never sees.
    #[rstest::rstest]
    #[case::trailing_slash_is_stripped("https://example.test/", Some("https://example.test"))]
    #[case::path_prefix_survives("https://example.test/api", Some("https://example.test/api"))]
    #[case::bare_host_is_not_a_base("example.test", None)]
    fn server_url_reads_what_the_file_holds(
        #[case] stored: &str,
        #[case] expected: Option<&str>,
    ) -> anyhow::Result<()> {
        let (_tmp, s) = store()?;
        write_registration(&s, "c1", stored)?;
        match expected {
            Some(want) => assert_eq!(s.server_url()?.to_string(), want),
            None => assert!(s.server_url().is_err()),
        }
        Ok(())
    }

    #[test]
    fn a_url_that_does_not_parse_is_still_readable_and_clearable() -> anyhow::Result<()> {
        let (_tmp, s) = store()?;
        write_registration(&s, "c1", "example.test")?;

        assert!(s.require().is_err(), "the typed read should refuse it");
        let stored = s
            .stored()?
            .ok_or_else(|| anyhow::anyhow!("the lenient read should still answer"))?;
        assert_eq!(stored.client_id, "c1");
        assert_eq!(stored.server_url, "example.test");

        s.clear()?;
        assert!(s.stored()?.is_none());
        Ok(())
    }

    /// Bypasses `put`, which would refuse these values.
    fn write_registration(
        s: &ClientRegistrationStore,
        client_id: &str,
        server_url: &str,
    ) -> anyhow::Result<()> {
        std::fs::create_dir_all(&s.root)?;
        std::fs::write(
            s.root.join(REGISTRATION_NAME),
            serde_json::to_vec(&serde_json::json!({
                "client_id": client_id,
                "server_url": server_url,
            }))?,
        )?;
        Ok(())
    }

    #[test]
    fn server_url_needs_a_registration() -> anyhow::Result<()> {
        let (_tmp, s) = store()?;
        assert!(matches!(s.server_url(), Err(Error::RegistrationMissing(_))));
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

    #[rstest::rstest]
    #[case::private_key(PRIVATE_KEY_NAME)]
    #[case::public_key(PUBLIC_KEY_NAME)]
    #[case::registration(REGISTRATION_NAME)]
    fn ensure_empty_rejects_existing_artifacts(#[case] name: &str) -> anyhow::Result<()> {
        let (_tmp, s) = store()?;
        s.ensure_empty()?;

        std::fs::create_dir_all(&s.root)?;
        std::fs::write(s.root.join(name), "x")?;

        assert!(matches!(s.ensure_empty(), Err(Error::IdentityExists(_))));
        Ok(())
    }

    #[test]
    fn ensure_empty_ignores_the_machine_scoped_files() -> anyhow::Result<()> {
        let (tmp, s) = store()?;
        ClientDeviceStore::new(identity_dir(&tmp)).put(&DeviceLabels::default())?;
        ClientSettingsStore::new(identity_dir(&tmp)).put(&ClientSettings {
            storage_quota_bytes: 1,
        })?;
        s.ensure_empty()?;
        Ok(())
    }

    #[test]
    fn require_reports_the_missing_file() -> anyhow::Result<()> {
        let (_tmp, s) = store()?;
        assert!(matches!(s.require(), Err(Error::RegistrationMissing(_))));
        Ok(())
    }

    #[test]
    fn clear_leaves_the_machine_scoped_files() -> anyhow::Result<()> {
        let (tmp, s) = store()?;
        let device = ClientDeviceStore::new(identity_dir(&tmp));
        let settings = ClientSettingsStore::new(identity_dir(&tmp));
        s.put_private_key(&key(KEY)?)?;
        s.put_public_key(&PublicKey::try_new("ca7e".repeat(16))?)?;
        s.put(&reg()?)?;
        device.put(&DeviceLabels {
            device_name: Some(TrimmedString::new("keep")),
            device_form_factor: None,
        })?;
        settings.put(&ClientSettings {
            storage_quota_bytes: 1,
        })?;
        s.clear()?;
        assert!(s.get_private_key()?.is_none());
        assert!(s.get()?.is_none());
        assert_eq!(
            device.get()?.device_name.as_ref().map(AsRef::as_ref),
            Some("keep")
        );
        assert!(settings.get()?.is_some());
        s.ensure_empty()?;
        Ok(())
    }

    #[test]
    fn old_registration_with_dropped_fields_still_deserializes() -> anyhow::Result<()> {
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
        assert_eq!(reg.client_id.as_ref(), "ev1_abc");
        assert_eq!(reg.server_url.to_string(), "https://mgmt.example.com");
        Ok(())
    }
}
