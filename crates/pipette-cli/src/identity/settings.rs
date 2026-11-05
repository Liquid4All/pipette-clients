//! Client-wide settings (`identity/settings.json`): local policy for this box,
//! read by [`PipetteWorkspace::open`] before any command runs.
//!
//! [`PipetteWorkspace::open`]: crate::workspace::PipetteWorkspace::open

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use super::files;
use crate::error::Result;

const SETTINGS_NAME: &str = "settings.json";

/// An absent file means the built-in defaults. No serde defaults on the fields:
/// a partial file is an error rather than a half-applied configuration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClientSettings {
    /// Cap on the disk the model and runtime artifact stores may occupy
    /// (`docs/storage-quota.md`).
    pub storage_quota_bytes: u64,
}

/// Minted from the workspace with `ws.settings()`.
#[derive(Debug, Clone)]
pub struct ClientSettingsStore {
    root: PathBuf,
}

impl ClientSettingsStore {
    /// `root` = workspace `identity/` directory.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// `None` when absent; the caller falls back to the built-in defaults.
    pub fn get(&self) -> Result<Option<ClientSettings>> {
        files::read_json(&self.root, SETTINGS_NAME)
    }

    pub fn put(&self, settings: &ClientSettings) -> Result<()> {
        files::write_json(&self.root, SETTINGS_NAME, settings)
    }

    /// Idempotent when absent.
    pub fn clear(&self) -> Result<()> {
        files::remove(&self.root, SETTINGS_NAME)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Error;

    fn store() -> anyhow::Result<(tempfile::TempDir, ClientSettingsStore)> {
        let tmp = tempfile::tempdir()?;
        let s = ClientSettingsStore::new(tmp.path().join("identity"));
        Ok((tmp, s))
    }

    #[test]
    fn settings_round_trip_and_clear() -> anyhow::Result<()> {
        let (_tmp, s) = store()?;
        assert!(s.get()?.is_none());
        let settings = ClientSettings {
            storage_quota_bytes: 42 * 1024 * 1024 * 1024,
        };
        s.put(&settings)?;
        assert_eq!(s.get()?, Some(settings));
        s.clear()?;
        assert!(s.get()?.is_none());
        s.clear()?;
        Ok(())
    }

    #[test]
    fn malformed_settings_surface_a_parse_error() -> anyhow::Result<()> {
        let (_tmp, s) = store()?;
        s.put(&ClientSettings {
            storage_quota_bytes: 1,
        })?;
        std::fs::write(s.root.join(SETTINGS_NAME), b"{not json")?;
        let err = s
            .get()
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
    fn settings_without_every_field_do_not_deserialize() {
        assert!(serde_json::from_value::<ClientSettings>(serde_json::json!({})).is_err());
    }
}
