//! File access under `identity/`, shared by the three stores. Free functions
//! over the root rather than a handle, which would be a second store under
//! each store. Every write asserts the directory's `0700` first; [`ensure`] is
//! private so a store cannot skip it.

use std::fs;
use std::path::{Path, PathBuf};

use serde::Serialize;
use serde::de::DeserializeOwned;

use pipette_ops::fs::{read_json_typed, write_json_typed};

use crate::error::Result;
use crate::secret_file;

pub(super) fn path(root: &Path, name: &str) -> PathBuf {
    root.join(name)
}

/// `None` only when the file is absent; one that is there but unreadable is an
/// error.
pub(super) fn read_json<T: DeserializeOwned>(root: &Path, name: &str) -> Result<Option<T>> {
    let path = path(root, name);
    if !path.exists() {
        return Ok(None);
    }
    Ok(Some(read_json_typed(&path)?))
}

pub(super) fn write_json<T: Serialize>(root: &Path, name: &str, value: &T) -> Result<()> {
    ensure(root)?;
    write_json_typed(&path(root, name), value).map_err(Into::into)
}

/// `None` when absent. For the key files, which hold hex rather than JSON.
pub(super) fn read_text(root: &Path, name: &str) -> Result<Option<String>> {
    let path = path(root, name);
    if !path.exists() {
        return Ok(None);
    }
    fs::read_to_string(&path)
        .map(Some)
        .map_err(|source| pipette_ops::Error::Io { path, source }.into())
}

pub(super) fn write_text(root: &Path, name: &str, contents: &str) -> Result<()> {
    ensure(root)?;
    let path = path(root, name);
    fs::write(&path, contents).map_err(|source| pipette_ops::Error::Io { path, source }.into())
}

/// For the private key, the one file here that is a credential.
pub(super) fn write_secret(root: &Path, name: &str, contents: &[u8]) -> Result<()> {
    ensure(root)?;
    secret_file::write(&path(root, name), contents)
}

/// Idempotent when the file is already gone.
pub(super) fn remove(root: &Path, name: &str) -> Result<()> {
    let path = path(root, name);
    if path.exists() {
        fs::remove_file(&path).map_err(|source| pipette_ops::Error::Io { path, source })?;
    }
    Ok(())
}

/// Create `identity/` at `0700`, reasserting the mode on an existing one, so
/// nothing lands in a directory the rest of the machine can list.
fn ensure(root: &Path) -> Result<()> {
    fs::create_dir_all(root).map_err(|source| pipette_ops::Error::Io {
        path: root.to_path_buf(),
        source,
    })?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(root, fs::Permissions::from_mode(0o700)).map_err(|source| {
            pipette_ops::Error::Io {
                path: root.to_path_buf(),
                source,
            }
        })?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_missing_file_reads_as_none() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let root = tmp.path().join("identity");

        assert!(read_text(&root, "id_ed25519")?.is_none());
        assert!(read_json::<serde_json::Value>(&root, "registration.json")?.is_none());
        Ok(())
    }

    #[test]
    fn text_and_json_round_trip() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let root = tmp.path().join("identity");

        write_text(&root, "id_ed25519.pub", "abcd")?;
        write_json(
            &root,
            "device.json",
            &serde_json::json!({"device_name": "box"}),
        )?;

        assert_eq!(read_text(&root, "id_ed25519.pub")?.as_deref(), Some("abcd"));
        assert_eq!(
            read_json::<serde_json::Value>(&root, "device.json")?,
            Some(serde_json::json!({"device_name": "box"}))
        );
        Ok(())
    }

    #[test]
    fn malformed_json_is_an_error_not_none() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let root = tmp.path().join("identity");
        write_text(&root, "settings.json", "{ not json")?;

        assert!(read_json::<serde_json::Value>(&root, "settings.json").is_err());
        Ok(())
    }

    /// `auth reset` has to finish a reset that failed partway.
    #[test]
    fn remove_is_idempotent() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let root = tmp.path().join("identity");
        write_text(&root, "id_ed25519.pub", "abcd")?;

        remove(&root, "id_ed25519.pub")?;
        remove(&root, "id_ed25519.pub")?;
        assert!(!path(&root, "id_ed25519.pub").exists());
        Ok(())
    }

    #[cfg(unix)]
    #[rstest::rstest]
    #[case::text(|root: &Path| write_text(root, "id_ed25519.pub", "abcd"))]
    #[case::json(|root: &Path| write_json(root, "device.json", &serde_json::json!({})))]
    #[case::secret(|root: &Path| write_secret(root, "id_ed25519", b"key"))]
    fn a_write_tightens_a_loose_identity_dir(
        #[case] write: fn(&Path) -> Result<()>,
    ) -> anyhow::Result<()> {
        use std::os::unix::fs::PermissionsExt;

        let tmp = tempfile::tempdir()?;
        let root = tmp.path().join("identity");
        fs::create_dir_all(&root)?;
        fs::set_permissions(&root, fs::Permissions::from_mode(0o755))?;

        write(&root)?;

        assert_eq!(fs::metadata(&root)?.permissions().mode() & 0o777, 0o700);
        Ok(())
    }
}
