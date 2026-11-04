//! The one file-creation rule for the two files that carry the signing key:
//! `identity/id_ed25519` and the `auth export --out` bundle.

use std::fs;
use std::path::Path;

use crate::error::Result;

/// Write a secret so it is never on disk world-readable, even briefly.
///
/// `fs::write` then `chmod` leaves the key at the umask default in between.
/// `.mode()` applies at creation; the reassert (an `fchmod` on the handle)
/// covers an existing file, where `.mode()` is ignored. Both run before
/// `write_all`, so the bytes only land in an already-restricted file.
/// A `0700` parent only narrows the window: modes are checked per open, so
/// anything already inside the directory can still catch it.
pub(crate) fn write(path: &Path, contents: &[u8]) -> Result<()> {
    write_owner_only(path, contents).map_err(|source| {
        pipette_ops::Error::Io {
            path: path.to_path_buf(),
            source,
        }
        .into()
    })
}

#[cfg(unix)]
fn write_owner_only(path: &Path, contents: &[u8]) -> std::io::Result<()> {
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

/// Windows has no creation mode, so the key rests on the profile directory's
/// ACL, not asserted here. Tightening it needs a Win32 ACL call (PIP-461).
#[cfg(not(unix))]
fn write_owner_only(path: &Path, contents: &[u8]) -> std::io::Result<()> {
    fs::write(path, contents)
}

#[cfg(all(test, unix))]
mod tests {
    use std::os::unix::fs::PermissionsExt;

    use anyhow::Context;

    use super::*;

    fn mode(path: &Path) -> anyhow::Result<u32> {
        Ok(fs::metadata(path)?.permissions().mode() & 0o777)
    }

    /// `.mode()` is ignored for an existing file, so this proves the reassert.
    #[test]
    fn an_existing_loose_file_is_tightened_and_replaced() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let path = tmp.path().join("id_ed25519");
        fs::write(&path, b"an older, longer key")?;
        fs::set_permissions(&path, fs::Permissions::from_mode(0o644))?;

        write(&path, b"key")?;

        assert_eq!(mode(&path)?, 0o600);
        assert_eq!(fs::read(&path)?, b"key");
        Ok(())
    }

    #[test]
    fn a_missing_parent_is_an_io_error_naming_the_path() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let path = tmp.path().join("absent").join("id_ed25519");

        let err = write(&path, b"key").err().context("write should fail")?;

        assert!(err.to_string().contains("id_ed25519"), "{err}");
        Ok(())
    }
}
