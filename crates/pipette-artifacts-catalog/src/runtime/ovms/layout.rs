//! Where an unpacked OVMS release keeps its server and libraries. The installer
//! checks an archive against this and the backend launches from it, so neither
//! re-derives the other's paths.

use std::path::{Path, PathBuf};

/// The archive's top-level directory, which is the whole payload.
const PAYLOAD_DIR: &str = "ovms";

/// An OVMS release unpacked under a runtime's install directory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OvmsInstall {
    payload: PathBuf,
}

impl OvmsInstall {
    /// The release unpacked into `install_dir`, the directory a stored runtime's
    /// `dir` names.
    pub fn under(install_dir: &Path) -> Self {
        Self {
            payload: install_dir.join(PAYLOAD_DIR),
        }
    }

    /// The server binary on this host.
    pub fn executable(&self) -> PathBuf {
        self.executable_for(cfg!(windows))
    }

    /// `ovms/ovms.exe` in the Windows zip, `ovms/bin/ovms` in the Linux tarballs.
    fn executable_for(&self, windows: bool) -> PathBuf {
        if windows {
            self.payload.join("ovms.exe")
        } else {
            self.payload.join("bin").join("ovms")
        }
    }

    /// The loader search variable the server needs, and the directory to put
    /// first on it.
    pub fn library_path(&self) -> (&'static str, PathBuf) {
        self.library_path_for(cfg!(windows))
    }

    /// The Windows zip keeps its DLLs beside `ovms.exe`, the Linux tarballs in
    /// `ovms/lib`.
    fn library_path_for(&self, windows: bool) -> (&'static str, PathBuf) {
        if windows {
            ("PATH", self.payload.clone())
        } else {
            ("LD_LIBRARY_PATH", self.payload.join("lib"))
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::windows(true, &["ovms", "ovms.exe"])]
    #[case::linux(false, &["ovms", "bin", "ovms"])]
    fn the_executable_sits_where_each_archive_puts_it(
        #[case] windows: bool,
        #[case] parts: &[&str],
    ) {
        let root = Path::new("/store/blobs");
        let expected = parts
            .iter()
            .fold(root.to_path_buf(), |p, part| p.join(part));
        assert_eq!(OvmsInstall::under(root).executable_for(windows), expected);
    }

    #[rstest]
    #[case::windows(true, "PATH", &["ovms"])]
    #[case::linux(false, "LD_LIBRARY_PATH", &["ovms", "lib"])]
    fn the_libraries_sit_where_each_archive_puts_them(
        #[case] windows: bool,
        #[case] var: &str,
        #[case] parts: &[&str],
    ) {
        let root = Path::new("/store/blobs");
        let expected = parts
            .iter()
            .fold(root.to_path_buf(), |p, part| p.join(part));
        assert_eq!(
            OvmsInstall::under(root).library_path_for(windows),
            (var, expected)
        );
    }
}
