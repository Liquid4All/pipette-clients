//! Dropping the page cache a directory's files hold. A model loaded with
//! `--load-mode none` leaves its file cached after the tool exits; on a phone
//! the next model's load then has to reclaim that cache, and the kernel can
//! swap the new process out instead
//! (`docs/methodology/galaxy-s26-ultra-memory-pressure.md`).

use std::path::Path;

/// What one [`drop_cached_files`] call did.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct DroppedCache {
    /// Regular files the kernel was told to drop.
    pub files: u64,
    /// Files or directories that could not be opened or advised.
    pub failed: u64,
}

/// Asks the kernel to drop the cached pages of every regular file under `dir`
/// (`POSIX_FADV_DONTNEED`), which needs no privilege, unlike `drop_caches`.
/// It drops only clean pages no process maps. Symlinks are not followed. A
/// file that fails is counted and skipped: a cache left in place costs memory,
/// not correctness. Log [`crate::MemoryPressure`] on either side to see what
/// it freed.
pub fn drop_cached_files(dir: &Path) -> DroppedCache {
    let mut dropped = DroppedCache::default();
    walk(dir, &mut dropped);
    tracing::info!(
        dir = %dir.display(),
        files = dropped.files,
        failed = dropped.failed,
        "dropped page cache"
    );
    dropped
}

fn walk(dir: &Path, dropped: &mut DroppedCache) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        dropped.failed += 1;
        return;
    };
    entries.for_each(|entry| {
        let Ok(entry) = entry else {
            dropped.failed += 1;
            return;
        };
        match entry.file_type() {
            Ok(kind) if kind.is_dir() => walk(&entry.path(), dropped),
            Ok(kind) if kind.is_file() => match advise_dont_need(&entry.path()) {
                Ok(()) => dropped.files += 1,
                Err(err) => {
                    tracing::debug!(path = %entry.path().display(), %err, "page cache not dropped");
                    dropped.failed += 1;
                }
            },
            Ok(_) => {}
            Err(_) => dropped.failed += 1,
        }
    });
}

fn advise_dont_need(path: &Path) -> std::io::Result<()> {
    use std::os::fd::AsRawFd;
    let file = std::fs::File::open(path)?;
    // SAFETY: `file` holds the descriptor open for the whole call, and the
    // call reads no memory of ours.
    let rc = unsafe { libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED) };
    // It returns the error number itself rather than -1 and `errno`.
    if rc == 0 {
        Ok(())
    } else {
        Err(std::io::Error::from_raw_os_error(rc))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn advises_every_file_below_the_directory() -> anyhow::Result<()> {
        let root = tempfile::tempdir()?;
        let nested = root.path().join("org__repo").join("blobs");
        std::fs::create_dir_all(&nested)?;
        std::fs::write(root.path().join("manifest.json"), "{}")?;
        std::fs::write(nested.join("model.gguf"), [0_u8; 4096])?;
        std::os::unix::fs::symlink(nested.join("model.gguf"), root.path().join("link"))?;

        assert_eq!(
            drop_cached_files(root.path()),
            DroppedCache {
                files: 2,
                failed: 0
            }
        );
        Ok(())
    }

    #[test]
    fn counts_a_missing_directory_as_failed() {
        assert_eq!(
            drop_cached_files(Path::new("/nonexistent/pipette-models")),
            DroppedCache {
                files: 0,
                failed: 1
            }
        );
    }
}
