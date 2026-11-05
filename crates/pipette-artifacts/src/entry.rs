use std::{
    ffi::OsString,
    fs::{self, File},
    io::{self, Write},
    path::{Path, PathBuf},
};

use anyhow::Context;
use serde::Serialize;
use uuid::Uuid;

/// Payload subdirectory of an entry, beside the manifest at the entry root.
pub const BLOBS_DIR_NAME: &str = "blobs";

/// TOML record at the entry root for directory artifacts (`models/` and
/// `runtimes/`).
pub const MANIFEST_NAME: &str = "manifest.toml";

/// Scratch directory under a store root, used by
/// [`install_dir_computing_manifest`].
pub const STAGING_DIR_NAME: &str = ".staging";

/// Directory under a store root holding each entry key's lock files, taken by
/// [`lock_entry`] and [`hold_entry`]. The quota survey skips it.
pub(crate) const LOCKS_DIR_NAME: &str = ".locks";

/// Suffix of a lock file: an entry's under [`LOCKS_DIR_NAME`], and the one
/// beside each staged dir that its fetch holds locked.
pub(crate) const LOCK_SUFFIX: &str = ".lock";

/// Suffix of an entry's in-use lock under [`LOCKS_DIR_NAME`]. Separate from
/// the fetch's lock so a miss never waits for a run.
pub(crate) const USE_SUFFIX: &str = ".use";

/// How many fresh staging names [`create_staged_dir`] tries when a sweep
/// removes the lock file before the fetch could lock it.
const STAGING_LOCK_ATTEMPTS: usize = 3;

fn child_manifest_path(target_dir: &Path, manifest_name: &str) -> anyhow::Result<PathBuf> {
    validate_manifest_name(manifest_name)?;
    Ok(target_dir.join(manifest_name))
}

/// Bytes a path occupies on disk: recursive, symlinks counted as the link itself
/// and never followed, `st_blocks * 512` on unix so the answer matches `du`.
/// Publish stamps it into the manifest as `blobs_bytes`, so a sweep normally
/// totals a store without walking a payload.
///
/// An unreadable child contributes 0: a size we cannot read is space we cannot
/// prove we would reclaim, and a permissions blip must not fail a sweep.
pub(crate) fn entry_size_bytes(path: &Path) -> u64 {
    let Ok(metadata) = fs::symlink_metadata(path) else {
        return 0;
    };
    let mut total = allocated_bytes(&metadata);
    // `symlink_metadata` reports a symlinked dir as a symlink, so recursion
    // stops at the link and never leaves the entry.
    if metadata.is_dir() {
        let Ok(children) = fs::read_dir(path) else {
            return total;
        };
        total = children.flatten().fold(total, |sum, child| {
            sum.saturating_add(entry_size_bytes(&child.path()))
        });
    }
    total
}

#[cfg(unix)]
fn allocated_bytes(metadata: &fs::Metadata) -> u64 {
    use std::os::unix::fs::MetadataExt;

    metadata.blocks().saturating_mul(512)
}

#[cfg(not(unix))]
fn allocated_bytes(metadata: &fs::Metadata) -> u64 {
    metadata.len()
}

/// Rewrite a *published* entry's manifest in place: write beside it, then
/// rename over. A torn write on the live path would strand a good entry.
///
/// The new bytes reach the disk before the rename, and the rename reaches the
/// disk before this returns. A crash at any point leaves the old manifest or the
/// new one. Without the first sync, a file system can keep the rename and lose
/// the data. The manifest then has the right size and holds only zeros.
pub(crate) fn rewrite_manifest<M: Serialize>(
    entry_dir: &Path,
    manifest_name: &str,
    manifest: &M,
) -> anyhow::Result<()> {
    let manifest_path = child_manifest_path(entry_dir, manifest_name)?;
    let temp_path = entry_dir.join(format!("{manifest_name}.{}.tmp", Uuid::new_v4()));
    if let Err(err) = write_toml_file(&temp_path, manifest) {
        let _ = fs::remove_file(&temp_path);
        return Err(err);
    }
    if let Err(source) = fs::rename(&temp_path, &manifest_path) {
        let _ = fs::remove_file(&temp_path);
        return Err(anyhow::Error::new(source)
            .context(format!("failed to replace {}", manifest_path.display())));
    }
    sync_dir(entry_dir)
}

/// Stage a directory, write the manifest the `prepare_dir` closure returns,
/// publish it atomically, and return the manifest.
///
/// The closure computes the manifest because some fields (`blobs_bytes`, a
/// runtime's binary paths and archive hash) are known only once the payload
/// has landed. Everything in the entry is synced before the rename, so a crash
/// cannot publish a manifest over a payload that holds no data.
pub fn install_dir_computing_manifest<M, F>(
    staging_root: &Path,
    target_dir: &Path,
    staging_name: &str,
    manifest_name: &str,
    prepare_dir: F,
) -> anyhow::Result<M>
where
    M: Serialize,
    F: FnOnce(&Path) -> anyhow::Result<M>,
{
    replace_dir_from_staged(staging_root, target_dir, staging_name, |staged_dir| {
        let manifest = prepare_dir(staged_dir)?;
        write_manifest_guarded(staged_dir, manifest_name, &manifest)?;
        Ok(manifest)
    })
}

/// Refuses to clobber a same-named file the prepared bundle produced: an
/// archive that shipped its own `manifest.toml` would otherwise be silently
/// overwritten by our record.
fn write_manifest_guarded<M: Serialize>(
    staged_dir: &Path,
    manifest_name: &str,
    manifest: &M,
) -> anyhow::Result<()> {
    let manifest_path = child_manifest_path(staged_dir, manifest_name)?;
    if manifest_path.exists() {
        anyhow::bail!("staged bundle already contains reserved name {manifest_name}");
    }
    write_toml_file(&manifest_path, manifest)
}

/// The bytes are on disk when this returns; the caller must sync the
/// directory entry. The parent is not created: recreating an entry a sweep
/// just moved away would publish a manifest with no payload.
fn write_toml_file<T: Serialize>(path: &Path, value: &T) -> anyhow::Result<()> {
    let rendered = toml::to_string_pretty(value)
        .with_context(|| format!("failed to serialize {}", path.display()))?;
    let mut file =
        File::create(path).with_context(|| format!("failed to write {}", path.display()))?;
    file.write_all(rendered.as_bytes())
        .and_then(|()| sync_file(&file))
        .with_context(|| format!("failed to write {}", path.display()))
}

/// Some file systems cannot sync a file or a directory and report
/// `Unsupported` or `InvalidInput` (EINVAL); that counts as success, as the
/// entry lock does for a file system without locks.
pub fn sync_file(file: &File) -> io::Result<()> {
    match file.sync_all() {
        Err(source)
            if matches!(
                source.kind(),
                io::ErrorKind::Unsupported | io::ErrorKind::InvalidInput
            ) =>
        {
            Ok(())
        }
        synced => synced,
    }
}

/// Make the entries of `dir` durable: a file created or renamed in it survives
/// a crash once this returns.
#[cfg(unix)]
pub fn sync_dir(dir: &Path) -> anyhow::Result<()> {
    File::open(dir)
        .and_then(|handle| sync_file(&handle))
        .with_context(|| format!("failed to sync {}", dir.display()))
}

/// A no-op: Windows cannot open a directory as a [`File`].
#[cfg(not(unix))]
pub fn sync_dir(_dir: &Path) -> anyhow::Result<()> {
    Ok(())
}

/// Make every file and directory under `path` durable, `path` included. A
/// symlink is not followed: the sync of its parent directory records it.
fn sync_tree(path: &Path) -> anyhow::Result<()> {
    let metadata = fs::symlink_metadata(path)
        .with_context(|| format!("failed to inspect {}", path.display()))?;
    if metadata.is_dir() {
        fs::read_dir(path)
            .with_context(|| format!("failed to read {}", path.display()))?
            .try_for_each(|child| {
                let child = child.with_context(|| format!("failed to read {}", path.display()))?;
                sync_tree(&child.path())
            })?;
        return sync_dir(path);
    }
    if !metadata.is_file() {
        return Ok(());
    }
    open_for_sync(path)
        .and_then(|file| sync_file(&file))
        .with_context(|| format!("failed to sync {}", path.display()))
}

#[cfg(unix)]
fn open_for_sync(path: &Path) -> io::Result<File> {
    File::open(path)
}

/// Windows flushes a file only through a handle with write access.
#[cfg(not(unix))]
fn open_for_sync(path: &Path) -> io::Result<File> {
    File::options().write(true).open(path)
}

fn replace_dir_from_staged<T, F>(
    staging_root: &Path,
    target: &Path,
    staging_name: &str,
    prepare: F,
) -> anyhow::Result<T>
where
    F: FnOnce(&Path) -> anyhow::Result<T>,
{
    // Must outlive the publish or cleanup: its lock keeps other processes'
    // sweeps off the dir.
    let staged = create_staged_dir(staging_root, staging_name)?;
    let prepared = match prepare(&staged.dir).and_then(|prepared| {
        sync_tree(&staged.dir)?;
        Ok(prepared)
    }) {
        Ok(prepared) => prepared,
        Err(err) => {
            return Err(preserve_with_cleanup(
                err,
                cleanup_path(&staged.dir),
                &staged.dir,
            ));
        }
    };
    if let Err(err) = replace_staged_path(&staged.dir, target) {
        let err = preserve_with_cleanup(err, cleanup_path(target), target);
        return Err(preserve_with_cleanup(
            err,
            cleanup_path(&staged.dir),
            &staged.dir,
        ));
    }
    // If a crash loses the rename of a publish, no entry exists. The next
    // `ensure` fetches it again, so a failed sync here does not fail the
    // install.
    if let Some(parent) = target.parent()
        && let Err(err) = sync_dir(parent)
    {
        tracing::warn!(
            path = %parent.display(),
            error = %format_args!("{err:#}"),
            "could not sync the store root after a publish"
        );
    }
    Ok(prepared)
}

/// A sibling, not a child: Windows refuses to rename a dir with a file open
/// inside it.
pub(crate) fn staging_lock_path(staged: &Path) -> PathBuf {
    let mut path = OsString::from(staged.as_os_str());
    path.push(LOCK_SUFFIX);
    PathBuf::from(path)
}

/// A staged dir and the exclusive lock its fetch holds on the file beside it.
/// The OS drops the lock however the process exits, so a sweep in another
/// process sharing the store that can take it knows the fetch is dead (see
/// [`crate::quota`]).
struct StagedDir {
    dir: PathBuf,
    lock_path: PathBuf,
    _lock: File,
}

impl StagedDir {
    fn create(dir: PathBuf) -> anyhow::Result<Option<Self>> {
        let lock_path = staging_lock_path(&dir);
        let lock = File::create_new(&lock_path)
            .with_context(|| format!("failed to create {}", lock_path.display()))?;
        Self::lock(dir, lock_path, lock)
    }

    /// `None` when a sweep removed the lock file before it was locked: it took
    /// the file for a dead fetch's, so this name is spent.
    fn lock(dir: PathBuf, lock_path: PathBuf, lock: File) -> anyhow::Result<Option<Self>> {
        match lock.lock() {
            Ok(()) => {}
            // A filesystem without locks: a sweep cannot take this lock either,
            // so it counts the fetch as running and leaves it alone.
            Err(source) if source.kind() == io::ErrorKind::Unsupported => {}
            Err(source) => {
                return Err(anyhow::Error::new(source)
                    .context(format!("failed to lock {}", lock_path.display())));
            }
        }
        // A sweep removes a lock file only while holding its lock, so one
        // still present now stays ours for as long as the lock is held.
        if !lock_path.exists() {
            return Ok(None);
        }
        let staged = Self {
            dir,
            lock_path,
            _lock: lock,
        };
        fs::create_dir_all(&staged.dir)
            .with_context(|| format!("failed to create {}", staged.dir.display()))?;
        Ok(Some(staged))
    }
}

impl Drop for StagedDir {
    /// Removed while still locked, so a sweep never finds the file unlocked
    /// while this fetch is alive.
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.lock_path);
    }
}

/// Exclusive lock on the entry `key` under `store_root`, blocking while
/// another process sharing the store holds it. A store's `ensure` takes it on
/// a miss and looks again before fetching, so two processes missing one entry
/// download it once.
///
/// The lock file is never removed: a waiter would then hold the removed file
/// while a newcomer locks a fresh one, both believing they own the entry. One
/// empty file per key ever fetched is the cost.
pub(crate) fn lock_entry(store_root: &Path, key: &str) -> anyhow::Result<File> {
    let (lock, path) = open_entry_lock(store_root, key, LOCK_SUFFIX)?;
    let source = match lock.try_lock() {
        Ok(()) => return Ok(lock),
        Err(fs::TryLockError::WouldBlock) => {
            tracing::info!(entry = key, "another process is fetching it; waiting");
            match lock.lock() {
                Ok(()) => return Ok(lock),
                Err(source) => source,
            }
        }
        Err(fs::TryLockError::Error(source)) => source,
    };
    // A filesystem without locks: fetch unlocked.
    if source.kind() == io::ErrorKind::Unsupported {
        return Ok(lock);
    }
    Err(anyhow::Error::new(source).context(format!("failed to lock {}", path.display())))
}

/// Shared lock on `.locks/<key>.use` while this process runs from entry `key`; a
/// sweep removes the entry only after taking it exclusively without waiting (see
/// [`crate::quota`]). `None` where it cannot be taken (no locks, a read-only
/// store without the file): the run goes on unlocked.
pub(crate) fn hold_entry(store_root: &Path, key: &str) -> Option<File> {
    let path = entry_lock_path(store_root, key, USE_SUFFIX);
    // A shared lock needs only read access, so a read-only store still locks.
    let held = match File::open(&path) {
        Ok(lock) => Ok(lock),
        Err(_) => open_entry_lock(store_root, key, USE_SUFFIX).map(|(lock, _)| lock),
    }
    .and_then(|lock| {
        lock.lock_shared()
            .with_context(|| format!("failed to lock {}", path.display()))?;
        Ok(lock)
    });
    match held {
        Ok(lock) => Some(lock),
        Err(err) => {
            tracing::warn!(
                entry = key,
                error = %format_args!("{err:#}"),
                "running it unlocked, so a sweep in another process may remove it"
            );
            None
        }
    }
}

/// The lock file `.locks/<key><suffix>` under `store_root`, created if missing.
pub(crate) fn open_entry_lock(
    store_root: &Path,
    key: &str,
    suffix: &str,
) -> anyhow::Result<(File, PathBuf)> {
    let locks = store_root.join(LOCKS_DIR_NAME);
    fs::create_dir_all(&locks).with_context(|| format!("failed to create {}", locks.display()))?;
    let path = entry_lock_path(store_root, key, suffix);
    let lock = File::options()
        .create(true)
        .write(true)
        .truncate(false)
        .open(&path)
        .with_context(|| format!("failed to open {}", path.display()))?;
    Ok((lock, path))
}

fn entry_lock_path(store_root: &Path, key: &str, suffix: &str) -> PathBuf {
    store_root
        .join(LOCKS_DIR_NAME)
        .join(format!("{key}{suffix}"))
}

fn validate_manifest_name(manifest_name: &str) -> anyhow::Result<()> {
    if manifest_name.is_empty() {
        anyhow::bail!("manifest name must not be empty");
    }
    if Path::new(manifest_name).is_absolute() {
        anyhow::bail!("manifest name must be a relative path");
    }
    if manifest_name.contains(['/', '\\']) {
        anyhow::bail!("manifest name must not contain path separators");
    }
    if manifest_name == "." || manifest_name == ".." {
        anyhow::bail!("manifest name must not be '.' or '..'");
    }
    Ok(())
}

fn create_staged_dir(staging_root: &Path, staging_name: &str) -> anyhow::Result<StagedDir> {
    fs::create_dir_all(staging_root)
        .with_context(|| format!("failed to create {}", staging_root.display()))?;
    (0..STAGING_LOCK_ATTEMPTS)
        .find_map(|_| StagedDir::create(staged_path(staging_root, staging_name)).transpose())
        .transpose()?
        .with_context(|| {
            format!(
                "a sweep removed the staging lock under {} {STAGING_LOCK_ATTEMPTS} times in a row",
                staging_root.display()
            )
        })
}

fn replace_staged_path(staged: &Path, target: &Path) -> anyhow::Result<()> {
    if let Some(parent) = target.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    if target.exists() {
        remove_path(target)?;
    }
    move_into_place(staged, target)
}

/// Characters of the entry name kept in a staging directory's name: enough to
/// recognize an orphan from a crashed install, short for Windows `MAX_PATH`
/// (see [`staged_path`]).
const STAGING_NAME_PREFIX_LEN: usize = 16;

/// Hex chars of uniqueness in a staging directory's name: 32 bits is ample to
/// tell concurrent installs of one entry apart.
const STAGING_UNIQUE_HEX: usize = 8;

/// Transient path under `staging_root`: an install before its atomic move into
/// place, or an entry a sweep moved out to delete.
///
/// Kept short: Windows caps a path at 260 characters unless both the OS and the
/// binary opt into long paths, and a staged venv
/// (`blobs/venv/Lib/site-packages/<dist>.dist-info/<file>`) is the longest path
/// this store creates. With the full entry name and a hyphenated UUID it reached
/// 261 characters under a user's home directory and `uv` failed mid-install
/// with "system cannot find the path specified".
/// Truncating here keeps the final entry name self-describing.
pub(crate) fn staged_path(staging_root: &Path, staging_name: &str) -> PathBuf {
    let safe_name = if staging_name.is_empty() {
        "asset".to_string()
    } else {
        staging_name.replace(['/', '\\'], "_")
    };
    // Char-wise, not byte-wise: an entry name is normalized ASCII today, but
    // slicing a multi-byte boundary would panic rather than truncate.
    let head: String = safe_name.chars().take(STAGING_NAME_PREFIX_LEN).collect();
    let mut unique = Uuid::new_v4().simple().to_string();
    unique.truncate(STAGING_UNIQUE_HEX);
    staging_root.join(format!("{head}.staged-{unique}"))
}

/// Attempts at the publish rename: on Windows an antivirus scan of a fresh file
/// refuses it for a moment (rustup and npm retry too); elsewhere a refusal lasts.
const RENAME_ATTEMPTS: u64 = 5;

/// Staging sits under the store root, so the rename never crosses a file
/// system; a copy instead would publish the manifest before the payload.
fn move_into_place(source: &Path, target: &Path) -> anyhow::Result<()> {
    let mut attempt = 1;
    loop {
        match fs::rename(source, target) {
            Err(err)
                if cfg!(windows)
                    && err.kind() == io::ErrorKind::PermissionDenied
                    && attempt < RENAME_ATTEMPTS =>
            {
                std::thread::sleep(std::time::Duration::from_millis(100 * attempt));
                attempt += 1;
            }
            result => {
                return result.with_context(|| {
                    format!(
                        "failed to stage {} into {}",
                        source.display(),
                        target.display()
                    )
                });
            }
        }
    }
}

fn remove_path(path: &Path) -> anyhow::Result<()> {
    if !path.exists() {
        return Ok(());
    }
    fs::remove_dir_all(path).with_context(|| format!("failed to remove {}", path.display()))
}

fn cleanup_path(path: &Path) -> anyhow::Result<()> {
    if !path.exists() {
        return Ok(());
    }
    let metadata = fs::symlink_metadata(path)
        .with_context(|| format!("failed to read metadata for {}", path.display()))?;
    if metadata.is_dir() {
        fs::remove_dir_all(path).with_context(|| format!("failed to remove {}", path.display()))
    } else {
        fs::remove_file(path).with_context(|| format!("failed to remove {}", path.display()))
    }
}

fn preserve_with_cleanup(
    err: anyhow::Error,
    cleanup: anyhow::Result<()>,
    path: &Path,
) -> anyhow::Error {
    match cleanup {
        Ok(()) => err,
        Err(cleanup_err) => anyhow::anyhow!(
            "{err:#}\nadditionally failed to clean up {}: {cleanup_err:#}",
            path.display()
        ),
    }
}

#[cfg(test)]
mod tests {
    use serde::{Deserialize, Serialize};

    use super::*;

    #[test]
    fn staged_names_are_bounded_for_windows_max_path() {
        let root = Path::new("/store/.staging");
        let long_key = format!("uv-openvino__3.11__pip__{}", "a".repeat(64));
        let staged = staged_path(root, &long_key);
        let name = staged
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or_default();

        // 16 head + ".staged-" + 8 hex.
        assert_eq!(name.len(), STAGING_NAME_PREFIX_LEN + 8 + STAGING_UNIQUE_HEX);
        assert!(name.starts_with("uv-openvino__3.1"), "got {name}");

        let deepest = "/blobs/venv/Lib/site-packages/             openvino_tokenizers-2026.4.1.0.dist-info/RECORD";
        let base = "C:\\Users\\somebody\\benchmarks\\.pipette\\runtimes\\.staging";
        assert!(
            base.len() + 1 + name.len() + deepest.len() < 260,
            "staged venv path would exceed MAX_PATH"
        );
    }

    #[test]
    fn staged_names_are_unique_per_call() {
        let root = Path::new("/store/.staging");
        assert_ne!(staged_path(root, "same-key"), staged_path(root, "same-key"));
    }

    #[test]
    fn staged_names_survive_a_short_or_empty_entry_name() {
        let root = Path::new("/store/.staging");
        let short = staged_path(root, "mlx");
        assert!(
            short
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or_default()
                .starts_with("mlx.staged-"),
            "got {short:?}"
        );
        assert!(
            staged_path(root, "")
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or_default()
                .starts_with("asset.staged-")
        );
    }

    #[derive(Debug, Deserialize, Serialize)]
    struct TestManifest {
        kind: String,
    }

    fn temp_root(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "pipette-store-assets-{label}-{}-{}",
            std::process::id(),
            Uuid::new_v4()
        ))
    }

    fn write_file(path: &Path, contents: &[u8]) -> anyhow::Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, contents)?;
        Ok(())
    }

    #[test]
    fn install_dir_overwrites_existing_tree_and_manifest() -> anyhow::Result<()> {
        let root = temp_root("install-dir");
        let staging_root = root.join(STAGING_DIR_NAME);
        let target = root.join("runtime");
        write_file(&target.join("old.txt"), b"old")?;
        write_file(
            &child_manifest_path(&target, MANIFEST_NAME)?,
            b"kind = \"old\"\n",
        )?;

        install_dir_computing_manifest(
            &staging_root,
            &target,
            "runtime",
            MANIFEST_NAME,
            |staged| {
                write_file(&staged.join("bin").join("llama-server"), b"server")?;
                Ok(TestManifest {
                    kind: "new".to_string(),
                })
            },
        )?;

        assert!(!target.join("old.txt").exists());
        assert_eq!(
            fs::read(target.join("bin").join("llama-server"))?,
            b"server"
        );
        assert_eq!(
            fs::read_to_string(child_manifest_path(&target, MANIFEST_NAME)?)?,
            "kind = \"new\"\n"
        );
        assert!(fs::read_dir(&staging_root)?.next().is_none());

        fs::remove_dir_all(&root)?;
        Ok(())
    }

    /// Publishing removes the lock file too.
    #[test]
    fn install_dir_holds_the_staging_lock_while_preparing() -> anyhow::Result<()> {
        let root = tempfile::tempdir()?;
        let staging_root = root.path().join(STAGING_DIR_NAME);
        let target = root.path().join("runtime");

        install_dir_computing_manifest(
            &staging_root,
            &target,
            "runtime",
            MANIFEST_NAME,
            |staged| {
                let lock = File::open(staging_lock_path(staged))?;
                anyhow::ensure!(
                    matches!(lock.try_lock(), Err(fs::TryLockError::WouldBlock)),
                    "the staging lock is free while the fetch runs"
                );
                Ok(TestManifest {
                    kind: "new".to_string(),
                })
            },
        )?;

        assert!(target.join(MANIFEST_NAME).exists());
        assert_eq!(fs::read_dir(&staging_root)?.count(), 0);
        Ok(())
    }

    /// A sweep that removed the lock file before the fetch locked it took it
    /// for a dead fetch's, so the fetch must not stage under that name.
    #[test]
    fn a_lock_file_removed_before_it_is_locked_spends_the_name() -> anyhow::Result<()> {
        let root = tempfile::tempdir()?;
        let dir = root.path().join("runtime.staged-1");
        let lock_path = staging_lock_path(&dir);
        let lock = File::create_new(&lock_path)?;
        fs::remove_file(&lock_path)?;

        anyhow::ensure!(
            StagedDir::lock(dir.clone(), lock_path, lock)?.is_none(),
            "staged under a lock file a sweep removed"
        );
        assert!(!dir.exists());
        Ok(())
    }

    /// A second taker of an entry lock waits for the first to release it, and
    /// the lock file stays for the next one.
    #[test]
    fn an_entry_lock_blocks_until_released() -> anyhow::Result<()> {
        let root = tempfile::tempdir()?;
        let held = lock_entry(root.path(), "llama")?;
        let (taken, taken_rx) = std::sync::mpsc::channel();

        std::thread::scope(|s| -> anyhow::Result<()> {
            let waiter = s.spawn(|| -> anyhow::Result<()> {
                let _lock = lock_entry(root.path(), "llama")?;
                taken.send(())?;
                Ok(())
            });
            let waited = taken_rx.recv_timeout(std::time::Duration::from_millis(200));
            anyhow::ensure!(waited.is_err(), "took a held entry lock");
            drop(held);
            taken_rx.recv_timeout(std::time::Duration::from_secs(10))?;
            waiter
                .join()
                .map_err(|_| anyhow::anyhow!("the waiter panicked"))?
        })?;

        let lock_file = root.path().join(LOCKS_DIR_NAME).join("llama.lock");
        anyhow::ensure!(lock_file.exists(), "the entry lock file was removed");
        Ok(())
    }

    #[test]
    fn an_entry_lock_does_not_wait_for_an_entry_in_use() -> anyhow::Result<()> {
        let root = tempfile::tempdir()?;
        let running = hold_entry(root.path(), "llama");
        anyhow::ensure!(running.is_some(), "could not hold the entry");
        let (taken, taken_rx) = std::sync::mpsc::channel();

        let waited = std::thread::scope(|s| {
            let waiter = s.spawn(|| -> anyhow::Result<()> {
                let _lock = lock_entry(root.path(), "llama")?;
                Ok(taken.send(())?)
            });
            let waited = taken_rx.recv_timeout(std::time::Duration::from_secs(10));
            // Released before the join, so a waiter blocked on it still returns.
            drop(running);
            waiter
                .join()
                .map_err(|_| anyhow::anyhow!("the waiter panicked"))??;
            anyhow::Ok(waited)
        })?;

        anyhow::ensure!(waited.is_ok(), "waited for an entry in use");
        Ok(())
    }

    #[test]
    fn install_dir_rejects_unsafe_manifest_name() -> anyhow::Result<()> {
        let root = temp_root("unsafe-manifest-name");
        let staging_root = root.join(STAGING_DIR_NAME);
        let target = root.join("runtime");

        let err = install_dir_computing_manifest(
            &staging_root,
            &target,
            "runtime",
            "../manifest.json",
            |_staged| {
                Ok(TestManifest {
                    kind: "new".to_string(),
                })
            },
        )
        .err()
        .context("unsafe manifest name should error")?;

        assert!(
            err.to_string()
                .contains("manifest name must not contain path separators")
        );
        assert!(!target.exists());
        Ok(())
    }

    #[test]
    fn install_dir_cleans_target_after_commit_failure() -> anyhow::Result<()> {
        let root = temp_root("commit-failure-dir");
        let staging_root = root.join(STAGING_DIR_NAME);
        let parent = root.join("parent");
        let target = parent.join("runtime");
        write_file(&target.join("old.txt"), b"old")?;

        let err = install_dir_computing_manifest(
            &staging_root,
            &target,
            "runtime",
            "manifest.json",
            |staged| {
                write_file(&staged.join("bin").join("llama-bench"), b"bench")?;
                fs::remove_dir_all(&parent)?;
                fs::write(&parent, b"blocker")?;
                Ok(TestManifest {
                    kind: "new".to_string(),
                })
            },
        )
        .err()
        .context("dir commit failure should error")?;

        let message = err.to_string();
        assert!(message.contains("failed to create") || message.contains("failed to stage"));
        assert!(!target.exists());
        assert!(parent.is_file());
        assert!(fs::read_dir(&staging_root)?.next().is_none());
        fs::remove_file(&parent)?;
        fs::remove_dir_all(&root)?;
        Ok(())
    }

    /// A failed rename must surface, not merge a copy into a live entry.
    #[test]
    fn move_into_place_fails_when_the_rename_fails() -> anyhow::Result<()> {
        let root = tempfile::tempdir()?;
        let source = root.path().join("staged");
        let target = root.path().join("runtime");
        write_file(&source.join("new.txt"), b"new")?;
        // A non-empty target makes the rename fail on every platform.
        write_file(&target.join("old.txt"), b"old")?;

        anyhow::ensure!(
            move_into_place(&source, &target).is_err(),
            "a failed rename should error"
        );
        assert!(!target.join("new.txt").exists());
        assert!(source.join("new.txt").exists());
        Ok(())
    }

    #[cfg(windows)]
    #[test]
    fn move_into_place_retries_a_briefly_refused_rename() -> anyhow::Result<()> {
        let root = tempfile::tempdir()?;
        let source = root.path().join("staged");
        let target = root.path().join("runtime");
        write_file(&source.join("new.txt"), b"new")?;
        // Windows refuses to rename a dir with a file open inside it, as a scan holds one.
        let scanned = File::open(source.join("new.txt"))?;

        let moved = std::thread::scope(|s| {
            s.spawn(move || {
                std::thread::sleep(std::time::Duration::from_millis(50));
                drop(scanned);
            });
            move_into_place(&source, &target)
        });

        moved?;
        assert!(target.join("new.txt").exists());
        Ok(())
    }

    #[test]
    fn install_dir_computing_manifest_returns_and_publishes() -> anyhow::Result<()> {
        let root = temp_root("computing-manifest");
        let staging_root = root.join(STAGING_DIR_NAME);
        let target = root.join("runtime");

        let manifest: TestManifest = install_dir_computing_manifest(
            &staging_root,
            &target,
            "runtime",
            MANIFEST_NAME,
            |staged| {
                write_file(&staged.join("bin").join("llama-server"), b"server")?;
                Ok(TestManifest {
                    kind: "computed".to_string(),
                })
            },
        )?;

        assert_eq!(manifest.kind, "computed");
        assert_eq!(
            fs::read(target.join("bin").join("llama-server"))?,
            b"server"
        );
        assert_eq!(
            fs::read_to_string(child_manifest_path(&target, MANIFEST_NAME)?)?,
            "kind = \"computed\"\n"
        );
        assert!(fs::read_dir(&staging_root)?.next().is_none());

        fs::remove_dir_all(&root)?;
        Ok(())
    }

    #[test]
    fn install_dir_rejects_archive_shipping_reserved_manifest() -> anyhow::Result<()> {
        let root = temp_root("reserved-collision");
        let staging_root = root.join(STAGING_DIR_NAME);
        let target = root.join("runtime");

        let err = install_dir_computing_manifest(
            &staging_root,
            &target,
            "runtime",
            "manifest.json",
            |staged| {
                fs::write(staged.join("manifest.json"), b"theirs")?;
                Ok(TestManifest {
                    kind: "ours".to_string(),
                })
            },
        )
        .err()
        .context("an archive-shipped manifest.json should collide")?;

        assert!(err.to_string().contains("reserved name manifest.json"));
        assert!(!target.exists());
        assert!(fs::read_dir(&staging_root)?.next().is_none());

        fs::remove_dir_all(&root)?;
        Ok(())
    }

    #[test]
    fn entry_size_bytes_sums_a_nested_tree() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let entry = tmp.path().join("entry");
        write_file(&entry.join("manifest.toml"), b"kind = \"x\"\n")?;
        write_file(&entry.join(BLOBS_DIR_NAME).join("a.gguf"), &[0u8; 9000])?;
        write_file(&entry.join(BLOBS_DIR_NAME).join("sub/b.gguf"), &[0u8; 9000])?;

        assert!(
            entry_size_bytes(&entry) >= 18_000,
            "both nested payload files counted"
        );
        Ok(())
    }

    #[test]
    fn entry_size_bytes_is_zero_for_a_missing_path() {
        assert_eq!(
            entry_size_bytes(Path::new("/pipette-does-not-exist-9d3f")),
            0
        );
    }

    #[cfg(unix)]
    #[test]
    fn entry_size_bytes_counts_allocated_blocks_not_length() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let file = tmp.path().join("payload.bin");
        fs::write(&file, [0u8; 5000])?;

        let size = entry_size_bytes(&file);
        assert!(size >= 5000, "allocation is at least the file length");
        assert_eq!(size % 512, 0, "reported in 512-byte blocks, like du");
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn entry_size_bytes_does_not_follow_symlinks() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let outside = tmp.path().join("outside.bin");
        fs::write(&outside, vec![0u8; 2 * 1024 * 1024])?;
        let entry = tmp.path().join("entry");
        fs::create_dir_all(&entry)?;
        std::os::unix::fs::symlink(&outside, entry.join("link.bin"))?;

        assert!(
            entry_size_bytes(&entry) < 1024 * 1024,
            "the link's target is another entry's space, not this one's"
        );
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn entry_size_bytes_does_not_descend_a_symlinked_dir() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let outside = tmp.path().join("outside");
        write_file(&outside.join("big.bin"), &vec![0u8; 2 * 1024 * 1024])?;
        let entry = tmp.path().join("entry");
        fs::create_dir_all(&entry)?;
        std::os::unix::fs::symlink(&outside, entry.join("weights"))?;

        assert!(entry_size_bytes(&entry) < 1024 * 1024);
        Ok(())
    }

    /// A dangling symlink does not fail it.
    #[cfg(unix)]
    #[test]
    fn sync_tree_syncs_a_nested_tree_and_skips_symlinks() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let entry = tmp.path().join("entry");
        write_file(&entry.join(BLOBS_DIR_NAME).join("sub/a.gguf"), &[1u8; 9000])?;
        write_file(&entry.join(MANIFEST_NAME), b"kind = \"x\"\n")?;
        std::os::unix::fs::symlink(tmp.path().join("missing"), entry.join("dangling"))?;

        sync_tree(&entry)
    }

    #[test]
    fn rewrite_manifest_replaces_a_published_record_and_leaves_no_temp_file() -> anyhow::Result<()>
    {
        let tmp = tempfile::tempdir()?;
        let entry = tmp.path().join("entry");
        write_file(&entry.join(MANIFEST_NAME), b"kind = \"old\"\n")?;

        rewrite_manifest(
            &entry,
            MANIFEST_NAME,
            &TestManifest {
                kind: "new".to_string(),
            },
        )?;

        assert_eq!(
            fs::read_to_string(entry.join(MANIFEST_NAME))?,
            "kind = \"new\"\n"
        );
        let names: Vec<String> = fs::read_dir(&entry)?
            .flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(names, vec![MANIFEST_NAME.to_owned()]);
        Ok(())
    }

    #[test]
    fn rewrite_manifest_does_not_recreate_a_removed_entry() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let entry = tmp.path().join("entry");

        let rewritten = rewrite_manifest(
            &entry,
            MANIFEST_NAME,
            &TestManifest {
                kind: "new".to_string(),
            },
        );

        assert!(rewritten.is_err());
        assert!(!entry.exists());
        Ok(())
    }
}
