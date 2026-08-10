use std::{
    fs,
    path::{Path, PathBuf},
};

use anyhow::Context;
use serde::Serialize;
use uuid::Uuid;

/// Payload subdirectory within a directory-artifact entry: files land under
/// `<entry>/blobs/`, isolated from the manifest at the entry root. Shared by
/// the model and runtime stores so a fetcher's placement and a manifest's
/// drift check can't silently disagree on the name.
pub const BLOBS_DIR_NAME: &str = "blobs";

/// TOML record at the entry root for directory artifacts (`models/` and
/// `runtimes/`).
pub const MANIFEST_NAME: &str = "manifest.toml";

/// Scratch directory name under a store root (`models/.staging`,
/// `runtimes/.staging`) used by [`install_dir_computing_manifest`].
pub const STAGING_DIR_NAME: &str = ".staging";

fn child_manifest_path(target_dir: &Path, manifest_name: &str) -> anyhow::Result<PathBuf> {
    validate_manifest_name(manifest_name)?;
    Ok(target_dir.join(manifest_name))
}

/// Bytes a path occupies on disk: recursive, symlinks counted as the link itself
/// and never followed, `st_blocks * 512` on unix so the answer matches `du`.
///
/// Called once per publish to stamp `blobs_bytes` into the manifest, and by
/// [`crate::quota`] only as the fallback for entries predating that field — so a
/// sweep normally totals a store without walking a payload at all.
///
/// Infallible: an unreadable child contributes 0, because a size we cannot read
/// is space we cannot prove we would reclaim, and a permissions blip must not
/// fail a sweep.
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
pub(crate) fn rewrite_manifest<M: Serialize>(
    entry_dir: &Path,
    manifest_name: &str,
    manifest: &M,
) -> anyhow::Result<()> {
    let manifest_path = child_manifest_path(entry_dir, manifest_name)?;
    let temp_path = entry_dir.join(format!("{manifest_name}.{}.tmp", Uuid::new_v4()));
    write_toml_file(&temp_path, manifest)?;
    if let Err(source) = fs::rename(&temp_path, &manifest_path) {
        let _ = fs::remove_file(&temp_path);
        return Err(anyhow::Error::new(source)
            .context(format!("failed to replace {}", manifest_path.display())));
    }
    Ok(())
}

/// Stage a directory, publish it atomically, and write the manifest the
/// `prepare_dir` closure returns.
///
/// The closure *computes* the manifest rather than receiving one, because both
/// stores only know some fields once the payload has landed — the fetched binary
/// paths and archive hash for a runtime, and `blobs_bytes` for either. The engine
/// writes the returned manifest into the staged dir, renames the whole dir into
/// place, and hands the manifest back so the caller needn't re-read it.
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

/// Write `manifest` (as TOML) into the staged dir at `manifest_name`, refusing to
/// clobber a same-named entry the prepared bundle already produced. The manifest
/// is written last (after prepare), so an archive that shipped its own
/// `manifest.toml` would otherwise silently overwrite our record — guard it.
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

/// Pretty-print TOML to `path`, creating parent directories if needed.
fn write_toml_file<T: Serialize>(path: &Path, value: &T) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    let rendered = toml::to_string_pretty(value)
        .with_context(|| format!("failed to serialize {}", path.display()))?;
    fs::write(path, rendered).with_context(|| format!("failed to write {}", path.display()))
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
    let staged = create_staged_dir(staging_root, staging_name)?;
    let prepared = match prepare(&staged) {
        Ok(prepared) => prepared,
        Err(err) => return Err(preserve_with_cleanup(err, cleanup_path(&staged), &staged)),
    };
    let replace_result = replace_staged_path(&staged, target);
    if let Err(err) = replace_result {
        let err = preserve_with_cleanup(err, cleanup_path(target), target);
        return Err(preserve_with_cleanup(err, cleanup_path(&staged), &staged));
    }
    Ok(prepared)
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

fn create_staged_dir(staging_root: &Path, staging_name: &str) -> anyhow::Result<PathBuf> {
    fs::create_dir_all(staging_root)
        .with_context(|| format!("failed to create {}", staging_root.display()))?;
    let staged = staged_path(staging_root, staging_name);
    if staged.exists() {
        cleanup_path(&staged)?;
    }
    fs::create_dir_all(&staged)
        .with_context(|| format!("failed to create {}", staged.display()))?;
    Ok(staged)
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

/// Characters of the entry name kept in a staging directory's name.
///
/// The staged directory is scaffolding: identity lives in the final path and
/// the manifest, and this name exists only so an orphan left by a crashed
/// install is recognizable. It is deliberately short because on Windows it sits
/// on the critical path for `MAX_PATH` — see [`staged_path`].
const STAGING_NAME_PREFIX_LEN: usize = 16;

/// Hex chars of uniqueness in a staging directory's name. Distinguishes
/// concurrent installs of the same entry; 32 bits is ample for that, where a
/// hyphenated UUID is 36 characters of mostly separator.
const STAGING_UNIQUE_HEX: usize = 8;

/// Transient path to install into before the atomic move to `target`.
///
/// Kept short on purpose. Windows caps a path at 260 characters unless both the
/// OS and the binary opt into long paths, and a staged venv is the longest path
/// this store ever creates: entry name, plus the staging suffix, plus
/// `blobs/venv/Lib/site-packages/<dist>.dist-info/<file>`. With a full entry
/// name and a hyphenated UUID that reached 261 characters for an ordinary
/// runtime install under a user's home directory, and `uv` failed mid-install
/// with a bare "system cannot find the path specified".
///
/// Truncating here rather than shortening the entry name keeps the *final*
/// layout self-describing — that is what the name is for — while taking the
/// transient path well clear of the limit.
fn staged_path(staging_root: &Path, staging_name: &str) -> PathBuf {
    let safe_name = if staging_name.is_empty() {
        "asset".to_string()
    } else {
        staging_name.replace(['/', '\\'], "_")
    };
    // Char-wise, not byte-wise: an entry name is normalized ASCII today, but
    // slicing a multi-byte boundary would panic rather than truncate.
    let head: String = safe_name.chars().take(STAGING_NAME_PREFIX_LEN).collect();
    let unique = Uuid::new_v4().simple().to_string();
    let unique = &unique[..STAGING_UNIQUE_HEX];
    staging_root.join(format!("{head}.staged-{unique}"))
}

fn move_into_place(source: &Path, target: &Path) -> anyhow::Result<()> {
    fs::rename(source, target)
        .or_else(|_| copy_tree(source, target))
        .with_context(|| {
            format!(
                "failed to stage {} into {}",
                source.display(),
                target.display()
            )
        })
}

fn copy_tree(source: &Path, target: &Path) -> anyhow::Result<()> {
    fs::create_dir_all(target).with_context(|| format!("failed to create {}", target.display()))?;
    for entry in
        fs::read_dir(source).with_context(|| format!("failed to read {}", source.display()))?
    {
        let entry = entry?;
        let src_path = entry.path();
        let dst_path = target.join(entry.file_name());
        if src_path.is_dir() {
            copy_tree(&src_path, &dst_path)?;
        } else {
            fs::copy(&src_path, &dst_path).with_context(|| {
                format!(
                    "failed to copy {} to {}",
                    src_path.display(),
                    dst_path.display()
                )
            })?;
        }
    }
    fs::remove_dir_all(source).with_context(|| format!("failed to remove {}", source.display()))?;
    Ok(())
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

    /// The regression this guards: a full entry name plus a hyphenated UUID
    /// pushed a staged venv path to 261 characters on Windows, two over the
    /// 260 limit, and `uv` failed mid-install with "system cannot find the
    /// path specified". The staged name is scaffolding, so it is bounded;
    /// the final entry name is not, and stays self-describing.
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

        // The deepest path a venv install writes has to clear the limit with
        // room for a realistic work-dir.
        let deepest = "/blobs/venv/Lib/site-packages/             openvino_tokenizers-2026.2.1.0.dist-info/RECORD";
        let base = "C:\\Users\\somebody\\benchmarks\\.pipette\\runtimes\\.staging";
        assert!(
            base.len() + 1 + name.len() + deepest.len() < 260,
            "staged venv path would exceed MAX_PATH"
        );
    }

    /// Two installs of the same entry must not collide in staging.
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
        assert!(staged_path(root, "")
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or_default()
            .starts_with("asset.staged-"));
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

        assert!(err
            .to_string()
            .contains("manifest name must not contain path separators"));
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

        // A prepared bundle that itself contains `manifest.json` must not clobber
        // our record — the engine bails and publishes nothing.
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
}
