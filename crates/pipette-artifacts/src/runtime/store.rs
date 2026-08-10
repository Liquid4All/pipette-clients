//! [`RuntimeArtifactStore`] — resolve a declared runtime to its installed
//! on-disk form, the runtime-side counterpart to [`crate::model`].
//!
//! The store owns *where* a runtime's binaries live (the workspace `runtimes/`
//! tree); the install callback owns *how* they arrive.
//! [`RuntimeArtifactStore::find`] is the read path (is it installed?);
//! [`RuntimeArtifactStore::ensure`] is find-or-fetch. Staging and atomic publish
//! are delegated to the shared [`crate::entry`] engine (the same one the model
//! store uses), so a partial or failed fetch never leaves a half-written entry
//! and a re-run resolves the already-installed runtime.

use std::fs;
use std::path::{Path, PathBuf};

use time::OffsetDateTime;

use pipette_plan_types::Runtime;

use super::key::{RuntimeStorageKey, RuntimeStorageKeyError};
use super::manifest::{RuntimeManifest, RuntimeManifestError};
use super::stored::to_stored;
use crate::entry::{
    entry_size_bytes, install_dir_computing_manifest, BLOBS_DIR_NAME, MANIFEST_NAME,
    STAGING_DIR_NAME,
};
use crate::quota::QuotaError;

/// A runtime-store failure.
#[derive(Debug, thiserror::Error)]
pub enum RuntimeStoreError {
    /// The runtime has no storage key (on-device or OS-bundled — not installed).
    #[error(transparent)]
    NotStorable(#[from] RuntimeStorageKeyError),
    /// A stored manifest is unusable — bad version or placement drift.
    #[error("installed runtime is corrupt: {0}")]
    Corrupt(String),
    /// The runtime cannot fit under the configured storage quota.
    #[error(transparent)]
    Quota(#[from] QuotaError),
    /// The fetcher failed to install the runtime's binaries (or the publish
    /// failed). The shared staging engine has already cleaned up.
    #[error("installing runtime `{runtime}` failed")]
    Fetch {
        runtime: String,
        #[source]
        source: anyhow::Error,
    },
    /// A filesystem operation on the manifest failed; `doing` names it.
    #[error("{doing}")]
    Io {
        doing: String,
        #[source]
        source: std::io::Error,
    },
    /// A manifest on disk didn't parse as TOML, or a typed field failed
    /// deserialization (e.g. non–RFC 3339 `fetched_at`) — distinct from a
    /// parseable-but-invalid manifest, which is `Corrupt`.
    #[error("parsing the runtime manifest at {path} failed")]
    Parse {
        path: PathBuf,
        #[source]
        source: toml::de::Error,
    },
}

// The manifest layer's failures fold into the store's own variants.
impl From<RuntimeManifestError> for RuntimeStoreError {
    fn from(err: RuntimeManifestError) -> Self {
        match err {
            RuntimeManifestError::NotStorable(e) => Self::NotStorable(e),
            RuntimeManifestError::Local(e) => Self::Corrupt(e.to_string()),
            RuntimeManifestError::Corrupt(reason) => Self::Corrupt(reason),
        }
    }
}

/// Filesystem runtime store rooted at a workspace `runtimes/` directory.
///
/// Assumes a single writer: [`Self::ensure`] stages an install, then swaps it
/// into place (replacing any stale copy first), so a reader never sees a
/// half-written entry. A crash mid-install can orphan a `.staging/<name>` dir;
/// those are safe to delete and are the operator's to garbage-collect.
///
/// The store owns *where* payloads land (`blobs/`); the `install` callback owns
/// *how*. Tools are resolved later via [`RuntimeManifest::resolve_tool`].
pub struct RuntimeArtifactStore {
    runtimes_dir: PathBuf,
}

impl RuntimeArtifactStore {
    pub fn new(runtimes_dir: PathBuf) -> Self {
        Self { runtimes_dir }
    }

    /// Workspace `runtimes/` root this store was opened on.
    pub fn runtimes_dir(&self) -> &Path {
        &self.runtimes_dir
    }

    /// Absolute entry dir for `manifest` under this store.
    pub fn entry_dir_for(
        &self,
        manifest: &RuntimeManifest,
    ) -> Result<PathBuf, RuntimeManifestError> {
        manifest.entry_dir(self.runtimes_dir())
    }

    /// Absolute install root for `manifest` under this store (`stored` form).
    pub fn install_dir_for(
        &self,
        manifest: &RuntimeManifest,
    ) -> Result<PathBuf, RuntimeManifestError> {
        manifest.install_dir(self.runtimes_dir())
    }

    /// The on-disk entry directory for a stored runtime: `runtimes_dir/<key>`.
    fn entry_dir(&self, key: &RuntimeStorageKey) -> PathBuf {
        self.runtimes_dir.join(key.relative_dir())
    }

    /// The stored manifest for `declared`, or `None` if it isn't installed. Pure
    /// read — no fetch, no mutation. Unparseable bytes or typed-field failures
    /// (including a non–RFC 3339 `fetched_at`) → `Err(Parse)`;
    /// parseable-but-invalid (bad version, placement drift) → `Err(Corrupt)`.
    ///
    /// Only `manifest.toml` is read — neither the `blobs/` payload nor, for a
    /// docker runtime, the daemon image is checked, so an out-of-band deletion
    /// isn't caught here (nor by [`Self::ensure`]). A docker entry records only
    /// that a pull once succeeded; a filesystem kind surfaces a missing payload
    /// when its tool is resolved at launch.
    pub fn find(&self, declared: &Runtime) -> Result<Option<RuntimeManifest>, RuntimeStoreError> {
        let entry = self.entry_dir(&RuntimeStorageKey::of(declared)?);
        if !entry.join(MANIFEST_NAME).exists() {
            return Ok(None);
        }
        Ok(Some(read_manifest(&entry)?))
    }

    /// Stamp `last_used_at = now` on a resolved entry — the eviction order.
    /// Best-effort: a read-only store must still resolve runtimes, so a failed
    /// write is logged and the in-memory value advances regardless.
    fn touch(&self, manifest: &mut RuntimeManifest) {
        manifest.last_used_at = OffsetDateTime::now_utc();
        let Ok(entry) = manifest.entry_dir(self.runtimes_dir()) else {
            return;
        };
        if let Err(err) = crate::entry::rewrite_manifest(&entry, MANIFEST_NAME, manifest) {
            log::debug!(
                "could not record last_used_at for {}: {err:#}",
                entry.display()
            );
        }
    }

    /// The manifest for `declared`, installing + recording it first if it isn't
    /// already stored. A re-run resolves the existing install without re-fetching.
    ///
    /// `install` writes into the store-owned `blobs/` dir and must fail if
    /// required tools are missing (same role as the model store's `fetch` callback).
    pub fn ensure<F>(
        &self,
        declared: &Runtime,
        mut install: F,
    ) -> Result<RuntimeManifest, RuntimeStoreError>
    where
        F: FnMut(&Runtime, &Path) -> anyhow::Result<()>,
    {
        if let Some(mut manifest) = self.find(declared)? {
            self.touch(&mut manifest);
            return Ok(manifest);
        }
        let key = RuntimeStorageKey::of(declared)?;
        let entry = self.entry_dir(&key);
        // Deliberately taken before the install, matching the model store: the
        // record dates when the pull started, not when it finished.
        let fetched_at = OffsetDateTime::now_utc();
        // The shared engine stages into `.staging/<key>.staged-<uuid>`, runs the
        // install, writes `manifest.toml` at the entry root beside the `blobs/`
        // payload, renames the whole dir into place — cleaning up on any failure.
        // The store creates and owns `blobs/`; the callback only sees that payload
        // dir. Required tools are checked by the installer; lookup at use time
        // walks install_dir. The manifest is built inside the closure (install
        // must run first), then handed back.
        let manifest = install_dir_computing_manifest(
            &self.runtimes_dir.join(STAGING_DIR_NAME),
            &entry,
            key.as_str(),
            MANIFEST_NAME,
            |staged| {
                let blobs = staged.join(BLOBS_DIR_NAME);
                fs::create_dir_all(&blobs)?;
                install(declared, &blobs)?;
                // Measured after the install, which is the only moment the size
                // is knowable for a venv nobody declared a size for.
                Ok(RuntimeManifest::new(
                    declared.clone(),
                    fetched_at,
                    entry_size_bytes(&blobs),
                )?)
            },
        )
        .map_err(|source| RuntimeStoreError::Fetch {
            runtime: declared.to_string(),
            source,
        })?;
        Ok(manifest)
    }

    /// Every recorded runtime, read from each `runtimes/<key>/manifest.toml`.
    /// Dirs without a manifest (partial installs, `.staging`, foreign entries)
    /// are skipped; sorted by the runtime's CLI ref for stable output. Includes
    /// docker runtimes — their entry is manifest-only. A malformed manifest or an
    /// unreadable dir entry surfaces as an error rather than being dropped.
    pub fn list(&self) -> Result<Vec<RuntimeManifest>, RuntimeStoreError> {
        if !self.runtimes_dir.exists() {
            return Ok(Vec::new());
        }
        let read = fs::read_dir(&self.runtimes_dir).map_err(|source| RuntimeStoreError::Io {
            doing: format!("reading {}", self.runtimes_dir.display()),
            source,
        })?;
        let mut manifests = read
            .map(|entry| {
                entry.map_err(|source| RuntimeStoreError::Io {
                    doing: format!("reading a {} entry", self.runtimes_dir.display()),
                    source,
                })
            })
            .filter_map(|entry| match entry {
                Err(e) => Some(Err(e)),
                Ok(entry) => {
                    let dir = entry.path();
                    // Skip non-dirs, reserved/hidden dirs (`.staging`), and dirs
                    // without a manifest.
                    let recorded = dir.is_dir()
                        && dir
                            .file_name()
                            .and_then(|n| n.to_str())
                            .is_some_and(|n| !n.starts_with('.'))
                        && dir.join(MANIFEST_NAME).exists();
                    if !recorded {
                        return None;
                    }
                    match classify_manifest_toml(&dir.join(MANIFEST_NAME)) {
                        Ok(ManifestKind::Engine) => None,
                        Ok(ManifestKind::Store) => Some(read_manifest(&dir)),
                        Ok(ManifestKind::Unknown) => Some(Err(RuntimeStoreError::Corrupt(
                            format!(
                                "runtime entry {} has a manifest.toml that is neither the store \
                                 schema (manifest_version) nor a known engine type; repair or remove it",
                                dir.display()
                            ),
                        ))),
                        Err(e) => Some(Err(e)),
                    }
                }
            })
            .collect::<Result<Vec<RuntimeManifest>, _>>()?;
        manifests.sort_by_key(|m| m.declared.cli_ref());
        Ok(manifests)
    }

    /// Remove a runtime's `runtimes/<key>/` record (and any `blobs/` payload).
    /// Returns whether an entry was present. Record-only: a docker runtime's
    /// daemon image is left in place — removing it is the docker installer's
    /// concern, not the filesystem store's.
    pub fn remove(&self, declared: &Runtime) -> Result<bool, RuntimeStoreError> {
        let entry = self.entry_dir(&RuntimeStorageKey::of(declared)?);
        if !entry.exists() {
            return Ok(false);
        }
        fs::remove_dir_all(&entry).map_err(|source| RuntimeStoreError::Io {
            doing: format!("removing {}", entry.display()),
            source,
        })?;
        Ok(true)
    }
}

/// Read + validate the `manifest.toml` at an entry dir: raw read → TOML /
/// typed deserialize (`Parse`) → validation (`Corrupt`). Typed, so callers keep
/// the unparseable-vs-invalid distinction. Free of the store because the entry
/// dir is the whole input — the quota survey reads entries the store never
/// resolved.
pub(crate) fn read_manifest(entry: &Path) -> Result<RuntimeManifest, RuntimeStoreError> {
    let manifest_path = entry.join(MANIFEST_NAME);
    let raw = fs::read_to_string(&manifest_path).map_err(|source| RuntimeStoreError::Io {
        doing: format!("reading {}", manifest_path.display()),
        source,
    })?;
    let m: RuntimeManifest = toml::from_str(&raw).map_err(|source| RuntimeStoreError::Parse {
        path: manifest_path.clone(),
        source,
    })?;
    if m.manifest_version != crate::runtime::manifest::MANIFEST_VERSION {
        return Err(RuntimeStoreError::Corrupt(format!(
            "unsupported manifest_version {} (expected {})",
            m.manifest_version,
            crate::runtime::manifest::MANIFEST_VERSION
        )));
    }
    RuntimeStorageKey::of(&m.declared)?;
    let expected = to_stored(&m.declared).map_err(|e| RuntimeStoreError::Corrupt(e.to_string()))?;
    if m.stored != expected {
        return Err(RuntimeStoreError::Corrupt(
            "stored form is not the declared runtime's local form".to_owned(),
        ));
    }
    Ok(m)
}

/// Peek-only owner of a `runtimes/*/manifest.toml` (full parse is per-owner).
///
/// `Engine` is the legacy torch-oai format — a serde-tagged `RuntimeEngine`,
/// written until #872 removed that writer. Its dir names, e.g.
/// `vllm@0.22.0+cu129.py3.12`, can't collide with a [`RuntimeStorageKey`], so
/// [`RuntimeArtifactStore::find`] never lands on one; only
/// [`RuntimeArtifactStore::list`] walks the dir blind and meets them, and
/// skipping beats failing a workspace that predates the unification.
pub(crate) enum ManifestKind {
    Store,
    Engine,
    Unknown,
}

pub(crate) fn classify_manifest_toml(path: &Path) -> Result<ManifestKind, RuntimeStoreError> {
    let raw = fs::read_to_string(path).map_err(|source| RuntimeStoreError::Io {
        doing: format!("reading {}", path.display()),
        source,
    })?;
    let value: toml::Value = toml::from_str(&raw).map_err(|source| RuntimeStoreError::Parse {
        path: path.to_path_buf(),
        source,
    })?;
    if value.get("manifest_version").is_some() {
        return Ok(ManifestKind::Store);
    }
    if value
        .get("type")
        .and_then(|t| t.as_str())
        .is_some_and(|ty| {
            matches!(
                ty,
                "docker_vllm" | "docker_sglang" | "uv_vllm" | "uv_sglang"
            )
        })
    {
        return Ok(ManifestKind::Engine);
    }
    Ok(ManifestKind::Unknown)
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::fs;

    use pipette_plan_types::{
        LlamaCppFlavor, LlamacppCliStockTools, LlamacppCliStockToolsSource, NonEmptyString,
        RepositoryUrl, SourceRepository,
    };

    use super::*;

    fn declared() -> anyhow::Result<Runtime> {
        Ok(Runtime::LlamacppCliStockTools(LlamacppCliStockTools {
            source: LlamacppCliStockToolsSource::GithubRelease(SourceRepository {
                repository_url: RepositoryUrl::new("github.com/ggml-org/llama.cpp"),
                repository_version: NonEmptyString::try_new("b9305".to_owned())?,
            }),
            flavor: LlamaCppFlavor::MacosArm64,
        }))
    }

    /// A runtime's entry dir under `root`, without needing an installed manifest.
    fn entry_dir(root: &std::path::Path, runtime: &Runtime) -> anyhow::Result<std::path::PathBuf> {
        Ok(root.join(RuntimeStorageKey::of(runtime)?.relative_dir()))
    }

    /// Writes a fake `llama-server` into the store-owned `blobs/` dir.
    fn install_stub(_declared: &Runtime, blobs_dir: &std::path::Path) -> anyhow::Result<()> {
        fs::write(blobs_dir.join("llama-server"), b"")?;
        Ok(())
    }

    #[test]
    fn find_is_none_until_ensured_then_fetches_once() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let store = RuntimeArtifactStore::new(tmp.path().to_path_buf());
        let calls = Cell::new(0usize);
        let runtime = declared()?;

        assert!(store.find(&runtime)?.is_none(), "not installed yet");

        let mut install = |d: &Runtime, blobs: &std::path::Path| {
            calls.set(calls.get() + 1);
            install_stub(d, blobs)
        };
        let manifest = store.ensure(&runtime, &mut install)?;
        // The manifest resolves its own on-disk entry dir.
        let entry = manifest.entry_dir(tmp.path())?;
        assert_eq!(entry, entry_dir(tmp.path(), &runtime)?);
        assert!(
            entry.join("blobs/llama-server").exists(),
            "binary landed under blobs/"
        );
        assert!(entry.join(MANIFEST_NAME).exists(), "manifest at entry root");
        assert_eq!(manifest.declared, runtime);
        assert_eq!(store.runtimes_dir(), tmp.path());
        assert_eq!(store.entry_dir_for(&manifest)?, entry);
        assert_eq!(
            manifest.resolve_tool(store.runtimes_dir(), "llama-server")?,
            entry.join("blobs/llama-server")
        );

        // `find` now resolves it, and a second `ensure` doesn't re-install.
        assert!(store.find(&runtime)?.is_some());
        store.ensure(&runtime, &mut install)?;
        assert_eq!(calls.get(), 1);
        Ok(())
    }

    #[test]
    fn a_failed_fetch_leaves_no_entry() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let store = RuntimeArtifactStore::new(tmp.path().to_path_buf());
        let runtime = declared()?;

        assert!(matches!(
            store.ensure(&runtime, |_, _| anyhow::bail!("network down")),
            Err(RuntimeStoreError::Fetch { .. })
        ));
        assert!(
            !entry_dir(tmp.path(), &runtime)?.exists(),
            "no entry published on failure"
        );
        let staging = tmp.path().join(STAGING_DIR_NAME);
        assert!(
            !staging.exists() || fs::read_dir(&staging)?.next().is_none(),
            "no staged remnant"
        );
        Ok(())
    }

    #[test]
    fn unparseable_manifest_is_a_parse_error() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let store = RuntimeArtifactStore::new(tmp.path().to_path_buf());
        let runtime = declared()?;
        let manifest = store.ensure(&runtime, install_stub)?;

        let manifest_path = manifest.entry_dir(tmp.path())?.join(MANIFEST_NAME);
        fs::write(&manifest_path, "not = valid = toml")?;
        assert!(matches!(
            store.find(&runtime),
            Err(RuntimeStoreError::Parse { .. })
        ));
        Ok(())
    }

    #[test]
    fn an_uninstallable_runtime_has_no_entry() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let store = RuntimeArtifactStore::new(tmp.path().to_path_buf());
        assert!(matches!(
            store.find(&Runtime::AppleFoundation(Default::default())),
            Err(RuntimeStoreError::NotStorable(_))
        ));
        Ok(())
    }

    #[test]
    fn a_payload_named_like_the_manifest_cannot_clobber_the_record() -> anyhow::Result<()> {
        // The payload lives under `blobs/`, isolated from the entry-root
        // `manifest.toml`. So even an archive that ships its own `manifest.toml`
        // lands at `blobs/manifest.toml` and can't overwrite our record.
        let tmp = tempfile::tempdir()?;
        let store = RuntimeArtifactStore::new(tmp.path().to_path_buf());
        let manifest = store.ensure(&declared()?, |_, blobs_dir| {
            fs::write(blobs_dir.join(MANIFEST_NAME), b"theirs = true")?;
            fs::write(blobs_dir.join("llama-server"), b"")?;
            Ok(())
        })?;
        assert_eq!(manifest.declared, declared()?);
        // Our record is intact and re-readable — not the payload's file.
        assert!(store.find(&declared()?)?.is_some());
        Ok(())
    }

    #[test]
    fn list_returns_recorded_runtimes() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let store = RuntimeArtifactStore::new(tmp.path().to_path_buf());
        assert!(store.list()?.is_empty(), "empty before any install");

        store.ensure(&declared()?, install_stub)?;
        let listed = store.list()?;
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].declared, declared()?);
        Ok(())
    }

    #[test]
    fn remove_deletes_the_entry_and_reports_presence() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let store = RuntimeArtifactStore::new(tmp.path().to_path_buf());
        let runtime = declared()?;

        assert!(!store.remove(&runtime)?, "nothing to remove yet");
        store.ensure(&runtime, install_stub)?;
        assert!(store.find(&runtime)?.is_some());

        assert!(store.remove(&runtime)?, "removed the installed entry");
        assert!(!entry_dir(tmp.path(), &runtime)?.exists());
        assert!(store.find(&runtime)?.is_none());
        Ok(())
    }

    /// `last_used_at` as it stands on disk, not as the store last returned it.
    fn stored_last_used_at(root: &Path, runtime: &Runtime) -> anyhow::Result<OffsetDateTime> {
        let raw = fs::read_to_string(entry_dir(root, runtime)?.join(MANIFEST_NAME))?;
        Ok(toml::from_str::<RuntimeManifest>(&raw)?.last_used_at)
    }

    #[test]
    fn ensure_seeds_last_used_at_on_publish() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let store = RuntimeArtifactStore::new(tmp.path().to_path_buf());

        let manifest = store.ensure(&declared()?, install_stub)?;

        assert_eq!(manifest.last_used_at, manifest.fetched_at);
        Ok(())
    }

    #[test]
    fn ensure_touches_last_used_at_on_a_cache_hit() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let store = RuntimeArtifactStore::new(tmp.path().to_path_buf());
        let runtime = declared()?;
        let calls = Cell::new(0usize);
        let mut install = |d: &Runtime, blobs: &std::path::Path| {
            calls.set(calls.get() + 1);
            install_stub(d, blobs)
        };
        let published = store.ensure(&runtime, &mut install)?;

        let resolved = store.ensure(&runtime, &mut install)?;

        assert_eq!(calls.get(), 1, "a hit does not re-install");
        assert_eq!(resolved.fetched_at, published.fetched_at);
        assert!(resolved.last_used_at > published.last_used_at);
        assert_eq!(
            stored_last_used_at(tmp.path(), &runtime)?,
            resolved.last_used_at
        );
        Ok(())
    }

    #[test]
    fn find_does_not_touch_last_used_at() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let store = RuntimeArtifactStore::new(tmp.path().to_path_buf());
        let runtime = declared()?;
        store.ensure(&runtime, install_stub)?;
        let before = stored_last_used_at(tmp.path(), &runtime)?;

        store.find(&runtime)?;

        assert_eq!(stored_last_used_at(tmp.path(), &runtime)?, before);
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn a_failed_touch_still_resolves() -> anyhow::Result<()> {
        use std::os::unix::fs::PermissionsExt;

        let tmp = tempfile::tempdir()?;
        let store = RuntimeArtifactStore::new(tmp.path().to_path_buf());
        let runtime = declared()?;
        let published = store.ensure(&runtime, install_stub)?;

        let entry = entry_dir(tmp.path(), &runtime)?;
        fs::set_permissions(&entry, fs::Permissions::from_mode(0o500))?;
        let resolved = store.ensure(&runtime, install_stub);
        fs::set_permissions(&entry, fs::Permissions::from_mode(0o700))?;

        let resolved = resolved?;
        assert_eq!(resolved.declared, published.declared);
        assert!(
            resolved.last_used_at > published.last_used_at,
            "the in-memory stamp advances even when the write cannot land"
        );
        Ok(())
    }

    /// No migration: the version bump strands v1 stores on purpose, and the
    /// missing field means they fail at deserialization rather than the version
    /// check. `pipette storage gc` is the recovery, not a compat shim.
    #[test]
    fn a_v1_manifest_is_unreadable() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let store = RuntimeArtifactStore::new(tmp.path().to_path_buf());
        let runtime = declared()?;
        store.ensure(&runtime, install_stub)?;

        let manifest_path = entry_dir(tmp.path(), &runtime)?.join(MANIFEST_NAME);
        let mut table: toml::Table = toml::from_str(&fs::read_to_string(&manifest_path)?)?;
        table.insert("manifest_version".into(), toml::Value::Integer(1));
        table.remove("last_used_at");
        fs::write(&manifest_path, toml::to_string(&table)?)?;

        assert!(matches!(
            store.find(&runtime),
            Err(RuntimeStoreError::Parse { .. })
        ));
        Ok(())
    }

    #[test]
    fn list_skips_engine_schema_but_errors_on_unknown_manifest() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let store = RuntimeArtifactStore::new(tmp.path().to_path_buf());
        store.ensure(&declared()?, install_stub)?;

        let engine_dir = tmp.path().join("vllm@0.22.0+cu129.py3.12");
        fs::create_dir_all(&engine_dir)?;
        fs::write(
            engine_dir.join(MANIFEST_NAME),
            r#"
            type = "uv_vllm"
            slug = "vllm@0.22.0+cu129.py3.12"
            server_version = "0.22.0"
            build = "cu129"
            python_version = "3.12"
            [source]
            type = "pip_requirements_text"
            requirements_text = "vllm==0.22.0"
            "#,
        )?;
        let listed = store.list()?;
        assert_eq!(listed.len(), 1, "store list keeps only store entries");
        assert_eq!(listed[0].declared, declared()?);

        let broken = tmp.path().join("broken-entry");
        fs::create_dir_all(&broken)?;
        fs::write(broken.join(MANIFEST_NAME), "foo = 1\n")?;
        let Err(err) = store.list() else {
            anyhow::bail!("unknown manifest must fail list");
        };
        assert!(matches!(err, RuntimeStoreError::Corrupt(_)), "got {err:?}");
        assert!(
            err.to_string().contains("neither the store schema"),
            "got {err}"
        );
        Ok(())
    }
}
