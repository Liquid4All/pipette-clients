//! [`RuntimeArtifactStore`]: the workspace `runtimes/` tree, the runtime-side
//! counterpart to [`crate::model`]'s store. Staging and atomic publish are the
//! shared [`crate::entry`] engine's.

use std::fs::{self, File};
use std::path::{Path, PathBuf};

use time::OffsetDateTime;

use pipette_plan_types::Runtime;

use super::key::{RuntimeStorageKey, RuntimeStorageKeyError};
use super::manifest::{RuntimeManifest, RuntimeManifestError};
use super::stored::to_stored;
use crate::entry::{
    BLOBS_DIR_NAME, MANIFEST_NAME, STAGING_DIR_NAME, entry_size_bytes, hold_entry,
    install_dir_computing_manifest, lock_entry,
};
use crate::quota::QuotaError;

#[derive(Debug, thiserror::Error)]
pub enum RuntimeStoreError {
    #[error(transparent)]
    NotStorable(#[from] RuntimeStorageKeyError),
    /// The manifest parses but is invalid: bad version or placement drift.
    #[error("installed runtime is corrupt: {0}")]
    Corrupt(String),
    /// A bundled runtime from an older client has different embedded sources or pins.
    #[error("runtime entry at {0} belongs to a different bundled build")]
    Obsolete(PathBuf),
    #[error(transparent)]
    Quota(#[from] QuotaError),
    /// The install or the publish failed; staging is already cleaned up.
    #[error("installing runtime `{runtime}` failed")]
    Fetch {
        runtime: String,
        #[source]
        source: anyhow::Error,
    },
    #[error("{doing}")]
    Io {
        doing: String,
        #[source]
        source: std::io::Error,
    },
    /// Not TOML, or a typed field failed to deserialize. `ensure` treats this
    /// as a torn write and reinstalls; `Corrupt` it does not.
    #[error("parsing the runtime manifest at {path} failed")]
    Parse {
        path: PathBuf,
        #[source]
        source: toml::de::Error,
    },
}

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
/// [`Self::ensure`] stages an install and renames it into place, so a reader
/// never sees a half-written entry. A crash mid-install leaves a staged dir
/// under `.staging`, which [`crate::quota`] reclaims as garbage.
pub struct RuntimeArtifactStore {
    runtimes_dir: PathBuf,
}

impl RuntimeArtifactStore {
    pub fn new(runtimes_dir: PathBuf) -> Self {
        Self { runtimes_dir }
    }

    pub fn runtimes_dir(&self) -> &Path {
        &self.runtimes_dir
    }

    pub fn entry_dir_for(
        &self,
        manifest: &RuntimeManifest,
    ) -> Result<PathBuf, RuntimeManifestError> {
        manifest.entry_dir(self.runtimes_dir())
    }

    pub fn install_dir_for(
        &self,
        manifest: &RuntimeManifest,
    ) -> Result<PathBuf, RuntimeManifestError> {
        manifest.install_dir(self.runtimes_dir())
    }

    fn entry_dir(&self, key: &RuntimeStorageKey) -> PathBuf {
        self.runtimes_dir.join(key.relative_dir())
    }

    /// The stored manifest for `declared`, or `None` if it isn't installed.
    /// Read-only.
    ///
    /// Only `manifest.toml` is read: an out-of-band deletion of the `blobs/`
    /// payload or the docker image is not caught here (nor by [`Self::ensure`]).
    /// A filesystem kind surfaces a missing payload when its tool is resolved.
    pub fn find(&self, declared: &Runtime) -> Result<Option<RuntimeManifest>, RuntimeStoreError> {
        let entry = self.entry_dir(&RuntimeStorageKey::of(declared)?);
        if !entry.join(MANIFEST_NAME).exists() {
            return Ok(None);
        }
        Ok(Some(read_manifest(&entry)?))
    }

    /// Like [`Self::find`], but `None` when the manifest does not parse
    /// ([`Self::ensure`] repairs that entry under its lock). `Corrupt` and
    /// other errors pass through.
    pub(crate) fn find_intact(
        &self,
        declared: &Runtime,
    ) -> Result<Option<RuntimeManifest>, RuntimeStoreError> {
        match self.find(declared) {
            Err(RuntimeStoreError::Parse { .. }) => Ok(None),
            found => found,
        }
    }

    /// The in-use lock on `declared`'s entry (see [`hold_entry`]), or `None` when
    /// a sweep removed the entry before the lock was taken. The inner `None` is a
    /// store that cannot be locked.
    pub(crate) fn hold(
        &self,
        declared: &Runtime,
    ) -> Result<Option<Option<File>>, RuntimeStoreError> {
        let key = RuntimeStorageKey::of(declared)?;
        let lock = hold_entry(&self.runtimes_dir, key.as_str());
        Ok(self.find_intact(declared)?.map(|_| lock))
    }

    /// Stamp `last_used_at`, the sweep's eviction order.
    /// Best-effort: a read-only store must still resolve runtimes, so a failed
    /// write is logged and the in-memory value advances regardless.
    fn touch(&self, manifest: &mut RuntimeManifest) {
        manifest.last_used_at = OffsetDateTime::now_utc();
        let Ok(entry) = manifest.entry_dir(self.runtimes_dir()) else {
            return;
        };
        if let Err(err) = crate::entry::rewrite_manifest(&entry, MANIFEST_NAME, manifest) {
            tracing::debug!(
                path = %entry.display(),
                error = %format_args!("{err:#}"),
                "could not record last_used_at"
            );
        }
    }

    fn touched(&self, found: Option<RuntimeManifest>) -> Option<RuntimeManifest> {
        found.map(|mut manifest| {
            self.touch(&mut manifest);
            manifest
        })
    }

    /// The manifest for `declared`, installing it first on a miss.
    ///
    /// An entry whose manifest does not parse (a crash mid-write, or an older
    /// version such as v1) is removed and reinstalled: the runtime spec lived
    /// in that manifest, so nothing else can repair it.
    ///
    /// `install` writes into the store-owned `blobs/` dir and must fail if
    /// required tools are missing.
    pub fn ensure<F>(
        &self,
        declared: &Runtime,
        mut install: F,
    ) -> Result<RuntimeManifest, RuntimeStoreError>
    where
        F: FnMut(&Runtime, &Path) -> anyhow::Result<()>,
    {
        if let Some(manifest) = self.touched(self.find_intact(declared)?) {
            return Ok(manifest);
        }
        let key = RuntimeStorageKey::of(declared)?;
        // Another process sharing the store may be installing it: wait for it,
        // then look again.
        let _installing = lock_entry(&self.runtimes_dir, key.as_str()).map_err(|source| {
            RuntimeStoreError::Fetch {
                runtime: declared.to_string(),
                source,
            }
        })?;
        let found = match self.find(declared) {
            Err(RuntimeStoreError::Parse { path, source }) => {
                tracing::warn!(
                    path = %path.display(),
                    error = %source,
                    "runtime manifest does not parse; removing the entry to reinstall it"
                );
                // Before the install, so that the disk does not hold two payloads.
                self.remove(declared)?;
                None
            }
            found => found?,
        };
        if let Some(manifest) = self.touched(found) {
            return Ok(manifest);
        }
        let entry = self.entry_dir(&key);
        // Before the install, matching the model store: the record dates when
        // the pull started.
        let fetched_at = OffsetDateTime::now_utc();
        let manifest = install_dir_computing_manifest(
            &self.runtimes_dir.join(STAGING_DIR_NAME),
            &entry,
            key.as_str(),
            MANIFEST_NAME,
            |staged| {
                let blobs = staged.join(BLOBS_DIR_NAME);
                fs::create_dir_all(&blobs)?;
                install(declared, &blobs)?;
                // A venv has no declared size; only the install reveals it.
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

    /// Every recorded runtime, sorted by CLI ref. Dirs without a manifest
    /// (partial installs, `.staging`) and legacy engine manifests are skipped;
    /// an unparseable manifest is an error.
    pub fn list(&self) -> Result<Vec<RuntimeManifest>, RuntimeStoreError> {
        self.walk(false)
    }

    /// Like [`Self::list`], but warn and skip an unparseable manifest, so one
    /// torn entry does not stop the worker's capability survey.
    pub fn list_intact(&self) -> Result<Vec<RuntimeManifest>, RuntimeStoreError> {
        self.walk(true)
    }

    fn walk(&self, skip_unparseable: bool) -> Result<Vec<RuntimeManifest>, RuntimeStoreError> {
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
            .filter(|read| match read {
                Err(RuntimeStoreError::Obsolete(path)) => {
                    tracing::warn!(
                        path = %path.display(),
                        "skipping a runtime entry from another bundled build; \
                         run `runtimes pull` to build this client's runtime"
                    );
                    false
                }
                Err(RuntimeStoreError::Parse { path, source }) if skip_unparseable => {
                    tracing::warn!(
                        path = %path.display(),
                        error = %source,
                        "runtime manifest does not parse; skipping its entry"
                    );
                    false
                }
                _ => true,
            })
            .collect::<Result<Vec<RuntimeManifest>, _>>()?;
        manifests.sort_by_key(|m| m.declared.cli_ref());
        Ok(manifests)
    }

    /// Remove a runtime's entry; returns whether one was present. A docker
    /// runtime's daemon image is left in place.
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

/// Free of the store because the quota survey reads entries the store never
/// resolved.
pub(crate) fn read_manifest(entry: &Path) -> Result<RuntimeManifest, RuntimeStoreError> {
    let manifest_path = entry.join(MANIFEST_NAME);
    let raw = fs::read(&manifest_path).map_err(|source| RuntimeStoreError::Io {
        doing: format!("reading {}", manifest_path.display()),
        source,
    })?;
    // Bytes that are not UTF-8 fail the TOML parse as `Parse`, not as `Io`.
    let m: RuntimeManifest = toml::from_str(&String::from_utf8_lossy(&raw)).map_err(|source| {
        RuntimeStoreError::Parse {
            path: manifest_path.clone(),
            source,
        }
    })?;
    if m.manifest_version != crate::runtime::manifest::MANIFEST_VERSION {
        return Err(RuntimeStoreError::Corrupt(format!(
            "unsupported manifest_version {} (expected {})",
            m.manifest_version,
            crate::runtime::manifest::MANIFEST_VERSION
        )));
    }
    let key = RuntimeStorageKey::of(&m.declared);
    if matches!(&m.declared, Runtime::AppleCoreAiMacosPipette(_)) {
        // Core AI includes this client's embedded sources in its key. An entry
        // from another build must not advertise the currently bundled runtime.
        let current = key
            .as_ref()
            .is_ok_and(|key| entry.file_name().is_some_and(|name| name == key.as_str()));
        if !current {
            return Err(RuntimeStoreError::Obsolete(entry.to_path_buf()));
        }
    }
    key?;
    let expected = to_stored(&m.declared).map_err(|e| RuntimeStoreError::Corrupt(e.to_string()))?;
    if m.stored != expected {
        return Err(RuntimeStoreError::Corrupt(
            "stored form is not the declared runtime's local form".to_owned(),
        ));
    }
    Ok(m)
}

/// Who wrote a `runtimes/*/manifest.toml`, from a peek.
///
/// `Engine` is a legacy serde-tagged `RuntimeEngine` record. Its dir names
/// (e.g. `vllm@0.22.0+cu129.py3.12`) can't collide with a
/// [`RuntimeStorageKey`], so `find` never lands on one. The list walk skips
/// them rather than fail an older workspace; [`crate::quota`] reclaims them.
pub(crate) enum ManifestKind {
    Store,
    Engine,
    Unknown,
}

pub(crate) fn classify_manifest_toml(path: &Path) -> Result<ManifestKind, RuntimeStoreError> {
    let raw = fs::read(path).map_err(|source| RuntimeStoreError::Io {
        doing: format!("reading {}", path.display()),
        source,
    })?;
    let value: toml::Value = toml::from_str(&String::from_utf8_lossy(&raw)).map_err(|source| {
        RuntimeStoreError::Parse {
            path: path.to_path_buf(),
            source,
        }
    })?;
    // A document with no keys is an empty file, which a crash during a write
    // can leave. The store reader reports it as `Parse`, as `find` does.
    if value.get("manifest_version").is_some() || value.as_table().is_some_and(|t| t.is_empty()) {
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
        LlamaCppFlavor, LlamacppCliStockTools, LlamacppCliStockToolsRelease,
        LlamacppCliStockToolsSource, NonEmptyString, RepositoryUrl,
    };

    use super::*;

    fn declared() -> anyhow::Result<Runtime> {
        Ok(Runtime::LlamacppCliStockTools(LlamacppCliStockTools {
            source: LlamacppCliStockToolsSource::GithubRelease(LlamacppCliStockToolsRelease {
                repository_url: RepositoryUrl::new("github.com/ggml-org/llama.cpp"),
                repository_version: NonEmptyString::try_new("b9305".to_owned())?,
                asset: None,
            }),
            flavor: LlamaCppFlavor::MacosArm64,
        }))
    }

    fn entry_dir(root: &std::path::Path, runtime: &Runtime) -> anyhow::Result<std::path::PathBuf> {
        Ok(root.join(RuntimeStorageKey::of(runtime)?.relative_dir()))
    }

    fn install_stub(_declared: &Runtime, blobs_dir: &std::path::Path) -> anyhow::Result<()> {
        fs::write(blobs_dir.join("llama-server"), b"")?;
        Ok(())
    }

    fn counting_install(
        calls: &Cell<usize>,
    ) -> impl FnMut(&Runtime, &std::path::Path) -> anyhow::Result<()> + '_ {
        move |declared, blobs_dir| {
            calls.set(calls.get() + 1);
            install_stub(declared, blobs_dir)
        }
    }

    #[test]
    fn find_is_none_until_ensured_then_fetches_once() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let store = RuntimeArtifactStore::new(tmp.path().to_path_buf());
        let calls = Cell::new(0usize);
        let runtime = declared()?;

        assert!(store.find(&runtime)?.is_none(), "not installed yet");

        let mut install = counting_install(&calls);
        let manifest = store.ensure(&runtime, &mut install)?;
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

    /// A crash mid-write left 444 zero bytes in a manifest on a handset; a torn
    /// write can also leave non-UTF-8 bytes.
    #[rstest::rstest]
    #[case::zero_bytes(&[0u8; 444])]
    #[case::not_utf8(&[0xFFu8; 444])]
    #[case::not_toml(b"not = [a manifest")]
    fn ensure_reinstalls_an_entry_whose_manifest_does_not_parse(
        #[case] manifest: &[u8],
    ) -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let store = RuntimeArtifactStore::new(tmp.path().to_path_buf());
        let runtime = declared()?;
        let calls = Cell::new(0usize);
        let manifest_path = entry_dir(tmp.path(), &runtime)?.join(MANIFEST_NAME);
        let mut counted = counting_install(&calls);
        let mut install = |d: &Runtime, blobs: &std::path::Path| {
            anyhow::ensure!(
                !manifest_path.exists(),
                "the old entry is gone before the install"
            );
            counted(d, blobs)
        };
        store.ensure(&runtime, &mut install)?;
        fs::write(&manifest_path, manifest)?;

        let (reinstalled, logged) =
            crate::test_support::logged_text(|| store.ensure(&runtime, &mut install))?;
        let reinstalled = reinstalled?;

        assert!(
            logged.contains("WARN") && logged.contains(&manifest_path.display().to_string()),
            "the damaged entry is logged with its path: {logged}"
        );
        assert_eq!(calls.get(), 2, "the damaged entry is installed again");
        assert_eq!(reinstalled.declared, runtime);
        assert_eq!(
            store.find(&runtime)?.map(|m| m.declared),
            Some(runtime),
            "the replaced manifest reads back"
        );
        Ok(())
    }

    #[rstest::rstest]
    #[case::zero_bytes(&[0u8; 444])]
    #[case::empty(&[])]
    #[case::not_utf8(&[0xFFu8; 444])]
    #[case::not_toml(b"not = [a manifest")]
    fn list_intact_skips_a_zero_byte_manifest_that_list_reports(
        #[case] manifest: &[u8],
    ) -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let store = RuntimeArtifactStore::new(tmp.path().to_path_buf());
        store.ensure(&declared()?, install_stub)?;
        let torn = tmp.path().join("llama-cpp__torn");
        fs::create_dir_all(&torn)?;
        fs::write(torn.join(MANIFEST_NAME), manifest)?;

        let intact = store.list_intact()?;

        assert_eq!(intact.len(), 1);
        assert_eq!(intact[0].declared, declared()?);
        assert!(matches!(store.list(), Err(RuntimeStoreError::Parse { .. })));
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
        let tmp = tempfile::tempdir()?;
        let store = RuntimeArtifactStore::new(tmp.path().to_path_buf());
        let manifest = store.ensure(&declared()?, |_, blobs_dir| {
            fs::write(blobs_dir.join(MANIFEST_NAME), b"theirs = true")?;
            fs::write(blobs_dir.join("llama-server"), b"")?;
            Ok(())
        })?;
        assert_eq!(manifest.declared, declared()?);
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
        let mut install = counting_install(&calls);
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

    /// A v1 manifest fails at deserialization (a missing field), not at the
    /// version check.
    #[test]
    fn a_v1_manifest_is_unreadable() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let store = RuntimeArtifactStore::new(tmp.path().to_path_buf());
        let runtime = declared()?;
        let calls = Cell::new(0usize);
        let mut install = counting_install(&calls);
        store.ensure(&runtime, &mut install)?;

        let manifest_path = entry_dir(tmp.path(), &runtime)?.join(MANIFEST_NAME);
        let mut table: toml::Table = toml::from_str(&fs::read_to_string(&manifest_path)?)?;
        table.insert("manifest_version".into(), toml::Value::Integer(1));
        table.remove("last_used_at");
        fs::write(&manifest_path, toml::to_string(&table)?)?;

        assert!(matches!(
            store.find(&runtime),
            Err(RuntimeStoreError::Parse { .. })
        ));
        store.ensure(&runtime, &mut install)?;
        assert_eq!(calls.get(), 2, "the v1 entry is installed again");
        assert!(
            store.find(&runtime)?.is_some(),
            "the replaced manifest reads back"
        );
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
