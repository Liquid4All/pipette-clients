//! [`ModelArtifactStore`] — resolve a declared model to its stored on-disk form.
//!
//! The store owns *where* a model's bytes live (the workspace `models/` tree);
//! a fetch callback owns *how* they arrive. [`ModelArtifactStore::find`] is the
//! read path (is it stored?); [`ModelArtifactStore::ensure`] is find-or-fetch:
//! it fetches once into a staging dir, records a [`ModelManifest`], and swaps the
//! entry into place with a rename (replacing any stale copy first) — so a partial
//! or failed fetch never leaves a half-written entry, and a re-run resolves the
//! already-pulled artifact.
//!
//! Both return the [`ModelManifest`]: `.declared` is the reporting identity,
//! `.bind_under(models_dir)` binds `stored` paths under the models dir.

use std::fs;
use std::path::{Path, PathBuf};

use time::OffsetDateTime;

use pipette_plan_types::Model;

use super::key::{ModelStorageKey, ModelStorageKeyError};
use super::manifest::{ModelManifest, MANIFEST_VERSION};
use super::stored::{to_stored, ModelStoredError};
use crate::entry::{
    entry_size_bytes, install_dir_computing_manifest, BLOBS_DIR_NAME, MANIFEST_NAME,
    STAGING_DIR_NAME,
};
use crate::quota::QuotaError;

/// A model-store failure.
#[derive(Debug, thiserror::Error)]
pub enum ModelStoreError {
    /// The model has no storage key (OS-bundled Apple Foundation — not stored).
    #[error(transparent)]
    NotStorable(#[from] ModelStorageKeyError),
    /// The declared source doesn't resolve to a usable on-disk path.
    #[error(transparent)]
    UnresolvedPath(#[from] ModelStoredError),
    /// A stored manifest is unusable — bad version, drift, or timestamp.
    #[error("stored model is corrupt: {0}")]
    Corrupt(String),
    /// The model cannot fit under the configured storage quota.
    #[error(transparent)]
    Quota(#[from] QuotaError),
    /// The fetcher failed to pull the model's bytes.
    #[error("fetching model `{model}` failed")]
    Fetch {
        model: String,
        #[source]
        source: anyhow::Error,
    },
    /// A filesystem operation failed; `doing` names it.
    #[error("{doing}")]
    Io {
        doing: String,
        #[source]
        source: std::io::Error,
    },
    /// A manifest on disk didn't parse as TOML, or a typed field failed
    /// deserialization (e.g. non-RFC 3339 `fetched_at`).
    #[error("parsing the manifest at {path} failed")]
    Parse {
        path: PathBuf,
        #[source]
        source: toml::de::Error,
    },
}

/// Filesystem model store rooted at a workspace `models/` directory.
///
/// Assumes a single writer: [`Self::ensure`] stages a fetch, then swaps it into
/// place with a rename (replacing any stale copy first), so a reader never sees a
/// half-written entry. Two concurrent `ensure` calls for the same model may both
/// fetch — the loser's entry is replaced, not corrupted.
///
/// A crash mid-fetch can orphan a `.staging/<uuid>` dir under `models/`; those are
/// safe to delete and are the operator's to garbage-collect.
pub struct ModelArtifactStore {
    models_dir: PathBuf,
}

impl ModelArtifactStore {
    pub fn new(models_dir: PathBuf) -> Self {
        Self { models_dir }
    }

    /// Root directory this store manages (`…/models`).
    pub fn models_dir(&self) -> &Path {
        &self.models_dir
    }

    /// The on-disk entry directory for a stored model: `models_dir/<key>`.
    fn entry_dir(&self, key: &ModelStorageKey) -> PathBuf {
        self.models_dir.join(key.relative_dir())
    }

    /// The stored manifest for `declared`, or `None` if it isn't stored. Pure
    /// read — no fetch, no mutation. `Err(Parse)` if the manifest bytes don't
    /// parse as TOML or a field fails type deserialization (including a
    /// non–RFC 3339 `fetched_at`); `Err(Corrupt)` if they parse but fail
    /// validation (version, `stored`/`declared` drift).
    ///
    /// Only `manifest.toml` is checked — the blob files it points at are not, so
    /// an out-of-band blob deletion isn't caught here (nor by [`Self::ensure`]);
    /// a loader surfaces it at load time.
    pub fn find(&self, declared: &Model) -> Result<Option<ModelManifest>, ModelStoreError> {
        let key = ModelStorageKey::of(declared)?;
        let manifest_path = self.entry_dir(&key).join(MANIFEST_NAME);
        if !manifest_path.exists() {
            return Ok(None);
        }
        Ok(Some(read_manifest(manifest_path)?))
    }

    /// Every stored model, read from each `models_dir/<key>/manifest.toml` and
    /// sorted by declared identity. The `.staging` scratch dir and any child
    /// without a manifest are skipped; a missing `models_dir` (fresh workspace) is
    /// an empty list. A corrupt or unparseable manifest surfaces as `Err`, the
    /// same strictness as [`Self::find`] — one bad entry fails the listing rather
    /// than being silently dropped.
    pub fn list(&self) -> Result<Vec<ModelManifest>, ModelStoreError> {
        let dir = match fs::read_dir(&self.models_dir) {
            Ok(dir) => dir,
            Err(source) if source.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(source) => {
                return Err(ModelStoreError::Io {
                    doing: format!("reading {}", self.models_dir.display()),
                    source,
                })
            }
        };
        let mut manifests = dir
            .filter_map(|entry| {
                let entry = match io(format!("reading {}", self.models_dir.display()), entry) {
                    Ok(entry) => entry,
                    Err(err) => return Some(Err(err)),
                };
                // Skip reserved / hidden dirs (`.staging`, etc.).
                if entry
                    .file_name()
                    .to_str()
                    .is_none_or(|name| name.starts_with('.'))
                {
                    return None;
                }
                let manifest_path = entry.path().join(MANIFEST_NAME);
                manifest_path.exists().then(|| read_manifest(manifest_path))
            })
            .collect::<Result<Vec<_>, _>>()?;
        manifests.sort_by_key(|manifest| manifest.declared.to_string());
        Ok(manifests)
    }

    /// Delete `declared`'s stored entry (`models/<key>/`), returning whether it
    /// was present. Removes the whole entry dir — manifest and blobs together.
    pub fn remove(&self, declared: &Model) -> Result<bool, ModelStoreError> {
        let entry = self.entry_dir(&ModelStorageKey::of(declared)?);
        if !entry.exists() {
            return Ok(false);
        }
        io(
            format!("removing {}", entry.display()),
            fs::remove_dir_all(&entry),
        )?;
        Ok(true)
    }

    /// The manifest for `declared`, fetching + recording it first if it isn't
    /// already stored. Call `.bind_under(models_dir)` on the result for loader paths.
    ///
    /// `fetch` puts `declared`'s files at the absolute Local paths in `into`
    /// (the store entry's `blobs/`): remote → download, local → copy. It verifies
    /// `sha256` when present — the store does not re-hash.
    ///
    /// A stored-but-corrupt entry (bad version, drift, tamper) surfaces as `Err`,
    /// not a re-fetch — surfacing the corruption over silently overwriting it.
    pub fn ensure<F>(&self, declared: &Model, fetch: F) -> Result<ModelManifest, ModelStoreError>
    where
        F: FnMut(&Model, &Model) -> anyhow::Result<()>,
    {
        match self.find(declared)? {
            Some(mut manifest) => {
                self.touch(&mut manifest);
                Ok(manifest)
            }
            None => self.fetch_and_publish(declared, fetch),
        }
    }

    /// Stamp `last_used_at = now` on a resolved entry — the eviction order.
    /// Best-effort: a read-only store must still resolve models, so a failed
    /// write is logged and the in-memory value advances regardless.
    fn touch(&self, manifest: &mut ModelManifest) {
        manifest.last_used_at = OffsetDateTime::now_utc();
        let Ok(key) = ModelStorageKey::of(&manifest.declared) else {
            return;
        };
        let entry = self.entry_dir(&key);
        if let Err(err) = crate::entry::rewrite_manifest(&entry, MANIFEST_NAME, manifest) {
            log::debug!(
                "could not record last_used_at for {}: {err:#}",
                entry.display()
            );
        }
    }

    /// Fetch into a staging dir, then swap it into place — delegating staging,
    /// the reserved-name guard, atomic publish, and cleanup-on-failure to the
    /// shared [`crate::entry`] engine.
    ///
    /// The manifest is computed rather than passed in, because `blobs_bytes` is
    /// only knowable once the fetch has landed. Measuring the staged payload
    /// here is what lets [`crate::quota`] total a store from manifests alone.
    fn fetch_and_publish<F>(
        &self,
        declared: &Model,
        mut fetch: F,
    ) -> Result<ModelManifest, ModelStoreError>
    where
        F: FnMut(&Model, &Model) -> anyhow::Result<()>,
    {
        // Store owns storable-key policy, auth strip, version stamp, and the
        // relative `stored` form. The strip covers only what the store keeps:
        // `fetch` gets the caller's `declared` so a gated download can still
        // authenticate.
        let storable = declared.without_auth_token();
        let key = ModelStorageKey::of(&storable)?;
        let entry = self.entry_dir(&key);
        let fetched_at = OffsetDateTime::now_utc();
        let stored = to_stored(&storable, &key.blobs_dir())?;
        install_dir_computing_manifest(
            &self.models_dir.join(STAGING_DIR_NAME),
            &entry,
            key.as_str(),
            MANIFEST_NAME,
            |staged| {
                let blobs = staged.join(BLOBS_DIR_NAME);
                std::fs::create_dir_all(&blobs)?;
                let into = to_stored(&storable, &blobs)?;
                fetch(declared, &into)?;
                Ok(ModelManifest {
                    manifest_version: MANIFEST_VERSION,
                    fetched_at,
                    last_used_at: fetched_at,
                    declared: storable.clone(),
                    stored,
                    blobs_bytes: Some(entry_size_bytes(&blobs)),
                })
            },
        )
        .map_err(|source| ModelStoreError::Fetch {
            model: storable.to_string(),
            source,
        })
    }
}

/// Deserialize + enforce store policy for the manifest at `manifest_path`
/// (assumed to exist).
///
/// - TOML / field types → [`ModelStoreError::Parse`]
/// - version, `stored` drift → [`ModelStoreError::Corrupt`]
/// - non-storable `declared` → [`ModelStoreError::NotStorable`]
pub(crate) fn read_manifest(manifest_path: PathBuf) -> Result<ModelManifest, ModelStoreError> {
    let raw = io(
        format!("reading {}", manifest_path.display()),
        fs::read_to_string(&manifest_path),
    )?;
    let mut m: ModelManifest = toml::from_str(&raw).map_err(|source| ModelStoreError::Parse {
        path: manifest_path,
        source,
    })?;

    // Drop any token so in-memory `declared` never carries secrets.
    m.declared = m.declared.without_auth_token();

    if m.manifest_version != MANIFEST_VERSION {
        return Err(ModelStoreError::Corrupt(format!(
            "unsupported manifest_version {} (expected {MANIFEST_VERSION})",
            m.manifest_version
        )));
    }
    let key = ModelStorageKey::of(&m.declared)?;
    let expected = to_stored(&m.declared, &key.blobs_dir())?;
    if m.stored != expected {
        return Err(ModelStoreError::Corrupt(
            "stored form is not the declared model's local form".to_owned(),
        ));
    }
    Ok(m)
}

/// Tag an [`std::io::Result`] with what filesystem operation produced it.
fn io<T>(doing: impl Into<String>, result: std::io::Result<T>) -> Result<T, ModelStoreError> {
    result.map_err(|source| ModelStoreError::Io {
        doing: doing.into(),
        source,
    })
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::path::Path;

    use rstest::rstest;

    use pipette_plan_types::{
        AbsolutePath, AuthToken, GgufText, GgufTextSource, HfOrg, HfRepo, HfRepoName, RepoSubpath,
    };

    use super::*;
    use crate::entry::BLOBS_DIR_NAME;

    fn declared() -> anyhow::Result<Model> {
        declared_named("llama")
    }

    /// A gguf-text model whose repo name varies so it lands in its own entry.
    fn declared_named(repo_name: &str) -> anyhow::Result<Model> {
        declared_named_with_auth(repo_name, None)
    }

    fn declared_named_with_auth(repo_name: &str, auth: Option<&str>) -> anyhow::Result<Model> {
        Ok(Model::GgufText(GgufText {
            source: GgufTextSource::HuggingFace {
                repo: HfRepo {
                    org: HfOrg::try_new("meta".to_owned())?,
                    repo_name: HfRepoName::try_new(repo_name.to_owned())?,
                    revision: None,
                    auth_token: auth.map(|t| AuthToken::try_new(t.to_owned())).transpose()?,
                },
                path: RepoSubpath::try_new("Q4.gguf")?,
                sha256: None,
            },
        }))
    }

    fn stored_manifest_path(tmp: &Path, model: &Model) -> anyhow::Result<PathBuf> {
        let key = ModelStorageKey::of(model)?;
        Ok(tmp.join(key.relative_dir()).join(MANIFEST_NAME))
    }

    /// Writes empty files at the `Local` paths in `into`, counting calls so a
    /// test can prove the second `ensure` resolves without re-fetching.
    #[derive(Default)]
    struct CallCounter {
        calls: Cell<usize>,
    }

    fn write_empty_gguf_text(into: &Model) -> anyhow::Result<()> {
        let Model::GgufText(GgufText {
            source: GgufTextSource::AbsoluteFile { path },
        }) = into
        else {
            anyhow::bail!("fake fetch only handles gguf-text");
        };
        let path = Path::new(path.as_ref());
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, b"")?;
        Ok(())
    }

    fn counting_fetch<'a>(
        counter: &'a CallCounter,
    ) -> impl FnMut(&Model, &Model) -> anyhow::Result<()> + 'a {
        move |_declared, into| {
            counter.calls.set(counter.calls.get() + 1);
            write_empty_gguf_text(into)
        }
    }

    #[test]
    fn find_is_none_until_ensured_then_fetches_once() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let store = ModelArtifactStore::new(tmp.path().to_path_buf());
        let counter = CallCounter::default();
        let model = declared()?;

        assert!(store.find(&model)?.is_none(), "not stored yet");

        let manifest = store.ensure(&model, counting_fetch(&counter))?;
        let entry = tmp.path().join("meta__llama__Q4.gguf");
        assert!(
            entry.join("blobs/Q4.gguf").exists(),
            "weights landed under blobs/"
        );
        assert!(
            entry.join(MANIFEST_NAME).exists(),
            "manifest at the entry root"
        );
        assert_eq!(
            manifest.bind_under(tmp.path())?,
            to_stored(&model, &entry.join(BLOBS_DIR_NAME))?
        );

        // `find` now resolves it, and a second `ensure` doesn't re-fetch.
        assert!(store.find(&model)?.is_some());
        store.ensure(&model, counting_fetch(&counter))?;
        assert_eq!(counter.calls.get(), 1);
        Ok(())
    }

    #[test]
    fn a_failed_fetch_leaves_no_entry() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let store = ModelArtifactStore::new(tmp.path().to_path_buf());

        assert!(store
            .ensure(&declared()?, |_d, _i| anyhow::bail!("network down"))
            .is_err());
        assert!(
            !tmp.path().join("meta__llama__Q4.gguf").exists(),
            "no entry published on failure"
        );
        // And the staging dir is cleaned up, not orphaned.
        let staging = tmp.path().join(STAGING_DIR_NAME);
        assert!(
            !staging.exists() || fs::read_dir(&staging)?.next().is_none(),
            "no staged remnant"
        );
        Ok(())
    }

    #[test]
    fn remove_deletes_a_stored_entry_and_reports_absence() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let store = ModelArtifactStore::new(tmp.path().to_path_buf());
        let model = declared()?;

        // Nothing stored yet: `remove` reports it wasn't there.
        assert!(!store.remove(&model)?, "nothing to remove yet");

        store.ensure(&model, counting_fetch(&CallCounter::default()))?;
        assert!(store.find(&model)?.is_some(), "stored after ensure");

        // `remove` deletes the entry, reports it was there, and a repeat is false.
        assert!(store.remove(&model)?, "stored entry removed");
        assert!(store.find(&model)?.is_none(), "entry gone after remove");
        assert!(!store.remove(&model)?, "already gone");
        Ok(())
    }

    /// Store one model, then overwrite its manifest with unparseable bytes.
    fn store_with_broken_manifest(tmp: &Path) -> anyhow::Result<(ModelArtifactStore, Model)> {
        let store = ModelArtifactStore::new(tmp.to_path_buf());
        let model = declared()?;
        store.ensure(&model, counting_fetch(&CallCounter::default()))?;
        let manifest_path = tmp.join("meta__llama__Q4.gguf").join(MANIFEST_NAME);
        fs::write(&manifest_path, "not = valid = toml")?;
        Ok((store, model))
    }

    /// Both read paths — the single-model `find` and the whole-store `list` —
    /// surface an unparseable manifest as `Parse` rather than skipping it.
    #[rstest]
    #[case::find(|store: &ModelArtifactStore, model: &Model| store.find(model).map(|_| ()))]
    #[case::list(|store: &ModelArtifactStore, _model: &Model| store.list().map(|_| ()))]
    fn a_broken_manifest_surfaces_as_parse_error(
        #[case] read: impl Fn(&ModelArtifactStore, &Model) -> Result<(), ModelStoreError>,
    ) -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let (store, model) = store_with_broken_manifest(tmp.path())?;
        assert!(matches!(
            read(&store, &model),
            Err(ModelStoreError::Parse { .. })
        ));
        Ok(())
    }

    #[test]
    fn a_drifted_manifest_is_corrupt() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let store = ModelArtifactStore::new(tmp.path().to_path_buf());
        let model = declared()?;
        store.ensure(&model, counting_fetch(&CallCounter::default()))?;

        // Tamper with the stored blob path so it no longer matches `declared` —
        // valid TOML, but the drift check must reject it as `Corrupt`.
        let manifest_path = stored_manifest_path(tmp.path(), &model)?;
        let good = fs::read_to_string(&manifest_path)?;
        let drifted = good.replace("blobs/Q4.gguf", "blobs/TAMPERED.gguf");
        assert_ne!(good, drifted, "the tamper must actually land");
        fs::write(&manifest_path, drifted)?;
        assert!(matches!(
            store.find(&model),
            Err(ModelStoreError::Corrupt(_))
        ));
        Ok(())
    }

    #[test]
    fn unsupported_manifest_version_is_corrupt() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let store = ModelArtifactStore::new(tmp.path().to_path_buf());
        let model = declared()?;
        store.ensure(&model, counting_fetch(&CallCounter::default()))?;

        let manifest_path = stored_manifest_path(tmp.path(), &model)?;
        let mut table: toml::Table = toml::from_str(&fs::read_to_string(&manifest_path)?)?;
        table.insert("manifest_version".into(), toml::Value::Integer(99));
        fs::write(&manifest_path, toml::to_string(&table)?)?;

        assert!(matches!(
            store.find(&model),
            Err(ModelStoreError::Corrupt(_))
        ));
        Ok(())
    }

    #[test]
    fn ensure_and_find_strip_auth_tokens() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let store = ModelArtifactStore::new(tmp.path().to_path_buf());
        let secret = "hf_secret_token";
        let model = declared_named_with_auth("llama", Some(secret))?;

        let written = store.ensure(&model, counting_fetch(&CallCounter::default()))?;
        assert!(written.declared.auth_token().is_none());
        let on_disk = fs::read_to_string(stored_manifest_path(tmp.path(), &model)?)?;
        assert!(
            !on_disk.contains(secret),
            "token must not be serialized to disk"
        );

        // Hand-edit a token into the flattened `declared` table; load strips it.
        let mut table: toml::Table = toml::from_str(&on_disk)?;
        let declared = table
            .get_mut("declared")
            .and_then(|v| v.as_table_mut())
            .ok_or_else(|| anyhow::anyhow!("declared table"))?;
        declared.insert("auth_token".into(), toml::Value::String(secret.to_owned()));
        fs::write(
            stored_manifest_path(tmp.path(), &model)?,
            toml::to_string(&table)?,
        )?;

        let loaded = store
            .find(&model.without_auth_token())?
            .ok_or_else(|| anyhow::anyhow!("expected stored entry"))?;
        assert!(loaded.declared.auth_token().is_none());
        Ok(())
    }

    /// The strip covers what the store persists, not what it fetches with — a
    /// strip that also reached `fetch` sent gated downloads out unauthenticated
    /// and every private repo answered 401.
    #[test]
    fn ensure_hands_the_auth_token_to_the_fetch() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let store = ModelArtifactStore::new(tmp.path().to_path_buf());
        let secret = "hf_secret_token";
        let model = declared_named_with_auth("llama", Some(secret))?;

        let seen: Cell<Option<String>> = Cell::new(None);
        store.ensure(&model, |declared, into| {
            seen.set(declared.auth_token().map(|t| t.as_ref().to_owned()));
            write_empty_gguf_text(into)
        })?;

        assert_eq!(seen.into_inner().as_deref(), Some(secret));
        Ok(())
    }

    #[test]
    fn garbage_fetched_at_is_a_parse_error() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let store = ModelArtifactStore::new(tmp.path().to_path_buf());
        let model = declared()?;
        store.ensure(&model, counting_fetch(&CallCounter::default()))?;

        let manifest_path = stored_manifest_path(tmp.path(), &model)?;
        let mut table: toml::Table = toml::from_str(&fs::read_to_string(&manifest_path)?)?;
        table.insert(
            "fetched_at".into(),
            toml::Value::String("not-a-timestamp".into()),
        );
        fs::write(&manifest_path, toml::to_string(&table)?)?;

        assert!(matches!(
            store.find(&model),
            Err(ModelStoreError::Parse { .. })
        ));
        Ok(())
    }

    /// `last_used_at` as it stands on disk, not as the store last returned it.
    fn stored_last_used_at(tmp: &Path, model: &Model) -> anyhow::Result<OffsetDateTime> {
        let raw = fs::read_to_string(stored_manifest_path(tmp, model)?)?;
        Ok(toml::from_str::<ModelManifest>(&raw)?.last_used_at)
    }

    #[test]
    fn ensure_seeds_last_used_at_on_publish() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let store = ModelArtifactStore::new(tmp.path().to_path_buf());
        let model = declared()?;

        let manifest = store.ensure(&model, counting_fetch(&CallCounter::default()))?;

        assert_eq!(manifest.last_used_at, manifest.fetched_at);
        Ok(())
    }

    #[test]
    fn ensure_touches_last_used_at_on_a_cache_hit() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let store = ModelArtifactStore::new(tmp.path().to_path_buf());
        let counter = CallCounter::default();
        let model = declared()?;
        let published = store.ensure(&model, counting_fetch(&counter))?;

        let resolved = store.ensure(&model, counting_fetch(&counter))?;

        assert_eq!(counter.calls.get(), 1, "a hit does not re-fetch");
        assert_eq!(resolved.fetched_at, published.fetched_at);
        assert!(resolved.last_used_at > published.last_used_at);
        assert_eq!(
            stored_last_used_at(tmp.path(), &model)?,
            resolved.last_used_at
        );
        Ok(())
    }

    #[test]
    fn find_does_not_touch_last_used_at() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let store = ModelArtifactStore::new(tmp.path().to_path_buf());
        let model = declared()?;
        store.ensure(&model, counting_fetch(&CallCounter::default()))?;
        let before = stored_last_used_at(tmp.path(), &model)?;

        store.find(&model)?;

        assert_eq!(stored_last_used_at(tmp.path(), &model)?, before);
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn a_failed_touch_still_resolves() -> anyhow::Result<()> {
        use std::os::unix::fs::PermissionsExt;

        let tmp = tempfile::tempdir()?;
        let store = ModelArtifactStore::new(tmp.path().to_path_buf());
        let model = declared()?;
        let published = store.ensure(&model, counting_fetch(&CallCounter::default()))?;

        let entry = tmp.path().join("meta__llama__Q4.gguf");
        fs::set_permissions(&entry, fs::Permissions::from_mode(0o500))?;
        let resolved = store.ensure(&model, counting_fetch(&CallCounter::default()));
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
        let store = ModelArtifactStore::new(tmp.path().to_path_buf());
        let model = declared()?;
        store.ensure(&model, counting_fetch(&CallCounter::default()))?;

        let manifest_path = stored_manifest_path(tmp.path(), &model)?;
        let mut table: toml::Table = toml::from_str(&fs::read_to_string(&manifest_path)?)?;
        table.insert("manifest_version".into(), toml::Value::Integer(1));
        table.remove("last_used_at");
        fs::write(&manifest_path, toml::to_string(&table)?)?;

        assert!(matches!(
            store.find(&model),
            Err(ModelStoreError::Parse { .. })
        ));
        Ok(())
    }

    #[test]
    fn list_is_empty_for_a_fresh_store() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        // `models_dir` doesn't exist yet — a fresh workspace lists nothing.
        let store = ModelArtifactStore::new(tmp.path().join("models"));
        assert!(store.list()?.is_empty());
        Ok(())
    }

    /// Local-authored models are storable: identity stays the authoring path,
    /// `stored` re-homes under `<key>/blobs/`, and the fetcher imports the file.
    #[test]
    fn ensure_imports_a_local_gguf_into_the_store() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let src = tmp.path().join("authoring/w.gguf");
        let parent = src
            .parent()
            .ok_or_else(|| anyhow::anyhow!("missing parent for {}", src.display()))?;
        fs::create_dir_all(parent)?;
        fs::write(&src, b"local-gguf")?;

        let model = Model::GgufText(GgufText {
            source: GgufTextSource::AbsoluteFile {
                path: AbsolutePath::try_new(src.to_string_lossy().into_owned())?,
            },
        });
        // Copy local plan path into store blobs (mirrors fetch_model local path).
        let copy_local = |declared: &Model, into: &Model| -> anyhow::Result<()> {
            let Model::GgufText(GgufText {
                source: GgufTextSource::AbsoluteFile { path: src },
            }) = declared
            else {
                anyhow::bail!("expected local declared");
            };
            let Model::GgufText(GgufText {
                source: GgufTextSource::AbsoluteFile { path: dest },
            }) = into
            else {
                anyhow::bail!("expected local into");
            };
            let dest = Path::new(dest.as_ref());
            if let Some(parent) = dest.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::copy(Path::new(src.as_ref()), dest)?;
            Ok(())
        };

        let models_dir = tmp.path().join("models");
        let store = ModelArtifactStore::new(models_dir.clone());
        let manifest = store.ensure(&model, copy_local)?;

        assert_eq!(manifest.declared, model);
        let Model::GgufText(GgufText {
            source: GgufTextSource::RelativeFile { path: stored },
        }) = &manifest.stored
        else {
            anyhow::bail!("expected relative stored");
        };
        // Manifest `stored` is models-root-relative, never absolute.
        assert!(
            !stored.as_ref().starts_with('/'),
            "stored must be relative to model storage root: {stored}"
        );
        assert!(
            stored.as_ref().ends_with("/blobs/w.gguf") || stored.as_ref().ends_with("blobs/w.gguf"),
            "stored path re-homed under blobs: {stored}"
        );

        let abs = match manifest.bind_under(&models_dir)? {
            Model::GgufText(GgufText {
                source: GgufTextSource::AbsoluteFile { path },
            }) => PathBuf::from(path.as_ref()),
            other => anyhow::bail!("unexpected bind_under: {other:?}"),
        };
        assert_eq!(fs::read(&abs)?, b"local-gguf");
        assert_eq!(fs::read(&src)?, b"local-gguf");
        Ok(())
    }

    #[test]
    fn list_returns_stored_models_sorted_and_skips_staging() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let store = ModelArtifactStore::new(tmp.path().to_path_buf());
        let counter = CallCounter::default();
        // Ensure out of sorted order; a crash-orphaned staging dir must not appear.
        store.ensure(&declared_named("zephyr")?, counting_fetch(&counter))?;
        store.ensure(&declared_named("llama")?, counting_fetch(&counter))?;
        fs::create_dir_all(
            tmp.path()
                .join(STAGING_DIR_NAME)
                .join("orphan")
                .join(BLOBS_DIR_NAME),
        )?;

        let ids: Vec<String> = store
            .list()?
            .iter()
            .map(|m| m.declared.to_string())
            .collect();
        assert_eq!(
            ids,
            vec![
                declared_named("llama")?.to_string(),
                declared_named("zephyr")?.to_string(),
            ],
            "sorted by declared identity, `.staging` skipped"
        );
        Ok(())
    }
}
