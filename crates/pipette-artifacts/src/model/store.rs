//! [`ModelArtifactStore`]: resolve a declared model to its entry under the
//! workspace `models/` tree. The store owns where the bytes live; a fetch
//! callback owns how they arrive.

use std::fs::{self, File};
use std::path::{Path, PathBuf};

use time::OffsetDateTime;

use pipette_plan_types::{Model, ModelType};

use super::key::{ModelStorageKey, ModelStorageKeyError};
use super::manifest::{MANIFEST_VERSION, ModelManifest};
use super::stored::{ModelStoredError, to_stored};
use crate::entry::{
    BLOBS_DIR_NAME, MANIFEST_NAME, STAGING_DIR_NAME, entry_size_bytes, hold_entry,
    install_dir_computing_manifest, lock_entry,
};
use crate::quota::QuotaError;

/// A model-store failure.
#[derive(Debug, thiserror::Error)]
pub enum ModelStoreError {
    /// The model has no storage key: Apple Foundation or a store-relative source.
    #[error(transparent)]
    NotStorable(#[from] ModelStorageKeyError),
    #[error(transparent)]
    UnresolvedPath(#[from] ModelStoredError),
    /// A manifest that decodes but has an unsupported version or a `stored`
    /// that drifted from `declared`.
    #[error("stored model is corrupt: {0}")]
    Corrupt(String),
    #[error(transparent)]
    Quota(#[from] QuotaError),
    #[error("fetching model `{model}` failed")]
    Fetch {
        model: String,
        #[source]
        source: anyhow::Error,
    },
    #[error("{doing}")]
    Io {
        doing: String,
        #[source]
        source: std::io::Error,
    },
    /// Not TOML, or a typed field failed to deserialize (e.g. a non-RFC 3339
    /// `fetched_at`). `ensure` treats this as a torn write and refetches.
    #[error("parsing the manifest at {path} failed")]
    Parse {
        path: PathBuf,
        #[source]
        source: toml::de::Error,
    },
    /// Two models of different types share one storage key (see
    /// [`ModelStorageKey`]).
    #[error(
        "the store entry `{key}` holds a {stored} model, but this cell names a {requested} \
         model; give each of the two a `prefix` so they address separate entries, or run \
         `models remove` to free this one"
    )]
    TypeCollision {
        key: String,
        stored: ModelType,
        requested: ModelType,
    },
}

/// Filesystem model store rooted at a workspace `models/` directory.
///
/// [`Self::ensure`] stages a fetch and renames it into place, so a reader never
/// sees a half-written entry. A crash mid-fetch leaves a staged dir under
/// `.staging`, which [`crate::quota`] reclaims as garbage.
pub struct ModelArtifactStore {
    models_dir: PathBuf,
}

impl ModelArtifactStore {
    pub fn new(models_dir: PathBuf) -> Self {
        Self { models_dir }
    }

    pub fn models_dir(&self) -> &Path {
        &self.models_dir
    }

    fn entry_dir(&self, key: &ModelStorageKey) -> PathBuf {
        self.models_dir.join(key.relative_dir())
    }

    /// The stored manifest for `declared`, or `None` if it isn't stored. Reads
    /// only: no fetch, no `last_used_at` touch.
    ///
    /// Checks `manifest.toml`, not the blobs it points at, so an out-of-band blob
    /// deletion surfaces when the model loads.
    ///
    /// `Err(TypeCollision)` if the entry holds another type: returning that
    /// manifest would bind a model the caller never asked for.
    pub fn find(&self, declared: &Model) -> Result<Option<ModelManifest>, ModelStoreError> {
        let key = ModelStorageKey::of(declared)?;
        let manifest_path = self.entry_dir(&key).join(MANIFEST_NAME);
        if !manifest_path.exists() {
            return Ok(None);
        }
        let manifest = read_manifest(manifest_path)?;
        let stored = ModelType::of(&manifest.declared);
        let requested = ModelType::of(declared);
        if stored != requested {
            return Err(ModelStoreError::TypeCollision {
                key: key.as_str().to_owned(),
                stored,
                requested,
            });
        }
        Ok(Some(manifest))
    }

    /// Like [`Self::find`], but `None` when the manifest does not parse: a
    /// publish replaces such an entry. [`Self::ensure`] repairs it under its
    /// lock.
    pub(crate) fn find_intact(
        &self,
        declared: &Model,
    ) -> Result<Option<ModelManifest>, ModelStoreError> {
        match self.find(declared) {
            Err(ModelStoreError::Parse { .. }) => Ok(None),
            found => found,
        }
    }

    /// The in-use lock on `declared`'s entry (see [`hold_entry`]), or `None` when
    /// a sweep removed the entry before the lock was taken. The inner `None` is a
    /// store that cannot be locked.
    pub(crate) fn hold(&self, declared: &Model) -> Result<Option<Option<File>>, ModelStoreError> {
        let key = ModelStorageKey::of(declared)?;
        let lock = hold_entry(&self.models_dir, key.as_str());
        Ok(self.find_intact(declared)?.map(|_| lock))
    }

    /// Every stored model, sorted by declared identity. Hidden dirs and children
    /// without a manifest are skipped. A bad manifest fails the listing, as in
    /// [`Self::find`], rather than being silently dropped.
    pub fn list(&self) -> Result<Vec<ModelManifest>, ModelStoreError> {
        let dir = match fs::read_dir(&self.models_dir) {
            Ok(dir) => dir,
            Err(source) if source.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(source) => {
                return Err(ModelStoreError::Io {
                    doing: format!("reading {}", self.models_dir.display()),
                    source,
                });
            }
        };
        let mut manifests = dir
            .filter_map(|entry| {
                let entry = match io(format!("reading {}", self.models_dir.display()), entry) {
                    Ok(entry) => entry,
                    Err(err) => return Some(Err(err)),
                };
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

    /// Delete `declared`'s whole entry dir, returning whether it was present.
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

    /// The manifest for `declared`, fetching it first if it isn't stored.
    ///
    /// `fetch(declared, into)` puts the files at `into`'s paths, under the
    /// staged `blobs/`, and verifies `sha256` itself: the store does not re-hash.
    ///
    /// A manifest that decodes but fails a check is `Err`, so the store reports
    /// the corruption rather than overwriting it. One that does not decode (a
    /// crash mid-write, or a v1 manifest) is removed and fetched again.
    pub fn ensure<F>(&self, declared: &Model, fetch: F) -> Result<ModelManifest, ModelStoreError>
    where
        F: FnMut(&Model, &Model) -> anyhow::Result<()>,
    {
        if let Some(manifest) = self.touched(self.find_intact(declared)?) {
            return Ok(manifest);
        }
        // Another process sharing the store may be fetching it: wait for it,
        // then look again.
        let key = ModelStorageKey::of(declared)?;
        let _fetching = lock_entry(&self.models_dir, key.as_str()).map_err(|source| {
            ModelStoreError::Fetch {
                model: declared.without_auth_token().to_string(),
                source,
            }
        })?;
        let found = match self.find(declared) {
            Err(ModelStoreError::Parse { path, source }) => {
                tracing::warn!(
                    path = %path.display(),
                    error = %source,
                    "model manifest does not parse; removing the entry to refetch it"
                );
                // Before the fetch, so that the disk does not hold two payloads.
                self.remove(declared)?;
                None
            }
            found => found?,
        };
        match self.touched(found) {
            Some(manifest) => Ok(manifest),
            None => self.fetch_and_publish(declared, fetch),
        }
    }

    fn touched(&self, found: Option<ModelManifest>) -> Option<ModelManifest> {
        found.map(|mut manifest| {
            self.touch(&mut manifest);
            manifest
        })
    }

    /// Stamp `last_used_at`, the sweep's eviction order.
    /// Best-effort: a read-only store must still resolve models, so a failed
    /// write is logged and the in-memory value advances regardless.
    fn touch(&self, manifest: &mut ModelManifest) {
        manifest.last_used_at = OffsetDateTime::now_utc();
        let Ok(key) = ModelStorageKey::of(&manifest.declared) else {
            return;
        };
        let entry = self.entry_dir(&key);
        if let Err(err) = crate::entry::rewrite_manifest(&entry, MANIFEST_NAME, manifest) {
            tracing::debug!(
                path = %entry.display(),
                error = %format_args!("{err:#}"),
                "could not record last_used_at"
            );
        }
    }

    fn fetch_and_publish<F>(
        &self,
        declared: &Model,
        mut fetch: F,
    ) -> Result<ModelManifest, ModelStoreError>
    where
        F: FnMut(&Model, &Model) -> anyhow::Result<()>,
    {
        // The auth strip covers only what the store keeps: `fetch` gets the
        // caller's `declared` so a gated download can still authenticate.
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

/// Decode the manifest at `manifest_path` and enforce store policy on it.
pub(crate) fn read_manifest(manifest_path: PathBuf) -> Result<ModelManifest, ModelStoreError> {
    let raw = io(
        format!("reading {}", manifest_path.display()),
        fs::read(&manifest_path),
    )?;
    // Bytes that are not UTF-8 fail the TOML parse as `Parse`, not as `Io`.
    let mut m: ModelManifest =
        toml::from_str(&String::from_utf8_lossy(&raw)).map_err(|source| {
            ModelStoreError::Parse {
                path: manifest_path,
                source,
            }
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
        AbsolutePath, AuthToken, GgufText, GgufTextSource, HfOrg, HfRepo, HfRepoName, ModelSource,
        OnnxGenai, Openvino, RepoSubpath,
    };

    use super::*;
    use crate::entry::BLOBS_DIR_NAME;

    fn declared() -> anyhow::Result<Model> {
        declared_named("llama")
    }

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

        assert!(store.find(&model)?.is_some());
        store.ensure(&model, counting_fetch(&counter))?;
        assert_eq!(counter.calls.get(), 1);
        Ok(())
    }

    /// A crash mid-write can leave zero bytes or non-UTF-8 bytes; that must not
    /// fail every job that names the model.
    #[rstest::rstest]
    #[case::zero_bytes(&[0u8; 444])]
    #[case::not_utf8(&[0xFFu8; 444])]
    #[case::not_toml(b"not = [a manifest")]
    fn ensure_refetches_an_entry_whose_manifest_does_not_parse(
        #[case] manifest: &[u8],
    ) -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let store = ModelArtifactStore::new(tmp.path().to_path_buf());
        let counter = CallCounter::default();
        let model = declared()?;
        store.ensure(&model, counting_fetch(&counter))?;
        let manifest_path = stored_manifest_path(tmp.path(), &model)?;
        fs::write(&manifest_path, manifest)?;

        let mut fetch = counting_fetch(&counter);
        let (refetched, logged) = crate::test_support::logged_text(|| {
            store.ensure(&model, |d: &Model, into: &Model| {
                anyhow::ensure!(
                    !manifest_path.exists(),
                    "the old entry is gone before the fetch"
                );
                fetch(d, into)
            })
        })?;
        let refetched = refetched?;

        assert!(
            logged.contains("WARN") && logged.contains(&manifest_path.display().to_string()),
            "the damaged entry is logged with its path: {logged}"
        );
        assert_eq!(counter.calls.get(), 2, "the damaged entry is fetched again");
        assert_eq!(refetched.declared, model);
        assert_eq!(
            store.find(&model)?.map(|m| m.declared),
            Some(model),
            "the replaced manifest reads back"
        );
        Ok(())
    }

    #[test]
    fn ensure_waits_for_a_fetch_in_flight_and_reuses_it() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let store = ModelArtifactStore::new(tmp.path().to_path_buf());
        let model = declared()?;
        let held = lock_entry(tmp.path(), ModelStorageKey::of(&model)?.as_str())?;
        let waiter_fetches = std::sync::atomic::AtomicUsize::new(0);

        let resolved = std::thread::scope(|s| -> anyhow::Result<ModelManifest> {
            let waiter = s.spawn(|| {
                store.ensure(&model, |_, into| {
                    waiter_fetches.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    write_empty_gguf_text(into)
                })
            });
            // Time for the waiter to miss and block on the lock. A slower one
            // finds the entry published below on its first look, so it never
            // fetches either way.
            std::thread::sleep(std::time::Duration::from_millis(200));
            store.fetch_and_publish(&model, |_, into| write_empty_gguf_text(into))?;
            drop(held);
            Ok(waiter
                .join()
                .map_err(|_| anyhow::anyhow!("the waiter panicked"))??)
        })?;

        assert_eq!(waiter_fetches.into_inner(), 0);
        assert_eq!(resolved.declared, model);
        Ok(())
    }

    #[test]
    fn a_second_model_type_on_one_repo_is_refused_by_name() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let store = ModelArtifactStore::new(tmp.path().to_path_buf());
        let repo = ModelSource::HuggingFace {
            repo: HfRepo {
                org: HfOrg::try_new("acme".to_owned())?,
                repo_name: HfRepoName::try_new("both-formats".to_owned())?,
                revision: None,
                auth_token: None,
            },
            prefix: None,
        };
        let openvino = Model::Openvino(Openvino {
            source: repo.clone(),
        });
        let onnx = Model::OnnxGenai(OnnxGenai { source: repo });
        assert_eq!(
            ModelStorageKey::of(&openvino)?.as_str(),
            ModelStorageKey::of(&onnx)?.as_str(),
            "the two types share one key, which is what makes the guard load-bearing"
        );

        store.ensure(&openvino, |_d, into| {
            let Model::Openvino(Openvino {
                source: ModelSource::AbsoluteDir { dir },
            }) = into
            else {
                anyhow::bail!("fake fetch only handles a bound openvino dir");
            };
            let dir = Path::new(dir.as_ref());
            fs::create_dir_all(dir)?;
            fs::write(dir.join("openvino_model.xml"), b"")?;
            Ok(())
        })?;

        let Err(ModelStoreError::TypeCollision {
            stored, requested, ..
        }) = store.find(&onnx)
        else {
            anyhow::bail!("the onnx model should be refused, not served the openvino entry");
        };
        assert_eq!(stored, ModelType::Openvino);
        assert_eq!(requested, ModelType::OnnxGenai);
        // The first type still resolves: the entry is its own.
        assert!(store.find(&openvino)?.is_some());
        Ok(())
    }

    #[test]
    fn a_failed_fetch_leaves_no_entry() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let store = ModelArtifactStore::new(tmp.path().to_path_buf());

        assert!(
            store
                .ensure(&declared()?, |_d, _i| anyhow::bail!("network down"))
                .is_err()
        );
        assert!(
            !tmp.path().join("meta__llama__Q4.gguf").exists(),
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
    fn remove_deletes_a_stored_entry_and_reports_absence() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let store = ModelArtifactStore::new(tmp.path().to_path_buf());
        let model = declared()?;

        assert!(!store.remove(&model)?, "nothing to remove yet");

        store.ensure(&model, counting_fetch(&CallCounter::default()))?;
        assert!(store.find(&model)?.is_some(), "stored after ensure");

        assert!(store.remove(&model)?, "stored entry removed");
        assert!(store.find(&model)?.is_none(), "entry gone after remove");
        assert!(!store.remove(&model)?, "already gone");
        Ok(())
    }

    fn store_with_broken_manifest(tmp: &Path) -> anyhow::Result<(ModelArtifactStore, Model)> {
        let store = ModelArtifactStore::new(tmp.to_path_buf());
        let model = declared()?;
        store.ensure(&model, counting_fetch(&CallCounter::default()))?;
        let manifest_path = tmp.join("meta__llama__Q4.gguf").join(MANIFEST_NAME);
        fs::write(&manifest_path, "not = valid = toml")?;
        Ok((store, model))
    }

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

    /// The strip covers what the store persists, not what it fetches with:
    /// otherwise every private repo answers 401.
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

    #[test]
    fn hold_finds_an_entry_removed_before_the_lock() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let store = ModelArtifactStore::new(tmp.path().to_path_buf());
        let model = declared()?;
        store.ensure(&model, counting_fetch(&CallCounter::default()))?;

        store.remove(&model)?;
        assert!(store.hold(&model)?.is_none());
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn a_read_only_store_holds_a_hit() -> anyhow::Result<()> {
        use std::os::unix::fs::PermissionsExt;

        let tmp = tempfile::tempdir()?;
        let store = ModelArtifactStore::new(tmp.path().to_path_buf());
        let model = declared()?;
        store.ensure(&model, counting_fetch(&CallCounter::default()))?;
        // The first hold, on a writable store, creates the lock file.
        drop(store.hold(&model)?);
        let locks = tmp.path().join(crate::entry::LOCKS_DIR_NAME);
        let lock = locks.join(format!(
            "{}{}",
            ModelStorageKey::of(&model)?.as_str(),
            crate::entry::USE_SUFFIX
        ));

        let set = |mode| {
            [lock.as_path(), locks.as_path(), tmp.path()]
                .iter()
                .try_for_each(|path| fs::set_permissions(path, fs::Permissions::from_mode(mode)))
        };
        set(0o500)?;
        let held = store.hold(&model);
        set(0o700)?;

        assert!(matches!(held?, Some(Some(_))));
        Ok(())
    }

    /// No migration: a v1 manifest is missing a field, so it fails to decode
    /// before the version check, and `ensure` refetches it.
    #[test]
    fn a_v1_manifest_is_unreadable() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let store = ModelArtifactStore::new(tmp.path().to_path_buf());
        let counter = CallCounter::default();
        let model = declared()?;
        store.ensure(&model, counting_fetch(&counter))?;

        let manifest_path = stored_manifest_path(tmp.path(), &model)?;
        let mut table: toml::Table = toml::from_str(&fs::read_to_string(&manifest_path)?)?;
        table.insert("manifest_version".into(), toml::Value::Integer(1));
        table.remove("last_used_at");
        fs::write(&manifest_path, toml::to_string(&table)?)?;

        assert!(matches!(
            store.find(&model),
            Err(ModelStoreError::Parse { .. })
        ));
        store.ensure(&model, counting_fetch(&counter))?;
        assert_eq!(counter.calls.get(), 2, "the v1 entry is fetched again");
        assert!(
            store.find(&model)?.is_some(),
            "the replaced manifest reads back"
        );
        Ok(())
    }

    #[test]
    fn list_is_empty_for_a_fresh_store() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let store = ModelArtifactStore::new(tmp.path().join("models"));
        assert!(store.list()?.is_empty());
        Ok(())
    }

    /// Identity stays the authoring path; `stored` re-homes under `<key>/blobs/`.
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
            other @ (Model::GgufText(_)
            | Model::GgufVision(_)
            | Model::GgufVisionSpeculative(_)
            | Model::Mlx(_)
            | Model::MlxSpeculative(_)
            | Model::TorchSpeculative(_)
            | Model::Torch(_)
            | Model::Openvino(_)
            | Model::OnnxGenai(_)
            | Model::AppleFoundationText
            | Model::AppleCoreAi(_)) => anyhow::bail!("unexpected bind_under: {other:?}"),
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
