//! Storage quota: a cap on the disk the artifact stores may occupy, enforced
//! after each publish (see `docs/storage-quota.md`).
//!
//! Three concrete steps — [`survey`] → [`plan`] → [`apply_sweep`]. The
//! split is what makes a dry run free and the eviction order testable without a
//! filesystem: `survey` is the only step that reads disk, `plan` is pure,
//! `apply_sweep` is the only step that deletes. `survey` reads one manifest per
//! entry and no payload file — entries record their own size at publish.
//!
//! The manifest is the unit of accounting: an entry counts toward the quota if
//! and only if it carries a manifest this build can read. Everything else under
//! a store root is garbage, and garbage is what the sweep reclaims first.
//!
//! Nothing here prints. [`apply_sweep`] reports every removal back to the
//! caller so no delete is silent.

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

use time::OffsetDateTime;

use pipette_plan_types::{Model, Runtime};

use crate::entry::{entry_size_bytes, BLOBS_DIR_NAME, MANIFEST_NAME, STAGING_DIR_NAME};
use crate::model::store::read_manifest as read_model_manifest;
use crate::model::ModelStorageKey;
use crate::runtime::store::{
    classify_manifest_toml, read_manifest as read_runtime_manifest, ManifestKind,
};
use crate::runtime::RuntimeStorageKey;

/// Why a fetch was refused on quota grounds.
#[derive(Debug, thiserror::Error)]
pub enum QuotaError {
    /// The artifact cannot fit even in an empty store, so fetching it would
    /// evict everything and still not fit.
    #[error(
        "`{artifact}` needs {needed_bytes} bytes but the whole storage quota is \
         {quota_bytes} bytes; raise the storage quota to fetch it"
    )]
    Oversize {
        artifact: String,
        needed_bytes: u64,
        quota_bytes: u64,
    },
}

/// What an entry under a store root is, for accounting purposes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EntryKind {
    /// No manifest this build can read: a `.staging/*` orphan, a manifest-less
    /// child, a legacy schema, or a corrupt record. Free to drop, no policy
    /// involved.
    Garbage { reason: String },
    Model {
        key: ModelStorageKey,
        last_used_at: OffsetDateTime,
        fetched_at: OffsetDateTime,
    },
    /// `evictable` is false for docker: the image lives in the daemon, so the
    /// entry measures ~0 and dropping it frees nothing.
    Runtime {
        key: RuntimeStorageKey,
        last_used_at: OffsetDateTime,
        fetched_at: OffsetDateTime,
        evictable: bool,
    },
}

impl EntryKind {
    /// The eviction timestamp, or `None` for garbage (which has no policy).
    pub fn last_used_at(&self) -> Option<OffsetDateTime> {
        match self {
            Self::Garbage { .. } => None,
            Self::Model { last_used_at, .. } | Self::Runtime { last_used_at, .. } => {
                Some(*last_used_at)
            }
        }
    }

    /// Publish time, breaking a `last_used_at` tie. Two entries resolved in the
    /// same second are otherwise ordered by directory-read order, which differs
    /// between runs and between clients.
    fn fetched_at(&self) -> Option<OffsetDateTime> {
        match self {
            Self::Garbage { .. } => None,
            Self::Model { fetched_at, .. } | Self::Runtime { fetched_at, .. } => Some(*fetched_at),
        }
    }

    /// Sweep rank: garbage, then models, then runtimes.
    fn rank(&self) -> u8 {
        match self {
            Self::Garbage { .. } => 0,
            Self::Model { .. } => 1,
            Self::Runtime { .. } => 2,
        }
    }
}

/// One thing under a store root that occupies disk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StorageEntry {
    /// What [`apply_sweep`] would remove.
    pub path: PathBuf,
    /// How the entry reads in a report: the model's identity, the runtime's
    /// `cli_ref`, or the store-relative path for garbage.
    pub label: String,
    /// What removing this entry would free. Garbage measures the whole
    /// directory; a live entry measures `blobs/` alone, because the size its
    /// manifest records cannot include the manifest carrying it — so a live
    /// entry under-reports by one manifest, erring low.
    pub size_bytes: u64,
    pub kind: EntryKind,
}

/// Everything under both store roots, already in sweep order.
#[derive(Debug, Clone, Default)]
pub struct StorageSurvey {
    pub entries: Vec<StorageEntry>,
    pub used_bytes: u64,
}

impl StorageSurvey {
    /// Order `entries` as the sweep would reclaim them and total their size.
    /// Garbage keeps discovery order; live entries sort least-recently-used
    /// first within their store.
    fn new(mut entries: Vec<StorageEntry>) -> Self {
        entries.sort_by_key(|entry| {
            (
                entry.kind.rank(),
                entry.kind.last_used_at(),
                entry.kind.fetched_at(),
            )
        });
        let used_bytes = entries
            .iter()
            .fold(0u64, |total, entry| total.saturating_add(entry.size_bytes));
        Self {
            entries,
            used_bytes,
        }
    }
}

/// The never-evict set: the entry just fetched plus whatever the in-flight plan
/// declares. A non-storable artifact has no key and is simply not pinned.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SweepPins {
    models: HashSet<ModelStorageKey>,
    runtimes: HashSet<RuntimeStorageKey>,
}

impl SweepPins {
    /// The artifacts one benchmark cell needs resolved at the same time — the
    /// runtime's ensure must not evict the model the model's ensure is about to
    /// resolve, and vice versa.
    pub fn for_cell(model: &Model, runtime: &Runtime) -> Self {
        Self::default().with_model(model).with_runtime(runtime)
    }

    pub(crate) fn with_model(mut self, model: &Model) -> Self {
        if let Ok(key) = ModelStorageKey::of(model) {
            self.models.insert(key);
        }
        self
    }

    pub(crate) fn with_runtime(mut self, runtime: &Runtime) -> Self {
        if let Ok(key) = RuntimeStorageKey::of(runtime) {
            self.runtimes.insert(key);
        }
        self
    }

    /// Absorb `other`'s pins.
    pub(crate) fn merge(&mut self, other: Self) {
        self.models.extend(other.models);
        self.runtimes.extend(other.runtimes);
    }
}

/// Both store roots plus the cap they share. Rides on
/// [`crate::ArtifactsContext`] because the sweep spans both stores while an
/// `ensure` sees only one of them.
#[derive(Debug, Clone)]
pub struct StoragePolicy {
    pub(crate) quota_bytes: u64,
    pub(crate) models_dir: PathBuf,
    pub(crate) runtimes_dir: PathBuf,
    pub(crate) pins: SweepPins,
}

impl StoragePolicy {
    /// Cap the stores rooted at `models_dir` / `runtimes_dir`, pinning nothing
    /// yet — the run layer adds the in-flight cell's pins.
    pub fn new(quota_bytes: u64, models_dir: PathBuf, runtimes_dir: PathBuf) -> Self {
        Self {
            quota_bytes,
            models_dir,
            runtimes_dir,
            pins: SweepPins::default(),
        }
    }
}

/// What a sweep would drop to bring the store back under the cap.
#[derive(Debug, Clone, Default)]
pub struct SweepPlan {
    pub evictions: Vec<StorageEntry>,
    pub freed_bytes: u64,
    /// Set when the candidates run out while still over — the caller warns and
    /// continues; a run never fails over disk bookkeeping.
    pub still_over_by_bytes: Option<u64>,
}

/// What a sweep actually did.
#[derive(Debug, Clone, Default)]
pub struct SweepReport {
    pub removed: Vec<StorageEntry>,
    /// Entries the sweep planned but could not remove, with the reason.
    pub failed: Vec<(StorageEntry, String)>,
    pub freed_bytes: u64,
}

/// Classify everything under both store roots, in sweep order.
///
/// Never fails on one bad entry: an unreadable manifest is garbage, which is
/// the whole point of the manifest-as-unit-of-accounting rule. That is also why
/// this walks the roots itself instead of going through the stores' `list`,
/// which fails the whole listing on one bad manifest — the resolve path keeps
/// its strictness, the accountant does not.
pub fn survey(models_dir: &Path, runtimes_dir: &Path) -> StorageSurvey {
    let mut entries = scan_store(models_dir, model_entry);
    entries.extend(scan_store(runtimes_dir, runtime_entry));
    StorageSurvey::new(entries)
}

/// What would be dropped to get `survey.used_bytes` at or under
/// `quota_bytes`, stopping the instant it fits. Pure: no filesystem access.
/// What would be dropped to bring `survey.used_bytes` to or under
/// `quota_bytes`: every piece of garbage, then live entries least-recently-used
/// first until it fits. Pure — no filesystem access.
///
/// Garbage goes unconditionally, whether or not the store is over. It is
/// unaccountable by definition and can never be pinned, so keeping it buys
/// nothing — and a store stranded by a manifest version bump is typically
/// *under* quota, where gating on the overage would leave the only recovery a
/// hand-deleted directory.
pub fn plan(survey: &StorageSurvey, quota_bytes: u64, pins: &SweepPins) -> SweepPlan {
    let mut freed_bytes = 0u64;
    let mut evictions = Vec::new();
    for entry in &survey.entries {
        let over_quota = survey.used_bytes.saturating_sub(freed_bytes) > quota_bytes;
        let garbage = matches!(entry.kind, EntryKind::Garbage { .. });
        // Garbage sorts ahead of every live entry, so reaching a live one while
        // the store fits means nothing later can qualify either.
        if !over_quota && !garbage {
            break;
        }
        if !is_candidate(entry, pins) {
            continue;
        }
        freed_bytes = freed_bytes.saturating_add(entry.size_bytes);
        evictions.push(entry.clone());
    }
    let remaining = survey.used_bytes.saturating_sub(freed_bytes);
    SweepPlan {
        evictions,
        freed_bytes,
        still_over_by_bytes: (remaining > quota_bytes).then(|| remaining - quota_bytes),
    }
}

/// Delete the planned entries, reporting each one and skipping any it cannot
/// remove — a failed unlink is bookkeeping, not a reason to fail a run.
pub fn apply_sweep(plan: &SweepPlan) -> SweepReport {
    plan.evictions
        .iter()
        .fold(SweepReport::default(), |mut report, entry| {
            match remove_entry(&entry.path) {
                Ok(()) => {
                    report.freed_bytes = report.freed_bytes.saturating_add(entry.size_bytes);
                    report.removed.push(entry.clone());
                }
                Err(source) => report.failed.push((entry.clone(), source.to_string())),
            }
            report
        })
}

/// Refuse before the fetch starts. Without this the fetch would evict the whole
/// store and still not fit. `declared_bytes` is `None` when the size isn't
/// knowable ahead of time (a uv/mlx venv), in which case the post-publish sweep
/// is the only enforcement.
pub(crate) fn refuse_if_oversize(
    artifact: &str,
    declared_bytes: Option<u64>,
    quota_bytes: u64,
) -> Result<(), QuotaError> {
    match declared_bytes {
        Some(needed_bytes) if needed_bytes > quota_bytes => Err(QuotaError::Oversize {
            artifact: artifact.to_owned(),
            needed_bytes,
            quota_bytes,
        }),
        _ => Ok(()),
    }
}

/// Bytes a live entry's payload occupies, preferring the size its manifest
/// recorded at publish over walking `blobs/`.
///
/// The recorded value is why a survey costs one small read per entry instead of
/// a traversal of every file in the store. It is absent only on entries
/// published before the field existed, which fall back to the walk — measuring
/// `blobs/` alone, as the recorded value does, so both paths agree.
fn payload_bytes(entry_dir: &Path, recorded: Option<u64>) -> u64 {
    recorded.unwrap_or_else(|| entry_size_bytes(&entry_dir.join(BLOBS_DIR_NAME)))
}

fn is_candidate(entry: &StorageEntry, pins: &SweepPins) -> bool {
    match &entry.kind {
        EntryKind::Garbage { .. } => true,
        EntryKind::Model { key, .. } => !pins.models.contains(key),
        EntryKind::Runtime { key, evictable, .. } => *evictable && !pins.runtimes.contains(key),
    }
}

fn remove_entry(path: &Path) -> std::io::Result<()> {
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.is_dir() => fs::remove_dir_all(path),
        Ok(_) => fs::remove_file(path),
        // Already gone is the outcome we wanted.
        Err(source) if source.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(source) => Err(source),
    }
}

/// The children of a store root, or an empty list for a fresh workspace.
fn store_children(root: &Path) -> Vec<PathBuf> {
    let Ok(read) = fs::read_dir(root) else {
        return Vec::new();
    };
    read.flatten().map(|child| child.path()).collect()
}

fn garbage(root: &Path, path: PathBuf, reason: &str) -> StorageEntry {
    StorageEntry {
        label: path
            .strip_prefix(root)
            .unwrap_or(&path)
            .display()
            .to_string(),
        size_bytes: entry_size_bytes(&path),
        kind: EntryKind::Garbage {
            reason: reason.to_owned(),
        },
        path,
    }
}

/// Each crashed fetch under `.staging` is its own garbage entry, so one orphan
/// doesn't hide behind the shared scratch dir's name.
fn staging_orphans(root: &Path, staging: &Path) -> Vec<StorageEntry> {
    store_children(staging)
        .into_iter()
        .map(|path| garbage(root, path, "orphaned staging dir"))
        .collect()
}

/// A child that is neither a live entry nor the staging root: whatever it is,
/// it carries no manifest, so it is garbage.
fn non_entry(root: &Path, path: &Path) -> Option<StorageEntry> {
    let name = path.file_name().and_then(|n| n.to_str())?;
    if !path.is_dir() {
        return Some(garbage(root, path.to_path_buf(), "not a store entry"));
    }
    if name.starts_with('.') {
        return Some(garbage(root, path.to_path_buf(), "not a store entry"));
    }
    if !path.join(MANIFEST_NAME).exists() {
        return Some(garbage(root, path.to_path_buf(), "no manifest"));
    }
    None
}

/// Classify every child of a store root: the staging dir yields one garbage
/// entry per crashed fetch, anything without a readable manifest is garbage,
/// and the rest goes to `entry_of` — the only part that differs between the
/// model and runtime stores.
fn scan_store(root: &Path, entry_of: impl Fn(&Path, PathBuf) -> StorageEntry) -> Vec<StorageEntry> {
    store_children(root)
        .into_iter()
        .flat_map(|path| {
            if path.file_name().is_some_and(|n| n == STAGING_DIR_NAME) {
                return staging_orphans(root, &path);
            }
            vec![non_entry(root, &path).unwrap_or_else(|| entry_of(root, path))]
        })
        .collect()
}

fn model_entry(models_dir: &Path, path: PathBuf) -> StorageEntry {
    let manifest = match read_model_manifest(path.join(MANIFEST_NAME)) {
        Ok(manifest) => manifest,
        Err(err) => return garbage(models_dir, path, &unreadable(&err)),
    };
    let Ok(key) = ModelStorageKey::of(&manifest.declared) else {
        return garbage(models_dir, path, "declared model has no storage key");
    };
    // A manifest whose payload is gone is a husk: with `blobs_bytes` recorded it
    // would otherwise hold quota against bytes that no longer exist, and never
    // be reclaimable. Publishing is atomic, so this only arises out of band.
    if !payload_present(&path, &manifest) {
        return garbage(models_dir, path, "manifest without its payload");
    }
    StorageEntry {
        label: manifest.declared.to_string(),
        size_bytes: payload_bytes(&path, manifest.blobs_bytes),
        kind: EntryKind::Model {
            key,
            last_used_at: manifest.last_used_at,
            fetched_at: manifest.fetched_at,
        },
        path,
    }
}

/// Whether the entry's payload is actually on disk. Mirrors the iOS husk rule
/// (`FileStorage.payloadURL`): the manifest names where the bytes should be, so
/// resolving it and finding nothing means the entry is no longer a model.
fn payload_present(entry_dir: &Path, manifest: &crate::model::ModelManifest) -> bool {
    let Some(models_dir) = entry_dir.parent() else {
        return true;
    };
    // No resolvable payload means nothing to miss: an OS-supplied model, or a
    // manifest the store already reports as corrupt.
    manifest
        .payload_paths(models_dir)
        .iter()
        .all(|path| path.exists())
}

fn runtime_entry(runtimes_dir: &Path, path: PathBuf) -> StorageEntry {
    // `list` skips the legacy torch-oai schema, but it still occupies disk, and
    // a manifest this build cannot read is reclaimable by rule.
    match classify_manifest_toml(&path.join(MANIFEST_NAME)) {
        Ok(ManifestKind::Store) => {}
        Ok(ManifestKind::Engine) => return garbage(runtimes_dir, path, "legacy engine manifest"),
        Ok(ManifestKind::Unknown) => {
            return garbage(runtimes_dir, path, "unrecognized manifest schema")
        }
        Err(err) => return garbage(runtimes_dir, path, &unreadable(&err)),
    }
    let manifest = match read_runtime_manifest(&path) {
        Ok(manifest) => manifest,
        Err(err) => return garbage(runtimes_dir, path, &unreadable(&err)),
    };
    let Ok(key) = RuntimeStorageKey::of(&manifest.declared) else {
        return garbage(runtimes_dir, path, "declared runtime has no storage key");
    };
    StorageEntry {
        label: manifest.declared.cli_ref(),
        size_bytes: payload_bytes(&path, manifest.blobs_bytes),
        kind: EntryKind::Runtime {
            key,
            last_used_at: manifest.last_used_at,
            fetched_at: manifest.fetched_at,
            // A docker image lives in the daemon, not here: evicting the entry
            // would free nothing.
            evictable: !matches!(
                manifest.declared,
                Runtime::DockerVllm(_) | Runtime::DockerSglang(_)
            ),
        },
        path,
    }
}

/// Generic over the error type: the model and runtime stores fail differently
/// but read the same in a garbage reason.
fn unreadable(err: &impl std::fmt::Display) -> String {
    format!("unreadable manifest: {err}")
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use rstest::rstest;
    use time::format_description::well_known::Rfc3339;

    use pipette_plan_types::{
        DockerVllm, GgufText, GgufTextSource, HfOrg, HfRepo, HfRepoName, LlamaCppFlavor,
        LlamacppCliStockTools, LlamacppCliStockToolsSource, NonEmptyString, RepoSubpath,
        RepositoryUrl, SourceRepository, VllmFlavor,
    };

    use super::*;
    use crate::model::ModelArtifactStore;
    use crate::runtime::RuntimeArtifactStore;

    fn at(rfc3339: &str) -> anyhow::Result<OffsetDateTime> {
        Ok(OffsetDateTime::parse(rfc3339, &Rfc3339)?)
    }

    fn model(repo_name: &str) -> anyhow::Result<Model> {
        Ok(Model::GgufText(GgufText {
            source: GgufTextSource::HuggingFace {
                repo: HfRepo {
                    org: HfOrg::try_new("meta".to_owned())?,
                    repo_name: HfRepoName::try_new(repo_name.to_owned())?,
                    revision: None,
                    auth_token: None,
                },
                path: RepoSubpath::try_new("Q4.gguf")?,
                sha256: None,
            },
        }))
    }

    fn llamacpp(version: &str) -> anyhow::Result<Runtime> {
        Ok(Runtime::LlamacppCliStockTools(LlamacppCliStockTools {
            source: LlamacppCliStockToolsSource::GithubRelease(SourceRepository {
                repository_url: RepositoryUrl::new("github.com/ggml-org/llama.cpp"),
                repository_version: NonEmptyString::try_new(version.to_owned())?,
            }),
            flavor: LlamaCppFlavor::MacosArm64,
        }))
    }

    fn docker() -> anyhow::Result<Runtime> {
        Ok(Runtime::DockerVllm(DockerVllm {
            image_name: NonEmptyString::try_new("vllm/vllm-openai".to_owned())?,
            image_tag: NonEmptyString::try_new("v0.22.0".to_owned())?,
            flavor: VllmFlavor::NvidiaGpu,
        }))
    }

    /// Store `declared` for real, then hand-edit `last_used_at` so a test can
    /// pin the eviction order without sleeping.
    fn store_model(models_dir: &Path, declared: &Model, last_used: &str) -> anyhow::Result<()> {
        let store = ModelArtifactStore::new(models_dir.to_path_buf());
        store.ensure(declared, |_d, into| {
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
            fs::write(path, b"weights")?;
            Ok(())
        })?;
        let key = ModelStorageKey::of(declared)?;
        set_last_used(&models_dir.join(key.relative_dir()), last_used)
    }

    fn store_runtime(
        runtimes_dir: &Path,
        declared: &Runtime,
        last_used: &str,
    ) -> anyhow::Result<()> {
        let store = RuntimeArtifactStore::new(runtimes_dir.to_path_buf());
        store.ensure(declared, |_d, blobs| {
            fs::write(blobs.join("llama-server"), b"server-binary")?;
            Ok(())
        })?;
        set_last_used(
            &runtimes_dir.join(RuntimeStorageKey::of(declared)?.relative_dir()),
            last_used,
        )
    }

    /// Overwrite the recorded payload size on a published entry.
    fn set_blobs_bytes(entry: &Path, bytes: i64) -> anyhow::Result<()> {
        edit_manifest(entry, |table| {
            table.insert("blobs_bytes".into(), toml::Value::Integer(bytes));
        })
    }

    /// Strip the recorded size, as an entry published before the field would be.
    fn drop_blobs_bytes(entry: &Path) -> anyhow::Result<()> {
        edit_manifest(entry, |table| {
            table.remove("blobs_bytes");
        })
    }

    fn edit_manifest(entry: &Path, edit: impl FnOnce(&mut toml::Table)) -> anyhow::Result<()> {
        let manifest_path = entry.join(MANIFEST_NAME);
        let mut table: toml::Table = toml::from_str(&fs::read_to_string(&manifest_path)?)?;
        edit(&mut table);
        fs::write(&manifest_path, toml::to_string(&table)?)?;
        Ok(())
    }

    fn set_last_used(entry: &Path, last_used: &str) -> anyhow::Result<()> {
        let manifest_path = entry.join(MANIFEST_NAME);
        let mut table: toml::Table = toml::from_str(&fs::read_to_string(&manifest_path)?)?;
        table.insert("last_used_at".into(), toml::Value::String(last_used.into()));
        fs::write(&manifest_path, toml::to_string(&table)?)?;
        Ok(())
    }

    /// A hand-built entry for the pure planner: `plan` never touches disk,
    /// so its cases are a table, not a filesystem.
    fn model_entry_of(
        declared: &Model,
        size_bytes: u64,
        last_used: &str,
    ) -> anyhow::Result<StorageEntry> {
        Ok(StorageEntry {
            path: PathBuf::from("/models").join(ModelStorageKey::of(declared)?.relative_dir()),
            label: declared.to_string(),
            size_bytes,
            kind: EntryKind::Model {
                key: ModelStorageKey::of(declared)?,
                last_used_at: at(last_used)?,
                fetched_at: at(last_used)?,
            },
        })
    }

    fn runtime_entry_of(
        declared: &Runtime,
        size_bytes: u64,
        last_used: &str,
        evictable: bool,
    ) -> anyhow::Result<StorageEntry> {
        Ok(StorageEntry {
            path: PathBuf::from("/runtimes").join(RuntimeStorageKey::of(declared)?.relative_dir()),
            label: declared.cli_ref(),
            size_bytes,
            kind: EntryKind::Runtime {
                key: RuntimeStorageKey::of(declared)?,
                last_used_at: at(last_used)?,
                fetched_at: at(last_used)?,
                evictable,
            },
        })
    }

    fn garbage_entry_of(name: &str, size_bytes: u64) -> StorageEntry {
        StorageEntry {
            path: PathBuf::from("/models").join(name),
            label: name.to_owned(),
            size_bytes,
            kind: EntryKind::Garbage {
                reason: "no manifest".to_owned(),
            },
        }
    }

    fn labels(entries: &[StorageEntry]) -> Vec<&str> {
        entries.iter().map(|e| e.label.as_str()).collect()
    }

    /// Two models and two runtimes, oldest-used first within each store, plus
    /// one garbage entry — 100 bytes each so quotas read as counts.
    fn mixed_survey() -> anyhow::Result<StorageSurvey> {
        Ok(StorageSurvey::new(vec![
            model_entry_of(&model("fresh")?, 100, "2026-05-01T00:00:00Z")?,
            runtime_entry_of(&llamacpp("b9000")?, 100, "2026-01-01T00:00:00Z", true)?,
            garbage_entry_of("orphan", 100),
            model_entry_of(&model("stale")?, 100, "2026-02-01T00:00:00Z")?,
            runtime_entry_of(&llamacpp("b9305")?, 100, "2026-06-01T00:00:00Z", true)?,
        ]))
    }

    #[test]
    fn survey_is_empty_for_missing_roots() {
        let survey = survey(
            Path::new("/pipette-missing-models"),
            Path::new("/pipette-missing-runtimes"),
        );
        assert!(survey.entries.is_empty());
        assert_eq!(survey.used_bytes, 0);
    }

    /// A manifest outliving its payload is a husk, not a model — the rule iOS
    /// applies in `isLive`. With `blobs_bytes` recorded it would otherwise hold
    /// quota against bytes that are gone, and never be reclaimable.
    #[test]
    fn survey_treats_a_manifest_without_its_payload_as_garbage() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let models = tmp.path().join("models");
        store_model(&models, &model("llama")?, "2026-03-01T00:00:00Z")?;
        let entry = models.join(ModelStorageKey::of(&model("llama")?)?.relative_dir());
        fs::remove_dir_all(entry.join(BLOBS_DIR_NAME))?;

        let survey = survey(&models, Path::new("/pipette-missing-runtimes"));

        assert!(
            matches!(survey.entries[0].kind, EntryKind::Garbage { .. }),
            "{:?}",
            survey.entries[0].kind
        );
        Ok(())
    }

    /// Two entries resolved in the same second fall back to publish time, so the
    /// order is the same on every run, and matches what iOS does.
    #[test]
    fn an_equal_last_used_tie_breaks_on_fetched_at() -> anyhow::Result<()> {
        let older = StorageEntry {
            kind: EntryKind::Model {
                key: ModelStorageKey::of(&model("older")?)?,
                last_used_at: at("2026-05-01T00:00:00Z")?,
                fetched_at: at("2026-01-01T00:00:00Z")?,
            },
            ..model_entry_of(&model("older")?, 100, "2026-05-01T00:00:00Z")?
        };
        let newer = StorageEntry {
            kind: EntryKind::Model {
                key: ModelStorageKey::of(&model("newer")?)?,
                last_used_at: at("2026-05-01T00:00:00Z")?,
                fetched_at: at("2026-04-01T00:00:00Z")?,
            },
            ..model_entry_of(&model("newer")?, 100, "2026-05-01T00:00:00Z")?
        };

        // Built newest-first so a stable sort alone could not produce the answer.
        let survey = StorageSurvey::new(vec![newer, older]);

        assert!(
            survey.entries[0].label.contains("older"),
            "{:?}",
            labels(&survey.entries)
        );
        Ok(())
    }

    /// The whole point of `blobs_bytes`: a survey totals a store from manifests,
    /// without reading a payload. Proven by lying in the manifest — the recorded
    /// number wins over what is actually on disk, which could only happen if the
    /// walk is skipped.
    #[test]
    fn survey_trusts_the_recorded_size_over_the_payload() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let models = tmp.path().join("models");
        store_model(&models, &model("llama")?, "2026-03-01T00:00:00Z")?;
        let entry = models.join(ModelStorageKey::of(&model("llama")?)?.relative_dir());
        set_blobs_bytes(&entry, 777_000)?;

        let survey = survey(&models, Path::new("/pipette-missing-runtimes"));

        assert_eq!(survey.entries[0].size_bytes, 777_000);
        assert_eq!(survey.used_bytes, 777_000);
        Ok(())
    }

    /// An entry published before the field falls back to measuring the payload —
    /// `blobs/` alone, so both paths report the same thing.
    #[test]
    fn survey_falls_back_to_the_walk_without_a_recorded_size() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let models = tmp.path().join("models");
        store_model(&models, &model("llama")?, "2026-03-01T00:00:00Z")?;
        let entry = models.join(ModelStorageKey::of(&model("llama")?)?.relative_dir());
        drop_blobs_bytes(&entry)?;

        let survey = survey(&models, Path::new("/pipette-missing-runtimes"));

        assert_eq!(
            survey.entries[0].size_bytes,
            entry_size_bytes(&entry.join(BLOBS_DIR_NAME)),
            "the fallback measures blobs/, not the entry dir"
        );
        assert!(survey.entries[0].size_bytes > 0);
        Ok(())
    }

    #[test]
    fn survey_reports_live_entries_with_size_and_last_used() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let models = tmp.path().join("models");
        let runtimes = tmp.path().join("runtimes");
        store_model(&models, &model("llama")?, "2026-03-01T00:00:00Z")?;
        store_runtime(&runtimes, &llamacpp("b9305")?, "2026-04-01T00:00:00Z")?;

        let survey = survey(&models, &runtimes);

        assert_eq!(
            labels(&survey.entries),
            vec![model("llama")?.to_string(), llamacpp("b9305")?.cli_ref()],
            "models sweep before runtimes"
        );
        assert!(survey.entries.iter().all(|e| e.size_bytes > 0));
        assert_eq!(
            survey.entries[0].kind.last_used_at(),
            Some(at("2026-03-01T00:00:00Z")?)
        );
        assert_eq!(
            survey.used_bytes,
            survey.entries.iter().map(|e| e.size_bytes).sum::<u64>()
        );
        Ok(())
    }

    #[test]
    fn survey_classifies_everything_without_a_readable_manifest_as_garbage() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let models = tmp.path().join("models");
        let runtimes = tmp.path().join("runtimes");
        store_model(&models, &model("llama")?, "2026-03-01T00:00:00Z")?;

        // A crashed fetch, a dir that never got a manifest, a stray file, and a
        // published entry whose manifest this build cannot read.
        fs::create_dir_all(models.join(STAGING_DIR_NAME).join("llama.staged-1"))?;
        fs::create_dir_all(models.join("no-manifest"))?;
        fs::create_dir_all(&runtimes)?;
        fs::write(models.join("stray.txt"), b"x")?;
        let corrupt = models.join("meta__zephyr__Q4.gguf");
        fs::create_dir_all(&corrupt)?;
        fs::write(corrupt.join(MANIFEST_NAME), "not = valid = toml")?;

        let survey = survey(&models, &runtimes);

        let reasons: Vec<(&str, &str)> = survey
            .entries
            .iter()
            .filter_map(|e| match &e.kind {
                EntryKind::Garbage { reason } => Some((e.label.as_str(), reason.as_str())),
                _ => None,
            })
            .collect();
        assert_eq!(reasons.len(), 4, "{reasons:?}");
        assert!(reasons
            .iter()
            .any(|(label, reason)| label.ends_with("llama.staged-1")
                && *reason == "orphaned staging dir"));
        assert!(reasons
            .iter()
            .any(|(label, reason)| *label == "no-manifest" && *reason == "no manifest"));
        assert!(reasons
            .iter()
            .any(|(label, reason)| *label == "stray.txt" && *reason == "not a store entry"));
        assert!(reasons
            .iter()
            .any(|(label, reason)| *label == "meta__zephyr__Q4.gguf"
                && reason.starts_with("unreadable manifest")));
        assert_eq!(
            survey.entries.last().map(|e| e.label.as_str()),
            Some(model("llama")?.to_string().as_str()),
            "garbage sweeps before the live model"
        );
        Ok(())
    }

    #[test]
    fn survey_classifies_a_legacy_engine_manifest_as_garbage() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let runtimes = tmp.path().join("runtimes");
        let legacy = runtimes.join("vllm@0.22.0+cu129.py3.12");
        fs::create_dir_all(&legacy)?;
        fs::write(
            legacy.join(MANIFEST_NAME),
            "type = \"uv_vllm\"\nslug = \"vllm@0.22.0+cu129.py3.12\"\n",
        )?;

        let survey = survey(&tmp.path().join("models"), &runtimes);

        assert!(matches!(
            survey.entries.as_slice(),
            [StorageEntry {
                kind: EntryKind::Garbage { reason },
                ..
            }] if reason == "legacy engine manifest"
        ));
        Ok(())
    }

    #[test]
    fn survey_marks_docker_runtimes_not_evictable() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let runtimes = tmp.path().join("runtimes");
        store_runtime(&runtimes, &docker()?, "2026-01-01T00:00:00Z")?;

        let survey = survey(&tmp.path().join("models"), &runtimes);

        assert!(matches!(
            survey.entries.as_slice(),
            [StorageEntry {
                kind: EntryKind::Runtime {
                    evictable: false,
                    ..
                },
                ..
            }]
        ));
        Ok(())
    }

    #[rstest]
    #[case::fits_but_garbage_still_goes(500, vec!["orphan"])]
    #[case::garbage_alone_is_enough(400, vec!["orphan"])]
    #[case::then_the_stalest_model(300, vec!["orphan", "stale"])]
    #[case::then_the_fresher_model(200, vec!["orphan", "stale", "fresh"])]
    #[case::runtimes_only_after_every_model(100, vec!["orphan", "stale", "fresh", "b9000"])]
    fn plan_takes_garbage_then_lru_until_it_fits(
        #[case] quota_bytes: u64,
        #[case] expected: Vec<&str>,
    ) -> anyhow::Result<()> {
        let survey = mixed_survey()?;
        let plan = plan(&survey, quota_bytes, &SweepPins::default());

        let evicted: Vec<String> = plan.evictions.iter().map(|e| e.label.clone()).collect();
        assert_eq!(evicted.len(), expected.len(), "{evicted:?}");
        evicted
            .iter()
            .zip(&expected)
            .for_each(|(actual, fragment)| {
                assert!(
                    actual.contains(fragment),
                    "{actual} should match {fragment}"
                );
            });
        assert_eq!(plan.freed_bytes, 100 * expected.len() as u64);
        assert!(plan.still_over_by_bytes.is_none());
        Ok(())
    }

    /// Garbage is unaccountable and can never be pinned, so keeping it buys
    /// nothing — it goes at any quota, which is also the recovery path for a
    /// store a manifest version bump stranded under quota.
    #[rstest]
    #[case::under_quota(500)]
    #[case::exactly_at_quota(400)]
    #[case::unlimited(u64::MAX)]
    fn plan_always_reclaims_garbage(#[case] quota_bytes: u64) -> anyhow::Result<()> {
        let survey = mixed_survey()?;

        let plan = plan(&survey, quota_bytes, &SweepPins::default());

        let evicted: Vec<&str> = labels(&plan.evictions);
        assert_eq!(evicted.len(), 1, "{evicted:?}");
        assert!(evicted[0].contains("orphan"), "{evicted:?}");
        assert!(plan.still_over_by_bytes.is_none());
        Ok(())
    }

    #[test]
    fn plan_never_evicts_a_pinned_model_or_runtime() -> anyhow::Result<()> {
        let survey = mixed_survey()?;
        let pins = SweepPins::for_cell(&model("stale")?, &llamacpp("b9000")?);

        let plan = plan(&survey, 100, &pins);

        let evicted: Vec<&str> = labels(&plan.evictions);
        assert!(
            !evicted
                .iter()
                .any(|l| l.contains("stale") || l.contains("b9000")),
            "{evicted:?}"
        );
        assert_eq!(evicted.len(), 3, "garbage, the unpinned model, then b9305");
        Ok(())
    }

    #[test]
    fn plan_never_evicts_a_docker_runtime() -> anyhow::Result<()> {
        // Docker measures ~0 and freeing it frees nothing, so a store made only
        // of docker entries stays over quota rather than being pointlessly torn
        // down.
        let survey = StorageSurvey::new(vec![runtime_entry_of(
            &docker()?,
            100,
            "2026-01-01T00:00:00Z",
            false,
        )?]);

        let plan = plan(&survey, 10, &SweepPins::default());

        assert!(plan.evictions.is_empty());
        assert_eq!(plan.still_over_by_bytes, Some(90));
        Ok(())
    }

    #[test]
    fn plan_reports_still_over_when_only_pinned_entries_remain() -> anyhow::Result<()> {
        let survey = StorageSurvey::new(vec![model_entry_of(
            &model("pinned")?,
            500,
            "2026-01-01T00:00:00Z",
        )?]);
        let pins = SweepPins::default().with_model(&model("pinned")?);

        let plan = plan(&survey, 100, &pins);

        assert!(plan.evictions.is_empty());
        assert_eq!(plan.freed_bytes, 0);
        assert_eq!(plan.still_over_by_bytes, Some(400));
        Ok(())
    }

    #[test]
    fn apply_sweep_removes_exactly_the_planned_paths_and_reports_each() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let models = tmp.path().join("models");
        store_model(&models, &model("stale")?, "2026-01-01T00:00:00Z")?;
        store_model(&models, &model("fresh")?, "2026-06-01T00:00:00Z")?;
        let survey = survey(&models, &tmp.path().join("runtimes"));
        let quota = survey.used_bytes - 1;

        let plan = plan(&survey, quota, &SweepPins::default());
        let report = apply_sweep(&plan);

        assert_eq!(labels(&report.removed), vec![model("stale")?.to_string()]);
        assert!(report.failed.is_empty());
        assert_eq!(report.freed_bytes, plan.freed_bytes);
        assert!(!models
            .join(ModelStorageKey::of(&model("stale")?)?.relative_dir())
            .exists());
        assert!(models
            .join(ModelStorageKey::of(&model("fresh")?)?.relative_dir())
            .exists());
        Ok(())
    }

    /// Unix-only because the *fixture* is, not the behaviour: an un-removable
    /// path is not constructible on Windows with std alone.
    ///
    /// Every lever that looks like it should work there is one Rust deliberately
    /// neutralizes to make Windows behave like Unix:
    ///
    /// - Nesting the target under a *file* fails `ENOTDIR` here, but Windows
    ///   reports that shape as plain `NotFound`, which [`remove_entry`] treats as
    ///   success — already gone is the outcome it wanted.
    /// - An open handle does not block it: std opens with `FILE_SHARE_DELETE`.
    /// - Nor does the read-only attribute: std removes with
    ///   `FILE_DISPOSITION_IGNORE_READONLY_ATTRIBUTE`.
    ///
    /// What is left is a DENY-DELETE ACL, which means a native dependency for one
    /// fixture. `apply_sweep`'s failure branch is plain platform-independent
    /// arithmetic over `remove_entry`'s `Err`, and the linux and macos legs cover
    /// it, so the coverage lost here is the fixture's reach, not the branch.
    #[cfg(unix)]
    #[test]
    fn apply_sweep_reports_a_removal_it_could_not_perform() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        // A path whose parent is a file: the unlink cannot succeed.
        fs::write(tmp.path().join("nested"), b"blocker")?;
        let plan = SweepPlan {
            evictions: vec![StorageEntry {
                path: tmp.path().join("nested").join("entry").join("child"),
                label: "blocked".to_owned(),
                size_bytes: 10,
                kind: EntryKind::Garbage {
                    reason: "no manifest".to_owned(),
                },
            }],
            freed_bytes: 10,
            still_over_by_bytes: None,
        };

        let report = apply_sweep(&plan);

        assert!(report.removed.is_empty());
        assert_eq!(report.freed_bytes, 0);
        assert_eq!(report.failed.len(), 1);
        Ok(())
    }

    #[test]
    fn a_cache_hit_changes_the_sweep_order() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let models = tmp.path().join("models");
        store_model(&models, &model("stale")?, "2026-01-01T00:00:00Z")?;
        store_model(&models, &model("fresh")?, "2026-06-01T00:00:00Z")?;
        let runtimes = tmp.path().join("runtimes");
        assert_eq!(
            labels(&survey(&models, &runtimes).entries),
            vec![model("stale")?.to_string(), model("fresh")?.to_string()]
        );

        // Resolving the stale model moves it to the back of the queue.
        let store = ModelArtifactStore::new(models.clone());
        let refetched = Cell::new(false);
        store.ensure(&model("stale")?, |_d, _into| {
            refetched.set(true);
            Ok(())
        })?;
        assert!(!refetched.get(), "a cache hit does not re-fetch");

        assert_eq!(
            labels(&survey(&models, &runtimes).entries),
            vec![model("fresh")?.to_string(), model("stale")?.to_string()]
        );
        Ok(())
    }

    #[test]
    fn refuse_if_oversize_names_both_numbers() -> anyhow::Result<()> {
        let err = refuse_if_oversize("meta/llama", Some(300), 200)
            .err()
            .ok_or_else(|| anyhow::anyhow!("an artifact over the whole quota must be refused"))?;
        let message = err.to_string();
        assert!(
            message.contains("300") && message.contains("200"),
            "{message}"
        );
        Ok(())
    }

    #[rstest]
    #[case::fits(Some(199))]
    #[case::exactly_the_quota(Some(200))]
    #[case::unknown_size(None)]
    fn refuse_if_oversize_allows_what_it_cannot_refuse(#[case] declared_bytes: Option<u64>) {
        assert!(refuse_if_oversize("meta/llama", declared_bytes, 200).is_ok());
    }
}
