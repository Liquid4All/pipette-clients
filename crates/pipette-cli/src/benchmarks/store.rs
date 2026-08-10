//! [`BenchmarkStore`] — filesystem capability handle under `benchmarks/`.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::Context;
use serde_json::Value;

use pipette_mgmt_client::types::BenchmarkSummary;
use pipette_ops::fs::{read_json_file, write_json_file};
use pipette_plan_types::benchmark::{BenchmarkDefinition, BenchmarkSource};
use pipette_plan_types::BenchmarkId;

use super::{RemoteSyncState, SourcedBenchmarkId};
use crate::error::{Error, Result};

/// Capability handle for the benchmark catalog (`root/` = workspace `benchmarks/`).
///
/// Layout (private): `local/<id>.json`, `remote/{index,sync,<id>}.json`.
/// Creates directories on write; reads OK if the tree is missing.
///
/// Minted by the workspace (`ws.benchmarks()`).
#[derive(Debug, Clone)]
pub struct BenchmarkStore {
    root: PathBuf,
}

impl BenchmarkStore {
    /// `root` = workspace `benchmarks/` directory.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    fn source_dir(&self, source: BenchmarkSource) -> PathBuf {
        let name = match source {
            BenchmarkSource::Local => "local",
            BenchmarkSource::Remote => "remote",
        };
        self.root.join(name)
    }

    fn entry_json(&self, source: BenchmarkSource, name: &str) -> PathBuf {
        self.source_dir(source).join(format!("{name}.json"))
    }

    /// Load by qualified [`SourcedBenchmarkId`] (`local/` or `remote/`).
    ///
    /// Returns `None` if missing. Source is [`SourcedBenchmarkId::source`], not returned.
    pub fn get(&self, reference: &SourcedBenchmarkId) -> Result<Option<BenchmarkDefinition>> {
        read_def_file(&self.entry_json(reference.source(), reference.id().as_ref()))
    }

    /// Create or replace a definition on the local or remote half of the catalog.
    ///
    /// Remote side expects a domain [`BenchmarkDefinition`] (convert wire
    /// `RemoteBenchmark` at the boundary first).
    pub fn put(&self, source: BenchmarkSource, def: &BenchmarkDefinition) -> Result<()> {
        write_json_file(&self.entry_json(source, def.benchmark_id()), def).map_err(Into::into)
    }

    /// List catalog entries for `source` as `(id, definition)`, sorted by id.
    ///
    /// Remote lists **detail** files only (`index.json` / `sync.json` skipped).
    pub fn list(&self, source: BenchmarkSource) -> Result<Vec<(BenchmarkId, BenchmarkDefinition)>> {
        let dir = self.source_dir(source);
        if !dir.exists() {
            return Ok(Vec::new());
        }
        let mut out: Vec<(BenchmarkId, BenchmarkDefinition)> = fs::read_dir(&dir)
            .with_context(|| format!("failed to read {}", dir.display()))
            .map_err(Error::Other)?
            .map(|e| e.map(|e| e.path()))
            .collect::<std::result::Result<Vec<_>, _>>()
            .with_context(|| format!("failed to read {}", dir.display()))
            .map_err(Error::Other)?
            .into_iter()
            .filter(|path| path.extension().and_then(|e| e.to_str()) == Some("json"))
            .filter(|path| {
                !matches!(source, BenchmarkSource::Remote)
                    || path
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .is_some_and(|stem| stem != "index" && stem != "sync")
            })
            .filter_map(|path| match read_def_file(&path) {
                Ok(Some(def)) => match BenchmarkId::try_new(def.benchmark_id().to_string()) {
                    Ok(id) => Some((id, def)),
                    Err(e) => {
                        log::warn!("skipping catalog file {}: invalid id ({e})", path.display());
                        None
                    }
                },
                Ok(None) => None,
                Err(e) => {
                    log::warn!("skipping corrupt catalog file {}: {e}", path.display());
                    None
                }
            })
            .collect();
        out.sort_by(|a, b| a.0.as_ref().cmp(b.0.as_ref()));
        Ok(out)
    }

    /// Remote index rows (sync metadata only).
    pub(crate) fn list_remote_index(&self) -> Result<Vec<BenchmarkSummary>> {
        let path = self.entry_json(BenchmarkSource::Remote, "index");
        if !path.exists() {
            return Ok(Vec::new());
        }
        read_json_file(&path).map_err(Into::into)
    }

    /// Replace the remote index.
    pub(crate) fn put_remote_index(&self, summaries: &[BenchmarkSummary]) -> Result<()> {
        write_json_file(
            &self.entry_json(BenchmarkSource::Remote, "index"),
            &summaries,
        )
        .map_err(Into::into)
    }

    /// Remote detail present? (conditional sync / etags).
    pub(crate) fn has_remote_detail(&self, id: &BenchmarkId) -> bool {
        self.entry_json(BenchmarkSource::Remote, id.as_ref())
            .exists()
    }

    /// Pull sync metadata (`sync.json`).
    pub(crate) fn get_sync_state(&self) -> Result<Option<RemoteSyncState>> {
        let path = self.entry_json(BenchmarkSource::Remote, "sync");
        if !path.exists() {
            return Ok(None);
        }
        Ok(Some(read_json_file(&path)?))
    }

    /// Write pull sync metadata.
    pub(crate) fn put_sync_state(&self, state: &RemoteSyncState) -> Result<()> {
        write_json_file(&self.entry_json(BenchmarkSource::Remote, "sync"), state)
            .map_err(Into::into)
    }

    /// Delete remote detail files for benchmarks no longer in the catalog.
    ///
    /// The index and the per-id ETag map both self-prune, since each sync rewrites them wholesale;
    /// the detail files do not, so without this a benchmark dropped server-side leaks its
    /// `{id}.json` forever and [`Self::list`] keeps returning it as a runnable remote benchmark.
    ///
    /// `keeping` must be the ids from the *list* response, not the post-detail-fetch kept set.
    /// A benchmark whose detail merely failed or 304'd-without-a-cache-entry this round is still in
    /// the catalog, and deleting its cached detail would force a re-fetch on every subsequent sync.
    /// Mirrors iOS `BenchmarkSync.pruneDetails(keeping:)`, which prunes at the same point.
    pub(crate) fn prune_remote_details(&self, keeping: &[String]) -> Result<()> {
        let dir = self.source_dir(BenchmarkSource::Remote);
        if !dir.exists() {
            return Ok(());
        }
        let entries = fs::read_dir(&dir)
            .with_context(|| format!("failed to read {}", dir.display()))
            .map_err(Error::Other)?;
        for entry in entries {
            let path = entry
                .with_context(|| format!("failed to read {}", dir.display()))
                .map_err(Error::Other)?
                .path();
            if path.extension().and_then(|e| e.to_str()) != Some("json") {
                continue;
            }
            let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
                continue;
            };
            // `index` and `sync` are the store's own bookkeeping, not benchmarks; the same two names
            // `list` skips. Removing them here would destroy the ETag state this prune supports.
            if stem == "index" || stem == "sync" || keeping.iter().any(|id| id == stem) {
                continue;
            }
            log::debug!("pruning remote benchmark detail `{stem}`: no longer in the catalog");
            fs::remove_file(&path)
                .with_context(|| format!("removing {}", path.display()))
                .map_err(Error::Other)?;
        }
        Ok(())
    }

    /// Drop remote index, details, and sync state.
    pub fn clear_remote(&self) -> Result<()> {
        let dir = self.source_dir(BenchmarkSource::Remote);
        if dir.exists() {
            fs::remove_dir_all(&dir)
                .with_context(|| format!("removing {}", dir.display()))
                .map_err(Error::Other)?;
        }
        Ok(())
    }
}

fn read_def_file(path: &Path) -> Result<Option<BenchmarkDefinition>> {
    if !path.exists() {
        return Ok(None);
    }
    let document: Value = read_json_file(path)?;
    if document.get("parameters").is_some() {
        return Err(Error::Other(anyhow::anyhow!(
            "benchmark file {} uses unsupported nested parameters object",
            path.display()
        )));
    }
    let def = serde_json::from_value(document)
        .with_context(|| format!("failed to parse benchmark document in {}", path.display()))
        .map_err(Error::Other)?;
    Ok(Some(def))
}

#[cfg(test)]
#[allow(clippy::panic)]
mod tests {
    use std::collections::BTreeMap;

    use pipette_plan_types::benchmark::BenchmarkSource;
    use pipette_plan_types::BenchmarkType;

    use super::*;
    use crate::benchmarks::SourcedBenchmarkId;

    fn temp_store(prefix: &str) -> (PathBuf, BenchmarkStore) {
        let root = std::env::temp_dir().join(format!(
            "pipette-benchmarks-{prefix}-{}-{}",
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        let _ = std::fs::remove_dir_all(&root);
        (root.clone(), BenchmarkStore::new(root))
    }

    /// `seed_standard_local` writes local ladder entries; get finds them under Local.
    #[test]
    fn seed_standard_then_get_local() -> anyhow::Result<()> {
        let (root, store) = temp_store("init-resolve");
        let summary =
            crate::benchmarks::seed_standard_local(&store, &[BenchmarkType::PrefillThroughput])?;
        assert!(summary.created > 0);
        assert_eq!(summary.updated, 0);

        let local = SourcedBenchmarkId::new(
            BenchmarkSource::Local,
            BenchmarkId::try_new("prefill_throughput_100".to_string())
                .map_err(|e| anyhow::anyhow!(e))?,
        );
        let def = match store.get(&local)? {
            Some(d) => d,
            None => panic!("seeded local def missing"),
        };
        assert_eq!(def.benchmark_id(), "prefill_throughput_100");
        assert!(store
            .list(BenchmarkSource::Local)?
            .iter()
            .any(|(_, b)| b.benchmark_id() == def.benchmark_id()));

        // second init updates rather than double-creating
        let again =
            crate::benchmarks::seed_standard_local(&store, &[BenchmarkType::PrefillThroughput])?;
        assert_eq!(again.created, 0);
        assert!(again.updated > 0);

        let _ = std::fs::remove_dir_all(&root);
        Ok(())
    }

    /// `clear_remote` removes only the remote tree; local defs stay.
    #[test]
    fn clear_remote_keeps_local() -> anyhow::Result<()> {
        let (root, store) = temp_store("clear-remote");
        crate::benchmarks::seed_standard_local(&store, &[BenchmarkType::PrefillThroughput])?;
        store.put_remote_index(&[])?;
        assert!(root.join("remote").exists());

        store.clear_remote()?;
        assert!(!root.join("remote").exists());
        assert!(!store.list(BenchmarkSource::Local)?.is_empty());

        let _ = std::fs::remove_dir_all(&root);
        Ok(())
    }

    fn remote_def(id: &str) -> anyhow::Result<BenchmarkDefinition> {
        Ok(serde_json::from_value(serde_json::json!({
            "benchmark_id": id,
            "benchmark_type": "prefill_throughput",
            "parameter_prefill_tokens": 128,
        }))?)
    }

    /// A benchmark that left the server catalog must lose its cached detail file (PIP-248).
    #[test]
    fn prune_remote_details_drops_only_departed_benchmarks() -> anyhow::Result<()> {
        let (root, store) = temp_store("prune-remote");
        store.put(
            BenchmarkSource::Remote,
            &remote_def("prefill_throughput_1")?,
        )?;
        store.put(
            BenchmarkSource::Remote,
            &remote_def("prefill_throughput_2")?,
        )?;
        store.put_remote_index(&[])?;
        store.put_sync_state(&RemoteSyncState {
            pulled_at: "2026-01-01T00:00:00Z".to_string(),
            benchmark_count: 2,
            benchmarks_etag: None,
            benchmark_etags: BTreeMap::new(),
        })?;

        store.prune_remote_details(&["prefill_throughput_1".to_string()])?;

        assert!(root.join("remote/prefill_throughput_1.json").exists());
        // The whole point: without this, a dropped benchmark stays runnable forever.
        assert!(!root.join("remote/prefill_throughput_2.json").exists());
        // The store's own bookkeeping must survive; deleting sync.json would destroy the ETag
        // state that makes the conditional pull conditional.
        assert!(root.join("remote/index.json").exists());
        assert!(root.join("remote/sync.json").exists());

        let _ = std::fs::remove_dir_all(&root);
        Ok(())
    }

    /// Pruning against an empty catalog clears every detail but still spares index/sync, and
    /// pruning a store that has never synced is a no-op rather than an error.
    #[test]
    fn prune_remote_details_handles_empty_and_missing() -> anyhow::Result<()> {
        let (root, store) = temp_store("prune-empty");
        store.prune_remote_details(&["anything".to_string()])?; // no remote dir yet

        store.put(
            BenchmarkSource::Remote,
            &remote_def("prefill_throughput_1")?,
        )?;
        store.put_remote_index(&[])?;
        store.prune_remote_details(&[])?;

        assert!(!root.join("remote/prefill_throughput_1.json").exists());
        assert!(root.join("remote/index.json").exists());

        let _ = std::fs::remove_dir_all(&root);
        Ok(())
    }
}
