//! [`BenchmarkStore`]: the benchmark catalog on disk under `benchmarks/`.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::Context;
use serde_json::Value;

use pipette_mgmt_client::types::{BenchmarkSummary, EntityTag};
use pipette_ops::fs::{read_json_file, write_json_file};
use pipette_ops::vl_spec_decode::text_sha256;
use pipette_plan_types::{BenchmarkDefinition, BenchmarkId, BenchmarkSource};

use super::{ConvertedRemote, RemoteSyncState, SourcedBenchmarkId};
use crate::error::{Error, Result};

/// Layout (private; why the subdirectories: `remote_dir`):
///
/// ```text
/// local/<id>.json          definitions this build generated
/// remote/detail/<id>.json  each remote definition as the server sent it
/// remote/detail/<id>.etag  the sha256 of that file, then the ETag the server sent with it
/// remote/state/index.json  the GET /benchmarks list as the server sent it, every row
/// remote/state/sync.json   RemoteSyncState
/// remote/*.json            written by builds before this layout: never read nor deleted
/// ```
///
/// Creates directories on write; reads succeed if the tree is missing.
#[derive(Debug, Clone)]
pub struct BenchmarkStore {
    root: PathBuf,
}

impl BenchmarkStore {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// Builds before this layout keep `*.json` files in `remote/`: their definitions,
    /// `index.json` and `sync.json`. Nothing here reads those: a definition may have lost a
    /// parameter its build did not model (public 2026.08.1 has no `parameter_num_images`), and
    /// an index holds only the rows its build could convert, so a `304` on its list ETag would
    /// hand this build that smaller list. Those builds touch only `remote/*.json`, never
    /// `detail/` or `state/`, and their files are left alone, since deleting one could leave
    /// such a build a list ETag without its list.
    fn remote_dir(&self) -> PathBuf {
        self.root.join("remote")
    }

    fn state_dir(&self) -> PathBuf {
        self.remote_dir().join("state")
    }

    fn index_json(&self) -> PathBuf {
        self.state_dir().join("index.json")
    }

    fn sync_json(&self) -> PathBuf {
        self.state_dir().join("sync.json")
    }

    fn definitions_dir(&self, source: BenchmarkSource) -> PathBuf {
        match source {
            BenchmarkSource::Local => self.root.join("local"),
            BenchmarkSource::Remote => self.remote_dir().join("detail"),
        }
    }

    fn definition_json(&self, source: BenchmarkSource, id: &BenchmarkId) -> PathBuf {
        self.definitions_dir(source).join(format!("{id}.json"))
    }

    fn detail_etag(&self, id: &BenchmarkId) -> PathBuf {
        self.definitions_dir(BenchmarkSource::Remote)
            .join(format!("{id}.etag"))
    }

    /// A remote id is read from `remote/detail/` only, so one an older build cached is `None`.
    pub fn get(&self, reference: &SourcedBenchmarkId) -> Result<Option<BenchmarkDefinition>> {
        read_def_file(
            &self.definition_json(reference.source(), reference.id()),
            reference.source(),
        )
    }

    /// A local definition is one this build generated, so its type is the whole of it.
    pub fn put_local(&self, def: &BenchmarkDefinition) -> Result<()> {
        write_json_file(
            &self.definition_json(BenchmarkSource::Local, def.benchmark_id()),
            def,
        )
        .map_err(Into::into)
    }

    /// The `.etag` file is `<sha256 of the body>\n<etag>`, and [`Self::get_remote`] honors it
    /// only for the body it hashes: a crash, or a pull and a claim caching the same id at once,
    /// can leave the two files from different writes, which then costs a full fetch, never a
    /// `304` for bytes that are not on disk.
    pub(crate) fn put_remote(
        &self,
        remote: &ConvertedRemote,
        etag: Option<&EntityTag>,
    ) -> Result<()> {
        let id = remote.definition().benchmark_id();
        let path = self.definition_json(BenchmarkSource::Remote, id);
        let etag_path = self.detail_etag(id);
        let body = serde_json::to_string_pretty(remote.sent())
            .context("failed to serialize the remote benchmark")?;
        remove_if_present(&etag_path)?;
        fs::create_dir_all(self.definitions_dir(BenchmarkSource::Remote))
            .and_then(|()| fs::write(&path, &body))
            .with_context(|| format!("failed to write {}", path.display()))?;
        if let Some(etag) = etag {
            let recorded = format!("{}\n{}", text_sha256(&body)?, etag.as_str());
            fs::write(&etag_path, recorded)
                .with_context(|| format!("failed to write {}", etag_path.display()))?;
        }
        Ok(())
    }

    /// One read of the body serves the definition and the digest check. The ETag is `None` when
    /// its file is missing, unreadable or records another body, so the next fetch is
    /// unconditional.
    pub(crate) fn get_remote(
        &self,
        id: &BenchmarkId,
    ) -> Result<Option<(BenchmarkDefinition, Option<EntityTag>)>> {
        let path = self.definition_json(BenchmarkSource::Remote, id);
        let Some(body) = read_if_present(&path)? else {
            return Ok(None);
        };
        let def = parse_def(&body, &path, BenchmarkSource::Remote)?;
        let etag = fs::read_to_string(self.detail_etag(id))
            .ok()
            .and_then(|recorded| {
                let (digest, etag) = recorded.split_once('\n')?;
                let current = text_sha256(&body).ok()?;
                (digest == current.as_ref())
                    .then(|| EntityTag::try_new(etag.to_owned()).ok())
                    .flatten()
            });
        Ok(Some((def, etag)))
    }

    /// Sorted by id. A file this build cannot read (a newer build's type, say) is skipped with a
    /// warning.
    pub fn list(&self, source: BenchmarkSource) -> Result<Vec<(BenchmarkId, BenchmarkDefinition)>> {
        let mut out: Vec<(BenchmarkId, BenchmarkDefinition)> =
            json_files(&self.definitions_dir(source))?
                .into_iter()
                .filter_map(|(_, path)| match read_def_file(&path, source) {
                    Ok(Some(def)) => match BenchmarkId::try_new(def.benchmark_id().to_string()) {
                        Ok(id) => Some((id, def)),
                        Err(e) => {
                            tracing::warn!(
                                path = %path.display(),
                                error = %e,
                                "skipping catalog file: invalid id"
                            );
                            None
                        }
                    },
                    Ok(None) => None,
                    Err(e) => {
                        tracing::warn!(
                            path = %path.display(),
                            error = %e,
                            "skipping corrupt catalog file"
                        );
                        None
                    }
                })
                .collect();
        out.sort_by(|a, b| a.0.as_ref().cmp(b.0.as_ref()));
        Ok(out)
    }

    /// Rows of a type this build does not know included: a reader filters them.
    pub(crate) fn list_remote_index(&self) -> Result<Vec<BenchmarkSummary>> {
        let path = self.index_json();
        if !path.exists() {
            return Ok(Vec::new());
        }
        read_json_file(&path).map_err(Into::into)
    }

    /// Every row the server sent, so the list ETag in the sync state validates exactly what is
    /// stored.
    pub(crate) fn put_remote_index(&self, rows: &[BenchmarkSummary]) -> Result<()> {
        write_json_file(&self.index_json(), &rows).map_err(Into::into)
    }

    /// Looks in `remote/detail/` only, so a file an older build cached never counts as one.
    pub(crate) fn has_remote_detail(&self, id: &BenchmarkId) -> bool {
        self.definition_json(BenchmarkSource::Remote, id).exists()
    }

    /// `None` also when the cached detail does not read.
    pub(crate) fn remote_detail_etag(&self, id: &BenchmarkId) -> Option<EntityTag> {
        self.get_remote(id).ok().flatten()?.1
    }

    pub(crate) fn get_sync_state(&self) -> Result<Option<RemoteSyncState>> {
        let path = self.sync_json();
        if !path.exists() {
            return Ok(None);
        }
        Ok(Some(read_json_file(&path)?))
    }

    pub(crate) fn put_sync_state(&self, state: &RemoteSyncState) -> Result<()> {
        write_json_file(&self.sync_json(), state).map_err(Into::into)
    }

    /// Delete the remote detail files of benchmarks the list no longer names. Each sync rewrites
    /// the list but not the details, so without this a benchmark dropped server-side stays
    /// listed as runnable.
    ///
    /// `keeping` must be every id of the *list*, rows this build does not fetch included: a
    /// detail that failed this round is still in the catalog, and deleting it would force a
    /// re-fetch on every sync; a detail a newer build cached for a type this build cannot read is
    /// that build's to keep. iOS `BenchmarkStore.pruneRemoteDetails(keeping:)` prunes at the same
    /// point.
    pub(crate) fn prune_remote_details(&self, keeping: &[String]) -> Result<()> {
        json_files(&self.definitions_dir(BenchmarkSource::Remote))?
            .into_iter()
            .filter(|(stem, _)| !keeping.contains(stem))
            .try_for_each(|(stem, path)| {
                tracing::debug!(
                    benchmark_id = %stem,
                    "pruning remote benchmark detail: no longer in the catalog"
                );
                remove_if_present(&path.with_extension("etag"))?;
                remove_if_present(&path)
            })
    }

    /// Older builds' files under `remote/` go too.
    pub fn clear_remote(&self) -> Result<()> {
        let dir = self.remote_dir();
        if dir.exists() {
            fs::remove_dir_all(&dir)
                .with_context(|| format!("removing {}", dir.display()))
                .map_err(Error::Other)?;
        }
        Ok(())
    }
}

fn json_files(dir: &Path) -> Result<Vec<(String, PathBuf)>> {
    if !dir.exists() {
        return Ok(Vec::new());
    }
    let paths = fs::read_dir(dir)
        .and_then(|entries| {
            entries
                .map(|entry| entry.map(|e| e.path()))
                .collect::<std::io::Result<Vec<_>>>()
        })
        .with_context(|| format!("failed to read {}", dir.display()))
        .map_err(Error::Other)?;
    Ok(paths
        .into_iter()
        .filter(|path| path.extension().and_then(|e| e.to_str()) == Some("json"))
        .filter_map(|path| Some((path.file_stem()?.to_str()?.to_owned(), path)))
        .collect())
}

/// Already gone is `Ok`: never written, or a concurrent claim or sync got there first.
fn remove_if_present(path: &Path) -> Result<()> {
    match fs::remove_file(path) {
        Err(source) if source.kind() != std::io::ErrorKind::NotFound => Err(Error::RemoveFile {
            path: path.to_owned(),
            source,
        }),
        _ => Ok(()),
    }
}

/// A local file with a top-level `parameters` object is the old nested format, refused. A
/// remote file ignores it, as on fetch; refusing would make every claim refetch it.
fn read_def_file(path: &Path, source: BenchmarkSource) -> Result<Option<BenchmarkDefinition>> {
    read_if_present(path)?
        .map(|text| parse_def(&text, path, source))
        .transpose()
}

fn read_if_present(path: &Path) -> Result<Option<String>> {
    if !path.exists() {
        return Ok(None);
    }
    fs::read_to_string(path)
        .map(Some)
        .with_context(|| format!("failed to read {}", path.display()))
        .map_err(Error::Other)
}

fn parse_def(text: &str, path: &Path, source: BenchmarkSource) -> Result<BenchmarkDefinition> {
    let document: Value = serde_json::from_str(text)
        .with_context(|| format!("failed to parse {}", path.display()))
        .map_err(Error::Other)?;
    if source == BenchmarkSource::Local && document.get("parameters").is_some() {
        return Err(Error::Other(anyhow::anyhow!(
            "benchmark file {} uses unsupported nested parameters object",
            path.display()
        )));
    }
    serde_json::from_value(document)
        .with_context(|| format!("failed to parse benchmark document in {}", path.display()))
        .map_err(Error::Other)
}

#[cfg(test)]
mod tests {
    use pipette_plan_types::{BenchmarkSource, BenchmarkType};

    use super::*;
    use crate::benchmarks::SourcedBenchmarkId;
    use crate::test_support::{
        IMG80, cache_remote, cache_remote_with_etag, corpus_wire, img80_wire, remote_detail_path,
        remote_state_path, write_older_build_bookkeeping, write_public_2026_08_1_img80_cache,
    };

    fn temp_store(prefix: &str) -> (PathBuf, BenchmarkStore) {
        let root = std::env::temp_dir().join(format!(
            "pipette-benchmarks-{prefix}-{}-{}",
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        (root.clone(), BenchmarkStore::new(root))
    }

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
        let Some(def) = store.get(&local)? else {
            anyhow::bail!("seeded local def missing");
        };
        assert_eq!(def.benchmark_id().as_str(), "prefill_throughput_100");
        assert!(
            store
                .list(BenchmarkSource::Local)?
                .iter()
                .any(|(_, b)| b.benchmark_id() == def.benchmark_id())
        );

        let again =
            crate::benchmarks::seed_standard_local(&store, &[BenchmarkType::PrefillThroughput])?;
        assert_eq!(again.created, 0);
        assert!(again.updated > 0);

        std::fs::remove_dir_all(&root)?;
        Ok(())
    }

    /// Older builds' files under `remote/` go too.
    #[test]
    fn clear_remote_keeps_local() -> anyhow::Result<()> {
        let (root, store) = temp_store("clear-remote");
        crate::benchmarks::seed_standard_local(&store, &[BenchmarkType::PrefillThroughput])?;
        store.put_remote_index(&[])?;
        store.put_sync_state(&sync_state())?;
        write_older_build_bookkeeping(&root, "https://mgmt.example.com", &["b1"])?;
        cache_prefill(&store, "prefill_throughput_1")?;
        assert!(remote_detail_path(&root, "prefill_throughput_1").exists());
        assert!(remote_state_path(&root, "sync").exists());

        store.clear_remote()?;
        assert!(!root.join("remote").exists());
        assert!(!store.list(BenchmarkSource::Local)?.is_empty());

        std::fs::remove_dir_all(&root)?;
        Ok(())
    }

    fn sync_state() -> RemoteSyncState {
        RemoteSyncState {
            pulled_at: "2026-01-01T00:00:00Z".to_string(),
            benchmark_count: 2,
            server_url: None,
            benchmarks_etag: None,
        }
    }

    fn cache_prefill(store: &BenchmarkStore, id: &str) -> anyhow::Result<()> {
        cache_remote(
            store,
            serde_json::json!({
                "benchmark_id": id,
                "benchmark_type": "prefill_throughput",
                "parameter_prefill_tokens": 128,
            }),
        )
    }

    fn remote_ref(id: &str) -> anyhow::Result<SourcedBenchmarkId> {
        Ok(SourcedBenchmarkId::new(
            BenchmarkSource::Remote,
            BenchmarkId::try_new(id.to_owned())?,
        ))
    }

    fn num_images(def: &BenchmarkDefinition) -> anyhow::Result<i32> {
        let BenchmarkDefinition::VlThroughput(vl) = def else {
            anyhow::bail!("expected vl_throughput, got {def:?}");
        };
        Ok(vl.parameter_num_images.into_inner())
    }

    fn img80_with_unmodelled_keys() -> Value {
        let mut sent = img80_wire();
        sent["parameter_temperature"] = serde_json::json!(0.7);
        sent["parameter_from_the_future"] = serde_json::json!(3);
        sent["org_id"] = serde_json::json!("org_test");
        sent
    }

    #[rstest::rstest]
    #[case::unmodelled_keys(img80_with_unmodelled_keys(), Some(80))]
    #[case::eval_samples(serde_json::json!({
        "benchmark_id": "eval_mmlu",
        "benchmark_type": "eval",
        "parameter_eval_id": "mmlu_pro",
        "parameter_dataset_name": "edge_2026.03.1",
        "parameter_max_tokens": 64,
        "samples": [{"id": "q1", "messages": []}],
    }), None)]
    #[case::corpus_dataset(corpus_wire("vl_spec_decode_mmspec_test_gqa_k8_bs8"), None)]
    fn a_remote_detail_keeps_what_the_server_sent(
        #[case] sent: Value,
        #[case] images: Option<i32>,
    ) -> anyhow::Result<()> {
        let (root, store) = temp_store("detail-verbatim");
        let id = sent["benchmark_id"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("the case has no benchmark_id"))?
            .to_owned();

        cache_remote(&store, sent.clone())?;

        let on_disk: Value = read_json_file(&remote_detail_path(&root, &id))?;
        assert_eq!(on_disk, sent);
        let def = store
            .get(&remote_ref(&id)?)?
            .ok_or_else(|| anyhow::anyhow!("the stored definition does not read back"))?;
        if let Some(images) = images {
            assert_eq!(num_images(&def)?, images);
        }
        std::fs::remove_dir_all(&root)?;
        Ok(())
    }

    #[test]
    fn what_an_older_build_wrote_is_never_read() -> anyhow::Result<()> {
        let (root, store) = temp_store("legacy-unread");
        write_public_2026_08_1_img80_cache(&root)?;
        write_older_build_bookkeeping(&root, "https://mgmt.example.com", &[IMG80])?;
        let id = BenchmarkId::try_new(IMG80.to_owned())?;

        assert!(store.get(&remote_ref(IMG80)?)?.is_none());
        assert!(store.list(BenchmarkSource::Remote)?.is_empty());
        assert!(!store.has_remote_detail(&id));
        assert!(store.list_remote_index()?.is_empty());
        assert!(store.get_sync_state()?.is_none());
        std::fs::remove_dir_all(&root)?;
        Ok(())
    }

    /// That build may still share the workspace.
    #[test]
    fn putting_a_remote_detail_leaves_its_legacy_file() -> anyhow::Result<()> {
        let (root, store) = temp_store("legacy-kept");
        let legacy = write_public_2026_08_1_img80_cache(&root)?;

        cache_remote(&store, img80_wire())?;

        assert!(legacy.exists(), "the older build's file went");
        assert!(remote_detail_path(&root, IMG80).exists());
        std::fs::remove_dir_all(&root)?;
        Ok(())
    }

    #[rstest::rstest]
    #[case::as_written(["index", "sync", "state"])]
    #[case::in_another_case(["Index", "SYNC", "State"])]
    fn caching_a_benchmark_named_like_the_bookkeeping_keeps_the_bookkeeping(
        #[case] names: [&str; 3],
    ) -> anyhow::Result<()> {
        let (root, store) = temp_store("bookkeeping-names");
        store.put_remote_index(&[serde_json::from_str(
            r#"{"benchmark_id":"index","benchmark_type":"prefill_throughput"}"#,
        )?])?;
        store.put_sync_state(&sync_state())?;
        let older = write_older_build_bookkeeping(&root, "https://mgmt.example.com", &["b1"])?;
        let older_state = root.join(format!("remote/{}.json", names[2]));
        std::fs::write(&older_state, "{}")?;

        names
            .into_iter()
            .try_for_each(|id| cache_prefill(&store, id))?;

        assert_eq!(store.list_remote_index()?.len(), 1, "the list was lost");
        assert!(store.get_sync_state()?.is_some(), "the sync state was lost");
        assert!(older.index.exists(), "an older build's index went");
        assert!(older.sync.exists(), "an older build's sync state went");
        assert!(older_state.exists(), "the older build's `state` went");
        names.into_iter().for_each(|name| {
            assert!(remote_detail_path(&root, name).exists(), "{name}");
        });
        std::fs::remove_dir_all(&root)?;
        Ok(())
    }

    #[test]
    fn prune_remote_details_drops_only_departed_benchmarks() -> anyhow::Result<()> {
        let (root, store) = temp_store("prune-remote");
        store.put_remote_index(&[])?;
        store.put_sync_state(&sync_state())?;
        let older = write_older_build_bookkeeping(
            &root,
            "https://mgmt.example.com",
            &["prefill_throughput_1"],
        )?;
        [
            "prefill_throughput_1",
            "prefill_throughput_2",
            "index",
            "sync",
            "state",
        ]
        .into_iter()
        .try_for_each(|id| cache_prefill(&store, id))?;
        std::fs::write(
            root.join("remote/detail/prefill_throughput_2.etag"),
            "\"v1\"",
        )?;
        std::fs::write(root.join("remote/prefill_throughput_1.json"), "{}")?;
        std::fs::write(root.join("remote/prefill_throughput_3.json"), "{}")?;

        store.prune_remote_details(&[
            "prefill_throughput_1".to_string(),
            "index".to_string(),
            "state".to_string(),
        ])?;

        assert!(remote_detail_path(&root, "prefill_throughput_1").exists());
        assert!(!remote_detail_path(&root, "prefill_throughput_2").exists());
        assert!(
            !root
                .join("remote/detail/prefill_throughput_2.etag")
                .exists()
        );
        assert!(remote_detail_path(&root, "index").exists());
        assert!(remote_detail_path(&root, "state").exists());
        assert!(!remote_detail_path(&root, "sync").exists());
        assert!(root.join("remote/prefill_throughput_1.json").exists());
        assert!(root.join("remote/prefill_throughput_3.json").exists());
        assert!(older.index.exists());
        assert!(older.sync.exists());
        assert!(remote_state_path(&root, "index").exists());
        assert!(remote_state_path(&root, "sync").exists());

        std::fs::remove_dir_all(&root)?;
        Ok(())
    }

    #[test]
    fn prune_remote_details_handles_empty_and_missing() -> anyhow::Result<()> {
        let (root, store) = temp_store("prune-empty");
        store.prune_remote_details(&["anything".to_string()])?;
        store.put_remote_index(&[])?;
        store.prune_remote_details(&["anything".to_string()])?;

        cache_prefill(&store, "prefill_throughput_1")?;
        store.prune_remote_details(&[])?;

        assert!(!remote_detail_path(&root, "prefill_throughput_1").exists());
        assert!(remote_state_path(&root, "index").exists());

        std::fs::remove_dir_all(&root)?;
        Ok(())
    }

    #[test]
    fn a_detail_keeps_the_etag_it_was_cached_with() -> anyhow::Result<()> {
        let (root, store) = temp_store("detail-etag");
        let id = BenchmarkId::try_new("prefill_throughput_1".to_owned())?;
        let body = || {
            serde_json::json!({
                "benchmark_id": "prefill_throughput_1",
                "benchmark_type": "prefill_throughput",
                "parameter_prefill_tokens": 128,
            })
        };
        let etag = |store: &BenchmarkStore| {
            store
                .remote_detail_etag(&id)
                .map(|etag| etag.as_str().to_owned())
        };
        assert_eq!(etag(&store), None, "nothing cached");

        cache_remote_with_etag(&store, body(), Some("\"v1\""))?;
        assert_eq!(etag(&store).as_deref(), Some("\"v1\""));
        cache_remote_with_etag(&store, body(), Some("\"v2\""))?;
        assert_eq!(etag(&store).as_deref(), Some("\"v2\""));
        cache_remote(&store, body())?;
        assert_eq!(etag(&store), None, "a body sent without an ETag keeps none");

        cache_remote_with_etag(&store, body(), Some("\"v3\""))?;
        let detail = remote_detail_path(&root, id.as_str());
        let (cached, cached_etag) = store
            .get_remote(&id)?
            .ok_or_else(|| anyhow::anyhow!("the detail does not read back"))?;
        assert_eq!(cached.benchmark_id(), &id);
        assert_eq!(
            cached_etag.map(|e| e.as_str().to_owned()).as_deref(),
            Some("\"v3\"")
        );
        let mut other = body();
        other["parameter_prefill_tokens"] = serde_json::json!(256);
        std::fs::write(&detail, serde_json::to_string_pretty(&other)?)?;
        assert_eq!(etag(&store), None, "an ETag recorded for another body");
        std::fs::write(detail.with_extension("etag"), "\"v3\"")?;
        assert_eq!(etag(&store), None, "an ETag file without a digest");

        std::fs::remove_file(&detail)?;
        assert_eq!(etag(&store), None, "an ETag without its detail");
        std::fs::remove_dir_all(&root)?;
        Ok(())
    }

    #[test]
    fn list_skips_what_it_cannot_read_and_its_own_bookkeeping() -> anyhow::Result<()> {
        let (root, store) = temp_store("list-tolerant");
        assert!(store.list(BenchmarkSource::Remote)?.is_empty());

        store.put_remote_index(&[])?;
        store.put_sync_state(&sync_state())?;
        write_older_build_bookkeeping(&root, "https://mgmt.example.com", &["b1"])?;
        ["prefill_throughput_512", "sync", "state"]
            .into_iter()
            .try_for_each(|id| cache_prefill(&store, id))?;
        let dir = root.join("remote/detail");
        std::fs::write(dir.join("broken.json"), "{ not json")?;
        std::fs::write(
            dir.join("badid.json"),
            r#"{"benchmark_id":"prefill throughput","benchmark_type":"prefill_throughput","parameter_prefill_tokens":1}"#,
        )?;
        std::fs::write(
            dir.join("b_new.json"),
            r#"{"benchmark_id":"b_new","benchmark_type":"quantum_latency"}"#,
        )?;
        std::fs::write(dir.join("notes.txt"), "not a catalog entry")?;
        std::fs::write(
            root.join("remote/prefill_throughput_2.json"),
            r#"{"benchmark_id":"prefill_throughput_2","benchmark_type":"prefill_throughput","parameter_prefill_tokens":1}"#,
        )?;

        let ids: Vec<String> = store
            .list(BenchmarkSource::Remote)?
            .into_iter()
            .map(|(id, _)| id.to_string())
            .collect();
        assert_eq!(ids, ["prefill_throughput_512", "state", "sync"]);
        assert!(remote_state_path(&root, "index").exists());
        assert!(remote_state_path(&root, "sync").exists());
        std::fs::remove_dir_all(&root)?;
        Ok(())
    }

    #[test]
    fn a_nested_parameters_object_is_refused_only_in_a_local_file() -> anyhow::Result<()> {
        let (root, store) = temp_store("nested-parameters");
        let body = serde_json::json!({
            "benchmark_id": "prefill_throughput_1",
            "benchmark_type": "prefill_throughput",
            "parameter_prefill_tokens": 1,
            "parameters": {"prefill_tokens": 1},
        });
        let local = root.join("local/prefill_throughput_1.json");
        std::fs::create_dir_all(root.join("local"))?;
        std::fs::write(&local, body.to_string())?;
        cache_remote(&store, body)?;
        let id = BenchmarkId::try_new("prefill_throughput_1".to_owned())?;

        let refused = store
            .get(&SourcedBenchmarkId::new(BenchmarkSource::Local, id.clone()))
            .err()
            .ok_or_else(|| anyhow::anyhow!("a nested parameters object is refused"))?;
        assert!(
            refused.to_string().contains("nested parameters"),
            "{refused}"
        );
        assert!(store.list(BenchmarkSource::Local)?.is_empty());
        assert!(store.get(&remote_ref(id.as_str())?)?.is_some());
        assert_eq!(store.list(BenchmarkSource::Remote)?.len(), 1);
        std::fs::remove_dir_all(&root)?;
        Ok(())
    }
}
