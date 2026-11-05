//! Concrete [`ResultsStore`] under workspace `results/`.

use std::{fs, path::Path, path::PathBuf};

use anyhow::Context;
use serde::Serialize;
use serde_json::Value;

use pipette_ops::fs::{read_json_file, write_json_file};
use pipette_plan_types::{BenchmarkSource, BenchmarkType};

use super::{
    BenchmarkResultListEntry, BenchmarkResultLocation, BenchmarkResultState, BenchmarkScoredResult,
};
use crate::benchmarks::SourcedBenchmarkId;
use crate::error::Result;

/// ```text
/// results/
///   local/<id>/{payload,extras}.json
///   remote/pending/<id>/...
///   remote/synced/<id>/{payload,extras,metrics}.json
/// ```
#[derive(Debug, Clone)]
pub struct ResultsStore {
    root: PathBuf,
}

impl ResultsStore {
    /// `root` = workspace `results/` directory.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn results_dir(&self) -> &Path {
        &self.root
    }

    pub fn location_dir(&self, location: BenchmarkResultLocation) -> PathBuf {
        match location {
            BenchmarkResultLocation::Local => self.root.join("local"),
            BenchmarkResultLocation::RemotePending => self.root.join("remote").join("pending"),
            BenchmarkResultLocation::RemoteSynced => self.root.join("remote").join("synced"),
        }
    }

    pub fn result_dir(&self, location: BenchmarkResultLocation, id: &str) -> PathBuf {
        self.location_dir(location).join(id)
    }

    pub fn payload_path(&self, location: BenchmarkResultLocation, id: &str) -> PathBuf {
        self.result_dir(location, id).join("payload.json")
    }

    pub fn extras_path(&self, location: BenchmarkResultLocation, id: &str) -> PathBuf {
        self.result_dir(location, id).join("extras.json")
    }

    pub fn metrics_path(&self, location: BenchmarkResultLocation, id: &str) -> PathBuf {
        self.result_dir(location, id).join("metrics.json")
    }

    pub fn save_payload(
        &self,
        location: BenchmarkResultLocation,
        id: &str,
        payload: &impl Serialize,
    ) -> Result<()> {
        write_json_file(&self.payload_path(location, id), payload).map_err(Into::into)
    }

    pub fn save_extras(
        &self,
        location: BenchmarkResultLocation,
        id: &str,
        extras: &impl Serialize,
    ) -> Result<()> {
        write_json_file(&self.extras_path(location, id), extras).map_err(Into::into)
    }

    /// Write the payload and its extras in one call so no caller forgets the
    /// extras. Not atomic: a failure between the two leaves the payload alone.
    pub fn save_result(
        &self,
        location: BenchmarkResultLocation,
        id: &str,
        payload: &impl Serialize,
        extras: &impl Serialize,
    ) -> Result<()> {
        self.save_payload(location, id, payload)?;
        self.save_extras(location, id, extras)
    }

    pub fn save_metrics(
        &self,
        location: BenchmarkResultLocation,
        id: &str,
        scored: &impl Serialize,
    ) -> Result<()> {
        write_json_file(&self.metrics_path(location, id), scored).map_err(Into::into)
    }

    pub fn load_payload(&self, location: BenchmarkResultLocation, id: &str) -> Result<Value> {
        read_json_file(&self.payload_path(location, id)).map_err(Into::into)
    }

    /// Corrupt metrics read as missing so a later refresh can replace them.
    pub fn has_valid_metrics(&self, location: BenchmarkResultLocation, id: &str) -> bool {
        let metrics_path = self.metrics_path(location, id);
        if !metrics_path.exists() {
            return false;
        }
        match read_json_file::<BenchmarkScoredResult>(&metrics_path) {
            Ok(_) => true,
            Err(err) => {
                tracing::warn!(
                    result_id = %id,
                    error = %format_args!("{err:#}"),
                    "ignoring corrupt metrics"
                );
                false
            }
        }
    }

    pub fn list_ids(&self, location: BenchmarkResultLocation) -> Result<Vec<String>> {
        let dir = self.location_dir(location);
        if !dir.exists() {
            return Ok(Vec::new());
        }
        let mut ids: Vec<String> = fs::read_dir(&dir)
            .with_context(|| format!("failed to read {}", dir.display()))?
            .filter_map(|entry| {
                let entry = entry.ok()?;
                if entry.path().is_dir() {
                    entry.file_name().to_str().map(ToString::to_string)
                } else {
                    None
                }
            })
            .collect();
        ids.sort();
        Ok(ids)
    }

    pub fn load_list_entry(
        &self,
        location: BenchmarkResultLocation,
        id: &str,
    ) -> Result<Option<BenchmarkResultListEntry>> {
        let payload = self.load_payload(location, id)?;
        let benchmark_id = payload
            .get("benchmark_id")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let Some(benchmark_type) = recorded_benchmark_type(&payload, id) else {
            tracing::warn!(result_id = %id, "skipping result: unknown benchmark type");
            return Ok(None);
        };
        let has_valid_metrics = matches!(location, BenchmarkResultLocation::RemoteSynced)
            && self.has_valid_metrics(location, id);
        let state = match location {
            BenchmarkResultLocation::RemoteSynced if has_valid_metrics => {
                BenchmarkResultState::Scored
            }
            BenchmarkResultLocation::RemoteSynced => BenchmarkResultState::Submitted,
            BenchmarkResultLocation::Local | BenchmarkResultLocation::RemotePending => {
                BenchmarkResultState::Local
            }
        };
        let ref_id = benchmark_id.as_deref().unwrap_or(id).to_string();
        let source = match location {
            BenchmarkResultLocation::Local => BenchmarkSource::Local,
            BenchmarkResultLocation::RemotePending | BenchmarkResultLocation::RemoteSynced => {
                BenchmarkSource::Remote
            }
        };
        let benchmark_ref = match pipette_plan_types::BenchmarkId::try_new(ref_id) {
            Ok(bid) => SourcedBenchmarkId::new(source, bid).to_string(),
            Err(_) => id.to_string(),
        };
        let created_at = payload
            .get("submitted_at")
            .and_then(|v| v.as_str())
            .unwrap_or("1970-01-01T00:00:00Z")
            .to_string();
        // Verbatim: the descriptor is opaque. An empty string is the legacy
        // default, so it becomes `None`.
        let runtime_descriptor = payload
            .get("runtime_descriptor")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .map(str::to_string);
        Ok(Some(BenchmarkResultListEntry {
            result_id: id.to_string(),
            benchmark_ref,
            benchmark_id,
            benchmark_type,
            state,
            created_at,
            runtime_descriptor,
        }))
    }
}

/// The id prefix types only payloads written before `benchmark_type` was recorded.
pub fn recorded_benchmark_type(payload: &Value, id: &str) -> Option<BenchmarkType> {
    match payload.get("benchmark_type").and_then(Value::as_str) {
        Some(recorded) => recorded.parse().ok(),
        None => BenchmarkType::from_id(
            payload
                .get("benchmark_id")
                .and_then(Value::as_str)
                .unwrap_or(id),
        ),
    }
}

/// Relocate a result directory (pending to synced). A no-op when `from` is
/// already gone, as on a re-run after a successful move.
pub fn move_result_dir(from: &Path, to: &Path) -> Result<()> {
    if !from.exists() {
        return Ok(());
    }
    if let Some(parent) = to.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    fs::rename(from, to)
        .with_context(|| format!("failed to move {} to {}", from.display(), to.display()))
        .map_err(Into::into)
}

/// A server-assigned job id, refused unless it names exactly one directory
/// under `synced/`: it becomes a path, and `..`, `""` or a Windows `C:x`
/// would escape it.
pub fn check_job_id(job_id: &str) -> Result<&str> {
    let mut parts = Path::new(job_id).components();
    if !matches!(
        (parts.next(), parts.next()),
        (Some(std::path::Component::Normal(_)), None)
    ) {
        return Err(anyhow::anyhow!("the server returned an unusable job id {job_id:?}").into());
    }
    Ok(job_id)
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn store() -> anyhow::Result<(tempfile::TempDir, ResultsStore)> {
        let tmp = tempfile::tempdir()?;
        let store = ResultsStore::new(tmp.path().join("results"));
        Ok((tmp, store))
    }

    fn seed(
        store: &ResultsStore,
        location: BenchmarkResultLocation,
        id: &str,
        payload: serde_json::Value,
    ) -> anyhow::Result<()> {
        store.save_payload(location, id, &payload)?;
        Ok(())
    }

    fn entry(
        store: &ResultsStore,
        location: BenchmarkResultLocation,
        id: &str,
    ) -> anyhow::Result<BenchmarkResultListEntry> {
        store
            .load_list_entry(location, id)?
            .ok_or_else(|| anyhow::anyhow!("result {id} should list"))
    }

    #[test]
    fn corrupt_metrics_read_as_missing() -> anyhow::Result<()> {
        let (_tmp, store) = store()?;
        let synced = BenchmarkResultLocation::RemoteSynced;
        assert!(!store.has_valid_metrics(synced, "job-1"));

        store.save_metrics(synced, "job-1", &BenchmarkScoredResult::default())?;
        assert!(store.has_valid_metrics(synced, "job-1"));

        fs::write(store.metrics_path(synced, "job-1"), "{ not json")?;
        assert!(!store.has_valid_metrics(synced, "job-1"));
        Ok(())
    }

    /// A pending result is `local` in the listing whatever it carries.
    #[rstest::rstest]
    #[case::local(BenchmarkResultLocation::Local, false, BenchmarkResultState::Local)]
    #[case::pending(
        BenchmarkResultLocation::RemotePending,
        true,
        BenchmarkResultState::Local
    )]
    #[case::synced(
        BenchmarkResultLocation::RemoteSynced,
        false,
        BenchmarkResultState::Submitted
    )]
    #[case::scored(
        BenchmarkResultLocation::RemoteSynced,
        true,
        BenchmarkResultState::Scored
    )]
    fn the_listed_state_follows_location_and_metrics(
        #[case] location: BenchmarkResultLocation,
        #[case] with_metrics: bool,
        #[case] expected: BenchmarkResultState,
    ) -> anyhow::Result<()> {
        let (_tmp, store) = store()?;
        seed(
            &store,
            location,
            "r1",
            json!({"benchmark_id": "prefill_throughput_512"}),
        )?;
        if with_metrics {
            store.save_metrics(location, "r1", &BenchmarkScoredResult::default())?;
        }

        assert_eq!(entry(&store, location, "r1")?.state, expected);
        Ok(())
    }

    #[rstest::rstest]
    #[case::plain("job-7", true)]
    #[case::empty("", false)]
    #[case::current(".", false)]
    #[case::parent("..", false)]
    #[case::nested("a/b", false)]
    #[case::absolute("/x", false)]
    #[cfg_attr(windows, case::drive("C:x", false))]
    #[cfg_attr(windows, case::backslash("a\\b", false))]
    fn only_a_single_name_is_a_usable_job_id(#[case] job_id: &str, #[case] usable: bool) {
        assert_eq!(check_job_id(job_id).is_ok(), usable, "{job_id:?}");
    }

    #[rstest::rstest]
    #[case(BenchmarkResultLocation::Local, "local/prefill_throughput_512")]
    #[case(
        BenchmarkResultLocation::RemotePending,
        "remote/prefill_throughput_512"
    )]
    #[case(BenchmarkResultLocation::RemoteSynced, "remote/prefill_throughput_512")]
    fn the_ref_names_the_source_catalog(
        #[case] location: BenchmarkResultLocation,
        #[case] expected: &str,
    ) -> anyhow::Result<()> {
        let (_tmp, store) = store()?;
        seed(
            &store,
            location,
            "r1",
            json!({"benchmark_id": "prefill_throughput_512"}),
        )?;

        assert_eq!(entry(&store, location, "r1")?.benchmark_ref, expected);
        Ok(())
    }

    /// A missing or empty `runtime_descriptor` is `None`, and a missing
    /// `submitted_at` reads as the epoch.
    #[rstest::rstest]
    #[case::absent(json!({"benchmark_id": "decode_throughput_256_100"}))]
    #[case::empty(json!({"benchmark_id": "decode_throughput_256_100", "runtime_descriptor": ""}))]
    fn a_legacy_payload_lists_with_defaults(
        #[case] payload: serde_json::Value,
    ) -> anyhow::Result<()> {
        let (_tmp, store) = store()?;
        let local = BenchmarkResultLocation::Local;
        seed(&store, local, "r1", payload)?;

        let listed = entry(&store, local, "r1")?;
        assert_eq!(listed.runtime_descriptor, None);
        assert_eq!(listed.created_at, "1970-01-01T00:00:00Z");
        assert_eq!(listed.benchmark_type, BenchmarkType::DecodeThroughput);
        Ok(())
    }

    /// Results written before `benchmark_id` existed are typed by directory name.
    #[test]
    fn the_result_id_types_a_payload_without_a_benchmark_id() -> anyhow::Result<()> {
        let (_tmp, store) = store()?;
        let local = BenchmarkResultLocation::Local;
        seed(&store, local, "end_to_end_latency_512_256", json!({}))?;

        let listed = entry(&store, local, "end_to_end_latency_512_256")?;
        assert_eq!(listed.benchmark_id, None);
        assert_eq!(listed.benchmark_type, BenchmarkType::EndToEndLatency);
        Ok(())
    }

    #[test]
    fn a_result_lists_under_its_recorded_type() -> anyhow::Result<()> {
        let (_tmp, store) = store()?;
        let local = BenchmarkResultLocation::Local;
        seed(
            &store,
            local,
            "r1",
            json!({
                "benchmark_id": "quill_orch_vl_throughput_512x512_text3396_decode47_img3",
                "benchmark_type": "vl_throughput",
            }),
        )?;

        assert_eq!(
            entry(&store, local, "r1")?.benchmark_type,
            BenchmarkType::VlThroughput
        );
        Ok(())
    }

    /// A newer client may have written it into a shared workspace.
    #[rstest::rstest]
    #[case::unknown_prefix(json!({"benchmark_id": "mystery_metric_1"}))]
    #[case::unknown_recorded(
        json!({"benchmark_id": "prefill_throughput_512", "benchmark_type": "quantum_latency"})
    )]
    fn an_unknown_benchmark_type_is_skipped(#[case] payload: Value) -> anyhow::Result<()> {
        let (_tmp, store) = store()?;
        let local = BenchmarkResultLocation::Local;
        seed(&store, local, "r1", payload)?;

        assert!(store.load_list_entry(local, "r1")?.is_none());
        Ok(())
    }

    #[test]
    fn list_ids_returns_sorted_directories_only() -> anyhow::Result<()> {
        let (_tmp, store) = store()?;
        let local = BenchmarkResultLocation::Local;
        assert!(store.list_ids(local)?.is_empty());

        ["b", "a", "c"]
            .into_iter()
            .try_for_each(|id| seed(&store, local, id, json!({})))?;
        fs::write(store.location_dir(local).join("stray.json"), "{}")?;

        assert_eq!(store.list_ids(local)?, ["a", "b", "c"]);
        Ok(())
    }

    #[test]
    fn save_result_writes_payload_and_extras() -> anyhow::Result<()> {
        let (_tmp, store) = store()?;
        let local = BenchmarkResultLocation::Local;
        let extras = crate::results::BenchmarkResultExtras {
            executable: Some("llama-bench".into()),
            command: vec!["llama-bench".into(), "-p".into(), "512".into()],
            stdout: "out".into(),
            stderr: String::new(),
            turns: Vec::new(),
        };

        store.save_result(local, "r1", &json!({"benchmark_id": "x"}), &extras)?;

        assert_eq!(store.load_payload(local, "r1")?["benchmark_id"], "x");
        let stored: crate::results::BenchmarkResultExtras =
            read_json_file(&store.extras_path(local, "r1"))?;
        assert_eq!(stored.command, extras.command);
        assert_eq!(stored.executable, extras.executable);
        Ok(())
    }

    #[test]
    fn move_result_dir_creates_the_parent_and_tolerates_a_missing_source() -> anyhow::Result<()> {
        let (_tmp, store) = store()?;
        let pending = BenchmarkResultLocation::RemotePending;
        let synced = BenchmarkResultLocation::RemoteSynced;
        seed(&store, pending, "r1", json!({"benchmark_id": "x"}))?;

        let from = store.result_dir(pending, "r1");
        let to = store.result_dir(synced, "job-9");
        move_result_dir(&from, &to)?;

        assert!(!from.exists());
        assert_eq!(store.load_payload(synced, "job-9")?["benchmark_id"], "x");
        move_result_dir(&from, &to)?;
        assert!(to.exists());
        Ok(())
    }
}
