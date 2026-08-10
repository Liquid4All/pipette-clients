//! Concrete [`ResultsStore`] under workspace `results/`.

use std::{fs, path::Path, path::PathBuf};

use anyhow::Context;
use serde::Serialize;
use serde_json::Value;

use pipette_ops::fs::{read_json_file, write_json_file};
use pipette_plan_types::benchmark::BenchmarkSource;
use pipette_plan_types::BenchmarkType;

use super::{
    BenchmarkResultListEntry, BenchmarkResultLocation, BenchmarkResultState, BenchmarkScoredResult,
};
use crate::benchmarks::SourcedBenchmarkId;
use crate::error::Result;

/// Capability handle for benchmark results (`root/` = workspace `results/`).
///
/// Layout (private):
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

    /// Write both halves of a result: the submission payload and the extras
    /// beside it. One call so a caller can't forget the second half — the worker
    /// went without its extras while each path wrote them itself. Not atomic: a
    /// failure between the two leaves the payload alone on disk.
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

    /// Returns true when a result has a scored metrics file that parses cleanly.
    /// Corrupt metrics are treated as missing so a later refresh can replace them.
    pub fn has_valid_metrics(&self, location: BenchmarkResultLocation, id: &str) -> bool {
        let metrics_path = self.metrics_path(location, id);
        if !metrics_path.exists() {
            return false;
        }
        match read_json_file::<BenchmarkScoredResult>(&metrics_path) {
            Ok(_) => true,
            Err(err) => {
                log::warn!("ignoring corrupt metrics for result {id}: {err:#}");
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
        let type_key = benchmark_id.as_deref().unwrap_or(id);
        let Some(benchmark_type) = BenchmarkType::from_id(type_key) else {
            log::warn!("skipping result {id}: unknown benchmark type");
            return Ok(None);
        };
        let has_valid_metrics = matches!(location, BenchmarkResultLocation::RemoteSynced)
            && self.has_valid_metrics(location, id);
        let state = match location {
            BenchmarkResultLocation::RemoteSynced if has_valid_metrics => {
                BenchmarkResultState::Scored
            }
            BenchmarkResultLocation::RemoteSynced => BenchmarkResultState::Submitted,
            _ => BenchmarkResultState::Local,
        };
        let ref_id = benchmark_id.as_deref().unwrap_or(id).to_string();
        let source = match location {
            BenchmarkResultLocation::Local => BenchmarkSource::Local,
            _ => BenchmarkSource::Remote,
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
        // Carry the `runtime_descriptor` JSON verbatim; the listing renders it as-is
        // (the server treats the descriptor opaquely, so we don't reshape it here).
        // Empty string is the legacy-payload default — normalize it to `None`.
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

/// Relocate a result directory (pending → synced). Free-standing because it
/// works on two paths, not on one store-owned id.
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
