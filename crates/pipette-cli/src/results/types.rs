//! Types describing a stored result's lifecycle, listing and scores.

use serde::{Deserialize, Serialize};
use strum::{Display, EnumString};

use pipette_plan_types::{BenchmarkType, TurnObservation};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BenchmarkResultLocation {
    Local,
    RemotePending,
    RemoteSynced,
}

impl From<pipette_plan_types::BenchmarkSource> for BenchmarkResultLocation {
    /// A body from the synced catalog is submittable, so its result waits for
    /// `sync`; one only this machine has is not, so it stays local.
    fn from(source: pipette_plan_types::BenchmarkSource) -> Self {
        match source {
            pipette_plan_types::BenchmarkSource::Local => Self::Local,
            pipette_plan_types::BenchmarkSource::Remote => Self::RemotePending,
        }
    }
}

impl BenchmarkResultLocation {
    pub fn label(self) -> &'static str {
        match self {
            Self::Local => "local",
            Self::RemotePending => "pending",
            Self::RemoteSynced => "synced",
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Display, EnumString)]
#[strum(serialize_all = "lowercase")]
pub enum BenchmarkResultState {
    Local,
    Submitted,
    Scored,
}

pub struct BenchmarkResultListEntry {
    pub result_id: String,
    pub benchmark_ref: String,
    pub benchmark_id: Option<String>,
    /// Validated when the entry is built, so consumers filter on it directly.
    pub benchmark_type: BenchmarkType,
    pub state: BenchmarkResultState,
    pub created_at: String,
    /// The payload's `runtime_descriptor` (canonical JSON of the run's
    /// `Runtime`), verbatim. `None` for payloads that predate it.
    pub runtime_descriptor: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkJobMetric {
    pub metric: String,
    pub value: f64,
    pub unit: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BenchmarkScoredResult {
    pub scored_at: Option<String>,
    pub metrics: Option<Vec<BenchmarkJobMetric>>,
}

/// `extras.json`, beside `payload.json`: the invocation and the streams as
/// captured. Local only, because the submission carries no invocation.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BenchmarkResultExtras {
    /// The runtime binary this cell invoked, when it invoked one. Can differ
    /// from `command[0]` when the preview is a shape rather than a literal argv.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub executable: Option<String>,
    pub command: Vec<String>,
    pub stdout: String,
    pub stderr: String,
    /// Per turn of a corpus traversal (`vl_spec_decode`), in order; empty for
    /// other kinds. Each reply's sha256, not its text, so two rows compare turn
    /// by turn without holding either's text; per-turn timings show drift that
    /// a summed row cannot. The submitted `observation_output_digest` summarises
    /// it.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub turns: Vec<TurnObservation>,
}
