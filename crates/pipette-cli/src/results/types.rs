//! Wire/entry types describing a stored result's lifecycle: where it lives,
//! its display state, its listing entry, and its pulled score metrics.

use serde::{Deserialize, Serialize};
use strum::{Display, EnumString};

use pipette_plan_types::BenchmarkType;

// ---------------------------------------------------------------------------
// BenchmarkResultLocation — where a result is stored
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BenchmarkResultLocation {
    Local,
    RemotePending,
    RemoteSynced,
}

impl From<pipette_plan_types::benchmark::BenchmarkSource> for BenchmarkResultLocation {
    /// A body from the synced catalog is submittable, so its result waits for
    /// `sync`; one only this machine has is not, so it stays local.
    fn from(source: pipette_plan_types::benchmark::BenchmarkSource) -> Self {
        match source {
            pipette_plan_types::benchmark::BenchmarkSource::Local => Self::Local,
            pipette_plan_types::benchmark::BenchmarkSource::Remote => Self::RemotePending,
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

// ---------------------------------------------------------------------------
// BenchmarkResultState — lifecycle state for display/filtering
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Display, EnumString)]
#[strum(serialize_all = "lowercase")]
pub enum BenchmarkResultState {
    Local,
    Submitted,
    Scored,
}

// ---------------------------------------------------------------------------
// BenchmarkResultListEntry — slim entry for result listing
// ---------------------------------------------------------------------------

pub struct BenchmarkResultListEntry {
    pub result_id: String,
    pub benchmark_ref: String,
    pub benchmark_id: Option<String>,
    /// The resolved benchmark type — validated when the entry is built, so
    /// consumers filter on it directly rather than re-parsing an id/ref.
    pub benchmark_type: BenchmarkType,
    pub state: BenchmarkResultState,
    pub created_at: String,
    /// The payload's `runtime_descriptor` — canonical JSON of the run's
    /// `pipette_plan_types::Runtime` — carried verbatim so a runtime-agnostic
    /// listing can show it without a second payload read. `None` for legacy
    /// payloads that predate the descriptor.
    pub runtime_descriptor: Option<String>,
}

// ---------------------------------------------------------------------------
// Scored results — metrics pulled from the management server
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// BenchmarkResultExtras — on-disk `extras.json` sidecar
// ---------------------------------------------------------------------------

/// Invocation preview and captured streams for a stored result, kept in
/// `extras.json` next to `payload.json` rather than on the submission wire.
///
/// Local only: [`BenchmarkSubmissionPayload`](pipette_plan_types::result::BenchmarkSubmissionPayload)
/// carries neither field, so what ran is recoverable on the box that ran it and
/// nowhere else. Putting either on the wire needs a `pipette-mgmt` schema
/// change first.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BenchmarkResultExtras {
    /// The runtime binary this cell invoked, when it invoked one. Distinct from
    /// `command[0]` for a cell whose preview is a shape rather than a literal
    /// argv — OpenVINO names its driver script there, not the temp path it ran
    /// from.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub executable: Option<String>,
    pub command: Vec<String>,
    pub stdout: String,
    pub stderr: String,
}
