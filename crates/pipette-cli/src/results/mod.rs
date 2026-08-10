//! Everything about a benchmark result: the [`ResultsStore`] on disk
//! (`store`), the lifecycle/entry types and the `extras.json` sidecar
//! (`types`), and the flow that records a submission and optionally submits it
//! (`record`). The wire payload itself is
//! [`pipette_plan_types::result::BenchmarkSubmissionPayload`] — the shape the
//! mobile clients mirror.

mod record;
mod store;
mod types;

pub(crate) use record::finished_run_payload;
pub use record::{record_and_maybe_submit_run, RecordSubmitOutcome};
pub use store::{move_result_dir, ResultsStore};
pub use types::{
    BenchmarkJobMetric, BenchmarkResultExtras, BenchmarkResultListEntry, BenchmarkResultLocation,
    BenchmarkResultState, BenchmarkScoredResult,
};
