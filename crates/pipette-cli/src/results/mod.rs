//! Benchmark results: the on-disk [`ResultsStore`], its lifecycle types, and
//! the flow that records a run and optionally submits it. The wire payload is
//! [`pipette_plan_types::BenchmarkSubmissionPayload`], which the mobile clients
//! mirror.

mod record;
mod store;
mod types;

pub(crate) use record::finished_run_payload;
pub use record::{RecordSubmitOutcome, record_and_maybe_submit_run};
pub use store::{ResultsStore, check_job_id, move_result_dir, recorded_benchmark_type};
pub use types::{
    BenchmarkJobMetric, BenchmarkResultExtras, BenchmarkResultListEntry, BenchmarkResultLocation,
    BenchmarkResultState, BenchmarkScoredResult,
};
