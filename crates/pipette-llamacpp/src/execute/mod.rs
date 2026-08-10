mod decode_throughput;
mod end_to_end_latency;
mod eval;
mod max_memory_usage;
mod prefill_throughput;
mod vl_throughput;

use pipette_ops::readiness::{ReadinessGate, RepObserver};
use pipette_ops::EvalCompletionsStore;
use pipette_plan_types::run::RunRequest;
pub use pipette_plan_types::run::RunResponse;

/// Top-level llamacpp **run** entry: dispatch a prepared [`RunRequest`] by [`BenchmarkType`](pipette_plan_types::BenchmarkType).
///
/// This module holds per-type implementations only; process helpers live in
/// [`crate::server`], [`crate::bench`], and [`crate::common`]. Bodies via
/// [`BenchmarkDefinition::as_*`](pipette_plan_types::benchmark::BenchmarkDefinition)
/// (`eval_completions` only for eval resume).
pub fn run(
    req: &RunRequest,
    eval_completions: &EvalCompletionsStore,
    readiness_gate: ReadinessGate,
    observer: &RepObserver,
) -> anyhow::Result<RunResponse> {
    // Route on type tag only; each `run` extracts its body with `as_*`.
    match req.benchmark.benchmark_type() {
        pipette_plan_types::BenchmarkType::PrefillThroughput => {
            prefill_throughput::run(req, readiness_gate, observer)
        }
        pipette_plan_types::BenchmarkType::DecodeThroughput => {
            decode_throughput::run(req, readiness_gate, observer)
        }
        pipette_plan_types::BenchmarkType::EndToEndLatency => {
            end_to_end_latency::run(req, readiness_gate, observer)
        }
        pipette_plan_types::BenchmarkType::MaxMemoryUsage => max_memory_usage::run(req),
        pipette_plan_types::BenchmarkType::Eval => eval::run_eval(req, eval_completions),
        pipette_plan_types::BenchmarkType::VlThroughput => {
            vl_throughput::run(req, readiness_gate, observer)
        }
    }
}
