//! lloom-serve execute: kind dispatch from a prepared [`RunRequest`].

mod decode_throughput;
mod end_to_end_latency;
mod eval;
mod max_memory_usage;
mod prefill_throughput;
pub(crate) mod server;
mod throughput_http;

use pipette_ops::readiness::{ReadinessGate, RepObserver};
use pipette_ops::EvalCompletionsStore;
use pipette_plan_types::run::RunRequest;
use pipette_plan_types::run::RunResponse;

/// Top-level lloom-serve dispatch: route a prepared [`RunRequest`] by kind.
///
/// CLI owns prepare/record; this crate only runs the cell and returns
/// [`RunResponse`]. `eval_completions` is used only for eval resume.
pub fn run(
    req: &RunRequest,
    eval_completions: &EvalCompletionsStore,
    readiness_gate: ReadinessGate,
    observer: &RepObserver,
) -> anyhow::Result<RunResponse> {
    if req.runtime_flags.is_some() {
        log::warn!(
            "runtime_flags are ignored for lloom-serve runs (the server takes no cell flags)"
        );
    }
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
        pipette_plan_types::BenchmarkType::Eval => eval::run(req, eval_completions),
        pipette_plan_types::BenchmarkType::VlThroughput => {
            anyhow::bail!("VL throughput benchmarks are not supported for lloom-serve")
        }
    }
}
