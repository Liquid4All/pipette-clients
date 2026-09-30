//! Core AI execute: kind dispatch from a prepared [`RunRequest`].

mod decode_throughput;
mod end_to_end_latency;
mod max_memory_usage;
mod prefill_throughput;
pub(crate) mod server;
mod throughput_http;

use pipette_ops::readiness::{ReadinessGate, RepObserver};
use pipette_plan_types::run::RunRequest;
use pipette_plan_types::run::RunResponse;
use pipette_plan_types::Runtime;

/// Top-level Core AI dispatch: route a prepared [`RunRequest`] by kind.
///
/// CLI owns prepare/record; this crate only runs the cell and returns
/// [`RunResponse`].
pub fn run(
    req: &RunRequest,
    readiness_gate: ReadinessGate,
    observer: &RepObserver,
) -> anyhow::Result<RunResponse> {
    // Fail closed on a non-bundled pin: the sidecar is built from the bundled
    // `Package.resolved` only, so running a cell whose runtime records a
    // different Swift stack would publish a number under a pin the executed
    // binary does not have. The CLI/URI parse rejects this too, but a plan
    // deserialized straight into a `RunRequest` reaches here without that gate.
    if let Runtime::AppleCoreAiMacosPipette(rt) = &req.runtime.bound {
        if !rt.is_bundled() {
            anyhow::bail!(
                "core-ai-macos-pipette runtime pins a Swift stack this client was \
                 not built against ({}); only the bundled pin is runnable",
                rt
            );
        }
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
        pipette_plan_types::BenchmarkType::Eval => {
            anyhow::bail!("eval is not yet supported for the Core AI runtime")
        }
        pipette_plan_types::BenchmarkType::VlThroughput => {
            anyhow::bail!("VL throughput benchmarks are not yet supported for Core AI")
        }
    }
}
