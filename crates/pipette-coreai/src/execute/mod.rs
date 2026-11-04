//! Core AI execute: kind dispatch from a prepared [`RunRequest`].

mod decode_throughput;
mod end_to_end_latency;
mod max_memory_usage;
mod prefill_throughput;

use pipette_ops::measurement::MeasurementObserver;
use pipette_ops::readiness::ReadinessGate;
use pipette_plan_types::{BenchmarkDefinition, RunRequest, RunResponse};

use crate::sidecar::require_coreai_sidecar;

/// Top-level Core AI dispatch: route a prepared [`RunRequest`] by kind.
///
/// Each supported arm resolves the sidecar before its cell runs, so a
/// first-use `swift build -c release`, which saturates every core for minutes,
/// finishes before the readiness gate certifies the device as idle.
pub fn run(
    req: &RunRequest,
    readiness_gate: ReadinessGate,
    observer: &MeasurementObserver<'_>,
) -> anyhow::Result<RunResponse> {
    match &req.benchmark {
        BenchmarkDefinition::PrefillThroughput(benchmark) => prefill_throughput::run(
            req,
            benchmark,
            require_coreai_sidecar()?,
            readiness_gate,
            observer,
        ),
        BenchmarkDefinition::DecodeThroughput(benchmark) => decode_throughput::run(
            req,
            benchmark,
            require_coreai_sidecar()?,
            readiness_gate,
            observer,
        ),
        BenchmarkDefinition::EndToEndLatency(benchmark) => end_to_end_latency::run(
            req,
            benchmark,
            require_coreai_sidecar()?,
            readiness_gate,
            observer,
        ),
        BenchmarkDefinition::MaxMemoryUsage(benchmark) => {
            max_memory_usage::run(req, benchmark, require_coreai_sidecar()?)
        }
        BenchmarkDefinition::Eval(_) => {
            anyhow::bail!("eval is not yet supported for the Core AI runtime")
        }
        BenchmarkDefinition::VlmEval(_)
        | BenchmarkDefinition::AsrEval(_)
        | BenchmarkDefinition::VlThroughput(_)
        | BenchmarkDefinition::VlMaxMemory(_)
        | BenchmarkDefinition::VlSpecDecode(_)
        | BenchmarkDefinition::AsrThroughput(_)
        | BenchmarkDefinition::AsrMaxMemory(_) => {
            anyhow::bail!("this benchmark is not yet supported for Core AI")
        }
    }
}
