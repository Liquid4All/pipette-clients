//! Core AI execute: kind dispatch from a prepared [`RunRequest`].

mod decode_throughput;
mod end_to_end_latency;
mod max_memory_usage;
mod prefill_throughput;

use pipette_ops::measurement::{self, MeasurementObserver};
use pipette_ops::readiness::ReadinessGate;
use pipette_plan_types::{BenchmarkDefinition, RunRequest, RunResponse};

use crate::sidecar::require_coreai_sidecar;

/// Why a batched entry fails, for [`pipette_ops::measurement::require_one_in_flight`].
const ONE_AT_A_TIME: &str = "the Core AI sidecar serves one request at a time";

/// Top-level Core AI dispatch: route a prepared [`RunRequest`] by kind.
///
/// Installation finishes during preparation. Dispatch only resolves the
/// executable from the bound runtime before the cell runs.
pub fn run(
    req: &RunRequest,
    readiness_gate: ReadinessGate,
    observer: &MeasurementObserver<'_>,
) -> anyhow::Result<RunResponse> {
    match &req.benchmark {
        BenchmarkDefinition::PrefillThroughput(benchmark) => {
            measurement::require_one_in_flight(&req.benchmark, ONE_AT_A_TIME)?;
            prefill_throughput::run(
                req,
                benchmark,
                require_coreai_sidecar(req)?,
                readiness_gate,
                observer,
            )
        }
        BenchmarkDefinition::DecodeThroughput(benchmark) => {
            measurement::require_one_in_flight(&req.benchmark, ONE_AT_A_TIME)?;
            decode_throughput::run(
                req,
                benchmark,
                require_coreai_sidecar(req)?,
                readiness_gate,
                observer,
            )
        }
        BenchmarkDefinition::EndToEndLatency(benchmark) => {
            measurement::require_one_in_flight(&req.benchmark, ONE_AT_A_TIME)?;
            end_to_end_latency::run(
                req,
                benchmark,
                require_coreai_sidecar(req)?,
                readiness_gate,
                observer,
            )
        }
        BenchmarkDefinition::MaxMemoryUsage(benchmark) => {
            max_memory_usage::run(req, benchmark, require_coreai_sidecar(req)?, observer)
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
