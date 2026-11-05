//! OpenVINO execution through a minimal OpenAI-compatible server
//! (`pipette_openvino_server.py`), one per cell, read on the client's clock.
//! The server compiles exactly one pipeline before it listens: compiling more
//! than once per process took an NPU down mid-run (`docs/openvino-ir.md`).
//! The blob cache keeps that compile to a load for a cell compiled before
//! (`docs/openvino-measurement.md`).

mod decode_throughput;
mod end_to_end_latency;
mod max_memory_usage;
mod prefill_throughput;

/// Why the timing cells refuse a batched entry on this engine. The server
/// runs one `LLMPipeline.generate` at a time, so a batched entry would be
/// measured serially and filed under a `_bs<n>` id that promises the
/// measurement under load. Refused before the pipeline compiles, since no
/// amount of waiting changes it.
const ONE_AT_A_TIME: &str = "the OpenVINO server generates one sequence at a time";

/// Request budget when the plan pins no `http_timeout`: a CPU prefill of a
/// long prompt on a small host runs to minutes.
const REQUEST_TIMEOUT_SECS: u64 = 3600;

use std::path::Path;
use std::time::Duration;

use pipette_ops::measurement::{self, MeasurementObserver};
use pipette_ops::readiness::ReadinessGate;
use pipette_plan_types::{
    BenchmarkDefinition, RunRequest, RunResponse, UvOpenvinoExtraRuntimeConfiguration,
};

use crate::flags::resolve_runtime_flags;
use crate::models::ResolvedOpenvinoModel;
use crate::runtimes::ResolvedOpenvinoPython;

/// Run a prepared cell; the CLI owns preparation and result recording.
/// `client_config` is applied whole to the server process, unread here.
pub fn run(
    req: &RunRequest,
    readiness_gate: ReadinessGate,
    observer: MeasurementObserver<'_>,
    compile_cache: &Path,
    client_config: Option<&UvOpenvinoExtraRuntimeConfiguration>,
) -> anyhow::Result<RunResponse> {
    let timeout = request_timeout(req);
    match &req.benchmark {
        BenchmarkDefinition::PrefillThroughput(benchmark) => {
            measurement::require_one_in_flight(&req.benchmark, ONE_AT_A_TIME)?;
            prefill_throughput::run(
                benchmark,
                resolve_runtime_flags(req)?.try_into()?,
                &ResolvedOpenvinoPython::try_from(&req.runtime.bound)?,
                &ResolvedOpenvinoModel::try_from(&req.model.bound)?,
                compile_cache,
                timeout,
                readiness_gate,
                observer,
                client_config,
            )
        }
        BenchmarkDefinition::DecodeThroughput(benchmark) => {
            measurement::require_one_in_flight(&req.benchmark, ONE_AT_A_TIME)?;
            decode_throughput::run(
                benchmark,
                resolve_runtime_flags(req)?.try_into()?,
                &ResolvedOpenvinoPython::try_from(&req.runtime.bound)?,
                &ResolvedOpenvinoModel::try_from(&req.model.bound)?,
                compile_cache,
                timeout,
                readiness_gate,
                observer,
                client_config,
            )
        }
        BenchmarkDefinition::EndToEndLatency(benchmark) => {
            measurement::require_one_in_flight(&req.benchmark, ONE_AT_A_TIME)?;
            end_to_end_latency::run(
                benchmark,
                resolve_runtime_flags(req)?.try_into()?,
                &ResolvedOpenvinoPython::try_from(&req.runtime.bound)?,
                &ResolvedOpenvinoModel::try_from(&req.model.bound)?,
                compile_cache,
                timeout,
                readiness_gate,
                observer,
                client_config,
            )
        }
        BenchmarkDefinition::MaxMemoryUsage(benchmark) => max_memory_usage::run(
            benchmark,
            resolve_runtime_flags(req)?.try_into()?,
            &ResolvedOpenvinoPython::try_from(&req.runtime.bound)?,
            &ResolvedOpenvinoModel::try_from(&req.model.bound)?,
            compile_cache,
            timeout,
            client_config,
        ),
        BenchmarkDefinition::Eval(_) => {
            anyhow::bail!("eval benchmarks are not yet supported for OpenVINO")
        }
        BenchmarkDefinition::VlThroughput(_)
        | BenchmarkDefinition::VlMaxMemory(_)
        | BenchmarkDefinition::VlmEval(_)
        | BenchmarkDefinition::VlSpecDecode(_) => {
            anyhow::bail!("VL benchmarks are not yet supported for OpenVINO")
        }
        BenchmarkDefinition::AsrEval(_) => {
            anyhow::bail!("asr_eval is not yet supported for OpenVINO")
        }
        BenchmarkDefinition::AsrThroughput(_) | BenchmarkDefinition::AsrMaxMemory(_) => {
            anyhow::bail!("ASR benchmarks are not yet supported for OpenVINO")
        }
    }
}

fn request_timeout(req: &RunRequest) -> Duration {
    measurement::http_timeout_with_default(
        req.benchmark_flags.as_ref().and_then(|f| f.http_timeout()),
        1,
        REQUEST_TIMEOUT_SECS,
    )
}

#[cfg(test)]
mod tests {
    use anyhow::Context;

    use pipette_plan_types::{
        AbsolutePath, BatchSize, BenchmarkId, BenchmarkType, DeclaredBound,
        DecodeThroughputBenchmark, Model, ModelSource, ModelType, NonNegativeCount, Openvino,
        OpenvinoDevice, PositiveCount, Runtime, RuntimeFlagRef, RuntimeFlags, RuntimeType,
    };

    use super::*;

    /// A reference entry gets as far as binding, which fails here for want of
    /// an installed runtime.
    #[rstest::rstest]
    #[case::batched(4, "decode_throughput_512_256_bs4")]
    #[case::batched_names_the_engine(4, "one sequence at a time")]
    #[case::reference(1, "python missing")]
    fn a_batched_entry_is_refused_before_the_runtime_binds(
        #[case] batch_size: u32,
        #[case] expected: &str,
    ) -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let runtime: Runtime = serde_json::from_value(serde_json::json!({
            "type": "uv_openvino", "server_version": "2026.4.1",
            "python_version": "3.11",
            "source": {"type": "absolute_preinstalled", "dir": dir.path()}
        }))?;
        let mut flags = RuntimeFlagRef::new(
            BenchmarkType::DecodeThroughput,
            RuntimeType::UvOpenvino,
            ModelType::Openvino,
        );
        flags.device = Some(OpenvinoDevice::Cpu);
        let req = RunRequest {
            runtime: DeclaredBound::already_bound(runtime),
            model: DeclaredBound::already_bound(Model::Openvino(Openvino {
                source: ModelSource::AbsoluteDir {
                    dir: AbsolutePath::try_new(dir.path().to_string_lossy().into_owned())?,
                },
            })),
            runtime_flags: Some(RuntimeFlags::try_from(flags)?),
            model_flags: None,
            benchmark_flags: None,
            benchmark: BenchmarkDefinition::DecodeThroughput(DecodeThroughputBenchmark {
                benchmark_id: BenchmarkId::try_new(format!(
                    "decode_throughput_512_256_bs{batch_size}"
                ))?,
                parameter_prefill_tokens: NonNegativeCount::try_new(512)?,
                parameter_decode_tokens: PositiveCount::try_new(256)?,
                parameter_batch_size: BatchSize::try_new(batch_size)?,
            }),
        };
        let observer = MeasurementObserver {
            on_before_rep_start: &|| Ok(()),
            on_after_rep_end: &|| Ok(()),
        };
        let err = run(&req, &|| Ok(()), observer, dir.path(), None)
            .err()
            .context("expected the cell to fail")?;
        let message = format!("{err:#}");
        assert!(message.contains(expected), "{message}");
        Ok(())
    }
}
