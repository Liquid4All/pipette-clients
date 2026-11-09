//! ONNX GenAI execution through a minimal OpenAI-compatible server
//! (`pipette_onnx_genai_server.py`): one server per cell, loaded before it
//! listens, so the model load stays outside every measured request. Cells are
//! timed on the client's clock, as on the other server-driven engines.

mod decode_throughput;
mod end_to_end_latency;
mod max_memory_usage;
mod prefill_throughput;

/// Why the timing cells refuse a batched entry on this engine. The server
/// generates one sequence at a time, so a batched entry would be measured
/// serially and filed under a `_bs<n>` id that promises the measurement under
/// load. Refused before the model loads, since no amount of waiting changes it.
const ONE_AT_A_TIME: &str = "the ONNX GenAI server generates one sequence at a time";

/// Request budget when the plan pins no `http_timeout`: a CPU prefill of a
/// long prompt on a small host runs to minutes.
const REQUEST_TIMEOUT_SECS: u64 = 3600;

use std::time::Duration;

use pipette_ops::measurement::{self, MeasurementObserver};
use pipette_ops::readiness::ReadinessGate;
use pipette_plan_types::{
    BenchmarkDefinition, RunRequest, RunResponse, UvOnnxGenaiExtraRuntimeConfiguration,
};

use crate::models::ResolvedOnnxGenaiModel;
use crate::runtimes::ResolvedOnnxGenaiInstall;

/// Run a prepared cell; the CLI owns preparation and result recording.
/// `client_config` is applied whole to the server process, unread here.
pub fn run(
    req: &RunRequest,
    readiness_gate: ReadinessGate,
    observer: MeasurementObserver<'_>,
    client_config: Option<&UvOnnxGenaiExtraRuntimeConfiguration>,
) -> anyhow::Result<RunResponse> {
    let timeout = request_timeout(req);
    match &req.benchmark {
        BenchmarkDefinition::PrefillThroughput(benchmark) => {
            measurement::require_one_in_flight(&req.benchmark, ONE_AT_A_TIME)?;
            let (install, model) = bind(req)?;
            prefill_throughput::run(
                benchmark,
                &install,
                &model,
                timeout,
                readiness_gate,
                observer,
                client_config,
            )
        }
        BenchmarkDefinition::DecodeThroughput(benchmark) => {
            measurement::require_one_in_flight(&req.benchmark, ONE_AT_A_TIME)?;
            let (install, model) = bind(req)?;
            decode_throughput::run(
                benchmark,
                &install,
                &model,
                timeout,
                readiness_gate,
                observer,
                client_config,
            )
        }
        BenchmarkDefinition::EndToEndLatency(benchmark) => {
            measurement::require_one_in_flight(&req.benchmark, ONE_AT_A_TIME)?;
            let (install, model) = bind(req)?;
            end_to_end_latency::run(
                benchmark,
                &install,
                &model,
                timeout,
                readiness_gate,
                observer,
                client_config,
            )
        }
        BenchmarkDefinition::MaxMemoryUsage(benchmark) => {
            let (install, model) = bind(req)?;
            max_memory_usage::run(
                benchmark,
                &install,
                &model,
                timeout,
                observer,
                client_config,
            )
        }
        BenchmarkDefinition::Eval(_) => {
            anyhow::bail!("eval benchmarks are not yet supported for ONNX GenAI")
        }
        BenchmarkDefinition::VlThroughput(_)
        | BenchmarkDefinition::VlMaxMemory(_)
        | BenchmarkDefinition::VlmEval(_)
        | BenchmarkDefinition::VlSpecDecode(_) => {
            anyhow::bail!("VL benchmarks are not yet supported for ONNX GenAI")
        }
        BenchmarkDefinition::AsrEval(_) => {
            anyhow::bail!("asr_eval is not yet supported for ONNX GenAI")
        }
        BenchmarkDefinition::AsrThroughput(_) | BenchmarkDefinition::AsrMaxMemory(_) => {
            anyhow::bail!("ASR benchmarks are not yet supported for ONNX GenAI")
        }
    }
}

/// Refused when the model declares no support for the install's provider.
fn bind(req: &RunRequest) -> anyhow::Result<(ResolvedOnnxGenaiInstall, ResolvedOnnxGenaiModel)> {
    let install = ResolvedOnnxGenaiInstall::try_from(&req.runtime.bound)?;
    let model = ResolvedOnnxGenaiModel::try_from(&req.model.bound)?;
    model.check_provider(&install.execution_provider)?;
    Ok((install, model))
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
    use std::cell::Cell;
    use std::path::Path;

    use rstest::rstest;

    use pipette_plan_types::{
        AbsolutePath, BatchSize, BenchmarkId, DeclaredBound, DecodeThroughputBenchmark,
        EndToEndLatencyBenchmark, MaxMemoryUsageBenchmark, Model, ModelSource, NonNegativeCount,
        OnnxGenai, PositiveCount, PrefillThroughputBenchmark, Runtime,
    };

    use super::*;
    use crate::runtimes::UnservableProviderError;

    fn request(
        venv: &Path,
        model: &Path,
        provider: &str,
        benchmark: BenchmarkDefinition,
    ) -> anyhow::Result<RunRequest> {
        let runtime: Runtime = serde_json::from_value(serde_json::json!({
            "type": "uv_onnx_genai", "server_version": "0.16.0",
            "python_version": "3.11", "execution_provider": provider,
            "source": {"type": "absolute_preinstalled", "dir": venv}
        }))?;
        Ok(RunRequest {
            runtime: DeclaredBound::already_bound(runtime),
            model: DeclaredBound::already_bound(Model::OnnxGenai(OnnxGenai {
                source: ModelSource::AbsoluteDir {
                    dir: AbsolutePath::try_new(model.to_string_lossy().into_owned())?,
                },
            })),
            runtime_flags: None,
            model_flags: None,
            benchmark_flags: None,
            benchmark,
        })
    }

    /// The reference entry reaches binding, which fails for want of a runtime.
    #[rstest]
    #[case::batched(4, "decode_throughput_512_256_bs4")]
    #[case::batched_names_the_engine(4, "one sequence at a time")]
    #[case::reference(1, "python missing")]
    fn a_batched_entry_is_refused_before_the_runtime_binds(
        #[case] batch_size: u32,
        #[case] expected: &str,
    ) -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let req = request(
            dir.path(),
            dir.path(),
            "cpu",
            BenchmarkDefinition::DecodeThroughput(DecodeThroughputBenchmark {
                benchmark_id: BenchmarkId::try_new(format!(
                    "decode_throughput_512_256_bs{batch_size}"
                ))?,
                parameter_prefill_tokens: NonNegativeCount::try_new(512)?,
                parameter_decode_tokens: PositiveCount::try_new(256)?,
                parameter_batch_size: BatchSize::try_new(batch_size)?,
            }),
        )?;
        let observer = MeasurementObserver::QUIET;
        let Err(err) = run(&req, &|| Ok(()), observer, None) else {
            anyhow::bail!("expected a batch refusal or a missing-runtime failure");
        };
        let message = format!("{err:#}");
        assert!(message.contains(expected), "{message}");
        Ok(())
    }

    /// Refused as the runtime binds, so before the gate too. The model declares
    /// `winml` and the venv has an interpreter, so only the provider refusal
    /// can stop the cell.
    #[rstest]
    #[case::prefill(|| Ok(BenchmarkDefinition::PrefillThroughput(PrefillThroughputBenchmark {
        benchmark_id: BenchmarkId::try_new("prefill_throughput_64".to_owned())?,
        parameter_prefill_tokens: NonNegativeCount::try_new(64)?,
        parameter_batch_size: BatchSize::one(),
    })))]
    #[case::decode(|| Ok(BenchmarkDefinition::DecodeThroughput(DecodeThroughputBenchmark {
        benchmark_id: BenchmarkId::try_new("decode_throughput_64_16".to_owned())?,
        parameter_prefill_tokens: NonNegativeCount::try_new(64)?,
        parameter_decode_tokens: PositiveCount::try_new(16)?,
        parameter_batch_size: BatchSize::one(),
    })))]
    #[case::end_to_end(|| Ok(BenchmarkDefinition::EndToEndLatency(EndToEndLatencyBenchmark {
        benchmark_id: BenchmarkId::try_new("end_to_end_latency_64_16".to_owned())?,
        parameter_prefill_tokens: NonNegativeCount::try_new(64)?,
        parameter_decode_tokens: PositiveCount::try_new(16)?,
        parameter_batch_size: BatchSize::one(),
    })))]
    #[case::memory(|| Ok(BenchmarkDefinition::MaxMemoryUsage(MaxMemoryUsageBenchmark {
        benchmark_id: BenchmarkId::try_new("max_memory_usage_64".to_owned())?,
        parameter_prefill_tokens: NonNegativeCount::try_new(64)?,
    })))]
    fn windows_ml_is_refused_before_a_server_starts(
        #[case] benchmark: fn() -> anyhow::Result<BenchmarkDefinition>,
    ) -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let venv = dir.path().join("venv");
        std::fs::create_dir_all(pipette_venv::venv_bin(&venv))?;
        std::fs::write(pipette_venv::venv_python(&venv), "")?;
        let model = dir.path().join("model");
        std::fs::create_dir_all(&model)?;
        std::fs::write(
            model.join("genai_config.json"),
            r#"{"model": {"decoder": {"filename": "m.onnx",
                "session_options": {"provider_options": [{"winml": {}}]}}}}"#,
        )?;
        std::fs::write(model.join("m.onnx"), b"graph")?;
        let req = request(&venv, &model, "winml", benchmark()?)?;
        let gated = Cell::new(false);
        let gate = || {
            gated.set(true);
            Ok(())
        };
        let observer = MeasurementObserver::QUIET;
        let Err(err) = run(&req, &gate, observer, None) else {
            anyhow::bail!("a Windows ML cell ran");
        };
        // The outermost error, not a cause under the memory cell's warm-up.
        let refused = err
            .chain()
            .next()
            .and_then(|outermost| outermost.downcast_ref::<UnservableProviderError>());
        assert_eq!(refused, Some(&UnservableProviderError::WinMl), "{err:#}");
        assert!(!gated.get(), "refused only after the gate");
        Ok(())
    }
}
