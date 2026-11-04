//! OVMS execute: kind dispatch from a prepared [`RunRequest`].
//!
//! Every cell launches its own server on the model directory; a batched timing
//! entry keeps `parameter_batch_size` requests in flight against it.

mod decode_throughput;
mod end_to_end_latency;
mod max_memory_usage;
mod prefill_throughput;

use std::path::Path;

use anyhow::Context;

use pipette_ops::measurement::MeasurementObserver;
use pipette_ops::readiness::ReadinessGate;
use pipette_plan_types::{
    BatchSize, BenchmarkDefinition, BenchmarkFlags, OpenvinoDevice, OvmsExtraRuntimeConfiguration,
    RunRequest, RunResponse,
};

use crate::flags::resolve_runtime_flags;
use crate::models::ResolvedOpenvinoModel;
use crate::runtimes::ResolvedOvmsInstall;

/// `compile_cache` is where OpenVINO keeps this runtime's compiled blobs and GPU
/// kernels, so a warmed `(model, device)` loads the model instead of compiling
/// it; `client_config` is applied whole to the server.
pub fn run(
    req: &RunRequest,
    readiness_gate: ReadinessGate,
    observer: MeasurementObserver<'_>,
    compile_cache: &Path,
    client_config: Option<&OvmsExtraRuntimeConfiguration>,
) -> anyhow::Result<RunResponse> {
    let http_timeout = req
        .benchmark_flags
        .as_ref()
        .and_then(BenchmarkFlags::http_timeout);
    match &req.benchmark {
        BenchmarkDefinition::PrefillThroughput(benchmark) => {
            require_batchable(req, benchmark.parameter_batch_size, available_threads())?;
            prefill_throughput::run(
                benchmark,
                resolve_runtime_flags(req)?.try_into()?,
                &ResolvedOvmsInstall::try_from(&req.runtime.bound)?,
                &ResolvedOpenvinoModel::try_from(&req.model.bound)?,
                compile_cache,
                http_timeout,
                readiness_gate,
                observer,
                client_config,
            )
        }
        BenchmarkDefinition::DecodeThroughput(benchmark) => {
            require_batchable(req, benchmark.parameter_batch_size, available_threads())?;
            decode_throughput::run(
                benchmark,
                resolve_runtime_flags(req)?.try_into()?,
                &ResolvedOvmsInstall::try_from(&req.runtime.bound)?,
                &ResolvedOpenvinoModel::try_from(&req.model.bound)?,
                compile_cache,
                http_timeout,
                readiness_gate,
                observer,
                client_config,
            )
        }
        BenchmarkDefinition::EndToEndLatency(benchmark) => {
            require_batchable(req, benchmark.parameter_batch_size, available_threads())?;
            end_to_end_latency::run(
                benchmark,
                resolve_runtime_flags(req)?.try_into()?,
                &ResolvedOvmsInstall::try_from(&req.runtime.bound)?,
                &ResolvedOpenvinoModel::try_from(&req.model.bound)?,
                compile_cache,
                http_timeout,
                readiness_gate,
                observer,
                client_config,
            )
        }
        BenchmarkDefinition::MaxMemoryUsage(benchmark) => max_memory_usage::run(
            benchmark,
            resolve_runtime_flags(req)?.try_into()?,
            &ResolvedOvmsInstall::try_from(&req.runtime.bound)?,
            &ResolvedOpenvinoModel::try_from(&req.model.bound)?,
            compile_cache,
            http_timeout,
            client_config,
        ),
        BenchmarkDefinition::Eval(_)
        | BenchmarkDefinition::VlmEval(_)
        | BenchmarkDefinition::AsrEval(_) => {
            anyhow::bail!("eval benchmarks are not yet supported for OVMS")
        }
        BenchmarkDefinition::VlThroughput(_)
        | BenchmarkDefinition::VlMaxMemory(_)
        | BenchmarkDefinition::VlSpecDecode(_) => {
            anyhow::bail!("VL benchmarks are not yet supported for OVMS")
        }
        BenchmarkDefinition::AsrThroughput(_) | BenchmarkDefinition::AsrMaxMemory(_) => {
            anyhow::bail!("ASR benchmarks are not yet supported for OVMS")
        }
    }
}

/// OVMS would run these serially: the NPU has no continuous batching, and past
/// one request per hardware thread the rest queue for a graph (`docs/ovms.md`).
fn require_batchable(
    req: &RunRequest,
    batch_size: BatchSize,
    threads: anyhow::Result<usize>,
) -> anyhow::Result<()> {
    let slots = batch_size.as_u32();
    if slots == 1 {
        return Ok(());
    }
    let id = req.benchmark.benchmark_id();
    let flags = resolve_runtime_flags(req)?;
    anyhow::ensure!(
        flags.device != Some(OpenvinoDevice::Npu),
        "{id} keeps {slots} requests in flight, and OVMS runs the NPU without \
         continuous batching; run it on cpu or gpu"
    );
    let threads = threads?;
    anyhow::ensure!(
        usize::try_from(slots)? <= threads,
        "{id} keeps {slots} requests in flight, and OVMS serves at most one per \
         hardware thread, {threads} on this host; the rest would queue"
    );
    Ok(())
}

/// OVMS sizes its graph pool and its streaming threads to
/// `hardware_concurrency()`, which this reads the same way.
fn available_threads() -> anyhow::Result<usize> {
    Ok(std::thread::available_parallelism()
        .context("counting the host's hardware threads")?
        .get())
}

#[cfg(test)]
mod tests {
    use pipette_artifacts_catalog::runtime::ovms::layout::OvmsInstall;
    use pipette_plan_types::{
        AbsolutePath, BenchmarkId, DeclaredBound, DecodeThroughputBenchmark,
        EndToEndLatencyBenchmark, Model, ModelSource, ModelType, NonNegativeCount, Openvino,
        PositiveCount, RuntimeFlagRef, RuntimeFlags, RuntimeType,
    };

    use super::*;
    use crate::test_support::{bound_at, request};

    fn e2e(batch_size: u32) -> anyhow::Result<BenchmarkDefinition> {
        Ok(BenchmarkDefinition::EndToEndLatency(
            EndToEndLatencyBenchmark {
                benchmark_id: BenchmarkId::try_new(format!(
                    "end_to_end_latency_512_256_bs{batch_size}"
                ))?,
                parameter_prefill_tokens: NonNegativeCount::try_new(512)?,
                parameter_decode_tokens: PositiveCount::try_new(256)?,
                parameter_batch_size: BatchSize::try_new(batch_size)?,
            },
        ))
    }

    fn decode(decode_tokens: u32) -> anyhow::Result<BenchmarkDefinition> {
        Ok(BenchmarkDefinition::DecodeThroughput(
            DecodeThroughputBenchmark {
                benchmark_id: BenchmarkId::try_new(format!(
                    "decode_throughput_512_{decode_tokens}"
                ))?,
                parameter_prefill_tokens: NonNegativeCount::try_new(512)?,
                parameter_decode_tokens: PositiveCount::try_from(decode_tokens)?,
                parameter_batch_size: BatchSize::one(),
            },
        ))
    }

    fn on(
        device: OpenvinoDevice,
        benchmark: BenchmarkDefinition,
        dir: &std::path::Path,
    ) -> anyhow::Result<RunRequest> {
        let mut flags = RuntimeFlagRef::new(
            benchmark.benchmark_type(),
            RuntimeType::Ovms,
            ModelType::Openvino,
        );
        flags.device = Some(device);
        request(
            bound_at(dir),
            Some(RuntimeFlags::try_from(flags)?),
            benchmark,
        )
    }

    /// [`on`] with a release and an IR directory under `dir` that bind: every
    /// file the binding checks, empty.
    fn bindable(
        device: OpenvinoDevice,
        benchmark: BenchmarkDefinition,
        dir: &std::path::Path,
    ) -> anyhow::Result<RunRequest> {
        let server = OvmsInstall::under(dir).executable();
        std::fs::create_dir_all(server.parent().context("the server has no parent")?)?;
        std::fs::write(&server, b"")?;
        let ir = dir.join("ir");
        std::fs::create_dir_all(&ir)?;
        [
            "openvino_model.xml",
            "openvino_model.bin",
            "openvino_tokenizer.xml",
            "openvino_detokenizer.xml",
        ]
        .into_iter()
        .try_for_each(|file| std::fs::write(ir.join(file), b""))?;
        Ok(RunRequest {
            model: DeclaredBound::already_bound(Model::Openvino(Openvino {
                source: ModelSource::AbsoluteDir {
                    dir: AbsolutePath::try_new(ir.to_string_lossy().into_owned())?,
                },
            })),
            ..on(device, benchmark, dir)?
        })
    }

    /// A batch OVMS would run serially is refused before the binding, and a
    /// runnable cell gets as far as the binding, which fails on this fixture.
    /// Two slots fit even a two-thread CI runner.
    #[rstest::rstest]
    #[case::batched_on_npu(OpenvinoDevice::Npu, e2e(2), &["end_to_end_latency_512_256_bs2", "without continuous batching"])]
    #[case::batched_on_gpu(OpenvinoDevice::Gpu, e2e(2), &["ovms server missing"])]
    #[case::reference_on_npu(OpenvinoDevice::Npu, e2e(1), &["ovms server missing"])]
    fn a_cell_is_refused_before_the_server_starts(
        #[case] device: OpenvinoDevice,
        #[case] benchmark: anyhow::Result<BenchmarkDefinition>,
        #[case] expected: &[&str],
    ) -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let observer = MeasurementObserver {
            on_before_rep_start: &|| Ok(()),
            on_after_rep_end: &|| Ok(()),
        };
        let req = on(device, benchmark?, dir.path())?;
        let Err(err) = run(&req, &|| Ok(()), observer, dir.path(), None) else {
            anyhow::bail!("expected the cell to fail");
        };
        let message = format!("{err:#}");
        expected.iter().try_for_each(|part| {
            anyhow::ensure!(message.contains(part), "{message}");
            Ok(())
        })
    }

    /// Refused before the readiness gate too, not only before the sizing server.
    #[test]
    fn a_one_token_decode_is_refused_before_a_server_starts() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let req = bindable(OpenvinoDevice::Gpu, decode(1)?, dir.path())?;
        let gated = std::cell::Cell::new(false);
        let observer = MeasurementObserver {
            on_before_rep_start: &|| Ok(()),
            on_after_rep_end: &|| Ok(()),
        };
        let gate = || {
            gated.set(true);
            Ok(())
        };
        let Err(err) = run(&req, &gate, observer, &dir.path().join("cache"), None) else {
            anyhow::bail!("a one-token decode cell was measured");
        };
        let message = format!("{err:#}");
        assert!(
            message.contains("parameter_decode_tokens must be at least 2"),
            "{message}"
        );
        assert!(!gated.get(), "refused only after the gate");
        Ok(())
    }

    /// This fixture's server cannot run, so a cell that started it before
    /// refusing would fail with another error.
    #[cfg(not(target_os = "windows"))]
    #[test]
    fn an_npu_memory_cell_is_refused_before_the_sizing_server() -> anyhow::Result<()> {
        use pipette_plan_types::MaxMemoryUsageBenchmark;

        let dir = tempfile::tempdir()?;
        let benchmark = BenchmarkDefinition::MaxMemoryUsage(MaxMemoryUsageBenchmark {
            benchmark_id: BenchmarkId::try_new("max_memory_usage_128".to_owned())?,
            parameter_prefill_tokens: NonNegativeCount::try_new(128)?,
        });
        let req = bindable(OpenvinoDevice::Npu, benchmark, dir.path())?;
        let observer = MeasurementObserver {
            on_before_rep_start: &|| Ok(()),
            on_after_rep_end: &|| Ok(()),
        };
        let Err(err) = run(&req, &|| Ok(()), observer, &dir.path().join("cache"), None) else {
            anyhow::bail!("an NPU memory cell ran off Windows");
        };
        let expected = pipette_memprobe::UnsupportedError {
            watch: pipette_memprobe::Watch::PeakWithNpu,
        };
        assert_eq!(
            err.downcast_ref::<pipette_memprobe::WatchError>(),
            Some(&expected.into()),
            "{err:#}"
        );
        Ok(())
    }

    #[rstest::rstest]
    #[case::fits(8, 8, true)]
    #[case::queues(8, 4, false)]
    fn a_batch_fits_the_hosts_threads(
        #[case] batch_size: u32,
        #[case] threads: usize,
        #[case] accepted: bool,
    ) -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let req = on(OpenvinoDevice::Gpu, e2e(batch_size)?, dir.path())?;
        match require_batchable(&req, BatchSize::try_new(batch_size)?, Ok(threads)) {
            Ok(()) => anyhow::ensure!(accepted, "expected a refusal"),
            Err(err) => {
                anyhow::ensure!(!accepted, "{err:#}");
                let message = format!("{err:#}");
                anyhow::ensure!(message.contains("4 on this host"), "{message}");
            }
        }
        Ok(())
    }
}
