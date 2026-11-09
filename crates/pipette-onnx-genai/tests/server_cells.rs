//! Every cell end to end through [`pipette_onnx_genai::run`] against
//! `fake_onnx_genai_server`, copied to where the venv's base interpreter would be:
//! server start, readiness, prompt sizing through `/tokenize`, the requests,
//! the checks, the reduction and the response.

use std::cell::Cell;

use anyhow::Context;
use rstest::rstest;

use pipette_ops::measurement::MeasurementObserver;
use pipette_plan_types::{
    AbsolutePath, BatchSize, BenchmarkDefinition, BenchmarkId, BenchmarkResultData, DeclaredBound,
    DecodeThroughputBenchmark, EndToEndLatencyBenchmark, MaxMemoryUsageBenchmark, Model,
    ModelSource, NonNegativeCount, OnnxExecutionProvider, OnnxGenai, PositiveCount,
    PrefillThroughputBenchmark, RunRequest, Runtime, RuntimeOutput,
};

/// A venv as uv lays one out, whose base interpreter is the fake server, and a
/// model directory that passes validation. On Windows the venv's `python.exe`
/// is an empty file, so a cell that spawns it instead of the base interpreter
/// fails.
fn request(dir: &std::path::Path, benchmark: BenchmarkDefinition) -> anyhow::Result<RunRequest> {
    let venv = dir.join("venv");
    let python = pipette_venv::venv_python(&venv);
    std::fs::create_dir_all(python.parent().context("interpreter has no parent")?)?;
    let home = dir.join("cpython");
    std::fs::create_dir_all(&home)?;
    std::fs::write(
        venv.join("pyvenv.cfg"),
        format!("home = {}\n", home.display()),
    )?;
    let interpreter = if cfg!(windows) {
        std::fs::write(&python, b"")?;
        home.join("python.exe")
    } else {
        python
    };
    std::fs::copy(env!("CARGO_BIN_EXE_fake_onnx_genai_server"), interpreter)?;
    let model = dir.join("model");
    std::fs::create_dir_all(&model)?;
    std::fs::write(
        model.join("genai_config.json"),
        r#"{"model": {"decoder": {"filename": "model.onnx"}}}"#,
    )?;
    std::fs::write(model.join("model.onnx"), b"graph")?;

    let runtime: Runtime = serde_json::from_value(serde_json::json!({
        "type": "uv_onnx_genai", "server_version": "0.16.0",
        "python_version": "3.11", "execution_provider": "cpu",
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

/// Each timing kind measures five reps off the server, gated before the
/// server starts and before each rep, and records the server's peak and the
/// script it ran.
#[rstest]
#[case::prefill("prefill_throughput_64", |benchmark_id| Ok(BenchmarkDefinition::PrefillThroughput(PrefillThroughputBenchmark {
    benchmark_id,
    parameter_prefill_tokens: NonNegativeCount::try_new(64)?,
    parameter_batch_size: BatchSize::one(),
})))]
#[case::decode("decode_throughput_64_16", |benchmark_id| Ok(BenchmarkDefinition::DecodeThroughput(DecodeThroughputBenchmark {
    benchmark_id,
    parameter_prefill_tokens: NonNegativeCount::try_new(64)?,
    parameter_decode_tokens: PositiveCount::try_new(16)?,
    parameter_batch_size: BatchSize::one(),
})))]
#[case::end_to_end("end_to_end_latency_64_16", |benchmark_id| Ok(BenchmarkDefinition::EndToEndLatency(EndToEndLatencyBenchmark {
    benchmark_id,
    parameter_prefill_tokens: NonNegativeCount::try_new(64)?,
    parameter_decode_tokens: PositiveCount::try_new(16)?,
    parameter_batch_size: BatchSize::one(),
})))]
fn a_timing_cell_runs_against_the_server(
    #[case] id: &str,
    #[case] build: fn(BenchmarkId) -> anyhow::Result<BenchmarkDefinition>,
) -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let req = request(dir.path(), build(BenchmarkId::try_new(id.to_owned())?)?)?;
    let (gates, reps) = (Cell::new(0), Cell::new(0));
    let observer = MeasurementObserver {
        on_before_rep_start: &|| {
            reps.set(reps.get() + 1);
            Ok(())
        },
        on_after_rep_end: &|| Ok(()),
        ..MeasurementObserver::QUIET
    };
    let gate = || {
        gates.set(gates.get() + 1);
        Ok(())
    };
    let response = pipette_onnx_genai::run(&req, &gate, observer, None)?;

    let ms = match response.result_data {
        BenchmarkResultData::PrefillThroughput {
            prefill_time_ms, ..
        } => prefill_time_ms,
        BenchmarkResultData::DecodeThroughput { decode_time_ms, .. } => decode_time_ms,
        BenchmarkResultData::EndToEndLatency { total_time_ms, .. } => total_time_ms,
        other @ (BenchmarkResultData::MaxMemoryUsage { .. }
        | BenchmarkResultData::Eval { .. }
        | BenchmarkResultData::VlThroughput { .. }
        | BenchmarkResultData::AsrThroughput { .. }
        | BenchmarkResultData::VlSpecDecode { .. }) => {
            anyhow::bail!("not a timing result: {other:?}")
        }
    };
    assert!(ms > 0.0, "{ms}");
    assert_eq!(reps.get(), 5);
    assert_eq!(gates.get(), 6, "one gate before the server, one per rep");
    // The server's own streams, which the fake leaves empty: no rep lines.
    assert_eq!(
        response.runtime_output,
        RuntimeOutput {
            runtime_stdout: Some(String::new()),
            runtime_stderr: Some(String::new()),
        }
    );
    assert!(
        response.memory.max_host_bytes.is_some(),
        "{:?}",
        response.memory
    );
    assert_eq!(
        response.command.get(1).map(String::as_str),
        Some("pipette_onnx_genai_server.py")
    );
    Ok(())
}

/// A single token is the prefill's own, with no decode step after it, so the
/// cell is refused before the gate and before the server loads the model.
#[test]
fn a_single_decode_token_is_refused_before_the_server_starts() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let req = request(
        dir.path(),
        BenchmarkDefinition::DecodeThroughput(DecodeThroughputBenchmark {
            benchmark_id: BenchmarkId::try_new("decode_throughput_64_1".to_owned())?,
            parameter_prefill_tokens: NonNegativeCount::try_new(64)?,
            parameter_decode_tokens: PositiveCount::try_new(1)?,
            parameter_batch_size: BatchSize::one(),
        }),
    )?;
    let gated = Cell::new(false);
    let gate = || {
        gated.set(true);
        Ok(())
    };
    let observer = MeasurementObserver::QUIET;
    let Err(err) = pipette_onnx_genai::run(&req, &gate, observer, None) else {
        anyhow::bail!("a decode cell under two tokens was measured");
    };
    assert!(format!("{err:#}").contains("at least 2"), "{err:#}");
    assert!(!gated.get(), "refused only after the gate");
    Ok(())
}

/// A cell that fails once the server is up stops it, says how it ended and
/// keeps its cause: the worker reads a cancelled run off that chain.
#[test]
fn a_failure_after_the_start_names_the_server_and_keeps_its_cause() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let req = request(
        dir.path(),
        BenchmarkDefinition::PrefillThroughput(PrefillThroughputBenchmark {
            benchmark_id: BenchmarkId::try_new("prefill_throughput_64".to_owned())?,
            parameter_prefill_tokens: NonNegativeCount::try_new(64)?,
            parameter_batch_size: BatchSize::one(),
        }),
    )?;
    let gates = Cell::new(0);
    // Open before the server starts, shut at the first rep.
    let gate = || -> anyhow::Result<()> {
        gates.set(gates.get() + 1);
        if gates.get() == 1 {
            Ok(())
        } else {
            Err(std::io::Error::from(std::io::ErrorKind::Interrupted).into())
        }
    };
    let observer = MeasurementObserver::QUIET;
    let Err(err) = pipette_onnx_genai::run(&req, &gate, observer, None) else {
        anyhow::bail!("expected the shut gate to fail the cell");
    };
    let interrupted = err.chain().any(|cause| {
        cause
            .downcast_ref::<std::io::Error>()
            .is_some_and(|err| err.kind() == std::io::ErrorKind::Interrupted)
    });
    assert!(interrupted, "{err:#}");
    assert!(
        format!("{err:#}").contains("pipette_onnx_genai_server.py was still running"),
        "{err:#}"
    );
    Ok(())
}

/// The fixture's config names no provider, which declares CPU: on a CPU install
/// the cell reads the measured server's peak, its observation spans the warm-up
/// server too, and it records the script by name; on a CUDA one it is refused
/// when it binds.
#[rstest]
#[case::declared_provider(OnnxExecutionProvider::Cpu)]
#[case::another_provider(OnnxExecutionProvider::Cuda)]
fn a_memory_cell_runs_only_on_a_provider_the_model_declares(
    #[case] provider: OnnxExecutionProvider,
) -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let mut req = request(
        dir.path(),
        BenchmarkDefinition::MaxMemoryUsage(MaxMemoryUsageBenchmark {
            benchmark_id: BenchmarkId::try_new("max_memory_usage_128".to_owned())?,
            parameter_prefill_tokens: NonNegativeCount::try_new(128)?,
        }),
    )?;
    let Runtime::UvOnnxGenai(runtime) = &mut req.runtime.bound else {
        anyhow::bail!("the fixture's runtime is uv_onnx_genai");
    };
    runtime.execution_provider = provider.clone();
    let observer = MeasurementObserver::QUIET;
    let result = pipette_onnx_genai::run(&req, &|| Ok(()), observer, None);
    match (provider, result) {
        (OnnxExecutionProvider::Cpu, Ok(response)) => {
            let BenchmarkResultData::MaxMemoryUsage {
                max_host_bytes,
                max_gpu_bytes: None,
                max_npu_bytes: None,
            } = response.result_data
            else {
                anyhow::bail!("not a host-only memory result: {:?}", response.result_data);
            };
            assert!(max_host_bytes > 0);
            assert!(
                response.memory.max_host_bytes >= Some(max_host_bytes),
                "{:?}",
                response.memory
            );
            assert_eq!(
                response.command.get(1).map(String::as_str),
                Some("pipette_onnx_genai_server.py")
            );
        }
        (OnnxExecutionProvider::Cuda, Err(err)) => {
            assert!(format!("{err:#}").contains("provider mismatch"), "{err:#}");
        }
        (provider, Ok(_)) => anyhow::bail!("{provider}: expected a provider mismatch"),
        (provider, Err(err)) => anyhow::bail!("{provider}: {err:#}"),
    }
    Ok(())
}

/// A memory cell on a provider whose device memory this OS cannot read is
/// refused before a server starts: on macOS a GPU provider, whose GPU the
/// footprint cannot show, and on Linux one that picks its own device, which
/// may be an NPU nothing reads there. The interpreter here cannot run, so a
/// cell that spawned one first would fail with another error.
#[cfg(not(target_os = "windows"))]
#[test]
fn an_unreadable_provider_memory_cell_is_refused_before_a_server_starts() -> anyhow::Result<()> {
    let (provider, watch) = if cfg!(target_os = "macos") {
        (
            OnnxExecutionProvider::Cuda,
            pipette_memprobe::Watch::PeakWithGpu,
        )
    } else {
        (
            OnnxExecutionProvider::Custom("qnn".to_owned()),
            pipette_memprobe::Watch::PeakWithAccelerators,
        )
    };
    let dir = tempfile::tempdir()?;
    let mut req = request(
        dir.path(),
        BenchmarkDefinition::MaxMemoryUsage(MaxMemoryUsageBenchmark {
            benchmark_id: BenchmarkId::try_new("max_memory_usage_128".to_owned())?,
            parameter_prefill_tokens: NonNegativeCount::try_new(128)?,
        }),
    )?;
    // Declares CUDA, so the provider check passes; a custom one skips it.
    std::fs::write(
        dir.path().join("model").join("genai_config.json"),
        r#"{"model": {"decoder": {"filename": "model.onnx",
            "session_options": {"provider_options": [{"cuda": {}}]}}}}"#,
    )?;
    let Runtime::UvOnnxGenai(runtime) = &mut req.runtime.bound else {
        anyhow::bail!("the fixture's runtime is uv_onnx_genai");
    };
    runtime.execution_provider = provider;
    // Still executable, so a spawn fails on the missing interpreter rather
    // than running an empty file through the shell.
    std::fs::write(
        pipette_venv::venv_python(&dir.path().join("venv")),
        "#!/nonexistent/interpreter\n",
    )?;
    let observer = MeasurementObserver::QUIET;
    let Err(err) = pipette_onnx_genai::run(&req, &|| Ok(()), observer, None) else {
        anyhow::bail!("a memory cell ran on a provider this OS cannot read");
    };
    let expected = pipette_memprobe::UnsupportedError { watch };
    assert_eq!(
        err.downcast_ref::<pipette_memprobe::WatchError>(),
        Some(&expected.into()),
        "{err:#}"
    );
    Ok(())
}
