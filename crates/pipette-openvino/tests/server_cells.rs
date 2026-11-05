//! Every cell end to end through [`pipette_openvino::run`] against
//! `fake_openvino_server`, copied to where the venv's base interpreter would be:
//! server start, readiness, prompt sizing through `/tokenize`, the requests,
//! the checks, the reduction and the response.

use std::cell::Cell;

use anyhow::Context;
use rstest::rstest;

use pipette_ops::measurement::MeasurementObserver;
use pipette_plan_types::{
    AbsolutePath, BatchSize, BenchmarkDefinition, BenchmarkId, BenchmarkResultData, DeclaredBound,
    DecodeThroughputBenchmark, EndToEndLatencyBenchmark, MaxMemoryUsageBenchmark, Model,
    ModelSource, ModelType, NonNegativeCount, Openvino, OpenvinoDevice, PositiveCount,
    PrefillThroughputBenchmark, RunRequest, Runtime, RuntimeFlagRef, RuntimeFlags, RuntimeType,
};

/// How the fake's one stderr line starts.
const LISTENING: &str = "fake_openvino_server: listening on http://127.0.0.1:";

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
    std::fs::copy(env!("CARGO_BIN_EXE_fake_openvino_server"), interpreter)?;
    let model = dir.join("model");
    std::fs::create_dir_all(&model)?;
    [
        "openvino_model.xml",
        "openvino_model.bin",
        "openvino_tokenizer.xml",
        "openvino_detokenizer.xml",
    ]
    .iter()
    .try_for_each(|file| std::fs::write(model.join(file), b"ir"))?;

    let runtime: Runtime = serde_json::from_value(serde_json::json!({
        "type": "uv_openvino", "server_version": "2026.4.1",
        "python_version": "3.11",
        "source": {"type": "absolute_preinstalled", "dir": venv}
    }))?;
    let mut flags = RuntimeFlagRef::new(
        benchmark.benchmark_type(),
        RuntimeType::UvOpenvino,
        ModelType::Openvino,
    );
    flags.device = Some(OpenvinoDevice::Cpu);
    Ok(RunRequest {
        runtime: DeclaredBound::already_bound(runtime),
        model: DeclaredBound::already_bound(Model::Openvino(Openvino {
            source: ModelSource::AbsoluteDir {
                dir: AbsolutePath::try_new(model.to_string_lossy().into_owned())?,
            },
        })),
        runtime_flags: Some(RuntimeFlags::try_from(flags)?),
        model_flags: None,
        benchmark_flags: None,
        benchmark,
    })
}

/// Each timing kind measures five reps off the server, gated before the
/// server starts and before each rep, and records the script it ran, the
/// flags the cell resolved and the memory the server held.
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
    };
    let gate = || {
        gates.set(gates.get() + 1);
        Ok(())
    };
    let response = pipette_openvino::run(&req, &gate, observer, &dir.path().join("cache"), None)?;

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
    // The server's own streams, the fake's one line: no rep lines.
    let output = &response.runtime_output;
    assert_eq!(output.runtime_stdout.as_deref(), Some(""));
    let stderr = output.runtime_stderr.as_deref().unwrap_or_default();
    assert!(
        stderr.starts_with(LISTENING) && stderr.lines().count() == 1,
        "{stderr}"
    );
    assert!(response.runtime_flags.is_some());
    assert_eq!(
        response.command.get(1).map(String::as_str),
        Some("pipette_openvino_server.py")
    );
    assert!(
        response.memory.max_host_bytes.is_some(),
        "{:?}",
        response.memory
    );
    Ok(())
}

/// A single token is the prefill's own, with no decode step after it, so the
/// cell is refused before the gate and before the server compiles the
/// pipeline.
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
    let observer = MeasurementObserver {
        on_before_rep_start: &|| Ok(()),
        on_after_rep_end: &|| Ok(()),
    };
    let Err(err) = pipette_openvino::run(&req, &gate, observer, &dir.path().join("cache"), None)
    else {
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
    let observer = MeasurementObserver {
        on_before_rep_start: &|| Ok(()),
        on_after_rep_end: &|| Ok(()),
    };
    let Err(err) = pipette_openvino::run(&req, &gate, observer, &dir.path().join("cache"), None)
    else {
        anyhow::bail!("expected the shut gate to fail the cell");
    };
    let interrupted = err.chain().any(|cause| {
        cause
            .downcast_ref::<std::io::Error>()
            .is_some_and(|err| err.kind() == std::io::ErrorKind::Interrupted)
    });
    assert!(interrupted, "{err:#}");
    assert!(
        format!("{err:#}").contains("pipette_openvino_server.py was still running"),
        "{err:#}"
    );
    Ok(())
}

/// Off Windows nothing reads the NPU's memory, so its memory cell is refused
/// before a server starts. The interpreter here cannot run, so a cell that
/// spawned one first would fail with another error.
#[cfg(not(target_os = "windows"))]
#[test]
fn an_npu_memory_cell_is_refused_before_a_server_starts() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let mut req = request(
        dir.path(),
        BenchmarkDefinition::MaxMemoryUsage(MaxMemoryUsageBenchmark {
            benchmark_id: BenchmarkId::try_new("max_memory_usage_128".to_owned())?,
            parameter_prefill_tokens: NonNegativeCount::try_new(128)?,
        }),
    )?;
    let mut flags = req.runtime_flags_ref()?;
    flags.device = Some(OpenvinoDevice::Npu);
    req.runtime_flags = Some(RuntimeFlags::try_from(flags)?);
    std::fs::write(pipette_venv::venv_python(&dir.path().join("venv")), b"")?;
    let observer = MeasurementObserver {
        on_before_rep_start: &|| Ok(()),
        on_after_rep_end: &|| Ok(()),
    };
    let Err(err) =
        pipette_openvino::run(&req, &|| Ok(()), observer, &dir.path().join("cache"), None)
    else {
        anyhow::bail!("an NPU memory cell ran off Windows");
    };
    let refused = err.downcast_ref::<pipette_memprobe::WatchError>();
    let expected = pipette_memprobe::UnsupportedError {
        watch: pipette_memprobe::Watch::PeakWithNpu,
    };
    assert_eq!(refused, Some(&expected.into()), "{err:#}");
    Ok(())
}

/// The measurement is the measured server's peak; the observation also covers
/// the warm-up server that compiled first, so it is never below it. The fake
/// refuses to run beside another copy, so the warm-up server must be gone
/// before the measured one starts. The script is recorded by name, as on the
/// timing cells.
#[test]
fn a_memory_cell_reads_the_servers_peak() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let req = request(
        dir.path(),
        BenchmarkDefinition::MaxMemoryUsage(MaxMemoryUsageBenchmark {
            benchmark_id: BenchmarkId::try_new("max_memory_usage_128".to_owned())?,
            parameter_prefill_tokens: NonNegativeCount::try_new(128)?,
        }),
    )?;
    let observer = MeasurementObserver {
        on_before_rep_start: &|| Ok(()),
        on_after_rep_end: &|| Ok(()),
    };
    let response =
        pipette_openvino::run(&req, &|| Ok(()), observer, &dir.path().join("cache"), None)?;
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
        response
            .memory
            .max_host_bytes
            .is_some_and(|observed| observed >= max_host_bytes),
        "{max_host_bytes} against {:?}",
        response.memory
    );
    assert_eq!(
        response.command.get(1).map(String::as_str),
        Some("pipette_openvino_server.py")
    );
    Ok(())
}

/// A failure once a server is up quotes that server's output around the
/// cell's whole error, the entry's marker inside, so a reading the entry
/// refuses is quoted as well. On Linux a `gpu` cell's sampler finds no GPU
/// memory on the fake, which fails the warm-up.
#[cfg(target_os = "linux")]
#[test]
fn a_memory_cell_failure_quotes_the_servers_tail() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let mut req = request(
        dir.path(),
        BenchmarkDefinition::MaxMemoryUsage(MaxMemoryUsageBenchmark {
            benchmark_id: BenchmarkId::try_new("max_memory_usage_128".to_owned())?,
            parameter_prefill_tokens: NonNegativeCount::try_new(128)?,
        }),
    )?;
    let mut flags = req.runtime_flags_ref()?;
    flags.device = Some(OpenvinoDevice::Gpu);
    req.runtime_flags = Some(RuntimeFlags::try_from(flags)?);
    let observer = MeasurementObserver {
        on_before_rep_start: &|| Ok(()),
        on_after_rep_end: &|| Ok(()),
    };
    let Err(err) =
        pipette_openvino::run(&req, &|| Ok(()), observer, &dir.path().join("cache"), None)
    else {
        anyhow::bail!("a gpu cell read GPU memory off the fake server");
    };
    let message = format!("{err:#}");
    let quoted = format!("pipette_openvino_server.py was still running\nstderr tail:\n{LISTENING}");
    assert!(message.starts_with(&quoted), "{message}");
    assert!(
        message.contains("\nerror: max_memory_usage_128 warm-up: "),
        "{message}"
    );
    let sampled = err.chain().any(|cause| {
        cause
            .downcast_ref::<pipette_memprobe::PeakError>()
            .is_some()
    });
    assert!(sampled, "{message}");
    Ok(())
}
