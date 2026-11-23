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
    VlMaxMemoryBenchmark, VlThroughputBenchmark,
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
    // A text IR and a VL one in one directory, so it passes either check.
    [
        "openvino_model.xml",
        "openvino_model.bin",
        "openvino_tokenizer.xml",
        "openvino_detokenizer.xml",
        "openvino_language_model.xml",
        "openvino_language_model.bin",
        "openvino_text_embeddings_model.xml",
        "openvino_text_embeddings_model.bin",
        "openvino_vision_embeddings_model.xml",
        "openvino_vision_embeddings_model.bin",
    ]
    .iter()
    .try_for_each(|file| std::fs::write(model.join(file), b"ir"))?;
    std::fs::write(model.join("config.json"), r#"{"model_type": "lfm3_vl"}"#)?;
    std::fs::write(
        model.join("preprocessor_config.json"),
        r#"{"patch_size": 16, "merge_size": 2,
            "size": {"shortest_edge": 65536, "longest_edge": 16777216}}"#,
    )?;

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
        ..MeasurementObserver::QUIET
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
        | BenchmarkResultData::VlSpecDecode { .. }
        | BenchmarkResultData::DecisionLatency { .. }
        | BenchmarkResultData::Session { .. }) => {
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

/// A `side` x `side` VL throughput cell of 16 text tokens, `images` images and
/// 8 output tokens.
fn vl_throughput(side: i32, images: i32) -> anyhow::Result<BenchmarkDefinition> {
    Ok(BenchmarkDefinition::VlThroughput(VlThroughputBenchmark {
        benchmark_id: BenchmarkId::try_new(format!(
            "vl_throughput_{side}x{side}_text16_decode8_img{images}"
        ))?,
        parameter_image_width: PositiveCount::try_new(side)?,
        parameter_image_height: PositiveCount::try_new(side)?,
        parameter_text_tokens: NonNegativeCount::try_new(16)?,
        parameter_decode_tokens: PositiveCount::try_new(8)?,
        parameter_num_images: NonNegativeCount::try_new(images)?,
    }))
}

/// The VL cell sends `pipette-llamacpp`'s request to the VL routes and reads
/// the server's counters; its image tokens are the model's rule, 256 for each
/// 512x512 image on LFM3-VL, which the fake's prompt count agrees with. A
/// cell of 0 images sends one, as llama.cpp's does.
#[rstest]
#[case::two_images(2, 10 + 16 + 2 * 258, 512)]
#[case::zero_reads_as_one(0, 10 + 16 + 258, 256)]
fn a_vl_throughput_cell_runs_against_the_server(
    #[case] images: i32,
    #[case] prompt: u32,
    #[case] image_tokens: u32,
) -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let req = request(dir.path(), vl_throughput(512, images)?)?;
    let gates = Cell::new(0);
    let gate = || {
        gates.set(gates.get() + 1);
        Ok(())
    };
    let observer = MeasurementObserver::QUIET;
    let response = pipette_openvino::run(&req, &gate, observer, &dir.path().join("cache"), None)?;
    let BenchmarkResultData::VlThroughput {
        prompt_tokens,
        image_tokens: counted,
        prompt_ms,
        predicted_ms,
        ..
    } = response.result_data
    else {
        anyhow::bail!("not a VL throughput result: {:?}", response.result_data);
    };
    assert_eq!(prompt_tokens, prompt);
    assert_eq!(counted, Some(image_tokens));
    assert!(
        prompt_ms > 0.0 && predicted_ms > 0.0,
        "{prompt_ms} {predicted_ms}"
    );
    assert_eq!(gates.get(), 6, "one gate before the server, one per rep");
    assert_eq!(
        response.command.get(1).map(String::as_str),
        Some("pipette_openvino_server.py")
    );
    Ok(())
}

/// The fake counts every image as a 512x512 one, so at 256x256 its prompt
/// disagrees with the client's rule: the cell fails naming both counts rather
/// than record image tokens the engine did not run.
#[test]
fn a_prompt_the_engine_counts_otherwise_fails_the_cell() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let req = request(dir.path(), vl_throughput(256, 2)?)?;
    let observer = MeasurementObserver::QUIET;
    let Err(err) =
        pipette_openvino::run(&req, &|| Ok(()), observer, &dir.path().join("cache"), None)
    else {
        anyhow::bail!("a prompt the rule does not count was measured");
    };
    let message = format!("{err:#}");
    let (engine, rule) = (10 + 16 + 2 * 258, 10 + 16 + 2 * 66);
    assert!(
        message.contains(&format!("GenAI prompted with {engine} tokens"))
            && message.contains(&format!("rule counts {rule}")),
        "{message}"
    );
    Ok(())
}

/// The VL memory cell runs the text memory cell's two servers, here each
/// serving one image and decoding the one token a cell without an output
/// length asks for.
#[test]
fn a_vl_memory_cell_reads_the_servers_peak() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let req = request(
        dir.path(),
        BenchmarkDefinition::VlMaxMemory(VlMaxMemoryBenchmark {
            benchmark_id: BenchmarkId::try_new("vl_max_memory_256x256_text0_img1".to_owned())?,
            parameter_image_width: PositiveCount::try_new(256)?,
            parameter_image_height: PositiveCount::try_new(256)?,
            parameter_text_tokens: NonNegativeCount::try_new(0)?,
            parameter_decode_tokens: None,
            parameter_num_images: NonNegativeCount::try_new(1)?,
        }),
    )?;
    let observer = MeasurementObserver::QUIET;
    let response =
        pipette_openvino::run(&req, &|| Ok(()), observer, &dir.path().join("cache"), None)?;
    let BenchmarkResultData::MaxMemoryUsage { max_host_bytes, .. } = response.result_data else {
        anyhow::bail!("not a memory result: {:?}", response.result_data);
    };
    assert!(max_host_bytes > 0);
    Ok(())
}

/// A single token is the prefill's own, with no decode step after it, so the
/// cell is refused before the gate and before the server compiles the
/// pipeline.
#[rstest]
#[case::decode("decode_throughput_64_1", |benchmark_id| Ok(BenchmarkDefinition::DecodeThroughput(DecodeThroughputBenchmark {
    benchmark_id,
    parameter_prefill_tokens: NonNegativeCount::try_new(64)?,
    parameter_decode_tokens: PositiveCount::try_new(1)?,
    parameter_batch_size: BatchSize::one(),
})))]
#[case::vl("vl_throughput_256x256_text0_decode1_img1", |benchmark_id| Ok(BenchmarkDefinition::VlThroughput(VlThroughputBenchmark {
    benchmark_id,
    parameter_image_width: PositiveCount::try_new(256)?,
    parameter_image_height: PositiveCount::try_new(256)?,
    parameter_text_tokens: NonNegativeCount::try_new(0)?,
    parameter_decode_tokens: PositiveCount::try_new(1)?,
    parameter_num_images: NonNegativeCount::try_new(1)?,
})))]
fn a_single_decode_token_is_refused_before_the_server_starts(
    #[case] id: &str,
    #[case] build: fn(BenchmarkId) -> anyhow::Result<BenchmarkDefinition>,
) -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let req = request(dir.path(), build(BenchmarkId::try_new(id.to_owned())?)?)?;
    let gated = Cell::new(false);
    let gate = || {
        gated.set(true);
        Ok(())
    };
    let observer = MeasurementObserver::QUIET;
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
    let observer = MeasurementObserver::QUIET;
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
    let observer = MeasurementObserver::QUIET;
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
    let observer = MeasurementObserver::QUIET;
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
    // On Linux the peak counts swap, which the observation carries beside its
    // resident term.
    let observed = response
        .memory
        .max_host_bytes
        .unwrap_or(0)
        .saturating_add(response.memory.max_swap_bytes.unwrap_or(0));
    assert!(
        observed >= max_host_bytes,
        "{max_host_bytes} against {:?}",
        response.memory
    );
    let mut extras = response
        .client_extras
        .keys()
        .map(String::as_str)
        .collect::<Vec<_>>();
    extras.sort_unstable();
    let expected: &[&str] = if cfg!(target_os = "linux") {
        &[
            "major_faults",
            "max_swap_kib",
            "peak_committed_kib",
            "peak_rss_kib",
        ]
    } else {
        &[]
    };
    assert_eq!(extras, expected);
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
    let observer = MeasurementObserver::QUIET;
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
