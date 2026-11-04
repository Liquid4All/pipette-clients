//! Pins what a `llama-bench` cell, or a batch's `llama-batched-bench` one,
//! keeps: the shape's argv with each knob on its own flag, a rep per
//! invocation after a decode's warm-up one (one for peak memory, after its
//! warm-up one), the mean of the readings, the first argv and the last
//! streams.
#![cfg(unix)]

mod common;

use std::cell::Cell;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};

use anyhow::Context;

use common::{
    assert_distinct_knobs, fake_model, fake_run_request, fake_runtime, ignore_reps, no_corpus,
    with_distinct_knobs,
};

use pipette_ops::measurement::MeasurementObserver;
use pipette_plan_types::{
    BatchSize, BenchmarkDefinition, BenchmarkId, BenchmarkResultData, DecodeThroughputBenchmark,
    LlamacppLoadMode, NonNegativeCount, PositiveCount, PrefillThroughputBenchmark, RunRequest,
    RuntimeFlagRef,
};

/// Invocation `n` reports `n` ms for each test it was asked for, as
/// `llama-bench` does: one per non-zero `--n-prompt`, one per non-zero
/// `--n-gen` and one per `-pg`. It says which invocation it was on stderr and
/// appends its arguments to `args`.
const BENCH_STUB: &str = r#"#!/bin/sh
count="$(dirname "$0")/count"
n=$(( $(cat "$count" 2>/dev/null || echo 0) + 1 ))
echo "$n" > "$count"
echo "$*" >> "$(dirname "$0")/args"
p=0; g=0; d=0; pg=""
while [ $# -gt 0 ]; do
  case "$1" in
    --n-prompt) p=$2; shift ;;
    --n-gen) g=$2; shift ;;
    --n-depth) d=$2; shift ;;
    -pg) pg="$pg $2"; shift ;;
  esac
  shift
done
echo "invocation $n" >&2
rows=""
row() {
  rows="$rows${rows:+, }$(printf '{"avg_ns": %d, "n_prompt": %s, "n_gen": %s, "n_depth": %s}' $((n * 1000000)) "$1" "$2" "$d")"
}
[ "$p" -gt 0 ] && row "$p" 0
[ "$g" -gt 0 ] && row 0 "$g"
for shape in $pg; do row "${shape%,*}" "${shape#*,}"; done
echo "[$rows]"
"#;

/// Invocation `n` prints a row per `-npl` value that fits `--ctx-size`, as
/// `llama-batched-bench` does: a slow first one, then `n` ms of prefill and
/// `10n` ms of decode. It says which invocation it was on stderr and appends
/// its arguments to `args`.
const BATCHED_STUB: &str = r#"#!/bin/sh
count="$(dirname "$0")/count"
n=$(( $(cat "$count" 2>/dev/null || echo 0) + 1 ))
echo "$n" > "$count"
echo "$*" >> "$(dirname "$0")/args"
pp=0; tg=0; pl=""; ctx=0
while [ $# -gt 0 ]; do
  case "$1" in
    -npp) pp=$2; shift ;;
    -ntg) tg=$2; shift ;;
    -npl) pl=$2; shift ;;
    --ctx-size) ctx=$2; shift ;;
  esac
  shift
done
echo "invocation $n" >&2
row=0
for b in $(echo "$pl" | tr ',' ' '); do
  row=$((row + 1))
  t_pp="0.00$n"; t_tg="0.0${n}0"
  [ $row -eq 1 ] && t_pp=9 && t_tg=9
  [ $((b * (pp + tg))) -le "$ctx" ] || continue
  printf '{"pp": %s, "tg": %s, "pl": %s, "t_pp": %s, "t_tg": %s}\n' "$pp" "$tg" "$b" "$t_pp" "$t_tg"
done
echo
"#;

const FAILING_STUB: &str = r#"#!/bin/sh
i=1
while [ $i -le 100 ]; do echo "load line $i" >&2; i=$((i + 1)); done
exit 3
"#;

/// Exits 0 with the rows of two tests, a 512-token prompt alone and
/// `-pg 512,1`, after a long log.
const TWO_ROW_STUB: &str = r#"#!/bin/sh
i=1
while [ $i -le 100 ]; do echo "load line $i" >&2; i=$((i + 1)); done
echo '[{"avg_ns": 1000000, "n_prompt": 512, "n_gen": 0, "n_depth": 0},
{"avg_ns": 9000000, "n_prompt": 512, "n_gen": 1, "n_depth": 0}]'
"#;

/// Exits 0 with the row of a 512-token prompt alone, after a long log.
#[cfg(any(target_os = "macos", target_os = "linux"))]
const PROMPT_ROW_STUB: &str = r#"#!/bin/sh
i=1
while [ $i -le 100 ]; do echo "load line $i" >&2; i=$((i + 1)); done
echo '[{"avg_ns": 1000000, "n_prompt": 512, "n_gen": 0, "n_depth": 0}]'
"#;

/// Exits 0 having printed one row, after a long log.
const ONE_ROW_STUB: &str = r#"#!/bin/sh
i=1
while [ $i -le 100 ]; do echo "load line $i" >&2; i=$((i + 1)); done
echo '{"pp": 512, "tg": 1, "pl": 4, "t_pp": 0.004, "t_tg": 0.001}'
"#;

/// Ends a stub by filling a [`ALLOCATED`] buffer, so the figure read as it
/// exits is its own: on Linux one no larger than the test process could be the
/// test process's image, and counts for nothing.
const ALLOCATE: &str = "exec dd if=/dev/zero of=/dev/null bs=67108864 count=1 2>/dev/null\n";

/// What [`ALLOCATE`] fills.
const ALLOCATED: u64 = 64 << 20;

/// Ends a [`BENCH_STUB`] by filling [`WARM_UP_ALLOCATED`] on its first
/// invocation and [`ALLOCATED`] on later ones.
#[cfg(any(target_os = "macos", target_os = "linux"))]
const ALLOCATE_MORE_FIRST: &str = r#"[ "$n" -eq 1 ] && exec dd if=/dev/zero of=/dev/null bs=268435456 count=1 2>/dev/null
exec dd if=/dev/zero of=/dev/null bs=67108864 count=1 2>/dev/null
"#;

/// What [`ALLOCATE_MORE_FIRST`]'s first invocation fills, more than [`ALLOCATED`].
#[cfg(any(target_os = "macos", target_os = "linux"))]
const WARM_UP_ALLOCATED: u64 = 256 << 20;

/// Starts a stub that sleeps 4 s before the rest of the script.
#[cfg(any(target_os = "macos", target_os = "linux"))]
const SLOW: &str = "#!/bin/sh\nsleep 4\n";

/// A deadline a [`SLOW`] run fits in once, with 2 s to spare under load, and
/// two runs miss by 2 s.
#[cfg(any(target_os = "macos", target_os = "linux"))]
const SLOW_RUN_DEADLINE: u64 = 6;

fn install_stub(root: &Path, tool: &str, script: &str) -> anyhow::Result<PathBuf> {
    let install = root.join("runtime-install");
    std::fs::create_dir_all(&install)?;
    let program = install.join(tool);
    std::fs::write(&program, script)?;
    std::fs::set_permissions(&program, std::fs::Permissions::from_mode(0o755))?;
    Ok(install)
}

fn prefill(batch_size: u32) -> anyhow::Result<BenchmarkDefinition> {
    Ok(BenchmarkDefinition::PrefillThroughput(
        PrefillThroughputBenchmark {
            benchmark_id: BenchmarkId::try_new(format!("prefill_throughput_512_bs{batch_size}"))?,
            parameter_prefill_tokens: NonNegativeCount::try_new(512)?,
            parameter_batch_size: BatchSize::try_new(batch_size)?,
        },
    ))
}

fn decode(
    prefill_tokens: i32,
    decode_tokens: i32,
    batch_size: u32,
) -> anyhow::Result<BenchmarkDefinition> {
    Ok(BenchmarkDefinition::DecodeThroughput(
        DecodeThroughputBenchmark {
            benchmark_id: BenchmarkId::try_new(format!(
                "decode_throughput_{prefill_tokens}_{decode_tokens}_bs{batch_size}"
            ))?,
            parameter_prefill_tokens: NonNegativeCount::try_new(prefill_tokens)?,
            parameter_decode_tokens: PositiveCount::try_new(decode_tokens)?,
            parameter_batch_size: BatchSize::try_new(batch_size)?,
        },
    ))
}

#[cfg(any(target_os = "macos", target_os = "linux"))]
fn max_memory(prefill_tokens: i32) -> anyhow::Result<BenchmarkDefinition> {
    Ok(BenchmarkDefinition::MaxMemoryUsage(
        pipette_plan_types::MaxMemoryUsageBenchmark {
            benchmark_id: BenchmarkId::try_new(format!("max_memory_usage_{prefill_tokens}"))?,
            parameter_prefill_tokens: NonNegativeCount::try_new(prefill_tokens)?,
        },
    ))
}

fn request(
    root: &Path,
    tool: &str,
    benchmark: BenchmarkDefinition,
    script: &str,
) -> anyhow::Result<RunRequest> {
    let model_path = root.join("fake-model.gguf");
    std::fs::write(&model_path, b"fake-model-bytes")?;
    with_distinct_knobs(fake_run_request(
        fake_runtime(&install_stub(root, tool, script)?)?,
        fake_model(&model_path)?,
        benchmark,
        None,
    ))
}

/// Invocation `n`'s reading is `n` times `ms`: llama-bench's one row, or a
/// batch's second, since its first row is the warm-up and is not averaged in.
/// A llama-bench decode first runs the same command once as its warm-up,
/// which is neither gated nor averaged in.
#[rstest::rstest]
#[case::prefill(
    "llama-bench",
    BENCH_STUB,
    prefill(1),
    &["--repetitions 1", "--n-prompt 512 --n-gen 0"],
    5,
    1.0,
    "\"avg_ns\": 5000000"
)]
#[case::decode(
    "llama-bench",
    BENCH_STUB,
    decode(512, 128, 1),
    &["--repetitions 1", "--n-prompt 0 --n-gen 128 --n-depth 512"],
    6,
    1.0,
    "\"avg_ns\": 6000000"
)]
#[case::decode_without_a_prompt(
    "llama-bench",
    BENCH_STUB,
    decode(0, 32, 1),
    &["--repetitions 1", "--n-prompt 0 --n-gen 32 --n-depth 0"],
    6,
    1.0,
    "\"avg_ns\": 6000000"
)]
#[case::batched_prefill(
    "llama-batched-bench",
    BATCHED_STUB,
    prefill(4),
    &["--output-format jsonl", "--ctx-size 2052 -npp 512 -ntg 1 -npl 4,4"],
    5,
    1.0,
    "\"t_pp\": 0.005"
)]
#[case::batched_decode(
    "llama-batched-bench",
    BATCHED_STUB,
    decode(512, 128, 4),
    &["--output-format jsonl", "--ctx-size 2560 -npp 512 -ntg 128 -npl 4,4"],
    5,
    10.0,
    "\"t_tg\": 0.050"
)]
fn a_timing_cell_measures_one_invocation_per_rep(
    #[case] tool: &str,
    #[case] script: &str,
    #[case] benchmark: anyhow::Result<BenchmarkDefinition>,
    #[case] shape: &[&str],
    #[case] invocations: u32,
    #[case] ms: f64,
    #[case] last_stdout: &str,
) -> anyhow::Result<()> {
    let root = tempfile::tempdir()?;
    let req = request(
        root.path(),
        tool,
        benchmark?,
        &format!("{script}{ALLOCATE}"),
    )?;

    let (gated, started, finished) = (Cell::new(0), Cell::new(0), Cell::new(0));
    let gate = || {
        gated.set(gated.get() + 1);
        Ok(())
    };
    let observer = MeasurementObserver {
        on_before_rep_start: &|| {
            started.set(started.get() + 1);
            Ok(())
        },
        on_after_rep_end: &|| {
            finished.set(finished.get() + 1);
            Ok(())
        },
    };
    let response = pipette_llamacpp::run(&req, &no_corpus(root.path()), &gate, observer, None)?;

    // Five gated reps; a decode's warm-up invocation is not one.
    assert_eq!((gated.get(), started.get(), finished.get()), (5, 5, 5));
    // Each exits before the observation's first tick, so only a sampler
    // stopped before the reap reads it.
    assert!(
        response
            .memory
            .max_host_bytes
            .is_some_and(|bytes| bytes >= ALLOCATED),
        "{:?}",
        response.memory
    );
    let (mean, stddev) = match response.result_data {
        BenchmarkResultData::PrefillThroughput {
            prefill_time_ms,
            prefill_time_ms_stddev,
        } => (prefill_time_ms, prefill_time_ms_stddev),
        BenchmarkResultData::DecodeThroughput {
            decode_time_ms,
            decode_time_ms_stddev,
        } => (decode_time_ms, decode_time_ms_stddev),
        other @ (BenchmarkResultData::EndToEndLatency { .. }
        | BenchmarkResultData::MaxMemoryUsage { .. }
        | BenchmarkResultData::Eval { .. }
        | BenchmarkResultData::VlThroughput { .. }
        | BenchmarkResultData::AsrThroughput { .. }
        | BenchmarkResultData::VlSpecDecode { .. }) => {
            anyhow::bail!("not a timing result: {other:?}")
        }
    };
    // The reps are the last five invocations, each reading narrowed to an f32.
    let expected = f64::from(invocations - 2) * ms;
    assert!((mean - expected).abs() < 1e-3, "{mean}");
    let stddev = stddev.unwrap_or_default();
    assert!((stddev - 2.5_f64.sqrt() * ms).abs() < 1e-3, "{stddev}");

    let install = root.path().join("runtime-install");
    let program = install.join(tool);
    assert_eq!(response.executable, Some(program.display().to_string()));
    // The warm-up runs the reps' own command.
    let args = std::fs::read_to_string(install.join("args"))?;
    let runs: Vec<&str> = args.lines().collect();
    assert_eq!(runs.len(), usize::try_from(invocations)?, "{args}");
    assert!(runs.iter().all(|run| Some(run) == runs.first()), "{args}");
    assert_distinct_knobs(&req.benchmark, &response.command)?;
    let argv = response.command.join(" ");
    assert!(shape.iter().all(|part| argv.contains(part)), "{argv}");
    // The cell left the load mode open, so the weights are pinned in RAM.
    assert!(argv.contains("--load-mode none"), "{argv}");
    let flags = RuntimeFlagRef::from(response.runtime_flags.context("no flags recorded")?);
    assert_eq!(flags.load_mode, Some(LlamacppLoadMode::Read));
    let output = &response.runtime_output;
    let stdout = output.runtime_stdout.as_deref().unwrap_or_default();
    assert!(stdout.contains(last_stdout), "{stdout}");
    assert_eq!(
        output.runtime_stderr.as_deref(),
        Some(format!("invocation {invocations}\n").as_str())
    );
    Ok(())
}

/// A batched decode from an empty cache is refused by name before the tool
/// runs.
#[test]
fn a_batched_decode_without_a_prompt_is_refused() -> anyhow::Result<()> {
    let root = tempfile::tempdir()?;
    let req = request(
        root.path(),
        "llama-batched-bench",
        decode(0, 128, 4)?,
        BATCHED_STUB,
    )?;
    let Err(err) = pipette_llamacpp::run(
        &req,
        &no_corpus(root.path()),
        &|| Ok(()),
        ignore_reps(),
        None,
    ) else {
        anyhow::bail!("expected the cell to be refused");
    };
    let message = format!("{err:#}");
    assert!(message.contains("decode_throughput_0_128_bs4"), "{message}");
    assert!(
        !root.path().join("runtime-install").join("count").exists(),
        "the tool ran"
    );
    Ok(())
}

/// A peak-memory cell samples one test per invocation, a prefill and one
/// decode step on it, with the weights pinned in RAM where the cell left the
/// load mode open, and records that overlay.
#[cfg(any(target_os = "macos", target_os = "linux"))]
#[test]
fn a_peak_memory_cell_samples_one_test() -> anyhow::Result<()> {
    use pipette_plan_types::RuntimeFlags;

    let root = tempfile::tempdir()?;
    let req = request(
        root.path(),
        "llama-bench",
        max_memory(512)?,
        &format!("{BENCH_STUB}{ALLOCATE}"),
    )?;
    let response = pipette_llamacpp::run(
        &req,
        &no_corpus(root.path()),
        &|| Ok(()),
        ignore_reps(),
        None,
    )?;

    let BenchmarkResultData::MaxMemoryUsage {
        max_host_bytes,
        max_gpu_bytes,
        ..
    } = response.result_data
    else {
        anyhow::bail!("not a peak-memory result: {:?}", response.result_data);
    };
    assert!(max_host_bytes >= ALLOCATED, "{max_host_bytes}");
    assert_eq!(max_gpu_bytes, None);
    assert_distinct_knobs(&req.benchmark, &response.command)?;
    let argv = response.command.join(" ");
    assert!(argv.contains("--n-prompt 0 --n-gen 0 -pg 512,1"), "{argv}");
    assert!(argv.contains("--load-mode none"), "{argv}");
    let Some(RuntimeFlags::MaxMemoryUsageLlamacppCliStockToolsGgufText(flags)) =
        response.runtime_flags
    else {
        anyhow::bail!(
            "not a peak-memory cell's flags: {:?}",
            response.runtime_flags
        );
    };
    assert_eq!(flags.load_mode, Some(LlamacppLoadMode::Read));
    Ok(())
}

/// A failure once the tool has run quotes its output around the cell's whole
/// error, the entry's marker inside, so a reading the entry refuses is quoted
/// as well. A Linux GPU build's sampler finds no GPU memory on the stub, so the
/// warm-up fails.
#[cfg(target_os = "linux")]
#[test]
fn a_peak_memory_failure_quotes_the_tools_tail() -> anyhow::Result<()> {
    use pipette_plan_types::{DeclaredBound, LlamaCppFlavor, Runtime};

    let root = tempfile::tempdir()?;
    let mut req = request(root.path(), "llama-bench", max_memory(512)?, BENCH_STUB)?;
    let Runtime::LlamacppCliStockTools(mut tools) =
        fake_runtime(&root.path().join("runtime-install"))?
    else {
        anyhow::bail!("the fake runtime is not llama.cpp's");
    };
    tools.flavor = LlamaCppFlavor::LinuxX64Vulkan;
    req.runtime = DeclaredBound::already_bound(Runtime::LlamacppCliStockTools(tools));

    let Err(err) = pipette_llamacpp::run(
        &req,
        &no_corpus(root.path()),
        &|| Ok(()),
        ignore_reps(),
        None,
    ) else {
        anyhow::bail!("a GPU build read GPU memory off the stub");
    };
    let message = format!("{err:#}");
    assert!(
        message.starts_with("llama-bench exited with exit status: 0\nstderr tail:\ninvocation 1\n"),
        "{message}"
    );
    assert!(
        message.contains("\nerror: max_memory_usage_512 warm-up: "),
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

/// A flavor whose peak cannot be watched is refused before anything runs,
/// the warm-up included. An operator's build is refused on every OS. This
/// tool cannot start, so a cell that started it first would fail with another
/// error.
#[cfg(any(target_os = "macos", target_os = "linux"))]
#[test]
fn a_refused_flavor_starts_no_tool() -> anyhow::Result<()> {
    use pipette_plan_types::{DeclaredBound, LlamaCppFlavor, Runtime};

    let root = tempfile::tempdir()?;
    let mut req = request(
        root.path(),
        "llama-bench",
        max_memory(512)?,
        "#!/nonexistent/interpreter\n",
    )?;
    let Runtime::LlamacppCliStockTools(mut tools) =
        fake_runtime(&root.path().join("runtime-install"))?
    else {
        anyhow::bail!("the fake runtime is not llama.cpp's");
    };
    tools.flavor = LlamaCppFlavor::Custom("my-build".to_owned());
    req.runtime = DeclaredBound::already_bound(Runtime::LlamacppCliStockTools(tools));

    let Err(err) = pipette_llamacpp::run(
        &req,
        &no_corpus(root.path()),
        &|| Ok(()),
        ignore_reps(),
        None,
    ) else {
        anyhow::bail!("an operator's build was measured");
    };
    let message = format!("{err:#}");
    assert!(message.contains("is an operator's build"), "{message}");
    Ok(())
}

/// A cell that sets both `mmap` and `load_mode` is refused before any tool
/// starts: its tool would fail on the interpreter.
#[test]
fn a_cell_setting_both_load_knobs_starts_no_tool() -> anyhow::Result<()> {
    use pipette_plan_types::{LlamacppLoadMode, RuntimeFlags};

    let root = tempfile::tempdir()?;
    let mut req = request(
        root.path(),
        "llama-bench",
        max_memory(512)?,
        "#!/nonexistent/interpreter\n",
    )?;
    let mut flags = req.runtime_flags_ref()?;
    flags.mmap = Some(false);
    flags.load_mode = Some(LlamacppLoadMode::Mmap);
    req.runtime_flags = Some(RuntimeFlags::try_from(flags)?);

    let Err(err) = pipette_llamacpp::run(
        &req,
        &no_corpus(root.path()),
        &|| Ok(()),
        ignore_reps(),
        None,
    ) else {
        anyhow::bail!("a cell with two load modes was measured");
    };
    let message = format!("{err:#}");
    assert!(
        message.contains("both `mmap` (false) and `load_mode` (mmap)"),
        "{message}"
    );
    Ok(())
}

/// A peak-memory cell first runs the measured command once and drops that
/// run's reading: the result is the measured run's, below the warm-up's larger
/// peak, and the observation covers both.
#[cfg(any(target_os = "macos", target_os = "linux"))]
#[test]
fn a_peak_memory_cell_warms_up_on_its_command() -> anyhow::Result<()> {
    let root = tempfile::tempdir()?;
    let req = request(
        root.path(),
        "llama-bench",
        max_memory(512)?,
        &format!("{BENCH_STUB}{ALLOCATE_MORE_FIRST}"),
    )?;
    let response = pipette_llamacpp::run(
        &req,
        &no_corpus(root.path()),
        &|| Ok(()),
        ignore_reps(),
        None,
    )?;

    let BenchmarkResultData::MaxMemoryUsage { max_host_bytes, .. } = response.result_data else {
        anyhow::bail!("not a peak-memory result: {:?}", response.result_data);
    };
    assert!(max_host_bytes >= ALLOCATED, "{max_host_bytes}");
    assert!(max_host_bytes < WARM_UP_ALLOCATED, "{max_host_bytes}");
    assert!(
        response
            .memory
            .max_host_bytes
            .is_some_and(|bytes| bytes >= WARM_UP_ALLOCATED),
        "{:?}",
        response.memory
    );
    let args = std::fs::read_to_string(root.path().join("runtime-install").join("args"))?;
    let runs: Vec<&str> = args.lines().collect();
    assert_eq!(runs.len(), 2, "{args}");
    assert_eq!(runs.first(), runs.last(), "{args}");
    assert_eq!(
        response.runtime_output.runtime_stderr.as_deref(),
        Some("invocation 2\n")
    );
    Ok(())
}

/// The error says it was the warm-up.
#[cfg(any(target_os = "macos", target_os = "linux"))]
#[test]
fn a_warm_up_that_cannot_start_fails_the_peak_memory_cell() -> anyhow::Result<()> {
    let root = tempfile::tempdir()?;
    let req = request(
        root.path(),
        "llama-bench",
        max_memory(512)?,
        "#!/nonexistent/interpreter\n",
    )?;
    let Err(err) = pipette_llamacpp::run(
        &req,
        &no_corpus(root.path()),
        &|| Ok(()),
        ignore_reps(),
        None,
    ) else {
        anyhow::bail!("expected the cell to fail");
    };
    let message = format!("{err:#}");
    assert!(
        message.contains("max_memory_usage_512 warm-up: failed to spawn llama-bench"),
        "{message}"
    );
    Ok(())
}

/// Two runs that each fit the deadline pass where their sum would not.
#[cfg(any(target_os = "macos", target_os = "linux"))]
#[test]
fn the_warm_up_and_the_measured_run_each_get_the_deadline() -> anyhow::Result<()> {
    use pipette_plan_types::{BenchmarkFlags, PositiveSeconds};

    let root = tempfile::tempdir()?;
    let mut req = request(
        root.path(),
        "llama-bench",
        max_memory(512)?,
        &format!("{SLOW}{BENCH_STUB}{ALLOCATE}"),
    )?;
    req.benchmark_flags = Some(
        BenchmarkFlags::MaxMemoryUsageLlamacppCliStockToolsGgufText {
            bench_timeout_seconds: Some(PositiveSeconds::try_new(SLOW_RUN_DEADLINE)?),
        },
    );
    let response = pipette_llamacpp::run(
        &req,
        &no_corpus(root.path()),
        &|| Ok(()),
        ignore_reps(),
        None,
    )?;
    assert_eq!(
        response.runtime_output.runtime_stderr.as_deref(),
        Some("invocation 2\n")
    );
    Ok(())
}

/// A failed invocation says why and quotes the end of its stderr, once. A
/// peak-memory cell's failed warm-up fails it before the measured run.
#[rstest::rstest]
#[case::exits_non_zero(
    "llama-bench",
    prefill(1),
    FAILING_STUB,
    "llama-bench exited with exit status: 3"
)]
#[case::a_decodes_warm_up_exits_non_zero(
    "llama-bench",
    decode(512, 128, 1),
    FAILING_STUB,
    "decode_throughput_512_128_bs1 warm-up: llama-bench exited with exit status: 3"
)]
#[case::prints_a_row_per_test(
    "llama-bench",
    prefill(1),
    TWO_ROW_STUB,
    "llama-bench printed 2 rows"
)]
#[case::prints_one_row_of_a_batch(
    "llama-batched-bench",
    prefill(4),
    ONE_ROW_STUB,
    "from llama-batched-bench (a warm-up and a reading), got 1"
)]
#[cfg_attr(
    any(target_os = "macos", target_os = "linux"),
    case::peak_memory_prints_a_row_per_test(
        "llama-bench",
        max_memory(512),
        TWO_ROW_STUB,
        "llama-bench printed 2 rows"
    )
)]
#[cfg_attr(
    any(target_os = "macos", target_os = "linux"),
    case::peak_memory_prints_another_prompts_row(
        "llama-bench",
        max_memory(256),
        PROMPT_ROW_STUB,
        "engine returned n_prompt 512, expected 256"
    )
)]
#[cfg_attr(
    any(target_os = "macos", target_os = "linux"),
    case::peak_memory_prints_a_row_without_the_decode(
        "llama-bench",
        max_memory(512),
        PROMPT_ROW_STUB,
        "engine returned n_gen 0, expected 1"
    )
)]
#[cfg_attr(
    any(target_os = "macos", target_os = "linux"),
    case::a_peak_memory_warm_up_exits_non_zero(
        "llama-bench",
        max_memory(512),
        FAILING_STUB,
        "max_memory_usage_512 warm-up: llama-bench exited with exit status: 3"
    )
)]
fn a_failed_invocation_quotes_its_output(
    #[case] tool: &str,
    #[case] benchmark: anyhow::Result<BenchmarkDefinition>,
    #[case] script: &str,
    #[case] reason: &str,
) -> anyhow::Result<()> {
    let root = tempfile::tempdir()?;
    let req = request(root.path(), tool, benchmark?, script)?;
    let Err(err) = pipette_llamacpp::run(
        &req,
        &no_corpus(root.path()),
        &|| Ok(()),
        ignore_reps(),
        None,
    ) else {
        anyhow::bail!("expected the cell to fail");
    };
    let message = format!("{err:#}");
    assert!(message.contains(reason), "{message}");
    assert_eq!(message.matches("load line 100").count(), 1, "{message}");
    assert!(!message.contains("load line 1\n"), "{message}");
    Ok(())
}

/// A gate failure between reps quotes no run, not even the successful one
/// before it.
#[test]
fn a_failure_between_reps_quotes_no_run() -> anyhow::Result<()> {
    let root = tempfile::tempdir()?;
    let req = request(root.path(), "llama-bench", prefill(1)?, BENCH_STUB)?;
    let gated = Cell::new(0);
    let gate = || {
        gated.set(gated.get() + 1);
        anyhow::ensure!(gated.get() < 2, "the host did not cool down");
        Ok(())
    };
    let Err(err) = pipette_llamacpp::run(&req, &no_corpus(root.path()), &gate, ignore_reps(), None)
    else {
        anyhow::bail!("expected the cell to fail");
    };
    let message = format!("{err:#}");
    assert!(message.contains("the host did not cool down"), "{message}");
    assert!(!message.contains("llama-bench"), "{message}");
    Ok(())
}
