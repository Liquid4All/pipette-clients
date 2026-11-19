//! Prefill throughput: `llama-bench` asked for the prompt alone, once per rep;
//! a batch is `llama-batched-bench` prefilling every sequence's prompt at once.

use std::num::NonZeroUsize;
use std::path::Path;
use std::time::{Duration, Instant};

use anyhow::Context;

use pipette_memprobe::{Sampler, SupportedWatch};
use pipette_ops::measurement::{
    self, MeasurementObserver, MeasurementWarmup, PrefillMeasurementPerPromptReading,
    PrefillMeasurementReading, PrefillMeasurementTotalsReading,
};
use pipette_ops::readiness::ReadinessGate;
use pipette_ops::run_response::RunResponseBuilder;
use pipette_plan_types::{
    LlamaCppFlavor, LlamacppCliStockToolsExtraRuntimeConfiguration, PositiveCount, PositiveMillis,
    PrefillThroughputBenchmark, PrefillThroughputLlamacppCliStockToolsGgufText, RunResponse,
    RuntimeFlags,
};
use pipette_subprocess::{OnStop, Process, Stream, start_process};

use crate::models::ResolvedGgufTextModel;
use crate::runtimes::{self, ResolvedLlamaBatchedBench, ResolvedLlamaBench};
use crate::{batched_bench, bench};

/// Where a prefill rep's warm-up runs. Each rep is its own llama-bench process.
/// A GPU or NPU build sets up its device at the process's first batch, so each
/// process warms itself with llama-bench's own pass over the prompt; a second
/// llama-bench test in the same process instead runs 7-9B models out of memory
/// on an 8 GB Orin Nano. A CPU-only build has nothing per process to set up,
/// and its own pass doubles each rep's time: one untimed process before the reps.
fn warmup(flavor: &LlamaCppFlavor) -> MeasurementWarmup {
    if runtimes::cpu_only(flavor) {
        MeasurementWarmup::Once
    } else {
        MeasurementWarmup::Inline
    }
}

/// Follows a GPU or NPU rep through llama-bench's `--progress` lines: logs how
/// long the model load and the untimed pass took, and starts the timed pass as
/// an attempt of its own, so the load with the untimed pass and the timed pass
/// each get the whole limit. A process that exits or is stopped first prints
/// no such line, and the rep stays one attempt.
fn follow_passes(
    llama_bench: &Process,
    started: Instant,
    rep: usize,
    observer: MeasurementObserver<'_>,
) {
    let printed = |marker: &str| {
        llama_bench
            .wait_for_output(Stream::Stderr, time_left(started), |stderr| {
                stderr.contains(marker)
            })
            .is_some()
    };
    if !printed(bench::UNTIMED_PROMPT_PASS) {
        return;
    }
    let loaded = started.elapsed();
    tracing::info!(
        rep,
        load_secs = loaded.as_secs_f64(),
        "llama-bench loaded the model; its untimed pass started"
    );
    if !printed(bench::TIMED_PROMPT_PASS) {
        return;
    }
    tracing::info!(
        rep,
        untimed_pass_secs = started.elapsed().saturating_sub(loaded).as_secs_f64(),
        "llama-bench's untimed pass ended; its timed pass started"
    );
    (observer.on_attempt_start)(&format!("rep {rep}, the timed pass"));
}

/// What is left of a rep's [`bench::TIMING_TIMEOUT`], shared by the waits on
/// its progress lines and on its exit.
fn time_left(started: Instant) -> Duration {
    bench::TIMING_TIMEOUT.saturating_sub(started.elapsed())
}

#[expect(
    clippy::too_many_arguments,
    reason = "the request's parts, and the seams its entry takes"
)]
pub fn run(
    benchmark: &PrefillThroughputBenchmark,
    runtime_flags: PrefillThroughputLlamacppCliStockToolsGgufText,
    flavor: &LlamaCppFlavor,
    program: &ResolvedLlamaBench,
    model: &ResolvedGgufTextModel,
    readiness_gate: ReadinessGate,
    observer: MeasurementObserver<'_>,
    client_config: Option<&LlamacppCliStockToolsExtraRuntimeConfiguration>,
) -> anyhow::Result<RunResponse> {
    let runtime_flags = PrefillThroughputLlamacppCliStockToolsGgufText {
        load_mode: runtimes::get_load_mode(runtime_flags.mmap, runtime_flags.load_mode),
        ..runtime_flags
    };
    let warmup = warmup(flavor);
    let launch = bench_launch(
        benchmark,
        &runtime_flags,
        program,
        model,
        client_config,
        warmup,
    );
    let mut runs = Vec::new();
    let mut rep = 0;
    let result = measurement::prefill(
        benchmark,
        NonZeroUsize::try_from(measurement::REPS)?,
        readiness_gate,
        observer,
        warmup,
        &mut runs,
        |runs| {
            let command = launch.build();
            let started = Instant::now();
            let mut llama_bench = start_process("llama-bench", command, OnStop::SIGTERM)
                .map_err(anyhow::Error::from)?;
            let sampler = Sampler::start(llama_bench.pid(), SupportedWatch::OBSERVE)
                .map_err(anyhow::Error::from)?;
            rep += 1;
            if warmup == MeasurementWarmup::Inline {
                follow_passes(&llama_bench, started, rep, observer);
            }
            let (memory, _) = llama_bench
                .wait_success_before_reap(time_left(started), || sampler.stop())
                .map_err(anyhow::Error::from)?;
            let stdout = llama_bench.output().runtime_stdout.unwrap_or_default();
            let row =
                bench::row(&stdout).map_err(|err| err.context(llama_bench.stop_with_tail()))?;
            runs.push((llama_bench, memory));
            Ok(PrefillMeasurementReading::PerPrompt(vec![
                PrefillMeasurementPerPromptReading::from_millis(
                    PositiveCount::try_from(row.n_prompt).context("invalid n_prompt")?,
                    row.avg_ns / 1_000_000.0,
                )
                .map_err(anyhow::Error::from)?,
            ]))
        },
    );
    let response = RunResponseBuilder::new(result?)
        .runtime_flags(RuntimeFlags::PrefillThroughputLlamacppCliStockToolsGgufText(runtime_flags));
    Ok(runs
        .iter_mut()
        .fold(response, |response, (run, memory)| {
            response.process(run).memory(*memory)
        })
        .build())
}

/// A batch: `llama-batched-bench` prefilling `parameter_batch_size` prompts
/// together, once per rep, read as their totals and the one time they share.
pub fn run_batched(
    benchmark: &PrefillThroughputBenchmark,
    runtime_flags: PrefillThroughputLlamacppCliStockToolsGgufText,
    program: &ResolvedLlamaBatchedBench,
    model: &ResolvedGgufTextModel,
    readiness_gate: ReadinessGate,
    observer: MeasurementObserver<'_>,
    client_config: Option<&LlamacppCliStockToolsExtraRuntimeConfiguration>,
) -> anyhow::Result<RunResponse> {
    let runtime_flags = PrefillThroughputLlamacppCliStockToolsGgufText {
        load_mode: runtimes::get_load_mode(runtime_flags.mmap, runtime_flags.load_mode),
        ..runtime_flags
    };
    let launch = batched_bench_launch(benchmark, &runtime_flags, program, model, client_config)?;
    let mut runs = Vec::new();
    let result = measurement::prefill(
        benchmark,
        NonZeroUsize::try_from(measurement::REPS)?,
        readiness_gate,
        observer,
        MeasurementWarmup::Inline,
        &mut runs,
        |runs| {
            let command = launch.build();
            let mut llama_batched_bench =
                start_process("llama-batched-bench", command, OnStop::SIGTERM)
                    .map_err(anyhow::Error::from)?;
            let sampler = Sampler::start(llama_batched_bench.pid(), SupportedWatch::OBSERVE)
                .map_err(anyhow::Error::from)?;
            let (memory, _) = llama_batched_bench
                .wait_success_before_reap(bench::TIMING_TIMEOUT, || sampler.stop())
                .map_err(anyhow::Error::from)?;
            let stdout = llama_batched_bench
                .output()
                .runtime_stdout
                .unwrap_or_default();
            let row = batched_bench::row(&stdout)
                .map_err(|err| err.context(llama_batched_bench.stop_with_tail()))?;
            runs.push((llama_batched_bench, memory));
            Ok(PrefillMeasurementReading::Totals(
                PrefillMeasurementTotalsReading {
                    prefill_tokens: PositiveCount::try_from(row.pl.saturating_mul(row.pp))
                        .context("invalid pl x pp")?,
                    prefill_time_ms: PositiveMillis::try_new(row.t_pp * 1000.0)
                        .context("invalid t_pp")?,
                },
            ))
        },
    );
    let response = RunResponseBuilder::new(result?)
        .runtime_flags(RuntimeFlags::PrefillThroughputLlamacppCliStockToolsGgufText(runtime_flags));
    Ok(runs
        .iter_mut()
        .fold(response, |response, (run, memory)| {
            response.process(run).memory(*memory)
        })
        .build())
}

/// Every field of the cell is named: a flag plan-types adds fails to compile
/// here until this cell renders it or sets it aside.
fn bench_launch(
    benchmark: &PrefillThroughputBenchmark,
    flags: &PrefillThroughputLlamacppCliStockToolsGgufText,
    program: &ResolvedLlamaBench,
    model: &Path,
    client_config: Option<&LlamacppCliStockToolsExtraRuntimeConfiguration>,
    warmup: MeasurementWarmup,
) -> bench::BenchCommandBuilder {
    let workload = bench::BenchWorkload::Prompt {
        n_prompt: benchmark
            .parameter_prefill_tokens
            .into_inner()
            .unsigned_abs(),
    };
    let PrefillThroughputLlamacppCliStockToolsGgufText {
        threads,
        number_gpu_layers,
        mmap,
        load_mode,
        flash_attention,
        envs,
        raw,
    } = flags.clone();
    bench::BenchCommandBuilder::new(program.clone(), model, workload)
        .threads(threads)
        .number_gpu_layers(number_gpu_layers)
        .mmap(mmap)
        .load_mode(load_mode)
        .flash_attention(flash_attention)
        .raw(raw)
        .envs(envs)
        .client_config(client_config.cloned())
        .no_warmup(warmup == MeasurementWarmup::Once)
        .progress(warmup == MeasurementWarmup::Inline)
}

/// Every field of the cell is named: a flag plan-types adds fails to compile
/// here until this cell renders it or sets it aside.
fn batched_bench_launch(
    benchmark: &PrefillThroughputBenchmark,
    flags: &PrefillThroughputLlamacppCliStockToolsGgufText,
    program: &ResolvedLlamaBatchedBench,
    model: &Path,
    client_config: Option<&LlamacppCliStockToolsExtraRuntimeConfiguration>,
) -> anyhow::Result<batched_bench::BatchedBenchCommandBuilder> {
    let workload = batched_bench::BatchedBenchWorkload {
        n_prompt: benchmark
            .parameter_prefill_tokens
            .into_inner()
            .unsigned_abs(),
        // One generated token, after the prefill's time stops: with none, the
        // tool's generation rate can be 0/0, which prints as `nan`.
        n_gen: 1,
        batch_size: benchmark.parameter_batch_size,
    };
    let PrefillThroughputLlamacppCliStockToolsGgufText {
        threads,
        number_gpu_layers,
        mmap,
        load_mode,
        flash_attention,
        envs,
        raw,
    } = flags.clone();
    Ok(
        batched_bench::BatchedBenchCommandBuilder::new(program.clone(), model, workload)?
            .threads(threads)
            .number_gpu_layers(number_gpu_layers)
            .mmap(mmap)
            .load_mode(load_mode)
            .flash_attention(flash_attention)
            .raw(raw)
            .envs(envs)
            .client_config(client_config.cloned()),
    )
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use pipette_plan_types::{
        BatchSize, BenchmarkId, LlamacppFlashAttention, LlamacppLoadMode, NonNegativeCount,
    };

    use super::*;
    use crate::runtimes::{test_llama_batched_bench, test_llama_bench};

    fn bench_args(
        benchmark: &PrefillThroughputBenchmark,
        flags: &PrefillThroughputLlamacppCliStockToolsGgufText,
        flavor: &LlamaCppFlavor,
    ) -> Vec<String> {
        bench_launch(
            benchmark,
            flags,
            &test_llama_bench(),
            Path::new("model.gguf"),
            None,
            warmup(flavor),
        )
        .args()
    }

    fn batched_bench_args(
        benchmark: &PrefillThroughputBenchmark,
        flags: &PrefillThroughputLlamacppCliStockToolsGgufText,
    ) -> anyhow::Result<Vec<String>> {
        Ok(batched_bench_launch(
            benchmark,
            flags,
            &test_llama_batched_bench(),
            Path::new("model.gguf"),
            None,
        )?
        .args())
    }

    /// Every knob set to its own value, so a knob that lands on another's flag shows.
    fn runtime_flags() -> PrefillThroughputLlamacppCliStockToolsGgufText {
        PrefillThroughputLlamacppCliStockToolsGgufText {
            threads: Some(3),
            number_gpu_layers: Some(7),
            mmap: None,
            load_mode: Some(LlamacppLoadMode::DirectIo),
            flash_attention: Some(LlamacppFlashAttention::On),
            envs: Vec::new(),
            raw: vec!["--poll".to_owned(), "0".to_owned()],
        }
    }

    fn benchmark() -> anyhow::Result<PrefillThroughputBenchmark> {
        prompt_of(512)
    }

    fn prompt_of(tokens: i32) -> anyhow::Result<PrefillThroughputBenchmark> {
        Ok(PrefillThroughputBenchmark {
            benchmark_id: BenchmarkId::try_new(format!("prefill_throughput_{tokens}"))?,
            parameter_prefill_tokens: NonNegativeCount::try_new(tokens)?,
            parameter_batch_size: BatchSize::try_new(4)?,
        })
    }

    fn with_raw(raw: &[&str]) -> PrefillThroughputLlamacppCliStockToolsGgufText {
        PrefillThroughputLlamacppCliStockToolsGgufText {
            raw: raw.iter().map(|&token| token.to_owned()).collect(),
            ..runtime_flags()
        }
    }

    /// A CPU-only build reads the prompt once per process after a warm-up
    /// process; any other build warms each process with llama-bench's own pass.
    #[rstest]
    #[case::cpu(
        LlamaCppFlavor::LinuxX64Cpu,
        512,
        &["--poll", "0"],
        "--flash-attn on --load-mode dio --n-gpu-layers 7 --no-warmup --poll 0 --repetitions 1 --threads 3 --n-prompt 512 --n-gen 0"
    )]
    #[case::cpu_past_a_batch(
        LlamaCppFlavor::AndroidArm64Cpu,
        4096,
        &["--poll", "0"],
        "--flash-attn on --load-mode dio --n-gpu-layers 7 --no-warmup --poll 0 --repetitions 1 --threads 3 --n-prompt 4096 --n-gen 0"
    )]
    #[case::gpu(
        LlamaCppFlavor::LinuxArm64Cuda13,
        512,
        &["--poll", "0"],
        "--flash-attn on --load-mode dio --n-gpu-layers 7 --poll 0 --progress --repetitions 1 --threads 3 --n-prompt 512 --n-gen 0"
    )]
    #[case::gpu_past_a_batch(
        LlamaCppFlavor::WindowsX64Vulkan,
        4096,
        &["--poll", "0"],
        "--flash-attn on --load-mode dio --n-gpu-layers 7 --poll 0 --progress --repetitions 1 --threads 3 --n-prompt 4096 --n-gen 0"
    )]
    #[case::npu_past_its_batch(
        LlamaCppFlavor::AndroidArm64Snapdragon,
        4096,
        &["--device", "HTP0", "-ub", "1024"],
        "--device HTP0 --flash-attn on --load-mode dio --n-gpu-layers 7 --progress --repetitions 1 --threads 3 -ub 1024 --n-prompt 4096 --n-gen 0"
    )]
    fn llama_bench_reads_the_prompt_alone(
        #[case] flavor: LlamaCppFlavor,
        #[case] tokens: i32,
        #[case] raw: &[&str],
        #[case] expected: &str,
    ) -> anyhow::Result<()> {
        assert_eq!(
            bench_args(&prompt_of(tokens)?, &with_raw(raw), &flavor).join(" "),
            expected
        );
        Ok(())
    }

    #[rstest]
    #[case::cpu(LlamaCppFlavor::LinuxX64Cpu, MeasurementWarmup::Once)]
    #[case::gpu(LlamaCppFlavor::MacosArm64, MeasurementWarmup::Inline)]
    #[case::npu(LlamaCppFlavor::AndroidArm64Snapdragon, MeasurementWarmup::Inline)]
    fn only_a_cpu_build_warms_in_a_process_of_its_own(
        #[case] flavor: LlamaCppFlavor,
        #[case] expected: MeasurementWarmup,
    ) {
        assert_eq!(warmup(&flavor), expected);
    }

    #[test]
    fn llama_batched_bench_prefills_every_sequence() -> anyhow::Result<()> {
        assert_eq!(
            batched_bench_args(&benchmark()?, &runtime_flags())?.join(" "),
            "--flash-attn on --load-mode dio --n-gpu-layers 7 --poll 0 --threads 3 --ctx-size 2052 -npp 512 -ntg 1 -npl 4,4"
        );
        Ok(())
    }
}
