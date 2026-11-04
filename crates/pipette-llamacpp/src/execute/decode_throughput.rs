//! Decode throughput: `llama-bench` generating past an untimed prompt in the
//! KV cache, once per rep; a batch is `llama-batched-bench` generating on every
//! sequence at once.

use std::num::NonZeroUsize;
use std::path::Path;
use std::time::Duration;

use anyhow::Context;

use pipette_memprobe::{Sampler, SupportedWatch};
use pipette_ops::measurement::{
    self, DecodeMeasurementPerPromptReading, DecodeMeasurementReading,
    DecodeMeasurementTotalsReading, MeasurementObserver, MeasurementWarmup,
};
use pipette_ops::readiness::ReadinessGate;
use pipette_ops::run_response::RunResponseBuilder;
use pipette_plan_types::{
    DecodeThroughputBenchmark, DecodeThroughputLlamacppCliStockToolsGgufText,
    LlamacppCliStockToolsExtraRuntimeConfiguration, PositiveCount, PositiveMillis, RunResponse,
    RuntimeFlags,
};
use pipette_subprocess::start_process;

use crate::models::ResolvedGgufTextModel;
use crate::runtimes::{self, ResolvedLlamaBatchedBench, ResolvedLlamaBench};
use crate::{batched_bench, bench};

pub fn run(
    benchmark: &DecodeThroughputBenchmark,
    runtime_flags: DecodeThroughputLlamacppCliStockToolsGgufText,
    program: &ResolvedLlamaBench,
    model: &ResolvedGgufTextModel,
    readiness_gate: ReadinessGate,
    observer: MeasurementObserver<'_>,
    client_config: Option<&LlamacppCliStockToolsExtraRuntimeConfiguration>,
) -> anyhow::Result<RunResponse> {
    let runtime_flags = DecodeThroughputLlamacppCliStockToolsGgufText {
        load_mode: runtimes::get_load_mode(runtime_flags.mmap, runtime_flags.load_mode),
        ..runtime_flags
    };
    let launch = bench_launch(benchmark, &runtime_flags, program, model, client_config);
    let mut runs = Vec::new();
    let result = measurement::decode(
        benchmark,
        NonZeroUsize::try_from(measurement::REPS)?,
        readiness_gate,
        observer,
        // llama-bench warms up on one token at depth 0, so a machine's first run
        // of a deep shape builds its attention kernel cold inside the timed decode.
        MeasurementWarmup::Once,
        &mut runs,
        |runs| {
            let command = launch.build();
            let mut llama_bench = start_process("llama-bench", command, Duration::ZERO)
                .map_err(anyhow::Error::from)?;
            let sampler = Sampler::start(llama_bench.pid(), SupportedWatch::OBSERVE)
                .map_err(anyhow::Error::from)?;
            let (memory, _) = llama_bench
                .wait_success_before_reap(bench::TIMING_TIMEOUT, || sampler.stop())
                .map_err(anyhow::Error::from)?;
            let stdout = llama_bench.output().runtime_stdout.unwrap_or_default();
            let row =
                bench::row(&stdout).map_err(|err| err.context(llama_bench.stop_with_tail()))?;
            runs.push((llama_bench, memory));
            Ok(DecodeMeasurementReading::PerPrompt(vec![
                DecodeMeasurementPerPromptReading::from_millis(
                    row.n_depth,
                    PositiveCount::try_from(row.n_gen).context("invalid n_gen")?,
                    row.avg_ns / 1_000_000.0,
                )
                .map_err(anyhow::Error::from)?,
            ]))
        },
    );
    let response = RunResponseBuilder::new(result?).runtime_flags(
        RuntimeFlags::DecodeThroughputLlamacppCliStockToolsGgufText(runtime_flags),
    );
    Ok(runs
        .iter_mut()
        .fold(response, |response, (run, memory)| {
            response.process(run).memory(*memory)
        })
        .build())
}

/// A batch: `llama-batched-bench` generating on `parameter_batch_size`
/// sequences in lockstep past their untimed prompts, once per rep, read as
/// their totals and the one time they share.
pub fn run_batched(
    benchmark: &DecodeThroughputBenchmark,
    runtime_flags: DecodeThroughputLlamacppCliStockToolsGgufText,
    program: &ResolvedLlamaBatchedBench,
    model: &ResolvedGgufTextModel,
    readiness_gate: ReadinessGate,
    observer: MeasurementObserver<'_>,
    client_config: Option<&LlamacppCliStockToolsExtraRuntimeConfiguration>,
) -> anyhow::Result<RunResponse> {
    // The totals carry a positive prompt count, and the tool's prompt rate
    // over an empty prompt can be 0/0, which prints as `nan`.
    anyhow::ensure!(
        benchmark.parameter_prefill_tokens.into_inner() > 0,
        "{} decodes {} sequences from an empty cache, and llama-batched-bench needs a prompt \
         to report a batch",
        benchmark.benchmark_id,
        benchmark.parameter_batch_size,
    );
    let runtime_flags = DecodeThroughputLlamacppCliStockToolsGgufText {
        load_mode: runtimes::get_load_mode(runtime_flags.mmap, runtime_flags.load_mode),
        ..runtime_flags
    };
    let launch = batched_bench_launch(benchmark, &runtime_flags, program, model, client_config)?;
    let mut runs = Vec::new();
    let result = measurement::decode(
        benchmark,
        NonZeroUsize::try_from(measurement::REPS)?,
        readiness_gate,
        observer,
        MeasurementWarmup::Inline,
        &mut runs,
        |runs| {
            let command = launch.build();
            let mut llama_batched_bench =
                start_process("llama-batched-bench", command, Duration::ZERO)
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
            Ok(DecodeMeasurementReading::Totals(
                DecodeMeasurementTotalsReading {
                    prefill_tokens: PositiveCount::try_from(row.pl.saturating_mul(row.pp))
                        .context("invalid pl x pp")?,
                    decode_tokens: PositiveCount::try_from(row.pl.saturating_mul(row.tg))
                        .context("invalid pl x tg")?,
                    decode_time_ms: PositiveMillis::try_new(row.t_tg * 1000.0)
                        .context("invalid t_tg")?,
                },
            ))
        },
    );
    let response = RunResponseBuilder::new(result?).runtime_flags(
        RuntimeFlags::DecodeThroughputLlamacppCliStockToolsGgufText(runtime_flags),
    );
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
    benchmark: &DecodeThroughputBenchmark,
    flags: &DecodeThroughputLlamacppCliStockToolsGgufText,
    program: &ResolvedLlamaBench,
    model: &Path,
    client_config: Option<&LlamacppCliStockToolsExtraRuntimeConfiguration>,
) -> bench::BenchCommandBuilder {
    let workload = bench::BenchWorkload::Generation {
        n_gen: benchmark
            .parameter_decode_tokens
            .into_inner()
            .unsigned_abs(),
        n_depth: benchmark
            .parameter_prefill_tokens
            .into_inner()
            .unsigned_abs(),
    };
    let DecodeThroughputLlamacppCliStockToolsGgufText {
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
}

/// Every field of the cell is named: a flag plan-types adds fails to compile
/// here until this cell renders it or sets it aside.
fn batched_bench_launch(
    benchmark: &DecodeThroughputBenchmark,
    flags: &DecodeThroughputLlamacppCliStockToolsGgufText,
    program: &ResolvedLlamaBatchedBench,
    model: &Path,
    client_config: Option<&LlamacppCliStockToolsExtraRuntimeConfiguration>,
) -> anyhow::Result<batched_bench::BatchedBenchCommandBuilder> {
    let workload = batched_bench::BatchedBenchWorkload {
        n_prompt: benchmark
            .parameter_prefill_tokens
            .into_inner()
            .unsigned_abs(),
        n_gen: benchmark
            .parameter_decode_tokens
            .into_inner()
            .unsigned_abs(),
        batch_size: benchmark.parameter_batch_size,
    };
    let DecodeThroughputLlamacppCliStockToolsGgufText {
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
    use pipette_plan_types::{
        BatchSize, BenchmarkId, LlamacppFlashAttention, LlamacppLoadMode, NonNegativeCount,
        PositiveCount,
    };

    use super::*;
    use crate::runtimes::{test_llama_batched_bench, test_llama_bench};

    fn bench_args(
        benchmark: &DecodeThroughputBenchmark,
        flags: &DecodeThroughputLlamacppCliStockToolsGgufText,
    ) -> Vec<String> {
        bench_launch(
            benchmark,
            flags,
            &test_llama_bench(),
            Path::new("model.gguf"),
            None,
        )
        .args()
    }

    fn batched_bench_args(
        benchmark: &DecodeThroughputBenchmark,
        flags: &DecodeThroughputLlamacppCliStockToolsGgufText,
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
    fn runtime_flags() -> DecodeThroughputLlamacppCliStockToolsGgufText {
        DecodeThroughputLlamacppCliStockToolsGgufText {
            threads: Some(3),
            number_gpu_layers: Some(7),
            mmap: None,
            load_mode: Some(LlamacppLoadMode::DirectIo),
            flash_attention: Some(LlamacppFlashAttention::On),
            envs: Vec::new(),
            raw: vec!["--poll".to_owned(), "0".to_owned()],
        }
    }

    fn benchmark() -> anyhow::Result<DecodeThroughputBenchmark> {
        Ok(DecodeThroughputBenchmark {
            benchmark_id: BenchmarkId::try_new("decode_throughput_512_128".to_owned())?,
            parameter_prefill_tokens: NonNegativeCount::try_new(512)?,
            parameter_decode_tokens: PositiveCount::try_new(128)?,
            parameter_batch_size: BatchSize::try_new(4)?,
        })
    }

    #[test]
    fn llama_bench_generates_at_the_prompts_depth() -> anyhow::Result<()> {
        assert_eq!(
            bench_args(&benchmark()?, &runtime_flags()).join(" "),
            "--flash-attn on --load-mode dio --n-gpu-layers 7 --poll 0 --repetitions 1 --threads 3 --n-prompt 0 --n-gen 128 --n-depth 512"
        );
        Ok(())
    }

    #[test]
    fn llama_batched_bench_runs_every_sequence() -> anyhow::Result<()> {
        assert_eq!(
            batched_bench_args(&benchmark()?, &runtime_flags())?.join(" "),
            "--flash-attn on --load-mode dio --n-gpu-layers 7 --poll 0 --threads 3 --ctx-size 2560 -npp 512 -ntg 128 -npl 4,4"
        );
        Ok(())
    }
}
