//! Peak memory for a text model: `llama-bench` runs twice under the flavor's
//! peak sampler ([`crate::memory::peak`]), the first reading dropped, so a build
//! that compiles into a cache (a GPU driver's kernels) is measured on a filled one.

use std::path::Path;
use std::time::Duration;

use pipette_memprobe::Sampler;
use pipette_ops::measurement::{self, MaxMemoryMeasurementReading, MemoryMeasurementWarmup};
use pipette_ops::run_response::RunResponseBuilder;
use pipette_plan_types::{
    BenchmarkFlagRef, LlamaCppFlavor, LlamacppCliStockToolsExtraRuntimeConfiguration,
    MaxMemoryUsageBenchmark, MaxMemoryUsageLlamacppCliStockToolsGgufText, MemoryObservation,
    RunResponse, RuntimeFlags,
};
use pipette_subprocess::{Process, start_process};

use crate::bench;
use crate::memory;
use crate::models::ResolvedGgufTextModel;
use crate::runtimes::{self, ResolvedLlamaBench};
#[cfg(any(target_os = "linux", target_os = "android"))]
use crate::swap_aware;

const DECODE_TOKENS: u32 = 1;

pub fn run(
    benchmark: &MaxMemoryUsageBenchmark,
    runtime_flags: MaxMemoryUsageLlamacppCliStockToolsGgufText,
    benchmark_flags: &BenchmarkFlagRef,
    flavor: &LlamaCppFlavor,
    program: &ResolvedLlamaBench,
    model: &ResolvedGgufTextModel,
    client_config: Option<&LlamacppCliStockToolsExtraRuntimeConfiguration>,
) -> anyhow::Result<RunResponse> {
    let watch = memory::peak(flavor, |name| {
        runtimes::launch_env(name, program, &runtime_flags.envs, client_config)
    })?;
    let runtime_flags = MaxMemoryUsageLlamacppCliStockToolsGgufText {
        load_mode: runtimes::get_load_mode(runtime_flags.mmap, runtime_flags.load_mode),
        ..runtime_flags
    };
    let launch = bench_launch(benchmark, &runtime_flags, program, model, client_config);
    // Per run: a warm-up gets a deadline of its own.
    let deadline = memory::bench_timeout(benchmark_flags.bench_timeout_seconds);
    let mut ran: Option<(Process, MemoryObservation)> = None;
    // The `/proc` terms behind the swap-aware peak, for the row.
    #[cfg(any(target_os = "linux", target_os = "android"))]
    let mut extras = serde_json::Map::new();
    let result = measurement::max_memory(
        &benchmark.benchmark_id,
        MemoryMeasurementWarmup::Once,
        &mut ran,
        |ran| {
            // A run an earlier attempt left has exited; the observation covers both.
            let earlier = ran.take().map(|(_, memory)| memory).unwrap_or_default();
            let command = launch.build();
            let mut llama_bench = start_process("llama-bench", command, Duration::ZERO)
                .map_err(anyhow::Error::from)?;
            let sampler = Sampler::start(llama_bench.pid(), watch).map_err(anyhow::Error::from)?;
            let (memory, peak) = llama_bench
                .wait_success_before_reap(deadline, || sampler.stop())
                .map_err(anyhow::Error::from)?;
            let stdout = llama_bench.output().runtime_stdout.unwrap_or_default();
            // Before the checks, so the cell quotes it on any failure from here.
            **ran = Some((llama_bench, earlier.merge_max(memory)));
            check_row(
                &stdout,
                benchmark
                    .parameter_prefill_tokens
                    .into_inner()
                    .unsigned_abs(),
            )?;
            let peak = peak.map_err(anyhow::Error::from)?;
            // The peak counts swap: zram or swap can hold part of the model mid-run.
            #[cfg(any(target_os = "linux", target_os = "android"))]
            let peak = {
                extras = swap_aware::extras(&peak.footprint);
                peak.counting_swap()
            };
            Ok(MaxMemoryMeasurementReading {
                max_host_bytes: peak.host_bytes,
                max_gpu_bytes: peak.gpu_bytes,
                max_npu_bytes: peak.npu_bytes,
            })
        },
    );
    let result_data = result.map_err(|err| match &mut ran {
        Some((llama_bench, _)) => err.context(llama_bench.stop_with_tail()),
        None => err,
    })?;
    let response = RunResponseBuilder::new(result_data).runtime_flags(
        RuntimeFlags::MaxMemoryUsageLlamacppCliStockToolsGgufText(runtime_flags),
    );
    #[cfg(any(target_os = "linux", target_os = "android"))]
    let response = response.client_extras(extras);
    Ok(ran
        .iter_mut()
        .fold(response, |response, (llama_bench, memory)| {
            response.process(llama_bench).memory(*memory)
        })
        .build())
}

/// The run was the cell's one test. A `raw` flag that adds tests (a `-pg`, a
/// comma list) would put their contexts in the peak too.
fn check_row(stdout: &str, prefill_tokens: u32) -> anyhow::Result<()> {
    let row = bench::row(stdout)?;
    measurement::expect_tokens("n_prompt", row.n_prompt, prefill_tokens)?;
    measurement::expect_tokens("n_gen", row.n_gen, DECODE_TOKENS)
}

/// Every field of the cell is named: a flag plan-types adds fails to compile
/// here until this cell renders it or sets it aside.
fn bench_launch(
    benchmark: &MaxMemoryUsageBenchmark,
    flags: &MaxMemoryUsageLlamacppCliStockToolsGgufText,
    program: &ResolvedLlamaBench,
    model: &Path,
    client_config: Option<&LlamacppCliStockToolsExtraRuntimeConfiguration>,
) -> bench::BenchCommandBuilder {
    let workload = bench::BenchWorkload::PromptThenGeneration {
        n_prompt: benchmark
            .parameter_prefill_tokens
            .into_inner()
            .unsigned_abs(),
        n_gen: DECODE_TOKENS,
    };
    let MaxMemoryUsageLlamacppCliStockToolsGgufText {
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

#[cfg(test)]
mod tests {
    use pipette_plan_types::{
        BenchmarkId, LlamacppFlashAttention, LlamacppLoadMode, NonNegativeCount,
    };

    use super::*;
    use crate::runtimes::test_llama_bench;

    fn bench_args(
        benchmark: &MaxMemoryUsageBenchmark,
        flags: &MaxMemoryUsageLlamacppCliStockToolsGgufText,
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

    /// Every knob set to its own value, so a knob that lands on another's flag shows.
    fn runtime_flags() -> MaxMemoryUsageLlamacppCliStockToolsGgufText {
        MaxMemoryUsageLlamacppCliStockToolsGgufText {
            threads: Some(3),
            number_gpu_layers: Some(7),
            mmap: None,
            load_mode: Some(LlamacppLoadMode::DirectIo),
            flash_attention: Some(LlamacppFlashAttention::On),
            envs: Vec::new(),
            raw: vec!["--poll".to_owned(), "0".to_owned()],
        }
    }

    #[test]
    fn llama_bench_decodes_one_token_on_the_prompt() -> anyhow::Result<()> {
        assert_eq!(
            bench_args(
                &MaxMemoryUsageBenchmark {
                    benchmark_id: BenchmarkId::try_new("max_memory_usage_512".to_owned())?,
                    parameter_prefill_tokens: NonNegativeCount::try_new(512)?,
                },
                &runtime_flags(),
            )
            .join(" "),
            "--flash-attn on --load-mode dio --n-gpu-layers 7 --poll 0 --repetitions 1 --threads 3 --n-prompt 0 --n-gen 0 -pg 512,1"
        );
        Ok(())
    }
}
