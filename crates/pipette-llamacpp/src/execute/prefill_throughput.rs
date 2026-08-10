//! Prefill throughput via `llama-bench` for a prepared [`RunRequest`].

use std::process::Command;

use pipette_ops::readiness::{ReadinessGate, RepObserver};
use pipette_plan_types::reserved_flags::llamacpp_cli_stock_tools as reserved;
use pipette_plan_types::result::BenchmarkResultData;
use pipette_plan_types::run::RunRequest;
use pipette_plan_types::run::RunResponse;

use crate::bench;
use crate::common::apply_dylib_search_env;
use crate::models::require_gguf_text;
use crate::runtime_flags;
use crate::runtimes::require_llama_bench;

/// Prefill cell: bound `llama-bench` + GGUF text, typed [`PrefillThroughput`] body.
pub fn run(
    req: &RunRequest,
    readiness_gate: ReadinessGate,
    observer: &RepObserver,
) -> anyhow::Result<RunResponse> {
    let benchmark = req
        .benchmark
        .as_prefill_throughput()
        .map_err(anyhow::Error::from)?;
    let llama_bench = require_llama_bench(req)?;
    let model_path = require_gguf_text(req)?;
    let flags = runtime_flags::for_bench(req)?;
    let extra_flags = bench::args_for(&flags).build(reserved::PREFILL, "prefill_throughput")?;

    let summary = bench::execute_reps(
        "prefill_throughput",
        readiness_gate,
        observer,
        || {
            let mut cmd = Command::new(&llama_bench);
            apply_dylib_search_env(&mut cmd, &llama_bench);
            cmd.arg("--output").arg("json");
            cmd.arg("--model").arg(&model_path);
            cmd.args(&extra_flags);
            cmd.arg("--n-prompt")
                .arg(benchmark.parameter_prefill_tokens.to_string());
            cmd.arg("--n-gen").arg("0");
            Ok(cmd)
        },
        |rows| {
            let n_prompt = benchmark.parameter_prefill_tokens;
            bench::select_row(
                rows,
                &format!("prefill_throughput(n_prompt={n_prompt}, n_gen=0)"),
                |row| row.n_prompt == n_prompt && row.n_gen == 0,
            )
        },
    )?;
    Ok(RunResponse {
        executable: Some(llama_bench.display().to_string()),
        command: summary.preview,
        runtime_flags: Some(flags),
        ..RunResponse::new(
            BenchmarkResultData::PrefillThroughput {
                prefill_time_ms: summary.mean_ms,
                prefill_time_ms_stddev: Some(summary.stddev_ms),
            },
            summary.stdout,
            summary.stderr,
        )
    })
}

#[cfg(test)]
mod tests {
    use anyhow::Context;

    use super::*;
    use crate::bench::LlamaBenchRow;

    #[test]
    fn selects_matching_prefill_row_when_not_first() -> anyhow::Result<()> {
        let rows = vec![
            LlamaBenchRow {
                n_prompt: 256,
                ..llama_bench_row()
            },
            LlamaBenchRow {
                n_prompt: 100,
                n_gen: 0,
                ..llama_bench_row()
            },
        ];

        let row = bench::select_row(&rows, "prefill", |row| {
            row.n_prompt == 100 && row.n_gen == 0
        })?;
        assert_eq!(row.n_prompt, 100);
        Ok(())
    }

    #[test]
    fn rejects_duplicate_prefill_rows() -> anyhow::Result<()> {
        let rows = vec![
            LlamaBenchRow {
                n_prompt: 100,
                n_gen: 0,
                ..llama_bench_row()
            },
            LlamaBenchRow {
                n_prompt: 100,
                n_gen: 0,
                ..llama_bench_row()
            },
        ];

        let err = bench::select_row(&rows, "prefill", |row| {
            row.n_prompt == 100 && row.n_gen == 0
        })
        .err()
        .context("expected duplicate-row rejection")?
        .to_string();

        assert!(err.contains("multiple rows"));
        Ok(())
    }

    fn llama_bench_row() -> LlamaBenchRow {
        LlamaBenchRow {
            n_prompt: 0,
            n_gen: 0,
            n_depth: 0,
            avg_ns: 0.0,
            stddev_ns: 0.0,
            avg_ts: 0.0,
            stddev_ts: 0.0,
        }
    }
}
