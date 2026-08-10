//! Decode throughput via `llama-bench` for a prepared [`RunRequest`].

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

/// Decode cell: bound `llama-bench` + GGUF text, typed [`DecodeThroughput`] body.
pub fn run(
    req: &RunRequest,
    readiness_gate: ReadinessGate,
    observer: &RepObserver,
) -> anyhow::Result<RunResponse> {
    let benchmark = req
        .benchmark
        .as_decode_throughput()
        .map_err(anyhow::Error::from)?;
    let llama_bench = require_llama_bench(req)?;
    let model_path = require_gguf_text(req)?;
    let flags = runtime_flags::for_bench(req)?;
    let extra_flags = bench::args_for(&flags).build(reserved::DECODE, "decode_throughput")?;

    let summary = bench::execute_reps(
        "decode_throughput",
        readiness_gate,
        observer,
        || {
            let mut cmd = Command::new(&llama_bench);
            apply_dylib_search_env(&mut cmd, &llama_bench);
            cmd.arg("--output").arg("json");
            cmd.arg("--model").arg(&model_path);
            cmd.args(&extra_flags);
            cmd.arg("--n-prompt").arg("0");
            cmd.arg("--n-gen")
                .arg(benchmark.parameter_decode_tokens.to_string());
            cmd.arg("--n-depth")
                .arg(benchmark.parameter_prefill_tokens.to_string());
            Ok(cmd)
        },
        |rows| {
            let n_gen = benchmark.parameter_decode_tokens;
            let n_depth = benchmark.parameter_prefill_tokens;
            bench::select_row(
                rows,
                &format!("decode_throughput(n_prompt=0, n_gen={n_gen}, n_depth={n_depth})"),
                |row| row.n_prompt == 0 && row.n_gen == n_gen && row.n_depth == n_depth,
            )
        },
    )?;
    Ok(RunResponse {
        executable: Some(llama_bench.display().to_string()),
        command: summary.preview,
        runtime_flags: Some(flags),
        ..RunResponse::new(
            BenchmarkResultData::DecodeThroughput {
                decode_time_ms: summary.mean_ms,
                decode_time_ms_stddev: Some(summary.stddev_ms),
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
    fn selects_matching_decode_row() -> anyhow::Result<()> {
        let rows = vec![
            LlamaBenchRow {
                n_prompt: 0,
                n_gen: 128,
                n_depth: 256,
                ..llama_bench_row()
            },
            LlamaBenchRow {
                n_prompt: 0,
                n_gen: 64,
                n_depth: 512,
                ..llama_bench_row()
            },
        ];
        let row = bench::select_row(&rows, "decode", |row| {
            row.n_prompt == 0 && row.n_gen == 64 && row.n_depth == 512
        })?;
        assert_eq!(row.n_gen, 64);
        assert_eq!(row.n_depth, 512);
        Ok(())
    }

    #[test]
    fn rejects_duplicate_decode_rows() -> anyhow::Result<()> {
        let rows = vec![
            LlamaBenchRow {
                n_prompt: 0,
                n_gen: 64,
                n_depth: 100,
                ..llama_bench_row()
            },
            LlamaBenchRow {
                n_prompt: 0,
                n_gen: 64,
                n_depth: 100,
                ..llama_bench_row()
            },
        ];
        let err = bench::select_row(&rows, "decode", |row| {
            row.n_prompt == 0 && row.n_gen == 64 && row.n_depth == 100
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
