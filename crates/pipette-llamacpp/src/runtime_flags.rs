//! The flags a llama.cpp cell runs with: the plan's [`RuntimeFlags`] entry plus
//! the defaults the benchmark overlays, kept in the same typed form so the
//! result records what ran. The argv builders render from this value
//! ([`crate::bench::args_for`], [`crate::server::args_for`]).
//!
//! Flags the benchmark fixes for every run (`-r 1`, `--no-warmup`) aren't here:
//! [`RuntimeFlags`] can't carry a reserved flag, by design. They're constants of
//! the benchmark, not of the cell, and the builders add them at argv time.

use pipette_plan_types::run::RunRequest;
use pipette_plan_types::RuntimeFlags;

/// Whether a benchmark pins the model in RAM. Latency and throughput cells do —
/// a page fault mid-measurement is measurement noise; eval leaves llama.cpp's
/// mmap-on default alone because its runs are long enough to amortize.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MmapPolicy {
    AsAuthored,
    PinInRam,
}

/// The flags a `llama-bench` cell runs with: the plan's, with mmap pinned off
/// unless the cell set it (llama-bench maps by default).
pub fn for_bench(req: &RunRequest) -> anyhow::Result<RuntimeFlags> {
    let mut r = req.runtime_flags_ref()?;
    r.mmap = r.mmap.or(Some(false));
    RuntimeFlags::try_from(r).map_err(anyhow::Error::from)
}

/// The flags a `llama-server` cell runs with: the plan's, with the context size
/// derived from the benchmark when the cell didn't pin one, and mmap resolved
/// per `policy`.
pub fn for_server(
    req: &RunRequest,
    default_ctx_size: u32,
    policy: MmapPolicy,
) -> anyhow::Result<RuntimeFlags> {
    let mut r = req.runtime_flags_ref()?;
    r.ctx_size = r.ctx_size.or(Some(default_ctx_size));
    if policy == MmapPolicy::PinInRam {
        r.mmap = r.mmap.or(Some(false));
    }
    RuntimeFlags::try_from(r).map_err(anyhow::Error::from)
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use pipette_plan_types::benchmark::{BenchmarkDefinition, EvalBenchmark, PrefillThroughput};
    use pipette_plan_types::run::DeclaredBound;
    use pipette_plan_types::{LlamacppFlashAttention, Model, Runtime, RuntimeFlagRef};

    use super::*;

    /// A llama.cpp CLI + GGUF-text run of `benchmark`, with no plan flags.
    fn req(benchmark: BenchmarkDefinition) -> anyhow::Result<RunRequest> {
        let runtime: Runtime = serde_json::from_value(serde_json::json!({
            "type": "llamacpp_cli_stock_tools",
            "source": "github_release",
            "version": "b5000",
            "flavor": "macos-arm64"
        }))?;
        let model: Model = serde_json::from_value(serde_json::json!({
            "type": "gguf_text",
            "source": "huggingface",
            "org": "o",
            "repo_name": "r",
            "path": "m-Q4_0.gguf"
        }))?;
        Ok(RunRequest {
            runtime: DeclaredBound::already_bound(runtime),
            model: DeclaredBound::already_bound(model),
            runtime_flags: None,
            model_flags: None,
            benchmark_flags: None,
            benchmark,
        })
    }

    fn bench_req() -> anyhow::Result<RunRequest> {
        req(BenchmarkDefinition::PrefillThroughput(PrefillThroughput {
            benchmark_id: "prefill_throughput_256".into(),
            parameter_prefill_tokens: 256,
        }))
    }

    fn server_req() -> anyhow::Result<RunRequest> {
        req(BenchmarkDefinition::Eval(EvalBenchmark {
            benchmark_id: "ifbench".into(),
            parameter_eval_id: "ifbench".into(),
            parameter_dataset_name: "ifbench".into(),
            parameter_max_tokens: 256,
            parameter_mcq_choices: None,
            samples: None,
        }))
    }

    /// The plan's fields survive alongside the overlay: the record is the whole
    /// picture of what ran, not just what the benchmark changed.
    #[test]
    fn for_bench_keeps_the_cells_values() -> anyhow::Result<()> {
        let mut req = bench_req()?;
        req.runtime_flags = Some(RuntimeFlags::PrefillLlamacppCliStockToolsGgufText {
            threads: Some(8),
            number_gpu_layers: None,
            mmap: None,
            flash_attention: Some(LlamacppFlashAttention::On),
            raw: vec![],
        });

        let r = RuntimeFlagRef::from(for_bench(&req)?);
        assert_eq!(r.threads, Some(8));
        assert_eq!(r.flash_attention, Some(LlamacppFlashAttention::On));
        Ok(())
    }

    fn prefill_flags(mmap: Option<bool>) -> RuntimeFlags {
        RuntimeFlags::PrefillLlamacppCliStockToolsGgufText {
            threads: None,
            number_gpu_layers: None,
            mmap,
            flash_attention: None,
            raw: vec![],
        }
    }

    /// llama-bench maps by default, so the harness pins it off — but only where
    /// the cell left the choice open.
    #[rstest]
    #[case::no_plan_entry(None, Some(false))]
    #[case::cell_left_it_unset(Some(prefill_flags(None)), Some(false))]
    #[case::cell_asked_for_mmap(Some(prefill_flags(Some(true))), Some(true))]
    #[case::cell_pinned_it_off(Some(prefill_flags(Some(false))), Some(false))]
    fn for_bench_resolves_mmap(
        #[case] plan: Option<RuntimeFlags>,
        #[case] expected: Option<bool>,
    ) -> anyhow::Result<()> {
        let mut req = bench_req()?;
        req.runtime_flags = plan;
        assert_eq!(RuntimeFlagRef::from(for_bench(&req)?).mmap, expected);
        Ok(())
    }

    #[test]
    fn for_server_derives_ctx_size_and_honors_the_mmap_policy() -> anyhow::Result<()> {
        let req = server_req()?;

        let pinned = RuntimeFlagRef::from(for_server(&req, 8448, MmapPolicy::PinInRam)?);
        assert_eq!(pinned.ctx_size, Some(8448));
        assert_eq!(pinned.mmap, Some(false));

        let as_authored = RuntimeFlagRef::from(for_server(&req, 8448, MmapPolicy::AsAuthored)?);
        assert_eq!(as_authored.ctx_size, Some(8448));
        assert_eq!(as_authored.mmap, None);
        Ok(())
    }

    #[test]
    fn for_server_keeps_an_authored_ctx_size() -> anyhow::Result<()> {
        let mut req = server_req()?;
        req.runtime_flags = Some(RuntimeFlags::EvalLlamacppCliStockToolsGgufText {
            threads: None,
            number_gpu_layers: None,
            mmap: None,
            flash_attention: None,
            ctx_size: Some(2048),
            no_cache: None,
            raw: vec![],
        });

        let r = RuntimeFlagRef::from(for_server(&req, 8448, MmapPolicy::PinInRam)?);
        assert_eq!(r.ctx_size, Some(2048));
        Ok(())
    }

    #[test]
    fn flags_for_another_cell_are_refused() -> anyhow::Result<()> {
        let mut req = bench_req()?;
        req.runtime_flags = Some(RuntimeFlags::EvalLlamacppCliStockToolsGgufText {
            threads: None,
            number_gpu_layers: None,
            mmap: None,
            flash_attention: None,
            ctx_size: None,
            no_cache: None,
            raw: vec![],
        });

        assert!(for_bench(&req).is_err());
        Ok(())
    }
}
