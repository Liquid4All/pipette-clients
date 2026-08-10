//! Cell flag derivation: the plan's [`RuntimeFlags`] plus the values this
//! client derives for the benchmark, kept in the same typed form so the result
//! records what ran — then rendered into the server's argv.
//!
//! Nothing here knows how the server is hosted — mounts, env forwarding and the
//! docker/uv split live in [`super::launch`].

use pipette_plan_types::benchmark::{
    BenchmarkDefinition, EndToEndLatency, EvalBenchmark, VlThroughput,
};
use pipette_plan_types::run::RunRequest;
use pipette_plan_types::{Runtime, RuntimeFlagRef, RuntimeFlags, RuntimeType, VllmFlavor};

use crate::flavor::sglang_to_vllm_flavor;
use crate::server;

/// `docker run --ipc` when the cell doesn't set one: vLLM's multi-GPU workers
/// need the host IPC namespace.
const DEFAULT_IPC: &str = "host";

/// The flags this cell runs with: the plan's entry plus every value the client
/// derives for it (context bound, prefix caching, the `docker run` settings).
/// The result is the same variant the plan authors, so it round-trips through
/// the wire form and reaches the submission as the record of the launch.
pub(super) fn resolve_runtime_flags(req: &RunRequest) -> anyhow::Result<RuntimeFlags> {
    let bound = &req.runtime.bound;
    // Ahead of the cell lookup: an unsupported runtime should say so, not
    // report that no flags are defined for its triple.
    if !is_docker(bound) && !is_uv(bound) {
        return Err(unsupported_runtime(bound));
    }
    let mut r = req.runtime_flags_ref()?;

    // vLLM context auto-sizing — mirror pipette-llamacpp's derived `ctx_size`:
    // when the cell hasn't pinned it, derive it from the benchmark so an eval's
    // `parameter_max_tokens` output always has prompt headroom. Without this,
    // the default model context can be smaller than (prompt + max_tokens) and
    // vLLM rejects the request. vLLM only (sglang uses `--context-length`,
    // which is left to the operator).
    if is_vllm(bound) {
        r.max_model_len = r
            .max_model_len
            .or_else(|| Some(default_vllm_max_model_len(&req.benchmark)));
    }

    // Prefix caching is off for every benchmark: they all measure cold
    // prefill+decode, so a reused cache would silently make it a warm-prefix
    // measurement. A cell asking for it on is refused rather than overridden.
    let benchmark_label = req.benchmark.benchmark_type().to_string();
    if r.prefix_caching == Some(true) {
        log::warn!(
            "{benchmark_label}: refusing prefix_caching = true; \
             the benchmark fixes it off for cross-run comparability"
        );
        anyhow::bail!(
            "{benchmark_label} benchmark does not accept prefix_caching = true; \
             it is fixed off by the benchmark; remove the override and re-run"
        );
    }
    r.prefix_caching = Some(false);

    if is_docker(bound) {
        r.gpus = resolve_gpus(bound, r.gpus.take());
        r.shm_size = r
            .shm_size
            .or_else(|| Some(server::DEFAULT_SHM_SIZE.to_string()));
        r.ipc = r.ipc.or_else(|| Some(DEFAULT_IPC.to_string()));
    }

    RuntimeFlags::try_from(r).map_err(anyhow::Error::from)
}

/// The `--gpus` value the launch site will use. Only `NvidiaGpu` takes one:
/// `AmdGpu` addresses GPUs via device mounts (`--device /dev/kfd …`) and `Cpu`
/// allocates none, so the cell's value is dropped there — and the record, built
/// from this, doesn't claim an allocation the container never got.
fn resolve_gpus(bound: &Runtime, authored: Option<String>) -> Option<String> {
    let flavor = match bound {
        Runtime::DockerVllm(d) => d.flavor,
        Runtime::DockerSglang(d) => sglang_to_vllm_flavor(d.flavor),
        _ => return None,
    };
    let gpus = authored.unwrap_or_else(|| server::DEFAULT_GPUS.to_string());
    match flavor {
        VllmFlavor::NvidiaGpu => (!gpus.is_empty()).then_some(gpus),
        VllmFlavor::AmdGpu => {
            if gpus != server::DEFAULT_GPUS {
                log::warn!(
                    "gpus {gpus:?} ignored for AMD runtime; \
                     use ROCR_VISIBLE_DEVICES via envs to restrict visible GPUs"
                );
            }
            None
        }
        VllmFlavor::Cpu => {
            if gpus != server::DEFAULT_GPUS {
                log::warn!("gpus {gpus:?} ignored for CPU runtime (no GPUs to allocate)");
            }
            None
        }
    }
}

/// Server argv for a cell's resolved flags. The launcher settings
/// (gpus/shm_size/ipc/envs) are handled at launch, not spliced on here.
///
/// Reads the flat [`RuntimeFlagRef`] form rather than per-variant patterns, so
/// the twelve torch cells share one renderer.
pub(super) fn render_cell(f: &RuntimeFlags) -> anyhow::Result<Vec<String>> {
    let r = RuntimeFlagRef::from(f.clone());
    let vllm = match r.runtime_type {
        RuntimeType::DockerVllm | RuntimeType::UvVllm => true,
        RuntimeType::DockerSglang | RuntimeType::UvSglang => false,
        other => anyhow::bail!("pipette-torch-oai received runtime flags for {other:?}"),
    };
    let mut out = Vec::new();
    let mut push = |flag: &str, value: String| {
        out.push(flag.to_string());
        out.push(value);
    };
    if let Some(tp) = r.tensor_parallel_size {
        push("--tensor-parallel-size", tp.to_string());
    }
    if let Some(d) = &r.dtype {
        push("--dtype", d.as_ref().to_string());
    }
    if let Some(n) = r.max_model_len {
        push("--max-model-len", n.to_string());
    }
    // vLLM caches by default and spells both directions; sglang's radix cache is
    // on by default with only an off switch, so `true` there is the absence of a
    // flag.
    match (r.prefix_caching, vllm) {
        (Some(true), true) => out.push("--enable-prefix-caching".to_string()),
        (Some(false), true) => out.push("--no-enable-prefix-caching".to_string()),
        (Some(false), false) => out.push("--disable-radix-cache".to_string()),
        (Some(true), false) | (None, _) => {}
    }
    out.extend(r.raw.iter().cloned());
    Ok(out)
}

pub(super) fn is_vllm(runtime: &Runtime) -> bool {
    matches!(runtime, Runtime::DockerVllm(_) | Runtime::UvVllm(_))
}

pub(super) fn is_docker(runtime: &Runtime) -> bool {
    matches!(runtime, Runtime::DockerVllm(_) | Runtime::DockerSglang(_))
}

pub(super) fn is_uv(runtime: &Runtime) -> bool {
    matches!(runtime, Runtime::UvVllm(_) | Runtime::UvSglang(_))
}

/// One spelling of the "wrong crate for this runtime" error, shared by the
/// flag derivation and the launch dispatch so they can't drift.
pub(super) fn unsupported_runtime(runtime: &Runtime) -> anyhow::Error {
    anyhow::anyhow!(
        "runtime `{}` is not a torch-oai (docker/uv vLLM/SGLang) runtime",
        runtime.headless_token()
    )
}

/// vLLM `--max-model-len` default, mirroring pipette-llamacpp's
/// `default_ctx_size` (the GGUF runner's `--ctx-size`): an 8 K-token
/// prompt budget on top of an eval's max output, prefill+decode for the
/// latency benchmark, an image-patch estimate for VL throughput, else a
/// 4096 fallback. `--max-model-len` bounds prompt+output, the same
/// semantics as llama.cpp's `--ctx-size`, so the formulae match exactly.
pub(super) fn default_vllm_max_model_len(benchmark: &BenchmarkDefinition) -> u32 {
    match benchmark {
        BenchmarkDefinition::EndToEndLatency(EndToEndLatency {
            parameter_prefill_tokens,
            parameter_decode_tokens,
            ..
        }) => parameter_prefill_tokens.saturating_add(*parameter_decode_tokens),
        BenchmarkDefinition::Eval(EvalBenchmark {
            parameter_max_tokens,
            ..
        }) => 8192u32.saturating_add(*parameter_max_tokens),
        BenchmarkDefinition::VlThroughput(VlThroughput {
            parameter_image_width,
            parameter_image_height,
            parameter_text_tokens,
            parameter_decode_tokens,
            ..
        }) => {
            // ~1 token per 14x14 patch + text + decode (matches llama.cpp).
            let image_tokens = (parameter_image_width / 14) * (parameter_image_height / 14);
            image_tokens
                .saturating_add(*parameter_text_tokens)
                .saturating_add(*parameter_decode_tokens)
                .max(8192)
        }
        _ => 4096,
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use pipette_plan_types::benchmark::{MaxMemoryUsage, PrefillThroughput};

    use super::*;

    fn eval_benchmark(max_tokens: u32) -> BenchmarkDefinition {
        BenchmarkDefinition::Eval(EvalBenchmark {
            benchmark_id: "eval_ifbench_original".into(),
            parameter_eval_id: "ifbench".into(),
            parameter_dataset_name: "original".into(),
            parameter_max_tokens: max_tokens,
            parameter_mcq_choices: None,
            samples: None,
        })
    }

    #[test]
    fn default_vllm_max_model_len_eval_is_8k_prompt_budget_plus_max_tokens() {
        // Matches pipette-llamacpp's default_ctx_size for evals: ifbench's
        // 8192 max_tokens -> 8192 + 8192 = 16384.
        assert_eq!(default_vllm_max_model_len(&eval_benchmark(8192)), 16384);
        assert_eq!(default_vllm_max_model_len(&eval_benchmark(256)), 8448);
    }

    #[test]
    fn default_vllm_max_model_len_latency_is_prefill_plus_decode() {
        let b = BenchmarkDefinition::EndToEndLatency(EndToEndLatency {
            benchmark_id: "e2e".into(),
            parameter_prefill_tokens: 256,
            parameter_decode_tokens: 192,
        });
        assert_eq!(default_vllm_max_model_len(&b), 448);
    }

    #[test]
    fn default_vllm_max_model_len_other_benchmarks_fall_back_to_4096() {
        let b = BenchmarkDefinition::PrefillThroughput(PrefillThroughput {
            benchmark_id: "prefill".into(),
            parameter_prefill_tokens: 512,
        });
        assert_eq!(default_vllm_max_model_len(&b), 4096);
    }

    fn vllm_flags(max_model_len: Option<u32>, prefix_caching: Option<bool>) -> RuntimeFlags {
        RuntimeFlags::EvalDockerVllmTorch {
            tensor_parallel_size: Some(2),
            dtype: None,
            max_model_len,
            prefix_caching,
            gpus: None,
            shm_size: None,
            ipc: None,
            envs: vec![],
            raw: vec!["--swap-space".to_string(), "8".to_string()],
        }
    }

    fn sglang_flags(prefix_caching: Option<bool>) -> RuntimeFlags {
        RuntimeFlags::MaxMemoryDockerSglangTorch {
            tensor_parallel_size: None,
            prefix_caching,
            gpus: None,
            shm_size: None,
            ipc: None,
            envs: vec![],
            raw: vec![],
        }
    }

    /// Each server spells the prefix-cache setting its own way, and the
    /// launcher-only fields never reach the server argv.
    #[rstest]
    #[case::vllm_off(
        vllm_flags(Some(8448), Some(false)),
        vec!["--tensor-parallel-size", "2", "--max-model-len", "8448",
             "--no-enable-prefix-caching", "--swap-space", "8"],
    )]
    #[case::vllm_on(
        vllm_flags(Some(4096), Some(true)),
        vec!["--tensor-parallel-size", "2", "--max-model-len", "4096",
             "--enable-prefix-caching", "--swap-space", "8"],
    )]
    #[case::vllm_unset(
        vllm_flags(None, None),
        vec!["--tensor-parallel-size", "2", "--swap-space", "8"],
    )]
    #[case::sglang_off(sglang_flags(Some(false)), vec!["--disable-radix-cache"])]
    // sglang's radix cache is on by default, so "on" renders nothing.
    #[case::sglang_on(sglang_flags(Some(true)), vec![])]
    fn render_cell_cases(
        #[case] flags: RuntimeFlags,
        #[case] expected: Vec<&str>,
    ) -> anyhow::Result<()> {
        assert_eq!(render_cell(&flags)?, expected);
        Ok(())
    }

    #[test]
    fn render_cell_rejects_a_non_torch_cell() {
        let flags = RuntimeFlags::PrefillMlxIosPipetteMlx { n_ubatch: None };
        assert!(render_cell(&flags).is_err());
    }

    fn max_memory_benchmark() -> BenchmarkDefinition {
        BenchmarkDefinition::MaxMemoryUsage(MaxMemoryUsage {
            benchmark_id: "mem".into(),
            parameter_prefill_tokens: 512,
        })
    }

    #[test]
    fn max_memory_benchmark_is_a_vllm_fallback_length() {
        assert_eq!(default_vllm_max_model_len(&max_memory_benchmark()), 4096);
    }
}
