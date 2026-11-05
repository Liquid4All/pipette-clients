//! Cell flag derivation: the plan's [`RuntimeFlags`] plus the values this
//! client derives for the benchmark, kept typed so the result records what
//! ran, then rendered into the server's argv. Hosting lives in
//! [`crate::launch`] (mounts, the model value, the drafter's argv) and
//! [`crate::server`] (env forwarding, the docker/uv split).

use pipette_plan_types::{
    AsrPromptProfile, BatchSize, BenchmarkDefinition, BenchmarkType, DecodeThroughputBenchmark,
    DockerIpcMode, DockerShmSize, EndToEndLatencyBenchmark, EvalBenchmark, KnownAsrPromptProfile,
    MaxMemoryUsageBenchmark, ModelType, PositiveCount, PrefillThroughputBenchmark, RunRequest,
    Runtime, RuntimeFlagRef, RuntimeFlags, RuntimeType, VlThroughputBenchmark, VllmFlavor,
    resolve_speculation,
};

use crate::execute::max_memory_usage;
use crate::flavor::sglang_to_vllm_flavor;
use crate::server;

/// `docker run --ipc` when the cell doesn't set one: vLLM's multi-GPU workers
/// need the host IPC namespace.
const DEFAULT_IPC: DockerIpcMode = DockerIpcMode::Host;

/// The flags this cell runs with: the plan's entry plus every value the client
/// derives for it (context bound, prefix caching, the `docker run` settings).
/// The result is the same variant the plan authors, so it round-trips through
/// the wire form and reaches the submission as the record of the launch.
pub(crate) fn resolve_runtime_flags(req: &RunRequest) -> anyhow::Result<RuntimeFlags> {
    let bound = &req.runtime.bound;
    // Ahead of the cell lookup: an unsupported runtime should say so, not
    // report that no flags are defined for its triple.
    if !is_docker(bound) && !is_uv(bound) {
        return Err(unsupported_runtime(bound));
    }
    let mut r = req.runtime_flags_ref()?;

    // Written back resolved, so the row records the drafter condition the run
    // was measured under rather than what the cell left unset.
    if req.benchmark.benchmark_type() == BenchmarkType::VlSpecDecode {
        r.speculative_enabled = Some(resolve_speculation(
            ModelType::of(&req.model.bound),
            r.speculative_enabled,
        )?);
    }

    // Unpinned, vLLM's context is sized from the benchmark (as pipette-llamacpp
    // derives `ctx_size`): the model default can be smaller than prompt plus
    // `max_tokens`, and vLLM rejects the request. SGLang's `--context-length`
    // is left to the operator. An eval pin at or under `parameter_max_tokens`
    // drops `max_tokens` instead (`execute::eval`).
    if is_vllm(bound)
        && r.max_model_len.is_none()
        && let Some(derived) = default_vllm_max_model_len(&req.benchmark)
    {
        r.max_model_len = Some(PositiveCount::try_from(derived)?);
    }

    // Prefix caching is off for every benchmark: they all measure cold
    // prefill+decode, so a reused cache would silently make it a warm-prefix
    // measurement. A cell asking for it on is refused rather than overridden.
    let benchmark_label = req.benchmark.benchmark_type().to_string();
    if r.prefix_caching == Some(true) {
        tracing::warn!(
            benchmark_type = %benchmark_label,
            "refusing prefix_caching = true; the benchmark fixes it off for cross-run \
             comparability"
        );
        anyhow::bail!(
            "{benchmark_label} benchmark does not accept prefix_caching = true; \
             it is fixed off by the benchmark; remove the override and re-run"
        );
    }
    r.prefix_caching = Some(false);

    if is_docker(bound) {
        r.gpus = resolve_gpus(bound, r.gpus.take());
        if r.shm_size.is_none() {
            r.shm_size = Some(DockerShmSize::try_new(server::DEFAULT_SHM_SIZE)?);
        }
        r.ipc = r.ipc.or(Some(DEFAULT_IPC));
    }

    RuntimeFlags::try_from(r).map_err(anyhow::Error::from)
}

/// The `--gpus` value the launch site will use. Only `NvidiaGpu` takes one:
/// `AmdGpu` addresses GPUs via device mounts (`--device /dev/kfd …`) and `Cpu`
/// allocates none, so the cell's value is dropped there, and the record built
/// from this doesn't claim an allocation the container never got.
fn resolve_gpus(bound: &Runtime, authored: Option<String>) -> Option<String> {
    let flavor = match bound {
        Runtime::DockerVllm(d) => d.flavor,
        Runtime::DockerSglang(d) => sglang_to_vllm_flavor(d.flavor),
        Runtime::LlamacppCliStockTools(_)
        | Runtime::LlamacppApkPipette(_)
        | Runtime::LlamacppIosPipette(_)
        | Runtime::MlxMacosPipette(_)
        | Runtime::MlxIosPipette(_)
        | Runtime::OnnxGenaiIosPipette(_)
        | Runtime::UvVllm(_)
        | Runtime::UvSglang(_)
        | Runtime::UvOpenvino(_)
        | Runtime::UvOnnxGenai(_)
        | Runtime::Ovms(_)
        | Runtime::TrtEdgellm(_)
        | Runtime::AppleCoreAiMacosPipette(_)
        | Runtime::AppleFoundation(_) => return None,
    };
    let gpus = authored.unwrap_or_else(|| server::DEFAULT_GPUS.to_string());
    match flavor {
        VllmFlavor::NvidiaGpu => (!gpus.is_empty()).then_some(gpus),
        VllmFlavor::AmdGpu => {
            if gpus != server::DEFAULT_GPUS {
                tracing::warn!(
                    %gpus,
                    "gpus ignored for AMD runtime; use ROCR_VISIBLE_DEVICES via envs to \
                     restrict visible GPUs"
                );
            }
            None
        }
        VllmFlavor::Cpu => {
            if gpus != server::DEFAULT_GPUS {
                tracing::warn!(%gpus, "gpus ignored for CPU runtime (no GPUs to allocate)");
            }
            None
        }
    }
}

/// Server argv for a cell's resolved flags. The launcher settings
/// (gpus/shm_size/ipc/envs) are handled at launch, not spliced on here.
///
/// Reads the flat [`RuntimeFlagRef`] form rather than per-variant patterns, so
/// every torch cell shares one renderer.
pub(crate) fn render_cell(f: &RuntimeFlags) -> anyhow::Result<Vec<String>> {
    let r = RuntimeFlagRef::from(f.clone());
    let vllm = match r.runtime_type {
        RuntimeType::DockerVllm | RuntimeType::UvVllm => true,
        RuntimeType::DockerSglang | RuntimeType::UvSglang => false,
        other @ (RuntimeType::LlamacppCliStockTools
        | RuntimeType::LlamacppApkPipette
        | RuntimeType::LlamacppIosPipette
        | RuntimeType::MlxMacosPipette
        | RuntimeType::MlxIosPipette
        | RuntimeType::OnnxGenaiIosPipette
        | RuntimeType::UvOpenvino
        | RuntimeType::UvOnnxGenai
        | RuntimeType::Ovms
        | RuntimeType::TrtEdgellm
        | RuntimeType::AppleCoreAiMacosPipette
        | RuntimeType::AppleFoundation) => {
            anyhow::bail!("pipette-torch-oai received runtime flags for {other:?}")
        }
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
        push("--dtype", d.as_str().to_string());
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

/// The slot cap for the benchmark's batch: vLLM's `--max-num-seqs`, SGLang's
/// `--max-running-requests`. Both default to far more than a timing cell keeps
/// in flight, so pinning the cap makes the row say the host served exactly
/// `parameter_batch_size` requests. The reference batch size emits nothing:
/// the argv is recorded with the row, and an unbatched run must not read as a
/// different configuration. `--max-model-len` is untouched: it bounds one
/// sequence's prompt plus generation, which the batch does not change.
///
/// The three timing cells only. `vl_spec_decode` reaches no argv with its
/// batch size: SGLang sizes its own slots there (48 under speculation) and the
/// traversal fills them, so a cap would contradict the engine's scheduler.
pub(crate) fn batch_argv(runtime: &Runtime, benchmark: &BenchmarkDefinition) -> Vec<String> {
    let batch_size = match benchmark {
        BenchmarkDefinition::PrefillThroughput(b) => b.parameter_batch_size,
        BenchmarkDefinition::DecodeThroughput(b) => b.parameter_batch_size,
        BenchmarkDefinition::EndToEndLatency(b) => b.parameter_batch_size,
        BenchmarkDefinition::MaxMemoryUsage(_)
        | BenchmarkDefinition::Eval(_)
        | BenchmarkDefinition::VlmEval(_)
        | BenchmarkDefinition::AsrEval(_)
        | BenchmarkDefinition::VlThroughput(_)
        | BenchmarkDefinition::VlMaxMemory(_)
        | BenchmarkDefinition::VlSpecDecode(_)
        | BenchmarkDefinition::AsrThroughput(_)
        | BenchmarkDefinition::AsrMaxMemory(_) => return Vec::new(),
    };
    if batch_size == BatchSize::one() {
        return Vec::new();
    }
    let flag = if is_vllm(runtime) {
        "--max-num-seqs"
    } else {
        "--max-running-requests"
    };
    vec![flag.to_owned(), batch_size.to_string()]
}

/// The argv that makes SGLang report what `--disable-radix-cache` is supposed
/// to have prevented: without `--enable-cache-report` it omits
/// `prompt_tokens_details` whatever happened, which is the shape a clean turn
/// has, so the corpus benchmark's refusal of a cached prefix can never fire.
///
/// The corpus cell alone, and SGLang alone: it is the only reader, vLLM has no
/// such flag, and the argv is recorded with the row, so a reporting flag on a
/// cell that reads nothing would make identical runs compare as different
/// configurations.
pub(crate) fn cache_report_argv(runtime: &Runtime, benchmark: &BenchmarkDefinition) -> Vec<String> {
    let sglang = matches!(runtime, Runtime::DockerSglang(_) | Runtime::UvSglang(_));
    if !sglang || !matches!(benchmark, BenchmarkDefinition::VlSpecDecode(_)) {
        return Vec::new();
    }
    vec!["--enable-cache-report".to_owned()]
}

/// The argv a vlm_eval's recipe needs of vLLM, which the plan types reserve
/// on that cell:
///
/// - `--generation-config vllm`, so a model repo's `generation_config.json`
///   sets no sampling default behind the recipe (vLLM's `auto` applies its
///   temperature, top-k and so on to every knob a request leaves out; each
///   request sends every knob anyway). The stop tokens the file names still
///   apply.
/// - `--trust-request-mm-kwargs` on an entry with an image budget, which each
///   request carries as `mm_processor_kwargs`: vLLM refuses a request's
///   processor kwargs otherwise.
///
/// vlm_eval alone, on vLLM alone: another cell's argv, recorded with its rows,
/// stays as it was.
pub(crate) fn vlm_eval_argv(runtime: &Runtime, benchmark: &BenchmarkDefinition) -> Vec<String> {
    let BenchmarkDefinition::VlmEval(vlm) = benchmark else {
        return Vec::new();
    };
    if !is_vllm(runtime) {
        return Vec::new();
    }
    let mut argv = vec!["--generation-config".to_owned(), "vllm".to_owned()];
    if vlm.parameter_image_tokens.is_some() {
        argv.push("--trust-request-mm-kwargs".to_owned());
    }
    argv
}

/// The argv an asr_eval's prompt profile needs of vLLM, which the plan types
/// reserve on that cell:
///
/// - `--generation-config vllm`, as on vlm_eval: a repo's
///   `generation_config.json` sets no sampling default behind the entry's
///   decoding, which each request sends whole.
/// - Voxtral's Mistral formats (`--tokenizer-mode`, `--config-format` and
///   `--load-format mistral`), as Mistral's card serves it: the transcription
///   request its prompt profile renders is mistral-common's.
/// - MiniCPM-o: `--trust-remote-code`, as its processor is its repo's code;
///   and `--chat-template-content-format string` with `--interleave-mm-strings`,
///   as its template renders each turn as one string and its prompt is the
///   instruction, then the clip (OmniEvalKit's `"<prompt>\n{audio}"`): without
///   them vLLM puts the clip's placeholder before the instruction.
///
/// asr_eval alone, on vLLM alone, as [`vlm_eval_argv`] is.
pub(crate) fn asr_eval_argv(runtime: &Runtime, benchmark: &BenchmarkDefinition) -> Vec<String> {
    let BenchmarkDefinition::AsrEval(asr) = benchmark else {
        return Vec::new();
    };
    if !is_vllm(runtime) {
        return Vec::new();
    }
    let mut argv = vec!["--generation-config", "vllm"];
    match &asr.parameter_prompt_profile {
        AsrPromptProfile::Known(KnownAsrPromptProfile::Voxtral) => argv.extend([
            "--tokenizer-mode",
            "mistral",
            "--config-format",
            "mistral",
            "--load-format",
            "mistral",
        ]),
        AsrPromptProfile::Known(KnownAsrPromptProfile::MinicpmO) => argv.extend([
            "--trust-remote-code",
            "--chat-template-content-format",
            "string",
            "--interleave-mm-strings",
        ]),
        AsrPromptProfile::Known(
            KnownAsrPromptProfile::Gemma4
            | KnownAsrPromptProfile::Qwen25Omni
            | KnownAsrPromptProfile::Lfm25Audio
            | KnownAsrPromptProfile::Lfm3,
        )
        | AsrPromptProfile::Unknown(_) => {}
    }
    argv.into_iter().map(str::to_owned).collect()
}

pub(crate) fn is_vllm(runtime: &Runtime) -> bool {
    matches!(runtime, Runtime::DockerVllm(_) | Runtime::UvVllm(_))
}

pub(crate) fn is_docker(runtime: &Runtime) -> bool {
    matches!(runtime, Runtime::DockerVllm(_) | Runtime::DockerSglang(_))
}

pub(crate) fn is_uv(runtime: &Runtime) -> bool {
    matches!(runtime, Runtime::UvVllm(_) | Runtime::UvSglang(_))
}

/// One spelling of the "wrong crate for this runtime" error, shared by the
/// flag derivation and the launch dispatch so they can't drift.
pub(crate) fn unsupported_runtime(runtime: &Runtime) -> anyhow::Error {
    anyhow::anyhow!(
        "runtime `{}` is not a torch-oai (docker/uv vLLM/SGLang) runtime",
        runtime.headless_token()
    )
}

/// vLLM `--max-model-len` default, which bounds prompt plus output as
/// llama.cpp's `--ctx-size` does: an 8 K-token prompt budget on top of an
/// eval's max output, prefill plus decode for the timing and max-memory cells
/// (prefill and max-memory generate one token), an image-patch estimate for
/// VL throughput, else 4096.
///
/// None for `vlm_eval`: the model's own context fits its longest image prompt
/// plus the recipe's output budget, where no fixed prompt budget would for
/// every family's image sizes.
pub(crate) fn default_vllm_max_model_len(benchmark: &BenchmarkDefinition) -> Option<u32> {
    Some(match benchmark {
        BenchmarkDefinition::PrefillThroughput(PrefillThroughputBenchmark {
            parameter_prefill_tokens,
            ..
        }) => parameter_prefill_tokens
            .into_inner()
            .unsigned_abs()
            .saturating_add(1),
        BenchmarkDefinition::DecodeThroughput(DecodeThroughputBenchmark {
            parameter_prefill_tokens,
            parameter_decode_tokens,
            ..
        })
        | BenchmarkDefinition::EndToEndLatency(EndToEndLatencyBenchmark {
            parameter_prefill_tokens,
            parameter_decode_tokens,
            ..
        }) => parameter_prefill_tokens
            .into_inner()
            .unsigned_abs()
            .saturating_add(parameter_decode_tokens.into_inner().unsigned_abs()),
        BenchmarkDefinition::Eval(EvalBenchmark {
            parameter_max_tokens,
            ..
        }) => 8192u32.saturating_add(parameter_max_tokens.into_inner().unsigned_abs()),
        BenchmarkDefinition::VlThroughput(VlThroughputBenchmark {
            parameter_image_width,
            parameter_image_height,
            parameter_text_tokens,
            parameter_decode_tokens,
            ..
        }) => {
            let [width, height, text, decode] = [
                parameter_image_width.into_inner(),
                parameter_image_height.into_inner(),
                parameter_text_tokens.into_inner(),
                parameter_decode_tokens.into_inner(),
            ]
            .map(i32::unsigned_abs);
            // ~1 token per 14x14 patch + text + decode (matches llama.cpp).
            let image_tokens = (width / 14) * (height / 14);
            image_tokens
                .saturating_add(text)
                .saturating_add(decode)
                .max(8192)
        }
        BenchmarkDefinition::MaxMemoryUsage(MaxMemoryUsageBenchmark {
            parameter_prefill_tokens,
            ..
        }) => parameter_prefill_tokens
            .into_inner()
            .unsigned_abs()
            .saturating_add(max_memory_usage::DECODE_TOKENS),
        BenchmarkDefinition::VlmEval(_) | BenchmarkDefinition::AsrEval(_) => return None,
        BenchmarkDefinition::VlMaxMemory(_)
        | BenchmarkDefinition::VlSpecDecode(_)
        | BenchmarkDefinition::AsrThroughput(_)
        | BenchmarkDefinition::AsrMaxMemory(_) => 4096,
    })
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use pipette_plan_types::{BenchmarkId, NonNegativeCount};

    use super::*;

    fn eval_benchmark(max_tokens: i32) -> anyhow::Result<BenchmarkDefinition> {
        Ok(BenchmarkDefinition::Eval(EvalBenchmark {
            benchmark_id: BenchmarkId::try_new("eval_ifbench_original".to_owned())?,
            parameter_eval_id: "ifbench".into(),
            parameter_dataset_name: "original".into(),
            parameter_max_tokens: PositiveCount::try_new(max_tokens)?,
            parameter_max_context: None,
            parameter_mcq_choices: None,
            parameter_temperature: None,
            parameter_top_p: None,
            parameter_top_k: None,
            parameter_min_p: None,
            parameter_seed: None,
            samples: None,
        }))
    }

    fn latency(prefill: i32, decode: i32) -> anyhow::Result<BenchmarkDefinition> {
        Ok(BenchmarkDefinition::EndToEndLatency(
            EndToEndLatencyBenchmark {
                benchmark_id: BenchmarkId::try_new("e2e".to_owned())?,
                parameter_prefill_tokens: NonNegativeCount::try_new(prefill)?,
                parameter_decode_tokens: PositiveCount::try_new(decode)?,
                parameter_batch_size: BatchSize::one(),
            },
        ))
    }

    fn prefill(prefill: i32) -> anyhow::Result<BenchmarkDefinition> {
        Ok(BenchmarkDefinition::PrefillThroughput(
            PrefillThroughputBenchmark {
                benchmark_id: BenchmarkId::try_new("prefill".to_owned())?,
                parameter_prefill_tokens: NonNegativeCount::try_new(prefill)?,
                parameter_batch_size: BatchSize::one(),
            },
        ))
    }

    fn decode(prefill: i32, decode: i32) -> anyhow::Result<BenchmarkDefinition> {
        Ok(BenchmarkDefinition::DecodeThroughput(
            DecodeThroughputBenchmark {
                benchmark_id: BenchmarkId::try_new("decode".to_owned())?,
                parameter_prefill_tokens: NonNegativeCount::try_new(prefill)?,
                parameter_decode_tokens: PositiveCount::try_new(decode)?,
                parameter_batch_size: BatchSize::one(),
            },
        ))
    }

    fn max_memory(prefill: i32) -> anyhow::Result<BenchmarkDefinition> {
        Ok(BenchmarkDefinition::MaxMemoryUsage(
            MaxMemoryUsageBenchmark {
                benchmark_id: BenchmarkId::try_new("mem".to_owned())?,
                parameter_prefill_tokens: NonNegativeCount::try_new(prefill)?,
            },
        ))
    }

    #[rstest]
    #[case::eval_ifbench(eval_benchmark(8192)?, 16384)]
    #[case::eval_small(eval_benchmark(256)?, 8448)]
    #[case::latency(latency(256, 192)?, 448)]
    #[case::prefill_is_prompt_plus_one(prefill(512)?, 513)]
    #[case::decode_is_prompt_plus_generation(decode(512, 128)?, 640)]
    #[case::max_memory_is_prompt_plus_one(max_memory(512)?, 513)]
    #[case::max_memory_past_the_old_fallback(max_memory(4096)?, 4097)]
    fn default_vllm_max_model_len_follows_the_shape(
        #[case] benchmark: BenchmarkDefinition,
        #[case] expected: u32,
    ) -> anyhow::Result<()> {
        assert_eq!(default_vllm_max_model_len(&benchmark), Some(expected));
        Ok(())
    }

    /// vlm_eval derives no context: unpinned, vLLM serves the model's own.
    #[test]
    fn a_vlm_eval_cell_derives_no_context() -> anyhow::Result<()> {
        let benchmark: BenchmarkDefinition = serde_json::from_value(serde_json::json!({
            "benchmark_type": "vlm_eval",
            "benchmark_id": "vlm_eval_mmstar",
            "parameter_lmms_task": "mmstar",
            "parameter_dataset_version": "2026.10.1",
            "parameter_sample_count": 1,
            "parameter_max_tokens": 32768,
            "samples": [{"id": "a", "messages": [{"role": "user", "content": "q"}]}],
        }))?;
        assert_eq!(default_vllm_max_model_len(&benchmark), None);
        Ok(())
    }

    /// vlm_eval's own argv goes to vlm_eval on vLLM alone, the trust flag only
    /// with an image budget: every other cell's argv, recorded with its rows,
    /// is what it was.
    #[rstest]
    #[case::vlm_eval_on_vllm("uv_vllm", Some(None), &["--generation-config", "vllm"])]
    #[case::vlm_eval_with_an_image_budget("uv_vllm", Some(Some(1120)), &["--generation-config", "vllm", "--trust-request-mm-kwargs"])]
    #[case::text_eval_on_vllm("uv_vllm", None, &[])]
    #[case::vlm_eval_elsewhere("uv_sglang", Some(Some(1120)), &[])]
    fn only_vlm_eval_on_vllm_takes_its_argv(
        #[case] runtime_type: &str,
        #[case] vlm_eval: Option<Option<i32>>,
        #[case] expected: &[&str],
    ) -> anyhow::Result<()> {
        let runtime: Runtime = serde_json::from_value(serde_json::json!({
            "type": runtime_type, "server_version": "0.31.0", "build": "rocm723",
            "python_version": "3.12", "source": {"type": "absolute_preinstalled", "dir": "/venv"},
        }))?;
        let benchmark = match vlm_eval {
            Some(image_tokens) => serde_json::from_value(serde_json::json!({
                "benchmark_type": "vlm_eval",
                "benchmark_id": "vlm_eval_mmstar",
                "parameter_lmms_task": "mmstar",
                "parameter_dataset_version": "2026.10.1",
                "parameter_sample_count": 1,
                "parameter_max_tokens": 64,
                "parameter_image_tokens": image_tokens,
                "samples": [{"id": "a", "messages": [{"role": "user", "content": "q"}]}],
            }))?,
            None => eval_benchmark(256)?,
        };
        assert_eq!(vlm_eval_argv(&runtime, &benchmark), expected);
        Ok(())
    }

    /// asr_eval's own argv goes to asr_eval on vLLM alone: the neutral
    /// generation config always, Voxtral's Mistral formats and MiniCPM-o's
    /// remote code by profile.
    #[rstest]
    #[case::qwen("uv_vllm", "qwen25_omni", &["--generation-config", "vllm"])]
    #[case::voxtral("uv_vllm", "voxtral", &["--generation-config", "vllm", "--tokenizer-mode", "mistral", "--config-format", "mistral", "--load-format", "mistral"])]
    #[case::minicpm_o("uv_vllm", "minicpm_o", &["--generation-config", "vllm", "--trust-remote-code", "--chat-template-content-format", "string", "--interleave-mm-strings"])]
    #[case::elsewhere("uv_sglang", "voxtral", &[])]
    fn only_asr_eval_on_vllm_takes_its_argv(
        #[case] runtime_type: &str,
        #[case] profile: &str,
        #[case] expected: &[&str],
    ) -> anyhow::Result<()> {
        let runtime: Runtime = serde_json::from_value(serde_json::json!({
            "type": runtime_type, "server_version": "0.32.0", "build": "rocm723",
            "python_version": "3.12", "source": {"type": "absolute_preinstalled", "dir": "/venv"},
        }))?;
        let benchmark: BenchmarkDefinition = serde_json::from_value(serde_json::json!({
            "benchmark_type": "asr_eval",
            "benchmark_id": "asr_eval_fleurs_nl_nl",
            "parameter_asr_task": "asr_fleurs_nl_nl",
            "parameter_dataset_version": "2026.10.1",
            "parameter_sample_count": 364,
            "parameter_prompt_profile": profile,
            "parameter_max_tokens": 256,
        }))?;
        assert_eq!(asr_eval_argv(&runtime, &benchmark), expected);
        assert!(vlm_eval_argv(&runtime, &benchmark).is_empty());
        Ok(())
    }

    fn vllm_flags(
        max_model_len: Option<i32>,
        prefix_caching: Option<bool>,
    ) -> anyhow::Result<RuntimeFlags> {
        Ok(RuntimeFlags::EvalDockerVllmTorch {
            tensor_parallel_size: Some(PositiveCount::try_new(2)?),
            dtype: None,
            max_model_len: max_model_len.map(PositiveCount::try_new).transpose()?,
            prefix_caching,
            gpus: None,
            shm_size: None,
            ipc: None,
            parallel: None,
            repetition_detection: None,
            envs: vec![],
            raw: vec!["--swap-space".to_string(), "8".to_string()],
        })
    }

    fn sglang_flags(prefix_caching: Option<bool>) -> RuntimeFlags {
        RuntimeFlags::MaxMemoryUsageDockerSglangTorch {
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
        vllm_flags(Some(8448), Some(false))?,
        vec!["--tensor-parallel-size", "2", "--max-model-len", "8448",
             "--no-enable-prefix-caching", "--swap-space", "8"],
    )]
    #[case::vllm_on(
        vllm_flags(Some(4096), Some(true))?,
        vec!["--tensor-parallel-size", "2", "--max-model-len", "4096",
             "--enable-prefix-caching", "--swap-space", "8"],
    )]
    #[case::vllm_unset(
        vllm_flags(None, None)?,
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

    /// Only above the reference size; the kinds with no batch take nothing.
    #[rstest]
    #[case::vllm_e2e(true, latency(512, 256)?, 4, vec!["--max-num-seqs", "4"])]
    #[case::vllm_prefill(true, prefill(512)?, 2, vec!["--max-num-seqs", "2"])]
    #[case::sglang_decode(false, decode(512, 256)?, 8, vec!["--max-running-requests", "8"])]
    #[case::reference_emits_nothing(true, latency(512, 256)?, 1, vec![])]
    #[case::eval_takes_none(true, eval_benchmark(256)?, 1, vec![])]
    #[case::max_memory_takes_none(false, max_memory(512)?, 1, vec![])]
    fn the_slot_cap_is_the_timing_cells_alone(
        #[case] vllm: bool,
        #[case] benchmark: BenchmarkDefinition,
        #[case] batch_size: u32,
        #[case] expected: Vec<&str>,
    ) -> anyhow::Result<()> {
        use pipette_plan_types::{DockerSglang, DockerVllm, NonEmptyString, SglangFlavor};

        let runtime = if vllm {
            Runtime::DockerVllm(DockerVllm {
                image_name: NonEmptyString::try_new("vllm/vllm-openai".to_string())?,
                image_tag: NonEmptyString::try_new("v0.20.2".to_string())?,
                flavor: VllmFlavor::NvidiaGpu,
            })
        } else {
            Runtime::DockerSglang(DockerSglang {
                image_name: NonEmptyString::try_new("lmsysorg/sglang".to_string())?,
                image_tag: NonEmptyString::try_new("v0.4.0".to_string())?,
                flavor: SglangFlavor::NvidiaGpu,
            })
        };
        let benchmark = with_batch_size(benchmark, BatchSize::try_new(batch_size)?);
        assert_eq!(batch_argv(&runtime, &benchmark), expected);
        Ok(())
    }

    fn sglang_runtime() -> anyhow::Result<Runtime> {
        use pipette_plan_types::{DockerSglang, NonEmptyString, SglangFlavor};

        Ok(Runtime::DockerSglang(DockerSglang {
            image_name: NonEmptyString::try_new("lmsysorg/sglang".to_string())?,
            image_tag: NonEmptyString::try_new("v0.4.0".to_string())?,
            flavor: SglangFlavor::NvidiaGpu,
        }))
    }

    fn vllm_runtime() -> anyhow::Result<Runtime> {
        use pipette_plan_types::{DockerVllm, NonEmptyString};

        Ok(Runtime::DockerVllm(DockerVllm {
            image_name: NonEmptyString::try_new("vllm/vllm-openai".to_string())?,
            image_tag: NonEmptyString::try_new("v0.20.2".to_string())?,
            flavor: VllmFlavor::NvidiaGpu,
        }))
    }

    fn corpus() -> anyhow::Result<BenchmarkDefinition> {
        use pipette_plan_types::{SamplingTemperature, VlSpecDecodeBenchmark};

        Ok(BenchmarkDefinition::VlSpecDecode(VlSpecDecodeBenchmark {
            benchmark_id: BenchmarkId::try_new("vl_spec_decode_mmspec_test_gqa_k8_bs8".to_owned())?,
            parameter_draft_length: PositiveCount::try_new(8)?,
            parameter_max_tokens: PositiveCount::try_new(2048)?,
            parameter_batch_size: BatchSize::try_new(8)?,
            parameter_temperature: SamplingTemperature::greedy(),
            dataset: serde_json::from_value(serde_json::json!({
                "format": "vl-spec-decode-corpus/1",
                "adapter": "mmspec@1",
                "bytes": 1,
                "digest": "0".repeat(64),
                "upstream": {
                    "source": "git",
                    "url": "github.com/acme/corpus",
                    "revision": "0123456789abcdef0123456789abcdef01234567",
                    "identity": "2923603292c7dd463ad01a119f57965817778fc3",
                    "digest": "2".repeat(64)
                }
            }))?,
        }))
    }

    #[test]
    fn the_corpus_benchmark_takes_no_slot_cap() -> anyhow::Result<()> {
        assert!(batch_argv(&sglang_runtime()?, &corpus()?).is_empty());
        Ok(())
    }

    #[rstest]
    #[case::sglang_corpus(false, true, vec!["--enable-cache-report"])]
    #[case::sglang_other_cell(false, false, vec![])]
    #[case::vllm_has_no_such_flag(true, true, vec![])]
    fn the_cache_report_is_the_corpus_benchmarks_alone(
        #[case] vllm: bool,
        #[case] corpus_cell: bool,
        #[case] expected: Vec<&str>,
    ) -> anyhow::Result<()> {
        let runtime = if vllm {
            vllm_runtime()?
        } else {
            sglang_runtime()?
        };
        let benchmark = if corpus_cell {
            corpus()?
        } else {
            decode(512, 256)?
        };
        assert_eq!(cache_report_argv(&runtime, &benchmark), expected);
        Ok(())
    }

    fn with_batch_size(
        benchmark: BenchmarkDefinition,
        batch_size: BatchSize,
    ) -> BenchmarkDefinition {
        match benchmark {
            BenchmarkDefinition::PrefillThroughput(b) => {
                BenchmarkDefinition::PrefillThroughput(PrefillThroughputBenchmark {
                    parameter_batch_size: batch_size,
                    ..b
                })
            }
            BenchmarkDefinition::DecodeThroughput(b) => {
                BenchmarkDefinition::DecodeThroughput(DecodeThroughputBenchmark {
                    parameter_batch_size: batch_size,
                    ..b
                })
            }
            BenchmarkDefinition::EndToEndLatency(b) => {
                BenchmarkDefinition::EndToEndLatency(EndToEndLatencyBenchmark {
                    parameter_batch_size: batch_size,
                    ..b
                })
            }
            other @ (BenchmarkDefinition::MaxMemoryUsage(_)
            | BenchmarkDefinition::Eval(_)
            | BenchmarkDefinition::VlmEval(_)
            | BenchmarkDefinition::AsrEval(_)
            | BenchmarkDefinition::VlThroughput(_)
            | BenchmarkDefinition::VlMaxMemory(_)
            | BenchmarkDefinition::VlSpecDecode(_)
            | BenchmarkDefinition::AsrThroughput(_)
            | BenchmarkDefinition::AsrMaxMemory(_)) => other,
        }
    }

    #[test]
    fn render_cell_rejects_a_non_torch_cell() {
        let flags = RuntimeFlags::PrefillThroughputMlxIosPipetteMlx { n_ubatch: None };
        assert!(render_cell(&flags).is_err());
    }
}
