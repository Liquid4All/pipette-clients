/// The argv tests both ASR cells share; `$flags` is the cell's runtime flags.
#[cfg(test)]
macro_rules! asr_launch_tests {
    ($flags:ident) => {
        mod tests {
            use pipette_plan_types::{LlamacppFlashAttention, LlamacppLoadMode};

            use super::*;
            use crate::asr::AudioEncoder;
            use crate::runtimes::test_llama_server;

            /// Every knob set to its own value, so a knob that lands on another's
            /// flag shows.
            fn every_knob_set(raw: &[&str]) -> $flags {
                $flags {
                    threads: Some(3),
                    number_gpu_layers: Some(7),
                    mmap: None,
                    load_mode: Some(LlamacppLoadMode::DirectIo),
                    flash_attention: Some(LlamacppFlashAttention::On),
                    ctx_size: Some(4096),
                    no_cache: Some(true),
                    envs: Vec::new(),
                    raw: raw.iter().map(|&t| t.to_owned()).collect(),
                }
            }

            fn server_args(projector: &str, raw: &[&str]) -> anyhow::Result<Vec<String>> {
                let flags = every_knob_set(raw);
                let recipe = Recipe::for_encoder(&AudioEncoder::from_name(projector, false))?;
                let model = ResolvedGgufVisionModel {
                    backbone: "model.gguf".into(),
                    mmproj: "mmproj.gguf".into(),
                };
                let extra = recipe.server_args(&flags.raw, &flags.envs, &[])?;
                Ok(launch(&flags, extra, &test_llama_server(), &model, None).args()?)
            }

            #[test]
            fn the_server_takes_every_knob() -> anyhow::Result<()> {
                assert_eq!(
                    server_args("gemma4a", &["--poll", "0"])?.join(" "),
                    "--ctx-checkpoints 0 --ctx-size 4096 --flash-attn on --load-mode dio \
                     --n-gpu-layers 7 --no-cache-prompt --no-warmup --parallel 1 --poll 0 \
                     --reasoning off --threads 3"
                );
                Ok(())
            }

            /// llama-server keeps the last of two spellings, so the final argv
            /// must turn thinking off exactly once and carry only the recipe's
            /// template flags, whatever else the cell sets.
            #[rstest::rstest]
            #[case::gemma("gemma4a", &[])]
            #[case::voxtral("voxtral", &["--chat-template", "--chat-template-kwargs", "--jinja"])]
            // Sorted before `--jinja`, which is harmless: llama.cpp stores the
            // template as it parses and checks it once every flag is read
            // (`common/arg.cpp`).
            #[case::lfm3("lfm3a", &["--chat-template", "--jinja"])]
            #[case::minicpm_o("minicpmoa", &["--chat-template-kwargs"])]
            #[case::qwen3_asr("qwen3a", &["--chat-template", "--chat-template-kwargs", "--jinja"])]
            fn the_final_argv_turns_thinking_off_once(
                #[case] projector: &str,
                #[case] template_flags: &[&str],
            ) -> anyhow::Result<()> {
                let args = server_args(projector, &["--threads", "4"])?;
                let reasoning: Vec<&str> = args
                    .iter()
                    .map(String::as_str)
                    .filter(|a| *a == "-rea" || a.contains("reasoning") || a.contains("think"))
                    .collect();
                assert_eq!(reasoning, ["--reasoning"], "{args:?}");
                assert!(
                    args.windows(2).any(|w| w == ["--reasoning", "off"]),
                    "{args:?}"
                );
                let templates: Vec<&str> = args
                    .iter()
                    .map(String::as_str)
                    .filter(|a| a.contains("jinja") || a.starts_with("--chat-template"))
                    .collect();
                assert_eq!(templates, template_flags, "{args:?}");
                Ok(())
            }
        }
    };
}

mod asr_eval;
mod asr_max_memory;
mod asr_throughput;
mod decode_throughput;
mod end_to_end_latency;
mod eval;
mod max_memory_usage;
mod prefill_throughput;
mod vl_max_memory;
mod vl_spec_decode;
mod vl_throughput;
mod vlm_eval;

use pipette_ops::measurement::MeasurementObserver;
use pipette_ops::readiness::ReadinessGate;
use pipette_ops::run_context::RunContext;
use pipette_plan_types::{
    BatchSize, BenchmarkDefinition, LlamacppCliStockToolsExtraRuntimeConfiguration, ModelType,
    RunRequest, RunResponse, VlSpecDecodeLlamacppCliStockTools, resolve_speculation,
};

use crate::models::{ResolvedGgufTextModel, ResolvedGgufVisionModel};
use crate::runtimes::{self, ResolvedLlamaBatchedBench, ResolvedLlamaBench, ResolvedLlamaServer};

/// Run a prepared [`RunRequest`]. The request stops here: each arm converts the
/// flags its benchmark reads and resolves the tool and model it runs.
/// `run_context` lends the eval checkpoints and the corpus download;
/// `client_config` is applied whole to every tool the run spawns.
pub fn run(
    req: &RunRequest,
    run_context: &dyn RunContext,
    readiness_gate: ReadinessGate,
    observer: MeasurementObserver<'_>,
    client_config: Option<&LlamacppCliStockToolsExtraRuntimeConfiguration>,
) -> anyhow::Result<RunResponse> {
    let flags = req.runtime_flags_ref()?;
    runtimes::check_load_mode(flags.mmap, flags.load_mode)?;
    match &req.benchmark {
        // `llama-bench` measures one sequence at a time, so a batch runs on
        // `llama-batched-bench`.
        BenchmarkDefinition::PrefillThroughput(benchmark)
            if benchmark.parameter_batch_size == BatchSize::one() =>
        {
            prefill_throughput::run(
                benchmark,
                req.runtime_flags_ref()?.try_into()?,
                &ResolvedLlamaBench::try_from(&req.runtime.bound)?,
                &ResolvedGgufTextModel::try_from(&req.model.bound)?,
                readiness_gate,
                observer,
                client_config,
            )
        }
        BenchmarkDefinition::PrefillThroughput(benchmark) => prefill_throughput::run_batched(
            benchmark,
            req.runtime_flags_ref()?.try_into()?,
            &ResolvedLlamaBatchedBench::try_from(&req.runtime.bound)?,
            &ResolvedGgufTextModel::try_from(&req.model.bound)?,
            readiness_gate,
            observer,
            client_config,
        ),
        BenchmarkDefinition::DecodeThroughput(benchmark)
            if benchmark.parameter_batch_size == BatchSize::one() =>
        {
            decode_throughput::run(
                benchmark,
                req.runtime_flags_ref()?.try_into()?,
                &ResolvedLlamaBench::try_from(&req.runtime.bound)?,
                &ResolvedGgufTextModel::try_from(&req.model.bound)?,
                readiness_gate,
                observer,
                client_config,
            )
        }
        BenchmarkDefinition::DecodeThroughput(benchmark) => decode_throughput::run_batched(
            benchmark,
            req.runtime_flags_ref()?.try_into()?,
            &ResolvedLlamaBatchedBench::try_from(&req.runtime.bound)?,
            &ResolvedGgufTextModel::try_from(&req.model.bound)?,
            readiness_gate,
            observer,
            client_config,
        ),
        BenchmarkDefinition::EndToEndLatency(benchmark) => end_to_end_latency::run(
            benchmark,
            req.runtime_flags_ref()?.try_into()?,
            &req.benchmark_flags_ref()?,
            &ResolvedLlamaServer::try_from(&req.runtime.bound)?,
            &ResolvedGgufTextModel::try_from(&req.model.bound)?,
            readiness_gate,
            observer,
            client_config,
        ),
        BenchmarkDefinition::MaxMemoryUsage(benchmark) => max_memory_usage::run(
            benchmark,
            req.runtime_flags_ref()?.try_into()?,
            &req.benchmark_flags_ref()?,
            runtimes::flavor(&req.runtime.bound)?,
            &ResolvedLlamaBench::try_from(&req.runtime.bound)?,
            &ResolvedGgufTextModel::try_from(&req.model.bound)?,
            observer,
            client_config,
        ),
        BenchmarkDefinition::Eval(benchmark) => eval::run(
            benchmark,
            req.runtime_flags_ref()?.try_into()?,
            req.model_flags.as_ref(),
            &req.benchmark_flags_ref()?,
            &ResolvedLlamaServer::try_from(&req.runtime.bound)?,
            &ResolvedGgufTextModel::try_from(&req.model.bound)?,
            // Resume is keyed by the request's portable identity.
            run_context.eval_completions().open(req)?,
            client_config,
        )
        // As authored: eval reports the cell's flags unchanged, none when it
        // carries none, not the context it sized.
        .map(|response| response.runtime_flags(req.runtime_flags.clone()).build()),
        BenchmarkDefinition::VlmEval(benchmark) => vlm_eval::run(
            // A catalog entry runs only its recipe's models, in its recipe's
            // thinking mode: refused before the corpus downloads.
            vlm_eval::on_recipe(benchmark, &req.model.declared, req.model_flags.as_ref())?,
            // A catalog entry's prompts are its corpus, downloaded and checked
            // against both its pins before anything starts.
            pipette_vl_spec_decode_corpus::vlm_eval::download(run_context, benchmark)?.as_ref(),
            req.runtime_flags_ref()?.try_into()?,
            req.model_flags.as_ref(),
            &req.benchmark_flags_ref()?,
            &ResolvedLlamaServer::try_from(&req.runtime.bound)?,
            &ResolvedGgufVisionModel::try_from(&req.model.bound)?,
            // Resume is keyed by the request's portable identity.
            run_context.eval_completions().open(req)?,
            readiness_gate,
            observer,
            client_config,
        ),
        BenchmarkDefinition::AsrEval(benchmark) => asr_eval::run(
            // A catalog entry runs only its recipe's models, thinking off:
            // refused before the corpus downloads, which the run does once it
            // knows its decode (it sizes the context from the clips).
            asr_eval::on_recipe(benchmark, &req.model.declared, req.model_flags.as_ref())?,
            req.runtime_flags_ref()?.try_into()?,
            &req.benchmark_flags_ref()?,
            &ResolvedLlamaServer::try_from(&req.runtime.bound)?,
            &ResolvedGgufVisionModel::try_from(&req.model.bound)?,
            // Resume is keyed by the request's portable identity.
            run_context.eval_completions().open(req)?,
            run_context,
            readiness_gate,
            observer,
            client_config,
        ),
        BenchmarkDefinition::VlThroughput(benchmark) => vl_throughput::run(
            benchmark,
            req.runtime_flags_ref()?.try_into()?,
            &req.benchmark_flags_ref()?,
            &ResolvedLlamaServer::try_from(&req.runtime.bound)?,
            &ResolvedGgufVisionModel::try_from(&req.model.bound)?,
            readiness_gate,
            observer,
            client_config,
        ),
        BenchmarkDefinition::AsrThroughput(benchmark) => asr_throughput::run(
            benchmark,
            req.runtime_flags_ref()?.try_into()?,
            &req.benchmark_flags_ref()?,
            &ResolvedLlamaServer::try_from(&req.runtime.bound)?,
            &ResolvedGgufVisionModel::try_from(&req.model.bound)?,
            readiness_gate,
            observer,
            client_config,
        ),
        BenchmarkDefinition::AsrMaxMemory(benchmark) => asr_max_memory::run(
            benchmark,
            req.runtime_flags_ref()?.try_into()?,
            &req.benchmark_flags_ref()?,
            runtimes::flavor(&req.runtime.bound)?,
            &ResolvedLlamaServer::try_from(&req.runtime.bound)?,
            &ResolvedGgufVisionModel::try_from(&req.model.bound)?,
            observer,
            client_config,
        ),
        BenchmarkDefinition::VlMaxMemory(benchmark) => vl_max_memory::run(
            benchmark,
            req.runtime_flags_ref()?.try_into()?,
            &req.benchmark_flags_ref()?,
            runtimes::flavor(&req.runtime.bound)?,
            &ResolvedLlamaServer::try_from(&req.runtime.bound)?,
            &ResolvedGgufVisionModel::try_from(&req.model.bound)?,
            observer,
            client_config,
        ),
        BenchmarkDefinition::VlSpecDecode(benchmark) => {
            let lease = observer.outside(|| {
                pipette_vl_spec_decode_corpus::download(
                    run_context,
                    &benchmark.benchmark_id,
                    &benchmark.dataset,
                )
            })?;
            // Resolved against the model, so the run loads the matching files and
            // the row records what ran rather than what the cell left unset.
            let runtime_flags: VlSpecDecodeLlamacppCliStockTools =
                req.runtime_flags_ref()?.try_into()?;
            let speculative = resolve_speculation(
                ModelType::of(&req.model.declared),
                runtime_flags.speculative_enabled,
            )?;
            vl_spec_decode::run(
                benchmark,
                VlSpecDecodeLlamacppCliStockTools {
                    speculative_enabled: Some(speculative),
                    ..runtime_flags
                },
                &req.benchmark_flags_ref()?,
                &ResolvedLlamaServer::try_from(&req.runtime.bound)?,
                &vl_spec_decode::LoadedModel::resolve(&req.model.bound, speculative)?,
                lease.corpus(),
                readiness_gate,
                observer,
                client_config,
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Context;
    use rstest::rstest;

    use pipette_plan_types::{
        AbsolutePath, BatchSize, BenchmarkDefinition, BenchmarkId, Dataset, DeclaredBound,
        DecodeThroughputBenchmark, LlamaCppFlavor, LlamacppCliStockTools,
        LlamacppCliStockToolsSource, Model, NonNegativeCount, PositiveCount,
        PrefillThroughputBenchmark, Runtime, RuntimeFlags,
    };

    use super::*;

    /// A [`RunContext`] with no corpus to lend: these cells pull none.
    struct NoCorpus;

    impl RunContext for NoCorpus {
        fn fetch_dataset(&self, _: &Dataset) -> anyhow::Result<Vec<u8>> {
            anyhow::bail!("this test lends no corpus")
        }

        fn datasets_dir(&self) -> std::path::PathBuf {
            std::env::temp_dir()
        }

        fn eval_completions(&self) -> pipette_ops::EvalCompletionsStore {
            pipette_ops::EvalCompletionsStore::new(std::env::temp_dir())
        }
    }

    fn prefill(batch_size: u32) -> anyhow::Result<BenchmarkDefinition> {
        Ok(BenchmarkDefinition::PrefillThroughput(
            PrefillThroughputBenchmark {
                benchmark_id: BenchmarkId::try_new(format!(
                    "prefill_throughput_512_bs{batch_size}"
                ))?,
                parameter_prefill_tokens: NonNegativeCount::try_new(512)?,
                parameter_batch_size: BatchSize::try_new(batch_size)?,
            },
        ))
    }

    fn decode(batch_size: u32) -> anyhow::Result<BenchmarkDefinition> {
        Ok(BenchmarkDefinition::DecodeThroughput(
            DecodeThroughputBenchmark {
                benchmark_id: BenchmarkId::try_new(format!(
                    "decode_throughput_512_128_bs{batch_size}"
                ))?,
                parameter_prefill_tokens: NonNegativeCount::try_new(512)?,
                parameter_decode_tokens: PositiveCount::try_new(128)?,
                parameter_batch_size: BatchSize::try_new(batch_size)?,
            },
        ))
    }

    /// `benchmark` on tools and weights under `root`, which holds neither.
    fn request(
        root: &std::path::Path,
        benchmark: BenchmarkDefinition,
    ) -> anyhow::Result<RunRequest> {
        let runtime = Runtime::LlamacppCliStockTools(LlamacppCliStockTools {
            source: LlamacppCliStockToolsSource::AbsoluteDir {
                dir: AbsolutePath::try_new(root.display().to_string())?,
            },
            flavor: LlamaCppFlavor::MacosArm64,
        });
        let model: Model = serde_json::from_value(serde_json::json!({
            "type": "gguf_text",
            "source": "absolute_file",
            "path": root.join("m.gguf"),
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

    fn failure(req: &RunRequest) -> anyhow::Result<String> {
        let observer = MeasurementObserver::QUIET;
        let err = run(req, &NoCorpus, &|| Ok(()), observer, None)
            .err()
            .context("expected the cell to fail")?;
        Ok(format!("{err:#}"))
    }

    /// The reference batch size runs on `llama-bench` and a batch on
    /// `llama-batched-bench`: the tool is the first thing an empty install root
    /// fails to resolve.
    #[rstest]
    #[case::prefill(prefill(1), "`llama-bench` not found")]
    #[case::decode(decode(1), "`llama-bench` not found")]
    #[case::prefill_batched(prefill(4), "`llama-batched-bench` not found")]
    #[case::decode_batched(decode(4), "`llama-batched-bench` not found")]
    fn the_batch_size_picks_the_tool(
        #[case] benchmark: anyhow::Result<BenchmarkDefinition>,
        #[case] expected: &str,
    ) -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let message = failure(&request(tmp.path(), benchmark?)?)?;
        assert!(message.contains(expected), "{message}");
        Ok(())
    }

    /// Converting the flags to the llama.cpp cell's type refuses another
    /// runtime, by name, before any tool is looked for.
    #[test]
    fn a_runtime_other_than_llamacpp_is_refused() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let mut req = request(tmp.path(), prefill(1)?)?;
        req.runtime = DeclaredBound::already_bound(Runtime::AppleFoundation(Default::default()));
        let message = failure(&req)?;
        assert!(message.contains("AppleFoundation"), "{message}");
        Ok(())
    }

    #[test]
    fn flags_for_another_cell_are_refused() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let mut req = request(tmp.path(), prefill(1)?)?;
        req.runtime_flags = Some(RuntimeFlags::EvalLlamacppCliStockToolsGgufText(
            pipette_plan_types::EvalLlamacppCliStockToolsGgufText {
                threads: None,
                number_gpu_layers: None,
                mmap: None,
                load_mode: None,
                flash_attention: None,
                parallel: None,
                ctx_size: None,
                no_cache: None,
                envs: Vec::new(),
                raw: vec![],
            },
        ));
        let message = failure(&req)?;
        assert!(message.contains("not for this cell"), "{message}");
        Ok(())
    }
}
