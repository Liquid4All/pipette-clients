use std::path::{Path, PathBuf};

use pipette_plan_types::{
    BenchmarkType, ModelType, RunRequest, Runtime, RuntimeFlags, resolve_speculation,
};

use crate::cell_flags::{
    asr_eval_argv, batch_argv, cache_report_argv, is_docker, render_cell, vlm_eval_argv,
};
use crate::models::{ResolvedTorchModel, ResolvedTorchSpeculativeModel};
use crate::server::docker::{CONTAINER_DRAFT_PATH, CONTAINER_MODEL_PATH};

/// The drafter a `vl_spec_decode` cell launches with when speculation is on.
/// Rendered here rather than in the cell's flags because the paths depend on
/// how the server is hosted (a container mount or the host directory), which
/// is also why plan-types reserves these flags in the cell's `raw`.
#[derive(Debug, Clone)]
pub(crate) struct Drafter {
    pub(crate) dir: PathBuf,
    pub(crate) block_size: u32,
}

/// Only the corpus benchmark on a `torch_speculative` pair has a drafter, and
/// only with speculation on; off, the pair's target runs alone. Every other
/// cell is a plain `torch` directory, and [`ResolvedTorchModel`] refuses a pair.
pub(crate) fn model_dirs(
    req: &RunRequest,
) -> anyhow::Result<(ResolvedTorchModel, Option<Drafter>)> {
    let model_type = ModelType::of(&req.model.bound);
    if req.benchmark.benchmark_type() != BenchmarkType::VlSpecDecode
        || model_type != ModelType::TorchSpeculative
    {
        return Ok((ResolvedTorchModel::try_from(&req.model.bound)?, None));
    }
    let ResolvedTorchSpeculativeModel { target, draft } =
        ResolvedTorchSpeculativeModel::try_from(&req.model.bound)?;
    let enabled = resolve_speculation(model_type, req.runtime_flags_ref()?.speculative_enabled)?;
    if !enabled {
        return Ok((target, None));
    }
    let block_size = req
        .benchmark
        .as_vl_spec_decode()
        .map_err(anyhow::Error::from)?
        .parameter_draft_length
        .into_inner()
        .unsigned_abs();
    Ok((
        target,
        Some(Drafter {
            dir: draft,
            block_size,
        }),
    ))
}

/// Only SGLang has a spec-decode path here, so a vLLM runtime with a drafter
/// is refused.
pub(crate) fn server_args(
    req: &RunRequest,
    flags: &RuntimeFlags,
    drafter: Option<&Drafter>,
) -> anyhow::Result<Vec<String>> {
    let bound = &req.runtime.bound;
    let mut server_args = render_cell(flags)?;
    // The batch is the benchmark's, not the cell's, so it is derived here
    // rather than rendered from the flags.
    server_args.extend(batch_argv(bound, &req.benchmark));
    // Likewise the benchmark's, not the cell's: SGLang reports the
    // cached-prefix counter only when the launch asks for it.
    server_args.extend(cache_report_argv(bound, &req.benchmark));
    // And what a vlm_eval's recipe needs of vLLM.
    server_args.extend(vlm_eval_argv(bound, &req.benchmark));
    // And what an asr_eval's prompt profile needs of it.
    server_args.extend(asr_eval_argv(bound, &req.benchmark));
    if let Some(drafter) = drafter {
        anyhow::ensure!(
            matches!(bound, Runtime::DockerSglang(_) | Runtime::UvSglang(_)),
            "a drafter is launched through SGLang's DSpark path; runtime `{}` has none",
            bound.headless_token()
        );
        let draft_value = if is_docker(bound) {
            CONTAINER_DRAFT_PATH.to_string()
        } else {
            drafter.dir.display().to_string()
        };
        server_args.extend(
            [
                "--speculative-algorithm",
                "DSPARK",
                "--speculative-draft-model-path",
                &draft_value,
                "--speculative-dspark-block-size",
                &drafter.block_size.to_string(),
            ]
            .map(str::to_string),
        );
    }
    Ok(server_args)
}

/// Also the OpenAI request's `model`: docker gets the container mount point,
/// uv the host directory.
pub(crate) fn model_value(runtime: &Runtime, model_dir: &Path) -> String {
    if is_docker(runtime) {
        CONTAINER_MODEL_PATH.to_string()
    } else {
        model_dir.display().to_string()
    }
}

pub(crate) fn mounts(model_dir: &Path, drafter: Option<&Drafter>) -> Vec<(PathBuf, PathBuf)> {
    let container = PathBuf::from(CONTAINER_MODEL_PATH);
    tracing::info!(
        from = %model_dir.display(),
        to = %container.display(),
        "mounting model dir"
    );
    std::iter::once((model_dir.to_path_buf(), container))
        .chain(drafter.map(|d| (d.dir.clone(), PathBuf::from(CONTAINER_DRAFT_PATH))))
        .collect()
}

#[cfg(test)]
mod tests {
    use anyhow::Context;
    use rstest::rstest;

    use pipette_plan_types::{
        BatchSize, BenchmarkDefinition, BenchmarkId, DockerIpcMode, DockerShmSize,
        EndToEndLatencyBenchmark, EnvForward, EvalBenchmark, NonNegativeCount, PositiveCount,
        PrefillThroughputBenchmark, VllmFlavor,
    };

    use super::*;
    use crate::cell_flags::resolve_runtime_flags;

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

    fn latency_benchmark() -> anyhow::Result<BenchmarkDefinition> {
        Ok(BenchmarkDefinition::EndToEndLatency(
            EndToEndLatencyBenchmark {
                benchmark_id: BenchmarkId::try_new("e2e".to_owned())?,
                parameter_prefill_tokens: NonNegativeCount::try_new(256)?,
                parameter_decode_tokens: PositiveCount::try_new(192)?,
                parameter_batch_size: BatchSize::one(),
            },
        ))
    }

    fn prefill_throughput_benchmark() -> anyhow::Result<BenchmarkDefinition> {
        Ok(BenchmarkDefinition::PrefillThroughput(
            PrefillThroughputBenchmark {
                benchmark_id: BenchmarkId::try_new("prefill".to_owned())?,
                parameter_prefill_tokens: NonNegativeCount::try_new(512)?,
                parameter_batch_size: BatchSize::one(),
            },
        ))
    }

    fn docker_vllm_runtime() -> anyhow::Result<Runtime> {
        use pipette_plan_types::{DockerVllm, NonEmptyString};

        Ok(Runtime::DockerVllm(DockerVllm {
            image_name: NonEmptyString::try_new("vllm/vllm-openai".to_string())?,
            image_tag: NonEmptyString::try_new("v0.20.2".to_string())?,
            flavor: VllmFlavor::NvidiaGpu,
        }))
    }

    fn docker_sglang_runtime() -> anyhow::Result<Runtime> {
        use pipette_plan_types::{DockerSglang, NonEmptyString, SglangFlavor};

        Ok(Runtime::DockerSglang(DockerSglang {
            image_name: NonEmptyString::try_new("lmsysorg/sglang".to_string())?,
            image_tag: NonEmptyString::try_new("v0.4.0".to_string())?,
            flavor: SglangFlavor::NvidiaGpu,
        }))
    }

    fn uv_vllm_runtime() -> anyhow::Result<Runtime> {
        use pipette_plan_types::{
            AbsolutePath, UvBuild, UvPythonVersion, UvRuntimeSource, UvServerVersion, UvVllm,
        };

        Ok(Runtime::UvVllm(UvVllm {
            server_version: UvServerVersion::try_new("0.21.0".to_string())?,
            build: UvBuild::try_new("cpu".to_string())?,
            python_version: UvPythonVersion::try_new("3.12".to_string())?,
            source: UvRuntimeSource::AbsolutePreinstalled {
                dir: AbsolutePath::try_new("/venv".to_string())?,
            },
        }))
    }

    /// All the launch reads is the runtime, the benchmark and the cell flags.
    fn stub_req(
        runtime: Runtime,
        model_dir: &Path,
        benchmark: BenchmarkDefinition,
    ) -> anyhow::Result<RunRequest> {
        use pipette_plan_types::{AbsolutePath, DeclaredBound, Model, ModelSource, Torch};

        let model = Model::Torch(Torch {
            source: ModelSource::AbsoluteDir {
                dir: AbsolutePath::try_new(model_dir.to_string_lossy().into_owned())?,
            },
        });
        Ok(RunRequest {
            runtime: DeclaredBound::already_bound(runtime),
            model: DeclaredBound::already_bound(model),
            runtime_flags: None,
            model_flags: None,
            benchmark_flags: None,
            benchmark,
        })
    }

    fn args_of(req: &RunRequest, drafter: Option<&Drafter>) -> anyhow::Result<Vec<String>> {
        server_args(req, &resolve_runtime_flags(req)?, drafter)
    }

    const MODEL_DIR: &str = "/data/models/abc/blobs";

    /// The reference batch size emits no cap, and `--max-model-len` does not
    /// grow with the batch.
    #[rstest]
    #[case::vllm_reference(true, 1, vec!["--max-model-len", "768", "--no-enable-prefix-caching"])]
    #[case::vllm_batched(
        true,
        4,
        vec!["--max-model-len", "768", "--no-enable-prefix-caching", "--max-num-seqs", "4"]
    )]
    #[case::sglang_reference(false, 1, vec!["--disable-radix-cache"])]
    #[case::sglang_batched(
        false,
        4,
        vec!["--disable-radix-cache", "--max-running-requests", "4"]
    )]
    fn the_slot_cap_follows_the_batch_size(
        #[case] vllm: bool,
        #[case] batch_size: u32,
        #[case] expected: Vec<&str>,
    ) -> anyhow::Result<()> {
        let runtime = if vllm {
            docker_vllm_runtime()?
        } else {
            docker_sglang_runtime()?
        };
        let benchmark = BenchmarkDefinition::EndToEndLatency(EndToEndLatencyBenchmark {
            benchmark_id: BenchmarkId::try_new(format!(
                "end_to_end_latency_512_256_bs{batch_size}"
            ))?,
            parameter_prefill_tokens: NonNegativeCount::try_new(512)?,
            parameter_decode_tokens: PositiveCount::try_new(256)?,
            parameter_batch_size: BatchSize::try_new(batch_size)?,
        });
        let req = stub_req(runtime, Path::new(MODEL_DIR), benchmark)?;
        assert_eq!(args_of(&req, None)?, expected);
        Ok(())
    }

    #[test]
    fn the_model_value_is_the_mount_on_docker_and_the_dir_on_uv() -> anyhow::Result<()> {
        let model_dir = Path::new(MODEL_DIR);
        assert_eq!(
            model_value(&docker_vllm_runtime()?, model_dir),
            CONTAINER_MODEL_PATH
        );
        assert_eq!(model_value(&uv_vllm_runtime()?, model_dir), MODEL_DIR);
        assert_eq!(
            mounts(model_dir, None),
            vec![(model_dir.to_path_buf(), PathBuf::from(CONTAINER_MODEL_PATH))]
        );
        Ok(())
    }

    /// 8448 is the 8192 prompt budget plus 256 `max_tokens`.
    #[test]
    fn the_record_and_the_argv_come_from_one_resolution() -> anyhow::Result<()> {
        let req = stub_req(
            docker_vllm_runtime()?,
            Path::new(MODEL_DIR),
            eval_benchmark(256)?,
        )?;
        let flags = resolve_runtime_flags(&req)?;
        assert_eq!(
            flags.submission_value(),
            serde_json::json!({
                "max_model_len": 8448,
                "prefix_caching": false,
                "gpus": "all",
                "shm_size": "16g",
                "ipc": "host",
            })
        );
        assert_eq!(
            server_args(&req, &flags, None)?,
            ["--max-model-len", "8448", "--no-enable-prefix-caching"]
        );
        Ok(())
    }

    /// Env forwards are recorded by name, since a value may be a token.
    #[test]
    fn the_record_keeps_the_cells_launcher_settings() -> anyhow::Result<()> {
        let mut req = stub_req(
            docker_vllm_runtime()?,
            Path::new(MODEL_DIR),
            eval_benchmark(256)?,
        )?;
        req.runtime_flags = Some(RuntimeFlags::EvalDockerVllmTorch {
            tensor_parallel_size: Some(PositiveCount::try_new(2)?),
            dtype: None,
            max_model_len: None,
            prefix_caching: None,
            gpus: Some("device=0,1".to_string()),
            shm_size: Some(DockerShmSize::try_new("32g")?),
            ipc: Some(DockerIpcMode::Private),
            parallel: None,
            repetition_detection: None,
            envs: vec![
                EnvForward::try_new("HF_TOKEN=secret")?,
                EnvForward::try_new("CUDA_VISIBLE_DEVICES")?,
            ],
            raw: vec![],
        });
        assert_eq!(
            resolve_runtime_flags(&req)?.submission_value(),
            serde_json::json!({
                "tensor_parallel_size": 2,
                "max_model_len": 8448,
                "prefix_caching": false,
                "gpus": "device=0,1",
                "shm_size": "32g",
                "ipc": "private",
                "envs": ["HF_TOKEN", "CUDA_VISIBLE_DEVICES"],
            })
        );
        Ok(())
    }

    #[test]
    fn a_cell_that_turns_prefix_caching_on_is_refused() -> anyhow::Result<()> {
        let mut req = stub_req(
            docker_vllm_runtime()?,
            Path::new(MODEL_DIR),
            eval_benchmark(256)?,
        )?;
        req.runtime_flags = Some(RuntimeFlags::EvalDockerVllmTorch {
            tensor_parallel_size: None,
            dtype: None,
            max_model_len: None,
            prefix_caching: Some(true),
            gpus: None,
            shm_size: None,
            ipc: None,
            parallel: None,
            repetition_detection: None,
            envs: vec![],
            raw: vec![],
        });
        let err = resolve_runtime_flags(&req)
            .err()
            .context("expected a rejection error")?
            .to_string();
        // The error names the benchmark that fixed it, not a hardcoded label.
        assert!(err.contains("eval"), "got {err}");
        assert!(err.contains("prefix_caching"), "got {err}");
        Ok(())
    }

    /// The record tracks the container that ran, not the request.
    #[test]
    fn the_record_omits_gpus_when_the_flavor_drops_them() -> anyhow::Result<()> {
        use pipette_plan_types::{DockerVllm, NonEmptyString};

        let runtime = Runtime::DockerVllm(DockerVllm {
            image_name: NonEmptyString::try_new("vllm/vllm-openai".to_string())?,
            image_tag: NonEmptyString::try_new("v0.20.2".to_string())?,
            flavor: VllmFlavor::Cpu,
        });
        let req = stub_req(runtime, Path::new(MODEL_DIR), latency_benchmark()?)?;
        let flags = resolve_runtime_flags(&req)?;
        assert_eq!(flags.gpus(), None);
        let record = flags.submission_value();
        assert!(record.get("gpus").is_none(), "got {record}");
        Ok(())
    }

    #[test]
    fn an_operator_pinned_max_model_len_is_kept() -> anyhow::Result<()> {
        let mut req = stub_req(
            docker_vllm_runtime()?,
            Path::new(MODEL_DIR),
            eval_benchmark(256)?,
        )?;
        req.runtime_flags = Some(RuntimeFlags::EvalDockerVllmTorch {
            tensor_parallel_size: None,
            dtype: None,
            max_model_len: Some(PositiveCount::try_new(2048)?),
            prefix_caching: None,
            gpus: None,
            shm_size: None,
            ipc: None,
            parallel: None,
            repetition_detection: None,
            envs: vec![],
            raw: vec![],
        });
        // The cell's value suppresses the derived 8448 entirely.
        assert_eq!(
            serde_json::to_value(resolve_runtime_flags(&req)?)?["max_model_len"],
            2048
        );
        Ok(())
    }

    /// 448 is 256 prefill plus 192 decode.
    #[test]
    fn a_uv_record_carries_no_docker_settings() -> anyhow::Result<()> {
        let req = stub_req(
            uv_vllm_runtime()?,
            Path::new(MODEL_DIR),
            latency_benchmark()?,
        )?;
        assert_eq!(
            resolve_runtime_flags(&req)?.submission_value(),
            serde_json::json!({
                "max_model_len": 448,
                "prefix_caching": false,
            })
        );
        Ok(())
    }

    /// SGLang's `--context-length` is left to the operator, so the record
    /// carries no context.
    #[test]
    fn sglang_takes_its_radix_cache_lever() -> anyhow::Result<()> {
        let req = stub_req(
            docker_sglang_runtime()?,
            Path::new(MODEL_DIR),
            latency_benchmark()?,
        )?;
        let flags = resolve_runtime_flags(&req)?;
        assert_eq!(
            flags.submission_value(),
            serde_json::json!({
                "prefix_caching": false,
                "gpus": "all",
                "shm_size": "16g",
                "ipc": "host",
            })
        );
        assert_eq!(server_args(&req, &flags, None)?, ["--disable-radix-cache"]);
        Ok(())
    }

    #[test]
    fn a_non_torch_runtime_is_refused() -> anyhow::Result<()> {
        let req = stub_req(
            Runtime::AppleFoundation(Default::default()),
            Path::new(MODEL_DIR),
            prefill_throughput_benchmark()?,
        )?;
        let err = resolve_runtime_flags(&req)
            .err()
            .context("expected a rejection error")?;
        assert!(err.to_string().contains("not a torch-oai"), "got {err:#}");
        Ok(())
    }

    fn corpus_benchmark(draft_length: i32) -> anyhow::Result<BenchmarkDefinition> {
        use pipette_plan_types::{SamplingTemperature, VlSpecDecodeBenchmark};

        Ok(BenchmarkDefinition::VlSpecDecode(VlSpecDecodeBenchmark {
            benchmark_id: BenchmarkId::try_new("vl_spec_decode_mmspec_test_gqa_k8".to_owned())?,
            parameter_draft_length: PositiveCount::try_new(draft_length)?,
            parameter_max_tokens: PositiveCount::try_new(2048)?,
            parameter_batch_size: BatchSize::one(),
            parameter_temperature: SamplingTemperature::greedy(),
            dataset: serde_json::from_value(serde_json::json!({
                "format": "vl-spec-decode-corpus/1",
                "adapter": "mmspec@1",
                "subset": "GQA",
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

    /// The pair on the corpus cell, with the drafter's flags as authored.
    fn spec_pair_req(
        runtime: Runtime,
        target: &Path,
        speculative_enabled: Option<bool>,
        raw: &[&str],
    ) -> anyhow::Result<RunRequest> {
        use pipette_plan_types::{
            AbsolutePath, DeclaredBound, Model, ModelSource, SpeculativePair,
        };

        let dir = |p: &Path| -> anyhow::Result<ModelSource> {
            Ok(ModelSource::AbsoluteDir {
                dir: AbsolutePath::try_new(p.to_string_lossy().into_owned())?,
            })
        };
        let model = Model::TorchSpeculative(SpeculativePair {
            target: dir(target)?,
            draft: dir(&target.with_file_name("draft"))?,
        });
        let flags: RuntimeFlags = serde_json::from_value(serde_json::json!({
            "runtime_type": pipette_plan_types::RuntimeType::of(&runtime),
            "model_type": "torch_speculative",
            "benchmark_type": "vl_spec_decode",
            "speculative_enabled": speculative_enabled,
            "raw": raw,
        }))?;
        Ok(RunRequest {
            runtime: DeclaredBound::already_bound(runtime),
            model: DeclaredBound::already_bound(model),
            runtime_flags: Some(flags),
            model_flags: None,
            benchmark_flags: None,
            benchmark: corpus_benchmark(8)?,
        })
    }

    /// The DSpark flags follow the cell's own `raw`, and the recorded flags
    /// say the run drafted.
    #[test]
    fn a_corpus_launch_with_a_drafter_mounts_it_and_derives_the_dspark_flags() -> anyhow::Result<()>
    {
        let target = Path::new("/data/models/spec__x/blobs/target");
        let req = spec_pair_req(
            docker_sglang_runtime()?,
            target,
            None,
            &["--speculative-draft-attention-backend", "flashinfer"],
        )?;
        let drafter = Drafter {
            dir: target.with_file_name("draft"),
            block_size: 8,
        };
        let flags = resolve_runtime_flags(&req)?;
        assert_eq!(
            mounts(target, Some(&drafter)),
            vec![
                (target.to_path_buf(), PathBuf::from(CONTAINER_MODEL_PATH)),
                (
                    target.with_file_name("draft"),
                    PathBuf::from(CONTAINER_DRAFT_PATH)
                ),
            ]
        );
        assert_eq!(
            server_args(&req, &flags, Some(&drafter))?,
            [
                "--disable-radix-cache",
                "--speculative-draft-attention-backend",
                "flashinfer",
                "--enable-cache-report",
                "--speculative-algorithm",
                "DSPARK",
                "--speculative-draft-model-path",
                CONTAINER_DRAFT_PATH,
                "--speculative-dspark-block-size",
                "8",
            ]
        );
        assert_eq!(flags.speculative_enabled(), Some(true));
        Ok(())
    }

    #[test]
    fn a_corpus_launch_with_speculation_off_launches_the_target_alone() -> anyhow::Result<()> {
        let target = Path::new("/data/models/spec__x/blobs/target");
        let req = spec_pair_req(docker_sglang_runtime()?, target, Some(false), &[])?;
        let flags = resolve_runtime_flags(&req)?;
        let server_args = server_args(&req, &flags, None)?;
        assert_eq!(mounts(target, None).len(), 1);
        assert!(
            !server_args.iter().any(|a| a.starts_with("--speculative-")),
            "{server_args:?}"
        );
        // The cache report is the benchmark's, not the drafter's.
        assert!(
            server_args.iter().any(|a| a == "--enable-cache-report"),
            "{server_args:?}"
        );
        assert_eq!(flags.speculative_enabled(), Some(false));
        Ok(())
    }

    #[test]
    fn a_drafter_on_vllm_is_refused() -> anyhow::Result<()> {
        let target = Path::new("/data/models/spec__x/blobs/target");
        let req = stub_req(docker_vllm_runtime()?, target, eval_benchmark(256)?)?;
        let drafter = Drafter {
            dir: target.with_file_name("draft"),
            block_size: 8,
        };
        let err = args_of(&req, Some(&drafter))
            .err()
            .context("expected a refusal")?;
        assert!(err.to_string().contains("DSpark path"), "got {err:#}");
        Ok(())
    }

    /// `model` on `benchmark` over SGLang, with the cell's `speculative_enabled`
    /// flag when set.
    fn sglang_req(
        model: pipette_plan_types::Model,
        benchmark: BenchmarkDefinition,
        speculative_enabled: Option<bool>,
    ) -> anyhow::Result<RunRequest> {
        use pipette_plan_types::{DeclaredBound, RuntimeType};

        let runtime = docker_sglang_runtime()?;
        let runtime_flags = speculative_enabled
            .map(|enabled| {
                serde_json::from_value(serde_json::json!({
                    "runtime_type": RuntimeType::of(&runtime),
                    "model_type": ModelType::of(&model),
                    "benchmark_type": benchmark.benchmark_type(),
                    "speculative_enabled": enabled,
                }))
            })
            .transpose()?;
        Ok(RunRequest {
            runtime: DeclaredBound::already_bound(runtime),
            model: DeclaredBound::already_bound(model),
            runtime_flags,
            model_flags: None,
            benchmark_flags: None,
            benchmark,
        })
    }

    // `expected`: `Ok(whether the drafter launches)`, or `Err(substring)`.
    #[rstest]
    #[case::unset_on_a_pair(corpus_benchmark(8), true, None, Ok(true))]
    #[case::on_a_pair(corpus_benchmark(8), true, Some(true), Ok(true))]
    #[case::off_on_a_pair(corpus_benchmark(8), true, Some(false), Ok(false))]
    #[case::plain_model(corpus_benchmark(8), false, None, Ok(false))]
    #[case::pair_on_another_benchmark(
        prefill_throughput_benchmark(),
        true,
        None,
        Err("expected a bound torch model")
    )]
    fn model_dirs_cases(
        #[case] benchmark: anyhow::Result<BenchmarkDefinition>,
        #[case] speculative: bool,
        #[case] speculative_enabled: Option<bool>,
        #[case] expected: Result<bool, &str>,
    ) -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let model = crate::models::test_support::bound_model(tmp.path(), speculative)?;
        let result = model_dirs(&sglang_req(model, benchmark?, speculative_enabled)?);
        match expected {
            Ok(drafter) => {
                let (target, draft) = result?;
                assert_eq!(
                    (target.to_path_buf(), draft.map(|d| (d.dir, d.block_size))),
                    (
                        tmp.path().join("target"),
                        drafter.then(|| (tmp.path().join("draft"), 8))
                    )
                );
            }
            Err(substr) => {
                let err = result.err().context("expected a rejection error")?;
                assert!(err.to_string().contains(substr), "got {err}");
            }
        }
        Ok(())
    }
}
