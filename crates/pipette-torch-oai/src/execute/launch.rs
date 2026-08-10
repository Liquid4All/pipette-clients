//! Server launch assembly for [`super::run`].
//!
//! Turns a bound runtime + model dir + derived cell flags into a docker or uv
//! launch pair. Flag derivation itself lives in [`super::cell_flags`].

use std::path::{Path, PathBuf};

use anyhow::Context;

use pipette_plan_types::run::RunRequest;
use pipette_plan_types::{Runtime, VllmFlavor};

use super::cell_flags::{is_docker, render_cell, resolve_runtime_flags, unsupported_runtime};
use crate::{
    flavor::sglang_to_vllm_flavor,
    runtimes,
    server::{uv, DockerLaunchEnv, LaunchSpec, ServerKind},
};

/// Fixed: the plan runner has no readiness-timeout setting.
const DEFAULT_READY_TIMEOUT_SECS: u64 = 600;

/// The cell's launcher-level settings, already resolved by
/// [`resolve_runtime_flags`]. Only the docker arm reads `gpus`/`shm_size`/`ipc`
/// (they're `None` on a uv cell); `envs` forwards apply to both engines.
struct LauncherSettings {
    envs: Vec<(String, String)>,
    gpus: Option<String>,
    shm_size: Option<String>,
    ipc: Option<String>,
}

/// Assemble [`super::Launch`] from bound plan runtime + model dir + cell flags.
/// `docker_bin` is resolved by the caller (and reused for orphan reaping); the
/// docker arms require it, the uv arms ignore it.
pub(crate) fn build_launch(
    req: &RunRequest,
    model_dir: &Path,
    workspace_label: String,
    docker_bin: Option<PathBuf>,
) -> anyhow::Result<super::Launch> {
    let bound = &req.runtime.bound;
    // One derivation, two consumers: the server argv below and the flags the
    // response carries back, so what we report can't drift from what we launch.
    let flags = resolve_runtime_flags(req)?;
    let server_args = render_cell(&flags)?;
    let model_value = model_value_for_runtime(bound, model_dir);

    let settings = LauncherSettings {
        envs: parse_envs(flags.envs()),
        gpus: flags.gpus().map(str::to_string),
        shm_size: flags.shm_size().map(str::to_string),
        ipc: flags.ipc().map(str::to_string),
    };

    let (launch_env, launch_spec) = match bound {
        Runtime::DockerVllm(d) => build_docker_launch(
            require_docker_bin(docker_bin)?,
            ServerKind::Vllm,
            d.flavor,
            format!("{}:{}", d.image_name.as_ref(), d.image_tag.as_ref()),
            workspace_label,
            &model_value,
            model_dir,
            server_args,
            settings,
        ),
        Runtime::DockerSglang(d) => build_docker_launch(
            require_docker_bin(docker_bin)?,
            ServerKind::Sglang,
            sglang_to_vllm_flavor(d.flavor),
            format!("{}:{}", d.image_name.as_ref(), d.image_tag.as_ref()),
            workspace_label,
            &model_value,
            model_dir,
            server_args,
            settings,
        ),
        Runtime::UvVllm(_) | Runtime::UvSglang(_) => {
            let venv_dir = runtimes::require_uv_venv(req)?;
            // Orphan marker: the `uv://…` install identity. Use `declared` —
            // bound `AbsolutePreinstalled` Displays as a path, which doesn't
            // identify the install.
            let runtime_ref = req.runtime.declared.to_string();
            let kind = if matches!(bound, Runtime::UvVllm(_)) {
                ServerKind::Vllm
            } else {
                ServerKind::Sglang
            };
            build_uv_launch(
                kind,
                workspace_label,
                &model_value,
                server_args,
                venv_dir,
                runtime_ref,
                settings.envs,
            )
        }
        other => return Err(unsupported_runtime(other)),
    };

    Ok(super::Launch {
        launch_env,
        launch_spec,
        model: model_value,
        ready_timeout_secs: DEFAULT_READY_TIMEOUT_SECS,
        runtime_flags: flags,
    })
}

fn require_docker_bin(docker_bin: Option<PathBuf>) -> anyhow::Result<PathBuf> {
    docker_bin.context("docker runtime reached launch build without a resolved docker binary")
}

/// Fixed container mount point for the resolved model directory. One model per
/// run, so a stable path avoids leaking the store's `<key>/blobs` basename into
/// the server's `--model` arg (which doubles as the OpenAI request `model` field).
const CONTAINER_MODEL_PATH: &str = "/models/model";

/// Parse the cell's `envs` entries into `(name, value)` pairs for the launch
/// site. `K=V` sets the value; a bare `K` is the inherit form — an empty value
/// tells the launcher to forward `$K` from the launching process's environment
/// (matching docker `-e K` and the uv launch site's convention). `K=` (explicit
/// empty) collapses to the same inherit form, so there's no way to author an
/// explicit empty value — acceptable for this domain (tokens, device lists).
fn parse_envs(envs: &[String]) -> Vec<(String, String)> {
    envs.iter()
        .map(|entry| match entry.split_once('=') {
            Some((name, value)) => (name.to_string(), value.to_string()),
            None => (entry.clone(), String::new()),
        })
        .collect()
}

/// Build the docker-side `LaunchEnv` + `LaunchSpec` pair, dispatched from
/// [`build_launch`]. The docker binary, `HF_HOME`, and trailing command args are
/// host/plan-driven defaults rather than cell config, so they're fixed here.
#[allow(clippy::too_many_arguments)]
fn build_docker_launch(
    docker_bin: PathBuf,
    kind: ServerKind,
    flavor: VllmFlavor,
    image_ref: String,
    workspace_label: String,
    model_value: &str,
    model_dir: &Path,
    server_args: Vec<String>,
    settings: LauncherSettings,
) -> (super::LaunchEnv, LaunchSpec) {
    let LauncherSettings {
        envs,
        gpus,
        shm_size,
        ipc,
    } = settings;
    let hf_home: Option<PathBuf> = None;
    let command_args: Vec<String> = Vec::new();
    // The resolved (store-materialized or local) dir is bind-mounted at the
    // fixed container path the engine loads from.
    let container = PathBuf::from(CONTAINER_MODEL_PATH);
    log::info!(
        "mounting model dir: {} -> {}",
        model_dir.display(),
        container.display()
    );
    let mounts: Vec<(PathBuf, PathBuf)> = vec![(model_dir.to_path_buf(), container)];
    let launch_env = DockerLaunchEnv {
        image_ref,
        workspace_label,
        hf_home,
        model: model_value.to_string(),
        flavor,
    };
    let launch_spec = LaunchSpec::Docker {
        kind,
        envs,
        gpus,
        shm_size,
        ipc,
        mounts,
        server_args,
        command_args,
    };
    (
        super::LaunchEnv::Docker {
            docker_bin,
            env: launch_env,
        },
        launch_spec,
    )
}

/// The `--model` value the server receives. Docker gets the fixed container
/// mount point (the resolved dir is bind-mounted there); uv runs on the host and
/// gets the resolved directory directly.
fn model_value_for_runtime(runtime: &Runtime, model_dir: &Path) -> String {
    if is_docker(runtime) {
        CONTAINER_MODEL_PATH.to_string()
    } else {
        model_dir.display().to_string()
    }
}

/// Build the uv-side `LaunchEnv` + `LaunchSpec`, dispatched from
/// [`build_launch`]. `envs` are the cell's env forwards; uv has no docker
/// `mounts` / `shm-size` / `ipc` settings. Host-path models are passed verbatim,
/// and `HF_HOME` / trailing command args are host/plan-driven defaults.
fn build_uv_launch(
    kind: ServerKind,
    workspace_label: String,
    model_value: &str,
    server_args: Vec<String>,
    venv_dir: PathBuf,
    runtime_ref: String,
    uv_envs: Vec<(String, String)>,
) -> (super::LaunchEnv, LaunchSpec) {
    let hf_home: Option<PathBuf> = None;
    let command_args: Vec<String> = Vec::new();
    let launch_env = uv::UvLaunchEnv {
        workspace_label,
        hf_home,
        model: model_value.to_string(),
        runtime_ref,
    };
    let launch_spec = LaunchSpec::Uv {
        kind,
        envs: uv_envs,
        server_args,
        command_args,
    };
    (
        super::LaunchEnv::Uv {
            venv_dir,
            env: launch_env,
        },
        launch_spec,
    )
}

#[cfg(test)]
mod tests {
    use pipette_plan_types::benchmark::{
        BenchmarkDefinition, EndToEndLatency, EvalBenchmark, PrefillThroughput,
    };
    use pipette_plan_types::RuntimeFlags;

    use super::*;

    #[test]
    fn parse_envs_splits_kv_and_inherit_forms() {
        // `K=V` sets the value; bare `K` is the inherit form (empty value ->
        // launcher forwards `$K`); a value may itself contain `=`.
        assert_eq!(
            parse_envs(&[
                "HF_TOKEN=secret".to_string(),
                "CUDA_VISIBLE_DEVICES".to_string(),
                "OPTS=a=b".to_string(),
            ]),
            vec![
                ("HF_TOKEN".to_string(), "secret".to_string()),
                ("CUDA_VISIBLE_DEVICES".to_string(), String::new()),
                ("OPTS".to_string(), "a=b".to_string()),
            ]
        );
    }

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

    fn latency_benchmark() -> BenchmarkDefinition {
        BenchmarkDefinition::EndToEndLatency(EndToEndLatency {
            benchmark_id: "e2e".into(),
            parameter_prefill_tokens: 256,
            parameter_decode_tokens: 192,
        })
    }

    fn prefill_throughput_benchmark() -> BenchmarkDefinition {
        BenchmarkDefinition::PrefillThroughput(PrefillThroughput {
            benchmark_id: "prefill".into(),
            parameter_prefill_tokens: 512,
        })
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

    #[test]
    fn model_value_for_runtime_maps_docker_to_mount_and_uv_to_host_dir() -> anyhow::Result<()> {
        use pipette_plan_types::{
            DockerVllm, NonEmptyString, UvBuild, UvPythonVersion, UvRuntimeSource, UvServerVersion,
            UvVllm,
        };

        let docker = Runtime::DockerVllm(DockerVllm {
            image_name: NonEmptyString::try_new("vllm/vllm-openai".to_string())?,
            image_tag: NonEmptyString::try_new("v0.20.2".to_string())?,
            flavor: VllmFlavor::Cpu,
        });
        let uv = Runtime::UvVllm(UvVllm {
            server_version: UvServerVersion::try_new("0.21.0".to_string())?,
            build: UvBuild::try_new("cpu".to_string())?,
            python_version: UvPythonVersion::try_new("3.12".to_string())?,
            source: UvRuntimeSource::AbsolutePreinstalled {
                dir: pipette_plan_types::AbsolutePath::try_new("/venv".to_string())?,
            },
        });

        // Docker gets the fixed container mount point regardless of the host dir
        // (the resolved dir is bind-mounted there); uv runs on the host and gets
        // the resolved directory directly.
        let model_dir = Path::new("/data/models/abc/blobs");
        assert_eq!(model_value_for_runtime(&docker, model_dir), "/models/model");
        assert_eq!(
            model_value_for_runtime(&uv, model_dir),
            "/data/models/abc/blobs"
        );
        Ok(())
    }

    /// A `RunRequest` carrying `runtime` as both declared and bound, and a
    /// torch model bound to `model_dir`. Enough for `build_launch`, which reads
    /// only the runtime, the model dir, the benchmark, and the cell flags.
    fn stub_req(
        runtime: Runtime,
        model_dir: &Path,
        benchmark: BenchmarkDefinition,
    ) -> anyhow::Result<RunRequest> {
        use pipette_plan_types::run::DeclaredBound;
        use pipette_plan_types::{AbsolutePath, Model, ModelSource, Torch};

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

    /// A uv runtime whose declared form is the `uv://…` install identity and
    /// whose bound form points at `venv` — the shape prepare produces.
    fn uv_declared_and_bound(venv: &Path) -> anyhow::Result<(Runtime, Runtime)> {
        use pipette_plan_types::{
            AbsolutePath, NonEmptyString, UvBuild, UvPythonVersion, UvRuntimeSource,
            UvServerVersion, UvVllm,
        };

        let base = |source: UvRuntimeSource| -> anyhow::Result<Runtime> {
            Ok(Runtime::UvVllm(UvVllm {
                server_version: UvServerVersion::try_new("0.21.0".to_string())?,
                build: UvBuild::try_new("cpu".to_string())?,
                python_version: UvPythonVersion::try_new("3.12".to_string())?,
                source,
            }))
        };
        let declared = base(UvRuntimeSource::PipRequirementsText {
            contents: NonEmptyString::try_new("vllm==0.21.0\n".to_string())?,
            install_flags: None,
        })?;
        let bound = base(UvRuntimeSource::AbsolutePreinstalled {
            dir: AbsolutePath::try_new(venv.to_string_lossy().into_owned())?,
        })?;
        Ok((declared, bound))
    }

    #[test]
    fn build_launch_docker_mounts_model_dir_and_derives_max_model_len() -> anyhow::Result<()> {
        let model_dir = Path::new("/data/models/abc/blobs");
        let req = stub_req(docker_vllm_runtime()?, model_dir, eval_benchmark(256))?;

        let launch = build_launch(
            &req,
            model_dir,
            "/ws/.pipette".to_string(),
            Some(PathBuf::from("/usr/bin/docker")),
        )?;

        // Docker arm: the container mount point is the API model id, and the
        // host dir is bind-mounted there.
        assert_eq!(launch.model, CONTAINER_MODEL_PATH);
        let crate::execute::LaunchEnv::Docker { docker_bin, env } = &launch.launch_env else {
            anyhow::bail!("expected a docker LaunchEnv");
        };
        assert_eq!(docker_bin, Path::new("/usr/bin/docker"));
        assert_eq!(env.image_ref, "vllm/vllm-openai:v0.20.2");
        assert_eq!(env.workspace_label, "/ws/.pipette");
        assert_eq!(env.flavor, VllmFlavor::NvidiaGpu);

        let LaunchSpec::Docker { kind, mounts, .. } = &launch.launch_spec else {
            anyhow::bail!("expected a docker LaunchSpec");
        };
        assert_eq!(*kind, ServerKind::Vllm);
        assert_eq!(
            mounts,
            &vec![(model_dir.to_path_buf(), PathBuf::from(CONTAINER_MODEL_PATH))]
        );

        // Defaulted launcher settings, plus the derived context (8192 prompt
        // budget + 256 max_tokens) and the prefix-cache setting.
        assert_eq!(
            launch.runtime_flags.submission_value(),
            serde_json::json!({
                "max_model_len": 8448,
                "prefix_caching": false,
                "gpus": "all",
                "shm_size": "16g",
                "ipc": "host",
            })
        );
        // …and the server argv is rendered from that same value.
        assert_eq!(
            launch.launch_spec.server_args(),
            ["--max-model-len", "8448", "--no-enable-prefix-caching"]
        );
        Ok(())
    }

    /// A cell's launcher settings survive into the record, env forwards
    /// included — by name, since a value may be a token.
    #[test]
    fn build_launch_records_the_cells_launcher_settings() -> anyhow::Result<()> {
        let model_dir = Path::new("/data/models/abc/blobs");
        let mut req = stub_req(docker_vllm_runtime()?, model_dir, eval_benchmark(256))?;
        req.runtime_flags = Some(RuntimeFlags::EvalDockerVllmTorch {
            tensor_parallel_size: Some(2),
            dtype: None,
            max_model_len: None,
            prefix_caching: None,
            gpus: Some("device=0,1".to_string()),
            shm_size: Some("32g".to_string()),
            ipc: Some("private".to_string()),
            envs: vec![
                "HF_TOKEN=secret".to_string(),
                "CUDA_VISIBLE_DEVICES".to_string(),
            ],
            raw: vec![],
        });

        let launch = build_launch(
            &req,
            model_dir,
            "/ws/.pipette".to_string(),
            Some(PathBuf::from("/usr/bin/docker")),
        )?;

        assert_eq!(
            launch.runtime_flags.submission_value(),
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

    /// The benchmarks measure cold prefill+decode, so a cell asking for prefix
    /// caching is refused rather than silently overridden.
    #[test]
    fn build_launch_refuses_a_cell_that_turns_prefix_caching_on() -> anyhow::Result<()> {
        let model_dir = Path::new("/data/models/abc/blobs");
        let mut req = stub_req(docker_vllm_runtime()?, model_dir, eval_benchmark(256))?;
        req.runtime_flags = Some(RuntimeFlags::EvalDockerVllmTorch {
            tensor_parallel_size: None,
            dtype: None,
            max_model_len: None,
            prefix_caching: Some(true),
            gpus: None,
            shm_size: None,
            ipc: None,
            envs: vec![],
            raw: vec![],
        });

        let err = build_launch(
            &req,
            model_dir,
            "/ws/.pipette".to_string(),
            Some(PathBuf::from("/usr/bin/docker")),
        )
        .err()
        .context("expected a rejection error")?
        .to_string();
        // The error names the benchmark that fixed it, not a hardcoded label.
        assert!(err.contains("eval"), "got {err}");
        assert!(err.contains("prefix_caching"), "got {err}");
        Ok(())
    }

    /// A CPU/AMD flavor allocates no GPUs at the launch site, so the record must
    /// not claim any — it tracks the container that ran, not the request.
    #[test]
    fn build_launch_omits_gpus_from_the_record_when_the_flavor_drops_it() -> anyhow::Result<()> {
        use pipette_plan_types::{DockerVllm, NonEmptyString};

        let model_dir = Path::new("/data/models/abc/blobs");
        let runtime = Runtime::DockerVllm(DockerVllm {
            image_name: NonEmptyString::try_new("vllm/vllm-openai".to_string())?,
            image_tag: NonEmptyString::try_new("v0.20.2".to_string())?,
            flavor: VllmFlavor::Cpu,
        });
        let req = stub_req(runtime, model_dir, latency_benchmark())?;

        let launch = build_launch(
            &req,
            model_dir,
            "/ws/.pipette".to_string(),
            Some(PathBuf::from("/usr/bin/docker")),
        )?;

        assert_eq!(launch.runtime_flags.gpus(), None);
        let record = launch.runtime_flags.submission_value();
        assert!(record.get("gpus").is_none(), "got {record}");
        Ok(())
    }

    #[test]
    fn build_launch_keeps_an_operator_pinned_max_model_len() -> anyhow::Result<()> {
        let model_dir = Path::new("/data/models/abc/blobs");
        let mut req = stub_req(docker_vllm_runtime()?, model_dir, eval_benchmark(256))?;
        req.runtime_flags = Some(RuntimeFlags::EvalDockerVllmTorch {
            tensor_parallel_size: None,
            dtype: None,
            max_model_len: Some(2048),
            prefix_caching: None,
            gpus: None,
            shm_size: None,
            ipc: None,
            envs: vec![],
            raw: vec![],
        });

        let launch = build_launch(
            &req,
            model_dir,
            "/ws/.pipette".to_string(),
            Some(PathBuf::from("/usr/bin/docker")),
        )?;

        // The cell's value suppresses the derived 8448 entirely.
        assert_eq!(
            serde_json::to_value(&launch.runtime_flags)?["max_model_len"],
            2048
        );
        Ok(())
    }

    #[test]
    fn build_launch_uv_binds_venv_and_tags_declared_runtime_ref() -> anyhow::Result<()> {
        // `require_uv_venv` stats `bin/python`, so the venv has to exist.
        let tmp = tempfile::tempdir()?;
        let venv = tmp.path().join("venv");
        std::fs::create_dir_all(venv.join("bin"))?;
        std::fs::write(venv.join("bin").join("python"), "")?;
        let model_dir = tmp.path().join("model");
        std::fs::create_dir_all(&model_dir)?;

        let (declared, bound) = uv_declared_and_bound(&venv)?;
        let mut req = stub_req(bound, &model_dir, latency_benchmark())?;
        req.runtime.declared = declared;

        // No docker binary available — the uv arm must not need one.
        let launch = build_launch(&req, &model_dir, "/ws/.pipette".to_string(), None)?;

        // uv runs on the host, so the model value is the host dir verbatim.
        assert_eq!(launch.model, model_dir.display().to_string());
        let crate::execute::LaunchEnv::Uv { venv_dir, env } = &launch.launch_env else {
            anyhow::bail!("expected a uv LaunchEnv");
        };
        assert_eq!(venv_dir, &venv);
        // The orphan marker is the install identity, not the bound venv path.
        assert_eq!(env.runtime_ref, "uv://vllm@0.21.0+cpu.py3.12");
        assert_eq!(env.model, model_dir.display().to_string());

        let LaunchSpec::Uv { kind, .. } = &launch.launch_spec else {
            anyhow::bail!("expected a uv LaunchSpec");
        };
        assert_eq!(*kind, ServerKind::Vllm);
        // 256 prefill + 192 decode. A uv cell has no `docker run` settings, so
        // the record carries none either.
        assert_eq!(
            launch.runtime_flags.submission_value(),
            serde_json::json!({
                "max_model_len": 448,
                "prefix_caching": false,
            })
        );
        Ok(())
    }

    #[test]
    fn build_launch_sglang_uses_sglang_kind_and_radix_lever() -> anyhow::Result<()> {
        let model_dir = Path::new("/data/models/abc/blobs");
        let req = stub_req(docker_sglang_runtime()?, model_dir, latency_benchmark())?;

        let launch = build_launch(
            &req,
            model_dir,
            "/ws/.pipette".to_string(),
            Some(PathBuf::from("/usr/bin/docker")),
        )?;

        assert_eq!(launch.launch_spec.kind(), ServerKind::Sglang);
        // sglang has no context field (it uses `--context-length`, left to the
        // operator), so the record is the prefix-cache setting plus the
        // launcher defaults — and the argv is sglang's own radix-cache spelling.
        assert_eq!(
            launch.runtime_flags.submission_value(),
            serde_json::json!({
                "prefix_caching": false,
                "gpus": "all",
                "shm_size": "16g",
                "ipc": "host",
            })
        );
        assert_eq!(launch.launch_spec.server_args(), ["--disable-radix-cache"]);
        Ok(())
    }

    #[test]
    fn build_launch_rejects_a_non_torch_runtime() -> anyhow::Result<()> {
        let model_dir = Path::new("/data/models/abc/blobs");
        let req = stub_req(
            Runtime::AppleFoundation(Default::default()),
            model_dir,
            prefill_throughput_benchmark(),
        )?;

        let err = build_launch(&req, model_dir, "/ws/.pipette".to_string(), None)
            .err()
            .context("expected a rejection error")?;
        assert!(err.to_string().contains("not a torch-oai"), "got {err:#}");
        Ok(())
    }

    #[test]
    fn build_launch_docker_without_a_resolved_binary_errors() -> anyhow::Result<()> {
        let model_dir = Path::new("/data/models/abc/blobs");
        let req = stub_req(docker_vllm_runtime()?, model_dir, latency_benchmark())?;

        let err = build_launch(&req, model_dir, "/ws/.pipette".to_string(), None)
            .err()
            .context("expected a rejection error")?;
        assert!(err.to_string().contains("docker binary"), "got {err:#}");
        Ok(())
    }
}
