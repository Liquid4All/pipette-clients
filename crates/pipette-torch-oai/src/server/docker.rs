//! The server in a container, run attached (`docker run --rm`) so the docker
//! CLI's output and exit are the container's.

use std::{
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

use anyhow::Context;

use pipette_plan_types::{
    EnvForward, Runtime, RuntimeFlags, TorchOaiExtraRuntimeConfiguration, VllmFlavor,
};
use pipette_subprocess::DockerRun;

use super::{Server, ServerKind, ServerState, Transport};
use crate::flavor::sglang_to_vllm_flavor;

/// Where the model directory is mounted. One model per run, so a stable path
/// keeps the store's `<key>/blobs` basename out of the server's `--model`,
/// which doubles as the OpenAI request's `model` field.
pub const CONTAINER_MODEL_PATH: &str = "/models/model";
pub const CONTAINER_DRAFT_PATH: &str = "/models/draft";

fn gpu_access_args(flavor: VllmFlavor, gpus_spec: Option<&str>) -> Vec<String> {
    match flavor {
        VllmFlavor::NvidiaGpu => nvidia_gpu_args(gpus_spec),
        VllmFlavor::AmdGpu => amd_gpu_args(gpus_spec),
        VllmFlavor::Cpu => cpu_args(gpus_spec),
    }
}

fn nvidia_gpu_args(gpus_spec: Option<&str>) -> Vec<String> {
    gpus_spec
        .map(|g| vec!["--gpus".to_string(), g.to_string()])
        .unwrap_or_default()
}

fn amd_gpu_args(gpus_spec: Option<&str>) -> Vec<String> {
    if gpus_spec.is_some() {
        tracing::debug!(
            "gpus is ignored for AMD runtimes; use ROCR_VISIBLE_DEVICES \
             via the cell's envs to restrict visible GPUs"
        );
    }
    vec![
        "--device".to_string(),
        "/dev/kfd".to_string(),
        "--device".to_string(),
        "/dev/dri".to_string(),
        "--group-add".to_string(),
        "video".to_string(),
    ]
}

fn cpu_args(gpus_spec: Option<&str>) -> Vec<String> {
    if gpus_spec.is_some() {
        tracing::debug!("gpus is ignored for CPU runtimes");
    }
    Vec::new()
}

/// Per-run container name. The uuid suffix only keeps concurrent runs from
/// colliding in `docker ps`; orphan reaping matches the workspace label, not
/// the name.
fn random_container_name() -> String {
    let suffix: String = uuid::Uuid::new_v4()
        .simple()
        .to_string()
        .chars()
        .take(8)
        .collect();
    format!("pipette-bench-{suffix}")
}

/// `workspace_label` marks the container for the next run's orphan reap.
pub fn launch(
    docker: &Path,
    runtime: &Runtime,
    flags: &RuntimeFlags,
    server_args: &[String],
    mounts: &[(PathBuf, PathBuf)],
    workspace_label: &str,
    client_config: Option<&TorchOaiExtraRuntimeConfiguration>,
) -> anyhow::Result<Server> {
    let kind = ServerKind::of(runtime)?;
    let port = pipette_subprocess::pick_server_port()?;
    let run = docker_run(
        runtime,
        flags,
        server_args,
        mounts,
        workspace_label,
        client_config,
        port,
    )?;
    ensure_image(docker, &run.image)?;
    tracing::info!(
        container = %run.name,
        image = %run.image,
        host_port = port,
        container_port = kind.default_port(),
        "starting docker container"
    );
    let state = ServerState {
        kind,
        port,
        executable: docker.to_path_buf(),
    };
    Ok(Server {
        state,
        transport: Transport::Docker(pipette_subprocess::start_docker_process(
            docker,
            run,
            super::STOP_GRACE,
        )?),
    })
}

/// Pull `image` if the daemon lacks it, so the pull is not timed as startup:
/// the store records a docker runtime once, and an image pruned since would
/// otherwise be pulled inside the ready wait, its progress shown only at `-v`.
fn ensure_image(docker: &Path, image: &str) -> anyhow::Result<()> {
    let mut inspect = Command::new(docker);
    inspect
        .args(["image", "inspect", image])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    pipette_subprocess::echo_debug(&inspect);
    #[expect(clippy::disallowed_methods, reason = "every stream is discarded")]
    let present = inspect
        .status()
        .with_context(|| format!("failed to run {}", docker.display()))?
        .success();
    if present {
        return Ok(());
    }
    tracing::info!(%image, "image is not in the docker daemon; pulling it");
    let mut pull = Command::new(docker);
    pull.args(["pull", image])
        .stdin(Stdio::null())
        .stdout(std::io::stderr());
    pipette_subprocess::echo_info(&pull);
    #[expect(
        clippy::disallowed_methods,
        reason = "docker reports the pull on stdout, sent to stderr above"
    )]
    let status = pull
        .status()
        .with_context(|| format!("failed to run {}", docker.display()))?;
    anyhow::ensure!(
        status.success(),
        "`docker pull {image}` failed with {status}"
    );
    Ok(())
}

/// Published on `127.0.0.1:<port>` only.
fn docker_run(
    runtime: &Runtime,
    flags: &RuntimeFlags,
    server_args: &[String],
    mounts: &[(PathBuf, PathBuf)],
    workspace_label: &str,
    client_config: Option<&TorchOaiExtraRuntimeConfiguration>,
    port: u16,
) -> anyhow::Result<DockerRun> {
    let kind = ServerKind::of(runtime)?;
    let (image, flavor) = match runtime {
        Runtime::DockerVllm(d) => (
            format!("{}:{}", d.image_name.as_ref(), d.image_tag.as_ref()),
            d.flavor,
        ),
        Runtime::DockerSglang(d) => (
            format!("{}:{}", d.image_name.as_ref(), d.image_tag.as_ref()),
            sglang_to_vllm_flavor(d.flavor),
        ),
        other @ (Runtime::LlamacppCliStockTools(_)
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
        | Runtime::AppleFoundation(_)) => {
            anyhow::bail!("`{}` is not a docker runtime", other.headless_token())
        }
    };
    let mut run = DockerRun {
        name: random_container_name(),
        image,
        options: vec![
            "--label".to_owned(),
            format!("{}={workspace_label}", crate::cleanup::WORKSPACE_LABEL),
            "--label".to_owned(),
            format!(
                "{}={}",
                crate::cleanup::OWNER_LABEL,
                pipette_subprocess::owner()
            ),
            "-p".to_owned(),
            format!("127.0.0.1:{port}:{}", kind.default_port()),
        ],
        args: kind
            .container_command()
            .iter()
            .copied()
            .chain([kind.model_arg_flag(), CONTAINER_MODEL_PATH])
            .map(str::to_owned)
            .chain(server_args.iter().cloned())
            .collect(),
        envs: Vec::new(),
    };
    // GPU access dispatches on the flavor; `gpus` applies only to `NvidiaGpu`.
    run.options.extend(gpu_access_args(flavor, flags.gpus()));
    if let Some(shm) = flags.shm_size() {
        run.options
            .extend(["--shm-size".to_owned(), shm.to_string()]);
    }
    if let Some(ipc) = flags.ipc() {
        run.options
            .extend(["--ipc".to_owned(), ipc.as_str().to_owned()]);
    }
    run.envs.extend(
        flags
            .envs()
            .iter()
            .map(|entry| cell_forward(entry.name(), entry.value().unwrap_or_default()))
            .collect::<anyhow::Result<Vec<_>>>()?,
    );
    // The client's configuration goes before the cell's forwards above, so the
    // cell's own win. A device pin indexes the container's view, which matches
    // the host's only when every device is exposed (`--gpus all`, or the
    // daemon's default runtime); nothing checks a narrower `gpus`.
    pipette_subprocess::forward_client_config(&mut run, client_config);
    run.options.extend(mounts.iter().flat_map(|(src, dst)| {
        [
            "-v".to_owned(),
            format!("{}:{}", src.display(), dst.display()),
        ]
    }));
    Ok(run)
}

/// A cell's env entry as a forward. An empty value is the inherit form, `K` and
/// `K=` alike, since `-e NAME` has no spelling for an explicit empty value; it
/// fails here when this process has nothing to forward rather than starting
/// the server without it.
fn cell_forward(name: &str, value: &str) -> anyhow::Result<EnvForward> {
    let entry = if value.is_empty() {
        match std::env::var(name) {
            Ok(_) => name.to_owned(),
            Err(std::env::VarError::NotPresent) => anyhow::bail!(
                "env `{name}`: variable not set in this process's environment; \
                 export it before running, or set `{name}=<value>` in the cell's envs"
            ),
            Err(std::env::VarError::NotUnicode(_)) => anyhow::bail!(
                "env `{name}`: variable is set but contains non-UTF-8 bytes; \
                 set `{name}=<value>` in the cell's envs with a valid UTF-8 value"
            ),
        }
    } else {
        format!("{name}={value}")
    };
    Ok(EnvForward::try_new(entry)?)
}

#[cfg(test)]
mod tests {
    #[cfg(unix)]
    use std::time::Duration;

    use rstest::rstest;

    #[cfg(unix)]
    use pipette_ops::run_response::RunResponseBuilder;
    #[cfg(unix)]
    use pipette_plan_types::BenchmarkResultData;
    use pipette_plan_types::{
        DockerIpcMode, DockerSglang, DockerShmSize, DockerVllm, NonEmptyString, SglangFlavor,
    };

    #[cfg(unix)]
    use super::super::test_support::{
        fake_docker, published_port, ready_once, serve_fake, write_executable,
    };
    use super::*;

    /// AMD addresses GPUs through device mounts, never `--gpus`.
    const AMD_ARGS: &[&str] = &[
        "--device",
        "/dev/kfd",
        "--device",
        "/dev/dri",
        "--group-add",
        "video",
    ];

    #[test]
    fn random_container_name_has_expected_shape() {
        let name = random_container_name();
        assert!(name.starts_with("pipette-bench-"), "got {name}");
        assert_eq!(name.len(), "pipette-bench-".len() + 8);
    }

    // A stray `gpus` value must not reach AMD or CPU even if it survives the
    // flag resolution's flavor gate.
    #[rstest]
    #[case::nvidia_all(VllmFlavor::NvidiaGpu, Some("all"), &["--gpus", "all"])]
    #[case::nvidia_device_list(VllmFlavor::NvidiaGpu, Some("device=0,1"), &["--gpus", "device=0,1"])]
    #[case::nvidia_unset_omits_flag(VllmFlavor::NvidiaGpu, None, &[])]
    #[case::amd_device_flags(VllmFlavor::AmdGpu, None, AMD_ARGS)]
    #[case::amd_ignores_gpus(VllmFlavor::AmdGpu, Some("all"), AMD_ARGS)]
    #[case::cpu_nothing(VllmFlavor::Cpu, None, &[])]
    #[case::cpu_ignores_gpus(VllmFlavor::Cpu, Some("all"), &[])]
    fn gpu_access_args_cases(
        #[case] flavor: VllmFlavor,
        #[case] gpus_spec: Option<&str>,
        #[case] expected: &[&str],
    ) {
        assert_eq!(gpu_access_args(flavor, gpus_spec), expected);
    }

    fn runtime() -> anyhow::Result<Runtime> {
        Ok(Runtime::DockerVllm(DockerVllm {
            image_name: NonEmptyString::try_new("vllm/vllm-openai".to_owned())?,
            image_tag: NonEmptyString::try_new("v0.20.2".to_owned())?,
            flavor: VllmFlavor::NvidiaGpu,
        }))
    }

    fn flags(envs: &[&str]) -> anyhow::Result<RuntimeFlags> {
        Ok(RuntimeFlags::EndToEndLatencyDockerVllmTorch {
            tensor_parallel_size: None,
            dtype: None,
            max_model_len: None,
            prefix_caching: None,
            gpus: Some("all".to_owned()),
            shm_size: Some(DockerShmSize::try_new("16g")?),
            ipc: Some(DockerIpcMode::Host),
            envs: envs
                .iter()
                .map(|entry| EnvForward::try_new((*entry).to_owned()))
                .collect::<Result<_, _>>()?,
            raw: vec![],
        })
    }

    fn model_mount() -> Vec<(PathBuf, PathBuf)> {
        vec![(
            PathBuf::from("/data/models/abc/blobs"),
            PathBuf::from(CONTAINER_MODEL_PATH),
        )]
    }

    fn server_args() -> Vec<String> {
        vec!["--max-model-len".to_owned(), "4096".to_owned()]
    }

    /// Every variable is a forward, none an option: the client's configuration
    /// first, then the cell's, so the later win.
    #[test]
    fn a_docker_run_publishes_on_loopback_and_orders_its_forwards() -> anyhow::Result<()> {
        let client_config = TorchOaiExtraRuntimeConfiguration::from(vec![EnvForward::try_new(
            "CUDA_VISIBLE_DEVICES=1".to_owned(),
        )?]);
        // `PATH` stands in for a forward the operator exported: every test
        // process has one.
        let run = docker_run(
            &runtime()?,
            &flags(&["PATH", "VLLM_USE_V1=1"])?,
            &server_args(),
            &model_mount(),
            "/ws/.pipette",
            Some(&client_config),
            20001,
        )?;

        assert!(run.name.starts_with("pipette-bench-"), "{}", run.name);
        assert_eq!(run.image, "vllm/vllm-openai:v0.20.2");
        assert_eq!(
            run.options,
            [
                "--label",
                "pipette.workspace=/ws/.pipette",
                "--label",
                format!("pipette.owner={}", pipette_subprocess::owner()).as_str(),
                "-p",
                "127.0.0.1:20001:8000",
                "--gpus",
                "all",
                "--shm-size",
                "16g",
                "--ipc",
                "host",
                "-v",
                "/data/models/abc/blobs:/models/model",
            ]
        );
        assert_eq!(
            run.args,
            ["--model", "/models/model", "--max-model-len", "4096"]
        );
        assert_eq!(
            run.envs.iter().map(AsRef::as_ref).collect::<Vec<&str>>(),
            ["CUDA_VISIBLE_DEVICES=1", "PATH", "VLLM_USE_V1=1"]
        );
        Ok(())
    }

    fn sglang_runtime() -> anyhow::Result<Runtime> {
        Ok(Runtime::DockerSglang(DockerSglang {
            image_name: NonEmptyString::try_new("lmsysorg/sglang".to_owned())?,
            image_tag: NonEmptyString::try_new("v0.5.21".to_owned())?,
            flavor: SglangFlavor::NvidiaGpu,
        }))
    }

    /// vLLM's image enters at its server; SGLang's at a wrapper that runs its
    /// arguments, so the launch names the server and binds every interface.
    #[rstest]
    #[case::vllm(runtime(), &["--model", "/models/model", "--max-model-len", "4096"])]
    #[case::sglang(sglang_runtime(), &[
        "python3", "-m", "sglang.launch_server", "--host", "0.0.0.0",
        "--model-path", "/models/model", "--max-model-len", "4096",
    ])]
    fn each_server_gets_its_container_command(
        #[case] runtime: anyhow::Result<Runtime>,
        #[case] expected: &[&str],
    ) -> anyhow::Result<()> {
        let run = docker_run(
            &runtime?,
            &flags(&[])?,
            &server_args(),
            &model_mount(),
            "/ws/.pipette",
            None,
            20001,
        )?;
        assert_eq!(run.args, expected);
        Ok(())
    }

    #[test]
    fn an_inherited_forward_that_is_not_set_is_refused() -> anyhow::Result<()> {
        let name = "PIPETTE_TORCH_OAI_TEST_NEVER_SET";
        let err = docker_run(
            &runtime()?,
            &flags(&[name])?,
            &server_args(),
            &model_mount(),
            "/ws/.pipette",
            None,
            20001,
        )
        .err()
        .context("expected a refusal")?;
        let message = format!("{err:#}");
        assert!(message.contains(name), "{message}");
        assert!(message.contains("not set"), "{message}");
        Ok(())
    }

    /// Through a stand-in `docker`: the run is attached and ready once the
    /// published port answers, the client's value reaches the CLI's
    /// environment, and the stop is `docker stop` with the 30 s grace.
    #[cfg(unix)]
    #[test]
    fn a_container_runs_attached_until_docker_stops_it() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let docker = fake_docker(dir.path(), r#"echo "up with $CUDA_VISIBLE_DEVICES""#)?;
        let http = serve_fake(&dir.path().join("argv"), published_port, ready_once)?;
        let client_config = TorchOaiExtraRuntimeConfiguration::from(vec![EnvForward::try_new(
            "CUDA_VISIBLE_DEVICES=1".to_owned(),
        )?]);

        let mut server = launch(
            &docker,
            &runtime()?,
            &flags(&[])?,
            &server_args(),
            &model_mount(),
            "/ws/.pipette",
            Some(&client_config),
        )?;
        server.wait_ready(Duration::from_secs(10))?;
        let argv = http
            .join()
            .map_err(|_| anyhow::anyhow!("the fake server panicked"))??;
        let crate::memprobe::ServerCgroup::Container { docker: bin, name } = server.cgroup() else {
            anyhow::bail!("a docker server has a container");
        };
        assert_eq!(bin, docker);
        let port = server.state().port;
        assert!(
            argv.starts_with(&format!(
                "run --rm --name {name} --label pipette.workspace=/ws/.pipette \
                 --label pipette.owner={} -p 127.0.0.1:{port}:8000 ",
                pipette_subprocess::owner()
            )),
            "{argv}"
        );
        server.stop();
        assert_eq!(
            std::fs::read_to_string(dir.path().join("calls"))?,
            format!("stop -t 30 {name}\n")
        );
        assert_eq!(
            server.output().runtime_stdout.as_deref(),
            Some("up with 1\nstopped\n")
        );
        Ok(())
    }

    /// A response records the container as it does a process: stopped first,
    /// so its output is whole, under the request and launcher the cell named.
    #[cfg(unix)]
    #[test]
    fn a_response_records_the_stopped_container() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let docker = fake_docker(dir.path(), "echo up")?;
        let http = serve_fake(&dir.path().join("argv"), published_port, ready_once)?;
        let mut server = launch(
            &docker,
            &runtime()?,
            &flags(&[])?,
            &server_args(),
            &model_mount(),
            "/ws/.pipette",
            None,
        )?;
        server.wait_ready(Duration::from_secs(10))?;
        http.join()
            .map_err(|_| anyhow::anyhow!("the fake server panicked"))??;

        let response = RunResponseBuilder::new(BenchmarkResultData::PrefillThroughput {
            prefill_time_ms: 1.0,
            prefill_time_ms_stddev: None,
        })
        .executable(docker.display().to_string())
        .command(vec!["POST /v1/completions".to_owned()])
        .process(&mut server)
        .build();

        assert_eq!(
            response.runtime_output.runtime_stdout.as_deref(),
            Some("up\nstopped\n")
        );
        assert_eq!(response.command, ["POST /v1/completions"]);
        assert_eq!(response.executable, Some(docker.display().to_string()));
        Ok(())
    }

    /// An image the daemon already has is not pulled.
    #[cfg(unix)]
    #[test]
    fn a_missing_image_is_pulled_before_the_wait() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let docker = write_executable(
            &dir.path().join("docker"),
            &format!(
                r#"#!/bin/sh
DIR='{}'
echo "$@" >> "$DIR/calls"
case "$1" in
  image) [ -e "$DIR/pulled" ] ;;
  pull) touch "$DIR/pulled" ;;
esac
"#,
                dir.path().display()
            ),
        )?;
        let image = "vllm/vllm-openai:v0.20.2";
        ensure_image(&docker, image)?;
        ensure_image(&docker, image)?;
        assert_eq!(
            std::fs::read_to_string(dir.path().join("calls"))?,
            format!("image inspect {image}\npull {image}\nimage inspect {image}\n")
        );
        Ok(())
    }

    #[test]
    fn a_uv_runtime_is_refused() -> anyhow::Result<()> {
        use pipette_plan_types::{
            AbsolutePath, UvBuild, UvPythonVersion, UvRuntimeSource, UvServerVersion, UvSglang,
        };

        let uv = Runtime::UvSglang(UvSglang {
            server_version: UvServerVersion::try_new("0.5.0".to_owned())?,
            build: UvBuild::try_new("cu129".to_owned())?,
            python_version: UvPythonVersion::try_new("3.12".to_owned())?,
            source: UvRuntimeSource::AbsolutePreinstalled {
                dir: AbsolutePath::try_new("/venv".to_owned())?,
            },
        });
        let err = docker_run(
            &uv,
            &flags(&[])?,
            &server_args(),
            &model_mount(),
            "/ws/.pipette",
            None,
            20001,
        )
        .err()
        .context("expected a refusal")?;
        assert!(
            err.to_string().contains("not a docker runtime"),
            "got {err:#}"
        );
        Ok(())
    }
}
