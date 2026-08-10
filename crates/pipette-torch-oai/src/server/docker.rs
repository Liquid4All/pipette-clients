//! Docker container launch, stop, and log tail for torch-oai servers.

use std::{
    fmt,
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

use anyhow::Context;
use serde::{Deserialize, Serialize};

use pipette_plan_types::VllmFlavor;
use pipette_subprocess::cleanup::TeardownToken;

use super::{pick_free_port, DockerLaunchEnv, LaunchSpec, ServerState};

/// The `docker` binary every function in this module takes a path to.
pub fn resolve_docker() -> anyhow::Result<PathBuf> {
    pipette_subprocess::which("docker").context("docker not found on PATH; install Docker Engine")
}

/// Newtype wrapping a docker container id. [`Display`](std::fmt::Display) formats as the
/// short id (first 12 chars) so log lines stay readable; use
/// `.0`/`AsRef<str>` for the full id.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContainerId(pub String);

impl ContainerId {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl AsRef<str> for ContainerId {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ContainerId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let short: String = self.0.chars().take(12).collect();
        write!(f, "{short}")
    }
}

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
        log::debug!(
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
        log::debug!("gpus is ignored for CPU runtimes");
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

/// Launch a detached container. Returns the in-memory state describing it
/// plus the [`TeardownToken`] the caller must hand to [`stop`] (or pass
/// directly to [`pipette_subprocess::cleanup::deregister`]) once it has torn the
/// container down on the happy path. Forgetting the token leaves the
/// signal handler firing a stale `docker stop` against an already-gone
/// container on the next ^C — harmless, but it violates the contract
/// documented on [`pipette_subprocess::cleanup::register`].
pub fn start(
    docker: &Path,
    env: DockerLaunchEnv,
    spec: LaunchSpec,
) -> anyhow::Result<(ServerState, TeardownToken)> {
    let container_port = spec.default_port();
    let host_port = pick_free_port()?;
    let container_name = random_container_name();

    let mut cmd = Command::new(docker);
    // `127.0.0.1` is hardcoded — the design pins pipette-torch-oai to
    // loopback-only since the tool exists to drive the engine for
    // measurement, not to host it for outside consumers. Operators who
    // need to publish to the network use `docker` directly.
    cmd.arg("run")
        .arg("-d")
        .arg("--rm")
        .arg("--name")
        .arg(&container_name)
        .arg("--label")
        .arg(format!(
            "{}={}",
            crate::cleanup::WORKSPACE_LABEL,
            env.workspace_label
        ))
        .arg("-p")
        .arg(format!("127.0.0.1:{host_port}:{container_port}"));

    let (kind, envs, gpus, shm_size, ipc, mounts) = match &spec {
        LaunchSpec::Docker {
            kind,
            envs,
            gpus,
            shm_size,
            ipc,
            mounts,
            ..
        } => (*kind, envs, gpus, shm_size, ipc, mounts),
        LaunchSpec::Uv { .. } => {
            anyhow::bail!("server::start (docker) called with a uv LaunchSpec");
        }
    };

    // GPU access flags dispatch on the launch env flavor — not on
    // `LaunchSpec.gpus`, which only applies to `NvidiaGpu`.
    cmd.args(gpu_access_args(env.flavor, gpus.as_deref()));
    if let Some(shm) = shm_size {
        cmd.arg("--shm-size").arg(shm);
    }
    if let Some(ipc) = ipc {
        cmd.arg("--ipc").arg(ipc);
    }

    if let Some(home) = &env.hf_home {
        std::fs::create_dir_all(home)
            .with_context(|| format!("failed to create hf-home dir {}", home.display()))?;
        cmd.arg("-v")
            .arg(format!("{}:/root/.cache/huggingface", home.display()))
            .arg("-e")
            .arg("HF_HOME=/root/.cache/huggingface");
    }
    envs.iter().try_for_each(|(k, v)| {
        if v.is_empty() {
            // Inherit form: `-e <NAME>` (no `=value`) tells docker to read
            // the value from the docker CLI's environment. Used both
            // intentionally (a bare `K` env entry forwards $K) and as the
            // secret-safe path — the value never hits argv.
            //
            // Fail fast if `$K` is unset or non-UTF-8: silently
            // forwarding an empty value would just push the misconfig
            // downstream (e.g. vLLM rejecting an empty HF_TOKEN with a
            // less helpful error). The cell clearly meant to forward
            // something.
            let value = match std::env::var(k) {
                Ok(v) => v,
                Err(std::env::VarError::NotPresent) => anyhow::bail!(
                    "env `{k}`: variable not set in this process's environment; \
                     export it before running, or set `{k}=<value>` in the cell's envs"
                ),
                Err(std::env::VarError::NotUnicode(_)) => anyhow::bail!(
                    "env `{k}`: variable is set but contains non-UTF-8 bytes; \
                     set `{k}=<value>` in the cell's envs with a valid UTF-8 value"
                ),
            };
            cmd.env(k, value).arg("-e").arg(k);
        } else {
            // Explicit form: the cell supplied `K=V`; the value is already on
            // this process's argv, so re-emitting it on docker's argv doesn't
            // widen the exposure. For secrets, prefer the bare `K` inherit
            // form so neither argv is touched.
            cmd.arg("-e").arg(format!("{k}={v}"));
        }
        Ok(())
    })?;
    cmd.args(mounts.iter().flat_map(|(src, dst)| {
        [
            "-v".to_string(),
            format!("{}:{}", src.display(), dst.display()),
        ]
    }));

    cmd.arg(&env.image_ref);
    // Wire order: server's model flag → server_args (shell-split) →
    // trailing `--` args. The model flag is emitted here (using `env.model`
    // and `LaunchSpec` kind) rather than threaded through assembled_args —
    // this keeps `command_args` a pure carrier of trailing `--` args and
    // matches `uv::launch`, which reads the model off `UvLaunchEnv`.
    cmd.arg(spec.model_arg_flag()).arg(&env.model);
    let args = spec.assembled_args();
    cmd.args(&args);

    log::info!(
        "[server] starting docker container {container_name} from {} \
         (host_port={host_port}, container_port={container_port})",
        env.image_ref,
    );
    pipette_subprocess::echo_info(&cmd);
    // Inherit stderr so docker daemon errors (image-not-found, NVIDIA hook
    // failure, …) stream through live. Capture stdout — that's just the
    // container id printed by `docker run -d`.
    let mut child = cmd
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .with_context(|| format!("failed to spawn {}", docker.display()))?;
    let mut stdout_buf = String::new();
    if let Some(mut out) = child.stdout.take() {
        use std::io::Read;
        out.read_to_string(&mut stdout_buf)
            .context("failed to read docker run stdout")?;
    }
    let status = child
        .wait()
        .with_context(|| format!("failed to wait for {}", docker.display()))?;
    if !status.success() {
        anyhow::bail!("docker run failed with {status}");
    }
    let container_id_raw = stdout_buf.trim().to_string();
    if container_id_raw.is_empty() {
        anyhow::bail!("docker run returned an empty container id");
    }
    let container_id = ContainerId(container_id_raw);

    log::info!("container {container_id} bound to host port {host_port}");
    // Register with the signal-handler registry as soon as docker
    // confirms the container exists — any later failure (readiness,
    // dispatch) is then covered by both the Drop guard and the SIGINT
    // handler. The token is returned to the caller so the Drop guard
    // can `cleanup::deregister` it after the happy-path stop succeeds;
    // otherwise the registry would grow unboundedly across runs and
    // SIGINT during run N would fire stale `docker stop` calls against
    // containers from runs 1..N-1.
    let teardown_token = crate::cleanup::register_docker(docker, container_id.as_str());

    let state = ServerState::Docker {
        kind,
        container_id,
        docker_bin: docker.to_path_buf(),
        port: host_port,
    };
    Ok((state, teardown_token))
}

/// Stop the container described by `state`. No-op if it's already gone.
pub fn stop(docker: &Path, state: &ServerState, timeout_secs: u32) -> anyhow::Result<()> {
    let container_id = state
        .container_id()
        .context("server::stop called on a non-docker ServerState; use server::uv::stop instead")?;
    // An inspect we couldn't interpret is treated as "nothing to stop", as it
    // always has been here: `docker stop` against a daemon that just refused
    // an inspect has nothing better to report than the inspect did.
    if container_liveness(docker, container_id.as_str())? != ContainerLiveness::Running {
        log::info!("container {container_id} already stopped");
        return Ok(());
    }
    log::info!("stopping container {container_id}");
    let mut cmd = Command::new(docker);
    cmd.args([
        "stop",
        "-t",
        &timeout_secs.to_string(),
        container_id.as_str(),
    ])
    .stdout(Stdio::null());
    pipette_subprocess::echo_info(&cmd);
    let status = cmd
        .status()
        .with_context(|| format!("failed to run docker stop {container_id}"))?;
    if !status.success() {
        anyhow::bail!("docker stop {container_id} exited with {status}");
    }
    Ok(())
}

/// What `docker inspect` was able to tell us about a container.
///
/// `Unknown` is a third answer rather than an error because `docker inspect`
/// exits non-zero both when the container is gone and when the CLI never
/// reached the daemon, and only the first says anything about the container.
/// A caller that aborts a run on `NotRunning` must not abort on a daemon hiccup.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContainerLiveness {
    Running,
    NotRunning,
    Unknown,
}

/// Ask the daemon whether `container_id` is still running.
///
/// `Err` only when the `docker` binary itself couldn't be spawned; a docker
/// that ran and failed answers [`ContainerLiveness::Unknown`] unless it said
/// the object no longer exists.
pub fn container_liveness(docker: &Path, container_id: &str) -> anyhow::Result<ContainerLiveness> {
    let mut cmd = Command::new(docker);
    cmd.args(["inspect", "--format", "{{.State.Running}}", container_id]);
    pipette_subprocess::echo_debug(&cmd);
    let output = cmd
        .output()
        .with_context(|| format!("failed to inspect container {container_id}"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Ok(inspect_failure_liveness(&stderr));
    }
    let s = String::from_utf8_lossy(&output.stdout).trim().to_string();
    Ok(if s == "true" {
        ContainerLiveness::Running
    } else {
        ContainerLiveness::NotRunning
    })
}

/// Classify a failed `docker inspect` from its stderr. Both docker and podman
/// spell a removed container "No such object" / "No such container"; anything
/// else (daemon unreachable, socket EPERM, CLI misuse) is no evidence.
fn inspect_failure_liveness(stderr: &str) -> ContainerLiveness {
    let stderr = stderr.to_lowercase();
    if stderr.contains("no such object") || stderr.contains("no such container") {
        ContainerLiveness::NotRunning
    } else {
        ContainerLiveness::Unknown
    }
}

/// Stream the container's stdout/stderr to the parent process's
/// stdout/stderr in the background. Returns the spawned `docker logs -f`
/// child — caller should kill it during teardown to avoid a brief race
/// where post-container output interleaves with the next step.
pub fn tail_logs(docker: &Path, container_id: &str) -> anyhow::Result<std::process::Child> {
    let mut cmd = Command::new(docker);
    cmd.args(["logs", "-f", container_id])
        .stdin(Stdio::null())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit());
    pipette_subprocess::echo_info(&cmd);
    cmd.spawn()
        .with_context(|| format!("failed to spawn `docker logs -f {container_id}`"))
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use pipette_plan_types::VllmFlavor;

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
    fn container_id_display_truncates_to_short_id() {
        let id = ContainerId("abcdef0123456789feedface".to_string());
        assert_eq!(id.to_string(), "abcdef012345");
    }

    #[test]
    fn random_container_name_has_expected_shape() {
        let name = random_container_name();
        assert!(name.starts_with("pipette-bench-"), "got {name}");
        // Fixed prefix + 8-char uuid suffix.
        assert_eq!(name.len(), "pipette-bench-".len() + 8);
    }

    // SweepPins the exact argv per flavor: a stray `gpus` value must not reach AMD
    // or CPU even if it survives the builder's flavor gate.
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

    // The readiness wait aborts a run on `NotRunning`, so an unreachable daemon or a
    // permission error has to land on `Unknown` instead.
    #[rstest]
    #[case::docker_removed("Error: No such object: abc", ContainerLiveness::NotRunning)]
    #[case::podman_removed("Error: no such container abc", ContainerLiveness::NotRunning)]
    #[case::daemon_down(
        "Cannot connect to the Docker daemon at unix:///var/run/docker.sock. Is the docker daemon running?",
        ContainerLiveness::Unknown
    )]
    #[case::socket_permission(
        "permission denied while trying to connect to the Docker daemon socket",
        ContainerLiveness::Unknown
    )]
    #[case::silent_failure("", ContainerLiveness::Unknown)]
    fn inspect_failure_liveness_cases(#[case] stderr: &str, #[case] expected: ContainerLiveness) {
        assert_eq!(inspect_failure_liveness(stderr), expected);
    }
}
