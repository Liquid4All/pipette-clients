//! In-process server lifecycle for `benchmarks run` (docker + shared types).
//!
//! Docker spawn/stop: [`docker`]. UV spawn: [`uv`] — the real setsid/PGID
//! implementation on Linux, an erroring stub elsewhere. Shared types and
//! readiness polling live here.

use std::{
    fmt,
    net::TcpListener,
    path::{Path, PathBuf},
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use anyhow::Context;

use pipette_plan_types::VllmFlavor;

pub mod docker;

// Only one of these ever compiles, so a macOS or Windows build type-checks the
// stub and never the real module. Verify changes to either arm against a Linux
// target before pushing — `cargo check` on a Mac cannot see a break in `uv.rs`.
#[cfg(target_os = "linux")]
pub mod uv;

#[cfg(not(target_os = "linux"))]
#[path = "uv_stub.rs"]
pub mod uv;

pub use docker::{
    container_liveness, resolve_docker, start, stop, tail_logs, ContainerId, ContainerLiveness,
};

pub const DEFAULT_SHM_SIZE: &str = "16g";
pub const DEFAULT_GPUS: &str = "all";

/// vLLM vs SGLang — ports, model flags, and entrypoints key off this.
/// Orthogonal to docker vs uv transport.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServerKind {
    Vllm,
    Sglang,
}

impl ServerKind {
    pub fn default_port(self) -> u16 {
        match self {
            Self::Vllm => 8000,
            Self::Sglang => 30000,
        }
    }

    /// Server CLI model flag: vLLM `--model`, SGLang `--model-path`.
    pub fn model_arg_flag(self) -> &'static str {
        match self {
            Self::Vllm => "--model",
            Self::Sglang => "--model-path",
        }
    }
}

/// Running server handle: transport (docker container vs uv process group)
/// plus [`ServerKind`].
///
/// [`base_url`](Self::base_url) is always loopback — `pipette-torch-oai`
/// drives the server for measurement, not for outside consumers, so there
/// is no publish-host setting.
///
/// `Uv` carries the leader's PID + PGID (equal because the leader `setsid`s
/// before exec); the optional `log_tail` owns the JoinHandle for the
/// background stdout/stderr tail and is drained by `uv::stop`.
///
/// Not `Clone` — uv holds a JoinHandle which is move-only.
#[derive(Debug)]
pub enum ServerState {
    Docker {
        kind: ServerKind,
        container_id: ContainerId,
        /// `docker` binary this container was launched with. Carried here so
        /// asking the daemon about the container needs nothing but the state.
        docker_bin: PathBuf,
        port: u16,
    },
    Uv {
        kind: ServerKind,
        /// The binary this server was launched with — the venv's `python`.
        /// Carried for the same reason `Docker` carries `docker_bin`: a stored
        /// result should name what produced it.
        executable: PathBuf,
        /// Leader PID (`setsid` in pre_exec ⇒ also the session id).
        pid: u32,
        /// Process group id — equal to `pid` after `setsid`.
        pgid: u32,
        port: u16,
        /// Background stdout/stderr tail; joined on stop.
        log_tail: Option<JoinHandle<()>>,
    },
}

impl ServerState {
    /// Default HTTP path polled to detect readiness.
    pub fn default_ready_path(&self) -> &'static str {
        "/v1/models"
    }

    pub fn port(&self) -> u16 {
        match self {
            Self::Docker { port, .. } | Self::Uv { port, .. } => *port,
        }
    }

    /// The binary that launched this server, for the result's provenance.
    pub fn executable(&self) -> &Path {
        match self {
            Self::Docker { docker_bin, .. } => docker_bin,
            Self::Uv { executable, .. } => executable,
        }
    }

    pub fn kind(&self) -> ServerKind {
        match self {
            Self::Docker { kind, .. } | Self::Uv { kind, .. } => *kind,
        }
    }

    /// Docker container id, when this is a docker server.
    pub fn container_id(&self) -> Option<&ContainerId> {
        match self {
            Self::Docker { container_id, .. } => Some(container_id),
            Self::Uv { .. } => None,
        }
    }

    /// Leader PID for a uv server tree.
    pub fn pid(&self) -> Option<u32> {
        match self {
            Self::Uv { pid, .. } => Some(*pid),
            Self::Docker { .. } => None,
        }
    }

    /// Process group id for a uv server tree.
    pub fn pgid(&self) -> Option<u32> {
        match self {
            Self::Uv { pgid, .. } => Some(*pgid),
            Self::Docker { .. } => None,
        }
    }

    pub fn base_url(&self) -> String {
        format!("http://127.0.0.1:{}", self.port())
    }

    pub fn ready_url(&self) -> String {
        format!("{}{}", self.base_url(), self.default_ready_path(),)
    }
}

/// Launch argv bags: docker vs uv transport, plus [`ServerKind`].
///
/// `gpus` is flavor-conditional at build (`None` for AmdGpu/Cpu); AMD
/// device flags are emitted by `docker::gpu_access_args` at argv time.
#[derive(Debug, Clone)]
pub enum LaunchSpec {
    Docker {
        kind: ServerKind,
        envs: Vec<(String, String)>,
        gpus: Option<String>,
        shm_size: Option<String>,
        ipc: Option<String>,
        mounts: Vec<(PathBuf, PathBuf)>,
        server_args: Vec<String>,
        command_args: Vec<String>,
    },
    Uv {
        kind: ServerKind,
        /// Operator `--env K[=V]`; bare `K` inherits from the parent env.
        envs: Vec<(String, String)>,
        server_args: Vec<String>,
        command_args: Vec<String>,
    },
}

impl LaunchSpec {
    pub fn kind(&self) -> ServerKind {
        match self {
            Self::Docker { kind, .. } | Self::Uv { kind, .. } => *kind,
        }
    }

    /// Concatenates `server_args ++ command_args` for the launch site.
    /// Model flag lives on `DockerLaunchEnv` / `UvLaunchEnv`, not here.
    /// Eval-checkpoint digest folds this via `execute::eval::eval_extras`.
    pub fn assembled_args(&self) -> Vec<String> {
        let (server_args, command_args) = match self {
            Self::Docker {
                server_args,
                command_args,
                ..
            }
            | Self::Uv {
                server_args,
                command_args,
                ..
            } => (server_args, command_args),
        };
        let mut out = Vec::with_capacity(server_args.len() + command_args.len());
        out.extend(server_args.iter().cloned());
        out.extend(command_args.iter().cloned());
        out
    }

    pub fn default_port(&self) -> u16 {
        self.kind().default_port()
    }

    pub fn model_arg_flag(&self) -> &'static str {
        self.kind().model_arg_flag()
    }

    pub fn server_args(&self) -> &[String] {
        match self {
            Self::Docker { server_args, .. } | Self::Uv { server_args, .. } => server_args,
        }
    }
}

/// Inputs the launch site needs that don't fit cleanly on `LaunchSpec`
/// because they're either workspace-global (label) or runtime-derived
/// (image ref, flavor, hf_home).
///
/// Nothing about the network binding is configurable: the host port is a free
/// ephemeral picked pre-spawn, the bind host is always `127.0.0.1`, the
/// container port comes from `LaunchSpec` kind, the container name is
/// generated, and the ready path is `ServerState::default_ready_path()`.
#[derive(Debug, Clone)]
pub struct DockerLaunchEnv {
    /// `<repo>:<tag>` string handed to `docker run`.
    pub image_ref: String,
    /// Workspace label value (absolute path of the workspace root). Set as
    /// the value of the `pipette-torch-oai.workspace` docker label so a
    /// subsequent run from this workspace can reap containers our process
    /// orphaned (Ctrl-C / SIGKILL / panic=abort).
    pub workspace_label: String,
    /// Host directory to bind-mount as `/root/.cache/huggingface`.
    pub hf_home: Option<PathBuf>,
    /// HF slug or in-container model path. `start` emits it via the
    /// server's native model flag (`--model` for vLLM, `--model-path`
    /// for SGLang) between the image ref and `assembled_args()` so the
    /// wire order is model flag, then runtime flags, then trailing args
    /// (model → `server_args` → trailing `--` args). Mirrors
    /// `UvLaunchEnv::model` on the uv engine path so `command_args`
    /// stays a pure carrier of trailing `--` args on both engines.
    pub model: String,
    /// GPU vendor flavor (canonicalized as [`VllmFlavor`]) for `docker run`
    /// device flags. Sourced from plan `DockerVllm`/`DockerSglang`.
    pub flavor: VllmFlavor,
}

/// Pick a free loopback port pre-spawn, shared by the docker and uv launch
/// sites so both allocate host ports the same way. There's a TOCTOU window
/// between this listener closing and the server binding the port — small
/// enough to be acceptable for a benchmark tool, and it matches the "client
/// picks a free ephemeral port" contract.
pub fn pick_free_port() -> anyhow::Result<u16> {
    let listener = TcpListener::bind("127.0.0.1:0")
        .context("failed to bind 127.0.0.1:0 for free-port discovery")?;
    let port = listener
        .local_addr()
        .context("failed to read assigned port")?
        .port();
    drop(listener);
    Ok(port)
}

/// How a server process ended, as reported by its transport's own liveness
/// primitive. Docker and uv answer different questions — see [`server_exited`]
/// — so each carries what its answer actually contains.
#[derive(Debug)]
enum ServerDeath {
    /// The leader was our direct child, so `waitpid` yields a decoded status.
    UvProcess { pid: u32, status: String },
    /// The daemon no longer lists the container as running.
    DockerContainer { container_id: ContainerId },
}

impl fmt::Display for ServerDeath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UvProcess { pid, status } => write!(
                f,
                "uv server pid {pid} is gone ({status}); anything it printed is \
                 above, prefixed `[uv-server pid={pid}]`"
            ),
            // `docker run --rm` means dockerd removed the container the moment
            // it exited, taking its exit code and `docker logs` buffer along —
            // all that survives is what `docker logs -f` already streamed.
            Self::DockerContainer { container_id } => write!(
                f,
                "container {container_id} is no longer running; it was started \
                 with --rm, so its exit code is gone with it and anything it \
                 printed is above"
            ),
        }
    }
}

/// `Some(_)` once the server process is gone, `None` while it is still up.
///
/// A probe with no answer to give reports `None`: a `docker inspect` that never
/// reached the daemon is no evidence the container died, and the readiness
/// deadline still bounds the wait either way.
fn server_exited(state: &ServerState) -> Option<ServerDeath> {
    match state {
        ServerState::Docker {
            container_id,
            docker_bin,
            ..
        } => match docker::container_liveness(docker_bin, container_id.as_str()) {
            Ok(ContainerLiveness::Running) => None,
            Ok(ContainerLiveness::NotRunning) => Some(ServerDeath::DockerContainer {
                container_id: container_id.clone(),
            }),
            Ok(ContainerLiveness::Unknown) => {
                log::debug!("liveness probe for container {container_id}: inspect gave no answer");
                None
            }
            Err(err) => {
                log::debug!("liveness probe for container {container_id}: {err:#}");
                None
            }
        },
        ServerState::Uv { pid, .. } => {
            uv::exit_status(*pid).map(|status| ServerDeath::UvProcess { pid: *pid, status })
        }
    }
}

/// Poll the server's ready URL until it answers, it dies, or `timeout` elapses.
///
/// Liveness is checked only after a failed HTTP probe, so a server that comes
/// up and answers never pays for it. A server that dies during startup fails in
/// seconds rather than holding the caller for the whole timeout.
pub fn wait_ready(state: &ServerState, timeout: Duration) -> anyhow::Result<()> {
    let ready_url = state.ready_url();
    let client = pipette_http::HttpClient::blocking_with_timeout("pipette", Duration::from_secs(5))
        .context("failed to build readiness HTTP client")?;
    let start = Instant::now();
    let deadline = start + timeout;
    let mut attempt: u32 = 0;
    loop {
        attempt += 1;
        match client.get(&ready_url).send() {
            Ok(resp) if resp.status().is_success() => return Ok(()),
            Ok(resp) => {
                log::debug!("readiness probe {attempt}: HTTP {}", resp.status());
            }
            Err(err) => {
                log::debug!("readiness probe {attempt}: {err}");
            }
        }
        if let Some(death) = server_exited(state) {
            anyhow::bail!(
                "server exited before becoming ready at {ready_url} after {:.1}s: {death}",
                start.elapsed().as_secs_f64()
            );
        }
        if Instant::now() >= deadline {
            anyhow::bail!(
                "server did not become ready at {ready_url} within {}s",
                timeout.as_secs()
            );
        }
        thread::sleep(Duration::from_secs(2));
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[test]
    fn pick_free_port_returns_a_non_privileged_port() -> anyhow::Result<()> {
        assert!(pick_free_port()? >= 1024);
        Ok(())
    }

    // The launch sites pass this straight to the server argv, so the exact
    // spelling is a wire contract with vLLM / SGLang.
    #[rstest]
    #[case::vllm(ServerKind::Vllm, "--model")]
    #[case::sglang(ServerKind::Sglang, "--model-path")]
    fn model_arg_flag_cases(#[case] kind: ServerKind, #[case] expected: &str) {
        assert_eq!(kind.model_arg_flag(), expected);
        assert_eq!(docker_spec(kind, &[], &[]).model_arg_flag(), expected);
    }

    /// A path no host can execute, so a state built with it never reaches a
    /// real docker daemon — including on a dev box that has one running.
    const UNRUNNABLE_DOCKER: &str = "/nonexistent/pipette-not-a-docker-binary";

    fn docker_state(kind: ServerKind, container_id: &str, port: u16) -> ServerState {
        ServerState::Docker {
            kind,
            container_id: ContainerId(container_id.to_string()),
            docker_bin: PathBuf::from(UNRUNNABLE_DOCKER),
            port,
        }
    }

    #[test]
    fn server_state_base_url_is_loopback() {
        let state = docker_state(ServerKind::Vllm, "abc", 41873);
        assert_eq!(state.base_url(), "http://127.0.0.1:41873");
        assert_eq!(state.ready_url(), "http://127.0.0.1:41873/v1/models");
        assert!(matches!(state, ServerState::Docker { .. }));
    }

    #[test]
    fn server_state_sglang_variant_carries_port() {
        let state = docker_state(ServerKind::Sglang, "xyz", 30000);
        assert_eq!(state.kind(), ServerKind::Sglang);
        assert_eq!(state.port(), 30000);
    }

    #[rstest]
    #[case::uv(
        ServerDeath::UvProcess {
            pid: 4242,
            status: "exit status 1".to_string(),
        },
        &["pid 4242", "exit status 1", "[uv-server pid=4242]"]
    )]
    #[case::docker(
        ServerDeath::DockerContainer {
            container_id: ContainerId("abcdef0123456789feedface".to_string()),
        },
        // Short id, and the --rm caveat: there is no exit code to report.
        &["container abcdef012345", "no longer running", "--rm"]
    )]
    fn server_death_message_cases(#[case] death: ServerDeath, #[case] expected: &[&str]) {
        let msg = death.to_string();
        for want in expected {
            assert!(msg.contains(want), "{msg:?} is missing {want:?}");
        }
    }

    /// A `docker` binary we can't execute is not evidence the container died,
    /// so the wait falls through to the deadline instead of reporting an exit.
    #[test]
    fn wait_ready_times_out_when_the_liveness_probe_cannot_run() -> anyhow::Result<()> {
        let state = docker_state(ServerKind::Vllm, "abc", pick_free_port()?);
        let err = wait_ready(&state, Duration::ZERO)
            .err()
            .context("expected a readiness error")?;
        let msg = format!("{err:#}");
        assert!(msg.contains("did not become ready"), "got {msg}");
        Ok(())
    }

    /// Write a stand-in `docker` that exits 1 with `stderr`, so the liveness
    /// path sees a CLI that ran and failed rather than one it couldn't spawn.
    /// Owned by the caller's tempdir and never on `PATH`, so the test doesn't
    /// depend on what the host image ships.
    #[cfg(unix)]
    fn failing_docker(dir: &std::path::Path, stderr: &str) -> anyhow::Result<PathBuf> {
        use std::os::unix::fs::PermissionsExt;

        let path = dir.join("docker");
        std::fs::write(&path, format!("#!/bin/sh\necho \"{stderr}\" >&2\nexit 1\n"))?;
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755))?;
        Ok(path)
    }

    #[cfg(unix)]
    fn docker_state_with_bin(docker_bin: PathBuf, port: u16) -> ServerState {
        ServerState::Docker {
            kind: ServerKind::Vllm,
            container_id: ContainerId("abc".to_string()),
            docker_bin,
            port,
        }
    }

    /// The point of the liveness check: a dead server ends the wait in
    /// milliseconds rather than after the (here, 10-minute) readiness timeout.
    #[cfg(unix)]
    #[test]
    fn wait_ready_fails_as_soon_as_the_server_is_gone() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let docker = failing_docker(dir.path(), "Error: No such object: abc")?;
        let state = docker_state_with_bin(docker, pick_free_port()?);
        let start = Instant::now();
        let err = wait_ready(&state, Duration::from_secs(600))
            .err()
            .context("expected a readiness error")?;
        let msg = format!("{err:#}");
        assert!(msg.contains("exited before becoming ready"), "got {msg}");
        assert!(msg.contains("no longer running"), "got {msg}");
        assert!(start.elapsed() < Duration::from_secs(30), "waited too long");
        Ok(())
    }

    /// An inspect that failed without saying the container is gone — a daemon
    /// restart mid-startup — must not abort a server that is still coming up.
    #[cfg(unix)]
    #[test]
    fn wait_ready_keeps_waiting_when_inspect_cannot_reach_the_daemon() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let docker = failing_docker(dir.path(), "Cannot connect to the Docker daemon")?;
        let state = docker_state_with_bin(docker, pick_free_port()?);
        let err = wait_ready(&state, Duration::ZERO)
            .err()
            .context("expected a readiness error")?;
        let msg = format!("{err:#}");
        assert!(msg.contains("did not become ready"), "got {msg}");
        Ok(())
    }

    fn docker_spec(kind: ServerKind, server_args: &[&str], command_args: &[&str]) -> LaunchSpec {
        LaunchSpec::Docker {
            kind,
            envs: vec![],
            gpus: None,
            shm_size: None,
            ipc: None,
            mounts: vec![],
            server_args: server_args.iter().map(|s| s.to_string()).collect(),
            command_args: command_args.iter().map(|s| s.to_string()).collect(),
        }
    }

    /// The uv variant assembles the same way. It lives here rather than with
    /// the uv integration test, which is `#![cfg(target_os = "linux")]` — a
    /// field renamed out from under a shape check should fail on the author's
    /// host, not three CI jobs later.
    #[test]
    fn launch_spec_uv_assembled_args() {
        let spec = LaunchSpec::Uv {
            kind: ServerKind::Vllm,
            envs: vec![("HF_TOKEN".to_string(), String::new())],
            server_args: vec!["--max-model-len".to_string(), "4096".to_string()],
            command_args: vec!["--extra".to_string()],
        };
        assert_eq!(
            spec.assembled_args(),
            ["--max-model-len", "4096", "--extra"]
        );
    }

    // Model flag is emitted by the launch site from `LaunchEnv`, never spliced
    // in here — so the assembled view is exactly `server_args ++ command_args`,
    // for either server.
    #[rstest]
    #[case::both(ServerKind::Vllm, &["--max-model-len", "4096"], &["extra-trailing"], &["--max-model-len", "4096", "extra-trailing"])]
    #[case::command_args_only(ServerKind::Sglang, &[], &["trail-a", "trail-b"], &["trail-a", "trail-b"])]
    #[case::server_args_only(ServerKind::Vllm, &["--dtype", "bfloat16"], &[], &["--dtype", "bfloat16"])]
    fn launch_spec_assembled_args_cases(
        #[case] kind: ServerKind,
        #[case] server_args: &[&str],
        #[case] command_args: &[&str],
        #[case] expected: &[&str],
    ) {
        assert_eq!(
            docker_spec(kind, server_args, command_args).assembled_args(),
            expected
        );
    }
}
