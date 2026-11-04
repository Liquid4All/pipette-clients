//! The server a cell runs against: vLLM or SGLang, in a container
//! ([`docker`]) or from a uv venv ([`uv`]), started on the
//! `pipette_subprocess` primitives, which keep its output, tear it down on ^C
//! and stop it when it is dropped.

use std::{
    path::{Path, PathBuf},
    time::Duration,
};

use anyhow::Context;

use pipette_plan_types::{Runtime, RuntimeFlags, RuntimeOutput, TorchOaiExtraRuntimeConfiguration};
use pipette_programs::ExternalPrograms;
use pipette_subprocess::{DockerProcess, NotReadyError, Process, Stopped};
use pipette_venv::require_bound_venv;

use crate::launch::{self, Drafter};
use crate::memprobe::ServerCgroup;
use crate::{cell_flags, cleanup};

pub mod docker;

// Only one of these compiles, so `cargo check` on a Mac never sees a break in
// `uv.rs`: check changes to either against a Linux target.
#[cfg(target_os = "linux")]
pub mod uv;

#[cfg(not(target_os = "linux"))]
#[path = "uv_stub.rs"]
pub mod uv;

pub const DEFAULT_SHM_SIZE: &str = "16g";
pub const DEFAULT_GPUS: &str = "all";

/// How long a start waits for the server to answer: the plan runner has no
/// readiness-timeout setting. Long enough for a cold start from a slow shared
/// store: vLLM 0.31 on a Nebius B300 node, its venv on a virtiofs mount, took
/// 800 s, about thirteen minutes, to answer (SLURM job 168989, 2026-10-11). A
/// server that exits ends the wait at once, so only a start that hangs waits
/// this long.
pub const READY_TIMEOUT: Duration = Duration::from_secs(1800);

/// Answers with a success once either server can take requests.
const READY_PATH: &str = "/v1/models";
/// Cap on one readiness request: the wait checks its own timeout only
/// between requests.
const READY_REQUEST_TIMEOUT: Duration = Duration::from_secs(5);
/// How long a stop waits for the server to exit before killing it.
const STOP_GRACE: Duration = Duration::from_secs(30);

/// Starts the server, after removing the containers a crashed run of this
/// workspace left behind; [`Server::wait_ready`] waits until it serves.
pub fn launch(
    runtime: &Runtime,
    server_args: &[String],
    flags: &RuntimeFlags,
    model_dir: &Path,
    drafter: Option<&Drafter>,
    client_config: Option<&TorchOaiExtraRuntimeConfiguration>,
    programs: &ExternalPrograms,
) -> anyhow::Result<Server> {
    let workspace_label = cleanup::workspace_label_from_store_path(model_dir)?;

    // Taken once, so the reap and `docker run` see the same binary. Only a
    // docker run requires it: the reap is workspace-scoped, so a uv run still
    // clears containers a crashed docker run left, when it can.
    let docker_bin = match programs.docker.path() {
        Ok(docker) => Some(docker.to_path_buf()),
        Err(err) if cell_flags::is_docker(runtime) => return Err(err),
        Err(err) => {
            tracing::debug!(
                error = %format_args!("{err:#}"),
                "skipping docker orphan reap"
            );
            None
        }
    };
    // The uv server spawns from the venv's own shims; this only asserts the
    // host can still manage that venv, before the server comes up rather than
    // mid-benchmark.
    if cell_flags::is_uv(runtime) {
        programs.uv.path()?;
    }
    if let Some(docker) = &docker_bin {
        // Before this launch, so a container this process owns is a leftover.
        cleanup::reap_workspace_orphans(docker, &workspace_label)?;
    }

    match runtime {
        Runtime::DockerVllm(_) | Runtime::DockerSglang(_) => docker::launch(
            docker_bin
                .as_deref()
                .context("docker runtime reached the launch without a resolved docker binary")?,
            runtime,
            flags,
            server_args,
            &launch::mounts(model_dir, drafter),
            &workspace_label,
            client_config,
        ),
        Runtime::UvVllm(rt) => uv::launch(
            &require_bound_venv(&rt.source)?,
            ServerKind::Vllm,
            model_dir,
            server_args,
            flags,
            client_config,
        ),
        Runtime::UvSglang(rt) => uv::launch(
            &require_bound_venv(&rt.source)?,
            ServerKind::Sglang,
            model_dir,
            server_args,
            flags,
            client_config,
        ),
        other @ (Runtime::LlamacppCliStockTools(_)
        | Runtime::LlamacppApkPipette(_)
        | Runtime::LlamacppIosPipette(_)
        | Runtime::MlxMacosPipette(_)
        | Runtime::MlxIosPipette(_)
        | Runtime::OnnxGenaiIosPipette(_)
        | Runtime::UvOpenvino(_)
        | Runtime::UvOnnxGenai(_)
        | Runtime::Ovms(_)
        | Runtime::TrtEdgellm(_)
        | Runtime::AppleCoreAiMacosPipette(_)
        | Runtime::AppleFoundation(_)) => Err(cell_flags::unsupported_runtime(other)),
    }
    .context("failed to start server for benchmark")
}

/// vLLM or SGLang: ports, model flags and entrypoints key off this, whether
/// docker or uv hosts it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServerKind {
    Vllm,
    Sglang,
}

impl ServerKind {
    pub fn of(runtime: &Runtime) -> anyhow::Result<Self> {
        match runtime {
            Runtime::DockerVllm(_) | Runtime::UvVllm(_) => Ok(Self::Vllm),
            Runtime::DockerSglang(_) | Runtime::UvSglang(_) => Ok(Self::Sglang),
            other @ (Runtime::LlamacppCliStockTools(_)
            | Runtime::LlamacppApkPipette(_)
            | Runtime::LlamacppIosPipette(_)
            | Runtime::MlxMacosPipette(_)
            | Runtime::MlxIosPipette(_)
            | Runtime::OnnxGenaiIosPipette(_)
            | Runtime::UvOpenvino(_)
            | Runtime::UvOnnxGenai(_)
            | Runtime::Ovms(_)
            | Runtime::TrtEdgellm(_)
            | Runtime::AppleCoreAiMacosPipette(_)
            | Runtime::AppleFoundation(_)) => anyhow::bail!(
                "runtime `{}` is not a torch-oai (docker/uv vLLM/SGLang) runtime",
                other.headless_token()
            ),
        }
    }

    pub fn default_port(self) -> u16 {
        match self {
            Self::Vllm => 8000,
            Self::Sglang => 30000,
        }
    }

    /// vLLM's image enters at `vllm serve` and binds every interface. SGLang's
    /// enters at a wrapper that execs its arguments, or at none, and the server
    /// binds 127.0.0.1, out of reach of the published port.
    pub fn container_command(self) -> &'static [&'static str] {
        match self {
            Self::Vllm => &[],
            Self::Sglang => &["python3", "-m", "sglang.launch_server", "--host", "0.0.0.0"],
        }
    }

    pub fn model_arg_flag(self) -> &'static str {
        match self {
            Self::Vllm => "--model",
            Self::Sglang => "--model-path",
        }
    }

    /// What logs and errors call a uv server: `sglang`, not the `python`
    /// running it.
    #[cfg(target_os = "linux")]
    pub fn name(self) -> &'static str {
        match self {
            Self::Vllm => "vllm",
            Self::Sglang => "sglang",
        }
    }
}

/// What a cell reads off a running server. Always loopback: pipette-torch-oai
/// drives the server for measurement, not for outside consumers.
#[derive(Debug, Clone)]
pub struct ServerState {
    pub kind: ServerKind,
    pub port: u16,
    /// What launched it, `docker` or the venv's server entrypoint, so a stored
    /// result names what produced it.
    pub executable: PathBuf,
}

impl ServerState {
    pub fn base_url(&self) -> String {
        format!("http://127.0.0.1:{}", self.port)
    }
}

/// A running server. Dropping it stops it.
pub struct Server {
    state: ServerState,
    transport: Transport,
}

enum Transport {
    Docker(DockerProcess),
    #[cfg_attr(
        not(target_os = "linux"),
        expect(dead_code, reason = "only the Linux uv module starts one")
    )]
    Uv(Process),
}

impl Server {
    pub fn state(&self) -> &ServerState {
        &self.state
    }

    /// Waits up to `timeout` until it serves; a server that exits first, or
    /// is still not serving then, is stopped.
    pub fn wait_ready(&mut self, timeout: Duration) -> anyhow::Result<()> {
        let ready = serving(&self.state)?;
        match &mut self.transport {
            Transport::Docker(process) => process.wait_ready(timeout, ready),
            Transport::Uv(process) => process.wait_ready(timeout, ready),
        }
        .context("failed to start server for benchmark")
    }

    /// Where its cgroup is, for the memory probe: the container's, or for a
    /// uv server pipette's own, which it shares.
    pub(crate) fn cgroup(&self) -> ServerCgroup {
        match &self.transport {
            Transport::Docker(process) => ServerCgroup::Container {
                docker: self.state.executable.clone(),
                name: process.container().to_owned(),
            },
            Transport::Uv(process) => ServerCgroup::Shared { pid: process.pid() },
        }
    }

    /// The `docker run` or server command line, without the environment.
    pub fn command(&self) -> &[String] {
        match &self.transport {
            Transport::Docker(process) => process.command(),
            Transport::Uv(process) => process.command(),
        }
    }

    /// What it has printed, the newest 256 KiB of each stream; final once
    /// stopped.
    pub fn output(&self) -> RuntimeOutput {
        match &self.transport {
            Transport::Docker(process) => process.output(),
            Transport::Uv(process) => process.output(),
        }
    }

    /// `docker stop`, or SIGTERM to its process group, then a kill of its
    /// session once [`STOP_GRACE`] has passed.
    pub fn stop(&mut self) {
        match &mut self.transport {
            Transport::Docker(process) => process.stop(),
            Transport::Uv(process) => process.stop(),
        }
    }

    /// [`Self::stop`] it over a failure, and say whether it had exited and the
    /// end of what it printed, which otherwise logs only at debug: context for
    /// that failure.
    pub fn stop_with_tail(&mut self) -> Stopped {
        match &mut self.transport {
            Transport::Docker(process) => process.stop_with_tail(),
            Transport::Uv(process) => process.stop_with_tail(),
        }
    }
}

/// The readiness check: `GET /v1/models` answers 2xx. Until then it is not
/// ready yet, and says why: a refused connection is a server still binding, an
/// error status one still loading.
fn serving(
    state: &ServerState,
) -> anyhow::Result<impl Fn(&RuntimeOutput) -> Result<(), NotReadyError>> {
    let url = format!("{}{READY_PATH}", state.base_url());
    let client = pipette_http::HttpClient::loopback("pipette", READY_REQUEST_TIMEOUT)
        .context("failed to build readiness HTTP client")?
        .client()
        .clone();
    Ok(move |_: &RuntimeOutput| {
        pipette_http::send(client.get(&url))
            .with_context(|| format!("GET {READY_PATH}"))
            .map_err(NotReadyError::yet)?;
        Ok(())
    })
}

/// Stand-ins for a server: the `docker` that starts it, and its network side.
#[cfg(test)]
pub(crate) mod test_support {
    use std::io::{BufRead, BufReader, Write};
    use std::net::TcpStream;
    use std::path::Path;
    #[cfg(unix)]
    use std::{net::TcpListener, path::PathBuf, thread::JoinHandle};

    #[cfg(unix)]
    use anyhow::Context;

    use pipette_memprobe::cgroup;

    /// Read one request and answer it: `status` on `GET /v1/models`, 404 on
    /// any other route.
    pub(crate) fn respond(mut stream: TcpStream, status: u16) -> std::io::Result<()> {
        let mut reader = BufReader::new(stream.try_clone()?);
        let mut request_line = String::new();
        reader.read_line(&mut request_line)?;
        // The rest of the head, so the close does not reset the connection
        // under a request the client is still sending.
        loop {
            let mut line = String::new();
            if reader.read_line(&mut line)? == 0 || line == "\r\n" {
                break;
            }
        }
        let status = if request_line.starts_with("GET /v1/models ") {
            status
        } else {
            404
        };
        write!(
            stream,
            "HTTP/1.1 {status} X\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
        )
    }

    /// The port a `docker run` publishes: `-p 127.0.0.1:<port>:<container>`.
    #[cfg(unix)]
    pub(crate) fn published_port(argv: &[&str]) -> Option<u16> {
        let flag = argv.iter().position(|word| *word == "-p")?;
        argv.get(flag + 1)?.split(':').nth(1)?.parse().ok()
    }

    /// The network side of a fake server binary, which serves nothing itself.
    /// `argv` is made a fifo the fake writes its arguments to; a thread reads
    /// them, binds the port `port_of` finds there and hands the listener to
    /// `serve`. The thread returns the arguments.
    #[cfg(unix)]
    pub(crate) fn serve_fake(
        argv: &Path,
        port_of: fn(&[&str]) -> Option<u16>,
        serve: impl FnOnce(TcpListener) -> anyhow::Result<()> + Send + 'static,
    ) -> anyhow::Result<JoinHandle<anyhow::Result<String>>> {
        #[expect(clippy::disallowed_methods, reason = "`mkfifo` prints nothing")]
        let made = std::process::Command::new("mkfifo").arg(argv).status()?;
        anyhow::ensure!(made.success(), "mkfifo {} failed", argv.display());
        let argv = argv.to_path_buf();
        Ok(std::thread::spawn(move || {
            let argv = std::fs::read_to_string(&argv)?;
            let words: Vec<&str> = argv.split_whitespace().collect();
            let port = port_of(&words).context("no port in the fake's arguments")?;
            serve(TcpListener::bind(("127.0.0.1", port))?)?;
            Ok(argv)
        }))
    }

    #[cfg(unix)]
    pub(crate) fn ready_once(listener: TcpListener) -> anyhow::Result<()> {
        Ok(respond(listener.accept()?.0, 200)?)
    }

    /// Written by a child: a write fd held here would pass to any child another
    /// test thread forks meanwhile, and exec'ing the script then fails with
    /// ETXTBSY.
    #[cfg(unix)]
    pub(crate) fn write_executable(path: &Path, script: &str) -> anyhow::Result<PathBuf> {
        #[expect(
            clippy::disallowed_methods,
            reason = "`cat` writes into the file, and nothing to stdout"
        )]
        let mut writer = std::process::Command::new("sh")
            .args(["-c", r#"cat > "$0" && chmod 755 "$0""#])
            .arg(path)
            .stdin(std::process::Stdio::piped())
            .spawn()?;
        writer
            .stdin
            .take()
            .context("the writer has no stdin")?
            .write_all(script.as_bytes())?;
        anyhow::ensure!(
            writer.wait()?.success(),
            "writing {} failed",
            path.display()
        );
        Ok(path.to_path_buf())
    }

    /// The pid [`fake_docker`]'s `container inspect` prints for the
    /// container's main process, which [`fake_host`] places in it.
    pub(crate) const FAKE_PID: u32 = 4242;

    /// The id [`fake_docker`]'s `container inspect` prints.
    pub(crate) const FAKE_CONTAINER_ID: &str =
        "3f7c9a1b2d4e6f8a3f7c9a1b2d4e6f8a3f7c9a1b2d4e6f8a3f7c9a1b2d4e6f8a";

    /// The container's `memory.stat` in [`fake_host`]'s tree.
    pub(crate) const FAKE_MEMORY_STAT: &str =
        "anon 1000\nfile 2000\nkernel 300\nkernel_stack 20\nshmem 40\n";

    /// `anon + shmem + kernel` of [`FAKE_MEMORY_STAT`].
    #[cfg(unix)]
    pub(crate) const FAKE_OWN_BYTES: u64 = 1_340;

    /// The container's `memory.peak` in [`fake_host`]'s tree.
    pub(crate) const FAKE_MEMORY_PEAK: u64 = 123_456_789;

    /// The container's `memory.swap.peak` in [`fake_host`]'s tree.
    pub(crate) const FAKE_SWAP_BYTES: u64 = 4_096;

    /// The container's cgroup in [`fake_host`]'s tree in `dir`: its systemd
    /// scope.
    #[cfg(unix)]
    pub(crate) fn fake_scope(dir: &Path) -> PathBuf {
        dir.join("cgroup")
            .join("system.slice")
            .join(format!("docker-{FAKE_CONTAINER_ID}.scope"))
    }

    /// A stand-in host in `dir`: `dir/proc` in `/proc`'s place, and a cgroup
    /// v2 tree at `dir/cgroup` that counts memory and holds no cgroup yet.
    pub(crate) fn fake_tree(dir: &Path) -> anyhow::Result<cgroup::Tree> {
        let root = dir.join("cgroup");
        std::fs::create_dir_all(&root)?;
        std::fs::write(root.join("cgroup.controllers"), "cpu io memory pids\n")?;
        Ok(cgroup::Tree::at(dir.join("proc"), root)?)
    }

    /// Writes `memory.stat`, `memory.peak` and `memory.swap.peak` holding
    /// the values above in `cgroup`, a directory of [`fake_tree`]'s.
    pub(crate) fn fill_cgroup(cgroup: &Path) -> anyhow::Result<()> {
        std::fs::create_dir_all(cgroup)?;
        std::fs::write(cgroup.join("memory.stat"), FAKE_MEMORY_STAT)?;
        std::fs::write(cgroup.join("memory.peak"), format!("{FAKE_MEMORY_PEAK}\n"))?;
        std::fs::write(
            cgroup.join("memory.swap.peak"),
            format!("{FAKE_SWAP_BYTES}\n"),
        )?;
        Ok(())
    }

    /// A stand-in docker host in `dir` for the container [`fake_docker`]
    /// names: [`FAKE_PID`] in its scope in [`fake_tree`]'s tree, which
    /// [`fill_cgroup`] fills.
    #[cfg(unix)]
    pub(crate) fn fake_host(dir: &Path) -> anyhow::Result<cgroup::Tree> {
        let pid = dir.join("proc").join(FAKE_PID.to_string());
        std::fs::create_dir_all(&pid)?;
        std::fs::write(
            pid.join("cgroup"),
            format!("0::/system.slice/docker-{FAKE_CONTAINER_ID}.scope\n"),
        )?;
        fill_cgroup(&fake_scope(dir))?;
        fake_tree(dir)
    }

    /// A stand-in `docker` in `dir`. `run` runs `greeting`, writes its argv to
    /// the `argv` fifo and serves until `stop` SIGTERMs it. Once `die` appears
    /// it crashes instead, as a container `--rm` then removes: a traceback, a
    /// write to the `crashed` fifo, and a `stop` that finds no container.
    /// `stop`, `kill` and `container inspect` are logged to `calls`. Once
    /// `no-container` appears, `container inspect` finds none, as before `run`
    /// creates it; once `not-started` does, it shows no process, as before
    /// `run` starts it; once `hang` does, it takes 5 s, as on a stalled
    /// daemon.
    #[cfg(unix)]
    pub(crate) fn fake_docker(dir: &Path, greeting: &str) -> anyhow::Result<PathBuf> {
        let script = format!(
            r#"#!/bin/sh
DIR='{}'
case "$1" in
  run)
    echo $$ > "$DIR/cli"
    trap 'echo stopped; exit 143' TERM
    {greeting}
    echo "$@" > "$DIR/argv"
    while [ ! -e "$DIR/die" ]; do sleep 0.05; done
    echo 'RuntimeError: CUDA error: out of memory' >&2
    touch "$DIR/gone"
    echo > "$DIR/crashed"
    exit 1 ;;
  container)
    if [ -e "$DIR/hang" ]; then exec sleep 5; fi
    echo "$@" >> "$DIR/calls"
    if [ -e "$DIR/no-container" ]; then
      echo "Error response from daemon: No such container: $5" >&2
      exit 1
    fi
    if [ -e "$DIR/not-started" ]; then echo "{FAKE_CONTAINER_ID} 0"; exit 0; fi
    echo "{FAKE_CONTAINER_ID} {FAKE_PID}" ;;
  stop|kill)
    echo "$@" >> "$DIR/calls"
    if [ -e "$DIR/gone" ]; then
      echo 'Error response from daemon: No such container' >&2
      exit 1
    fi
    kill -TERM "$(cat "$DIR/cli")" ;;
esac
"#,
            dir.display()
        );
        write_executable(&dir.join("docker"), &script)
    }

    /// pipette's own cgroup in [`fake_session`]'s tree: a login session's.
    pub(crate) const FAKE_SESSION: &str = "/user.slice/user-1000.slice/session-7.scope";

    /// The session's `memory.stat` in [`fake_session`]'s tree.
    pub(crate) const FAKE_SESSION_MEMORY_STAT: &str =
        "anon 3000000\nfile 2000\nkernel 300000\nkernel_stack 20\nshmem 40000\n";

    /// What [`fake_session`] gives pipette of its own: `RssAnon` 1000 kB.
    #[cfg(target_os = "linux")]
    pub(crate) const FAKE_PIPETTE_BYTES: u64 = 1_024_000;

    /// The session's own memory less pipette's: what a uv server's probe in
    /// [`fake_session`] reads.
    #[cfg(target_os = "linux")]
    pub(crate) const FAKE_SERVER_BYTES: u64 = 3_340_000 - FAKE_PIPETTE_BYTES;

    /// A process in [`FAKE_SESSION`] that is not the server's, which NVML
    /// lists beside it.
    pub(crate) const FAKE_OTHER_PID: u32 = 41;

    /// pipette's host in `dir`: it runs in [`FAKE_SESSION`] with 1000 kB of
    /// its own and no swap, beside [`FAKE_OTHER_PID`], and the session swaps
    /// [`FAKE_SWAP_BYTES`]; without `counted` the session has no
    /// `memory.current`, as the host root has none.
    pub(crate) fn fake_session(dir: &Path, counted: bool) -> anyhow::Result<cgroup::Tree> {
        let tree = fake_tree(dir)?;
        let session = dir
            .join("cgroup")
            .join(FAKE_SESSION.trim_start_matches('/'));
        fill_cgroup(&session)?;
        std::fs::write(session.join("memory.stat"), FAKE_SESSION_MEMORY_STAT)?;
        std::fs::write(
            session.join("memory.swap.current"),
            format!("{FAKE_SWAP_BYTES}\n"),
        )?;
        if counted {
            std::fs::write(session.join("memory.current"), "3340000\n")?;
        }
        let own = std::process::id();
        std::fs::write(
            session.join("cgroup.procs"),
            format!("{own}\n{FAKE_OTHER_PID}\n"),
        )?;
        let proc = dir.join("proc");
        [
            ("self".to_owned(), 1000),
            (own.to_string(), 1000),
            (FAKE_OTHER_PID.to_string(), 0),
        ]
        .into_iter()
        .try_for_each(|(pid, rss_anon_kib)| -> anyhow::Result<()> {
            let pid_dir = proc.join(&pid);
            std::fs::create_dir_all(&pid_dir)?;
            std::fs::write(pid_dir.join("cgroup"), format!("0::{FAKE_SESSION}\n"))?;
            std::fs::write(
                pid_dir.join("status"),
                format!("RssAnon:\t{rss_anon_kib} kB\nRssShmem:\t0 kB\n"),
            )?;
            Ok(())
        })?;
        std::fs::write(
            proc.join(FAKE_OTHER_PID.to_string()).join("stat"),
            format!("{FAKE_OTHER_PID} (sh) S 1 {FAKE_OTHER_PID} {FAKE_OTHER_PID} 0\n"),
        )?;
        Ok(tree)
    }

    /// Crash the `fake_docker` in `dir`, returning once it has.
    #[expect(clippy::disallowed_methods, reason = "`mkfifo` prints nothing")]
    pub(crate) fn crash_fake_docker(dir: &std::path::Path) -> std::io::Result<()> {
        let crashed = dir.join("crashed");
        if !std::process::Command::new("mkfifo")
            .arg(&crashed)
            .status()?
            .success()
        {
            return Err(std::io::Error::other(format!(
                "mkfifo {} failed",
                crashed.display()
            )));
        }
        std::fs::write(dir.join("die"), "")?;
        std::fs::read(&crashed).map(drop)
    }
}

#[cfg(test)]
mod tests {
    use std::net::TcpListener;

    use rstest::rstest;

    use super::test_support::respond;
    use super::*;

    // The launch sites pass this straight to the server argv, so the exact
    // spelling is a wire contract with vLLM / SGLang.
    #[rstest]
    #[case::vllm(ServerKind::Vllm, "--model")]
    #[case::sglang(ServerKind::Sglang, "--model-path")]
    fn model_arg_flag_cases(#[case] kind: ServerKind, #[case] expected: &str) {
        assert_eq!(kind.model_arg_flag(), expected);
    }

    #[test]
    fn server_state_base_url_is_loopback() {
        let state = ServerState {
            kind: ServerKind::Vllm,
            port: 41873,
            executable: PathBuf::from("docker"),
        };
        assert_eq!(state.base_url(), "http://127.0.0.1:41873");
    }

    /// Only a success is ready: a 503 is a server still loading its weights,
    /// and a refused connection one that has not bound its port yet.
    #[test]
    fn only_a_success_from_the_models_route_is_ready() -> anyhow::Result<()> {
        let listener = TcpListener::bind("127.0.0.1:0")?;
        let state = ServerState {
            kind: ServerKind::Vllm,
            port: listener.local_addr()?.port(),
            executable: PathBuf::from("docker"),
        };
        // One connection per status, then the port closes.
        let server = std::thread::spawn(move || {
            [503, 200]
                .into_iter()
                .try_for_each(|status| respond(listener.accept()?.0, status))
        });

        let ready = serving(&state)?;
        let printed = RuntimeOutput::default();
        let [loading, up] = [ready(&printed), ready(&printed)];
        server
            .join()
            .map_err(|_| anyhow::anyhow!("the fake server panicked"))??;
        let Some(NotReadyError::Yet(loading)) = loading.err() else {
            anyhow::bail!("a 503 is a server still loading, not a failed one");
        };
        let loading = anyhow::anyhow!(loading);
        assert!(format!("{loading:#}").contains("503"), "{loading:#}");
        up?;
        assert!(
            matches!(ready(&printed), Err(NotReadyError::Yet(_))),
            "a closed port answered ready"
        );
        Ok(())
    }
}
