//! uv engine: the venv's `vllm serve` or `python -m sglang.launch_server` as a
//! [`pipette_subprocess::Process`], whose session takes vLLM's ray workers,
//! SGLang's scheduler and detokenizer and the build steps of a JIT compile with
//! it, at a stop and, when a killed run left it behind, at the next start.
//!
//! The server shares pipette's own cgroup, as any child does, so a memory
//! cell's probe reads that cgroup for the server's session and what it starts,
//! less pipette's own memory ([`crate::memprobe`]).
//!
//! Linux-only: the non-Linux build swaps in `server/uv_stub.rs`.

#![cfg(target_os = "linux")]

use std::{
    ffi::{OsStr, OsString},
    path::{Path, PathBuf},
    process::Command,
};

use anyhow::Context;

use pipette_plan_types::{RuntimeFlags, TorchOaiExtraRuntimeConfiguration};
use pipette_venv::{venv_bin, venv_python as venv_python_in};

use super::{Server, ServerKind, ServerState, Transport};

/// Private so this module's public surface stays identical to `uv_stub.rs`.
fn venv_vllm_in(venv: &Path) -> PathBuf {
    venv_bin(venv).join("vllm")
}

/// The server binary and its args before the model, `--host`, `--port` and
/// `server_args`.
fn entrypoint(venv: &Path, kind: ServerKind) -> anyhow::Result<(PathBuf, Vec<PathBuf>)> {
    let (program, args) = match kind {
        ServerKind::Vllm => (venv_vllm_in(venv), vec![PathBuf::from("serve")]),
        ServerKind::Sglang => (
            venv_python_in(venv),
            vec![PathBuf::from("-m"), PathBuf::from("sglang.launch_server")],
        ),
    };
    if !program.exists() {
        anyhow::bail!(
            "{} not found; venv at {} is missing or broken. \
             Re-run `pipette runtimes pull` / `benchmarks run` with a bundled \
             `uv-vllm://` / `uv-sglang://` ref (see `pipette runtimes catalog uv_vllm`)",
            program.display(),
            venv.display(),
        );
    }
    Ok((program, args))
}

pub fn launch(
    venv: &Path,
    kind: ServerKind,
    model: &Path,
    server_args: &[String],
    flags: &RuntimeFlags,
    client_config: Option<&TorchOaiExtraRuntimeConfiguration>,
) -> anyhow::Result<Server> {
    let port = pipette_subprocess::pick_server_port()?;
    let command = command(venv, kind, model, server_args, flags, client_config, port)?;
    let state = ServerState {
        kind,
        port,
        executable: PathBuf::from(command.get_program()),
    };
    tracing::info!(
        program = %state.executable.display(),
        port,
        "starting uv server on 127.0.0.1"
    );
    Ok(Server {
        state,
        transport: Transport::Uv(pipette_subprocess::start_process(
            kind.name(),
            command,
            super::STOP_GRACE,
        )?),
    })
}

/// The server's command line and environment, bound to `127.0.0.1:<port>`.
fn command(
    venv: &Path,
    kind: ServerKind,
    model: &Path,
    server_args: &[String],
    flags: &RuntimeFlags,
    client_config: Option<&TorchOaiExtraRuntimeConfiguration>,
    port: u16,
) -> anyhow::Result<Command> {
    let (program, args) = entrypoint(venv, kind)?;

    let mut cmd = Command::new(&program);
    cmd.args(&args);
    // vLLM also accepts `--model`, but every shipping `vllm serve` documents
    // the positional form, so that is what it gets.
    match kind {
        ServerKind::Sglang => {
            cmd.arg("--model-path").arg(model);
        }
        ServerKind::Vllm => {
            cmd.arg(model);
        }
    }
    cmd.args(server_args)
        .arg("--host")
        .arg("127.0.0.1")
        .arg("--port")
        .arg(port.to_string());

    // First, so the cell's own forwards win.
    pipette_subprocess::apply_client_config(&mut cmd, client_config);

    // Same inherit-form semantics as docker: an empty value forwards this
    // process's `$K`.
    for entry in flags.envs() {
        let (k, v) = (entry.name(), entry.value().unwrap_or_default());
        if v.is_empty() {
            let value = match std::env::var(k) {
                Ok(v) => v,
                Err(std::env::VarError::NotPresent) => anyhow::bail!(
                    "--env {k}: variable not set in this process's environment; \
                     export it before running, or pass `--env {k}=<value>` explicitly"
                ),
                Err(std::env::VarError::NotUnicode(_)) => anyhow::bail!(
                    "--env {k}: variable is set but contains non-UTF-8 bytes; \
                     pass `--env {k}=<value>` with a valid UTF-8 value"
                ),
            };
            cmd.env(k, value);
        } else {
            cmd.env(k, v);
        }
    }
    // The venv's `bin/` leads PATH so the server's children find venv tools:
    // PyTorch's `cpp_extension` JIT (LFM2/Mamba) shells out to `ninja` and
    // fails with `No such file or directory: 'ninja'` without it. Set after
    // the cell's forwards so a forwarded `PATH` cannot drop it.
    let forwarded_path = cmd
        .get_envs()
        .find_map(|(k, v)| (k == "PATH").then(|| v.map(OsStr::to_owned)))
        .flatten();
    cmd.env("PATH", server_path(venv_bin(venv), forwarded_path)?);
    Ok(cmd)
}

/// The server's `PATH`: the venv's `bin/` in front of whatever the cell
/// forwarded as `PATH`, or of this process's own when it forwarded none.
fn server_path(venv_bin_dir: PathBuf, forwarded: Option<OsString>) -> anyhow::Result<OsString> {
    let existing = forwarded
        .or_else(|| std::env::var_os("PATH"))
        .unwrap_or_default();
    let parts = std::iter::once(venv_bin_dir).chain(std::env::split_paths(&existing));
    std::env::join_paths(parts).context("failed to prepend venv/bin to PATH")
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::time::Duration;

    use rstest::rstest;

    use super::super::test_support::{ready_once, serve_fake, write_executable};
    use super::*;

    /// Otherwise a forwarded `PATH` replaces the prepend and the server cannot
    /// find `ninja`.
    #[test]
    fn a_forwarded_path_keeps_the_venv_in_front() -> anyhow::Result<()> {
        let venv_bin = PathBuf::from("/store/venv/bin");
        let path = server_path(
            venv_bin.clone(),
            Some(OsString::from("/opt/cuda/bin:/usr/bin")),
        )?;
        let parts: Vec<PathBuf> = std::env::split_paths(&path).collect();
        assert_eq!(
            parts,
            vec![
                venv_bin,
                PathBuf::from("/opt/cuda/bin"),
                PathBuf::from("/usr/bin")
            ]
        );
        Ok(())
    }

    /// A venv root with both shims present so `entrypoint` gets past its
    /// existence check and returns the argv it picked.
    fn populated_venv() -> anyhow::Result<tempfile::TempDir> {
        let tmp = tempfile::tempdir()?;
        let bin = tmp.path().join("bin");
        fs::create_dir_all(&bin)?;
        fs::write(bin.join("vllm"), "")?;
        fs::write(bin.join("python"), "")?;
        Ok(tmp)
    }

    #[rstest]
    #[case::vllm(ServerKind::Vllm, "bin/vllm", &["serve"])]
    #[case::sglang(ServerKind::Sglang, "bin/python", &["-m", "sglang.launch_server"])]
    fn entrypoint_argv_cases(
        #[case] kind: ServerKind,
        #[case] expected_program: &str,
        #[case] expected_args: &[&str],
    ) -> anyhow::Result<()> {
        let tmp = populated_venv()?;
        let (program, args) = entrypoint(tmp.path(), kind)?;

        assert_eq!(program, tmp.path().join(expected_program));
        let want: Vec<PathBuf> = expected_args.iter().map(PathBuf::from).collect();
        assert_eq!(args, want);
        Ok(())
    }

    #[rstest]
    #[case::vllm(ServerKind::Vllm)]
    #[case::sglang(ServerKind::Sglang)]
    fn entrypoint_bails_when_the_venv_shim_is_missing(
        #[case] kind: ServerKind,
    ) -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let err = entrypoint(tmp.path(), kind)
            .err()
            .context("expected an error")?;
        let msg = format!("{err:#}");
        assert!(msg.contains("not found"), "got {msg}");
        assert!(msg.contains("venv"), "got {msg}");
        Ok(())
    }

    /// Through a stand-in `vllm` in the venv: bound to the port it was given,
    /// ready once `/v1/models` answers, and stopped with SIGTERM.
    #[test]
    fn a_server_starts_from_the_venv_and_stops_on_sigterm() -> anyhow::Result<()> {
        let venv = tempfile::tempdir()?;
        let bin = venv_bin(venv.path());
        fs::create_dir_all(&bin)?;
        let vllm = write_executable(
            &bin.join("vllm"),
            r#"#!/bin/sh
DIR=$(dirname "$0")
trap 'echo stopping; exit 0' TERM
echo loading
echo "$@" > "$DIR/argv"
while :; do sleep 0.05; done
"#,
        )?;
        let port_flag = |words: &[&str]| {
            let flag = words.iter().position(|word| *word == "--port")?;
            words.get(flag + 1)?.parse().ok()
        };
        let http = serve_fake(&bin.join("argv"), port_flag, ready_once)?;
        let flags = RuntimeFlags::EndToEndLatencyUvVllmTorch {
            tensor_parallel_size: None,
            dtype: None,
            max_model_len: None,
            prefix_caching: None,
            envs: vec![],
            raw: vec![],
        };

        let mut server = launch(
            venv.path(),
            ServerKind::Vllm,
            Path::new("/data/models/abc/blobs"),
            &["--max-model-len".to_owned(), "4096".to_owned()],
            &flags,
            None,
        )?;
        server.wait_ready(Duration::from_secs(10))?;
        let argv = http
            .join()
            .map_err(|_| anyhow::anyhow!("the fake server panicked"))??;
        let port = server.state().port;
        assert_eq!(
            argv.trim(),
            format!(
                "serve /data/models/abc/blobs --max-model-len 4096 \
                 --host 127.0.0.1 --port {port}"
            )
        );
        assert_eq!(server.state().executable, vllm);

        server.stop();
        assert_eq!(
            server.output().runtime_stdout.as_deref(),
            Some("loading\nstopping\n")
        );
        Ok(())
    }

    /// A real uv install launches: a stand-in `vllm` runs the venv python's
    /// http.server over the model directory, where `v1/models` is a file.
    ///
    /// `cargo test -p pipette-torch-oai --lib uv_install_launch_stop_round_trip -- --ignored --nocapture`
    #[test]
    #[ignore = "needs uv + network; run with --ignored"]
    fn uv_install_launch_stop_round_trip() -> anyhow::Result<()> {
        use pipette_artifacts::runtime::RuntimeArtifactStore;
        use pipette_artifacts::{ArtifactsContext, ensure_runtime};
        use pipette_http::HttpClient;
        use pipette_plan_types::{
            NonEmptyString, Runtime, UvBuild, UvPythonVersion, UvRuntimeSource, UvServerVersion,
            UvVllm,
        };
        use pipette_programs::ExternalPrograms;

        let _ = tracing_subscriber::fmt()
            .with_env_filter(
                tracing_subscriber::EnvFilter::try_from_default_env()
                    .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
            )
            .with_test_writer()
            .try_init();
        let programs = ExternalPrograms::from_path();
        if !programs.uv.is_resolved() {
            eprintln!("skipping: uv not on PATH");
            return Ok(());
        }

        let work = tempfile::tempdir()?;
        let declared = Runtime::UvVllm(UvVllm {
            server_version: UvServerVersion::try_new("0.1.0".to_string())?,
            build: UvBuild::try_new("cpu".to_string())?,
            python_version: UvPythonVersion::try_new("3.12".to_string())?,
            source: UvRuntimeSource::PipRequirementsText {
                contents: NonEmptyString::try_new("# empty\n".to_owned())?,
                install_flags: None,
            },
        });
        let store = RuntimeArtifactStore::new(work.path().join("runtimes"));
        let ctx = ArtifactsContext::new(HttpClient::new("pipette")?, programs);
        ensure_runtime(&ctx, &store, &declared)?;
        let manifest = store
            .find(&declared)?
            .context("store has no manifest for the runtime it just installed")?;
        let venv = store.install_dir_for(&manifest)?;

        // `vllm serve <model> ... --port <port>` becomes the venv python
        // serving the model directory.
        let vllm = write_executable(
            &venv_bin(&venv).join("vllm"),
            r#"#!/bin/sh
model=$2
while [ $# -gt 0 ]; do
  [ "$1" = --port ] && port=$2
  shift
done
exec "$(dirname "$0")/python" -m http.server --bind 127.0.0.1 --directory "$model" "$port"
"#,
        )?;
        let model = work.path().join("model");
        fs::create_dir_all(model.join("v1"))?;
        fs::write(model.join("v1/models"), r#"{"data":[]}"#)?;

        let flags = RuntimeFlags::EndToEndLatencyUvVllmTorch {
            tensor_parallel_size: None,
            dtype: None,
            max_model_len: None,
            prefix_caching: None,
            envs: vec![],
            raw: vec![],
        };
        let mut server = launch(&venv, ServerKind::Vllm, &model, &[], &flags, None)?;
        server.wait_ready(Duration::from_secs(60))?;
        let port = server.state().port;
        assert_eq!(server.state().executable, vllm);

        server.stop();

        // After stop, the port must be free again (give the kernel a beat).
        let closed = (0..20).any(|_| {
            let free = std::net::TcpStream::connect(("127.0.0.1", port)).is_err();
            if !free {
                std::thread::sleep(Duration::from_millis(100));
            }
            free
        });
        assert!(closed, "port {port} still open after stop");
        // http.server logs each request to stderr, the readiness probe's included.
        let stderr = server.output().runtime_stderr.unwrap_or_default();
        assert!(stderr.contains("GET /v1/models"), "{stderr}");
        Ok(())
    }
}
