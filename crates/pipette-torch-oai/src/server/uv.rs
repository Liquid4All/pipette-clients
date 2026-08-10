//! uv engine: install, launch, stop, and cross-invocation orphan reaping.
//!
//! **Linux-only** — the docker engine continues to work wherever Docker
//! does. The non-Linux build swaps in `server/uv_stub.rs`, which bails at
//! the public entrypoints so the rest of the crate still compiles
//! cross-platform.
//!
//! ## Install
//!
//! `uv venv <venv>`, `uv pip install -r <req>`, then best-effort
//! `uv pip compile -o <lock>` to record the resolved dependency set.
//!
//! The requirements passed by the operator or bundled catalog are
//! copied into the per-runtime workspace dir so subsequent
//! `runtimes status` reads see the same bytes that were installed.
//!
//! ## Launch
//!
//! 1. Pick a free port via `TcpListener::bind("127.0.0.1:0")`.
//! 2. Spawn `<venv>/bin/{vllm|python}` with a `setsid(2)` pre-exec so the
//!    server tree (vLLM ray workers, sglang scheduler-tokenizer-detok
//!    workers) lives in its own process group — `kill -- -<pgid>` reaps
//!    the whole tree atomically.
//! 3. Inject env-markers so the cross-invocation reaper can recognise our
//!    server tree without an on-disk pid file (`PIPETTE_WORKSPACE_LABEL`,
//!    `PIPETTE_RUNTIME_REF`, `PIPETTE_PARENT_PID`, `HF_HOME`).
//! 4. Tail stdout/stderr to the logger from a background thread; the
//!    thread joins automatically when the child closes its pipes.
//!
//! ## Stop (Drop-guard / explicit)
//!
//! SIGTERM to `-<pgid>`, poll `waitpid` up to `grace_secs`, escalate to
//! SIGKILL on overrun, then join the log-tail thread.
//!
//! ## Orphan reaper (cross-invocation)
//!
//! Linux-only via `/proc`. Walk it, read each `environ`, keep PIDs where
//!
//! 1. `PIPETTE_WORKSPACE_LABEL` matches the current workspace, **and**
//! 2. `PIPETTE_PARENT_PID` no longer refers to a live `pipette-torch-oai`
//!    binary (i.e. the spawning CLI is dead).
//!
//! Dedupe survivors by PGID (one tree of workers shares one PGID), send
//! `SIGTERM` to `-<pgid>` for each. Sleep briefly, escalate to `SIGKILL`
//! on anything still alive. Single-digit-ms cost on a typical box.

#![cfg(target_os = "linux")]

use std::{
    collections::{HashMap, HashSet},
    ffi::OsStr,
    fs,
    io::{BufRead, BufReader, ErrorKind, Read},
    os::unix::process::CommandExt,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    str,
    sync::Arc,
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use anyhow::Context;

use pipette_subprocess::cleanup::{self, TeardownToken};
use pipette_venv::{venv_bin, venv_python as venv_python_in};

use super::{LaunchSpec, ServerKind, ServerState};

/// Env-marker keys injected on every spawned uv server child. Read back
/// during cross-invocation orphan reaping to identify our processes.
pub const ENV_WORKSPACE_LABEL: &str = "PIPETTE_WORKSPACE_LABEL";
pub const ENV_RUNTIME_REF: &str = "PIPETTE_RUNTIME_REF";
pub const ENV_PARENT_PID: &str = "PIPETTE_PARENT_PID";

/// In-venv `vllm` shim under a venv root. Private so this module's public
/// surface stays identical to the non-Linux stub in `uv_stub.rs`.
fn venv_vllm_in(venv: &Path) -> PathBuf {
    venv_bin(venv).join("vllm")
}

/// Per-server entrypoint argv (the binary + its pre-positional args, not
/// the trailing `--host`/`--port`/server_args). Splitting it out makes
/// the launch site engine-agnostic.
fn entrypoint(venv: &Path, spec: &LaunchSpec) -> anyhow::Result<Vec<PathBuf>> {
    let argv: Vec<PathBuf> = match spec {
        LaunchSpec::Uv {
            kind: ServerKind::Vllm,
            ..
        } => vec![venv_vllm_in(venv), PathBuf::from("serve")],
        LaunchSpec::Uv {
            kind: ServerKind::Sglang,
            ..
        } => vec![
            venv_python_in(venv),
            PathBuf::from("-m"),
            PathBuf::from("sglang.launch_server"),
        ],
        LaunchSpec::Docker { .. } => {
            anyhow::bail!("entrypoint() called with a docker LaunchSpec; uv-only");
        }
    };
    if !argv[0].exists() {
        anyhow::bail!(
            "{} not found; venv at {} is missing or broken. \
             Re-run `pipette runtimes pull` / `benchmarks run` with a bundled \
             `uv-vllm://` / `uv-sglang://` ref (see `pipette runtimes catalog uv_vllm`)",
            argv[0].display(),
            venv.display(),
        );
    }
    Ok(argv)
}

/// Re-export of [`super::pick_free_port`] so the docker and uv
/// launch sites allocate host ports via the exact same code path.
pub use super::pick_free_port;

/// Optional inputs for the launch site that aren't part of `LaunchSpec`
/// because they're either workspace-global (the label) or runtime-derived
/// (`hf_home`).
#[derive(Debug, Clone)]
pub struct UvLaunchEnv {
    /// Workspace label percent-encoded into a single line. Echoes to the
    /// child's `PIPETTE_WORKSPACE_LABEL` env var and matched against in
    /// the orphan reaper.
    pub workspace_label: String,
    /// HuggingFace cache dir; child sees this as `HF_HOME`. `None` skips
    /// the env var (uv inherits the parent's `HF_HOME`, if any).
    pub hf_home: Option<PathBuf>,
    /// Path to the model on disk (HF cache snapshot dir or host path).
    /// Passed verbatim to the server via its `--model` / `--model-path`
    /// flag.
    pub model: String,
    /// Install identity for `PIPETTE_RUNTIME_REF` (plan `declared` Display,
    /// e.g. `uv://vllm@…`).
    pub runtime_ref: String,
}

/// Launch a uv server in its own process group.
///
/// Returns the in-memory state + a teardown token registered with the
/// signal-handler registry. The token must be deregistered after a
/// clean stop (mirrors the docker side). `execute`'s `ServerGuard` is that
/// pattern: it owns the returned `TeardownToken` for the duration of the run
/// and its `Drop` does the stop, then deregisters — keeping the token
/// registered if the stop failed, so ^C can still retry.
pub fn launch(
    venv: &Path,
    spec: &LaunchSpec,
    env: &UvLaunchEnv,
) -> anyhow::Result<(ServerState, TeardownToken)> {
    let (spec_envs, server_args, command_args) = match spec {
        LaunchSpec::Uv {
            envs,
            server_args,
            command_args,
            ..
        } => (envs, server_args, command_args),
        LaunchSpec::Docker { .. } => {
            anyhow::bail!("uv::launch called with a docker LaunchSpec");
        }
    };

    let port = pick_free_port()?;
    let argv = entrypoint(venv, spec)?;

    let mut cmd = Command::new(&argv[0]);
    for arg in &argv[1..] {
        cmd.arg(arg);
    }
    // Prepend the runtime venv's `bin/` to PATH so the server's own
    // subprocesses find venv-installed build tools — notably `ninja`,
    // which PyTorch's `cpp_extension` shells out to via `shutil.which`
    // when JIT-compiling kernels for models like LFM2/Mamba. Without
    // this, `pip install ninja` lands in the venv but the EngineCore
    // process still gets a parent PATH that omits it and fails with
    // `FileNotFoundError: [Errno 2] No such file or directory: 'ninja'`.
    let venv_bin_dir = venv_bin(venv);
    let existing_path = std::env::var_os("PATH").unwrap_or_default();
    let parts = std::iter::once(venv_bin_dir).chain(std::env::split_paths(&existing_path));
    let new_path = std::env::join_paths(parts).context("failed to prepend venv/bin to PATH")?;
    cmd.env("PATH", new_path);
    // Model goes in the position the server's CLI wants it:
    //   vllm:   `vllm serve <model>` — positional, after `serve`.
    //   sglang: `python -m sglang.launch_server --model-path <model>` — flagged.
    // Matches the literal entrypoint shape each server documents.
    // vLLM also accepts `--model`, but the positional form is what every
    // currently-shipping `vllm serve` documents — keeping the literal
    // form is the conservative call.
    match spec.kind() {
        ServerKind::Sglang => {
            cmd.arg("--model-path").arg(&env.model);
        }
        ServerKind::Vllm => {
            cmd.arg(&env.model);
        }
    }
    // user-supplied runtime flags (already shell-split), then bind
    // 127.0.0.1:<port>.
    for f in server_args {
        cmd.arg(f);
    }
    cmd.arg("--host").arg("127.0.0.1");
    cmd.arg("--port").arg(port.to_string());
    // trailing `--` args
    for a in command_args {
        cmd.arg(a);
    }

    // env-markers and HF-related env so the orphan reaper can find this
    // tree and the in-venv HF client uses the workspace cache.
    cmd.env(ENV_WORKSPACE_LABEL, &env.workspace_label);
    cmd.env(ENV_RUNTIME_REF, &env.runtime_ref);
    cmd.env(ENV_PARENT_PID, std::process::id().to_string());
    if let Some(home) = &env.hf_home {
        fs::create_dir_all(home)
            .with_context(|| format!("failed to create hf-home dir {}", home.display()))?;
        cmd.env("HF_HOME", home);
    }
    // operator-supplied envs from --env K[=V]. Same inherit-form
    // semantics as docker: empty value means "read $K from this
    // process's env and forward it".
    for (k, v) in spec_envs {
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

    // setsid(2) puts the child (and every fork) in a new session, so its
    // PID == its PGID. `kill -- -<pgid>` then reaps the whole tree
    // atomically.
    //
    // SAFETY: setsid is async-signal-safe per POSIX and we don't touch any
    // not-async-signal-safe state in pre_exec.
    unsafe {
        cmd.pre_exec(|| {
            if libc::setsid() < 0 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }

    cmd.stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    pipette_subprocess::echo_info(&cmd);
    log::info!(
        "[server] starting uv server for {} on 127.0.0.1:{port}",
        env.runtime_ref
    );
    let mut child = cmd
        .spawn()
        .with_context(|| format!("failed to spawn {}", argv[0].display()))?;
    let pid = child.id();
    // After setsid in the child, getpgid(pid) == pid. We trust this rather
    // than calling getpgid because (a) the child is already past
    // pre_exec by the time spawn returns, (b) we can't safely call libc
    // from this thread until we hold a stable pid.
    let pgid = pid;

    // Register the teardown closure with the signal handler before any
    // post-spawn step that could fail — that way a SIGINT in the
    // log-tailing or port wait kills the server cleanly via the same
    // path the Drop guard uses.
    let teardown_token = cleanup::register(move || {
        // Best effort: kill the whole group via -PGID. Ignore errors
        // (the group may already be gone).
        unsafe {
            libc::kill(-(pgid as i32), libc::SIGTERM);
        }
        // Brief grace, then SIGKILL.
        thread::sleep(Duration::from_millis(500));
        unsafe {
            libc::kill(-(pgid as i32), libc::SIGKILL);
        }
    });

    let log_tail = spawn_log_tail(pid, &mut child).ok();

    let state = match spec {
        LaunchSpec::Uv { kind, .. } => ServerState::Uv {
            kind: *kind,
            executable: PathBuf::from(&argv[0]),
            pid,
            pgid,
            port,
            log_tail,
        },
        LaunchSpec::Docker { .. } => {
            anyhow::bail!("uv::launch called with a docker LaunchSpec");
        }
    };

    // We deliberately drop `child` here. The child is in its own session
    // now (setsid), so even though Rust would call waitpid on Drop, we
    // own the reap responsibility ourselves via the explicit
    // wait-on-stop in `stop()`. The kernel will keep the process around
    // until either we wait for it or our process exits (in which case
    // init reaps it).
    //
    // Forgetting `child` here would leak the file descriptor for its
    // stdout/stderr pipes — but we already took those into the log-tail
    // thread, so dropping `child` is safe and reclaims its small
    // bookkeeping struct.
    drop(child);

    Ok((state, teardown_token))
}

/// Stream the child's stdout and stderr line-by-line to the logger. The
/// returned JoinHandle covers both pipes; it joins itself once the child
/// closes them (which happens automatically when the child exits).
fn spawn_log_tail(pid: u32, child: &mut std::process::Child) -> anyhow::Result<JoinHandle<()>> {
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| anyhow::anyhow!("child stdout pipe missing"))?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| anyhow::anyhow!("child stderr pipe missing"))?;
    let prefix = Arc::new(format!("[uv-server pid={pid}]"));
    let h = thread::Builder::new()
        .name(format!("pipette-torch-oai-uv-log-{pid}"))
        .spawn(move || {
            let prefix_out = Arc::clone(&prefix);
            let t_out = thread::Builder::new()
                .name(format!("pipette-torch-oai-uv-stdout-{pid}"))
                .spawn(move || tail_pipe(stdout, &prefix_out, false))
                .ok();
            let prefix_err = Arc::clone(&prefix);
            let t_err = thread::Builder::new()
                .name(format!("pipette-torch-oai-uv-stderr-{pid}"))
                .spawn(move || tail_pipe(stderr, &prefix_err, true))
                .ok();
            if let Some(h) = t_out {
                let _ = h.join();
            }
            if let Some(h) = t_err {
                let _ = h.join();
            }
        })
        .context("failed to spawn uv log-tail thread")?;
    Ok(h)
}

fn tail_pipe<R: Read>(pipe: R, prefix: &str, is_stderr: bool) {
    let reader = BufReader::new(pipe);
    for line in reader.lines() {
        match line {
            Ok(s) => {
                if is_stderr {
                    log::warn!("{prefix} {s}");
                } else {
                    log::info!("{prefix} {s}");
                }
            }
            Err(err) => {
                log::debug!("{prefix} log-tail read error: {err}");
                break;
            }
        }
    }
}

/// `Some(status)` once the leader has exited, `None` while it is still
/// running. Used by the readiness wait to stop polling a port whose server
/// is already dead.
///
/// This reaps the leader. That is what makes the status readable, and it is
/// safe: [`stop`]'s later `waitpid` then reports `ECHILD`, which it already
/// treats as "already gone".
pub(super) fn exit_status(pid: u32) -> Option<String> {
    let mut raw: libc::c_int = 0;
    let r = unsafe { libc::waitpid(pid as i32, &mut raw, libc::WNOHANG) };
    if r == pid as i32 {
        return Some(wait_status_reason(raw));
    }
    if r < 0 && std::io::Error::last_os_error().raw_os_error() == Some(libc::ECHILD) {
        return Some("already reaped, exit status unavailable".to_string());
    }
    None
}

/// Decode a raw `waitpid` status into the phrase the readiness error carries.
fn wait_status_reason(raw: libc::c_int) -> String {
    if libc::WIFEXITED(raw) {
        format!("exit status {}", libc::WEXITSTATUS(raw))
    } else if libc::WIFSIGNALED(raw) {
        format!("killed by signal {}", libc::WTERMSIG(raw))
    } else {
        format!("unrecognised wait status {raw}")
    }
}

/// Stop a uv server tree: SIGTERM the PGID, poll waitpid up to
/// `grace_secs`, escalate to SIGKILL on overrun, then join the log
/// tail.
///
/// Idempotent — already-dead trees return Ok without raising.
pub fn stop(state: &mut ServerState, grace_secs: u32) -> anyhow::Result<()> {
    let (pid, pgid) = match state {
        ServerState::Uv { pid, pgid, .. } => (*pid as i32, *pgid as i32),
        ServerState::Docker { .. } => {
            anyhow::bail!("uv::stop called with a docker ServerState");
        }
    };

    // Phase 1: SIGTERM to the whole process group.
    log::info!("stopping uv server pgid={pgid} (SIGTERM)");
    let rc = unsafe { libc::kill(-pgid, libc::SIGTERM) };
    if rc != 0 {
        let err = std::io::Error::last_os_error();
        if err.raw_os_error() != Some(libc::ESRCH) {
            // Anything other than ESRCH (process group gone) is a real
            // problem worth surfacing.
            log::warn!("kill -SIGTERM -{pgid} failed: {err}");
        }
    }

    // Phase 2: poll waitpid for the leader. The other group members are
    // reaped by init once they exit (we don't own them as direct
    // children).
    let deadline = Instant::now() + Duration::from_secs(grace_secs as u64);
    let mut leader_gone = false;
    while Instant::now() < deadline {
        let mut status: libc::c_int = 0;
        let r = unsafe { libc::waitpid(pid, &mut status, libc::WNOHANG) };
        if r == pid {
            leader_gone = true;
            break;
        }
        if r < 0 {
            let err = std::io::Error::last_os_error();
            // ECHILD: not our child / already reaped. ESRCH should not
            // happen for waitpid but treat both as "already done".
            if matches!(err.raw_os_error(), Some(libc::ECHILD) | Some(libc::ESRCH)) {
                leader_gone = true;
                break;
            }
            log::warn!("waitpid({pid}) failed: {err}");
            break;
        }
        thread::sleep(Duration::from_millis(100));
    }

    // Phase 3: escalate to SIGKILL if the leader is still alive.
    if !leader_gone {
        log::warn!("uv server pid={pid} did not exit within {grace_secs}s; SIGKILL -{pgid}");
        let rc = unsafe { libc::kill(-pgid, libc::SIGKILL) };
        if rc != 0 {
            let err = std::io::Error::last_os_error();
            if err.raw_os_error() != Some(libc::ESRCH) {
                log::warn!("kill -SIGKILL -{pgid} failed: {err}");
            }
        }
        // Final non-blocking waitpid to clean up the zombie.
        let mut status: libc::c_int = 0;
        let _ = unsafe { libc::waitpid(pid, &mut status, libc::WNOHANG) };
    }

    // Phase 4: join the log tail. The child's stdout/stderr pipes close
    // when the child exits, so the tail thread sees EOF and returns.
    let log_tail = match state {
        ServerState::Uv { log_tail, .. } => log_tail.take(),
        ServerState::Docker { .. } => {
            anyhow::bail!("uv::stop called with a docker ServerState");
        }
    };
    if let Some(handle) = log_tail {
        let _ = handle.join();
    }

    Ok(())
}

/// Reap stale uv server trees from a previous (crashed) run of *this*
/// workspace.
///
/// Filter chain:
///
/// 1. Same `PIPETTE_WORKSPACE_LABEL` as us — disambiguates other
///    workspaces on the host.
/// 2. Spawning CLI (`PIPETTE_PARENT_PID`) is no longer alive as a
///    `pipette-torch-oai` binary — so concurrent runs in the same
///    workspace coexist.
/// 3. Dedupe by PGID — one server tree presents as many PIDs sharing
///    one group.
///
/// Then `kill -- -<pgid>` for each survivor (SIGTERM, brief sleep,
/// SIGKILL). Soft-fails: a broken `/proc` or unkillable process logs a
/// warning instead of aborting the caller's run.
pub fn reap_workspace_orphans(workspace_label: &str) -> anyhow::Result<()> {
    let mut survivors: HashSet<u32> = HashSet::new();
    let proc_dir = Path::new("/proc");
    if !proc_dir.exists() {
        // Not Linux (or `/proc` not mounted) — no orphans to reap. The uv
        // engine is Linux-only, so this is a graceful no-op for unsupported
        // hosts.
        return Ok(());
    }
    let entries = fs::read_dir(proc_dir).context("failed to read /proc")?;
    for entry in entries {
        let Ok(entry) = entry else { continue };
        let name = entry.file_name();
        let Some(name_str) = name.to_str() else {
            continue;
        };
        let Ok(pid) = name_str.parse::<u32>() else {
            continue;
        };
        // Skip our own process — we obviously have the same workspace
        // label but we're not an orphan.
        if pid == std::process::id() {
            continue;
        }

        let environ_path = proc_dir.join(name).join("environ");
        let environ = match fs::read(&environ_path) {
            Ok(b) => b,
            Err(err)
                if matches!(
                    err.kind(),
                    ErrorKind::PermissionDenied | ErrorKind::NotFound
                ) =>
            {
                continue;
            }
            Err(err) => {
                log::debug!("failed to read {}: {err}", environ_path.display());
                continue;
            }
        };
        let env = parse_environ(&environ);

        // (a) Same workspace.
        let label = env.get(ENV_WORKSPACE_LABEL.as_bytes());
        if label.map(|b| b.as_slice()) != Some(workspace_label.as_bytes()) {
            continue;
        }

        // (b) Spawning CLI is dead.
        let ppid_alive = env
            .get(ENV_PARENT_PID.as_bytes())
            .and_then(|b| str::from_utf8(b).ok())
            .and_then(|s| s.parse::<i32>().ok())
            .map(parent_still_running_as_pipette)
            .unwrap_or(false);
        if ppid_alive {
            continue;
        }

        // (c) Read PGID and add to survivors. Failures here are
        // logged-and-skipped because /proc is intentionally racy.
        match read_pgid(pid) {
            Ok(pgid) => {
                survivors.insert(pgid);
            }
            Err(err) => {
                log::debug!("failed to read pgid for {pid}: {err}");
            }
        }
    }

    if survivors.is_empty() {
        return Ok(());
    }
    log::warn!(
        "reaping {} stale uv server tree(s) from a prior crashed run",
        survivors.len()
    );
    // SIGTERM first, brief grace, then SIGKILL any survivors. Tracking
    // which PGIDs were already dead between phases avoids spurious
    // SIGKILLs; we just send to all and let the kernel ignore ESRCH.
    for pgid in &survivors {
        let signed = -(*pgid as i32);
        unsafe {
            libc::kill(signed, libc::SIGTERM);
        }
    }
    thread::sleep(Duration::from_millis(500));
    for pgid in &survivors {
        let signed = -(*pgid as i32);
        unsafe {
            libc::kill(signed, libc::SIGKILL);
        }
    }
    Ok(())
}

/// Parse a NUL-separated `K=V` environ blob into a key->value map.
/// Returns an empty map for empty / malformed input.
fn parse_environ(blob: &[u8]) -> HashMap<Vec<u8>, Vec<u8>> {
    let mut out = HashMap::new();
    for chunk in blob.split(|b| *b == 0) {
        if chunk.is_empty() {
            continue;
        }
        if let Some(eq) = chunk.iter().position(|b| *b == b'=') {
            let (k, v) = chunk.split_at(eq);
            out.insert(k.to_vec(), v[1..].to_vec());
        }
    }
    out
}

/// Read the process group id from `/proc/<pid>/stat`.
///
/// Field 5 (1-indexed) is `pgrp`. Field 2 is `comm`, surrounded by `()`
/// and may itself contain spaces — read up to the last `)`, then split
/// the remainder on whitespace.
fn read_pgid(pid: u32) -> anyhow::Result<u32> {
    let path = format!("/proc/{pid}/stat");
    let raw = fs::read_to_string(&path).with_context(|| format!("failed to read {path}"))?;
    let close = raw
        .rfind(')')
        .with_context(|| format!("malformed {path}: no ')' in {raw:?}"))?;
    let tail = &raw[close + 1..];
    // After `)`: fields are space-separated, starting with field 3 (state).
    // We want field 5 (pgrp) which is the 3rd field after `)`.
    let pgid = tail
        .split_whitespace()
        .nth(2)
        .with_context(|| format!("malformed {path}: too few fields after ')': {tail:?}"))?
        .parse::<u32>()
        .with_context(|| format!("malformed {path}: pgrp not a u32: {tail:?}"))?;
    Ok(pgid)
}

/// Check whether `ppid` still names a live `pipette` process.
///
/// Two-layer: `kill(pid, 0)` for liveness (returns 0 if alive, ESRCH
/// otherwise), then `/proc/<ppid>/exe` symlink confirmation to defend
/// against PID reuse (a reused PID for e.g. an editor wouldn't symlink
/// to our binary).
fn parent_still_running_as_pipette(ppid: i32) -> bool {
    if ppid <= 0 {
        return false;
    }
    if unsafe { libc::kill(ppid, 0) } != 0 {
        // ESRCH or EPERM — either way, not a live process we care about.
        return false;
    }
    let link = format!("/proc/{ppid}/exe");
    matches!(
        fs::read_link(&link),
        Ok(path) if path
            .file_name()
            .and_then(OsStr::to_str)
            == Some("pipette")
    )
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[test]
    fn parse_environ_extracts_key_value_pairs() {
        let blob = b"FOO=bar\0PATH=/usr/bin\0EMPTY=\0";
        let env = parse_environ(blob);
        assert_eq!(env.get(b"FOO".as_ref()), Some(&b"bar".to_vec()));
        assert_eq!(env.get(b"PATH".as_ref()), Some(&b"/usr/bin".to_vec()));
        assert_eq!(env.get(b"EMPTY".as_ref()), Some(&Vec::new()));
    }

    #[test]
    fn parse_environ_skips_chunks_without_equals() {
        // Some kernels leave a trailing NUL with garbage before the first =.
        let blob = b"\0noequals\0K=V\0";
        let env = parse_environ(blob);
        assert_eq!(env.get(b"K".as_ref()), Some(&b"V".to_vec()));
        assert_eq!(env.len(), 1);
    }

    #[test]
    fn parse_environ_empty_blob_is_empty_map() {
        assert!(parse_environ(b"").is_empty());
    }

    #[test]
    fn read_pgid_of_self_matches_libc() -> anyhow::Result<()> {
        // Our own PGID per libc must match what we parse from /proc/self/stat.
        // Guards against the field-counting bug where `comm` (field 2) was
        // assumed to be a single token but is actually `(name with spaces)`.
        let want = unsafe { libc::getpgrp() } as u32;
        let got = read_pgid(std::process::id())?;
        assert_eq!(got, want);
        Ok(())
    }

    #[test]
    fn parent_still_running_recognises_self() {
        // Our own PID is alive and exes to the `pipette` binary only when the
        // test runner is that binary. Under `cargo test` the exe is the test
        // harness (e.g. `pipette_torch_oai-<hash>`), which doesn't match, so
        // skip rather than false-trip.
        let me = std::process::id() as i32;
        let exe = fs::read_link(format!("/proc/{me}/exe")).ok();
        let is_pipette = exe
            .as_ref()
            .and_then(|p| p.file_name())
            .and_then(OsStr::to_str)
            == Some("pipette");
        if !is_pipette {
            // Test runner has a different binary name — the function would
            // correctly return false, which is what we'd assert. Skip with
            // a note so this test stays useful when run as part of the
            // `pipette` binary.
            eprintln!("skipping: test process exe is not pipette (got {exe:?})");
            return;
        }
        assert!(parent_still_running_as_pipette(me));
    }

    #[test]
    fn parent_still_running_rejects_dead_pid() {
        // A PID we know doesn't exist. PID 0 is special (kernel) so use a
        // very high one that's unlikely to clash on a normal box.
        let unlikely = 999_999;
        // It's possible (1 in many millions) this is live; if so, the
        // test is harmless.
        if unsafe { libc::kill(unlikely, 0) } == 0 {
            eprintln!("skipping: PID {unlikely} happens to be live");
            return;
        }
        assert!(!parent_still_running_as_pipette(unlikely));
    }

    #[test]
    fn parent_still_running_rejects_invalid_pid_values() {
        assert!(!parent_still_running_as_pipette(0));
        assert!(!parent_still_running_as_pipette(-1));
    }

    #[test]
    fn pick_free_port_returns_a_non_privileged_port() -> anyhow::Result<()> {
        assert!(pick_free_port()? >= 1024);
        Ok(())
    }

    // Raw wait statuses as the kernel encodes them: a normal exit is the code
    // in the high byte, a signal death is the signal number in the low bits.
    #[rstest]
    #[case::exit_zero(0, "exit status 0")]
    #[case::exit_one(1 << 8, "exit status 1")]
    #[case::oom_kill(libc::SIGKILL, "killed by signal 9")]
    #[case::segfault(libc::SIGSEGV, "killed by signal 11")]
    fn wait_status_reason_cases(#[case] raw: libc::c_int, #[case] expected: &str) {
        assert_eq!(wait_status_reason(raw), expected);
    }

    // A pid that was never our child reports ECHILD, which means gone — the
    // readiness wait must not read that as "still starting up".
    #[test]
    fn exit_status_reports_a_pid_we_never_spawned_as_gone() -> anyhow::Result<()> {
        let reason = exit_status(1).context("expected pid 1 to report as gone")?;
        assert!(reason.contains("already reaped"), "got {reason}");
        Ok(())
    }

    fn uv_spec(kind: ServerKind) -> LaunchSpec {
        LaunchSpec::Uv {
            kind,
            envs: vec![],
            server_args: vec![],
            command_args: vec![],
        }
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

    // SweepPins the dispatch table: vLLM runs the venv's `vllm serve` shim, sglang
    // runs `python -m sglang.launch_server`.
    #[rstest]
    #[case::vllm(ServerKind::Vllm, &["bin/vllm", "serve"])]
    #[case::sglang(
        ServerKind::Sglang,
        &["bin/python", "-m", "sglang.launch_server"]
    )]
    fn entrypoint_argv_cases(
        #[case] kind: ServerKind,
        #[case] expected_tail: &[&str],
    ) -> anyhow::Result<()> {
        let tmp = populated_venv()?;
        let argv = entrypoint(tmp.path(), &uv_spec(kind))?;

        let mut want: Vec<PathBuf> = vec![tmp.path().join(expected_tail[0])];
        want.extend(expected_tail[1..].iter().map(PathBuf::from));
        assert_eq!(argv, want);
        Ok(())
    }

    #[rstest]
    #[case::vllm(ServerKind::Vllm)]
    #[case::sglang(ServerKind::Sglang)]
    fn entrypoint_bails_when_the_venv_shim_is_missing(
        #[case] kind: ServerKind,
    ) -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let err = entrypoint(tmp.path(), &uv_spec(kind))
            .err()
            .context("expected an error")?;
        let msg = format!("{err:#}");
        assert!(msg.contains("not found"), "got {msg}");
        assert!(msg.contains("venv"), "got {msg}");
        Ok(())
    }

    #[test]
    fn entrypoint_rejects_docker_spec() -> anyhow::Result<()> {
        let spec = LaunchSpec::Docker {
            kind: ServerKind::Vllm,
            envs: vec![],
            gpus: None,
            shm_size: None,
            ipc: None,
            mounts: vec![],
            server_args: vec![],
            command_args: vec![],
        };
        let tmp = populated_venv()?;
        let err = entrypoint(tmp.path(), &spec)
            .err()
            .context("expected an error")?;
        let msg = format!("{err:#}");
        assert!(msg.contains("docker"), "got {msg}");
        assert!(msg.contains("uv-only"), "got {msg}");
        Ok(())
    }

    #[test]
    fn reap_workspace_orphans_is_noop_when_proc_missing() -> anyhow::Result<()> {
        // On a host without /proc this returns Ok without doing anything;
        // we can't easily fake the absence so this is really a smoke test
        // that the function is callable.
        let label = format!(
            "pipette-torch-oai-reap-test-{}",
            uuid::Uuid::new_v4().simple()
        );
        // With a label nothing matches, the reaper should return Ok without
        // killing anything we don't own.
        reap_workspace_orphans(&label)?;
        Ok(())
    }
}
