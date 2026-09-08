//! Spawn `lloom-serve`, wait for its ready marker, hand back a base URL.
//!
//! The binary prints `{"kind":"ready","host":"127.0.0.1","port":N}` on stdout
//! once the model is loaded and the pipelines are warm; everything after that
//! line is captured for the result's stdout, stderr throughout. Drop kills the
//! process. The shape is the MLX sidecar driver's, minus the script.

use std::{
    io::{BufRead, BufReader},
    net::TcpListener,
    path::Path,
    process::{Child, ChildStderr, ChildStdout, Command, Stdio},
    sync::{mpsc, Arc, Mutex},
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use anyhow::Context;
use serde::Deserialize;

use pipette_ops::prompt_seed::PROMPT_SEED_TEXT;

/// The seed passage `lloom-serve` tiles into its throughput prompts, so its
/// cells and the MLX sidecar's read the same text.
const PROMPT_SEED_TEXT_ENV: &str = "PIPETTE_PROMPT_SEED_TEXT";
/// Loading a 2.6B f16 checkpoint and warming ~30 pipelines takes seconds; a
/// cold page-in of the weights can take a minute. The budget is the MLX one.
const READY_TIMEOUT: Duration = Duration::from_secs(3600);
const READY_POLL_INTERVAL: Duration = Duration::from_millis(100);
const EXIT_POLL_INTERVAL: Duration = Duration::from_millis(100);
const OUTPUT_CAPTURE_BYTES: usize = 256 * 1024;

pub struct ServerHandle {
    child: Child,
    exited: bool,
    stdout_thread: Option<JoinHandle<()>>,
    stderr_thread: Option<JoinHandle<()>>,
    stdout_buf: Arc<Mutex<String>>,
    stderr_buf: Arc<Mutex<String>>,
    _cleanup_guard: Option<pipette_subprocess::cleanup::Guard>,
    pub base_url: String,
    pub executable: String,
    pub command_preview: Vec<String>,
}

impl Drop for ServerHandle {
    fn drop(&mut self) {
        self.shutdown();
    }
}

impl ServerHandle {
    pub fn pid(&self) -> u32 {
        self.child.id()
    }

    pub fn stdout(&self) -> String {
        self.stdout_buf
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    pub fn stderr(&self) -> String {
        self.stderr_buf
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// Wait for an exit the server was asked for (`/shutdown`).
    pub fn wait_for_exit(&mut self, timeout: Duration) -> anyhow::Result<()> {
        let deadline = Instant::now() + timeout;
        loop {
            if let Some(status) = self
                .child
                .try_wait()
                .context("failed to inspect lloom-serve status during shutdown")?
            {
                self.exited = true;
                self.join_output_threads();
                if !status.success() {
                    anyhow::bail!("lloom-serve exited with {status}");
                }
                return Ok(());
            }
            let now = Instant::now();
            if now >= deadline {
                anyhow::bail!("timed out waiting for lloom-serve exit after {timeout:?}");
            }
            let remaining = deadline.saturating_duration_since(now);
            thread::sleep(remaining.min(EXIT_POLL_INTERVAL));
        }
    }

    fn shutdown(&mut self) {
        if !self.exited {
            let _ = self.child.kill();
            let _ = self.child.wait();
            self.exited = true;
        }
        self.join_output_threads();
    }

    fn join_output_threads(&mut self) {
        if let Some(handle) = self.stdout_thread.take() {
            let _ = handle.join();
        }
        if let Some(handle) = self.stderr_thread.take() {
            let _ = handle.join();
        }
    }
}

pub fn start_server(
    binary: &Path,
    model_dir: &Path,
    port_hint: Option<u16>,
) -> anyhow::Result<ServerHandle> {
    start_server_with_command_config(binary, model_dir, port_hint, READY_TIMEOUT, |_| Ok(()))
}

pub(crate) fn start_server_with_command_config(
    binary: &Path,
    model_dir: &Path,
    port_hint: Option<u16>,
    ready_timeout: Duration,
    configure: impl FnOnce(&mut Command) -> anyhow::Result<()>,
) -> anyhow::Result<ServerHandle> {
    let requested_port = choose_port(port_hint)?;
    log::info!(
        "spawning lloom-serve for {} on 127.0.0.1:{requested_port}",
        model_dir.display()
    );
    let mut command = Command::new(binary);
    command
        .arg("--model-dir")
        .arg(model_dir)
        .arg("--port")
        .arg(requested_port.to_string())
        .env(PROMPT_SEED_TEXT_ENV, PROMPT_SEED_TEXT);
    configure(&mut command).context("failed to configure the lloom-serve command")?;
    command.stdout(Stdio::piped()).stderr(Stdio::piped());
    pipette_subprocess::echo_info(&command);
    let command_preview = pipette_subprocess::argv(&command);
    let mut child = command
        .spawn()
        .with_context(|| format!("failed to spawn lloom-serve at {}", binary.display()))?;
    let cleanup_guard = pipette_subprocess::cleanup::Guard::for_pid(child.id());
    let stdout = take_child_stdout(&mut child)?;
    let stderr = take_child_stderr(&mut child)?;
    let stdout_buf = Arc::new(Mutex::new(String::new()));
    let (ready_rx, stdout_thread) = spawn_stdout_reader(stdout, Arc::clone(&stdout_buf));
    let stderr_buf = Arc::new(Mutex::new(String::new()));
    let stderr_thread = spawn_stderr_reader(stderr, Arc::clone(&stderr_buf));
    let port = match wait_for_ready_marker(&mut child, &ready_rx, requested_port, ready_timeout) {
        Ok(port) => port,
        Err(err) => {
            // Reap first, then report: the stderr thread has read everything
            // the process wrote only once the process is gone, and the tail
            // is the diagnostic (a missing model dir, a Metal error).
            let _ = child.kill();
            let _ = child.wait();
            let _ = stdout_thread.join();
            let _ = stderr_thread.join();
            return Err(err.context(stderr_tail_hint(&stderr_buf)));
        }
    };
    log::info!("lloom-serve is ready at http://127.0.0.1:{port}");
    Ok(ServerHandle {
        child,
        exited: false,
        stdout_thread: Some(stdout_thread),
        stderr_thread: Some(stderr_thread),
        stdout_buf,
        stderr_buf,
        _cleanup_guard: Some(cleanup_guard),
        base_url: format!("http://127.0.0.1:{port}"),
        executable: binary.display().to_string(),
        command_preview,
    })
}

fn take_child_stdout(child: &mut Child) -> anyhow::Result<ChildStdout> {
    match child.stdout.take() {
        Some(stdout) => Ok(stdout),
        None => {
            let _ = child.kill();
            let _ = child.wait();
            anyhow::bail!("lloom-serve stdout missing")
        }
    }
}

fn take_child_stderr(child: &mut Child) -> anyhow::Result<ChildStderr> {
    match child.stderr.take() {
        Some(stderr) => Ok(stderr),
        None => {
            let _ = child.kill();
            let _ = child.wait();
            anyhow::bail!("lloom-serve stderr missing")
        }
    }
}

fn choose_port(port_hint: Option<u16>) -> anyhow::Result<u16> {
    match port_hint {
        Some(port) => Ok(port),
        None => pick_free_port(),
    }
}

fn pick_free_port() -> anyhow::Result<u16> {
    let listener = TcpListener::bind("127.0.0.1:0")
        .context("failed to bind 127.0.0.1:0 for free-port discovery")?;
    Ok(listener
        .local_addr()
        .context("failed to read assigned port")?
        .port())
}

type ReadyRead = std::result::Result<String, ReadyReadError>;

#[derive(Debug, thiserror::Error)]
enum ReadyReadError {
    #[error("failed reading stdout: {0}")]
    Io(#[from] std::io::Error),
    #[error("lloom-serve stdout closed before the ready marker")]
    ClosedBeforeReady,
}

/// The first stdout line goes to the ready channel; the rest is captured.
fn spawn_stdout_reader(
    stdout: ChildStdout,
    stdout_buf: Arc<Mutex<String>>,
) -> (mpsc::Receiver<ReadyRead>, JoinHandle<()>) {
    let (tx, rx) = mpsc::channel();
    let handle = thread::spawn(move || {
        let mut ready_tx = Some(tx);
        let read_result = BufReader::new(stdout).lines().try_for_each(|line| {
            let line = line?;
            if let Some(tx) = ready_tx.take() {
                let _ = tx.send(Ok(line));
            } else if !line.trim().is_empty() {
                log::info!(target: "pipette_lloom::server", "{line}");
                push_capped_line(&stdout_buf, &line);
            }
            Ok::<_, std::io::Error>(())
        });
        match (read_result, ready_tx.take()) {
            (Err(err), Some(tx)) => {
                let _ = tx.send(Err(ReadyReadError::Io(err)));
            }
            (Ok(()), Some(tx)) => {
                let _ = tx.send(Err(ReadyReadError::ClosedBeforeReady));
            }
            _ => {}
        }
    });
    (rx, handle)
}

fn spawn_stderr_reader(stderr: ChildStderr, stderr_buf: Arc<Mutex<String>>) -> JoinHandle<()> {
    thread::spawn(move || {
        BufReader::new(stderr)
            .lines()
            .map_while(std::result::Result::ok)
            .for_each(|line| {
                log::info!(target: "pipette_lloom::server", "{line}");
                push_capped_line(&stderr_buf, &line);
            });
    })
}

fn push_capped_line(buf: &Arc<Mutex<String>>, line: &str) {
    let mut buf = buf.lock().unwrap_or_else(|e| e.into_inner());
    buf.push_str(line);
    buf.push('\n');
    if buf.len() > OUTPUT_CAPTURE_BYTES {
        let over = buf.len() - OUTPUT_CAPTURE_BYTES;
        let cutoff = buf[over..].find('\n').map(|i| over + i + 1).unwrap_or(over);
        buf.drain(..cutoff);
    }
}

fn wait_for_ready_marker(
    child: &mut Child,
    ready_rx: &mpsc::Receiver<ReadyRead>,
    requested_port: u16,
    timeout: Duration,
) -> anyhow::Result<u16> {
    let deadline = Instant::now() + timeout;
    loop {
        if let Some(status) = child
            .try_wait()
            .context("failed to inspect lloom-serve status")?
        {
            anyhow::bail!("lloom-serve exited before the ready marker with {status}");
        }
        let now = Instant::now();
        if now >= deadline {
            anyhow::bail!("timed out waiting for the lloom-serve ready marker after {timeout:?}");
        }
        let remaining = deadline.saturating_duration_since(now);
        match ready_rx.recv_timeout(remaining.min(READY_POLL_INTERVAL)) {
            Ok(Ok(line)) => return parse_ready_marker(&line, requested_port),
            // stdout closed: the process is exiting (or exited between the
            // `try_wait` above and here). Its status is the message.
            Ok(Err(ReadyReadError::ClosedBeforeReady))
            | Err(mpsc::RecvTimeoutError::Disconnected) => {
                let status = child
                    .wait()
                    .context("failed to reap lloom-serve after it closed stdout")?;
                anyhow::bail!("lloom-serve exited before the ready marker with {status}");
            }
            Ok(Err(err)) => anyhow::bail!("failed reading the lloom-serve ready marker: {err}"),
            Err(mpsc::RecvTimeoutError::Timeout) => {}
        }
    }
}

#[derive(Debug, Deserialize)]
struct ReadyMarker {
    kind: String,
    port: u16,
}

fn parse_ready_marker(line: &str, requested_port: u16) -> anyhow::Result<u16> {
    let marker: ReadyMarker = serde_json::from_str(line.trim())
        .with_context(|| format!("failed to parse the lloom-serve ready marker: {line:?}"))?;
    if marker.kind != "ready" {
        anyhow::bail!("unexpected lloom-serve marker kind {:?}", marker.kind);
    }
    if requested_port != 0 && marker.port != requested_port {
        anyhow::bail!(
            "lloom-serve reported port {}, expected {}",
            marker.port,
            requested_port
        );
    }
    Ok(marker.port)
}

fn stderr_tail_hint(stderr_buf: &Arc<Mutex<String>>) -> String {
    let buf = stderr_buf.lock().unwrap_or_else(|e| e.into_inner());
    let tail: Vec<&str> = buf.lines().rev().take(20).collect();
    if tail.is_empty() {
        "lloom-serve wrote nothing to stderr".to_string()
    } else {
        let mut lines = tail;
        lines.reverse();
        format!("lloom-serve stderr tail:\n{}", lines.join("\n"))
    }
}

#[cfg(test)]
mod tests {
    use std::{fs, os::unix::fs::PermissionsExt, path::PathBuf, process::Command};

    use serde_json::{json, Value};

    use super::*;

    /// A stand-in binary with the contract's shape: the same flags, a ready
    /// marker on stdout once bound, `/health` and `/tokenize`. Python does
    /// the serving; the shell wrapper is what makes it a "binary".
    const FAKE_SERVER_PY: &str = r#"
import argparse, json, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
p = argparse.ArgumentParser()
p.add_argument("--model-dir", required=True)
p.add_argument("--port", type=int, required=True)
a = p.parse_args()
class H(BaseHTTPRequestHandler):
    def _json(self, status, payload):
        body = json.dumps(payload).encode()
        self.send_response(status); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
    def do_GET(self):
        self._json(200, {}) if self.path == "/health" else self._json(404, {"error": self.path})
    def do_POST(self):
        if self.path != "/tokenize":
            self._json(404, {"error": self.path}); return
        n = int(self.headers.get("Content-Length", "0"))
        req = json.loads(self.rfile.read(n).decode() or "{}")
        toks = list(range(len(req.get("prompt", ""))))
        self._json(200, {"tokens": toks, "count": len(toks)})
    def log_message(self, *args): pass
class S(ThreadingHTTPServer):
    # HTTPServer.server_bind does a reverse DNS lookup of the host, which can
    # hang for seconds; bind without it, as the MLX fake does.
    def server_bind(self):
        from socketserver import TCPServer
        TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address[:2]
srv = S(("127.0.0.1", a.port), H)
print(json.dumps({"kind": "ready", "host": "127.0.0.1", "port": srv.server_address[1]}), flush=True)
print("loaded", file=sys.stderr, flush=True)
srv.serve_forever()
"#;

    fn python3() -> Option<PathBuf> {
        Command::new("/usr/bin/env")
            .arg("python3")
            .arg("--version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
            .then(|| PathBuf::from("python3"))
    }

    /// Materialize the fake as an executable `lloom-serve` under a temp dir.
    fn fake_binary(dir: &Path, script: &str) -> anyhow::Result<PathBuf> {
        let py = dir.join("fake.py");
        fs::write(&py, script)?;
        let bin = dir.join("lloom-serve");
        fs::write(
            &bin,
            format!("#!/bin/sh\nexec python3 {} \"$@\"\n", py.display()),
        )?;
        fs::set_permissions(&bin, fs::Permissions::from_mode(0o755))?;
        Ok(bin)
    }

    fn http() -> anyhow::Result<reqwest::blocking::Client> {
        pipette_http::HttpClient::blocking_with_timeout("pipette", Duration::from_secs(5))
            .context("failed to build test HTTP client")
    }

    #[test]
    fn parses_the_ready_marker_and_checks_the_port() -> anyhow::Result<()> {
        assert_eq!(
            parse_ready_marker(r#"{"kind":"ready","host":"127.0.0.1","port":18080}"#, 18080)?,
            18080
        );
        assert_eq!(parse_ready_marker(r#"{"kind":"ready","port":5}"#, 0)?, 5);
        assert!(parse_ready_marker(r#"{"kind":"ready","port":5}"#, 6).is_err());
        assert!(parse_ready_marker(r#"{"kind":"loaded","port":5}"#, 5).is_err());
        assert!(parse_ready_marker("[loaded LFM2.5-2.6B]", 5).is_err());
        Ok(())
    }

    #[test]
    fn start_server_waits_for_ready_and_serves_health_and_tokenize() -> anyhow::Result<()> {
        let Some(_) = python3() else {
            eprintln!("python3 not found; skipping");
            return Ok(());
        };
        let tmp = tempfile::tempdir()?;
        let bin = fake_binary(tmp.path(), FAKE_SERVER_PY)?;
        let server = start_server_with_command_config(
            &bin,
            Path::new("fake/model"),
            None,
            Duration::from_secs(10),
            |_| Ok(()),
        )?;
        let client = http()?;
        let health = client.get(format!("{}/health", server.base_url)).send()?;
        assert!(health.status().is_success());
        let tok: Value = client
            .post(format!("{}/tokenize", server.base_url))
            .json(&json!({"prompt": "abcd"}))
            .send()?
            .json()?;
        assert_eq!(tok["count"], 4);
        assert!(server.command_preview.iter().any(|a| a == "--model-dir"));
        Ok(())
    }

    #[test]
    fn a_binary_that_never_reports_ready_is_killed_at_the_deadline() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let bin = fake_binary(tmp.path(), "import time\ntime.sleep(30)\n")?;
        let err = start_server_with_command_config(
            &bin,
            Path::new("fake/model"),
            None,
            Duration::from_millis(500),
            |_| Ok(()),
        )
        .err()
        .map(|e| format!("{e:#}"))
        .unwrap_or_default();
        assert!(err.contains("timed out"), "got: {err}");
        Ok(())
    }

    #[test]
    fn a_binary_that_exits_early_is_reported_with_its_stderr() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let bin = fake_binary(
            tmp.path(),
            "import sys\nprint('no model at that path', file=sys.stderr)\nsys.exit(3)\n",
        )?;
        let err = start_server_with_command_config(
            &bin,
            Path::new("fake/model"),
            None,
            Duration::from_secs(10),
            |_| Ok(()),
        )
        .err()
        .map(|e| format!("{e:#}"))
        .unwrap_or_default();
        assert!(err.contains("exited before the ready marker"), "got: {err}");
        assert!(err.contains("no model at that path"), "got: {err}");
        Ok(())
    }
}
