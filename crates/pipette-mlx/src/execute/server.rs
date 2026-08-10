use std::{
    fs,
    io::{BufRead, BufReader},
    net::TcpListener,
    path::{Path, PathBuf},
    process::{Child, ChildStderr, ChildStdout, Command, Stdio},
    sync::{mpsc, Arc, Mutex},
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use anyhow::Context;
use serde::Deserialize;

use pipette_ops::prompt_seed::PROMPT_SEED_TEXT;

const MLX_SERVER_SCRIPT: &str = include_str!("../python/pipette_mlx_server.py");
const PROMPT_SEED_TEXT_ENV: &str = "PIPETTE_MLX_PROMPT_SEED_TEXT";
const PYTHON_UNBUFFERED_ENV: &str = "PYTHONUNBUFFERED";
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
    /// The interpreter this server runs under, and the argv it was spawned
    /// with — recorded on the outcome so a stored result says what produced it,
    /// the same as a `llama-bench` cell does.
    pub executable: String,
    pub command_preview: Vec<String>,
    #[cfg(test)]
    port: u16,
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

    /// Captured server stdout since readiness (the ready marker is excluded),
    /// up to the capture cap — recorded on the benchmark outcome for provenance.
    pub fn stdout(&self) -> String {
        self.stdout_buf
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// Captured server stderr (where mlx-lm logs), up to the capture cap.
    pub fn stderr(&self) -> String {
        self.stderr_buf
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    pub fn wait_for_exit(&mut self, timeout: Duration) -> anyhow::Result<()> {
        let deadline = Instant::now() + timeout;
        loop {
            if let Some(status) = self
                .child
                .try_wait()
                .context("failed to inspect pipette_mlx_server status during shutdown")?
            {
                self.exited = true;
                self.join_output_threads();
                if !status.success() {
                    anyhow::bail!("pipette_mlx_server exited with {status}");
                }
                return Ok(());
            }

            let now = Instant::now();
            if now >= deadline {
                anyhow::bail!("timed out waiting for pipette_mlx_server exit after {timeout:?}");
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
    venv_python: &Path,
    model_dir: &Path,
    port_hint: Option<u16>,
) -> anyhow::Result<ServerHandle> {
    start_server_with_command_config(venv_python, model_dir, port_hint, |_| Ok(()))
}

pub(crate) fn start_server_with_command_config(
    venv_python: &Path,
    model_dir: &Path,
    port_hint: Option<u16>,
    configure: impl FnOnce(&mut Command) -> anyhow::Result<()>,
) -> anyhow::Result<ServerHandle> {
    start_server_with_script(
        venv_python,
        model_dir,
        port_hint,
        MLX_SERVER_SCRIPT,
        READY_TIMEOUT,
        configure,
    )
}

fn start_server_with_script(
    venv_python: &Path,
    model_dir: &Path,
    port_hint: Option<u16>,
    script: &str,
    ready_timeout: Duration,
    configure: impl FnOnce(&mut Command) -> anyhow::Result<()>,
) -> anyhow::Result<ServerHandle> {
    let requested_port = choose_port(port_hint)?;
    log::info!(
        "spawning pipette_mlx_server for {} on 127.0.0.1:{requested_port}",
        model_dir.display()
    );
    let script_path = materialize_server_script(script)?;

    let mut command = Command::new(venv_python);
    command
        .arg(&script_path)
        .arg("--model")
        .arg(model_dir)
        .arg("--port")
        .arg(requested_port.to_string())
        .env(PROMPT_SEED_TEXT_ENV, PROMPT_SEED_TEXT)
        .env(PYTHON_UNBUFFERED_ENV, "1");
    configure(&mut command).context("failed to configure pipette_mlx_server command")?;
    command.stdout(Stdio::piped()).stderr(Stdio::piped());

    pipette_subprocess::echo_info(&command);
    let command_preview = pipette_subprocess::argv(&command);
    let mut child = command.spawn().with_context(|| {
        format!(
            "failed to spawn pipette_mlx_server via {}",
            venv_python.display()
        )
    })?;
    let cleanup_guard = pipette_subprocess::cleanup::Guard::for_pid(child.id());

    let stdout = take_child_stdout(&mut child)?;
    let stderr = take_child_stderr(&mut child)?;
    let stdout_buf = Arc::new(Mutex::new(String::new()));
    let (ready_rx, stdout_thread) = spawn_stdout_reader(stdout, Arc::clone(&stdout_buf));
    let stderr_buf = Arc::new(Mutex::new(String::new()));
    let stderr_thread = spawn_stderr_reader(stderr, Arc::clone(&stderr_buf));

    let port = match wait_for_ready_marker(
        &mut child,
        &ready_rx,
        requested_port,
        ready_timeout,
        &stderr_buf,
    ) {
        Ok(port) => port,
        Err(err) => {
            let _ = child.kill();
            let _ = child.wait();
            let _ = stdout_thread.join();
            let _ = stderr_thread.join();
            return Err(err);
        }
    };

    log::info!("pipette_mlx_server is ready at http://127.0.0.1:{port}");
    Ok(ServerHandle {
        child,
        exited: false,
        stdout_thread: Some(stdout_thread),
        stderr_thread: Some(stderr_thread),
        stdout_buf,
        stderr_buf,
        _cleanup_guard: Some(cleanup_guard),
        base_url: format!("http://127.0.0.1:{port}"),
        executable: venv_python.display().to_string(),
        command_preview,
        #[cfg(test)]
        port,
    })
}

fn take_child_stdout(child: &mut Child) -> anyhow::Result<ChildStdout> {
    match child.stdout.take() {
        Some(stdout) => Ok(stdout),
        None => {
            let _ = child.kill();
            let _ = child.wait();
            anyhow::bail!("pipette_mlx_server stdout missing")
        }
    }
}

fn take_child_stderr(child: &mut Child) -> anyhow::Result<ChildStderr> {
    match child.stderr.take() {
        Some(stderr) => Ok(stderr),
        None => {
            let _ = child.kill();
            let _ = child.wait();
            anyhow::bail!("pipette_mlx_server stderr missing")
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
    let port = listener
        .local_addr()
        .context("failed to read assigned port")?
        .port();
    Ok(port)
}

type ReadyRead = std::result::Result<String, ReadyReadError>;

#[derive(Debug, thiserror::Error)]
enum ReadyReadError {
    #[error("failed reading stdout: {0}")]
    Io(#[from] std::io::Error),
    #[error("pipette_mlx_server stdout closed before ready marker")]
    ClosedBeforeReady,
}

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
                log::info!(target: "pipette_mlx::server", "{line}");
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
                log::info!(target: "pipette_mlx::server", "{line}");
                push_capped_line(&stderr_buf, &line);
            });
    })
}

/// Append `line` to a captured-output buffer, trimming whole lines off the
/// front once it exceeds the cap so the buffer stays bounded (keeps the tail —
/// the most relevant part when something fails).
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
    stderr_buf: &Arc<Mutex<String>>,
) -> anyhow::Result<u16> {
    let deadline = Instant::now() + timeout;
    loop {
        if let Some(status) = child
            .try_wait()
            .context("failed to inspect pipette_mlx_server status")?
        {
            anyhow::bail!(
                "pipette_mlx_server exited before ready marker with {status}{}",
                stderr_tail_hint(stderr_buf)
            );
        }

        let now = Instant::now();
        if now >= deadline {
            anyhow::bail!(
                "timed out waiting for pipette_mlx_server ready marker after {timeout:?}{}",
                ready_timeout_diagnostics(child.id(), requested_port, stderr_buf)
            );
        }
        let remaining = deadline.saturating_duration_since(now);
        let poll = remaining.min(READY_POLL_INTERVAL);
        match ready_rx.recv_timeout(poll) {
            Ok(Ok(line)) => return parse_ready_marker(&line, requested_port),
            Ok(Err(err)) => anyhow::bail!(
                "failed reading pipette_mlx_server ready marker: {err}{}",
                stderr_tail_hint(stderr_buf)
            ),
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => anyhow::bail!(
                "pipette_mlx_server stdout reader stopped before ready marker{}",
                stderr_tail_hint(stderr_buf)
            ),
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
        .with_context(|| format!("failed to parse pipette_mlx_server ready marker: {line:?}"))?;
    if marker.kind != "ready" {
        anyhow::bail!(
            "unexpected pipette_mlx_server marker kind {:?}",
            marker.kind
        );
    }
    if requested_port != 0 && marker.port != requested_port {
        anyhow::bail!(
            "pipette_mlx_server reported port {}, expected {}",
            marker.port,
            requested_port
        );
    }
    Ok(marker.port)
}

fn stderr_snapshot(stderr_buf: &Arc<Mutex<String>>) -> String {
    stderr_buf
        .lock()
        .map(|buf| buf.clone())
        .unwrap_or_else(|poisoned| poisoned.into_inner().clone())
}

fn stderr_tail_hint(stderr_buf: &Arc<Mutex<String>>) -> String {
    let stderr = stderr_snapshot(stderr_buf);
    let tail: Vec<&str> = stderr
        .lines()
        .filter(|line| !line.trim().is_empty())
        .rev()
        .take(5)
        .collect();
    if tail.is_empty() {
        return "\nstderr tail: <none captured>".to_string();
    }
    let mut ordered = tail;
    ordered.reverse();
    format!("\nstderr tail:\n  {}", ordered.join("\n  "))
}

fn ready_timeout_diagnostics(
    child_pid: u32,
    requested_port: u16,
    stderr_buf: &Arc<Mutex<String>>,
) -> String {
    format!(
        "\nchild pid: {child_pid}\nrequested port: {requested_port}{}{}",
        stderr_tail_hint(stderr_buf),
        process_snapshot_hint(child_pid)
    )
}

fn process_snapshot_hint(pid: u32) -> String {
    let pid = pid.to_string();
    let mut command = Command::new("ps");
    command.args([
        "-p", &pid, "-o", "pid=", "-o", "ppid=", "-o", "stat=", "-o", "etime=", "-o", "command=",
    ]);
    pipette_subprocess::echo_debug(&command);
    let output = command.output();

    match output {
        Ok(output) if output.status.success() => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let snapshot = stdout.trim();
            if snapshot.is_empty() {
                "\nprocess snapshot: <empty>".to_string()
            } else {
                format!("\nprocess snapshot:\n  {}", snapshot.replace('\n', "\n  "))
            }
        }
        Ok(output) => {
            let stderr = String::from_utf8_lossy(&output.stderr);
            format!(
                "\nprocess snapshot: unavailable (`ps` exited with {}; stderr: {})",
                output.status,
                stderr.trim()
            )
        }
        Err(err) => format!("\nprocess snapshot: unavailable (failed to run `ps`: {err})"),
    }
}

/// Prefix every materialized server script path shares.
///
/// The orphan reaper matches command lines against this, so it must stay the
/// literal path [`materialize_server_script`] builds — hence one definition
/// rather than two spellings. `temp_dir()` is per-user on macOS, which scopes
/// reaping to this user's own servers.
pub(crate) fn server_script_marker() -> String {
    script_dir()
        .join(SCRIPT_FILE_PREFIX)
        .to_string_lossy()
        .into_owned()
}

const SCRIPT_FILE_PREFIX: &str = "pipette_mlx_server-";

fn script_dir() -> PathBuf {
    std::env::temp_dir().join("pipette-mlx")
}

#[cfg(test)]
pub(crate) fn materialize_server_script_for_test(script: &str) -> anyhow::Result<PathBuf> {
    materialize_server_script(script)
}

fn materialize_server_script(script: &str) -> anyhow::Result<PathBuf> {
    let dir = script_dir();
    fs::create_dir_all(&dir).with_context(|| format!("failed to create {}", dir.display()))?;
    let fingerprint = script_fingerprint(script);
    let path = dir.join(format!("{SCRIPT_FILE_PREFIX}{fingerprint:016x}.py"));

    if let Ok(existing) = fs::read_to_string(&path) {
        if existing == script {
            return Ok(path);
        }
    }

    let tmp = dir.join(format!(
        ".pipette_mlx_server-{fingerprint:016x}-{}.tmp",
        uuid::Uuid::new_v4()
    ));
    fs::write(&tmp, script).with_context(|| format!("failed to write {}", tmp.display()))?;
    fs::rename(&tmp, &path).with_context(|| {
        let _ = fs::remove_file(&tmp);
        format!("failed to move {} to {}", tmp.display(), path.display())
    })?;
    Ok(path)
}

fn script_fingerprint(script: &str) -> u64 {
    script
        .as_bytes()
        .iter()
        .fold(0xcbf29ce484222325, |hash, byte| {
            (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
        })
}

#[cfg(test)]
mod tests {
    use std::{net::TcpStream, path::PathBuf, thread};

    use reqwest::blocking::Client;
    use serde_json::{json, Value};

    use super::*;

    const FAKE_SERVER_SCRIPT: &str = r#"
import argparse
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from socketserver import TCPServer

def diagnostic(stage):
    print(
        f"fake-server diagnostic: {stage}; t={time.monotonic():.6f}; exe={sys.executable}",
        file=sys.stderr,
        flush=True,
    )

diagnostic("script-start")
parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--port", type=int, required=True)
args = parser.parse_args()
diagnostic(f"args-parsed model={args.model} port={args.port}")

class LocalThreadingHTTPServer(ThreadingHTTPServer):
    def server_bind(self):
        diagnostic("server-bind-start")
        TCPServer.server_bind(self)
        host, port = self.server_address[:2]
        self.server_name = host
        self.server_port = port
        diagnostic(f"server-bind-done host={host} port={port}")

class Handler(BaseHTTPRequestHandler):
    def _json(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._json(200, {})
        else:
            self._json(404, {"error": self.path})

    def do_POST(self):
        if self.path != "/tokenize":
            self._json(404, {"error": self.path})
            return
        length = int(self.headers.get("Content-Length", "0"))
        request = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
        tokens = list(range(len(request.get("prompt", ""))))
        self._json(200, {"tokens": tokens, "count": len(tokens)})

    def log_message(self, fmt, *args):
        pass

diagnostic("handler-defined")
diagnostic(f"binding 127.0.0.1:{args.port}")
server = LocalThreadingHTTPServer(("127.0.0.1", args.port), Handler)
diagnostic(f"server-bound port={server.server_address[1]}")
ready = json.dumps({"kind": "ready", "port": server.server_address[1]})
print(ready, flush=True)
diagnostic(f"ready-marker-printed {ready}")
server.serve_forever()
"#;

    const NEVER_READY_SCRIPT: &str = r#"
import sys
import time
print("never-ready diagnostic: script-start; sleeping", file=sys.stderr, flush=True)
time.sleep(30)
"#;

    fn find_python3() -> Option<PathBuf> {
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

    fn http_client() -> anyhow::Result<Client> {
        pipette_http::HttpClient::blocking_with_timeout("pipette", Duration::from_secs(5))
            .context("failed to build test HTTP client")
    }

    fn start_fake_server(python: &Path) -> anyhow::Result<ServerHandle> {
        start_server_with_script(
            python,
            Path::new("fake/model"),
            None,
            FAKE_SERVER_SCRIPT,
            Duration::from_secs(5),
            |_| Ok(()),
        )
    }

    #[test]
    fn materialized_server_script_keeps_rendered_command_concise() -> anyhow::Result<()> {
        let script = "print('server body should stay out of logs')";
        let path = materialize_server_script(script)?;
        let path_str = path.to_str().context("script path is not UTF-8")?;

        let mut command = Command::new("python3");
        command
            .arg(&path)
            .arg("--model")
            .arg("fake/model")
            .arg("--port")
            .arg("1234");
        let rendered = pipette_subprocess::render(&command);

        assert!(
            rendered.contains(path_str),
            "rendered command should include script path: {rendered}"
        );
        assert!(
            !rendered.contains(script),
            "rendered command should not include inline script body: {rendered}"
        );
        assert_eq!(fs::read_to_string(path)?, script);
        Ok(())
    }

    fn wait_for_port_closed(port: u16, timeout: Duration) -> bool {
        let deadline = Instant::now() + timeout;
        while Instant::now() < deadline {
            if TcpStream::connect(("127.0.0.1", port)).is_err() {
                return true;
            }
            thread::sleep(Duration::from_millis(50));
        }
        false
    }

    #[test]
    fn start_server_waits_for_ready_and_serves_health_and_tokenize() -> anyhow::Result<()> {
        let Some(python) = find_python3() else {
            eprintln!("skipping: python3 not on this host");
            return Ok(());
        };

        let server = start_fake_server(&python)?;
        let client = http_client()?;
        let health: Value = client
            .get(format!("{}/health", server.base_url))
            .send()
            .context("GET /health")?
            .error_for_status()
            .context("/health status")?
            .json()
            .context("/health json")?;
        assert_eq!(health, json!({}));

        let tokenized: Value = client
            .post(format!("{}/tokenize", server.base_url))
            .json(&json!({"prompt": "hello"}))
            .send()
            .context("POST /tokenize")?
            .error_for_status()
            .context("/tokenize status")?
            .json()
            .context("/tokenize json")?;
        assert_eq!(tokenized["count"].as_u64(), Some(5));
        assert_eq!(tokenized["tokens"].as_array().map(Vec::len), Some(5));
        Ok(())
    }

    #[test]
    fn drop_kills_and_reaps_server_process() -> anyhow::Result<()> {
        let Some(python) = find_python3() else {
            eprintln!("skipping: python3 not on this host");
            return Ok(());
        };

        let port = {
            let server = start_fake_server(&python)?;
            assert!(
                TcpStream::connect(("127.0.0.1", server.port)).is_ok(),
                "server should be listening before drop"
            );
            server.port
        };

        assert!(
            wait_for_port_closed(port, Duration::from_secs(2)),
            "server port {port} still accepts connections after handle drop"
        );
        Ok(())
    }

    #[test]
    fn ready_wait_times_out_and_kills_child() -> anyhow::Result<()> {
        let Some(python) = find_python3() else {
            eprintln!("skipping: python3 not on this host");
            return Ok(());
        };

        let err = start_server_with_script(
            &python,
            Path::new("fake/model"),
            None,
            NEVER_READY_SCRIPT,
            Duration::from_millis(200),
            |_| Ok(()),
        )
        .err()
        .context("server should time out before ready marker")?;
        let msg = format!("{err:#}");
        assert!(
            msg.contains("timed out waiting for pipette_mlx_server ready marker"),
            "unexpected error: {msg}"
        );
        assert!(
            msg.contains("child pid:")
                && msg.contains("requested port:")
                && msg.contains("stderr tail:")
                && msg.contains("process snapshot:"),
            "timeout error should include diagnostics, got: {msg}"
        );
        Ok(())
    }
}
