use std::{
    io::{BufRead, BufReader},
    net::TcpListener,
    process::{Child, ChildStderr, ChildStdout, Command, Stdio},
    sync::{mpsc, Arc, Mutex},
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use anyhow::Context;

use pipette_plan_types::run::RunRequest;

use crate::models::require_coreai_model_dir;
use crate::runtimes::require_coreai_sidecar;

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

    pub fn wait_for_exit(&mut self, timeout: Duration) -> anyhow::Result<()> {
        let deadline = Instant::now() + timeout;
        loop {
            if let Some(status) = self
                .child
                .try_wait()
                .context("failed to inspect pipette-coreai-sidecar status during shutdown")?
            {
                self.exited = true;
                self.join_output_threads();
                if !status.success() {
                    anyhow::bail!("pipette-coreai-sidecar exited with {status}");
                }
                return Ok(());
            }
            let now = Instant::now();
            if now >= deadline {
                anyhow::bail!("timed out waiting for pipette-coreai-sidecar exit after {timeout:?}");
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

/// Start the Swift sidecar for the bound Core AI model, waiting for its ready
/// marker on stderr.
pub fn start_server(req: &RunRequest, port_hint: Option<u16>) -> anyhow::Result<ServerHandle> {
    let sidecar = require_coreai_sidecar(req)?;
    let model_dir = require_coreai_model_dir(req)?;
    let port = choose_port(port_hint)?;

    log::info!(
        "spawning pipette-coreai-sidecar for {} on 127.0.0.1:{port}",
        model_dir.display()
    );

    let mut command = Command::new(&sidecar);
    command
        .arg("--model")
        .arg(&model_dir)
        .arg("--port")
        .arg(port.to_string())
        .arg("--seed")
        .arg("42");
    command.stdout(Stdio::piped()).stderr(Stdio::piped());

    pipette_subprocess::echo_info(&command);
    let command_preview = pipette_subprocess::argv(&command);
    let mut child = command
        .spawn()
        .with_context(|| format!("failed to spawn pipette-coreai-sidecar at {}", sidecar.display()))?;

    let stdout = take_child_stdout(&mut child)?;
    let stderr = take_child_stderr(&mut child)?;
    let stdout_buf = Arc::new(Mutex::new(String::new()));
    let (ready_rx, stdout_thread) = spawn_stdout_reader(stdout, Arc::clone(&stdout_buf));
    let stderr_buf = Arc::new(Mutex::new(String::new()));
    let stderr_thread = spawn_stderr_reader(stderr, Arc::clone(&stderr_buf));

    let ready_port = match wait_for_ready_marker(
        &mut child,
        &ready_rx,
        port,
        READY_TIMEOUT,
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

    log::info!("pipette-coreai-sidecar is ready at http://127.0.0.1:{ready_port}");
    Ok(ServerHandle {
        child,
        exited: false,
        stdout_thread: Some(stdout_thread),
        stderr_thread: Some(stderr_thread),
        stdout_buf,
        stderr_buf,
        base_url: format!("http://127.0.0.1:{ready_port}"),
        executable: sidecar.display().to_string(),
        command_preview,
    })
}

fn take_child_stdout(child: &mut Child) -> anyhow::Result<ChildStdout> {
    match child.stdout.take() {
        Some(stdout) => Ok(stdout),
        None => {
            let _ = child.kill();
            let _ = child.wait();
            anyhow::bail!("pipette-coreai-sidecar stdout missing")
        }
    }
}

fn take_child_stderr(child: &mut Child) -> anyhow::Result<ChildStderr> {
    match child.stderr.take() {
        Some(stderr) => Ok(stderr),
        None => {
            let _ = child.kill();
            let _ = child.wait();
            anyhow::bail!("pipette-coreai-sidecar stderr missing")
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
    #[error("pipette-coreai-sidecar stdout closed before ready marker")]
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
                log::info!(target: "pipette_coreai::server", "{line}");
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
                log::info!(target: "pipette_coreai::server", "{line}");
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
    stderr_buf: &Arc<Mutex<String>>,
) -> anyhow::Result<u16> {
    let deadline = Instant::now() + timeout;
    loop {
        if let Some(status) = child
            .try_wait()
            .context("failed to inspect pipette-coreai-sidecar status")?
        {
            anyhow::bail!(
                "pipette-coreai-sidecar exited before ready marker with {status}{}",
                stderr_tail_hint(stderr_buf)
            );
        }
        let now = Instant::now();
        if now >= deadline {
            anyhow::bail!(
                "timed out waiting for pipette-coreai-sidecar ready marker after {timeout:?}{}",
                stderr_tail_hint(stderr_buf)
            );
        }
        let remaining = deadline.saturating_duration_since(now);
        let poll = remaining.min(READY_POLL_INTERVAL);
        match ready_rx.recv_timeout(poll) {
            Ok(Ok(line)) => return parse_ready_marker(&line, requested_port),
            Ok(Err(err)) => anyhow::bail!(
                "failed reading pipette-coreai-sidecar ready marker: {err}{}",
                stderr_tail_hint(stderr_buf)
            ),
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => anyhow::bail!(
                "pipette-coreai-sidecar stdout reader stopped before ready marker{}",
                stderr_tail_hint(stderr_buf)
            ),
        }
    }
}

fn parse_ready_marker(line: &str, requested_port: u16) -> anyhow::Result<u16> {
    let trimmed = line.trim();
    let port = trimmed
        .split_whitespace()
        .find_map(|tok| tok.strip_prefix("port="))
        .context("ready marker missing port=")?;
    let port: u16 = port
        .parse()
        .with_context(|| format!("invalid port in ready marker: {line:?}"))?;
    if requested_port != 0 && port != requested_port {
        anyhow::bail!("pipette-coreai-sidecar reported port {port}, expected {requested_port}");
    }
    Ok(port)
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
