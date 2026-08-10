use std::{
    io::{BufRead, BufReader},
    net::TcpListener,
    path::Path,
    process::{Child, Command, ExitStatus, Stdio},
    sync::{Arc, Mutex},
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use anyhow::Context;

use pipette_http::HttpClient;
use pipette_plan_types::{LlamacppFlashAttention, RuntimeFlagRef, RuntimeFlags};
use pipette_subprocess::{argv, echo_info};

use crate::common::apply_dylib_search_env;
use crate::flags::{canonicalize_flag_order, reject_reserved_flags};

pub struct RunningLlamaServer {
    pub child: Child,
    pub base_url: String,
    pub command_preview: Vec<String>,
    stderr_buf: Arc<Mutex<String>>,
    stderr_thread: Option<JoinHandle<()>>,
    /// Some(_) when the child pid is in the SIGINT/SIGTERM cleanup
    /// registry. Drop on the field auto-deregisters; the test-only
    /// construction site below leaves it `None`.
    _cleanup_guard: Option<pipette_subprocess::cleanup::Guard>,
}

impl Drop for RunningLlamaServer {
    /// Guarantee the child is reaped and the stderr-reader thread is
    /// joined even on panic or early `?` return. Operations are
    /// idempotent: if `shutdown_and_collect_stderr` already ran, these
    /// calls are harmless no-ops. `_cleanup_guard` deregisters on its
    /// own Drop — no manual handling here.
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        if let Some(handle) = self.stderr_thread.take() {
            let _ = handle.join();
        }
    }
}

/// llama-server extra argv for a server cell's resolved flags — the plan's
/// entry with the benchmark's derived context and mmap policy already overlaid
/// ([`crate::runtime_flags::for_server`]). Reads the flat [`RuntimeFlagRef`]
/// form rather than per-variant patterns, so the server cells share one renderer.
pub fn args_for(flags: &RuntimeFlags) -> LlamaServerArgsBuilder {
    let r = RuntimeFlagRef::from(flags.clone());
    LlamaServerArgsBuilder::new()
        .threads(r.threads)
        .gpu_layers(r.number_gpu_layers)
        .flash_attention(r.flash_attention)
        .mmap(r.mmap)
        .ctx_size(r.ctx_size)
        .no_cache(r.no_cache)
        .raw(&r.raw)
}

/// Builds llama-server extra argv from plan knobs; finalizes on [`Self::build`].
///
/// Reached through [`args_for`]; `build(reserved, label)` then adds the flags
/// the benchmark fixes for every run.
pub struct LlamaServerArgsBuilder {
    tokens: Vec<String>,
}

impl LlamaServerArgsBuilder {
    pub fn new() -> Self {
        Self { tokens: Vec::new() }
    }

    pub fn threads(mut self, threads: Option<u32>) -> Self {
        if let Some(t) = threads {
            self.push_pair("-t", t.to_string());
        }
        self
    }

    pub fn gpu_layers(mut self, gpu_layers: Option<u32>) -> Self {
        if let Some(n) = gpu_layers {
            self.push_pair("-ngl", n.to_string());
        }
        self
    }

    pub fn flash_attention(mut self, flash_attention: Option<LlamacppFlashAttention>) -> Self {
        if let Some(fa) = flash_attention {
            self.push_pair("-fa", fa.as_str().to_string());
        }
        self
    }

    /// llama-server is mmap-on by default; only bare `--no-mmap` disables.
    pub fn mmap(mut self, mmap: Option<bool>) -> Self {
        if mmap == Some(false) {
            self.tokens.push("--no-mmap".to_string());
        }
        self
    }

    pub fn ctx_size(mut self, ctx_size: Option<u32>) -> Self {
        if let Some(c) = ctx_size {
            self.push_pair("-c", c.to_string());
        }
        self
    }

    pub fn no_cache(mut self, no_cache: Option<bool>) -> Self {
        if no_cache == Some(true) {
            self.tokens.push("--no-cache-prompt".to_string());
        }
        self
    }

    pub fn raw(mut self, raw: &[String]) -> Self {
        self.tokens.extend(raw.iter().cloned());
        self
    }

    /// Reject reserved flags, add the benchmark's fixed `--no-warmup`,
    /// canonicalize order. Cell-level defaults (context size, mmap) arrive
    /// typed, from `runtime_flags`.
    pub fn build(self, reserved_list: &[&str], label: &str) -> anyhow::Result<Vec<String>> {
        reject_reserved_flags(&self.tokens, reserved_list, label)?;
        let mut out = Vec::with_capacity(self.tokens.len() + 1);
        out.push("--no-warmup".to_string());
        out.extend(self.tokens);
        Ok(canonicalize_flag_order(&out))
    }

    fn push_pair(&mut self, flag: &str, val: String) {
        self.tokens.push(flag.to_string());
        self.tokens.push(val);
    }
}

/// Spawn `llama-server` with finalized `extra_flags` (from
/// [`LlamaServerArgsBuilder::build`]). `mmproj` is VL-only.
pub fn start(
    llama_server: &Path,
    model_path: &Path,
    mmproj: Option<&Path>,
    extra_flags: &[String],
) -> anyhow::Result<RunningLlamaServer> {
    let port = reserve_local_port()?;
    let mut command = Command::new(llama_server);
    command.arg("--model").arg(model_path);
    if let Some(mmproj_path) = mmproj {
        command.arg("--mmproj").arg(mmproj_path);
    }
    command.arg("--host").arg("127.0.0.1");
    command.arg("--port").arg(port.to_string());
    command.args(extra_flags);
    apply_dylib_search_env(&mut command, llama_server);
    command.stdout(Stdio::null()).stderr(Stdio::piped());
    let preview = argv(&command);
    echo_info(&command);
    let mut child = command.spawn().context("failed to spawn llama-server")?;
    let cleanup_guard = pipette_subprocess::cleanup::Guard::for_process_group(child.id());

    let stderr_buf = Arc::new(Mutex::new(String::new()));
    let stderr_thread = {
        let buf = Arc::clone(&stderr_buf);
        let stderr = child.stderr.take().context("stderr was piped")?;
        thread::spawn(move || {
            BufReader::new(stderr)
                .lines()
                .map_while(anyhow::Result::ok)
                .for_each(|line| {
                    if let Ok(mut b) = buf.lock() {
                        b.push_str(&line);
                        b.push('\n');
                    }
                });
        })
    };

    Ok(RunningLlamaServer {
        child,
        base_url: format!("http://127.0.0.1:{port}"),
        command_preview: preview,
        stderr_buf,
        stderr_thread: Some(stderr_thread),
        _cleanup_guard: Some(cleanup_guard),
    })
}

pub fn wait_until_ready(
    base_url: &str,
    child: &mut Child,
    request_timeout: Duration,
) -> anyhow::Result<()> {
    log::info!("waiting for llama-server at {base_url}");
    let client = HttpClient::blocking_with_timeout("pipette", request_timeout)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let deadline = Instant::now() + request_timeout;
    let mut last_logged = Instant::now();
    while Instant::now() < deadline {
        if let Some(status) = child
            .try_wait()
            .context("failed to check llama-server status")?
        {
            anyhow::bail!("llama-server exited early with {status}");
        }
        match client.get(format!("{base_url}/health")).send() {
            Ok(response) if response.status().is_success() => {
                log::info!("llama-server is ready");
                return Ok(());
            }
            _ => {
                if last_logged.elapsed() >= Duration::from_secs(5) {
                    log::info!("still waiting for llama-server");
                    last_logged = Instant::now();
                }
                thread::sleep(Duration::from_millis(200));
            }
        }
    }
    anyhow::bail!("timed out waiting for llama-server at {base_url}")
}

impl RunningLlamaServer {
    /// Return a snapshot of stderr collected so far.
    pub fn stderr_snapshot(&self) -> String {
        match self.stderr_buf.lock() {
            Ok(buf) => buf.clone(),
            Err(poisoned) => poisoned.into_inner().clone(),
        }
    }

    /// Consuming variant of [`shutdown_and_collect_stderr`]: kill +
    /// reap + join the stderr reader, returning the captured stderr
    /// by value. Use when the caller has no further need for the
    /// server (e.g. after a mid-completion crash, where the next
    /// step is to spawn a fresh one).
    pub fn shutdown_collect_stderr(mut self) -> String {
        shutdown_and_collect_stderr(&mut self)
    }
}

/// Parse EOG token IDs from llama-server stderr output.
///
/// The server prints lines like:
///   `print_info: EOG token             = 2 '<|endoftext|>'`
/// during model loading.  We extract all such IDs.
pub fn parse_eog_token_ids(stderr: &str) -> Vec<u32> {
    stderr
        .lines()
        .filter_map(|line| line.find("EOG token").map(|pos| &line[pos..]))
        .filter_map(|rest| rest.find('=').map(|eq| rest[eq + 1..].trim_start()))
        .filter_map(|after_eq| after_eq.split_whitespace().next()?.parse::<u32>().ok())
        .fold(Vec::new(), |mut ids, id| {
            if !ids.contains(&id) {
                ids.push(id);
            }
            ids
        })
}

/// Discover EOG token IDs by parsing llama-server stderr for
/// `EOG token = <id>` lines printed during model loading.
///
/// This is a workaround for llama.cpp server where `ignore_eos`
/// does not reliably suppress all EOG tokens.
pub fn discover_eog_token_ids(server: &RunningLlamaServer) -> Vec<u32> {
    // Poll up to 2s for the stderr reader thread to capture EOG lines.
    let ids = (0..20)
        .map(|i| {
            if i > 0 {
                thread::sleep(Duration::from_millis(100));
            }
            parse_eog_token_ids(&server.stderr_snapshot())
        })
        .find(|ids| !ids.is_empty())
        .unwrap_or_default();

    if ids.is_empty() {
        log::warn!("could not discover any EOG tokens from stderr");
    } else {
        log::info!("discovered EOG token ids from stderr: {ids:?}");
    }
    ids
}

#[derive(Debug, serde::Deserialize)]
pub struct TokenizeResponse {
    pub tokens: Vec<u32>,
}

pub fn shutdown_and_collect_stderr(server: &mut RunningLlamaServer) -> String {
    let _ = server.child.kill();
    let _ = server.child.wait();
    // Join the reader thread so all buffered data is flushed before we read.
    if let Some(handle) = server.stderr_thread.take() {
        let _ = handle.join();
    }
    server.stderr_snapshot()
}

/// Poll `child.try_wait()` until the child has exited or `timeout` elapses.
/// Returns `Some(status)` if the child exited within the window, `None` if
/// it is still running. Used after a `/completion` failure to decide
/// which recovery path the eval loop takes:
///
///   * `Some(_)` → the server crashed; the eval loop drains its stderr
///     and spawns a fresh one before the next sample.
///   * `None`    → the server is still alive; the eval loop keeps the
///     same process and only marks the sample failed.
///
/// On Windows the exit can take a second or more to surface to the
/// parent (STATUS_STACK_OVERFLOW typically lands in 100-500 ms; remote
/// connection resets observed under SSH can lag further), so callers
/// pass a multi-second timeout rather than a bare `try_wait`.
pub fn poll_child_exit(child: &mut Child, timeout: Duration) -> anyhow::Result<Option<ExitStatus>> {
    let deadline = Instant::now() + timeout;
    loop {
        if let Some(status) = child
            .try_wait()
            .context("failed to check llama-server status")?
        {
            return Ok(Some(status));
        }
        if Instant::now() >= deadline {
            return Ok(None);
        }
        thread::sleep(Duration::from_millis(50));
    }
}

fn reserve_local_port() -> anyhow::Result<u16> {
    let listener = TcpListener::bind(("127.0.0.1", 0)).context("failed to bind local port")?;
    let port = listener
        .local_addr()
        .context("failed to get local addr")?
        .port();
    drop(listener);
    Ok(port)
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[cfg(unix)]
    #[test]
    fn shutdown_and_collect_stderr_terminates_child_before_reading() -> anyhow::Result<()> {
        use std::{process::Stdio, thread};

        let mut child = Command::new("sh")
            .arg("-c")
            .arg("printf '%s\n' stderr-line >&2; sleep 5")
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .context("failed to spawn test child")?;

        let stderr_buf = Arc::new(Mutex::new(String::new()));
        let stderr_thread = {
            let buf = Arc::clone(&stderr_buf);
            let stderr = child.stderr.take().context("stderr was piped")?;
            thread::spawn(move || {
                BufReader::new(stderr)
                    .lines()
                    .map_while(anyhow::Result::ok)
                    .for_each(|line| {
                        if let Ok(mut b) = buf.lock() {
                            b.push_str(&line);
                            b.push('\n');
                        }
                    });
            })
        };

        thread::sleep(Duration::from_millis(100));
        let mut server = RunningLlamaServer {
            child,
            base_url: String::new(),
            command_preview: Vec::new(),
            stderr_buf,
            stderr_thread: Some(stderr_thread),
            _cleanup_guard: None,
        };
        let stderr = shutdown_and_collect_stderr(&mut server);

        assert!(stderr.contains("stderr-line"));
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn poll_child_exit_returns_status_for_exited_child() -> anyhow::Result<()> {
        use std::process::Stdio;

        let mut child = Command::new("sh")
            .arg("-c")
            .arg("exit 7")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .context("failed to spawn test child")?;
        let status =
            poll_child_exit(&mut child, Duration::from_secs(2)).context("try_wait failed")?;
        let status = status.context("expected child to have exited within the window")?;
        assert_eq!(status.code(), Some(7));
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn poll_child_exit_returns_none_when_still_running() -> anyhow::Result<()> {
        use std::process::Stdio;

        let mut child = Command::new("sh")
            .arg("-c")
            .arg("sleep 5")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .context("failed to spawn test child")?;
        let status =
            poll_child_exit(&mut child, Duration::from_millis(150)).context("try_wait failed")?;
        assert!(
            status.is_none(),
            "child should still be running after a tight timeout"
        );
        // Clean up: kill the sleeper so the test doesn't leak.
        let _ = child.kill();
        let _ = child.wait();
        Ok(())
    }

    #[test]
    fn parse_eog_token_ids_extracts_ids_from_stderr() {
        let stderr = "\
print_info: EOG token             = 2 '<|endoftext|>'
print_info: EOG token             = 7 '<|im_end|>'
some other line
";
        assert_eq!(parse_eog_token_ids(stderr), vec![2, 7]);
    }

    #[test]
    fn parse_eog_token_ids_returns_empty_for_no_matches() {
        assert_eq!(parse_eog_token_ids("no eog here\n"), Vec::<u32>::new());
    }

    fn server_flags(
        ctx_size: Option<u32>,
        mmap: Option<bool>,
        no_cache: Option<bool>,
        raw: &[&str],
    ) -> RuntimeFlags {
        RuntimeFlags::EvalLlamacppCliStockToolsGgufText {
            threads: None,
            number_gpu_layers: None,
            mmap,
            flash_attention: None,
            ctx_size,
            no_cache,
            raw: raw.iter().map(|s| (*s).to_string()).collect(),
        }
    }

    /// Every typed field reaches argv in canonical order, and `build` adds the
    /// `--no-warmup` the benchmark fixes. The cell's values arrive already
    /// resolved — llama-server's `-c` and mmap spellings are closed to `raw`,
    /// so the typed field is the only source.
    #[rstest]
    #[case::pinned_in_ram(
        server_flags(Some(8448), Some(false), None, &[]),
        &["--no-mmap", "--no-warmup", "-c", "8448"]
    )]
    #[case::mmap_left_on(
        server_flags(Some(8196), None, None, &[]),
        &["--no-warmup", "-c", "8196"]
    )]
    #[case::mmap_asked_for(
        server_flags(Some(8196), Some(true), None, &[]),
        &["--no-warmup", "-c", "8196"]
    )]
    #[case::no_cache_and_raw(
        server_flags(Some(4096), None, Some(true), &["--parallel", "2"]),
        &["--no-cache-prompt", "--no-warmup", "--parallel", "2", "-c", "4096"]
    )]
    fn args_for_renders_the_typed_fields(
        #[case] flags: RuntimeFlags,
        #[case] expected: &[&str],
    ) -> anyhow::Result<()> {
        let got = args_for(&flags).build(&[], "test")?;
        assert_eq!(got, expected);
        Ok(())
    }
}
