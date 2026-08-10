use std::path::Path;
use std::process::Command;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    mpsc, Arc,
};
use std::thread;
use std::time::Duration;

use anyhow::Context;
use reqwest::blocking::RequestBuilder;
use serde::de::DeserializeOwned;

// ──────────────────────────────────────────────────────────────────────
// Per-benchmark-type wall-clock deadlines.
//
// llama-bench (and llama-server, via `eval`) can wedge on driver bugs,
// Vulkan layer faults, or platform-specific hangs (Android KGSL, AMDVLK
// command-buffer issues, Metal pipeline state). The parent's
// `wait_with_output` blocks forever without a deadline. Each timeout is
// generous enough not to false-kill the slowest legitimate run on the
// largest model we benchmark, and tight enough to recover an
// orchestration loop from a wedge in the same hour.
//
// Per-type rather than blanket: `max_memory_usage` does a single
// 1-token decode and exits, so a 10-minute deadline is plenty. The
// throughput benches run multiple repetitions with full prefill/decode
// across contexts up to 8192 tokens — those are sized for the slowest
// device we expect to run on (Android phone, large model, cold cache),
// where individual runs can take 20+ minutes.
// ──────────────────────────────────────────────────────────────────────

/// Deadline for `max_memory_usage` runs: one prefill + one decode, one
/// repetition. Cold-cache LFM2-350M takes <60 s; we budget 10 min as
/// headroom for slow Android devices loading large models with mmap=0.
///
/// Cfg-gated to the OSes that have a `max_memory_usage` per-platform
/// module; on any other target it would otherwise be dead code.
#[cfg(any(
    target_os = "macos",
    target_os = "android",
    target_os = "windows",
    target_os = "linux"
))]
pub const MAX_MEMORY_USAGE_TIMEOUT: Duration = Duration::from_secs(600);

/// Deadline for `prefill_throughput`, `decode_throughput`, and
/// `end_to_end_latency`. Each does multiple repetitions over prompts up
/// to 8192 tokens; 1 hour covers the slowest currently-benchmarked
/// configuration with margin.
pub const LLAMA_BENCH_TIMING_TIMEOUT: Duration = Duration::from_secs(3600);

/// RAII handle for a timeout-killer thread. Drop to cancel — the
/// killer's `recv_timeout` returns `Err(Disconnected)` once the
/// `Sender` drops and the thread exits without firing.
///
/// Closing the PID-recycling window is the reason this is a handle and
/// not a fire-and-forget thread: a child that exits in milliseconds
/// must not leave a detached killer sitting on the now-recyclable pid
/// until the long deadline elapses.
pub struct TimeoutKiller {
    _tx: mpsc::Sender<()>,
    /// Set by the killer thread when the deadline elapses. Lets the
    /// caller distinguish "killed by us" from "child failed on its
    /// own" after `wait_with_output` returns.
    fired: Arc<AtomicBool>,
}

impl TimeoutKiller {
    /// `true` if the killer thread fired its `kill_fn` (deadline
    /// elapsed before the handle was dropped).
    pub fn fired(&self) -> bool {
        self.fired.load(Ordering::SeqCst)
    }
}

/// Spawn a killer thread that fires `kill_fn` after `deadline` unless
/// the returned [`TimeoutKiller`] is dropped first. `kill_fn` runs on
/// the killer thread and must be `Send + 'static`; it takes no
/// arguments because the caller closes-over whatever pid/handle is
/// needed.
pub fn spawn_timeout_killer<F>(deadline: Duration, kill_fn: F) -> TimeoutKiller
where
    F: FnOnce() + Send + 'static,
{
    use std::sync::mpsc::RecvTimeoutError;
    let (tx, rx) = mpsc::channel();
    let fired = Arc::new(AtomicBool::new(false));
    let fired_thread = Arc::clone(&fired);
    thread::spawn(move || match rx.recv_timeout(deadline) {
        Ok(_) | Err(RecvTimeoutError::Disconnected) => {}
        Err(RecvTimeoutError::Timeout) => {
            log::warn!("llama-bench deadline elapsed ({deadline:?}); firing kill");
            fired_thread.store(true, Ordering::SeqCst);
            kill_fn();
        }
    });
    TimeoutKiller { _tx: tx, fired }
}

/// Convenience: format a "deadline elapsed" error message for callers
/// that detect `TimeoutKiller::fired()` after `wait_with_output`. The
/// child's own exit status will be "signal killed" / "TerminateProcess
/// exit 1" — without this context the operator can't tell that *we*
/// killed it.
pub fn deadline_error_message(deadline: Duration, stderr_tail: &str) -> String {
    let tail: Vec<&str> = stderr_tail
        .lines()
        .filter(|l| !l.trim().is_empty())
        .rev()
        .take(20)
        .collect();
    let mut ordered = tail;
    ordered.reverse();
    format!(
        "llama-bench exceeded {deadline:?} and was killed by the parent's deadline. \
         A wedge inside llama.cpp or a GPU runtime (Vulkan / Metal / Android driver) \
         most commonly causes this. Last 20 stderr lines:\n  {}",
        if ordered.is_empty() {
            "(no stderr captured)".to_string()
        } else {
            ordered.join("\n  ")
        }
    )
}

/// Send an HTTP request and parse the JSON body, including the response
/// body text in the error when the server returns a non-2xx status. Plain
/// `.error_for_status()?` drops the body — for llama-server 4xx/5xx the
/// body usually carries a JSON error payload that tells you why.
pub fn send_json<T: DeserializeOwned>(req: RequestBuilder, what: &str) -> anyhow::Result<T> {
    let response = req
        .send()
        .with_context(|| format!("failed to call {what}"))?;
    let status = response.status();
    if !status.is_success() {
        let body = response.text().unwrap_or_default();
        anyhow::bail!("{what} failed: HTTP {} ({})", status, body.trim());
    }
    response
        .json()
        .with_context(|| format!("failed to parse {what} response"))
}

/// Configure `cmd` so the dynamic linker can find the shared
/// libraries colocated with `binary_path`. Platform-aware:
///
/// - **Linux / Android** — sets `LD_LIBRARY_PATH` to the binary's
///   directory, preserving any value the parent process inherited.
///   bionic (Android) and glibc (Linux) ld.so both honor this.
/// - **macOS / others** — no-op. Mach-O binaries shipped by upstream
///   llama.cpp use `LC_RPATH = @loader_path`, so dyld locates dylibs
///   in the binary's directory automatically. `LD_LIBRARY_PATH` is
///   ignored on macOS, and `DYLD_LIBRARY_PATH` is blocked by
///   Hardened Runtime / Library Validation on signed binaries —
///   relying on rpath is the only portable answer.
/// - **Windows** — currently a no-op. When a Windows code path
///   lands, add a `#[cfg(target_os = "windows")]` definition that
///   prepends `binary_path.parent()` to `PATH` (PE loader has no
///   rpath equivalent).
#[cfg(any(target_os = "linux", target_os = "android"))]
pub fn apply_dylib_search_env(cmd: &mut Command, binary_path: &Path) {
    use std::ffi::OsString;
    let runtime_dir = binary_path.parent().unwrap_or(Path::new("."));
    let mut value = OsString::from(runtime_dir);
    if let Some(existing) = std::env::var_os("LD_LIBRARY_PATH") {
        if !existing.is_empty() {
            value.push(":");
            value.push(existing);
        }
    }
    cmd.env("LD_LIBRARY_PATH", value);
}

#[cfg(not(any(target_os = "linux", target_os = "android")))]
pub fn apply_dylib_search_env(_cmd: &mut Command, _binary_path: &Path) {}

#[cfg(test)]
mod tests {
    use super::*;

    /// Dropping the handle before the deadline must cancel the
    /// watchdog cleanly — `kill_fn` must NOT run. Closes the
    /// PID-recycling window for short-lived children.
    #[test]
    fn killer_doesnt_fire_when_dropped_before_deadline() {
        let fired = Arc::new(AtomicBool::new(false));
        let f = Arc::clone(&fired);
        let killer = spawn_timeout_killer(Duration::from_secs(60), move || {
            f.store(true, Ordering::SeqCst);
        });
        drop(killer);
        // Give the watchdog thread a moment to observe Disconnected
        // and exit. Without sync the assertion below could race a
        // not-yet-scheduled killer thread that's about to bail —
        // empirically 200ms is enough on every CI runner we use.
        thread::sleep(Duration::from_millis(200));
        assert!(!fired.load(Ordering::SeqCst));
    }

    /// A handle that's NOT dropped before the deadline must fire its
    /// `kill_fn` exactly once and `fired()` must observe the change.
    #[test]
    fn killer_fires_after_deadline_elapses() {
        let counter = Arc::new(std::sync::atomic::AtomicU32::new(0));
        let c = Arc::clone(&counter);
        let killer = spawn_timeout_killer(Duration::from_millis(50), move || {
            c.fetch_add(1, Ordering::SeqCst);
        });
        thread::sleep(Duration::from_millis(300));
        assert!(killer.fired(), "fired() should observe the kill_fn run");
        assert_eq!(
            counter.load(Ordering::SeqCst),
            1,
            "kill_fn must run exactly once"
        );
        drop(killer);
    }

    /// `fired()` returning false in the typical happy path: a long
    /// deadline, a short-lived "child" (simulated by an immediate
    /// drop), no kill_fn invocation.
    #[test]
    fn fired_reports_false_when_killer_was_cancelled() {
        let ran = Arc::new(AtomicBool::new(false));
        let r = Arc::clone(&ran);
        let killer = spawn_timeout_killer(Duration::from_secs(60), move || {
            r.store(true, Ordering::SeqCst);
        });
        thread::sleep(Duration::from_millis(50));
        assert!(!killer.fired());
        assert!(!ran.load(Ordering::SeqCst), "kill_fn must not run");
        drop(killer);
    }

    #[test]
    fn deadline_error_message_includes_stderr_tail() {
        let stderr = "first line\nmiddle line\nlast line\n";
        let msg = deadline_error_message(Duration::from_secs(600), stderr);
        assert!(msg.contains("600s") || msg.contains("600"));
        assert!(msg.contains("last line"));
        assert!(msg.contains("middle line"));
    }

    #[test]
    fn deadline_error_message_handles_empty_stderr() {
        let msg = deadline_error_message(Duration::from_secs(600), "");
        assert!(msg.contains("(no stderr captured)"));
    }
}
