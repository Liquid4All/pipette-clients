//! Shared ^C / SIGTERM teardown registry for spawned subprocesses.
//!
//! Rust's default ^C handler calls `process::exit` without unwinding, so any
//! [`Drop`] guard on an in-flight child never runs and the spawned server /
//! container outlives us. This module installs a `ctrlc`-backed handler that
//! drains a registry of in-flight teardown callbacks (each spawn site
//! registers one and deregisters when its happy-path teardown is done), then
//! exits 130.
//!
//! Each runtime crate registers a teardown appropriate to how it spawned its
//! child: a process-group SIGKILL (llamacpp's `llama-bench`/`llama-server`,
//! torch's uv server), a single-pid SIGKILL (mlx's python child), or a
//! `docker stop` (torch's container engine).

use std::sync::{Mutex, OnceLock};

/// Opaque token returned by [`register`] and consumed by [`deregister`]
/// when the happy-path teardown has already run. Holding it in an RAII
/// guard (`Drop` calls [`deregister`]) keeps the registry from growing
/// per run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TeardownToken(u64);

type Teardown = Box<dyn FnOnce() + Send + 'static>;

struct Entry {
    token: TeardownToken,
    teardown: Teardown,
}

fn registry() -> &'static Mutex<Vec<Entry>> {
    static REG: OnceLock<Mutex<Vec<Entry>>> = OnceLock::new();
    REG.get_or_init(|| Mutex::new(Vec::new()))
}

fn next_token() -> TeardownToken {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(1);
    TeardownToken(NEXT.fetch_add(1, Ordering::Relaxed))
}

/// Add a teardown callback to the in-flight set. Call [`deregister`] with
/// the returned token once the normal-path teardown has finished so the
/// signal handler can't double-fire on it.
pub fn register<F>(teardown: F) -> TeardownToken
where
    F: FnOnce() + Send + 'static,
{
    let token = next_token();
    // A poisoned lock only means a prior holder panicked; the registry vec
    // itself is still consistent, so recover the guard rather than abort.
    let mut guard = registry().lock().unwrap_or_else(|p| p.into_inner());
    guard.push(Entry {
        token,
        teardown: Box::new(teardown),
    });
    token
}

/// Drop a callback from the in-flight set after its happy-path teardown
/// has already run. No-op on a token that was never registered.
pub fn deregister(token: TeardownToken) {
    // See `register`: recover a poisoned guard; the vec stays consistent.
    let mut guard = registry().lock().unwrap_or_else(|p| p.into_inner());
    guard.retain(|e| e.token != token);
}

/// Install the SIGINT/SIGTERM handler once per process. Subsequent calls
/// are no-ops. Failure to install is logged but non-fatal.
pub fn install_signal_handler() {
    static INSTALLED: OnceLock<()> = OnceLock::new();
    INSTALLED.get_or_init(|| {
        if let Err(err) = ctrlc::set_handler(handle_signal) {
            log::warn!("failed to install signal handler: {err}");
        }
    });
}

fn handle_signal() {
    // Atomically swap out the registry so each callback runs exactly once
    // even if the handler somehow re-enters.
    let entries: Vec<Entry> = registry()
        .lock()
        .map(|mut g| std::mem::take(&mut *g))
        .unwrap_or_default();
    if !entries.is_empty() {
        eprintln!(
            "\n[pipette] interrupt received; tearing down {} resource(s) before exit",
            entries.len()
        );
        for entry in entries {
            (entry.teardown)();
        }
    }
    // 130 = 128 + SIGINT, the conventional shell exit code for ^C.
    std::process::exit(130);
}

/// Best-effort single-process SIGKILL. The right primitive when the child
/// stays in the parent's process group (no `setpgid`), so a targeted
/// `kill(pid, SIGKILL)` reaches it without touching siblings.
pub fn kill_pid(pid: u32) {
    #[cfg(unix)]
    // SAFETY: `kill(2)` is signal-safe; targeting a non-existent pid is
    // benign (sets errno = ESRCH).
    unsafe {
        libc::kill(pid as libc::pid_t, libc::SIGKILL);
    }
    #[cfg(windows)]
    {
        let _ = std::process::Command::new("taskkill")
            .args(["/F", "/PID", &pid.to_string()])
            .status();
    }
}

/// Best-effort process-group kill on Unix, single-process subtree kill on
/// Windows. The right primitive when the child leads its own process group
/// (`setpgid`), so `kill(-pid, SIGKILL)` reaches the child and any grandchildren.
pub fn kill_process_group(pid: u32) {
    #[cfg(unix)]
    // SAFETY: `kill(2)` is signal-safe; targeting a non-existent pgid is
    // benign (sets errno = ESRCH). The pgid leader's pgid equals its pid.
    unsafe {
        libc::kill(-(pid as libc::pid_t), libc::SIGKILL);
    }
    #[cfg(windows)]
    {
        // Windows has no process-group concept; `taskkill /F /T /PID`
        // walks the subtree (the `/T`) and force-terminates each — the
        // closest equivalent to `kill(-pid, SIGKILL)` on the Unix arm.
        let _ = std::process::Command::new("taskkill")
            .args(["/F", "/T", "/PID", &pid.to_string()])
            .status();
    }
}

/// RAII wrapper around a teardown registration. The signal handler's callback
/// is only reached on ^C / SIGTERM — never via this Drop, which simply
/// removes the entry so the registry doesn't grow per run. Spawn sites
/// hold one of these in a `_cleanup_guard` (or an `Option` field) so
/// every exit path — success, `?`-return, panic-unwind — deregisters.
pub struct Guard {
    token: TeardownToken,
}

impl Guard {
    /// Register `f` as a teardown callback, wrapped in a `Guard`.
    pub fn register<F>(f: F) -> Self
    where
        F: FnOnce() + Send + 'static,
    {
        Self { token: register(f) }
    }

    /// Convenience: a single-pid SIGKILL ([`kill_pid`]) wrapped in a `Guard`.
    pub fn for_pid(pid: u32) -> Self {
        Self::register(move || kill_pid(pid))
    }

    /// Convenience: a process-group SIGKILL ([`kill_process_group`]) wrapped
    /// in a `Guard`.
    pub fn for_process_group(pid: u32) -> Self {
        Self::register(move || kill_process_group(pid))
    }

    /// The token under this guard. Exposed for tests that need to assert
    /// registry state without consuming the guard.
    #[cfg(test)]
    fn token(&self) -> TeardownToken {
        self.token
    }
}

impl Drop for Guard {
    fn drop(&mut self) {
        deregister(self.token);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{atomic::AtomicBool, Arc};

    use super::*;

    fn registry_contains(token: TeardownToken) -> anyhow::Result<bool> {
        let guard = registry()
            .lock()
            .map_err(|e| anyhow::anyhow!("registry mutex poisoned: {e}"))?;
        Ok(guard.iter().any(|e| e.token == token))
    }

    #[test]
    fn register_and_deregister_round_trips() -> anyhow::Result<()> {
        let fired = Arc::new(AtomicBool::new(false));
        let f = fired.clone();
        let token = register(move || {
            f.store(true, std::sync::atomic::Ordering::SeqCst);
        });
        assert!(registry_contains(token)?);
        deregister(token);
        assert!(!registry_contains(token)?);
        // The callback was never invoked because we deregistered before
        // the signal-handler path.
        assert!(!fired.load(std::sync::atomic::Ordering::SeqCst));
        Ok(())
    }

    #[test]
    fn deregister_unknown_is_noop() {
        deregister(TeardownToken(u64::MAX));
    }

    /// `Guard`'s Drop is the load-bearing invariant for every spawn site.
    /// Pin the round trip: registered while held, deregistered on Drop, the
    /// teardown callback never fires (signals are the only path that reaches
    /// it).
    #[test]
    fn guard_drop_deregisters_token_from_registry() -> anyhow::Result<()> {
        // We can't go through `Guard::for_pid` here because we want to
        // observe whether the teardown callback was invoked — that wires a
        // real SIGKILL, not a test sentinel. So we build a `Guard` by hand
        // from a sentinel registration; the Drop path is the same code
        // under test.
        let fired = Arc::new(AtomicBool::new(false));
        let f = fired.clone();
        let token = register(move || {
            f.store(true, std::sync::atomic::Ordering::SeqCst);
        });
        let guard = Guard { token };
        assert!(registry_contains(guard.token())?);
        drop(guard);
        assert!(!registry_contains(token)?);
        assert!(!fired.load(std::sync::atomic::Ordering::SeqCst));
        Ok(())
    }
}
