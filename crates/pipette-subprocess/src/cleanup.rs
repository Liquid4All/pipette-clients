//! The teardowns of spawned children, run on ^C, SIGTERM, or a closed stderr
//! for a command that asks ([`stop_on_closed_output`]).
//!
//! A signal ends the process without unwinding, so no [`Drop`] on a running
//! child would run and the server or container would outlive it. Each spawn
//! holds a [`Guard`] while its child runs, and the handler runs every
//! registered teardown, then exits 130.
//!
//! A looping command can ask ([`register_stop_flag`]) for the first signal to
//! stop it instead, while every registered teardown is bounded: a short command
//! that ends on its own deadline ([`crate::output_within`]), or a
//! [`register_bounded_teardown`] cleanup.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock, PoisonError, RwLock, RwLockReadGuard, Weak};

/// Names a registration for [`deregister`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct TeardownToken(u64);

type Teardown = Box<dyn FnOnce() + Send + 'static>;

struct Entry {
    token: TeardownToken,
    teardown: Teardown,
    /// Does not hold off a graceful stop.
    bounded: bool,
    /// Stops a running child, so [`stop_children`] runs it too.
    child: bool,
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

/// Call [`deregister`] with the token once the normal-path teardown has run,
/// so the handler does not run it again.
fn register<F>(teardown: F, bounded: bool, child: bool) -> TeardownToken
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
        bounded,
        child,
    });
    token
}

/// Stop every child this process runs, as ^C would, but stay alive: for a run
/// past its attempt limit, whose engine then fails on whatever it waited on.
/// Teardowns that are not a child's (an in-flight job's record, a corpus
/// file) stay registered. Returns how many children were stopped.
pub fn stop_children() -> usize {
    let children = {
        // As in `handle_signal`: a child mid-start registers before this runs.
        let _starts = STARTS.write().unwrap_or_else(PoisonError::into_inner);
        take_children(&mut registry().lock().unwrap_or_else(|p| p.into_inner()))
    };
    let count = children.len();
    // Unlocked, as in `handle_signal`: `docker kill` starts a child of its own.
    children.into_iter().for_each(|entry| (entry.teardown)());
    count
}

/// Remove the children's teardowns from `entries`, so each runs once.
fn take_children(entries: &mut Vec<Entry>) -> Vec<Entry> {
    let (children, rest) = std::mem::take(entries).into_iter().partition(|e| e.child);
    *entries = rest;
    children
}

/// Each start holds a read lock from spawn until it registers its teardown.
/// The signal handler holds a write lock while it selects teardowns. A signal
/// that lands mid-start waits for that child's teardown and runs it too.
static STARTS: RwLock<()> = RwLock::new(());

/// Hold off a signal's teardown until the guard drops.
/// Spawn a child while you hold the guard.
/// Register its teardown before you drop the guard.
/// Signal cleanup waits without a deadline. Code that holds this guard must
/// finish promptly, including any caller-supplied Unix `pre_exec` callbacks.
pub(crate) fn hold_off_teardown() -> RwLockReadGuard<'static, ()> {
    STARTS.read().unwrap_or_else(PoisonError::into_inner)
}

/// No-op on a token that was never registered.
fn deregister(token: TeardownToken) {
    // See `register`: recover a poisoned guard; the vec stays consistent.
    let mut guard = registry().lock().unwrap_or_else(|p| p.into_inner());
    guard.retain(|e| e.token != token);
}

/// Flags a long-running command asks the handler to set instead of exiting.
/// Held weakly: a command that returns without deregistering leaves nothing
/// behind, and a flag whose owner is gone is pruned at the next signal.
fn stop_flags() -> &'static Mutex<Vec<Weak<AtomicBool>>> {
    static FLAGS: OnceLock<Mutex<Vec<Weak<AtomicBool>>>> = OnceLock::new();
    FLAGS.get_or_init(|| Mutex::new(Vec::new()))
}

/// Ask for the first ^C / SIGTERM to set `flag` and return, rather than exit
/// the process, so a loop with a wait it can cut short (`pipette worker`
/// between jobs) leaves by its own exit path. A second signal, or one while an
/// unbounded teardown such as a child in flight is registered, tears down and
/// exits 130 as usual.
pub fn register_stop_flag(flag: &Arc<AtomicBool>) {
    // See `register`: recover a poisoned guard; the vec stays consistent.
    let mut guard = stop_flags().lock().unwrap_or_else(|p| p.into_inner());
    guard.push(Arc::downgrade(flag));
}

/// Set every live stop flag, answering whether at least one was still unset.
/// `false` means the caller exits instead.
fn request_graceful_stop() -> bool {
    // See `register`: recover a poisoned guard; the vec stays consistent.
    let mut guard = stop_flags().lock().unwrap_or_else(|p| p.into_inner());
    guard.retain(|flag| flag.strong_count() > 0);
    let mut absorbed = false;
    for flag in guard.iter().filter_map(Weak::upgrade) {
        // A loop, not an `any`: every registered flag is set, and a
        // short-circuit would leave the ones behind the first still running.
        absorbed |= !flag.swap(true, Ordering::SeqCst);
    }
    absorbed
}

/// Whether every registered teardown is bounded.
fn teardowns_idle() -> bool {
    registry()
        .lock()
        .map(|g| g.iter().all(|e| e.bounded))
        .unwrap_or(false)
}

/// Install the SIGINT/SIGTERM handler once per process. A failure is logged.
pub fn install_signal_handler() {
    static INSTALLED: OnceLock<()> = OnceLock::new();
    INSTALLED.get_or_init(|| {
        if let Err(err) = ctrlc::set_handler(handle_signal) {
            tracing::warn!(error = %err, "failed to install signal handler");
        }
    });
}

fn handle_signal() {
    let entries = {
        let _starts = STARTS.write().unwrap_or_else(PoisonError::into_inner);
        // A command that asked for a graceful stop gets one, but only while
        // every registered teardown is bounded: a child must not keep burning
        // the machine while its parent finishes a loop, so ^C during a run
        // still means "stop now".
        if teardowns_idle() && request_graceful_stop() {
            say("interrupt received; stopping after the current wait (^C again to exit now)");
            return;
        }
        take_teardowns()
    };
    // Run with the lock released: a teardown such as `docker kill` starts a
    // child of its own.
    // 130 = 128 + SIGINT, the conventional shell exit code for ^C.
    run_and_exit(entries, "interrupt received", 130);
}

/// Whether [`stop_on_closed_output`] asked a closed output to stop the process.
static STOP_ON_CLOSED_OUTPUT: AtomicBool = AtomicBool::new(false);

/// Whether a teardown before exit has started, from a signal or a closed
/// output.
static DRAINING: AtomicBool = AtomicBool::new(false);

/// Ask a closed stderr to stop the process. When stderr loses its reader, the
/// process runs every registered teardown and then exits 141.
///
/// On Unix a thread watches stderr for that, so a run that writes nothing
/// still stops. A write to [`crate::log_stderr`] that finds no reader also
/// stops the process. This also ignores `SIGPIPE`, so that write fails with
/// `EPIPE`, and the process stays alive to run the teardowns.
///
/// Use it in a command that has only its output pipe as a link to its starter,
/// such as `pipette worker` or `pipette benchmarks run` over adb or ssh. Every
/// other command keeps the default that [`crate::reset_sigpipe`] restores.
pub fn stop_on_closed_output() {
    crate::sigpipe::ignore_sigpipe();
    STOP_ON_CLOSED_OUTPUT.store(true, Ordering::SeqCst);
    watch_stderr();
}

/// Wait on a thread of its own until stderr loses its reader, then stop the
/// process. A pipe, a pty or a socket on stderr reports the loss as a hangup.
/// A file or a device never does, and the thread then waits forever or
/// returns.
#[cfg(unix)]
fn watch_stderr() {
    let spawned = std::thread::Builder::new()
        .name("stderr-watch".into())
        .spawn(|| {
            // macOS reports a hangup only when it is asked for. Linux always
            // reports it, and reports a pipe with no reader as an error.
            let mut stderr = libc::pollfd {
                fd: libc::STDERR_FILENO,
                events: libc::POLLHUP,
                revents: 0,
            };
            // SAFETY: one live `pollfd`, and the count is one.
            while unsafe { libc::poll(&raw mut stderr, 1, -1) } < 0 {
                if std::io::Error::last_os_error().kind() != std::io::ErrorKind::Interrupted {
                    return;
                }
            }
            if stderr.revents & (libc::POLLHUP | libc::POLLERR) != 0 {
                output_closed();
            }
        });
    if let Err(err) = spawned {
        tracing::warn!(error = %err, "failed to start the thread that watches stderr for a closed reader");
    }
}

#[cfg(not(unix))]
fn watch_stderr() {}

/// A write to stderr found no reader. If [`stop_on_closed_output`] asked for
/// it, and no teardown has started, this call tears down and exits. Every other
/// call returns.
fn output_closed() {
    // A teardown that logs on the closed stderr comes back here. The flag lets
    // that call return, so the teardown that is in progress finishes.
    if STOP_ON_CLOSED_OUTPUT.load(Ordering::SeqCst) && !DRAINING.swap(true, Ordering::SeqCst) {
        // 141 = 128 + SIGPIPE, the status the closed pipe would have given.
        tear_down_and_exit("output closed", 141);
    }
}

/// Stderr as the writer for a log subscriber (`with_writer(log_stderr)`). A
/// write that finds no reader fails as a write to stderr does. If
/// [`stop_on_closed_output`] asked for it, that write also stops the
/// process.
pub fn log_stderr() -> LogStderr {
    LogStderr(std::io::stderr())
}

/// The writer [`log_stderr`] returns.
pub struct LogStderr(std::io::Stderr);

impl LogStderr {
    fn checked<T>(result: std::io::Result<T>) -> std::io::Result<T> {
        if let Err(err) = &result
            && err.kind() == std::io::ErrorKind::BrokenPipe
        {
            output_closed();
        }
        result
    }
}

impl std::io::Write for LogStderr {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        Self::checked(self.0.write(buf))
    }

    /// One stderr lock for the whole buffer, so another thread's write does
    /// not land inside a log line.
    fn write_all(&mut self, buf: &[u8]) -> std::io::Result<()> {
        Self::checked(self.0.write_all(buf))
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Self::checked(self.0.flush())
    }
}

/// Run every registered teardown once, then exit with `code`.
fn tear_down_and_exit(why: &str, code: i32) -> ! {
    run_and_exit(take_teardowns(), why, code)
}

/// Remove every teardown from the registry so each runs once, even on re-entry.
fn take_teardowns() -> Vec<Entry> {
    DRAINING.store(true, Ordering::SeqCst);
    registry()
        .lock()
        .map(|mut g| std::mem::take(&mut *g))
        .unwrap_or_default()
}

fn run_and_exit(entries: Vec<Entry>, why: &str, code: i32) -> ! {
    if !entries.is_empty() {
        say(&format!(
            "{why}; tearing down {} resource(s) before exit",
            entries.len()
        ));
        for entry in entries {
            (entry.teardown)();
        }
    }
    std::process::exit(code);
}

/// Write a note to stderr, and drop a failed write. Under the `SIG_IGN` that
/// [`stop_on_closed_output`] installs, the teardown after the note then runs
/// on a closed stderr. Under the default `SIGPIPE`, a closed stderr stops the
/// process at this write, before the teardown.
fn say(message: &str) {
    use std::io::Write;

    let _ = writeln!(std::io::stderr(), "\n[pipette] {message}");
}

/// Kill `pid` and its descendants: Windows has no process groups, so
/// `taskkill /T` walks the tree.
#[cfg(windows)]
pub(crate) fn kill_tree(pid: u32) {
    taskkill(&["/F", "/T", "/PID", &pid.to_string()]);
}

/// Its `SUCCESS: …` line would otherwise land on the CLI's stdout, which
/// carries only what the command produces.
#[cfg(windows)]
fn taskkill(args: &[&str]) {
    use std::process::{Command, Stdio};

    let _ = Command::new("taskkill")
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();
}

/// A teardown registration. Drop only deregisters: the teardown runs only on a
/// signal or a closed output.
pub(crate) struct Guard {
    token: TeardownToken,
}

impl Guard {
    pub(crate) fn register<F>(f: F) -> Self
    where
        F: FnOnce() + Send + 'static,
    {
        Self {
            token: register(f, false, false),
        }
    }

    /// As [`Guard::register`], for a teardown that does not hold off a
    /// graceful stop.
    pub(crate) fn register_bounded<F>(f: F) -> Self
    where
        F: FnOnce() + Send + 'static,
    {
        Self {
            token: register(f, true, false),
        }
    }

    /// As [`Guard::register`], for the teardown that stops a running child.
    pub(crate) fn child<F>(f: F) -> Self
    where
        F: FnOnce() + Send + 'static,
    {
        Self {
            token: register(f, false, true),
        }
    }

    /// As [`Guard::child`], for a short child that ends on its own deadline.
    pub(crate) fn child_bounded<F>(f: F) -> Self
    where
        F: FnOnce() + Send + 'static,
    {
        Self {
            token: register(f, true, true),
        }
    }

    #[cfg(test)]
    fn token(&self) -> TeardownToken {
        self.token
    }
}

/// Keeps a [`register_bounded_teardown`] cleanup registered until dropped.
pub struct BoundedTeardownGuard {
    _guard: Guard,
}

/// Register `cleanup` to run when a signal or a closed output exits the
/// process, without holding off a [`register_stop_flag`] graceful stop. Drop
/// only deregisters, so a normal exit runs its own cleanup first.
pub fn register_bounded_teardown<F>(cleanup: F) -> BoundedTeardownGuard
where
    F: FnOnce() + Send + 'static,
{
    BoundedTeardownGuard {
        _guard: Guard::register_bounded(cleanup),
    }
}

impl Drop for Guard {
    fn drop(&mut self) {
        deregister(self.token);
    }
}

/// A teardown for something other than a child, such as a file only the
/// run's own `Drop` would delete: ^C and SIGTERM run it before the exit that
/// skips that `Drop`. Dropping the guard deregisters it without running it.
#[must_use = "dropping the guard deregisters the teardown at once"]
pub struct SignalTeardown(Guard);

impl std::fmt::Debug for SignalTeardown {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("SignalTeardown")
            .field(&self.0.token)
            .finish()
    }
}

/// Register `teardown` to run if ^C or SIGTERM ends the process while the
/// returned guard is held.
pub fn on_signal<F>(teardown: F) -> SignalTeardown
where
    F: FnOnce() + Send + 'static,
{
    SignalTeardown(Guard::register(teardown))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(unix)]
    #[test]
    fn an_interrupt_waits_for_startup_registration() -> anyhow::Result<()> {
        use std::sync::mpsc::{self, RecvTimeoutError};
        use std::time::Duration;

        use anyhow::Context;

        const CHILD: &str = "PIPETTE_TEST_STARTUP_TEARDOWN";
        const TEST: &str = "cleanup::tests::an_interrupt_waits_for_startup_registration";
        if let Some(marker) = std::env::var_os(CHILD) {
            // Bound a broken handshake or a handler that never exits.
            std::thread::spawn(|| {
                std::thread::sleep(Duration::from_secs(10));
                std::process::exit(124);
            });

            let starting = hold_off_teardown();
            let (entered, entering) = mpsc::channel();
            let (drained, draining) = mpsc::channel();
            let _early = Guard::register(move || {
                let _ = drained.send(());
            });
            ctrlc::set_handler(move || {
                let _ = entered.send(());
                handle_signal();
            })?;
            // SAFETY: SIGINT is valid, and the handler is installed above.
            anyhow::ensure!(unsafe { libc::raise(libc::SIGINT) } == 0, "raise failed");
            entering.recv_timeout(Duration::from_secs(5))?;
            assert_eq!(
                draining.recv_timeout(Duration::from_millis(200)),
                Err(RecvTimeoutError::Timeout),
                "the handler drained teardowns before startup finished"
            );

            let _late = Guard::register(move || {
                if std::fs::write(&marker, "torn down").is_err() {
                    std::process::exit(1);
                }
            });
            drop(starting);
            // Keep both registrations alive until the handler exits.
            std::thread::sleep(Duration::from_secs(10));
            anyhow::bail!("the signal handler did not exit");
        }

        let dir = tempfile::tempdir()?;
        let marker = dir.path().join("teardown");
        let output = std::process::Command::new(std::env::current_exe()?)
            .args(["--exact", TEST, "--nocapture", "--test-threads=1"])
            .env(CHILD, &marker)
            .output()?;
        assert_eq!(
            output.status.code(),
            Some(130),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(
            std::fs::read_to_string(marker)
                .context("the handler exited without running the startup teardown")?,
            "torn down"
        );
        Ok(())
    }

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
        let token = register(
            move || {
                f.store(true, std::sync::atomic::Ordering::SeqCst);
            },
            false,
            false,
        );
        assert!(registry_contains(token)?);
        deregister(token);
        assert!(!registry_contains(token)?);
        assert!(!fired.load(std::sync::atomic::Ordering::SeqCst));
        Ok(())
    }

    #[test]
    fn deregister_unknown_is_noop() {
        deregister(TeardownToken(u64::MAX));
    }

    /// On a local list: the process-wide registry holds other tests' children.
    #[test]
    fn stopping_children_leaves_other_teardowns_registered() -> anyhow::Result<()> {
        let stopped = Arc::new(AtomicBool::new(false));
        let kept = Arc::new(AtomicBool::new(false));
        let entry = |flag: &Arc<AtomicBool>, child| {
            let flag = flag.clone();
            Entry {
                token: next_token(),
                teardown: Box::new(move || flag.store(true, Ordering::SeqCst)),
                bounded: false,
                child,
            }
        };
        let mut entries = vec![entry(&kept, false), entry(&stopped, true)];
        let children = take_children(&mut entries);
        anyhow::ensure!(children.len() == 1 && entries.len() == 1);
        children.into_iter().for_each(|e| (e.teardown)());
        anyhow::ensure!(stopped.load(Ordering::SeqCst), "the child's teardown ran");
        anyhow::ensure!(!kept.load(Ordering::SeqCst), "the other teardown did not");
        Ok(())
    }

    /// A closed output stops only a process that asked for it. The armed path
    /// exits, so `tests/closed_output.rs` covers it.
    #[test]
    fn a_closed_output_stops_nothing_unasked() {
        output_closed();
    }

    #[test]
    fn guard_drop_deregisters_token_from_registry() -> anyhow::Result<()> {
        let fired = Arc::new(AtomicBool::new(false));
        let f = fired.clone();
        let token = register(
            move || {
                f.store(true, std::sync::atomic::Ordering::SeqCst);
            },
            false,
            false,
        );
        let guard = Guard { token };
        assert!(registry_contains(guard.token())?);
        drop(guard);
        assert!(!registry_contains(token)?);
        assert!(!fired.load(std::sync::atomic::Ordering::SeqCst));
        Ok(())
    }

    /// A signal teardown is held in the registry while its guard lives and
    /// leaves it, unrun, when the guard drops.
    #[test]
    fn a_signal_teardown_lives_with_its_guard() -> anyhow::Result<()> {
        let fired = Arc::new(AtomicBool::new(false));
        let f = fired.clone();
        let teardown = on_signal(move || f.store(true, Ordering::SeqCst));
        let token = teardown.0.token();
        assert!(registry_contains(token)?);
        drop(teardown);
        assert!(!registry_contains(token)?);
        assert!(!fired.load(Ordering::SeqCst));
        Ok(())
    }

    /// One test, as the flags are process-wide and a parallel test would set
    /// this one's. Drives `request_graceful_stop`, as `handle_signal` exits.
    #[test]
    fn a_stop_flag_absorbs_one_signal_each() {
        assert!(!request_graceful_stop(), "no flag to absorb the signal");

        let stop = Arc::new(AtomicBool::new(false));
        register_stop_flag(&stop);
        assert!(request_graceful_stop(), "the first signal is absorbed");
        assert!(stop.load(Ordering::SeqCst), "and asks the command to stop");

        assert!(
            !request_graceful_stop(),
            "the second signal is not absorbed"
        );

        drop(stop);
        assert!(!request_graceful_stop(), "a dead flag absorbs nothing");
        let live = stop_flags().lock().unwrap_or_else(|p| p.into_inner()).len();
        assert_eq!(live, 0, "and is pruned: {live} left");
    }
}
