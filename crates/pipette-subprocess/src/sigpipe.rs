//! Restore the default `SIGPIPE` disposition for CLI binaries.
//!
//! Rust installs `SIG_IGN` for `SIGPIPE` at startup so a closed pipe
//! surfaces as a `BrokenPipe` `io::Error` instead of killing the
//! process. Convenient for libraries — fatal for CLIs that get piped
//! into `head`, `less`, `awk -n`, etc.: each `println!` after the
//! consumer closes the pipe panics on the broken `write` (default
//! `println!` uses `expect` semantics for the underlying write).
//!
//! Calling [`reset_sigpipe`] once at the very top of `main` (before any
//! `println!` and before any threads are spawned) restores the default
//! disposition (`SIG_DFL`), which terminates the process on
//! `EPIPE` — the POSIX-shell-friendly behavior every CLI expects.
//!
//! No-op on non-unix targets (Windows has no `SIGPIPE`).

#[cfg(unix)]
pub fn reset_sigpipe() {
    // SAFETY: setting a signal disposition to `SIG_DFL` is async-signal-safe
    // and called once before any threads are spawned.
    unsafe {
        libc::signal(libc::SIGPIPE, libc::SIG_DFL);
    }
}

#[cfg(not(unix))]
pub fn reset_sigpipe() {}
