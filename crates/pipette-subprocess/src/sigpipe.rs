//! The `SIGPIPE` disposition: restoring the default for CLI binaries, and
//! keeping a write to a child that has gone from raising it.
//!
//! Rust ignores `SIGPIPE` at startup, so once a reader such as `head` closes
//! the pipe, the next `println!` panics. [`reset_sigpipe`], called at the top
//! of `main` before any `println!` or thread, restores `SIG_DFL` so the process
//! ends quietly instead. A no-op on Windows, which has no `SIGPIPE`.
//!
//! `pipette worker` and `pipette benchmarks run` ignore it again
//! ([`crate::stop_on_closed_output`]), so a closed stderr stops what they
//! started before they exit.

#[cfg(unix)]
pub fn reset_sigpipe() {
    // SAFETY: `SIG_DFL` installs no handler, so there is no handler code or
    // state to get wrong; the disposition is process-wide by design.
    unsafe {
        libc::signal(libc::SIGPIPE, libc::SIG_DFL);
    }
}

#[cfg(not(unix))]
pub fn reset_sigpipe() {}

/// Make a write to a closed pipe fail with `EPIPE` in every thread.
#[cfg(unix)]
pub(crate) fn ignore_sigpipe() {
    // SAFETY: `SIG_IGN` installs no handler; the disposition is process-wide
    // by design.
    unsafe {
        libc::signal(libc::SIGPIPE, libc::SIG_IGN);
    }
}

#[cfg(not(unix))]
pub(crate) fn ignore_sigpipe() {}

/// Make a write to `pipe` after its reader has gone fail with `EPIPE` rather
/// than raise the `SIGPIPE` [`reset_sigpipe`] made fatal. Call it only on a
/// thread spawned for the write: on Linux it blocks the signal for the rest of
/// the calling thread's life.
pub fn no_sigpipe_on(pipe: &std::process::ChildStdin) -> std::io::Result<()> {
    imp::no_sigpipe_on(pipe)
}

/// macOS marks the pipe itself: it would hand a blocked signal to the process
/// once the thread exits.
#[cfg(target_vendor = "apple")]
mod imp {
    use std::os::fd::AsRawFd;

    /// `<sys/fcntl.h>`; the libc crate does not export it for Apple.
    const F_SETNOSIGPIPE: libc::c_int = 73;

    pub(super) fn no_sigpipe_on(pipe: &std::process::ChildStdin) -> std::io::Result<()> {
        // SAFETY: sets a flag on a descriptor `pipe` keeps open; no memory is passed.
        if unsafe { libc::fcntl(pipe.as_raw_fd(), F_SETNOSIGPIPE, 1) } == -1 {
            return Err(std::io::Error::last_os_error());
        }
        Ok(())
    }
}

/// Linux sends the signal to the writing thread, so blocking it there is enough.
#[cfg(all(unix, not(target_vendor = "apple")))]
mod imp {
    pub(super) fn no_sigpipe_on(_pipe: &std::process::ChildStdin) -> std::io::Result<()> {
        // SAFETY: `sigset_t` is plain data, and a zeroed one is valid input to `sigemptyset`.
        let mut set: libc::sigset_t = unsafe { std::mem::zeroed() };
        // SAFETY: `set` is a local `sigset_t` the call fills.
        unsafe { libc::sigemptyset(&raw mut set) };
        // SAFETY: `set` is an initialized local `sigset_t` the call updates.
        unsafe { libc::sigaddset(&raw mut set, libc::SIGPIPE) };
        // SAFETY: `set` is initialized and only read; a null old-set pointer is allowed.
        let rc =
            unsafe { libc::pthread_sigmask(libc::SIG_BLOCK, &raw const set, std::ptr::null_mut()) };
        if rc != 0 {
            return Err(std::io::Error::from_raw_os_error(rc));
        }
        Ok(())
    }
}

/// Windows has no `SIGPIPE`.
#[cfg(not(unix))]
mod imp {
    pub(super) fn no_sigpipe_on(_pipe: &std::process::ChildStdin) -> std::io::Result<()> {
        Ok(())
    }
}
