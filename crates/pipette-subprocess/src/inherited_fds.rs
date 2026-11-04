//! Descriptors a parent left open across `exec`.
//!
//! On Android, adbd registers its `_adb-tls-connect._tcp` mDNS service through
//! libdns_sd, whose socket to mdnsd has no `O_CLOEXEC`, so every process adbd
//! forks inherits a copy and keeps it through `exec`. mdnsd withdraws the
//! registration only when the last copy closes. A worker holding one keeps the
//! handset's old record alive after wireless debugging restarts, and adbd's
//! next registration is renamed `<name> (2)`, a name the host does not look for.

use std::os::fd::RawFd;

/// Close every descriptor from 3 up that lacks `FD_CLOEXEC`.
///
/// Nothing this process opens lacks the flag (std sets it on every descriptor,
/// bionic on the few it keeps), so one without it was inherited. That includes
/// one a launcher passes on purpose, which can only survive `exec` without the
/// flag. One that carries it is this process's own, or closes at the next
/// `exec` anyway, and stays open.
///
/// # Errors
///
/// Listing `/proc/self/fd` failed; nothing was closed.
///
/// # Safety
///
/// Nothing else in the process may own a descriptor without `FD_CLOEXEC`, or
/// open one while this runs: call it at the top of `main`, before any thread
/// starts.
pub unsafe fn close_inherited_fds() -> std::io::Result<()> {
    std::fs::read_dir("/proc/self/fd")?
        .map(|entry| entry.map(|entry| entry.file_name()))
        // Listed in full before the first close, so the listing's own
        // descriptor is shut by then rather than closed under the iterator;
        // `fcntl` answers `EBADF` for it.
        .collect::<std::io::Result<Vec<_>>>()?
        .into_iter()
        .filter_map(|name| name.to_str()?.parse::<RawFd>().ok())
        .filter(|&fd| fd > 2)
        .filter(|&fd| {
            // SAFETY: reads a descriptor's flags; no memory is passed.
            let flags = unsafe { libc::fcntl(fd, libc::F_GETFD) };
            flags != -1 && flags & libc::FD_CLOEXEC == 0
        })
        .for_each(|fd| {
            // SAFETY: the caller guarantees nothing in the process owns a
            // descriptor without `FD_CLOEXEC`. A failed close leaves nothing to
            // retry: Linux frees the number even on `EINTR`.
            unsafe { libc::close(fd) };
        });
    Ok(())
}
