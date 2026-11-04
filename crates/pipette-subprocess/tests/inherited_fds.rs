//! Closing the descriptors a parent leaked. Its own test binary because the
//! close is process-wide: beside other tests it would close a descriptor one of
//! them opened without `FD_CLOEXEC`.
#![cfg(any(target_os = "linux", target_os = "android"))]

use std::io;
use std::os::fd::{AsRawFd, RawFd};

fn is_open(fd: RawFd) -> bool {
    // SAFETY: reads a descriptor's flags; no memory is passed.
    let flags = unsafe { libc::fcntl(fd, libc::F_GETFD) };
    flags != -1
}

#[test]
fn closes_a_descriptor_without_cloexec_and_keeps_one_with_it() -> anyhow::Result<()> {
    let mut leaked: [RawFd; 2] = [-1; 2];
    // SAFETY: `leaked` has room for the two descriptors `pipe` writes.
    if unsafe { libc::pipe(leaked.as_mut_ptr()) } == -1 {
        return Err(io::Error::last_os_error().into());
    }
    // std opens with `O_CLOEXEC`.
    let kept = std::fs::File::open("/proc/self/stat")?;

    // SAFETY: the binary's one test, and nothing else in it holds a descriptor
    // without `FD_CLOEXEC`.
    unsafe { pipette_subprocess::close_inherited_fds() }?;

    leaked.iter().for_each(|&fd| {
        assert!(!is_open(fd), "fd {fd} has no FD_CLOEXEC and stayed open");
    });
    assert!(
        is_open(kept.as_raw_fd()),
        "the FD_CLOEXEC descriptor was closed"
    );
    Ok(())
}
