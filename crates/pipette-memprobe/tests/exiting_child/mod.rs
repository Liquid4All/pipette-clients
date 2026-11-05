//! A child that fills a buffer and exits, for a sampler to read before and
//! after its reap. Shared by the unit tests and the integration tests.

use std::io::Write as _;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

/// The buffer the child fills, in bytes.
pub const ALLOCATED: u64 = 64 << 20;

/// A child that, once [`release`]d, fills an [`ALLOCATED`] buffer `blocks`
/// times and exits: within a few ms for one. `exec` keeps its pid.
pub fn command(blocks: u32) -> Command {
    let script =
        format!("read _ && exec dd if=/dev/zero of=/dev/null bs={ALLOCATED} count={blocks}");
    let mut command = Command::new("sh");
    command
        .args(["-c", &script])
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    command
}

#[expect(clippy::disallowed_methods, reason = "its output goes to null")]
pub fn spawn(blocks: u32) -> anyhow::Result<Child> {
    Ok(command(blocks).spawn()?)
}

/// Let it run.
pub fn release(child: &mut Child) -> anyhow::Result<()> {
    let mut stdin = child
        .stdin
        .take()
        .ok_or_else(|| anyhow::anyhow!("no stdin"))?;
    Ok(stdin.write_all(b"\n")?)
}

/// Wait for `pid`, a child of this process, to exit, leaving it unreaped.
pub fn wait_unreaped(pid: u32) -> anyhow::Result<()> {
    let started = Instant::now();
    while started.elapsed() < Duration::from_secs(30) {
        let mut info = std::mem::MaybeUninit::<libc::siginfo_t>::zeroed();
        // SAFETY: `info` is a live, writable `siginfo_t`, and `WNOWAIT`
        // leaves the child to be reaped by its owner.
        let rc = unsafe {
            libc::waitid(
                libc::P_PID,
                pid,
                info.as_mut_ptr(),
                libc::WEXITED | libc::WNOWAIT | libc::WNOHANG,
            )
        };
        anyhow::ensure!(rc == 0, "waitid: {}", std::io::Error::last_os_error());
        // SAFETY: it started zeroed, a valid `siginfo_t`.
        if unsafe { info.assume_init() }.si_signo == libc::SIGCHLD {
            return Ok(());
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    anyhow::bail!("{pid} did not exit")
}
