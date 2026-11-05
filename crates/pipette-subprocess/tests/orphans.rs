//! A session whose starter was SIGKILLed outlives it until the next start.
#![cfg(any(target_os = "linux", target_os = "android", target_os = "macos"))]

use std::process::{Command, Stdio};
use std::time::Duration;

use pipette_subprocess::start_process;

mod common;

#[test]
fn the_next_start_kills_what_a_killed_run_left() -> anyhow::Result<()> {
    #[expect(clippy::disallowed_methods, reason = "stdout is piped to read the pid")]
    let mut run = Command::new(env!("CARGO_BIN_EXE_start_and_wait"))
        .stdout(Stdio::piped())
        .spawn()?;
    let left: libc::pid_t = common::read_pid(&mut run)?;
    // SIGKILL: neither a drop nor the ^C handler runs.
    run.kill()?;
    run.wait()?;
    anyhow::ensure!(left > 1, "not a child's pid: {left}");

    // The reap runs before the spawn, so it has run even if the spawn failed.
    let next = start_process("true", Command::new("true"), Duration::ZERO);
    if !common::gone(left) {
        // SAFETY: `kill` takes no pointers; `left` leads the group the run made,
        // as it leads its session.
        unsafe { libc::kill(-left, libc::SIGKILL) };
        anyhow::bail!("{left} outlived the next start");
    }
    next?;
    Ok(())
}
