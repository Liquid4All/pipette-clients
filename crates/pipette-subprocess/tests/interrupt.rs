//! What a short command started, in any group, goes with a ^C, and after a
//! SIGKILL at the next reap, even once the command itself is gone.
#![cfg(any(target_os = "linux", target_os = "android", target_os = "macos"))]

use std::process::{Child, Command};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::Context;

mod common;

/// The helper mid-command, and the sleeper its command left in a group of its
/// own.
fn run_mid_command(
    pid_file: &std::path::Path,
    args: &[&str],
) -> anyhow::Result<(Child, libc::pid_t)> {
    #[expect(clippy::disallowed_methods, reason = "the helper prints nothing")]
    let run = Command::new(env!("CARGO_BIN_EXE_output_and_wait"))
        .arg(pid_file)
        .args(args)
        .spawn()?;
    let started = Instant::now();
    loop {
        if let Some(pid) = std::fs::read_to_string(pid_file)
            .ok()
            .and_then(|text| text.trim().parse().ok())
        {
            return Ok((run, pid));
        }
        anyhow::ensure!(
            started.elapsed() < Duration::from_secs(10),
            "the command never started its sleeper"
        );
        thread::sleep(Duration::from_millis(20));
    }
}

/// Return the command's session after the run records it.
/// If SIGKILL stops the run before that record exists, the next reap cannot find it.
fn noted_session(run: &Child, left: libc::pid_t) -> anyhow::Result<libc::pid_t> {
    // SAFETY: `getsid` takes no pointers; `left` is the sleeper the run made.
    let command = unsafe { libc::getsid(left) };
    anyhow::ensure!(command > 1, "no session for {left}");
    // SAFETY: `getuid` takes nothing and cannot fail.
    let uid = unsafe { libc::getuid() };
    let notes = std::env::temp_dir().join(format!("pipette-{uid}-processes"));
    let (owner, pid) = (run.id().to_string(), command.to_string());
    let started = Instant::now();
    // A note is named `<owner>-<start>-<pid>-<start>`.
    while !std::fs::read_dir(&notes)
        .into_iter()
        .flatten()
        .filter_map(|entry| entry.ok())
        .any(|entry| {
            let name = entry.file_name().to_string_lossy().into_owned();
            let mut parts = name.split('-');
            parts.next() == Some(owner.as_str()) && parts.nth(1) == Some(pid.as_str())
        })
    {
        anyhow::ensure!(
            started.elapsed() < Duration::from_secs(10),
            "the run never noted the session of {command}"
        );
        thread::sleep(Duration::from_millis(20));
    }
    Ok(command)
}

fn ensure_gone(left: libc::pid_t, after: &str) -> anyhow::Result<()> {
    if !common::gone(left) {
        // SAFETY: `kill` takes no pointers; `left` is the sleeper the run made.
        unsafe { libc::kill(left, libc::SIGKILL) };
        anyhow::bail!("{left} outlived {after}");
    }
    Ok(())
}

#[test]
fn an_interrupt_kills_what_a_short_command_started() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let (mut run, left) = run_mid_command(&dir.path().join("left"), &[])?;
    let run_pid = libc::pid_t::try_from(run.id()).context("run pid")?;
    // SAFETY: `kill` takes no pointers; `run_pid` is the helper's.
    unsafe { libc::kill(run_pid, libc::SIGINT) };
    let status = run.wait()?;
    assert_eq!(status.code(), Some(130), "{status}");
    ensure_gone(left, "the interrupt")
}

/// A short command ends by its own deadline, so a ^C goes to the stop flag: a
/// readiness probe must not kill the worker.
#[test]
fn a_short_command_leaves_the_stop_to_the_flag() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let (mut run, _) = run_mid_command(&dir.path().join("left"), &["--stop"])?;
    let run_pid = libc::pid_t::try_from(run.id()).context("run pid")?;
    // SAFETY: `kill` takes no pointers; `run_pid` is the helper's.
    unsafe { libc::kill(run_pid, libc::SIGINT) };
    let status = run.wait()?;
    assert_eq!(status.code(), Some(0), "{status}");
    Ok(())
}

/// A SIGKILL runs no handler; the command's note leaves it to the next reap.
#[rstest::rstest]
#[case::live_leader(false)]
#[case::killed_leader(true)]
fn a_killed_runs_short_command_goes_at_the_next_reap(
    #[case] kill_leader: bool,
) -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let (mut run, left) = run_mid_command(&dir.path().join("left"), &[])?;
    let command = noted_session(&run, left)?;
    run.kill()?;
    run.wait()?;
    if kill_leader {
        // A dead leader's pid could name another session. The sleeper's token
        // still identifies the run that owns it.
        // SAFETY: `kill` takes no pointers; `command` is the `sh` the run started.
        unsafe { libc::kill(command, libc::SIGKILL) };
        anyhow::ensure!(
            common::gone(command),
            "the command {command} was not reaped"
        );
    }
    pipette_subprocess::reap_orphans();
    ensure_gone(left, "the next reap")
}
