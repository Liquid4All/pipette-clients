//! A run whose stderr reader closes, as when the adb or ssh session of a
//! worker ends. This is a test binary of its own, because the `SIGPIPE`
//! disposition is process-wide.
#![cfg(any(target_os = "linux", target_os = "android", target_os = "macos"))]

use std::os::unix::process::ExitStatusExt;
use std::process::{Command, ExitStatus, Stdio};

use rstest::rstest;

mod common;

/// How the run ends after its stderr reader closes.
#[derive(Debug, Clone, Copy)]
enum Ends {
    /// Exits with this code.
    Code(i32),
    /// Dies by this signal.
    Signal(i32),
}

impl Ends {
    fn matches(self, status: ExitStatus) -> bool {
        match self {
            Self::Code(code) => status.code() == Some(code),
            Self::Signal(signal) => status.signal() == Some(signal),
        }
    }
}

/// An armed run stops its sleep when the pipe closes, also if it writes
/// nothing to the pipe. An unarmed run dies by `SIGPIPE` as any command does.
/// The next start's reap gets the sleep of an unarmed run, and a test beside
/// this one can start at any moment. So the test checks only that the unarmed
/// run dies.
#[rstest]
#[case::armed("armed", Ends::Code(141))]
#[case::unarmed("unarmed", Ends::Signal(libc::SIGPIPE))]
#[case::quiet("quiet", Ends::Code(141))]
fn a_closed_stderr_stops_what_an_armed_run_started(
    #[case] mode: &str,
    #[case] ends: Ends,
) -> anyhow::Result<()> {
    #[expect(clippy::disallowed_methods, reason = "stdout is piped to read the pid")]
    let mut run = Command::new(env!("CARGO_BIN_EXE_log_and_wait"))
        .arg(mode)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    let left: libc::pid_t = common::read_pid(&mut run)?;
    anyhow::ensure!(left > 1, "not a child's pid: {left}");

    drop(run.stderr.take());
    let status = run.wait()?;
    // A run that exits by a code ran its teardown.
    let stops = matches!(ends, Ends::Code(_));
    let stopped = stops && common::gone(left);
    if !stopped {
        // SAFETY: `kill` takes no pointers; `left` leads the group the run made.
        unsafe { libc::kill(-left, libc::SIGKILL) };
    }
    anyhow::ensure!(ends.matches(status), "{status}, expected {ends:?}");
    anyhow::ensure!(
        stopped == stops,
        "whether the run stopped {left}: got {stopped}, expected {stops}"
    );
    Ok(())
}
