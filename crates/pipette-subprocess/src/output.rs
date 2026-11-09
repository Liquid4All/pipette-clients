//! A short command run to its exit with both streams kept, such as a program
//! check or a `docker` call; a server is [`crate::start_process`]'s.

use std::io::{self, Read};
use std::process::{Command, Output, Stdio};
use std::sync::mpsc::{self, Receiver};
use std::sync::{Arc, Mutex, PoisonError};
use std::thread;
use std::time::{Duration, Instant};

use crate::cleanup::{self, Guard};
use crate::orphans::{Record, Token};
use crate::session::Session;

const EXIT_POLL: Duration = Duration::from_millis(20);
/// How long the streams get to close once it is gone: a process that left its
/// session can hold them open.
const DRAIN_LIMIT: Duration = Duration::from_secs(5);

/// Why [`output_within`] has no output.
#[derive(Debug, thiserror::Error)]
pub enum OutputError {
    #[error("could not start it")]
    Spawn(#[source] io::Error),
    #[error("could not wait on it")]
    Wait(#[source] io::Error),
    #[error("it did not exit within {0:?}, so it was killed")]
    TimedOut(Duration),
}

/// Run `cmd` with an empty stdin until it exits, leading a session of its own
/// that is killed once it exits, at `timeout`, on ^C, or at the next start
/// after pipette dies, so nothing it started outlives it. A ^C that a stop flag
/// absorbs leaves it to finish, as its deadline bounds it.
///
/// The timeout starts after spawn and teardown registration. Signal cleanup
/// waits for both without a deadline. On Unix, caller-supplied `pre_exec`
/// callbacks must return promptly; a blocked callback can stall signal cleanup.
pub fn output_within(mut cmd: Command, timeout: Duration) -> Result<Output, OutputError> {
    cmd.stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    #[cfg(unix)]
    Session::lead(&mut cmd);
    let token = Token::mark(&mut cmd);
    let starting = cleanup::hold_off_teardown();
    let mut child = cmd.spawn().map_err(OutputError::Spawn)?;
    #[cfg(windows)]
    crate::job::assign(&child);
    let session = Session::led_by(child.id());
    let _record = Record::write(child.id(), &cmd.get_program().to_string_lossy(), &token);
    let _teardown = Guard::child_bounded(move || {
        session.kill();
    });
    drop(starting);
    // Read as it runs, so a full pipe cannot stall it.
    let stdout = read_all(child.stdout.take());
    let stderr = read_all(child.stderr.take());
    let deadline = Instant::now() + timeout;
    let status = loop {
        // What is left in its session would hold the pipes open.
        match session.reap(&mut child) {
            Ok(Some(status)) => break Ok(status),
            Ok(None) => {
                let left = deadline.saturating_duration_since(Instant::now());
                if left.is_zero() {
                    break Err(OutputError::TimedOut(timeout));
                }
                thread::sleep(EXIT_POLL.min(left));
            }
            Err(err) => break Err(OutputError::Wait(err)),
        }
    };
    if status.is_err() {
        session.kill();
        let _ = child.kill();
        let _ = child.wait();
    }
    let until = Instant::now() + DRAIN_LIMIT;
    Ok(Output {
        status: status?,
        stdout: stdout.drained(until),
        stderr: stderr.drained(until),
    })
}

/// A stream read on a thread of its own as it arrives.
struct Reader {
    bytes: Arc<Mutex<Vec<u8>>>,
    done: Option<Receiver<()>>,
}

fn read_all(pipe: Option<impl Read + Send + 'static>) -> Reader {
    let bytes = Arc::new(Mutex::new(Vec::new()));
    let Some(mut pipe) = pipe else {
        return Reader { bytes, done: None };
    };
    let (finished, done) = mpsc::channel();
    let kept = Arc::clone(&bytes);
    thread::spawn(move || {
        let mut chunk = [0u8; 8192];
        while let Ok(read @ 1..) = pipe.read(&mut chunk) {
            if let Some(read) = chunk.get(..read) {
                kept.lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .extend_from_slice(read);
            }
        }
        let _ = finished.send(());
    });
    Reader {
        bytes,
        done: Some(done),
    }
}

impl Reader {
    /// What arrived once the stream closes, or by `until`, leaving the thread
    /// to whatever still holds it.
    fn drained(self, until: Instant) -> Vec<u8> {
        if let Some(done) = self.done {
            let _ = done.recv_timeout(until.saturating_duration_since(Instant::now()));
        }
        std::mem::take(&mut *self.bytes.lock().unwrap_or_else(PoisonError::into_inner))
    }
}

#[cfg(all(test, unix))]
mod tests {
    use std::os::unix::process::ExitStatusExt;

    use super::*;

    fn sh(script: &str) -> Command {
        let mut cmd = Command::new("sh");
        cmd.args(["-c", script]);
        cmd
    }

    #[test]
    fn it_keeps_both_streams_and_the_status() -> anyhow::Result<()> {
        let output = output_within(
            sh("echo out; echo err >&2; exit 3"),
            Duration::from_secs(10),
        )?;
        assert_eq!(output.status.code(), Some(3));
        assert_eq!(output.stdout, b"out\n");
        assert_eq!(output.stderr, b"err\n");
        Ok(())
    }

    /// More than a pipe holds, which a read after the exit would deadlock on.
    #[test]
    fn a_long_output_does_not_stall_it() -> anyhow::Result<()> {
        let output = output_within(sh("head -c 1000000 /dev/zero"), Duration::from_secs(10))?;
        assert!(output.status.success());
        assert_eq!(output.stdout.len(), 1_000_000);
        Ok(())
    }

    /// At the deadline it is killed with what it started, which holds the
    /// pipes, so the wait ends then.
    #[cfg(any(target_os = "linux", target_os = "android", target_os = "macos"))]
    #[test]
    fn it_is_killed_at_the_deadline_with_what_it_started() -> anyhow::Result<()> {
        use anyhow::Context;

        use crate::session::test_support::ends_within;

        let dir = tempfile::tempdir()?;
        let pid_file = dir.path().join("left");
        let mut cmd = sh(r#"sleep 300 & echo $! > "$0"; wait"#);
        cmd.arg(&pid_file);
        let started = Instant::now();
        let result = output_within(cmd, Duration::from_millis(500));
        assert!(
            matches!(result, Err(OutputError::TimedOut(_))),
            "{result:?}"
        );
        assert!(started.elapsed() < Duration::from_secs(5));
        let left: u32 = std::fs::read_to_string(&pid_file)?
            .trim()
            .parse()
            .context("no pid written")?;
        assert!(
            ends_within(left, Duration::from_secs(5)),
            "pid {left} outlived it"
        );
        Ok(())
    }

    /// What it left running after its exit is killed, so the wait does not
    /// hang on the pipes it holds.
    #[cfg(any(target_os = "linux", target_os = "android", target_os = "macos"))]
    #[test]
    fn what_it_leaves_after_its_exit_is_killed() -> anyhow::Result<()> {
        use anyhow::Context;

        use crate::session::test_support::{ends_within, require_perl};

        require_perl()?;
        let started = Instant::now();
        let output = output_within(
            sh("perl -e 'setpgrp; exec @ARGV' sleep 20 & echo $!"),
            Duration::from_secs(30),
        )?;
        assert!(output.status.success());
        assert!(started.elapsed() < Duration::from_secs(5));
        let left: u32 = String::from_utf8_lossy(&output.stdout)
            .trim()
            .parse()
            .context("no pid printed")?;
        assert!(
            ends_within(left, Duration::from_secs(5)),
            "pid {left} outlived it"
        );
        Ok(())
    }

    /// A process that left its session and holds the pipes past the exit
    /// cannot hold the call: what arrived comes back once the streams have had
    /// their time.
    #[cfg(any(target_os = "linux", target_os = "android", target_os = "macos"))]
    #[test]
    fn a_process_outside_its_session_cannot_hold_it() -> anyhow::Result<()> {
        use anyhow::Context;

        crate::session::test_support::require_perl()?;
        let started = Instant::now();
        let output = output_within(
            sh("perl -e 'use POSIX; POSIX::setsid(); exec @ARGV' sleep 30 & echo $!"),
            Duration::from_secs(60),
        )?;
        assert!(started.elapsed() < DRAIN_LIMIT + Duration::from_secs(5));
        let left: libc::pid_t = String::from_utf8_lossy(&output.stdout)
            .trim()
            .parse()
            .context("no pid printed")?;
        // SAFETY: no pointers; `left` is the `sleep` this test started.
        unsafe { libc::kill(left, libc::SIGKILL) };
        Ok(())
    }

    #[test]
    fn a_program_that_cannot_start_says_so() {
        let result = output_within(Command::new("/no/such/program"), Duration::from_secs(1));
        assert!(matches!(result, Err(OutputError::Spawn(_))), "{result:?}");
    }

    #[test]
    fn a_signalled_exit_is_its_status() -> anyhow::Result<()> {
        let output = output_within(sh("kill -KILL $$"), Duration::from_secs(10))?;
        assert_eq!(output.status.signal(), Some(libc::SIGKILL));
        Ok(())
    }
}
