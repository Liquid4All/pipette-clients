//! A local tool run under a deadline, both streams drained on threads of their
//! own so neither pipe can fill and stall it.

use std::io::{self, Read};
use std::path::PathBuf;
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::mpsc;
use std::time::Duration;

use wait_timeout::ChildExt;

/// How long a finished child's output may take to arrive: a helper that left
/// its process group can hold a pipe open.
const DRAIN_GRACE: Duration = Duration::from_secs(5);

#[derive(Debug)]
pub(crate) struct Ran {
    pub(crate) status: ExitStatus,
    pub(crate) stdout: String,
    pub(crate) stderr: String,
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum ToolError {
    #[error("running `{program}`")]
    Run {
        program: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("`{program}` did not finish within {}s and was killed", .deadline.as_secs())]
    TimedOut {
        program: PathBuf,
        deadline: Duration,
        stderr: String,
    },
}

/// Overrides any stdin set on `command` with null.
pub(crate) fn run_bounded(command: &mut Command, deadline: Duration) -> Result<Ran, ToolError> {
    let program = PathBuf::from(command.get_program());
    #[expect(
        clippy::disallowed_methods,
        reason = "both output streams are piped and drained below"
    )]
    let mut child = command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|source| ToolError::Run {
            program: program.clone(),
            source,
        })?;
    let stdout = child.stdout.take().map(drain);
    let stderr = child.stderr.take().map(drain);
    let status = wait_bounded(&mut child, deadline);
    let stdout = collect(stdout);
    let stderr = collect(stderr);
    match status {
        Some(status) => Ok(Ran {
            status,
            stdout,
            stderr,
        }),
        None => Err(ToolError::TimedOut {
            program,
            deadline,
            stderr,
        }),
    }
}

pub(crate) fn wait_bounded(child: &mut Child, deadline: Duration) -> Option<ExitStatus> {
    match child.wait_timeout(deadline) {
        Ok(Some(status)) => Some(status),
        Ok(None) | Err(_) => {
            let _ = child.kill();
            let _ = child.wait();
            None
        }
    }
}

pub(crate) fn drain<R: Read + Send + 'static>(mut pipe: R) -> mpsc::Receiver<String> {
    let (sender, receiver) = mpsc::channel();
    std::thread::spawn(move || {
        let mut buffer = Vec::new();
        // A read error loses the stream, not the call.
        let _ = pipe.read_to_end(&mut buffer);
        let _ = sender.send(String::from_utf8_lossy(&buffer).into_owned());
    });
    receiver
}

/// Empty when the stream is still open after [`DRAIN_GRACE`].
pub(crate) fn collect(stream: Option<mpsc::Receiver<String>>) -> String {
    stream
        .and_then(|receiver| receiver.recv_timeout(DRAIN_GRACE).ok())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use anyhow::Context;

    use super::*;

    #[test]
    fn a_child_past_its_deadline_is_killed() -> anyhow::Result<()> {
        let error = run_bounded(
            Command::new("/bin/sh").args(["-c", "echo held >&2; exec sleep 30"]),
            // Long enough for a loaded shell to print before the kill.
            Duration::from_secs(2),
        )
        .err()
        .context("a child that outlives the deadline must be an error")?;
        let ToolError::TimedOut { stderr, .. } = &error else {
            anyhow::bail!("not a timeout: {error:#}");
        };
        assert_eq!(stderr.trim(), "held");
        assert!(
            error.to_string().contains("did not finish within"),
            "{error}"
        );
        Ok(())
    }

    #[test]
    fn a_prompt_child_returns_both_streams() -> anyhow::Result<()> {
        let ran = run_bounded(
            Command::new("/bin/sh").args(["-c", "printf out; printf err >&2; exit 3"]),
            Duration::from_secs(10),
        )?;
        assert_eq!(ran.status.code(), Some(3));
        assert_eq!(ran.stdout, "out");
        assert_eq!(ran.stderr, "err");
        Ok(())
    }
}
