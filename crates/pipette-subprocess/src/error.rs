//! What the process handles fail with. A caller's own error stays its own: a
//! check's is kept as the cause, and one a program is stopped over gets
//! [`Stopped`] as context.

use std::error::Error;
use std::fmt;
use std::io;
use std::process::ExitStatus;
use std::time::Duration;

/// Why [`start_process`](crate::start_process) or
/// [`start_docker_process`](crate::start_docker_process) failed.
#[derive(Debug, thiserror::Error)]
pub enum StartError {
    #[error("failed to spawn {name}: {program}")]
    Spawn {
        name: String,
        program: String,
        source: io::Error,
    },
    /// Its output could not be read, so it was killed.
    #[error("failed to read the output of {name}")]
    Output { name: String, source: io::Error },
}

/// Why a started program's exit status could not be read. From then on it is
/// lost, and the program is not signalled again.
#[derive(Debug, thiserror::Error)]
pub enum StatusError {
    #[error("failed to check {name}")]
    Check { name: String, source: io::Error },
    #[error("the exit status of {name} is lost")]
    Lost { name: String },
}

/// Why [`Process::wait_success`](crate::Process::wait_success) failed. `tail`
/// is the end of each stream, as the message quotes it.
#[derive(Debug, thiserror::Error)]
pub enum ExitError {
    #[error(transparent)]
    Status(#[from] StatusError),
    #[error("{name} exited with {status}{tail}")]
    Failed {
        name: String,
        status: ExitStatus,
        tail: String,
    },
    /// It was stopped.
    #[error("{name} did not exit within {timeout:?}{tail}")]
    TimedOut {
        name: String,
        timeout: Duration,
        tail: String,
    },
}

/// Why `wait_ready` failed on a [`Process`](crate::Process) or a
/// [`DockerProcess`](crate::DockerProcess). `tail` is the end of each stream,
/// as the message quotes it.
#[derive(Debug, thiserror::Error)]
pub enum ReadyError {
    #[error(transparent)]
    Status(#[from] StatusError),
    #[error("{name} exited with {status} before it was ready{tail}")]
    Exited {
        name: String,
        status: ExitStatus,
        tail: String,
    },
    /// The check answered [`NotReadyError::Failed`] with `source`, and it was
    /// stopped.
    #[error("{name} failed before it was ready{tail}\ncheck")]
    Failed {
        name: String,
        tail: String,
        source: Box<dyn Error + Send + Sync>,
    },
    /// `source` is the check's last answer; it was stopped.
    #[error("{name} was not ready within {timeout:?}{tail}\nlast check")]
    TimedOut {
        name: String,
        timeout: Duration,
        tail: String,
        source: Box<dyn Error + Send + Sync>,
    },
}

/// Why a readiness check did not pass.
#[derive(Debug, thiserror::Error)]
pub enum NotReadyError {
    /// Not yet: it is asked again, and a timeout keeps the last one as its
    /// cause.
    #[error(transparent)]
    Yet(Box<dyn Error + Send + Sync>),
    /// Never will be: the wait stops the program at once and fails with this
    /// as its cause.
    #[error(transparent)]
    Failed(Box<dyn Error + Send + Sync>),
}

/// From any error a check has: `anyhow::Error`, `io::Error`, a message. A bare
/// `anyhow::Error` hides its outermost type from a chain walk; `.context()` it
/// first if that type matters.
impl NotReadyError {
    pub fn yet(cause: impl Into<Box<dyn Error + Send + Sync>>) -> Self {
        Self::Yet(cause.into())
    }

    pub fn failed(cause: impl Into<Box<dyn Error + Send + Sync>>) -> Self {
        Self::Failed(cause.into())
    }
}

/// What [`Process::stop_with_tail`](crate::Process::stop_with_tail) says about
/// a program stopped over a failure: whether it had exited, and the end of its
/// output. It is context, so the failure keeps its type; `{:#}` then reads
/// `llama-server was still running\nstderr tail:\n...\nerror: <err>`.
#[derive(Debug)]
#[must_use = "attach it to the failure: `err.context(process.stop_with_tail())`"]
pub struct Stopped {
    pub(crate) headline: String,
    pub(crate) tail: String,
}

impl fmt::Display for Stopped {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{}\nerror", self.headline, self.tail)
    }
}

/// An error and its causes, joined as anyhow's `{:#}` joins them, for a log.
pub(crate) struct Chain<'a>(pub(crate) &'a (dyn Error + 'static));

impl fmt::Display for Chain<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)?;
        std::iter::successors(self.0.source(), |&err| err.source())
            .try_for_each(|err| write!(f, ": {err}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_chain_reads_like_anyhows_alternate_form() {
        let refused = io::Error::from(io::ErrorKind::ConnectionRefused);
        let err = ReadyError::TimedOut {
            name: "sh".to_owned(),
            timeout: Duration::from_secs(1),
            tail: String::new(),
            source: anyhow::Error::new(refused).context("GET /health").into(),
        };
        let chain = Chain(&err).to_string();
        assert_eq!(
            chain,
            "sh was not ready within 1s\nlast check: GET /health: connection refused"
        );
        assert_eq!(chain, format!("{:#}", anyhow::Error::new(err)));
    }
}
