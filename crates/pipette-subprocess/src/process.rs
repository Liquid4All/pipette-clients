//! A program run to its exit, or serving requests until it is stopped.

use std::process::{Command, ExitStatus};
use std::sync::Arc;
use std::time::Duration;

use pipette_plan_types::RuntimeOutput;

use crate::error::{ExitError, NotReadyError, ReadyError, StartError, StatusError, Stopped};
use crate::spawned::{Spawned, Stop, Stream};

/// A program started by [`start_process`]: one that runs to its exit, such as
/// `llama-bench`, or one that serves requests until it is stopped, such as
/// `llama-server`. Dropping it stops it.
pub struct Process {
    spawned: Spawned,
}

/// Spawn `command` leading a session of its own, with an empty stdin and both
/// output streams kept, so a program that talks over stdin and stdout needs a
/// plain `Command` instead. Its environment is the command's:
/// [`apply_client_config`](crate::apply_client_config), then
/// [`apply_envs`](crate::apply_envs). Returns while it runs.
///
/// `name` is what logs and errors call it: `mlx_lm.server` rather than the
/// `python` running it. `stop_grace` is how long [`Process::stop`] waits after
/// SIGTERM before SIGKILL; zero kills at once. Each line it prints goes to
/// [`log_line`](crate::log_line) until [`Process::on_line`] names another
/// callback.
///
/// Whatever it starts goes with it when it is stopped or exits, a daemon
/// included.
///
/// First it kills the sessions that pipette processes left running when they
/// died (Linux, Android and macOS).
///
/// Signal cleanup waits for spawn and teardown registration without a deadline.
/// On Unix, caller-supplied `pre_exec` callbacks must return promptly; a blocked
/// callback can stall signal cleanup.
pub fn start_process(
    name: &str,
    command: Command,
    stop_grace: Duration,
) -> Result<Process, StartError> {
    let stop = if stop_grace.is_zero() {
        Stop::Kill
    } else {
        Stop::Terminate(stop_grace)
    };
    Ok(Process {
        spawned: Spawned::start(command, name.to_owned(), stop)?,
    })
}

impl Process {
    /// The pid it runs, or ran, as.
    pub fn pid(&self) -> u32 {
        self.spawned.pid()
    }

    /// The program and its arguments, without the environment.
    pub fn command(&self) -> &[String] {
        self.spawned.command()
    }

    pub fn is_running(&mut self) -> bool {
        self.spawned.is_running()
    }

    /// Wait up to `timeout` for it to exit: `Some(status)` if it did, `None`
    /// if it is still running. Never stops it.
    pub fn wait(&mut self, timeout: Duration) -> Result<Option<ExitStatus>, StatusError> {
        self.spawned.wait(timeout)
    }

    /// Wait up to `timeout` for it to exit with success. Otherwise it is
    /// stopped, and the error says how it ended and quotes its output.
    pub fn wait_success(&mut self, timeout: Duration) -> Result<(), ExitError> {
        self.spawned.wait_success(timeout)
    }

    /// [`Self::wait_success`], calling `before_reap` once it has exited, or
    /// `timeout` has passed, and before it is reaped: its pid still names it,
    /// so a memory sampler stopped there takes its last read from the exited
    /// process rather than missing it. `before_reap`'s value on success. A
    /// process already seen to exit ([`Self::is_running`], [`Self::wait`],
    /// [`Self::wait_ready`]) has been reaped, so `before_reap` runs after it.
    pub fn wait_success_before_reap<T>(
        &mut self,
        timeout: Duration,
        before_reap: impl FnOnce() -> T,
    ) -> Result<T, ExitError> {
        self.spawned.wait_success_before_reap(timeout, before_reap)
    }

    /// Ask `ready` every 200 ms about what it has printed so far until it
    /// returns `Ok`. Fails once [`NotReadyError::Failed`] is the answer, it exits, or
    /// `timeout` has passed, stopping it; the error quotes the end of its
    /// output and keeps the check's last error as its cause.
    pub fn wait_ready(
        &mut self,
        timeout: Duration,
        mut ready: impl FnMut(&RuntimeOutput) -> Result<(), NotReadyError>,
    ) -> Result<(), ReadyError> {
        self.spawned.wait_ready(timeout, &mut ready)
    }

    /// The newest 256 KiB of each stream: what it has printed so far while it
    /// runs, and all of that once it has exited or been stopped.
    pub fn output(&self) -> RuntimeOutput {
        self.spawned.output()
    }

    /// Wait up to `timeout` for `found` to hold on what `stream` has printed
    /// (its newest 256 KiB, as `output` keeps): that text, or `None` once the
    /// timeout passes or the stream closes without it.
    pub fn wait_for_output(
        &self,
        stream: Stream,
        timeout: Duration,
        found: impl Fn(&str) -> bool,
    ) -> Option<String> {
        self.spawned.wait_for_output(stream, timeout, &found)
    }

    /// Hand each line it prints to `on_line` instead of
    /// [`log_line`](crate::log_line): the lines kept so far now, stdout first,
    /// then each new one on its stream's reader thread, all inside its span.
    pub fn on_line(&self, on_line: impl Fn(Stream, &str) + Send + Sync + 'static) {
        self.spawned.on_line(Arc::new(on_line));
    }

    /// SIGTERM its process group and SIGKILL its session once the stop grace
    /// has passed; with no grace, SIGKILL at once. Windows kills the process
    /// tree at once. Nothing to do once it has exited.
    pub fn stop(&mut self) {
        self.spawned.stop();
    }

    /// [`Self::stop`] it over a failure, and say whether it had exited and the
    /// end of its output, which otherwise logs only at debug. Put it on the
    /// failure as context, `err.context(process.stop_with_tail())`, which keeps
    /// `err`'s chain.
    pub fn stop_with_tail(&mut self) -> Stopped {
        self.spawned.stop_with_tail()
    }
}

/// Unix-only: the programs are shell one-liners.
#[cfg(all(test, unix))]
mod tests {
    use std::cell::Cell;
    use std::os::unix::process::ExitStatusExt;
    use std::path::Path;
    use std::sync::{Mutex, PoisonError};
    use std::thread;
    use std::time::Instant;

    use anyhow::Context;

    use super::*;
    use crate::spawned::OUTPUT_CAP;

    const LONG: Duration = Duration::from_secs(10);

    fn sh(script: &str) -> Command {
        let mut command = Command::new("sh");
        command.arg("-c").arg(script);
        command
    }

    fn not_yet(_: &RuntimeOutput) -> Result<(), NotReadyError> {
        Err(NotReadyError::yet("not listening yet"))
    }

    fn touched(path: &Path) -> Result<(), NotReadyError> {
        path.exists()
            .then_some(())
            .ok_or_else(|| NotReadyError::yet("not touched yet"))
    }

    /// Whether `pid` is gone, giving a reparented descendant a moment to be
    /// reaped.
    fn gone(pid: u32) -> anyhow::Result<bool> {
        let pid = libc::pid_t::try_from(pid)?;
        let started = Instant::now();
        while started.elapsed() < Duration::from_secs(5) {
            // SAFETY: signal 0 only asks whether the pid exists.
            if unsafe { libc::kill(pid, 0) } != 0 {
                return Ok(true);
            }
            thread::sleep(Duration::from_millis(20));
        }
        Ok(false)
    }

    #[test]
    fn an_exit_reports_its_status_and_what_it_printed() -> anyhow::Result<()> {
        let mut process =
            start_process("sh", sh("echo out; echo err >&2; exit 3"), Duration::ZERO)?;
        let status = process
            .wait(Duration::from_secs(10))?
            .context("still running")?;
        assert_eq!(status.code(), Some(3));
        assert!(!process.is_running());
        let output = process.output();
        assert_eq!(output.runtime_stdout.as_deref(), Some("out\n"));
        assert_eq!(output.runtime_stderr.as_deref(), Some("err\n"));
        Ok(())
    }

    /// A line that is not UTF-8 neither ends the capture nor breaks the pipe
    /// the program goes on writing to.
    #[test]
    fn output_after_a_line_that_is_not_utf8_is_kept() -> anyhow::Result<()> {
        let mut process = start_process("sh", sh(r"printf 'a\n\377\nb\n' >&2"), Duration::ZERO)?;
        let status = process
            .wait(Duration::from_secs(10))?
            .context("still running")?;
        assert!(status.success(), "{status}");
        assert_eq!(
            process.output().runtime_stderr.as_deref(),
            Some("a\n\u{fffd}\nb\n")
        );
        Ok(())
    }

    #[test]
    fn a_wait_that_runs_out_leaves_it_running() -> anyhow::Result<()> {
        let mut process = start_process("sh", sh("exec sleep 30"), Duration::ZERO)?;
        assert_eq!(process.wait(Duration::from_millis(100))?, None);
        assert!(process.is_running());
        process.stop();
        assert!(!process.is_running());
        Ok(())
    }

    #[test]
    fn wait_success_returns_once_it_exits_with_success() -> anyhow::Result<()> {
        let mut process = start_process("sh", sh("echo done"), Duration::ZERO)?;
        process.wait_success(LONG)?;
        assert_eq!(process.output().runtime_stdout.as_deref(), Some("done\n"));
        Ok(())
    }

    #[rstest::rstest]
    #[case::failed("exit 3", LONG, "exited with exit status: 3")]
    #[case::past_the_timeout(
        "exec sleep 30",
        Duration::from_millis(300),
        "did not exit within 300ms"
    )]
    fn wait_success_fails_saying_how_it_ended(
        #[case] then: &str,
        #[case] timeout: Duration,
        #[case] ended: &str,
    ) -> anyhow::Result<()> {
        let script = format!("echo 'unknown argument: --bogus' >&2; {then}");
        let mut process = start_process("llama-bench", sh(&script), Duration::ZERO)?;
        process
            .wait_for_output(Stream::Stderr, LONG, |text| text.contains("--bogus"))
            .context("the line never arrived")?;
        let Err(err) = process.wait_success(timeout) else {
            anyhow::bail!("expected the wait to fail");
        };
        assert_eq!(
            format!("{err:#}"),
            format!("llama-bench {ended}\nstderr tail:\nunknown argument: --bogus")
        );
        let pid = process.pid();
        assert!(gone(pid)?, "{pid} outlived the failed wait");
        Ok(())
    }

    /// Whether `pid` has exited and is still waiting to be reaped.
    fn unreaped(pid: u32) -> bool {
        let mut info = std::mem::MaybeUninit::<libc::siginfo_t>::zeroed();
        // SAFETY: `info` is a live, writable `siginfo_t`, and `WNOWAIT` leaves
        // the child to its owner.
        let rc = unsafe {
            libc::waitid(
                libc::P_PID,
                pid,
                info.as_mut_ptr(),
                libc::WEXITED | libc::WNOWAIT | libc::WNOHANG,
            )
        };
        // SAFETY: it started zeroed, a valid `siginfo_t`.
        rc == 0 && unsafe { info.assume_init() }.si_signo == libc::SIGCHLD
    }

    #[test]
    fn before_reap_runs_on_the_exited_process_before_it_is_reaped() -> anyhow::Result<()> {
        let mut process = start_process("sh", sh("echo done"), Duration::ZERO)?;
        let pid = process.pid();
        assert!(process.wait_success_before_reap(LONG, || unreaped(pid))?);
        assert!(gone(pid)?, "{pid} was not reaped after");
        assert_eq!(process.output().runtime_stdout.as_deref(), Some("done\n"));
        Ok(())
    }

    #[rstest::rstest]
    #[case::failed("exit 3", LONG, "sh exited with exit status: 3")]
    #[case::past_the_timeout("exec sleep 30", Duration::from_millis(300), "sh did not exit")]
    fn before_reap_fails_as_wait_success_does(
        #[case] script: &str,
        #[case] timeout: Duration,
        #[case] ended: &str,
    ) -> anyhow::Result<()> {
        let mut process = start_process("sh", sh(script), Duration::ZERO)?;
        let called = Cell::new(false);
        let Err(err) = process.wait_success_before_reap(timeout, || called.set(true)) else {
            anyhow::bail!("expected the wait to fail");
        };
        assert!(format!("{err:#}").starts_with(ended), "{err:#}");
        assert!(called.get(), "before_reap was skipped");
        let pid = process.pid();
        assert!(gone(pid)?, "{pid} outlived the failed wait");
        Ok(())
    }

    #[test]
    fn stop_kills_what_it_started() -> anyhow::Result<()> {
        let mut process = start_process("sh", sh("sleep 30 & echo $!; wait"), Duration::ZERO)?;
        let child: u32 = process
            .wait_for_output(Stream::Stdout, LONG, |text| {
                text.trim().parse::<u32>().is_ok()
            })
            .context("no pid printed")?
            .trim()
            .parse()?;
        process.stop();
        assert!(gone(child)?, "{child} outlived the stop");
        Ok(())
    }

    #[test]
    fn a_wait_for_output_returns_once_the_line_arrives() -> anyhow::Result<()> {
        let process = start_process(
            "sh",
            sh("sleep 0.2; echo up; exec sleep 30"),
            Duration::ZERO,
        )?;
        let started = Instant::now();
        let printed = process
            .wait_for_output(Stream::Stdout, LONG, |text| text.contains("up"))
            .context("the line never arrived")?;
        assert_eq!(printed, "up\n");
        assert!(started.elapsed() < LONG, "waited out the timeout");
        Ok(())
    }

    /// Each stream is waited on alone, and one that closed without the text
    /// ends the wait rather than the timeout.
    #[test]
    fn a_wait_for_output_ends_when_its_stream_closes() -> anyhow::Result<()> {
        let mut process = start_process("sh", sh("echo up >&2"), Duration::ZERO)?;
        let started = Instant::now();
        let seen = process.wait_for_output(Stream::Stdout, LONG, |text| text.contains("up"));
        assert_eq!(seen, None);
        assert!(started.elapsed() < LONG, "waited on a closed stream");
        process.wait(LONG)?.context("still running")?;
        assert_eq!(
            process
                .wait_for_output(Stream::Stderr, LONG, |text| text.contains("up"))
                .as_deref(),
            Some("up\n")
        );
        Ok(())
    }

    /// Set after the start, a callback gets the lines printed before it
    /// (replayed) and after it, each once, with its stream.
    #[test]
    fn a_callback_set_after_the_start_gets_every_line_once() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let gate = dir.path().join("gate");
        let script = format!(
            "echo early; while [ ! -e '{}' ]; do sleep 0.05; done; echo late; echo err >&2",
            gate.display()
        );
        let mut process = start_process("sh", sh(&script), Duration::ZERO)?;
        process
            .wait_for_output(Stream::Stdout, LONG, |text| text.contains("early"))
            .context("the first line never arrived")?;
        let seen = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&seen);
        process.on_line(move |stream, line| {
            sink.lock()
                .unwrap_or_else(PoisonError::into_inner)
                .push(format!("{stream}: {line}"));
        });
        std::fs::write(&gate, "")?;
        process.wait(LONG)?.context("still running")?;
        let mut seen = seen.lock().unwrap_or_else(PoisonError::into_inner).clone();
        seen.sort();
        assert_eq!(seen, ["stderr: err", "stdout: early", "stdout: late"]);
        Ok(())
    }

    #[test]
    fn a_wait_for_output_gives_up_at_the_timeout() -> anyhow::Result<()> {
        let process = start_process("sh", sh("exec sleep 30"), Duration::ZERO)?;
        let timeout = Duration::from_millis(200);
        let started = Instant::now();
        let seen = process.wait_for_output(Stream::Stdout, timeout, |text| !text.is_empty());
        assert_eq!(seen, None);
        assert!(started.elapsed() >= timeout);
        Ok(())
    }

    #[test]
    fn dropping_it_stops_it() -> anyhow::Result<()> {
        let process = start_process("sh", sh("exec sleep 30"), Duration::ZERO)?;
        let pid = process.pid();
        drop(process);
        assert!(gone(pid)?, "{pid} outlived the drop");
        Ok(())
    }

    /// Left running, the child would also hold the output open until the drain
    /// gave up on it.
    #[test]
    fn an_exit_takes_what_it_left_running_with_it() -> anyhow::Result<()> {
        let mut process = start_process("sh", sh("sleep 30 & echo $!; exit 0"), Duration::ZERO)?;
        let started = Instant::now();
        let status = process.wait(LONG)?.context("still running")?;
        assert!(status.success(), "{status}");
        assert!(
            started.elapsed() < Duration::from_secs(2),
            "the output stayed open"
        );
        let left: u32 = process
            .output()
            .runtime_stdout
            .unwrap_or_default()
            .trim()
            .parse()?;
        assert!(gone(left)?, "{left} outlived its parent");
        Ok(())
    }

    #[test]
    fn a_chatty_process_keeps_its_newest_whole_lines() -> anyhow::Result<()> {
        let mut process = start_process("sh", sh("seq 1 300000"), Duration::ZERO)?;
        process.wait(LONG)?.context("still running")?;
        let stdout = process.output().runtime_stdout.unwrap_or_default();
        assert!(stdout.len() <= OUTPUT_CAP, "kept {} bytes", stdout.len());
        assert!(stdout.ends_with("\n300000\n"), "the newest line is missing");
        let mut lines = stdout.lines().map(str::parse::<u32>);
        let first = lines.next().context("nothing kept")??;
        let second = lines.next().context("one line kept")??;
        assert_eq!(second, first + 1, "the first line kept was cut");
        Ok(())
    }

    #[test]
    fn a_carriage_return_before_the_newline_is_dropped() -> anyhow::Result<()> {
        let mut process = start_process("sh", sh(r"printf 'a\r\nb\n'"), Duration::ZERO)?;
        process.wait(LONG)?.context("still running")?;
        assert_eq!(process.output().runtime_stdout.as_deref(), Some("a\nb\n"));
        Ok(())
    }

    /// A program that reads its stdin gets end of file rather than waiting on
    /// the terminal.
    #[test]
    fn its_stdin_is_empty() -> anyhow::Result<()> {
        let mut process = start_process(
            "sh",
            sh("read line; echo \"read exited $?\""),
            Duration::ZERO,
        )?;
        process.wait(LONG)?.context("waited on stdin")?;
        assert_eq!(
            process.output().runtime_stdout.as_deref(),
            Some("read exited 1\n")
        );
        Ok(())
    }

    #[test]
    fn a_missing_program_is_named_in_the_error() -> anyhow::Result<()> {
        let Err(err) = start_process(
            "no-such-server",
            Command::new("/nonexistent/pipette-no-such-program"),
            Duration::ZERO,
        ) else {
            anyhow::bail!("expected the spawn to fail");
        };
        assert!(
            format!("{err:#}")
                .contains("failed to spawn no-such-server: /nonexistent/pipette-no-such-program"),
            "{err:#}"
        );
        Ok(())
    }

    #[test]
    fn wait_ready_returns_once_it_is_ready() -> anyhow::Result<()> {
        let asked = Cell::new(0);
        let ask = || {
            asked.set(asked.get() + 1);
            anyhow::ensure!(asked.get() == 3, "asked {} times", asked.get());
            Ok(())
        };
        let mut service = start_process("sh", sh("exec sleep 30"), Duration::ZERO)?;
        service.wait_ready(LONG, |_| ask().map_err(NotReadyError::yet))?;
        assert_eq!(asked.get(), 3);
        assert!(service.is_running());
        service.stop();
        assert!(!service.is_running());
        Ok(())
    }

    #[test]
    fn a_service_that_exits_before_it_is_ready_fails_at_once_with_its_output() -> anyhow::Result<()>
    {
        let script = "echo loading failed >&2; exit 4";
        let mut service = start_process("mlx_lm.server", sh(script), Duration::ZERO)?;
        assert_eq!(service.command(), ["sh", "-c", script]);
        let started = Instant::now();
        let Err(err @ ReadyError::Exited { .. }) =
            service.wait_ready(Duration::from_secs(60), not_yet)
        else {
            anyhow::bail!("expected the wait to fail on the exit");
        };
        let message = format!("{err:#}");
        assert!(
            message.starts_with("mlx_lm.server exited with"),
            "{message}"
        );
        assert!(message.contains("loading failed"), "{message}");
        assert!(started.elapsed() < LONG, "waited out the timeout");
        Ok(())
    }

    /// A server that prints why it will never answer, and stays up.
    #[test]
    fn a_check_that_fails_on_its_output_stops_it_at_once() -> anyhow::Result<()> {
        let script = "echo 'OSError: [Errno 48] Address already in use' >&2; exec sleep 30";
        let mut service = start_process("mlx_lm.server", sh(script), Duration::ZERO)?;
        let started = Instant::now();
        let Err(err @ ReadyError::Failed { .. }) =
            service.wait_ready(Duration::from_secs(60), |output| {
                let stderr = output.runtime_stderr.as_deref().unwrap_or_default();
                if stderr.contains("Address already in use") {
                    return Err(NotReadyError::failed("its port is taken"));
                }
                not_yet(output)
            })
        else {
            anyhow::bail!("expected the check to fail the wait");
        };
        assert!(started.elapsed() < LONG, "waited out the timeout");
        let message = format!("{:#}", anyhow::Error::new(err));
        assert!(
            message.starts_with("mlx_lm.server failed before it was ready"),
            "{message}"
        );
        assert!(message.contains("[Errno 48]"), "{message}");
        assert!(message.ends_with("check: its port is taken"), "{message}");
        let pid = service.pid();
        assert!(gone(pid)?, "{pid} outlived the failed check");
        Ok(())
    }

    #[test]
    fn a_service_never_ready_is_stopped_at_the_timeout() -> anyhow::Result<()> {
        let mut service = start_process(
            "sh",
            sh("echo still loading >&2; exec sleep 30"),
            Duration::ZERO,
        )?;
        let timeout = Duration::from_millis(300);
        let started = Instant::now();
        let Err(err @ ReadyError::TimedOut { .. }) = service.wait_ready(timeout, not_yet) else {
            anyhow::bail!("expected the wait to time out");
        };
        let err = anyhow::Error::new(err);
        let message = format!("{err:#}");
        assert!(message.contains("was not ready within"), "{message}");
        assert!(message.contains("still loading"), "{message}");
        assert!(
            message.contains("last check: not listening yet"),
            "{message}"
        );
        assert_eq!(err.root_cause().to_string(), "not listening yet");
        assert!(started.elapsed() >= timeout);
        assert!(!service.is_running());
        let pid = service.pid();
        assert!(gone(pid)?, "{pid} outlived the failed wait");
        Ok(())
    }

    #[test]
    fn stop_lets_a_service_exit_within_the_grace() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let armed = dir.path().join("armed");
        let script = format!(
            "trap 'echo terminated; exit 0' TERM; touch '{}'; while :; do sleep 0.05; done",
            armed.display()
        );
        let mut service = start_process("sh", sh(&script), LONG)?;
        service.wait_ready(LONG, |_| touched(&armed))?;
        let started = Instant::now();
        service.stop();
        assert!(started.elapsed() < LONG, "waited out the grace");
        assert_eq!(
            service.output().runtime_stdout.as_deref(),
            Some("terminated\n")
        );
        Ok(())
    }

    #[test]
    fn stop_kills_a_service_once_the_grace_has_passed() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let armed = dir.path().join("armed");
        let script = format!(
            "trap '' TERM; touch '{}'; while :; do sleep 0.05; done",
            armed.display()
        );
        let grace = Duration::from_millis(300);
        let mut service = start_process("sh", sh(&script), grace)?;
        service.wait_ready(LONG, |_| touched(&armed))?;
        let started = Instant::now();
        service.stop();
        assert!(started.elapsed() >= grace, "killed before the grace");
        assert!(!service.is_running());
        Ok(())
    }

    /// The caller's error stays whole, under what the server printed.
    #[rstest::rstest]
    #[case::still_running("exec sleep 30", false, "was still running")]
    #[case::exited("exit 3", true, "exited with exit status: 3")]
    fn a_failure_after_the_start_carries_its_output(
        #[case] then: &str,
        #[case] exits: bool,
        #[case] state: &str,
    ) -> anyhow::Result<()> {
        let script = format!("echo loading >&2; {then}");
        let mut server = start_process("llama-server", sh(&script), Duration::ZERO)?;
        server
            .wait_for_output(Stream::Stderr, LONG, |text| text.contains("loading"))
            .context("the line never arrived")?;
        if exits {
            server.wait(LONG)?.context("still running")?;
        }
        let refused = std::io::Error::from(std::io::ErrorKind::ConnectionRefused);
        let err = anyhow::Error::new(refused)
            .context("warm-up")
            .context(server.stop_with_tail());
        assert!(!server.is_running());
        assert_eq!(
            format!("{err:#}"),
            format!(
                "llama-server {state}\nstderr tail:\nloading\nerror: warm-up: connection refused"
            )
        );
        assert!(err.downcast_ref::<std::io::Error>().is_some(), "{err:?}");
        Ok(())
    }

    #[test]
    fn one_that_died_is_started_again() -> anyhow::Result<()> {
        let start = || start_process("sh", sh("exec sleep 30"), Duration::ZERO);
        let mut process = start()?;
        let first = process.pid();
        let pid = libc::pid_t::try_from(first)?;
        // SAFETY: `kill` takes no pointers; the pid is this test's own child.
        unsafe { libc::kill(pid, libc::SIGKILL) };
        let status = process.wait(LONG)?.context("still running")?;
        assert_eq!(status.signal(), Some(libc::SIGKILL));

        process = start()?;
        assert_ne!(process.pid(), first);
        assert!(process.is_running());
        Ok(())
    }
}
