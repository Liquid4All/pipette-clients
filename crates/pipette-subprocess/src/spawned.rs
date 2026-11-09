//! A child leading its own session, both streams kept, stopped the way it was
//! started: what [`crate::Process`] and [`crate::DockerProcess`] are made of.

use std::fmt;
use std::io::{BufRead, BufReader, Read};
use std::path::PathBuf;
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::thread;
use std::time::{Duration, Instant};

use pipette_plan_types::RuntimeOutput;

use crate::cleanup;
use crate::command::{argv, echo_info};
use crate::error::{Chain, ExitError, NotReadyError, ReadyError, StartError, StatusError, Stopped};
use crate::session::Session;
use crate::{docker_process, orphans};

/// Where [`log_line`] logs each line, at debug, inside the child's
/// `process{program, pid}` or `container{container, pid}` span.
const OUTPUT_TARGET: &str = "pipette_subprocess::output";
/// The newest output kept per stream.
pub(crate) const OUTPUT_CAP: usize = 256 * 1024;
/// How often a wait checks whether the child has exited.
const EXIT_POLL: Duration = Duration::from_millis(50);
/// How often a service is asked whether it is ready.
const READY_POLL: Duration = Duration::from_millis(200);
/// How often a readiness wait says it is still waiting.
const READY_NOTICE: Duration = Duration::from_secs(5);
/// How long an exited child's streams get to reach end of file. Only a
/// descendant that left its session can hold them open.
const DRAIN_LIMIT: Duration = Duration::from_secs(5);
/// The most of its grace a service gets on ^C.
const INTERRUPT_GRACE: Duration = Duration::from_millis(500);
/// Lines of each stream a failure quotes: where a program says why, without
/// its whole log.
const FAILURE_LINES: usize = 40;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stream {
    Stdout,
    Stderr,
}

impl Stream {
    fn name(self) -> &'static str {
        match self {
            Self::Stdout => "stdout",
            Self::Stderr => "stderr",
        }
    }
}

impl fmt::Display for Stream {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Called with each line a started program prints, inside the program's span.
pub(crate) type OnLine = Arc<dyn Fn(Stream, &str) + Send + Sync>;

/// Where a started program's lines go until its `on_line` names another
/// callback: a `tracing` debug event, target `pipette_subprocess::output`, with
/// a `stream` field.
pub fn log_line(stream: Stream, line: &str) {
    tracing::debug!(target: OUTPUT_TARGET, %stream, "{line}");
}

#[derive(Clone)]
pub(crate) enum Stop {
    /// SIGKILL its session.
    Kill,
    /// SIGTERM its process group, SIGKILL its session once the grace has
    /// passed.
    Terminate(Duration),
    /// `docker stop` the container the CLI is attached to, then SIGKILL
    /// whatever is left.
    Docker {
        docker: PathBuf,
        container: String,
        grace: Duration,
    },
}

#[derive(Clone, Copy)]
enum State {
    Running,
    Exited(ExitStatus),
    /// Its status could not be read, so it is not signalled again.
    Lost,
}

pub(crate) struct Spawned {
    child: Child,
    name: String,
    command: Vec<String>,
    stdout: Tail,
    stderr: Tail,
    stop: Stop,
    state: State,
    /// Killed before the streams drain, which its members can hold open.
    session: Session,
    teardown: Option<cleanup::Guard>,
    /// Its session's note, for a later start to reap if this process dies;
    /// dropped with `teardown`. None for a container: docker keeps it running
    /// whatever happens to its CLI.
    record: Option<orphans::Record>,
    /// Entered by the output readers and around a readiness wait and a stop,
    /// so what they log names the child.
    span: tracing::Span,
}

impl Spawned {
    /// `name` is what logs and errors call it.
    pub(crate) fn start(
        mut command: Command,
        name: String,
        stop: Stop,
    ) -> Result<Self, StartError> {
        // Before the new program has to share the machine with them.
        orphans::reap_orphans();
        echo_info(&command);
        let argv = argv(&command);
        command
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        #[cfg(unix)]
        Session::lead(&mut command);
        let token = orphans::Token::mark(&mut command);
        let starting = cleanup::hold_off_teardown();
        let mut child = command.spawn().map_err(|source| StartError::Spawn {
            name: name.clone(),
            program: command.get_program().to_string_lossy().into_owned(),
            source,
        })?;
        #[cfg(windows)]
        crate::job::assign(&child);
        let pid = child.id();
        let session = Session::led_by(pid);
        let teardown = cleanup::Guard::child(on_interrupt(&stop, session));
        let record = match &stop {
            Stop::Kill | Stop::Terminate(_) => orphans::Record::write(pid, &name, &token),
            Stop::Docker { .. } => None,
        };
        drop(starting);
        // Opened here, so it nests under the caller's span (a worker's `job`).
        let span = match &stop {
            Stop::Kill | Stop::Terminate(_) => tracing::info_span!(
                target: OUTPUT_TARGET,
                "process",
                program = %name,
                pid
            ),
            // Not `name`: JSON output lists a span's own name under that key.
            Stop::Docker { container, .. } => tracing::info_span!(
                target: OUTPUT_TARGET,
                "container",
                container = %container,
                pid
            ),
        };
        let tails = match (child.stdout.take(), child.stderr.take()) {
            (Some(stdout), Some(stderr)) => Tail::read(stdout, Stream::Stdout, &span, pid)
                .and_then(|stdout| Ok((stdout, Tail::read(stderr, Stream::Stderr, &span, pid)?))),
            _ => Err(std::io::Error::other("its output was not piped")),
        };
        match tails {
            Ok((stdout, stderr)) => Ok(Self {
                child,
                name,
                command: argv,
                stdout,
                stderr,
                stop,
                state: State::Running,
                session,
                teardown: Some(teardown),
                record,
                span,
            }),
            Err(source) => {
                session.kill();
                let _ = child.kill();
                let _ = child.wait();
                Err(StartError::Output { name, source })
            }
        }
    }

    pub(crate) fn pid(&self) -> u32 {
        self.child.id()
    }

    pub(crate) fn command(&self) -> &[String] {
        &self.command
    }

    /// Hand each line to `on_line` from now on, after replaying the lines
    /// kept so far to it, stdout first.
    pub(crate) fn on_line(&self, on_line: OnLine) {
        let _entered = self.span.enter();
        self.stdout
            .set_on_line(Stream::Stdout, Arc::clone(&on_line));
        self.stderr.set_on_line(Stream::Stderr, on_line);
    }

    pub(crate) fn is_running(&mut self) -> bool {
        match self.poll() {
            Ok(status) => status.is_none(),
            Err(err) => {
                tracing::debug!(
                    error = %Chain(&err),
                    "exit status lost; reporting it as not running"
                );
                false
            }
        }
    }

    pub(crate) fn wait(&mut self, timeout: Duration) -> Result<Option<ExitStatus>, StatusError> {
        let started = Instant::now();
        loop {
            if let Some(status) = self.poll()? {
                return Ok(Some(status));
            }
            let left = timeout.saturating_sub(started.elapsed());
            if left.is_zero() {
                return Ok(None);
            }
            thread::sleep(EXIT_POLL.min(left));
        }
    }

    pub(crate) fn wait_success(&mut self, timeout: Duration) -> Result<(), ExitError> {
        let status = self.wait(timeout)?;
        self.success(status, timeout)
    }

    pub(crate) fn wait_success_before_reap<T>(
        &mut self,
        timeout: Duration,
        before_reap: impl FnOnce() -> T,
    ) -> Result<T, ExitError> {
        self.wait_unreaped(timeout)?;
        let value = before_reap();
        let status = self.poll()?;
        self.success(status, timeout)?;
        Ok(value)
    }

    /// [`Self::wait`], but an exit leaves it a zombie until the next
    /// [`Self::poll`] reaps it.
    #[cfg(unix)]
    fn wait_unreaped(&mut self, timeout: Duration) -> Result<(), StatusError> {
        let started = Instant::now();
        while matches!(self.state, State::Running) && !self.exited_unreaped()? {
            let left = timeout.saturating_sub(started.elapsed());
            if left.is_zero() {
                break;
            }
            thread::sleep(EXIT_POLL.min(left));
        }
        Ok(())
    }

    /// [`Self::wait`]: on Windows the handle `Child` keeps holds the pid
    /// however it is waited on.
    #[cfg(not(unix))]
    fn wait_unreaped(&mut self, timeout: Duration) -> Result<(), StatusError> {
        self.wait(timeout).map(drop)
    }

    /// Whether it has exited, without reaping it.
    #[cfg(unix)]
    fn exited_unreaped(&mut self) -> Result<bool, StatusError> {
        crate::session::exited_unreaped(self.child.id()).map_err(|source| {
            self.lost();
            StatusError::Check {
                name: self.name.clone(),
                source,
            }
        })
    }

    fn success(&mut self, status: Option<ExitStatus>, timeout: Duration) -> Result<(), ExitError> {
        match status {
            Some(status) if status.success() => Ok(()),
            Some(status) => Err(ExitError::Failed {
                name: self.name.clone(),
                status,
                tail: failure_tail(&self.output()),
            }),
            None => {
                self.stop();
                Err(ExitError::TimedOut {
                    name: self.name.clone(),
                    timeout,
                    tail: failure_tail(&self.output()),
                })
            }
        }
    }

    pub(crate) fn output(&self) -> RuntimeOutput {
        RuntimeOutput {
            runtime_stdout: Some(self.stdout.snapshot()),
            runtime_stderr: Some(self.stderr.snapshot()),
        }
    }

    pub(crate) fn wait_for_output(
        &self,
        stream: Stream,
        timeout: Duration,
        found: &dyn Fn(&str) -> bool,
    ) -> Option<String> {
        match stream {
            Stream::Stdout => &self.stdout,
            Stream::Stderr => &self.stderr,
        }
        .wait_for(timeout, found)
    }

    /// Stop it as [`Stop`] says and reap it. Nothing to do once it has exited.
    pub(crate) fn stop(&mut self) {
        if !matches!(self.state, State::Running) {
            return;
        }
        let span = self.span.clone();
        let _entered = span.enter();
        let stopped = match self.stop.clone() {
            Stop::Kill => false,
            Stop::Terminate(grace) => {
                self.session.terminate();
                self.exits_within(grace)
            }
            Stop::Docker {
                docker,
                container,
                grace,
            } => {
                docker_process::stop_container(&docker, &container, grace)
                    && self.exits_within(docker_process::CLI_EXIT)
            }
        };
        if stopped {
            return;
        }
        if let Stop::Docker {
            docker, container, ..
        } = &self.stop
        {
            docker_process::kill_container(docker, container);
        }
        // While it runs, so what it started in a session of its own is still
        // below it.
        self.session.kill();
        let _ = self.child.kill();
        match self.child.wait() {
            Ok(status) => self.finish(State::Exited(status)),
            Err(err) => {
                tracing::warn!(error = %err, "failed to reap");
                self.finish(State::Lost);
            }
        }
    }

    /// Ask `ready` about the output so far until it answers `Ok`. Fails as
    /// soon as the child exits or the check fails, or once `timeout` has
    /// passed, stopping it; the error quotes its output and keeps the check's
    /// error as its cause.
    pub(crate) fn wait_ready(
        &mut self,
        timeout: Duration,
        ready: &mut dyn FnMut(&RuntimeOutput) -> Result<(), NotReadyError>,
    ) -> Result<(), ReadyError> {
        let span = self.span.clone();
        let _entered = span.enter();
        let started = Instant::now();
        let mut noticed = started;
        tracing::info!(?timeout, "waiting until ready");
        loop {
            if let Some(status) = self.poll()? {
                return Err(ReadyError::Exited {
                    name: self.name.clone(),
                    status,
                    tail: failure_tail(&self.output()),
                });
            }
            let not_ready = match ready(&self.output()) {
                Ok(()) => {
                    tracing::info!(elapsed = ?started.elapsed(), "ready");
                    return Ok(());
                }
                Err(NotReadyError::Yet(not_yet)) => not_yet,
                Err(NotReadyError::Failed(source)) => {
                    self.stop();
                    return Err(ReadyError::Failed {
                        name: self.name.clone(),
                        tail: failure_tail(&self.output()),
                        source,
                    });
                }
            };
            let elapsed = started.elapsed();
            if elapsed >= timeout {
                self.stop();
                // A cause, not formatted in: callers walk the chain.
                return Err(ReadyError::TimedOut {
                    name: self.name.clone(),
                    timeout,
                    tail: failure_tail(&self.output()),
                    source: not_ready,
                });
            }
            if noticed.elapsed() >= READY_NOTICE {
                tracing::info!(?elapsed, error = %Chain(&*not_ready), "still not ready");
                noticed = Instant::now();
            }
            thread::sleep(READY_POLL.min(timeout.saturating_sub(elapsed)));
        }
    }

    /// Stop it, and say whether it had exited and how its output ended.
    pub(crate) fn stop_with_tail(&mut self) -> Stopped {
        let headline = match self.poll() {
            Ok(Some(status)) => format!("{} exited with {status}", self.name),
            Ok(None) => format!("{} was still running", self.name),
            Err(lost) => Chain(&lost).to_string(),
        };
        self.stop();
        Stopped {
            headline,
            tail: failure_tail(&self.output()),
        }
    }

    /// The exit status once the child has exited, reaping it; `None` while it
    /// runs.
    fn poll(&mut self) -> Result<Option<ExitStatus>, StatusError> {
        if matches!(self.state, State::Running) {
            match self.session.reap(&mut self.child) {
                Ok(Some(status)) => self.finish(State::Exited(status)),
                Ok(None) => {}
                Err(source) => {
                    self.lost();
                    return Err(StatusError::Check {
                        name: self.name.clone(),
                        source,
                    });
                }
            }
        }
        match self.state {
            State::Running => Ok(None),
            State::Exited(status) => Ok(Some(status)),
            State::Lost => Err(StatusError::Lost {
                name: self.name.clone(),
            }),
        }
    }

    /// Whether it exits within `grace`. A lost status counts as exited: it is
    /// not signalled again.
    fn exits_within(&mut self, grace: Duration) -> bool {
        match self.wait(grace) {
            Ok(status) => status.is_some(),
            Err(err) => {
                tracing::warn!(error = %Chain(&err), "exit status lost; counting it as exited");
                true
            }
        }
    }

    /// Its status could not be read. That may mean a reused pid, which the
    /// kill checks. On Windows an exited pid names no tree.
    fn lost(&mut self) {
        #[cfg(unix)]
        self.session.kill();
        self.finish(State::Lost);
    }

    fn finish(&mut self, state: State) {
        self.state = state;
        self.teardown = None;
        self.record = None;
        let until = Instant::now() + DRAIN_LIMIT;
        self.stdout.drain(until);
        self.stderr.drain(until);
    }
}

impl Drop for Spawned {
    fn drop(&mut self) {
        self.stop();
    }
}

/// What ^C does to a child still running: its stop, without waiting out a
/// long grace.
pub(crate) fn on_interrupt(stop: &Stop, session: Session) -> Box<dyn FnOnce() + Send> {
    match stop {
        Stop::Kill => Box::new(move || {
            session.kill();
        }),
        Stop::Terminate(grace) => {
            let grace = (*grace).min(INTERRUPT_GRACE);
            Box::new(move || {
                session.terminate();
                thread::sleep(grace);
                session.kill();
            })
        }
        Stop::Docker {
            docker, container, ..
        } => {
            let (docker, container) = (docker.clone(), container.clone());
            Box::new(move || {
                docker_process::kill_container(&docker, &container);
                session.kill();
            })
        }
    }
}

/// The last lines of each stream, for a failure message.
fn failure_tail(output: &RuntimeOutput) -> String {
    [
        ("stderr", &output.runtime_stderr),
        ("stdout", &output.runtime_stdout),
    ]
    .into_iter()
    .filter_map(|(stream, text)| {
        let lines = last_lines(text.as_deref()?, FAILURE_LINES);
        (!lines.is_empty()).then(|| format!("\n{stream} tail:\n{lines}"))
    })
    .collect()
}

fn last_lines(text: &str, count: usize) -> String {
    let mut lines: Vec<&str> = text
        .lines()
        .rev()
        .filter(|line| !line.trim().is_empty())
        .take(count)
        .collect();
    lines.reverse();
    lines.join("\n")
}

struct Kept {
    text: String,
    closed: bool,
    on_line: OnLine,
}

/// A stream's [`Kept`] and the signal its reader gives after each line and at
/// the end.
struct Shared {
    kept: Mutex<Kept>,
    changed: Condvar,
}

impl Shared {
    fn lock(&self) -> MutexGuard<'_, Kept> {
        self.kept.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// Marks the stream closed when its reader ends, however it ends.
struct Closing(Arc<Shared>);

impl Drop for Closing {
    fn drop(&mut self) {
        self.0.lock().closed = true;
        self.0.changed.notify_all();
    }
}

/// One stream, read to its end on its own thread: each line handed to its
/// [`OnLine`], the newest [`OUTPUT_CAP`] bytes kept.
struct Tail {
    shared: Arc<Shared>,
}

impl Tail {
    fn read(
        pipe: impl Read + Send + 'static,
        stream: Stream,
        span: &tracing::Span,
        pid: u32,
    ) -> std::io::Result<Self> {
        let shared = Arc::new(Shared {
            kept: Mutex::new(Kept {
                text: String::new(),
                closed: false,
                on_line: Arc::new(log_line),
            }),
            changed: Condvar::new(),
        });
        let closing = Closing(Arc::clone(&shared));
        let span = span.clone();
        thread::Builder::new()
            .name(format!("{pid} {}", stream.name()))
            .spawn(move || {
                let _entered = span.enter();
                let Closing(shared) = &closing;
                // Split on bytes: `lines()` ends at the first line that is not
                // UTF-8, and the child then writes into a closed pipe.
                for line in BufReader::new(pipe).split(b'\n').map_while(Result::ok) {
                    let line = String::from_utf8_lossy(&line);
                    let line = line.strip_suffix('\r').unwrap_or(&line);
                    let mut kept = shared.lock();
                    kept.text.push_str(line);
                    kept.text.push('\n');
                    // Trimmed at twice the cap, so a chatty child does not move
                    // the whole buffer on every line.
                    if kept.text.len() > 2 * OUTPUT_CAP {
                        keep_last(&mut kept.text, OUTPUT_CAP);
                    }
                    // Taken as the line is kept, so a callback set meanwhile
                    // gets it here or in its replay, never both.
                    let on_line = Arc::clone(&kept.on_line);
                    drop(kept);
                    shared.changed.notify_all();
                    on_line(stream, line);
                }
            })?;
        Ok(Self { shared })
    }

    fn snapshot(&self) -> String {
        newest(&self.shared.lock().text).to_owned()
    }

    fn set_on_line(&self, stream: Stream, on_line: OnLine) {
        let mut kept = self.shared.lock();
        newest(&kept.text)
            .lines()
            .for_each(|line| on_line(stream, line));
        kept.on_line = on_line;
    }

    /// The stream's text once `found` holds for it; `None` once `timeout` has
    /// passed or the stream has closed without it.
    fn wait_for(&self, timeout: Duration, found: &dyn Fn(&str) -> bool) -> Option<String> {
        let started = Instant::now();
        let mut kept = self.shared.lock();
        loop {
            let text = newest(&kept.text);
            if found(text) {
                return Some(text.to_owned());
            }
            let left = timeout.saturating_sub(started.elapsed());
            if kept.closed || left.is_zero() {
                return None;
            }
            kept = match self.shared.changed.wait_timeout(kept, left) {
                Ok((kept, _)) => kept,
                Err(poisoned) => poisoned.into_inner().0,
            };
        }
    }

    /// Wait until `until` for the reader to reach end of file.
    fn drain(&self, until: Instant) {
        let kept = self.shared.lock();
        let left = until.saturating_duration_since(Instant::now());
        let (kept, _) = self
            .shared
            .changed
            .wait_timeout_while(kept, left, |kept| !kept.closed)
            .unwrap_or_else(PoisonError::into_inner);
        if !kept.closed {
            tracing::warn!(
                "a stream is still open after its process exited: a process it started holds it"
            );
        }
    }
}

/// The last [`OUTPUT_CAP`] bytes of `text`, as [`keep_last`] would leave them.
fn newest(text: &str) -> &str {
    text.get(tail_start(text, OUTPUT_CAP)..).unwrap_or(text)
}

/// Keep the last `cap` bytes, from the start of a line where that leaves some.
fn keep_last(text: &mut String, cap: usize) {
    let start = tail_start(text, cap);
    text.drain(..start);
}

fn tail_start(text: &str, cap: usize) -> usize {
    if text.len() <= cap {
        return 0;
    }
    let cut = text.ceil_char_boundary(text.len() - cap);
    text.get(cut..)
        .and_then(|rest| rest.find('\n'))
        .map(|i| cut + i + 1)
        .filter(|&start| start < text.len())
        .unwrap_or(cut)
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::under_the_cap("a\nb\n", 8, "a\nb\n")]
    #[case::from_a_line_start("aaaa\nbb\ncc\n", 7, "bb\ncc\n")]
    #[case::a_line_longer_than_the_cap("xxxxxxxxxx\n", 4, "xxx\n")]
    fn keep_last_keeps_the_newest_bytes(
        #[case] text: &str,
        #[case] cap: usize,
        #[case] kept: &str,
    ) {
        let mut text = text.to_owned();
        keep_last(&mut text, cap);
        assert_eq!(text, kept);
    }

    /// Output is not ASCII; a cut inside an `é` moves to the next character.
    #[test]
    fn keep_last_off_a_char_boundary_does_not_panic() {
        let mut text = "é".repeat(8);
        keep_last(&mut text, 5);
        assert_eq!(text, "éé");
    }

    #[test]
    fn a_failure_quotes_the_last_lines_of_each_stream() {
        let output = RuntimeOutput {
            runtime_stdout: Some(String::new()),
            runtime_stderr: Some((1..=50).map(|i| format!("line {i}\n")).collect()),
        };
        let tail = failure_tail(&output);
        assert!(tail.starts_with("\nstderr tail:\nline 11\n"), "{tail}");
        assert!(tail.ends_with("line 50"), "{tail}");
        assert!(!tail.contains("stdout"), "{tail}");
    }

    /// The llama.cpp eval pool's workers share one server.
    #[test]
    fn the_handles_can_move_to_another_thread() {
        fn send<T: Send>() {}
        send::<crate::Process>();
        send::<crate::DockerProcess>();
    }

    /// Unix-only: the children are shell one-liners.
    #[cfg(unix)]
    mod children {
        use std::os::unix::process::ExitStatusExt;
        use std::process::Command;
        use std::time::{Duration, Instant};

        use anyhow::Context;
        use rstest::rstest;

        use super::super::{Spawned, Stop, Stream, on_interrupt};
        use crate::StatusError;

        const LONG: Duration = Duration::from_secs(10);

        fn sh(script: &str) -> Command {
            let mut command = Command::new("sh");
            command.arg("-c").arg(script);
            command
        }

        fn start(script: &str, stop: Stop) -> anyhow::Result<Spawned> {
            Ok(Spawned::start(sh(script), "sh".to_owned(), stop)?)
        }

        /// Once reaped, its pid no longer names its group, so neither ^C nor a
        /// later run's reap may signal it.
        #[test]
        fn it_is_torn_down_and_noted_only_until_it_is_reaped() -> anyhow::Result<()> {
            let mut running = start("exec sleep 30", Stop::Kill)?;
            assert!(running.teardown.is_some());
            assert!(running.record.is_some());
            running.stop();
            assert!(running.teardown.is_none());
            assert!(running.record.is_none());

            let mut exited = start("exit 0", Stop::Kill)?;
            exited.wait(LONG)?.context("still running")?;
            assert!(exited.teardown.is_none());
            assert!(exited.record.is_none());
            Ok(())
        }

        #[test]
        fn one_reaped_behind_its_back_is_not_signalled_again() -> anyhow::Result<()> {
            let mut spawned = start("exit 0", Stop::Kill)?;
            let pid = libc::pid_t::try_from(spawned.pid())?;
            let mut status = 0;
            // SAFETY: `status` is a live local the call writes; `pid` is this
            // test's own child.
            let reaped = unsafe { libc::waitpid(pid, &raw mut status, 0) };
            assert_eq!(reaped, pid);

            assert!(!spawned.is_running());
            assert!(
                matches!(spawned.wait(LONG), Err(StatusError::Lost { .. })),
                "a status nobody read"
            );
            assert!(spawned.teardown.is_none());
            spawned.stop();
            Ok(())
        }

        #[test]
        fn an_interrupt_kills_a_process_at_once() -> anyhow::Result<()> {
            let mut spawned = start("exec sleep 30", Stop::Kill)?;
            on_interrupt(&Stop::Kill, spawned.session)();
            let status = spawned.wait(LONG)?.context("still running")?;
            assert_eq!(status.signal(), Some(libc::SIGKILL));
            Ok(())
        }

        /// On ^C a service gets SIGTERM and at most half a second of its grace.
        #[rstest]
        #[case::it_exits_on_sigterm("trap 'exit 0' TERM", Some(0), None)]
        #[case::it_ignores_sigterm("trap '' TERM", None, Some(libc::SIGKILL))]
        fn an_interrupt_cuts_a_services_grace_short(
            #[case] trap: &str,
            #[case] code: Option<i32>,
            #[case] signal: Option<i32>,
        ) -> anyhow::Result<()> {
            let script = format!("{trap}; echo armed; while :; do sleep 0.05; done");
            let stop = Stop::Terminate(LONG);
            let mut spawned = start(&script, stop.clone())?;
            spawned
                .wait_for_output(Stream::Stdout, LONG, &|text| text.contains("armed"))
                .context("the trap was never armed")?;

            let started = Instant::now();
            on_interrupt(&stop, spawned.session)();
            assert!(started.elapsed() < LONG / 2, "waited out the grace");
            let status = spawned.wait(LONG)?.context("still running")?;
            assert_eq!((status.code(), status.signal()), (code, signal));
            Ok(())
        }

        /// A `sleep` that ignores SIGTERM, in a group of its own as `ninja`
        /// runs a build step, goes with the service however it ends; one in a
        /// session of its own goes with a kill while the service runs. Needs
        /// `perl` on the host.
        #[cfg(any(target_os = "linux", target_os = "android", target_os = "macos"))]
        mod session {
            use super::*;
            use crate::session::test_support::{alone_within, ends_within, runs};

            struct Service {
                spawned: Spawned,
                left: u32,
            }

            #[derive(Debug, Clone, Copy)]
            enum Shape {
                /// Started by the service, which runs until SIGTERM.
                RunsUntilStopped,
                /// Started by a member that stays; the service exits after 1 s.
                ParentStays,
                /// Started by the service, which exits after 1 s.
                ParentExits,
                /// Started by the service in a session of its own.
                OwnSession,
            }

            /// `perl` puts the `sleep` in a group of its own.
            fn service(stop: Stop, shape: Shape) -> anyhow::Result<Service> {
                crate::session::test_support::require_perl()?;
                let script = match shape {
                    Shape::RunsUntilStopped => {
                        "trap 'exit 0' TERM; \\
                         (trap '' TERM; exec perl -e 'setpgrp; exec @ARGV' sleep 300) & \\
                         echo \"left $!\"; while :; do sleep 0.05; done"
                    }
                    Shape::ParentStays => {
                        "(trap '' TERM; perl -e 'setpgrp; exec @ARGV' sleep 300 & \\
                         echo \"left $!\"; wait) & sleep 1; exit 3"
                    }
                    Shape::ParentExits => {
                        "(trap '' TERM; exec perl -e 'setpgrp; exec @ARGV' sleep 300) & \\
                         echo \"left $!\"; sleep 1; exit 3"
                    }
                    Shape::OwnSession => {
                        "(trap '' TERM; exec perl -e 'use POSIX; POSIX::setsid(); exec @ARGV' \\
                         sleep 300) & echo \"left $!\"; while :; do sleep 0.05; done"
                    }
                };
                let spawned = start(script, stop)?;
                let line = spawned
                    .wait_for_output(Stream::Stdout, LONG, &|text| text.contains("left "))
                    .context("the service never started its sleep")?;
                let left = line
                    .split_whitespace()
                    .skip_while(|word| *word != "left")
                    .nth(1)
                    .and_then(|pid| pid.parse().ok())
                    .context("no pid printed")?;
                // A signal to the group before it leaves reaches it there.
                anyhow::ensure!(alone_within(left, LONG), "pid {left} never left the group");
                Ok(Service { spawned, left })
            }

            #[derive(Debug, Clone, Copy)]
            enum Act {
                Stop,
                Interrupt,
                Exit,
            }

            #[rstest]
            #[case::stop(Shape::RunsUntilStopped, Act::Stop)]
            #[case::interrupt(Shape::RunsUntilStopped, Act::Interrupt)]
            #[case::exit_while_its_parent_stays(Shape::ParentStays, Act::Exit)]
            #[case::exit_after_its_parent(Shape::ParentExits, Act::Exit)]
            fn what_left_the_group_goes_with_it(
                #[case] shape: Shape,
                #[case] act: Act,
            ) -> anyhow::Result<()> {
                let stop = Stop::Terminate(LONG);
                let Service { mut spawned, left } = service(stop.clone(), shape)?;
                assert!(runs(left));
                let started = Instant::now();
                match act {
                    Act::Stop => spawned.stop(),
                    Act::Interrupt => {
                        on_interrupt(&stop, spawned.session)();
                        spawned.wait(LONG)?.context("still running")?;
                    }
                    Act::Exit => {
                        let status = spawned.wait(LONG)?.context("still running")?;
                        assert_eq!(status.code(), Some(3));
                    }
                }
                // Killed before the streams it holds open drain.
                assert!(
                    started.elapsed() < LONG / 2,
                    "{shape:?} {act:?} waited on its streams"
                );
                assert!(
                    ends_within(left, Duration::from_secs(5)),
                    "pid {left} outlived the {shape:?} {act:?}"
                );
                Ok(())
            }

            /// Found below the service, which still runs when the kill comes.
            #[test]
            fn a_kill_takes_a_session_the_service_started() -> anyhow::Result<()> {
                let Service { mut spawned, left } = service(Stop::Kill, Shape::OwnSession)?;
                spawned.stop();
                assert!(ends_within(left, Duration::from_secs(5)));
                Ok(())
            }
        }
    }
}
