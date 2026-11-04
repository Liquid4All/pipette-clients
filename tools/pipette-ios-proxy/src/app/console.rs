//! One `--console` launch, streamed to completion and ended on the deadline, a
//! lost lease, or the stop signal.

use std::io::{BufRead, BufReader};
use std::process::{ChildStderr, ChildStdout, Command, Stdio};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use anyhow::Context;
use tracing::Level;

use super::launch::LaunchPlan;
use super::{APP_LINE, ConsoleRun, EngineClock, SentinelStream, scan_sentinel};
use crate::device::{self, DeviceControl, Devicectl};
use crate::secret;
use crate::timing;

/// How long after SIGTERM the device-side terminate backstop fires. Generous:
/// the app is given room to finish uploading the cell in flight.
const TERMINATE_GRACE: Duration = Duration::from_secs(10);

/// Launch the app and drain both streams to completion, terminating it on the
/// deadline, a lost lease, the stop signal (launchd SIGKILLs us 20 s after
/// SIGTERM, which would orphan a run that outlived it), or no line from the app
/// by `first_line`.
#[expect(
    clippy::too_many_arguments,
    reason = "the launch, its two deadlines, its signals and its line sink"
)]
pub(crate) fn run_console(
    device: &Devicectl,
    bundle_id: &str,
    app_args: &[String],
    deadline: Duration,
    first_line: Duration,
    sentinel_on: SentinelStream,
    signals: Signals<'_>,
    on_stdout_line: &mut dyn FnMut(&str),
) -> anyhow::Result<ConsoleRun> {
    let abort = signals.abort;
    let argv = device.launch_argv(bundle_id, app_args);
    // Never log the argv: `model=` can carry a gated repository's access token.
    // The app can print it back, so every line read is scrubbed of it.
    let secrets = secret::argv_secrets(app_args);
    let arg_count = app_args.len().to_string();
    let deadline_secs = deadline.as_secs().to_string();
    tracing::debug!(
        device = %device.udid(),
        bundle_id = %bundle_id,
        app_args = %arg_count,
        deadline_secs = %deadline_secs,
        "launching the app headless"
    );

    #[expect(
        clippy::disallowed_methods,
        reason = "both output streams are piped and read below"
    )]
    let mut child = Command::new(device::PROGRAM)
        .args(&argv)
        // Inheriting stdin would let the app consume the operator's terminal input.
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| {
            format!(
                "spawning `{} devicectl device process launch` ({})",
                device::PROGRAM,
                device::INSTALL_HINT
            )
        })?;

    let pid = child.id();
    let stdout_pipe = child
        .stdout
        .take()
        .context("the devicectl launch's stdout was not piped")?;
    let stderr_pipe = child
        .stderr
        .take()
        .context("the devicectl launch's stderr was not piped")?;

    let finished = AtomicBool::new(false);
    let timed_out = AtomicBool::new(false);
    let stopped = AtomicBool::new(false);
    let silent = AtomicBool::new(false);
    let spoke = AtomicBool::new(false);
    let stderr_text = Mutex::new(String::new());
    let mut stdout_text = String::new();

    // Scoped threads: the stderr drain and the watchdog borrow caller-owned state.
    // Both log, so both carry the caller's span and subscriber.
    let span = tracing::Span::current();
    let dispatch = tracing::dispatcher::get_default(tracing::Dispatch::clone);
    let engine_time = std::thread::scope(|scope| {
        let stderr_drain = scope.spawn(|| {
            let _dispatch = tracing::dispatcher::set_default(&dispatch);
            let _span = span.enter();
            drain_stderr(stderr_pipe, &stderr_text, &secrets, &spoke)
        });
        scope.spawn(|| {
            let _dispatch = tracing::dispatcher::set_default(&dispatch);
            let _span = span.enter();
            watchdog(
                device,
                bundle_id,
                pid,
                deadline,
                first_line,
                signals,
                Verdict {
                    finished: &finished,
                    timed_out: &timed_out,
                    stopped: &stopped,
                    silent: &silent,
                    spoke: &spoke,
                },
            )
        });
        drain_stdout(
            stdout_pipe,
            &mut stdout_text,
            &secrets,
            abort,
            &spoke,
            on_stdout_line,
        );
        // Before the child is reaped, so the watchdog cannot signal a reused pid.
        finished.store(true, Ordering::SeqCst);
        stderr_drain.join().ok().flatten()
    });

    let status = child
        .wait()
        .context("waiting for the devicectl launch to exit")?;
    let stderr = stderr_text.into_inner().unwrap_or_default();
    let sentinel = match sentinel_on {
        SentinelStream::Stdout => scan_sentinel(&stdout_text),
        SentinelStream::Stderr => scan_sentinel(&stderr),
    };

    Ok(ConsoleRun {
        stdout: stdout_text,
        stderr,
        sentinel,
        exit_code: status.code(),
        timed_out: timed_out.load(Ordering::SeqCst),
        engine_time,
        aborted: abort.load(Ordering::SeqCst),
        stopped: stopped.load(Ordering::SeqCst),
        silent: silent.load(Ordering::SeqCst),
    })
}

/// `on_submitted` runs as each line saying the app's upload was accepted arrives.
/// A silent launch is retried once, under the same lease ([`relaunch_if_silent`]).
pub(crate) fn run_benchmark(
    device: &dyn DeviceControl,
    bundle_id: &str,
    plan: &LaunchPlan,
    run_timeout: Duration,
    first_line: Duration,
    signals: Signals<'_>,
    on_submitted: &dyn Fn(),
) -> anyhow::Result<ConsoleRun> {
    let mut forward = |line: &str| {
        // The only record of which cells ran, over a run that can take an hour.
        tracing::info!(line = %line, "app");
        if reports_a_submission(line) {
            on_submitted();
        }
    };
    relaunch_if_silent("the run", first_line, signals, || {
        device.launch_console(
            bundle_id,
            &plan.app_args,
            run_timeout,
            first_line,
            SentinelStream::Stdout,
            signals,
            &mut forward,
        )
    })
}

/// `launch`, then once more if the app printed nothing by `first_line`. Single
/// App Mode reopens the app with no arguments after every exit, and when that
/// lands between `--terminate-existing`'s terminate and launch steps, `--console`
/// attaches to the UI copy, which prints nothing and never exits. The relaunch's
/// `--terminate-existing` ends it. A second silent launch comes back
/// [`ConsoleRun::silent`]. Every caller's launch is safe to repeat: a run that
/// printed nothing never had the cell, and each verb reads, or deletes one result.
pub(crate) fn relaunch_if_silent(
    what: &str,
    first_line: Duration,
    signals: Signals<'_>,
    mut launch: impl FnMut() -> anyhow::Result<ConsoleRun>,
) -> anyhow::Result<ConsoleRun> {
    let run = launch()?;
    if !run.silent {
        return Ok(run);
    }
    if signals.raised() {
        // Stopping, or the lease is gone: no relaunch and no failure to report.
        return Ok(ConsoleRun {
            stopped: true,
            ..run
        });
    }
    tracing::warn!(
        launch = %what,
        silent_secs = %first_line.as_secs(),
        "the app printed nothing after launch; relaunching"
    );
    let run = launch()?;
    if run.silent {
        tracing::warn!(
            launch = %what,
            silent_secs = %first_line.as_secs(),
            "the app printed nothing after the relaunch either; giving up"
        );
    }
    Ok(run)
}

/// The app's filing line for a cell the server accepted:
/// `[HEADLESS] result cell=<id> … submitted_as_job=<server job id>`
/// (`ResultReport.line`).
fn reports_a_submission(line: &str) -> bool {
    line.split_once("[HEADLESS] result cell=")
        .is_some_and(|(_, fields)| {
            fields
                .split_whitespace()
                .any(|field| field.starts_with("submitted_as_job="))
        })
}

/// Drain stdout on the CALLER's thread: `on_stdout_line` is a borrowed
/// `&mut dyn FnMut` and cannot cross a thread boundary.
fn drain_stdout(
    pipe: ChildStdout,
    into: &mut String,
    secrets: &[String],
    abort: &AtomicBool,
    spoke: &AtomicBool,
    on_stdout_line: &mut dyn FnMut(&str),
) {
    BufReader::new(pipe)
        .split(b'\n')
        .map_while(Result::ok)
        .for_each(|bytes| {
            let line = line_text(&bytes, secrets);
            heard(&line, spoke);
            // A lost lease stops FORWARDING, not draining: an unread pipe would
            // deadlock the app before it could exit.
            if !abort.load(Ordering::SeqCst) {
                on_stdout_line(&line);
            }
            into.push_str(&line);
            into.push('\n');
        });
}

/// Drain stderr on its own thread: a full pipe deadlocks the app, and every
/// multi-minute readiness wait is on this stream while stdout stays silent.
/// Returns how long the engine had the cell when the stream closed.
fn drain_stderr(
    pipe: ChildStderr,
    into: &Mutex<String>,
    secrets: &[String],
    spoke: &AtomicBool,
) -> Option<Duration> {
    let mut clock = EngineClock::default();
    BufReader::new(pipe)
        .split(b'\n')
        .map_while(Result::ok)
        .for_each(|bytes| {
            let line = line_text(&bytes, secrets);
            heard(&line, spoke);
            log_app_stderr(&line);
            clock.line(&line, Instant::now());
            // Dropping a line beats taking the launch down over a log buffer.
            if let Ok(mut text) = into.lock() {
                text.push_str(&line);
                text.push('\n');
            }
        });
    clock.time(Instant::now())
}

/// At the level the app gave the line: `AppLog` mirrors each of its lines as
/// `[HEADLESS] <level> <category> <message>` (`mirrorHeadless`,
/// `ios/Pipette/Pipette/Support/AppLog.swift`). The rest of stderr (the
/// engines' own output, `devicectl`'s) stays at debug.
fn log_app_stderr(line: &str) {
    match app_log_level(line) {
        Some(Level::ERROR) => tracing::error!(line = %line, "app stderr"),
        Some(Level::WARN) => tracing::warn!(line = %line, "app stderr"),
        Some(Level::INFO) => tracing::info!(line = %line, "app stderr"),
        _ => tracing::debug!(line = %line, "app stderr"),
    }
}

fn app_log_level(line: &str) -> Option<Level> {
    let level = line.strip_prefix("[HEADLESS] ")?.split(' ').next()?;
    match level {
        "error" => Some(Level::ERROR),
        "warning" => Some(Level::WARN),
        "info" => Some(Level::INFO),
        "debug" => Some(Level::DEBUG),
        _ => None,
    }
}

/// Raise `spoke` on the app's own line, which ends the first-line deadline.
fn heard(line: &str, spoke: &AtomicBool) {
    if line.starts_with(APP_LINE) {
        spoke.store(true, Ordering::SeqCst);
    }
}

/// One console line as text, lossily: a model name is not ours to validate.
/// Scrubbed before anything logs, forwards or keeps it.
fn line_text(bytes: &[u8], secrets: &[String]) -> String {
    let line = String::from_utf8_lossy(bytes)
        .trim_end_matches('\r')
        .to_string();
    secret::scrub(line, secrets)
}

#[derive(Clone, Copy)]
pub(crate) struct Signals<'a> {
    pub(crate) abort: &'a AtomicBool,
    pub(crate) stop: &'a AtomicBool,
}

impl Signals<'_> {
    /// The lease is lost or the process is stopping.
    pub(crate) fn raised(&self) -> bool {
        self.abort.load(Ordering::SeqCst) || self.stop.load(Ordering::SeqCst)
    }
}

struct Verdict<'a> {
    finished: &'a AtomicBool,
    timed_out: &'a AtomicBool,
    stopped: &'a AtomicBool,
    silent: &'a AtomicBool,
    /// Raised by the drains on the app's first line.
    spoke: &'a AtomicBool,
}

/// The deadline, first-line, lost-lease and stop-signal enforcer, on its own
/// thread because the stdout drain blocks until the app exits. Polling on
/// [`timing::SLEEP_SLICE`] costs one slice of latency per call and needs no channel.
fn watchdog(
    device: &dyn DeviceControl,
    bundle_id: &str,
    pid: u32,
    deadline: Duration,
    first_line: Duration,
    signals: Signals<'_>,
    verdict: Verdict<'_>,
) {
    let start = Instant::now();
    let pid_text = pid.to_string();
    let mut signalled: Option<Instant> = None;
    let mut backstopped = false;
    let mut warned = false;
    while !verdict.finished.load(Ordering::SeqCst) {
        match signalled {
            None => {
                if let Some(reason) =
                    ending(signals, &verdict, start.elapsed(), deadline, first_line)
                {
                    tracing::warn!(
                        reason = %reason,
                        pid = %pid_text,
                        "terminating the launch"
                    );
                    send_sigterm(pid);
                    signalled = Some(Instant::now());
                }
            }
            Some(at) if !backstopped && at.elapsed() >= TERMINATE_GRACE => {
                backstopped = true;
                terminate_on_device(device, bundle_id);
            }
            Some(at) if backstopped && !warned && at.elapsed() >= TERMINATE_GRACE * 2 => {
                warned = true;
                tracing::warn!(
                    "the launch has not exited after SIGTERM and a device-side terminate; still waiting"
                );
            }
            _ => {}
        }
        std::thread::sleep(timing::SLEEP_SLICE);
    }
}

/// Why the launch ends at `elapsed`, if it does, with the verdict's flag for that
/// raised. The first-line deadline only ends a launch the app has said nothing
/// on; once it has, the run deadline alone applies. It is checked first, so a
/// silent launch whose two deadlines fall together (a verb's 60 s
/// `call_timeout_secs` default) is relaunched rather than timed out.
fn ending(
    signals: Signals<'_>,
    verdict: &Verdict<'_>,
    elapsed: Duration,
    deadline: Duration,
    first_line: Duration,
) -> Option<&'static str> {
    if signals.abort.load(Ordering::SeqCst) {
        Some("the lease was lost")
    } else if signals.stop.load(Ordering::SeqCst) {
        verdict.stopped.store(true, Ordering::SeqCst);
        Some("the process was asked to stop")
    } else if elapsed >= first_line && !verdict.spoke.load(Ordering::SeqCst) {
        verdict.silent.store(true, Ordering::SeqCst);
        Some("the app printed nothing by the first-line deadline")
    } else if elapsed >= deadline {
        verdict.timed_out.store(true, Ordering::SeqCst);
        Some("the run deadline elapsed")
    } else {
        None
    }
}

/// SIGTERM to the `xcrun` child, which `--console` forwards to the app so an
/// in-flight cell can finish uploading. Never SIGKILL: it orphans the app.
fn send_sigterm(pid: u32) {
    use rustix::process::{Pid, Signal, kill_process};

    let Some(target) = i32::try_from(pid).ok().and_then(Pid::from_raw) else {
        return;
    };
    if let Err(error) = kill_process(target, Signal::TERM) {
        tracing::debug!(pid = %pid, error = %error, "SIGTERM to the launch failed");
    }
}

/// The backstop for an app that ignored the forwarded SIGTERM.
fn terminate_on_device(device: &dyn DeviceControl, bundle_id: &str) {
    match device.pid_for_bundle(bundle_id) {
        Ok(Some(pid)) => match device.terminate_process(pid) {
            Ok(()) => {
                tracing::warn!(pid = %pid, "terminated the app on the device")
            }
            Err(error) => tracing::warn!(
                error = %format!("{error:#}"),
                "terminating the app on the device failed"
            ),
        },
        Ok(None) => tracing::debug!("the app is no longer running on the device"),
        Err(error) => tracing::warn!(
            error = %format!("{error:#}"),
            "looking up the app's pid on the device failed"
        ),
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    /// Only a cell's filing line naming the server's job is an accepted upload.
    #[rstest]
    #[case::accepted(
        "[HEADLESS] result cell=cell-1 benchmark=decode_throughput_256 location=remote/synced \
         payload=/x/payload.json submitted_as_job=job-1",
        true
    )]
    #[case::pending(
        "[HEADLESS] result cell=cell-1 benchmark=decode_throughput_256 location=remote/pending \
         payload=/x/payload.json",
        false
    )]
    #[case::the_count("[HEADLESS] result submitted cells=1", false)]
    #[case::a_listing("[HEADLESS] result result=job-a/cell-1 state=submitted", false)]
    fn an_accepted_upload_is_read_off_its_filing_line(#[case] line: &str, #[case] accepted: bool) {
        assert_eq!(reports_a_submission(line), accepted);
    }

    const SILENT: &str = "the app printed nothing by the first-line deadline";

    const DEADLINE: &str = "the run deadline elapsed";

    /// The first-line deadline ends a launch the app has printed nothing on, and
    /// only that one: after the app's first line the run deadline alone applies.
    /// A silent verb whose call deadline falls at the same moment is silent, so it
    /// is relaunched; one whose call deadline comes first times out as before.
    #[rstest]
    #[case::silent(false, 60, 10800, Some(SILENT))]
    #[case::not_yet(false, 59, 10800, None)]
    #[case::after_a_line(true, 60, 10800, None)]
    #[case::after_a_line_the_run_deadline(true, 10800, 10800, Some(DEADLINE))]
    #[case::a_verb_at_both_deadlines(false, 60, 60, Some(SILENT))]
    #[case::a_verb_whose_call_deadline_is_sooner(false, 30, 30, Some(DEADLINE))]
    fn the_first_line_deadline_ends_only_a_silent_launch(
        #[case] spoke: bool,
        #[case] elapsed: u64,
        #[case] deadline: u64,
        #[case] reason: Option<&str>,
    ) {
        let flags: [AtomicBool; 4] = Default::default();
        let [never, finished, timed_out, stopped] = &flags;
        let (silent, spoke) = (AtomicBool::new(false), AtomicBool::new(spoke));
        let verdict = Verdict {
            finished,
            timed_out,
            stopped,
            silent: &silent,
            spoke: &spoke,
        };
        let ended = ending(
            Signals {
                abort: never,
                stop: never,
            },
            &verdict,
            Duration::from_secs(elapsed),
            Duration::from_secs(deadline),
            Duration::from_secs(60),
        );
        assert_eq!(ended, reason);
        assert_eq!(silent.load(Ordering::SeqCst), reason == Some(SILENT));
    }

    /// Only the app's own line, on either stream, ends the silence; `devicectl`'s
    /// lines around a launch do not.
    #[rstest]
    #[case::launched(
        "Launched application with ai.liquid.liquid-pipette bundle identifier.",
        false
    )]
    #[case::waiting("Waiting for the application to terminate…", false)]
    #[case::terminated("App terminated due to signal 15.", false)]
    #[case::stdout("[HEADLESS] bench ensuring unsloth/Qwen3.5-2B-GGUF", true)]
    #[case::stderr(
        "[HEADLESS] info benchmarkSync benchmark decode_throughput_512_100: outcome=changed",
        true
    )]
    // The line every headless launch prints first, on stderr; the iOS test
    // `theLaunchMarkerNamesTheProcess` asserts the same literal.
    #[case::the_launch_marker("[HEADLESS] hello pid=4242", true)]
    fn only_the_apps_own_line_breaks_the_silence(#[case] line: &str, #[case] app: bool) {
        let spoke = AtomicBool::new(false);
        heard(line, &spoke);
        assert_eq!(spoke.load(Ordering::SeqCst), app);
    }

    /// `AppLog`'s mirrored lines keep their level; anything else is debug.
    #[rstest]
    #[case::error("[HEADLESS] error jobRun run failed", Some(Level::ERROR))]
    #[case::warning("[HEADLESS] warning jobRun slow", Some(Level::WARN))]
    #[case::info("[HEADLESS] info jobRun dispatch: request", Some(Level::INFO))]
    #[case::debug("[HEADLESS] debug jobRun poll", Some(Level::DEBUG))]
    #[case::usage_error("[HEADLESS] ERROR invalid value", None)]
    #[case::engine_output("ggml_metal_init: found device", None)]
    fn an_app_line_keeps_its_level(#[case] line: &str, #[case] expected: Option<Level>) {
        assert_eq!(app_log_level(line), expected);
    }

    /// A usage error echoes the whole argv (`HeadlessRunner.swift`).
    #[test]
    fn a_line_echoing_the_argv_loses_its_token() {
        let model = r#"model={"source":"huggingface","auth_token":"hf_gated"}"#;
        let secrets = secret::argv_secrets(&[model.to_string()]);
        let echoed = format!("[HEADLESS] ERROR invalid value: {model}\r");
        assert_eq!(
            line_text(echoed.as_bytes(), &secrets),
            r#"[HEADLESS] ERROR invalid value: model={"source":"huggingface","auth_token":"<redacted>"}"#
        );
    }
}
