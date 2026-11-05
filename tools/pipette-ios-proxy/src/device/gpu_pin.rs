//! Xcode's "GPU Performance State" device condition, held for one run.
//!
//! Left to itself, iOS runs some small-model decode reps at a GPU state about
//! 1.3x slower than the rest, so a cell's reps come out at two speeds. The
//! Instruments ConditionInducer pins the state, but only while the connection
//! that set it stays open, so a `pymobiledevice3 developer dvt condition set`
//! child is kept for the whole run and SIGTERMed after it. Not SIGINT: a child
//! started with SIGINT ignored (as a non-interactive shell starts `&` jobs)
//! never sees it, while SIGTERM's default action ends it either way.

use std::io::{BufRead, BufReader};
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::sync::mpsc;
use std::time::Duration;

use anyhow::Context;

use crate::tool;

/// What the CLI prints once the condition is set, before it waits for a signal.
const READY: &str = "Press Ctrl+C";

/// How long the child may take to exit after SIGTERM before it is killed.
const RELEASE_GRACE: Duration = Duration::from_secs(5);

/// How much of a failed child's stderr rides along on the error.
const DETAIL_CHARS: usize = 400;

/// The condition held. `Mid` ran 1.9x slower and `Min` slower still, so `Max`
/// is the only state a benchmark wants.
const PROFILE: &str = "GPUPerformanceStateMax";

/// The held condition. Dropping it ends the connection, which releases it.
#[derive(Debug)]
pub(crate) struct GpuPin {
    child: Child,
    stderr: Option<mpsc::Receiver<String>>,
}

impl GpuPin {
    /// Set the condition on the phone with hardware UDID `udid`, through Apple's
    /// own tunnel (`--native`: no root, no pairing record). Errs when it is not
    /// set within `ready_within`.
    pub(crate) fn hold(program: &Path, udid: &str, ready_within: Duration) -> anyhow::Result<Self> {
        #[expect(
            clippy::disallowed_methods,
            reason = "both output streams are piped and drained below"
        )]
        let mut child = Command::new(program)
            .args(["developer", "dvt", "condition", "set", PROFILE])
            .args(["--native", "--udid", udid])
            // A piped stdout is block-buffered, which would hold the prompt back.
            .env("PYTHONUNBUFFERED", "1")
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .with_context(|| format!("running `{}`", program.display()))?;
        let stderr = child.stderr.take().map(tool::drain);
        let (ready, prompted) = mpsc::channel();
        if let Some(stdout) = child.stdout.take() {
            std::thread::spawn(move || {
                for line in BufReader::new(stdout).lines().map_while(Result::ok) {
                    if line.contains(READY) {
                        let _ = ready.send(());
                    }
                }
            });
        }
        // An early exit closes stdout, which ends the wait at once.
        let timed_out = match prompted.recv_timeout(ready_within) {
            Ok(()) => return Ok(Self { child, stderr }),
            Err(mpsc::RecvTimeoutError::Timeout) => true,
            Err(mpsc::RecvTimeoutError::Disconnected) => false,
        };
        let grace = if timed_out {
            Duration::ZERO
        } else {
            RELEASE_GRACE
        };
        let status = tool::wait_bounded(&mut child, grace);
        let detail = last_line(&tool::collect(stderr));
        match status {
            Some(status) if !timed_out => {
                anyhow::bail!("`{}` exited ({status}): {detail}", program.display())
            }
            _ => anyhow::bail!(
                "`{}` did not set the condition within {}s: {detail}",
                program.display(),
                ready_within.as_secs()
            ),
        }
    }

    /// End the hold. Errs when the helper had already exited: the phone dropped
    /// the condition then, so the rest of the run went unpinned.
    pub(crate) fn release(mut self) -> anyhow::Result<()> {
        let Ok(Some(status)) = self.child.try_wait() else {
            // Still holding: Drop ends it.
            return Ok(());
        };
        let detail = last_line(&tool::collect(self.stderr.take()));
        anyhow::bail!("the GPU performance state helper exited during the run ({status}): {detail}")
    }
}

impl Drop for GpuPin {
    fn drop(&mut self) {
        use rustix::process::{Pid, Signal, kill_process};

        // Already reaped: its pid may belong to another process by now.
        if let Ok(Some(_)) = self.child.try_wait() {
            return;
        }
        if let Some(pid) = i32::try_from(self.child.id()).ok().and_then(Pid::from_raw) {
            let _ = kill_process(pid, Signal::TERM);
        }
        if tool::wait_bounded(&mut self.child, RELEASE_GRACE).is_none() {
            tracing::warn!(
                "the GPU performance state helper outlived SIGTERM by {}s; killed it",
                RELEASE_GRACE.as_secs()
            );
        }
    }
}

/// The CLI a pin runs must run: a missing one would fail every job claimed.
pub(crate) fn check_cli(program: &Path, deadline: Duration) -> anyhow::Result<()> {
    let checked = tool::run_bounded(Command::new(program).arg("version"), deadline)
        .map_err(anyhow::Error::from)
        .and_then(|ran| {
            anyhow::ensure!(
                ran.status.success(),
                "`version` exited ({}): {}",
                ran.status,
                last_line(&ran.stderr)
            );
            Ok(())
        });
    checked.with_context(|| {
        format!(
            "checking [devicectl] pymobiledevice3 `{}`, which the GPU pin needs",
            program.display()
        )
    })
}

/// SIGTERM a pin a proxy left behind when it died without unwinding (an ssh drop
/// kills it by SIGPIPE), so this one does not run pinned against its config.
/// Called with the phone's lock held: no live proxy owns such a helper.
pub(crate) fn release_leftover(udid: &str, deadline: Duration) {
    let pattern = format!("dvt condition set GPUPerformanceState[A-Za-z]+ --native --udid {udid}");
    // pkill exits 1 when nothing matched, the usual case.
    if let Ok(ran) = tool::run_bounded(
        Command::new("/usr/bin/pkill").args(["-TERM", "-f", &pattern]),
        deadline,
    ) && ran.status.success()
    {
        tracing::warn!(udid = %udid, "released a GPU performance state pin a previous proxy left");
    }
}

/// The CLI ends a failure with a one-line exception under a framed traceback.
fn last_line(stderr: &str) -> String {
    let line = stderr
        .lines()
        .map(str::trim)
        .rfind(|line| !line.is_empty())
        .unwrap_or("no output");
    line.chars().take(DETAIL_CHARS).collect()
}

#[cfg(test)]
mod tests {
    use std::os::unix::fs::PermissionsExt;
    use std::path::PathBuf;
    use std::time::Instant;

    use super::*;

    /// A stand-in CLI: `body` runs under /bin/sh with the real argv in `$@`.
    fn fake_cli(dir: &Path, body: &str) -> anyhow::Result<PathBuf> {
        let path = dir.join("pymobiledevice3");
        std::fs::write(&path, format!("#!/bin/sh\n{body}\n"))?;
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755))?;
        Ok(path)
    }

    #[test]
    fn the_pin_is_held_until_released_by_sigterm() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let marker = dir.path().join("released");
        let program = fake_cli(
            dir.path(),
            &format!(
                "echo \"$@\" > {args}\ntrap 'touch {marker}; exit 0' TERM\n\
                 echo \"Press Ctrl+C to send a SIGINT\"\nwhile :; do sleep 0.1; done",
                args = dir.path().join("args").display(),
                marker = marker.display(),
            ),
        )?;
        let pin = GpuPin::hold(
            &program,
            "00008150-000E1D1C2247401C",
            Duration::from_secs(10),
        )?;
        assert!(!marker.exists(), "released before the drop");
        let args = std::fs::read_to_string(dir.path().join("args"))?;
        assert_eq!(
            args.trim(),
            "developer dvt condition set GPUPerformanceStateMax --native --udid \
             00008150-000E1D1C2247401C"
        );
        pin.release()?;
        assert!(marker.exists(), "SIGTERM did not reach the helper");
        Ok(())
    }

    #[test]
    fn a_helper_that_dies_during_the_run_fails_the_release() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let program = fake_cli(
            dir.path(),
            "echo 'Press Ctrl+C to send a SIGINT'\necho 'ConnectionAbortedError' >&2\nexit 1",
        )?;
        let mut pin = GpuPin::hold(&program, "u", Duration::from_secs(10))?;
        // The run outlasts the helper, which exits right after its prompt.
        let started = Instant::now();
        while pin.child.try_wait()?.is_none() && started.elapsed() < Duration::from_secs(5) {
            std::thread::sleep(Duration::from_millis(50));
        }
        let error = pin
            .release()
            .err()
            .context("a helper gone before the release must fail it")?;
        let message = format!("{error:#}");
        assert!(message.contains("exited during the run"), "{message}");
        assert!(message.contains("ConnectionAbortedError"), "{message}");
        Ok(())
    }

    #[test]
    fn a_leftover_pin_is_found_by_the_argv_hold_gives_it() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let marker = dir.path().join("released");
        let program = fake_cli(
            dir.path(),
            &format!(
                "trap 'touch {marker}; exit 0' TERM\n\
                 echo \"Press Ctrl+C to send a SIGINT\"\nwhile :; do sleep 0.1; done",
                marker = marker.display(),
            ),
        )?;
        // Unique, so the system-wide pkill reaches only this test's helper.
        let udid = format!("leftover-test-{}", std::process::id());
        let pin = GpuPin::hold(&program, &udid, Duration::from_secs(10))?;
        release_leftover(&udid, Duration::from_secs(10));
        let started = Instant::now();
        while !marker.exists() && started.elapsed() < Duration::from_secs(5) {
            std::thread::sleep(Duration::from_millis(50));
        }
        assert!(marker.exists(), "pkill did not match the helper's argv");
        drop(pin);
        Ok(())
    }

    #[test]
    fn a_helper_that_exits_before_the_prompt_is_an_error_naming_why() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let program = fake_cli(
            dir.path(),
            "echo 'Traceback' >&2\necho 'ConnectionRefusedError: [Errno 61]' >&2\nexit 1",
        )?;
        let started = Instant::now();
        let error = GpuPin::hold(&program, "u", Duration::from_secs(30))
            .err()
            .context("an exit before the prompt must fail")?;
        assert!(
            started.elapsed() < Duration::from_secs(10),
            "waited out the deadline"
        );
        assert!(
            format!("{error:#}").contains("ConnectionRefusedError: [Errno 61]"),
            "{error:#}"
        );
        Ok(())
    }

    #[test]
    fn the_startup_check_refuses_a_missing_or_failing_cli() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let deadline = Duration::from_secs(10);
        let error = check_cli(&dir.path().join("absent"), deadline)
            .err()
            .context("a missing CLI must be refused")?;
        assert!(format!("{error:#}").contains("absent"), "{error:#}");

        check_cli(&fake_cli(dir.path(), "echo 11.43.1")?, deadline)?;
        let error = check_cli(&fake_cli(dir.path(), "echo broken >&2; exit 1")?, deadline)
            .err()
            .context("a failing CLI must be refused")?;
        assert!(format!("{error:#}").contains("broken"), "{error:#}");
        Ok(())
    }

    #[test]
    fn a_helper_that_never_prompts_times_out_and_is_killed() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let program = fake_cli(dir.path(), "exec sleep 30")?;
        let error = GpuPin::hold(&program, "u", Duration::from_secs(1))
            .err()
            .context("a helper that never prompts must fail")?;
        assert!(
            format!("{error:#}").contains("did not set the condition within 1s"),
            "{error:#}"
        );
        Ok(())
    }
}
