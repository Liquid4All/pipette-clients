//! The Windows side of the process primitives: no process groups and no
//! SIGTERM, so a stop kills the process tree at once. The children are
//! PowerShell one-liners, quoted with `'` only.
#![cfg(windows)]

use std::cell::Cell;
use std::process::Command;
use std::thread;
use std::time::{Duration, Instant};

use anyhow::Context;

use pipette_plan_types::RuntimeOutput;
use pipette_subprocess::{NotReadyError, Stream, start_process};

mod common;

/// PowerShell can take seconds to start on a busy host.
const LONG: Duration = Duration::from_secs(30);
/// How long a killed tree gets to disappear. Every child a test starts must
/// outlive it, or it could pass by exiting on its own.
const GONE: Duration = Duration::from_secs(10);

fn powershell(script: &str) -> Command {
    let mut command = Command::new("powershell");
    command.args(["-NoProfile", "-NonInteractive", "-Command", script]);
    command
}

fn not_yet(_: &RuntimeOutput) -> Result<(), NotReadyError> {
    Err(NotReadyError::yet("not listening yet"))
}

fn alive(pid: u32) -> anyhow::Result<bool> {
    let filter = format!("PID eq {pid}");
    let output = Command::new("tasklist")
        .args(["/FI", filter.as_str(), "/NH", "/FO", "CSV"])
        .output()?;
    Ok(String::from_utf8_lossy(&output.stdout).contains(&format!("\"{pid}\"")))
}

/// Whether `pid` is gone, giving the tree kill a moment.
fn gone(pid: u32) -> anyhow::Result<bool> {
    let started = Instant::now();
    while started.elapsed() < GONE {
        if !alive(pid)? {
            return Ok(true);
        }
        thread::sleep(Duration::from_millis(100));
    }
    Ok(false)
}

#[test]
fn an_exit_reports_its_status_and_what_it_printed() -> anyhow::Result<()> {
    let mut process = start_process(
        "powershell",
        powershell("[Console]::Out.WriteLine('out'); [Console]::Error.WriteLine('err'); exit 3"),
        Duration::ZERO,
    )?;
    let status = process.wait(LONG)?.context("still running")?;
    assert_eq!(status.code(), Some(3));
    assert!(!process.is_running());
    // Written with CRLF: the line keeps no carriage return.
    let output = process.output();
    assert_eq!(output.runtime_stdout.as_deref(), Some("out\n"));
    assert_eq!(output.runtime_stderr.as_deref(), Some("err\n"));
    Ok(())
}

#[test]
fn a_wait_that_runs_out_leaves_it_running() -> anyhow::Result<()> {
    let mut process = start_process("powershell", powershell("Start-Sleep 600"), Duration::ZERO)?;
    assert_eq!(process.wait(Duration::from_millis(200))?, None);
    assert!(process.is_running());
    process.stop();
    assert!(!process.is_running());
    Ok(())
}

#[test]
fn a_run_past_its_timeout_is_stopped() -> anyhow::Result<()> {
    let mut process = start_process("powershell", powershell("Start-Sleep 600"), Duration::ZERO)?;
    let Err(err) = process.wait_success(Duration::from_secs(3)) else {
        anyhow::bail!("expected the wait to fail");
    };
    assert!(
        format!("{err:#}").contains("powershell did not exit within 3s"),
        "{err:#}"
    );
    let pid = process.pid();
    assert!(gone(pid)?, "{pid} outlived the failed wait");
    Ok(())
}

#[test]
fn stop_kills_what_it_started() -> anyhow::Result<()> {
    let mut process = start_process(
        "powershell",
        powershell(
            "$p = Start-Process -PassThru -NoNewWindow ping -ArgumentList '-n','600','127.0.0.1'; \
             [Console]::Out.WriteLine($p.Id); Start-Sleep 600",
        ),
        Duration::ZERO,
    )?;
    let pid_of = |text: &str| {
        text.lines()
            .find_map(|line| line.trim().parse::<u32>().ok())
    };
    let printed = process
        .wait_for_output(Stream::Stdout, LONG, |text| pid_of(text).is_some())
        .context("no pid printed")?;
    let child = pid_of(&printed).context("no pid printed")?;
    process.stop();
    assert!(gone(child)?, "{child} outlived the stop");
    Ok(())
}

#[test]
fn dropping_it_stops_it() -> anyhow::Result<()> {
    let process = start_process("powershell", powershell("Start-Sleep 600"), Duration::ZERO)?;
    let pid = process.pid();
    drop(process);
    assert!(gone(pid)?, "{pid} outlived the drop");
    Ok(())
}

#[test]
fn its_stdin_is_empty() -> anyhow::Result<()> {
    let mut process = start_process(
        "powershell",
        powershell("[Console]::Out.WriteLine($null -eq [Console]::In.ReadLine())"),
        Duration::ZERO,
    )?;
    process.wait(LONG)?.context("waited on stdin")?;
    assert_eq!(process.output().runtime_stdout.as_deref(), Some("True\n"));
    Ok(())
}

#[test]
fn a_missing_program_is_named_in_the_error() -> anyhow::Result<()> {
    let Err(err) = start_process(
        "no-such-server",
        Command::new(r"C:\nonexistent\pipette-no-such-program.exe"),
        Duration::ZERO,
    ) else {
        anyhow::bail!("expected the spawn to fail");
    };
    assert!(
        format!("{err:#}").contains(
            r"failed to spawn no-such-server: C:\nonexistent\pipette-no-such-program.exe"
        ),
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
    let mut service = start_process("powershell", powershell("Start-Sleep 600"), Duration::ZERO)?;
    service.wait_ready(LONG, |_| ask().map_err(NotReadyError::yet))?;
    assert_eq!(asked.get(), 3);
    assert!(service.is_running());
    service.stop();
    assert!(!service.is_running());
    Ok(())
}

#[test]
fn a_service_that_exits_before_it_is_ready_fails_with_its_output() -> anyhow::Result<()> {
    let mut service = start_process(
        "powershell",
        powershell("[Console]::Error.WriteLine('loading failed'); exit 4"),
        Duration::ZERO,
    )?;
    let Err(err) = service.wait_ready(LONG, not_yet) else {
        anyhow::bail!("expected the wait to fail");
    };
    let message = format!("{err:#}");
    assert!(message.contains("exited with"), "{message}");
    assert!(message.contains("loading failed"), "{message}");
    Ok(())
}

#[test]
fn a_service_never_ready_is_stopped_at_the_timeout() -> anyhow::Result<()> {
    let mut service = start_process("powershell", powershell("Start-Sleep 600"), Duration::ZERO)?;
    let Err(err) = service.wait_ready(Duration::from_secs(3), not_yet) else {
        anyhow::bail!("expected the wait to fail");
    };
    assert!(
        format!("{err:#}").contains("was not ready within"),
        "{err:#}"
    );
    let pid = service.pid();
    assert!(gone(pid)?, "{pid} outlived the failed wait");
    Ok(())
}

/// The `taskkill` behind a stop prints nothing on the CLI's own streams.
#[test]
fn a_stop_prints_nothing_on_the_clis_streams() -> anyhow::Result<()> {
    let run = Command::new(env!("CARGO_BIN_EXE_start_and_stop")).output()?;
    anyhow::ensure!(run.status.success(), "{run:?}");
    assert_eq!(String::from_utf8_lossy(&run.stdout), "");
    assert_eq!(String::from_utf8_lossy(&run.stderr), "");
    Ok(())
}

/// A hard kill (`TerminateProcess`, as `Stop-Process -Force`) runs no
/// teardown. The job that holds the children of the run stops them.
#[test]
fn a_hard_kill_takes_what_the_run_started_with_it() -> anyhow::Result<()> {
    use std::process::Stdio;

    #[expect(clippy::disallowed_methods, reason = "stdout is piped to read the pid")]
    let mut run = Command::new(env!("CARGO_BIN_EXE_start_and_wait"))
        .stdout(Stdio::piped())
        .spawn()?;
    let left: u32 = common::read_pid(&mut run)?;
    run.kill()?;
    run.wait()?;
    if !gone(left)? {
        Command::new("taskkill")
            .args(["/F", "/T", "/PID", &left.to_string()])
            .output()?;
        anyhow::bail!("{left} outlived the run");
    }
    Ok(())
}

/// Windows has no SIGTERM, so a stop grace is not waited out.
#[test]
fn a_service_is_killed_at_once_whatever_its_grace() -> anyhow::Result<()> {
    let mut service = start_process("powershell", powershell("Start-Sleep 600"), LONG)?;
    let started = Instant::now();
    service.stop();
    assert!(started.elapsed() < LONG, "waited out the grace");
    assert!(!service.is_running());
    Ok(())
}

#[test]
fn one_that_died_is_started_again() -> anyhow::Result<()> {
    let start = || start_process("powershell", powershell("Start-Sleep 600"), Duration::ZERO);
    let mut process = start()?;
    let first = process.pid();
    let killed = Command::new("taskkill")
        .args(["/F", "/PID", &first.to_string()])
        .output()?;
    anyhow::ensure!(killed.status.success(), "taskkill failed: {killed:?}");
    process.wait(LONG)?.context("still running")?;

    process = start()?;
    assert_ne!(process.pid(), first);
    assert!(process.is_running());
    Ok(())
}
