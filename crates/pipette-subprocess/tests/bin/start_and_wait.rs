//! Starts a `sleep` in its own session, prints its pid, and waits to be killed:
//! a run that dies without stopping what it started. On Windows the sleep is
//! PowerShell's `Start-Sleep`.

use std::process::Command;
use std::time::Duration;

fn main() -> Result<(), pipette_subprocess::StartError> {
    let process = pipette_subprocess::start_process("sleep", sleep(), Duration::ZERO)?;
    println!("{}", process.pid());
    std::thread::sleep(Duration::from_secs(300));
    Ok(())
}

#[cfg(unix)]
fn sleep() -> Command {
    let mut sleep = Command::new("sleep");
    sleep.arg("300");
    sleep
}

#[cfg(windows)]
fn sleep() -> Command {
    let mut sleep = Command::new("powershell");
    sleep.args([
        "-NoProfile",
        "-NonInteractive",
        "-Command",
        "Start-Sleep 300",
    ]);
    sleep
}
