//! Starts a PowerShell that would sleep for ten minutes, stops it, and exits.

use std::process::Command;

fn main() -> Result<(), pipette_subprocess::StartError> {
    let mut sleep = Command::new("powershell");
    sleep.args([
        "-NoProfile",
        "-NonInteractive",
        "-Command",
        "Start-Sleep 600",
    ]);
    let mut process = pipette_subprocess::start_process(
        "powershell",
        sleep,
        pipette_subprocess::OnStop::SIGKILL,
    )?;
    process.stop();
    Ok(())
}
