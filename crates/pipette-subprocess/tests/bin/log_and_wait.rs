//! Starts a `sleep` in its own session and prints its pid. Then it logs a line
//! to stderr every 50 ms until the stderr reader closes. This is a worker whose
//! adb or ssh session ended.
//!
//! The first argument is the mode:
//! - `armed`: asks for a closed output to stop it, as `pipette worker` does.
//! - `unarmed`: keeps the default disposition, as every other command does.
//! - `quiet`: armed, but writes nothing, so only the watch on stderr can stop
//!   it.

use std::io::Write;
use std::process::Command;
use std::time::Duration;

fn main() -> Result<(), pipette_subprocess::StartError> {
    let mode = std::env::args().nth(1).unwrap_or_default();
    pipette_subprocess::reset_sigpipe();
    pipette_subprocess::install_signal_handler();
    if mode != "unarmed" {
        pipette_subprocess::stop_on_closed_output();
    }
    let mut sleep = Command::new("sleep");
    sleep.arg("300");
    let process = pipette_subprocess::start_process("sleep", sleep, Duration::ZERO)?;
    println!("{}", process.pid());
    for _ in 0..6000 {
        if mode != "quiet" {
            let _ = writeln!(pipette_subprocess::log_stderr(), "still running");
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    Ok(())
}
