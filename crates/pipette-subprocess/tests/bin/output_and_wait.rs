//! Runs a short command that starts a copy of this helper sleeping in a group
//! of its own, writes its pid to the file named first, and waits: a run ^C
//! ends mid-command. The sleeper is this helper, not `sleep`: some macOS
//! releases let no one read a system binary's environment, which carries its
//! session's token.
//!
//! With `--stop`, it runs a two-second command under a stop flag instead and
//! exits 0 if a ^C set the flag.

use std::process::Command;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

fn main() -> Result<(), pipette_subprocess::OutputError> {
    let arg = std::env::args().nth(1).unwrap_or_default();
    if arg == "--sleep" {
        // Out of the command's group, as a `ninja` build step leaves it.
        #[cfg(unix)]
        // SAFETY: no pointers.
        unsafe {
            libc::setpgid(0, 0)
        };
        std::thread::sleep(Duration::from_secs(300));
        return Ok(());
    }
    pipette_subprocess::install_signal_handler();
    if std::env::args().nth(2).as_deref() == Some("--stop") {
        let stop = Arc::new(AtomicBool::new(false));
        pipette_subprocess::register_stop_flag(&stop);
        let mut cmd = Command::new("sh");
        cmd.args(["-c", r#"echo $$ > "$0"; sleep 2"#, &arg]);
        pipette_subprocess::output_within(cmd, Duration::from_secs(300))?;
        std::process::exit(if stop.load(Ordering::SeqCst) { 0 } else { 2 });
    }
    let mut cmd = Command::new("sh");
    cmd.args(["-c", r#""$1" --sleep & echo $! > "$0"; wait"#, &arg]);
    cmd.arg(std::env::current_exe().unwrap_or_default());
    pipette_subprocess::output_within(cmd, Duration::from_secs(300))?;
    // The ^C handler kills the command, then exits; returning first would end
    // the run with another status.
    std::thread::sleep(Duration::from_secs(300));
    Ok(())
}
