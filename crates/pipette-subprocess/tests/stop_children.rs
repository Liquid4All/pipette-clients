//! A test binary of its own: [`pipette_subprocess::stop_children`] stops every
//! child the process runs.
#![cfg(unix)]

use std::os::unix::process::ExitStatusExt;
use std::process::Command;

/// A helper talked to over its pipes is a child like any other: the stop
/// reaches it.
#[test]
fn a_piped_helper_is_stopped_with_the_children() -> anyhow::Result<()> {
    let (mut helper, _teardown) = pipette_subprocess::spawn_piped(Command::new("cat"))?;
    let stopped = pipette_subprocess::stop_children();
    let status = helper.wait()?;
    anyhow::ensure!(stopped == 1, "stopped {stopped} children");
    anyhow::ensure!(status.signal() == Some(libc::SIGKILL), "{status}");
    Ok(())
}
