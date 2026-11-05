//! A write to a child that has closed its stdin, under the default `SIGPIPE`
//! disposition the CLIs restore. Its own test binary because that disposition
//! is process-wide: the harness ignores `SIGPIPE`, so an in-crate test would
//! pass without the guard.
#![cfg(unix)]

use std::io::{ErrorKind, Write};
use std::process::{Command, Stdio};

use anyhow::Context;

/// Larger than any pipe buffer, so the write is still in flight when the
/// reader goes.
const LARGE: usize = 2 * 1024 * 1024;

#[test]
fn a_guarded_write_to_a_closed_stdin_fails_instead_of_killing_the_process() -> anyhow::Result<()> {
    pipette_subprocess::reset_sigpipe();
    #[expect(
        clippy::disallowed_methods,
        reason = "the shell and its sleep print nothing"
    )]
    let mut child = Command::new("/bin/sh")
        .arg("-c")
        .arg("exec 0<&-; exec sleep 30")
        .stdin(Stdio::piped())
        .spawn()?;
    let mut stdin = child.stdin.take().context("stdin was piped")?;
    let written = std::thread::spawn(move || -> std::io::Result<()> {
        pipette_subprocess::no_sigpipe_on(&stdin)?;
        stdin.write_all(&vec![b'x'; LARGE])
    })
    .join()
    .map_err(|_| anyhow::anyhow!("the writer thread panicked"))?;
    child.kill()?;
    child.wait()?;

    let Err(err) = written else {
        anyhow::bail!("expected the write to fail");
    };
    assert_eq!(err.kind(), ErrorKind::BrokenPipe, "{err}");
    Ok(())
}
