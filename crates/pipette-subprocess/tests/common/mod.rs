//! Helpers for the tests that start a helper bin and watch what it leaves
//! behind.
#![allow(dead_code, reason = "each test crate uses a subset of these helpers")]

use std::io::{BufRead, BufReader};
use std::process::Child;
use std::str::FromStr;

use anyhow::Context;

/// Read the pid that `run` prints on its first stdout line. If the read
/// fails, kill `run`, so it does not outlive the test.
pub fn read_pid<T>(run: &mut Child) -> anyhow::Result<T>
where
    T: FromStr,
    T::Err: std::error::Error + Send + Sync + 'static,
{
    let mut line = String::new();
    let pid = run
        .stdout
        .take()
        .context("stdout was piped")
        .and_then(|stdout| Ok(BufReader::new(stdout).read_line(&mut line)?))
        .and_then(|_| line.trim().parse().context("no pid printed"));
    if pid.is_err() {
        let _ = run.kill();
        let _ = run.wait();
    }
    pid
}

/// Whether `pid` is gone within a few seconds.
#[cfg(unix)]
pub fn gone(pid: libc::pid_t) -> bool {
    use std::thread;
    use std::time::{Duration, Instant};

    let started = Instant::now();
    while started.elapsed() < Duration::from_secs(5) {
        // SAFETY: signal 0 only asks whether the pid exists.
        if unsafe { libc::kill(pid, 0) } != 0 {
            return true;
        }
        thread::sleep(Duration::from_millis(20));
    }
    false
}
