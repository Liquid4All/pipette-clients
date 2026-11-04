//! Each line a child prints is logged at debug, inside a span with the name it
//! was started under and its pid. Its own test binary: the subscriber is
//! process-wide.

use std::io::Write;
use std::process::Command;
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use anyhow::Context;
use serde_json::{Value, json};

use pipette_subprocess::start_process;

/// Collects what the subscriber writes.
#[derive(Clone)]
struct Sink(Arc<Mutex<Vec<u8>>>);

impl Write for Sink {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[cfg(unix)]
const PROGRAM: &str = "sh";
#[cfg(windows)]
const PROGRAM: &str = "powershell";

fn prints_to_stderr(line: &str) -> Command {
    let mut command = Command::new(PROGRAM);
    #[cfg(unix)]
    command.arg("-c").arg(format!("echo {line} >&2"));
    #[cfg(windows)]
    command.args([
        "-NoProfile",
        "-NonInteractive",
        "-Command",
        &format!("[Console]::Error.WriteLine('{line}')"),
    ]);
    command
}

#[test]
fn each_line_is_logged_inside_the_processs_span() -> anyhow::Result<()> {
    let sink = Sink(Arc::new(Mutex::new(Vec::new())));
    let writer = sink.clone();
    tracing_subscriber::fmt()
        .json()
        .flatten_event(true)
        .with_current_span(false)
        .with_span_list(true)
        .with_max_level(tracing::Level::DEBUG)
        .with_writer(move || writer.clone())
        .try_init()
        .map_err(|err| anyhow::anyhow!("{err}"))?;

    let mut process = start_process(
        "echo-server",
        prints_to_stderr("hello-from-the-child"),
        Duration::ZERO,
    )?;
    process
        .wait(Duration::from_secs(30))?
        .context("still running")?;
    let pid = process.pid();

    let written = sink
        .0
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .clone();
    let event = String::from_utf8(written)?
        .lines()
        .filter_map(|line| serde_json::from_str::<Value>(line).ok())
        .find(|event| event["message"] == "hello-from-the-child")
        .context("the line was not logged")?;
    assert_eq!(event["level"], "DEBUG");
    assert_eq!(event["target"], "pipette_subprocess::output");
    assert_eq!(event["stream"], "stderr");
    assert_eq!(
        event["spans"],
        json!([{ "name": "process", "program": "echo-server", "pid": pid }])
    );
    Ok(())
}
