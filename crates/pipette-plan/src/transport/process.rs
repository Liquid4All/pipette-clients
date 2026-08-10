use std::{
    io::{BufRead, BufReader, Write},
    process::{Command, Stdio},
};

use anyhow::Context;

use crate::transport::ExecOutput;

/// Spawn a process and stream its stdout/stderr to the local terminal in real
/// time.  When `prefix` is set, each output line is tagged with `[prefix] `.
/// Returns only the exit code.
pub(crate) fn run_streaming(
    program: &str,
    args: &[&str],
    prefix: Option<&str>,
) -> anyhow::Result<ExecOutput> {
    run_streaming_scanning(program, args, prefix, None).map(|(out, _)| out)
}

/// Like [`run_streaming`], but also scans stdout for a `sentinel` line
/// of the form `<sentinel> <int>` (or a bare `<sentinel>`, read as `0`)
/// and returns the last such value alongside the process exit code.
///
/// iOS uses this: `xcrun devicectl … --console` does not reliably
/// propagate the launched app's exit code, but the app prints
/// `BENCH_DONE <status>` as its result contract, so the caller trusts
/// the scanned status over the process exit code when present.
///
/// "Not reliably" is intermittent, not absent, which is the trap: six identical refused
/// launches on the fleet's Mac (Xcode 26.2, 2026-08-03) exited `2, 2, 2, 2, 2, 0` while
/// every one of them printed its refusal. A handful of green samples — three on Xcode
/// 26.6 — reads as "the exit code works, drop the sentinel", and the sixth run then
/// records a refused cell as measured. Measure a hundred before believing that.
pub(crate) fn run_streaming_scanning(
    program: &str,
    args: &[&str],
    prefix: Option<&str>,
    sentinel: Option<&str>,
) -> anyhow::Result<(ExecOutput, Option<i32>)> {
    let mut scanned: Option<i32> = None;
    let mut child = Command::new(program)
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| format!("spawning {program}"))?;

    let child_stdout = child.stdout.take().context("stdout not piped")?;
    let child_stderr = child.stderr.take().context("stderr not piped")?;

    let prefix_bytes: Option<Vec<u8>> = prefix.map(|p| format!("[{p}] ").into_bytes());

    // Stream stderr on a background thread so both pipes drain concurrently.
    // Filter out PowerShell CLIXML noise that Windows SSH emits on stderr.
    let err_prefix = prefix_bytes.clone();
    let stderr_thread = std::thread::spawn(move || {
        let reader = BufReader::new(child_stderr);
        let mut in_clixml = false;
        for line in reader.split(b'\n') {
            match line {
                Ok(bytes) => {
                    if is_clixml_line(&bytes, &mut in_clixml) {
                        continue;
                    }
                    let mut local_err = std::io::stderr().lock();
                    if let Some(pfx) = &err_prefix {
                        let _ = local_err.write_all(pfx);
                    }
                    let _ = local_err.write_all(&bytes);
                    let _ = local_err.write_all(b"\n");
                    let _ = local_err.flush();
                }
                Err(_) => break,
            }
        }
    });

    // Stream stdout on the main thread.
    {
        let reader = BufReader::new(child_stdout);
        for line in reader.split(b'\n') {
            match line {
                Ok(bytes) => {
                    if let Some(sent) = sentinel {
                        if let Some(status) = scan_sentinel(&bytes, sent) {
                            scanned = Some(status);
                        }
                    }
                    let mut local_out = std::io::stdout().lock();
                    if let Some(pfx) = &prefix_bytes {
                        let _ = local_out.write_all(pfx);
                    }
                    let _ = local_out.write_all(&bytes);
                    let _ = local_out.write_all(b"\n");
                    let _ = local_out.flush();
                }
                Err(_) => break,
            }
        }
    }

    let _ = stderr_thread.join();
    let exit = child.wait().context("waiting for child process")?;
    let status = exit.code().unwrap_or(1);
    Ok((ExecOutput { status }, scanned))
}

/// Parse a `<sentinel> [<int>]` stdout line into its status: the
/// trailing integer, or `0` when the sentinel carries no number.
/// Returns `None` for lines without the sentinel.
///
/// The sentinel is matched anywhere in the line, not just at the start:
/// the iOS app emits it through a logger that prepends `[HEADLESS] `, so
/// the real line is `[HEADLESS] BENCH_DONE <status>`.
fn scan_sentinel(bytes: &[u8], sentinel: &str) -> Option<i32> {
    let text = std::str::from_utf8(bytes).ok()?;
    let after = text.split(sentinel).nth(1)?;
    match after.split_whitespace().next() {
        None => Some(0),
        Some(tok) => tok.parse().ok(),
    }
}

/// Spawn a process and discard its stdout/stderr. Returns only the exit code.
pub(crate) fn run_quiet(program: &str, args: &[&str]) -> anyhow::Result<ExecOutput> {
    let status = Command::new(program)
        .args(args)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .with_context(|| format!("spawning {program}"))?
        .code()
        .unwrap_or(1);
    Ok(ExecOutput { status })
}

/// Detect and skip PowerShell CLIXML blocks on stderr.
///
/// Windows SSH wraps progress/error output as `#< CLIXML` followed by XML
/// ending with `</Objs>`. Skip the entire block.
fn is_clixml_line(bytes: &[u8], in_clixml: &mut bool) -> bool {
    if bytes.starts_with(b"#< CLIXML") {
        *in_clixml = true;
        return true;
    }
    if *in_clixml {
        if bytes.windows(7).any(|w| w == b"</Objs>") {
            *in_clixml = false;
        }
        return true;
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scan_sentinel_parses_status() {
        assert_eq!(scan_sentinel(b"BENCH_DONE 0", "BENCH_DONE"), Some(0));
        assert_eq!(scan_sentinel(b"BENCH_DONE 3", "BENCH_DONE"), Some(3));
        // Bare sentinel with no number reads as success. The iOS client always writes a
        // status now; this stays for a device still on an older build, where the app's
        // own exit code is the only other signal and `devicectl` does not pass it on.
        // A refused invocation there is indistinguishable from a completed cell.
        assert_eq!(scan_sentinel(b"BENCH_DONE", "BENCH_DONE"), Some(0));
        // Surrounding whitespace is tolerated.
        assert_eq!(scan_sentinel(b"  BENCH_DONE 1  ", "BENCH_DONE"), Some(1));
        // The real device form: the app's logger prepends `[HEADLESS] `.
        assert_eq!(
            scan_sentinel(b"[HEADLESS] BENCH_DONE 0", "BENCH_DONE"),
            Some(0)
        );
        assert_eq!(
            scan_sentinel(b"[HEADLESS] BENCH_DONE 2", "BENCH_DONE"),
            Some(2)
        );
        assert_eq!(
            scan_sentinel(b"[HEADLESS] BENCH_DONE", "BENCH_DONE"),
            Some(0)
        );
        // Non-sentinel lines don't match.
        assert_eq!(scan_sentinel(b"[HEADLESS] running", "BENCH_DONE"), None);
        // A non-numeric tail is ignored (no false status).
        assert_eq!(scan_sentinel(b"BENCH_DONE ok", "BENCH_DONE"), None);
    }
}
