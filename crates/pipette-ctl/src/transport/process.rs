use std::{
    io::{BufRead, BufReader, Write},
    process::{Command, Stdio},
};

use anyhow::Context;

use crate::transport::ExecOutput;

/// Stream stdout/stderr to the local terminal, each line tagged `[prefix] `
/// when set.
pub(crate) fn run_streaming(
    program: &str,
    args: &[&str],
    prefix: Option<&str>,
) -> anyhow::Result<ExecOutput> {
    run_streaming_scanning(program, args, prefix, None).map(|(out, _)| out)
}

/// Like [`run_streaming`], but also returns the last `<sentinel> <int>` value
/// on stdout (a bare `<sentinel>` reads as `0`).
///
/// iOS needs this: `xcrun devicectl … --console` does not reliably pass on the
/// app's exit code, so the caller trusts the app's `BENCH_DONE <status>`.
/// The failure is intermittent: six identical refused launches (Xcode 26.2,
/// 2026-08-03) exited `2, 2, 2, 2, 2, 0`. A few green samples (three on Xcode
/// 26.6) do not prove the exit code works; measure a hundred before dropping
/// the sentinel.
pub(crate) fn run_streaming_scanning(
    program: &str,
    args: &[&str],
    prefix: Option<&str>,
    sentinel: Option<&str>,
) -> anyhow::Result<(ExecOutput, Option<i32>)> {
    let mut scanned: Option<i32> = None;
    #[expect(
        clippy::disallowed_methods,
        reason = "both streams are piped and read below"
    )]
    let mut child = Command::new(program)
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| format!("spawning {program}"))?;

    let child_stdout = child.stdout.take().context("stdout not piped")?;
    let child_stderr = child.stderr.take().context("stderr not piped")?;

    let prefix_bytes: Option<Vec<u8>> = prefix.map(|p| format!("[{p}] ").into_bytes());

    // stderr on its own thread so both pipes drain concurrently. Drops the
    // PowerShell CLIXML noise Windows SSH emits on stderr.
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

    {
        let reader = BufReader::new(child_stdout);
        for line in reader.split(b'\n') {
            match line {
                Ok(bytes) => {
                    if let Some(sent) = sentinel
                        && let Some(status) = scan_sentinel(&bytes, sent)
                    {
                        scanned = Some(status);
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

/// Matched anywhere in the line: the iOS app's logger prepends `[HEADLESS] `.
fn scan_sentinel(bytes: &[u8], sentinel: &str) -> Option<i32> {
    let text = std::str::from_utf8(bytes).ok()?;
    let after = text.split(sentinel).nth(1)?;
    match after.split_whitespace().next() {
        None => Some(0),
        Some(tok) => tok.parse().ok(),
    }
}

/// Discards stdout/stderr.
pub(crate) fn run_quiet(program: &str, args: &[&str]) -> anyhow::Result<ExecOutput> {
    #[expect(clippy::disallowed_methods, reason = "both streams are discarded")]
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

/// Windows SSH wraps progress/error output as `#< CLIXML` followed by XML
/// ending with `</Objs>`; true for every line of that block.
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
        // Bare sentinel reads as success, for an older iOS build that writes no
        // status; there a refused launch is indistinguishable from a finished cell.
        assert_eq!(scan_sentinel(b"BENCH_DONE", "BENCH_DONE"), Some(0));
        assert_eq!(scan_sentinel(b"  BENCH_DONE 1  ", "BENCH_DONE"), Some(1));
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
        assert_eq!(scan_sentinel(b"[HEADLESS] running", "BENCH_DONE"), None);
        assert_eq!(scan_sentinel(b"BENCH_DONE ok", "BENCH_DONE"), None);
    }
}
