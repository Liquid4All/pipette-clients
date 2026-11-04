//! The venv's tokenizer: exact-length prompts and EOS ids.
//!
//! `mlx_lm.server` exposes no `/tokenize` endpoint and takes string prompts,
//! so exact-length prompts and EOS suppression need the tokenizer directly.
//! One helper process per session loads only the tokenizer (no model weights)
//! and answers a count per line: prompt building makes dozens of counts, and
//! re-importing `transformers` for each would dominate the cell's setup.

use std::{
    io::{BufRead, BufReader, Read, Write},
    path::Path,
    process::{Child, ChildStdin, ChildStdout, Command, Stdio},
    thread::{self, JoinHandle},
};

use anyhow::Context;
use serde::Deserialize;

const TOKENIZE_SCRIPT: &str = include_str!("python/pipette_mlx_tokenize.py");

/// A prompt the server tokenizes to exactly `tokens`, counted with the venv's
/// tokenizer (the one the server runs, so the count agrees with its `usage`
/// block), and that tokenizer's EOS ids. `trust_remote_code` is the server's
/// own: a model's own code can load a different tokenizer.
pub(crate) fn prompt(
    venv_python: &Path,
    model_dir: &Path,
    tokens: u32,
    trust_remote_code: bool,
) -> anyhow::Result<(String, Vec<u32>)> {
    let mut tokenizer = TokenizerSession::start(venv_python, model_dir, trust_remote_code)?;
    let text = pipette_ops::prompt_seed::build_prompt_text(tokens, |text| tokenizer.count(text))?;
    Ok((text, tokenizer.eos_ids().to_vec()))
}

#[derive(Debug, Deserialize)]
struct Greeting {
    eos_ids: Vec<u32>,
}

#[derive(Debug, Deserialize)]
struct CountReply {
    count: usize,
}

/// A live helper process with the tokenizer loaded; dropping it kills and
/// reaps the helper.
struct TokenizerSession {
    child: Child,
    stdin: Option<ChildStdin>,
    stdout: BufReader<ChildStdout>,
    // Drained on its own thread so a chatty tokenizer load cannot fill the
    // pipe and stall the helper; read back only to explain a failure.
    stderr: Option<JoinHandle<String>>,
    eos_ids: Vec<u32>,
}

impl TokenizerSession {
    fn start(
        venv_python: &Path,
        model_dir: &Path,
        trust_remote_code: bool,
    ) -> anyhow::Result<Self> {
        let mut command = Command::new(venv_python);
        command
            .arg("-c")
            .arg(TOKENIZE_SCRIPT)
            .arg("--model-dir")
            .arg(model_dir);
        if trust_remote_code {
            command.arg("--trust-remote-code");
        }
        #[expect(
            clippy::disallowed_methods,
            reason = "every stream is piped: the helper talks over stdin and stdout"
        )]
        let mut child = command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .with_context(|| {
                format!(
                    "failed to spawn tokenizer helper via {}",
                    venv_python.display()
                )
            })?;
        let stdin = child.stdin.take();
        let stdout = child.stdout.take();
        let stderr = child.stderr.take().map(|mut pipe| {
            thread::spawn(move || {
                let mut text = String::new();
                let _ = pipe.read_to_string(&mut text);
                text
            })
        });
        let mut session = Self {
            child,
            stdin,
            stdout: BufReader::new(stdout.context("tokenizer helper stdout missing")?),
            stderr,
            eos_ids: Vec::new(),
        };
        // A write to a helper that has gone fails with EPIPE rather than
        // raising the SIGPIPE the CLI makes fatal. This crate builds for macOS
        // alone, where the guard sits on the pipe, so once covers every write.
        if let Some(stdin) = &session.stdin {
            pipette_subprocess::no_sigpipe_on(stdin)
                .context("failed to guard the tokenizer helper's stdin")?;
        }
        let greeting: Greeting = session.read_reply()?;
        session.eos_ids = greeting.eos_ids;
        Ok(session)
    }

    /// Tokenizer EOS ids, for `logit_bias` suppression of early stops.
    fn eos_ids(&self) -> &[u32] {
        &self.eos_ids
    }

    /// Token count of `text` under the venv's tokenizer.
    fn count(&mut self, text: &str) -> anyhow::Result<usize> {
        let line = serde_json::json!({ "text": text }).to_string();
        let stdin = self
            .stdin
            .as_mut()
            .context("tokenizer helper stdin missing")?;
        let written = writeln!(stdin, "{line}").and_then(|()| stdin.flush());
        if let Err(err) = written {
            return Err(self.failure(&format!("failed to write request: {err}")));
        }
        self.read_reply::<CountReply>().map(|reply| reply.count)
    }

    fn read_reply<T: for<'de> Deserialize<'de>>(&mut self) -> anyhow::Result<T> {
        let mut line = String::new();
        match self.stdout.read_line(&mut line) {
            Ok(0) => Err(self.failure("exited without replying")),
            Ok(_) => serde_json::from_str(&line).with_context(|| {
                format!("failed to parse tokenizer helper reply: {}", line.trim())
            }),
            Err(err) => Err(self.failure(&format!("failed to read reply: {err}"))),
        }
    }

    /// Reap the helper and report `what` with its stderr, which holds the
    /// Python traceback when the tokenizer failed to load or a request broke.
    fn failure(&mut self, what: &str) -> anyhow::Error {
        self.stdin = None;
        let _ = self.child.kill();
        let status = self
            .child
            .wait()
            .map_or_else(|err| format!("unknown status ({err})"), |s| s.to_string());
        let stderr = self
            .stderr
            .take()
            .and_then(|handle| handle.join().ok())
            .unwrap_or_default();
        anyhow::anyhow!(
            "tokenizer helper {what} (exited with {status}): {}",
            stderr.trim()
        )
    }
}

impl Drop for TokenizerSession {
    fn drop(&mut self) {
        self.stdin = None;
        let _ = self.child.kill();
        let _ = self.child.wait();
        if let Some(handle) = self.stderr.take() {
            let _ = handle.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::os::unix::fs::PermissionsExt;
    use std::path::PathBuf;

    use super::*;

    /// Stands in for the venv interpreter, running `body` and ignoring the
    /// script argument, so the session is tested without Python.
    fn fake_python(dir: &Path, body: &str) -> anyhow::Result<PathBuf> {
        let path = dir.join("python");
        fs::write(&path, format!("#!/bin/sh\n{body}\n"))?;
        fs::set_permissions(&path, fs::Permissions::from_mode(0o755))?;
        Ok(path)
    }

    /// The fake counts each request line's length less its 11-byte JSON wrapper.
    #[test]
    fn session_reports_eos_ids_and_answers_counts_in_order() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let python = fake_python(
            dir.path(),
            r#"echo '{"eos_ids": [2, 7]}'
while IFS= read -r line; do echo "{\"count\": $(( ${#line} - 11 ))}"; done"#,
        )?;
        let mut session = TokenizerSession::start(&python, dir.path(), true)?;
        assert_eq!(session.eos_ids(), &[2, 7]);
        assert_eq!(session.count("abc")?, 3);
        assert_eq!(session.count("hello")?, 5);
        Ok(())
    }

    /// `--trust-remote-code` reaches the helper exactly when asked for.
    #[rstest::rstest]
    #[case::text_server(true, 1)]
    #[case::vision_server(false, 0)]
    fn the_servers_remote_code_setting_reaches_the_helper(
        #[case] trust_remote_code: bool,
        #[case] flagged: u32,
    ) -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let python = fake_python(
            dir.path(),
            r#"case " $* " in *" --trust-remote-code "*) echo '{"eos_ids": [1]}';; *) echo '{"eos_ids": [0]}';; esac"#,
        )?;
        let session = TokenizerSession::start(&python, dir.path(), trust_remote_code)?;
        assert_eq!(session.eos_ids(), &[flagged]);
        Ok(())
    }

    #[test]
    fn a_helper_that_dies_reports_its_stderr() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let python = fake_python(dir.path(), "echo 'no tokenizer here' >&2\nexit 3")?;
        let err = TokenizerSession::start(&python, dir.path(), true)
            .err()
            .context("a helper that exits before its greeting must fail")?;
        let msg = format!("{err:#}");
        assert!(
            msg.contains("exited without replying") && msg.contains("no tokenizer here"),
            "unexpected error: {msg}"
        );
        Ok(())
    }
}
