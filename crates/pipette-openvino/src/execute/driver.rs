//! Invoking the one-shot Python driver: one process, one compiled pipeline,
//! one JSON result.
//!
//! The driver script ships in the binary via `include_str!` and is written to a
//! temp file per invocation, the same way the MLX backend ships its server.
//! Unlike MLX there is no long-lived process and no HTTP — see `super` for the
//! reason — so stdin/stdout is enough.

use std::io::{BufRead, Read, Write};
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::Context;
use serde::{Deserialize, Serialize};

const DRIVER_PY: &str = include_str!("../python/pipette_openvino_driver.py");
pub(super) const DRIVER_FILENAME: &str = "pipette_openvino_driver.py";

/// Hard ceiling on one driver invocation.
///
/// A hang detector, not a performance bound — deliberately generous, because a
/// large model's NPU compile is legitimately slow (17.7s for a 350M at the
/// default shape, 62.8s at `MAX_PROMPT_LEN` 4096, and unmeasured above that).
/// It exists because the device this backend targets is documented to wedge:
/// `ZE_RESULT_ERROR_DEVICE_LOST` took it down during bring-up. Isolating each
/// rep in its own process survives a device *loss*; without this it would still
/// hang forever on a device that stops responding. Matches the budget
/// `pipette-mlx` puts on its own engine calls.
const DRIVER_TIMEOUT: Duration = Duration::from_secs(3600);

/// How long to wait for a tokenizer to exit once its stdin is closed. Short
/// because nothing is left to finish: the driver's read loop returns on EOF.
const TOKENIZER_SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(30);

/// How often to check whether the driver has exited. Coarse on purpose: the
/// work being waited on is seconds to minutes long.
const POLL_INTERVAL: Duration = Duration::from_millis(200);

/// Marker the driver prefixes its one result line with, so a stray write to
/// stdout by OpenVINO or its telemetry cannot be parsed as the result.
const RESULT_PREFIX: &str = "PIPETTE_RESULT ";

/// What the driver should do with the pipeline it compiles.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum Mode {
    /// Generate a single token: the reported time is time-to-first-token, i.e.
    /// prefill. There is no prefill-only entry point in GenAI.
    Prefill,
    /// Generate `decode_tokens` and report per-token timing.
    Decode,
    /// Load the tokenizer only and answer token counts until stdin closes.
    /// Compiles no pipeline, so it is safe to hold open next to the NPU.
    Tokenize,
    /// Compile the pipeline and exit, so the cache holds a blob before the
    /// first measured rep. Generates nothing and reports no timing.
    Compile,
}

/// The shape of a warm-up pass — the shared one, not the cell's, so an 8k
/// prefill does not pay for its own rehearsal.
#[derive(Debug, Clone, Copy, Serialize)]
pub(super) struct WarmupShape {
    pub prefill_tokens: u32,
    pub decode_tokens: u32,
}

/// One driver invocation.
#[derive(Debug, Serialize)]
pub(super) struct DriverRequest<'a> {
    pub model_dir: &'a str,
    pub device: &'a str,
    pub mode: Mode,
    pub prefill_tokens: u32,
    pub decode_tokens: u32,
    /// An untimed generate before the measured one, or `None` to go straight
    /// to the measurement.
    ///
    /// Per rep, not per series: each rep is a fresh process, so there is
    /// nothing warm to carry. `None` only for `max_memory_usage`, which is not
    /// a timing cell.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub warmup: Option<WarmupShape>,
    /// The prompt as text, for a cell that measures tokenization as part of
    /// its number. The driver encodes it inside the timed region; absent, the
    /// driver builds the prompt from repeated seed token ids instead.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt: Option<&'a str>,
    /// Extra `LLMPipeline` properties (`MAX_PROMPT_LEN`, `GENERATE_HINT`, …).
    pub properties: serde_json::Map<String, serde_json::Value>,
    /// The shared prompt corpus every backend builds its prompts from.
    ///
    /// Sent on stdin rather than in the environment: it is ~24 KB against a
    /// documented Windows ceiling of 32,767 characters for one variable, and
    /// Windows is this runtime's primary host. stdin has no such bound.
    pub prompt_seed: &'static str,
}

/// What the driver reports back.
///
/// `tpot_ms` / `throughput_tps` are absent for a single generated token, where
/// a per-token rate is undefined — the driver omits them rather than
/// forwarding OpenVINO's placeholder.
#[derive(Debug, Deserialize)]
pub(super) struct DriverResult {
    pub compile_s: f64,
    pub wall_ms: f64,
    pub input_tokens: u32,
    pub generated_tokens: u32,
    pub ttft_ms: f64,
    #[serde(default)]
    pub tpot_ms: Option<f64>,
    #[serde(default)]
    pub throughput_tps: Option<f64>,
    /// `None` when the platform peak-RSS counter was unavailable — a missing
    /// number rather than a wrong one.
    #[serde(default)]
    pub peak_host_bytes: Option<u64>,
}

/// The driver's failure form. It reports errors as data so a compile refusal on
/// the NPU reads as a benchmark failure with a reason, not a parse error.
#[derive(Debug, Deserialize)]
struct DriverError {
    kind: String,
    error: String,
}

/// Captured output of one invocation, carried into the `RunResponse` so an
/// operator sees what the driver saw.
#[derive(Debug, Default)]
pub(super) struct DriverOutput {
    pub stdout: String,
    pub stderr: String,
}

/// The driver script materialized next to the run, kept alive for as long as
/// invocations need it.
pub(super) struct DriverScript {
    _dir: tempfile::TempDir,
    path: std::path::PathBuf,
}

impl DriverScript {
    pub fn materialize() -> anyhow::Result<Self> {
        Self::write(DRIVER_PY)
    }

    fn write(contents: &str) -> anyhow::Result<Self> {
        let dir = tempfile::tempdir().context("creating a temp dir for the OpenVINO driver")?;
        let path = dir.path().join(DRIVER_FILENAME);
        std::fs::write(&path, contents)
            .with_context(|| format!("writing the OpenVINO driver to {}", path.display()))?;
        Ok(Self { _dir: dir, path })
    }

    /// Compile the pipeline once and throw the result away, so the blob cache
    /// holds an entry before any measured rep runs.
    ///
    /// Its own process, like every other compile: the device is documented to
    /// fail when one process compiles more than once.
    pub fn precompile(&self, python: &Path, request: &DriverRequest<'_>) -> anyhow::Result<()> {
        let (result, _) = self
            .invoke(python, request)
            .context("warming the OpenVINO blob cache")?;
        log::info!("openvino cache warm: compile {:.1}s", result.compile_s);
        Ok(())
    }

    /// Run one invocation to completion and parse its result.
    ///
    /// Every call is a fresh process. That is the point: it is what keeps two
    /// NPU cells in one plan from sharing a compiled pipeline.
    pub fn invoke(
        &self,
        python: &Path,
        request: &DriverRequest<'_>,
    ) -> anyhow::Result<(DriverResult, DriverOutput)> {
        // Newline-terminated: the driver reads its request with `readline`, so
        // that tokenize mode can keep reading afterwards.
        let mut body = serde_json::to_vec(request).context("serializing the driver request")?;
        body.push(b'\n');

        let mut cmd = Command::new(python);
        cmd.arg(&self.path)
            // OpenVINO emits non-ASCII in some diagnostics and Windows consoles
            // default to a legacy code page, which turns a log line into a
            // UnicodeEncodeError that kills the run.
            .env("PYTHONIOENCODING", "utf-8")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        pipette_subprocess::echo_info(&cmd);

        let mut child = cmd
            .spawn()
            .with_context(|| format!("spawning the OpenVINO driver via {}", python.display()))?;

        // Both pipes are drained on their own threads for the whole run. Polling
        // for exit while leaving them buffered would deadlock a driver that
        // out-writes the pipe capacity — and OpenVINO is chatty on stderr.
        let mut stdout = child
            .stdout
            .take()
            .context("the OpenVINO driver has no stdout")?;
        let mut stderr = child
            .stderr
            .take()
            .context("the OpenVINO driver has no stderr")?;
        let stdout_reader = thread::spawn(move || {
            let mut buf = Vec::new();
            let _ = stdout.read_to_end(&mut buf);
            buf
        });
        let stderr_reader = thread::spawn(move || {
            let mut buf = Vec::new();
            let _ = stderr.read_to_end(&mut buf);
            buf
        });

        child
            .stdin
            .take()
            .context("the OpenVINO driver has no stdin")?
            .write_all(&body)
            .context("writing the request to the OpenVINO driver")?;

        let waited = wait_with_deadline(&mut child, DRIVER_TIMEOUT);

        let join = |reader: thread::JoinHandle<Vec<u8>>| {
            reader
                .join()
                .map(|buf| String::from_utf8_lossy(&buf).into_owned())
                .unwrap_or_default()
        };
        let output = DriverOutput {
            stdout: join(stdout_reader),
            stderr: join(stderr_reader),
        };

        let status = waited.with_context(|| {
            format!(
                "the OpenVINO driver did not finish within {}s and was killed{}",
                DRIVER_TIMEOUT.as_secs(),
                tail(&output.stderr)
            )
        })?;
        let result = parse_result(&output.stdout).with_context(|| {
            format!(
                "the OpenVINO driver exited {} without a usable result{}",
                status,
                tail(&output.stderr)
            )
        })?;
        // Compile time is the number that differs most by device — about a
        // second on CPU against ~18s on NPU — and none of it reaches the
        // reported metric, so surface it here or it is invisible.
        log::info!(
            "openvino driver: compile {:.1}s, wall {:.0}ms, {} in / {} out{}",
            result.compile_s,
            result.wall_ms,
            result.input_tokens,
            result.generated_tokens,
            result
                .throughput_tps
                .map(|tps| format!(", {tps:.1} tok/s"))
                .unwrap_or_default()
        );
        Ok((result, output))
    }
}

/// A driver held open in tokenize mode, answering token counts.
///
/// The convergence loop stays in Rust — `pipette_ops::prompt_seed` owns it, and
/// llamacpp/MLX/torch-oai all drive it the same way — so what this adds is only
/// the transport the other three get from their server's `/tokenize`. One
/// process per cell, not per rep: it compiles no pipeline, so holding it open
/// costs nothing the measured reps care about.
pub(super) struct TokenizeSession {
    child: Child,
    /// `None` only while dropping: closing it is what ends the driver's read
    /// loop, so it has to be dropped before the wait.
    stdin: Option<std::process::ChildStdin>,
    /// Lines the reader thread has drained, marked and unmarked alike.
    lines: std::sync::mpsc::Receiver<String>,
    reader: Option<thread::JoinHandle<()>>,
}

impl TokenizeSession {
    /// Start a tokenizer for `model_dir` under `python`, returning once it has
    /// confirmed the tokenizer loaded.
    ///
    /// Waiting for that confirmation is what keeps a bad model directory from
    /// being reported against the first prompt that happens to be measured
    /// against it.
    pub fn start(script: &DriverScript, python: &Path, model_dir: &str) -> anyhow::Result<Self> {
        let mut cmd = Command::new(python);
        cmd.arg(&script.path)
            .env("PYTHONIOENCODING", "utf-8")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            // Inherited: the tokenizer's diagnostics belong in the run log, and
            // nothing here parses them.
            .stderr(Stdio::inherit());
        pipette_subprocess::echo_info(&cmd);

        let mut child = cmd
            .spawn()
            .with_context(|| format!("spawning the OpenVINO tokenizer via {}", python.display()))?;
        let mut stdin = child
            .stdin
            .take()
            .context("the OpenVINO tokenizer has no stdin")?;
        let stdout = child
            .stdout
            .take()
            .context("the OpenVINO tokenizer has no stdout")?;

        // Drained continuously on its own thread for the same reason `invoke`
        // does it: this side reads only when it has asked a question, and a
        // driver that filled the pipe in between would block mid-answer.
        let (tx, lines) = std::sync::mpsc::channel();
        let reader = thread::spawn(move || {
            let _ = std::io::BufReader::new(stdout)
                .lines()
                .map_while(Result::ok)
                .try_for_each(|line| tx.send(line));
        });

        // The mode word comes from the enum, so it cannot drift from the one
        // the measured requests send.
        let request = serde_json::json!({ "mode": Mode::Tokenize, "model_dir": model_dir });
        writeln!(stdin, "{request}").context("starting the OpenVINO tokenizer")?;
        stdin.flush().context("starting the OpenVINO tokenizer")?;

        let session = Self {
            child,
            stdin: Some(stdin),
            lines,
            reader: Some(reader),
        };
        session.next_result("load its tokenizer")?;
        Ok(session)
    }

    /// Tokens in `text`, under the same settings the measured pass will use.
    pub fn count(&mut self, text: &str) -> anyhow::Result<usize> {
        let stdin = self
            .stdin
            .as_mut()
            .context("the OpenVINO tokenizer is shutting down")?;
        let request = serde_json::json!({ "text": text });
        writeln!(stdin, "{request}").context("asking the OpenVINO tokenizer")?;
        stdin.flush().context("asking the OpenVINO tokenizer")?;

        self.next_result("answer")?
            .get("tokens")
            .and_then(serde_json::Value::as_u64)
            .map(|n| n as usize)
            .context("the tokenizer's answer carries no token count")
    }

    /// The next marked result line, within [`DRIVER_TIMEOUT`].
    ///
    /// Bounded for the same reason [`DriverScript::invoke`] is: this backend's
    /// device is documented to wedge, and a tokenizer that stops answering
    /// would otherwise hang the cell with no diagnostic. Unmarked lines are
    /// skipped — the driver's imports print before its first answer.
    fn next_result(&self, what: &str) -> anyhow::Result<serde_json::Value> {
        let deadline = Instant::now() + DRIVER_TIMEOUT;
        loop {
            let left = deadline.saturating_duration_since(Instant::now());
            anyhow::ensure!(
                !left.is_zero(),
                "the OpenVINO tokenizer did not {what} within {}s",
                DRIVER_TIMEOUT.as_secs()
            );
            let line = match self.lines.recv_timeout(left) {
                Ok(line) => line,
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                    anyhow::bail!("the OpenVINO tokenizer exited before it could {what}")
                }
            };
            let Some(marked) = line.trim().strip_prefix(RESULT_PREFIX) else {
                continue;
            };
            let value: serde_json::Value =
                serde_json::from_str(marked).context("the tokenizer's answer is not JSON")?;
            if value.get("ok").and_then(serde_json::Value::as_bool) == Some(false) {
                let err: DriverError =
                    serde_json::from_value(value).context("malformed tokenizer error")?;
                anyhow::bail!(
                    "the OpenVINO tokenizer failed at {}: {}",
                    err.kind,
                    err.error
                );
            }
            return Ok(value);
        }
    }
}

impl Drop for TokenizeSession {
    /// Close stdin so the driver's read loop ends, then reap it. A tokenizer
    /// left running would outlive the run as an orphan.
    fn drop(&mut self) {
        // Dropping our end signals EOF, which returns the child from `main`.
        drop(self.stdin.take());
        // Deadlined, not a bare `wait`: there is nothing left for the child to
        // finish, so one that does not exit is wedged and gets killed rather
        // than blocking the cell that is about to be measured.
        if let Err(err) = wait_with_deadline(&mut self.child, TOKENIZER_SHUTDOWN_TIMEOUT) {
            log::warn!("the OpenVINO tokenizer did not shut down cleanly: {err}");
        }
        if let Some(reader) = self.reader.take() {
            let _ = reader.join();
        }
    }
}

/// Wait for `child`, killing it once `timeout` elapses.
///
/// `std` has no wait-with-deadline, so poll. The child is reaped after the kill
/// so it cannot be left a zombie; the kill error is folded into the timeout
/// error rather than replacing it, because "it hung" is the fact worth
/// reporting either way.
fn wait_with_deadline(
    child: &mut Child,
    timeout: Duration,
) -> anyhow::Result<std::process::ExitStatus> {
    let deadline = Instant::now() + timeout;
    loop {
        if let Some(status) = child.try_wait().context("polling the OpenVINO driver")? {
            return Ok(status);
        }
        if Instant::now() >= deadline {
            let killed = child.kill().and_then(|()| child.wait().map(|_| ()));
            return Err(match killed {
                Ok(()) => anyhow::anyhow!("timed out"),
                Err(err) => anyhow::anyhow!("timed out, and killing it failed: {err}"),
            });
        }
        thread::sleep(POLL_INTERVAL);
    }
}

/// Pull the result object out of the driver's stdout.
///
/// Scans for the marker rather than parsing the whole stream: OpenVINO's
/// telemetry and the occasional library `print` share stdout, so the last
/// marked line is the only thing guaranteed to be ours.
fn parse_result(stdout: &str) -> anyhow::Result<DriverResult> {
    let line = stdout
        .lines()
        .filter_map(|l| l.trim().strip_prefix(RESULT_PREFIX))
        .next_back()
        .context("no PIPETTE_RESULT line on stdout")?;

    let value: serde_json::Value =
        serde_json::from_str(line).context("the PIPETTE_RESULT line is not valid JSON")?;
    if value.get("ok").and_then(serde_json::Value::as_bool) == Some(false) {
        let err: DriverError =
            serde_json::from_value(value).context("malformed driver error object")?;
        anyhow::bail!("the OpenVINO driver failed at {}: {}", err.kind, err.error);
    }
    serde_json::from_value(value).context("malformed driver result object")
}

/// Last few stderr lines, for the context line on a failure.
fn tail(stderr: &str) -> String {
    let lines: Vec<&str> = stderr.lines().filter(|l| !l.trim().is_empty()).collect();
    if lines.is_empty() {
        return String::new();
    }
    let start = lines.len().saturating_sub(5);
    format!("; stderr tail:\n{}", lines[start..].join("\n"))
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    fn request(warmup: Option<WarmupShape>) -> DriverRequest<'static> {
        DriverRequest {
            model_dir: "/tmp/ir",
            device: "CPU",
            mode: Mode::Decode,
            prefill_tokens: 512,
            decode_tokens: 100,
            warmup,
            prompt: None,
            properties: serde_json::Map::new(),
            prompt_seed: "seed",
        }
    }

    /// The driver reads the warm-up shape off the request, so a cell that warms
    /// up has to send both numbers — the driver has no fallback and would skip
    /// the pass entirely.
    #[test]
    fn a_warming_request_carries_the_shape() -> anyhow::Result<()> {
        let json = serde_json::to_value(request(Some(WarmupShape {
            prefill_tokens: 512,
            decode_tokens: 100,
        })))?;
        assert_eq!(json["warmup"]["prefill_tokens"], 512);
        assert_eq!(json["warmup"]["decode_tokens"], 100);
        Ok(())
    }

    /// The key is omitted rather than sent as null: the driver tests it for
    /// truthiness, and `null` reads the same as absent only by luck.
    #[test]
    fn a_non_warming_request_omits_the_key() -> anyhow::Result<()> {
        let json = serde_json::to_value(request(None))?;
        assert!(json.get("warmup").is_none(), "got {json}");
        Ok(())
    }

    #[test]
    fn parses_the_marked_result_line_among_noise() -> anyhow::Result<()> {
        let stdout = "loading model...\n\
             PIPETTE_RESULT {\"ok\":true,\"compile_s\":1.0,\"wall_ms\":10.0,\
             \"input_tokens\":512,\"generated_tokens\":100,\"ttft_ms\":5.0,\
             \"tpot_ms\":2.0,\"throughput_tps\":50.0,\"peak_host_bytes\":123}\n\
             telemetry: done\n";
        let result = parse_result(stdout)?;
        assert_eq!(result.input_tokens, 512);
        assert_eq!(result.generated_tokens, 100);
        assert_eq!(result.peak_host_bytes, Some(123));
        assert_eq!(result.throughput_tps, Some(50.0));
        Ok(())
    }

    /// A prefill rep generates one token, where a per-token rate is undefined.
    /// The absent fields must parse as absent, not default to zero — zero would
    /// look like a real (terrible) measurement.
    #[test]
    fn a_single_token_result_has_no_rate_fields() -> anyhow::Result<()> {
        let stdout = "PIPETTE_RESULT {\"ok\":true,\"compile_s\":1.0,\"wall_ms\":10.0,\
             \"input_tokens\":512,\"generated_tokens\":1,\"ttft_ms\":9.0,\
             \"peak_host_bytes\":null}\n";
        let result = parse_result(stdout)?;
        assert_eq!(result.tpot_ms, None);
        assert_eq!(result.throughput_tps, None);
        assert_eq!(result.peak_host_bytes, None);
        Ok(())
    }

    #[test]
    fn a_driver_error_surfaces_its_kind_and_message() -> anyhow::Result<()> {
        let stdout = "PIPETTE_RESULT {\"ok\":false,\"kind\":\"compile\",\
             \"error\":\"NPU refused the model\"}\n";
        let Err(err) = parse_result(stdout) else {
            anyhow::bail!("expected the driver error to propagate");
        };
        let msg = err.to_string();
        assert!(msg.contains("compile"), "got {msg}");
        assert!(msg.contains("NPU refused the model"), "got {msg}");
        Ok(())
    }

    /// The wire split between the two prompt methods: a cell that measures
    /// tokenization sends `prompt`, and one that measures raw continuation
    /// sends none, leaving the driver to repeat seed ids. Pinned here because
    /// the difference is invisible in the request otherwise — and getting it
    /// backwards would silently change what a benchmark id means.
    #[test]
    fn only_a_text_cell_puts_a_prompt_on_the_wire() -> anyhow::Result<()> {
        let ids = serde_json::to_value(DriverRequest {
            prompt: None,
            ..sample_request()
        })?;
        assert!(ids.get("prompt").is_none(), "{ids}");

        let text = serde_json::to_value(DriverRequest {
            prompt: Some("a prompt"),
            ..sample_request()
        })?;
        assert_eq!(
            text.get("prompt").and_then(serde_json::Value::as_str),
            Some("a prompt")
        );
        Ok(())
    }

    fn sample_request() -> DriverRequest<'static> {
        DriverRequest {
            model_dir: "/tmp/ir",
            device: "CPU",
            mode: Mode::Decode,
            prefill_tokens: 512,
            decode_tokens: 100,
            prompt: None,
            warmup: None,
            properties: serde_json::Map::new(),
            prompt_seed: "seed",
        }
    }

    /// Anything that is not a complete marked line has to fail rather than be
    /// half-read: a crashed rep can leave a truncated line behind, and an
    /// earlier good line must not be resurrected in its place.
    #[rstest]
    #[case::no_marker("loading...\nsegfault\n")]
    #[case::truncated("PIPETTE_RESULT {\"ok\":true,\"compile_s\":")]
    #[case::empty("")]
    fn unusable_stdout_is_an_error(#[case] stdout: &str) {
        assert!(parse_result(stdout).is_err());
    }

    /// The invocation path itself — spawn, feed stdin, drain both pipes, parse.
    /// Driven with `sh` standing in for the interpreter so it runs on a box with
    /// no python and no OpenVINO, the same trick `pipette-llamacpp` plays with
    /// its fake server.
    ///
    /// Unix-only: the stand-in is a shell script, and the code under test is
    /// platform-independent.
    #[cfg(unix)]
    mod invocation {
        use super::*;

        const SHELL: &str = "/bin/sh";

        /// Reads the request off stdin, then answers the way the real driver
        /// does: chatter on both streams around one marked line.
        const FAKE_DRIVER: &str = "\
cat > /dev/null
echo 'loading model...'
echo 'openvino telemetry' >&2
echo 'PIPETTE_RESULT {\"ok\":true,\"compile_s\":1.0,\"wall_ms\":10.0,\"input_tokens\":512,\
\"generated_tokens\":1,\"ttft_ms\":9.0}'
";

        #[test]
        fn a_completed_invocation_yields_its_result_and_both_streams() -> anyhow::Result<()> {
            let script = DriverScript::write(FAKE_DRIVER)?;
            let (result, output) = script.invoke(Path::new(SHELL), &sample_request())?;
            assert_eq!(result.input_tokens, 512);
            assert!(output.stdout.contains("loading model..."), "{output:?}");
            assert!(output.stderr.contains("openvino telemetry"), "{output:?}");
            Ok(())
        }

        /// A driver that exits without a result must name what it printed:
        /// stderr is the only clue when the device or the wheel is at fault.
        #[test]
        fn a_driver_that_prints_no_result_reports_its_stderr() -> anyhow::Result<()> {
            let script = DriverScript::write("cat > /dev/null\necho 'ZE_RESULT_ERROR' >&2\n")?;
            let Err(err) = script.invoke(Path::new(SHELL), &sample_request()) else {
                anyhow::bail!("expected a missing-result error");
            };
            assert!(
                format!("{err:#}").contains("ZE_RESULT_ERROR"),
                "got {err:#}"
            );
            Ok(())
        }

        /// A tokenizer stand-in: confirms it loaded, then answers one count
        /// per candidate — the byte count, which is enough for the convergence
        /// loop to move in the right direction.
        const FAKE_TOKENIZER: &str = r#"
read -r _request
echo 'PIPETTE_RESULT {"ok":true,"ready":true}'
while IFS= read -r line; do
  n=$(printf '%s' "$line" | wc -c)
  echo "PIPETTE_RESULT {\"ok\":true,\"tokens\":$n}"
done
"#;

        /// The session answers repeated counts over one process, which is what
        /// lets the shared convergence loop run against it at all.
        #[test]
        fn a_session_answers_successive_counts() -> anyhow::Result<()> {
            let script = DriverScript::write(FAKE_TOKENIZER)?;
            let mut session = TokenizeSession::start(&script, Path::new(SHELL), "/tmp/ir")?;
            let short = session.count("aa")?;
            let long = session.count("aaaaaaaa")?;
            assert!(short > 0 && long > short, "got {short} then {long}");
            Ok(())
        }

        /// A model directory the tokenizer cannot load fails at `start`, not
        /// against whichever prompt happened to be asked first.
        #[test]
        fn a_session_fails_to_start_when_the_tokenizer_does_not_load() -> anyhow::Result<()> {
            let script = DriverScript::write(
                "read -r _request\n\
                 echo 'PIPETTE_RESULT {\"ok\":false,\"kind\":\"tokenizer\",\"error\":\"no IR\"}'\n\
                 cat > /dev/null\n",
            )?;
            let Err(err) = TokenizeSession::start(&script, Path::new(SHELL), "/tmp/ir") else {
                anyhow::bail!("expected the reported failure to propagate");
            };
            assert!(err.to_string().contains("no IR"), "got {err}");
            Ok(())
        }

        /// A driver that exits without confirming is a failure too — silence
        /// must not read as a healthy tokenizer.
        #[test]
        fn a_session_that_never_confirms_is_an_error() -> anyhow::Result<()> {
            let script = DriverScript::write("read -r _request\n")?;
            let Err(err) = TokenizeSession::start(&script, Path::new(SHELL), "/tmp/ir") else {
                anyhow::bail!("expected a session with no confirmation to fail");
            };
            assert!(err.to_string().contains("exited"), "got {err}");
            Ok(())
        }

        /// A tokenizer that dies mid-session fails the next count rather than
        /// blocking forever on an answer that is not coming.
        #[test]
        fn a_session_that_exits_mid_run_fails_the_next_count() -> anyhow::Result<()> {
            let script = DriverScript::write(
                "read -r _request\n\
                 echo 'PIPETTE_RESULT {\"ok\":true,\"ready\":true}'\n\
                 read -r _first\n\
                 echo 'PIPETTE_RESULT {\"ok\":true,\"tokens\":4}'\n",
            )?;
            let mut session = TokenizeSession::start(&script, Path::new(SHELL), "/tmp/ir")?;
            assert_eq!(session.count("aaaa")?, 4);
            assert!(session.count("aaaa").is_err());
            Ok(())
        }

        /// The hang case the timeout exists for: the child is killed *and*
        /// reaped, so a wedged device leaves no zombie behind.
        #[test]
        fn a_hung_child_is_killed_and_reaped() -> anyhow::Result<()> {
            let mut child = Command::new(SHELL)
                .arg("-c")
                .arg("sleep 30")
                .stdout(Stdio::null())
                .spawn()?;
            let Err(err) = wait_with_deadline(&mut child, Duration::from_millis(50)) else {
                anyhow::bail!("expected the deadline to fire");
            };
            assert!(err.to_string().contains("timed out"), "got {err}");
            // Already reaped: a second wait would block on a live child.
            assert!(child.try_wait()?.is_some());
            Ok(())
        }
    }
}
