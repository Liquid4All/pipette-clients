//! Running the iOS client's headless mode over `devicectl … process launch
//! --console`, and reading its console contract back.
//!
//! The app's contract, and every part of it that a naive reader gets wrong:
//!
//! - [`HEADLESS_TOKEN`] must be the first app argument, or the app opens its UI
//!   and `--console` waits forever.
//! - Drain stdout AND stderr concurrently: the thermal-readiness gate (up to
//!   300 s per measured rep by default) is on stderr, and a full pipe deadlocks
//!   the app.
//! - The sentinel is `BENCH_DONE <int>` as a SUBSTRING; the real line is
//!   `[HEADLESS] BENCH_DONE 0 completed`.
//! - Trust the sentinel over the exit code: six identical refused launches exited
//!   2, 2, 2, 2, 2, 0.
//! - On `results show` and `job export` the sentinel moves to STDERR.
//! - A current build's first line is `[HEADLESS] hello pid=<pid>` on stderr. A
//!   launch with no [`APP_LINE`] by `first_line_timeout_secs` is taken for the UI
//!   copy Single App Mode reopened and devicectl attached to, and relaunched once.
//!
//! A run launches with `submit=1 claim=<job id>`: the app uploads its result
//! with the claim echoed, under its Keychain identity, which is the client that
//! claimed the job.

mod console;
mod launch;
mod verbs;

use std::time::{Duration, Instant};

pub(crate) use self::console::{Signals, run_benchmark, run_console};
pub(crate) use self::launch::{LaunchPlan, launch_plan_from_claim};
pub(crate) use self::verbs::{
    ListedResult, ResultRef, RuntimesListFailedError, app_version, delete_result, list_results,
    list_runtimes, local_client_id, show_result,
};

/// The token `HeadlessRunner` looks for; everything after it is parsed.
pub(crate) const HEADLESS_TOKEN: &str = "headlessrun";

pub(crate) const SENTINEL: &str = "BENCH_DONE";

/// How every line the app itself prints begins, on either stream; `devicectl`'s
/// own lines around them do not.
pub(crate) const APP_LINE: &str = "[HEADLESS] ";

pub(crate) const SUBMIT_ON: &str = "submit=1";

/// The key naming the server job a run answers; the app echoes it as `job_id`.
pub(crate) const CLAIM_KEY: &str = "claim";

/// Sentinel status for a malformed invocation (unknown verb or `key=`). Terminal.
pub(crate) const EXIT_MALFORMED: i32 = 2;

/// devicectl 642.16's last stdout line when the app died on SIGKILL. Nothing sends
/// SIGKILL to a run the proxy was not told to stop (the farm unit's `--kill`
/// follows its `pkill -TERM` of the proxy), so in practice it is iOS at the
/// per-app memory limit.
pub(crate) const KILLED_BY_IOS: &str = "App terminated due to signal 9.";

/// The stderr line `JobExecutor` logs for a cell that failed:
/// `[HEADLESS] error jobRun <benchmark> / <model> failed: <error>`.
const CELL_FAILED: &str = "[HEADLESS] error jobRun ";

/// How the app words an attempt it stopped past the plan's
/// `attempt_timeout_seconds` (`AttemptTimeoutError`): `rep 2 ran 300.4s, past
/// attempt_timeout_seconds (300s), and was stopped`.
const ATTEMPT_TIMED_OUT: &str = "s, past attempt_timeout_seconds (";

/// How `formatJobError` words a `RuntimeError.outOfMemory`: the model and its
/// context did not fit in the app's memory.
pub(crate) const OUT_OF_MEMORY: &str = "exceeded available device memory";

/// How the app words an engine failure (`RuntimeError`, `formatJobError`): an
/// engine operation (load, tokenize, decode) or its backend failed, it ran out
/// of memory, or an attempt ran past its limit.
const ENGINE_FAILED: [&str; 4] = [
    " failed: Engine error: ",
    " failed: Engine backend failed: ",
    OUT_OF_MEMORY,
    ATTEMPT_TIMED_OUT,
];

/// Engine errors that are not the cell's: llama.cpp's load failures on the file
/// (missing, not a GGUF, cut short), which a fresh download repairs, and the
/// Apple foundation model not ready on this phone (`AFMRuntime.run`: its assets
/// still downloading, Apple Intelligence off, an ineligible device).
const NOT_THE_CELLS: [&str; 4] = [
    "failed to open",
    "failed to read magic",
    "corrupted or incomplete",
    "Apple foundation model unavailable",
];

/// `RunCell.dispatchRun`'s first line: the model is on the phone, its checks are
/// over, and the engine has the cell.
const ENGINE_STARTED: &str = "[HEADLESS] info jobRun dispatch: ";

/// A `vl_spec_decode` cell fetches its corpus after dispatch
/// (`Workspace.fetchDataset`) and logs it unpacked (`VlSpecDecodeCorpus.download`).
const CORPUS_FETCH: &str = "[HEADLESS] info corpus fetching a corpus: ";
const CORPUS_READY: &str = "[HEADLESS] info corpus corpus for ";

/// A readiness wait's polls (`ReadinessStatus.description`): `→ waiting 1.8s`
/// while it holds the rep, then `→ proceeding` or `→ timed out after 300s`.
const READINESS: &str = "[HEADLESS] info jobRun readiness: ";

/// The app's last failed-cell line in `stderr`, joined with the lines below it
/// (llama.cpp's log tail, which says why a load failed, and is also checked for
/// a failure on the file), when the engine failed on the cell: any phone running
/// that build fails the same way. Only after the dispatch line, so a failure
/// before the engine had the cell (a download, the power-state refusal) is not one.
pub(crate) fn engine_failure(stderr: &str) -> Option<String> {
    let lines: Vec<&str> = stderr.lines().map(str::trim).collect();
    let at = lines
        .iter()
        .rposition(|line| line.starts_with(CELL_FAILED))?;
    let (before, from) = lines.split_at(at);
    let (line, after) = from.split_first()?;
    let entry: Vec<&str> = std::iter::once(*line)
        .chain(
            after
                .iter()
                .copied()
                .take_while(|next| !next.starts_with("[HEADLESS] ")),
        )
        .filter(|text| !text.is_empty())
        .collect();
    let the_cells = before.iter().any(|l| l.starts_with(ENGINE_STARTED))
        && ENGINE_FAILED.iter().any(|m| line.contains(m))
        && !entry
            .iter()
            .any(|text| NOT_THE_CELLS.iter().any(|m| text.contains(m)));
    the_cells.then(|| entry.join(" | "))
}

/// Whether an [`engine_failure`] entry is an attempt the app stopped at its
/// `attempt_timeout_seconds` rather than an engine error.
pub(crate) fn attempt_timed_out(entry: &str) -> bool {
    entry.contains(ATTEMPT_TIMED_OUT)
}

/// How long the app's engine has had the cell, read off stderr as each line
/// arrives: from the dispatch line on, less a corpus fetch and the readiness
/// waits, which another try or another phone may get through sooner.
#[derive(Debug, Default)]
pub(crate) struct EngineClock {
    since: Option<Instant>,
    /// When the fetch or wait in progress began.
    held_from: Option<Instant>,
    held: Duration,
}

impl EngineClock {
    pub(crate) fn line(&mut self, line: &str, at: Instant) {
        let line = line.trim();
        let readiness = line.starts_with(READINESS);
        if line.starts_with(ENGINE_STARTED) {
            *self = Self {
                since: Some(at),
                ..Self::default()
            };
        } else if line.starts_with(CORPUS_FETCH) || (readiness && line.contains(" → waiting ")) {
            self.held_from.get_or_insert(at);
        } else if line.starts_with(CORPUS_READY)
            || (readiness && (line.contains(" → proceeding") || line.contains(" → timed out")))
        {
            self.held += self
                .held_from
                .take()
                .map_or(Duration::ZERO, |from| at.saturating_duration_since(from));
        }
    }

    /// The engine's time at `at`, `None` before the dispatch line.
    pub(crate) fn time(&self, at: Instant) -> Option<Duration> {
        let held = self.held
            + self
                .held_from
                .map_or(Duration::ZERO, |from| at.saturating_duration_since(from));
        Some(
            at.saturating_duration_since(self.since?)
                .saturating_sub(held),
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SentinelStream {
    Stdout,
    /// `results show` and `job export`, whose stdout is the payload verbatim.
    Stderr,
}

#[derive(Debug, Clone)]
pub(crate) struct ConsoleRun {
    /// Captured in full: on `results show` these bytes are the submit body.
    pub(crate) stdout: String,
    pub(crate) stderr: String,
    /// The LAST sentinel seen on the expected stream, `None` when none appeared.
    pub(crate) sentinel: Option<i32>,
    pub(crate) exit_code: Option<i32>,
    pub(crate) timed_out: bool,
    /// How long the app's engine had the cell by the time stderr closed
    /// ([`EngineClock`]), `None` if it never did.
    pub(crate) engine_time: Option<Duration>,
    /// The lease was lost mid-run.
    pub(crate) aborted: bool,
    /// The process was asked to stop mid-run. Unlike `aborted`, the lease is
    /// still ours; there is just no result to submit.
    pub(crate) stopped: bool,
    /// Ended at the first-line deadline: the app had printed no [`APP_LINE`].
    pub(crate) silent: bool,
}

impl ConsoleRun {
    /// Whether the app reported success: the sentinel decides, the exit code is
    /// the fallback when none arrived.
    pub(crate) fn succeeded(&self) -> bool {
        match self.sentinel {
            Some(status) => status == 0,
            None => {
                self.exit_code == Some(0)
                    && !self.timed_out
                    && !self.aborted
                    && !self.stopped
                    && !self.silent
            }
        }
    }

    pub(crate) fn killed_by_ios(&self) -> bool {
        self.sentinel.is_none()
            && self
                .stdout
                .lines()
                .map(str::trim)
                .rfind(|line| !line.is_empty())
                == Some(KILLED_BY_IOS)
    }
}

/// The status in the LAST `BENCH_DONE` in `stream`. A bare one means 0, and text
/// after the integer is ignored (`BENCH_DONE 1 cancelled` is status 1).
pub(crate) fn scan_sentinel(stream: &str) -> Option<i32> {
    stream.lines().filter_map(sentinel_in_line).next_back()
}

/// The sentinel status in one line, matched as a substring so the `[HEADLESS] `
/// prefix and any trailing detail are tolerated. A non-numeric token
/// (`BENCH_DONE ok`) is NO status rather than 0, so it falls back to the exit code.
fn sentinel_in_line(line: &str) -> Option<i32> {
    let (_, after) = line.split_once(SENTINEL)?;
    match after.split_whitespace().next() {
        // As `pipette-ctl`'s `scan_sentinel` reads it.
        None => Some(0),
        Some(token) => token.parse().ok(),
    }
}

/// A complete `mlx_ios_pipette` descriptor as the server sends it, keys in the
/// server's order rather than alphabetised.
#[cfg(test)]
pub(crate) const RUNTIME_WIRE: &str = r#"{"type":"mlx_ios_pipette","packages":{"mlx_swift":{"repository_url":"github.com/ml-explore/mlx-swift","repository_version":"0.21.0"},"mlx_swift_lm":{"repository_url":"github.com/ml-explore/mlx-swift-examples","repository_version":"2.25.4"},"swift_transformers":{"repository_url":"github.com/huggingface/swift-transformers","repository_version":"0.1.17"}},"flavor":"ios-arm64"}"#;

/// What `devicectl --console` prints ahead of the app's own lines.
#[cfg(test)]
pub(crate) const LAUNCHED: &str = "Launched application with ai.liquid.liquid-pipette bundle \
                                   identifier.\nWaiting for the application to terminate…\n";

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::real_line("[HEADLESS] BENCH_DONE 0 completed", Some(0))]
    #[case::bare("[HEADLESS] BENCH_DONE", Some(0))]
    #[case::failed("[HEADLESS] BENCH_DONE 1 cancelled", Some(1))]
    #[case::malformed("[HEADLESS] BENCH_DONE 2", Some(2))]
    #[case::progress("[HEADLESS] result cell=c1 benchmark=b location=remotePending", None)]
    #[case::non_numeric("[HEADLESS] BENCH_DONE ok", None)]
    fn sentinel_in_line_cases(#[case] line: &str, #[case] expected: Option<i32>) {
        assert_eq!(sentinel_in_line(line), expected);
    }

    #[test]
    fn the_last_sentinel_wins() {
        let stream = "[HEADLESS] BENCH_DONE 1 cancelled\n[HEADLESS] start runtime=MLX\n[HEADLESS] BENCH_DONE 0 completed\n";
        assert_eq!(scan_sentinel(stream), Some(0));
    }

    #[test]
    fn a_stream_with_no_sentinel_has_no_status() {
        assert_eq!(scan_sentinel("[HEADLESS] worker status=idle\n"), None);
    }

    fn framed(app_lines: &str) -> String {
        format!("{LAUNCHED}{app_lines}")
    }

    /// Pinned to devicectl 642.16's wording: a reworded line stops matching,
    /// and every kill is retriable again.
    #[rstest]
    #[case::killed(
        framed(
            "[HEADLESS] start runtime=llama.cpp model=gemma-4-E4B-it-Q4_K_M.gguf batch=512 \
             benchmarks=[\"vl_throughput_512x512_text0_decode256_img1\"] submit=true\n\
             App terminated due to signal 9.\n"
        ),
        None,
        true
    )]
    #[case::crlf(framed("App terminated due to signal 9.\r\n"), None, true)]
    #[case::blank_after(framed("App terminated due to signal 9.\n\n"), None, true)]
    #[case::proxy_sigterm(framed("App terminated due to signal 15.\n"), None, false)]
    #[case::exited(framed("The app terminated with the exit code 1.\n"), None, false)]
    #[case::not_last(
        framed("App terminated due to signal 9.\n[HEADLESS] later\n"),
        None,
        false
    )]
    #[case::sentinel_first(
        framed("[HEADLESS] BENCH_DONE 1\nApp terminated due to signal 9.\n"),
        Some(1),
        false
    )]
    #[case::lost_console(framed(""), None, false)]
    #[case::empty_stdout(String::new(), None, false)]
    fn a_kill_by_ios_is_devicectls_last_stdout_line(
        #[case] stdout: String,
        #[case] sentinel: Option<i32>,
        #[case] killed: bool,
    ) {
        let run = ConsoleRun {
            stdout,
            stderr: String::new(),
            sentinel,
            exit_code: Some(1),
            timed_out: false,
            engine_time: None,
            aborted: false,
            stopped: false,
            silent: false,
        };
        assert_eq!(run.killed_by_ios(), killed);
    }

    #[test]
    fn the_sentinel_beats_the_exit_code_in_both_directions() {
        let run = |sentinel, exit_code| ConsoleRun {
            stdout: String::new(),
            stderr: String::new(),
            sentinel,
            exit_code,
            timed_out: false,
            engine_time: None,
            aborted: false,
            stopped: false,
            silent: false,
        };
        // The refused launches exited 2, 2, 2, 2, 2, 0; the sixth is not a success.
        assert!(!run(Some(EXIT_MALFORMED), Some(0)).succeeded());
        assert!(run(Some(0), Some(1)).succeeded());
        assert!(!run(None, Some(1)).succeeded());
        assert!(run(None, Some(0)).succeeded());
    }

    #[rstest]
    #[case::dispatched(&[(0, DISPATCH)], 1000, Some(1000))]
    #[case::never_dispatched(&[(0, "[HEADLESS] info jobRun fetching")], 1000, None)]
    #[case::corpus_and_a_wait(
        &[
            (100, DISPATCH),
            (110, "[HEADLESS] info corpus fetching a corpus: 9 bytes from /d"),
            (210, "[HEADLESS] info corpus corpus for vl_spec_decode_x: 4 samples / 8 turns"),
            (300, "[HEADLESS] info jobRun readiness: ios soc=41C → waiting 2s"),
            (302, "[HEADLESS] info jobRun readiness: ios soc=40C → waiting 2s"),
            (600, "[HEADLESS] info jobRun readiness: ios soc=35C → proceeding"),
        ],
        1000,
        Some(500)
    )]
    #[case::still_fetching(
        &[(0, DISPATCH), (100, "[HEADLESS] info corpus fetching a corpus: 9 bytes from /d")],
        1000,
        Some(100)
    )]
    #[case::a_wait_that_timed_out(
        &[
            (0, DISPATCH),
            (100, "[HEADLESS] info jobRun readiness: ios soc=45C → waiting 2s"),
            (400, "[HEADLESS] info jobRun readiness: ios soc=45C → timed out after 300s"),
        ],
        1000,
        Some(700)
    )]
    fn the_engines_time_leaves_out_fetches_and_waits(
        #[case] lines: &[(u64, &str)],
        #[case] end: u64,
        #[case] expected: Option<u64>,
    ) {
        let start = Instant::now();
        let at = |secs| start + Duration::from_secs(secs);
        let mut clock = EngineClock::default();
        lines
            .iter()
            .for_each(|(secs, line)| clock.line(line, at(*secs)));
        assert_eq!(clock.time(at(end)), expected.map(Duration::from_secs));
    }

    const DISPATCH: &str = "[HEADLESS] info jobRun dispatch: RunRequest()";

    /// The app's line for an attempt it stopped is the engine's failure, read as a timeout.
    #[test]
    fn an_attempt_past_its_limit_is_the_engines() {
        let line = "[HEADLESS] error jobRun decode_throughput_256 / LiquidAI/LFM2-350M-GGUF \
                    failed: rep 2 ran 300.4s, past attempt_timeout_seconds (300s), and was \
                    stopped";
        let stderr = format!("{DISPATCH}\n{line}\n");
        let entry = engine_failure(&stderr);
        assert_eq!(entry.as_deref(), Some(line));
        assert!(entry.as_deref().is_some_and(attempt_timed_out));
    }

    #[rstest]
    #[case::dispatched(DISPATCH, true)]
    #[case::before_dispatch("[HEADLESS] info jobRun fetching", false)]
    fn an_engine_failure_quotes_the_engines_lines(#[case] first: &str, #[case] found: bool) {
        let stderr = format!(
            "{first}\n[HEADLESS] error jobRun decode_throughput_256 / m failed: Engine error: \
             model load failed: /m/q.gguf: llama_model_loader: loaded meta data\n\
             llama_model_load: error loading model architecture: unknown model architecture: \
             'lfm3'\n\n[HEADLESS] info jobRun sweep done\n"
        );
        let entry = engine_failure(&stderr);
        assert_eq!(
            entry,
            found.then(|| {
                "[HEADLESS] error jobRun decode_throughput_256 / m failed: Engine error: model \
                 load failed: /m/q.gguf: llama_model_loader: loaded meta data | \
                 llama_model_load: error loading model architecture: unknown model \
                 architecture: 'lfm3'"
                    .to_owned()
            })
        );
        assert!(!entry.as_deref().is_some_and(attempt_timed_out));
    }
}
