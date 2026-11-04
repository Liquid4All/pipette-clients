//! The short headless verbs: the result queue, `runtimes list`, `version`.

use std::sync::atomic::AtomicBool;
use std::time::Duration;

use anyhow::Context;
use serde::de::IgnoredAny;
use serde_json::Value;

use pipette_plan_types::Runtime;

use super::console::{Signals, relaunch_if_silent};
use super::{ConsoleRun, HEADLESS_TOKEN, SENTINEL, SentinelStream};
use crate::device::DeviceControl;

/// A headless verb that did not answer.
#[derive(Debug, thiserror::Error)]
enum HeadlessError {
    #[error("`{HEADLESS_TOKEN} {verb}` failed: {outcome}")]
    VerbFailed { verb: String, outcome: String },
    #[error("the phone printed nothing on stdout")]
    NoOutput,
    #[error("the phone's stdout held no line opening a JSON document")]
    NoDocument,
}

/// The app ran `runtimes list` and answered failure: a llama.cpp build it carries does
/// not load on this phone, which no relaunch fixes.
#[derive(Debug, thiserror::Error)]
#[error("`{HEADLESS_TOKEN} runtimes list` failed on the phone: {outcome}")]
pub(crate) struct RuntimesListFailedError {
    outcome: String,
}

fn verb_failed(verb: impl Into<String>, run: &ConsoleRun) -> anyhow::Error {
    HeadlessError::VerbFailed {
        verb: verb.into(),
        outcome: outcome(run),
    }
    .into()
}

/// One `<jobId>/<cellId>` pair, the `result=` argument's two non-empty segments.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ResultRef {
    pub(crate) job_id: String,
    pub(crate) cell_id: String,
}

impl ResultRef {
    pub(crate) fn wire(&self) -> String {
        format!("{}/{}", self.job_id, self.cell_id)
    }
}

/// One line of `results list`: the result, and whether the server holds it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ListedResult {
    pub(crate) result: ResultRef,
    /// The app's upload was accepted.
    pub(crate) submitted: bool,
}

/// The `state=` words of a result the server accepted (`BenchmarkResultState`).
const SUBMITTED_STATES: [&str; 2] = ["submitted", "scored"];

/// `headlessrun results list`, parsed from the `result=<jobId>/<cellId>` and
/// `state=` fields of each line.
pub(crate) fn list_results(
    device: &dyn DeviceControl,
    bundle_id: &str,
    timeout: Duration,
    first_line: Duration,
) -> anyhow::Result<Vec<ListedResult>> {
    let run = run_verb(
        device,
        bundle_id,
        &["results", "list"],
        timeout,
        first_line,
        SentinelStream::Stdout,
    )?;
    if !run.succeeded() {
        return Err(verb_failed("results list", &run));
    }
    Ok(parse_listed_results(&run.stdout))
}

/// `headlessrun results show result=<jobId>/<cellId>`: stdout is `payload.json`
/// verbatim, and the sentinel is on stderr.
pub(crate) fn show_result(
    device: &dyn DeviceControl,
    bundle_id: &str,
    result: &ResultRef,
    timeout: Duration,
    first_line: Duration,
) -> anyhow::Result<Value> {
    let argument = format!("result={}", result.wire());
    let run = run_verb(
        device,
        bundle_id,
        &["results", "show", &argument],
        timeout,
        first_line,
        SentinelStream::Stderr,
    )?;
    if !run.succeeded() {
        return Err(verb_failed(format!("results show {argument}"), &run));
    }
    first_json_document(&run.stdout)
        .with_context(|| format!("reading the payload the phone printed for {argument}"))
}

/// `headlessrun results delete result=<jobId>/<cellId>`, so the phone's queue
/// does not accumulate payloads the proxy already submitted.
pub(crate) fn delete_result(
    device: &dyn DeviceControl,
    bundle_id: &str,
    result: &ResultRef,
    timeout: Duration,
    first_line: Duration,
) -> anyhow::Result<()> {
    let argument = format!("result={}", result.wire());
    let run = run_verb(
        device,
        bundle_id,
        &["results", "delete", &argument],
        timeout,
        first_line,
        SentinelStream::Stdout,
    )?;
    if !run.succeeded() {
        return Err(verb_failed(format!("results delete {argument}"), &run));
    }
    Ok(())
}

/// `headlessrun runtimes list`: one `runtime … descriptor=<json> …` line per engine
/// compiled into the INSTALLED build, in the same encoding a claim's
/// `spec.runtime` carries, so the two compare as typed [`Runtime`]s. The set is
/// fixed at compile time, so one reading per build lasts that build's life.
pub(crate) fn list_runtimes(
    device: &dyn DeviceControl,
    bundle_id: &str,
    timeout: Duration,
    first_line: Duration,
) -> anyhow::Result<RuntimeListing> {
    let run = run_verb(
        device,
        bundle_id,
        &["runtimes", "list"],
        timeout,
        first_line,
        SentinelStream::Stdout,
    )?;
    if !run.succeeded() {
        // A sentinel means the app ran the verb; without one the launch never answered.
        return Err(match run.sentinel {
            Some(_) => RuntimesListFailedError {
                outcome: outcome(&run),
            }
            .into(),
            None => verb_failed("runtimes list", &run),
        });
    }
    Ok(parse_runtime_listing(&run.stdout))
}

#[derive(Debug, Default, PartialEq)]
pub(crate) struct RuntimeListing {
    pub(crate) runtimes: Vec<Runtime>,
    /// Descriptors this binary's `pipette-plan-types` cannot name, kept verbatim
    /// so a build newer than the proxy does not look like one with fewer runtimes.
    pub(crate) unreadable: Vec<String>,
}

/// Every runtime row on a listing, whose real shape is
/// `[HEADLESS] runtime type=<type> digest=<digest> descriptor={…} runtime=<cli ref>`. The
/// descriptor is read as one JSON document: a string in it may carry a space, and
/// `runtime=` is not JSON. A row without one (an older build) is reported whole rather
/// than read as a build that provides nothing.
fn parse_runtime_listing(stdout: &str) -> RuntimeListing {
    stdout
        .lines()
        .filter(|line| line.contains(RUNTIME_ROW))
        .map(|line| {
            line.split_once(DESCRIPTOR_FIELD)
                .map_or_else(|| line.trim(), |(_, rest)| leading_json(rest))
        })
        .fold(RuntimeListing::default(), |mut listing, raw| {
            match serde_json::from_str::<Runtime>(raw) {
                Ok(runtime) => listing.runtimes.push(runtime),
                Err(_) => listing.unreadable.push(raw.to_string()),
            }
            listing
        })
}

/// Opens every row; the `runtimes count=N` header lacks the space.
const RUNTIME_ROW: &str = "[HEADLESS] runtime ";

const DESCRIPTOR_FIELD: &str = "descriptor=";

/// The JSON document `text` begins with, as written; all of `text` when it begins with
/// none, so a malformed descriptor is still reported.
fn leading_json(text: &str) -> &str {
    let mut documents = serde_json::Deserializer::from_str(text).into_iter::<IgnoredAny>();
    let end = match documents.next() {
        Some(Ok(_)) => documents.byte_offset(),
        Some(Err(_)) | None => text.len(),
    };
    text.get(..end).unwrap_or(text).trim()
}

/// `headlessrun version`. The line is
/// `[HEADLESS] version client_version=<version> private_thermal=<on|off> …`.
/// The value runs up to ` private_thermal=`, not the next space: a build that names no
/// release answers the local form `0.1.0 (1, a1b2c3d)`, which a whitespace split would
/// truncate to something that looks like a release.
pub(crate) fn app_version(
    device: &dyn DeviceControl,
    bundle_id: &str,
    timeout: Duration,
    first_line: Duration,
) -> anyhow::Result<Option<String>> {
    let run = run_verb(
        device,
        bundle_id,
        &["version"],
        timeout,
        first_line,
        SentinelStream::Stdout,
    )?;
    if !run.succeeded() {
        return Err(verb_failed("version", &run));
    }
    Ok(parse_app_version(&run.stdout))
}

fn parse_app_version(stdout: &str) -> Option<String> {
    stdout.lines().find_map(|line| {
        let (_, rest) = line.split_once(VERSION_FIELD)?;
        // A line with no following field is still a value.
        let value = rest
            .split_once(THERMAL_FIELD)
            .map_or(rest, |(value, _)| value)
            .trim();
        (!value.is_empty()).then(|| value.to_string())
    })
}

const VERSION_FIELD: &str = "client_version=";

/// The field that follows it, which bounds a value that may contain spaces.
const THERMAL_FIELD: &str = " private_thermal=";

/// `headlessrun auth me local=1`: the client the INSTALLED app holds, answered
/// from its Keychain with no request. The line is
/// `[HEADLESS] auth me client_id=<id> server_url=<url> public_key=<hex>`.
pub(crate) fn local_client_id(
    device: &dyn DeviceControl,
    bundle_id: &str,
    timeout: Duration,
    first_line: Duration,
) -> anyhow::Result<String> {
    let run = run_verb(
        device,
        bundle_id,
        &["auth", "me", "local=1"],
        timeout,
        first_line,
        SentinelStream::Stdout,
    )?;
    if !run.succeeded() {
        return Err(verb_failed("auth me local=1", &run));
    }
    parse_local_client_id(&run.stdout).ok_or_else(|| {
        HeadlessError::VerbFailed {
            verb: "auth me local=1".to_owned(),
            outcome: "the app printed no client_id".to_owned(),
        }
        .into()
    })
}

fn parse_local_client_id(stdout: &str) -> Option<String> {
    stdout.lines().find_map(|line| {
        let (_, rest) = line.split_once("auth me client_id=")?;
        let id = rest.split_whitespace().next()?;
        (!id.is_empty()).then(|| id.to_owned())
    })
}

/// One short-lived headless verb: the same `--console` launch with nothing to
/// forward and `timeout` as the deadline (the launch carries no `--timeout`).
/// A silent launch is retried once ([`relaunch_if_silent`]).
fn run_verb(
    device: &dyn DeviceControl,
    bundle_id: &str,
    verb: &[&str],
    timeout: Duration,
    first_line: Duration,
    sentinel_on: SentinelStream,
) -> anyhow::Result<ConsoleRun> {
    let mut app_args = Vec::with_capacity(verb.len() + 1);
    app_args.push(HEADLESS_TOKEN.to_string());
    app_args.extend(verb.iter().map(|token| (*token).to_string()));
    // These verbs read what is already on disk and finish in seconds.
    let never = AtomicBool::new(false);
    let signals = Signals {
        abort: &never,
        stop: &never,
    };
    let mut ignore = |_: &str| {};
    relaunch_if_silent(&verb.join(" "), first_line, signals, || {
        device.launch_console(
            bundle_id,
            &app_args,
            timeout,
            first_line,
            sentinel_on,
            signals,
            &mut ignore,
        )
    })
}

fn outcome(run: &ConsoleRun) -> String {
    let mut detail = match (run.sentinel, run.exit_code) {
        (Some(status), _) => format!("{SENTINEL} {status}"),
        (None, Some(code)) => format!("no {SENTINEL} and devicectl exited {code}"),
        (None, None) => format!("no {SENTINEL} and devicectl was killed by a signal"),
    };
    if run.timed_out {
        detail.push_str(", after our own deadline elapsed");
    }
    if run.aborted {
        detail.push_str(", after the lease was lost");
    }
    if run.stopped {
        detail.push_str(", after the process was asked to stop");
    }
    if run.silent {
        detail.push_str(", after the app printed nothing by the first-line deadline, twice");
    }
    detail
}

/// The FIRST JSON document on stdout: `results show` prints `payload.json` and
/// then, for a cell that has one, `metrics.json`. Two documents back to back are
/// not one JSON value, and the payload alone is the submit body. `devicectl
/// --console` prints its own lines ahead of the app's, so the document starts at
/// the first line that opens an object.
fn first_json_document(stdout: &str) -> anyhow::Result<Value> {
    if stdout.trim().is_empty() {
        return Err(HeadlessError::NoOutput.into());
    }
    let document = std::iter::once(0)
        .chain(stdout.match_indices('\n').map(|(at, _)| at + 1))
        .filter_map(|at| stdout.get(at..))
        .find(|rest| rest.trim_start_matches([' ', '\t']).starts_with('{'))
        .ok_or(HeadlessError::NoDocument)?;
    let mut documents = serde_json::Deserializer::from_str(document).into_iter::<Value>();
    match documents.next() {
        Some(first) => first.context("the phone's stdout did not hold a whole JSON document"),
        None => Err(HeadlessError::NoDocument.into()),
    }
}

fn parse_listed_results(stdout: &str) -> Vec<ListedResult> {
    stdout.lines().filter_map(parse_result_line).collect()
}

/// One listing line, whose real shape is
/// `[HEADLESS] result result=<jobId>/<cellId> state=… benchmark=… …`. The per-cell
/// progress line prints `cell=<cellId>` alone, so the listing is the only source.
fn parse_result_line(line: &str) -> Option<ListedResult> {
    let field = |key: &str| {
        line.split_whitespace()
            .find_map(|token| token.strip_prefix(key))
    };
    Some(ListedResult {
        result: parse_result_value(field("result=")?)?,
        submitted: field("state=").is_some_and(|state| SUBMITTED_STATES.contains(&state)),
    })
}

/// `<jobId>/<cellId>`: exactly two non-empty segments, as `ResultId.parse` has it.
/// A job id alone addresses a job, not a result.
fn parse_result_value(raw: &str) -> Option<ResultRef> {
    let mut segments = raw.split('/');
    let job_id = segments.next()?;
    let cell_id = segments.next()?;
    if segments.next().is_some() || job_id.is_empty() || cell_id.is_empty() {
        return None;
    }
    Some(ResultRef {
        job_id: job_id.to_string(),
        cell_id: cell_id.to_string(),
    })
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;
    use crate::app::RUNTIME_WIRE;
    use crate::device::fake::FakeDevice;

    #[test]
    fn the_local_identity_line_yields_its_client_id() {
        let out = "[HEADLESS] auth me client_id=ev1_cb8e78ec server_url=https://c public_key=5f\n\
                   [HEADLESS] BENCH_DONE 0\n";
        assert_eq!(parse_local_client_id(out).as_deref(), Some("ev1_cb8e78ec"));
        assert_eq!(
            parse_local_client_id("[HEADLESS] auth me ERROR not registered\n"),
            None
        );
    }

    /// The rows as `RuntimesCommands.runtimeRows` prints them: the descriptor, then the CLI name
    /// under `runtime=`, which is not JSON. What does not parse is reported, not dropped.
    #[rstest]
    #[case::one_per_row(
        format!(
            "[HEADLESS] runtimes count=2\n\
             [HEADLESS] runtime type=apple_foundation digest=5d41402a \
             descriptor={{\"type\":\"apple_foundation\"}} runtime=apple_foundation\n\
             [HEADLESS] runtime type=mlx_ios_pipette digest=7d793037 \
             descriptor={RUNTIME_WIRE} runtime=mlx-swift=0.21.0 mlx-swift-lm=2.25.4\n\
             [HEADLESS] BENCH_DONE 0\n"
        ),
        &["apple_foundation", "mlx_ios_pipette"],
        &[]
    )]
    #[case::an_unknown_type(
        "[HEADLESS] runtime type=quantum_ios_pipette digest=- \
         descriptor={\"type\":\"quantum_ios_pipette\"} runtime=quantum\n"
            .to_owned(),
        &[],
        &[r#"{"type":"quantum_ios_pipette"}"#]
    )]
    // The WHOLE document reaches the parser, and nothing after it.
    #[case::a_space_in_a_string(
        "[HEADLESS] runtime type=llamacpp_ios_pipette digest=- \
         descriptor={\"repository_url\":\"a b\",\"type\":\"llamacpp_ios_pipette\"} \
         runtime=b1:ios-arm64\n"
            .to_owned(),
        &[],
        &[r#"{"repository_url":"a b","type":"llamacpp_ios_pipette"}"#]
    )]
    #[case::before_the_descriptor_column(
        "[HEADLESS] runtimes count=1\n\
         [HEADLESS] runtime runtime={\"type\":\"apple_foundation\"}\n"
            .to_owned(),
        &[],
        &[r#"[HEADLESS] runtime runtime={"type":"apple_foundation"}"#]
    )]
    fn a_runtimes_listing_yields_each_rows_descriptor(
        #[case] stdout: String,
        #[case] runtimes: &[&str],
        #[case] unreadable: &[&str],
    ) {
        let listing = parse_runtime_listing(&stdout);
        let tokens: Vec<_> = listing
            .runtimes
            .iter()
            .map(Runtime::headless_token)
            .collect();
        assert_eq!(tokens, runtimes, "{listing:?}");
        assert_eq!(listing.unreadable, unreadable, "{listing:?}");
    }

    #[rstest]
    #[case::current(
        "[HEADLESS] version client_version=2026.09.25-357c24d-internal private_thermal=on \
         thermal=die\n",
        Some("2026.09.25-357c24d-internal")
    )]
    #[case::with_the_memory_flags(
        "[HEADLESS] version client_version=2026.09.26-abc1234-internal private_thermal=on \
         private_memory=on thermal=die memory=kernel\n",
        Some("2026.09.26-abc1234-internal")
    )]
    #[case::local_build(
        "[HEADLESS] version client_version=0.1.0 (1, a1b2c3d) private_thermal=off\n",
        Some("0.1.0 (1, a1b2c3d)")
    )]
    #[case::no_version("[HEADLESS] BENCH_DONE 0 completed\n", None)]
    fn the_version_line_yields_the_client_version(
        #[case] stdout: &str,
        #[case] expected: Option<&str>,
    ) {
        assert_eq!(parse_app_version(stdout).as_deref(), expected);
    }

    /// The app's own `state=` words (`BenchmarkResultState`).
    #[test]
    fn results_and_their_state_come_off_the_listing_lines() {
        let stdout = "[HEADLESS] results count=3\n\
             [HEADLESS] result result=job-a/cell-1 state=local benchmark=b type=timing\n\
             [HEADLESS] result result=job-a/cell-2 state=submitted benchmark=b type=timing\n\
             [HEADLESS] result result=job-b/cell-3 state=scored benchmark=b type=timing\n\
             [HEADLESS] BENCH_DONE 0\n";
        let listed = |job: &str, cell: &str, submitted: bool| ListedResult {
            result: ResultRef {
                job_id: job.to_string(),
                cell_id: cell.to_string(),
            },
            submitted,
        };
        assert_eq!(
            parse_listed_results(stdout),
            vec![
                listed("job-a", "cell-1", false),
                listed("job-a", "cell-2", true),
                listed("job-b", "cell-3", true),
            ]
        );
    }

    #[rstest]
    #[case::job_only("job-a")]
    #[case::three("job-a/cell-1/extra")]
    #[case::empty_cell("job-a/")]
    #[case::empty_job("/cell-1")]
    fn a_result_value_that_is_not_a_pair_is_not_a_result(#[case] raw: &str) {
        assert_eq!(parse_result_value(raw), None);
    }

    #[test]
    fn the_payload_is_the_first_document_even_with_metrics_behind_it() -> anyhow::Result<()> {
        let stdout =
            "{\"benchmark_id\":\"b\",\"client_version\":\"0.1.1\"}\n{\"tokens_per_second\":42}\n";
        let payload = first_json_document(stdout)?;
        assert_eq!(payload.get("benchmark_id"), Some(&json!("b")));
        assert_eq!(payload.get("tokens_per_second"), None);
        Ok(())
    }

    /// `devicectl --console` around a real `results show`: its two lines, the
    /// app's documents, then its exit line.
    #[test]
    fn the_payload_is_found_past_the_lines_devicectl_prints_first() -> anyhow::Result<()> {
        let stdout = "Launched application with ai.liquid.liquid-pipette bundle identifier.\n\
                      Waiting for the application to terminate…\n\
                      {\"benchmark_id\":\"b\"}\n\
                      {\"tokens_per_second\":42}\n\
                      The app terminated with the exit code 0.\n";
        assert_eq!(first_json_document(stdout)?, json!({"benchmark_id": "b"}));
        Ok(())
    }

    #[rstest]
    #[case::nothing("", "nothing on stdout")]
    #[case::only_devicectl(
        "Launched application with ai.liquid.liquid-pipette bundle identifier.\n\
         Waiting for the application to terminate…\n",
        "no line opening a JSON document"
    )]
    fn stdout_without_a_document_is_refused(#[case] stdout: &str, #[case] expected: &str) {
        let error = first_json_document(stdout)
            .err()
            .map(|error| error.to_string())
            .unwrap_or_default();
        assert!(error.contains(expected), "{error}");
    }

    /// A verb's launch the app printed nothing on is relaunched once, since every
    /// verb reads, or deletes one result. A second silent launch fails the verb
    /// there, at the first-line deadline, and is never launched a third time.
    #[rstest]
    #[case::the_relaunch_answers(false)]
    #[case::silent_twice(true)]
    fn a_silent_verb_is_relaunched_once(#[case] silent_again: bool) -> anyhow::Result<()> {
        let listing = "[HEADLESS] results count=1\n\
                       [HEADLESS] result result=job-a/cell-1 state=local benchmark=b\n\
                       [HEADLESS] BENCH_DONE 0\n";
        let relaunch = if silent_again {
            console("", true)
        } else {
            console(listing, false)
        };
        let device = FakeDevice::new("phone-a")
            .then_console(console("", true))
            .then_console(relaunch);
        let log = device.log();

        let listed = list_results(
            &device,
            "ai.liquid.liquid-pipette",
            Duration::from_secs(300),
            Duration::from_secs(60),
        );

        match listed {
            Ok(listed) => {
                assert!(!silent_again);
                assert_eq!(listed.len(), 1);
            }
            Err(error) => {
                assert!(silent_again);
                let error = error.to_string();
                assert!(
                    error.contains("printed nothing by the first-line deadline, twice"),
                    "{error}"
                );
            }
        }
        assert_eq!(log.calls()?.len(), 2);
        Ok(())
    }

    /// A verb's launch as the fake replays it: one the app answered, or one the
    /// watchdog ended at the first-line deadline, with no sentinel.
    fn console(stdout: &str, silent: bool) -> ConsoleRun {
        ConsoleRun {
            stdout: stdout.to_owned(),
            stderr: String::new(),
            sentinel: (!silent).then_some(0),
            exit_code: Some(i32::from(silent)),
            timed_out: false,
            engine_time: None,
            aborted: false,
            stopped: false,
            silent,
        }
    }
}
