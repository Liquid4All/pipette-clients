//! Getting the result off the phone and onto the server, and reporting a failure
//! when there is none. The app uploads its result itself, the claimed `job_id`
//! echoed, which closes the lease. A result it could not upload is sent from
//! here, as the phone wrote it, with `job_id` and the run's output the only
//! insertions.

use std::time::Duration;

use serde_json::Value;

use pipette_mgmt_client::types::{ClaimedJob, FailureSubmission};
use pipette_plan_types::{FailureCode, RuntimeOutput};

use crate::app::{self, ListedResult, ResultRef};
use crate::device::DeviceControl;
use crate::server::{self, Session};

/// `client_version` on a `FailureSubmission`. Names the proxy, not the phone.
const CLIENT_VERSION_NAME: &str = "pipette-ios-proxy";

/// `pipette-ios-proxy/<version>`, the version `--version` prints.
pub(crate) fn client_version() -> String {
    format!("{CLIENT_VERSION_NAME}/{}", crate::cli::VERSION)
}

/// Why a job failed, and with it whether the server should hand it to someone
/// else. Terminal means the job can never run anywhere, or the cell met its end
/// in the app's engine: iOS killed the app, the engine failed, or the engine
/// ran past the deadline or an attempt's limit. The snake_case slug is what the
/// log carries; the table test pins every one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, strum::IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
pub(crate) enum FailureCause {
    DeviceUnreachable,
    /// `devicectl` itself failed, timed out, or hung.
    DevicectlFailure,
    /// A failed run no other cause explains, or a reported success that left
    /// no result payload or one naming another build.
    RunFailed,
    /// The deadline elapsed with the cell in the app's engine for under half of
    /// it, or never: a download or a wait may finish sooner on another try.
    RunTimedOut,
    /// The deadline elapsed with the cell in the app's engine for half of it or
    /// more ([`crate::app::ConsoleRun::engine_time`]).
    EngineTimedOut,
    /// The app stopped an attempt past the plan's `attempt_timeout_seconds`
    /// ([`crate::app::attempt_timed_out`]).
    AttemptTimedOut,
    /// See [`crate::app::KILLED_BY_IOS`].
    KilledByIos,
    /// The app printed nothing by the first-line deadline, on the launch and on
    /// its one relaunch: it never had the cell (`crate::app::run_benchmark`).
    LaunchSilent,
    /// See [`crate::app::engine_failure`].
    EngineFailed,
    /// The engine failure the app reported as out of memory.
    EngineOutOfMemory,
    /// Exit 2 with no sentinel: malformed argv.
    MalformedInvocation,
    /// Pool mode: no build in the pool provides the job's runtime, or the one
    /// that does would not install.
    InstallFailed,
    UninterpretableSpec,
    /// The server answered `404` for the claim's benchmark.
    UnknownBenchmark,
    /// Version mode: no build this phone offers is one the claim admits.
    NoAdmittedVersion,
    /// Version mode: the chosen build does not carry the job's runtime.
    RuntimeNotInBuild,
    /// Version mode: a listing of the whole releases bucket, taken during the
    /// claim, shows no build carries the job's runtime, so no phone can run it.
    NoBuildCarriesRuntime,
    /// Version mode: the build refused the invocation, which another build
    /// may accept.
    RefusedByBuild,
    /// Version mode: the build could not be pulled, signed or checked yet.
    BuildNotReady,
    /// Version mode: the build fails in a way that would repeat on this phone.
    BuildUnusable,
    /// Version mode: the phone holds another client after an install.
    IdentityReplaced,
}

impl FailureCause {
    /// The classification tables (DESIGN sections 4 and 9). In version mode every failure
    /// that depends on the build is retriable: the proxy, not the job, picked it. A runtime
    /// no build in the bucket carries is the plan's mistake. The cell's end in the engine
    /// (a kill by iOS, an engine failure, the deadline) is the exception: it is terminal
    /// in version mode too, so the cell fails instead of waiting out its deadline, and a
    /// build that fixes it needs `campaigns rerun`.
    pub(crate) fn retriable(self) -> bool {
        match self {
            Self::DeviceUnreachable
            | Self::DevicectlFailure
            | Self::RunFailed
            | Self::RunTimedOut
            | Self::LaunchSilent
            | Self::NoAdmittedVersion
            | Self::RuntimeNotInBuild
            | Self::RefusedByBuild
            | Self::BuildNotReady
            | Self::BuildUnusable
            | Self::IdentityReplaced => true,
            Self::KilledByIos
            | Self::EngineFailed
            | Self::EngineOutOfMemory
            | Self::EngineTimedOut
            | Self::AttemptTimedOut
            | Self::MalformedInvocation
            | Self::InstallFailed
            | Self::UninterpretableSpec
            | Self::UnknownBenchmark
            | Self::NoBuildCarriesRuntime => false,
        }
    }

    /// A failure of the phone, the Mac or the release pipeline, which an
    /// operator fixes, rather than of the job, its plan or the app. A replaced
    /// identity is the phone's, and a pool build the phone cannot take is the
    /// pool's or the phone's.
    pub(crate) fn infrastructure(self) -> bool {
        match self {
            Self::DeviceUnreachable
            | Self::DevicectlFailure
            | Self::BuildNotReady
            | Self::BuildUnusable
            | Self::IdentityReplaced
            | Self::InstallFailed
            | Self::LaunchSilent => true,
            Self::RunFailed
            | Self::RunTimedOut
            | Self::EngineTimedOut
            | Self::AttemptTimedOut
            | Self::KilledByIos
            | Self::EngineFailed
            | Self::EngineOutOfMemory
            | Self::MalformedInvocation
            | Self::UninterpretableSpec
            | Self::UnknownBenchmark
            | Self::NoAdmittedVersion
            | Self::RuntimeNotInBuild
            | Self::NoBuildCarriesRuntime
            | Self::RefusedByBuild => false,
        }
    }

    /// mgmt's grouping (its `docs/httpapi.md`, `failure_code`); `None` where
    /// the cause does not say, which mgmt records as `unspecified`.
    pub(crate) fn code(self) -> Option<FailureCode> {
        match self {
            Self::AttemptTimedOut => Some(FailureCode::AttemptTimeout),
            Self::KilledByIos | Self::EngineOutOfMemory => Some(FailureCode::OutOfMemory),
            // EngineTimedOut is the proxy's deadline, not the attempt limit: hung and
            // slow look the same.
            Self::EngineFailed | Self::EngineTimedOut => Some(FailureCode::RuntimeError),
            Self::DeviceUnreachable
            | Self::DevicectlFailure
            | Self::IdentityReplaced
            | Self::LaunchSilent => Some(FailureCode::DeviceError),
            // RunTimedOut: under half the window in the engine, so a download or a
            // wait ran out.
            Self::RunTimedOut
            | Self::MalformedInvocation
            | Self::InstallFailed
            | Self::UninterpretableSpec
            | Self::UnknownBenchmark
            | Self::NoAdmittedVersion
            | Self::RuntimeNotInBuild
            | Self::NoBuildCarriesRuntime
            | Self::RefusedByBuild
            | Self::BuildNotReady
            | Self::BuildUnusable => Some(FailureCode::HarnessError),
            Self::RunFailed => None,
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct JobFailure {
    pub(crate) cause: FailureCause,
    /// One line, no secrets. Goes into `failure_reason` behind a timestamp.
    pub(crate) detail: String,
    /// The app's console, for a failure that launched it.
    pub(crate) output: RuntimeOutput,
    /// The build the phone reported running, in version mode; else the
    /// submission names this proxy.
    pub(crate) client_version: Option<String>,
}

impl JobFailure {
    pub(crate) fn new(cause: FailureCause, detail: impl Into<String>) -> Self {
        Self {
            cause,
            detail: detail.into(),
            output: RuntimeOutput::default(),
            client_version: None,
        }
    }

    pub(crate) fn with_output(self, output: RuntimeOutput) -> Self {
        Self { output, ..self }
    }

    /// A failure on the build a version-mode job chose: named in the reason,
    /// and the submission's `client_version` once the phone `reported` it.
    pub(crate) fn on_build(self, version: &str, reported: bool) -> Self {
        Self {
            detail: format!("build {version}: {}", self.detail),
            client_version: reported.then(|| version.to_owned()),
            ..self
        }
    }

    pub(crate) fn retriable(&self) -> bool {
        self.cause.retriable()
    }
}

/// `[{RFC 3339 UTC}] {detail}`, the convention every client's `failure_reason` follows.
fn format_failure_reason(detail: &str) -> String {
    format!("[{}] {detail}", crate::timing::rfc3339_now())
}

/// Build the failure body. Reads only the claim ENVELOPE, so a job whose spec
/// never parsed still reports.
fn failure_submission(job: &ClaimedJob, failure: &JobFailure) -> FailureSubmission {
    FailureSubmission {
        failure_code: failure.cause.code(),
        runtime_output: failure.output.clone(),
        ..FailureSubmission::from_claim(
            job,
            format_failure_reason(&failure.detail),
            failure.retriable(),
            failure
                .client_version
                .clone()
                .unwrap_or_else(client_version),
        )
    }
}

/// Insert `job_id` and change nothing else: the descriptors already on the
/// payload are what the server's claim-binding check compares.
fn attach_job_id(payload: &mut Value, job_id: &str) -> anyhow::Result<()> {
    match payload.as_object_mut() {
        Some(object) => {
            object.insert("job_id".to_string(), Value::String(job_id.to_string()));
            Ok(())
        }
        None => anyhow::bail!(
            "the payload the phone printed is not a JSON object, so `job_id` cannot be attached"
        ),
    }
}

/// Add the run's streams, unless the phone already put its own there.
fn attach_runtime_output(payload: &mut Value, output: &RuntimeOutput) -> anyhow::Result<()> {
    let Value::Object(streams) = serde_json::to_value(output)? else {
        anyhow::bail!("runtime output does not serialize to an object");
    };
    match payload.as_object_mut() {
        Some(object) => {
            for (key, value) in streams {
                object.entry(key).or_insert(value);
            }
            Ok(())
        }
        None => anyhow::bail!(
            "the payload the phone printed is not a JSON object, so the run's output cannot be \
             attached"
        ),
    }
}

/// One result a run left on the phone.
#[derive(Debug)]
pub(crate) struct Collected {
    pub(crate) result: ResultRef,
    /// The app's own upload was accepted.
    pub(crate) submitted: bool,
    pub(crate) payload: Value,
}

/// The results this run produced: the `results list` pairs not present before
/// the run, then `results show` for each. A pair's `jobId` is the app's OWN local
/// manifest id, unrelated to the server's, so before/after is the only filter.
pub(crate) fn collect_payloads(
    device: &dyn DeviceControl,
    bundle_id: &str,
    before: &[ResultRef],
    timeout: Duration,
    first_line: Duration,
) -> anyhow::Result<Vec<Collected>> {
    let after = app::list_results(device, bundle_id, timeout, first_line)?;
    let fresh: Vec<ListedResult> = after
        .into_iter()
        .filter(|listed| !before.contains(&listed.result))
        .collect();
    tracing::debug!(
        before = %before.len(),
        fresh = %fresh.len(),
        "collecting results the run produced"
    );
    // Stop on the first unreadable payload: submitting the rest would file a
    // partial run as a complete one.
    fresh
        .into_iter()
        .map(|ListedResult { result, submitted }| {
            app::show_result(device, bundle_id, &result, timeout, first_line).map(|payload| {
                Collected {
                    result,
                    submitted,
                    payload,
                }
            })
        })
        .collect()
}

/// Submit one success payload with the claim echoed onto it. `Ok(None)` is the
/// server's `409`: a closed job, not an error.
pub(crate) fn submit_success(
    ctx: &Session,
    job: &ClaimedJob,
    payload: Value,
    output: &RuntimeOutput,
) -> anyhow::Result<Option<String>> {
    let mut payload = payload;
    attach_job_id(&mut payload, &job.job_id)?;
    attach_runtime_output(&mut payload, output)?;
    let accepted = server::submit_with_backoff(ctx, &job.job_id, &payload, std::thread::sleep)?;
    match accepted {
        Some(response) => {
            tracing::info!("submitted the result");
            Ok(Some(response.job_id))
        }
        None => {
            tracing::warn!("the server dropped the result");
            Ok(None)
        }
    }
}

/// Submit a classified failure. Best effort: `Ok` once the retry budget is spent,
/// since the loop must move past the job either way.
pub(crate) fn submit_failure(
    ctx: &Session,
    job: &ClaimedJob,
    failure: &JobFailure,
) -> anyhow::Result<()> {
    let body = failure_submission(job, failure);
    tracing::warn!(
        cause = %<&str>::from(failure.cause),
        retriable = %if failure.retriable() { "yes" } else { "no" },
        detail = %failure.detail,
        "reporting a job failure"
    );
    if server::submit_with_backoff(ctx, &job.job_id, &body, std::thread::sleep)?.is_none() {
        tracing::warn!("the server dropped the failure report");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use anyhow::Context;
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    fn claim() -> ClaimedJob {
        ClaimedJob {
            job_id: "job-1".to_string(),
            benchmark_id: "decode_throughput_256".to_string(),
            time_window: "PT10M".to_string(),
            expires_at: None,
            client_versions: None,
            spec: Value::Null,
        }
    }

    /// The slug is what an operator greps for: a variant rename must not change it.
    #[rstest]
    #[case::device_absent(FailureCause::DeviceUnreachable, true, true, "device_unreachable")]
    #[case::devicectl_error(FailureCause::DevicectlFailure, true, true, "devicectl_failure")]
    #[case::run_failed(FailureCause::RunFailed, true, false, "run_failed")]
    #[case::run_timed_out(FailureCause::RunTimedOut, true, false, "run_timed_out")]
    #[case::malformed_argv(
        FailureCause::MalformedInvocation,
        false,
        false,
        "malformed_invocation"
    )]
    #[case::install_failed(FailureCause::InstallFailed, false, true, "install_failed")]
    #[case::uninterpretable_spec(
        FailureCause::UninterpretableSpec,
        false,
        false,
        "uninterpretable_spec"
    )]
    #[case::unknown_benchmark(FailureCause::UnknownBenchmark, false, false, "unknown_benchmark")]
    #[case::no_admitted_version(
        FailureCause::NoAdmittedVersion,
        true,
        false,
        "no_admitted_version"
    )]
    #[case::runtime_not_in_build(
        FailureCause::RuntimeNotInBuild,
        true,
        false,
        "runtime_not_in_build"
    )]
    #[case::no_build_carries_runtime(
        FailureCause::NoBuildCarriesRuntime,
        false,
        false,
        "no_build_carries_runtime"
    )]
    #[case::killed_by_ios(FailureCause::KilledByIos, false, false, "killed_by_ios")]
    #[case::launch_silent(FailureCause::LaunchSilent, true, true, "launch_silent")]
    #[case::engine_failed(FailureCause::EngineFailed, false, false, "engine_failed")]
    #[case::engine_out_of_memory(
        FailureCause::EngineOutOfMemory,
        false,
        false,
        "engine_out_of_memory"
    )]
    #[case::engine_timed_out(FailureCause::EngineTimedOut, false, false, "engine_timed_out")]
    #[case::attempt_timed_out(FailureCause::AttemptTimedOut, false, false, "attempt_timed_out")]
    #[case::refused_by_build(FailureCause::RefusedByBuild, true, false, "refused_by_build")]
    #[case::build_not_ready(FailureCause::BuildNotReady, true, true, "build_not_ready")]
    #[case::build_unusable(FailureCause::BuildUnusable, true, true, "build_unusable")]
    #[case::identity_replaced(FailureCause::IdentityReplaced, true, true, "identity_replaced")]
    fn the_classification_table(
        #[case] cause: FailureCause,
        #[case] retriable: bool,
        #[case] infrastructure: bool,
        #[case] slug: &str,
    ) {
        assert_eq!(cause.retriable(), retriable);
        assert_eq!(cause.infrastructure(), infrastructure);
        assert_eq!(<&str>::from(cause), slug);
        assert_eq!(JobFailure::new(cause, "detail").retriable(), retriable);
    }

    #[rstest]
    #[case::attempt(FailureCause::AttemptTimedOut, "attempt_timeout")]
    #[case::killed_by_ios(FailureCause::KilledByIos, "out_of_memory")]
    #[case::engine_out_of_memory(FailureCause::EngineOutOfMemory, "out_of_memory")]
    #[case::engine_failed(FailureCause::EngineFailed, "runtime_error")]
    #[case::engine_deadline(FailureCause::EngineTimedOut, "runtime_error")]
    #[case::run_deadline(FailureCause::RunTimedOut, "harness_error")]
    #[case::devicectl(FailureCause::DevicectlFailure, "device_error")]
    #[case::launch_silent(FailureCause::LaunchSilent, "device_error")]
    fn a_failure_body_carries_its_code(
        #[case] cause: FailureCause,
        #[case] code: &str,
    ) -> anyhow::Result<()> {
        let wire =
            serde_json::to_value(failure_submission(&claim(), &JobFailure::new(cause, "d")))?;
        assert_eq!(wire["failure_code"], code);
        Ok(())
    }

    #[test]
    fn a_failure_with_no_code_leaves_the_key_out() -> anyhow::Result<()> {
        let failure = JobFailure::new(FailureCause::RunFailed, "exit 1");
        let wire = serde_json::to_value(failure_submission(&claim(), &failure))?;
        assert!(wire.get("failure_code").is_none(), "{wire}");
        Ok(())
    }

    #[test]
    fn a_failure_body_carries_the_envelope_and_this_binarys_version() {
        let failure = JobFailure::new(FailureCause::RunFailed, "BENCH_DONE 1 cancelled");
        let body = failure_submission(&claim(), &failure);
        assert_eq!(body.message_type, "failure");
        assert_eq!(body.job_id, "job-1");
        assert_eq!(body.benchmark_id, "decode_throughput_256");
        assert!(body.retriable);
        assert_eq!(
            body.client_version,
            format!("{CLIENT_VERSION_NAME}/{}", crate::cli::VERSION)
        );
        // Only the device that ran the job times it, and the phone sent no failure.
        assert_eq!(body.device_time_ms, None);
        assert!(
            body.failure_reason.ends_with("] BENCH_DONE 1 cancelled"),
            "{}",
            body.failure_reason
        );
    }

    #[rstest]
    #[case::reported(true, "2026.10.02-e6a9434-internal")]
    #[case::not_yet(false, "pipette-ios-proxy/")]
    fn a_failure_on_a_build_names_it(#[case] reported: bool, #[case] version_prefix: &str) {
        let failure = JobFailure::new(FailureCause::BuildNotReady, "the export timed out")
            .on_build("2026.10.02-e6a9434-internal", reported);
        let body = failure_submission(&claim(), &failure);
        assert!(
            body.failure_reason
                .ends_with("] build 2026.10.02-e6a9434-internal: the export timed out"),
            "{}",
            body.failure_reason
        );
        assert!(
            body.client_version.starts_with(version_prefix),
            "{}",
            body.client_version
        );
    }

    #[test]
    fn the_reason_is_stamped_in_the_one_timestamp_shape_both_binaries_print() {
        let reason = format_failure_reason("devicectl CoreDeviceError 1011");
        assert!(reason.starts_with('['), "{reason}");
        assert_eq!(reason.split(']').next().map(str::len), Some(21));
        assert!(reason.contains("Z]"), "{reason}");
    }

    /// The phone's `device_time_ms` passes through unchanged: the proxy does
    /// not time a job.
    #[test]
    fn the_claim_echo_is_the_only_mutation() -> anyhow::Result<()> {
        let mut payload = json!({
            "benchmark_id": "decode_throughput_256",
            "device_chip_model": "A18 Pro",
            "client_version": "0.1.1 (1, a1b2c3d)",
            "device_time_ms": 184_250,
        });
        attach_job_id(&mut payload, "job-1")?;
        assert_eq!(
            payload,
            json!({
                "benchmark_id": "decode_throughput_256",
                "device_chip_model": "A18 Pro",
                "client_version": "0.1.1 (1, a1b2c3d)",
                "device_time_ms": 184_250,
                "job_id": "job-1",
            })
        );
        Ok(())
    }

    /// Key order survives and `job_id` is appended: needs `serde_json`'s
    /// `preserve_order`. Asserted on bytes, since `Value == Value` ignores order.
    #[test]
    fn the_submit_body_keeps_the_phones_key_order_with_the_echo_appended() -> anyhow::Result<()> {
        let source = r#"{"tokens_per_second":42.5,"benchmark_id":"b","client_version":"0.1.1","device":{"z":1,"a":2}}"#;
        let mut payload: Value =
            serde_json::from_str(source).context("the phone's payload parses")?;
        attach_job_id(&mut payload, "job-1")?;
        assert_eq!(
            serde_json::to_string(&payload).context("the body serialises")?,
            r#"{"tokens_per_second":42.5,"benchmark_id":"b","client_version":"0.1.1","device":{"z":1,"a":2},"job_id":"job-1"}"#
        );
        Ok(())
    }

    #[test]
    fn a_failure_body_carries_the_run_output() -> anyhow::Result<()> {
        let output = RuntimeOutput {
            runtime_stdout: Some(String::new()),
            runtime_stderr: Some("ggml abort\n".to_string()),
        };
        let failure = JobFailure::new(FailureCause::RunFailed, "exit 1").with_output(output);
        let wire = serde_json::to_value(failure_submission(&claim(), &failure))?;
        assert_eq!(wire["runtime_stdout"], "");
        assert_eq!(wire["runtime_stderr"], "ggml abort\n");
        Ok(())
    }

    /// The phone's own streams, should it ever send them, win over the console.
    #[test]
    fn the_run_output_fills_only_the_streams_the_phone_left_out() -> anyhow::Result<()> {
        let mut payload = json!({"benchmark_id": "b", "runtime_stderr": "from the app"});
        let output = RuntimeOutput {
            runtime_stdout: Some("console out".to_string()),
            runtime_stderr: Some("console err".to_string()),
        };
        attach_runtime_output(&mut payload, &output)?;
        assert_eq!(
            payload,
            json!({
                "benchmark_id": "b",
                "runtime_stderr": "from the app",
                "runtime_stdout": "console out",
            })
        );
        Ok(())
    }

    #[test]
    fn a_payload_that_is_not_an_object_cannot_carry_the_echo() -> anyhow::Result<()> {
        let mut payload = json!(["not", "a", "payload"]);
        let error = attach_job_id(&mut payload, "job-1")
            .err()
            .context("an array cannot carry the echo")?;
        assert!(
            error.to_string().contains("is not a JSON object"),
            "{error:#}"
        );
        Ok(())
    }

    #[test]
    fn a_job_id_already_on_the_payload_is_replaced_by_the_claims() -> anyhow::Result<()> {
        // An already-echoed payload must still end up bound to THIS claim.
        let mut payload = json!({"job_id": "someone-elses"});
        attach_job_id(&mut payload, "job-1")?;
        assert_eq!(payload.get("job_id"), Some(&json!("job-1")));
        Ok(())
    }
}
