//! One claimed job: install the build it needs, run it, classify the outcome,
//! submit.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use anyhow::Context;

use pipette_mgmt_client::types::ClaimedJob;
use pipette_plan_types::{MAX_RUNTIME_OUTPUT_BYTES, Runtime, RuntimeOutput, keep_tail};

use super::Builds;
use super::fleet::Phone;
use super::heartbeat::{HeartbeatGuard, spawn_heartbeat};
use super::install;
use super::summary::{self, CHOSEN_BY_POOL, JobSummary, Outcome, Unreported};
use super::version;
use crate::app::{self, ConsoleRun, LaunchPlan, ResultRef, Signals};
use crate::device::gpu_pin::GpuPin;
use crate::device::{DeviceControl, DevicectlFaultError};
use crate::pool::{self, Pool};
use crate::releases::BuildVersion;
use crate::secret;
use crate::server::submit::{self, Collected, FailureCause, JobFailure};
use crate::server::{self, Session};
use crate::state::DeviceFacts;
use crate::timing;

/// How much of the app's output rides along on a `failure_reason`.
const FAILURE_DETAIL_LINES: usize = 3;

const FAILURE_DETAIL_CHARS: usize = 400;

/// One claimed job, end to end: install, launch with a heartbeat, submit, and
/// the `job finished` line. `Ok(())` means the job reached a conclusion the
/// server was told about (or deliberately not told about, for a lost lease or a
/// stop); a reported failure is a finished job. `Err` is a proxy-side problem;
/// one that [`super::halts_phone`] recognises means this phone must not be
/// driven further, and the claim loop stops.
pub(crate) fn run_claimed_job(
    phone: &mut Phone,
    builds: &mut Builds,
    ctx: &Arc<Session>,
    job: &ClaimedJob,
) -> anyhow::Result<()> {
    let started = Instant::now();
    let mut summary = JobSummary::default();
    let ended = match builds {
        Builds::Pool(pool) => execute_job(phone, pool, ctx, job, &mut summary),
        Builds::Releases(releases) => version::execute_job(phone, releases, ctx, job, &mut summary),
    };
    let outcome = match &ended {
        Ok(outcome) => *outcome,
        Err(JobEnd {
            report: Ok(failure),
            ..
        }) => Outcome::Failed(failure.cause),
        Err(JobEnd {
            report: Err(unreported),
            ..
        }) => Outcome::from(*unreported),
    };
    let concluded = match ended {
        Ok(_) => Ok(()),
        Err(end) => conclude(ctx, job, end, &mut summary.submit),
    };
    summary.log(job, outcome, started.elapsed());
    concluded
}

/// Report a failed job, and hand back the halt it earned, if any.
fn conclude(
    ctx: &Session,
    job: &ClaimedJob,
    end: JobEnd,
    submit: &mut Option<Duration>,
) -> anyhow::Result<()> {
    let JobEnd { report, halt } = end;
    let reported = report.as_ref().map_or(Ok(()), |failure| {
        summary::timed(submit, || report_failure(ctx, job, failure))
    });
    let Some(halt) = halt else {
        return reported;
    };
    // The halt outlives a failed report: the next claim would fail the same way.
    if let Err(error) = reported {
        tracing::error!(
            error = %format!("{error:#}"),
            "reporting the job failed; stopping the phone anyway"
        );
    }
    // Here, inside the job's span: `device stopped` names only the phone.
    tracing::error!(
        error = %format!("{halt:#}"),
        "the job stops this phone until an operator fixes it"
    );
    Err(halt)
}

/// How a failed step ends the job: the failure the server is told about, and
/// whether the phone can be driven again once it has been.
pub(super) struct JobEnd {
    /// The failure the server is told about, or why it is told nothing.
    pub(super) report: Result<JobFailure, Unreported>,
    pub(super) halt: Option<anyhow::Error>,
}

impl From<JobFailure> for JobEnd {
    fn from(failure: JobFailure) -> Self {
        Self {
            report: Ok(failure),
            halt: None,
        }
    }
}

pub(super) struct Lease {
    pub(super) heartbeat: HeartbeatGuard,
    pub(super) lost: Arc<AtomicBool>,
    pub(super) every: Duration,
}

impl Lease {
    pub(super) fn start(ctx: &Arc<Session>, job: &ClaimedJob, every: Duration) -> Self {
        let lost = Arc::new(AtomicBool::new(false));
        Self {
            heartbeat: spawn_heartbeat(
                Arc::clone(ctx),
                job.job_id.clone(),
                every,
                Arc::clone(&lost),
            ),
            lost,
            every,
        }
    }
}

/// Pool mode's steps; any step that fails ends the job with a classified failure.
fn execute_job(
    phone: &mut Phone,
    pool: &Pool,
    ctx: &Arc<Session>,
    job: &ClaimedJob,
    summary: &mut JobSummary,
) -> Result<Outcome, JobEnd> {
    let plan = launch_plan(job)?;
    known_benchmark(ctx, job)?;

    summary.chosen_by = Some(CHOSEN_BY_POOL);
    // Terminal unless the phone went away: the next phone resolves the same pool
    // and fails identically. A replaced identity also ends THIS phone: every
    // further claim would install, fail the same way, and burn another job.
    ensure_installed(phone, pool, &plan.runtime, summary).map_err(|error| {
        let failure = JobFailure::new(
            failure_cause(&error, FailureCause::InstallFailed),
            format!("installing the build: {error:#}"),
        );
        let halt = install::replaced_identity(&error).then_some(error);
        JobEnd {
            report: Ok(failure),
            halt,
        }
    })?;

    let every = heartbeat_interval(phone, job);
    let before = results_before(phone)?;
    let lease = Lease::start(ctx, job, every);
    run_and_submit(phone, ctx, job, &plan, &before, lease, None, summary)
}

/// A benchmark the server does not know is terminal, as `pipette-cli`'s worker makes it.
/// Asked before the launch because the app can only report it as a failed run, which is
/// retriable.
pub(super) fn known_benchmark(ctx: &Session, job: &ClaimedJob) -> Result<(), JobEnd> {
    if server::benchmark_is_unknown(ctx, &job.benchmark_id) {
        return Err(JobFailure::new(
            FailureCause::UnknownBenchmark,
            format!("GET /benchmarks/{}: benchmark not found", job.benchmark_id),
        )
        .into());
    }
    Ok(())
}

/// The claim as the app's arguments; a claim that cannot be read is terminal.
pub(super) fn launch_plan(job: &ClaimedJob) -> Result<LaunchPlan, JobEnd> {
    app::launch_plan_from_claim(job).map_err(|refusal| {
        JobFailure::new(FailureCause::UninterpretableSpec, refusal.to_string()).into()
    })
}

pub(super) fn heartbeat_interval(phone: &Phone, job: &ClaimedJob) -> Duration {
    let protocol = timing::resolve_heartbeat_interval(&job.time_window, None);
    let every = timing::resolve_heartbeat_interval(
        &job.time_window,
        Some(phone.config.loop_config.heartbeat.as_secs()),
    );
    if every > protocol {
        // Honoured, not clamped, with a warning saying what it costs.
        tracing::warn!(
            heartbeat_secs = %every.as_secs(),
            time_window = %job.time_window,
            half_window_secs = %protocol.as_secs(),
            "configured heartbeat_secs is longer than half the claim's time_window; the lease may \
             lapse between ticks"
        );
    }
    every
}

/// So the payloads THIS run produced can be told from the ones already there.
pub(super) fn results_before(phone: &Phone) -> Result<Vec<ResultRef>, JobEnd> {
    app::list_results(
        phone.control.as_ref(),
        &phone.entry.bundle_id,
        phone.config.devicectl.call_timeout,
        phone.config.devicectl.first_line_timeout,
    )
    .map(|listed| listed.into_iter().map(|listed| listed.result).collect())
    .or_fail(
        FailureCause::DevicectlFailure,
        "listing the phone's existing results",
    )
    .map_err(JobEnd::from)
}

/// The app uploads the run's result under the claim; the proxy then deletes the
/// phone's copy, and submits a result the app could not. `version` is the
/// build a version-mode job chose; every payload the proxy submits must report it.
#[expect(
    clippy::too_many_arguments,
    reason = "the run's inputs, each from another step of the claim"
)]
pub(super) fn run_and_submit(
    phone: &mut Phone,
    ctx: &Arc<Session>,
    job: &ClaimedJob,
    plan: &LaunchPlan,
    before: &[ResultRef],
    lease: Lease,
    version: Option<&BuildVersion>,
    summary: &mut JobSummary,
) -> Result<Outcome, JobEnd> {
    let bundle_id = phone.entry.bundle_id.clone();
    let bundle_id = bundle_id.as_str();
    let run_timeout = phone.config.devicectl.run_timeout;
    let first_line = phone.config.devicectl.first_line_timeout;
    tracing::info!(
        heartbeat_secs = %lease.every.as_secs(),
        run_timeout_secs = %run_timeout.as_secs(),
        "starting the run"
    );

    let pin = pin_gpu(phone)?;
    let started = Instant::now();
    let heartbeat = &lease.heartbeat;
    let outcome = app::run_benchmark(
        phone.control.as_ref(),
        bundle_id,
        plan,
        run_timeout,
        first_line,
        Signals {
            abort: &lease.lost,
            stop: &phone.stop,
        },
        &|| heartbeat.stop(),
    );
    let elapsed = started.elapsed();
    summary.run = Some(elapsed);
    // Stop the thread first, so "lost" is settled before submit is decided.
    drop(lease.heartbeat);
    // Before the post-run verbs, which are not measured.
    let pinned = pin.map_or(Ok(()), GpuPin::release);

    if lease.lost.load(Ordering::SeqCst) {
        // No submission at all: the server drops a result whose lease has gone,
        // and a dropped submit reads like a transport fault.
        tracing::warn!(
            elapsed_secs = %elapsed.as_secs(),
            "the lease was lost during the run; skipping submit"
        );
        // A beat can land after the app's upload closed the lease: still clear
        // what went up.
        settle_uploaded(phone, before);
        return Ok(Outcome::LeaseLost);
    }

    let run = outcome.or_fail(FailureCause::DevicectlFailure, "carrying out the launch")?;
    if run.stopped {
        // Stopped with the lease still ours: nothing to submit and no failure on
        // the phone, so the lease lapses and the server hands the job out again.
        tracing::warn!(
            elapsed_secs = %elapsed.as_secs(),
            "the run was terminated by a stop signal; nothing to submit"
        );
        return Ok(Outcome::Stopped);
    }
    if run.silent {
        return Err(JobFailure::new(
            FailureCause::LaunchSilent,
            format!(
                "the app printed nothing for {}s (first_line_timeout_secs) after its launch and \
                 again after one relaunch, so it never received the cell; giving the job back",
                first_line.as_secs()
            ),
        )
        .with_output(runtime_output(&run))
        .into());
    }
    if !run.succeeded() {
        return Err(classify_run(&run, elapsed, run_timeout, version.is_some())
            .with_output(runtime_output(&run))
            .into());
    }
    // A pin lost mid-run leaves reps at the state it exists to rule out.
    pinned.map_err(|error| {
        JobFailure::new(
            FailureCause::DevicectlFailure,
            format!("holding the GPU performance state: {error:#}"),
        )
    })?;
    summary::timed(&mut summary.submit, || {
        submit_results(phone, ctx, job, before, version, &run, elapsed)
    })
}

/// The GPU performance state, held for the run when `[devicectl] pymobiledevice3`
/// is set. A pin that cannot be set fails the job, retriable: an unpinned run
/// would publish rows that read like pinned ones.
fn pin_gpu(phone: &Phone) -> Result<Option<GpuPin>, JobEnd> {
    let Some(program) = phone.config.devicectl.pymobiledevice3.as_deref() else {
        return Ok(None);
    };
    phone
        .details
        .hardware_udid
        .as_deref()
        .context("`device info details` named no hardware UDID")
        .and_then(|udid| GpuPin::hold(program, udid, phone.config.devicectl.call_timeout))
        .map(|pin| {
            tracing::info!("holding the GPU performance state for the run");
            Some(pin)
        })
        .map_err(|error| {
            JobFailure::new(
                FailureCause::DevicectlFailure,
                format!("pinning the GPU performance state: {error:#}"),
            )
            .into()
        })
}

/// The payloads this run left on the phone: each the app uploaded leaves the
/// phone, and the rest are submitted against the job, leaving it once accepted.
fn submit_results(
    phone: &mut Phone,
    ctx: &Session,
    job: &ClaimedJob,
    before: &[ResultRef],
    version: Option<&BuildVersion>,
    run: &ConsoleRun,
    elapsed: Duration,
) -> Result<Outcome, JobEnd> {
    let output = &runtime_output(run);
    let bundle_id = phone.entry.bundle_id.clone();
    let bundle_id = bundle_id.as_str();
    let devicectl = &phone.config.devicectl;
    let collected = submit::collect_payloads(
        phone.control.as_ref(),
        bundle_id,
        before,
        devicectl.call_timeout,
        devicectl.first_line_timeout,
    )
    .or_fail(
        FailureCause::DevicectlFailure,
        "reading the run's result off the phone",
    )
    .map_err(|failure| failure.with_output(output.clone()))?;
    if collected.is_empty() {
        // Success with nothing to show for it: a build that reports its cell's
        // failure as success still logs it. Not terminal otherwise: a rerun may work.
        // The whole stderr, not the output's tail, which can lose the dispatch line.
        let failure = match app::engine_failure(&run.stderr) {
            Some(entry) => engine_failed(&entry, elapsed),
            None => JobFailure::new(
                FailureCause::RunFailed,
                format!(
                    "the run reported success after {}s but left no new result payload on the \
                     phone",
                    elapsed.as_secs()
                ),
            ),
        };
        return Err(failure.with_output(output.clone()).into());
    }
    if collected.len() > 1 {
        tracing::warn!(
            payloads = %collected.len(),
            "the run left more than one new result payload; each answers this job"
        );
    }
    if let Some(version) = version
        && let Some(reported) = collected
            .iter()
            .filter(|collected| !collected.submitted)
            .map(|collected| payload_version(&collected.payload))
            .find(|reported| *reported != Some(version.as_str()))
    {
        // Before the proxy submits: the server files a row under the build it names.
        return Err(JobFailure::new(
            FailureCause::RunFailed,
            format!(
                "the run's result reports client_version {}, not the build installed for it; \
                 the proxy submitted nothing and the phone's copies are kept",
                reported.unwrap_or("none")
            ),
        )
        .with_output(output.clone())
        .into());
    }

    let refused = collected
        .into_iter()
        .map(|collected| settle_collected(phone, ctx, job, collected, output, elapsed))
        .filter(|left_the_phone| !left_the_phone)
        .count();
    Ok(if refused == 0 {
        Outcome::Accepted
    } else {
        Outcome::NotAccepted
    })
}

/// One result the run left: the app's upload or an accepted submit takes it off
/// the phone (`true`). Every other outcome leaves the copy in place and moves
/// on, so one refusal cannot strand the rest.
fn settle_collected(
    phone: &mut Phone,
    ctx: &Session,
    job: &ClaimedJob,
    collected: Collected,
    output: &RuntimeOutput,
    elapsed: Duration,
) -> bool {
    let Collected {
        result,
        submitted,
        payload,
    } = collected;
    let wire = result.wire();
    // Read before the payload is handed over: `device_ram_bytes` reaches this
    // proxy no other way, since `devicectl` reports no memory size.
    harvest_device_facts(phone, &payload);
    if submitted {
        tracing::info!(
            result = %wire,
            elapsed_secs = %elapsed.as_secs(),
            "the app's upload was accepted"
        );
        delete_result(phone, &result);
        return true;
    }
    match submit::submit_success(ctx, job, payload, output) {
        Ok(Some(accepted)) => {
            tracing::info!(
                server_job_id = %accepted,
                result = %wire,
                elapsed_secs = %elapsed.as_secs(),
                "result accepted"
            );
            delete_result(phone, &result);
            true
        }
        Ok(None) => {
            tracing::warn!(
                result = %wire,
                "the result was not accepted; leaving the phone's copy in place"
            );
            false
        }
        Err(error) => {
            tracing::error!(
                result = %wire,
                error = %format!("{error:#}"),
                "submitting the result failed; leaving the phone's copy in place"
            );
            false
        }
    }
}

/// After a lost lease: the results the app uploaded leave the phone, and the
/// rest stay, unsubmitted. Best effort, since the job is over either way.
fn settle_uploaded(phone: &mut Phone, before: &[ResultRef]) {
    let collected = submit::collect_payloads(
        phone.control.as_ref(),
        &phone.entry.bundle_id,
        before,
        phone.config.devicectl.call_timeout,
        phone.config.devicectl.first_line_timeout,
    );
    match collected {
        Ok(collected) => collected
            .into_iter()
            .filter(|collected| collected.submitted)
            .for_each(|collected| {
                tracing::info!(
                    result = %collected.result.wire(),
                    "the app's upload was accepted before the lease was lost"
                );
                harvest_device_facts(phone, &collected.payload);
                delete_result(phone, &collected.result);
            }),
        Err(error) => tracing::warn!(
            error = %format!("{error:#}"),
            "listing the run's results failed; any the app uploaded stay on the phone"
        ),
    }
}

fn delete_result(phone: &Phone, result: &ResultRef) {
    if let Err(error) = app::delete_result(
        phone.control.as_ref(),
        &phone.entry.bundle_id,
        result,
        phone.config.devicectl.call_timeout,
        phone.config.devicectl.first_line_timeout,
    ) {
        // A payload left on the phone costs disk, not correctness.
        tracing::warn!(
            result = %result.wire(),
            error = %format!("{error:#}"),
            "submitted, but deleting the phone's copy failed"
        );
    }
}

fn payload_version(payload: &serde_json::Value) -> Option<&str> {
    payload
        .get("client_version")
        .and_then(serde_json::Value::as_str)
}

/// A failed job step becomes a classified [`JobFailure`]: `cause`, unless the
/// phone turned out to be gone (see [`failure_cause`]).
pub(super) trait OrFail<T> {
    fn or_fail(self, cause: FailureCause, doing: &str) -> Result<T, JobFailure>;
}

impl<T> OrFail<T> for anyhow::Result<T> {
    fn or_fail(self, cause: FailureCause, doing: &str) -> Result<T, JobFailure> {
        self.map_err(|error| {
            JobFailure::new(failure_cause(&error, cause), format!("{doing}: {error:#}"))
        })
    }
}

/// Turn a finished-but-unsuccessful run into a classified failure. The deadline
/// outranks everything, terminal once the engine had half the window, then a kill by
/// iOS (terminal), then the sentinel, then the engine's failure (terminal); the
/// exit code is consulted only when no sentinel appeared. In version mode the
/// proxy chose the build, so a refusal another build may not make is retriable.
fn classify_run(
    run: &ConsoleRun,
    elapsed: Duration,
    run_timeout: Duration,
    version_mode: bool,
) -> JobFailure {
    let detail = output_tail(run);
    if run.timed_out {
        // Under half the window in the engine, a download or wait ate the rest,
        // and the next try, its model on the phone, may fit.
        let (cause, stage) = match run.engine_time {
            Some(time) if time >= run_timeout / 2 => (
                FailureCause::EngineTimedOut,
                format!("{}s of it in the app's engine", time.as_secs()),
            ),
            Some(time) => (
                FailureCause::RunTimedOut,
                format!(
                    "only {}s of it in the app's engine, under half the limit",
                    time.as_secs()
                ),
            ),
            None => (
                FailureCause::RunTimedOut,
                "before the app's engine had the cell".to_owned(),
            ),
        };
        return JobFailure::new(
            cause,
            format!(
                "the run exceeded run_timeout_secs ({}s) after {}s, {stage}, and was \
                 terminated: {detail}",
                run_timeout.as_secs(),
                elapsed.as_secs()
            ),
        );
    }
    if run.killed_by_ios() {
        return JobFailure::new(
            FailureCause::KilledByIos,
            format!(
                "the app was killed (signal 9) after {}s, most likely by iOS at its per-app \
                 memory limit: {detail}",
                elapsed.as_secs()
            ),
        );
    }
    let malformed = match run.sentinel {
        Some(status) => status == app::EXIT_MALFORMED,
        None => run.exit_code == Some(app::EXIT_MALFORMED),
    };
    if malformed {
        return JobFailure::new(
            if version_mode {
                FailureCause::RefusedByBuild
            } else {
                FailureCause::MalformedInvocation
            },
            format!("the app refused the invocation as malformed: {detail}"),
        );
    }
    if let Some(entry) = app::engine_failure(&run.stderr) {
        return engine_failed(&entry, elapsed);
    }
    JobFailure::new(
        FailureCause::RunFailed,
        format!(
            "the run failed after {}s (sentinel={}, exit={}): {detail}",
            elapsed.as_secs(),
            run.sentinel
                .map(|status| status.to_string())
                .unwrap_or_else(|| "none".to_string()),
            run.exit_code
                .map(|code| code.to_string())
                .unwrap_or_else(|| "signal".to_string()),
        ),
    )
}

/// The app's engine failed on the cell, as its failed-cell `entry` says, or the
/// app stopped an attempt at `attempt_timeout_seconds`.
fn engine_failed(entry: &str, elapsed: Duration) -> JobFailure {
    if app::attempt_timed_out(entry) {
        return JobFailure::new(
            FailureCause::AttemptTimedOut,
            format!(
                "the app stopped the cell after {}s, an attempt past its \
                 attempt_timeout_seconds: {entry}",
                elapsed.as_secs()
            ),
        );
    }
    JobFailure::new(
        if entry.contains(app::OUT_OF_MEMORY) {
            FailureCause::EngineOutOfMemory
        } else {
            FailureCause::EngineFailed
        },
        format!(
            "the app's engine failed on the cell after {}s: {entry}",
            elapsed.as_secs()
        ),
    )
}

/// An absent phone is its own failure cause, whatever call noticed it:
/// `CoreDeviceError 1011` can surface from any `devicectl` call, and filing it as
/// the TERMINAL [`FailureCause::InstallFailed`] would burn every queued job over
/// one unplugged cable. `otherwise` is what the call site would have reported.
pub(super) fn failure_cause(error: &anyhow::Error, otherwise: FailureCause) -> FailureCause {
    let absent = error
        .chain()
        .filter_map(|cause| cause.downcast_ref::<DevicectlFaultError>())
        .any(DevicectlFaultError::is_device_absent);
    if absent {
        FailureCause::DeviceUnreachable
    } else {
        otherwise
    }
}

/// Install the build a claimed job needs, unless it is already the one on the
/// phone. The claim names a runtime and the build providing it is the one that
/// can run it; a runtime no build provides is TERMINAL, since retrying cannot
/// make a build appear. Only an install attempt goes to `summary`, and
/// `installed` is raised once `device install app` succeeds, so a check after
/// it that fails still reports the install.
fn ensure_installed(
    phone: &mut Phone,
    pool: &Pool,
    runtime: &Runtime,
    summary: &mut JobSummary,
) -> anyhow::Result<()> {
    let Some(build) = pool.build_for(runtime).map(|known| known.path.clone()) else {
        return Err(pool::NoBuildForError {
            dir: pool.dir().to_path_buf(),
            runtime: runtime.headless_token().to_owned(),
            advertised: pool.capabilities()?,
        }
        .into());
    };

    // Either alone is wrong: an operator who deleted the app leaves our record
    // true and `device info apps` false.
    let bundle_id = &phone.entry.bundle_id;
    if phone.installed.as_deref() == Some(build.as_path())
        && bundle_present(phone.control.as_ref(), bundle_id)?
    {
        tracing::debug!(
            build = %build.display(),
            runtime = %runtime.headless_token(),
            "the build this job needs is already on the phone"
        );
        return Ok(());
    }

    tracing::info!(
        build = %build.display(),
        runtime = %runtime.headless_token(),
        device = %phone.control.udid(),
        "installing the build this job's runtime needs"
    );
    summary::timed(&mut summary.install, || {
        // Forget BEFORE, remember after: a failure between leaves "unknown".
        phone.set_installed(None)?;
        let control = phone.control.as_ref();
        install::check_before(control, &build)?;
        control
            .install_app(&build)
            .with_context(|| format!("installing {} on {}", build.display(), control.udid()))?;
        summary.installed = true;
        install::check_after(
            control,
            &phone.entry,
            &format!("after installing {}", build.display()),
            phone.config.devicectl.call_timeout,
            phone.config.devicectl.first_line_timeout,
        )?;
        phone.set_installed(Some(&build))
    })
}

fn bundle_present(devicectl: &dyn DeviceControl, bundle_id: &str) -> anyhow::Result<bool> {
    let installed = devicectl.info_apps(Some(bundle_id)).with_context(|| {
        format!(
            "listing apps on {} to decide on an install",
            devicectl.udid()
        )
    })?;
    // Matched against its OWN column: the text listing ends with an
    // `Apps installed:` line a naive `tail -1` turns into a bogus app.
    let found = installed.iter().find(|app| app.bundle_id == *bundle_id);
    if let Some(app) = found {
        // Both numbers: `version` names the release, `bundleVersion` distinguishes
        // two builds of it.
        tracing::debug!(
            bundle_id = %bundle_id,
            build_version = %app.build_version.as_deref().unwrap_or("unknown"),
            version = %app.version.as_deref().unwrap_or("unknown"),
            "the bundle is on the phone"
        );
    }
    Ok(found.is_some())
}

/// The runtime runs inside the app, so the app's console is its output.
fn runtime_output(run: &ConsoleRun) -> RuntimeOutput {
    let stream = |text: &str| Some(keep_tail(text, MAX_RUNTIME_OUTPUT_BYTES));
    RuntimeOutput {
        runtime_stdout: stream(&run.stdout),
        runtime_stderr: stream(&run.stderr),
    }
}

/// The last few non-empty output lines for a `failure_reason`, stderr first:
/// that is where the app puts its refusals.
fn output_tail(run: &ConsoleRun) -> String {
    let source = if run.stderr.trim().is_empty() {
        &run.stdout
    } else {
        &run.stderr
    };
    let mut lines: Vec<&str> = source
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .rev()
        .take(FAILURE_DETAIL_LINES)
        .collect();
    lines.reverse();
    let joined = lines.join(" | ");
    if joined.chars().count() <= FAILURE_DETAIL_CHARS {
        return joined;
    }
    // By chars, not bytes: slicing UTF-8 in half would panic on the failure path.
    let kept: String = joined
        .chars()
        .skip(joined.chars().count() - FAILURE_DETAIL_CHARS)
        .collect();
    format!("...{kept}")
}

/// Post a classified failure. Nothing is logged here:
/// `crate::server::submit::submit_failure` narrates it with the stable cause slug.
fn report_failure(ctx: &Session, job: &ClaimedJob, failure: &JobFailure) -> anyhow::Result<()> {
    submit::submit_failure(ctx, job, failure)
        .with_context(|| format!("submitting the failure for job {}", job.job_id))
}

/// The claim, at info, and its spec at debug with every `auth_token` replaced: a
/// plan injects a HuggingFace token into the cell it dispatches.
pub(crate) fn log_claimed_job(job: &ClaimedJob) {
    tracing::info!(
        time_window = %job.time_window,
        expires_at = %job.expires_at.as_ref().map_or("-", AsRef::as_ref),
        "claimed a job"
    );
    if tracing::enabled!(tracing::Level::DEBUG) {
        tracing::debug!(
            spec = %secret::redacted_spec(&job.spec).to_string(),
            "claim spec"
        );
    }
}

/// Record what the phone said about itself on a completed run. Best effort:
/// failing a submitted result over a cache write would trade a real measurement
/// for a cosmetic one, and a payload with none of the fields changes nothing.
fn harvest_device_facts(phone: &mut Phone, payload: &serde_json::Value) {
    let Some(facts) = DeviceFacts::from_payload(payload) else {
        return;
    };
    if phone.facts.as_ref() == Some(&facts) {
        return;
    }
    match facts.store(&phone.paths.device_facts) {
        Ok(()) => tracing::info!(
            chip = %facts.chip_model.as_deref().unwrap_or("-"),
            ram_bytes = %facts.ram_bytes.map(|b| b.to_string()).unwrap_or_default(),
            "learned what the phone reports about itself"
        ),
        Err(error) => tracing::warn!(
            error = %format!("{error:#}"),
            "could not record what the phone reports about itself"
        ),
    }
    // Remembered even when the write failed: what the phone said is true either way.
    phone.facts = Some(facts);
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use httpmock::Method::{GET, POST};
    use httpmock::MockServer;
    use rstest::rstest;
    use serde_json::{Value, json};

    use super::*;
    use crate::device::fake::{Call, CallLog, FakeDevice};
    use crate::logging::testing::captured;
    use crate::state::InstalledBuild;
    use crate::worker::summary::testing::{has_fields, summary_fields};
    use crate::worker::testing::{
        BUNDLE, CLIENT_ID, identity_of, install_calls, knows_every_benchmark, llamacpp,
        memory_claim, phone, session,
    };

    #[test]
    fn a_build_already_on_the_phone_is_not_reinstalled() -> anyhow::Result<()> {
        let state = tempfile::tempdir()?;
        let device = FakeDevice::new("phone-a").with_installed(BUNDLE);
        let log = device.log();
        let (mut phone, mut pool) = phone(device, state.path())?;
        let build = state.path().join("builds/A.app");
        pool.seed(&build, vec![llamacpp("b10216")?]);
        phone.set_installed(Some(&build))?;

        let mut summary = JobSummary::default();
        ensure_installed(&mut phone, &pool, &llamacpp("b10216")?, &mut summary)?;

        assert_eq!(log.calls()?, vec![], "nothing should be installed");
        assert!(!summary.installed);
        assert_eq!(summary.install, None, "no install, no time");
        assert_eq!(phone.installed, Some(build));
        Ok(())
    }

    #[test]
    fn a_recorded_build_the_phone_no_longer_has_is_reinstalled() -> anyhow::Result<()> {
        let state = tempfile::tempdir()?;
        let device = FakeDevice::new("phone-a").then_console(identity_of(CLIENT_ID)); // nothing installed
        let log = device.log();
        let (mut phone, mut pool) = phone(device, state.path())?;
        let build = state.path().join("builds/A.app");
        pool.seed(&build, vec![llamacpp("b10216")?]);
        phone.set_installed(Some(&build))?;

        let mut summary = JobSummary::default();
        ensure_installed(&mut phone, &pool, &llamacpp("b10216")?, &mut summary)?;

        assert_eq!(log.calls()?, install_calls(&build));
        assert!(summary.installed);
        assert!(summary.install.is_some());
        assert_eq!(phone.installed, Some(build));
        Ok(())
    }

    #[test]
    fn a_job_needing_another_build_installs_it() -> anyhow::Result<()> {
        let state = tempfile::tempdir()?;
        let device = FakeDevice::new("phone-a")
            .with_installed(BUNDLE)
            .then_console(identity_of(CLIENT_ID));
        let log = device.log();
        let (mut phone, mut pool) = phone(device, state.path())?;
        let a = state.path().join("builds/A.app");
        let b = state.path().join("builds/B.app");
        pool.seed(&a, vec![llamacpp("b1")?]);
        pool.seed(&b, vec![llamacpp("b2")?]);
        phone.set_installed(Some(&a))?;

        ensure_installed(
            &mut phone,
            &pool,
            &llamacpp("b2")?,
            &mut JobSummary::default(),
        )?;

        assert_eq!(log.calls()?, install_calls(&b));
        assert_eq!(phone.installed, Some(b.clone()));
        assert_eq!(InstalledBuild::load(&phone.paths.installed)?.path, Some(b));
        Ok(())
    }

    /// A build signed by another team installs cleanly and mints the phone a new
    /// identity. The install is refused after the fact, loudly, and the record
    /// stays "unknown" so nothing trusts the build.
    #[test]
    fn a_build_that_replaces_the_phones_identity_fails_the_install() -> anyhow::Result<()> {
        let state = tempfile::tempdir()?;
        let device = FakeDevice::new("phone-a").then_console(identity_of("ev1_someone_else"));
        let (mut phone, mut pool) = phone(device, state.path())?;
        let build = state.path().join("builds/Other.app");
        pool.seed(&build, vec![llamacpp("b10216")?]);

        let mut summary = JobSummary::default();
        let error = ensure_installed(&mut phone, &pool, &llamacpp("b10216")?, &mut summary)
            .err()
            .context("the install must be refused")?;
        assert!(summary.installed, "the build did go on the phone");
        assert!(summary.install.is_some());
        let message = format!("{error:#}");
        assert!(
            message.contains("ev1_someone_else") && message.contains(CLIENT_ID),
            "{message}"
        );
        assert!(
            install::replaced_identity(&error),
            "typed, so the claim loop stops this phone: {error:#}"
        );
        assert_eq!(phone.installed, None);
        assert_eq!(InstalledBuild::load(&phone.paths.installed)?.path, None);
        Ok(())
    }

    #[test]
    fn a_failed_install_leaves_the_phone_unknown() -> anyhow::Result<()> {
        let state = tempfile::tempdir()?;
        let device = FakeDevice::new("phone-a")
            .with_installed(BUNDLE)
            .failing_install("devicectl: the device disconnected during install");
        let log = device.log();
        let (mut phone, mut pool) = phone(device, state.path())?;
        let a = state.path().join("builds/A.app");
        let b = state.path().join("builds/B.app");
        pool.seed(&a, vec![llamacpp("b1")?]);
        pool.seed(&b, vec![llamacpp("b2")?]);
        phone.set_installed(Some(&a))?;

        let mut summary = JobSummary::default();
        let error = ensure_installed(&mut phone, &pool, &llamacpp("b2")?, &mut summary)
            .err()
            .context("the install was scripted to fail")?;
        assert!(!summary.installed);
        assert!(
            summary.install.is_some(),
            "a failed install still took time"
        );
        assert!(format!("{error:#}").contains("disconnected"), "{error:#}");

        assert_eq!(log.calls()?, vec![Call::Install(b)]);
        assert_eq!(phone.installed, None, "in memory");
        assert_eq!(
            InstalledBuild::load(&phone.paths.installed)?.path,
            None,
            "on disk"
        );
        Ok(())
    }

    #[test]
    fn a_runtime_no_build_provides_installs_nothing() -> anyhow::Result<()> {
        let state = tempfile::tempdir()?;
        let device = FakeDevice::new("phone-a");
        let log = device.log();
        let (mut phone, mut pool) = phone(device, state.path())?;
        pool.seed(&state.path().join("builds/A.app"), vec![llamacpp("b1")?]);

        let mut summary = JobSummary::default();
        let error = ensure_installed(&mut phone, &pool, &llamacpp("b9999")?, &mut summary)
            .err()
            .context("no build provides b9999")?;
        assert_eq!(summary.install, None, "nothing was installed");
        let message = format!("{error:#}");
        assert!(
            message.contains("llamacpp_ios_pipette"),
            "must name the runtime: {message}"
        );
        assert_eq!(log.calls()?, vec![]);
        Ok(())
    }

    /// A memory cell's figure is its process's footprint, so the cell gets a launch of
    /// its own: the queue verbs around it are separate launches, and nothing else runs
    /// in it. A payload the app could not upload goes up from here with the claim's
    /// job id, then leaves the phone. Pool mode reads no `client_versions`, `[]`
    /// included, and its line names no version.
    #[rstest]
    #[case::older_server(None)]
    #[case::none_admitted(Some(vec![]))]
    #[case::pinned(Some(vec!["2026.10.02-e6a9434-internal".to_owned()]))]
    fn a_memory_cell_runs_in_a_launch_of_its_own_and_is_submitted(
        #[case] client_versions: Option<Vec<String>>,
    ) -> anyhow::Result<()> {
        let state = tempfile::tempdir()?;
        let payload = json!({
            "benchmark_id": "max_memory_usage_512",
            "observation_max_host_bytes": 461_373_440,
            "device_ram_bytes": 12_884_901_888_u64,
        });
        let (mut phone, pool, log) =
            memory_cell_phone(state.path(), [completed("")], "local", &payload)?;
        let server = MockServer::start();
        let submit = server.mock(|when, then| {
            when.method(POST)
                .path("/benchmarks")
                .json_body_includes(r#"{"job_id": "job-mem"}"#)
                .json_body_includes(r#"{"observation_max_host_bytes": 461373440}"#)
                .body_includes("BENCH_DONE 0 completed");
            then.status(202).json_body(json!({ "job_id": "job-mem" }));
        });
        let claim = ClaimedJob {
            client_versions,
            ..memory_claim()?
        };

        let ended = run_pool_claim(&mut phone, pool, &server, &claim)?;

        ended.result?;
        submit.assert();
        has_fields(
            &ended.fields,
            &[
                "outcome=accepted",
                "version=-",
                "chosen_by=pool",
                "installed=no",
                "install_secs=-",
            ],
        );
        let verb = |args: &[&str]| Call::Launch {
            bundle_id: BUNDLE.to_owned(),
            app_args: args.iter().map(|arg| (*arg).to_owned()).collect(),
        };
        let cell = Call::Launch {
            bundle_id: BUNDLE.to_owned(),
            app_args: app::launch_plan_from_claim(&claim)?.app_args,
        };
        assert_eq!(
            log.calls()?,
            vec![
                verb(&["headlessrun", "results", "list"]),
                cell,
                verb(&["headlessrun", "results", "list"]),
                verb(&["headlessrun", "results", "show", "result=local-1/cell-1"]),
                verb(&["headlessrun", "results", "delete", "result=local-1/cell-1"]),
            ]
        );
        assert_eq!(
            phone.facts.and_then(|facts| facts.ram_bytes),
            Some(12_884_901_888)
        );
        Ok(())
    }

    /// A benchmark the server does not know is reported terminal before the phone is
    /// touched.
    #[test]
    fn an_unknown_benchmark_is_terminal_and_touches_no_phone() -> anyhow::Result<()> {
        let state = tempfile::tempdir()?;
        let (mut phone, pool, log) =
            memory_cell_phone(state.path(), [completed("")], "local", &json!({}))?;
        let server = MockServer::start();
        let lookup = server.mock(|when, then| {
            when.method(GET).path("/benchmarks/max_memory_usage_512");
            then.status(404);
        });
        let failed = server.mock(|when, then| {
            when.method(POST)
                .path("/benchmarks")
                .json_body_includes(r#"{"retriable": false}"#)
                .body_includes("benchmark not found");
            then.status(202).json_body(json!({ "job_id": "job-mem" }));
        });

        let ended = run_pool_claim(&mut phone, pool, &server, &memory_claim()?)?;

        ended.result?;
        lookup.assert();
        failed.assert();
        assert_eq!(log.calls()?, vec![]);
        has_fields(
            &ended.fields,
            &["outcome=failed", "cause=unknown_benchmark"],
        );
        Ok(())
    }

    /// With `[devicectl] pymobiledevice3` set, a run it cannot pin never launches:
    /// the job is given back retriable instead of running at a free GPU state.
    #[rstest]
    #[case::no_hardware_udid(None, "named no hardware UDID")]
    #[case::the_cli_fails(Some("job-pin-test-udid"), "ConnectionRefusedError")]
    fn a_configured_pin_that_cannot_be_set_fails_the_job_before_the_launch(
        #[case] hardware_udid: Option<&str>,
        #[case] reason: &str,
    ) -> anyhow::Result<()> {
        use std::os::unix::fs::PermissionsExt;

        let state = tempfile::tempdir()?;
        let cli = state.path().join("pymobiledevice3");
        std::fs::write(
            &cli,
            "#!/bin/sh\necho 'ConnectionRefusedError: [Errno 61]' >&2\nexit 1\n",
        )?;
        std::fs::set_permissions(&cli, std::fs::Permissions::from_mode(0o755))?;
        let (mut phone, pool, log) =
            memory_cell_phone(state.path(), [completed("")], "local", &json!({}))?;
        let mut config = (*phone.config).clone();
        config.devicectl.pymobiledevice3 = Some(cli);
        phone.config = Arc::new(config);
        phone.details.hardware_udid = hardware_udid.map(str::to_owned);
        let server = MockServer::start();
        let failed = server.mock(|when, then| {
            when.method(POST)
                .path("/benchmarks")
                .json_body_includes(r#"{"retriable": true}"#)
                .body_includes("pinning the GPU performance state")
                .body_includes(reason);
            then.status(202).json_body(json!({ "job_id": "job-mem" }));
        });
        let claim = memory_claim()?;
        let cell = Call::Launch {
            bundle_id: BUNDLE.to_owned(),
            app_args: app::launch_plan_from_claim(&claim)?.app_args,
        };

        let ended = run_pool_claim(&mut phone, pool, &server, &claim)?;

        ended.result?;
        failed.assert();
        assert!(!log.calls()?.contains(&cell), "launched unpinned");
        has_fields(
            &ended.fields,
            &["outcome=failed", "cause=devicectl_failure"],
        );
        Ok(())
    }

    /// How `run_claimed_job` ended, and the fields of its `job finished` line.
    struct Ended {
        result: anyhow::Result<()>,
        fields: Vec<String>,
    }

    fn run_pool_claim(
        phone: &mut Phone,
        pool: Pool,
        server: &MockServer,
        claim: &ClaimedJob,
    ) -> anyhow::Result<Ended> {
        knows_every_benchmark(server);
        let ctx = Arc::new(session(phone, &server.base_url())?);
        let (result, logged) =
            captured(|| run_claimed_job(phone, &mut Builds::Pool(Arc::new(pool)), &ctx, claim));
        Ok(Ended {
            result,
            fields: summary_fields(&logged)?,
        })
    }

    /// How a run ended, as the line reports it: a refused result is kept on the
    /// phone, and a lost lease or a stop submits nothing. A lease lost while the
    /// proxy stops still reads as lost.
    #[rstest]
    #[case::refused(completed(""), false, 1, "outcome=not_accepted")]
    #[case::lease_lost(
        ConsoleRun { aborted: true, ..console("", "", None, None) },
        false,
        0,
        "outcome=lease_lost"
    )]
    #[case::lease_lost_while_stopping(
        ConsoleRun { aborted: true, ..console("", "", None, None) },
        true,
        0,
        "outcome=lease_lost"
    )]
    #[case::stopped(
        ConsoleRun { stopped: true, ..console("", "", None, None) },
        false,
        0,
        "outcome=stopped"
    )]
    fn the_line_says_how_the_run_ended(
        #[case] cell: ConsoleRun,
        #[case] stopping: bool,
        #[case] submits: usize,
        #[case] outcome: &str,
    ) -> anyhow::Result<()> {
        let state = tempfile::tempdir()?;
        let payload = json!({ "benchmark_id": "max_memory_usage_512" });
        let (mut phone, pool, log) = memory_cell_phone(state.path(), [cell], "local", &payload)?;
        phone.stop.store(stopping, Ordering::SeqCst);
        let server = MockServer::start();
        let refused = server.mock(|when, then| {
            when.method(POST).path("/benchmarks");
            then.status(409);
        });

        let ended = run_pool_claim(&mut phone, pool, &server, &memory_claim()?)?;

        ended.result?;
        refused.assert_calls(submits);
        has_fields(&ended.fields, &[outcome, "cause=-"]);
        assert!(!deleted_the_copy(&log)?, "the phone keeps its copy");
        Ok(())
    }

    /// A pool job that put its build on the phone says so, even when the check
    /// after the install refuses it.
    #[rstest]
    #[case::checked(CLIENT_ID, 0, &["outcome=stopped", "installed=yes", "install_secs=#"])]
    #[case::identity_replaced(
        "ev1_someone_else",
        1,
        &["outcome=failed", "cause=install_failed", "installed=yes", "install_secs=#"]
    )]
    fn a_pool_job_that_installs_its_build_says_so(
        #[case] client: &str,
        #[case] reports: usize,
        #[case] expected: &[&str],
    ) -> anyhow::Result<()> {
        let state = tempfile::tempdir()?;
        let device = FakeDevice::new("phone-a")
            .then_console(identity_of(client))
            .then_console(console(
                "[HEADLESS] results count=0\n",
                "",
                Some(0),
                Some(0),
            ))
            .then_console(ConsoleRun {
                stopped: true,
                ..console("", "", None, None)
            });
        let (mut phone, mut pool) = phone(device, state.path())?;
        pool.seed(
            &state.path().join("builds/A.app"),
            vec![llamacpp("b10216")?],
        );
        let server = MockServer::start();
        let reported = server.mock(|when, then| {
            when.method(POST).path("/benchmarks");
            then.status(202).json_body(json!({ "job_id": "job-mem" }));
        });

        let ended = run_pool_claim(&mut phone, pool, &server, &memory_claim()?)?;

        reported.assert_calls(reports);
        has_fields(&ended.fields, expected);
        Ok(())
    }

    /// The app uploaded the result under the claim: the proxy sends nothing, still
    /// reads the device facts off it, and deletes the phone's copy. So it does when
    /// a beat found the lease gone after that upload closed it.
    #[rstest]
    #[case::lease_held(completed(FILED), "outcome=accepted")]
    #[case::lease_lost_after(
        ConsoleRun { aborted: true, ..completed(FILED) },
        "outcome=lease_lost"
    )]
    fn a_result_the_app_uploaded_is_not_sent_again(
        #[case] cell: ConsoleRun,
        #[case] outcome: &str,
    ) -> anyhow::Result<()> {
        let state = tempfile::tempdir()?;
        let payload = json!({
            "benchmark_id": "max_memory_usage_512",
            "device_ram_bytes": 12_884_901_888_u64,
        });
        let (mut phone, pool, log) =
            memory_cell_phone(state.path(), [cell], "submitted", &payload)?;
        let server = MockServer::start();
        let submit = server.mock(|when, then| {
            when.method(POST).path("/benchmarks");
            then.status(202).json_body(json!({ "job_id": "job-mem" }));
        });

        let ended = run_pool_claim(&mut phone, pool, &server, &memory_claim()?)?;

        ended.result?;
        submit.assert_calls(0);
        has_fields(&ended.fields, &[outcome]);
        assert!(deleted_the_copy(&log)?, "{:?}", log.calls()?);
        assert_eq!(
            phone.facts.and_then(|facts| facts.ram_bytes),
            Some(12_884_901_888)
        );
        Ok(())
    }

    /// A launch the app printed nothing on is relaunched once under the same lease.
    /// A second silent one gives the job back, retriable: the app never had the cell.
    /// Without the lease, or on a stop, there is no relaunch.
    #[rstest]
    #[case::the_relaunch_runs_the_cell(
        vec![silent(), completed(FILED)],
        false,
        2,
        0,
        &["outcome=accepted"]
    )]
    #[case::silent_twice(
        vec![silent(), silent()],
        false,
        2,
        1,
        &["outcome=failed", "cause=launch_silent", "infra=yes"]
    )]
    #[case::the_lease_lost(
        vec![ConsoleRun { aborted: true, ..silent() }],
        false,
        1,
        0,
        &["outcome=lease_lost"]
    )]
    #[case::stopping(vec![silent()], true, 1, 0, &["outcome=stopped"])]
    fn a_silent_launch_is_relaunched_once(
        #[case] cells: Vec<ConsoleRun>,
        #[case] stopping: bool,
        #[case] launches: usize,
        #[case] reports: usize,
        #[case] expected: &[&str],
    ) -> anyhow::Result<()> {
        let state = tempfile::tempdir()?;
        let payload = json!({ "benchmark_id": "max_memory_usage_512" });
        let (mut phone, pool, log) = memory_cell_phone(state.path(), cells, "submitted", &payload)?;
        phone.stop.store(stopping, Ordering::SeqCst);
        let server = MockServer::start();
        let given_back = server.mock(|when, then| {
            when.method(POST)
                .path("/benchmarks")
                .json_body_includes(r#"{"retriable": true}"#)
                .body_includes("printed nothing for 60s");
            then.status(202).json_body(json!({ "job_id": "job-mem" }));
        });
        let claim = memory_claim()?;

        let ended = run_pool_claim(&mut phone, pool, &server, &claim)?;

        ended.result?;
        given_back.assert_calls(reports);
        has_fields(&ended.fields, expected);
        let cell = Call::Launch {
            bundle_id: BUNDLE.to_owned(),
            app_args: app::launch_plan_from_claim(&claim)?.app_args,
        };
        assert_eq!(
            log.calls()?.iter().filter(|call| **call == cell).count(),
            launches
        );
        Ok(())
    }

    /// A launch the watchdog ended at the first-line deadline: `devicectl`'s own
    /// lines and nothing from the app.
    fn silent() -> ConsoleRun {
        ConsoleRun {
            silent: true,
            ..console(
                &format!("{}App terminated due to signal 15.\n", app::LAUNCHED),
                "",
                None,
                None,
            )
        }
    }

    /// The app's filing line for a cell its upload answered.
    const FILED: &str = "[HEADLESS] result cell=cell-1 benchmark=max_memory_usage_512 \
                         location=remote/synced submitted_as_job=job-mem\n";

    /// A memory cell's phone, holding pool build A: an empty listing before the
    /// run, `cells` for its launches, then one `state=<listed>` result holding
    /// `payload`, and a bare sentinel for a delete.
    fn memory_cell_phone(
        state: &Path,
        cells: impl IntoIterator<Item = ConsoleRun>,
        listed: &str,
        payload: &Value,
    ) -> anyhow::Result<(Phone, Pool, CallLog)> {
        let done = "[HEADLESS] BENCH_DONE 0\n";
        let ok = |stdout: &str, stderr: &str| console(stdout, stderr, Some(0), Some(0));
        let listing = format!(
            "[HEADLESS] results count=1\n[HEADLESS] result result=local-1/cell-1 \
             state={listed} benchmark=max_memory_usage_512 type=max_memory_usage\n{done}"
        );
        let device = cells
            .into_iter()
            .fold(
                FakeDevice::new("phone-a")
                    .with_installed(BUNDLE)
                    .then_console(ok("[HEADLESS] results count=0\n", "")),
                FakeDevice::then_console,
            )
            .then_console(ok(&listing, ""))
            .then_console(ok(&payload.to_string(), done))
            .then_console(ok(done, ""));
        let log = device.log();
        let (mut phone, mut pool) = phone(device, state)?;
        let build = state.join("builds/A.app");
        pool.seed(&build, vec![llamacpp("b10216")?]);
        phone.set_installed(Some(&build))?;
        Ok((phone, pool, log))
    }

    /// A cell's launch that completed, after `stdout`.
    fn completed(stdout: &str) -> ConsoleRun {
        console(
            &format!("{stdout}[HEADLESS] BENCH_DONE 0 completed\n"),
            "",
            Some(0),
            Some(0),
        )
    }

    fn deleted_the_copy(log: &CallLog) -> anyhow::Result<bool> {
        Ok(log.calls()?.into_iter().any(|call| {
            matches!(call, Call::Launch { app_args, .. } if app_args.iter().any(|arg| arg == "delete"))
        }))
    }

    fn console(stdout: &str, stderr: &str, sentinel: Option<i32>, exit: Option<i32>) -> ConsoleRun {
        ConsoleRun {
            stdout: stdout.to_string(),
            stderr: stderr.to_string(),
            sentinel,
            exit_code: exit,
            timed_out: false,
            engine_time: None,
            aborted: false,
            stopped: false,
            silent: false,
        }
    }

    const REFUSED_RUNTIME: &str =
        "runtime= names llamacpp_ios_pipette b10216, but this build is b10100\n";

    /// The fleet's `run_timeout_secs`.
    const RUN_TIMEOUT: Duration = Duration::from_secs(10800);

    /// `RunCell.dispatchRun`'s line: the cell is in the engine.
    const DISPATCHED: &str = "[HEADLESS] info jobRun dispatch: RunRequest(benchmark: \
                              decode_throughput_4096_100)\n";

    /// The Bonsai-27B Q1_0 failure on the PrismML build, as 17pro-10 logged it.
    const ENGINE_FAILED: &str = "[HEADLESS] error jobRun decode_throughput_4096_100 / \
                                 prism-ml/Bonsai-27B-gguf failed: Engine backend failed: \
                                 prefill decode rc=-3.\n";

    /// A load `LlamaCpp.swift` judged out of memory from llama.cpp's log, as
    /// `formatJobError` words it: the app's
    /// `formatJobErrorAugmentsOutOfMemoryWithContextSizeAndDetail` holds the same message.
    const ENGINE_OUT_OF_MEMORY: &str = "[HEADLESS] error jobRun decode_throughput_4096_100 / \
                                        prism-ml/Bonsai-27B-gguf failed: Model + context size \
                                        4096 exceeded available device memory. Try a smaller \
                                        quant (e.g. Q4_0 instead of Q5_K_M), a smaller model, or \
                                        a smaller context length. Underlying error: failed to \
                                        allocate buffer\n";

    /// An attempt the app stopped at its `attempt_timeout_seconds` (`JobExecutor.abandon`):
    /// the app's `theFailureNamesTheAttemptAndTheLimit` holds the same message.
    const ATTEMPT_TIMED_OUT: &str = "[HEADLESS] error jobRun decode_throughput_4096_100 / \
                                     prism-ml/Bonsai-27B-gguf failed: rep 2 ran 300.4s, past \
                                     attempt_timeout_seconds (300s), and was stopped\n";

    /// A cell whose download failed: the same line, from before the engine.
    const DOWNLOAD_FAILED: &str = "[HEADLESS] error jobRun decode_throughput_4096_100 / \
                                   prism-ml/Bonsai-27B-gguf failed: The network connection \
                                   was lost.\n";

    /// `RunCell.refuseOnBattery`, before the engine has the cell.
    const POWER_REFUSED: &str = "[HEADLESS] error jobRun vl_spec_decode_x / m failed: Device \
                                 not ready: `vl_spec_decode_x` measures two arms against each \
                                 other, so it refuses to start while the power state can \
                                 move under it (device_power_state=charging, battery 80%). \
                                 Run it on AC with the battery near full..\n";

    /// `AFMRuntime.run` on a phone whose Apple Intelligence assets are still
    /// downloading.
    const AFM_UNAVAILABLE: &str = "[HEADLESS] error jobRun decode_throughput_256 / \
                                   apple-foundation failed: Engine error: Apple foundation \
                                   model unavailable: modelNotReady.\n";

    /// A llama.cpp load that failed on a file cut short, its log tail below the
    /// app's line.
    const TRUNCATED: &str = "[HEADLESS] error jobRun decode_throughput_4096_100 / m failed: \
                             Engine error: failed to load model: llama_model_load: error \
                             loading model:\ntensor 'output.weight' data is not within the \
                             file bounds, model is corrupted or incomplete.\n";

    /// The same load, as `RuntimeError.unreadableModel` words it.
    const UNREADABLE_MODEL: &str = "[HEADLESS] error jobRun decode_throughput_4096_100 / m \
                                    failed: Model file unreadable: failed to load model: \
                                    llama_model_load: error loading model:\ntensor \
                                    'output.weight' data is not within the file bounds, model \
                                    is corrupted or incomplete.\n";

    /// The same refusal as `AFM_UNAVAILABLE`, as `RuntimeError.readiness` words it.
    const AFM_NOT_READY: &str = "[HEADLESS] error jobRun decode_throughput_256 / \
                                 apple-foundation failed: Device not ready: Apple foundation \
                                 model unavailable: modelNotReady.\n";

    /// The DESIGN section 4 table: the sentinel decides, the exit code is
    /// consulted only when none appeared, and the deadline outranks both. In
    /// version mode the proxy chose the build, so the app's refusals are
    /// retriable causes of their own. A refused runtime is one more status 2.
    #[rstest]
    #[case::sentinel_one(
        "[HEADLESS] BENCH_DONE 1 cancelled\n",
        Some(1),
        Some(0),
        false,
        false,
        FailureCause::RunFailed
    )]
    // Exit 0 on a run that reported malformed: the sentinel still decides.
    #[case::sentinel_two("", Some(2), Some(0), false, false, FailureCause::MalformedInvocation)]
    #[case::exit_two(
        "unknown parameter\n",
        None,
        Some(2),
        false,
        false,
        FailureCause::MalformedInvocation
    )]
    #[case::exit_one(
        "it went wrong\n",
        None,
        Some(1),
        false,
        false,
        FailureCause::RunFailed
    )]
    #[case::deadline("", None, None, true, false, FailureCause::RunTimedOut)]
    #[case::refused_runtime(
        REFUSED_RUNTIME,
        Some(2),
        Some(2),
        false,
        false,
        FailureCause::MalformedInvocation
    )]
    #[case::version_sentinel_one(
        "[HEADLESS] BENCH_DONE 1 cancelled\n",
        Some(1),
        Some(0),
        false,
        true,
        FailureCause::RunFailed
    )]
    #[case::version_sentinel_two("", Some(2), Some(0), false, true, FailureCause::RefusedByBuild)]
    #[case::version_exit_two(
        "unknown parameter\n",
        None,
        Some(2),
        false,
        true,
        FailureCause::RefusedByBuild
    )]
    #[case::version_refused_runtime(
        REFUSED_RUNTIME,
        Some(2),
        Some(2),
        false,
        true,
        FailureCause::RefusedByBuild
    )]
    #[case::version_deadline("", None, None, true, true, FailureCause::RunTimedOut)]
    #[case::engine_failed(
        &format!("{DISPATCHED}{ENGINE_FAILED}"),
        Some(1),
        Some(0),
        false,
        false,
        FailureCause::EngineFailed
    )]
    #[case::version_engine_failed(
        &format!("{DISPATCHED}{ENGINE_FAILED}"),
        Some(1),
        Some(0),
        false,
        true,
        FailureCause::EngineFailed
    )]
    #[case::engine_out_of_memory(
        &format!("{DISPATCHED}{ENGINE_OUT_OF_MEMORY}"),
        Some(1),
        Some(0),
        false,
        false,
        FailureCause::EngineOutOfMemory
    )]
    #[case::attempt_timed_out(
        &format!("{DISPATCHED}{ATTEMPT_TIMED_OUT}"),
        Some(1),
        Some(1),
        false,
        false,
        FailureCause::AttemptTimedOut
    )]
    #[case::version_attempt_timed_out(
        &format!("{DISPATCHED}{ATTEMPT_TIMED_OUT}"),
        Some(1),
        Some(1),
        false,
        true,
        FailureCause::AttemptTimedOut
    )]
    #[case::download_failed(
        DOWNLOAD_FAILED,
        Some(1),
        Some(0),
        false,
        true,
        FailureCause::RunFailed
    )]
    #[case::engine_failed_before_dispatch(
        ENGINE_FAILED,
        Some(1),
        Some(0),
        false,
        true,
        FailureCause::RunFailed
    )]
    #[case::power_refused(POWER_REFUSED, Some(1), Some(0), false, true, FailureCause::RunFailed)]
    #[case::truncated(
        &format!("{DISPATCHED}{TRUNCATED}"),
        Some(1),
        Some(0),
        false,
        true,
        FailureCause::RunFailed
    )]
    #[case::afm_unavailable(
        &format!("{DISPATCHED}{AFM_UNAVAILABLE}"),
        Some(1),
        Some(0),
        false,
        true,
        FailureCause::RunFailed
    )]
    #[case::unreadable_model(
        &format!("{DISPATCHED}{UNREADABLE_MODEL}"),
        Some(1),
        Some(0),
        false,
        true,
        FailureCause::RunFailed
    )]
    #[case::afm_not_ready(
        &format!("{DISPATCHED}{AFM_NOT_READY}"),
        Some(1),
        Some(0),
        false,
        true,
        FailureCause::RunFailed
    )]
    fn a_failed_run_is_classified_by_the_sentinel_first(
        #[case] stderr: &str,
        #[case] sentinel: Option<i32>,
        #[case] exit: Option<i32>,
        #[case] timed_out: bool,
        #[case] version_mode: bool,
        #[case] cause: FailureCause,
    ) {
        let run = ConsoleRun {
            timed_out,
            ..console("", stderr, sentinel, exit)
        };
        let failure = classify_run(&run, Duration::ZERO, RUN_TIMEOUT, version_mode);
        assert_eq!(failure.cause, cause);
        assert_eq!(failure.retriable(), cause.retriable());
    }

    #[rstest]
    fn a_kill_by_ios_is_terminal_and_names_the_memory_limit(
        #[values(false, true)] version_mode: bool,
    ) {
        let run = console(
            &format!("{}App terminated due to signal 9.\n", app::LAUNCHED),
            "llama_kv_cache:       MTL0 KV buffer size =  3060.00 MiB\n",
            None,
            Some(1),
        );
        let failure = classify_run(&run, Duration::from_secs(3), RUN_TIMEOUT, version_mode);
        assert_eq!(failure.cause, FailureCause::KilledByIos);
        assert!(!failure.retriable());
        [
            "after 3s",
            "signal 9",
            "per-app memory limit",
            "3060.00 MiB",
        ]
        .iter()
        .for_each(|needle| assert!(failure.detail.contains(needle), "{}", failure.detail));
    }

    /// A deadline is the engine's once it had half the window.
    #[rstest]
    #[case::in_the_engine(
        Some(9000),
        FailureCause::EngineTimedOut,
        "9000s of it in the app's engine"
    )]
    #[case::mostly_downloading(
        Some(600),
        FailureCause::RunTimedOut,
        "only 600s of it in the app's engine, under half the limit"
    )]
    #[case::never_dispatched(
        None,
        FailureCause::RunTimedOut,
        "before the app's engine had the cell"
    )]
    fn a_deadline_names_its_limit_and_the_engines_share(
        #[case] engine_secs: Option<u64>,
        #[case] cause: FailureCause,
        #[case] stage: &str,
    ) {
        let timed_out = ConsoleRun {
            timed_out: true,
            engine_time: engine_secs.map(Duration::from_secs),
            ..console(
                "",
                &format!("{DISPATCHED}llama_context: n_ctx = 4196\n"),
                Some(1),
                None,
            )
        };
        let failure = classify_run(&timed_out, Duration::from_secs(10810), RUN_TIMEOUT, true);
        assert_eq!(failure.cause, cause);
        [
            "run_timeout_secs (10800s)",
            "after 10810s",
            stage,
            "n_ctx = 4196",
        ]
        .iter()
        .for_each(|needle| assert!(failure.detail.contains(needle), "{}", failure.detail));
    }

    #[test]
    fn an_engine_failure_quotes_the_apps_line() {
        let failed = classify_run(
            &console(
                "",
                &format!("{DISPATCHED}{ENGINE_FAILED}"),
                Some(1),
                Some(0),
            ),
            Duration::from_secs(34),
            RUN_TIMEOUT,
            true,
        );
        assert!(
            failed
                .detail
                .ends_with(&format!("after 34s: {}", ENGINE_FAILED.trim())),
            "{}",
            failed.detail
        );
    }

    /// Terminal, the plan's limit named, the app's line quoted: the cell ran out of time
    /// in the engine well inside `run_timeout_secs`, which stays the outer bound.
    #[test]
    fn an_attempt_the_app_stopped_quotes_its_line() {
        let stopped = classify_run(
            &console(
                "[HEADLESS] BENCH_DONE 1 failed: rep 2 ran 300.4s, past \
                 attempt_timeout_seconds (300s), and was stopped\n",
                &format!("{DISPATCHED}{ATTEMPT_TIMED_OUT}"),
                Some(1),
                Some(1),
            ),
            Duration::from_secs(612),
            RUN_TIMEOUT,
            false,
        );
        assert_eq!(stopped.cause, FailureCause::AttemptTimedOut);
        assert!(!stopped.retriable());
        assert!(
            stopped.detail.ends_with(&format!(
                "after 612s, an attempt past its attempt_timeout_seconds: {}",
                ATTEMPT_TIMED_OUT.trim()
            )),
            "{}",
            stopped.detail
        );
    }

    /// The proxy's own terminate can end in devicectl's signal-9 line.
    #[test]
    fn the_deadline_outranks_a_kill_by_ios() {
        let run = ConsoleRun {
            timed_out: true,
            ..console(
                &format!("{}App terminated due to signal 9.\n", app::LAUNCHED),
                "",
                None,
                Some(1),
            )
        };
        let failure = classify_run(&run, Duration::ZERO, RUN_TIMEOUT, true);
        assert_eq!(failure.cause, FailureCause::RunTimedOut);
        assert!(failure.retriable());
    }

    /// An absent device is RETRIABLE and outranks the terminal cause the call site
    /// would have reported: an unplugged cable must not burn every queued job.
    #[test]
    fn an_absent_phone_outranks_the_call_site_cause() {
        let absent = anyhow::Error::new(DevicectlFaultError {
            domain: "com.apple.dt.CoreDeviceError".to_string(),
            code: crate::device::devicectl::DEVICE_NOT_FOUND_CODE,
            description: "The specified device was not found.".to_string(),
        })
        .context("listing apps on 00008150-000 to decide on an install");
        assert_eq!(
            failure_cause(&absent, FailureCause::InstallFailed),
            FailureCause::DeviceUnreachable
        );
        assert!(FailureCause::DeviceUnreachable.retriable());

        let refused = anyhow::Error::new(DevicectlFaultError {
            domain: "com.apple.dt.CoreDeviceError".to_string(),
            code: 3002,
            description: "Unable to install.".to_string(),
        })
        .context("installing ./build/Pipette.app on 00008150-000");
        assert_eq!(
            failure_cause(&refused, FailureCause::InstallFailed),
            FailureCause::InstallFailed
        );

        let ours = anyhow::anyhow!("the configured build does not exist");
        assert_eq!(
            failure_cause(&ours, FailureCause::InstallFailed),
            FailureCause::InstallFailed
        );
    }

    #[test]
    fn the_failure_detail_prefers_stderr_and_stays_bounded() {
        let run = console(
            "stdout line\n",
            "  first\n\nsecond\nthird\nfourth\n",
            None,
            Some(1),
        );
        assert_eq!(output_tail(&run), "second | third | fourth");

        let quiet_stderr = console("only stdout had anything\n", "   \n", None, Some(1));
        assert_eq!(output_tail(&quiet_stderr), "only stdout had anything");

        let long = console("", &"é".repeat(FAILURE_DETAIL_CHARS * 2), None, Some(1));
        let tail = output_tail(&long);
        assert!(tail.starts_with("..."));
        assert_eq!(tail.chars().count(), FAILURE_DETAIL_CHARS + 3);
    }

    /// Both streams go out, an empty one as `""` and a long one cut to its tail.
    #[test]
    fn the_runtime_output_is_the_console_tail() {
        let long = format!("head{}", "x".repeat(MAX_RUNTIME_OUTPUT_BYTES));
        let run = console(&long, "", None, Some(1));
        let output = runtime_output(&run);
        let stdout = output.runtime_stdout.unwrap_or_default();
        assert_eq!(stdout.len(), MAX_RUNTIME_OUTPUT_BYTES);
        assert!(!stdout.contains("head"));
        assert_eq!(output.runtime_stderr.as_deref(), Some(""));
    }
}
