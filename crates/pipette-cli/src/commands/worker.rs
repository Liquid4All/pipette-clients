//! `pipette worker`: the planner claim loop, implementing the client obligations
//! in pipette-mgmt's `docs/client-integration.md`.
//!
//! ## Deliberate v1 limits
//!
//! - **No mid-run cancellation.** Runtimes have no cancel token. A heartbeat
//!   that loses the lease sets a shared flag, the benchmark keeps running, and
//!   the worker skips the submit the server would drop.
//! - **No pending artifact for an unsubmitted result.** A run is submitted at
//!   once, and only one the server holds is filed locally (`RemoteSynced`). An
//!   exhausted submit backoff drops the outcome, as a crash mid-run would: the
//!   lease lapses and the job is served again, an eval resuming from its
//!   checkpoint. `benchmarks run` + `sync` is the path that keeps pending
//!   results on disk.
//! - **Claim 5xx retries forever** at a capped backoff. A `403` stops the loop,
//!   and so does a failure report that cannot be submitted: the lease outlives
//!   the job, so a re-claim is handed the same job back (mgmt's
//!   `client-integration.md` §7) and would re-run it against the same refusal
//!   forever. A refused result is first reported as a retriable failure, which
//!   returns the job to the queue.
//! - **Reboot detection.** While a job runs, a marker in the workspace holds
//!   its job ID and kernel boot ID, refreshed every 30 seconds. A claim that
//!   hands back the same job under a new boot ID, with a marker that shows the
//!   job alive at the reboot, is failed permanently instead of run again. The
//!   marker is removed when the job ends, on a signal, and when the session
//!   closes, so an orderly shutdown (^C, or an OS reboot that sends SIGTERM)
//!   is never reported as a reboot. Detection needs a workspace that survives
//!   a reboot.
//!
//! ## Stopping
//!
//! ^C / SIGTERM between jobs sets a flag
//! ([`pipette_subprocess::register_stop_flag`]) that every wait here watches in
//! slices, and the loop returns `Ok(())` at the next check. Inside a run only
//! the readiness gate reads it, through `run_cell`'s `cancelled`: the handler
//! sets the flag only while no child but a bounded short command (such as a
//! readiness probe) is in flight, and two throughput cells enter the gate
//! before they spawn anything, so without it a ^C there would wait out a
//! cooldown plus the whole benchmark. That cell is dropped rather than failed,
//! see [`submit_finished_run`]. A signal with a server up still tears it down
//! and exits 130.
//!
//! A closed stderr stops the worker too
//! ([`pipette_subprocess::stop_on_closed_output`]): over adb or ssh the output
//! pipe is the session, so the worker tears down what it started and exits 141,
//! at once on Unix and at the next log line on Windows.
//!
//! ## Eval shards
//!
//! Under `--eval-shard I/N` the worker takes one claim, runs that group of the
//! cell's samples into a part file beside its checkpoint, and returns the
//! run's outcome: success exits 0 and an error exits 1. It heartbeats as for
//! any job, but writes no in-flight marker, submits no result or failure and
//! finalizes nothing. It runs only behind the pool dispatcher, which holds the
//! lease and later has the run without the flag fold the parts in and submit
//! the cell.

use std::ops::ControlFlow;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::Context;
use clap::{Args, ValueEnum};

use pipette_http::HttpClient;
use pipette_http::retry::{exponential_backoff, jittered};
use pipette_mgmt_client::ConditionalResponse;
use pipette_mgmt_client::types::{ClaimedJob, IfNoneMatch};
use pipette_mgmt_client::{ClientIdentity, ClientStatus, ServerUrl, client_api};
use pipette_ops::eval_completions::EvalRunDigest;
use pipette_plan_types::FailureCode;
use pipette_programs::{ExternalProgram, ExternalPrograms};

use crate::client::claim::{redacted_spec, run_spec_from_claim};
use crate::client::worker::in_flight::{self, Boot, ClaimDecision, InFlight, InFlightRun};
use crate::client::worker::{
    ClaimPoll, DEFAULT_IDLE_JITTER, DEFAULT_IDLE_WAIT, LeaseKeepalive, ReindexCallbacks,
    ReindexStatus, ReindexWait, SubmitDisposition, SubmitRefusedError,
    attach_claim_to_success_payload, classify_run_error, device_time_ms, failure_code,
    failure_from_claim, format_failure_reason, idle_wait_with_jitter,
    installed_runtime_capabilities, keepalive_lease, poll_claim, refresh_profile_at_startup,
    resolve_heartbeat_interval, submit_plan_result_with_backoff,
};
use crate::client_configuration::ClientConfiguration;
use crate::identity::DeviceLabels;
use crate::progress::ReindexProgress;
use crate::run::{finalize_eval_checkpoint, run_cell};
use crate::workspace::PipetteWorkspace;

pub const WORKER_LONG_ABOUT: &str = "\
Run the planner claim loop: claim a job, run it, submit the result, repeat.

Instead of an operator naming the benchmark, runtime and model, this machine
asks the management server for the next job it is eligible for. It runs until
stopped, or until --max-jobs jobs have finished.

At startup it detects the device profile and the runtimes already installed,
reports them so the server can match jobs, waits out a server reindex if one is
pending, and then starts claiming. Only runtimes in the local store are
advertised, so install what this machine should be offered before starting.

Requires a workspace that is initialized, registered, and approved. A claim
refused with 403 means this client is not approved yet: the worker says so and
exits rather than spinning.";

pub const WORKER_AFTER_HELP: &str = "\
Examples:
  pipette worker                     # run until stopped
  pipette worker --max-jobs 1        # take one job, then exit
  pipette worker --idle-secs 60      # poll more often while debugging
  pipette worker --strict docker     # prove the docker daemon answers
  pipette worker --strict uv --strict docker --strict nvidia-smi

By default the worker requires no external program up front: a job that needs
a missing one fails when it would have spawned it. Name the programs this box is
supposed to have, one --strict each, and every one of them is run before the
worker claims anything:

  uv  docker  python  nvidia-smi  rocm-smi

`pipette check <program>` runs the same checks without starting a worker.

While a job runs the worker heartbeats to hold its lease. If the lease is lost,
the benchmark still finishes but the result is not submitted, because runtimes
have no cancel seam in v1. Use `benchmarks run` + `sync` when a result must
survive a failed submit.

Ctrl-C (or SIGTERM) while the worker is waiting, whether between jobs or
backing off from a failing server, stops the loop within a second instead of
killing the process where it stands. During a run it still stops the benchmark
and exits.";

/// Run the planner claim loop until stopped or not-approved.
#[derive(Args, Debug)]
pub struct WorkerArgs {
    /// Base idle wait after a 204 claim (seconds). Default 300 (5 min).
    #[arg(long, default_value_t = DEFAULT_IDLE_WAIT.as_secs())]
    pub idle_secs: u64,

    /// Max jitter added to the idle wait (seconds). Default 60.
    #[arg(long, default_value_t = DEFAULT_IDLE_JITTER.as_secs())]
    pub idle_jitter_secs: u64,

    /// Heartbeat period in seconds while a job is running. Default: half the
    /// claim's `time_window` (protocol recommendation). Values longer than
    /// half the window risk the lease expiring between ticks.
    #[arg(long, value_parser = clap::value_parser!(u64).range(1..))]
    pub heartbeat_secs: Option<u64>,

    /// Exit after finishing this many claimed jobs, failed runs included
    /// (0 = run forever).
    #[arg(long, default_value_t = 0)]
    pub max_jobs: u64,

    /// Skip the startup profile PATCH (use only for debugging against a
    /// server that already has an accurate profile).
    #[arg(long, default_value_t = false)]
    pub skip_profile_refresh: bool,

    /// A program that must answer its own check before this worker claims
    /// anything, rather than only being on PATH. Repeatable:
    /// `--strict uv --strict docker`.
    #[arg(long, value_name = "PROGRAM")]
    pub strict: Vec<StrictProgram>,
}

/// A program `--strict` can name: one variant per [`ExternalProgram`].
#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum StrictProgram {
    /// `uv --version`
    Uv,
    /// `docker info`, which needs a running daemon and not just the client
    Docker,
    /// `python3 --version`
    Python,
    /// `nvidia-smi -L`, the NVML probe a bad driver fails
    NvidiaSmi,
    /// `rocm-smi --version`
    RocmSmi,
}

impl StrictProgram {
    fn program(self) -> ExternalProgram {
        match self {
            Self::Uv => ExternalProgram::UV,
            Self::Docker => ExternalProgram::DOCKER,
            Self::Python => ExternalProgram::PYTHON,
            Self::NvidiaSmi => ExternalProgram::NVIDIA_SMI,
            Self::RocmSmi => ExternalProgram::ROCM_SMI,
        }
    }
}

impl WorkerArgs {
    /// The programs `--strict` named. Named rather than inferred: only the
    /// operator knows what this box is supposed to have.
    pub fn strict_programs(&self) -> Vec<ExternalProgram> {
        self.strict
            .iter()
            .copied()
            .map(StrictProgram::program)
            .collect()
    }

    /// `client_config` constrains the process, not a claim, so it applies to
    /// every job. The root command has already gated `programs` on
    /// [`Self::strict_programs`].
    pub fn execute(
        self,
        ws: &PipetteWorkspace,
        http: &HttpClient,
        client_config: &ClientConfiguration,
        programs: &ExternalPrograms,
    ) -> anyhow::Result<()> {
        // The signal handler sets this instead of exiting while no unbounded
        // child is in flight (see the module docs).
        let stop = Arc::new(AtomicBool::new(false));
        pipette_subprocess::register_stop_flag(&stop);
        self.run_until_stopped(ws, http, client_config, programs, &stop)
    }

    /// [`WorkerArgs::execute`] with the stop flag named, so a test can ask the
    /// loop to stop without raising a signal.
    fn run_until_stopped(
        self,
        ws: &PipetteWorkspace,
        http: &HttpClient,
        client_config: &ClientConfiguration,
        programs: &ExternalPrograms,
        stop: &Arc<AtomicBool>,
    ) -> anyhow::Result<()> {
        self.run_until_stopped_on_boot(ws, http, client_config, programs, stop, Boot::current())
    }

    /// Takes the boot so loop tests can model a reboot on any host.
    fn run_until_stopped_on_boot(
        self,
        ws: &PipetteWorkspace,
        http: &HttpClient,
        client_config: &ClientConfiguration,
        programs: &ExternalPrograms,
        stop: &Arc<AtomicBool>,
        boot: Boot,
    ) -> anyhow::Result<()> {
        let registration = ws.registration();
        let auth = &registration.signing_identity()?;
        let server = &registration.server_url()?;
        // Read once, like the key: a job can outlive its workspace (a preempted
        // SLURM job loses `/tmp`), and a read at submit would fall back to DMI.
        let labels = &ws.device().get()?;

        // An absent heartbeat_secs is half the claim's time_window, an absent
        // max_jobs is no limit.
        tracing::info!(
            client_id = %auth.client_id,
            url = %server,
            idle_secs = self.idle_secs,
            idle_jitter_secs = self.idle_jitter_secs,
            heartbeat_secs = self.heartbeat_secs,
            max_jobs = (self.max_jobs > 0).then_some(self.max_jobs),
            "worker starting"
        );

        if self.skip_profile_refresh {
            // Info, not warn: the dispatcher passes it on every start.
            tracing::info!(
                "skipping startup profile refresh (--skip-profile-refresh); \
                 matching may be stale until the next PATCH"
            );
        } else {
            let caps = installed_runtime_capabilities(&ws.runtimes())?;
            tracing::info!(
                flags = ?caps,
                "refreshing profile with the installed-runtime capability flags"
            );
            // The gate can hold startup for five minutes, so a ^C reaches it as
            // it reaches a readiness gate.
            let stop_requested = || stop.load(Ordering::SeqCst);
            // One rewriting row on a terminal; the usual lines in a journal.
            let reindex_progress = ReindexProgress::new();
            let report = |status: &ReindexStatus| reindex_progress.report(status);
            let profile = refresh_profile_at_startup(
                &registration,
                labels,
                http,
                caps,
                ReindexWait::default(),
                &ReindexCallbacks::new()
                    .with_status(&report)
                    .with_cancel(&stop_requested),
            )?;
            // `None` only comes back from a wait that read the flag, so this
            // line cannot claim a wait that never ran.
            let Some(profile) = profile else {
                tracing::info!("stop requested during the reindex wait; not claiming");
                return Ok(());
            };
            if profile.status != ClientStatus::Approved {
                anyhow::bail!(
                    "client not approved (status={}); an operator must run \
                     `pipette-mgmt clients approve {}`, then restart this client",
                    profile.status,
                    profile.client_id
                );
            }
            // Approval is not the whole gate: registration seeds
            // `submit_results` alone and claiming takes `perform_work` (mgmt's
            // `client-integration.md` §1), so an approved client can still be
            // refused every claim.
            if let Some(missing) = missing_worker_permission(profile.permissions.as_deref()) {
                anyhow::bail!(
                    "client {} is approved but does not hold `{missing}`; an operator must \
                     run `pipette-mgmt clients permission add {} {missing}`, then restart \
                     this client",
                    profile.client_id,
                    profile.client_id
                );
            }
            tracing::info!(
                client_status = %profile.status,
                reindex_pending = profile.reindex_pending,
                tags = ?profile.tags,
                flags = ?profile.capabilities,
                "profile ready"
            );
        }

        let idle_base = Duration::from_secs(self.idle_secs);
        let idle_jitter = Duration::from_secs(self.idle_jitter_secs);
        let mut completed = 0u64;
        let mut idle_rounds = 0u64;

        // Fixed for the worker's lifetime: every claim below runs under it.
        tracing::info!(
            client_config = client_config.to_string().as_str(),
            "client configuration"
        );
        // Before the first claim: a job's prepare would otherwise share the
        // device with what a dead worker left running.
        pipette_subprocess::reap_orphans();
        #[cfg(target_os = "android")]
        crate::client::worker::android_doze::disable_doze()?;
        tracing::info!("entering claim loop");
        loop {
            if stop.load(Ordering::SeqCst) {
                tracing::info!(completed, "stop requested; leaving claim loop");
                return Ok(());
            }
            tracing::debug!(completed, idle_rounds, "polling claim");
            let Some(poll) = claim_with_retry(server, http, auth, |d| sleep_interruptible(d, stop))
            else {
                tracing::info!("stop requested during claim backoff; leaving claim loop");
                return Ok(());
            };
            match poll {
                ClaimPoll::Job(job) => {
                    // The job's `device_time_ms` counts from here.
                    let claimed_at = Instant::now();
                    idle_rounds = 0;
                    log_claimed_job(&job);
                    // Every line the job leaves, from this crate, the runtime
                    // and store crates and the heartbeat thread, carries its id,
                    // so no call site below prefixes it.
                    let span = tracing::info_span!("job", id = %job.job_id);
                    let _job = span.enter();
                    if let Some(shard) = ws.eval_shard() {
                        tracing::info!(
                            %shard,
                            "eval shard: running its group into a part file; nothing is reported"
                        );
                        // Keeps the call heard of on the pool, where the listener answers
                        // it; a lost lease changes nothing for a run that reports nothing.
                        let _heartbeat = spawn_heartbeat(
                            server.clone(),
                            http.clone(),
                            auth.clone(),
                            job.job_id.clone(),
                            resolve_heartbeat_interval(&job.time_window, self.heartbeat_secs),
                            Arc::default(),
                        );
                        // No in-flight marker: a killed shard's would later fail the
                        // parent's job for good. One claim whatever --max-jobs says, so a
                        // failed group exits 1 instead of claiming the job again.
                        return execute_job(
                            ws,
                            server,
                            http,
                            auth,
                            labels,
                            &job,
                            client_config,
                            programs,
                            stop,
                        )
                        .map(drop);
                    }
                    let marker_store = ws.in_flight_marker();
                    let previous = match marker_store.read() {
                        Ok(marker) => marker,
                        Err(err) => {
                            tracing::warn!(error = %format_args!("{err:#}"), "could not read in-flight marker; running claim without reboot detection");
                            None
                        }
                    };
                    let decision = in_flight::decide(previous.as_ref(), &job.job_id, &boot);
                    match decision {
                        ClaimDecision::StaleJob { old_job_id } => {
                            tracing::warn!(
                                old_job_id,
                                "stale in-flight marker from an earlier lease; running claimed job"
                            );
                        }
                        ClaimDecision::SameBoot => {
                            tracing::warn!(boot_id = ?boot.id, "same job has an in-flight marker on this boot; running claimed job");
                        }
                        ClaimDecision::EarlierBoot => {
                            tracing::warn!(
                                alive_at = ?previous.as_ref().and_then(|m| m.alive_at),
                                boot_started_at = ?boot.started_at,
                                "in-flight marker from an earlier boot does not show the job alive at the reboot; running claimed job"
                            );
                        }
                        ClaimDecision::Run | ClaimDecision::Rebooted { .. } => {}
                    }
                    let outcome = if let ClaimDecision::Rebooted {
                        old_boot_id,
                        new_boot_id,
                    } = decision
                    {
                        let reason = reboot_failure_reason(old_boot_id, new_boot_id);
                        tracing::error!(
                            old_boot_id,
                            new_boot_id,
                            "device rebooted during this job; reporting failure without running it again"
                        );
                        submit_failure(
                            server,
                            http,
                            auth,
                            &job,
                            format_failure_reason(&reason),
                            false,
                            Some(FailureCode::DeviceError),
                            Some(device_time_ms(claimed_at.elapsed())),
                            stop,
                        )
                        .map(|delivered| {
                            if !delivered {
                                tracing::warn!(
                                    "reboot failure not delivered; keeping the in-flight marker to report it on the next claim"
                                );
                            } else if let Err(err) = marker_store.remove() {
                                tracing::warn!(error = %format_args!("{err:#}"), "could not remove in-flight marker after the reboot failure");
                            }
                        })
                    } else {
                        let marker = InFlight::new(&job.job_id, boot.id.clone());
                        let in_flight = match InFlightRun::start(marker_store, &marker, stop) {
                            Ok(in_flight) => {
                                tracing::info!(boot_id = ?marker.boot_id, started_at = %marker.started_at, "in-flight marker written");
                                Some(in_flight)
                            }
                            Err(err) => {
                                tracing::warn!(error = %format_args!("{err:#}"), "could not write in-flight marker; running claim without reboot detection");
                                None
                            }
                        };
                        run_claimed_job(
                            ws,
                            server,
                            http,
                            auth,
                            labels,
                            &job,
                            claimed_at,
                            self.heartbeat_secs,
                            client_config,
                            programs,
                            stop,
                            in_flight,
                        )
                    };
                    match outcome {
                        Ok(()) => {
                            completed += 1;
                            tracing::info!(
                                completed,
                                max_jobs = (self.max_jobs > 0).then_some(self.max_jobs),
                                "done"
                            );
                            if self.max_jobs > 0 && completed >= self.max_jobs {
                                tracing::info!(
                                    max_jobs = self.max_jobs,
                                    "reached --max-jobs; leaving claim loop"
                                );
                                return Ok(());
                            }
                        }
                        // The job cannot be handed back; see [`SubmitRefusedError`].
                        Err(e) if e.downcast_ref::<SubmitRefusedError>().is_some() => {
                            tracing::error!(
                                error = %format_args!("{e:#}"),
                                "this job's failure report could not be submitted; stopping \
                                 rather than re-claiming the same job"
                            );
                            return Err(e);
                        }
                        Err(e) if stopped_by_held_touch(&e) => {
                            tracing::error!(
                                error = %format_args!("{e:#}"),
                                "a held touch on the touchscreen failed the readiness gate; \
                                 stopping the worker"
                            );
                            return Err(e);
                        }
                        Err(e) => {
                            // run_claimed_job already reported failure/abort when
                            // it could; log and continue the loop.
                            tracing::error!(
                                error = %format_args!("{e:#}"),
                                "ended with error (continuing claim loop)"
                            );
                        }
                    }
                }
                ClaimPoll::Idle => {
                    idle_rounds += 1;
                    let wait = idle_wait_with_jitter(idle_base, idle_jitter);
                    tracing::info!(
                        retry_in = ?wait,
                        ?idle_base,
                        ?idle_jitter,
                        idle_round = idle_rounds,
                        "no work available (204); sleeping"
                    );
                    if sleep_interruptible(wait, stop).is_break() {
                        tracing::info!("stop requested during the idle wait; leaving claim loop");
                        return Ok(());
                    }
                }
                ClaimPoll::Refused(err) => {
                    // The server's own line, because the two refusals answer
                    // alike and need different commands.
                    tracing::error!(
                        client_id = %auth.client_id,
                        error = %err,
                        "claim returned 403; stopping"
                    );
                    return Err(anyhow::Error::new(err).context(format!(
                        "the management server refused client {}. Its reason says which \
                         of the two an operator has to clear: approve the client \
                         (`pipette-mgmt clients approve {}`), or grant it `perform_work` \
                         (`pipette-mgmt clients permission add {} perform_work`). Restart \
                         this client afterwards",
                        auth.client_id, auth.client_id, auth.client_id
                    )));
                }
                ClaimPoll::Transient(e) => {
                    // claim_with_retry absorbs transients; this arm is defensive.
                    tracing::error!(error = %e, "unexpected unhandled claim transient");
                    return Err(e.into());
                }
                ClaimPoll::Fatal(err) => {
                    tracing::error!(error = %err, "fatal claim error");
                    return Err(err.into());
                }
            }
        }
    }
}

fn log_claimed_job(job: &ClaimedJob) {
    tracing::info!(
        job_id = %job.job_id,
        benchmark_id = job.benchmark_id.as_str(),
        time_window = %job.time_window,
        expires_at = job.expires_at.as_ref().map(tracing::field::display),
        "claimed job"
    );
    // Untyped here: `execute_job` logs the model and runtime once the payload
    // has parsed. This is the copy an operator reads when it does not, so any
    // plan-carried access token is stripped.
    tracing::debug!(
        job_id = %job.job_id,
        spec = %redacted_spec(&job.spec),
        "claimed job spec"
    );
}

fn reboot_failure_reason(old_boot_id: &str, new_boot_id: &str) -> String {
    let reason = format!(
        "device rebooted during this job (old boot ID: {old_boot_id}; new boot ID: {new_boot_id})"
    );
    #[cfg(target_os = "android")]
    if let Some(boot_reason) = pipette_device::boot_reason() {
        return format!("{reason}; sys.boot.reason: {boot_reason}");
    }
    reason
}

/// Claim until the server answers something the loop can act on, absorbing
/// transients at [`exponential_backoff`] from a second to a minute: a worker is
/// a daemon, and a server that is down now will be up later. `None` means
/// `sleep` said stop. `sleep` is injected so a test can watch the retry
/// without living through the wait; production passes [`sleep_interruptible`].
fn claim_with_retry(
    server: &ServerUrl,
    http: &HttpClient,
    auth: &ClientIdentity,
    mut sleep: impl FnMut(Duration) -> ControlFlow<()>,
) -> Option<ClaimPoll> {
    const BACKOFF_FIRST: Duration = Duration::from_secs(1);
    const BACKOFF_CAP: Duration = Duration::from_secs(60);
    let mut attempt = 0u32;
    loop {
        attempt += 1;
        tracing::debug!(attempt, "claim attempt");
        match poll_claim(server, http, auth) {
            ClaimPoll::Transient(e) => {
                let backoff = exponential_backoff(attempt, BACKOFF_FIRST, BACKOFF_CAP);
                tracing::warn!(
                    attempt,
                    error = %e,
                    retry_in = ?backoff,
                    "claim attempt transient failure; retrying"
                );
                if sleep(backoff).is_break() {
                    return None;
                }
            }
            other @ (ClaimPoll::Job(_)
            | ClaimPoll::Idle
            | ClaimPoll::Refused(_)
            | ClaimPoll::Fatal(_)) => return Some(other),
        }
    }
}

/// Execute one claimed job end-to-end: heartbeat thread + run + submit. The
/// submission's `device_time_ms` runs from `claimed_at` to the end of the run,
/// and `in_flight` covers the run but not the submit.
#[expect(
    clippy::too_many_arguments,
    reason = "the claim, what runs it, its in-flight marker, and the flag that ends its waits"
)]
fn run_claimed_job(
    ws: &PipetteWorkspace,
    server: &ServerUrl,
    http: &HttpClient,
    auth: &ClientIdentity,
    labels: &DeviceLabels,
    job: &ClaimedJob,
    claimed_at: Instant,
    heartbeat_secs: Option<u64>,
    client_config: &ClientConfiguration,
    programs: &ExternalPrograms,
    stop: &AtomicBool,
    in_flight: Option<InFlightRun>,
) -> anyhow::Result<()> {
    let protocol_hb = resolve_heartbeat_interval(&job.time_window, None);
    let hb_every = resolve_heartbeat_interval(&job.time_window, heartbeat_secs);
    if heartbeat_secs.is_some() && hb_every > protocol_hb {
        tracing::warn!(
            heartbeat_interval = ?hb_every,
            time_window = %job.time_window,
            half_time_window = ?protocol_hb,
            "--heartbeat-secs is longer than half time_window; lease may expire between \
             heartbeats"
        );
    }
    // heartbeat_secs is present only when the interval came from the flag.
    tracing::info!(
        heartbeat_interval = ?hb_every,
        heartbeat_secs,
        time_window = %job.time_window,
        "starting run"
    );
    // Shared with the heartbeat thread: set on lease abort so the submit is
    // skipped after the (uncancellable) run finishes.
    let lease_lost = Arc::new(AtomicBool::new(false));
    // Dropping `hb` disconnects the channel and joins the thread.
    let hb = spawn_heartbeat(
        server.clone(),
        http.clone(),
        auth.clone(),
        job.job_id.clone(),
        hb_every,
        Arc::clone(&lease_lost),
    );

    let run_result = execute_job(
        ws,
        server,
        http,
        auth,
        labels,
        job,
        client_config,
        programs,
        stop,
    );
    // The result, or the error, is ready: the clock stops before the submit.
    let elapsed = claimed_at.elapsed();

    drop(hb);
    tracing::debug!("heartbeat thread stopped");
    // A reboot during the submit loses only the report, so the next claim runs
    // the job again rather than failing it.
    drop(in_flight);

    submit_finished_run(
        ws,
        server,
        http,
        auth,
        job,
        run_result,
        elapsed,
        lease_lost.load(Ordering::SeqCst),
        stop,
    )
}

/// What a finished run owes the server, once the heartbeat thread has stopped.
/// Split from [`run_claimed_job`] so the lost-lease rule (report nothing at
/// all) is testable without executing a benchmark.
#[expect(
    clippy::too_many_arguments,
    reason = "the claim, its run, and the report sinks"
)]
fn submit_finished_run(
    ws: &PipetteWorkspace,
    server: &ServerUrl,
    http: &HttpClient,
    auth: &ClientIdentity,
    job: &ClaimedJob,
    run_result: anyhow::Result<(
        pipette_plan_types::BenchmarkSubmissionPayload,
        crate::results::BenchmarkResultExtras,
        Option<EvalRunDigest>,
    )>,
    elapsed: Duration,
    lease_lost: bool,
    stop: &AtomicBool,
) -> anyhow::Result<()> {
    // A held touch stays until an operator removes the cause, so every later
    // claim would fail the same way. Not submitted, as for a ^C, so the cell
    // keeps its attempts; the claim loop then stops the worker.
    let run_result = match run_result {
        Err(err) if stopped_by_held_touch(&err) => {
            tracing::warn!(
                ?elapsed,
                error = %format_args!("{err:#}"),
                "run stopped by a held touch; skipping submit (the cell keeps its attempts)"
            );
            return Err(err);
        }
        other => other,
    };
    if lease_lost {
        tracing::warn!(
            ?elapsed,
            "lease was lost during the run; skipping submit \
             (result would be rejected — see module docs)"
        );
        return Ok(());
    }

    match run_result {
        Ok((mut payload, extras, eval_checkpoint)) => {
            tracing::info!(?elapsed, "run succeeded");
            payload.device_time_ms = Some(device_time_ms(elapsed));
            submit_success(
                ws,
                server,
                http,
                auth,
                job,
                &payload,
                &extras,
                eval_checkpoint.as_ref(),
                stop,
            )
        }
        // A run the operator stopped is not a cell that failed: the server
        // reads a failure as evidence about the cell and spends one of its
        // attempts on it. Dropped rather than submitted, as a lost lease is.
        Err(err) if stop.load(Ordering::SeqCst) && was_cancelled(&err) => {
            tracing::warn!(
                ?elapsed,
                error = %format_args!("{err:#}"),
                "run stopped by request; skipping submit (the cell keeps its attempts)"
            );
            Ok(())
        }
        Err(err) => {
            let retriable = classify_run_error(&err);
            let code = failure_code(&err);
            let reason = format_failure_reason(&format!("{err:#}"));
            tracing::warn!(
                ?elapsed,
                retriable,
                ?code,
                error = %reason,
                "run failed"
            );
            let device_time = Some(device_time_ms(elapsed));
            submit_failure(
                server,
                http,
                auth,
                job,
                reason,
                retriable,
                code,
                device_time,
                stop,
            )
            .map(|_delivered| ())
        }
    }
}

/// Owns the heartbeat background thread. Dropping it disconnects the channel
/// and joins: the thread lives in `recv_timeout`, so it costs no CPU between
/// ticks and wakes at once when the run finishes.
struct HeartbeatGuard {
    /// Dropping this disconnects the receiver and unblocks `recv_timeout`.
    stop_tx: Option<mpsc::Sender<()>>,
    join: Option<thread::JoinHandle<()>>,
}

impl Drop for HeartbeatGuard {
    fn drop(&mut self) {
        // Disconnect first so a thread blocked in recv_timeout wakes now,
        // then join so we don't tear down mid-HTTP call without waiting.
        drop(self.stop_tx.take());
        if let Some(handle) = self.join.take() {
            let _ = handle.join();
        }
    }
}

fn spawn_heartbeat(
    server: ServerUrl,
    http: HttpClient,
    auth: ClientIdentity,
    job_id: String,
    every: Duration,
    lease_lost: Arc<AtomicBool>,
) -> HeartbeatGuard {
    // Unit channel: disconnect (dropping the sender) is the stop signal, and
    // `recv_timeout` sleeps the full interval rather than polling.
    let (stop_tx, stop_rx) = mpsc::channel::<()>();
    // Spans are per thread: the `job` span is entered again on this one.
    let span = tracing::Span::current();
    let join = thread::spawn(move || {
        let _job = span.enter();
        // First heartbeat after `every`, not immediately — the claim itself
        // just granted a fresh window.
        tracing::info!(
            heartbeat_interval = ?every,
            "heartbeat thread started (first tick after one interval, sleeping via recv_timeout)"
        );
        let mut wait = every;
        let mut ticks = 0u32;
        let mut failures = 0u32;
        loop {
            match stop_rx.recv_timeout(wait) {
                Ok(()) | Err(RecvTimeoutError::Disconnected) => {
                    tracing::debug!(ticks, "heartbeat thread stopping");
                    return;
                }
                Err(RecvTimeoutError::Timeout) => {
                    tracing::debug!(tick = ticks + 1, "sending heartbeat");
                    match keepalive_lease(&server, &http, &auth, &job_id) {
                        LeaseKeepalive::Ok => {
                            ticks += 1;
                            tracing::info!(
                                tick = ticks,
                                heartbeat_interval = ?every,
                                "heartbeat ok"
                            );
                            failures = 0;
                            wait = every;
                        }
                        LeaseKeepalive::Abort => {
                            lease_lost.store(true, Ordering::SeqCst);
                            tracing::error!(
                                tick = ticks + 1,
                                "heartbeat aborted lease; flagging lease_lost — run continues \
                                 (no cancel path) but submit will be skipped"
                            );
                            return;
                        }
                        LeaseKeepalive::Retry => {
                            // Jittered so a fleet facing one outage does not retry in
                            // lockstep; capped at a quarter interval so several tries land
                            // in the time_window/2 slack before the lease lapses.
                            const RETRY_FIRST: Duration = Duration::from_secs(2);
                            failures = failures.saturating_add(1);
                            wait = jittered(exponential_backoff(failures, RETRY_FIRST, every / 4));
                            tracing::warn!(
                                tick = ticks + 1,
                                failures,
                                retry_in = ?wait,
                                "heartbeat transient; retrying"
                            );
                        }
                    }
                }
            }
        }
    });
    HeartbeatGuard {
        stop_tx: Some(stop_tx),
        join: Some(join),
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "the claim, what it runs against, and the stop flag"
)]
fn execute_job(
    ws: &PipetteWorkspace,
    server: &ServerUrl,
    http: &HttpClient,
    auth: &ClientIdentity,
    labels: &DeviceLabels,
    job: &ClaimedJob,
    client_config: &ClientConfiguration,
    programs: &ExternalPrograms,
    stop: &AtomicBool,
) -> anyhow::Result<(
    pipette_plan_types::BenchmarkSubmissionPayload,
    crate::results::BenchmarkResultExtras,
    Option<EvalRunDigest>,
)> {
    let spec = run_spec_from_claim(job)?;
    tracing::info!(model = %spec.model, runtime = %spec.runtime, "cell");

    let benchmark = ensure_claim_benchmark_cached(ws, server, http, auth, &job.benchmark_id)?;
    tracing::info!(
        benchmark_id = %job.benchmark_id,
        benchmark_type = %benchmark.benchmark_type(),
        "benchmark ready"
    );

    tracing::info!("dispatching via shared ClientRunSpec path");
    // `prepare` adds the cell's pins; the policy only carries the cap here.
    let artifacts = ws.artifacts(http, programs);
    run_cell(
        &spec,
        benchmark,
        &artifacts,
        ws,
        labels,
        client_config,
        // A worker takes the mode from `PIPETTE_READINESS_HELD_TOUCH` or the
        // host's `[readiness.android]` table.
        None,
        &|| stop.load(Ordering::SeqCst),
        programs,
    )
}

fn ensure_claim_benchmark_cached(
    ws: &PipetteWorkspace,
    server: &ServerUrl,
    http: &HttpClient,
    auth: &ClientIdentity,
    benchmark_id: &str,
) -> anyhow::Result<pipette_plan_types::BenchmarkDefinition> {
    // A worker never syncs, and mgmt refuses a result whose `benchmark_params` differ from its
    // catalog's, so revalidate the cache by its ETag, which hashes the whole body (corpus pin
    // and eval samples included). Nothing here writes the sync state.
    let id = pipette_plan_types::BenchmarkId::try_new(benchmark_id.to_owned())
        .with_context(|| format!("invalid benchmark id \"{}\"", benchmark_id.escape_default()))?;
    let store = ws.benchmarks();
    let (cached, cached_etag) = match store.get_remote(&id) {
        Ok(Some((def, etag))) => (Some(def), etag),
        Ok(None) => (None, None),
        Err(e) => {
            tracing::warn!(
                %benchmark_id,
                error = %format_args!("{e:#}"),
                "could not read the cached benchmark; fetching it from the server"
            );
            (None, None)
        }
    };
    let if_none_match = cached_etag.as_ref().map(IfNoneMatch::from);
    // Signed: a claim can name a benchmark only this device's principal sees.
    let fetched = client_api::get_benchmark_conditional(
        server,
        http,
        Some(auth),
        benchmark_id,
        if_none_match.as_ref(),
    );
    let (remote, etag) = match (fetched, cached) {
        (Ok(ConditionalResponse::Modified { value, etag }), cached) => {
            match (cached.is_some(), if_none_match.is_some()) {
                (true, true) => {
                    tracing::info!(%benchmark_id, outcome = "changed", "the server changed the cached benchmark; running its new definition")
                }
                (true, false) => {
                    tracing::info!(%benchmark_id, outcome = "fetched", "the cached benchmark has no ETag to revalidate; running the server's definition")
                }
                (false, _) => {}
            }
            (value, etag)
        }
        (Ok(ConditionalResponse::NotModified { .. }), Some(cached)) => {
            tracing::info!(%benchmark_id, outcome = "current", "cached benchmark is current; running it");
            return Ok(cached);
        }
        (Ok(ConditionalResponse::NotModified { .. }), None) => {
            anyhow::bail!("GET /benchmarks/{benchmark_id}: 304 without a cached definition")
        }
        // An outage must not stop a host that holds the definition. A `404` does: mgmt refuses
        // the result of a benchmark it dropped, so it fails the claim as it does uncached.
        (Err(error), Some(cached)) if error.http_status() != Some(404) => {
            tracing::warn!(
                %benchmark_id,
                outcome = "unchecked",
                error = %error,
                "could not revalidate the cached benchmark; running it unchecked"
            );
            return Ok(cached);
        }
        (Err(error), _) => {
            return Err(error).with_context(|| format!("GET /benchmarks/{benchmark_id}"));
        }
    };
    let converted = crate::benchmarks::ConvertedRemote::try_from(remote)
        .with_context(|| format!("converting remote benchmark `{benchmark_id}`"))?;
    match store.put_remote(&converted, etag.as_ref()) {
        Ok(()) => tracing::debug!(%benchmark_id, "benchmark cached in catalog store"),
        Err(e) => tracing::warn!(%benchmark_id, error = %e, "failed to cache remote benchmark"),
    }
    Ok(converted.into_definition())
}

/// Post a finished run against its claim, then keep a local copy and retire
/// an eval's checkpoint; a refused or abandoned submit leaves the checkpoint
/// for the job's next run. The failure that reports a refused submit carries
/// the payload's `device_time_ms`.
#[expect(
    clippy::too_many_arguments,
    reason = "the claim, the result, and the flag that ends the submit backoff"
)]
fn submit_success(
    ws: &PipetteWorkspace,
    server: &ServerUrl,
    http: &HttpClient,
    auth: &ClientIdentity,
    job: &ClaimedJob,
    payload: &pipette_plan_types::BenchmarkSubmissionPayload,
    extras: &crate::results::BenchmarkResultExtras,
    eval_checkpoint: Option<&EvalRunDigest>,
    stop: &AtomicBool,
) -> anyhow::Result<()> {
    tracing::info!(
        device_time_ms = payload.device_time_ms,
        "posting the success body to the server"
    );
    let mut body = serde_json::to_value(payload).context("serialize success payload")?;
    body = attach_claim_to_success_payload(body, job)?;
    tracing::debug!(
        keys = ?body.as_object().map(|o| o.keys().cloned().collect::<Vec<_>>()),
        "success payload"
    );
    match post_completion(server, http, auth, &job.job_id, &body, |d| {
        sleep_interruptible(d, stop)
    }) {
        Ok(true) => {
            store_completed_result(&ws.results(), &job.job_id, payload, extras);
            finalize_eval_checkpoint(ws, eval_checkpoint);
            Ok(())
        }
        // Never reached the server: the lease lapses and the job is served
        // again, so keep the checkpoint for that run and file nothing as synced.
        Ok(false) => Ok(()),
        Err(e) => match e.downcast::<SubmitRefusedError>() {
            Ok(refused) => {
                let device_time = payload.device_time_ms;
                report_refused_submission(server, http, auth, job, &refused, device_time, stop)
            }
            Err(other) => Err(other),
        },
    }
}

/// Report a result that could not be submitted as a retriable failure, so the
/// job returns to the queue: the server refused this build's body, not the cell
/// (mgmt's `client-integration.md` §6). The reason carries the server's answer,
/// the only record of why the result was lost.
fn report_refused_submission(
    server: &ServerUrl,
    http: &HttpClient,
    auth: &ClientIdentity,
    job: &ClaimedJob,
    refused: &SubmitRefusedError,
    device_time_ms: Option<u64>,
    stop: &AtomicBool,
) -> anyhow::Result<()> {
    let reason = format_failure_reason(&format!(
        "this run's result could not be submitted: {refused}"
    ));
    tracing::warn!(
        error = %reason,
        "reporting the refused result as a retriable failure"
    );
    submit_failure(
        server,
        http,
        auth,
        job,
        reason,
        true,
        Some(FailureCode::HarnessError),
        device_time_ms,
        stop,
    )
    .map(|_delivered| ())
}

/// Keep a completed job's result on disk under the id and location `sync`
/// promotes a CLI result to, so `pipette results` shows both alike. Stored only
/// once the server holds it, and never pending, since a pending copy would let
/// `sync` post it again through the generic endpoint. Best-effort: a failed
/// write must not fail the job.
fn store_completed_result(
    results: &crate::results::ResultsStore,
    job_id: &str,
    payload: &pipette_plan_types::BenchmarkSubmissionPayload,
    extras: &crate::results::BenchmarkResultExtras,
) {
    let location = crate::results::BenchmarkResultLocation::RemoteSynced;
    match crate::results::check_job_id(job_id)
        .and_then(|job_id| results.save_result(location, job_id, payload, extras))
    {
        Ok(()) => tracing::info!(?location, "result stored"),
        Err(e) => tracing::warn!(
            error = %format_args!("{e:#}"),
            "submitted but not stored locally"
        ),
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "the claim, the failure, and the flag that ends the submit backoff"
)]
fn submit_failure(
    server: &ServerUrl,
    http: &HttpClient,
    auth: &ClientIdentity,
    job: &ClaimedJob,
    reason: String,
    retriable: bool,
    failure_code: Option<FailureCode>,
    device_time_ms: Option<u64>,
    stop: &AtomicBool,
) -> anyhow::Result<bool> {
    let body = failure_from_claim(job, reason, retriable, failure_code, device_time_ms);
    tracing::info!(
        retriable,
        ?failure_code,
        device_time_ms = body.device_time_ms,
        "posting the failure body to the server"
    );
    post_completion(server, http, auth, &job.job_id, &body, |d| {
        sleep_interruptible(d, stop)
    })
}

/// The retrying POST both outcomes share. `sleep` is injected, as for
/// [`claim_with_retry`], because living through an exhausted backoff costs
/// about 14 minutes. A `sleep` that says stop ends the backoff with the same
/// answer; the log says which, since one is a broken server and the other an
/// operator.
///
/// `Ok(true)`: the server has the report (accepted, or dropped for a
/// superseded lease). `Ok(false)`: it never reached the server.
fn post_completion<P: serde::Serialize>(
    server: &ServerUrl,
    http: &HttpClient,
    auth: &ClientIdentity,
    job_id: &str,
    payload: &P,
    mut sleep: impl FnMut(Duration) -> ControlFlow<()>,
) -> anyhow::Result<bool> {
    tracing::info!("POST /benchmarks (plan-attached)");
    let mut stopped = false;
    match submit_plan_result_with_backoff(server, http, auth, job_id, payload, |d| {
        tracing::warn!(retry_in = ?d, "submit transient; backing off before retry");
        let flow = sleep(d);
        stopped = flow.is_break();
        flow
    })? {
        SubmitDisposition::Accepted {
            job_id: accepted_id,
        } => {
            // The server's answer, not the span's claim id: they differ only
            // if the server filed the result somewhere else.
            tracing::info!(server_job_id = %accepted_id, "submit accepted");
            Ok(true)
        }
        SubmitDisposition::Dropped => {
            tracing::warn!("submit dropped (409 superseded, or reclaim after 404 failed)");
            Ok(true)
        }
        SubmitDisposition::RetryTransient if stopped => {
            // Same loss as below, but not the same cause: this one is the
            // operator's ^C landing in the backoff, not a server that stayed
            // down for the whole of it.
            tracing::warn!(
                "stop requested while backing off; result dropped \
                 (no local pending artifact)"
            );
            Ok(false)
        }
        SubmitDisposition::RetryTransient => {
            // v1: a result never submitted leaves no pending artifact (module docs).
            tracing::warn!(
                "gave up submitting after repeated transient failures; \
                 result dropped (no local pending artifact)"
            );
            Ok(false)
        }
        SubmitDisposition::Fatal(e) => {
            tracing::error!(error = %e, "submit fatal error");
            Err(SubmitRefusedError(e).into())
        }
    }
}

/// Whether a run ended because the readiness gate read the stop flag, rather
/// than failing on its own while a stop happened to be pending. The gate is the
/// only cancel seam in a run and every runtime propagates its error with `?`;
/// if one ever stops preserving it this reads `false` and the failure is
/// submitted.
fn was_cancelled(err: &anyhow::Error) -> bool {
    // The chain, not the top frame, so a runtime that folds the gate's error
    // into its own type is read the same way `classify_run_error` reads it.
    err.chain().any(|cause| {
        matches!(
            cause.downcast_ref::<pipette_readiness::ReadinessError>(),
            Some(pipette_readiness::ReadinessError::Cancelled)
        )
    })
}

/// Whether the readiness gate failed the run on a held touch
/// (`[readiness.android] held_touch = "fail"`). Read along the chain, as
/// [`was_cancelled`] reads it.
fn stopped_by_held_touch(err: &anyhow::Error) -> bool {
    err.chain().any(|cause| {
        matches!(
            cause.downcast_ref::<pipette_readiness::ReadinessError>(),
            Some(pipette_readiness::ReadinessError::HeldTouch { .. })
        )
    })
}

/// The permission a worker needs but the profile says it does not hold.
///
/// `None` for a profile that carries no `permissions` at all: an older server
/// omits the field, and refusing there would be this client deciding a grant
/// it cannot see is missing. An empty array is the server saying it holds
/// none, which is a refusal.
fn missing_worker_permission(permissions: Option<&[String]>) -> Option<&'static str> {
    const PERFORM_WORK: &str = "perform_work";
    let held = permissions?;
    (!held.iter().any(|p| p == PERFORM_WORK)).then_some(PERFORM_WORK)
}

/// Sleep `total`, stopping at the first
/// [`pipette_wait::CANCEL_CHECK_INTERVAL`] boundary after `stop` is set.
/// [`ControlFlow::Break`] when cut short, the shape the backoff callbacks take.
/// The slices are what make the flag worth setting: a worker's waits run from a
/// 60 s claim backoff to the minutes a submit backoff can reach.
fn sleep_interruptible(total: Duration, stop: &AtomicBool) -> ControlFlow<()> {
    pipette_wait::sleep_cancellably(total, &|| stop.load(Ordering::SeqCst), &thread::sleep)
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use httpmock::prelude::*;
    use serde_json::json;

    use pipette_mgmt_client::PrivateKey;
    use pipette_ops::eval_completions::run_request_digest;
    use pipette_ops::run_context::RunContext;
    use pipette_plan_types::{
        BenchmarkDefinition, BenchmarkEvalCompletion, BenchmarkId, BenchmarkParams,
        BenchmarkSubmissionPayload, ClientId, DeclaredBound, EvalBenchmark, EvalId, Model,
        PositiveCount, RunRequest, Runtime,
    };

    use super::*;
    use crate::client::claim::UnrunnableClaimError;
    use crate::identity::IdentityRegistration;
    use crate::results::{BenchmarkResultExtras, BenchmarkResultLocation};
    use crate::test_support::{
        IMG80, TempWorkspace, cache_remote, cache_remote_with_etag, img80_wire, logged_text,
        remote_detail_path, remote_state_path, write_public_2026_08_1_img80_cache,
    };

    /// A claim with a runnable cell — the body `poll_claim` decodes in
    /// `client/worker/protocol.rs`, kept in the same shape here.
    const CLAIM_JOB_BODY: &str = r#"{
        "job_id": "job-abc",
        "benchmark_id": "prefill_throughput_256",
        "time_window": "PT10M",
        "model_name": "m",
        "spec": {
            "benchmark": "prefill_throughput_256",
            "model": {
                "type": "gguf_text",
                "source": "huggingface",
                "org": "o",
                "repo_name": "r",
                "path": "m-Q4_0.gguf"
            },
            "runtime": {
                "type": "llamacpp_cli_stock_tools",
                "source": "github_release",
                "repository_url": "github.com/ggml-org/llama.cpp",
                "version": "b5000",
                "flavor": "macos-arm64"
            }
        }
    }"#;

    /// The same envelope carrying a cell no client can read. A claim the worker
    /// can answer without running anything, which is what lets the loop tests
    /// below drive `execute` end to end.
    const UNRUNNABLE_CLAIM_JOB_BODY: &str = r#"{
        "job_id": "job-abc",
        "benchmark_id": "prefill_throughput_256",
        "time_window": "PT10M",
        "spec": { "model": "not-a-model" }
    }"#;

    fn test_http() -> anyhow::Result<HttpClient> {
        HttpClient::new("pipette-test/0").map_err(|e| anyhow::anyhow!("{e}"))
    }

    fn server_url(server: &MockServer) -> anyhow::Result<ServerUrl> {
        Ok(ServerUrl::try_new(server.base_url())?)
    }

    fn test_auth() -> anyhow::Result<ClientIdentity> {
        Ok(ClientIdentity {
            client_id: ClientId::try_new("ev1_test")?,
            private_key_hex: PrivateKey::try_new("11".repeat(32))?,
        })
    }

    /// Give the workspace the identity `execute` signs with. No request in these
    /// tests is verified, so any 32 bytes of hex is a usable Ed25519 seed.
    fn register(ws: &PipetteWorkspace, server: &MockServer) -> anyhow::Result<()> {
        let registration = ws.registration();
        registration.put(&IdentityRegistration {
            client_id: ClientId::try_new("ev1_test")?,
            server_url: server_url(server)?,
        })?;
        registration.put_private_key(&PrivateKey::try_new("11".repeat(32))?)?;
        Ok(())
    }

    fn claimed_job(body: &str) -> anyhow::Result<ClaimedJob> {
        Ok(serde_json::from_str(body)?)
    }

    /// A finished run's payload, in the shape `run_cell` returns it — the metric
    /// sits at the top level because `result` is `#[serde(flatten)]`.
    fn finished_payload() -> anyhow::Result<BenchmarkSubmissionPayload> {
        Ok(serde_json::from_value(json!({
            "benchmark_id": "prefill_throughput_256",
            "device": {},
            "thermal": {},
            "submitted_at": "2026-07-30T00:00:00Z",
            "prefill_time_ms": 1.0,
        }))?)
    }

    fn benchmark_body() -> serde_json::Value {
        json!({
            "benchmark_id": "decode_throughput_512_100",
            "benchmark_type": "decode_throughput",
            "parameter_prefill_tokens": 512,
            "parameter_decode_tokens": 100,
        })
    }

    /// A `PATCH /clients/me` response carrying `status`, with the reindex gate
    /// already lifted — the reindex wait itself is `client/worker/profile.rs`'s.
    fn profile_body(status: &str) -> String {
        profile_body_with(status, Some(&["perform_work", "submit_results"]))
    }

    /// The profile a server returns, with `permissions` spelled as the case
    /// needs: `None` omits the key, as a server predating the field does.
    fn profile_body_with(status: &str, permissions: Option<&[&str]>) -> String {
        let mut body = json!({
            "client_id": "ev1_test",
            "organization": "acme",
            "client_details": "a box",
            "contact_email": "ops@example.com",
            "status": status,
        });
        if let (Some(permissions), Some(object)) = (permissions, body.as_object_mut()) {
            object.insert("permissions".into(), json!(permissions));
        }
        body.to_string()
    }

    /// The flags the loop tests start from: no startup PATCH, so a case that is
    /// about claiming is only about claiming, and a zero idle wait, so a `204`
    /// nobody expected cannot park the test for five minutes.
    fn worker_args() -> WorkerArgs {
        WorkerArgs {
            idle_secs: 0,
            idle_jitter_secs: 0,
            heartbeat_secs: None,
            max_jobs: 0,
            skip_profile_refresh: true,
            // The gate this asks for runs before dispatch, so a loop test
            // never reaches it.
            strict: Vec::new(),
        }
    }

    /// Poll `condition` until it holds, or fail after a generous deadline, so a
    /// heartbeat test waits for state instead of a fixed span a loaded machine
    /// would overrun.
    fn wait_until(what: &str, mut condition: impl FnMut() -> bool) -> anyhow::Result<()> {
        let deadline = Instant::now() + Duration::from_secs(10);
        while Instant::now() < deadline {
            if condition() {
                return Ok(());
            }
            thread::sleep(Duration::from_millis(5));
        }
        anyhow::bail!("timed out waiting for {what}")
    }

    /// Heartbeat period the tests below run the thread at.
    const FAST_TICK: Duration = Duration::from_millis(20);
    /// Long enough for many [`FAST_TICK`] intervals to pass, so "nothing more
    /// happened" is a claim about the thread rather than about the clock.
    const SEVERAL_TICKS: Duration = Duration::from_millis(300);

    // --- the claim retry schedule ---

    /// The loop sees only the answer that followed the transients, and the
    /// waits are the claim's schedule: one second, doubling, capped at a minute.
    #[test]
    fn claim_with_retry_absorbs_transients_and_returns_the_next_answer() -> anyhow::Result<()> {
        let server = MockServer::start();
        let mut down = server.mock(|when, then| {
            when.method(POST).path("/plans/claim");
            then.status(503).body("down");
        });

        let mut sleeps = Vec::new();
        let poll = claim_with_retry(&server_url(&server)?, &test_http()?, &test_auth()?, |d| {
            sleeps.push(d);
            // Eight transients, then the answer. httpmock has no response
            // sequencing and its matchers take no captured state, so the
            // server is re-pointed from inside the wait the loop is already
            // taking.
            if sleeps.len() == 8 {
                down.delete();
                server.mock(|when, then| {
                    when.method(POST).path("/plans/claim");
                    then.status(204);
                });
            }
            ControlFlow::Continue(())
        });

        assert!(matches!(poll, Some(ClaimPoll::Idle)), "got {poll:?}");
        assert_eq!(
            sleeps,
            [1, 2, 4, 8, 16, 32, 60, 60]
                .map(Duration::from_secs)
                .to_vec(),
            "one wait per absorbed transient, on the schedule"
        );
        Ok(())
    }

    /// Everything that is not a transient is the loop's to act on at once — a
    /// refusal waited out would be a worker that never reports it is unapproved.
    #[rstest::rstest]
    #[case::not_approved(403, "pending")]
    #[case::fatal(400, "bad")]
    fn a_claim_answer_that_is_not_transient_is_returned_without_waiting(
        #[case] status: u16,
        #[case] body: &str,
    ) -> anyhow::Result<()> {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(POST).path("/plans/claim");
            then.status(status).body(body);
        });
        let mut sleeps = Vec::new();
        let poll = claim_with_retry(&server_url(&server)?, &test_http()?, &test_auth()?, |d| {
            sleeps.push(d);
            ControlFlow::Continue(())
        });
        match status {
            403 => assert!(matches!(poll, Some(ClaimPoll::Refused(_))), "got {poll:?}"),
            _ => assert!(matches!(poll, Some(ClaimPoll::Fatal(_))), "got {poll:?}"),
        }
        assert!(sleeps.is_empty(), "nothing to wait for: {sleeps:?}");
        Ok(())
    }

    /// The one exception to "retries forever": the loop hears `None` rather
    /// than an answer it would go on to act on.
    #[test]
    fn claim_with_retry_gives_up_when_the_wait_is_stopped() -> anyhow::Result<()> {
        let server = MockServer::start();
        let down = server.mock(|when, then| {
            when.method(POST).path("/plans/claim");
            then.status(503).body("down");
        });

        let poll = claim_with_retry(&server_url(&server)?, &test_http()?, &test_auth()?, |_| {
            ControlFlow::Break(())
        });

        assert!(poll.is_none(), "got {poll:?}");
        assert_eq!(down.calls(), 1, "no attempt after the stop");
        Ok(())
    }

    // --- the heartbeat thread that holds the lease ---

    #[test]
    fn the_heartbeat_ticks_until_the_guard_is_dropped() -> anyhow::Result<()> {
        let server = MockServer::start();
        let beat = server.mock(|when, then| {
            when.method(PUT).path("/plans/job-abc/heartbeat");
            then.status(200);
        });
        let lease_lost = Arc::new(AtomicBool::new(false));
        let hb = spawn_heartbeat(
            server_url(&server)?,
            test_http()?,
            test_auth()?,
            "job-abc".to_owned(),
            FAST_TICK,
            Arc::clone(&lease_lost),
        );

        wait_until("two heartbeats", || beat.calls() >= 2)?;
        drop(hb);

        // The guard joins before it returns, so no tick can land after this read.
        let ticks = beat.calls();
        thread::sleep(SEVERAL_TICKS);
        assert_eq!(
            beat.calls(),
            ticks,
            "the thread kept beating after the drop"
        );
        assert!(
            !lease_lost.load(Ordering::SeqCst),
            "a renewed lease was never lost"
        );
        Ok(())
    }

    /// A superseded lease sets the flag the submit consults, and the thread
    /// stops: with no cancel path it has nothing left to do.
    #[test]
    fn a_superseded_lease_flags_lease_lost_and_stops_the_thread() -> anyhow::Result<()> {
        let server = MockServer::start();
        let beat = server.mock(|when, then| {
            when.method(PUT).path("/plans/job-abc/heartbeat");
            then.status(409).body("superseded");
        });
        let lease_lost = Arc::new(AtomicBool::new(false));
        let hb = spawn_heartbeat(
            server_url(&server)?,
            test_http()?,
            test_auth()?,
            "job-abc".to_owned(),
            FAST_TICK,
            Arc::clone(&lease_lost),
        );

        wait_until("the lease-lost flag", || lease_lost.load(Ordering::SeqCst))?;
        thread::sleep(SEVERAL_TICKS);
        assert_eq!(
            beat.calls(),
            1,
            "the thread went on beating on a lease the server had given away"
        );
        drop(hb);
        Ok(())
    }

    /// Disconnecting the channel is what wakes `recv_timeout` mid-wait.
    #[test]
    fn a_transient_heartbeat_backs_off_within_the_slack_and_the_guard_still_stops_at_once()
    -> anyhow::Result<()> {
        let server = MockServer::start();
        let beat = server.mock(|when, then| {
            when.method(PUT).path("/plans/job-abc/heartbeat");
            then.status(503).body("down");
        });
        let lease_lost = Arc::new(AtomicBool::new(false));
        // Four ticks, so retries wait up to one FAST_TICK (a quarter interval).
        let hb = spawn_heartbeat(
            server_url(&server)?,
            test_http()?,
            test_auth()?,
            "job-abc".to_owned(),
            FAST_TICK * 4,
            Arc::clone(&lease_lost),
        );

        let started = Instant::now();
        wait_until("three heartbeats", || beat.calls() >= 3)?;
        // One 80 ms tick then two retries of at most 20 ms; a fixed 2 s wait takes 4 s.
        assert!(
            started.elapsed() < Duration::from_secs(3),
            "a retry waited past a quarter interval ({:?})",
            started.elapsed()
        );
        // Jittered waits of up to 20 ms give about 25 retries here; a tight loop, well over 100.
        thread::sleep(SEVERAL_TICKS);
        assert!(
            beat.calls() < 60,
            "a transient was retried without backing off ({} heartbeats)",
            beat.calls()
        );

        let stopping = Instant::now();
        drop(hb);
        assert!(
            stopping.elapsed() < Duration::from_secs(1),
            "the drop waited out the retry timer ({:?})",
            stopping.elapsed()
        );
        assert!(
            !lease_lost.load(Ordering::SeqCst),
            "a transient is not a lost lease"
        );
        Ok(())
    }

    // --- what a finished run owes the server ---

    /// Not submitted, and not filed locally either: the server has already
    /// given the job to someone else.
    #[rstest::rstest]
    #[case::a_successful_run(true)]
    #[case::a_failed_run(false)]
    fn a_run_whose_lease_was_lost_is_not_submitted(#[case] succeeded: bool) -> anyhow::Result<()> {
        let temp = TempWorkspace::new("worker-lease-lost")?;
        let server = MockServer::start();
        let posted = server.mock(|when, then| {
            when.method(POST).path("/benchmarks");
            then.status(202).body(r#"{"job_id":"job-abc"}"#);
        });
        let run_result = if succeeded {
            Ok((finished_payload()?, BenchmarkResultExtras::default(), None))
        } else {
            Err(anyhow::anyhow!("out of memory"))
        };

        submit_finished_run(
            &temp.ws,
            &server_url(&server)?,
            &test_http()?,
            &test_auth()?,
            &claimed_job(CLAIM_JOB_BODY)?,
            run_result,
            Duration::from_secs(3),
            true,
            &AtomicBool::new(false),
        )?;

        assert_eq!(posted.calls(), 0, "a lost lease must not be submitted");
        assert!(
            temp.ws
                .results()
                .list_ids(BenchmarkResultLocation::RemoteSynced)?
                .is_empty(),
            "nothing was submitted, so nothing is filed as synced"
        );
        Ok(())
    }

    /// A held touch is not submitted, so the cell keeps its attempts, and the
    /// error comes back so the claim loop stops the worker. A lost lease does
    /// not change that, and neither does context added on the way up.
    #[rstest::rstest]
    #[case::bare(false, false)]
    #[case::with_context(true, false)]
    #[case::lease_lost(false, true)]
    fn a_held_touch_is_not_submitted_and_stops_the_worker(
        #[case] with_context: bool,
        #[case] lease_lost: bool,
    ) -> anyhow::Result<()> {
        let temp = TempWorkspace::new("worker-held-touch")?;
        let server = MockServer::start();
        let posted = server.mock(|when, then| {
            when.method(POST);
            then.status(202).body(r#"{"job_id":"job-abc"}"#);
        });
        let err = anyhow::Error::from(pipette_readiness::ReadinessError::HeldTouch {
            pointers: 1,
            irq_per_sec: 120,
        });
        let err = if with_context {
            err.context("prefill_throughput_256 readiness gate")
        } else {
            err
        };

        let returned = submit_finished_run(
            &temp.ws,
            &server_url(&server)?,
            &test_http()?,
            &test_auth()?,
            &claimed_job(CLAIM_JOB_BODY)?,
            Err(err),
            Duration::from_secs(3),
            lease_lost,
            &AtomicBool::new(false),
        )
        .err()
        .context("a held touch must come back as an error")?;

        assert!(stopped_by_held_touch(&returned), "{returned:#}");
        assert_eq!(posted.calls(), 0, "a held touch must not be submitted");
        Ok(())
    }

    /// How a run ended, for the cases below.
    #[derive(Clone, Copy)]
    enum RunOutcome {
        Succeeded,
        /// The readiness gate read the stop flag.
        Cancelled,
        /// The run failed for a reason of its own.
        Failed,
    }

    /// The gate's cancellation is dropped, so the ^C costs the cell none of its
    /// attempts; a run that finished first is still submitted, and so is one
    /// that failed on its own.
    #[rstest::rstest]
    #[case::the_gate_cancelled_the_run(RunOutcome::Cancelled, 0)]
    #[case::the_run_finished_first(RunOutcome::Succeeded, 1)]
    #[case::the_run_failed_on_its_own(RunOutcome::Failed, 1)]
    fn a_stop_drops_only_the_cell_it_interrupted(
        #[case] outcome: RunOutcome,
        #[case] want_posts: usize,
    ) -> anyhow::Result<()> {
        let temp = TempWorkspace::new("worker-stopped-run")?;
        let server = MockServer::start();
        let posted = server.mock(|when, then| {
            when.method(POST).path("/benchmarks");
            then.status(202).body(r#"{"job_id":"job-abc"}"#);
        });
        let run_result = match outcome {
            RunOutcome::Succeeded => {
                Ok((finished_payload()?, BenchmarkResultExtras::default(), None))
            }
            RunOutcome::Cancelled => Err(anyhow::Error::from(
                pipette_readiness::ReadinessError::Cancelled,
            )),
            RunOutcome::Failed => Err(anyhow::anyhow!("out of memory")),
        };

        submit_finished_run(
            &temp.ws,
            &server_url(&server)?,
            &test_http()?,
            &test_auth()?,
            &claimed_job(CLAIM_JOB_BODY)?,
            run_result,
            Duration::from_secs(3),
            false,
            &AtomicBool::new(true),
        )?;

        assert_eq!(posted.calls(), want_posts);
        Ok(())
    }

    #[test]
    fn a_successful_run_that_kept_its_lease_is_submitted_and_stored() -> anyhow::Result<()> {
        let temp = TempWorkspace::new("worker-submit-success")?;
        let server = MockServer::start();
        let posted = server.mock(|when, then| {
            when.method(POST)
                .path("/benchmarks")
                // The claim echo the server's binding check looks for, and
                // the time from the claim.
                .json_body_includes(r#"{"job_id": "job-abc", "device_time_ms": 3000}"#);
            then.status(202).body(r#"{"job_id":"job-abc"}"#);
        });

        submit_finished_run(
            &temp.ws,
            &server_url(&server)?,
            &test_http()?,
            &test_auth()?,
            &claimed_job(CLAIM_JOB_BODY)?,
            Ok((finished_payload()?, BenchmarkResultExtras::default(), None)),
            Duration::from_secs(3),
            false,
            &AtomicBool::new(false),
        )?;

        posted.assert();
        let results = temp.ws.results();
        let stored = results.load_payload(BenchmarkResultLocation::RemoteSynced, "job-abc")?;
        assert_eq!(stored["benchmark_id"], "prefill_throughput_256");
        assert_eq!(stored["device_time_ms"], 3000);
        Ok(())
    }

    /// The disposition is `classify_run_error`'s. The reason is the whole error
    /// chain, since the context names the step that died and the limit it ran
    /// past; asserted on the wire.
    #[rstest::rstest]
    #[case::a_local_failure_is_retriable(
        anyhow::anyhow!("out of memory").context("loading the model"),
        true,
        "out_of_memory"
    )]
    #[case::an_unreadable_claim_is_terminal(
        anyhow::Error::from(UnrunnableClaimError::NoSpec),
        false,
        "harness_error"
    )]
    #[case::a_warm_up_past_its_limit_is_terminal(
        anyhow::Error::from(pipette_subprocess::ExitError::TimedOut {
            name: "llama-bench".to_owned(),
            timeout: Duration::from_secs(3600),
            tail: String::new(),
        })
        .context("decode_throughput_4096_100 warm-up"),
        false,
        "runtime_error"
    )]
    #[case::an_attempt_past_attempt_timeout_seconds_is_terminal(
        anyhow::Error::from(pipette_ops::attempt::AttemptTimeoutError {
            overrun: pipette_ops::attempt::Overrun {
                what: "rep 1".to_owned(),
                limit: Duration::from_secs(10),
                elapsed: Duration::from_millis(10_020),
            },
            engine_error: Some("llama-server exited with signal: 9 (SIGKILL)".to_owned()),
        }),
        false,
        "attempt_timeout"
    )]
    fn a_failed_run_is_reported_against_the_claim(
        #[case] error: anyhow::Error,
        #[case] retriable: bool,
        #[case] failure_code: &str,
    ) -> anyhow::Result<()> {
        let temp = TempWorkspace::new("worker-submit-failure")?;
        let server = MockServer::start();
        // Timestamp's closing bracket, the chain verbatim, then the field's end
        // quote: a prefix of the error, or a stray elision marker, both fail.
        // Matched against the raw body, so the fixture must not need JSON escaping.
        let expected_reason = format!("] {error:#}\"");
        let posted = server.mock(|when, then| {
            when.method(POST)
                .path("/benchmarks")
                .json_body_includes(
                    json!({
                        "message_type": "failure",
                        "job_id": "job-abc",
                        "benchmark_id": "prefill_throughput_256",
                        "retriable": retriable,
                        "failure_code": failure_code,
                        "device_time_ms": 3000,
                    })
                    .to_string(),
                )
                .body_includes(&expected_reason);
            then.status(202).body(r#"{"job_id":"job-abc"}"#);
        });

        submit_finished_run(
            &temp.ws,
            &server_url(&server)?,
            &test_http()?,
            &test_auth()?,
            &claimed_job(CLAIM_JOB_BODY)?,
            Err(error),
            Duration::from_secs(3),
            false,
            &AtomicBool::new(false),
        )?;

        posted.assert();
        assert!(
            temp.ws
                .results()
                .list_ids(BenchmarkResultLocation::RemoteSynced)?
                .is_empty(),
            "a failure carries no result to file"
        );
        Ok(())
    }

    // --- the four answers a submit can come back with ---

    /// Accepted and dropped are both answers the loop moves past; only a
    /// definitive refusal stops the job. Each is reached on the first attempt, so
    /// no backoff is spent.
    #[rstest::rstest]
    #[case::accepted(202, r#"{"job_id":"job-abc"}"#, false)]
    #[case::dropped_as_superseded(409, "superseded", false)]
    #[case::refused_outright(400, "bad", true)]
    fn post_completion_translates_the_servers_answer(
        #[case] status: u16,
        #[case] body: &str,
        #[case] expect_error: bool,
    ) -> anyhow::Result<()> {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(POST).path("/benchmarks");
            then.status(status).body(body);
        });
        let mut sleeps = Vec::new();
        let outcome = post_completion(
            &server_url(&server)?,
            &test_http()?,
            &test_auth()?,
            "job-abc",
            &json!({ "job_id": "job-abc" }),
            |d| {
                sleeps.push(d);
                ControlFlow::Continue(())
            },
        );
        assert_eq!(outcome.is_err(), expect_error, "{outcome:?}");
        assert!(sleeps.is_empty(), "answered first time: {sleeps:?}");
        Ok(())
    }

    /// An exhausted backoff drops the outcome rather than failing the job: the
    /// run is over, the module keeps no pending artifact for it, and the loop
    /// goes on to the next claim (v1 limits, module docs).
    #[test]
    fn post_completion_gives_up_when_the_backoff_is_exhausted() -> anyhow::Result<()> {
        let server = MockServer::start();
        let posted = server.mock(|when, then| {
            when.method(POST).path("/benchmarks");
            then.status(503).body("down");
        });
        let mut sleeps = Vec::new();
        post_completion(
            &server_url(&server)?,
            &test_http()?,
            &test_auth()?,
            "job-abc",
            &json!({ "job_id": "job-abc" }),
            |d| {
                sleeps.push(d);
                ControlFlow::Continue(())
            },
        )?;

        assert_eq!(posted.calls(), 32, "the bounded retry in `protocol.rs`");
        assert_eq!(
            sleeps.iter().take(6).copied().collect::<Vec<_>>(),
            [1, 2, 4, 8, 16, 30].map(Duration::from_secs).to_vec(),
            "doubling to the submit cap"
        );
        Ok(())
    }

    /// The same drop, reached in a second: the result is lost either way, so
    /// the job still returns `Ok` and the loop leaves on its own terms.
    #[test]
    fn post_completion_stops_backing_off_when_asked_to_stop() -> anyhow::Result<()> {
        let server = MockServer::start();
        let posted = server.mock(|when, then| {
            when.method(POST).path("/benchmarks");
            then.status(503).body("down");
        });

        post_completion(
            &server_url(&server)?,
            &test_http()?,
            &test_auth()?,
            "job-abc",
            &json!({ "job_id": "job-abc" }),
            |_| ControlFlow::Break(()),
        )?;

        assert_eq!(posted.calls(), 1, "no resubmit after the stop");
        Ok(())
    }

    // --- the claim's benchmark ---

    fn cached_detail(ws: &PipetteWorkspace, benchmark_id: &str) -> std::path::PathBuf {
        remote_detail_path(&ws.root().join("benchmarks"), benchmark_id)
    }

    /// Matches only a signed `v1` request; `None` requires no `If-None-Match` at all.
    fn mock_signed<'s>(
        server: &'s MockServer,
        path: &str,
        if_none_match: Option<&str>,
        status: u16,
        body: &str,
        etag: Option<&str>,
    ) -> httpmock::Mock<'s> {
        server.mock(|when, then| {
            let when = when
                .method(GET)
                .path(path)
                .header_exists("X-Client-Id")
                .header_exists("X-Timestamp")
                .header_exists("X-Nonce")
                .header_exists("X-Signature");
            match if_none_match {
                Some(tag) => when.header("if-none-match", tag),
                None => when.header_missing("if-none-match"),
            };
            let then = then.status(status).body(body);
            if let Some(etag) = etag {
                then.header("etag", etag);
            }
        })
    }

    fn mock_detail<'s>(
        server: &'s MockServer,
        id: &str,
        status: u16,
        body: &str,
    ) -> httpmock::Mock<'s> {
        mock_signed(
            server,
            &format!("/benchmarks/{id}"),
            None,
            status,
            body,
            None,
        )
    }

    fn mock_revalidated<'s>(
        server: &'s MockServer,
        id: &str,
        etag: &str,
        status: u16,
        body: &str,
        new_etag: Option<&str>,
    ) -> httpmock::Mock<'s> {
        mock_signed(
            server,
            &format!("/benchmarks/{id}"),
            Some(etag),
            status,
            body,
            new_etag,
        )
    }

    fn cached_etag(ws: &PipetteWorkspace, id: &str) -> anyhow::Result<Option<String>> {
        Ok(ws
            .benchmarks()
            .remote_detail_etag(&BenchmarkId::try_new(id.to_owned())?)
            .map(|etag| etag.as_str().to_owned()))
    }

    /// A sync state on disk, which a claim must leave byte for byte.
    struct SyncStateOnDisk {
        path: std::path::PathBuf,
        bytes: Vec<u8>,
    }

    impl SyncStateOnDisk {
        fn write(ws: &PipetteWorkspace) -> anyhow::Result<Self> {
            let path = remote_state_path(&ws.root().join("benchmarks"), "sync");
            if let Some(dir) = path.parent() {
                std::fs::create_dir_all(dir)?;
            }
            let bytes = serde_json::to_vec_pretty(&json!({
                "pulled_at": "2026-10-01T00:00:00Z",
                "benchmark_count": 1,
                "server_url": "https://mgmt.example.com",
                "benchmarks_etag": "\"list\"",
            }))?;
            std::fs::write(&path, &bytes)?;
            Ok(Self { path, bytes })
        }

        fn assert_unchanged(&self) -> anyhow::Result<()> {
            assert_eq!(
                std::fs::read(&self.path)?,
                self.bytes,
                "a claim wrote the sync state"
            );
            Ok(())
        }
    }

    struct LoggedClaim {
        definition: anyhow::Result<BenchmarkDefinition>,
        log: String,
    }

    fn claim_logged(
        ws: &PipetteWorkspace,
        server: &ServerUrl,
        id: &str,
    ) -> anyhow::Result<LoggedClaim> {
        let (definition, log) = logged_text(|| {
            ensure_claim_benchmark_cached(ws, server, &test_http()?, &test_auth()?, id)
        })?;
        Ok(LoggedClaim { definition, log })
    }

    fn prefill_body(id: &str, prefill_tokens: u32) -> serde_json::Value {
        json!({
            "benchmark_id": id,
            "benchmark_type": "prefill_throughput",
            "parameter_prefill_tokens": prefill_tokens,
        })
    }

    fn vl_body(id: &str, num_images: u32) -> serde_json::Value {
        json!({
            "benchmark_id": id,
            "benchmark_type": "vl_throughput",
            "parameter_image_width": 512,
            "parameter_image_height": 512,
            "parameter_text_tokens": 1024,
            "parameter_decode_tokens": 64,
            "parameter_num_images": num_images,
        })
    }

    fn eval_body() -> serde_json::Value {
        json!({
            "benchmark_id": "eval_mmlu",
            "benchmark_type": "eval",
            "parameter_eval_id": "mmlu_pro",
            "parameter_dataset_name": "edge_2026.03.1",
            "parameter_max_tokens": 64,
            "samples": [{"id": "q1", "messages": []}],
        })
    }

    fn id_of(body: &serde_json::Value) -> anyhow::Result<String> {
        Ok(body["benchmark_id"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("fixture without an id"))?
            .to_owned())
    }

    /// An eval keeps the samples it was cached with: the ETag covers the whole body.
    #[rstest::rstest]
    #[case::decode(benchmark_body())]
    #[case::eval(eval_body())]
    fn a_current_cache_is_revalidated_and_run(
        #[case] cached: serde_json::Value,
    ) -> anyhow::Result<()> {
        let temp = TempWorkspace::new("worker-benchmark-current")?;
        let id = id_of(&cached)?;
        cache_remote_with_etag(&temp.ws.benchmarks(), cached.clone(), Some("\"v1\""))?;
        let detail = cached_detail(&temp.ws, &id);
        let cached_bytes = std::fs::read(&detail)?;
        let sync_state = SyncStateOnDisk::write(&temp.ws)?;
        let server = MockServer::start();
        let revalidated = mock_revalidated(&server, &id, "\"v1\"", 304, "", None);
        let fetched = mock_detail(&server, &id, 500, "an unconditional fetch");

        let LoggedClaim { definition, log } = claim_logged(&temp.ws, &server_url(&server)?, &id)?;

        assert_eq!(
            definition?,
            serde_json::from_value::<BenchmarkDefinition>(cached)?
        );
        revalidated.assert_calls(1);
        fetched.assert_calls(0);
        assert_eq!(std::fs::read(&detail)?, cached_bytes, "the cache changed");
        assert_eq!(cached_etag(&temp.ws, &id)?.as_deref(), Some("\"v1\""));
        assert!(log.contains("outcome=\"current\""), "{log}");
        assert!(!log.contains("WARN"), "{log}");
        sync_state.assert_unchanged()
    }

    /// The new body runs, so the result declares what mgmt checks it against.
    #[rstest::rstest]
    #[case::prefill_tokens(
        prefill_body("prefill_throughput_x", 256),
        prefill_body("prefill_throughput_x", 512)
    )]
    #[case::vl_num_images(
        vl_body("vl_throughput_512x512_text1024_decode64_img5", 1),
        vl_body("vl_throughput_512x512_text1024_decode64_img5", 5)
    )]
    fn a_changed_benchmark_replaces_the_cache(
        #[case] cached: serde_json::Value,
        #[case] served: serde_json::Value,
    ) -> anyhow::Result<()> {
        let temp = TempWorkspace::new("worker-benchmark-changed")?;
        let id = id_of(&cached)?;
        cache_remote_with_etag(&temp.ws.benchmarks(), cached, Some("\"v1\""))?;
        let sync_state = SyncStateOnDisk::write(&temp.ws)?;
        let server = MockServer::start();
        let revalidated = mock_revalidated(
            &server,
            &id,
            "\"v1\"",
            200,
            &served.to_string(),
            Some("\"v2\""),
        );

        let LoggedClaim { definition, log } = claim_logged(&temp.ws, &server_url(&server)?, &id)?;

        assert_eq!(
            definition?,
            serde_json::from_value::<BenchmarkDefinition>(served.clone())?
        );
        revalidated.assert_calls(1);
        let on_disk: serde_json::Value =
            pipette_ops::fs::read_json_file(&cached_detail(&temp.ws, &id))?;
        assert_eq!(on_disk, served, "the new body was not cached");
        assert_eq!(cached_etag(&temp.ws, &id)?.as_deref(), Some("\"v2\""));
        assert!(log.contains("outcome=\"changed\""), "{log}");
        assert!(!log.contains("WARN"), "{log}");
        sync_state.assert_unchanged()
    }

    /// The next claim revalidates the ETag the whole fetch came with.
    #[test]
    fn a_cache_without_an_etag_is_fetched_whole() -> anyhow::Result<()> {
        let temp = TempWorkspace::new("worker-benchmark-no-etag")?;
        let id = "prefill_throughput_x";
        cache_remote(&temp.ws.benchmarks(), prefill_body(id, 256))?;
        let server = MockServer::start();
        let fetched = mock_signed(
            &server,
            &format!("/benchmarks/{id}"),
            None,
            200,
            &prefill_body(id, 512).to_string(),
            Some("\"v1\""),
        );
        let revalidated = mock_revalidated(&server, id, "\"v1\"", 304, "", None);

        let LoggedClaim { definition, log } = claim_logged(&temp.ws, &server_url(&server)?, id)?;
        assert_eq!(
            definition?,
            serde_json::from_value::<BenchmarkDefinition>(prefill_body(id, 512))?
        );
        assert!(log.contains("outcome=\"fetched\""), "{log}");
        fetched.assert_calls(1);
        assert_eq!(cached_etag(&temp.ws, id)?.as_deref(), Some("\"v1\""));

        claim_logged(&temp.ws, &server_url(&server)?, id)?.definition?;
        fetched.assert_calls(1);
        revalidated.assert_calls(1);
        Ok(())
    }

    #[rstest::rstest]
    #[case::server_error(500, "boom")]
    #[case::unauthorized(401, r#"{"error":"invalid signature"}"#)]
    #[case::refused(403, r#"{"error":"client is pending approval"}"#)]
    #[case::malformed(200, "not json")]
    fn a_revalidation_without_an_answer_runs_the_cache(
        #[case] status: u16,
        #[case] body: &str,
    ) -> anyhow::Result<()> {
        let temp = TempWorkspace::new("worker-benchmark-unchecked")?;
        let id = "decode_throughput_512_100";
        cache_remote_with_etag(&temp.ws.benchmarks(), benchmark_body(), Some("\"v1\""))?;
        let detail = cached_detail(&temp.ws, id);
        let cached_bytes = std::fs::read(&detail)?;
        let sync_state = SyncStateOnDisk::write(&temp.ws)?;
        let server = MockServer::start();
        let revalidated = mock_revalidated(&server, id, "\"v1\"", status, body, None);

        let LoggedClaim { definition, log } = claim_logged(&temp.ws, &server_url(&server)?, id)?;

        assert_eq!(
            definition?,
            serde_json::from_value::<BenchmarkDefinition>(benchmark_body())?
        );
        revalidated.assert_calls(1);
        assert_eq!(std::fs::read(&detail)?, cached_bytes, "the cache changed");
        assert_eq!(cached_etag(&temp.ws, id)?.as_deref(), Some("\"v1\""));
        assert!(
            log.contains("WARN") && log.contains("outcome=\"unchecked\""),
            "{log}"
        );
        sync_state.assert_unchanged()
    }

    /// mgmt refuses the result of a benchmark it dropped. Only mgmt's own `404` is terminal; a
    /// proxy's page returns the job to the queue. The cache is left for `sync` to prune.
    #[rstest::rstest]
    #[case::not_found(r#"{"error":"benchmark not found"}"#, false)]
    #[case::proxy_404("<html>Not Found</html>", true)]
    fn a_cached_benchmark_the_server_does_not_find_fails_the_claim(
        #[case] body: &str,
        #[case] retriable: bool,
    ) -> anyhow::Result<()> {
        let temp = TempWorkspace::new("worker-benchmark-dropped")?;
        let id = "decode_throughput_512_100";
        cache_remote_with_etag(&temp.ws.benchmarks(), benchmark_body(), Some("\"v1\""))?;
        let server = MockServer::start();
        let revalidated = mock_revalidated(&server, id, "\"v1\"", 404, body, None);

        let LoggedClaim { definition, .. } = claim_logged(&temp.ws, &server_url(&server)?, id)?;

        let error = match definition {
            Ok(definition) => anyhow::bail!("a 404 ran {definition:?}"),
            Err(error) => error,
        };
        revalidated.assert_calls(1);
        let text = format!("{error:#}");
        assert!(text.contains(&format!("GET /benchmarks/{id}")), "{text}");
        assert_eq!(classify_run_error(&error), retriable, "{text}");
        assert!(cached_detail(&temp.ws, id).exists());
        Ok(())
    }

    /// Fails as on iOS: the cache no longer declares what mgmt checks the result against.
    #[test]
    fn a_cached_benchmark_changed_into_one_this_build_cannot_run_fails_the_claim()
    -> anyhow::Result<()> {
        let temp = TempWorkspace::new("worker-benchmark-unconvertible-change")?;
        let id = "decode_throughput_512_100";
        cache_remote_with_etag(&temp.ws.benchmarks(), benchmark_body(), Some("\"v1\""))?;
        let cached_bytes = std::fs::read(cached_detail(&temp.ws, id))?;
        let server = MockServer::start();
        let served = json!({"benchmark_id": id, "benchmark_type": "mystery_metric"});
        let revalidated = mock_revalidated(
            &server,
            id,
            "\"v1\"",
            200,
            &served.to_string(),
            Some("\"v2\""),
        );

        let LoggedClaim { definition, .. } = claim_logged(&temp.ws, &server_url(&server)?, id)?;

        let error = match definition {
            Ok(definition) => anyhow::bail!("an unconvertible change ran {definition:?}"),
            Err(error) => error,
        };
        revalidated.assert_calls(1);
        assert!(!classify_run_error(&error), "{error:#}");
        assert_eq!(std::fs::read(cached_detail(&temp.ws, id))?, cached_bytes);
        assert_eq!(cached_etag(&temp.ws, id)?.as_deref(), Some("\"v1\""));
        Ok(())
    }

    #[test]
    fn a_check_that_cannot_reach_the_server_runs_the_cache() -> anyhow::Result<()> {
        let temp = TempWorkspace::new("worker-benchmark-unreachable")?;
        cache_remote_with_etag(&temp.ws.benchmarks(), benchmark_body(), Some("\"v1\""))?;
        let sync_state = SyncStateOnDisk::write(&temp.ws)?;
        // A port nothing listens on: bound, then released.
        let port = std::net::TcpListener::bind("127.0.0.1:0")?
            .local_addr()?
            .port();
        let server = ServerUrl::try_new(format!("http://127.0.0.1:{port}"))?;

        let LoggedClaim { definition, log } =
            claim_logged(&temp.ws, &server, "decode_throughput_512_100")?;

        assert_eq!(
            definition?,
            serde_json::from_value::<BenchmarkDefinition>(benchmark_body())?
        );
        assert!(cached_detail(&temp.ws, "decode_throughput_512_100").exists());
        assert!(
            log.contains("WARN") && log.contains("outcome=\"unchecked\""),
            "{log}"
        );
        sync_state.assert_unchanged()
    }

    /// Its ETag is not sent, and the warning says why rather than calling it a miss.
    #[test]
    fn an_unreadable_cached_benchmark_is_refetched_with_a_warning() -> anyhow::Result<()> {
        let temp = TempWorkspace::new("worker-benchmark-unreadable")?;
        let id = "decode_throughput_512_100";
        let detail = cached_detail(&temp.ws, id);
        std::fs::create_dir_all(
            detail
                .parent()
                .ok_or_else(|| anyhow::anyhow!("a detail path without a parent"))?,
        )?;
        std::fs::write(&detail, "{ not json")?;
        std::fs::write(detail.with_extension("etag"), "\"v1\"")?;
        let server = MockServer::start();
        let fetched = mock_detail(&server, id, 200, &benchmark_body().to_string());

        let LoggedClaim { definition, log } = claim_logged(&temp.ws, &server_url(&server)?, id)?;

        assert_eq!(
            definition?,
            serde_json::from_value::<BenchmarkDefinition>(benchmark_body())?
        );
        fetched.assert_calls(1);
        assert!(
            log.contains("WARN") && log.contains("could not read the cached benchmark"),
            "{log}"
        );
        assert!(!log.contains("benchmark not cached"), "{log}");
        let on_disk: serde_json::Value = pipette_ops::fs::read_json_file(&detail)?;
        assert_eq!(on_disk, benchmark_body());
        Ok(())
    }

    #[rstest::rstest]
    #[case::not_found(404, r#"{"error":"benchmark not found"}"#, false)]
    #[case::down(500, "down", true)]
    fn an_uncached_benchmark_that_does_not_fetch_fails_the_claim(
        #[case] status: u16,
        #[case] body: &str,
        #[case] retriable: bool,
    ) -> anyhow::Result<()> {
        let temp = TempWorkspace::new("worker-benchmark-fetch-fails")?;
        let id = "prefill_throughput_x";
        let server = MockServer::start();
        let fetched = mock_detail(&server, id, status, body);

        let LoggedClaim { definition, .. } = claim_logged(&temp.ws, &server_url(&server)?, id)?;

        let error = match definition {
            Ok(definition) => anyhow::bail!("a failed fetch ran {definition:?}"),
            Err(error) => error,
        };
        fetched.assert_calls(1);
        let text = format!("{error:#}");
        assert!(
            text.contains(&format!("GET /benchmarks/{id}")) && text.contains(&status.to_string()),
            "{text}"
        );
        assert_eq!(classify_run_error(&error), retriable, "{text}");
        assert!(!cached_detail(&temp.ws, id).exists());
        Ok(())
    }

    /// The mock matches only a request carrying the `v1` headers. What is kept is
    /// the server's object, keys this build does not model included.
    #[test]
    fn an_uncached_benchmark_is_fetched_signed_and_cached() -> anyhow::Result<()> {
        let temp = TempWorkspace::new("worker-benchmark-fetch")?;
        let server = MockServer::start();
        let mut sent = benchmark_body();
        sent["org_id"] = json!("org_test");
        sent["parameter_from_the_future"] = json!(3);
        let fetched = server.mock(|when, then| {
            when.method(GET)
                .path("/benchmarks/decode_throughput_512_100")
                .header_exists("X-Client-Id")
                .header_exists("X-Timestamp")
                .header_exists("X-Nonce")
                .header_exists("X-Signature");
            then.status(200).body(sent.to_string());
        });

        let definition = ensure_claim_benchmark_cached(
            &temp.ws,
            &server_url(&server)?,
            &test_http()?,
            &test_auth()?,
            "decode_throughput_512_100",
        )?;

        fetched.assert();
        assert_eq!(
            definition.benchmark_id().as_str(),
            "decode_throughput_512_100"
        );
        let reference = crate::benchmarks::SourcedBenchmarkId::new(
            pipette_plan_types::BenchmarkSource::Remote,
            pipette_plan_types::BenchmarkId::try_new("decode_throughput_512_100".to_owned())?,
        );
        assert!(
            temp.ws.benchmarks().get(&reference)?.is_some(),
            "the fetched definition was not cached"
        );
        let on_disk: serde_json::Value =
            pipette_ops::fs::read_json_file(&cached_detail(&temp.ws, "decode_throughput_512_100"))?;
        assert_eq!(on_disk, sent, "the cache is not what the server sent");
        Ok(())
    }

    /// Public 2026.08.1 cached this 80-image VL benchmark at `remote/<id>.json` with
    /// no image count; a build that read it would run and declare one image, which
    /// mgmt refuses. The second claim revalidates what the first one fetched.
    #[test]
    fn a_cache_file_an_older_build_wrote_is_refetched_not_run() -> anyhow::Result<()> {
        let temp = TempWorkspace::new("worker-benchmark-legacy")?;
        let legacy = write_public_2026_08_1_img80_cache(&temp.ws.root().join("benchmarks"))?;
        let server = MockServer::start();
        let fetched = mock_signed(
            &server,
            &format!("/benchmarks/{IMG80}"),
            None,
            200,
            &img80_wire().to_string(),
            Some("\"img80\""),
        );
        let revalidated = mock_revalidated(&server, IMG80, "\"img80\"", 304, "", None);
        let claim = || {
            ensure_claim_benchmark_cached(
                &temp.ws,
                &server_url(&server)?,
                &test_http()?,
                &test_auth()?,
                IMG80,
            )
        };

        let definition = claim()?;

        let BenchmarkDefinition::VlThroughput(vl) = &definition else {
            anyhow::bail!("expected vl_throughput, got {definition:?}");
        };
        assert_eq!(vl.parameter_num_images.into_inner(), 80);
        let declared = BenchmarkParams::from(&definition);
        let BenchmarkParams::VlThroughput { num_images, .. } = declared else {
            anyhow::bail!("expected vl_throughput params, got {declared:?}");
        };
        assert_eq!(num_images.into_inner(), 80, "the result would declare it");
        fetched.assert_calls(1);
        revalidated.assert_calls(0);
        assert!(legacy.exists(), "the older build's file went");

        claim()?;
        fetched.assert_calls(1);
        revalidated.assert_calls(1);
        Ok(())
    }

    /// The cache is an optimization for the next claim, not a precondition for
    /// this one: a catalog that cannot be written warns and the job runs anyway.
    #[test]
    fn a_benchmark_cache_write_failure_is_a_warning_not_a_refusal() -> anyhow::Result<()> {
        let temp = TempWorkspace::new("worker-benchmark-uncachable")?;
        // Put a file where the remote half of the catalog goes, so both the read
        // and the write below fail on it.
        let remote = temp.ws.root().join("benchmarks/remote");
        std::fs::remove_dir_all(&remote)?;
        std::fs::write(&remote, "not a directory")?;
        let server = MockServer::start();
        let fetched = server.mock(|when, then| {
            when.method(GET)
                .path("/benchmarks/decode_throughput_512_100");
            then.status(200).body(benchmark_body().to_string());
        });

        let definition = ensure_claim_benchmark_cached(
            &temp.ws,
            &server_url(&server)?,
            &test_http()?,
            &test_auth()?,
            "decode_throughput_512_100",
        )?;

        fetched.assert();
        assert_eq!(
            definition.benchmark_id().as_str(),
            "decode_throughput_512_100"
        );
        Ok(())
    }

    /// Cached, it would make `list` warn and every claim refetch it.
    #[test]
    fn an_unconvertible_benchmark_is_refused_and_not_cached() -> anyhow::Result<()> {
        let temp = TempWorkspace::new("worker-benchmark-unconvertible")?;
        let server = MockServer::start();
        let fetched = server.mock(|when, then| {
            when.method(GET).path("/benchmarks/mystery_1");
            then.status(200).body(
                json!({
                    "benchmark_id": "mystery_1",
                    "benchmark_type": "mystery_metric",
                    "parameter_foo": 1,
                })
                .to_string(),
            );
        });

        let error = ensure_claim_benchmark_cached(
            &temp.ws,
            &server_url(&server)?,
            &test_http()?,
            &test_auth()?,
            "mystery_1",
        )
        .err()
        .ok_or_else(|| anyhow::anyhow!("an unknown benchmark type converted"))?;

        fetched.assert_calls(1);
        assert!(
            format!("{error:#}").starts_with("converting remote benchmark `mystery_1`"),
            "unexpected error: {error:#}"
        );
        assert!(
            !cached_detail(&temp.ws, "mystery_1").exists(),
            "a body that does not convert was cached"
        );
        Ok(())
    }

    /// An id the catalog could not file is refused before the fetch: the store is
    /// keyed by `BenchmarkId`, so there would be nowhere to put the answer. The
    /// refusal shows the server's id escaped.
    #[test]
    fn an_unfileable_benchmark_id_is_refused_before_the_fetch() -> anyhow::Result<()> {
        let temp = TempWorkspace::new("worker-benchmark-bad-id")?;
        let server = MockServer::start();
        let fetched = server.mock(|when, then| {
            when.method(GET);
            then.status(200).body(benchmark_body().to_string());
        });

        let error = ensure_claim_benchmark_cached(
            &temp.ws,
            &server_url(&server)?,
            &test_http()?,
            &test_auth()?,
            "not a\nbenchmark id",
        )
        .err()
        .ok_or_else(|| anyhow::anyhow!("expected a refusal"))?;

        assert!(
            format!("{error:#}").starts_with(
                r#"invalid benchmark id "not a\nbenchmark id": benchmark id contains ' '"#
            ),
            "unexpected error: {error:#}"
        );
        assert_eq!(fetched.calls(), 0, "the server was asked anyway");
        Ok(())
    }

    /// The claim is logged before its benchmark id is checked, so the id goes in
    /// quoted and a newline in it cannot start a line of its own.
    #[test]
    fn a_claimed_job_logs_its_benchmark_id_quoted() -> anyhow::Result<()> {
        let job =
            claimed_job(r#"{"job_id": "j1", "benchmark_id": "a\nb", "time_window": "PT10M"}"#)?;
        let ((), log) = logged_text(|| log_claimed_job(&job))?;
        assert_eq!(log.lines().count(), 1, "{log}");
        assert!(log.contains(r#"benchmark_id="a\nb""#), "{log}");
        Ok(())
    }

    /// The wait every pacing decision above is finally spent in. Asserted because
    /// it is the loop's only sleep, and one that returned early would turn a
    /// five-minute idle wait — or a capped claim backoff — into a spin against the
    /// server.
    #[test]
    fn sleep_interruptible_waits_out_the_span_it_was_given() {
        let started = Instant::now();
        let flow = sleep_interruptible(Duration::from_millis(40), &AtomicBool::new(false));
        assert_eq!(flow, ControlFlow::Continue(()));
        assert!(
            started.elapsed() >= Duration::from_millis(40),
            "returned after {:?}",
            started.elapsed()
        );
    }

    /// An hour, because the span it was given must no longer decide when it
    /// returns. This asserts the binding to this flag; the slice width is
    /// `pipette-wait`'s own test.
    #[test]
    fn a_wait_that_is_already_stopped_returns_at_once() {
        let started = Instant::now();
        let flow = sleep_interruptible(Duration::from_secs(3600), &AtomicBool::new(true));
        assert_eq!(flow, ControlFlow::Break(()));
        assert!(
            started.elapsed() < Duration::from_secs(1),
            "waited {:?} on a flag that was already set",
            started.elapsed()
        );
    }

    // --- the claim loop ---

    /// A refusal, because nothing the client does next changes it; an undefined
    /// status, because retrying a server that refuses it is not a daemon
    /// posture. The `403` cases carry the server's message, the only thing that
    /// tells the two refusals apart.
    #[rstest::rstest]
    #[case::not_approved(403, "client is pending approval", Some("client is pending approval"))]
    #[case::no_permission(
        403,
        "client lacks the perform_work permission",
        Some("client lacks the perform_work permission")
    )]
    #[case::fatal(400, "bad request", None)]
    fn a_claim_the_loop_cannot_act_on_stops_the_worker(
        #[case] status: u16,
        #[case] body: &str,
        #[case] says: Option<&str>,
    ) -> anyhow::Result<()> {
        let temp = TempWorkspace::new("worker-claim-stops")?;
        let server = MockServer::start();
        register(&temp.ws, &server)?;
        server.mock(|when, then| {
            when.method(POST).path("/plans/claim");
            then.status(status).body(body);
        });

        let error = worker_args()
            .execute(
                &temp.ws,
                &test_http()?,
                &ClientConfiguration::default(),
                &ExternalPrograms::at_missing_paths(),
            )
            .err()
            .ok_or_else(|| anyhow::anyhow!("expected the worker to stop"))?;

        if let Some(fragment) = says {
            assert!(
                format!("{error:#}").contains(fragment),
                "unexpected error: {error:#}"
            );
        }
        Ok(())
    }

    /// The startup profile is also a gate, so an unapproved client never holds
    /// a lease it cannot use.
    #[test]
    fn a_profile_that_reads_back_unapproved_stops_before_claiming() -> anyhow::Result<()> {
        let temp = TempWorkspace::new("worker-profile-pending")?;
        let server = MockServer::start();
        register(&temp.ws, &server)?;
        let patched = server.mock(|when, then| {
            when.method(httpmock::Method::PATCH).path("/clients/me");
            then.status(200).body(profile_body("pending"));
        });
        let claimed = server.mock(|when, then| {
            when.method(POST).path("/plans/claim");
            then.status(204);
        });

        let args = WorkerArgs {
            skip_profile_refresh: false,
            ..worker_args()
        };
        let error = args
            .execute(
                &temp.ws,
                &test_http()?,
                &ClientConfiguration::default(),
                &ExternalPrograms::at_missing_paths(),
            )
            .err()
            .ok_or_else(|| anyhow::anyhow!("expected the worker to stop"))?;

        patched.assert();
        assert!(
            format!("{error:#}").contains("client not approved"),
            "unexpected error: {error:#}"
        );
        assert_eq!(claimed.calls(), 0, "an unapproved client claimed anyway");
        Ok(())
    }

    /// Claiming takes `perform_work`, which registration does not seed, so an
    /// approved client without it is refused at startup rather than on every
    /// claim. A server that sends no `permissions` at all is not second-guessed:
    /// the `403` on claim still catches it.
    #[rstest::rstest]
    #[case::granted(Some(&["perform_work", "submit_results"][..]), None)]
    #[case::holds_none(Some(&[][..]), Some("does not hold `perform_work`"))]
    #[case::only_submit_results(Some(&["submit_results"][..]), Some("permission add"))]
    #[case::server_does_not_say(None, None)]
    fn an_approved_client_without_perform_work_is_refused_at_startup(
        #[case] permissions: Option<&[&str]>,
        #[case] says: Option<&str>,
    ) -> anyhow::Result<()> {
        let temp = TempWorkspace::new("worker-perform-work")?;
        let server = MockServer::start();
        register(&temp.ws, &server)?;
        let patched = server.mock(|when, then| {
            when.method(httpmock::Method::PATCH).path("/clients/me");
            then.status(200)
                .body(profile_body_with("approved", permissions));
        });
        // A job the worker answers without running anything, so `--max-jobs`
        // ends the loop. A `204` would not: it completes no job, so the loop
        // would poll forever.
        let claimed = server.mock(|when, then| {
            when.method(POST).path("/plans/claim");
            then.status(200).body(UNRUNNABLE_CLAIM_JOB_BODY);
        });
        server.mock(|when, then| {
            when.method(POST).path("/benchmarks");
            then.status(202).body(r#"{"job_id":"job-abc"}"#);
        });

        let args = WorkerArgs {
            max_jobs: 1,
            skip_profile_refresh: false,
            ..worker_args()
        };
        let outcome = args.execute(
            &temp.ws,
            &test_http()?,
            &ClientConfiguration::default(),
            &ExternalPrograms::at_missing_paths(),
        );

        patched.assert();
        match says {
            Some(fragment) => {
                let error = outcome
                    .err()
                    .ok_or_else(|| anyhow::anyhow!("expected the worker to stop"))?;
                assert!(
                    format!("{error:#}").contains(fragment),
                    "unexpected error: {error:#}"
                );
            }
            None => outcome?,
        }
        assert_eq!(
            claimed.calls() > 0,
            says.is_none(),
            "claimed={}",
            claimed.calls()
        );
        Ok(())
    }

    /// `--max-jobs` counts every claim the worker answered, whatever the answer.
    /// Started through the real startup path, so an approved profile is what
    /// lets the loop reach a claim at all.
    #[test]
    fn max_jobs_leaves_the_loop_once_that_many_claims_are_answered() -> anyhow::Result<()> {
        let temp = TempWorkspace::new("worker-max-jobs")?;
        let server = MockServer::start();
        register(&temp.ws, &server)?;
        let patched = server.mock(|when, then| {
            when.method(httpmock::Method::PATCH).path("/clients/me");
            then.status(200).body(profile_body("approved"));
        });
        let claimed = server.mock(|when, then| {
            when.method(POST).path("/plans/claim");
            then.status(200).body(UNRUNNABLE_CLAIM_JOB_BODY);
        });
        let reported = server.mock(|when, then| {
            when.method(POST)
                .path("/benchmarks")
                .json_body_includes(r#"{"message_type": "failure", "retriable": false}"#);
            then.status(202).body(r#"{"job_id":"job-abc"}"#);
        });

        let args = WorkerArgs {
            max_jobs: 1,
            skip_profile_refresh: false,
            ..worker_args()
        };
        args.execute(
            &temp.ws,
            &test_http()?,
            &ClientConfiguration::default(),
            &ExternalPrograms::at_missing_paths(),
        )?;

        patched.assert();
        reported.assert();
        assert_eq!(
            claimed.calls(),
            1,
            "the loop claimed again after reaching --max-jobs"
        );
        Ok(())
    }

    /// The reason carries the server's own words, because the denial marker
    /// keeps it and the all-denied summary reads it back. If that report is
    /// refused too, the refusal reaches the loop as itself, which stops the
    /// worker.
    #[rstest::rstest]
    #[case::report_accepted(202, false)]
    #[case::report_refused_too(400, true)]
    fn a_refused_result_is_reported_as_a_retriable_failure(
        #[case] report_status: u16,
        #[case] stops: bool,
    ) -> anyhow::Result<()> {
        let temp = TempWorkspace::new("worker-refused-result")?;
        let server = MockServer::start();
        // Matched on a measurement, which only a success body carries:
        // `message_type` is omitted on success, since the server defaults it.
        let refused = server.mock(|when, then| {
            when.method(POST)
                .path("/benchmarks")
                .json_body_includes(r#"{"prefill_time_ms": 1.0}"#);
            then.status(400).body("unknown field `prefill_time_ms`");
        });
        let reported = server.mock(|when, then| {
            when.method(POST)
                .path("/benchmarks")
                .json_body_includes(
                    json!({
                        "message_type": "failure",
                        "job_id": "job-abc",
                        "retriable": true,
                        "device_time_ms": 3000,
                    })
                    .to_string(),
                )
                // The server's own refusal, so the denial marker records why.
                .body_includes("unknown field `prefill_time_ms`");
            then.status(report_status).body(r#"{"job_id":"job-abc"}"#);
        });

        let outcome = submit_finished_run(
            &temp.ws,
            &server_url(&server)?,
            &test_http()?,
            &test_auth()?,
            &claimed_job(CLAIM_JOB_BODY)?,
            Ok((finished_payload()?, BenchmarkResultExtras::default(), None)),
            Duration::from_secs(3),
            false,
            &AtomicBool::new(false),
        );

        refused.assert_calls(1);
        reported.assert_calls(1);
        match outcome {
            Ok(()) => assert!(!stops, "a refused failure report was swallowed"),
            Err(error) => assert!(
                stops && error.downcast_ref::<SubmitRefusedError>().is_some(),
                "unexpected error: {error:#}"
            ),
        }
        assert!(
            temp.ws
                .results()
                .list_ids(BenchmarkResultLocation::RemoteSynced)?
                .is_empty(),
            "a result the server never took must not be filed as synced"
        );
        Ok(())
    }

    /// An eval request, only to key a checkpoint.
    fn eval_request() -> anyhow::Result<RunRequest> {
        Ok(RunRequest {
            runtime: DeclaredBound::already_bound(Runtime::AppleFoundation(Default::default())),
            model: DeclaredBound::already_bound(Model::AppleFoundationText),
            runtime_flags: None,
            model_flags: None,
            benchmark_flags: None,
            benchmark: BenchmarkDefinition::Eval(EvalBenchmark {
                benchmark_id: BenchmarkId::try_new("eval_smoke".to_owned())?,
                parameter_eval_id: EvalId::from("math_500"),
                parameter_dataset_name: "d".to_owned(),
                parameter_max_tokens: PositiveCount::try_new(8)?,
                parameter_max_context: None,
                parameter_mcq_choices: None,
                parameter_temperature: None,
                parameter_top_p: None,
                parameter_top_k: None,
                parameter_min_p: None,
                parameter_seed: None,
                samples: Some(vec![json!({"id": "s0", "messages": []})]),
            }),
        })
    }

    /// So a vlm_eval's or an asr_eval's checkpoint is retired with its result,
    /// as an eval's is.
    #[test]
    fn a_vlm_eval_hands_back_its_checkpoint() -> anyhow::Result<()> {
        let eval = eval_request()?;
        assert_eq!(
            crate::run::eval_checkpoint(&eval)?,
            Some(run_request_digest(&eval)?)
        );
        let mut vlm = eval_request()?;
        vlm.runtime = DeclaredBound::already_bound(serde_json::from_value(json!({
            "type": "llamacpp_cli_stock_tools", "source": "remote_archive",
            "url": "http://127.0.0.1/llama.tar.gz", "flavor": "linux-x64-rocm",
        }))?);
        vlm.model = DeclaredBound::already_bound(serde_json::from_value(json!({
            "type": "gguf_vision", "source": "huggingface", "org": "org", "repo_name": "repo",
            "model": "m.gguf", "mmproj": "mmproj.gguf",
        }))?);
        vlm.benchmark = serde_json::from_value(json!({
            "benchmark_type": "vlm_eval",
            "benchmark_id": "vlm_eval_smoke",
            "parameter_lmms_task": "mmstar",
            "parameter_dataset_version": "2026.10.1",
            "parameter_sample_count": 1,
            "parameter_max_tokens": 8,
            "samples": [{"id": "s0", "messages": []}],
        }))?;
        assert_eq!(
            crate::run::eval_checkpoint(&vlm)?,
            Some(run_request_digest(&vlm)?)
        );
        // An asr_eval's too: a checkpoint left behind would hand every
        // transcript to the next run of the cell without running it.
        let mut asr = vlm.clone();
        asr.benchmark = serde_json::from_value(json!({
            "benchmark_type": "asr_eval",
            "benchmark_id": "asr_eval_smoke",
            "parameter_asr_task": "asr_fleurs_nl_nl",
            "parameter_dataset_version": "2026.10.1",
            "parameter_sample_count": 1,
            "parameter_prompt_profile": "voxtral",
            "parameter_max_tokens": 8,
        }))?;
        asr.runtime_flags = None;
        assert_eq!(
            crate::run::eval_checkpoint(&asr)?,
            Some(run_request_digest(&asr)?)
        );
        Ok(())
    }

    /// An eval's checkpoint is retired once the server holds its result, and
    /// only then: a refused result, a lost lease or a submit abandoned in its
    /// backoff leaves it, so the job's next run resumes every sample instead of
    /// generating them again. Only a result the server holds is filed as synced.
    #[rstest::rstest]
    #[case::accepted(202, false, false, false)]
    #[case::refused(400, false, false, true)]
    #[case::lease_lost(202, true, false, true)]
    // The stop ends the backoff at its first sleep, as an exhausted one would.
    #[case::abandoned_in_the_backoff(503, false, true, true)]
    fn an_eval_checkpoint_is_kept_until_the_server_has_the_result(
        #[case] status: u16,
        #[case] lease_lost: bool,
        #[case] stop: bool,
        #[case] kept: bool,
    ) -> anyhow::Result<()> {
        let temp = TempWorkspace::new("worker-eval-checkpoint")?;
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(POST)
                .path("/benchmarks")
                .json_body_includes(r#"{"prefill_time_ms": 1.0}"#);
            then.status(status).body(r#"{"job_id":"job-abc"}"#);
        });
        // The retriable failure a refused result is reported as.
        server.mock(|when, then| {
            when.method(POST)
                .path("/benchmarks")
                .json_body_includes(r#"{"message_type": "failure"}"#);
            then.status(202).body(r#"{"job_id":"job-abc"}"#);
        });
        let req = eval_request()?;
        let evals = temp.ws.eval_completions();
        evals.open(&req)?.append(BenchmarkEvalCompletion {
            id: "s0".to_owned(),
            completion: "answer".to_owned(),
            ..Default::default()
        })?;

        submit_finished_run(
            &temp.ws,
            &server_url(&server)?,
            &test_http()?,
            &test_auth()?,
            &claimed_job(CLAIM_JOB_BODY)?,
            Ok((
                finished_payload()?,
                BenchmarkResultExtras::default(),
                Some(run_request_digest(&req)?),
            )),
            Duration::from_secs(3),
            lease_lost,
            &AtomicBool::new(stop),
        )?;

        assert_eq!(evals.open(&req)?.contains("s0"), kept);
        let synced = temp
            .ws
            .results()
            .list_ids(BenchmarkResultLocation::RemoteSynced)?;
        assert_eq!(synced.is_empty(), kept);
        Ok(())
    }

    #[test]
    fn a_shard_workspace_opens_a_part_of_its_own() -> anyhow::Result<()> {
        let temp = TempWorkspace::new("worker-eval-shard")?;
        let req = eval_request()?;
        let session = temp.as_shard("1/2")?.eval_completions().open(&req)?;
        let digest16: String = run_request_digest(&req)?
            .as_ref()
            .chars()
            .take(16)
            .collect();
        let name = session
            .path()
            .file_name()
            .and_then(|name| name.to_str())
            .context("a part name")?;
        assert!(
            name.starts_with(&format!("{digest16}.part-1-of-2-")),
            "{name}"
        );
        assert_eq!(
            session.path().extension(),
            Some(std::ffi::OsStr::new("jsonl"))
        );
        Ok(())
    }

    /// A refused failure report stops the worker after exactly one claim,
    /// rather than re-claiming the same leased job (see [`SubmitRefusedError`]).
    /// Driven through `execute`, so it is the real loop that has to stop.
    #[test]
    fn a_refused_failure_report_stops_the_worker_instead_of_re_claiming() -> anyhow::Result<()> {
        let temp = TempWorkspace::new("worker-submit-refused")?;
        let server = MockServer::start();
        register(&temp.ws, &server)?;
        // Always the same job, as a live lease makes the server answer.
        let claimed = server.mock(|when, then| {
            when.method(POST).path("/plans/claim");
            then.status(200).body(UNRUNNABLE_CLAIM_JOB_BODY);
        });
        let refused = server.mock(|when, then| {
            when.method(POST).path("/benchmarks");
            then.status(400).body("failure: unknown field `retriable`");
        });

        let error = worker_args()
            .execute(
                &temp.ws,
                &test_http()?,
                &ClientConfiguration::default(),
                &ExternalPrograms::at_missing_paths(),
            )
            .err()
            .ok_or_else(|| anyhow::anyhow!("expected the worker to stop"))?;

        assert_eq!(claimed.calls(), 1, "the loop re-claimed the refused job");
        assert_eq!(
            refused.calls(),
            1,
            "a refused failure report was sent a second time"
        );
        assert!(
            error.downcast_ref::<SubmitRefusedError>().is_some(),
            "the refusal did not reach the loop as itself: {error:#}"
        );
        Ok(())
    }

    #[test]
    fn a_reclaimed_job_after_reboot_is_failed_once_without_running() -> anyhow::Result<()> {
        let temp = TempWorkspace::new("worker-reboot")?;
        let server = MockServer::start();
        register(&temp.ws, &server)?;
        let store = temp.ws.in_flight_marker();
        store.write(&InFlight::new("job-abc", Some("boot-old".into())))?;
        let claimed = server.mock(|when, then| {
            when.method(POST).path("/plans/claim");
            then.status(200).body(CLAIM_JOB_BODY);
        });
        let failure = server.mock(|when, then| {
            when.method(POST)
                .path("/benchmarks")
                .body_includes("device rebooted during this job")
                .body_includes("boot-old")
                .body_includes("boot-new")
                .body_includes("\"retriable\":false");
            then.status(202).body(r#"{"job_id":"job-abc"}"#);
        });
        let benchmark = server.mock(|when, then| {
            when.method(httpmock::Method::GET)
                .path_includes("/benchmarks/");
            then.status(200).body("{}");
        });

        WorkerArgs {
            max_jobs: 1,
            ..worker_args()
        }
        .run_until_stopped_on_boot(
            &temp.ws,
            &test_http()?,
            &ClientConfiguration::default(),
            &ExternalPrograms::at_missing_paths(),
            &Arc::new(AtomicBool::new(false)),
            Boot {
                id: Some("boot-new".into()),
                started_at: Some(time::OffsetDateTime::now_utc()),
            },
        )?;

        assert_eq!(claimed.calls(), 1);
        assert_eq!(failure.calls(), 1);
        assert_eq!(benchmark.calls(), 0, "the job started again");
        assert!(store.read()?.is_none(), "reboot marker was not removed");
        Ok(())
    }

    /// Where an ordinary worker would fail the job for the reboot its marker
    /// shows, a shard leaves the marker as it was, heartbeats while its run
    /// lasts, reports nothing, not even a run that failed, and exits with that
    /// failure after its one claim.
    #[test]
    fn a_shard_claims_once_heartbeats_and_never_touches_the_marker_or_a_result()
    -> anyhow::Result<()> {
        let temp = TempWorkspace::new("worker-shard")?;
        let server = MockServer::start();
        register(&temp.ws, &server)?;
        let store = temp.ws.in_flight_marker();
        let marker = InFlight::new("job-abc", Some("boot-old".into()));
        store.write(&marker)?;
        let claimed = server.mock(|when, then| {
            when.method(POST).path("/plans/claim");
            then.status(200).body(CLAIM_JOB_BODY);
        });
        // Fails the run a second after its first heartbeat is due.
        server.mock(|when, then| {
            when.method(GET).path("/benchmarks/prefill_throughput_256");
            then.status(404).delay(Duration::from_secs(2));
        });
        let beat = server.mock(|when, then| {
            when.method(PUT).path("/plans/job-abc/heartbeat");
            then.status(200);
        });
        let posted = server.mock(|when, then| {
            when.method(POST).path("/benchmarks");
            then.status(202).body(r#"{"job_id":"job-abc"}"#);
        });

        let stop = Arc::new(AtomicBool::new(false));
        let outcome = thread::scope(|scope| -> anyhow::Result<anyhow::Result<()>> {
            // Ends a loop that claims again, which --max-jobs 0 would not.
            let signaller = scope.spawn(|| -> anyhow::Result<()> {
                let waited = wait_until("the first claim", || claimed.calls() >= 1);
                stop.store(true, Ordering::SeqCst);
                waited
            });
            let outcome = WorkerArgs {
                heartbeat_secs: Some(1),
                ..worker_args()
            }
            .run_until_stopped_on_boot(
                &temp.as_shard("0/2")?,
                &test_http()?,
                &ClientConfiguration::default(),
                &ExternalPrograms::at_missing_paths(),
                &stop,
                Boot {
                    id: Some("boot-new".into()),
                    started_at: Some(time::OffsetDateTime::now_utc()),
                },
            );
            signaller
                .join()
                .map_err(|_| anyhow::anyhow!("signaller panicked"))??;
            Ok(outcome)
        })?;
        let error = outcome.err().context("a shard whose run failed exited 0")?;

        assert_eq!(claimed.calls(), 1, "{error:#}");
        assert!(beat.calls() >= 1, "a shard's run sent no heartbeat");
        assert_eq!(posted.calls(), 0, "a shard reported to the server");
        assert_eq!(store.read()?, Some(marker));
        Ok(())
    }

    #[test]
    fn an_undelivered_reboot_failure_keeps_its_marker() -> anyhow::Result<()> {
        let temp = TempWorkspace::new("worker-reboot-undelivered")?;
        let server = MockServer::start();
        register(&temp.ws, &server)?;
        let store = temp.ws.in_flight_marker();
        let previous = InFlight::new("job-abc", Some("boot-old".into()));
        store.write(&previous)?;
        server.mock(|when, then| {
            when.method(POST).path("/plans/claim");
            then.status(200).body(CLAIM_JOB_BODY);
        });
        let failure = server.mock(|when, then| {
            when.method(POST).path("/benchmarks");
            then.status(503).body("down");
        });
        let stop = Arc::new(AtomicBool::new(false));
        thread::scope(|scope| -> anyhow::Result<()> {
            // The operator's ^C, landing in the submit backoff.
            let signaller = scope.spawn(|| -> anyhow::Result<()> {
                let waited = wait_until("the reboot failure", || failure.calls() >= 1);
                stop.store(true, Ordering::SeqCst);
                waited
            });
            WorkerArgs {
                max_jobs: 1,
                ..worker_args()
            }
            .run_until_stopped_on_boot(
                &temp.ws,
                &test_http()?,
                &ClientConfiguration::default(),
                &ExternalPrograms::at_missing_paths(),
                &stop,
                Boot {
                    id: Some("boot-new".into()),
                    started_at: Some(time::OffsetDateTime::now_utc()),
                },
            )?;
            signaller
                .join()
                .map_err(|_| anyhow::anyhow!("signaller panicked"))??;
            Ok(())
        })?;
        assert_eq!(
            store.read()?,
            Some(previous),
            "an undelivered reboot failure lost its marker"
        );
        Ok(())
    }

    #[test]
    fn a_clean_stop_during_a_job_removes_its_marker() -> anyhow::Result<()> {
        let temp = TempWorkspace::new("worker-stop-in-flight")?;
        let server = MockServer::start();
        register(&temp.ws, &server)?;
        let claimed = server.mock(|when, then| {
            when.method(POST).path("/plans/claim");
            then.status(200).body(UNRUNNABLE_CLAIM_JOB_BODY);
        });
        // The slow submit holds the loop inside the first job until the
        // signaller has asked it to stop.
        let submitted = server.mock(|when, then| {
            when.method(POST).path("/benchmarks");
            then.status(202)
                .body(r#"{"job_id":"job-abc"}"#)
                .delay(Duration::from_millis(200));
        });
        let marker = temp.ws.in_flight_marker();
        let stop = Arc::new(AtomicBool::new(false));
        thread::scope(|scope| -> anyhow::Result<()> {
            let signaller = scope.spawn(|| -> anyhow::Result<()> {
                let waited = wait_until("the first claim", || claimed.calls() >= 1);
                stop.store(true, Ordering::SeqCst);
                waited
            });
            worker_args().run_until_stopped(
                &temp.ws,
                &test_http()?,
                &ClientConfiguration::default(),
                &ExternalPrograms::at_missing_paths(),
                &stop,
            )?;
            signaller
                .join()
                .map_err(|_| anyhow::anyhow!("signaller panicked"))??;
            Ok(())
        })?;
        assert_eq!(submitted.calls(), 1);
        assert!(marker.read()?.is_none(), "clean stop left a marker");
        Ok(())
    }

    #[test]
    fn a_stale_marker_for_another_job_is_replaced_and_the_claim_runs() -> anyhow::Result<()> {
        let temp = TempWorkspace::new("worker-stale-marker")?;
        let server = MockServer::start();
        register(&temp.ws, &server)?;
        let store = temp.ws.in_flight_marker();
        store.write(&InFlight::new("old-job", Some("boot-old".into())))?;
        let claimed = server.mock(|when, then| {
            when.method(POST).path("/plans/claim");
            then.status(200).body(UNRUNNABLE_CLAIM_JOB_BODY);
        });
        let rebooted = server.mock(|when, then| {
            when.method(POST)
                .path("/benchmarks")
                .body_includes("device rebooted");
            then.status(202).body(r#"{"job_id":"job-abc"}"#);
        });
        let submitted = server.mock(|when, then| {
            when.method(POST).path("/benchmarks");
            then.status(202).body(r#"{"job_id":"job-abc"}"#);
        });

        WorkerArgs {
            max_jobs: 1,
            ..worker_args()
        }
        .run_until_stopped_on_boot(
            &temp.ws,
            &test_http()?,
            &ClientConfiguration::default(),
            &ExternalPrograms::at_missing_paths(),
            &Arc::new(AtomicBool::new(false)),
            Boot {
                id: Some("boot-new".into()),
                started_at: Some(time::OffsetDateTime::now_utc()),
            },
        )?;

        assert_eq!(claimed.calls(), 1);
        assert_eq!(
            rebooted.calls(),
            0,
            "a stale marker was reported as a reboot"
        );
        assert_eq!(submitted.calls(), 1, "the claimed job did not run");
        assert!(store.read()?.is_none(), "the job left its marker");
        Ok(())
    }

    /// The counterpart to `--max-jobs`: a stop the operator asks for, which the
    /// loop honours at the same seam — between claims, before taking one it
    /// would then owe the server a result for.
    #[test]
    fn a_worker_already_asked_to_stop_never_claims() -> anyhow::Result<()> {
        let temp = TempWorkspace::new("worker-stop-before-claim")?;
        let server = MockServer::start();
        register(&temp.ws, &server)?;
        let marker = temp.ws.in_flight_marker();
        marker.write(&InFlight::new("old-job", Some("old-boot".into())))?;
        let claimed = server.mock(|when, then| {
            when.method(POST).path("/plans/claim");
            then.status(200).body(UNRUNNABLE_CLAIM_JOB_BODY);
        });

        worker_args().run_until_stopped(
            &temp.ws,
            &test_http()?,
            &ClientConfiguration::default(),
            &ExternalPrograms::at_missing_paths(),
            &Arc::new(AtomicBool::new(true)),
        )?;

        assert_eq!(claimed.calls(), 0, "claimed after being asked to stop");
        assert!(
            marker.read()?.is_some(),
            "a stop before the first claim removed an inherited marker"
        );
        Ok(())
    }

    /// A quiet planner keeps a worker in this wait almost all the time. The wait
    /// here is an hour, so a loop that only checked between waits would still be
    /// running when this test gave up.
    #[test]
    fn a_stop_during_the_idle_wait_leaves_the_loop() -> anyhow::Result<()> {
        let temp = TempWorkspace::new("worker-stop-idle")?;
        let server = MockServer::start();
        register(&temp.ws, &server)?;
        let claimed = server.mock(|when, then| {
            when.method(POST).path("/plans/claim");
            then.status(204);
        });

        let args = WorkerArgs {
            idle_secs: 3600,
            ..worker_args()
        };
        let stop = Arc::new(AtomicBool::new(false));
        let waited = thread::scope(|scope| -> anyhow::Result<Duration> {
            // The signal an operator would send, raised once the worker has
            // asked for work and been told there is none — it is in the wait
            // by then, or about to be, the two orderings the loop must cover.
            let signaller = scope.spawn(|| -> anyhow::Result<()> {
                wait_until("the first claim", || claimed.calls() >= 1)?;
                stop.store(true, Ordering::SeqCst);
                Ok(())
            });
            let started = Instant::now();
            args.run_until_stopped(
                &temp.ws,
                &test_http()?,
                &ClientConfiguration::default(),
                &ExternalPrograms::at_missing_paths(),
                &stop,
            )?;
            let waited = started.elapsed();
            signaller
                .join()
                .map_err(|_| anyhow::anyhow!("the signaller panicked"))??;
            Ok(waited)
        })?;

        assert_eq!(claimed.calls(), 1, "the loop went round again");
        assert!(
            waited < Duration::from_secs(30),
            "the hour-long wait was not cut short: {waited:?}"
        );
        Ok(())
    }
}
