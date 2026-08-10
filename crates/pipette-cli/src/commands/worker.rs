//! `pipette worker` — planner claim loop.
//!
//! Implements the client obligations in pipette-mgmt's
//! `docs/client-integration.md`:
//!
//! 1. Refresh device profile + capabilities at startup; wait out reindex.
//! 2. Claim → run (with heartbeats) → submit → repeat.
//! 3. On 403 (not approved), stop. On 204, idle-wait with jitter.
//! 4. Heartbeat 404 → reclaim; 409 → abort. Submit failures follow the same
//!    reclaim/drop rules. Transient errors retry with backoff.
//!
//! ## Deliberate v1 limits (client-integration §3 / §6)
//!
//! - **No mid-run cancellation.** Runtime engines have no cancel token. When the
//!   heartbeat thread loses the lease (`409` / failed reclaim), it sets a
//!   shared flag and exits; the benchmark keeps running. On completion the
//!   worker **skips submit** rather than posting a result the server will drop.
//!   Wasted compute is accepted until engines grow cancellation.
//! - **No pending artifact for an unsubmitted result.** A successful or failed
//!   run is submitted immediately; a *completed* one is then copied to the
//!   results store as `RemoteSynced`, so a worker's runs are inspectable with
//!   `pipette results`. Exhausted submit backoff drops the outcome (warn log
//!   only) — same as a process crash mid-run. Ad-hoc `benchmarks run` + `sync`
//!   remains the path that keeps *pending* results on disk.
//! - **Claim 5xx retries forever** at a capped backoff (daemon posture).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use anyhow::Context;
use clap::Args;

use pipette_http::HttpClient;
use pipette_mgmt_client::types::ClaimedJob;
use pipette_mgmt_client::{AuthIdentity, MgmtClient};
use pipette_plan_types::is_compatible;

use crate::client::claim::{redacted_spec, run_spec_from_claim, UnrunnableClaim};
use crate::client::worker::{
    attach_claim_to_success_payload, classify_run_error, failure_from_claim, format_failure_reason,
    idle_wait_with_jitter, installed_runtime_capabilities, keepalive_lease, poll_claim,
    refresh_profile_at_startup, resolve_heartbeat_interval, submit_plan_result_with_backoff,
    ClaimPoll, LeaseKeepalive, SubmitDisposition, DEFAULT_IDLE_JITTER, DEFAULT_IDLE_WAIT,
};
use crate::run::run_cell;
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

While a job runs the worker heartbeats to hold its lease. If the lease is lost,
the benchmark still finishes but the result is not submitted, because runtimes
have no cancel seam in v1. Use `benchmarks run` + `sync` when a result must
survive a failed submit.";

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

    /// Exit after successfully finishing this many jobs (0 = run forever).
    #[arg(long, default_value_t = 0)]
    pub max_jobs: u64,

    /// Skip the startup profile PATCH (use only for debugging against a
    /// server that already has an accurate profile).
    #[arg(long, default_value_t = false)]
    pub skip_profile_refresh: bool,
}

impl WorkerArgs {
    pub fn execute(self, ws: &PipetteWorkspace, http: &HttpClient) -> anyhow::Result<()> {
        let session = ws.mgmt_session()?;
        let (identity, client, auth) = (&session.identity, &session.client, &session.auth);

        log::info!(
            "worker starting (client_id={}, server={}, idle={}s+0..{}s jitter, \
             heartbeat={}, max_jobs={})",
            auth.client_id,
            session.server_url,
            self.idle_secs,
            self.idle_jitter_secs,
            self.heartbeat_secs
                .map(|s| format!("{s}s"))
                .unwrap_or_else(|| "half time_window".into()),
            if self.max_jobs == 0 {
                "unlimited".to_string()
            } else {
                self.max_jobs.to_string()
            }
        );

        if self.skip_profile_refresh {
            log::warn!(
                "skipping startup profile refresh (--skip-profile-refresh); \
                 matching may be stale until the next PATCH"
            );
        } else {
            let caps = installed_runtime_capabilities(&ws.runtimes())?;
            log::info!(
                "refreshing profile ({} capability flag(s): [{}])",
                caps.len(),
                caps.join(", ")
            );
            let profile = refresh_profile_at_startup(identity, client, http, caps)?;
            if profile.status != "approved" {
                anyhow::bail!(
                    "client not approved (status={}); an operator must run \
                     `pipette-mgmt clients approve {}`, then restart this client",
                    profile.status,
                    profile.client_id
                );
            }
            log::info!(
                "profile ready (status={}, reindex_pending={}, tags={:?}, caps={:?})",
                profile.status,
                profile.reindex_pending,
                profile.tags,
                profile.capabilities
            );
        }

        let idle_base = Duration::from_secs(self.idle_secs);
        let idle_jitter = Duration::from_secs(self.idle_jitter_secs);
        let mut completed = 0u64;
        let mut idle_rounds = 0u64;

        log::info!("entering claim loop");
        loop {
            log::debug!("polling claim (completed={completed}, idle_rounds={idle_rounds})");
            match claim_with_retry(client, http, auth) {
                ClaimPoll::Job(job) => {
                    idle_rounds = 0;
                    log_claimed_job(&job);
                    match run_claimed_job(ws, client, http, auth, &job, self.heartbeat_secs) {
                        Ok(()) => {
                            completed += 1;
                            log::info!(
                                "job {} done (session completed={completed}{})",
                                job.job_id,
                                if self.max_jobs > 0 {
                                    format!(", max_jobs={}", self.max_jobs)
                                } else {
                                    String::new()
                                }
                            );
                            if self.max_jobs > 0 && completed >= self.max_jobs {
                                log::info!(
                                    "reached --max-jobs={}; leaving claim loop",
                                    self.max_jobs
                                );
                                return Ok(());
                            }
                        }
                        Err(e) => {
                            // run_claimed_job already reported failure/abort when
                            // it could; log and continue the loop.
                            log::error!(
                                "job {} ended with error (continuing claim loop): {e:#}",
                                job.job_id
                            );
                        }
                    }
                }
                ClaimPoll::Idle => {
                    idle_rounds += 1;
                    let wait = idle_wait_with_jitter(idle_base, idle_jitter);
                    log::info!(
                        "no work available (204); sleeping {}s \
                         (base={}s, jitter=0..{}s, idle_round={idle_rounds})",
                        wait.as_secs(),
                        idle_base.as_secs(),
                        idle_jitter.as_secs()
                    );
                    sleep_interruptible(wait);
                }
                ClaimPoll::NotApproved => {
                    log::error!(
                        "claim returned 403 — client {} is not approved; stopping",
                        auth.client_id
                    );
                    anyhow::bail!(
                        "client not approved; an operator must run \
                         `pipette-mgmt clients approve {}`, then restart this client",
                        auth.client_id
                    );
                }
                ClaimPoll::Transient(e) => {
                    // claim_with_retry absorbs transients; this arm is defensive.
                    log::error!("unexpected unhandled claim transient: {e}");
                    return Err(e.into());
                }
                ClaimPoll::Fatal(err) => {
                    log::error!("fatal claim error: {err}");
                    return Err(err.into());
                }
            }
        }
    }
}

fn log_claimed_job(job: &ClaimedJob) {
    log::info!(
        "claimed {} (benchmark={}, time_window={}, expires_at={})",
        job.job_id,
        job.benchmark_id,
        job.time_window,
        job.expires_at.as_deref().unwrap_or("-"),
    );
    // Untyped here — `execute_job` logs the model and runtime once the payload
    // has parsed. This is the copy an operator reads when it does not, so it
    // goes out with any plan-carried access token stripped.
    log::debug!("{} spec={}", job.job_id, redacted_spec(&job.spec));
}

fn claim_with_retry(client: &MgmtClient, http: &HttpClient, auth: &AuthIdentity) -> ClaimPoll {
    let mut backoff = Duration::from_secs(1);
    const BACKOFF_CAP: Duration = Duration::from_secs(60);
    let mut attempt = 0u32;
    loop {
        attempt += 1;
        log::debug!("claim attempt {attempt}");
        match poll_claim(client, http, auth) {
            ClaimPoll::Transient(e) => {
                log::warn!(
                    "claim attempt {attempt} transient failure ({e}); retrying in {}s",
                    backoff.as_secs()
                );
                sleep_interruptible(backoff);
                backoff = (backoff * 2).min(BACKOFF_CAP);
            }
            other => return other,
        }
    }
}

/// Execute one claimed job end-to-end: heartbeat thread + run + submit.
fn run_claimed_job(
    ws: &PipetteWorkspace,
    client: &MgmtClient,
    http: &HttpClient,
    auth: &AuthIdentity,
    job: &ClaimedJob,
    heartbeat_secs: Option<u64>,
) -> anyhow::Result<()> {
    let protocol_hb = resolve_heartbeat_interval(&job.time_window, None);
    let hb_every = resolve_heartbeat_interval(&job.time_window, heartbeat_secs);
    if heartbeat_secs.is_some() && hb_every > protocol_hb {
        log::warn!(
            "{}: --heartbeat-secs={}s is longer than half time_window \
             ({} = {}s); lease may expire between heartbeats",
            job.job_id,
            hb_every.as_secs(),
            job.time_window,
            protocol_hb.as_secs()
        );
    }
    log::info!(
        "{}: starting run (heartbeat every {}s{}; time_window={})",
        job.job_id,
        hb_every.as_secs(),
        if heartbeat_secs.is_some() {
            ", from --heartbeat-secs"
        } else {
            ", half time_window"
        },
        job.time_window
    );
    // Shared with the heartbeat thread: set on lease abort so we skip submit
    // after the (uncancellable) run finishes.
    let lease_lost = Arc::new(AtomicBool::new(false));
    // Heartbeat thread sleeps the full interval via recv_timeout (no polling).
    // Dropping `hb` disconnects the channel and joins the thread.
    let hb = spawn_heartbeat(
        client.clone(),
        http.clone(),
        auth.clone(),
        job.job_id.clone(),
        hb_every,
        Arc::clone(&lease_lost),
    );

    let started = std::time::Instant::now();
    let run_result = execute_job(ws, client, http, auth, job);
    let elapsed = started.elapsed();

    drop(hb);
    log::debug!("{}: heartbeat thread stopped", job.job_id);

    if lease_lost.load(Ordering::SeqCst) {
        log::warn!(
            "{}: lease was lost during the run ({:.1}s elapsed); skipping submit \
             (result would be rejected — see module docs)",
            job.job_id,
            elapsed.as_secs_f64()
        );
        return Ok(());
    }

    match run_result {
        Ok((payload, extras)) => {
            log::info!(
                "{}: run succeeded in {:.1}s; submitting success",
                job.job_id,
                elapsed.as_secs_f64()
            );
            submit_success(ws, client, http, auth, job, &payload, &extras)
        }
        Err(err) => {
            let retriable = classify_run_error(&err);
            let reason = format_failure_reason(&format!("{err:#}"));
            log::warn!(
                "{}: run failed after {:.1}s (retriable={retriable}); submitting failure: {reason}",
                job.job_id,
                elapsed.as_secs_f64()
            );
            submit_failure(client, http, auth, job, reason, retriable)
        }
    }
}

/// Owns the heartbeat background thread. Dropping it signals stop (channel
/// disconnect) and joins — the thread spends its life in `recv_timeout`, so it
/// costs no CPU between ticks and wakes immediately when the run finishes.
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
    client: MgmtClient,
    http: HttpClient,
    auth: AuthIdentity,
    job_id: String,
    every: Duration,
    lease_lost: Arc<AtomicBool>,
) -> HeartbeatGuard {
    // Unit channel: we never send a value; disconnect (`drop` of the sender)
    // is the stop signal. `recv_timeout` sleeps the full interval in the
    // kernel — no 200 ms poll loop.
    let (stop_tx, stop_rx) = mpsc::channel::<()>();
    let join = thread::spawn(move || {
        // First heartbeat after `every`, not immediately — the claim itself
        // just granted a fresh window.
        log::info!(
            "{job_id}: heartbeat thread started (first tick in {}s, sleeping via recv_timeout)",
            every.as_secs()
        );
        let mut wait = every;
        let mut ticks = 0u32;
        loop {
            match stop_rx.recv_timeout(wait) {
                // Stop requested, or the sender was dropped.
                Ok(()) | Err(RecvTimeoutError::Disconnected) => {
                    log::debug!(
                        "{job_id}: heartbeat thread stopping after {ticks} successful tick(s)"
                    );
                    return;
                }
                Err(RecvTimeoutError::Timeout) => {
                    log::debug!("{job_id}: sending heartbeat (tick {})", ticks + 1);
                    match keepalive_lease(&client, &http, &auth, &job_id) {
                        LeaseKeepalive::Ok => {
                            ticks += 1;
                            log::info!(
                                "{job_id}: heartbeat ok (tick={ticks}, next in {}s)",
                                every.as_secs()
                            );
                            wait = every;
                        }
                        LeaseKeepalive::Abort => {
                            lease_lost.store(true, Ordering::SeqCst);
                            log::error!(
                                "{job_id}: heartbeat aborted lease (tick would be {}); \
                                 flagging lease_lost — run continues (no cancel path) \
                                 but submit will be skipped",
                                ticks + 1
                            );
                            return;
                        }
                        LeaseKeepalive::Retry => {
                            // Brief retry — still a single kernel sleep, not a poll.
                            // Slack is up to time_window/2 before the lease lapses.
                            const RETRY_WAIT: Duration = Duration::from_secs(2);
                            log::warn!(
                                "{job_id}: heartbeat transient (tick would be {}); \
                                 retrying in {}s",
                                ticks + 1,
                                RETRY_WAIT.as_secs()
                            );
                            wait = RETRY_WAIT;
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

fn execute_job(
    ws: &PipetteWorkspace,
    client: &MgmtClient,
    http: &HttpClient,
    auth: &AuthIdentity,
    job: &ClaimedJob,
) -> anyhow::Result<(
    pipette_plan_types::result::BenchmarkSubmissionPayload,
    crate::results::BenchmarkResultExtras,
)> {
    let spec = run_spec_from_claim(job)?;
    log::info!(
        "{}: cell model=`{}`, runtime=`{}`",
        job.job_id,
        spec.model,
        spec.runtime
    );
    if !is_compatible(&spec.model, &spec.runtime) {
        return Err(UnrunnableClaim::Incompatible {
            model: spec.model.to_string(),
            runtime: spec.runtime.to_string(),
        }
        .into());
    }

    // Populate the local remote benchmark cache (if needed); prepare/run
    // resolves the body and ensures artifacts from storage.
    let benchmark = ensure_claim_benchmark_cached(ws, client, http, auth, &job.benchmark_id)?;
    log::info!(
        "{}: benchmark ready (id={}, type={:?})",
        job.job_id,
        job.benchmark_id,
        benchmark.benchmark_type()
    );

    log::info!("{}: dispatching via shared ClientRunSpec path", job.job_id);
    // `prepare` adds the cell's pins; the policy only carries the cap here.
    let artifacts = ws.artifacts(http);
    run_cell(&spec, benchmark, &artifacts, ws)
}

fn ensure_claim_benchmark_cached(
    ws: &PipetteWorkspace,
    client: &MgmtClient,
    http: &HttpClient,
    auth: &AuthIdentity,
    benchmark_id: &str,
) -> anyhow::Result<pipette_plan_types::benchmark::BenchmarkDefinition> {
    // Prefer the remote cache (populated by `sync` or a prior claim).
    let remote_ref = crate::benchmarks::SourcedBenchmarkId::new(
        pipette_plan_types::benchmark::BenchmarkSource::Remote,
        pipette_plan_types::BenchmarkId::try_new(benchmark_id.to_owned())
            .with_context(|| format!("invalid benchmark id `{benchmark_id}`"))?,
    );
    if let Ok(Some(def)) = ws.benchmarks().get(&remote_ref) {
        log::info!("benchmark {benchmark_id}: loaded from local remote cache");
        return Ok(def);
    }
    log::info!("benchmark {benchmark_id}: not cached; GET /benchmarks/{benchmark_id}");
    let remote = client
        .get_benchmark(http, auth, benchmark_id)
        .with_context(|| format!("GET /benchmarks/{benchmark_id}"))?;
    let def = crate::benchmarks::benchmark_definition_from_remote(remote)
        .with_context(|| format!("converting remote benchmark `{benchmark_id}`"))?;
    match ws
        .benchmarks()
        .put(pipette_plan_types::benchmark::BenchmarkSource::Remote, &def)
    {
        Ok(()) => log::debug!("benchmark {benchmark_id}: cached in catalog store"),
        Err(e) => log::warn!("failed to cache remote benchmark {benchmark_id}: {e}"),
    }
    Ok(def)
}

/// Post a finished run against its claim, then keep a local copy.
#[allow(clippy::too_many_arguments)]
fn submit_success(
    ws: &PipetteWorkspace,
    client: &MgmtClient,
    http: &HttpClient,
    auth: &AuthIdentity,
    job: &ClaimedJob,
    payload: &pipette_plan_types::result::BenchmarkSubmissionPayload,
    extras: &crate::results::BenchmarkResultExtras,
) -> anyhow::Result<()> {
    log::info!("{}: submitting the completed run", job.job_id);
    let mut body = serde_json::to_value(payload).context("serialize success payload")?;
    body = attach_claim_to_success_payload(body, job)?;
    log::debug!(
        "{}: success payload keys={:?}",
        job.job_id,
        body.as_object()
            .map(|o| o.keys().cloned().collect::<Vec<_>>())
    );
    post_completion(client, http, auth, &job.job_id, &body)?;
    store_completed_result(&ws.results(), &job.job_id, payload, extras);
    Ok(())
}

/// Keep a completed job's result on disk, so a worker's runs are inspectable
/// with `pipette results` like a `benchmarks run` result — under the id and
/// location `sync` promotes a CLI result to, so both paths leave one shape.
///
/// After the submit: storing first would file work as submitted before it was,
/// and storing it pending on a failed submit would let `sync` post it again
/// through the generic endpoint. Best-effort, because the job is already
/// complete server-side and a write failure must not fail it.
fn store_completed_result(
    results: &crate::results::ResultsStore,
    job_id: &str,
    payload: &pipette_plan_types::result::BenchmarkSubmissionPayload,
    extras: &crate::results::BenchmarkResultExtras,
) {
    let location = crate::results::BenchmarkResultLocation::RemoteSynced;
    match results.save_result(location, job_id, payload, extras) {
        Ok(()) => log::info!("{job_id}: result stored under {location:?}"),
        Err(e) => log::warn!("{job_id}: submitted but not stored locally: {e:#}"),
    }
}

fn submit_failure(
    client: &MgmtClient,
    http: &HttpClient,
    auth: &AuthIdentity,
    job: &ClaimedJob,
    reason: String,
    retriable: bool,
) -> anyhow::Result<()> {
    log::info!(
        "{}: building failure submission (retriable={retriable})",
        job.job_id
    );
    let body = failure_from_claim(job, reason, retriable);
    post_completion(client, http, auth, &job.job_id, &body)
}

/// The retrying POST both outcomes share: a success body or a failure body.
fn post_completion<P: serde::Serialize>(
    client: &MgmtClient,
    http: &HttpClient,
    auth: &AuthIdentity,
    job_id: &str,
    payload: &P,
) -> anyhow::Result<()> {
    log::info!("{job_id}: POST /benchmarks (plan-attached)");
    match submit_plan_result_with_backoff(client, http, auth, job_id, payload, |d| {
        log::warn!(
            "{job_id}: submit transient; backing off {}s before retry",
            d.as_secs()
        );
        sleep_interruptible(d);
    })? {
        SubmitDisposition::Accepted {
            job_id: accepted_id,
        } => {
            log::info!("{job_id}: submit accepted (server job_id={accepted_id})");
            Ok(())
        }
        SubmitDisposition::Dropped => {
            log::warn!("{job_id}: submit dropped (409 superseded, or reclaim after 404 failed)");
            Ok(())
        }
        SubmitDisposition::RetryTransient => {
            // v1: a result never submitted leaves no pending artifact (module docs).
            log::warn!(
                "{job_id}: gave up submitting after repeated transient failures; \
                 result dropped (no local pending artifact)"
            );
            Ok(())
        }
        SubmitDisposition::Fatal(e) => {
            log::error!("{job_id}: submit fatal error: {e}");
            Err(e.into())
        }
    }
}

/// Sleep in 1 s chunks so a signal handler can interrupt a multi-minute wait promptly.
fn sleep_interruptible(total: Duration) {
    let deadline = std::time::Instant::now() + total;
    while std::time::Instant::now() < deadline {
        let remaining = deadline.saturating_duration_since(std::time::Instant::now());
        thread::sleep(remaining.min(Duration::from_secs(1)));
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use pipette_plan_types::result::BenchmarkSubmissionPayload;

    use super::store_completed_result;
    use crate::results::{BenchmarkResultExtras, BenchmarkResultLocation, ResultsStore};

    /// A completed job's result lands under the synced location keyed by the job
    /// id — where `sync` promotes a CLI result to, so `results list` sees both
    /// paths as one shape.
    #[test]
    fn store_completed_result_files_under_the_job_id() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let results = ResultsStore::new(tmp.path().join("results"));
        let payload: BenchmarkSubmissionPayload = serde_json::from_value(json!({
            "benchmark_id": "prefill_throughput_512",
            "device": {},
            "thermal": {},
            "submitted_at": "2026-07-30T00:00:00Z",
            // `result` is `#[serde(flatten)]`, so the metric sits at the top level.
            "prefill_time_ms": 1.0,
        }))?;

        store_completed_result(
            &results,
            "job-1",
            &payload,
            &BenchmarkResultExtras::default(),
        );

        let location = BenchmarkResultLocation::RemoteSynced;
        assert_eq!(
            results.load_payload(location, "job-1")?["benchmark_id"],
            "prefill_throughput_512"
        );
        assert!(results.extras_path(location, "job-1").exists());
        Ok(())
    }
}
