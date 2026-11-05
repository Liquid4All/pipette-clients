//! `run`: one thread per phone, each restarted on transient failures, each
//! claiming until stopped (DESIGN section 1).

mod fleet;
mod heartbeat;
mod install;
mod job;
mod pause;
mod summary;
mod version;

#[cfg(test)]
pub(crate) use self::fleet::testing;
pub(crate) use self::fleet::{
    DeviceWait, Exclusivity, Fleet, Phone, StoppedError, client_details, device_profile, prepare,
    refresh_pool,
};
pub(crate) use self::version::listed_catalog;

use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::thread;

use anyhow::Context;
use serde_json::Value;

use pipette_mgmt_client::types::{ClientProfile, DeviceProfileFields};
use pipette_plan_types::RuntimeBuildFlagError;

use self::job::{log_claimed_job, run_claimed_job};
use self::pause::{pause_claims, resume_claims, wait_until_runnable};
use self::version::Releases;
use crate::config::BuildSource;
use crate::device::{Reachability, gpu_pin};
use crate::pool::Pool;
use crate::server::{self, ClaimPoll, Session};
use crate::timing;

/// Where one phone's builds come from, by the config's mode.
pub(crate) enum Builds {
    /// `[builds]`: shared, and read once before the phones start.
    Pool(Arc<Pool>),
    /// `[signing]`: this phone's catalog of the bucket.
    Releases(Box<Releases>),
}

impl Builds {
    fn capabilities(&self) -> Result<Vec<String>, RuntimeBuildFlagError> {
        match self {
            Self::Pool(pool) => pool.capabilities(),
            Self::Releases(releases) => releases.catalog.advertised(),
        }
    }
}

/// `run`: claim for every configured device until `max_jobs` or a signal.
///
/// One thread per device, independent all the way down, so a twenty-minute
/// benchmark on one phone does not idle the rest. A thread that fails takes its
/// own device out of service and leaves the others running; the process exits
/// non-zero if ANY device failed ([`run_outcome`]).
pub(crate) fn run(fleet: Fleet) -> anyhow::Result<()> {
    let Fleet {
        mut phones,
        config,
        state,
        stop,
        locks,
    } = fleet;
    // Bound to a name: dropping them early releases the locks mid-run.
    let _held = locks;
    let call_timeout = config.devicectl.call_timeout;
    let pin_cli = config.devicectl.pymobiledevice3.as_deref();
    for phone in &phones {
        let udid = phone.details.hardware_udid.as_deref();
        if let Some(udid) = udid {
            gpu_pin::release_leftover(udid, call_timeout);
        }
        if pin_cli.is_some() {
            udid.with_context(|| {
                format!(
                    "{}: `device info details` named no hardware UDID, which the GPU pin \
                     ([devicectl] pymobiledevice3) needs",
                    phone.control.udid()
                )
            })?;
        }
    }
    if let Some(program) = pin_cli {
        gpu_pin::check_cli(program, call_timeout)?;
    }

    let count = phones.len();
    let Some(first) = phones.first_mut() else {
        anyhow::bail!("no devices to run for");
    };
    let mut builds = match &config.builds {
        BuildSource::Pool(source) => {
            // Once, before anything is shared; the pool never changes after this.
            let mut pool = Pool::open(source, &state.build_index)?;
            refresh_pool(first, &mut pool)?;
            pool.check_advertisable()?;
            let pool = Arc::new(pool);
            (0..count)
                .map(|_| Builds::Pool(Arc::clone(&pool)))
                .collect::<Vec<_>>()
        }
        BuildSource::Releases(_) => {
            version::fleet_builds(config.signer()?, &config, &state, count)?
        }
    };

    let stopped: Vec<DeviceStop> = thread::scope(|scope| {
        let handles: Vec<_> = phones
            .iter_mut()
            .zip(builds.iter_mut())
            .map(|(phone, builds)| {
                let label = phone.control.udid().to_owned();
                (
                    label,
                    scope.spawn(move || -> anyhow::Result<()> { run_device(phone, builds) }),
                )
            })
            .collect();
        handles
            .into_iter()
            .filter_map(|(label, handle)| match handle.join() {
                Ok(Ok(())) => None,
                Ok(Err(error)) => {
                    tracing::error!(
                        device = %label,
                        error = %format!("{error:#}"),
                        "device stopped"
                    );
                    Some(if halts_phone(&error) {
                        DeviceStop::Operator
                    } else {
                        DeviceStop::Other
                    })
                }
                Err(_) => {
                    tracing::error!(device = %label, "device thread panicked");
                    Some(DeviceStop::Other)
                }
            })
            .collect()
    });

    run_outcome(&stopped, stop.load(Ordering::SeqCst))
}

/// `run` ended with every device that failed stopped for a fault only an
/// operator can fix, one a restart would hit again, possibly after failing
/// another job. Typed so `main` can give it its own exit status, which a
/// supervisor reads as "do not restart" (DESIGN section 7).
#[derive(Debug, thiserror::Error)]
#[error("{stopped} of the configured devices stopped for a fault only an operator can fix")]
pub(crate) struct OperatorStopError {
    pub(crate) stopped: usize,
}

/// Why a device's thread ended with an error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DeviceStop {
    /// [`halts_phone`]: a restart would stop the same way, possibly after
    /// failing another job.
    Operator,
    /// Anything else, a panic included. A `4xx` too: it never holds a lease,
    /// so a restart costs no job and recovers once the server, the approval or
    /// the clock is fixed.
    Other,
}

/// How `run` ends, from the devices that stopped with an error. A stop signal
/// makes it a clean exit whatever they were.
fn run_outcome(stopped: &[DeviceStop], signalled: bool) -> anyhow::Result<()> {
    if stopped.is_empty() || signalled {
        return Ok(());
    }
    if stopped.iter().all(|stop| *stop == DeviceStop::Operator) {
        return Err(OperatorStopError {
            stopped: stopped.len(),
        }
        .into());
    }
    anyhow::bail!(
        "{} of the configured devices stopped with an error",
        stopped.len()
    );
}

/// One device for the life of the process, restarted with the claim loop's
/// backoff whenever it stops for a reason the next attempt might not hit (a blip
/// during startup's `PATCH /clients/me` must not need a human). It stops on the
/// stop signal and on [`is_operator_actionable`] errors, which a restart would
/// hit again.
fn run_device(phone: &mut Phone, builds: &mut Builds) -> anyhow::Result<()> {
    let mut backoff = phone.config.loop_config.claim_backoff;
    let mut tally = Tally::default();
    loop {
        match run_one(phone, builds, &mut tally) {
            Ok(()) => return Ok(()),
            Err(error) if is_operator_actionable(&error) => return Err(error),
            Err(error) => {
                if phone.stop.load(Ordering::SeqCst) {
                    return Ok(());
                }
                tracing::warn!(
                    error = %format!("{error:#}"),
                    retry_secs = %backoff.as_secs(),
                    "device stopped transiently; restarting it"
                );
                if !timing::sleep_interruptible(backoff, &phone.stop) {
                    return Ok(());
                }
                backoff = timing::next_backoff(backoff, timing::CLAIM_BACKOFF_CAP);
            }
        }
    }
}

/// A stop that only an operator can undo. Typed rather than a message, because
/// `run_device` decides by TYPE and the server's status has already been consumed
/// into a `ClaimPoll` variant by then.
#[derive(Debug, thiserror::Error)]
enum OperatorActionError {
    #[error(
        "claim refused with 403: client {client_id} is not approved. This worker adopts the \
         phone's own registered client rather than creating one, so approval is a property of \
         that client: have an operator approve it, then restart"
    )]
    NotApproved { client_id: String },
    #[error(
        "claim refused with 401: the request signature or its timestamp was rejected for client \
         {client_id}. Check this Mac's clock is synchronised (the server accepts a few minutes of \
         skew), and that this device's private_key is the one that client registered with"
    )]
    SignatureRejected { client_id: String },
}

/// Whether a failed job means this phone must not claim again: its identity
/// was replaced, or its build needs an operator.
fn halts_phone(error: &anyhow::Error) -> bool {
    install::replaced_identity(error) || version::needs_operator(error)
}

/// Whether an error is one only an operator can fix: the claim loop's typed
/// stops, one [`halts_phone`] names, and any `4xx` from the calls before it.
/// Everything else can clear on its own and is retried.
fn is_operator_actionable(error: &anyhow::Error) -> bool {
    halts_phone(error)
        || error.chain().any(|cause| {
            cause.downcast_ref::<OperatorActionError>().is_some()
                || cause
                    .downcast_ref::<pipette_mgmt_client::Error>()
                    .and_then(pipette_mgmt_client::Error::http_status)
                    .is_some_and(|status| (400..500).contains(&status))
        })
}

/// The jobs one device has taken, kept across its restarts so a restart does not
/// refill the `max_jobs` budget.
#[derive(Debug, Default)]
struct Tally {
    /// The `max_jobs` budget. Every job CLAIMED, so a phone that fails every job
    /// still retires it and `max_jobs = 1` returns.
    attempted: u32,
    /// Only the jobs the server was told about.
    completed: u32,
}

fn run_one(phone: &mut Phone, builds: &mut Builds, tally: &mut Tally) -> anyhow::Result<()> {
    // Every line this thread (and the threads it hands the span to) logs names its
    // phone and client: the operator's spelling, and what the server keys on.
    let _span = tracing::info_span!(
        "phone",
        device = %phone.control.udid(),
        client_id = %phone.entry.credentials.client_id.as_ref(),
    )
    .entered();
    match builds {
        // Before the first server call: an unread pool would advertise nothing.
        Builds::Pool(pool) => pool.check_advertisable()?,
        Builds::Releases(releases) => {
            // `start` runs the app: a phone that cannot would fail it and
            // restart every minute instead of pausing.
            if !wait_until_runnable(phone) {
                return Ok(());
            }
            releases.start(phone)?;
            if phone.stop.load(Ordering::SeqCst) {
                return Ok(());
            }
        }
    }
    let (ctx, profile) = connect(phone, &builds.capabilities()?)?;

    // The sleep's return value is the stop signal: dropping it turns the
    // remaining polls into a spin and then a bogus "not approved after 300s".
    let waited =
        server::wait_until_claimable(&ctx, |wait| timing::sleep_interruptible(wait, &phone.stop))
            .with_context(|| {
            format!(
                "waiting for client {} to become claimable",
                profile.client_id
            )
        })?;
    let waited = match waited {
        server::ClaimableWait::Claimable(profile) => profile,
        server::ClaimableWait::Stopped => {
            tracing::info!(completed = %tally.completed, "stopping on a signal");
            return Ok(());
        }
    };
    tracing::info!(
        client_id = %waited.client_id,
        status = %waited.status,
        capabilities = %waited.capabilities.len(),
        "client is claimable"
    );

    claim_loop(phone, builds, &Arc::new(ctx), tally)
}

/// `PATCH /clients/me` with the profile and `capabilities`, returning the
/// signing context and the profile the server answered.
pub(crate) fn connect(
    phone: &Phone,
    capabilities: &[String],
) -> anyhow::Result<(Session, ClientProfile)> {
    let http = server::http_client()?;
    let (details, device) = report(phone, capabilities);
    // Adopted, not minted (`ClientCredentials::signer`).
    let auth = phone.entry.credentials.signer();
    let client_id = auth.client_id.to_string();
    tracing::info!(
        client_id = %client_id,
        device = %phone.control.udid(),
        state_dir = %phone.paths.dir.display(),
        "acting as the phone's own registered client"
    );

    let ctx = Session {
        server: phone.config.server.url.clone(),
        http,
        auth,
    };
    let profile = server::refresh_profile(&ctx, Some(details), device)
        .with_context(|| format!("PATCH /clients/me as client {client_id}"))?;
    tracing::info!(
        client_id = %profile.client_id,
        status = %profile.status,
        reindex_pending = %profile.reindex_pending,
        capabilities = %profile.capabilities.join(","),
        "profile refreshed"
    );
    Ok((ctx, profile))
}

/// What this proxy reports for `phone`, capabilities included.
fn report(phone: &Phone, capabilities: &[String]) -> (String, DeviceProfileFields) {
    (client_details(phone), device_profile(phone, capabilities))
}

/// What re-reporting the profile between jobs found.
#[derive(Debug, PartialEq, Eq)]
enum Reasserted {
    /// The server held this report already: a no-op.
    Unchanged,
    /// This proxy's report changed: what a run taught it, or a new iOS version.
    Updated,
    /// The report had not changed, but the server's copy had.
    Replaced,
}

/// Re-report the profile between jobs, holding no lease, and wait out any reindex it
/// causes. Anything else signing as this phone replaces the profile (an older app build
/// at every launch, or the app opened by hand), relinquishing this proxy's lease.
/// `last` is the previous report.
fn reassert_profile(
    phone: &Phone,
    capabilities: &[String],
    ctx: &Session,
    last: &mut Value,
) -> anyhow::Result<Reasserted> {
    let (details, device) = report(phone, capabilities);
    let sent = serde_json::to_value((&details, &device))?;
    let profile = server::refresh_profile(ctx, Some(details), device)
        .context("re-reporting the profile between jobs")?;
    let reasserted = match (profile.reindex_pending, sent == *last) {
        (false, _) => Reasserted::Unchanged,
        (true, true) => Reasserted::Replaced,
        (true, false) => Reasserted::Updated,
    };
    *last = sent;
    match reasserted {
        Reasserted::Unchanged => return Ok(reasserted),
        Reasserted::Updated => {
            tracing::info!("reported what the phone says about itself; waiting for the reindex");
        }
        Reasserted::Replaced => tracing::warn!(
            "the server held another profile for this client than the one this proxy reported: \
             something else signing as the phone replaced it and relinquished any lease this \
             proxy held. A Pipette build that reports its profile on every headless launch does \
             this, as does opening the app by hand. Reported again; waiting for the reindex"
        ),
    }
    // A signal ends the wait; the claim loop stops on it next.
    server::wait_until_claimable(ctx, |wait| timing::sleep_interruptible(wait, &phone.stop))?;
    Ok(reasserted)
}

fn claim_loop(
    phone: &mut Phone,
    builds: &mut Builds,
    ctx: &Arc<Session>,
    tally: &mut Tally,
) -> anyhow::Result<()> {
    // Owned copies: `run_claimed_job` takes `phone` mutably, so a borrow could
    // not be held across the call.
    let pacing = phone.config.loop_config.clone();
    let stop = Arc::clone(&phone.stop);
    let mut backoff = pacing.claim_backoff;
    let mut paused = None;
    // What `connect` just sent, so a reindex after an unchanged report reads as a
    // replaced profile.
    let mut reported = serde_json::to_value(report(phone, &builds.capabilities()?))?;

    tracing::info!(
        device = %phone.control.udid(),
        idle_wait_secs = %pacing.idle_wait.as_secs(),
        idle_jitter_secs = %pacing.idle_jitter.as_secs(),
        max_jobs = %if pacing.max_jobs == 0 { "unlimited".to_string() } else { pacing.max_jobs.to_string() },
        "entering the claim loop"
    );

    while !stop.load(Ordering::SeqCst) {
        if let Builds::Releases(releases) = builds
            && releases.refresh()?
        {
            reassert_profile(phone, &builds.capabilities()?, ctx, &mut reported)?;
        }
        match phone.control.reachability(&phone.entry.bundle_id) {
            Reachability::Unreachable(reason) => {
                pause_claims(&mut paused, &reason);
                // Jittered like the 204 wait: after a mass restart every proxy
                // finds its phone still coming back, and a flat pause would put
                // them in lockstep on the first claim.
                let wait = timing::idle_wait_with_jitter(pacing.idle_wait, pacing.idle_jitter);
                let _ = timing::sleep_interruptible(wait, &stop);
                continue;
            }
            Reachability::Reachable { os_version } => {
                resume_claims(&mut paused);
                // The server matches `os_version:` against the report, so a
                // new iOS goes up before the next claim.
                if phone.note_os_version(os_version) {
                    reassert_profile(phone, &builds.capabilities()?, ctx, &mut reported)?;
                    continue;
                }
            }
        }

        match server::poll_claim(ctx) {
            ClaimPoll::Job(job) => {
                backoff = pacing.claim_backoff;
                tally.attempted += 1;
                // Every line of this job, the heartbeat thread's included, carries
                // its id, up to `job finished`. A claim names no plan: one job can
                // serve several.
                let ran = tracing::info_span!("job", id = %job.job_id).in_scope(|| {
                    log_claimed_job(&job);
                    run_claimed_job(phone, builds, ctx, &job)
                });
                match ran {
                    Ok(()) => tally.completed += 1,
                    Err(error) if halts_phone(&error) => {
                        // The next claim would only burn another job the same way.
                        return Err(error);
                    }
                    Err(error) => {
                        // Anything reaching here is a proxy-side problem, and one
                        // bad job must not take a fleet daemon down.
                        tracing::error!(
                            error = %format!("{error:#}"),
                            "the job ended with an error; continuing the claim loop"
                        );
                    }
                }
                // The budget counts attempts, so an erroring job retires one too.
                if pacing.max_jobs > 0 && tally.attempted >= pacing.max_jobs {
                    tracing::info!(
                        max_jobs = %pacing.max_jobs,
                        "reached max_jobs; leaving the claim loop"
                    );
                    return Ok(());
                }
                if !stop.load(Ordering::SeqCst) {
                    reassert_profile(phone, &builds.capabilities()?, ctx, &mut reported)?;
                }
                if let Builds::Releases(releases) = builds {
                    releases.pace(&stop);
                }
            }
            ClaimPoll::Idle => {
                let wait = timing::idle_wait_with_jitter(pacing.idle_wait, pacing.idle_jitter);
                tracing::info!(
                    sleep_secs = %wait.as_secs(),
                    base_secs = %pacing.idle_wait.as_secs(),
                    jitter_secs = %pacing.idle_jitter.as_secs(),
                    "no work available (204)"
                );
                let _ = timing::sleep_interruptible(wait, &stop);
            }
            // Two unrelated fixes, so two messages. Typed rather than `bail!`ed
            // so `run_device` can recognise them as the errors NOT to retry.
            ClaimPoll::NotApproved => {
                return Err(OperatorActionError::NotApproved {
                    client_id: phone.entry.credentials.client_id.as_ref().to_owned(),
                }
                .into());
            }
            ClaimPoll::SignatureRejected => {
                return Err(OperatorActionError::SignatureRejected {
                    client_id: phone.entry.credentials.client_id.as_ref().to_owned(),
                }
                .into());
            }
            ClaimPoll::Transient(error) => {
                tracing::warn!(
                    error = %error,
                    sleep_secs = %backoff.as_secs(),
                    "claim failed transiently; backing off"
                );
                let _ = timing::sleep_interruptible(backoff, &stop);
                backoff = timing::next_backoff(backoff, timing::CLAIM_BACKOFF_CAP);
            }
            ClaimPoll::Fatal(error) => return Err(error.into()),
        }
    }

    tracing::info!(completed = %tally.completed, "stopping on a signal");
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::sync::atomic::{AtomicBool, AtomicUsize};
    use std::time::Duration;

    use httpmock::Method::{GET, PATCH, POST};
    use httpmock::{HttpMockResponse, Mock, MockServer};
    use rstest::rstest;
    use serde_json::json;

    use super::*;
    use crate::device::fake::FakeDevice;
    use crate::logging::testing::{captured, said};
    use crate::state::DeviceFacts;
    use crate::worker::testing::{
        fast_paced, identity_of, knows_every_benchmark, llamacpp, memory_claim, phone,
        profile_body, session,
    };

    /// What the phone told the proxy since its last report.
    #[derive(Debug, Clone, Copy)]
    enum Learned {
        Nothing,
        Ram,
        Ios,
    }

    /// Re-reporting between jobs puts the pool's capabilities back. A reindex after an
    /// unchanged report means something else replaced them, and the proxy waits it out
    /// before the next claim; after a changed report, the change was this proxy's own.
    /// The same iOS version reported again is no change.
    #[rstest]
    #[case::held_already(false, Learned::Nothing, Reasserted::Unchanged, 0)]
    #[case::replaced(true, Learned::Nothing, Reasserted::Replaced, 1)]
    #[case::facts_learned(true, Learned::Ram, Reasserted::Updated, 1)]
    #[case::ios_updated(true, Learned::Ios, Reasserted::Updated, 1)]
    fn the_profile_is_reported_again_between_jobs(
        #[case] reindex_pending: bool,
        #[case] learned: Learned,
        #[case] expected: Reasserted,
        #[case] gate_polls: usize,
    ) -> anyhow::Result<()> {
        let state = tempfile::tempdir()?;
        let (mut phone, mut pool) = phone(FakeDevice::new("phone-a"), state.path())?;
        pool.seed(
            &state.path().join("builds/A.app"),
            vec![llamacpp("b10216")?],
        );
        phone.details.os_version = Some("26.5.1".to_owned());
        let mut last = serde_json::to_value(report(&phone, &pool.capabilities()?))?;
        match learned {
            Learned::Nothing => {}
            Learned::Ram => {
                phone.facts = Some(DeviceFacts {
                    ram_bytes: Some(12 * 1024 * 1024 * 1024),
                    ..DeviceFacts::default()
                });
            }
            Learned::Ios => {
                assert!(phone.note_os_version(Some("26.6".to_owned())));
            }
        }
        let ios = json!({
            "device_os_name": "iOS",
            "device_os_version": phone.details.os_version,
        });
        let server = MockServer::start();
        let patch = server.mock(|when, then| {
            when.method(PATCH)
                .path("/clients/me")
                // The build flag is the plan-types vector for this llama.cpp descriptor.
                .body_includes(concat!(
                    r#""capabilities":["#,
                    r#""runtime:descriptor:6cef53d3f274d3f2a4c449b830556b5c7abadbe43b88c90dac8600d4a6442dc8","#,
                    r#""runtime:llama_cpp","runtime:llama_cpp:b10216"]"#,
                ))
                .json_body_includes(ios.to_string());
            then.status(200)
                .json_body(profile_body("approved", reindex_pending));
        });
        let gate = server.mock(|when, then| {
            when.method(GET).path("/clients/me");
            then.status(200).json_body(profile_body("approved", false));
        });

        let reasserted = reassert_profile(
            &phone,
            &pool.capabilities()?,
            &session(&phone, &server.base_url())?,
            &mut last,
        )?;

        assert_eq!(reasserted, expected);
        patch.assert_calls(1);
        gate.assert_calls(gate_polls);
        assert_eq!(
            last,
            serde_json::to_value(report(&phone, &pool.capabilities()?))?
        );
        Ok(())
    }

    fn http_status(status: u16) -> anyhow::Error {
        pipette_mgmt_client::Error::HttpStatus {
            method: "GET".to_owned(),
            url: "https://collector.example/clients/me".to_owned(),
            status,
            body: String::new(),
        }
        .into()
    }

    /// `PATCH /clients/me` carries the phone's iOS as `devicectl` read it, with
    /// its name, or neither when `devicectl` read none.
    #[rstest]
    #[case::read(Some("26.5.1"))]
    #[case::not_read(None)]
    fn the_report_carries_the_ios_version_with_its_name_or_neither(
        #[case] read: Option<&str>,
    ) -> anyhow::Result<()> {
        let state = tempfile::tempdir()?;
        let (mut phone, mut pool) = phone(FakeDevice::new("phone-a"), state.path())?;
        pool.seed(
            &state.path().join("builds/A.app"),
            vec![llamacpp("b10216")?],
        );
        phone.details.os_version = read.map(str::to_owned);
        let mut last = serde_json::to_value(report(&phone, &pool.capabilities()?))?;
        let server = MockServer::start();
        let patch = server.mock(|when, then| {
            let when = when
                .method(PATCH)
                .path("/clients/me")
                .json_body_includes(json!({ "device_form_factor": "phone" }).to_string());
            match read {
                Some(version) => when.json_body_includes(
                    json!({ "device_os_name": "iOS", "device_os_version": version }).to_string(),
                ),
                None => when
                    .body_excludes("device_os_name")
                    .body_excludes("device_os_version"),
            };
            then.status(200).json_body(profile_body("approved", false));
        });

        reassert_profile(
            &phone,
            &pool.capabilities()?,
            &session(&phone, &server.base_url())?,
            &mut last,
        )?;

        patch.assert_calls(1);
        Ok(())
    }

    /// A phone that retires after `max_jobs`, with a pool its claims can name.
    fn retiring_phone(
        device: FakeDevice,
        state: &Path,
        max_jobs: u32,
    ) -> anyhow::Result<(Phone, Pool)> {
        let (mut phone, mut pool) = phone(device, state)?;
        pool.seed(&state.join("builds/A.app"), vec![llamacpp("b10216")?]);
        let mut config = (*phone.config).clone();
        config.loop_config.max_jobs = max_jobs;
        phone.config = Arc::new(config);
        Ok((phone, pool))
    }

    /// Claims the proxy cannot read, and the failure each is reported as.
    fn unreadable_claims(server: &MockServer) -> (Mock<'_>, Mock<'_>) {
        let claim = server.mock(|when, then| {
            when.method(POST).path("/plans/claim");
            then.status(200).json_body(json!({
                "job_id": "job-mem",
                "benchmark_id": "max_memory_usage_512",
                "time_window": "PT10M",
                "spec": null,
            }));
        });
        let failed = server.mock(|when, then| {
            when.method(POST)
                .path("/benchmarks")
                .json_body_includes(r#"{"retriable": false}"#);
            then.status(202).json_body(json!({ "job_id": "job-mem" }));
        });
        (claim, failed)
    }

    /// Every job but the last is followed by one profile report, however it ended.
    #[test]
    fn the_claim_loop_reports_the_profile_between_jobs() -> anyhow::Result<()> {
        let state = tempfile::tempdir()?;
        let (mut phone, pool) = retiring_phone(FakeDevice::new("phone-a"), state.path(), 2)?;
        let server = MockServer::start();
        let (claim, failed) = unreadable_claims(&server);
        let patch = server.mock(|when, then| {
            when.method(PATCH).path("/clients/me");
            then.status(200).json_body(profile_body("approved", false));
        });
        let ctx = Arc::new(session(&phone, &server.base_url())?);

        claim_loop(
            &mut phone,
            &mut Builds::Pool(Arc::new(pool)),
            &ctx,
            &mut Tally::default(),
        )?;

        claim.assert_calls(2);
        failed.assert_calls(2);
        patch.assert_calls(1);
        Ok(())
    }

    /// Each outage pauses claims and each return resumes them, so a second
    /// outage starts a pause of its own.
    #[test]
    fn the_claim_loop_pauses_and_resumes_for_each_outage() -> anyhow::Result<()> {
        let state = tempfile::tempdir()?;
        let device = FakeDevice::new("phone-a")
            .then_gone("CoreDeviceError 1011")
            .then_back()
            .then_gone("developer mode is disabled")
            .then_back();
        let (mut phone, pool) = retiring_phone(device, state.path(), 2)?;
        fast_paced(&mut phone);
        let server = MockServer::start();
        let (claim, _) = unreadable_claims(&server);
        server.mock(|when, then| {
            when.method(PATCH).path("/clients/me");
            then.status(200).json_body(profile_body("approved", false));
        });
        let ctx = Arc::new(session(&phone, &server.base_url())?);

        let (ended, log) = captured(|| {
            claim_loop(
                &mut phone,
                &mut Builds::Pool(Arc::new(pool)),
                &ctx,
                &mut Tally::default(),
            )
        });

        ended?;
        claim.assert_calls(2);
        assert_eq!(
            said(&log)
                .into_iter()
                .filter(|line| line.contains("claims"))
                .collect::<Vec<_>>(),
            [
                "WARN the phone cannot run a job; pausing claims (no job is failed for this) \
                 reason=CoreDeviceError 1011",
                "INFO the phone can run a job again; resuming claims paused_mins=0",
                "WARN the phone cannot run a job; pausing claims (no job is failed for this) \
                 reason=developer mode is disabled",
                "INFO the phone can run a job again; resuming claims paused_mins=0",
            ]
        );
        Ok(())
    }

    /// The reachability probe before each claim reads the iOS version. A new
    /// one is reported before the claim, and the reindex it causes waited out;
    /// the same one (padded or not), none, or a blank one is not reported
    /// again.
    #[rstest]
    #[case::same(Some("26.5.1"), 0, "26.5.1")]
    #[case::padded(Some(" 26.5.1\n"), 0, "26.5.1")]
    #[case::not_read(None, 0, "26.5.1")]
    #[case::blank(Some("  "), 0, "26.5.1")]
    #[case::updated(Some("26.6"), 1, "26.6")]
    fn a_new_ios_version_is_reported_before_the_next_claim(
        #[case] read: Option<&str>,
        #[case] reports: usize,
        #[case] expected: &str,
    ) -> anyhow::Result<()> {
        let state = tempfile::tempdir()?;
        let device = read.map_or_else(
            || FakeDevice::new("phone-a"),
            |version| FakeDevice::new("phone-a").running_ios(version),
        );
        let (mut phone, pool) = retiring_phone(device, state.path(), 1)?;
        phone.details.os_version = Some("26.5.1".to_owned());
        let server = MockServer::start();
        let (claim, failed) = unreadable_claims(&server);
        let patch = server.mock(|when, then| {
            when.method(PATCH).path("/clients/me").json_body_includes(
                json!({ "device_os_name": "iOS", "device_os_version": expected }).to_string(),
            );
            then.status(200).json_body(profile_body("approved", true));
        });
        let gate = server.mock(|when, then| {
            when.method(GET).path("/clients/me");
            then.status(200).json_body(profile_body("approved", false));
        });
        let ctx = Arc::new(session(&phone, &server.base_url())?);

        claim_loop(
            &mut phone,
            &mut Builds::Pool(Arc::new(pool)),
            &ctx,
            &mut Tally::default(),
        )?;

        patch.assert_calls(reports);
        gate.assert_calls(reports);
        claim.assert_calls(1);
        failed.assert_calls(1);
        assert_eq!(phone.details.os_version.as_deref(), Some(expected));
        Ok(())
    }

    /// A failed report between jobs restarts the device, and the restart spends what is
    /// left of `max_jobs` rather than a fresh budget.
    #[test]
    fn a_restart_spends_the_remaining_job_budget() -> anyhow::Result<()> {
        let state = tempfile::tempdir()?;
        let (mut phone, pool) = retiring_phone(FakeDevice::new("phone-a"), state.path(), 2)?;
        let server = MockServer::start();
        let (claim, failed) = unreadable_claims(&server);
        let patch = server.mock(|when, then| {
            when.method(PATCH).path("/clients/me");
            then.status(500);
        });
        let ctx = Arc::new(session(&phone, &server.base_url())?);
        let mut tally = Tally::default();
        let mut builds = Builds::Pool(Arc::new(pool));

        let first = claim_loop(&mut phone, &mut builds, &ctx, &mut tally)
            .err()
            .context("the failed report should have restarted the device")?;
        // What `run_device` runs next, with the tally it kept.
        claim_loop(&mut phone, &mut builds, &ctx, &mut tally)?;

        assert!(
            format!("{first:#}").contains("re-reporting the profile between jobs"),
            "{first:#}"
        );
        claim.assert_calls(2);
        failed.assert_calls(2);
        patch.assert_calls(1);
        Ok(())
    }

    /// A job that stops its phone says why under its own id: the `device
    /// stopped` line that follows names only the phone.
    #[test]
    fn a_halt_names_the_job_that_earned_it() -> anyhow::Result<()> {
        let state = tempfile::tempdir()?;
        let device = FakeDevice::new("phone-a").then_console(identity_of("ev1_someone_else"));
        let (mut phone, pool) = retiring_phone(device, state.path(), 0)?;
        let job = memory_claim()?;
        let server = MockServer::start();
        knows_every_benchmark(&server);
        let claim = server.mock(|when, then| {
            when.method(POST).path("/plans/claim");
            then.status(200).json_body(json!({
                "job_id": job.job_id,
                "benchmark_id": job.benchmark_id,
                "time_window": job.time_window,
                "spec": job.spec,
            }));
        });
        let failed = server.mock(|when, then| {
            when.method(POST).path("/benchmarks");
            then.status(202).json_body(json!({ "job_id": "job-mem" }));
        });
        let ctx = Arc::new(session(&phone, &server.base_url())?);

        let (ended, log) = captured(|| {
            claim_loop(
                &mut phone,
                &mut Builds::Pool(Arc::new(pool)),
                &ctx,
                &mut Tally::default(),
            )
        });

        let error = ended.err().context("the phone must halt")?;
        assert!(halts_phone(&error), "{error:#}");
        claim.assert_calls(1);
        failed.assert_calls(1);
        let halted = log
            .lines()
            .find(|line| line.contains("the job stops this phone until an operator fixes it"))
            .context("no halt line")?;
        assert!(halted.contains(" job{id=job-mem}: "), "{halted}");
        assert!(halted.contains("ev1_someone_else"), "{halted}");
        Ok(())
    }

    #[rstest]
    #[case::not_approved(403, true)]
    #[case::bad_signature(401, true)]
    #[case::protocol_mismatch(422, true)]
    #[case::server_error(500, false)]
    #[case::gateway(502, false)]
    fn a_4xx_is_operator_actionable_and_a_5xx_is_not(#[case] status: u16, #[case] terminal: bool) {
        assert_eq!(is_operator_actionable(&http_status(status)), terminal);
    }

    #[test]
    fn the_claim_loops_typed_stops_are_operator_actionable() {
        let not_approved: anyhow::Error = OperatorActionError::NotApproved {
            client_id: "ev1_x".to_owned(),
        }
        .into();
        let bad_key: anyhow::Error = OperatorActionError::SignatureRejected {
            client_id: "ev1_x".to_owned(),
        }
        .into();
        assert!(is_operator_actionable(&not_approved));
        assert!(is_operator_actionable(&bad_key));
    }

    #[test]
    fn a_replaced_identity_is_operator_actionable() {
        let replaced: anyhow::Error = install::IdentityReplacedError {
            checked: "after installing /pool/Other.app".to_owned(),
            device: "phone-a".to_owned(),
            expected: "ev1_test".to_owned(),
            found: "ev1_someone_else".to_owned(),
        }
        .into();
        assert!(is_operator_actionable(
            &replaced.context("installing the build")
        ));
    }

    #[test]
    fn a_wrapped_4xx_is_still_recognised() {
        let wrapped: anyhow::Error = http_status(403)
            .context("PATCH /clients/me")
            .context("connecting phone-a");
        assert!(is_operator_actionable(&wrapped));
    }

    #[test]
    fn a_connection_failure_is_retried() {
        let network = anyhow::anyhow!("connection refused (os error 61)");
        assert!(!is_operator_actionable(&network));
    }

    /// The operator's error only when every device that failed needs one: a
    /// panic beside it is worth a restart. A stop signal is a clean exit.
    #[rstest]
    #[case::none(&[], false, None)]
    #[case::one_operator(&[DeviceStop::Operator], false, Some(true))]
    #[case::all_operator(&[DeviceStop::Operator, DeviceStop::Operator], false, Some(true))]
    #[case::mixed(&[DeviceStop::Operator, DeviceStop::Other], false, Some(false))]
    #[case::other(&[DeviceStop::Other], false, Some(false))]
    #[case::signalled(&[DeviceStop::Operator], true, None)]
    fn run_ends_on_the_operators_error_only_when_every_failure_needs_one(
        #[case] stopped: &[DeviceStop],
        #[case] signalled: bool,
        #[case] operator: Option<bool>,
    ) {
        let ended = run_outcome(stopped, signalled)
            .err()
            .map(|error| error.is::<OperatorStopError>());
        assert_eq!(ended, operator);
    }

    /// `run` over one pool-mode phone whose server is `server`, retiring after
    /// one job, so a phone that should have stopped returns instead of
    /// claiming forever.
    fn run_one_phone(
        device: FakeDevice,
        state: &Path,
        server: &MockServer,
        stop: &Arc<AtomicBool>,
    ) -> anyhow::Result<()> {
        let (mut phone, _) = phone(device, state)?;
        phone.stop = Arc::clone(stop);
        let bundle = state.join("builds/A.app");
        std::fs::create_dir_all(&bundle)?;
        std::fs::write(bundle.join("Info.plist"), "<plist/>")?;
        let mut read = Pool::open(
            &crate::config::BuildsConfig {
                dir: state.join("builds"),
            },
            &state.join("builds.json"),
        )?;
        read.refresh(&state.join("cfg.toml"), |_| {
            Ok(crate::pool::BuildReading {
                runtimes: vec![llamacpp("b10216")?],
            })
        })?;
        let mut config = (*phone.config).clone();
        config.server.url = pipette_mgmt_client::types::ServerUrl::try_new(server.base_url())
            .map_err(|error| anyhow::anyhow!("test server url: {error}"))?;
        config.loop_config.max_jobs = 1;
        config.loop_config.claim_backoff = Duration::from_millis(10);
        let config = Arc::new(config);
        phone.config = Arc::clone(&config);
        let stop = Arc::clone(&phone.stop);
        run(Fleet {
            phones: vec![phone],
            locks: Vec::new(),
            config,
            state: crate::state::StatePaths::resolve(Some(state))?,
            stop,
        })
    }

    /// With `[devicectl] pymobiledevice3` set, `run` refuses to start rather than
    /// claim jobs it would fail one by one.
    #[rstest]
    #[case::no_hardware_udid(None, "named no hardware UDID")]
    #[case::the_cli_fails(Some("startup-pin-test-udid"), "`version` exited")]
    fn a_configured_pin_that_cannot_work_stops_run_before_any_claim(
        #[case] hardware_udid: Option<&str>,
        #[case] reason: &str,
    ) -> anyhow::Result<()> {
        use std::os::unix::fs::PermissionsExt;

        let state = tempfile::tempdir()?;
        let cli = state.path().join("pymobiledevice3");
        std::fs::write(&cli, "#!/bin/sh\necho broken >&2\nexit 1\n")?;
        std::fs::set_permissions(&cli, std::fs::Permissions::from_mode(0o755))?;
        let (mut phone, _) = phone(FakeDevice::new("phone-a"), state.path())?;
        phone.details.hardware_udid = hardware_udid.map(str::to_owned);
        let server = MockServer::start();
        let claim = server.mock(|when, then| {
            when.method(POST).path("/plans/claim");
            then.status(204);
        });
        let mut config = (*phone.config).clone();
        config.server.url = pipette_mgmt_client::types::ServerUrl::try_new(server.base_url())
            .map_err(|error| anyhow::anyhow!("test server url: {error}"))?;
        config.devicectl.pymobiledevice3 = Some(cli);
        let config = Arc::new(config);
        phone.config = Arc::clone(&config);
        let stop = Arc::clone(&phone.stop);

        let error = run(Fleet {
            phones: vec![phone],
            locks: Vec::new(),
            config,
            state: crate::state::StatePaths::resolve(Some(state.path()))?,
            stop,
        })
        .err()
        .context("run must refuse to start")?;

        let message = format!("{error:#}");
        assert!(message.contains(reason), "{message}");
        assert!(message.contains("[devicectl] pymobiledevice3"), "{message}");
        claim.assert_calls(0);
        Ok(())
    }

    /// A build that replaced the phone's identity stops the phone after the
    /// one job it failed, and `run` ends on the operator's error: a restart
    /// would fail another job the same way.
    #[test]
    fn a_phone_that_needs_a_person_ends_run_on_the_operators_error() -> anyhow::Result<()> {
        let state = tempfile::tempdir()?;
        let device = FakeDevice::new("phone-a").then_console(identity_of("ev1_someone_else"));
        let server = MockServer::start();
        knows_every_benchmark(&server);
        let patch = server.mock(|when, then| {
            when.method(PATCH).path("/clients/me");
            then.status(200).json_body(profile_body("approved", false));
        });
        let gate = server.mock(|when, then| {
            when.method(GET).path("/clients/me");
            then.status(200).json_body(profile_body("approved", false));
        });
        let spec = json!({
            "benchmark": "max_memory_usage_512",
            "model": {
                "type": "gguf_text",
                "source": "huggingface",
                "org": "LiquidAI",
                "repo_name": "LFM2.5-350M-GGUF",
                "path": "LFM2.5-350M-Q4_0.gguf",
            },
            "runtime": serde_json::to_value(llamacpp("b10216")?)?,
            "runtime_flags": null,
            "model_flags": null,
            "benchmark_flags": null,
        });
        let claim = server.mock(|when, then| {
            when.method(POST).path("/plans/claim");
            then.status(200).json_body(json!({
                "job_id": "job-mem",
                "benchmark_id": "max_memory_usage_512",
                "time_window": "PT10M",
                "spec": spec,
            }));
        });
        let failed = server.mock(|when, then| {
            when.method(POST)
                .path("/benchmarks")
                .body_includes("ev1_someone_else");
            then.status(202).json_body(json!({ "job_id": "job-mem" }));
        });

        let ended = run_one_phone(device, state.path(), &server, &Arc::default())
            .err()
            .context("a phone that needs a person must end `run` with an error")?;

        patch.assert_calls(1);
        gate.assert_calls(1);
        claim.assert_calls(1);
        failed.assert_calls(1);
        let operator = ended
            .downcast_ref::<OperatorStopError>()
            .context("the operator's error")?;
        assert_eq!(operator.stopped, 1);
        Ok(())
    }

    /// A `4xx` stops the phone before any claim, so `run` ends on the generic
    /// error: a restart costs no job and recovers once the server is fixed.
    #[test]
    fn a_refused_phone_ends_run_on_the_generic_error() -> anyhow::Result<()> {
        let state = tempfile::tempdir()?;
        let server = MockServer::start();
        let stop = Arc::new(AtomicBool::new(false));
        let patch = server.mock(|when, then| {
            when.method(PATCH).path("/clients/me");
            let stop = Arc::clone(&stop);
            let calls = AtomicUsize::new(0);
            then.respond_with(move |_| {
                // A retry means the 4xx was not recognised: stop, so the test
                // fails instead of retrying forever.
                if calls.fetch_add(1, Ordering::SeqCst) > 0 {
                    stop.store(true, Ordering::SeqCst);
                }
                HttpMockResponse::builder().status(403).build()
            });
        });

        let ended = run_one_phone(FakeDevice::new("phone-a"), state.path(), &server, &stop)
            .err()
            .context("a refused phone must end `run` with an error")?;

        patch.assert_calls(1);
        assert_eq!(
            format!("{ended:#}"),
            "1 of the configured devices stopped with an error"
        );
        Ok(())
    }
}
