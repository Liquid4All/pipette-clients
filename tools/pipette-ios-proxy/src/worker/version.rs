//! Version mode (DESIGN section 9): every claim runs on a build its
//! `client_versions` admits, pulled from the releases bucket, signed on this
//! Mac, and installed when the phone runs another.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::{Duration, SystemTime};

use anyhow::Context;

use pipette_http::HttpClient;
use pipette_mgmt_client::types::ClaimedJob;
use pipette_plan_types::{Runtime, RuntimeType};

use super::Builds;
use super::fleet::Phone;
use super::install;
use super::job::{self, JobEnd, Lease};
use super::summary::{JobSummary, Outcome, Unreported, timed};
use crate::app::{self, LaunchPlan, Signals};
use crate::config::{Config, ReleasesBucket};
use crate::device::Reachability;
use crate::pool::bundle::{self, FleetSigned};
use crate::releases::catalog::{Catalog, CatalogError, ChosenBy, NoChoice};
use crate::releases::sign::{self, Signer};
use crate::releases::store;
use crate::releases::{Build, BuildVersion, Fault, UnreadyError, descriptor_key, plan_toml, s3};
use crate::server::Session;
use crate::server::submit::{FailureCause, JobFailure};
use crate::state::{InstalledBuild, ReleasePaths, StatePaths};
use crate::timing;

/// A failure only an operator can clear (DESIGN section 9): the phone stops,
/// at startup or during a claim.
#[derive(Debug, thiserror::Error)]
#[error("this phone stops until an operator fixes it")]
pub(crate) struct OperatorNeededError(#[source] anyhow::Error);

pub(crate) fn needs_operator(error: &anyhow::Error) -> bool {
    error.chain().any(|cause| cause.is::<OperatorNeededError>())
}

#[derive(Debug, thiserror::Error)]
#[error("the lease was lost or a stop arrived before the install")]
struct StoppedBeforeInstallError;

/// The phone runs another build than the one just installed: the archive's
/// version label is not the one it is published under.
#[derive(Debug, thiserror::Error)]
#[error("after installing build {installed}, the phone reports {reported}")]
struct ReportsAnotherVersionError {
    installed: BuildVersion,
    reported: String,
}

/// The phone's own listing of an installed build is not what its manifest
/// names: the release wrote the manifest wrong, so no job can trust it.
#[derive(Debug, thiserror::Error)]
#[error("build {version} reports the runtimes {reported}, but its manifest names {published}")]
struct RuntimesDifferError {
    version: BuildVersion,
    published: String,
    reported: String,
}

/// Why a build did not get onto the phone, as the claim reports it.
#[derive(Debug)]
struct Failed {
    cause: FailureCause,
    fault: Fault,
    error: anyhow::Error,
    /// The phone had reported running the chosen version.
    reported: bool,
}

impl From<UnreadyError> for Failed {
    fn from(unready: UnreadyError) -> Self {
        Self {
            cause: match unready.fault {
                Fault::Version => FailureCause::BuildUnusable,
                Fault::Operator | Fault::Transient => FailureCause::BuildNotReady,
            },
            fault: unready.fault,
            error: unready.error,
            reported: false,
        }
    }
}

impl Failed {
    /// A `devicectl` call or an app verb that failed.
    fn device(error: anyhow::Error) -> Self {
        Self {
            cause: job::failure_cause(&error, FailureCause::DevicectlFailure),
            fault: Fault::Transient,
            error,
            reported: false,
        }
    }

    /// The client check after an install, or before trusting a build this
    /// proxy did not record installing.
    fn identity(error: anyhow::Error) -> Self {
        if install::replaced_identity(&error) {
            Self {
                cause: FailureCause::IdentityReplaced,
                fault: Fault::Operator,
                error,
                reported: false,
            }
        } else {
            Self::device(error)
        }
    }

    fn after_report(self) -> Self {
        Self {
            reported: true,
            ..self
        }
    }

    /// Before the phone claims, where nothing is reported: an operator's fault
    /// stops the phone, anything else restarts it.
    fn at_startup(self, doing: String) -> anyhow::Error {
        match self.fault {
            Fault::Operator => OperatorNeededError(self.error).into(),
            Fault::Version | Fault::Transient => self.error.context(doing),
        }
    }
}

/// The build a claim runs on, and why that one.
struct Chosen {
    build: Build,
    by: ChosenBy,
}

/// One phone's side of the bucket in `run`.
pub(crate) struct Releases {
    pub(crate) catalog: Catalog,
    bucket: ReleasesBucket,
    http: HttpClient,
    paths: ReleasePaths,
    signer: Arc<Signer>,
    /// The wait before the next claim ([`backoff_after`]).
    backoff: Option<Duration>,
}

/// One [`Builds::Releases`] per phone, after proving this Mac can sign with
/// `signer`: a wrong keychain password stops `run` before any phone starts.
pub(crate) fn fleet_builds(
    signer: Signer,
    config: &Config,
    state: &StatePaths,
    phones: usize,
) -> anyhow::Result<Vec<Builds>> {
    signer
        .preflight()
        .context("checking that this Mac can sign builds")?;
    let signer = Arc::new(signer);
    let bucket = config.releases_bucket()?;
    let http = s3::http_client()?;
    Ok((0..phones)
        .map(|_| {
            Builds::Releases(Box::new(Releases::new(
                bucket.clone(),
                http.clone(),
                state.releases(),
                Arc::clone(&signer),
                config.default_version().cloned(),
            )))
        })
        .collect())
}

/// The versions `register` and `probe` advertise: no secret but the bucket's
/// is read, and nothing is installed.
pub(crate) fn listed_catalog(config: &Config, state: &StatePaths) -> anyhow::Result<Catalog> {
    Catalog::listed(
        &s3::http_client()?,
        &config.releases_bucket()?,
        config.default_version(),
        &state.releases(),
    )
}

impl Releases {
    pub(crate) fn new(
        bucket: ReleasesBucket,
        http: HttpClient,
        paths: ReleasePaths,
        signer: Arc<Signer>,
        default: Option<BuildVersion>,
    ) -> Self {
        Self {
            catalog: Catalog::new(default),
            bucket,
            http,
            paths,
            signer,
            backoff: None,
        }
    }

    /// List the bucket again when it is due, holding no lease. `true` when what
    /// this phone advertises changed.
    pub(crate) fn refresh(&mut self) -> Result<bool, OperatorNeededError> {
        self.check_offered()?;
        if !self.catalog.due() {
            return Ok(false);
        }
        let before = self.advertised()?;
        if let Err(error) = self.catalog.relist(&self.http, &self.bucket, &self.paths) {
            tracing::warn!(
                error = %format!("{error:#}"),
                "listing the releases bucket failed; advertising what it listed before"
            );
            return Ok(false);
        }
        self.check_offered()?;
        Ok(self.advertised()? != before)
    }

    /// The catalog's flags. If the proxy cannot digest a descriptor, the proxy or a
    /// manifest has a defect. A retry returns the same error.
    fn advertised(&self) -> Result<Vec<String>, OperatorNeededError> {
        self.catalog
            .advertised()
            .map_err(|error| OperatorNeededError(error.into()))
    }

    fn check_offered(&self) -> Result<(), OperatorNeededError> {
        self.catalog
            .check_offered(&self.bucket)
            .map_err(|error| OperatorNeededError(error.into()))
    }

    /// Before the phone connects: list the bucket, and install a build when the
    /// phone runs none this proxy offers. A phone on any offered version is
    /// left alone, so a restart does not undo the last job's install.
    pub(crate) fn start(&mut self, phone: &mut Phone) -> anyhow::Result<()> {
        // A restart after a server blip needs no bucket.
        if let Err(error) = self.catalog.relist(&self.http, &self.bucket, &self.paths) {
            if !self.catalog.has_listing() {
                return Err(error.context("listing the releases bucket"));
            }
            tracing::warn!(
                error = %format!("{error:#}"),
                "listing the releases bucket failed; offering what it listed before"
            );
        }
        self.check_offered()?;
        let running = running_version(phone)
            .map_err(|failed| failed.error.context("reading the version the phone runs"))?;
        if let Some(build) = running
            .as_deref()
            .and_then(|version| self.catalog.offers(version))
            .cloned()
        {
            tracing::info!(version = %build.version, "the phone runs a version this proxy offers");
            return match trust_running(phone, &build) {
                Ok(()) => Ok(()),
                // As for a startup install: each claim installs a build it can run.
                Err(failed) if failed.fault == Fault::Version => {
                    tracing::warn!(
                        version = %build.version,
                        error = %format!("{:#}", failed.error),
                        "the build found running fails its runtime listing; withdrawing it and claiming"
                    );
                    self.catalog.withdraw(&build.version);
                    self.check_offered()?;
                    Ok(())
                }
                Err(failed) => {
                    Err(failed
                        .at_startup(format!("checking the client of build {}", build.version)))
                }
            };
        }
        let build = self
            .catalog
            .choose(None, None)
            .map(|choice| choice.build.clone())
            .map_err(|_| CatalogError::Empty {
                bucket: self.bucket.url.to_string(),
            })?;
        tracing::info!(
            from = %running.as_deref().unwrap_or("none"),
            to = %build.version,
            "the phone runs no version this proxy offers; installing one before claiming"
        );
        let never = AtomicBool::new(false);
        let stop = Arc::clone(&phone.stop);
        let signals = Signals {
            abort: &never,
            stop: &stop,
        };
        // Not a job: nothing reads these times.
        let unlogged = &mut JobSummary::default();
        match self.install(phone, &build, running.as_deref(), signals, unlogged) {
            Ok(()) => Ok(()),
            Err(_) if signals.raised() => Ok(()),
            Err(failed) if failed.fault == Fault::Version => {
                tracing::warn!(
                    version = %build.version,
                    error = %format!("{:#}", failed.error),
                    "the startup build cannot go onto this phone; claiming without it"
                );
                self.catalog.withdraw(&build.version);
                self.check_offered()?;
                Ok(())
            }
            // Left to the claims, under their backoff: a restart would install
            // again every minute and never report the profile.
            Err(failed) if failed.cause == FailureCause::DevicectlFailure => {
                tracing::warn!(
                    version = %build.version,
                    error = %format!("{:#}", failed.error),
                    "the phone did not take the startup build; claiming, and each claim installs its own"
                );
                Ok(())
            }
            Err(failed) => Err(failed.at_startup(format!(
                "installing build {} before claiming",
                build.version
            ))),
        }
    }

    /// Sleeps the backoff the last job set, if any; a stop cuts it short.
    pub(crate) fn pace(&self, stop: &AtomicBool) {
        if let Some(wait) = self.backoff {
            tracing::info!(
                wait_secs = %wait.as_secs(),
                "the last job failed in a way the next claim would likely repeat; waiting"
            );
            let _ = timing::sleep_interruptible(wait, stop);
        }
    }

    /// The build a claim runs on (DESIGN section 9, Choosing the version). A
    /// runtime no admitted build carries is checked against a listing of the
    /// whole bucket taken now: only one in which no build carries it shows no
    /// phone can run the job.
    fn choose(
        &mut self,
        admitted: Option<&[String]>,
        plan: &LaunchPlan,
    ) -> Result<Chosen, JobFailure> {
        let mut chosen = self.catalog.choose(admitted, Some(&plan.runtime));
        let mut fresh = false;
        if let Err(NoChoice::RuntimeNotCarried { .. }) = chosen {
            fresh = match self
                .catalog
                .relist_whole(&self.http, &self.bucket, &self.paths)
            {
                Ok(()) => true,
                Err(error) => {
                    tracing::warn!(
                        error = %format!("{error:#}"),
                        "listing the releases bucket again failed; the job stays retriable"
                    );
                    false
                }
            };
            chosen = self.catalog.choose(admitted, Some(&plan.runtime));
        }
        match chosen {
            Ok(choice) => {
                tracing::info!(
                    version = %choice.build.version,
                    chosen_by = %<&str>::from(choice.by),
                    admitted = %describe(admitted),
                    "chose the build for this job"
                );
                Ok(Chosen {
                    build: choice.build.clone(),
                    by: choice.by,
                })
            }
            Err(NoChoice::NoneAdmitted) => Err(JobFailure::new(
                FailureCause::NoAdmittedVersion,
                format!(
                    "the claim admits {}, and this phone offers {}",
                    describe(admitted),
                    describe(Some(&self.catalog.version_flags())),
                ),
            )),
            Err(NoChoice::RuntimeNotCarried { admitted: offered }) => {
                if fresh
                    && let Some(admitted) = admitted
                    && let Some(carried) = self.catalog.carried_instead(admitted, &plan.runtime)
                {
                    return Err(JobFailure::new(
                        FailureCause::NoBuildCarriesRuntime,
                        format!(
                            "the job's runtime {} is in no build in the releases bucket, so no \
                             phone can run it. The builds the claim admits ({}) carry, in plan \
                             TOML: {}",
                            plan_toml(&plan.runtime),
                            admitted.join(", "),
                            alternatives(&carried, &plan.runtime),
                        ),
                    ));
                }
                Err(JobFailure::new(
                    FailureCause::RuntimeNotInBuild,
                    format!(
                        "the job's runtime {} is in the manifest of no build the claim admits \
                         here ({})",
                        plan.runtime_json,
                        offered
                            .iter()
                            .map(BuildVersion::as_str)
                            .collect::<Vec<_>>()
                            .join(", "),
                    ),
                ))
            }
        }
    }

    /// DESIGN section 9's claim up to the run: choose the build, and have the
    /// phone run it.
    fn prepare(
        &mut self,
        phone: &mut Phone,
        job: &ClaimedJob,
        plan: &LaunchPlan,
        signals: Signals<'_>,
        summary: &mut JobSummary,
    ) -> Result<Build, JobEnd> {
        let Chosen { build, by } = timed(&mut summary.choose, || {
            self.choose(job.client_versions.as_deref(), plan)
        })?;
        summary.version = Some(build.version.clone());
        summary.chosen_by = Some(by.into());

        let ready = match running_version(phone) {
            Ok(running) if running.as_deref() == Some(build.version.as_str()) => {
                trust_running(phone, &build).map_err(Failed::after_report)
            }
            Ok(running) => self.install(phone, &build, running.as_deref(), signals, summary),
            Err(failed) => Err(failed),
        };
        if let Err(failed) = ready {
            let failure = JobFailure::new(failed.cause, format!("{:#}", failed.error))
                .on_build(build.version.as_str(), failed.reported);
            let halt = match failed.fault {
                Fault::Operator => Some(OperatorNeededError(failed.error).into()),
                Fault::Version => {
                    self.catalog.withdraw(&build.version);
                    self.check_offered().err().map(anyhow::Error::from)
                }
                Fault::Transient => None,
            };
            return Err(JobEnd {
                report: Ok(failure),
                halt,
            });
        }
        Ok(build)
    }

    /// Pull, sign and install `build`, each step's time going to `summary`.
    fn install(
        &self,
        phone: &mut Phone,
        build: &Build,
        from: Option<&str>,
        signals: Signals<'_>,
        summary: &mut JobSummary,
    ) -> Result<(), Failed> {
        let dir = timed(&mut summary.pull, || {
            store::ensure_pulled(&self.http, &self.bucket, &self.paths, build, signals)
        })?;
        let staged = timed(&mut summary.sign, || self.sign(phone, build, &dir, signals))?;

        // The last point a stop or a lost lease is honoured: an install half
        // done is worse.
        if signals.raised() {
            return Err(UnreadyError::transient(StoppedBeforeInstallError).into());
        }
        tracing::info!(
            from = %from.unwrap_or("none"),
            to = %build.version,
            "installing the build"
        );
        timed(&mut summary.install, || {
            phone
                .set_installed(None)
                .map_err(|error| Failed::from(UnreadyError::transient(error)))?;
            phone
                .control
                .install_app(&staged)
                .with_context(|| {
                    format!(
                        "installing {} on {}",
                        staged.display(),
                        phone.control.udid()
                    )
                })
                .map_err(Failed::device)?;
            summary.installed = true;
            check_installed(phone, build, staged)
        })
    }

    /// Check the archived app, then sign it for this phone and stage it.
    fn sign(
        &self,
        phone: &Phone,
        build: &Build,
        dir: &Path,
        signals: Signals<'_>,
    ) -> Result<PathBuf, UnreadyError> {
        let archived = store::archived(dir)?;
        bundle::check_archived(&archived.app, &build.version, &phone.entry.bundle_id)?;
        let control = phone.control.as_ref();
        let gate = |app: &Path, expiry_margin: Duration| {
            let fleet = FleetSigned {
                team_id: &self.signer.team_id,
                bundle_id: &phone.entry.bundle_id,
                now: SystemTime::now(),
                expiry_margin,
            };
            control.verify_installable(app, Some(&fleet))
        };
        sign::signed_build(
            &self.signer,
            &self.paths,
            build,
            &archived.archive,
            &gate,
            &phone.paths.staged,
            signals,
        )
    }
}

/// The checks after an install: the phone holds its client, runs `build`, and
/// lists the runtimes its manifest names. Then it is recorded.
fn check_installed(phone: &mut Phone, build: &Build, staged: PathBuf) -> Result<(), Failed> {
    let call_timeout = phone.config.devicectl.call_timeout;
    let first_line = phone.config.devicectl.first_line_timeout;
    install::check_after(
        phone.control.as_ref(),
        &phone.entry,
        &format!("after installing {}", staged.display()),
        call_timeout,
        first_line,
    )
    .map_err(Failed::identity)?;
    let control = phone.control.as_ref();
    let bundle_id = &phone.entry.bundle_id;
    let reported = app::app_version(control, bundle_id, call_timeout, first_line)
        .context("reading the version the phone runs after the install")
        .map_err(Failed::device)?;
    if reported.as_deref() != Some(build.version.as_str()) {
        return Err(Failed::from(UnreadyError::new(
            Fault::Version,
            ReportsAnotherVersionError {
                installed: build.version.clone(),
                reported: reported.unwrap_or_else(|| "no version".to_owned()),
            },
        )));
    }
    check_runtimes(phone, build)?;
    phone
        .record_installed(InstalledBuild {
            path: Some(staged),
            version: Some(build.version.to_string()),
        })
        .map_err(|error| Failed::from(UnreadyError::transient(error)).after_report())
}

/// The phone's listing of the runtimes `build` provides against its manifest.
fn check_runtimes(phone: &Phone, build: &Build) -> Result<(), Failed> {
    let listing = app::list_runtimes(
        phone.control.as_ref(),
        &phone.entry.bundle_id,
        phone.config.devicectl.call_timeout,
        phone.config.devicectl.first_line_timeout,
    )
    .context("listing the runtimes the phone runs")
    .map_err(|error| {
        // The listing loads every llama.cpp build the app carries, so a refusal is the
        // build's own fault.
        if error
            .chain()
            .any(|cause| cause.is::<app::RuntimesListFailedError>())
        {
            Failed::from(UnreadyError::new(Fault::Version, error))
        } else {
            Failed::device(error)
        }
        .after_report()
    })?;
    let reported = reported_keys(&listing.runtimes, &listing.unreadable);
    let published = build.runtime_keys();
    if reported != published {
        return Err(Failed::from(UnreadyError::new(
            Fault::Version,
            RuntimesDifferError {
                version: build.version.clone(),
                published: published.join(", "),
                reported: reported.join(", "),
            },
        ))
        .after_report());
    }
    Ok(())
}

/// One claim in version mode: the heartbeat starts before the build is chosen,
/// so a pull, an export and an install all run under the lease.
pub(super) fn execute_job(
    phone: &mut Phone,
    releases: &mut Releases,
    ctx: &Arc<Session>,
    job: &ClaimedJob,
    summary: &mut JobSummary,
) -> Result<Outcome, JobEnd> {
    let ended = run_job(phone, releases, ctx, job, summary);
    releases.backoff = backoff_after(
        releases.backoff,
        ended
            .as_ref()
            .err()
            .and_then(|end| end.report.as_ref().ok())
            .map(|failure| failure.cause),
        phone.config.loop_config.claim_backoff,
    );
    ended
}

/// The wait before the next claim: the claim backoff after a job that ended
/// `build_not_ready` or `devicectl_failure`, which a claim right after would
/// likely meet again, doubling while they repeat; none after anything else.
fn backoff_after(
    previous: Option<Duration>,
    ended: Option<FailureCause>,
    base: Duration,
) -> Option<Duration> {
    matches!(
        ended,
        Some(FailureCause::BuildNotReady | FailureCause::DevicectlFailure)
    )
    .then(|| {
        previous.map_or(base, |wait| {
            timing::next_backoff(wait, timing::PREPARE_BACKOFF_CAP)
        })
    })
}

fn run_job(
    phone: &mut Phone,
    releases: &mut Releases,
    ctx: &Arc<Session>,
    job: &ClaimedJob,
    summary: &mut JobSummary,
) -> Result<Outcome, JobEnd> {
    let plan = job::launch_plan(job)?;
    job::known_benchmark(ctx, job)?;
    let lease = Lease::start(ctx, job, job::heartbeat_interval(phone, job));
    let lost = Arc::clone(&lease.lost);
    let stop = Arc::clone(&phone.stop);
    let signals = Signals {
        abort: &lost,
        stop: &stop,
    };
    let prepared = releases.prepare(phone, job, &plan, signals, summary);
    if signals.raised() {
        // The lease is gone, or lapses: nothing is reported either way, but a
        // halt still stops the phone.
        drop(lease);
        tracing::warn!(
            "the lease was lost or a stop arrived while preparing the build; nothing to report"
        );
        return match prepared {
            Err(JobEnd {
                halt: Some(halt), ..
            }) => Err(JobEnd {
                report: Err(Unreported::of(signals)),
                halt: Some(halt),
            }),
            Ok(_) | Err(JobEnd { halt: None, .. }) => Ok(Unreported::of(signals).into()),
        };
    }
    let build = prepared?;
    let on_build = |end: JobEnd| JobEnd {
        report: end
            .report
            .map(|failure| failure.on_build(build.version.as_str(), true)),
        halt: end.halt,
    };
    let before = job::results_before(phone).map_err(on_build)?;
    job::run_and_submit(
        phone,
        ctx,
        job,
        &plan,
        &before,
        lease,
        Some(&build.version),
        summary,
    )
    .map_err(on_build)
}

/// What `headlessrun version` names, or `None` when the app does not answer
/// on a reachable phone (not installed). A phone that is gone fails here,
/// before anything is pulled or signed for it.
fn running_version(phone: &Phone) -> Result<Option<String>, Failed> {
    let bundle_id = &phone.entry.bundle_id;
    app::app_version(
        phone.control.as_ref(),
        bundle_id,
        phone.config.devicectl.call_timeout,
        phone.config.devicectl.first_line_timeout,
    )
    .or_else(|error| match phone.control.reachability(bundle_id) {
        Reachability::Reachable { .. } => {
            tracing::debug!(error = %format!("{error:#}"), "the app named no version");
            Ok(None)
        }
        Reachability::Unreachable(reason) => Err(Failed {
            cause: FailureCause::DeviceUnreachable,
            fault: Fault::Transient,
            error: error.context(format!("the phone is unreachable: {reason}")),
            reported: false,
        }),
    })
}

/// The phone runs `version`. Unless `installed.json` names it, this proxy did
/// not see the install through (one cut short, or a new state directory), so
/// the phone proves its client first, as after an install.
fn trust_running(phone: &mut Phone, build: &Build) -> Result<(), Failed> {
    let version = &build.version;
    let recorded = InstalledBuild::load(&phone.paths.installed)
        .ok()
        .and_then(|installed| installed.version);
    if recorded.as_deref() == Some(version.as_str()) {
        return Ok(());
    }
    install::check_after(
        phone.control.as_ref(),
        &phone.entry,
        &format!("on build {version} found running"),
        phone.config.devicectl.call_timeout,
        phone.config.devicectl.first_line_timeout,
    )
    .map_err(Failed::identity)?;
    // Unrecorded, so its check after the install did not pass, or never ran.
    check_runtimes(phone, build)?;
    let record = InstalledBuild {
        path: None,
        version: Some(version.to_string()),
    };
    if let Err(error) = phone.record_installed(record) {
        tracing::warn!(error = %format!("{error:#}"), "could not record the version on the phone");
    }
    Ok(())
}

/// The phone's `runtimes list`, spelled as [`Build::runtime_keys`] spells a
/// manifest.
fn reported_keys(runtimes: &[Runtime], unreadable: &[String]) -> Vec<String> {
    let readable = runtimes.iter().map(|runtime| {
        serde_json::to_value(runtime)
            .map_or_else(|_| runtime.to_string(), |value| descriptor_key(&value))
    });
    let unreadable = unreadable.iter().map(|raw| {
        serde_json::from_str::<serde_json::Value>(raw)
            .map_or_else(|_| raw.clone(), |value| descriptor_key(&value))
    });
    let mut keys: Vec<String> = readable.chain(unreadable).collect();
    keys.sort();
    keys
}

/// The runtimes a plan could name instead of `wanted`, in plan TOML: those of
/// its type when the builds carry any, else every one.
fn alternatives(carried: &[Runtime], wanted: &Runtime) -> String {
    let same_type: Vec<&Runtime> = carried
        .iter()
        .filter(|runtime| RuntimeType::of(runtime) == RuntimeType::of(wanted))
        .collect();
    let shown = if same_type.is_empty() {
        carried.iter().collect()
    } else {
        same_type
    };
    if shown.is_empty() {
        return "none".to_owned();
    }
    shown
        .into_iter()
        .map(plan_toml)
        .collect::<Vec<_>>()
        .join(" or ")
}

/// A `client_versions` list, as the log and a failure reason name it.
fn describe(versions: Option<&[String]>) -> String {
    match versions {
        None => "every version (the server predates client_versions)".to_owned(),
        Some([]) => "no version".to_owned(),
        Some(versions) => versions.join(", "),
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::process::Command;
    use std::sync::atomic::Ordering;

    use anyhow::Context;
    use httpmock::Method::{GET, PATCH, POST, PUT};
    use httpmock::{Mock, MockServer};
    use serde_json::json;

    use pipette_plan_types::Runtime;

    use super::*;
    use crate::app::{ConsoleRun, LAUNCHED};
    use crate::device::DevicectlFaultError;
    use crate::device::devicectl::DEVICE_NOT_FOUND_CODE;
    use crate::device::fake::{Call, CallLog, FakeDevice, GateLog};
    use crate::logging::testing::captured;
    use crate::pool::bundle::BundleError;
    use crate::releases::catalog::ADVERTISED_BUILDS;
    use crate::releases::sign::testing::{
        PASSWORD, adhoc_app, ipa, logged, security, signer, xcodebuild,
    };
    use crate::releases::testing::{
        B10216_BUILD_FLAG, FakeBucket, Published, PublishedBuild, b10216, mlx,
    };
    use crate::worker::job::run_claimed_job;
    use crate::worker::summary::testing::{field_matches, has_fields, summary_fields};
    use crate::worker::testing::{
        BUNDLE, CLIENT_ID, identity_of, knows_every_benchmark, llamacpp, phone, profile_body,
        session,
    };
    use crate::worker::{Tally, claim_loop, run_one};

    const V1: &str = "2026.10.01-0000000-internal";
    const V2: &str = "2026.10.02-e6a9434-internal";
    /// A version the bucket does not publish.
    const UNOFFERED: &str = "2025.01.01-0000000-internal";

    /// What `devicectl --console` prints after the app's own lines.
    const TERMINATED: &str = "The app terminated with the exit code 0.\n";
    /// devicectl 642.16's line for an app iOS killed, in place of [`TERMINATED`].
    const KILLED: &str = "App terminated due to signal 9.\n";

    /// A launch that exited 0, its stdout between devicectl's lines.
    fn ok(stdout: &str, stderr: &str) -> ConsoleRun {
        framed(ConsoleRun {
            stdout: stdout.to_owned(),
            stderr: stderr.to_owned(),
            sentinel: Some(0),
            exit_code: Some(0),
            timed_out: false,
            engine_time: None,
            aborted: false,
            stopped: false,
            silent: false,
        })
    }

    fn framed(run: ConsoleRun) -> ConsoleRun {
        ConsoleRun {
            stdout: format!("{LAUNCHED}{}{TERMINATED}", run.stdout),
            ..run
        }
    }

    fn identity(client: &str) -> ConsoleRun {
        framed(identity_of(client))
    }

    /// The `headlessrun version` line (`HeadlessRunner.runVersion`), its value
    /// bounded by ` private_thermal=`.
    fn version_line(version: &str) -> ConsoleRun {
        ok(
            &format!(
                "[HEADLESS] version client_version={version} private_thermal=on \
                 private_memory=on thermal=private SoC die temp (internal build) \
                 memory=private reclaim-ring flush + kernel footprint peak + 20 ms sampler \
                 (internal build)\n[HEADLESS] BENCH_DONE 0\n"
            ),
            "",
        )
    }

    /// `headlessrun runtimes list` for a build carrying `runtime`.
    fn runtimes_line(runtime: &Runtime) -> anyhow::Result<ConsoleRun> {
        Ok(ok(
            &format!(
                "[HEADLESS] runtimes count=1\n[HEADLESS] runtime type=llamacpp_ios_pipette \
                 digest=0a1b2c3d descriptor={} runtime=b10216:ios-arm64\n\
                 [HEADLESS] BENCH_DONE 0\n",
                serde_json::to_string(runtime)?
            ),
            "",
        ))
    }

    const LISTED: &str = "[HEADLESS] results count=1\n[HEADLESS] result result=local-1/cell-1 \
                          state=local benchmark=max_memory_usage_512 type=max_memory_usage\n\
                          [HEADLESS] BENCH_DONE 0\n";

    /// A phone found running `version` unrecorded, whose client check passes and whose
    /// listing names the runtimes that every fixture build provides.
    fn found_running(version: &str) -> anyhow::Result<FakeDevice> {
        Ok(FakeDevice::new("phone-a")
            .then_console(version_line(version))
            .then_console(identity(CLIENT_ID))
            .then_console(runtimes_line(&llamacpp("b10216")?)?))
    }

    /// The app's console for one cell, start to finish, its payload naming
    /// `client_version`.
    fn a_run(device: FakeDevice, client_version: &str) -> FakeDevice {
        let payload = json!({
            "benchmark_id": "max_memory_usage_512",
            "client_version": client_version,
            "observation_max_host_bytes": 461_373_440,
        });
        device
            .then_console(ok(
                "[HEADLESS] results count=0\n[HEADLESS] BENCH_DONE 0\n",
                "",
            ))
            .then_console(ok("[HEADLESS] BENCH_DONE 0 completed\n", ""))
            .then_console(ok(LISTED, ""))
            .then_console(ok(&format!("{payload}\n"), "[HEADLESS] BENCH_DONE 0\n"))
            .then_console(ok("[HEADLESS] BENCH_DONE 0\n", ""))
    }

    fn launch(args: &[&str]) -> Call {
        Call::Launch {
            bundle_id: BUNDLE.to_owned(),
            app_args: args.iter().map(|arg| (*arg).to_owned()).collect(),
        }
    }

    fn run_calls(claim: &ClaimedJob) -> anyhow::Result<Vec<Call>> {
        Ok(vec![
            launch(&["headlessrun", "results", "list"]),
            Call::Launch {
                bundle_id: BUNDLE.to_owned(),
                app_args: app::launch_plan_from_claim(claim)?.app_args,
            },
            launch(&["headlessrun", "results", "list"]),
            launch(&["headlessrun", "results", "show", "result=local-1/cell-1"]),
            launch(&["headlessrun", "results", "delete", "result=local-1/cell-1"]),
        ])
    }

    fn claim(client_versions: Option<&[&str]>) -> anyhow::Result<ClaimedJob> {
        claim_wanting(client_versions, "b10216")
    }

    fn claim_wanting(
        client_versions: Option<&[&str]>,
        runtime: &str,
    ) -> anyhow::Result<ClaimedJob> {
        Ok(ClaimedJob {
            job_id: "job-1".to_owned(),
            benchmark_id: "max_memory_usage_512".to_owned(),
            time_window: "PT10M".to_owned(),
            expires_at: None,
            client_versions: client_versions
                .map(|versions| versions.iter().map(|v| (*v).to_owned()).collect()),
            spec: json!({
                "benchmark": "max_memory_usage_512",
                "model": {
                    "type": "gguf_text",
                    "source": "huggingface",
                    "org": "LiquidAI",
                    "repo_name": "LFM2.5-350M-GGUF",
                    "path": "LFM2.5-350M-Q4_0.gguf",
                },
                "runtime": serde_json::to_value(llamacpp(runtime)?)?,
            }),
        })
    }

    /// A release archive of `version` as the release job zips one, its app
    /// ad-hoc signed with the memory entitlement when `memory`.
    fn release_zip(version: &str, memory: bool) -> anyhow::Result<Vec<u8>> {
        let stage = tempfile::tempdir()?;
        let folder = stage.path().join(format!("pipette-ios-{version}"));
        adhoc_app(
            &folder.join("Pipette.xcarchive/Products/Applications/Pipette.app"),
            BUNDLE,
            version.trim_end_matches("-internal"),
            memory,
        )?;
        let zip = stage.path().join("archive.zip");
        let packed = Command::new("ditto")
            .args(["-c", "-k", "--keepParent"])
            .arg(&folder)
            .arg(&zip)
            .output()?;
        anyhow::ensure!(packed.status.success(), "ditto: {packed:?}");
        Ok(std::fs::read(zip)?)
    }

    /// A phone in version mode, the bucket publishing [`V1`] then [`V2`], and
    /// stand-ins for `security` and `xcodebuild`.
    struct Fixture {
        dir: tempfile::TempDir,
        bucket: FakeBucket,
        server: MockServer,
        published: Vec<Published>,
    }

    struct FakePhone {
        phone: Phone,
        log: CallLog,
        gated: GateLog,
    }

    impl FakePhone {
        /// The calls the phone saw, an install named relative to the staged
        /// directory, as [`installed`] names it.
        fn calls(&self) -> anyhow::Result<Vec<Call>> {
            let staged = &self.phone.paths.staged;
            Ok(self
                .log
                .calls()?
                .into_iter()
                .map(|mut call| {
                    if let Call::Install(path) = &mut call
                        && let Ok(relative) = path.strip_prefix(staged)
                    {
                        *path = relative.to_path_buf();
                    }
                    call
                })
                .collect())
        }
    }

    /// The install of the staged build, as [`FakePhone::calls`] names it.
    fn installed() -> Call {
        Call::Install(PathBuf::from("Pipette.app"))
    }

    impl Fixture {
        /// `memory[i]` is whether the i-th version's archive carries the
        /// memory entitlement. Both manifests name llama.cpp `b10216`.
        fn new(memory: [bool; 2]) -> anyhow::Result<Self> {
            Self::carrying(memory, &[b10216()])
        }

        /// As [`Fixture::new`], both manifests naming `runtimes`.
        fn carrying(memory: [bool; 2], runtimes: &[serde_json::Value]) -> anyhow::Result<Self> {
            let zips = [release_zip(V1, memory[0])?, release_zip(V2, memory[1])?];
            let published = [V1, V2]
                .iter()
                .zip(&zips)
                .zip([1, 2])
                .flat_map(|((version, zip), day)| {
                    Published::build_carrying(
                        &format!("{version}/pipette-ios-{version}.xcarchive.zip"),
                        zip,
                        day,
                        runtimes,
                    )
                    .objects()
                })
                .collect();
            let fixture = Self {
                dir: tempfile::tempdir()?,
                bucket: FakeBucket::start(),
                server: MockServer::start(),
                published,
            };
            fixture.bucket.publish(&fixture.published, 1000);
            Ok(fixture)
        }

        fn state(&self) -> PathBuf {
            self.dir.path().join("state")
        }

        /// The phone, with no hardware UDID and an iOS below the archives'
        /// `MinimumOSVersion` (18.0): neither is the proxy's to check.
        fn phone(&self, device: FakeDevice) -> anyhow::Result<FakePhone> {
            let log = device.log();
            let gated = device.gated();
            let (mut phone, _pool) = phone(device, &self.state())?;
            phone.details.os_version = Some("17.6".to_owned());
            Ok(FakePhone { phone, log, gated })
        }

        /// This phone's side of the bucket, never listed; the export
        /// stand-in runs `before` first.
        fn unlisted(
            &self,
            default: Option<&str>,
            unlock_status: i32,
            before: &str,
        ) -> anyhow::Result<Releases> {
            let tools = self.dir.path().join("tools");
            std::fs::create_dir_all(&tools)?;
            let ipa = ipa(&tools)?;
            let signer = signer(
                &tools,
                security(&tools, unlock_status, 1)?,
                xcodebuild(&tools, &ipa, before)?,
            );
            Ok(Releases::new(
                self.bucket.config()?,
                s3::http_client()?,
                StatePaths::resolve(Some(&self.state()))?.releases(),
                Arc::new(signer),
                default.map(str::parse).transpose()?,
            ))
        }

        fn releases(
            &self,
            default: Option<&str>,
            unlock_status: i32,
            before: &str,
        ) -> anyhow::Result<Releases> {
            let mut releases = self.unlisted(default, unlock_status, before)?;
            releases
                .catalog
                .relist(&releases.http, &releases.bucket, &releases.paths)?;
            Ok(releases)
        }

        fn exports(&self) -> anyhow::Result<usize> {
            Ok(logged(&self.dir.path().join("tools"), "xcodebuild.argv")?
                .lines()
                .count())
        }

        fn submitted(&self, includes: serde_json::Value) -> Mock<'_> {
            self.server.mock(|when, then| {
                when.method(POST)
                    .path("/benchmarks")
                    .json_body_includes(includes.to_string());
                then.status(202).json_body(json!({ "job_id": "job-1" }));
            })
        }

        /// A retriable failure report, as [`Fixture::failure`] matches one.
        fn failed(&self, client_version: Option<&str>, reasons: &[&str]) -> Mock<'_> {
            self.failure(true, client_version, reasons)
        }

        /// A failure report filed under `client_version` (or a prefix of it;
        /// any version when `None`), its `failure_reason` naming every one of
        /// `reasons`.
        fn failure(
            &self,
            retriable: bool,
            client_version: Option<&str>,
            reasons: &[&str],
        ) -> Mock<'_> {
            let filed = client_version.map(|version| format!(r#""client_version":"{version}"#));
            let reasons: Vec<String> = reasons.iter().map(|&reason| reason.to_owned()).collect();
            self.server.mock(|when, then| {
                filed
                    .iter()
                    .fold(
                        when.method(POST)
                            .path("/benchmarks")
                            .json_body_includes(json!({ "retriable": retriable }).to_string()),
                        |when, needle| when.body_includes(needle),
                    )
                    .is_true(move |request| {
                        let body: serde_json::Value =
                            serde_json::from_str(&request.body_string()).unwrap_or_default();
                        let reason = body
                            .get("failure_reason")
                            .and_then(serde_json::Value::as_str)
                            .unwrap_or_default();
                        reasons
                            .iter()
                            .all(|needle| reason.contains(needle.as_str()))
                    });
                then.status(202).json_body(json!({ "job_id": "job-1" }));
            })
        }

        /// `POST /plans/claim` answering `claim` every time.
        fn claims(&self, claim: &ClaimedJob) -> Mock<'_> {
            let body = json!({
                "job_id": claim.job_id,
                "benchmark_id": claim.benchmark_id,
                "time_window": claim.time_window,
                "client_versions": claim.client_versions,
                "spec": claim.spec,
            });
            self.server.mock(|when, then| {
                when.method(POST).path("/plans/claim");
                then.status(200).json_body(body);
            })
        }

        /// `PATCH /clients/me` with a body holding `includes`.
        fn reported(&self, includes: &str) -> Mock<'_> {
            self.server.mock(|when, then| {
                when.method(PATCH)
                    .path("/clients/me")
                    .body_includes(includes);
                then.status(200).json_body(profile_body("approved", false));
            })
        }

        /// Mounted after a test's own mocks, so one of them can answer the benchmark lookup.
        fn session(&self, phone: &Phone) -> anyhow::Result<Arc<Session>> {
            knows_every_benchmark(&self.server);
            Ok(Arc::new(session(phone, &self.server.base_url())?))
        }
    }

    fn version_mode(releases: Releases) -> Builds {
        Builds::Releases(Box::new(releases))
    }

    fn releases_of(builds: &Builds) -> anyhow::Result<&Releases> {
        match builds {
            Builds::Releases(releases) => Ok(releases),
            Builds::Pool(_) => anyhow::bail!("the builds changed mode"),
        }
    }

    fn with_max_jobs(phone: &mut Phone, max_jobs: u32) {
        let mut config = (*phone.config).clone();
        config.loop_config.max_jobs = max_jobs;
        phone.config = Arc::new(config);
    }

    fn with_heartbeat(phone: &mut Phone, every: Duration) {
        let mut config = (*phone.config).clone();
        config.loop_config.heartbeat = every;
        phone.config = Arc::new(config);
    }

    /// A claim runs on the version it admits: pulled, signed, gated, installed
    /// and confirmed when the phone runs another, left alone when it runs that
    /// one. The phone proves its client either way, since nothing recorded
    /// the version, and the result goes up as that version.
    #[rstest::rstest]
    #[case::pinned_to_another(
        V1,
        Some(V1),
        Some(&[V2][..]),
        vec![
            launch(&["headlessrun", "version"]),
            installed(),
            launch(&["headlessrun", "auth", "me", "local=1"]),
            launch(&["headlessrun", "version"]),
            launch(&["headlessrun", "runtimes", "list"]),
        ]
    )]
    #[case::already_on_the_newest(
        V2,
        None,
        None,
        vec![
            launch(&["headlessrun", "version"]),
            launch(&["headlessrun", "auth", "me", "local=1"]),
            launch(&["headlessrun", "runtimes", "list"]),
        ]
    )]
    fn a_claim_runs_on_the_version_it_admits(
        #[case] running: &str,
        #[case] default: Option<&str>,
        #[case] admitted: Option<&[&str]>,
        #[case] before_the_run: Vec<Call>,
    ) -> anyhow::Result<()> {
        let installs = before_the_run.contains(&installed());
        let fixture = Fixture::new([true, true])?;
        let device = FakeDevice::new("phone-a")
            .then_console(version_line(running))
            .then_console(identity(CLIENT_ID));
        let device = if installs {
            device.then_console(version_line(V2))
        } else {
            device
        };
        // After an install, or before trusting a build found running unrecorded.
        let device = a_run(
            device.then_console(runtimes_line(&llamacpp("b10216")?)?),
            V2,
        );
        let mut fake = fixture.phone(device)?;
        let mut builds = version_mode(fixture.releases(default, 0, "")?);
        let submit = fixture.submitted(json!({ "job_id": "job-1", "client_version": V2 }));
        let ctx = fixture.session(&fake.phone)?;
        let claim = claim(admitted)?;

        run_claimed_job(&mut fake.phone, &mut builds, &ctx, &claim)?;

        submit.assert();
        let expected: Vec<Call> = before_the_run
            .into_iter()
            .chain(run_calls(&claim)?)
            .collect();
        assert_eq!(fake.calls()?, expected);
        assert_eq!(fixture.exports()?, usize::from(installs));
        assert_eq!(fake.gated.builds()?.len(), usize::from(installs));
        let record = InstalledBuild::load(&fake.phone.paths.installed)?;
        assert_eq!(record.version.as_deref(), Some(V2));
        assert_eq!(
            record.path,
            installs.then(|| fake.phone.paths.staged.join("Pipette.app"))
        );
        Ok(())
    }

    /// Only a runtime no listed build carries fails the job for good: the
    /// listing taken again says so, and the claim names the builds. Without
    /// `client_versions` the admitted builds are not known.
    #[rstest::rstest]
    #[case::none_admitted(Some(&[][..]), "b10216", true, "the claim admits no version")]
    #[case::only_unknown_versions(Some(&["9.9"][..]), "b10216", true, "the claim admits 9.9")]
    #[case::runtime_in_no_build(
        Some(&[V2][..]),
        "b1",
        false,
        "is in no build in the releases bucket, so no phone can run it. The builds the claim \
         admits (2026.10.02-e6a9434-internal) carry"
    )]
    #[case::older_server(
        None,
        "b1",
        true,
        "admits here (2026.10.02-e6a9434-internal, 2026.10.01-0000000-internal)"
    )]
    fn a_claim_that_cannot_run_here_touches_no_phone(
        #[case] admitted: Option<&[&str]>,
        #[case] runtime: &str,
        #[case] retriable: bool,
        #[case] reason: &str,
    ) -> anyhow::Result<()> {
        let fixture = Fixture::new([true, true])?;
        let mut fake = fixture.phone(FakeDevice::new("phone-a"))?;
        let releases = fixture.releases(None, 0, "")?;
        let failed = fixture.failure(retriable, None, &[reason]);
        let ctx = fixture.session(&fake.phone)?;

        run_claimed_job(
            &mut fake.phone,
            &mut version_mode(releases),
            &ctx,
            &claim_wanting(admitted, runtime)?,
        )?;

        failed.assert();
        assert_eq!(fake.calls()?, vec![]);
        Ok(())
    }

    /// A benchmark the server does not know is terminal in version mode too, and no
    /// build is chosen for it.
    #[test]
    fn an_unknown_benchmark_touches_no_phone() -> anyhow::Result<()> {
        let fixture = Fixture::new([true, true])?;
        let mut fake = fixture.phone(FakeDevice::new("phone-a"))?;
        let releases = fixture.releases(None, 0, "")?;
        let lookup = fixture.server.mock(|when, then| {
            when.method(GET).path("/benchmarks/max_memory_usage_512");
            then.status(404);
        });
        let failed = fixture.failure(false, None, &["benchmark not found"]);
        let ctx = fixture.session(&fake.phone)?;

        run_claimed_job(
            &mut fake.phone,
            &mut version_mode(releases),
            &ctx,
            &claim(Some(&[V2]))?,
        )?;

        lookup.assert();
        failed.assert();
        assert_eq!(fake.calls()?, vec![]);
        Ok(())
    }

    /// The documented trap: a plan's runtime without `private_thermal`. The
    /// reason offers, in plan TOML, what the builds carry of the job's type.
    #[test]
    fn a_runtime_in_no_build_names_what_the_builds_carry() -> anyhow::Result<()> {
        let mut gated = b10216();
        gated["private_thermal"] = true.into();
        let fixture = Fixture::carrying([true, true], &[gated, mlx()])?;
        let mut fake = fixture.phone(FakeDevice::new("phone-a"))?;
        let mut builds = version_mode(fixture.releases(None, 0, "")?);
        let failed = fixture.failure(
            false,
            None,
            &[&format!(
                "] the job's runtime {{ type = \"llamacpp_ios_pipette\", flavor = \"ios-arm64\", \
                 repository_url = \"github.com/ggml-org/llama.cpp\", repository_version = \
                 \"b10216\" }} is in no build in the releases bucket, so no phone can run it. \
                 The builds the claim admits ({V2}, {V1}) carry, in plan TOML: {{ type = \
                 \"llamacpp_ios_pipette\", \
                 flavor = \"ios-arm64\", private_thermal = true, repository_url = \
                 \"github.com/ggml-org/llama.cpp\", repository_version = \"b10216\" }}"
            )],
        );
        let ctx = fixture.session(&fake.phone)?;

        run_claimed_job(&mut fake.phone, &mut builds, &ctx, &claim(Some(&[V2, V1]))?)?;

        failed.assert();
        assert_eq!(fake.calls()?, vec![]);
        assert_eq!(releases_of(&builds)?.backoff, None);
        Ok(())
    }

    /// A listing that cannot be taken again proves nothing: the job stays
    /// retriable, for a phone whose listing is fresher.
    #[test]
    fn a_runtime_not_carried_on_a_stale_listing_stays_retriable() -> anyhow::Result<()> {
        let fixture = Fixture::new([true, true])?;
        let mut fake = fixture.phone(FakeDevice::new("phone-a"))?;
        let releases = fixture.releases(None, 0, "")?;
        fixture.bucket.server.reset();
        let failed = fixture.failed(
            None,
            &[&format!(
                "is in the manifest of no build the claim admits here ({V2})"
            )],
        );
        let ctx = fixture.session(&fake.phone)?;

        run_claimed_job(
            &mut fake.phone,
            &mut version_mode(releases),
            &ctx,
            &claim_wanting(Some(&[V2]), "b1")?,
        )?;

        failed.assert();
        assert_eq!(fake.calls()?, vec![]);
        Ok(())
    }

    /// A manifest corrected since the last listing is seen before the job is
    /// failed: the listing taken again finds the build, and the job runs.
    #[test]
    fn a_fresh_listing_finds_a_build_the_stale_one_missed() -> anyhow::Result<()> {
        let fixture = Fixture::new([true, true])?;
        let device = a_run(
            FakeDevice::new("phone-a")
                .then_console(version_line(V2))
                .then_console(identity(CLIENT_ID)),
            V2,
        );
        let mut fake = fixture.phone(device)?;
        let mut builds = version_mode(fixture.releases(None, 0, "")?);
        let corrected = naming(&fixture.published, &[V1, V2], "b1")?;
        fixture.bucket.server.reset();
        fixture.bucket.publish(&corrected, 1000);
        let submit = fixture.submitted(json!({ "job_id": "job-1", "client_version": V2 }));
        let ctx = fixture.session(&fake.phone)?;
        let claim = claim_wanting(Some(&[V2]), "b1")?;

        run_claimed_job(&mut fake.phone, &mut builds, &ctx, &claim)?;

        submit.assert();
        Ok(())
    }

    /// `client_versions` names only the versions this phone advertised: a
    /// build uploaded since, carrying the runtime, may be one the job admits
    /// on a phone that advertises it, so the job stays retriable.
    #[test]
    fn a_build_uploaded_since_the_claim_keeps_the_job_retriable() -> anyhow::Result<()> {
        const V3: &str = "2026.10.03-3333333-internal";
        let fixture = Fixture::new([true, true])?;
        let mut fake = fixture.phone(FakeDevice::new("phone-a"))?;
        let releases = fixture.releases(None, 0, "")?;
        let mut published = fixture.published.clone();
        published.extend(
            Published::build_carrying(
                &format!("{V3}/pipette-ios-{V3}.xcarchive.zip"),
                V3.as_bytes(),
                3,
                &[serde_json::to_value(llamacpp("b1")?)?],
            )
            .objects(),
        );
        fixture.bucket.server.reset();
        fixture.bucket.publish(&published, 1000);
        let failed = fixture.failed(
            None,
            &[&format!(
                "is in the manifest of no build the claim admits here ({V2}, {V1})"
            )],
        );
        let ctx = fixture.session(&fake.phone)?;

        run_claimed_job(
            &mut fake.phone,
            &mut version_mode(releases),
            &ctx,
            &claim_wanting(Some(&[V2, V1]), "b1")?,
        )?;

        failed.assert();
        assert_eq!(fake.calls()?, vec![]);
        Ok(())
    }

    /// A build this phone withdrew is never in its `client_versions`, yet
    /// another phone may run the job on it: the job stays retriable.
    #[test]
    fn a_carrier_withdrawn_here_keeps_the_job_retriable() -> anyhow::Result<()> {
        let fixture = Fixture::new([true, true])?;
        let published = naming(&fixture.published, &[V2], "b1")?;
        fixture.bucket.server.reset();
        fixture.bucket.publish(&published, 1000);
        let mut fake = fixture.phone(FakeDevice::new("phone-a"))?;
        let mut releases = fixture.releases(None, 0, "")?;
        releases.catalog.withdraw(&V2.parse()?);
        let failed = fixture.failed(
            None,
            &[&format!(
                "is in the manifest of no build the claim admits here ({V1})"
            )],
        );
        let ctx = fixture.session(&fake.phone)?;

        run_claimed_job(
            &mut fake.phone,
            &mut version_mode(releases),
            &ctx,
            &claim_wanting(Some(&[V1]), "b1")?,
        )?;

        failed.assert();
        assert_eq!(fake.calls()?, vec![]);
        Ok(())
    }

    /// Another proxy may offer a build this phone does not advertise, so the
    /// bucket is read whole first. `V1` sits past twenty newer uploads, and the
    /// claim admits the default, `V2`.
    #[rstest::rstest]
    #[case::an_older_build_carries_it(
        true,
        true,
        "is in the manifest of no build the claim admits here (2026.10.02-e6a9434-internal)"
    )]
    #[case::none_carries_it(false, false, "is in no build in the releases bucket")]
    fn a_job_fails_for_good_only_on_the_whole_bucket(
        #[case] older_carries: bool,
        #[case] retriable: bool,
        #[case] reason: &str,
    ) -> anyhow::Result<()> {
        let fixture = Fixture::new([true, true])?;
        let mut published = if older_carries {
            naming(&fixture.published, &[V1], "b1")?
        } else {
            fixture.published.clone()
        };
        published.extend((3u8..).take(ADVERTISED_BUILDS).flat_map(|day| {
            let version = format!("2026.10.{day:02}-{day:07}-internal");
            Published::build(
                &format!("{version}/pipette-ios-{version}.xcarchive.zip"),
                version.as_bytes(),
                day,
            )
            .objects()
        }));
        fixture.bucket.server.reset();
        fixture.bucket.publish(&published, 1000);
        let mut fake = fixture.phone(FakeDevice::new("phone-a"))?;
        let releases = fixture.releases(Some(V2), 0, "")?;
        let failed = fixture.failure(retriable, None, &[reason]);
        let ctx = fixture.session(&fake.phone)?;

        run_claimed_job(
            &mut fake.phone,
            &mut version_mode(releases),
            &ctx,
            &claim_wanting(Some(&[V2]), "b1")?,
        )?;

        failed.assert();
        assert_eq!(fake.calls()?, vec![]);
        Ok(())
    }

    /// A build in the bucket that the listing could not read.
    #[derive(Debug, Clone, Copy)]
    enum Unread {
        /// Its manifest does not parse.
        Unparseable,
        /// Its archive alone, the upload not finished.
        Uploading,
    }

    /// A build the listing could not read may carry the runtime, so the job
    /// stays retriable.
    #[rstest::rstest]
    #[case::unparseable_manifest(Unread::Unparseable)]
    #[case::still_uploading(Unread::Uploading)]
    fn a_build_that_cannot_be_read_keeps_the_job_retriable(
        #[case] unread: Unread,
    ) -> anyhow::Result<()> {
        const V3: &str = "2026.10.03-3333333-internal";
        let fixture = Fixture::new([true, true])?;
        let PublishedBuild {
            archive,
            sidecar,
            mut manifest,
        } = Published::build(
            &format!("{V3}/pipette-ios-{V3}.xcarchive.zip"),
            V3.as_bytes(),
            3,
        );
        let mut published = fixture.published.clone();
        published.push(archive);
        if let Unread::Unparseable = unread {
            manifest.body = b"{".to_vec();
            published.extend([sidecar, manifest]);
        }
        fixture.bucket.server.reset();
        fixture.bucket.publish(&published, 1000);
        let mut fake = fixture.phone(FakeDevice::new("phone-a"))?;
        let releases = fixture.releases(None, 0, "")?;
        let failed = fixture.failed(
            None,
            &[&format!(
                "is in the manifest of no build the claim admits here ({V2}, {V1})"
            )],
        );
        let ctx = fixture.session(&fake.phone)?;

        run_claimed_job(
            &mut fake.phone,
            &mut version_mode(releases),
            &ctx,
            &claim_wanting(Some(&[V2, V1]), "b1")?,
        )?;

        failed.assert();
        assert_eq!(fake.calls()?, vec![]);
        Ok(())
    }

    fn apple_foundation() -> anyhow::Result<Runtime> {
        Ok(serde_json::from_value(
            json!({ "type": "apple_foundation" }),
        )?)
    }

    /// What a reason offers in place of llama.cpp `b1`: the runtimes of its
    /// type when the builds carry any, else every one, else none.
    #[rstest::rstest]
    #[case::its_type(
        vec![llamacpp("b2")?, apple_foundation()?, llamacpp("b3")?],
        concat!(
            r#"{ type = "llamacpp_ios_pipette", flavor = "ios-arm64", "#,
            r#"repository_url = "github.com/ggml-org/llama.cpp", repository_version = "b2" } or "#,
            r#"{ type = "llamacpp_ios_pipette", flavor = "ios-arm64", "#,
            r#"repository_url = "github.com/ggml-org/llama.cpp", repository_version = "b3" }"#,
        )
    )]
    #[case::only_other_types(
        vec![apple_foundation()?],
        r#"{ type = "apple_foundation" }"#
    )]
    #[case::none(vec![], "none")]
    fn alternatives_prefer_the_job_type(
        #[case] carried: Vec<Runtime>,
        #[case] expected: &str,
    ) -> anyhow::Result<()> {
        assert_eq!(alternatives(&carried, &llamacpp("b1")?), expected);
        Ok(())
    }

    /// `published`, with the manifests of `versions` naming llama.cpp
    /// `runtime` alone.
    fn naming(
        published: &[Published],
        versions: &[&str],
        runtime: &str,
    ) -> anyhow::Result<Vec<Published>> {
        let named = serde_json::to_value(llamacpp(runtime)?)?;
        Ok(published
            .iter()
            .map(|object| {
                let manifest =
                    |version: &&str| object.key == format!("{version}/pipette-ios-{version}.json");
                match versions.iter().copied().find(manifest) {
                    Some(version) => Published::new(
                        &object.key,
                        json!({ "client_version": version, "runtimes": [named] })
                            .to_string()
                            .as_bytes(),
                        object.day,
                    ),
                    None => object.clone(),
                }
            })
            .collect())
    }

    #[test]
    fn a_gone_phone_is_reported_before_anything_is_pulled() -> anyhow::Result<()> {
        let fixture = Fixture::new([true, true])?;
        let mut fake = fixture.phone(FakeDevice::new("phone-a").gone("CoreDeviceError 1011"))?;
        let mut builds = version_mode(fixture.releases(None, 0, "")?);
        let failed = fixture.failed(None, &["the phone is unreachable: CoreDeviceError 1011"]);
        let ctx = fixture.session(&fake.phone)?;

        run_claimed_job(&mut fake.phone, &mut builds, &ctx, &claim(Some(&[V2]))?)?;

        failed.assert();
        assert_eq!(fake.calls()?, vec![launch(&["headlessrun", "version"])]);
        let releases = releases_of(&builds)?;
        assert!(!releases.paths.records.exists(), "nothing was pulled");
        assert_eq!(fixture.exports()?, 0);
        assert_eq!(releases.catalog.version_flags().len(), 2);
        Ok(())
    }

    /// What a build that cannot be made ready leaves behind, past the
    /// retriable report.
    #[derive(Debug, PartialEq)]
    struct Unprepared {
        advertised: usize,
        exports: usize,
        backs_off: bool,
    }

    /// A build that cannot be made ready is reported retriable, and who can
    /// fix it decides the rest: an operator's fault halts the phone, the
    /// version's withdraws it, and a fault that may clear backs the next claim
    /// off.
    #[rstest::rstest]
    #[case::archive_without_the_memory_limit(
        false,
        0,
        None,
        None,
        Ok(()),
        Unprepared { advertised: 1, exports: 0, backs_off: false }
    )]
    #[case::keychain_that_does_not_unlock(
        true,
        51,
        None,
        None,
        Err("with `security -i` failed"),
        Unprepared { advertised: 2, exports: 0, backs_off: true }
    )]
    #[case::fresh_export_of_another_team(
        true,
        0,
        Some(BundleError::WrongTeam { path: "a".into(), place: "profile", found: "ABCDE12345".into(), expected: "T".into() }),
        None,
        Err("carries team ABCDE12345 in its profile"),
        Unprepared { advertised: 2, exports: 1, backs_off: true }
    )]
    #[case::fresh_export_of_another_bundle(
        true,
        0,
        Some(BundleError::WrongBundleId { path: "a".into(), found: "ai.liquid.other".into(), expected: BUNDLE.into() }),
        None,
        Ok(()),
        Unprepared { advertised: 1, exports: 1, backs_off: false }
    )]
    #[case::install_refused_by_the_phone(
        true,
        0,
        None,
        Some("devicectl: there is not enough storage on the device"),
        Ok(()),
        Unprepared { advertised: 2, exports: 1, backs_off: true }
    )]
    fn a_build_that_cannot_be_made_ready(
        #[case] memory_limit: bool,
        #[case] unlock_status: i32,
        #[case] gate_refusal: Option<BundleError>,
        #[case] install_failure: Option<&str>,
        #[case] halt: Result<(), &str>,
        #[case] expected: Unprepared,
    ) -> anyhow::Result<()> {
        let fixture = Fixture::new([true, memory_limit])?;
        let device = FakeDevice::new("phone-a").then_console(version_line(V1));
        let device = match gate_refusal {
            Some(refusal) => device.then_gate_refuses(refusal),
            None => device,
        };
        let device = match install_failure {
            Some(message) => device.failing_install(message),
            None => device,
        };
        let mut fake = fixture.phone(device)?;
        let mut builds = version_mode(fixture.releases(None, unlock_status, "")?);
        let failed = fixture.failed(None, &[]);
        let ctx = fixture.session(&fake.phone)?;

        let outcome = run_claimed_job(&mut fake.phone, &mut builds, &ctx, &claim(Some(&[V2]))?);

        failed.assert();
        match (outcome, halt) {
            (Ok(()), Ok(())) => {}
            (Err(error), Err(needle)) => {
                assert!(needs_operator(&error), "{error:#}");
                assert!(format!("{error:#}").contains(needle), "{error:#}");
                assert!(!format!("{error:#}").contains(PASSWORD));
            }
            (got, expected) => anyhow::bail!("got {got:?}, expected {expected:?}"),
        }
        let calls = std::iter::once(launch(&["headlessrun", "version"]))
            .chain(install_failure.map(|_| installed()))
            .collect::<Vec<_>>();
        assert_eq!(fake.calls()?, calls);
        let releases = releases_of(&builds)?;
        assert_eq!(
            Unprepared {
                advertised: releases.catalog.version_flags().len(),
                exports: fixture.exports()?,
                backs_off: releases.backoff == Some(fake.phone.config.loop_config.claim_backoff),
            },
            expected
        );
        Ok(())
    }

    #[rstest::rstest]
    #[case::already_running_it(V2, V2)]
    #[case::just_installed(V1, "pipette-ios-proxy/")]
    fn a_failure_names_the_version_the_phone_reported(
        #[case] running: &str,
        #[case] client_version: &str,
    ) -> anyhow::Result<()> {
        let fixture = Fixture::new([true, true])?;
        let mut fake = fixture.phone(
            FakeDevice::new("phone-a")
                .then_console(version_line(running))
                .then_console(identity("ev1_someone_else")),
        )?;
        let mut builds = version_mode(fixture.releases(None, 0, "")?);
        let failed = fixture.failed(Some(client_version), &[]);
        let ctx = fixture.session(&fake.phone)?;

        let error = run_claimed_job(&mut fake.phone, &mut builds, &ctx, &claim(Some(&[V2]))?)
            .err()
            .context("a replaced identity halts the phone")?;

        failed.assert();
        assert!(install::replaced_identity(&error), "{error:#}");
        Ok(())
    }

    #[test]
    fn the_backoff_doubles_to_its_cap_and_resets() {
        let base = Duration::from_secs(5);
        let endings = [
            Some(FailureCause::BuildNotReady),
            Some(FailureCause::BuildNotReady),
            Some(FailureCause::DevicectlFailure),
            Some(FailureCause::BuildNotReady),
            Some(FailureCause::BuildNotReady),
            Some(FailureCause::BuildNotReady),
            Some(FailureCause::BuildNotReady),
            Some(FailureCause::BuildNotReady),
            Some(FailureCause::BuildNotReady),
            Some(FailureCause::BuildNotReady),
            Some(FailureCause::BuildUnusable),
            Some(FailureCause::DevicectlFailure),
            None,
        ];
        let waits: Vec<Option<u64>> = endings
            .iter()
            .scan(None, |previous, ended| {
                *previous = backoff_after(*previous, *ended, base);
                Some(previous.map(|wait| wait.as_secs()))
            })
            .collect();
        assert_eq!(
            waits,
            [
                Some(5),
                Some(10),
                Some(20),
                Some(40),
                Some(80),
                Some(160),
                Some(320),
                Some(640),
                Some(900),
                Some(900),
                None,
                Some(5),
                None,
            ]
        );
    }

    #[test]
    fn a_result_from_another_build_is_not_submitted() -> anyhow::Result<()> {
        let fixture = Fixture::new([true, true])?;
        let device = a_run(
            FakeDevice::new("phone-a")
                .then_console(version_line(V2))
                .then_console(identity(CLIENT_ID)),
            V1,
        );
        let mut fake = fixture.phone(device)?;
        let mut builds = version_mode(fixture.releases(None, 0, "")?);
        let accepted = fixture.submitted(json!({ "observation_max_host_bytes": 461_373_440 }));
        let failed = fixture.failed(Some(V2), &[]);
        let ctx = fixture.session(&fake.phone)?;

        run_claimed_job(&mut fake.phone, &mut builds, &ctx, &claim(Some(&[V2]))?)?;

        accepted.assert_calls(0);
        failed.assert();
        Ok(())
    }

    /// What `runtimes list` prints when a llama.cpp build the app carries does not load.
    const LISTING_FAILS: &str = "[HEADLESS] runtimes list ERROR llama.cpp build llamacpp-asr \
                                 does not load: image not found\n[HEADLESS] BENCH_DONE 1\n";

    /// A build whose own listing is not what its manifest names, or fails because a
    /// llama.cpp build it carries does not load here, was published wrong: the job is
    /// retriable under that version, which this phone stops offering, and nothing runs.
    /// So is one found running unrecorded, whose check after its install never passed.
    #[rstest::rstest]
    #[case::other_runtimes_than_its_manifest(
        V1,
        None,
        &["reports the runtimes", "but its manifest names"],
    )]
    #[case::listing_fails(V1, Some(LISTING_FAILS), &["runtimes list` failed on the phone"])]
    #[case::found_running_listing_fails(
        V2,
        Some(LISTING_FAILS),
        &["runtimes list` failed on the phone"],
    )]
    fn a_build_whose_listing_is_wrong_is_withdrawn(
        #[case] running: &str,
        #[case] failure: Option<&str>,
        #[case] needles: &[&str],
    ) -> anyhow::Result<()> {
        let listing = match failure {
            None => runtimes_line(&llamacpp("b10100")?)?,
            Some(stdout) => ConsoleRun {
                sentinel: Some(1),
                exit_code: Some(1),
                ..ok(stdout, "")
            },
        };
        let installs = running != V2;
        let fixture = Fixture::new([true, true])?;
        let device = FakeDevice::new("phone-a")
            .then_console(version_line(running))
            .then_console(identity(CLIENT_ID));
        let device = if installs {
            device.then_console(version_line(V2))
        } else {
            device
        };
        let mut fake = fixture.phone(device.then_console(listing))?;
        let mut builds = version_mode(fixture.releases(None, 0, "")?);
        let failed = fixture.failed(Some(V2), needles);
        let ctx = fixture.session(&fake.phone)?;

        run_claimed_job(&mut fake.phone, &mut builds, &ctx, &claim(Some(&[V2]))?)?;

        failed.assert();
        let expected: Vec<Call> = [launch(&["headlessrun", "version"])]
            .into_iter()
            .chain(installs.then(installed))
            .chain([launch(&["headlessrun", "auth", "me", "local=1"])])
            .chain(installs.then(|| launch(&["headlessrun", "version"])))
            .chain([launch(&["headlessrun", "runtimes", "list"])])
            .collect();
        assert_eq!(fake.calls()?, expected);
        assert_eq!(
            releases_of(&builds)?.catalog.version_flags(),
            [format!("client_version:{V1}")]
        );
        assert_eq!(
            InstalledBuild::load(&fake.phone.paths.installed)?.version,
            None
        );
        Ok(())
    }

    /// The build refusing the job's runtime is retriable here: another version
    /// may carry it. The refusal is the app's, between devicectl's lines, and is
    /// one more status 2.
    #[test]
    fn a_runtime_the_build_refuses_is_retriable() -> anyhow::Result<()> {
        let fixture = Fixture::new([true, true])?;
        let claim = claim(Some(&[V2]))?;
        let tokens = app::launch_plan_from_claim(&claim)?.app_args.join(" ");
        let refused = framed(ConsoleRun {
            stdout: format!(
                "[HEADLESS] ERROR `runtime=` rejected: names {}, but this build is {}: \
                 {tokens}\n[HEADLESS] BENCH_DONE 2\n",
                serde_json::to_string(&llamacpp("b10216")?)?,
                serde_json::to_string(&llamacpp("b10100")?)?,
            ),
            stderr: String::new(),
            sentinel: Some(2),
            exit_code: Some(2),
            timed_out: false,
            engine_time: None,
            aborted: false,
            stopped: false,
            silent: false,
        });
        let device = found_running(V2)?
            .then_console(ok(
                "[HEADLESS] results count=0\n[HEADLESS] BENCH_DONE 0\n",
                "",
            ))
            .then_console(refused);
        let mut fake = fixture.phone(device)?;
        let mut builds = version_mode(fixture.releases(None, 0, "")?);
        let failed = fixture.failed(Some(V2), &["refused the invocation as malformed"]);
        let ctx = fixture.session(&fake.phone)?;

        run_claimed_job(&mut fake.phone, &mut builds, &ctx, &claim)?;

        failed.assert();
        Ok(())
    }

    /// Unlike a refusal, a kill by iOS is terminal, so mgmt ends the job for every phone.
    #[test]
    fn an_ios_kill_is_filed_non_retriable_under_the_build() -> anyhow::Result<()> {
        let fixture = Fixture::new([true, true])?;
        let claim = claim(Some(&[V2]))?;
        let killed = ConsoleRun {
            stdout: format!(
                "{LAUNCHED}[HEADLESS] start runtime=llama.cpp model=gemma-4-E4B-it-Q4_K_M.gguf \
                 batch=512 benchmarks=[\"vl_max_memory_256x256_text0_img1\"] submit=true\n{KILLED}"
            ),
            stderr: "load_hparams: model size:         944.39 MiB\n".to_owned(),
            sentinel: None,
            exit_code: Some(1),
            timed_out: false,
            engine_time: None,
            aborted: false,
            stopped: false,
            silent: false,
        };
        let device = found_running(V2)?
            .then_console(ok(
                "[HEADLESS] results count=0\n[HEADLESS] BENCH_DONE 0\n",
                "",
            ))
            .then_console(killed);
        let mut fake = fixture.phone(device)?;
        let mut builds = version_mode(fixture.releases(None, 0, "")?);
        let failed = fixture.failure(
            false,
            Some(V2),
            &[&format!("build {V2}"), "signal 9", "memory limit"],
        );
        let ctx = fixture.session(&fake.phone)?;

        run_claimed_job(&mut fake.phone, &mut builds, &ctx, &claim)?;

        failed.assert();
        Ok(())
    }

    /// A build that reports its failed cell as `BENCH_DONE 0 completed` and leaves
    /// no payload still logs the engine's failure; the reason names the build and
    /// quotes the app's line.
    #[test]
    fn an_engine_failure_reported_as_success_is_filed_non_retriable() -> anyhow::Result<()> {
        let fixture = Fixture::new([true, true])?;
        let claim = claim(Some(&[V2]))?;
        let line = "[HEADLESS] error jobRun decode_throughput_4096_100 / prism-ml/Bonsai-27B-gguf \
                    failed: Engine backend failed: prefill decode rc=-3.";
        let device = found_running(V2)?
            .then_console(ok(
                "[HEADLESS] results count=0\n[HEADLESS] BENCH_DONE 0\n",
                "",
            ))
            .then_console(ok(
                "[HEADLESS] BENCH_DONE 0 completed\n",
                &format!("[HEADLESS] info jobRun dispatch: RunRequest()\n{line}\n"),
            ))
            .then_console(ok(
                "[HEADLESS] results count=0\n[HEADLESS] BENCH_DONE 0\n",
                "",
            ));
        let mut fake = fixture.phone(device)?;
        let mut builds = version_mode(fixture.releases(None, 0, "")?);
        let failed = fixture.failure(false, Some(V2), &[&format!("build {V2}"), line]);
        let ctx = fixture.session(&fake.phone)?;

        run_claimed_job(&mut fake.phone, &mut builds, &ctx, &claim)?;

        failed.assert();
        Ok(())
    }

    #[test]
    fn the_lease_is_kept_while_the_build_is_signed() -> anyhow::Result<()> {
        let fixture = Fixture::new([true, true])?;
        let device = a_run(
            FakeDevice::new("phone-a")
                .then_console(version_line(V1))
                .then_console(identity(CLIENT_ID))
                .then_console(version_line(V2))
                .then_console(runtimes_line(&llamacpp("b10216")?)?),
            V2,
        );
        let mut fake = fixture.phone(device)?;
        with_heartbeat(&mut fake.phone, Duration::from_secs(1));
        let mut builds = version_mode(fixture.releases(None, 0, "sleep 3")?);
        let heartbeat = fixture.server.mock(|when, then| {
            when.method(PUT).path("/plans/job-1/heartbeat");
            then.status(200);
        });
        let submit = fixture.submitted(json!({ "client_version": V2 }));
        let ctx = fixture.session(&fake.phone)?;

        run_claimed_job(&mut fake.phone, &mut builds, &ctx, &claim(Some(&[V2]))?)?;

        submit.assert();
        assert!(heartbeat.calls() >= 2, "{} heartbeats", heartbeat.calls());
        Ok(())
    }

    /// A lease lost during an install that no signal interrupts reports
    /// nothing, but a halt the build earned still stops the phone: a replaced
    /// identity, or the last offered version withdrawn.
    #[rstest::rstest]
    #[case::identity_replaced(
        vec![identity("ev1_someone_else")],
        "replaced the phone's identity"
    )]
    #[case::last_version_withdrawn(
        vec![
            identity(CLIENT_ID),
            version_line(V2),
            runtimes_line(&llamacpp("b10100")?)?,
        ],
        "withdrew every build"
    )]
    fn a_halt_outlives_a_lost_lease(
        #[case] after_the_install: Vec<ConsoleRun>,
        #[case] needle: &str,
    ) -> anyhow::Result<()> {
        let fixture = Fixture::new([true, true])?;
        let installing = Arc::new(AtomicBool::new(false));
        let device = after_the_install.into_iter().fold(
            FakeDevice::new("phone-a")
                .then_console(version_line(V1))
                .installing_slowly(Arc::clone(&installing), Duration::from_secs(3)),
            FakeDevice::then_console,
        );
        let mut fake = fixture.phone(device)?;
        with_max_jobs(&mut fake.phone, 3);
        with_heartbeat(&mut fake.phone, Duration::from_secs(1));
        let mut releases = fixture.releases(None, 0, "")?;
        releases.catalog.withdraw(&V1.parse()?);
        let claims = fixture.claims(&claim(Some(&[V2]))?);
        let _kept = {
            let installing = Arc::clone(&installing);
            fixture.server.mock(|when, then| {
                when.method(PUT)
                    .path("/plans/job-1/heartbeat")
                    .is_false(move |_| installing.load(Ordering::SeqCst));
                then.status(200);
            })
        };
        let lost = {
            let installing = Arc::clone(&installing);
            fixture.server.mock(|when, then| {
                when.method(PUT)
                    .path("/plans/job-1/heartbeat")
                    .is_true(move |_| installing.load(Ordering::SeqCst));
                then.status(409);
            })
        };
        let reported = fixture.server.mock(|when, then| {
            when.method(POST).path("/benchmarks");
            then.status(202).json_body(json!({ "job_id": "job-1" }));
        });
        let ctx = fixture.session(&fake.phone)?;

        let error = claim_loop(
            &mut fake.phone,
            &mut version_mode(releases),
            &ctx,
            &mut Tally::default(),
        )
        .err()
        .context("the phone must halt")?;

        assert!(needs_operator(&error), "{error:#}");
        assert!(format!("{error:#}").contains(needle), "{error:#}");
        claims.assert_calls(1);
        assert!(lost.calls() >= 1, "the lease was never lost");
        reported.assert_calls(0);
        Ok(())
    }

    /// A stop mid-export ends it, and one once the build is signed is honoured
    /// before the install: nothing is installed or reported, the lease lapses,
    /// and the job's line says it stopped.
    #[rstest::rstest]
    #[case::during_the_export("sleep 30", false, 0)]
    #[case::after_signing("", true, 1)]
    fn a_stop_before_the_install_reports_nothing(
        #[case] export: &str,
        #[case] at_the_gate: bool,
        #[case] gated: usize,
    ) -> anyhow::Result<()> {
        let fixture = Fixture::new([true, true])?;
        let stop = Arc::new(AtomicBool::new(false));
        let device = FakeDevice::new("phone-a").then_console(version_line(V1));
        let device = if at_the_gate {
            device.raising_at_gate(Arc::clone(&stop))
        } else {
            device
        };
        let mut fake = fixture.phone(device)?;
        fake.phone.stop = Arc::clone(&stop);
        let mut builds = version_mode(fixture.releases(None, 0, export)?);
        let reported = fixture.server.mock(|when, then| {
            when.method(POST).path("/benchmarks");
            then.status(202).json_body(json!({ "job_id": "job-1" }));
        });
        let ctx = fixture.session(&fake.phone)?;
        let claim = claim(Some(&[V2]))?;
        let started = std::time::Instant::now();

        let (ended, log) = captured(|| {
            std::thread::scope(|scope| {
                if !at_the_gate {
                    scope.spawn(|| {
                        std::thread::sleep(Duration::from_millis(500));
                        stop.store(true, Ordering::SeqCst);
                    });
                }
                run_claimed_job(&mut fake.phone, &mut builds, &ctx, &claim)
            })
        });

        ended?;
        reported.assert_calls(0);
        assert!(started.elapsed() < Duration::from_secs(20));
        assert_eq!(fake.gated.builds()?.len(), gated);
        assert_eq!(fake.calls()?, vec![launch(&["headlessrun", "version"])]);
        has_fields(
            &summary_fields(&log)?,
            &[
                "outcome=stopped",
                "cause=-",
                "installed=no",
                "install_secs=-",
            ],
        );
        Ok(())
    }

    /// At startup a phone on any offered version is left alone, whatever the
    /// default, once it proves its client; one on none gets the default.
    #[rstest::rstest]
    #[case::on_an_offered_version(
        V1,
        vec![
            launch(&["headlessrun", "version"]),
            launch(&["headlessrun", "auth", "me", "local=1"]),
            launch(&["headlessrun", "runtimes", "list"]),
        ],
        V1
    )]
    #[case::on_none_offered(
        UNOFFERED,
        vec![
            launch(&["headlessrun", "version"]),
            installed(),
            launch(&["headlessrun", "auth", "me", "local=1"]),
            launch(&["headlessrun", "version"]),
            launch(&["headlessrun", "runtimes", "list"]),
        ],
        V2
    )]
    fn startup_installs_only_when_the_phone_runs_no_offered_version(
        #[case] running: &str,
        #[case] calls: Vec<Call>,
        #[case] recorded: &str,
    ) -> anyhow::Result<()> {
        let fixture = Fixture::new([true, true])?;
        let device = FakeDevice::new("phone-a")
            .then_console(version_line(running))
            .then_console(identity(CLIENT_ID));
        // An install reads its version back; a build found running goes straight to the
        // listing.
        let device = if running == UNOFFERED {
            device.then_console(version_line(V2))
        } else {
            device
        };
        let device = device.then_console(runtimes_line(&llamacpp("b10216")?)?);
        let mut fake = fixture.phone(device)?;
        let mut releases = fixture.releases(Some(V2), 0, "")?;

        releases.start(&mut fake.phone)?;

        assert_eq!(fake.calls()?, calls);
        assert_eq!(
            InstalledBuild::load(&fake.phone.paths.installed)?
                .version
                .as_deref(),
            Some(recorded)
        );
        Ok(())
    }

    /// At startup a build found running unrecorded, whose listing fails, is withdrawn
    /// and the phone claims: no restart, and no install before the first claim.
    #[test]
    fn startup_withdraws_a_build_found_running_whose_listing_fails() -> anyhow::Result<()> {
        let fixture = Fixture::new([true, true])?;
        let device = FakeDevice::new("phone-a")
            .then_console(version_line(V2))
            .then_console(identity(CLIENT_ID))
            .then_console(ConsoleRun {
                sentinel: Some(1),
                exit_code: Some(1),
                ..ok(LISTING_FAILS, "")
            });
        let mut fake = fixture.phone(device)?;
        let mut releases = fixture.releases(Some(V2), 0, "")?;

        releases.start(&mut fake.phone)?;

        assert_eq!(
            fake.calls()?,
            [
                launch(&["headlessrun", "version"]),
                launch(&["headlessrun", "auth", "me", "local=1"]),
                launch(&["headlessrun", "runtimes", "list"]),
            ]
        );
        assert_eq!(
            releases.catalog.version_flags(),
            [format!("client_version:{V1}")]
        );
        assert_eq!(
            InstalledBuild::load(&fake.phone.paths.installed)?.version,
            None
        );
        Ok(())
    }

    /// At startup an operator is needed, before any profile is reported, for
    /// a phone on an offered version that holds another client, and one whose
    /// install withdraws the last version left.
    #[rstest::rstest]
    #[case::another_client(
        [true, true],
        vec![version_line(V1), identity("ev1_someone_else")],
        None,
        "replaced the phone's identity",
        vec![
            launch(&["headlessrun", "version"]),
            launch(&["headlessrun", "auth", "me", "local=1"]),
        ]
    )]
    #[case::last_version_withdrawn(
        [true, false],
        vec![version_line(UNOFFERED)],
        Some(V1),
        "withdrew every build",
        vec![launch(&["headlessrun", "version"])]
    )]
    fn startup_that_needs_an_operator(
        #[case] memory: [bool; 2],
        #[case] consoles: Vec<ConsoleRun>,
        #[case] withdrawn: Option<&str>,
        #[case] needle: &str,
        #[case] calls: Vec<Call>,
    ) -> anyhow::Result<()> {
        let fixture = Fixture::new(memory)?;
        let device = consoles
            .into_iter()
            .fold(FakeDevice::new("phone-a"), FakeDevice::then_console);
        let mut fake = fixture.phone(device)?;
        let mut releases = fixture.releases(None, 0, "")?;
        if let Some(version) = withdrawn {
            releases.catalog.withdraw(&version.parse()?);
        }

        let error = releases
            .start(&mut fake.phone)
            .err()
            .context("the start must be refused")?;

        assert!(needs_operator(&error), "{error:#}");
        assert!(format!("{error:#}").contains(needle), "{error:#}");
        assert_eq!(fake.calls()?, calls);
        assert_eq!(
            InstalledBuild::load(&fake.phone.paths.installed)?.version,
            None
        );
        Ok(())
    }

    /// At startup an install the phone refuses (a phone missing from the
    /// profile, an iOS below the build's floor) does not restart the phone: it
    /// claims, every version still offered, and each claim installs its own
    /// build. A phone gone mid-install restarts.
    #[rstest::rstest]
    #[case::refused(anyhow::anyhow!("devicectl: Unable to install \"Pipette\""), Ok(()))]
    #[case::gone(
        anyhow::Error::new(DevicectlFaultError {
            domain: "com.apple.dt.CoreDeviceError".to_owned(),
            code: DEVICE_NOT_FOUND_CODE,
            description: "The specified device was not found.".to_owned(),
        }),
        Err("installing build")
    )]
    fn a_startup_install_the_phone_refuses_does_not_restart_it(
        #[case] install_failure: anyhow::Error,
        #[case] expected: Result<(), &str>,
    ) -> anyhow::Result<()> {
        let fixture = Fixture::new([true, true])?;
        let device = FakeDevice::new("phone-a")
            .then_console(version_line(UNOFFERED))
            .failing_install_with(install_failure);
        let mut fake = fixture.phone(device)?;
        let mut releases = fixture.releases(None, 0, "")?;

        match (releases.start(&mut fake.phone), expected) {
            (Ok(()), Ok(())) => {}
            (Err(error), Err(needle)) => {
                assert!(!needs_operator(&error), "{error:#}");
                assert!(format!("{error:#}").contains(needle), "{error:#}");
            }
            (got, expected) => anyhow::bail!("got {got:?}, expected {expected:?}"),
        }
        assert_eq!(
            fake.calls()?,
            vec![launch(&["headlessrun", "version"]), installed()]
        );
        assert_eq!(releases.catalog.version_flags().len(), 2);
        assert_eq!(
            InstalledBuild::load(&fake.phone.paths.installed)?.version,
            None
        );
        Ok(())
    }

    /// A restart keeps the last listing when the bucket cannot be listed; a
    /// phone that never listed it cannot start. `Ok` counts the phone's calls.
    #[rstest::rstest]
    #[case::listed_before(true, Ok(3))]
    #[case::never_listed(false, Err("listing the releases bucket"))]
    fn a_restart_needs_no_bucket_once_listed(
        #[case] listed_before: bool,
        #[case] expected: Result<usize, &str>,
    ) -> anyhow::Result<()> {
        let fixture = Fixture::new([true, true])?;
        let mut fake = fixture.phone(found_running(V1)?)?;
        let mut releases = if listed_before {
            fixture.releases(None, 0, "")?
        } else {
            fixture.unlisted(None, 0, "")?
        };
        fixture.bucket.server.reset();

        let started = releases.start(&mut fake.phone);

        match (started, expected) {
            (Ok(()), Ok(calls)) => assert_eq!(fake.calls()?.len(), calls),
            (Err(error), Err(needle)) => {
                assert!(format!("{error:#}").contains(needle), "{error:#}");
                assert_eq!(fake.calls()?, vec![]);
            }
            (got, expected) => anyhow::bail!("got {got:?}, expected {expected:?}"),
        }
        Ok(())
    }

    /// A phone that cannot run a job pauses before the start runs the app,
    /// rather than failing the start and restarting.
    #[test]
    fn a_phone_that_cannot_run_a_job_pauses_before_the_start() -> anyhow::Result<()> {
        let fixture = Fixture::new([true, true])?;
        let mut fake = fixture.phone(FakeDevice::new("phone-a").gone("CoreDeviceError 1011"))?;
        let mut builds = version_mode(fixture.releases(None, 0, "")?);
        fake.phone.stop.store(true, Ordering::SeqCst);

        run_one(&mut fake.phone, &mut builds, &mut Tally::default())?;

        assert_eq!(fake.calls()?, vec![]);
        Ok(())
    }

    #[test]
    fn a_mac_that_cannot_sign_starts_no_phone() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let signer = signer(
            dir.path(),
            security(dir.path(), 51, 1)?,
            xcodebuild(dir.path(), Path::new("/none.ipa"), "")?,
        );
        let state = dir.path().join("state");
        let (phone, _pool) = phone(FakeDevice::new("phone-a"), &state)?;

        let error = fleet_builds(
            signer,
            &phone.config,
            &StatePaths::resolve(Some(&state))?,
            2,
        )
        .err()
        .context("run must not start")?;

        assert!(
            format!("{error:#}").starts_with("checking that this Mac can sign builds: unlocking"),
            "{error:#}"
        );
        Ok(())
    }

    /// A version withdrawn by one job leaves the next profile report, before
    /// the next claim; a phone that withdraws its last one stops rather than
    /// claim with nothing to offer, inside its job budget.
    #[rstest::rstest]
    #[case::one_left([true, false], &[V1][..], &[V2][..], 2, Ok(()))]
    #[case::none_left(
        [false, false],
        &[UNOFFERED, UNOFFERED][..],
        &[V1, V2][..],
        3,
        Err("withdrew every build")
    )]
    fn a_withdrawn_version_leaves_the_next_report(
        #[case] memory: [bool; 2],
        #[case] running: &[&str],
        #[case] admitted: &[&str],
        #[case] max_jobs: u32,
        #[case] expected: Result<(), &str>,
    ) -> anyhow::Result<()> {
        let fixture = Fixture::new(memory)?;
        let device = running
            .iter()
            .fold(FakeDevice::new("phone-a"), |device, version| {
                device.then_console(version_line(version))
            });
        let mut fake = fixture.phone(device)?;
        with_max_jobs(&mut fake.phone, max_jobs);
        let releases = fixture.releases(None, 0, "")?;
        let claims = fixture.claims(&claim(Some(admitted))?);
        let failed = fixture.failed(None, &[]);
        let report = fixture.reported(&format!(
            r#""capabilities":["client_version:{V1}","{B10216_BUILD_FLAG}"]"#
        ));
        let ctx = fixture.session(&fake.phone)?;

        let ended = claim_loop(
            &mut fake.phone,
            &mut version_mode(releases),
            &ctx,
            &mut Tally::default(),
        );

        match (ended, expected) {
            (Ok(()), Ok(())) => {}
            (Err(error), Err(needle)) => {
                assert!(needs_operator(&error), "{error:#}");
                assert!(format!("{error:#}").contains(needle), "{error:#}");
            }
            (got, expected) => anyhow::bail!("got {got:?}, expected {expected:?}"),
        }
        claims.assert_calls(2);
        failed.assert_calls(2);
        report.assert_calls(1);
        Ok(())
    }

    #[rstest::rstest]
    #[case::reported(202)]
    #[case::report_refused(422)]
    fn withdrawing_the_last_version_halts_whatever_the_report(
        #[case] status: u16,
    ) -> anyhow::Result<()> {
        let fixture = Fixture::new([true, false])?;
        let mut fake = fixture.phone(FakeDevice::new("phone-a").then_console(version_line(V1)))?;
        let mut releases = fixture.releases(None, 0, "")?;
        releases.catalog.withdraw(&V1.parse()?);
        let report = fixture.server.mock(|when, then| {
            when.method(POST).path("/benchmarks");
            then.status(status).json_body(json!({ "job_id": "job-1" }));
        });
        let ctx = fixture.session(&fake.phone)?;

        let error = run_claimed_job(
            &mut fake.phone,
            &mut version_mode(releases),
            &ctx,
            &claim(Some(&[V2]))?,
        )
        .err()
        .context("the phone must halt")?;

        report.assert();
        assert!(needs_operator(&error), "{error:#}");
        assert!(
            format!("{error:#}").contains("withdrew every build"),
            "{error:#}"
        );
        Ok(())
    }

    /// Every line from the claim on names the job, and one sums it up, field by
    /// field: `#` stands for seconds, `-` for a phase the job never reached.
    #[rstest::rstest]
    #[case::installed_and_accepted(
        "b10216",
        [
            "outcome=accepted",
            "cause=-",
            "infra=-",
            "benchmark=max_memory_usage_512",
            "version=2026.10.02-e6a9434-internal",
            "chosen_by=newest",
            "installed=yes",
            "choose_secs=#",
            "pull_secs=#",
            "sign_secs=#",
            "install_secs=#",
            "run_secs=#",
            "submit_secs=#",
            "total_secs=#",
        ]
    )]
    #[case::runtime_in_no_build(
        "b1",
        [
            "outcome=failed",
            "cause=no_build_carries_runtime",
            "infra=no",
            "benchmark=max_memory_usage_512",
            "version=-",
            "chosen_by=-",
            "installed=no",
            "choose_secs=#",
            "pull_secs=-",
            "sign_secs=-",
            "install_secs=-",
            "run_secs=-",
            "submit_secs=#",
            "total_secs=#",
        ]
    )]
    fn every_line_of_a_job_names_it(
        #[case] runtime: &str,
        #[case] expected: [&str; 14],
    ) -> anyhow::Result<()> {
        let fixture = Fixture::new([true, true])?;
        let device = a_run(
            FakeDevice::new("phone-a")
                .then_console(version_line(V1))
                .then_console(identity(CLIENT_ID))
                .then_console(version_line(V2))
                .then_console(runtimes_line(&llamacpp("b10216")?)?),
            V2,
        );
        let mut fake = fixture.phone(device)?;
        with_max_jobs(&mut fake.phone, 1);
        let releases = fixture.releases(None, 0, "")?;
        let claims = fixture.claims(&claim_wanting(Some(&[V2]), runtime)?);
        let reported = fixture.server.mock(|when, then| {
            when.method(POST).path("/benchmarks");
            then.status(202).json_body(json!({ "job_id": "job-1" }));
        });
        let ctx = fixture.session(&fake.phone)?;

        let (ended, log) = captured(|| {
            claim_loop(
                &mut fake.phone,
                &mut version_mode(releases),
                &ctx,
                &mut Tally::default(),
            )
        });

        ended?;
        claims.assert_calls(1);
        reported.assert_calls(1);
        let from_the_claim: Vec<&str> = log
            .lines()
            .skip_while(|line| !line.contains("claimed a job"))
            .collect();
        let finished = from_the_claim
            .iter()
            .position(|line| line.contains(" job finished "))
            .context("no `job finished` line")?;
        let (of_the_job, after) = from_the_claim.split_at(finished + 1);
        assert!(of_the_job.len() > 2, "{log}");
        of_the_job
            .iter()
            .for_each(|line| assert!(line.contains(" job{id=job-1}: "), "{line}"));
        // The span closes with the summary: the loop's own lines name no job.
        assert!(
            after.iter().any(|line| line.contains("reached max_jobs")),
            "{log}"
        );
        after
            .iter()
            .for_each(|line| assert!(!line.contains("job{id="), "{line}"));
        let fields = summary_fields(&log)?;
        assert_eq!(fields.len(), expected.len(), "{fields:?}");
        fields.iter().zip(expected).for_each(|(field, expected)| {
            assert!(field_matches(field, expected), "{field}, not {expected}");
        });
        Ok(())
    }

    /// At the top of the claim loop a stale listing is listed again: a changed
    /// set is reported before the next claim, and a failed listing keeps the
    /// set and reports nothing.
    #[rstest::rstest]
    #[case::a_version_added(true, 1)]
    #[case::listing_fails(false, 0)]
    fn a_stale_listing_is_refreshed_before_a_claim(
        #[case] lists: bool,
        #[case] reports: usize,
    ) -> anyhow::Result<()> {
        let fixture = Fixture::new([true, true])?;
        let mut fake = fixture.phone(FakeDevice::new("phone-a"))?;
        with_max_jobs(&mut fake.phone, 1);
        let mut releases = fixture.releases(None, 0, "")?;
        releases.catalog.expire();
        fixture.bucket.server.reset();
        let v3 = "2026.10.03-0000003-internal";
        if lists {
            let mut published = fixture.published.clone();
            published.extend(
                Published::build(&format!("{v3}/pipette-ios-{v3}.xcarchive.zip"), b"zip", 3)
                    .objects(),
            );
            fixture.bucket.publish(&published, 1000);
        }
        let unreadable = ClaimedJob {
            spec: serde_json::Value::Null,
            ..claim(None)?
        };
        let claims = fixture.claims(&unreadable);
        let failed = fixture.submitted(json!({ "retriable": false }));
        let report = fixture.reported(&format!(
            r#""capabilities":["client_version:{V1}","client_version:{V2}","client_version:{v3}","{B10216_BUILD_FLAG}"]"#
        ));
        let ctx = fixture.session(&fake.phone)?;

        claim_loop(
            &mut fake.phone,
            &mut version_mode(releases),
            &ctx,
            &mut Tally::default(),
        )?;

        report.assert_calls(reports);
        claims.assert_calls(1);
        failed.assert_calls(1);
        Ok(())
    }
}
