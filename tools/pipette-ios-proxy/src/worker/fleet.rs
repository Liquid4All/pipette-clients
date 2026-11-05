//! The phones one process drives: config, probe, per-phone lock, and the shared
//! build pool's refresh.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use anyhow::Context;

use pipette_mgmt_client::types::DeviceProfileFields;

use super::install;
use super::pause::{Pause, pause_or_remind};
use crate::app::Signals;
use crate::cli::Cli;
use crate::config::{self, Config, ConfigError, DeviceConfig};
use crate::device::{DeviceControl, DeviceDetails, Devicectl};
use crate::logging;
use crate::pool::{self, Pool};
use crate::profile;
use crate::state::{DeviceFacts, DevicePaths, InstalledBuild, StateLock, StatePaths};
use crate::timing;

/// The stop signal arrived while waiting: a clean exit, not a failure.
#[derive(Debug, thiserror::Error)]
#[error("stopped while {0}")]
pub(crate) struct StoppedError(pub &'static str);

/// Whether a verb must be the only process driving its phones. The verbs that
/// install or write state take each selected phone's lock; the read-only ones run
/// beside the agent on purpose.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Exclusivity {
    Exclusive,
    Shared,
}

/// Whether a verb waits for the phone or fails when it is not there. `run` waits:
/// a fleet Mac reboots faster than a phone re-enumerates, and exiting would turn
/// a boot-order race into a restart loop. Every other verb answers a question
/// about right now, where waiting would hang a command an operator is watching.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DeviceWait {
    Wait,
    Fail,
}

/// How long to wait between probes for an absent phone. Fixed rather than backing
/// off: a phone is attached by a human, which does not get likelier with time.
const DEVICE_WAIT_INTERVAL: Duration = Duration::from_secs(10);

/// One configured phone, probed; built before any server call.
pub(crate) struct Phone {
    pub(crate) config: Arc<Config>,
    /// This phone's `[[devices]]` entry.
    pub(crate) entry: DeviceConfig,
    pub(crate) control: Box<dyn DeviceControl>,
    /// The probe result, taken at startup: it names this phone's state
    /// directory and the reported profile is built from it. Its `os_version`
    /// follows each later reachability probe ([`Phone::note_os_version`]).
    pub(crate) details: DeviceDetails,
    /// Where this phone's own state lives; the build index is shared, above it.
    pub(crate) paths: DevicePaths,
    pub(crate) installed: Option<PathBuf>,
    pub(crate) facts: Option<DeviceFacts>,
    /// Set by the signal handler; every wait checks it.
    pub(crate) stop: Arc<AtomicBool>,
}

impl std::fmt::Debug for Phone {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Phone")
            .field("device", &self.control.udid())
            .field("client_id", &self.entry.credentials.client_id)
            .field("installed", &self.installed)
            .finish_non_exhaustive()
    }
}

impl Phone {
    /// Record which build is on this phone, in memory and on disk together.
    /// `None` is "unknown", which is what an install in flight leaves behind.
    pub(crate) fn set_installed(&mut self, build: Option<&Path>) -> anyhow::Result<()> {
        self.record_installed(InstalledBuild {
            path: build.map(Path::to_path_buf),
            version: None,
        })
    }

    pub(crate) fn record_installed(&mut self, record: InstalledBuild) -> anyhow::Result<()> {
        record.store(&self.paths.installed)?;
        self.installed = record.path;
        Ok(())
    }

    /// Keep the iOS version a reachability probe read, so the next report
    /// carries it; `true` when it changed. A probe that read none, or a blank
    /// one, keeps the last; padding is no change.
    pub(crate) fn note_os_version(&mut self, read: Option<String>) -> bool {
        let Some(read) = read
            .map(|version| version.trim().to_owned())
            .filter(|version| !version.is_empty())
        else {
            return false;
        };
        if self.details.os_version.as_deref().map(str::trim) == Some(read.as_str()) {
            return false;
        }
        tracing::info!(
            from = %self.details.os_version.as_deref().unwrap_or("unknown"),
            to = %read,
            "the phone runs another iOS version"
        );
        self.details.os_version = Some(read);
        true
    }
}

/// Every phone this worker acts for, plus the config and state they share.
/// Each verb opens the pool itself, when `config.builds` names one.
pub(crate) struct Fleet {
    pub(crate) phones: Vec<Phone>,
    /// One per phone for the verbs that drive phones; empty for the read-only ones.
    pub(crate) locks: Vec<StateLock>,
    pub(crate) config: Arc<Config>,
    pub(crate) state: StatePaths,
    pub(crate) stop: Arc<AtomicBool>,
}

/// Read the config, probe every selected device, and install the signal handler.
/// Touches no server.
///
/// Every device is probed here: the resolved identifier names its state
/// directory, and two entries reaching ONE phone can only be caught after the
/// probe (`devicectl` accepts several spellings for one device).
///
/// `--udid` SELECTS one configured device rather than overriding a device's own.
pub(crate) fn prepare(
    cli: &Cli,
    wait_for_device: DeviceWait,
    exclusivity: Exclusivity,
) -> anyhow::Result<Fleet> {
    // The flag and the variable meet here, not inside clap, so an empty variable
    // can count as unset.
    let config_path =
        config::resolve_config_path(cli.config_path(), config::config_path_env().as_deref());
    // No context: `Config::load` already names the file it was reading.
    let config = Arc::new(Config::load(&config_path)?);
    logging::init(
        cli.requested_log_level().unwrap_or(config.log.level),
        config.log.format,
    );

    // Before any waiting, or Ctrl-C mid-probe takes the default disposition.
    let stop = Arc::new(AtomicBool::new(false));
    let signalled = Arc::clone(&stop);
    // A flag rather than an exit: only the main thread can drop the heartbeat
    // guard and leave the phone without an orphaned app.
    ctrlc::set_handler(move || signalled.store(true, Ordering::SeqCst)).map_err(|error| {
        anyhow::anyhow!("installing the Ctrl-C/SIGTERM handler failed: {error}")
    })?;

    let paths = StatePaths::resolve(cli.global.state_dir.as_deref())?;

    let selected = cli
        .global
        .udid
        .as_deref()
        .map(str::trim)
        .filter(|raw| !raw.is_empty());
    let wanted = select_devices(&config, selected)?;

    let phones = wanted
        .into_iter()
        .map(|entry| {
            let (target, devicectl, details) = {
                // `run_one` opens this span only once the phone is resolved, and
                // the wait's lines must name the phone too.
                let _span = tracing::info_span!(
                    "phone",
                    device = %entry.udid.as_deref().unwrap_or("any"),
                    client_id = %entry.credentials.client_id.as_ref(),
                )
                .entered();
                resolve_device(&config, entry.udid.as_deref(), wait_for_device, &stop)?
            };
            tracing::info!(
                device = %target,
                client_id = %entry.credentials.client_id.as_ref(),
                product_type = %details.product_type,
                os_version = %details.os_version.as_deref().unwrap_or("unknown"),
                "device resolved"
            );
            let device_paths = paths.device(&target);
            Ok(Phone {
                config: Arc::clone(&config),
                entry,
                control: Box::new(devicectl),
                details,
                installed: InstalledBuild::load(&device_paths.installed)?.path,
                facts: DeviceFacts::load(&device_paths.device_facts)?,
                paths: device_paths,
                stop: Arc::clone(&stop),
            })
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    // Differing spellings can still be one phone; the probe's `identifier` is the
    // CoreDevice UUID whatever the entry said. Two claim loops on one device
    // install over each other mid-run.
    if let Some(shared) =
        config::first_duplicate(phones.iter().map(|phone| phone.details.identifier.as_str()))
    {
        return Err(ConfigError::SamePhone {
            file: config.source.clone(),
            entries: phones
                .iter()
                .filter(|phone| phone.details.identifier == shared)
                .map(|phone| phone.control.udid().to_owned())
                .collect(),
            phone: shared.clone(),
        }
        .into());
    }

    // Per phone, so one process per phone can share this state root.
    let locks = match exclusivity {
        Exclusivity::Exclusive => phones
            .iter()
            .map(|phone| StateLock::for_device(&phone.paths, phone.control.udid()))
            .collect::<anyhow::Result<Vec<_>>>()?,
        Exclusivity::Shared => Vec::new(),
    };

    Ok(Fleet {
        phones,
        locks,
        config,
        state: paths,
        stop,
    })
}

/// The devices a verb acts on: all of them, or the one `--udid` names. A `--udid`
/// matching nothing is an error, not a silent "act on all".
fn select_devices(config: &Config, selected: Option<&str>) -> anyhow::Result<Vec<DeviceConfig>> {
    let Some(selected) = selected else {
        return Ok(config.devices.clone());
    };
    let matched: Vec<DeviceConfig> = config
        .devices
        .iter()
        .filter(|device| device.udid.as_deref() == Some(selected))
        .cloned()
        .collect();
    if matched.is_empty() {
        let known: Vec<&str> = config
            .devices
            .iter()
            .filter_map(|device| device.udid.as_deref())
            .collect();
        return Err(ConfigError::UnknownUdid {
            file: config.source.clone(),
            selected: selected.to_owned(),
            known: known.into_iter().map(str::to_owned).collect(),
        }
        .into());
    }
    Ok(matched)
}

/// Resolve the target and probe it. Under [`DeviceWait::Wait`] both steps retry:
/// resolution fails with no phone attached, the probe fails for one not yet
/// unlocked, and both are cured by the same thing happening.
fn resolve_device(
    config: &Config,
    requested: Option<&str>,
    wait: DeviceWait,
    stop: &Arc<AtomicBool>,
) -> anyhow::Result<(String, Devicectl, DeviceDetails)> {
    let mut paused: Option<Pause> = None;
    loop {
        let attempt = Devicectl::resolve_target(requested, config.devicectl.call_timeout).and_then(
            |target| {
                let devicectl = Devicectl::new(target.clone(), config.devicectl.call_timeout);
                let identity = devicectl.identity().with_context(|| {
                    format!("probing device `{target}` with `devicectl device info details`")
                })?;
                Ok((target, devicectl, identity))
            },
        );

        let error = match attempt {
            Ok((target, devicectl, identity)) => {
                if let Some(pause) = paused {
                    tracing::info!(
                        device = %target,
                        paused_mins = %pause.minutes(),
                        "the phone is attached again"
                    );
                }
                return Ok((target, devicectl, identity));
            }
            Err(error) => error,
        };

        if wait == DeviceWait::Fail {
            return Err(error);
        }
        // Not once per probe: a phone left unplugged overnight would bury the log.
        let reason = format!("{error:#}");
        pause_or_remind(&mut paused, &reason, || {
            tracing::warn!(
                error = %reason,
                retry_secs = %DEVICE_WAIT_INTERVAL.as_secs(),
                "waiting for the phone; the claim loop starts when it is attached"
            );
        });
        if !timing::sleep_interruptible(DEVICE_WAIT_INTERVAL, stop) {
            // A supervisor stopping a proxy that never saw its phone is a clean stop.
            return Err(StoppedError("waiting for the phone to be attached").into());
        }
    }
}

/// Learn anything not yet known about the pool, reading builds on THIS phone.
/// Runs once before any device thread starts. Reading installs, so whichever
/// build the refresh left behind becomes this phone's installed build.
pub(crate) fn refresh_pool(phone: &mut Phone, pool: &mut Pool) -> anyhow::Result<()> {
    // One process per phone share the pool: the first reads new builds, the rest
    // wait and then find them already read.
    let _pool_lock = wait_for_pool_lock(pool, &phone.stop)?;
    pool.reload()?;

    let devicectl = phone.control.as_ref();
    let bundle_id = &phone.entry.bundle_id;
    let run_timeout = phone.config.devicectl.run_timeout;
    let call_timeout = phone.config.devicectl.call_timeout;
    let first_line = phone.config.devicectl.first_line_timeout;
    let installed_path = phone.paths.installed.clone();

    let last = pool.refresh(&phone.config.source, |build| {
        tracing::info!(
            build = %build.display(),
            device = %devicectl.udid(),
            "reading a build to learn which runtimes it provides"
        );
        install::check_before(devicectl, build)?;
        // From here the old answer is wrong, and a part-way failure must leave
        // "unknown" rather than a name the next job would trust.
        InstalledBuild::default().store(&installed_path)?;
        let reading = pool::read_build(devicectl, bundle_id, build, run_timeout, first_line)?;
        install::check_after(
            devicectl,
            &phone.entry,
            &format!("after installing {}", build.display()),
            call_timeout,
            first_line,
        )?;
        Ok(reading)
    })?;
    if let Some(build) = last {
        phone.set_installed(Some(&build))?;
    }
    Ok(())
}

fn wait_for_pool_lock(pool: &Pool, stop: &AtomicBool) -> anyhow::Result<StateLock> {
    let never = AtomicBool::new(false);
    let signals = Signals {
        abort: &never,
        stop,
    };
    let waited = StateLock::wait_for(&pool.lock_path(), Duration::MAX, signals);
    // Never cleared: a stop that ended the wait is still set here.
    if waited.is_err() && stop.load(Ordering::SeqCst) {
        return Err(StoppedError("waiting for the build pool lock").into());
    }
    waited
}

pub(crate) fn device_profile(phone: &Phone, capabilities: &[String]) -> DeviceProfileFields {
    profile::device_profile(
        &phone.entry,
        &phone.details,
        phone.facts.as_ref(),
        capabilities,
    )
}

pub(crate) fn client_details(phone: &Phone) -> String {
    profile::render_client_details(
        &phone.config.server.client_details_template,
        &phone.details,
        &profile::host_name(),
    )
}

#[cfg(test)]
pub(crate) mod testing {
    use pipette_mgmt_client::types::{ClaimedJob, PrivateKey, ServerUrl};
    use pipette_plan_types::Runtime;
    use pipette_plan_types::{
        ClientId, LlamacppIosPipette, LlamacppIosPipetteFlavor, NonEmptyString, PrivateApis,
        RepositoryUrl, SourceRepository,
    };

    use super::*;
    use crate::app::ConsoleRun;
    use crate::config::{BuildsConfig, ClientCredentials, DeviceConfig};
    use crate::device::fake::Call;
    use crate::device::fake::FakeDevice;
    use crate::server::Session;

    pub(crate) const BUNDLE: &str = "ai.liquid.liquid-pipette";
    pub(crate) const CLIENT_ID: &str = "ev1_test";

    /// What `headlessrun auth me local=1` prints for a phone holding `client`.
    pub(crate) fn identity_of(client: &str) -> ConsoleRun {
        ConsoleRun {
            stdout: format!(
                "[HEADLESS] auth me client_id={client} server_url=https://c public_key=5f\n\
                 [HEADLESS] BENCH_DONE 0\n"
            ),
            stderr: String::new(),
            sentinel: Some(0),
            exit_code: Some(0),
            timed_out: false,
            engine_time: None,
            aborted: false,
            stopped: false,
            silent: false,
        }
    }

    /// A pool-mode claim for one memory cell on the `b10216` build.
    pub(crate) fn memory_claim() -> anyhow::Result<ClaimedJob> {
        Ok(ClaimedJob {
            job_id: "job-mem".to_string(),
            benchmark_id: "max_memory_usage_512".to_string(),
            time_window: "PT10M".to_string(),
            expires_at: None,
            client_versions: None,
            spec: serde_json::json!({
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
            }),
        })
    }

    /// A `/clients/me` answer.
    pub(crate) fn profile_body(status: &str, reindex_pending: bool) -> serde_json::Value {
        serde_json::json!({
            "client_id": CLIENT_ID,
            "organization": "liquid",
            "client_details": "relaying iPhone16,1",
            "contact_email": "ops@example.com",
            "status": status,
            "reindex_pending": reindex_pending,
        })
    }

    /// The server knows every benchmark, so the lookup before each launch passes.
    pub(crate) fn knows_every_benchmark(server: &httpmock::MockServer) {
        server.mock(|when, then| {
            when.method(httpmock::Method::GET)
                .path_prefix("/benchmarks/");
            then.status(200).json_body(serde_json::json!({
                "benchmark_id": "max_memory_usage_512",
                "benchmark_type": "max_memory_usage",
            }));
        });
    }

    pub(crate) fn session(phone: &Phone, base_url: &str) -> anyhow::Result<Session> {
        Ok(Session {
            server: ServerUrl::try_new(base_url)
                .map_err(|e| anyhow::anyhow!("test server url: {e}"))?,
            http: crate::server::http_client()?,
            auth: phone.entry.credentials.signer(),
        })
    }

    /// The install-and-verify sequence a pool-mode install produces.
    pub(crate) fn install_calls(build: &Path) -> Vec<Call> {
        vec![
            Call::Install(build.to_path_buf()),
            Call::Launch {
                bundle_id: BUNDLE.to_owned(),
                app_args: ["headlessrun", "auth", "me", "local=1"]
                    .map(str::to_owned)
                    .to_vec(),
            },
        ]
    }

    pub(crate) fn llamacpp(version: &str) -> anyhow::Result<Runtime> {
        Ok(Runtime::LlamacppIosPipette(LlamacppIosPipette {
            source: SourceRepository {
                repository_url: RepositoryUrl::new("github.com/ggml-org/llama.cpp"),
                repository_version: NonEmptyString::try_new(version.to_owned())?,
            },
            flavor: LlamacppIosPipetteFlavor::IosArm64,
            private_apis: PrivateApis::default(),
        }))
    }

    /// No wait between probes, so a test can walk a phone through a pause.
    pub(crate) fn fast_paced(phone: &mut Phone) {
        let mut config = (*phone.config).clone();
        config.loop_config.idle_wait = Duration::ZERO;
        config.loop_config.idle_jitter = Duration::ZERO;
        phone.config = Arc::new(config);
    }

    /// A `Phone` over a fake, with its own state directory. The pool sits beside
    /// it, as the real code holds it.
    pub(crate) fn phone(device: FakeDevice, state: &Path) -> anyhow::Result<(Phone, Pool)> {
        let key = "1122334455667788990011223344556677889900112233445566778899001122";
        let device_config = DeviceConfig {
            udid: Some(device.udid().to_owned()),
            device_name: None,
            bundle_id: BUNDLE.to_owned(),
            credentials: ClientCredentials {
                client_id: ClientId::try_new("ev1_test")?,
                private_key: PrivateKey::try_new(key)?,
            },
        };
        let text = format!(
            "[server]\nurl = \"https://collector.example\"\n\n[[devices]]\nudid = \"{}\"\n\
             client_id = \"ev1_test\"\nprivate_key = \"{key}\"\n\n[builds]\ndir = \"{}\"\n",
            device.udid(),
            state.join("builds").display()
        );
        std::fs::create_dir_all(state.join("builds"))?;
        let config = Arc::new(Config::parse_str(&text, &state.join("cfg.toml"))?);
        let paths = StatePaths::resolve(Some(state))?.device(device.udid());
        let pool = Pool::open(
            &BuildsConfig {
                dir: state.join("builds"),
            },
            &state.join("builds.json"),
        )?;
        let phone = Phone {
            config,
            entry: device_config,
            control: Box::new(device),
            details: DeviceDetails {
                identifier: "1ACFAA4F-A9FD-5840-B495-072D33D2A5F5".to_owned(),
                hardware_udid: None,
                product_type: "iPhone16,1".to_owned(),
                marketing_name: None,
                os_version: None,
                os_build: None,
                cpu_type: None,
                device_name: None,
                tunnel_state: None,
                developer_mode: None,
            },
            paths,
            installed: None,
            facts: None,
            stop: Arc::new(AtomicBool::new(false)),
        };
        Ok((phone, pool))
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::app::ConsoleRun;
    use crate::config::BuildsConfig;
    use crate::device::fake::{Call, FakeDevice};
    use crate::worker::testing::{CLIENT_ID, identity_of, llamacpp, phone};

    fn console(stdout: &str) -> ConsoleRun {
        ConsoleRun {
            stdout: stdout.to_owned(),
            stderr: String::new(),
            sentinel: Some(0),
            exit_code: Some(0),
            timed_out: false,
            engine_time: None,
            aborted: false,
            stopped: false,
            silent: false,
        }
    }

    /// Every new build is read on the phone that refreshes the pool, in order.
    #[test]
    fn every_new_build_is_read_on_this_phone() -> anyhow::Result<()> {
        let state = tempfile::tempdir()?;
        let builds = state.path().join("builds");
        let device = ["b10216", "b10217"].iter().zip(["A.app", "B.app"]).try_fold(
            FakeDevice::new("phone-a"),
            |device, (version, name)| -> anyhow::Result<FakeDevice> {
                std::fs::create_dir_all(builds.join(name))?;
                std::fs::write(builds.join(name).join("Info.plist"), "<plist/>")?;
                let descriptor = format!(
                    r#"{{"type":"llamacpp_ios_pipette","repository_url":"github.com/ggml-org/llama.cpp","repository_version":"{version}","flavor":"ios-arm64"}}"#
                );
                Ok(device
                    .then_console(console(&format!(
                        "[HEADLESS] runtimes count=1\n[HEADLESS] runtime \
                         type=llamacpp_ios_pipette digest=0a1b2c3d descriptor={descriptor} \
                         runtime={version}:ios-arm64\n[HEADLESS] BENCH_DONE 0 completed\n"
                    )))
                    .then_console(identity_of(CLIENT_ID)))
            },
        )?;
        let log = device.log();
        let (mut phone, mut pool) = phone(device, state.path())?;

        refresh_pool(&mut phone, &mut pool)?;

        let installs: Vec<Call> = log
            .calls()?
            .into_iter()
            .filter(|call| matches!(call, Call::Install(_)))
            .collect();
        assert_eq!(
            installs,
            vec![
                Call::Install(builds.join("A.app")),
                Call::Install(builds.join("B.app"))
            ]
        );
        let read = ["b10216", "b10217"]
            .iter()
            .map(|version| {
                Ok(pool
                    .build_for(&llamacpp(version)?)
                    .map(|known| known.path.clone()))
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        assert_eq!(
            read,
            vec![Some(builds.join("A.app")), Some(builds.join("B.app"))]
        );
        assert_eq!(phone.installed, Some(builds.join("B.app")));
        Ok(())
    }

    /// Another process read the pool while this one waited for `builds.lock`: its
    /// reading is picked up rather than repeated with an install on this phone.
    #[test]
    fn a_pool_read_by_another_process_is_not_read_again() -> anyhow::Result<()> {
        let state = tempfile::tempdir()?;
        let device = FakeDevice::new("phone-a");
        let log = device.log();
        let (mut phone, mut pool) = phone(device, state.path())?;
        let bundle = state.path().join("builds/A.app");
        std::fs::create_dir_all(&bundle)?;
        std::fs::write(bundle.join("Info.plist"), "<plist/>")?;

        let mut other = Pool::open(
            &BuildsConfig {
                dir: state.path().join("builds"),
            },
            &state.path().join("builds.json"),
        )?;
        other.refresh(&state.path().join("cfg.toml"), |_| {
            Ok(pool::BuildReading {
                runtimes: vec![llamacpp("b10216")?],
            })
        })?;

        refresh_pool(&mut phone, &mut pool)?;

        assert_eq!(log.calls()?, vec![], "no install");
        assert!(pool.build_for(&llamacpp("b10216")?).is_some());
        Ok(())
    }

    #[test]
    fn a_stop_while_waiting_for_the_pool_lock_touches_nothing() -> anyhow::Result<()> {
        let state = tempfile::tempdir()?;
        let device = FakeDevice::new("phone-a");
        let log = device.log();
        let (mut phone, mut pool) = phone(device, state.path())?;
        let _other = StateLock::try_acquire(&pool.lock_path())?
            .map_err(|holder| anyhow::anyhow!("pool lock busy: {holder}"))?;
        phone.stop.store(true, Ordering::SeqCst);

        let error = refresh_pool(&mut phone, &mut pool)
            .err()
            .context("a stop while waiting must end the refresh")?;
        assert!(
            format!("{error:#}").contains("build pool lock"),
            "{error:#}"
        );
        assert_eq!(log.calls()?, vec![]);
        Ok(())
    }
}
