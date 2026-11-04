//! The one-shot verbs: `register`, `status`, `scan`, `uninstall`, `probe`,
//! `check`, `builds`.

use std::collections::BTreeSet;
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::Context;

use crate::cli::{BuildsCommand, Cli};
use crate::config::{self, BuildSource, Config, ConfigError};
use crate::device::{Reachability, gpu_pin};
use crate::logging;
use crate::pool::{self, Pool, display_name};
use crate::profile;
use crate::releases::catalog::Catalog;
use crate::releases::sign::Signer;
use crate::releases::{self, Build, s3, store};
use crate::server::{self, Session};
use crate::state::{self, InstalledBuild, ReleasePaths, StateLock, StatePaths};
use crate::timing;
use crate::worker::{
    Fleet, Phone, client_details, connect, device_profile, listed_catalog, refresh_pool,
};

const FIELD_LABEL_WIDTH: usize = 14;

/// `register`: refresh every configured device's server-side profile, then exit.
/// Nothing registers; each phone's client row already exists (`ClientCredentials::signer`).
pub(crate) fn register(fleet: Fleet) -> anyhow::Result<()> {
    let Fleet {
        mut phones,
        config,
        state,
        stop: _,
        locks,
    } = fleet;
    let _held = locks;
    let Some(first) = phones.first_mut() else {
        anyhow::bail!("no devices to refresh");
    };
    let capabilities = match &config.builds {
        BuildSource::Pool(source) => {
            let mut pool = Pool::open(source, &state.build_index)?;
            refresh_pool(first, &mut pool)?;
            pool.check_advertisable()?;
            pool.capabilities()?
        }
        // Lists the bucket and installs nothing.
        BuildSource::Releases(_) => listed_catalog(&config, &state)?.advertised()?,
    };

    phones.iter_mut().try_for_each(|phone| {
        let (_, profile) = connect(phone, &capabilities)?;
        print_field("device:", phone.control.udid());
        print_field("client:", &profile.client_id);
        print_field("status:", &profile.status.to_string());
        print_field("reindex:", &profile.reindex_pending.to_string());
        print_field("state_dir:", &phone.paths.dir.display().to_string());
        print_field("details:", &profile.client_details);
        print_field("capabilities:", &profile.capabilities.join(", "));
        println!();
        Ok(())
    })
}

/// `check`: everything `run` needs that can be known without a phone: the config
/// parses, every key resolves at a safe mode, the pool holds a build or this Mac
/// can sign, a configured GPU pin's CLI runs; plus which phones another process
/// is driving, and the `[client_releases]` secret. It cannot tell whether a key is the RIGHT key
/// (only the server knows) or what the builds provide (only a phone can say):
/// those are `register` and `scan`.
pub(crate) fn check(cli: &Cli) -> anyhow::Result<()> {
    let config = load_config(cli)?;

    print_field("config:", &config.source.display().to_string());
    print_field("server:", &config.server.url.to_string());
    print_field("devices:", &config.devices.len().to_string());
    for (index, device) in config.devices.iter().enumerate() {
        print_field(
            &format!("  #{}:", index + 1),
            &format!(
                "{} as {}",
                device.udid.as_deref().unwrap_or("(the one attached phone)"),
                device.credentials.client_id.as_ref()
            ),
        );
    }

    print_field("releases:", &releases_field(&config)?);

    match &config.devicectl.pymobiledevice3 {
        Some(program) => {
            gpu_pin::check_cli(program, config.devicectl.call_timeout)?;
            print_field("gpu pin:", &format!("max, through {}", program.display()));
        }
        None => print_field("gpu pin:", "off"),
    }

    match &config.builds {
        BuildSource::Pool(pool) => {
            // Present and non-empty; its CONTENTS need a phone, so `scan` reports those.
            let builds = pool::discover(&pool.dir)?;
            if builds.is_empty() {
                return Err(pool::UnadvertisablePoolError::Empty {
                    dir: pool.dir.clone(),
                    config: config.source.clone(),
                }
                .into());
            }
            print_field(
                "pool:",
                &format!("{} ({} builds)", pool.dir.display(), builds.len()),
            );
        }
        BuildSource::Releases(_) => print_field("signing:", &signing_field(&config.signer()?)?),
    }

    let paths = StatePaths::resolve(cli.global.state_dir.as_deref())?;
    print_field("state:", &paths.root.display().to_string());
    // A free lock is taken and released on the way: `check` is not a claimant.
    let driven = try_lock_all(paths.phones()?.into_iter().map(|phone| phone.lock))?.held;
    print_field(
        "running:",
        &if driven.is_empty() {
            "no phone is being driven".to_string()
        } else {
            driven
                .iter()
                .map(|held| {
                    let phone = held.path.parent().and_then(Path::file_name);
                    format!(
                        "{} ({})",
                        phone.unwrap_or(held.path.as_os_str()).to_string_lossy(),
                        held.holder
                    )
                })
                .collect::<Vec<_>>()
                .join(", ")
        },
    );
    println!("ok");
    Ok(())
}

/// Reads the secret and checks its file's mode; whether the bucket accepts the
/// pair is `builds list`.
fn releases_field(config: &Config) -> anyhow::Result<String> {
    if config.client_releases.is_none() {
        return Ok("no [client_releases]; the `builds` verbs are off".to_owned());
    }
    let releases = config.releases_bucket()?;
    Ok(format!(
        "{} in {}, access key {}",
        releases.url, releases.region, releases.access_key_id
    ))
}

fn signing_field(signer: &Signer) -> anyhow::Result<String> {
    signer
        .preflight()
        .context("checking that this Mac can sign builds")?;
    Ok(format!(
        "team {}, keychain {} and xcodebuild pass the preflight, App Store Connect key {}",
        signer.team_id,
        signer.keychain.display(),
        signer.api_key_id
    ))
}

/// The config and the logs, for the verbs that do not go through `prepare`.
fn load_config(cli: &Cli) -> anyhow::Result<Config> {
    let config_path =
        config::resolve_config_path(cli.config_path(), config::config_path_env().as_deref());
    let config = Config::load(&config_path)?;
    logging::init(
        cli.requested_log_level().unwrap_or(config.log.level),
        config.log.format,
    );
    Ok(config)
}

/// The `builds` verbs: the bucket and the store, with no phone and no
/// management server.
pub(crate) fn builds(cli: &Cli, action: &BuildsCommand) -> anyhow::Result<()> {
    builds_with(
        &load_config(cli)?,
        cli.global.state_dir.as_deref(),
        action,
        &mut std::io::stdout().lock(),
    )
}

fn builds_with(
    config: &Config,
    state_dir: Option<&Path>,
    action: &BuildsCommand,
    out: &mut dyn Write,
) -> anyhow::Result<()> {
    let bucket = &config.releases_bucket()?;
    let state = StatePaths::resolve(state_dir)?;
    let paths = state.releases();
    let http = s3::http_client()?;
    match action {
        BuildsCommand::List => {
            let builds = releases::list_builds(&http, bucket)?;
            if builds.is_empty() {
                tracing::info!(bucket = %bucket.url, "the bucket publishes no app build");
            }
            build_lines(&builds, &store::Records::load(&paths)?, &paths)
                .iter()
                .try_for_each(|line| print_line(out, line))
        }
        BuildsCommand::Pull { version } => {
            print_line(out, store::pull(&http, bucket, &paths, version)?.display())
        }
        BuildsCommand::Prune { dry_run } => {
            let catalog = || Catalog::listed(&http, bucket, config.default_version(), &paths);
            prune(&state, *dry_run, catalog, out)
        }
    }
}

fn print_line(out: &mut dyn Write, line: impl std::fmt::Display) -> anyhow::Result<()> {
    writeln!(out, "{line}").context("writing to stdout")
}

/// A lock another process holds.
struct HeldLock {
    path: PathBuf,
    holder: state::Holder,
}

/// Every lock asked for: taken where free, else who holds it.
struct Locking {
    taken: Vec<StateLock>,
    held: Vec<HeldLock>,
}

fn try_lock_all(paths: impl IntoIterator<Item = PathBuf>) -> anyhow::Result<Locking> {
    paths.into_iter().try_fold(
        Locking {
            taken: Vec::new(),
            held: Vec::new(),
        },
        |mut locking, path| {
            match StateLock::try_acquire(&path)? {
                Ok(lock) => locking.taken.push(lock),
                Err(holder) => locking.held.push(HeldLock { path, holder }),
            }
            Ok(locking)
        },
    )
}

#[derive(Debug, thiserror::Error)]
#[error(
    "another pipette-ios-proxy is using {root} ({held}): stop every process using the state \
     directory, then prune"
)]
struct StateInUseError {
    root: PathBuf,
    held: String,
}

/// `builds prune`: removes every version directory that neither the bucket's
/// catalog nor a phone's `installed.json` names, and every scratch directory.
/// The pull, signing and phone locks are held first, so no process reads what
/// it removes; `catalog` lists the bucket after that.
fn prune(
    state: &StatePaths,
    dry_run: bool,
    catalog: impl FnOnce() -> anyhow::Result<Catalog>,
    out: &mut dyn Write,
) -> anyhow::Result<()> {
    let paths = state.releases();
    let phones = state.phones()?;
    let _locks = if dry_run {
        Vec::new()
    } else {
        let locking = try_lock_all(
            [paths.lock.clone(), paths.signed_lock.clone()]
                .into_iter()
                .chain(phones.iter().map(|phone| phone.lock.clone())),
        )?;
        if !locking.held.is_empty() {
            return Err(StateInUseError {
                root: state.root.clone(),
                held: locking
                    .held
                    .iter()
                    .map(|held| format!("{} holds {}", held.holder, held.path.display()))
                    .collect::<Vec<_>>()
                    .join(", "),
            }
            .into());
        }
        locking.taken
    };
    let installed = phones
        .iter()
        .map(|phone| Ok(InstalledBuild::load(&phone.installed)?.version))
        .filter_map(anyhow::Result::transpose);
    let keep = catalog()?
        .versions()
        .map(|version| Ok(version.to_string()))
        .chain(installed)
        .collect::<anyhow::Result<BTreeSet<String>>>()?;
    tracing::info!(
        versions = %keep.iter().map(String::as_str).collect::<Vec<_>>().join(", "),
        "keeping these versions"
    );
    store::prunable(&paths, &keep)?.iter().try_for_each(|dir| {
        if !dry_run {
            std::fs::remove_dir_all(dir).with_context(|| format!("removing {}", dir.display()))?;
        }
        print_line(out, dir.display())
    })
}

const SHORT_SHA256_CHARS: usize = 12;

/// No value holds a space, so the lines also split on whitespace.
fn build_lines(builds: &[Build], records: &store::Records, paths: &ReleasePaths) -> Vec<String> {
    let width = |column: fn(&Build) -> &str| {
        builds
            .iter()
            .map(|build| column(build).len())
            .max()
            .unwrap_or_default()
    };
    let version_width = width(|build| build.version.as_str());
    let tag_width = width(|build| build.tag.as_str());
    builds
        .iter()
        .map(|build| {
            let sha256: &str = build.sha256.as_ref();
            format!(
                "{:<version_width$}  {:<tag_width$}  {:>9}  {}  {}  {}",
                build.version.as_str(),
                build.tag,
                mebibytes(build.bytes),
                timing::rfc3339(build.uploaded_at),
                sha256.get(..SHORT_SHA256_CHARS).unwrap_or(sha256),
                records.held(paths, build),
            )
        })
        .collect()
}

/// `104.9MiB`: tenths of a mebibyte, rounded down.
fn mebibytes(bytes: u64) -> String {
    const MEBIBYTE: u64 = 1024 * 1024;
    let tenths = bytes.saturating_mul(10) / MEBIBYTE;
    format!("{}.{}MiB", tenths / 10, tenths % 10)
}

/// `status`: print the server-side status and the phone's reachability. Never
/// writes to the server.
pub(crate) fn status(fleet: Fleet) -> anyhow::Result<()> {
    fleet.phones.into_iter().try_for_each(|phone| {
        status_one(phone)?;
        println!();
        Ok(())
    })
}

fn status_one(phone: Phone) -> anyhow::Result<()> {
    let reachability = phone.control.reachability(&phone.entry.bundle_id);
    print_field("device:", phone.control.udid());
    print_field(
        "hardware:",
        &format!(
            "{} ({})",
            phone.details.product_type,
            phone
                .details
                .marketing_name
                .as_deref()
                .unwrap_or("unknown model")
        ),
    );
    print_field(
        "os:",
        &format!(
            "{} {} ({})",
            profile::OS_NAME,
            phone.details.os_version.as_deref().unwrap_or("unknown"),
            phone.details.os_build.as_deref().unwrap_or("unknown")
        ),
    );
    let reachable = match &reachability {
        Reachability::Reachable { .. } => "yes".to_string(),
        Reachability::Unreachable(reason) => format!("no ({reason})"),
    };
    print_field("reachable:", &reachable);
    print_field("state_dir:", &phone.paths.dir.display().to_string());
    print_field("server:", &phone.config.server.url.to_string());

    let auth = phone.entry.credentials.signer();
    let client_id = auth.client_id.to_string();
    let http = server::http_client()?;
    let ctx = Session {
        server: phone.config.server.url.clone(),
        http,
        auth,
    };
    // Propagated: a `status` that cannot reach the server must exit non-zero.
    let profile = server::fetch_profile(&ctx)
        .with_context(|| format!("GET /clients/me as client {client_id}"))?;
    print_field("client:", &profile.client_id);
    print_field("status:", &profile.status.to_string());
    print_field("reindex:", &profile.reindex_pending.to_string());
    print_field("capabilities:", &profile.capabilities.join(", "));
    if !profile.tags.is_empty() {
        print_field("tags:", &profile.tags.join(", "));
    }
    Ok(())
}

fn print_field(label: &str, value: &str) {
    println!("{label:<FIELD_LABEL_WIDTH$}{value}");
}

/// `scan`: read every build in the pool and record what it provides. Reads on the
/// FIRST configured device; the answer is a property of the build, so every other
/// device inherits it.
pub(crate) fn scan(fleet: &mut Fleet) -> anyhow::Result<()> {
    let BuildSource::Pool(source) = &fleet.config.builds else {
        return Err(ConfigError::NoPool {
            file: fleet.config.source.clone(),
            verb: "scan",
        }
        .into());
    };
    let Some(phone) = fleet.phones.first_mut() else {
        anyhow::bail!("no devices to scan with");
    };
    let mut pool = Pool::open(source, &fleet.state.build_index)?;
    refresh_pool(phone, &mut pool)?;

    print_field("pool:", &pool.dir().display().to_string());
    pool.known().for_each(|known| {
        let runtimes: Vec<&str> = known
            .runtimes
            .iter()
            .map(pipette_plan_types::Runtime::headless_token)
            .collect();
        print_field(
            &format!("{}:", display_name(pool.dir(), &known.path)),
            &if runtimes.is_empty() {
                "no runtimes".to_string()
            } else {
                runtimes.join(", ")
            },
        );
    });
    print_field("capabilities:", &pool.capabilities()?.join(", "));
    pool.check_advertisable()
}

/// `uninstall`: remove the configured bundle id from every selected phone.
pub(crate) fn uninstall(fleet: &mut Fleet) -> anyhow::Result<()> {
    fleet.phones.iter_mut().try_for_each(uninstall_one)
}

fn uninstall_one(phone: &mut Phone) -> anyhow::Result<()> {
    let bundle_id = phone.entry.bundle_id.clone();
    phone
        .control
        .uninstall_app(&bundle_id)
        .with_context(|| format!("uninstalling {bundle_id} from {}", phone.control.udid()))?;
    // A later job that skipped its install on the old record would launch nothing.
    phone.set_installed(None)?;
    println!("uninstalled {bundle_id} from {}", phone.control.udid());
    Ok(())
}

pub(crate) fn probe(fleet: &Fleet) -> anyhow::Result<()> {
    let capabilities = probed_capabilities(fleet)?;
    fleet
        .phones
        .iter()
        .try_for_each(|phone| probe_one(phone, &capabilities))
}

/// The pool's flags as last read, or the bucket's versions; nothing is installed.
fn probed_capabilities(fleet: &Fleet) -> anyhow::Result<Vec<String>> {
    Ok(match &fleet.config.builds {
        BuildSource::Pool(source) => {
            Pool::open(source, &fleet.state.build_index)?.capabilities()?
        }
        BuildSource::Releases(_) => listed_catalog(&fleet.config, &fleet.state)?.advertised()?,
    })
}

fn probe_one(phone: &Phone, capabilities: &[String]) -> anyhow::Result<()> {
    let device = device_profile(phone, capabilities);
    // Pretty JSON in declaration order: the verb exists to be diffed against what
    // the fleet expects.
    let rendered =
        serde_json::to_string_pretty(&device).context("rendering the device profile as JSON")?;
    println!("{rendered}");
    // The fields with no profile slot go to stderr, so stdout stays one document.
    tracing::info!(
        client_details = %client_details(phone),
        device_identifier = %phone.details.identifier,
        hardware_udid = %phone.details.hardware_udid.as_deref().unwrap_or("-"),
        os_build = %phone.details.os_build.as_deref().unwrap_or("-"),
        // The one field that says whether the cable goes to the expected phone.
        device_name = %phone.details.device_name.as_deref().unwrap_or("-"),
        cpu_type = %phone.details.cpu_type.as_deref().unwrap_or("-"),
        tunnel_state = %phone.details.tunnel_state.as_deref().unwrap_or("-"),
        "devicectl reported"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::os::unix::fs::PermissionsExt;
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;

    use httpmock::Method::PATCH;
    use httpmock::MockServer;
    use rstest::rstest;
    use time::macros::datetime;

    use pipette_plan_types::Sha256;

    use super::*;
    use crate::config::ConfigError;
    use crate::device::DeviceControl;
    use crate::device::fake::{Call, FakeDevice};
    use crate::releases::testing::{B10216_BUILD_FLAG, FakeBucket, SECRET, build};
    use crate::worker::testing::{BUNDLE, phone, profile_body};

    fn version_fleet(
        dir: &Path,
        device: FakeDevice,
        server: &str,
        endpoint: &str,
    ) -> anyhow::Result<Fleet> {
        let secret = dir.join("releases.secret");
        std::fs::write(&secret, SECRET)?;
        std::fs::set_permissions(&secret, std::fs::Permissions::from_mode(0o600))?;
        let text = format!(
            "[server]\nurl = \"{server}\"\n\n[[devices]]\nudid = \"{}\"\n\
             client_id = \"ev1_test\"\nprivate_key = \"{}\"\n\n\
             [client_releases]\nbucket = \"builds\"\nregion = \"us-east-1\"\n\
             endpoint = \"{endpoint}\"\naccess_key_id = \"{}\"\n\
             secret_access_key_file = \"releases.secret\"\n\n\
             [signing]\nteam_id = \"353N5JSDGZ\"\n\
             keychain_password_file = \"keychain.password\"\napi_key_file = \"AuthKey.p8\"\n\
             api_key_id = \"TESTKEYID1\"\napi_issuer_id = \"issuer\"\n",
            device.udid(),
            "1122334455667788990011223344556677889900112233445566778899001122",
            FakeBucket::ACCESS_KEY_ID,
        );
        let config = Arc::new(Config::parse_str(
            &text,
            &dir.join("pipette-ios-proxy.toml"),
        )?);
        let (mut phone, _pool) = phone(device, dir)?;
        phone.config = Arc::clone(&config);
        Ok(Fleet {
            phones: vec![phone],
            locks: Vec::new(),
            config,
            state: StatePaths::resolve(Some(dir))?,
            stop: Arc::new(AtomicBool::new(false)),
        })
    }

    fn two_versions() -> FakeBucket {
        let bucket = FakeBucket::start();
        bucket.publish_builds(&[("a", "1.0", 1), ("b", "2.0", 2)], 1000);
        bucket
    }

    #[test]
    fn version_mode_advertises_the_bucket_and_installs_nothing() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let bucket = two_versions();
        let server = MockServer::start();
        let patch = server.mock(|when, then| {
            when.method(PATCH)
                .path("/clients/me")
                .body_includes(format!(
                    r#""capabilities":["client_version:1.0","client_version:2.0","{B10216_BUILD_FLAG}"]"#
                ));
            then.status(200).json_body(profile_body("approved", false));
        });
        let device = FakeDevice::new("phone-a");
        let calls = device.log();
        let fleet = version_fleet(
            dir.path(),
            device,
            &server.base_url(),
            &bucket.server.base_url(),
        )?;

        assert_eq!(
            probed_capabilities(&fleet)?,
            [
                "client_version:1.0",
                "client_version:2.0",
                B10216_BUILD_FLAG
            ]
        );
        register(fleet)?;

        patch.assert();
        assert_eq!(calls.calls()?, vec![]);
        Ok(())
    }

    #[test]
    fn scan_is_refused_in_version_mode() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let mut fleet = version_fleet(
            dir.path(),
            FakeDevice::new("phone-a"),
            "https://collector.example",
            "https://bucket.example",
        )?;

        let error = scan(&mut fleet).err().context("scan must be refused")?;

        assert!(
            matches!(
                error.downcast_ref::<ConfigError>(),
                Some(ConfigError::NoPool { verb: "scan", .. })
            ),
            "{error:#}"
        );
        Ok(())
    }

    /// One build in each of the three states, with records as `builds pull` leaves them.
    #[test]
    fn a_build_line_carries_every_column() -> anyhow::Result<()> {
        let state = tempfile::tempdir()?;
        let paths = StatePaths::resolve(Some(state.path()))?.releases();
        let pulled = "7f5693b172a03d00c59e0fdf0b5b9d490d7198f1dc031567cdc8010c16a975c4";
        let other = "1b2c3d4e5f60718293a4b5c6d7e8f9000102030405060708090a0b0c0d0e0f10";
        let builds = [
            (
                "2026.10.02-5c435ba-internal",
                "2026.10.02-5c435ba",
                pulled,
                110_000_000,
            ),
            ("feature-x.1", "feature-x", other, 1_048_576),
            (
                "2026.10.01-0000000-internal",
                "2026.10.01-0000000",
                other,
                99,
            ),
        ]
        .into_iter()
        .map(|(version, tag, sha256, bytes)| -> anyhow::Result<Build> {
            Ok(Build {
                sha256: Sha256::try_new(sha256)?,
                bytes,
                uploaded_at: datetime!(2026-10-02 05:14:15 +02:00),
                ..build(tag, version, b"")?
            })
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
        std::fs::create_dir_all(paths.dir.join("2026.10.02-5c435ba-internal"))?;
        std::fs::write(
            &paths.records,
            serde_json::json!({
                "schema": 1,
                "versions": {
                    "2026.10.02-5c435ba-internal": {
                        "sha256": pulled, "key": "k", "pulled_at": "2026-10-02T04:00:00Z",
                    },
                    "2026.10.01-0000000-internal": {
                        "sha256": pulled, "key": "k", "pulled_at": "2026-10-01T04:00:00Z",
                    },
                },
            })
            .to_string(),
        )?;

        let lines = build_lines(&builds, &store::Records::load(&paths)?, &paths);

        assert_eq!(
            lines,
            [
                "2026.10.02-5c435ba-internal  2026.10.02-5c435ba   104.9MiB  \
                 2026-10-02T03:14:15Z  7f5693b172a0  pulled",
                "feature-x.1                  feature-x              1.0MiB  \
                 2026-10-02T03:14:15Z  1b2c3d4e5f60  not-pulled",
                "2026.10.01-0000000-internal  2026.10.01-0000000     0.0MiB  \
                 2026-10-02T03:14:15Z  1b2c3d4e5f60  changed",
            ]
        );
        Ok(())
    }

    /// A 0600 config: [`MINIMAL`](crate::config::tests::MINIMAL) followed by
    /// `releases`.
    fn config_with(dir: &Path, releases: &str) -> anyhow::Result<Config> {
        let path = dir.join("pipette-ios-proxy.toml");
        std::fs::write(
            &path,
            format!("{}{releases}", crate::config::tests::MINIMAL),
        )?;
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600))?;
        Config::load(&path)
    }

    #[test]
    fn builds_is_refused_without_the_client_releases_table() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;

        let error = builds_with(
            &config_with(dir.path(), "")?,
            Some(dir.path()),
            &BuildsCommand::List,
            &mut Vec::new(),
        )
        .err()
        .context("there is no bucket to list")?;

        assert!(
            matches!(
                error.downcast_ref::<ConfigError>(),
                Some(ConfigError::NoClientReleases { .. })
            ),
            "{error:#}"
        );
        Ok(())
    }

    fn bucket_config(dir: &Path, bucket: &FakeBucket, default: &str) -> anyhow::Result<Config> {
        config_with(
            dir,
            &format!(
                "\n[client_releases]\nbucket = \"builds\"\nregion = \"us-east-1\"\n\
                 endpoint = \"{}\"\naccess_key_id = \"{}\"\nsecret_access_key = \"{SECRET}\"\n\
                 default = \"{default}\"\n",
                bucket.server.base_url(),
                FakeBucket::ACCESS_KEY_ID,
            ),
        )
    }

    /// Pulled and signed builds, scratch left by killed processes, and one
    /// phone on `gone-1`, which the bucket no longer publishes.
    fn stock(state: &Path) -> anyhow::Result<()> {
        [
            "releases/v01",
            "releases/v02",
            "releases/v22",
            "releases/gone-1",
            "releases/old-1",
            "releases/.partial-abc",
            "releases/Not-A-Version",
            "signed/v02",
            "signed/v03",
            "signed/gone-1",
            "signed/.partial-x",
            "PHONE-A/staged/Pipette.app",
        ]
        .iter()
        .try_for_each(|dir| std::fs::create_dir_all(state.join(dir)))?;
        std::fs::write(state.join("releases.json"), "{}")?;
        std::fs::write(state.join("PHONE-A/lock"), "")?;
        InstalledBuild {
            path: None,
            version: Some("gone-1".to_owned()),
        }
        .store(&state.join("PHONE-A/installed.json"))
    }

    /// The bucket's newest 20 of 22 builds are `v03` to `v22`; `v01` stays as
    /// the default and `gone-1` as what a phone runs.
    #[rstest]
    #[case::removes(false)]
    #[case::dry_run(true)]
    fn prune_keeps_the_catalog_and_what_phones_run(#[case] dry_run: bool) -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let state = dir.path().join("state");
        stock(&state)?;
        let bucket = FakeBucket::start();
        let versions: Vec<String> = (1..=22).map(|n| format!("v{n:02}")).collect();
        let published: Vec<(&str, &str, u8)> = versions
            .iter()
            .zip(1u8..)
            .map(|(version, day)| (version.as_str(), version.as_str(), day))
            .collect();
        bucket.publish_builds(&published, 1000);
        let mut out = Vec::new();

        builds_with(
            &bucket_config(dir.path(), &bucket, "v01")?,
            Some(&state),
            &BuildsCommand::Prune { dry_run },
            &mut out,
        )?;

        let removed = [
            "releases/.partial-abc",
            "releases/old-1",
            "releases/v02",
            "signed/.partial-x",
            "signed/v02",
        ];
        assert_eq!(
            String::from_utf8(out)?,
            removed
                .iter()
                .map(|path| format!("{}\n", state.join(path).display()))
                .collect::<String>()
        );
        removed
            .iter()
            .for_each(|path| assert_eq!(state.join(path).exists(), dry_run, "{path}"));
        [
            "releases/v01",
            "releases/v22",
            "releases/gone-1",
            "releases/Not-A-Version",
            "signed/v03",
            "signed/gone-1",
            "releases.json",
            "PHONE-A/installed.json",
            "PHONE-A/staged/Pipette.app",
        ]
        .iter()
        .for_each(|path| assert!(state.join(path).exists(), "{path}"));
        Ok(())
    }

    #[rstest]
    #[case::pulling("releases.lock")]
    #[case::signing("signed.lock")]
    #[case::driving_a_phone("PHONE-A/lock")]
    fn prune_refuses_while_another_process_holds_a_lock(#[case] lock: &str) -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let state = dir.path().join("state");
        stock(&state)?;
        let bucket = FakeBucket::start();
        let held = state.join(lock);
        let _other = StateLock::try_acquire(&held)?
            .map_err(|holder| anyhow::anyhow!("{lock} is busy: {holder}"))?;
        let mut out = Vec::new();

        let error = builds_with(
            &bucket_config(dir.path(), &bucket, "v01")?,
            Some(&state),
            &BuildsCommand::Prune { dry_run: false },
            &mut out,
        )
        .err()
        .context("a held lock refuses the prune")?;

        assert_eq!(
            error.to_string(),
            format!(
                "another pipette-ios-proxy is using {} (pid {} holds {}): stop every process \
                 using the state directory, then prune",
                state.display(),
                std::process::id(),
                held.display()
            )
        );
        assert!(out.is_empty());
        assert!(state.join("releases/old-1").is_dir());
        Ok(())
    }

    #[rstest]
    #[case::no_table("", "no [client_releases]; the `builds` verbs are off")]
    #[case::table(
        "\n[client_releases]\nbucket = \"releases\"\nregion = \"eu-west-1\"\n\
         access_key_id = \"AKIAIOSFODNN7EXAMPLE\"\nsecret_access_key = \"secret\"\n",
        "https://releases.s3.eu-west-1.amazonaws.com/ in eu-west-1, access key AKIAIOSFODNN7EXAMPLE"
    )]
    fn check_reports_the_releases_bucket(
        #[case] releases: &str,
        #[case] expected: &str,
    ) -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        assert_eq!(
            releases_field(&config_with(dir.path(), releases)?)?,
            expected
        );
        Ok(())
    }

    #[test]
    fn uninstalling_forgets_what_was_installed() -> anyhow::Result<()> {
        let state = tempfile::tempdir()?;
        let device = FakeDevice::new("phone-a").with_installed(BUNDLE);
        let log = device.log();
        let (mut phone, _pool) = phone(device, state.path())?;
        let a = state.path().join("builds/A.app");
        phone.set_installed(Some(&a))?;

        uninstall_one(&mut phone)?;

        assert_eq!(log.calls()?, vec![Call::Uninstall(BUNDLE.to_owned())]);
        assert_eq!(phone.installed, None);
        assert_eq!(InstalledBuild::load(&phone.paths.installed)?.path, None);
        Ok(())
    }
}
