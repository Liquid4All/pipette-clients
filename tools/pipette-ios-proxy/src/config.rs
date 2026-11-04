//! The TOML file, parsed and validated once. [`FileConfig`] is the raw text
//! (every table `deny_unknown_fields`); [`Config`] is the resolved value, with
//! every default and exclusivity rule decided in [`Config::parse_str`].

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::time::Duration;

use anyhow::Context;
use serde::Deserialize;
use tracing::level_filters::LevelFilter;
use url::Url;

use pipette_mgmt_client::types::PrivateKey;
use pipette_mgmt_client::{ClientIdentity, ServerUrl};
use pipette_plan_types::ClientId;

use crate::logging::LogFormat;
use crate::releases::BuildVersion;
use crate::releases::sign::Signer;
use crate::secret::{self, KeyKind, Secret};
use crate::timing;

const DEFAULT_CONFIG_FILE: &str = "pipette-ios-proxy.toml";

/// A config the proxy refuses, naming the file and the mistake.
#[derive(Debug, thiserror::Error)]
pub(crate) enum ConfigError {
    #[error(
        "{file} declares no [[devices]]: this worker acts on behalf of phones, and with none named \
         there is nothing for it to claim for"
    )]
    NoDevices { file: PathBuf },
    #[error(
        "[[devices]] #{entry} of {file} omits `udid`, which means \"the one connected phone\"; with \
         {count} devices configured every entry has to name its phone"
    )]
    UdidRequired {
        file: PathBuf,
        entry: usize,
        count: usize,
    },
    #[error(
        "{file} names the device `{udid}` in more than one [[devices]]: one phone cannot be driven by \
         two claim loops, which would install over each other mid-run"
    )]
    DuplicateUdid { file: PathBuf, udid: String },
    #[error(
        "{file} uses client_id `{client_id}` for more than one device: each phone has its own \
         registered client, and sharing one would file every result under a profile that \
         describes another phone"
    )]
    SharedClientId { file: PathBuf, client_id: String },
    #[error("{key} in {file} is empty")]
    Empty { file: PathBuf, key: String },
    #[error("{key} `{value}` in {file}: {message}")]
    Invalid {
        file: PathBuf,
        key: String,
        value: String,
        message: String,
    },
    #[error("{key} in {file} must be at least 1")]
    NotPositive { file: PathBuf, key: String },
    #[error(
        "[[devices]] entries `{}` in {file} resolve to the same phone ({phone}); one phone cannot be \
         driven by two claim loops",
        .entries.join("` and `")
    )]
    SamePhone {
        file: PathBuf,
        entries: Vec<String>,
        phone: String,
    },
    #[error(
        "--udid {selected} names no configured device; {file} declares [{}]",
        .known.join(", ")
    )]
    UnknownUdid {
        file: PathBuf,
        selected: String,
        known: Vec<String>,
    },
    #[error(
        "[client_releases] in {file} gives both secret_access_key and secret_access_key_file: one \
         of them is stale, and which one won would not show up anywhere, so give exactly one"
    )]
    TwoReleaseSecrets { file: PathBuf },
    #[error(
        "[client_releases] in {file} gives no secret: set secret_access_key_file to a 0600 file \
         holding the secret access key, or secret_access_key inline"
    )]
    NoReleaseSecret { file: PathBuf },
    #[error("{file} has no [client_releases] table, so there is no bucket to read app builds from")]
    NoClientReleases { file: PathBuf },
    #[error(
        "{file} has neither [builds] nor [signing]: give [builds] to install pre-signed builds \
         from a directory, or [signing] with [client_releases] to install the build each job pins"
    )]
    NoBuildSource { file: PathBuf },
    #[error(
        "{file} has both [builds] and [signing]: a config installs from one of them. To run some \
         phones each way, render two config files that share one --state-dir"
    )]
    TwoBuildSources { file: PathBuf },
    #[error(
        "[signing] in {file} needs [client_releases]: it signs the builds that bucket publishes"
    )]
    SigningWithoutReleases { file: PathBuf },
    #[error("{file} has no [signing] table: this proxy installs from its [builds] pool")]
    NoSigning { file: PathBuf },
    #[error(
        "{file} installs from the releases bucket ([signing]), so there is no [builds] pool to \
         {verb}; `builds list` prints the versions it can install"
    )]
    NoPool { file: PathBuf, verb: &'static str },
    #[error("[signing] keychain {keychain} in {file} is not a file")]
    NoKeychain { file: PathBuf, keychain: PathBuf },
    #[error("invalid config {file}, line {line}: {message}")]
    Syntax {
        file: PathBuf,
        line: usize,
        message: String,
    },
}

/// Environment override for the config path. An empty value counts as unset.
pub(crate) const CONFIG_PATH_ENV: &str = "PIPETTE_IOS_PROXY_CONFIG";

/// `[[devices]] bundle_id` default: the one id the iOS client ships under (no
/// `.debug` variant, unlike Android).
const DEFAULT_BUNDLE_ID: &str = "ai.liquid.liquid-pipette";

/// `[server] client_details` default; marks the row as proxied.
const DEFAULT_CLIENT_DETAILS: &str =
    "pipette-ios-proxy relaying {product_type} ({udid}) from {host}";

/// The validated configuration: paths `~`-expanded, durations typed, keys as
/// redacting newtypes. Deliberately small; everything knowable from the phone or
/// a build is read rather than declared.
#[derive(Debug, Clone)]
pub(crate) struct Config {
    pub(crate) server: ServerConfig,
    /// The phones this worker acts for, in file order. At least one; each with
    /// its OWN `client_id` and key, sharing every other table.
    pub(crate) devices: Vec<DeviceConfig>,
    pub(crate) builds: BuildSource,
    pub(crate) client_releases: Option<ClientReleases>,
    pub(crate) loop_config: LoopConfig,
    pub(crate) devicectl: DevicectlConfig,
    pub(crate) log: LogConfig,
    pub(crate) source: PathBuf,
    /// Whether any `[[devices]]` entry wrote its key or token INLINE, making this
    /// file the credential: [`Self::load`] then refuses it at any mode beyond `0600`.
    pub(crate) holds_a_key: bool,
}

#[derive(Debug, Clone)]
pub(crate) struct ServerConfig {
    pub(crate) url: ServerUrl,
    /// `client_details` template, rendered by `crate::profile::render_client_details`
    /// once the device is probed.
    pub(crate) client_details_template: String,
}

#[derive(Debug, Clone)]
pub(crate) struct DeviceConfig {
    /// `--device` argument when set, used VERBATIM: `devicectl` accepts a UDID,
    /// CoreDevice UUID, ECID, serial, name or DNS name. `None` means "require
    /// exactly one connected device".
    pub(crate) udid: Option<String>,
    pub(crate) device_name: Option<String>,
    pub(crate) bundle_id: String,
    /// The phone's own client, from `headlessrun auth export`.
    pub(crate) credentials: ClientCredentials,
}

#[derive(Debug, Clone)]
pub(crate) struct ClientCredentials {
    pub(crate) client_id: ClientId,
    /// Never persisted or logged; renders `<redacted>`.
    pub(crate) private_key: PrivateKey,
}

impl ClientCredentials {
    /// What requests for this phone are signed with. One value, so a caller
    /// cannot pair one phone's id with another's key (which signs fine and answers
    /// `401`, indistinguishable from clock drift). Adopted, never minted:
    /// `POST /clients/register` would split the phone's history.
    pub(crate) fn signer(&self) -> ClientIdentity {
        ClientIdentity {
            client_id: self.client_id.clone(),
            private_key_hex: self.private_key.clone(),
        }
    }
}

/// Where the builds a phone installs come from (DESIGN section 2): one per config.
#[derive(Debug, Clone)]
pub(crate) enum BuildSource {
    /// `[builds]`, "pool mode": pre-signed bundles, picked by the job's runtime.
    Pool(BuildsConfig),
    /// `[signing]`, "version mode": the `[client_releases]` archives, picked by
    /// the version a claim admits and signed on this Mac at install time.
    Releases(SigningConfig),
}

/// The pool of installable app builds, shared by every device. What each build
/// PROVIDES is read off the build (`crate::pool`), not declared.
#[derive(Debug, Clone)]
pub(crate) struct BuildsConfig {
    pub(crate) dir: PathBuf,
}

/// `[signing]` as loaded: paths resolved, no secret read yet ([`Config::signer`]).
#[derive(Debug, Clone)]
pub(crate) struct SigningConfig {
    pub(crate) team_id: String,
    pub(crate) keychain: PathBuf,
    pub(crate) keychain_password_file: PathBuf,
    pub(crate) api_key_file: PathBuf,
    pub(crate) api_key_id: String,
    pub(crate) api_issuer_id: String,
    pub(crate) export_timeout: Duration,
}

const DEFAULT_KEYCHAIN: &str = "~/Library/Keychains/login.keychain-db";

/// By absolute path, so the keychain password never goes to whichever
/// `security` comes first on `PATH`.
const SECURITY: &str = "/usr/bin/security";

const XCODEBUILD: &str = "/usr/bin/xcodebuild";

/// `[client_releases]` as loaded: its shape checked, its secret not read yet,
/// so a bad secret file stops only what reads the bucket, never pool mode's `run`.
#[derive(Debug, Clone)]
pub(crate) struct ClientReleases {
    url: Url,
    region: String,
    access_key_id: String,
    secret: ReleaseSecret,
    /// The version unpinned jobs install when a claim admits it. Read only in
    /// version mode.
    default: Option<BuildVersion>,
}

#[derive(Debug, Clone)]
enum ReleaseSecret {
    /// Written in the config, which is then the credential.
    Inline(Secret),
    File(PathBuf),
}

/// The bucket each release publishes the app builds to, and the read-only key
/// pair this proxy lists and reads it with.
#[derive(Debug, Clone)]
pub(crate) struct ReleasesBucket {
    /// Where the bucket's keys hang, ending in `/`: its own AWS host, or a host
    /// with the bucket as the first path segment.
    pub(crate) url: Url,
    pub(crate) region: String,
    pub(crate) access_key_id: String,
    pub(crate) secret_access_key: Secret,
}

#[derive(Debug, Clone)]
pub(crate) struct LoopConfig {
    pub(crate) idle_wait: Duration,
    pub(crate) idle_jitter: Duration,
    /// Heartbeat cadence, the override for `crate::timing::resolve_heartbeat_interval`.
    pub(crate) heartbeat: Duration,
    pub(crate) claim_backoff: Duration,
    pub(crate) max_jobs: u32,
}

#[derive(Debug, Clone)]
pub(crate) struct DevicectlConfig {
    /// `devicectl --timeout` on every parsed call, the proxy's own deadline for a
    /// verb's launch, which carries no `--timeout` (`crate::device::devicectl`),
    /// and the deadline for each GPU-pin CLI call (`crate::device::gpu_pin`).
    pub(crate) call_timeout: Duration,
    pub(crate) run_timeout: Duration,
    /// How long any launch, a run's or a verb's, may go without a line from the
    /// app before it is ended and tried once more (`crate::app::console::relaunch_if_silent`).
    pub(crate) first_line_timeout: Duration,
    /// When set, every phone's runs hold the GPU performance state through this
    /// CLI, and a run that cannot is a failure (`crate::device::gpu_pin`).
    pub(crate) pymobiledevice3: Option<PathBuf>,
}

#[derive(Debug, Clone)]
pub(crate) struct LogConfig {
    pub(crate) level: LevelFilter,
    pub(crate) format: LogFormat,
}

impl Default for LogConfig {
    fn default() -> Self {
        Self {
            level: LevelFilter::INFO,
            format: LogFormat::default(),
        }
    }
}

impl Default for LoopConfig {
    fn default() -> Self {
        Self {
            idle_wait: timing::DEFAULT_IDLE_WAIT,
            idle_jitter: timing::DEFAULT_IDLE_JITTER,
            heartbeat: timing::DEFAULT_HEARTBEAT,
            claim_backoff: timing::DEFAULT_CLAIM_BACKOFF,
            max_jobs: 0,
        }
    }
}

impl Default for DevicectlConfig {
    fn default() -> Self {
        Self {
            call_timeout: timing::DEFAULT_CALL_TIMEOUT,
            run_timeout: timing::DEFAULT_RUN_TIMEOUT,
            first_line_timeout: timing::DEFAULT_FIRST_LINE_TIMEOUT,
            pymobiledevice3: None,
        }
    }
}

impl Config {
    pub(crate) fn load(path: &Path) -> anyhow::Result<Self> {
        let text = fs::read_to_string(path)
            .with_context(|| format!("reading config {}", path.display()))?;
        let config = Self::parse_str(&text, path)?;
        // A config carrying a key IS the credential and gets a key file's rule.
        if config.holds_a_key {
            secret::refuse_beyond_owner(path, KeyKind::Device).with_context(|| {
                format!(
                    "{} holds a device's private_key inline, so its mode is the key's mode",
                    path.display()
                )
            })?;
        }
        Ok(config)
    }

    /// The parse and validate half of [`Self::load`]. `source` names the file in
    /// diagnostics and anchors its relative paths. Reads any key file.
    pub(crate) fn parse_str(text: &str, source: &Path) -> anyhow::Result<Self> {
        // toml's own rendering quotes the offending line, which can be a key's.
        let file: FileConfig = toml::from_str(text).map_err(|error| ConfigError::Syntax {
            file: source.to_path_buf(),
            line: error.span().map_or(1, |span| {
                text.get(..span.start)
                    .map_or(0, |head| head.matches('\n').count())
                    + 1
            }),
            message: error.message().trim().to_owned(),
        })?;

        // Relative paths resolve against the config file, not the cwd (launchd's is `/`).
        let base = source.parent().unwrap_or_else(|| Path::new(""));

        let url = ServerUrl::try_new(file.server.url.as_str())
            .with_context(|| format!("[server] url in {}", source.display()))?;

        if file.devices.is_empty() {
            return Err(ConfigError::NoDevices {
                file: source.to_path_buf(),
            }
            .into());
        }
        let devices = file
            .devices
            .iter()
            .enumerate()
            .map(|(index, device)| resolve_device(device, index, base, source))
            .collect::<anyhow::Result<Vec<_>>>()?;
        let holds_a_key = file
            .devices
            .iter()
            .any(|device| device.private_key.is_some() || device.identity_token.is_some());
        // An omitted `udid` means "the one connected phone", which only works
        // with one entry.
        if devices.len() > 1
            && let Some(index) = devices.iter().position(|d| d.udid.is_none())
        {
            return Err(ConfigError::UdidRequired {
                file: source.to_path_buf(),
                entry: index + 1,
                count: devices.len(),
            }
            .into());
        }
        // Two entries for one phone would be two claim loops installing over each
        // other. Same-spelling duplicates only; `worker::prepare` catches two
        // spellings of one phone after the probe.
        if let Some(dup) = first_duplicate(devices.iter().filter_map(|d| d.udid.as_deref())) {
            return Err(ConfigError::DuplicateUdid {
                file: source.to_path_buf(),
                udid: dup.clone(),
            }
            .into());
        }
        // Two phones signing as one client: the profile would flip between them.
        if let Some(dup) = first_duplicate(devices.iter().map(|d| d.credentials.client_id.as_ref()))
        {
            return Err(ConfigError::SharedClientId {
                file: source.to_path_buf(),
                client_id: dup.clone(),
            }
            .into());
        }

        let client_releases = file
            .client_releases
            .as_ref()
            .map(|table| resolve_client_releases(table, base, source))
            .transpose()?;

        let file_path = || source.to_path_buf();
        let builds = match (&file.builds, &file.signing, &client_releases) {
            (Some(table), None, _) => BuildSource::Pool(BuildsConfig {
                dir: resolve_relative(
                    base,
                    required_text(&table.dir, "[builds] dir", source)?.as_str(),
                ),
            }),
            (None, Some(table), Some(_)) => {
                BuildSource::Releases(resolve_signing(table, base, source)?)
            }
            (None, Some(_), None) => {
                return Err(ConfigError::SigningWithoutReleases { file: file_path() }.into());
            }
            (Some(_), Some(_), _) => {
                return Err(ConfigError::TwoBuildSources { file: file_path() }.into());
            }
            (None, None, _) => {
                return Err(ConfigError::NoBuildSource { file: file_path() }.into());
            }
        };

        let defaults = LoopConfig::default();
        let loop_config = LoopConfig {
            // Zero is allowed for the two idle knobs.
            idle_wait: file
                .loop_table
                .idle_wait_secs
                .map_or(defaults.idle_wait, Duration::from_secs),
            idle_jitter: file
                .loop_table
                .idle_jitter_secs
                .map_or(defaults.idle_jitter, Duration::from_secs),
            heartbeat: positive_secs(
                file.loop_table.heartbeat_secs,
                defaults.heartbeat,
                "[loop] heartbeat_secs",
                source,
            )?,
            claim_backoff: positive_secs(
                file.loop_table.claim_backoff_secs,
                defaults.claim_backoff,
                "[loop] claim_backoff_secs",
                source,
            )?,
            max_jobs: file.loop_table.max_jobs.unwrap_or(defaults.max_jobs),
        };

        let devicectl_defaults = DevicectlConfig::default();
        let devicectl = DevicectlConfig {
            call_timeout: positive_secs(
                file.devicectl.call_timeout_secs,
                devicectl_defaults.call_timeout,
                "[devicectl] call_timeout_secs",
                source,
            )?,
            run_timeout: positive_secs(
                file.devicectl.run_timeout_secs,
                devicectl_defaults.run_timeout,
                "[devicectl] run_timeout_secs",
                source,
            )?,
            first_line_timeout: positive_secs(
                file.devicectl.first_line_timeout_secs,
                devicectl_defaults.first_line_timeout,
                "[devicectl] first_line_timeout_secs",
                source,
            )?,
            // A bare name is looked up on PATH; a path is the config file's, like the others.
            pymobiledevice3: trimmed(file.devicectl.pymobiledevice3.as_deref()).map(|raw| {
                if raw.contains('/') {
                    resolve_relative(base, &raw)
                } else {
                    PathBuf::from(raw)
                }
            }),
        };

        let log = LogConfig {
            level: parse_value::<LevelFilter>(
                file.log.level.as_deref(),
                "[log] level",
                source,
                LevelFilter::INFO,
            )?,
            format: parse_value::<LogFormat>(
                file.log.format.as_deref(),
                "[log] format",
                source,
                LogFormat::default(),
            )?,
        };

        Ok(Self {
            server: ServerConfig {
                url,
                client_details_template: match file.server.client_details.as_deref() {
                    Some(value) if !value.trim().is_empty() => value.to_owned(),
                    _ => DEFAULT_CLIENT_DETAILS.to_owned(),
                },
            },
            devices,
            builds,
            client_releases,
            loop_config,
            devicectl,
            log,
            source: source.to_path_buf(),
            holds_a_key,
        })
    }

    /// The releases bucket, with its secret, which is read here and only here.
    pub(crate) fn releases_bucket(&self) -> anyhow::Result<ReleasesBucket> {
        let table = self
            .client_releases
            .as_ref()
            .ok_or_else(|| ConfigError::NoClientReleases {
                file: self.source.clone(),
            })?;
        let secret_access_key = match &table.secret {
            ReleaseSecret::Inline(secret) => {
                secret::refuse_beyond_owner(&self.source, KeyKind::AccessKey).with_context(
                    || {
                        format!(
                            "{} holds [client_releases] secret_access_key inline, so its mode \
                             is the key's mode",
                            self.source.display()
                        )
                    },
                )?;
                secret.clone()
            }
            ReleaseSecret::File(path) => {
                let secret = Secret::from(
                    secret::read_secret_file(path, KeyKind::AccessKey).with_context(|| {
                        format!(
                            "[client_releases] secret_access_key_file in {}",
                            self.source.display()
                        )
                    })?,
                );
                if secret.expose().is_empty() {
                    return Err(ConfigError::Empty {
                        file: self.source.clone(),
                        key: format!(
                            "[client_releases] secret_access_key_file {}",
                            path.display()
                        ),
                    }
                    .into());
                }
                secret
            }
        };
        Ok(ReleasesBucket {
            url: table.url.clone(),
            region: table.region.clone(),
            access_key_id: table.access_key_id.clone(),
            secret_access_key,
        })
    }

    /// Reads the keychain password and mode-checks both secret files, here and
    /// only here, so only `run` and `check` touch them.
    pub(crate) fn signer(&self) -> anyhow::Result<Signer> {
        let BuildSource::Releases(signing) = &self.builds else {
            return Err(ConfigError::NoSigning {
                file: self.source.clone(),
            }
            .into());
        };
        let at = |key: &str| format!("[signing] {key} in {}", self.source.display());
        let password = secret::read_password_file(&signing.keychain_password_file)
            .with_context(|| at("keychain_password_file"))?;
        secret::refuse_beyond_owner(&signing.api_key_file, KeyKind::ApiKey)
            .with_context(|| at("api_key_file"))?;
        if !signing.keychain.is_file() {
            return Err(ConfigError::NoKeychain {
                file: self.source.clone(),
                keychain: signing.keychain.clone(),
            }
            .into());
        }
        Ok(Signer {
            team_id: signing.team_id.clone(),
            keychain: signing.keychain.clone(),
            password,
            api_key_file: signing.api_key_file.clone(),
            api_key_id: signing.api_key_id.clone(),
            api_issuer_id: signing.api_issuer_id.clone(),
            export_timeout: signing.export_timeout,
            security: PathBuf::from(SECURITY),
            xcodebuild: PathBuf::from(XCODEBUILD),
        })
    }

    pub(crate) fn default_version(&self) -> Option<&BuildVersion> {
        self.client_releases
            .as_ref()
            .and_then(|releases| releases.default.as_ref())
    }
}

/// `$PIPETTE_IOS_PROXY_CONFIG`; non-UTF-8 counts as unset. Read here rather than
/// via clap `env`, whose `PathBuf` parser would refuse an empty value.
pub(crate) fn config_path_env() -> Option<String> {
    std::env::var_os(CONFIG_PATH_ENV).and_then(|value| value.into_string().ok())
}

/// Config path resolution: `--config <p>`, then `$PIPETTE_IOS_PROXY_CONFIG`, then
/// `./pipette-ios-proxy.toml`. An empty or blank env value counts as unset.
pub(crate) fn resolve_config_path(flag: Option<&Path>, env_value: Option<&str>) -> PathBuf {
    if let Some(path) = flag {
        return path.to_path_buf();
    }
    if let Some(raw) = env_value.map(str::trim).filter(|v| !v.is_empty()) {
        return PathBuf::from(raw);
    }
    PathBuf::from(DEFAULT_CONFIG_FILE)
}

/// Expand a leading `~` or `~/…` against `$HOME`. `~user` and an unset `$HOME`
/// leave the path literal.
fn expand_tilde(raw: &str) -> PathBuf {
    let trimmed = raw.trim();
    let home = || std::env::var_os("HOME").filter(|value| !value.is_empty());
    if trimmed == "~" {
        return home().map_or_else(|| PathBuf::from(trimmed), PathBuf::from);
    }
    if let Some(rest) = trimmed.strip_prefix("~/") {
        return match home() {
            Some(home) => PathBuf::from(home).join(rest),
            None => PathBuf::from(trimmed),
        };
    }
    PathBuf::from(trimmed)
}

/// One `[[devices]]` entry, validated. `index` names it in diagnostics.
fn resolve_device(
    table: &DeviceTable,
    index: usize,
    base: &Path,
    source: &Path,
) -> anyhow::Result<DeviceConfig> {
    let ordinal = index + 1;
    let at = |what: &str| format!("{what} in [[devices]] #{ordinal} of {}", source.display());

    let key_file = table
        .private_key_file
        .as_deref()
        .map(|raw| resolve_relative(base, raw));
    let token_file = table
        .identity_token_file
        .as_deref()
        .map(|raw| resolve_relative(base, raw));
    let (client_id, private_key) = secret::resolve_credentials(secret::CredentialFields {
        client_id: table.client_id.as_deref(),
        private_key: table.private_key.as_deref(),
        private_key_file: key_file.as_deref(),
        identity_token: table.identity_token.as_deref(),
        identity_token_file: token_file.as_deref(),
    })
    .with_context(|| at("the credentials"))?;

    Ok(DeviceConfig {
        udid: trimmed(table.udid.as_deref()),
        device_name: trimmed(table.device_name.as_deref()),
        bundle_id: trimmed(table.bundle_id.as_deref())
            .unwrap_or_else(|| DEFAULT_BUNDLE_ID.to_owned()),
        credentials: ClientCredentials {
            client_id,
            private_key,
        },
    })
}

/// `[client_releases]`, its shape validated.
fn resolve_client_releases(
    table: &ClientReleasesTable,
    base: &Path,
    source: &Path,
) -> anyhow::Result<ClientReleases> {
    let bucket = s3_name(&table.bucket, "[client_releases] bucket", source)?;
    let region = s3_name(&table.region, "[client_releases] region", source)?;
    let url = match trimmed(table.endpoint.as_deref()) {
        Some(endpoint) => bucket_url_at(&endpoint, &bucket, source)?,
        // AWS's certificate covers one label below its host, so a bucket name
        // with a `.` is reached path-style.
        None if bucket.contains('.') => {
            Url::parse(&format!("https://s3.{region}.amazonaws.com/{bucket}/"))
                .with_context(|| format!("[client_releases] bucket in {}", source.display()))?
        }
        None => Url::parse(&format!("https://{bucket}.s3.{region}.amazonaws.com/"))
            .with_context(|| format!("[client_releases] bucket in {}", source.display()))?,
    };

    let secret = match (&table.secret_access_key, &table.secret_access_key_file) {
        (Some(inline), None) if inline.expose().is_empty() => {
            return Err(ConfigError::Empty {
                file: source.to_path_buf(),
                key: "[client_releases] secret_access_key".to_owned(),
            }
            .into());
        }
        (Some(inline), None) => ReleaseSecret::Inline(inline.clone()),
        (None, Some(raw)) => ReleaseSecret::File(resolve_relative(
            base,
            &required_text(raw, "[client_releases] secret_access_key_file", source)?,
        )),
        (Some(_), Some(_)) => {
            return Err(ConfigError::TwoReleaseSecrets {
                file: source.to_path_buf(),
            }
            .into());
        }
        (None, None) => {
            return Err(ConfigError::NoReleaseSecret {
                file: source.to_path_buf(),
            }
            .into());
        }
    };

    let default = trimmed(table.default.as_deref())
        .map(|raw| {
            raw.parse::<BuildVersion>()
                .map_err(|error| ConfigError::Invalid {
                    file: source.to_path_buf(),
                    key: "[client_releases] default".to_owned(),
                    value: raw.clone(),
                    message: error.to_string(),
                })
        })
        .transpose()?;

    Ok(ClientReleases {
        url,
        region,
        access_key_id: required_text(
            &table.access_key_id,
            "[client_releases] access_key_id",
            source,
        )?,
        secret,
        default,
    })
}

/// `[signing]`, its shape validated; its files are read by [`Config::signer`].
fn resolve_signing(
    table: &SigningTable,
    base: &Path,
    source: &Path,
) -> anyhow::Result<SigningConfig> {
    let invalid = |key: &str, value: &str, message: &str| ConfigError::Invalid {
        file: source.to_path_buf(),
        key: key.to_owned(),
        value: value.to_owned(),
        message: message.to_owned(),
    };
    let team_id = required_text(&table.team_id, "[signing] team_id", source)?;
    if team_id.len() != TEAM_ID_CHARS
        || !team_id
            .chars()
            .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit())
    {
        return Err(invalid(
            "[signing] team_id",
            &team_id,
            "an Apple team id is 10 characters of A-Z and 0-9",
        )
        .into());
    }
    let keychain = resolve_relative(
        base,
        trimmed(table.keychain.as_deref())
            .as_deref()
            .unwrap_or(DEFAULT_KEYCHAIN),
    );
    let keychain_text = keychain.display().to_string();
    if keychain_text.contains(['"', '\\', '\r', '\n']) {
        return Err(invalid(
            "[signing] keychain",
            &keychain_text,
            "the `security -i` line cannot carry a double quote, a backslash or a line break",
        )
        .into());
    }
    let file = |raw: &str, key: &str| -> anyhow::Result<PathBuf> {
        Ok(resolve_relative(base, &required_text(raw, key, source)?))
    };
    Ok(SigningConfig {
        team_id,
        keychain,
        keychain_password_file: file(
            &table.keychain_password_file,
            "[signing] keychain_password_file",
        )?,
        api_key_file: file(&table.api_key_file, "[signing] api_key_file")?,
        api_key_id: required_text(&table.api_key_id, "[signing] api_key_id", source)?,
        api_issuer_id: required_text(&table.api_issuer_id, "[signing] api_issuer_id", source)?,
        export_timeout: positive_secs(
            table.export_timeout_secs,
            timing::DEFAULT_EXPORT_TIMEOUT,
            "[signing] export_timeout_secs",
            source,
        )?,
    })
}

const TEAM_ID_CHARS: usize = 10;

/// A bucket or a region. Both land in a host name and in the signed scope, so
/// only what S3 itself allows in a bucket name gets through.
fn s3_name(raw: &str, key: &str, source: &Path) -> anyhow::Result<String> {
    let name = required_text(raw, key, source)?;
    let allowed = |c: char| c.is_ascii_lowercase() || c.is_ascii_digit() || "-.".contains(c);
    if name.chars().all(allowed) && !name.starts_with(['-', '.']) {
        return Ok(name);
    }
    Err(ConfigError::Invalid {
        file: source.to_path_buf(),
        key: key.to_owned(),
        value: name,
        message: "only lowercase letters, digits, `-` and `.`, starting with a letter or digit"
            .to_owned(),
    }
    .into())
}

/// The bucket under `endpoint`, path-style. The endpoint is a bare origin: what
/// follows it is the bucket and the key, and its query is the signature's.
fn bucket_url_at(endpoint: &str, bucket: &str, source: &Path) -> anyhow::Result<Url> {
    let invalid = |message: String| ConfigError::Invalid {
        file: source.to_path_buf(),
        key: "[client_releases] endpoint".to_owned(),
        value: without_credentials(endpoint),
        message,
    };
    let url = Url::parse(endpoint).map_err(|error| invalid(error.to_string()))?;
    let is_origin = matches!(url.scheme(), "http" | "https")
        && url.has_host()
        && url.username().is_empty()
        && url.password().is_none()
        && url.path() == "/"
        && url.query().is_none()
        && url.fragment().is_none();
    if !is_origin {
        return Err(invalid(
            "expected http(s)://host[:port], with no credentials, path, query or fragment"
                .to_owned(),
        )
        .into());
    }
    url.join(&format!("{bucket}/"))
        .map_err(|error| invalid(error.to_string()).into())
}

/// `endpoint` as a refusal may echo it: without the user, password, query or
/// fragment, any of which can hold a credential.
fn without_credentials(endpoint: &str) -> String {
    use url::Position;

    match Url::parse(endpoint) {
        Ok(url) if url.has_host() => format!(
            "{}://{}",
            &url[..Position::AfterScheme],
            &url[Position::BeforeHost..Position::AfterPath]
        ),
        // No host to anchor on, as in `user:pass@host` without a scheme.
        Ok(_) | Err(_) => endpoint
            .rsplit_once('@')
            .map_or(endpoint, |(_, after)| after)
            .split(['?', '#'])
            .next()
            .unwrap_or_default()
            .to_owned(),
    }
}

fn trimmed(value: Option<&str>) -> Option<String> {
    value
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
}

pub(crate) fn first_duplicate<'a>(mut values: impl Iterator<Item = &'a str>) -> Option<String> {
    let mut seen = HashSet::new();
    values.find(|value| !seen.insert(*value)).map(str::to_owned)
}

/// `deny_unknown_fields` on every table: a mistyped key must not run at a default.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FileConfig {
    server: ServerTable,
    /// `[[devices]]`, one table per phone. Required and non-empty.
    devices: Vec<DeviceTable>,
    /// Exactly one of `builds` and `signing`.
    builds: Option<BuildsTable>,
    signing: Option<SigningTable>,
    client_releases: Option<ClientReleasesTable>,
    #[serde(default, rename = "loop")]
    loop_table: LoopTable,
    #[serde(default)]
    devicectl: DevicectlTable,
    #[serde(default)]
    log: LogTable,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ServerTable {
    url: String,
    client_details: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DeviceTable {
    udid: Option<String>,
    device_name: Option<String>,
    bundle_id: Option<String>,
    client_id: Option<String>,
    private_key: Option<String>,
    private_key_file: Option<String>,
    identity_token: Option<String>,
    identity_token_file: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct BuildsTable {
    dir: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ClientReleasesTable {
    bucket: String,
    region: String,
    endpoint: Option<String>,
    access_key_id: String,
    secret_access_key: Option<Secret>,
    secret_access_key_file: Option<String>,
    default: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SigningTable {
    team_id: String,
    keychain: Option<String>,
    keychain_password_file: String,
    api_key_file: String,
    api_key_id: String,
    api_issuer_id: String,
    export_timeout_secs: Option<u64>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct LoopTable {
    idle_wait_secs: Option<u64>,
    idle_jitter_secs: Option<u64>,
    heartbeat_secs: Option<u64>,
    claim_backoff_secs: Option<u64>,
    max_jobs: Option<u32>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct DevicectlTable {
    call_timeout_secs: Option<u64>,
    run_timeout_secs: Option<u64>,
    first_line_timeout_secs: Option<u64>,
    pymobiledevice3: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct LogTable {
    level: Option<String>,
    format: Option<String>,
}

/// `~`-expand, then rebase a still-relative path on the config file's directory.
fn resolve_relative(base: &Path, raw: &str) -> PathBuf {
    let expanded = expand_tilde(raw);
    if expanded.is_absolute() {
        expanded
    } else {
        base.join(expanded)
    }
}

/// Blank is the same mistake as absent: `dir = ""` would read the pool from the
/// config file's own directory.
fn required_text(raw: &str, key: &str, source: &Path) -> anyhow::Result<String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err(ConfigError::Empty {
            file: source.to_path_buf(),
            key: key.to_owned(),
        }
        .into());
    }
    Ok(trimmed.to_owned())
}

/// Run one TOML value through the same `FromStr` the matching CLI flag uses.
fn parse_value<T: FromStr>(
    raw: Option<&str>,
    key: &str,
    source: &Path,
    default: T,
) -> anyhow::Result<T>
where
    T::Err: std::fmt::Display,
{
    match raw {
        Some(value) => T::from_str(value.trim()).map_err(|message| {
            ConfigError::Invalid {
                file: source.to_path_buf(),
                key: key.to_owned(),
                value: value.to_owned(),
                message: message.to_string(),
            }
            .into()
        }),
        None => Ok(default),
    }
}

/// A duration that must not be zero (`heartbeat_secs = 0` spins;
/// `call_timeout_secs = 0` is an expired `devicectl --timeout`).
fn positive_secs(
    raw: Option<u64>,
    default: Duration,
    key: &str,
    source: &Path,
) -> anyhow::Result<Duration> {
    match raw {
        Some(0) => Err(ConfigError::NotPositive {
            file: source.to_path_buf(),
            key: key.to_owned(),
        }
        .into()),
        Some(secs) => Ok(Duration::from_secs(secs)),
        None => Ok(default),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use anyhow::Context;
    use rstest::rstest;

    use super::*;

    /// The shipped example, compiled in so it cannot drift from the parser.
    const EXAMPLE: &str = include_str!("../pipette-ios-proxy.toml.example");

    const KEY: &str = "1122334455667788990011223344556677889900112233445566778899001122";

    /// Every required key and nothing else.
    pub(crate) const MINIMAL: &str = r#"
[server]
url = "https://collector.pipette.liquid.ai"

[[devices]]
udid = "00008130-0011223344556677"
client_id = "ev1_9f2c1d3e4b5a6c7d"
private_key = "1122334455667788990011223344556677889900112233445566778899001122"

[builds]
dir = "./builds"
"#;

    fn source() -> &'static Path {
        Path::new("pipette-ios-proxy.toml")
    }

    fn pool_dir(config: &Config) -> anyhow::Result<&Path> {
        match &config.builds {
            BuildSource::Pool(pool) => Ok(&pool.dir),
            BuildSource::Releases(_) => anyhow::bail!("a pool config read as version mode"),
        }
    }

    #[test]
    fn the_shipped_example_parses_to_the_documented_values() -> anyhow::Result<()> {
        let config = Config::parse_str(EXAMPLE, source())?;

        assert_eq!(
            config.server.url.to_string(),
            "https://collector.pipette.liquid.ai"
        );
        assert_eq!(
            config.server.client_details_template,
            DEFAULT_CLIENT_DETAILS
        );
        assert_eq!(config.devices.len(), 2);
        assert_eq!(
            config.devices[0].udid.as_deref(),
            Some("00008130-0011223344556677")
        );
        assert_eq!(
            config.devices[0].credentials.client_id.as_ref(),
            "ev1_9f2c1d3e4b5a6c7d"
        );
        assert_eq!(
            config.devices[1].credentials.client_id.as_ref(),
            "ev1_1a2b3c4d5e6f7a8b"
        );
        assert_ne!(
            config.devices[0].credentials.client_id.as_ref(),
            config.devices[1].credentials.client_id.as_ref(),
            "each device signs as its own client"
        );
        assert_eq!(config.devices[0].device_name, None);
        assert_eq!(config.devices[0].bundle_id, DEFAULT_BUNDLE_ID);
        assert_eq!(pool_dir(&config)?, Path::new("./builds"));
        let releases = config
            .client_releases
            .context("the example has the table")?;
        assert_eq!(
            releases.url.as_str(),
            "https://example-client-releases.s3.us-east-1.amazonaws.com/"
        );
        assert_eq!(releases.region, "us-east-1");
        assert_eq!(releases.access_key_id, "AKIAIOSFODNN7EXAMPLE");
        assert!(
            matches!(&releases.secret, ReleaseSecret::File(path) if path == Path::new("client-releases.secret")),
            "{:?}",
            releases.secret
        );
        assert_eq!(config.loop_config.idle_wait, Duration::from_secs(30));
        assert_eq!(config.loop_config.idle_jitter, Duration::from_secs(10));
        assert_eq!(config.loop_config.heartbeat, Duration::from_secs(20));
        assert_eq!(config.loop_config.claim_backoff, Duration::from_secs(5));
        assert_eq!(config.loop_config.max_jobs, 0);
        assert_eq!(config.devicectl.call_timeout, Duration::from_secs(60));
        assert_eq!(config.devicectl.run_timeout, Duration::from_secs(1800));
        assert_eq!(config.devicectl.first_line_timeout, Duration::from_secs(60));
        assert_eq!(config.log.level, LevelFilter::INFO);
        assert_eq!(config.log.format, LogFormat::Text);
        Ok(())
    }

    fn with_second_device(udid: &str, client_id: &str, key: &str) -> String {
        format!(
            "{MINIMAL}\n[[devices]]\nudid = \"{udid}\"\nclient_id = \"{client_id}\"\n\
             private_key = \"{key}\"\n"
        )
    }

    #[rstest]
    #[case::unknown_key(format!("{MINIMAL}\n[loop]\nidle_wait_sec = 30\n"), "idle_wait_sec")]
    #[case::unknown_table(format!("{MINIMAL}\n[logging]\nlevel = \"info\"\n"), "logging")]
    #[case::missing_required_key(MINIMAL.replace("dir = \"./builds\"", ""), "dir")]
    #[case::url_with_a_query(
        MINIMAL.replace(
            "url = \"https://collector.pipette.liquid.ai\"",
            "url = \"https://collector.pipette.liquid.ai/?org=liquid\"",
        ),
        "[server] url"
    )]
    #[case::one_phone_named_twice(
        with_second_device("00008130-0011223344556677", "ev1_other", KEY),
        "00008130-0011223344556677"
    )]
    #[case::two_phones_sharing_a_client(
        with_second_device("00008140-0011223344556677", "ev1_9f2c1d3e4b5a6c7d", KEY),
        "ev1_9f2c1d3e4b5a6c7d"
    )]
    #[case::udid_omitted_beside_another(
        format!("{MINIMAL}\n[[devices]]\nclient_id = \"ev1_other\"\nprivate_key = \"{KEY}\"\n"),
        "[[devices]] #2"
    )]
    #[case::no_devices(
        "[server]\nurl = \"https://collector.pipette.liquid.ai\"\n\n[builds]\ndir = \"./builds\"\n"
            .to_string(),
        "devices"
    )]
    fn a_bad_config_is_refused_naming_the_mistake(
        #[case] text: String,
        #[case] needle: &str,
    ) -> anyhow::Result<()> {
        let error = Config::parse_str(&text, source())
            .err()
            .context("the config must be refused")?;
        let message = format!("{error:#}");
        assert!(
            message.contains(needle),
            "the message must name `{needle}`: {message}"
        );
        Ok(())
    }

    #[test]
    fn a_trailing_comment_with_a_quote_and_utf8_parses() -> anyhow::Result<()> {
        let text = MINIMAL.replace(
            "dir = \"./builds\"",
            "dir = \"./builds\"   # naïve \"quoted\" note, ünicode ok",
        );
        let config = Config::parse_str(&text, source())?;
        assert_eq!(pool_dir(&config)?, Path::new("./builds"));
        Ok(())
    }

    #[test]
    fn the_server_url_keeps_its_path_prefix_and_loses_its_trailing_slash() -> anyhow::Result<()> {
        let text = MINIMAL.replace(
            "url = \"https://collector.pipette.liquid.ai\"",
            "url = \"  https://collector.pipette.liquid.ai/api/  \"",
        );
        let config = Config::parse_str(&text, source())?;
        assert_eq!(
            config.server.url.to_string(),
            "https://collector.pipette.liquid.ai/api"
        );

        let bare = MINIMAL.replace(
            "url = \"https://collector.pipette.liquid.ai\"",
            "url = \"https://collector.pipette.liquid.ai/\"",
        );
        assert_eq!(
            Config::parse_str(&bare, source())?.server.url.to_string(),
            "https://collector.pipette.liquid.ai"
        );

        // A surviving slash would sign and send `//plans/claim`.
        let doubled = MINIMAL.replace(
            "url = \"https://collector.pipette.liquid.ai\"",
            "url = \"https://collector.pipette.liquid.ai/api//\"",
        );
        assert_eq!(
            Config::parse_str(&doubled, source())?
                .server
                .url
                .to_string(),
            "https://collector.pipette.liquid.ai/api"
        );
        Ok(())
    }

    #[test]
    fn each_device_carries_its_own_identity() -> anyhow::Result<()> {
        let other = "aa11bb22cc33dd44ee55ff6600778899aa11bb22cc33dd44ee55ff6600778899";
        let text = with_second_device("00008140-0011223344556677", "ev1_1a2b3c4d5e6f7a8b", other);
        let config = Config::parse_str(&text, source())?;
        assert_eq!(config.devices.len(), 2);
        assert_ne!(
            config.devices[0].credentials.client_id.as_ref(),
            config.devices[1].credentials.client_id.as_ref()
        );
        assert!(
            config
                .devices
                .iter()
                .all(|device| device.credentials.private_key.to_string() == crate::secret::REDACTED)
        );
        Ok(())
    }

    #[test]
    fn a_lone_device_may_omit_its_udid() -> anyhow::Result<()> {
        let text = MINIMAL.replace("udid = \"00008130-0011223344556677\"\n", "");
        let config = Config::parse_str(&text, source())?;
        assert_eq!(config.devices[0].udid, None);
        Ok(())
    }

    #[test]
    fn a_config_holding_a_key_inline_must_be_owner_only() -> anyhow::Result<()> {
        use std::os::unix::fs::PermissionsExt;

        let dir = tempfile::tempdir().context("a temp dir")?;
        let path = dir.path().join("pipette-ios-proxy.toml");
        fs::write(&path, MINIMAL).context("writing the config")?;

        fs::set_permissions(&path, fs::Permissions::from_mode(0o600)).context("tightening")?;
        let config = Config::load(&path)?;
        assert!(config.holds_a_key);

        fs::set_permissions(&path, fs::Permissions::from_mode(0o644)).context("loosening")?;
        let error = Config::load(&path)
            .err()
            .context("a readable config holding a key must be refused")?;
        let message = format!("{error:#}");
        assert!(
            message.contains("private_key inline") && message.contains("0644"),
            "{message}"
        );
        Ok(())
    }

    #[test]
    fn a_config_pointing_at_a_key_file_may_be_readable() -> anyhow::Result<()> {
        use std::os::unix::fs::PermissionsExt;

        let dir = tempfile::tempdir().context("a temp dir")?;
        let key_path = dir.path().join("phone.key");
        fs::write(&key_path, KEY).context("writing the key")?;
        fs::set_permissions(&key_path, fs::Permissions::from_mode(0o600)).context("key mode")?;

        let text = MINIMAL.replace(
            &format!("private_key = \"{KEY}\""),
            &format!("private_key_file = \"{}\"", key_path.display()),
        );
        let path = dir.path().join("pipette-ios-proxy.toml");
        fs::write(&path, text).context("writing the config")?;
        fs::set_permissions(&path, fs::Permissions::from_mode(0o644)).context("config mode")?;

        let config = Config::load(&path)?;
        assert!(!config.holds_a_key);
        Ok(())
    }

    const SECRET: &str = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY";

    fn with_releases() -> String {
        format!(
            "{MINIMAL}\n[client_releases]\nbucket = \"example-client-releases\"\n\
             region = \"us-east-1\"\naccess_key_id = \"AKIAIOSFODNN7EXAMPLE\"\n\
             secret_access_key = \"{SECRET}\"\n"
        )
    }

    #[test]
    fn without_the_table_the_bucket_is_refused_by_name() -> anyhow::Result<()> {
        let config = Config::parse_str(MINIMAL, source())?;
        assert!(config.client_releases.is_none());
        let error = config
            .releases_bucket()
            .err()
            .context("there is no bucket")?;
        assert_eq!(
            error.to_string(),
            "pipette-ios-proxy.toml has no [client_releases] table, so there is no bucket to read \
             app builds from"
        );
        Ok(())
    }

    #[rstest]
    #[case::aws(
        "example-client-releases",
        "",
        "https://example-client-releases.s3.us-east-1.amazonaws.com/"
    )]
    #[case::dotted_bucket_at_aws(
        "releases.liquid.ai",
        "",
        "https://s3.us-east-1.amazonaws.com/releases.liquid.ai/"
    )]
    #[case::endpoint(
        "example-client-releases",
        "endpoint = \" http://127.0.0.1:9000/ \"\n",
        "http://127.0.0.1:9000/example-client-releases/"
    )]
    #[case::dotted_bucket_at_an_endpoint(
        "releases.liquid.ai",
        "endpoint = \"https://minio.local:9000\"\n",
        "https://minio.local:9000/releases.liquid.ai/"
    )]
    fn the_bucket_is_addressed_at_aws_or_under_the_endpoint(
        #[case] bucket: &str,
        #[case] endpoint: &str,
        #[case] url: &str,
    ) -> anyhow::Result<()> {
        let text = format!(
            "{}{endpoint}",
            with_releases().replace("example-client-releases", bucket)
        );
        let releases = Config::parse_str(&text, source())?
            .client_releases
            .context("the table is there")?;
        assert_eq!(releases.url.as_str(), url);
        assert_eq!(releases.region, "us-east-1");
        assert_eq!(releases.access_key_id, "AKIAIOSFODNN7EXAMPLE");
        Ok(())
    }

    #[test]
    fn the_secret_does_not_print_with_the_config() -> anyhow::Result<()> {
        let config = Config::parse_str(&with_releases(), source())?;
        let printed = format!("{config:?}");
        assert!(!printed.contains(SECRET), "{printed}");
        assert!(printed.contains("secret: Inline(<redacted>)"), "{printed}");
        Ok(())
    }

    /// Quoting the line would echo the key.
    #[test]
    fn a_misspelt_key_is_refused_without_quoting_its_line() -> anyhow::Result<()> {
        let text = MINIMAL.replace("private_key =", "privat_key =");
        let error = Config::parse_str(&text, source())
            .err()
            .context("an unknown key must be refused")?;
        let message = format!("{error:#}");
        assert!(
            message.starts_with(
                "invalid config pipette-ios-proxy.toml, line 8: unknown field `privat_key`"
            ),
            "{message}"
        );
        assert!(!message.contains(KEY), "{message}");
        Ok(())
    }

    /// Each case rewrites one line; no refusal may echo the secret or an
    /// endpoint's password.
    #[rstest]
    #[case::misspelt_key(
        "secret_access_key =",
        "secret_acces_key =",
        "line 17: unknown field `secret_acces_key`"
    )]
    #[case::no_bucket("bucket = \"example-client-releases\"", "", "bucket")]
    #[case::bucket_that_is_no_host_name(
        "\"example-client-releases\"",
        "\"Liquid_Releases\"",
        "[client_releases] bucket `Liquid_Releases`"
    )]
    #[case::bucket_that_starts_with_a_dot(
        "\"example-client-releases\"",
        "\".releases\"",
        "[client_releases] bucket `.releases`"
    )]
    #[case::region_with_a_path(
        "\"us-east-1\"",
        "\"us-east-1/x\"",
        "[client_releases] region `us-east-1/x`"
    )]
    #[case::blank_access_key(
        "\"AKIAIOSFODNN7EXAMPLE\"",
        "\" \"",
        "[client_releases] access_key_id"
    )]
    #[case::two_secrets(
        "region = \"us-east-1\"",
        "region = \"us-east-1\"\nsecret_access_key_file = \"releases.secret\"",
        "both secret_access_key and secret_access_key_file"
    )]
    #[case::no_secret(
        "secret_access_key = \"wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY\"",
        "",
        "gives no secret"
    )]
    #[case::blank_secret(
        "\"wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY\"",
        "\"  \"",
        "[client_releases] secret_access_key in pipette-ios-proxy.toml is empty"
    )]
    #[case::blank_secret_file(
        "secret_access_key = \"wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY\"",
        "secret_access_key_file = \" \"",
        "[client_releases] secret_access_key_file in pipette-ios-proxy.toml is empty"
    )]
    #[case::endpoint_with_a_path(
        "region = \"us-east-1\"",
        "region = \"us-east-1\"\nendpoint = \"http://127.0.0.1:9000/minio\"",
        "[client_releases] endpoint `http://127.0.0.1:9000/minio`"
    )]
    #[case::endpoint_with_a_query(
        "region = \"us-east-1\"",
        "region = \"us-east-1\"\nendpoint = \"http://127.0.0.1:9000/?token=MinioPassw0rd\"",
        "[client_releases] endpoint `http://127.0.0.1:9000/`"
    )]
    #[case::endpoint_with_credentials(
        "region = \"us-east-1\"",
        "region = \"us-east-1\"\nendpoint = \"http://minio:MinioPassw0rd@127.0.0.1:9000\"",
        "[client_releases] endpoint `http://127.0.0.1:9000/`"
    )]
    #[case::endpoint_with_credentials_and_no_scheme(
        "region = \"us-east-1\"",
        "region = \"us-east-1\"\nendpoint = \"minio:MinioPassw0rd@127.0.0.1:9000\"",
        "[client_releases] endpoint `127.0.0.1:9000`"
    )]
    #[case::endpoint_that_is_not_http(
        "region = \"us-east-1\"",
        "region = \"us-east-1\"\nendpoint = \"s3://example-client-releases\"",
        "expected http(s)://host[:port]"
    )]
    #[case::endpoint_without_a_scheme(
        "region = \"us-east-1\"",
        "region = \"us-east-1\"\nendpoint = \"127.0.0.1:9000\"",
        "[client_releases] endpoint `127.0.0.1:9000`"
    )]
    fn a_bad_client_releases_table_is_refused_naming_the_mistake(
        #[case] line: &str,
        #[case] rewritten: &str,
        #[case] needle: &str,
    ) -> anyhow::Result<()> {
        let text = with_releases().replace(line, rewritten);
        let error = Config::parse_str(&text, source())
            .err()
            .context("the table must be refused")?;
        let message = format!("{error:#}");
        assert!(
            message.contains(needle),
            "the message must name `{needle}`: {message}"
        );
        [SECRET, "Passw0rd"].into_iter().for_each(|credential| {
            assert!(
                !message.contains(credential),
                "echoes {credential}: {message}"
            );
        });
        Ok(())
    }

    /// A config whose keys all sit in files, with `secret` (when given) in its
    /// secret file at `mode`.
    fn config_with_a_secret_file(
        dir: &Path,
        secret: Option<&str>,
        mode: u32,
    ) -> anyhow::Result<PathBuf> {
        use std::os::unix::fs::PermissionsExt;

        let write_at = |name: &str, contents: &str, mode: u32| -> anyhow::Result<PathBuf> {
            let path = dir.join(name);
            fs::write(&path, contents).with_context(|| format!("writing {name}"))?;
            fs::set_permissions(&path, fs::Permissions::from_mode(mode))
                .with_context(|| format!("setting the mode of {name}"))?;
            Ok(path)
        };
        write_at("phone.key", KEY, 0o600)?;
        if let Some(secret) = secret {
            write_at("releases.secret", &format!("{secret}\n"), mode)?;
        }
        let text = with_releases()
            .replace(
                &format!("private_key = \"{KEY}\""),
                "private_key_file = \"phone.key\"",
            )
            .replace(
                &format!("secret_access_key = \"{SECRET}\""),
                "secret_access_key_file = \"releases.secret\"",
            );
        write_at("pipette-ios-proxy.toml", &text, 0o644)
    }

    /// The secret file is read for the bucket alone: a missing, blank or exposed
    /// one is refused there, and the config still loads for `run`.
    #[rstest]
    #[case::owner_only(Some(SECRET), 0o600, Ok(()))]
    #[case::group_readable(
        Some(SECRET),
        0o640,
        Err("releases.secret is mode 0640, readable beyond its owner")
    )]
    #[case::blank(Some(" "), 0o600, Err("is empty"))]
    #[case::missing(None, 0o600, Err("releases.secret: No such file or directory"))]
    fn the_secret_file_is_read_for_the_bucket_alone(
        #[case] secret: Option<&str>,
        #[case] mode: u32,
        #[case] expected: Result<(), &str>,
    ) -> anyhow::Result<()> {
        let dir = tempfile::tempdir().context("a temp dir")?;
        let config = Config::load(&config_with_a_secret_file(dir.path(), secret, mode)?)?;
        match (config.releases_bucket(), expected) {
            (Ok(releases), Ok(())) => assert_eq!(releases.secret_access_key.expose(), SECRET),
            (Err(error), Err(needle)) => {
                let message = format!("{error:#}");
                assert!(
                    message.contains("[client_releases] secret_access_key_file")
                        && message.contains(needle),
                    "{message}"
                );
                assert!(!message.contains(SECRET), "{message}");
            }
            (got, expected) => anyhow::bail!("got {got:?}, expected {expected:?}"),
        }
        Ok(())
    }

    #[test]
    fn a_config_holding_the_secret_inline_must_be_owner_only() -> anyhow::Result<()> {
        use std::os::unix::fs::PermissionsExt;

        let dir = tempfile::tempdir().context("a temp dir")?;
        let path = config_with_a_secret_file(dir.path(), None, 0o600)?;
        let text = fs::read_to_string(&path)
            .context("re-reading the config")?
            .replace(
                "secret_access_key_file = \"releases.secret\"",
                &format!("secret_access_key = \"{SECRET}\""),
            );
        fs::write(&path, text).context("rewriting the config")?;

        let error = Config::load(&path)?
            .releases_bucket()
            .err()
            .context("a readable config holding the secret must be refused")?;
        let message = format!("{error:#}");
        assert!(
            message.contains("[client_releases] secret_access_key inline")
                && message.contains("0644")
                && message.contains("create a new access key"),
            "{message}"
        );
        assert!(!message.contains(SECRET), "{message}");

        fs::set_permissions(&path, fs::Permissions::from_mode(0o600)).context("tightening")?;
        assert_eq!(
            Config::load(&path)?
                .releases_bucket()?
                .secret_access_key
                .expose(),
            SECRET
        );
        Ok(())
    }

    #[test]
    fn relative_paths_resolve_against_the_config_file() -> anyhow::Result<()> {
        let config = Config::parse_str(MINIMAL, Path::new("/etc/pipette/ios-proxy.toml"))?;
        assert_eq!(
            pool_dir(&config)?,
            Path::new("/etc/pipette/./builds"),
            "a launchd job's cwd is / , so a checked-in config's relative path has to mean \
             `beside the config`"
        );
        Ok(())
    }

    #[rstest]
    #[case::heartbeat("[loop] heartbeat_secs", "[loop]\nheartbeat_secs = 0")]
    #[case::claim_backoff("[loop] claim_backoff_secs", "[loop]\nclaim_backoff_secs = 0")]
    #[case::call_timeout("[devicectl] call_timeout_secs", "[devicectl]\ncall_timeout_secs = 0")]
    #[case::run_timeout("[devicectl] run_timeout_secs", "[devicectl]\nrun_timeout_secs = 0")]
    #[case::first_line_timeout(
        "[devicectl] first_line_timeout_secs",
        "[devicectl]\nfirst_line_timeout_secs = 0"
    )]
    fn a_zero_timeout_is_refused_naming_the_key(
        #[case] key: &str,
        #[case] table: &str,
    ) -> anyhow::Result<()> {
        let text = format!("{MINIMAL}\n{table}\n");
        let error = Config::parse_str(&text, source())
            .err()
            .context("zero would spin")?;
        let message = format!("{error:#}");
        assert!(
            message.contains(key),
            "the message must name {key}: {message}"
        );
        Ok(())
    }

    /// A config from before the key still loads, with the built-in default.
    #[rstest]
    #[case::absent("", 60)]
    #[case::set("[devicectl]\nfirst_line_timeout_secs = 120\n", 120)]
    fn the_first_line_timeout_defaults_and_can_be_set(
        #[case] table: &str,
        #[case] secs: u64,
    ) -> anyhow::Result<()> {
        let config = Config::parse_str(&format!("{MINIMAL}\n{table}"), source())?;
        assert_eq!(
            config.devicectl.first_line_timeout,
            Duration::from_secs(secs)
        );
        Ok(())
    }

    #[rstest]
    #[case::unset("", None)]
    #[case::blank("[devicectl]\npymobiledevice3 = \"  \"\n", None)]
    #[case::set(
        "[devicectl]\npymobiledevice3 = \"/opt/uv/bin/pymobiledevice3\"\n",
        Some("/opt/uv/bin/pymobiledevice3")
    )]
    #[case::relative_to_the_config(
        "[devicectl]\npymobiledevice3 = \"venv/bin/pymobiledevice3\"\n",
        Some("/srv/proxy/venv/bin/pymobiledevice3")
    )]
    #[case::bare_name_on_path(
        "[devicectl]\npymobiledevice3 = \"pymobiledevice3\"\n",
        Some("pymobiledevice3")
    )]
    fn the_gpu_pin_is_on_only_when_its_cli_is_configured(
        #[case] table: &str,
        #[case] expected: Option<&str>,
    ) -> anyhow::Result<()> {
        let config = Config::parse_str(
            &format!("{MINIMAL}\n{table}"),
            Path::new("/srv/proxy/pipette-ios-proxy.toml"),
        )?;
        assert_eq!(
            config.devicectl.pymobiledevice3.as_deref(),
            expected.map(Path::new)
        );
        Ok(())
    }

    #[test]
    fn an_idle_wait_of_zero_is_allowed() -> anyhow::Result<()> {
        let text = format!("{MINIMAL}\n[loop]\nidle_wait_secs = 0\nidle_jitter_secs = 0\n");
        let config = Config::parse_str(&text, source())?;
        assert_eq!(config.loop_config.idle_wait, Duration::ZERO);
        assert_eq!(config.loop_config.idle_jitter, Duration::ZERO);
        Ok(())
    }

    #[test]
    fn a_bad_log_level_is_refused_naming_the_key_and_value() -> anyhow::Result<()> {
        let text = format!("{MINIMAL}\n[log]\nlevel = \"verbose\"\n");
        let error = Config::parse_str(&text, source())
            .err()
            .context("`verbose` is not a level")?;
        let message = format!("{error:#}");
        assert!(message.contains("[log] level `verbose`"), "{message}");
        Ok(())
    }

    #[test]
    fn the_config_path_falls_back_flag_then_env_then_cwd() -> anyhow::Result<()> {
        assert_eq!(
            resolve_config_path(Some(Path::new("/tmp/a.toml")), Some("/tmp/b.toml")),
            PathBuf::from("/tmp/a.toml"),
            "the flag wins over the variable"
        );
        assert_eq!(
            resolve_config_path(None, Some("/tmp/b.toml")),
            PathBuf::from("/tmp/b.toml")
        );
        assert_eq!(
            resolve_config_path(None, Some("")),
            PathBuf::from(DEFAULT_CONFIG_FILE)
        );
        assert_eq!(
            resolve_config_path(None, None),
            PathBuf::from(DEFAULT_CONFIG_FILE)
        );
        Ok(())
    }

    #[test]
    fn tilde_expands_only_as_a_leading_segment() -> anyhow::Result<()> {
        let home = std::env::var_os("HOME")
            .filter(|value| !value.is_empty())
            .map(PathBuf::from)
            .context("HOME is set in every environment this tool runs in")?;
        assert_eq!(expand_tilde("~"), home);
        assert_eq!(expand_tilde("~/.config/x"), home.join(".config/x"));
        assert_eq!(
            expand_tilde("~other/.config"),
            PathBuf::from("~other/.config"),
            "another user's home is not ours to resolve"
        );
        assert_eq!(expand_tilde("/abs/~/x"), PathBuf::from("/abs/~/x"));
        Ok(())
    }

    const SIGNING: &str = "team_id = \"353N5JSDGZ\"\n\
         keychain_password_file = \"keychain.password\"\n\
         api_key_file = \"AuthKey_TEST.p8\"\n\
         api_key_id = \"TESTKEYID1\"\n\
         api_issuer_id = \"00000000-0000-0000-0000-000000000000\"\n";

    const DEFAULT_VERSION: &str = "2026.10.02-e6a9434-internal";

    /// [`MINIMAL`] in version mode: `[signing]` (`signing` its body) instead of
    /// `[builds]`, beside a `[client_releases]` with a `default`.
    fn version_mode(signing: &str) -> String {
        format!(
            "{}\n[client_releases]\nbucket = \"example-client-releases\"\n\
             region = \"us-east-1\"\naccess_key_id = \"AKIAIOSFODNN7EXAMPLE\"\n\
             secret_access_key_file = \"releases.secret\"\ndefault = \"{DEFAULT_VERSION}\"\n\n\
             [signing]\n{signing}",
            MINIMAL.replace("[builds]\ndir = \"./builds\"\n", "")
        )
    }

    #[test]
    fn a_signing_config_resolves_its_files_beside_the_config() -> anyhow::Result<()> {
        let config = Config::parse_str(
            &version_mode(SIGNING),
            Path::new("/etc/pipette/ios-proxy.toml"),
        )?;
        let BuildSource::Releases(signing) = &config.builds else {
            anyhow::bail!("[signing] must read as version mode: {:?}", config.builds);
        };
        assert_eq!(
            signing.keychain_password_file,
            Path::new("/etc/pipette/keychain.password")
        );
        assert_eq!(
            signing.api_key_file,
            Path::new("/etc/pipette/AuthKey_TEST.p8")
        );
        assert_eq!(signing.api_key_id, "TESTKEYID1");
        assert_eq!(
            signing.api_issuer_id,
            "00000000-0000-0000-0000-000000000000"
        );
        Ok(())
    }

    /// The example's commented `[signing]` block and `default`, uncommented in
    /// place of `[builds]`.
    fn the_examples_signing_block() -> String {
        let mut in_signing = false;
        EXAMPLE
            .replace("[builds]\n", "")
            .replace("dir = \"./builds\"\n", "")
            .lines()
            .map(|line| {
                in_signing = (in_signing && !line.is_empty()) || line == "# [signing]";
                match line.strip_prefix("# ") {
                    Some(rest) if in_signing || rest.starts_with("default ") => {
                        format!("{rest}\n")
                    }
                    _ => format!("{line}\n"),
                }
            })
            .collect()
    }

    #[rstest]
    #[case::signing(version_mode(SIGNING), Some("353N5JSDGZ"), DEFAULT_VERSION)]
    #[case::the_examples_signing_block(
        the_examples_signing_block(),
        Some("ABCDE12345"),
        "2026.10.02-5c435ba-internal"
    )]
    #[case::default_without_signing(
        format!("{}default = \"{DEFAULT_VERSION}\"\n", with_releases()),
        None,
        DEFAULT_VERSION
    )]
    fn a_config_names_its_build_source_and_default(
        #[case] text: String,
        #[case] team: Option<&str>,
        #[case] default: &str,
    ) -> anyhow::Result<()> {
        let config = Config::parse_str(&text, source())?;
        match (&config.builds, team) {
            (BuildSource::Releases(signing), Some(team)) => {
                assert_eq!(signing.team_id, team);
                assert_eq!(signing.keychain, expand_tilde(DEFAULT_KEYCHAIN));
                assert_eq!(signing.export_timeout, timing::DEFAULT_EXPORT_TIMEOUT);
            }
            (BuildSource::Pool(pool), None) => assert_eq!(pool.dir, Path::new("./builds")),
            (builds, team) => anyhow::bail!("got {builds:?}, expected team {team:?}"),
        }
        assert_eq!(
            config.default_version().map(BuildVersion::as_str),
            Some(default)
        );
        Ok(())
    }

    #[rstest]
    #[case::both(
        format!("{}\n[builds]\ndir = \"./builds\"\n", version_mode(SIGNING)),
        "has both [builds] and [signing]"
    )]
    #[case::neither(
        MINIMAL.replace("[builds]\ndir = \"./builds\"\n", ""),
        "has neither [builds] nor [signing]"
    )]
    #[case::signing_without_releases(
        format!("{}\n[signing]\n{SIGNING}", MINIMAL.replace("[builds]\ndir = \"./builds\"\n", "")),
        "[signing] in pipette-ios-proxy.toml needs [client_releases]"
    )]
    #[case::lowercase_team(
        version_mode(&SIGNING.replace("353N5JSDGZ", "353n5jsdgz")),
        "[signing] team_id"
    )]
    #[case::short_team(version_mode(&SIGNING.replace("353N5JSDGZ", "353N5")), "[signing] team_id")]
    #[case::keychain_with_a_quote(
        version_mode(&format!("{SIGNING}keychain = \"/k/a\\\"b.keychain-db\"\n")),
        "[signing] keychain"
    )]
    #[case::unknown_signing_key(version_mode(&format!("{SIGNING}team = \"x\"\n")), "team")]
    #[case::missing_key_id(
        version_mode(&SIGNING.replace("api_key_id = \"TESTKEYID1\"\n", "")),
        "api_key_id"
    )]
    #[case::blank_issuer(
        version_mode(&SIGNING.replace("00000000-0000-0000-0000-000000000000", " ")),
        "[signing] api_issuer_id"
    )]
    #[case::zero_export_timeout(
        version_mode(&format!("{SIGNING}export_timeout_secs = 0\n")),
        "[signing] export_timeout_secs"
    )]
    #[case::default_not_a_version(
        version_mode(SIGNING).replace(DEFAULT_VERSION, "Not A Version"),
        "[client_releases] default"
    )]
    fn a_bad_build_source_is_refused_naming_the_mistake(
        #[case] text: String,
        #[case] needle: &str,
    ) -> anyhow::Result<()> {
        let error = Config::parse_str(&text, source())
            .err()
            .context("the config must be refused")?;
        let message = format!("{error:#}");
        assert!(
            message.contains(needle),
            "the message must name `{needle}`: {message}"
        );
        Ok(())
    }

    fn signing_files(
        dir: &Path,
        password_mode: u32,
        key: bool,
        keychain: bool,
    ) -> anyhow::Result<PathBuf> {
        use std::os::unix::fs::PermissionsExt;

        let write_at = |name: &str, contents: &str, mode: u32| -> anyhow::Result<()> {
            let path = dir.join(name);
            fs::write(&path, contents).with_context(|| format!("writing {name}"))?;
            fs::set_permissions(&path, fs::Permissions::from_mode(mode))
                .with_context(|| format!("setting the mode of {name}"))
        };
        write_at("keychain.password", "correct horse\n", password_mode)?;
        if key {
            write_at("AuthKey_TEST.p8", "-----BEGIN PRIVATE KEY-----\n", 0o600)?;
        }
        if keychain {
            write_at("login.keychain-db", "", 0o600)?;
        }
        write_at("phone.key", KEY, 0o600)?;
        let text = version_mode(&format!("{SIGNING}keychain = \"login.keychain-db\"\n")).replace(
            &format!("private_key = \"{KEY}\""),
            "private_key_file = \"phone.key\"",
        );
        write_at("pipette-ios-proxy.toml", &text, 0o644)?;
        Ok(dir.join("pipette-ios-proxy.toml"))
    }

    #[rstest]
    #[case::ready(0o600, true, true, Ok(()))]
    #[case::password_group_readable(0o640, true, true, Err("[signing] keychain_password_file"))]
    #[case::no_key_file(0o600, false, true, Err("[signing] api_key_file"))]
    #[case::no_keychain(0o600, true, false, Err("is not a file"))]
    fn the_signer_reads_its_files_only_when_asked(
        #[case] password_mode: u32,
        #[case] key: bool,
        #[case] keychain: bool,
        #[case] expected: Result<(), &str>,
    ) -> anyhow::Result<()> {
        let dir = tempfile::tempdir().context("a temp dir")?;
        let config = Config::load(&signing_files(dir.path(), password_mode, key, keychain)?)?;
        match (config.signer(), expected) {
            (Ok(signer), Ok(())) => {
                assert_eq!(signer.password.expose(), "correct horse");
                assert_eq!(signer.security, Path::new("/usr/bin/security"));
                assert_eq!(signer.xcodebuild, Path::new("/usr/bin/xcodebuild"));
                assert_eq!(signer.keychain, dir.path().join("login.keychain-db"));
                assert!(!format!("{signer:?}").contains("correct horse"));
            }
            (Err(error), Err(needle)) => {
                let message = format!("{error:#}");
                assert!(message.contains(needle), "{message}");
                assert!(!message.contains("correct horse"), "{message}");
            }
            (got, expected) => anyhow::bail!("got {got:?}, expected {expected:?}"),
        }
        Ok(())
    }

    #[test]
    fn a_pool_config_has_no_signer() -> anyhow::Result<()> {
        let error = Config::parse_str(MINIMAL, source())?
            .signer()
            .err()
            .context("a pool config signs nothing")?;
        assert!(
            error.to_string().contains("has no [signing] table"),
            "{error:#}"
        );
        Ok(())
    }

    #[test]
    fn the_signer_is_the_configured_pair() -> anyhow::Result<()> {
        let credentials = ClientCredentials {
            client_id: ClientId::try_new("ev1_9f2c1d3e4b5a6c7d")
                .context("a well-formed client id")?,
            private_key: PrivateKey::try_new(KEY).context("a well-formed key")?,
        };
        let signer = credentials.signer();
        assert_eq!(signer.client_id.as_ref(), "ev1_9f2c1d3e4b5a6c7d");
        assert_eq!(signer.private_key_hex, credentials.private_key);
        Ok(())
    }
}
