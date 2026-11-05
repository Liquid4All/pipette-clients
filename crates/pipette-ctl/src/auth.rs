//! `pipette-ctl auth`: the user identity a plan is submitted under.
//!
//! A plan submitter is a user (`user_…`), not a client (`ev1_…`): plan and
//! fleet permissions are grantable only to a user (pipette-mgmt's
//! `docs/authentication.md` §5.11). These verbs cover the account's own routes;
//! the rest of the account surface is [`crate::mgmt`].
//!
//! Admitting an account is the operator's decision: `register` presents the
//! user pre-auth key an operator minted, or asks for an account an operator
//! then admits.

use std::{
    ffi::OsString,
    fmt,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    str::FromStr,
    time::{Duration, Instant},
};

use anyhow::Context;
use clap::{Args, Subcommand};
use tabled::{Tabled, settings::Style};

use pipette_http::HttpClient;
use pipette_mgmt_client::{
    AddUserKeyRequest, ContactEmail, Error, KeyDetails, KeyOnlyIdentity, PrivateKey, PublicKey,
    RegisterUserRequest, ServerUrl, Timestamp, UserPreauthKey, UserProfile, UserSigner, UserStatus,
    generate_keypair, key_id_for, parse_contact_email,
    user_api::{self, KeyApproval, RefusalError},
};
use pipette_plan_types::{UserId, UserKeyId};
use pipette_ssh::{Ed25519PublicKey, inventory::SystemKey};
use pipette_workspace::resolve_work_dir;

use crate::{
    host,
    identity::{
        self, DEFAULT_SERVER_URL, KeyToStore, RegisteredIdentity, SshKeyRecord, StoredKey,
        SubmitIdentity,
    },
    system_keys,
};

const REGISTER_LONG_ABOUT: &str = "\
Register this machine's key against a user account, on either path in.

Signs with a key this machine already has, from an agent or from ~/.ssh, and
sends its public half. `auth keys available` lists them. With one
usable key there is nothing to choose; with several, --ssh-key names it. Pass
--generate instead to mint a key that lives with this working directory.

Nothing is copied by registering an SSH key: the identity records which public
key signs for the account, and the private half is read from where it already
lives, or never read at all when an agent holds it.

What goes with it decides which account it lands on:

  --user-preauth-key   an account that already exists, named by the key, and
                       admitted already approved. Nothing else on the command
                       line sets a field of the record, and there is no
                       approval step to wait for.

  --contact-email      no account yet: this asks for one. It comes up pending,
                       and an operator has to admit it before the account can
                       file a plan. The key registered here is the account's
                       from the start, so nothing is re-run once it is.

The two are exclusive. An invite carries the account this key joins, so an
address beside it would describe a record this route may not touch.

A user's pre-auth key is not a device's. The server keeps the two in separate
records, mints them with separate commands, and admits each on its own
registration route, so a `client_preauth_...` token handed out for a device does
not register an account here.

Refuses to run if this working directory already holds a user identity. Revoke
that key with `auth keys revoke` before replacing it, or the account keeps a
registration nothing here can sign for. It also refuses a working directory
with no workspace in it: run `pipette-ctl init` there first, so the identity
lands beside the plan state rather than in a directory nothing else reads.";

const LOGIN_LONG_ABOUT: &str = "\
Sign this machine in by approving its key from a browser.

Picks a key the way `auth register` does: the machine's one usable key, the
one --ssh-key names, or a new one with --generate. Then it prints a page on the
management app and a code, and opens the page in the default browser, unless
--no-browser is passed or no browser is likely to reach you (an SSH session,
or Linux with no display). Signed in there, type the code, choose whether the
key expires, and approve. This command waits for that, then writes the
identity where `auth register` would have.

The code is the start of the key's digest, so a page whose code differs is
approving another key. It is a short check, not a proof: approve only a page you
opened from this command yourself.

Nothing is written until the key is approved: Ctrl-C, or --timeout passing
(counted from when the page is printed), leaves the working directory as it
was.

In a working directory that already holds an identity, it asks for that
identity's key to be approved again, after it expired or was revoked. If the
key still works, it says so and stops.";

const ADOPT_LONG_ABOUT: &str = "\
Prepare a second machine to sign for an account it is already on.

An account holds one key per machine, and a new machine cannot register its own:
`POST /api/v1/users/me/keys` is signed by a key the account already holds. So this
picks the key this machine already has, or mints one with --generate, and
prints its public key to add with `auth keys add` on a machine that is already
registered, or on the management app's Account page.

Until that add lands, requests from this machine are refused: the account does
not know this key yet. `auth login` does both halves from a browser, and `auth
register` with a user pre-auth key needs no second machine.";

const KEYS_ADD_LONG_ABOUT: &str = "\
Register another machine's public key under this account.

Signed by the key this machine holds, so a second laptop needs no operator
round-trip. Run `auth adopt` on the new machine first to get the public key to
pass here. The management app's sign-in page (<server>/app/) prints this
command with a key it made for the browser, to sign in without this machine's
key leaving it.

--expires-in makes the key stop signing after that many days, at most what the
server allows (30 today). The sign-in page passes it, so the browser's key
expires on its own.";

const KEYS_AVAILABLE_LONG_ABOUT: &str = "\
List the Ed25519 keys this machine can register, and where each one lives.

Reads every ssh-agent on the host, not only the one SSH_AUTH_SOCK names, and
every key under ~/.ssh. A key in both places is listed once, with both.

Keys this machine cannot sign with are listed too, because the reason is what
an operator acts on: an encrypted file with no agent holding it needs `ssh-add`,
not a new key. So is anything that could not be read at all, which is what a
locked vault and an RSA key both look like from here.

Reaches no server, and names the keys by the `SHA256:...` fingerprint
`ssh-add -l` prints, which is what --ssh-key takes.";

const KEYS_REVOKE_LONG_ABOUT: &str = "\
Revoke one key registration.

Revoking the last key, or the key this request is signed with, is allowed on
purpose: a holder whose only key is compromised has to be able to retire it at
once, and a fresh user pre-auth key re-registers. Nothing local is deleted, so
delete the identity directory too when retiring this machine's own key.";

/// Manage the user identity that files campaigns
#[derive(Args, Debug)]
pub struct AuthArgs {
    #[command(subcommand)]
    pub command: AuthCommand,
}

#[derive(Subcommand, Debug)]
pub enum AuthCommand {
    /// Register this machine's key against an account, or ask for a new account
    #[command(long_about = REGISTER_LONG_ABOUT)]
    Register(RegisterArgs),
    /// Approve this machine's key from a browser signed in to the account
    #[command(long_about = LOGIN_LONG_ABOUT)]
    Login(LoginArgs),
    /// Print the account, its permissions, and its registered keys
    Me(MeArgs),
    /// Prepare this machine's key for a machine already on the account to add
    #[command(long_about = ADOPT_LONG_ABOUT)]
    Adopt(AdoptArgs),
    /// Add or revoke the account's key registrations
    Keys(KeysArgs),
    /// Print this machine's public key, to hand to `auth keys add`
    Pubkey,
}

/// How a registration gets the key it signs with. Flattened into the verbs
/// that start an identity, which face the same choice.
#[derive(Args, Debug)]
pub struct KeyArgs {
    /// Which of this machine's keys to sign with: a `SHA256:...` fingerprint
    /// (`ssh-add -l`), the key's comment, its hex public key, or a key file in
    /// `~/.ssh` (`ssh-add` one kept elsewhere). `auth keys available` lists
    /// them; omit it when this machine offers exactly one usable key.
    #[arg(
        long,
        value_name = "FINGERPRINT|COMMENT|PATH",
        conflicts_with = "generate"
    )]
    pub ssh_key: Option<String>,

    /// Mint a key for this working directory instead of using one the machine
    /// already has. It is a second private key on this disk: prefer an agent's.
    #[arg(long)]
    pub generate: bool,
}

/// The key a fresh registration will use.
enum ChosenKey {
    /// Minted here, and about to be written beside the identity.
    Generated {
        private_key_hex: PrivateKey,
        public_key_hex: PublicKey,
    },
    /// One the machine already had, recorded rather than copied.
    System {
        public_key_hex: PublicKey,
        fingerprint: String,
        found_at: String,
    },
}

impl ChosenKey {
    /// The key `--ssh-key` names, or a fresh one under `--generate`.
    fn resolve(args: &KeyArgs) -> anyhow::Result<Self> {
        if args.generate {
            let (private_key_hex, public_key_hex) = generate_keypair()?;
            return Ok(Self::Generated {
                private_key_hex,
                public_key_hex,
            });
        }
        Self::from_system(system_keys::choose(args.ssh_key.as_deref())?)
    }

    fn from_system(key: SystemKey) -> anyhow::Result<Self> {
        Ok(Self::System {
            public_key_hex: system_keys::public_key_hex(&key)?,
            fingerprint: key.fingerprint(),
            found_at: key.found_at(),
        })
    }

    /// A keypair the caller already holds, so a test knows the key id its mock
    /// body has to echo.
    #[cfg(test)]
    fn generated((private_key_hex, public_key_hex): (PrivateKey, PublicKey)) -> Self {
        Self::Generated {
            private_key_hex,
            public_key_hex,
        }
    }

    fn public_key_hex(&self) -> &PublicKey {
        match self {
            Self::Generated { public_key_hex, .. } | Self::System { public_key_hex, .. } => {
                public_key_hex
            }
        }
    }

    fn to_store(&self) -> KeyToStore<'_> {
        match self {
            Self::Generated {
                private_key_hex, ..
            } => KeyToStore::Generated(private_key_hex.as_ref()),
            Self::System {
                public_key_hex,
                found_at,
                ..
            } => KeyToStore::System(SshKeyRecord {
                public_key: public_key_hex.to_string(),
                found_at: found_at.clone(),
            }),
        }
    }

    /// Which key the account now answers to, for the run's output.
    fn describe(&self) -> String {
        match self {
            Self::Generated { .. } => {
                "generated, and written to the identity directory".to_string()
            }
            Self::System {
                fingerprint,
                found_at,
                ..
            } => format!("{fingerprint} ({found_at})"),
        }
    }
}

#[derive(Args, Debug)]
pub struct RegisterArgs {
    /// The account's pre-auth key, which admits this registration already
    /// approved. Minted for the account it registers, over HTTP by an
    /// administrator of it or on the server host, and a user's own credential:
    /// a device's `client_preauth_...` token names no account and registers
    /// nothing here.
    /// Also read from PIPETTE_USER_PREAUTH_KEY to keep it out of shell history;
    /// it is sent with the request and never written to disk.
    ///
    /// Omit it to ask for an account instead, with --contact-email.
    #[arg(long, env = "PIPETTE_USER_PREAUTH_KEY", hide_env_values = true)]
    pub user_preauth_key: Option<UserPreauthKey>,

    /// Where to reach whoever is asking, when there is no pre-auth key to
    /// present. It is the whole of what an operator decides on, so it is
    /// required on that path and refused on the other: an invite already names
    /// an account, and that account already carries an address.
    #[arg(
        long,
        value_parser = parse_contact_email,
        required_unless_present = "user_preauth_key",
        conflicts_with = "user_preauth_key"
    )]
    pub contact_email: Option<ContactEmail>,

    /// What this machine is, for whoever later decides whether to revoke the
    /// key: a key id is 64 hex characters and settles nothing. Defaults to
    /// `user@hostname`, which is what tells one machine's row of `auth me`
    /// from another's.
    #[arg(long)]
    pub key_details: Option<KeyDetails>,

    /// Management server URL.
    #[arg(long, env = "PIPETTE_MGMT_SERVER", default_value = DEFAULT_SERVER_URL)]
    pub server: ServerUrl,

    #[command(flatten)]
    pub key: KeyArgs,
}

#[derive(Args, Debug)]
pub struct LoginArgs {
    /// Management server URL. Defaults to the server of the identity this
    /// working directory already holds, then to the production server.
    #[arg(long, env = "PIPETTE_MGMT_SERVER")]
    pub server: Option<ServerUrl>,

    /// What this machine is, shown on the approval page and kept with the
    /// key. Defaults to `user@hostname`.
    #[arg(long)]
    pub key_details: Option<KeyDetails>,

    /// Print the page's address without opening a browser.
    #[arg(long)]
    pub no_browser: bool,

    /// How long to wait for the approval, in whole seconds, minutes or hours,
    /// such as `90s`, `10m` or `1h`.
    #[arg(long, default_value = "10m")]
    pub timeout: LoginTimeout,

    #[command(flatten)]
    pub key: KeyArgs,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LoginTimeout(Duration);

#[derive(Debug, thiserror::Error)]
#[error("expected whole seconds, minutes or hours, such as 90s, 10m or 1h")]
pub struct TimeoutError;

impl FromStr for LoginTimeout {
    type Err = TimeoutError;

    fn from_str(raw: &str) -> Result<Self, Self::Err> {
        [("s", 1), ("m", 60), ("h", 60 * 60)]
            .into_iter()
            .find_map(|(unit, seconds)| raw.strip_suffix(unit).map(|count| (count, seconds)))
            .filter(|(count, _)| count.bytes().all(|byte| byte.is_ascii_digit()))
            .and_then(|(count, seconds)| Some((count.parse::<u32>().ok()?, seconds)))
            .filter(|(count, _)| *count >= 1)
            .map(|(count, seconds)| Self(Duration::from_secs(u64::from(count) * seconds)))
            .ok_or(TimeoutError)
    }
}

/// In the largest unit that divides it, as it was most likely typed.
impl fmt::Display for LoginTimeout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let seconds = self.0.as_secs();
        if seconds.is_multiple_of(60 * 60) {
            write!(f, "{}h", seconds / (60 * 60))
        } else if seconds.is_multiple_of(60) {
            write!(f, "{}m", seconds / 60)
        } else {
            write!(f, "{seconds}s")
        }
    }
}

#[derive(Args, Debug)]
pub struct MeArgs {
    /// Print the identity on disk without calling the server.
    #[arg(long)]
    pub local: bool,
}

#[derive(Args, Debug)]
pub struct AdoptArgs {
    /// The account this machine is joining, as `auth me` reports it.
    #[arg(long)]
    pub user_id: UserId,

    /// Management server URL.
    #[arg(long, env = "PIPETTE_MGMT_SERVER", default_value = DEFAULT_SERVER_URL)]
    pub server: ServerUrl,

    #[command(flatten)]
    pub key: KeyArgs,
}

#[derive(Args, Debug)]
pub struct KeysArgs {
    #[command(subcommand)]
    pub command: KeysCommand,
}

#[derive(Subcommand, Debug)]
pub enum KeysCommand {
    /// Register another machine's public key under this account
    #[command(long_about = KEYS_ADD_LONG_ABOUT)]
    Add(AddKeyArgs),
    /// Revoke one key registration
    #[command(long_about = KEYS_REVOKE_LONG_ABOUT)]
    Revoke(RevokeKeyArgs),
    /// List the keys this machine could register, from its agents and ~/.ssh
    #[command(long_about = KEYS_AVAILABLE_LONG_ABOUT)]
    Available,
}

#[derive(Args, Debug)]
pub struct AddKeyArgs {
    /// Hex public key, as `auth adopt` printed it on the other machine.
    #[arg(long)]
    pub public_key: PublicKey,

    /// What that machine is, for whoever later decides whether to revoke it.
    #[arg(long)]
    pub key_details: KeyDetails,

    /// Whole days until the key stops signing, such as `7d`. The server caps
    /// it (30 days today). Without it the key never expires.
    #[arg(long)]
    pub expires_in: Option<KeyLifetime>,
}

/// `--expires-in`: whole days, at least `1d`. The upper bound is the server's
/// to set and refuse, so a change to it needs no client release.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KeyLifetime(u16);

impl std::str::FromStr for KeyLifetime {
    type Err = String;

    fn from_str(raw: &str) -> Result<Self, Self::Err> {
        raw.strip_suffix('d')
            .filter(|days| days.bytes().all(|byte| byte.is_ascii_digit()))
            .and_then(|days| days.parse::<u16>().ok())
            .filter(|days| *days >= 1)
            .map(Self)
            .ok_or_else(|| "expected whole days, at least 1d, such as 7d".to_string())
    }
}

impl KeyLifetime {
    fn expires_at(self, now: time::OffsetDateTime) -> anyhow::Result<Timestamp> {
        let at = now
            .checked_add(time::Duration::days(i64::from(self.0)))
            .context("the expiry is past the last date this machine can represent")?
            .format(&time::format_description::well_known::Rfc3339)?;
        Ok(Timestamp::try_new(at)?)
    }
}

#[derive(Args, Debug)]
pub struct RevokeKeyArgs {
    /// The key id to revoke, as `auth me` lists it.
    pub user_key_id: UserKeyId,
}

/// One row of the key table `auth me` prints.
#[derive(Tabled)]
struct KeyRow {
    #[tabled(rename = "KEY ID")]
    user_key_id: String,
    #[tabled(rename = "DETAILS")]
    key_details: String,
    #[tabled(rename = "ADDED")]
    added_at: String,
    #[tabled(rename = "EXPIRES")]
    expires_at: String,
    /// Which registration is signing from this machine. A revocation is
    /// decided from this table, and revoking the local key is allowed, so the
    /// row that would lock this machine out is the one to mark.
    #[tabled(rename = "THIS MACHINE")]
    local: &'static str,
}

impl AuthArgs {
    /// Each arm builds the HTTP client only if it needs one: `adopt` and
    /// `pubkey` reach no server and run while a machine is still being set up,
    /// where a client that cannot be built is likeliest.
    pub fn execute(self, work_dir_arg: Option<&Path>) -> anyhow::Result<()> {
        // The identity lives in the `--work-dir` workspace, so an uninitialized
        // one is refused rather than given an identity beside no plan state.
        // Resolved per arm: `keys available` runs before there is a workspace.
        let workspace = || -> anyhow::Result<PathBuf> {
            let work_dir = resolve_work_dir(work_dir_arg)?;
            crate::workspace::require_workspace(&work_dir, work_dir_arg)?;
            Ok(work_dir)
        };
        let http = |timeout| {
            HttpClient::with_request_timeout("pipette-ctl", timeout)
                .context("to build the HTTP client")
        };
        // The one-shot verbs are single, unretried calls, so they share mgmt's bound.
        let json_http = || http(crate::mgmt::JSON_TIMEOUT);
        match self.command {
            AuthCommand::Adopt(args) => adopt(&workspace()?, args),
            AuthCommand::Pubkey => pubkey(&workspace()?),
            AuthCommand::Register(args) => register(&workspace()?, args, &json_http()?),
            AuthCommand::Login(args) => login(&workspace()?, args, &http(LOGIN_REQUEST_TIMEOUT)?),
            AuthCommand::Me(args) => me(&workspace()?, args, &json_http()?),
            AuthCommand::Keys(args) => match args.command {
                KeysCommand::Add(add) => add_key(
                    &workspace()?,
                    add,
                    &json_http()?,
                    time::OffsetDateTime::now_utc(),
                ),
                KeysCommand::Revoke(revoke) => revoke_key(&workspace()?, revoke, &json_http()?),
                KeysCommand::Available => {
                    system_keys::print_available();
                    Ok(())
                }
            },
        }
    }
}

fn register(work_dir: &Path, args: RegisterArgs, http: &HttpClient) -> anyhow::Result<()> {
    identity::ensure_empty(work_dir)?;
    let key = ChosenKey::resolve(&args.key)?;
    register_with_key(work_dir, args, http, key)
}

/// Takes the key rather than choosing one, so a test can echo the key id the
/// public half derives from a static mock body.
fn register_with_key(
    work_dir: &Path,
    args: RegisterArgs,
    http: &HttpClient,
    key: ChosenKey,
) -> anyhow::Result<()> {
    let public_key_hex = key.public_key_hex().clone();
    let key_details = match args.key_details {
        Some(details) => details,
        None => default_key_details()?,
    };
    let key_id = key_id_for(&public_key_hex)?;

    // Register before writing anything: a refused pre-auth key then leaves no
    // identity on disk, so the operator retries without clearing the store
    // first.
    let response = user_api::register(
        &args.server,
        http,
        RegisterUserRequest {
            public_key: public_key_hex,
            user_preauth_key: args.user_preauth_key,
            contact_email: args.contact_email,
            key_details: key_details.clone(),
        },
    )
    .map_err(|error| match error {
        // Only reachable after a 2xx, so the account holds the key and the
        // pre-auth key is spent, while the private half is dropped here.
        Error::DecodeJson { .. } => anyhow::Error::new(error).context(
            "the server accepted the registration, but its receipt could not be read. \
             The account holds a key this machine cannot sign for. Revoke it from \
             another machine before you register again",
        ),
        // The request may have landed and only its answer been lost.
        Error::Transport { .. } => {
            let next = match &key {
                ChosenKey::Generated { .. } => format!(
                    "the key this run generated was not saved, so revoke {key_id} if the \
                     Account page lists it, or with `pipette-ctl auth keys revoke {key_id}` on \
                     another machine of the account, before you register again"
                ),
                // Named in full: a bare login reaches the default server and may pick another key.
                ChosenKey::System { fingerprint, .. } => format!(
                    "run `pipette-ctl auth login --server {} --ssh-key {fingerprint}`: it \
                     writes the identity at once if the key was registered",
                    args.server
                ),
            };
            anyhow::Error::new(error).context(format!(
                "the server may have registered {key_id} before the connection failed. \
                 Nothing was written; {next}"
            ))
        }
        error @ (Error::InvalidUrl { .. }
        | Error::Signer { .. }
        | Error::TimestampFormat(_)
        | Error::UnusableTimestamp { .. }
        | Error::InvalidPrivateKey(_)
        | Error::InvalidPublicKey(_)
        | Error::UnusableKeyId { .. }
        | Error::OsEntropy(_)
        | Error::DotSegment { .. }
        | Error::EncodeJson(_)
        | Error::HttpStatus { .. }
        | Error::StreamEnded { .. }
        | Error::BuildClient(_)) => anyhow::Error::new(error),
    })?;
    reconcile_key_id(&key_id, &response.user_key_id)?;

    identity::write(
        work_dir,
        &response.user_id,
        &key_id,
        &args.server,
        &key.to_store(),
    )
    .with_context(|| {
        format!(
            "the key is registered as {key_id} under {}, but could not be saved. \
             Revoke it from another machine before registering again",
            response.user_id
        )
    })?;

    // An account that was asked for is not one that was joined, and the
    // difference is the whole of what the operator does next.
    let pending = response.status == UserStatus::Pending;
    println!(
        "{} {}",
        if pending { "requested" } else { "registered" },
        response.user_id
    );
    println!("status:       {}", response.status);
    println!("user_key_id:  {key_id}");
    // The one field the operator may not have written, so the run shows what
    // the account will carry.
    println!("key_details:  {key_details}");
    // Which of the machine's keys was picked, for an operator who did not name one.
    println!("key:          {}", key.describe());
    println!(
        "identity:     {}",
        identity::identity_dir(work_dir).display()
    );
    if pending {
        println!();
        println!("An operator has to admit the account before it can file a plan.");
        println!("`auth me` reports the status. This key is already registered to");
        println!("it, so nothing here is re-run once the account is admitted.");
    }
    Ok(())
}

/// What `--key-details` says when the operator says nothing: `user@host`.
fn default_key_details() -> anyhow::Result<KeyDetails> {
    let label = key_details_label(host::current_user(), host::hostname());
    KeyDetails::try_new(label).context("building the default --key-details")
}

/// The label itself, from what the host was willing to say.
///
/// The description is read next to other rows of `auth me`, and the operator
/// and the machine are what tell those rows apart. A host that names neither
/// still registers: a blank description is refused by [`KeyDetails`], and
/// failing a registration over a label would be worse than a vague one.
///
/// Takes both values rather than reading them, so every arm is testable
/// without an environment to arrange.
fn key_details_label(user: Option<String>, hostname: Option<String>) -> String {
    match (user, hostname) {
        (Some(user), Some(hostname)) => format!("{user}@{hostname}"),
        (Some(user), None) => user,
        (None, Some(hostname)) => hostname,
        (None, None) => "unknown machine".to_string(),
    }
}

const POLL_INTERVAL: Duration = Duration::from_secs(3);

/// Bounds each poll, since the wait checks its deadline only between them and
/// a connection gone half-open would otherwise hold one past it.
const LOGIN_REQUEST_TIMEOUT: Duration = Duration::from_secs(15);

enum LoginKey {
    /// Picked as `auth register` picks one, for a directory with no identity.
    Chosen(ChosenKey),
    /// The key of the identity this directory already holds, approved again
    /// in place.
    Kept {
        key: StoredKey,
        public_key_hex: PublicKey,
    },
}

impl LoginKey {
    fn resolve(
        work_dir: &Path,
        stored: &RegisteredIdentity,
        args: &KeyArgs,
    ) -> anyhow::Result<Self> {
        let Some(key) = stored.key() else {
            identity::ensure_empty(work_dir)?;
            return Ok(Self::Chosen(ChosenKey::resolve(args)?));
        };
        // Swapping the key would strand the one the account may still hold.
        if args.generate || args.ssh_key.is_some() {
            anyhow::bail!(
                "{} already holds an identity, and `auth login` asks for its key to be \
                 approved again. Remove the directory to log in with a different key",
                identity::identity_dir(work_dir).display()
            );
        }
        Ok(Self::Kept {
            key: key.clone(),
            public_key_hex: identity::public_key_of(key)?,
        })
    }

    fn public_key_hex(&self) -> &PublicKey {
        match self {
            Self::Chosen(key) => key.public_key_hex(),
            Self::Kept { public_key_hex, .. } => public_key_hex,
        }
    }

    fn signer(&self) -> anyhow::Result<UserSigner> {
        match self {
            Self::Chosen(ChosenKey::Generated {
                private_key_hex, ..
            }) => Ok(UserSigner::PrivateKey(private_key_hex.clone())),
            Self::Chosen(ChosenKey::System { public_key_hex, .. }) => {
                system_keys::signer_for(public_key_hex.as_ref())
            }
            Self::Kept { key, .. } => identity::signer(key),
        }
    }

    fn to_store(&self) -> KeyToStore<'_> {
        match self {
            Self::Chosen(key) => key.to_store(),
            Self::Kept {
                key: StoredKey::Local(_),
                ..
            } => KeyToStore::InPlace,
            Self::Kept {
                key: StoredKey::System(record),
                ..
            } => KeyToStore::System(record.clone()),
        }
    }

    fn describe(&self) -> String {
        match self {
            Self::Chosen(key) => key.describe(),
            Self::Kept {
                key: StoredKey::Local(path),
                ..
            } => path.display().to_string(),
            Self::Kept {
                key: StoredKey::System(record),
                ..
            } => {
                let fingerprint = Ed25519PublicKey::from_hex(&record.public_key)
                    .map_or_else(|_| record.public_key.clone(), |key| key.fingerprint());
                format!("{fingerprint} ({})", record.found_at)
            }
        }
    }

    /// What to do when the identity is not saved. A generated key that never
    /// reached disk is lost, so an approval of it signs nothing.
    fn if_unsaved(&self, key_id: &UserKeyId, key_on_disk: bool) -> String {
        let again = "run `pipette-ctl auth login` again";
        match self {
            Self::Chosen(ChosenKey::Generated { .. }) if !key_on_disk => format!(
                "{again}, and revoke {key_id} if the Account page lists it: the key this run \
                 generated was not saved"
            ),
            Self::Chosen(_) | Self::Kept { .. } => again.to_string(),
        }
    }
}

/// Injected so a test records the address instead of opening it.
type OpenBrowser<'a> = &'a dyn Fn(&str) -> std::io::Result<()>;

struct LoginRequest {
    server: ServerUrl,
    key_details: KeyDetails,
    timeout: LoginTimeout,
}

/// What the approval wait reads time from and sleeps on, so a test can run
/// ten minutes of polling without waiting for them.
trait Clock {
    /// Time since a fixed start.
    fn elapsed(&self) -> Duration;
    fn sleep(&mut self, duration: Duration);
}

struct WallClock(Instant);

impl Clock for WallClock {
    fn elapsed(&self) -> Duration {
        self.0.elapsed()
    }

    fn sleep(&mut self, duration: Duration) {
        std::thread::sleep(duration);
    }
}

fn login(work_dir: &Path, args: LoginArgs, http: &HttpClient) -> anyhow::Result<()> {
    let mut clock = WallClock(Instant::now());
    let stored = RegisteredIdentity::read(work_dir)?;
    let key = LoginKey::resolve(work_dir, &stored, &args.key)?;
    let server = stored.server(args.server)?;
    let previous = previous_identity(stored);
    let request = LoginRequest {
        server,
        key_details: match args.key_details {
            Some(details) => details,
            None => default_key_details()?,
        },
        timeout: args.timeout,
    };
    let opener: OpenBrowser<'_> = &open_in_browser;
    let browser = (!args.no_browser
        && browser_is_likely(
            |name| std::env::var_os(name),
            cfg!(all(unix, not(target_os = "macos"))),
        ))
    .then_some(opener);
    login_with(
        work_dir,
        previous.as_ref(),
        &request,
        http,
        &key,
        &mut clock,
        browser,
    )
}

/// The identity on disk, to tell a renewal from a rewrite. One that does not
/// resolve is reported here, since approval replaces it.
fn previous_identity(stored: RegisteredIdentity) -> Option<SubmitIdentity> {
    let empty = stored.is_empty();
    match stored.resolve(None) {
        Ok(previous) => Some(previous),
        Err(_) if empty => None,
        Err(error) => {
            eprintln!(
                "the stored identity could not be read ({error:#}); it is replaced once the key \
                 is approved"
            );
            None
        }
    }
}

/// Split from [`login`] so a test drives the whole wait against a mock server.
fn login_with(
    work_dir: &Path,
    previous: Option<&SubmitIdentity>,
    request: &LoginRequest,
    http: &HttpClient,
    key: &LoginKey,
    clock: &mut dyn Clock,
    browser: Option<OpenBrowser<'_>>,
) -> anyhow::Result<()> {
    let public_key_hex = key.public_key_hex();
    let key_id = key_id_for(public_key_hex)?;
    let auth = KeyOnlyIdentity {
        user_key_id: key_id.clone(),
        signer: key.signer()?,
    };

    // Asked once before any page is shown, so a key that already works needs
    // no approval.
    let (profile, already) = match classify(user_api::me_by_key(&request.server, http, &auth)) {
        Answer::Approved(profile) => (profile, true),
        Answer::Refused(error) => return Err(error),
        Answer::Waiting(failure) => {
            // Said before the page, so a mistyped --server is not ten silent minutes.
            if let Some(failure) = failure {
                eprintln!(
                    "could not ask {} yet ({failure}); waiting anyway",
                    request.server
                );
            }
            let url = approval_url(&request.server, public_key_hex, &request.key_details);
            println!("Approve this machine's key in the management app:");
            println!();
            println!("  {url}");
            println!();
            println!("Enter this code on the page: {}", approval_code(&key_id)?);
            println!();
            match browser.map(|open| open(&url)) {
                Some(Ok(())) => println!("Opened the page in your browser."),
                Some(Err(error)) => {
                    println!("Could not open a browser ({error}); open the address above.");
                }
                None => println!("Open the address above in a browser."),
            }
            // The page's own advice, `--generate`, is refused over an identity.
            if matches!(key, LoginKey::Kept { .. }) {
                println!(
                    "If the page says another account or a device holds this key, remove {} \
                     and run `pipette-ctl auth login --generate`.",
                    identity::identity_dir(work_dir).display()
                );
            }
            println!(
                "Waiting for the approval, up to {} (Ctrl-C to stop)...",
                request.timeout
            );
            let profile = wait_for_approval(request, http, key, &auth, clock)?;
            (profile, false)
        }
    };
    save_login(
        work_dir,
        previous,
        &request.server,
        key,
        &key_id,
        &profile,
        already,
    )
}

fn wait_for_approval(
    request: &LoginRequest,
    http: &HttpClient,
    key: &LoginKey,
    auth: &KeyOnlyIdentity,
    clock: &mut dyn Clock,
) -> anyhow::Result<UserProfile> {
    // Counted from here, so a vault's signing prompt does not eat into it.
    let start = clock.elapsed();
    let mut last_failure = None;
    loop {
        // The last poll lands on the deadline, however short the timeout.
        let remaining = request
            .timeout
            .0
            .saturating_sub(clock.elapsed().saturating_sub(start));
        if remaining.is_zero() {
            break;
        }
        clock.sleep(POLL_INTERVAL.min(remaining));
        match classify(user_api::me_by_key(&request.server, http, auth)) {
            Answer::Approved(profile) => return Ok(profile),
            Answer::Waiting(failure) => last_failure = failure,
            Answer::Refused(error) => return Err(error),
        }
    }
    let waited = format!(
        "the key was not approved within {}. Nothing was written; {}",
        request.timeout,
        key.if_unsaved(&auth.user_key_id, false)
    );
    match last_failure {
        Some(failure) => anyhow::bail!("{waited}. The last attempt failed: {failure}"),
        None => anyhow::bail!(waited),
    }
}

enum Answer {
    Approved(UserProfile),
    /// Not approved yet (`None`), or a transient failure (transport, 5xx or
    /// 429), kept for the timeout to report.
    Waiting(Option<String>),
    Refused(anyhow::Error),
}

fn classify(answer: Result<KeyApproval, Error>) -> Answer {
    match answer {
        Ok(KeyApproval::Approved(profile)) => Answer::Approved(profile),
        Ok(KeyApproval::Pending) => Answer::Waiting(None),
        Ok(KeyApproval::Refused(refusal)) => Answer::Refused(refused(&refusal)),
        // Polling every few seconds is what a rate limit catches; the next poll may pass.
        Err(error) if error.is_transient() || error.http_status() == Some(429) => {
            Answer::Waiting(Some(error.to_string()))
        }
        // A proxy's page rather than mgmt's reason: still final, and still nothing written.
        Err(error) if matches!(error.http_status(), Some(401 | 403)) => {
            Answer::Refused(anyhow::Error::new(error).context(
                "the server refused this key. Nothing was written. Run `pipette-ctl auth login` \
                 again once that is resolved",
            ))
        }
        Err(error) => Answer::Refused(error.into()),
    }
}

fn refused(refusal: &RefusalError) -> anyhow::Error {
    let next = match refusal {
        RefusalError::UserDisabled => "Ask an operator to enable the account",
        RefusalError::TimestampExpired => "Correct this machine's clock and run it again",
        RefusalError::Other(_) => "Run `pipette-ctl auth login` again once that is resolved",
    };
    anyhow::anyhow!("the server refused this key: {refusal}. Nothing was written. {next}")
}

/// Write the identity the server confirmed, unless it is the one already on
/// disk, and report it.
fn save_login(
    work_dir: &Path,
    previous: Option<&SubmitIdentity>,
    server: &ServerUrl,
    key: &LoginKey,
    key_id: &UserKeyId,
    profile: &UserProfile,
    already: bool,
) -> anyhow::Result<()> {
    let unchanged = previous.is_some_and(|previous| {
        previous.user_id == profile.user_id
            && previous.user_key_id == *key_id
            && previous.server == *server
    });
    if !unchanged {
        // The wait can be minutes long, and another command may have written
        // an identity here meanwhile.
        match key {
            LoginKey::Chosen(chosen) => identity::ensure_empty(work_dir).with_context(|| {
                let why = match chosen {
                    ChosenKey::Generated { .. } => {
                        "the key this run generated was not saved and signs nothing"
                    }
                    ChosenKey::System { .. } => "if it is not wanted",
                };
                format!(
                    "{key_id} is approved for {}, but another command wrote an identity here \
                     while this one waited; nothing was overwritten. Revoke {key_id} on the \
                     Account page: {why}",
                    profile.user_id
                )
            })?,
            // Compared by public key: a replaced `id_ed25519` keeps its path.
            LoginKey::Kept { public_key_hex, .. } => {
                let current = RegisteredIdentity::read(work_dir)?;
                if !current.key().is_some_and(|current| {
                    identity::public_key_of(current).is_ok_and(|current| current == *public_key_hex)
                }) {
                    anyhow::bail!(
                        "{key_id} is approved for {}, but the identity here changed while this \
                         run waited; nothing was overwritten. Run `pipette-ctl auth login` again",
                        profile.user_id
                    );
                }
            }
        }
        identity::write(work_dir, &profile.user_id, key_id, server, &key.to_store()).with_context(
            || {
                format!(
                    "{key_id} is approved for {}, but the identity could not be saved; {}",
                    profile.user_id,
                    // `write` puts the key down first, so a later failure leaves it recoverable.
                    key.if_unsaved(key_id, identity::key_file(work_dir).exists())
                )
            },
        )?;
    }

    // Printed: it changes which account files the next plan.
    match previous.filter(|previous| previous.user_id != profile.user_id) {
        _ if already && unchanged => println!("already signed in: this key works"),
        Some(previous) => println!(
            "this key is approved for {}, not {}; rewrote the identity",
            profile.user_id, previous.user_id
        ),
        None if already => println!("this key is already approved; wrote the identity"),
        None => {}
    }
    println!(
        "signed in as {} ({})",
        profile.user_id, profile.contact_email
    );
    println!("status:       {}", profile.status);
    println!("user_key_id:  {key_id}");
    println!("key:          {}", key.describe());
    println!("server:       {server}");
    println!(
        "identity:     {}",
        identity::identity_dir(work_dir).display()
    );
    if profile.status == UserStatus::Pending {
        println!();
        println!("An operator has to admit the account before it can file a plan.");
    }
    Ok(())
}

/// The app page that approves `public_key_hex`. The code is not in it: the
/// person approving types it from this terminal, which catches a page opened
/// for another key.
fn approval_url(
    server: &ServerUrl,
    public_key_hex: &PublicKey,
    key_details: &KeyDetails,
) -> String {
    format!(
        "{server}/app/login/cli?key={public_key_hex}&details={}",
        url::form_urlencoded::byte_serialize(key_details.as_ref().as_bytes()).collect::<String>()
    )
}

/// The code the page asks for; it must match mgmt's `keyNames`
/// (`ui/src/lib/clilogin.ts`).
fn approval_code(key_id: &UserKeyId) -> anyhow::Result<String> {
    let Some((first, second)) = key_id
        .as_ref()
        .strip_prefix("userkey_")
        .and_then(|digest| Some((digest.get(..4)?, digest.get(4..8)?)))
    else {
        anyhow::bail!("{key_id} is not a key id a code can be derived from");
    };
    Ok(format!(
        "{}-{}",
        first.to_ascii_uppercase(),
        second.to_ascii_uppercase()
    ))
}

/// SSH or a headless Linux session would open a browser nobody sees.
fn browser_is_likely(var: impl Fn(&str) -> Option<OsString>, needs_display: bool) -> bool {
    let set = |name: &str| var(name).is_some_and(|value| !value.is_empty());
    if set("SSH_CONNECTION") || set("SSH_TTY") {
        return false;
    }
    !needs_display || set("DISPLAY") || set("WAYLAND_DISPLAY")
}

/// Hand `url` to the desktop's opener as one argument, never through a shell.
/// Reaped on a thread, since `xdg-open` can wait on the browser it starts.
fn open_in_browser(url: &str) -> std::io::Result<()> {
    let mut command = if cfg!(target_os = "macos") {
        Command::new("open")
    } else if cfg!(windows) {
        let mut command = Command::new("rundll32");
        command.arg("url.dll,FileProtocolHandler");
        command
    } else {
        Command::new("xdg-open")
    };
    // Its own process group, so Ctrl-C on the wait does not reach a browser
    // `xdg-open` started as its child.
    #[cfg(unix)]
    std::os::unix::process::CommandExt::process_group(&mut command, 0);
    #[expect(clippy::disallowed_methods, reason = "every stream goes to null")]
    let mut child = command
        .arg(url)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()?;
    std::thread::spawn(move || child.wait());
    Ok(())
}

fn me(work_dir: &Path, args: MeArgs, http: &HttpClient) -> anyhow::Result<()> {
    let (server, credential) = identity::registered(work_dir)?;
    if args.local {
        println!("user_id:      {}", credential.user_id);
        println!("user_key_id:  {}", credential.user_key_id);
        println!("server:       {server}");
        println!(
            "identity:     {}",
            identity::identity_dir(work_dir).display()
        );
        return Ok(());
    }

    let profile = user_api::me(&server, http, &credential)?;
    print_profile(
        &profile,
        &credential.user_key_id,
        time::OffsetDateTime::now_utc(),
    );
    Ok(())
}

fn adopt(work_dir: &Path, args: AdoptArgs) -> anyhow::Result<()> {
    identity::ensure_empty(work_dir)?;
    let key = ChosenKey::resolve(&args.key)?;
    let public_key_hex = key.public_key_hex().clone();
    let key_id = key_id_for(&public_key_hex)?;
    identity::write(
        work_dir,
        &args.user_id,
        &key_id,
        &args.server,
        &key.to_store(),
    )?;

    println!("prepared a key for {}", args.user_id);
    println!("user_key_id: {key_id}");
    println!("public key: {public_key_hex}");
    println!("key:        {}", key.describe());
    println!(
        "identity:    {}",
        identity::identity_dir(work_dir).display()
    );
    println!();
    println!("This machine cannot sign yet. Add its key to the account on the");
    println!(
        "Account page of {}/app/, or on a machine already",
        args.server
    );
    println!("registered to it, run:");
    println!();
    println!("  pipette-ctl auth keys add --public-key {public_key_hex} \\");
    println!("    --key-details \"<what this machine is>\"");
    println!();
    println!("`pipette-ctl auth login` does both from a browser in one step.");
    Ok(())
}

fn add_key(
    work_dir: &Path,
    args: AddKeyArgs,
    http: &HttpClient,
    now: time::OffsetDateTime,
) -> anyhow::Result<()> {
    // Derived before the round trip, so the id this machine will report is
    // fixed before the server is asked to agree with it. The value itself is
    // already checked: `--public-key` parses into a `PublicKey`.
    let key_id = key_id_for(&args.public_key)?;
    let expires_at = args
        .expires_in
        .map(|lifetime| lifetime.expires_at(now))
        .transpose()?;
    let (server, credential) = identity::registered(work_dir)?;

    let response = user_api::add_key(
        &server,
        http,
        &credential,
        AddUserKeyRequest {
            public_key: args.public_key,
            key_details: args.key_details,
            expires_at: expires_at.clone(),
        },
    )?;
    reconcile_key_id(&key_id, &response.user_key_id)?;

    match expires_at {
        Some(at) => println!("added {key_id} to {}, expiring {at}", credential.user_id),
        None => println!("added {key_id} to {}", credential.user_id),
    }
    Ok(())
}

fn revoke_key(work_dir: &Path, args: RevokeKeyArgs, http: &HttpClient) -> anyhow::Result<()> {
    let (server, credential) = identity::registered(work_dir)?;
    user_api::revoke_key(&server, http, &credential, &args.user_key_id)?;

    println!("revoked {} from {}", args.user_key_id, credential.user_id);
    if args.user_key_id == credential.user_key_id {
        println!();
        println!("That was this machine's key, so nothing here can sign any more.");
        println!(
            "Delete {} and register again with a",
            identity::identity_dir(work_dir).display()
        );
        println!("fresh pre-auth key, or run `auth adopt` and have another machine add it.");
    }
    Ok(())
}

/// `local_key_id` marks which row of the account's key table is this machine,
/// which is the one an operator must not revoke by accident.
fn print_profile(profile: &UserProfile, local_key_id: &UserKeyId, now: time::OffsetDateTime) {
    println!("user_id:        {}", profile.user_id);
    println!("contact_email:  {}", profile.contact_email);
    if let Some(account_type) = &profile.account_type {
        println!("account_type:   {account_type}");
    }
    println!("status:         {}", profile.status);
    if let Some(created_at) = &profile.created_at {
        println!("created_at:     {created_at}");
    }
    if let Some(org_id) = &profile.org_id {
        println!("org_id:         {org_id}");
    }
    if let Some(org_slug) = &profile.org_slug {
        println!("org_slug:       {org_slug}");
    }
    println!(
        "permissions:    {}",
        if profile.permissions.is_empty() {
            "(none)".to_string()
        } else {
            profile.permissions.join(", ")
        }
    );
    println!();
    // Reported before the table, so an operator reads the count before the rows
    // it is missing from: an entry this build cannot read is still a
    // registration the account holds.
    let unreadable = profile.keys.unreadable;
    if unreadable > 0 {
        let noun = if unreadable == 1 { "key" } else { "keys" };
        println!("this build could not read {unreadable} registered {noun}");
        println!();
    }
    if profile.keys.entries.is_empty() {
        if unreadable == 0 {
            // Only reachable when the signing key was revoked between the request
            // being signed and the record being read.
            println!("no registered keys");
        } else {
            println!("no readable keys");
        }
        return;
    }
    let rows: Vec<KeyRow> = profile
        .keys
        .entries
        .iter()
        .map(|key| KeyRow {
            user_key_id: key.user_key_id.to_string(),
            key_details: if key.key_details.is_empty() {
                "(none)".to_string()
            } else {
                key.key_details.clone()
            },
            added_at: key
                .added_at
                .as_ref()
                .map_or_else(|| "(unknown)".to_string(), Timestamp::to_string),
            expires_at: expiry_cell(key.expires_at.as_ref(), now),
            local: if &key.user_key_id == local_key_id {
                "yes"
            } else {
                ""
            },
        })
        .collect();
    println!("{}", tabled::Table::new(rows).with(Style::sharp()));
}

/// The EXPIRES cell: `never`, the timestamp, or `expired` once it has passed,
/// so a lapsed browser key is not read as one still signing.
fn expiry_cell(expires_at: Option<&Timestamp>, now: time::OffsetDateTime) -> String {
    let Some(at) = expires_at else {
        return "never".to_string();
    };
    // `Timestamp` already parsed with this description, so this only fails
    // if the two drift apart; the raw value is still worth showing then.
    let passed = time::OffsetDateTime::parse(
        at.as_ref(),
        &time::format_description::well_known::Iso8601::PARSING,
    )
    .is_ok_and(|at| at <= now);
    if passed {
        format!("expired {at}")
    } else {
        at.to_string()
    }
}

/// Fail when the server names a different key id than the one derived here.
///
/// Both sides digest the same public key, so a disagreement means one of them
/// is not deriving what the spec says. Writing the local value anyway would
/// produce an identity that signs with a `keyid` the server cannot resolve,
/// and every later request would 401 with nothing pointing here.
fn reconcile_key_id(derived: &UserKeyId, returned: &UserKeyId) -> anyhow::Result<()> {
    if returned != derived {
        anyhow::bail!(
            "the server registered the key as {returned}, but its public key derives {derived}. \
             Refusing to save an identity that cannot sign. The account now holds that \
             registration; revoke {returned} before you try again"
        );
    }
    Ok(())
}

/// Reprint what `auth adopt` printed once.
///
/// The public half is derived, never stored beside the private one, so this is
/// how a machine whose adopt output scrolled away recovers the value the other
/// machine needs. It reads the key alone rather than the whole identity: on a
/// machine waiting to be adopted, the key is registered nowhere yet.
fn pubkey(work_dir: &Path) -> anyhow::Result<()> {
    let public_key_hex = identity::public_key(work_dir)?;
    println!("user_key_id: {}", key_id_for(&public_key_hex)?);
    println!("public key: {public_key_hex}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use httpmock::{Method::DELETE, Method::GET, Method::POST, MockServer};
    use rstest::rstest;

    use super::*;

    fn work_dir() -> anyhow::Result<tempfile::TempDir> {
        Ok(tempfile::tempdir()?)
    }

    /// A working directory that has adopted `user_9f2c` against `server`, plus
    /// the credential it now signs with. Not an `#[rstest]` fixture: the caller
    /// needs the same `MockServer` afterwards.
    fn adopted(
        server: &MockServer,
    ) -> anyhow::Result<(tempfile::TempDir, pipette_mgmt_client::UserIdentity)> {
        let tmp = work_dir()?;
        adopt(
            tmp.path(),
            AdoptArgs {
                user_id: UserId::try_new("user_9f2c")?,
                server: ServerUrl::try_new(server.base_url())?,
                key: generating(),
            },
        )?;
        let (_, credential) = identity::registered(tmp.path())?;
        Ok((tmp, credential))
    }

    /// `--generate`, which is what a test wants: the machine running it has
    /// keys of its own, and none of them are this test's business.
    fn generating() -> KeyArgs {
        KeyArgs {
            ssh_key: None,
            generate: true,
        }
    }

    fn user_preauth_key() -> anyhow::Result<UserPreauthKey> {
        Ok(UserPreauthKey::try_new("user_preauth_k1.secret")?)
    }

    fn http_ctx() -> anyhow::Result<HttpClient> {
        HttpClient::new("pipette-test/0").map_err(|error| anyhow::anyhow!(error))
    }

    fn register_args(server: ServerUrl) -> anyhow::Result<RegisterArgs> {
        Ok(RegisterArgs {
            user_preauth_key: Some(user_preauth_key()?),
            contact_email: None,
            key_details: Some(KeyDetails::try_new("work laptop")?),
            server,
            key: generating(),
        })
    }

    #[test]
    fn registering_a_machine_key_copies_no_private_key() -> anyhow::Result<()> {
        let tmp = work_dir()?;
        let work_dir = tmp.path();
        let (_, public_key_hex) = generate_keypair()?;
        let key_id = key_id_for(&public_key_hex)?;

        let server = MockServer::start();
        let mock = server.mock(|when, then| {
            when.method(POST)
                .path("/api/v1/users/register")
                .body_includes(public_key_hex.as_ref());
            then.status(201).json_body(serde_json::json!({
                "user_id": "user_9f2c",
                "status": "active",
                "key_id": key_id.as_ref(),
            }));
        });

        register_with_key(
            work_dir,
            register_args(ServerUrl::try_new(server.base_url())?)?,
            &http_ctx()?,
            ChosenKey::System {
                public_key_hex: public_key_hex.clone(),
                fingerprint: "SHA256:fixture".to_string(),
                found_at: "agent (1Password)".to_string(),
            },
        )?;
        mock.assert();

        assert!(
            !identity::key_file(work_dir).exists(),
            "an agent's key must not be copied beside the working directory"
        );
        assert_eq!(identity::public_key(work_dir)?, public_key_hex);
        Ok(())
    }

    #[test]
    fn register_writes_the_identity_the_server_confirmed() -> anyhow::Result<()> {
        let tmp = work_dir()?;
        let work_dir = tmp.path();
        let (private_key_hex, public_key_hex) = generate_keypair()?;
        let key_id = key_id_for(&public_key_hex)?;

        let server = MockServer::start();
        let mock = server.mock(|when, then| {
            when.method(POST).path("/api/v1/users/register").json_body_includes(
                r#"{"user_preauth_key": "user_preauth_k1.secret", "key_details": "work laptop"}"#,
            );
            then.status(201).json_body(serde_json::json!({
                "user_id": "user_9f2c",
                "status": "active",
                "key_id": key_id.as_ref(),
            }));
        });

        register_with_key(
            work_dir,
            register_args(ServerUrl::try_new(server.base_url())?)?,
            &http_ctx()?,
            ChosenKey::generated((private_key_hex, public_key_hex)),
        )?;
        mock.assert();

        let (stored_server, credential) = identity::registered(work_dir)?;
        assert_eq!(credential.user_id.as_ref(), "user_9f2c");
        assert_eq!(stored_server, ServerUrl::try_new(server.base_url())?);
        assert_eq!(credential.user_key_id, key_id);
        assert_eq!(
            credential.user_key_id,
            key_id_for(&identity::public_key(work_dir)?)?
        );
        Ok(())
    }

    #[test]
    fn register_labels_the_key_with_this_machine_by_default() -> anyhow::Result<()> {
        let tmp = work_dir()?;
        let server = MockServer::start();
        let details = default_key_details()?;
        let (private_key_hex, public_key_hex) = generate_keypair()?;
        let key_id = key_id_for(&public_key_hex)?;
        let mock = server.mock(|when, then| {
            when.method(POST)
                .path("/api/v1/users/register")
                .json_body_includes(
                    serde_json::json!({ "key_details": details.as_ref() }).to_string(),
                );
            then.status(201).json_body(serde_json::json!({
                "user_id": "user_9f2c",
                "status": "active",
                "key_id": key_id.as_ref(),
            }));
        });

        register_with_key(
            tmp.path(),
            RegisterArgs {
                key_details: None,
                ..register_args(ServerUrl::try_new(server.base_url())?)?
            },
            &http_ctx()?,
            ChosenKey::generated((private_key_hex, public_key_hex)),
        )?;
        mock.assert();
        Ok(())
    }

    #[rstest]
    #[case::user_and_host(Some("yk"), Some("box"), "yk@box")]
    #[case::user_only(Some("yk"), None, "yk")]
    #[case::host_only(None, Some("box"), "box")]
    #[case::neither(None, None, "unknown machine")]
    fn the_label_names_whatever_the_host_offered(
        #[case] user: Option<&str>,
        #[case] hostname: Option<&str>,
        #[case] expected: &str,
    ) -> anyhow::Result<()> {
        let label = key_details_label(user.map(str::to_string), hostname.map(str::to_string));
        anyhow::ensure!(label == expected, "unexpected label: {label}");
        KeyDetails::try_new(label)?;
        Ok(())
    }

    #[test]
    fn a_request_registers_the_key_under_a_pending_account() -> anyhow::Result<()> {
        let tmp = work_dir()?;
        let work_dir = tmp.path();
        let server = MockServer::start();
        let (private_key_hex, public_key_hex) = generate_keypair()?;
        let key_id = key_id_for(&public_key_hex)?;
        let mock = server.mock(|when, then| {
            when.method(POST)
                .path("/api/v1/users/register")
                .json_body_includes(r#"{"contact_email": "ops@example.com"}"#)
                // The invite is absent rather than null, so the server can tell
                // a request from a malformed invited registration.
                .body_excludes("user_preauth_key");
            then.status(201).json_body(serde_json::json!({
                "user_id": "user_9f2c",
                "status": "pending",
                "key_id": key_id.as_ref(),
            }));
        });

        register_with_key(
            work_dir,
            RegisterArgs {
                user_preauth_key: None,
                contact_email: Some(ContactEmail::try_new("ops@example.com")?),
                ..register_args(ServerUrl::try_new(server.base_url())?)?
            },
            &http_ctx()?,
            ChosenKey::generated((private_key_hex, public_key_hex)),
        )?;
        mock.assert();

        let (_, credential) = identity::registered(work_dir)?;
        assert_eq!(credential.user_id.as_ref(), "user_9f2c");
        assert_eq!(
            credential.user_key_id,
            key_id_for(&identity::public_key(work_dir)?)?
        );
        Ok(())
    }

    /// Checked on `pubkey`, which reaches no server. Every verb but `keys
    /// available` calls the same `workspace()` guard in `execute`.
    #[test]
    fn auth_refuses_a_working_directory_with_no_workspace() -> anyhow::Result<()> {
        let tmp = work_dir()?;
        let error = AuthArgs {
            command: AuthCommand::Pubkey,
        }
        .execute(Some(tmp.path()))
        .err()
        .ok_or_else(|| anyhow::anyhow!("auth should refuse an uninitialized directory"))?;
        let message = format!("{error:#}");
        anyhow::ensure!(
            message.contains("no pipette-ctl workspace") && message.contains("init"),
            "unexpected error: {message}"
        );

        crate::workspace::PipetteCtlWorkspace::init(tmp.path())?;
        let error = format!(
            "{:#}",
            AuthArgs {
                command: AuthCommand::Pubkey,
            }
            .execute(Some(tmp.path()))
            .err()
            .ok_or_else(|| anyhow::anyhow!("pubkey should refuse without a key"))?
        );
        anyhow::ensure!(
            !error.contains("no pipette-ctl workspace"),
            "unexpected error: {error}"
        );
        Ok(())
    }

    #[rstest]
    #[case::refused_user_preauth_key(403, "pre-auth key already spent")]
    #[case::key_id_disagrees(
        201,
        r#"{"user_id":"user_9f2c","status":"active","key_id":"userkey_something_else"}"#
    )]
    fn a_rejected_registration_leaves_no_identity_on_disk(
        #[case] status: u16,
        #[case] body: &str,
    ) -> anyhow::Result<()> {
        let tmp = work_dir()?;
        let work_dir = tmp.path();
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(POST).path("/api/v1/users/register");
            then.status(status).body(body);
        });

        assert!(
            register(
                work_dir,
                register_args(ServerUrl::try_new(server.base_url())?)?,
                &http_ctx()?
            )
            .is_err()
        );
        identity::ensure_empty(work_dir)?;
        Ok(())
    }

    /// A dropped answer may still have registered the key, so the error names the way back.
    #[rstest]
    #[case::generated_key_is_lost(true, "pipette-ctl auth keys revoke")]
    #[case::machine_key_logs_in(
        false,
        "pipette-ctl auth login --server {server} --ssh-key SHA256:test"
    )]
    fn a_lost_register_answer_names_the_recovery(
        #[case] generated: bool,
        #[case] recovery: &str,
    ) -> anyhow::Result<()> {
        let tmp = work_dir()?;
        let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
        let server = ServerUrl::try_new(format!("http://{}", listener.local_addr()?))?;
        // Reads the request, then hangs up without answering.
        std::thread::spawn(move || {
            if let Ok((mut socket, _)) = listener.accept() {
                let _ = std::io::Read::read(&mut socket, &mut [0; 4096]);
            }
        });
        let (private_key_hex, public_key_hex) = generate_keypair()?;
        let key = if generated {
            ChosenKey::generated((private_key_hex, public_key_hex))
        } else {
            ChosenKey::System {
                public_key_hex,
                fingerprint: "SHA256:test".to_string(),
                found_at: "test".to_string(),
            }
        };

        let recovery = recovery.replace("{server}", &server.to_string());

        let Err(error) = register_with_key(tmp.path(), register_args(server)?, &http_ctx()?, key)
        else {
            anyhow::bail!("registered with no answer");
        };
        let error = format!("{error:#}");
        anyhow::ensure!(
            error.contains("may have registered") && error.contains(&recovery),
            "unexpected error: {error}"
        );
        identity::ensure_empty(tmp.path())?;
        Ok(())
    }

    /// The stored key id must match the printed public key, so `keys add` on
    /// the other machine registers the same id.
    #[test]
    fn adopt_writes_an_identity_the_account_does_not_know_yet() -> anyhow::Result<()> {
        let tmp = work_dir()?;
        let work_dir = tmp.path();
        adopt(
            work_dir,
            AdoptArgs {
                user_id: UserId::try_new("user_9f2c")?,
                server: ServerUrl::try_new("https://mgmt.example.com")?,
                key: generating(),
            },
        )?;

        let (_, credential) = identity::registered(work_dir)?;
        assert_eq!(credential.user_id.as_ref(), "user_9f2c");
        assert_eq!(
            credential.user_key_id,
            key_id_for(&identity::public_key(work_dir)?)?
        );
        Ok(())
    }

    #[test]
    fn a_public_key_of_the_wrong_length_is_refused_at_the_flag() -> anyhow::Result<()> {
        let error = PublicKey::try_new("00".repeat(31))
            .err()
            .ok_or_else(|| anyhow::anyhow!("a short public key should be refused"))?;
        anyhow::ensure!(
            error.to_string().contains("public key must be 32 bytes"),
            "unexpected error: {error}"
        );
        Ok(())
    }

    #[test]
    fn add_key_registers_the_pasted_key_signed_by_this_machine() -> anyhow::Result<()> {
        let server = MockServer::start();
        let (tmp, credential) = adopted(&server)?;
        let (_, public_key) = generate_keypair()?;
        let derived = key_id_for(&public_key)?;
        let signer = credential.user_key_id.to_string();

        let mock = server.mock(|when, then| {
            when.method(POST)
                .path("/api/v1/users/me/keys")
                .is_true(move |request| crate::test_support::signed_as(request, &signer))
                .json_body_includes(format!(
                    r#"{{"public_key": "{public_key}", "key_details": "build server"}}"#
                ));
            then.status(201)
                .json_body(serde_json::json!({"key_id": derived.as_ref()}));
        });

        add_key(
            tmp.path(),
            AddKeyArgs {
                public_key,
                key_details: KeyDetails::try_new("build server")?,
                expires_in: None,
            },
            &http_ctx()?,
            time::OffsetDateTime::now_utc(),
        )?;
        mock.assert();
        Ok(())
    }

    #[test]
    fn add_key_sends_when_an_expiring_key_stops_signing() -> anyhow::Result<()> {
        let server = MockServer::start();
        let (tmp, _) = adopted(&server)?;
        let (_, public_key) = generate_keypair()?;
        let derived = key_id_for(&public_key)?;

        let mock = server.mock(|when, then| {
            when.method(POST)
                .path("/api/v1/users/me/keys")
                .json_body(serde_json::json!({
                    "public_key": public_key.as_ref(),
                    "key_details": "console app, Firefox on Linux",
                    "expires_at": "2026-10-11T12:00:00Z",
                }));
            then.status(201)
                .json_body(serde_json::json!({"key_id": derived.as_ref()}));
        });

        add_key(
            tmp.path(),
            AddKeyArgs {
                public_key: public_key.clone(),
                key_details: KeyDetails::try_new("console app, Firefox on Linux")?,
                expires_in: Some(KeyLifetime(7)),
            },
            &http_ctx()?,
            time::OffsetDateTime::parse(
                "2026-10-04T12:00:00Z",
                &time::format_description::well_known::Rfc3339,
            )?,
        )?;
        mock.assert();
        Ok(())
    }

    #[rstest]
    #[case::one_day("1d", Some(1))]
    #[case::a_week("7d", Some(7))]
    #[case::the_servers_cap("30d", Some(30))]
    #[case::past_the_servers_cap("31d", Some(31))]
    #[case::zero("0d", None)]
    #[case::no_unit("7", None)]
    #[case::hours("12h", None)]
    #[case::signed("+7d", None)]
    #[case::empty("d", None)]
    #[case::overflows_the_type("70000d", None)]
    fn expires_in_takes_whole_days_and_leaves_the_cap_to_the_server(
        #[case] raw: &str,
        #[case] days: Option<u16>,
    ) {
        assert_eq!(
            raw.parse::<KeyLifetime>().ok(),
            days.map(KeyLifetime),
            "{raw}"
        );
    }

    #[rstest]
    #[case::no_expiry(None, "never")]
    #[case::ahead(Some("2026-10-11T12:00:00Z"), "2026-10-11T12:00:00Z")]
    #[case::passed(Some("2026-10-01T12:00:00Z"), "expired 2026-10-01T12:00:00Z")]
    #[case::exactly_now(Some("2026-10-04T12:00:00Z"), "expired 2026-10-04T12:00:00Z")]
    fn expiry_cell_marks_a_key_past_its_expiry(
        #[case] expires_at: Option<&str>,
        #[case] cell: &str,
    ) -> anyhow::Result<()> {
        let now = time::OffsetDateTime::parse(
            "2026-10-04T12:00:00Z",
            &time::format_description::well_known::Rfc3339,
        )?;
        let expires_at = expires_at.map(Timestamp::try_new).transpose()?;
        assert_eq!(expiry_cell(expires_at.as_ref(), now), cell);
        Ok(())
    }

    #[test]
    fn add_key_refuses_a_key_id_the_server_disagrees_on() -> anyhow::Result<()> {
        let server = MockServer::start();
        let (tmp, _) = adopted(&server)?;
        let (_, public_key) = generate_keypair()?;
        server.mock(|when, then| {
            when.method(POST).path("/api/v1/users/me/keys");
            then.status(201)
                .json_body(serde_json::json!({"key_id": "userkey_something_else"}));
        });

        assert!(
            add_key(
                tmp.path(),
                AddKeyArgs {
                    public_key,
                    key_details: KeyDetails::try_new("build server")?,
                    expires_in: None,
                },
                &http_ctx()?,
                time::OffsetDateTime::now_utc(),
            )
            .is_err()
        );
        Ok(())
    }

    #[test]
    fn revoke_deletes_the_named_key() -> anyhow::Result<()> {
        let server = MockServer::start();
        let (tmp, credential) = adopted(&server)?;
        let work_dir = tmp.path();
        let signer = credential.user_key_id.to_string();

        let mock = server.mock(|when, then| {
            when.method(DELETE)
                .path("/api/v1/users/me/keys/userkey_retired")
                .is_true(move |request| crate::test_support::signed_as(request, &signer));
            then.status(204);
        });
        revoke_key(
            work_dir,
            RevokeKeyArgs {
                user_key_id: UserKeyId::try_new("userkey_retired")?,
            },
            &http_ctx()?,
        )?;
        mock.assert();
        Ok(())
    }

    #[test]
    fn a_machine_with_no_account_is_told_how_to_get_one() -> anyhow::Result<()> {
        let tmp = work_dir()?;
        let work_dir = tmp.path();
        let error = me(work_dir, MeArgs { local: true }, &http_ctx()?)
            .err()
            .ok_or_else(|| anyhow::anyhow!("me should refuse without an identity"))?
            .to_string();
        anyhow::ensure!(error.contains("auth register"), "unexpected error: {error}");
        anyhow::ensure!(error.contains("adopt"), "unexpected error: {error}");
        Ok(())
    }

    /// The RFC 8032 test-1 key, which the app's tests pin the same code for.
    const RFC8032_KEY: &str = "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a";

    #[test]
    fn the_code_is_the_start_of_the_key_digest() -> anyhow::Result<()> {
        let key_id = key_id_for(&PublicKey::try_new(RFC8032_KEY)?)?;
        assert!(key_id.as_ref().starts_with("userkey_21fe31df"), "{key_id}");
        assert_eq!(approval_code(&key_id)?, "21FE-31DF");
        Ok(())
    }

    /// The server is normalized without its trailing slash, so a prefix stays
    /// and no `//` appears; `details` is form-encoded, as `URLSearchParams` reads it.
    #[rstest]
    #[case::bare("https://mgmt.example.test", "https://mgmt.example.test/app/login/cli")]
    #[case::trailing_slash(
        "https://mgmt.example.test/",
        "https://mgmt.example.test/app/login/cli"
    )]
    #[case::path_prefix(
        "https://mgmt.example.test/pipette/",
        "https://mgmt.example.test/pipette/app/login/cli"
    )]
    fn the_url_names_the_key_and_its_details(
        #[case] server: &str,
        #[case] page: &str,
    ) -> anyhow::Result<()> {
        let url = approval_url(
            &ServerUrl::try_new(server)?,
            &PublicKey::try_new(RFC8032_KEY)?,
            &KeyDetails::try_new("yk@box name/é&x=1")?,
        );
        assert_eq!(
            url,
            format!("{page}?key={RFC8032_KEY}&details=yk%40box+name%2F%C3%A9%26x%3D1")
        );
        Ok(())
    }

    #[rstest]
    #[case::ssh_connection(&[("SSH_CONNECTION", "10.0.0.1 22 10.0.0.2 22"), ("DISPLAY", ":0")], true, false)]
    #[case::ssh_tty(&[("SSH_TTY", "/dev/pts/1")], false, false)]
    #[case::linux_with_x(&[("DISPLAY", ":0")], true, true)]
    #[case::linux_with_wayland(&[("WAYLAND_DISPLAY", "wayland-0")], true, true)]
    #[case::linux_headless(&[], true, false)]
    #[case::linux_empty_display(&[("DISPLAY", "")], true, false)]
    #[case::macos(&[], false, true)]
    fn a_browser_is_opened_only_where_it_reaches_someone(
        #[case] vars: &[(&str, &str)],
        #[case] needs_display: bool,
        #[case] expected: bool,
    ) -> anyhow::Result<()> {
        let var = |name: &str| {
            vars.iter()
                .find(|(key, _)| *key == name)
                .map(|(_, value)| OsString::from(value))
        };
        assert_eq!(browser_is_likely(var, needs_display), expected);
        Ok(())
    }

    const NOT_A_TIMEOUT: &str = "expected whole seconds, minutes or hours";

    /// `Ok` is the seconds parsed and how they print back.
    #[rstest]
    #[case::the_default("10m", Ok((600, "10m")))]
    #[case::seconds("90s", Ok((90, "90s")))]
    #[case::an_hour("1h", Ok((3600, "1h")))]
    #[case::minutes_typed_as_seconds("120s", Ok((120, "2m")))]
    #[case::zero("0m", Err(NOT_A_TIMEOUT))]
    #[case::no_unit("10", Err(NOT_A_TIMEOUT))]
    #[case::days("1d", Err(NOT_A_TIMEOUT))]
    #[case::signed("+5m", Err(NOT_A_TIMEOUT))]
    #[case::empty("m", Err(NOT_A_TIMEOUT))]
    fn timeout_takes_whole_seconds_minutes_or_hours(
        #[case] raw: &str,
        #[case] expected: Result<(u64, &str), &str>,
    ) -> anyhow::Result<()> {
        match (raw.parse::<LoginTimeout>(), expected) {
            (Ok(parsed), Ok((seconds, shown))) => {
                assert_eq!(parsed, LoginTimeout(Duration::from_secs(seconds)), "{raw}");
                assert_eq!(parsed.to_string(), shown);
            }
            (Err(error), Err(reason)) => {
                assert!(error.to_string().contains(reason), "{raw}: {error}");
            }
            (got, expected) => anyhow::bail!("{raw}: got {got:?}, expected {expected:?}"),
        }
        Ok(())
    }

    /// Advances only when slept on, and runs `on_sleep` with the count, so a
    /// test can change the server's answer between polls.
    struct FakeClock<F: FnMut(u32)> {
        elapsed: Duration,
        sleeps: u32,
        on_sleep: F,
    }

    impl<F: FnMut(u32)> Clock for FakeClock<F> {
        fn elapsed(&self) -> Duration {
            self.elapsed
        }

        fn sleep(&mut self, duration: Duration) {
            self.elapsed += duration;
            self.sleeps += 1;
            (self.on_sleep)(self.sleeps);
        }
    }

    /// Starts an hour in, so the deadline cases pin that `--timeout` counts
    /// from the page rather than from launch.
    fn clock<F: FnMut(u32)>(on_sleep: F) -> FakeClock<F> {
        FakeClock {
            elapsed: Duration::from_secs(60 * 60),
            sleeps: 0,
            on_sleep,
        }
    }

    fn login_request(server: &MockServer, timeout: &str) -> anyhow::Result<LoginRequest> {
        Ok(LoginRequest {
            server: ServerUrl::try_new(server.base_url())?,
            key_details: KeyDetails::try_new("yk@box")?,
            timeout: timeout.parse()?,
        })
    }

    fn me_body(user_id: &str) -> serde_json::Value {
        serde_json::json!({
            "user_id": user_id,
            "contact_email": "ops@example.com",
            "account_type": "human",
            "status": "active",
            "created_at": "2026-01-02T03:04:05Z",
            "permissions": [],
            "keys": [],
            "may": {
                "read_roster": "none",
                "administer_roster": "none",
                "read_fleet": false,
                "administer_fleet": "none",
                "read_plans": "none",
                "file_plans": false,
                "create_orgs": false,
                "grant": [],
                "grant_service": [],
            },
        })
    }

    const NOT_FOUND: (u16, &str) = (401, "key not found");
    const EXPIRED: (u16, &str) = (401, "key expired");
    const BAD_GATEWAY: (u16, &str) = (502, "bad gateway");
    const RATE_LIMITED: (u16, &str) = (429, "too many requests");
    const DISABLED: (u16, &str) = (403, "user is disabled");

    struct LoginRun {
        result: anyhow::Result<()>,
        sleeps: u32,
        hits: usize,
        opened: Vec<String>,
    }

    /// `/api/v1/users/me` answers `answers[n]` after the n-th sleep and keeps the
    /// last one, to key-only requests only. A `200` names the account.
    fn run_login(
        tmp: &Path,
        server: &MockServer,
        key: &LoginKey,
        timeout: &str,
        answers: &[(u16, &str)],
    ) -> anyhow::Result<LoginRun> {
        let key_id = key_id_for(key.public_key_hex())?.to_string();
        let answer = |&(status, text): &(u16, &str)| {
            let expected = key_id.clone();
            let body = if status == 200 {
                me_body(text)
            } else {
                serde_json::json!({ "error": text })
            };
            server.mock(move |when, then| {
                when.method(GET)
                    .path("/api/v1/users/me")
                    .is_true(move |request| crate::test_support::signed_as(request, &expected));
                then.status(status).json_body(body);
            })
        };
        let [first, rest @ ..] = answers else {
            anyhow::bail!("a login needs a first answer");
        };
        let (mut current, mut rest, mut hits) = (answer(first), rest.iter(), 0);
        let opened = std::cell::RefCell::new(Vec::new());
        let open = |url: &str| {
            opened.borrow_mut().push(url.to_string());
            Ok(())
        };
        let mut clock = clock(|_| {
            if let Some(next) = rest.next() {
                hits += current.calls();
                current.delete();
                current = answer(next);
            }
        });
        let previous = previous_identity(RegisteredIdentity::read(tmp)?);
        let result = login_with(
            tmp,
            previous.as_ref(),
            &login_request(server, timeout)?,
            &http_ctx()?,
            key,
            &mut clock,
            Some(&open),
        );
        // The last use of `clock`, which ends its borrows of `hits` and `current`.
        let sleeps = clock.sleeps;
        Ok(LoginRun {
            result,
            sleeps,
            hits: hits + current.calls(),
            opened: opened.into_inner(),
        })
    }

    /// The key of an identity adopted as `user_9f2c`, as `auth login` keeps it.
    fn kept_key(tmp: &Path) -> anyhow::Result<LoginKey> {
        LoginKey::resolve(
            tmp,
            &RegisteredIdentity::read(tmp)?,
            &KeyArgs {
                ssh_key: None,
                generate: false,
            },
        )
    }

    /// The identity's two files as bytes, `None` where absent.
    #[derive(Debug, PartialEq, Eq)]
    struct IdentityFiles {
        user_json: Option<Vec<u8>>,
        key: Option<Vec<u8>>,
    }

    fn identity_files(tmp: &Path) -> anyhow::Result<IdentityFiles> {
        let read = |path: PathBuf| -> anyhow::Result<Option<Vec<u8>>> {
            Ok(path.exists().then(|| std::fs::read(&path)).transpose()?)
        };
        Ok(IdentityFiles {
            user_json: read(identity::identity_dir(tmp).join("user.json"))?,
            key: read(identity::key_file(tmp))?,
        })
    }

    /// `kept` logs in with the key of an identity adopted as `user_9f2c`, else
    /// with a generated one in an empty directory. Polls are `sleeps + 1`.
    #[rstest]
    #[case::approved_after_two_polls(false, &[NOT_FOUND, NOT_FOUND, (200, "user_9f2c")], "10m", Ok("user_9f2c"), 2, true)]
    #[case::waits_through_errors(false, &[BAD_GATEWAY, EXPIRED, BAD_GATEWAY, (200, "user_9f2c")], "10m", Ok("user_9f2c"), 3, true)]
    #[case::waits_through_a_rate_limit(false, &[RATE_LIMITED, NOT_FOUND, RATE_LIMITED, (200, "user_9f2c")], "10m", Ok("user_9f2c"), 3, true)]
    #[case::reports_the_last_rate_limit(false, &[NOT_FOUND, RATE_LIMITED], "6s", Err("The last attempt failed: request returned 429"), 2, true)]
    #[case::already_approved(false, &[(200, "user_9f2c")], "10m", Ok("user_9f2c"), 0, false)]
    #[case::disabled_account(false, &[DISABLED], "10m", Err("user is disabled. Nothing was written. Ask an operator"), 0, false)]
    #[case::disabled_mid_wait(false, &[NOT_FOUND, DISABLED], "10m", Err("user is disabled. Nothing was written. Ask an operator"), 1, true)]
    #[case::skewed_clock(false, &[(401, "timestamp expired")], "10m", Err("Correct this machine's clock"), 0, false)]
    #[case::gives_up_at_the_timeout(false, &[NOT_FOUND], "9s", Err("not approved within 9s"), 3, true)]
    // The last poll lands on the deadline: 3 + 3 + 3 + 1.
    #[case::polls_at_the_deadline(false, &[NOT_FOUND], "10s", Err("not approved within 10s"), 4, true)]
    #[case::polls_under_a_short_timeout(false, &[NOT_FOUND], "1s", Err("not approved within 1s"), 1, true)]
    #[case::working_identity(true, &[(200, "user_9f2c")], "10m", Ok("user_9f2c"), 0, false)]
    #[case::renews_an_expired_key(true, &[EXPIRED, EXPIRED, (200, "user_9f2c")], "10m", Ok("user_9f2c"), 2, true)]
    #[case::revoked_key_approved_by_another_account(true, &[NOT_FOUND, (200, "user_a1b2")], "10m", Ok("user_a1b2"), 1, true)]
    #[case::renewal_times_out(true, &[EXPIRED], "6s", Err("not approved within 6s"), 2, true)]
    #[case::renewal_refused(true, &[NOT_FOUND, DISABLED], "10m", Err("user is disabled"), 1, true)]
    fn login_settles_on_the_servers_answers(
        #[case] kept: bool,
        #[case] answers: &[(u16, &str)],
        #[case] timeout: &str,
        #[case] outcome: Result<&str, &str>,
        #[case] sleeps: u32,
        #[case] opens_browser: bool,
    ) -> anyhow::Result<()> {
        let server = MockServer::start();
        let (tmp, key) = if kept {
            let (tmp, _) = adopted(&server)?;
            let key = kept_key(tmp.path())?;
            (tmp, key)
        } else {
            (
                work_dir()?,
                LoginKey::Chosen(ChosenKey::generated(generate_keypair()?)),
            )
        };
        let before = identity_files(tmp.path())?;

        let run = run_login(tmp.path(), &server, &key, timeout, answers)?;
        let after = identity_files(tmp.path())?;
        assert_eq!(run.sleeps, sleeps);
        assert_eq!(run.hits, usize::try_from(sleeps)? + 1);
        let page = format!(
            "{}/app/login/cli?key={}&details=yk%40box",
            server.base_url(),
            key.public_key_hex()
        );
        assert_eq!(run.opened, if opens_browser { vec![page] } else { vec![] });
        match outcome {
            Ok(user_id) => {
                run.result?;
                let (stored_server, credential) = identity::registered(tmp.path())?;
                assert_eq!(credential.user_id.as_ref(), user_id);
                assert_eq!(credential.user_key_id, key_id_for(key.public_key_hex())?);
                assert_eq!(stored_server, ServerUrl::try_new(server.base_url())?);
                assert!(after.key.is_some());
                // The kept key stays as it was, and so does an account it renews.
                if kept {
                    assert_eq!(after.key, before.key);
                    assert_eq!(after.user_json == before.user_json, user_id == "user_9f2c");
                }
            }
            Err(reason) => {
                let error = run
                    .result
                    .err()
                    .ok_or_else(|| anyhow::anyhow!("the login should fail"))?;
                assert!(format!("{error:#}").contains(reason), "{error:#}");
                assert_eq!(after, before);
            }
        }
        Ok(())
    }

    /// What the working directory holds before a generated-key login.
    #[derive(Debug, Clone, Copy)]
    enum Before {
        Empty,
        /// An identity another command wrote while the login waited.
        Adopted,
        /// A dangling link where the identity directory goes: it reads as no
        /// identity, and no directory can be created there.
        Unwritable,
    }

    #[rstest]
    #[case::times_out(Before::Empty, &[NOT_FOUND], "3s", "revoke")]
    #[case::identity_written_meanwhile(Before::Adopted, &[NOT_FOUND, (200, "user_a1b2")], "10m", "Revoke")]
    #[cfg_attr(not(unix), ignore = "the setup is a symlink")]
    #[case::cannot_save(Before::Unwritable, &[NOT_FOUND, (200, "user_9f2c")], "10m", "revoke")]
    fn an_unsaved_generated_key_is_named_for_revocation(
        #[case] before: Before,
        #[case] answers: &[(u16, &str)],
        #[case] timeout: &str,
        #[case] verb: &str,
    ) -> anyhow::Result<()> {
        let server = MockServer::start();
        let tmp = match before {
            Before::Empty => work_dir()?,
            Before::Adopted => adopted(&server)?.0,
            Before::Unwritable => {
                let tmp = work_dir()?;
                let dir = identity::identity_dir(tmp.path());
                std::fs::create_dir_all(dir.parent().ok_or_else(|| anyhow::anyhow!("no parent"))?)?;
                #[cfg(unix)]
                std::os::unix::fs::symlink(tmp.path().join("gone"), &dir)?;
                tmp
            }
        };
        let before_login = identity_files(tmp.path())?;

        let key = LoginKey::Chosen(ChosenKey::generated(generate_keypair()?));
        let run = run_login(tmp.path(), &server, &key, timeout, answers)?;
        let error = run
            .result
            .err()
            .ok_or_else(|| anyhow::anyhow!("{before:?}: the login should fail"))?;
        let key_id = key_id_for(key.public_key_hex())?;
        assert!(
            error.to_string().contains(&format!("{verb} {key_id}")),
            "{before:?}: {error}"
        );
        assert_eq!(identity_files(tmp.path())?, before_login);
        Ok(())
    }

    #[test]
    fn a_kept_key_on_another_server_rewrites_the_server() -> anyhow::Result<()> {
        let stored_server = MockServer::start();
        let (tmp, _) = adopted(&stored_server)?;
        let key_before = std::fs::read(identity::key_file(tmp.path()))?;
        let key = kept_key(tmp.path())?;
        let server = MockServer::start();
        run_login(tmp.path(), &server, &key, "10m", &[(200, "user_9f2c")])?.result?;
        let (written_server, credential) = identity::registered(tmp.path())?;
        assert_eq!(written_server, ServerUrl::try_new(server.base_url())?);
        assert_eq!(credential.user_id.as_ref(), "user_9f2c");
        assert_eq!(std::fs::read(identity::key_file(tmp.path()))?, key_before);
        Ok(())
    }

    /// A kept key replaced under the same path while the login waited is not
    /// written over: `user.json` would name one key and the file hold another.
    #[test]
    fn a_kept_key_replaced_meanwhile_is_not_overwritten() -> anyhow::Result<()> {
        let server = MockServer::start();
        let (tmp, _) = adopted(&server)?;
        let key = kept_key(tmp.path())?;
        std::fs::remove_dir_all(identity::identity_dir(tmp.path()))?;
        adopt(
            tmp.path(),
            AdoptArgs {
                user_id: UserId::try_new("user_9f2c")?,
                server: ServerUrl::try_new(server.base_url())?,
                key: generating(),
            },
        )?;
        let before = identity_files(tmp.path())?;
        let key_id = key_id_for(key.public_key_hex())?;
        let profile: UserProfile = serde_json::from_value(me_body("user_a1b2"))?;
        let error = save_login(
            tmp.path(),
            None,
            &ServerUrl::try_new(server.base_url())?,
            &key,
            &key_id,
            &profile,
            false,
        )
        .err()
        .ok_or_else(|| anyhow::anyhow!("a replaced key should not be written over"))?;
        assert!(
            error.to_string().contains("changed while this run waited"),
            "{error}"
        );
        assert_eq!(identity_files(tmp.path())?, before);
        Ok(())
    }

    #[test]
    fn a_refusal_without_mgmts_reason_still_ends_the_wait() -> anyhow::Result<()> {
        let proxy = Error::HttpStatus {
            method: "GET".to_string(),
            url: "https://mgmt.example.test/api/v1/users/me".to_string(),
            status: 403,
            body: ": <html>Forbidden</html>".to_string(),
        };
        let Answer::Refused(error) = classify(Err(proxy)) else {
            anyhow::bail!("a 403 from a proxy should end the wait");
        };
        let message = format!("{error:#}");
        assert!(message.contains("Nothing was written"), "{message}");
        assert!(message.contains("403"), "{message}");
        Ok(())
    }

    #[rstest]
    #[case::generate(generating())]
    #[case::ssh_key(KeyArgs { ssh_key: Some("SHA256:abc".to_string()), generate: false })]
    fn login_refuses_a_new_key_over_an_existing_identity(
        #[case] args: KeyArgs,
    ) -> anyhow::Result<()> {
        let server = MockServer::start();
        let (tmp, _) = adopted(&server)?;
        let stored = RegisteredIdentity::read(tmp.path())?;
        let error = LoginKey::resolve(tmp.path(), &stored, &args)
            .err()
            .ok_or_else(|| anyhow::anyhow!("{args:?} over an identity should be refused"))?;
        assert!(
            error.to_string().contains("already holds an identity"),
            "{error}"
        );
        Ok(())
    }
}
