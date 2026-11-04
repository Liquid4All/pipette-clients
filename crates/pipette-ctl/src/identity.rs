//! The user identity this working directory files campaigns as, read from
//! `<work_dir>/.pipette-ctl/identity/`.
//!
//! The management verbs sign as a user, not a client: a client cannot be
//! granted `submit_plans` (pipette-mgmt's `docs/authentication.md`
//! §5.11). So the identity is not borrowed from the `pipette` client's
//! `.pipette/identity/`, which holds a device's registration.
//!
//! The layout mirrors `pipette-cli`'s identity store file for file, so an
//! operator who has seen one recognizes the other.

use std::{
    fs,
    io::ErrorKind,
    path::{Path, PathBuf},
};

use anyhow::Context;
use serde::{Deserialize, Serialize};

use pipette_mgmt_client::{PrivateKey, ServerUrl, UserIdentity, UserSigner};
use pipette_plan_types::{UserId, UserKeyId};

use crate::workspace::ctl_dir;

const IDENTITY_DIR: &str = "identity";
const USER_NAME: &str = "user.json";
const PRIVATE_KEY_NAME: &str = "id_ed25519";

/// The management server a submission goes to when the operator names none and
/// the working directory holds no identity to name one.
pub const DEFAULT_SERVER_URL: &str = "https://collector.pipette.liquid.ai";

/// `user.json`. Unknown fields are dropped, so a file carrying server-owned
/// extras still reads.
#[derive(Debug, Serialize, Deserialize)]
struct UserRecord {
    user_id: String,
    user_key_id: String,
    server_url: String,
    /// Present only when the account signs with a key this machine already
    /// had; a generated identity writes only the three fields above.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    ssh_key: Option<SshKeyRecord>,
}

/// Which key on this machine signs for the account, by its public half rather
/// than a path or socket: a key moves between agent and disk, and an identity
/// that pinned one would break for a reason nobody changed.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SshKeyRecord {
    pub public_key: String,
    /// Where it was when it was registered, for the message when it is gone.
    pub found_at: String,
}

/// The key a working directory signs with.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StoredKey {
    /// The hex private key `--generate` writes beside the identity.
    Local(PathBuf),
    /// A key this machine already had, named by its public half.
    System(SshKeyRecord),
}

/// The key the written identity signs with, as [`write()`] takes it.
pub enum KeyToStore<'a> {
    /// Hex to write to `id_ed25519` beside the identity.
    Generated(&'a str),
    /// A key already on this machine, recorded rather than copied.
    System(SshKeyRecord),
    /// The `id_ed25519` already beside the identity, left as it is.
    InPlace,
}

/// The public key this working directory registers as: derived from a
/// generated key, stored for one the machine already had.
pub fn public_key(work_dir: &Path) -> anyhow::Result<pipette_mgmt_client::PublicKey> {
    let key = RegisteredIdentity::read(work_dir)?.key.with_context(|| {
        format!(
            "{} holds no key. Run `pipette-ctl auth register` or `auth adopt` first",
            identity_dir(work_dir).display()
        )
    })?;
    public_key_of(&key)
}

pub fn public_key_of(key: &StoredKey) -> anyhow::Result<pipette_mgmt_client::PublicKey> {
    match key {
        StoredKey::Local(path) => Ok(pipette_mgmt_client::public_key_hex_from_private(
            &read_key(path)?,
        )?),
        StoredKey::System(record) => Ok(pipette_mgmt_client::PublicKey::try_new(
            record.public_key.clone(),
        )
        .context("the stored identity does not name a usable public key")?),
    }
}

/// The signer a stored key resolves to, at the moment it is needed.
pub fn signer(key: &StoredKey) -> anyhow::Result<UserSigner> {
    match key {
        StoredKey::Local(path) => Ok(UserSigner::PrivateKey(read_key(path)?)),
        StoredKey::System(record) => crate::system_keys::signer_for(&record.public_key),
    }
}

/// What `<work_dir>/.pipette-ctl/identity/` offers a signed request.
///
/// Every field is `None` on a working directory that holds no identity, which
/// is a legitimate state: the public catalog reads need none.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct RegisteredIdentity {
    user_id: Option<String>,
    user_key_id: Option<String>,
    server_url: Option<String>,
    key: Option<StoredKey>,
}

/// Where `<work_dir>/.pipette-ctl/identity/` lives.
///
/// Public because `auth` writes into it and the management verbs read out of
/// it, and both name the same directory.
pub fn identity_dir(work_dir: &Path) -> PathBuf {
    ctl_dir(work_dir).join(IDENTITY_DIR)
}

/// The key the stored identity signs with. Named here so nothing else has to
/// spell `id_ed25519`.
pub fn key_file(work_dir: &Path) -> PathBuf {
    identity_dir(work_dir).join(PRIVATE_KEY_NAME)
}

/// Refuse to write over an identity this directory already holds.
///
/// Registering again mints a key the account does not know, which would strand
/// the key that *is* registered with no local copy, and revoking that one must
/// be signed by a key the account holds.
pub fn ensure_empty(work_dir: &Path) -> anyhow::Result<()> {
    let dir = identity_dir(work_dir);
    if let Some(path) = [USER_NAME, PRIVATE_KEY_NAME]
        .into_iter()
        .map(|name| dir.join(name))
        .find(|path| path.exists())
    {
        anyhow::bail!(
            "{} already holds an identity ({}). \
             Remove the directory to register a different account",
            dir.display(),
            path.display()
        );
    }
    Ok(())
}

/// Persist a registration and the key that signs for it.
///
/// The key is written first, and `0600`. It is the one value nothing can
/// rebuild: a crash between the two writes leaves a key with no `user.json`,
/// which reads as a recoverable half, where the reverse order loses the key
/// while the account still carries its registration.
pub fn write(
    work_dir: &Path,
    user_id: &UserId,
    user_key_id: &UserKeyId,
    server_url: &ServerUrl,
    key: &KeyToStore<'_>,
) -> anyhow::Result<()> {
    let dir = identity_dir(work_dir);
    create_identity_dir(&dir)?;
    let ssh_key = match key {
        KeyToStore::Generated(private_key_hex) => {
            write_key_file(&key_file(work_dir), private_key_hex)?;
            None
        }
        // No private key is written: the point of this path is that none
        // lands on disk.
        KeyToStore::System(record) => Some(record.clone()),
        KeyToStore::InPlace => None,
    };
    let record = UserRecord {
        user_id: user_id.to_string(),
        user_key_id: user_key_id.to_string(),
        server_url: server_url.to_string(),
        ssh_key,
    };
    let path = dir.join(USER_NAME);
    let json = serde_json::to_vec_pretty(&record).context("encoding user.json")?;
    // Write-then-rename: a renewal rewrites a live record, and a torn one
    // could neither be read by `auth login` nor replaced by `auth register`.
    let tmp = path.with_extension("json.tmp");
    fs::write(&tmp, json).with_context(|| format!("writing {}", tmp.display()))?;
    fs::rename(&tmp, &path).with_context(|| format!("replacing {}", path.display()))?;
    // Makes the key file's and user.json's directory entries durable as well.
    #[cfg(unix)]
    fs::File::open(&dir)
        .and_then(|handle| sync(&handle))
        .with_context(|| format!("syncing {}", dir.display()))?;
    Ok(())
}

/// Create the identity directory, narrowed to `0700`.
///
/// The key inside is `0600`, but a `0755` directory still exposes `user.json`
/// (the account, its key id and its server). `pipette-cli` closes its identity
/// root the same way (`identity/files.rs`).
fn create_identity_dir(dir: &Path) -> anyhow::Result<()> {
    fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(dir, fs::Permissions::from_mode(0o700))
            .with_context(|| format!("narrowing {} to 0700", dir.display()))?;
    }
    Ok(())
}

/// Write a secret so it is never world-readable, even briefly: created with
/// the mode rather than chmod'd after, and `create_new` so it cannot land on
/// top of a key already there.
fn write_key_file(path: &Path, hex: &str) -> anyhow::Result<()> {
    let mut options = fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options
        .open(path)
        .with_context(|| format!("creating {}", path.display()))?;
    // The server already holds the public half, so a key lost to a crash strands it.
    std::io::Write::write_all(&mut file, hex.as_bytes())
        .and_then(|()| sync(&file))
        .with_context(|| format!("writing {}", path.display()))
}

/// A file system that cannot sync counts as synced, as in pipette-artifacts' `sync_file`.
fn sync(file: &fs::File) -> std::io::Result<()> {
    match file.sync_all() {
        Err(e) if matches!(e.kind(), ErrorKind::Unsupported | ErrorKind::InvalidInput) => Ok(()),
        synced => synced,
    }
}

/// Read a stored private key, accepting the newline the identity store writes.
///
/// Typed on the way in, so a file that is not a key is reported against its
/// path rather than as a 401 that reads as the server not knowing this client.
pub fn read_key(path: &Path) -> anyhow::Result<PrivateKey> {
    let stored = fs::read_to_string(path)
        .with_context(|| format!("to read private key {}", path.display()))?;
    PrivateKey::try_new(stored).with_context(|| format!("in private key {}", path.display()))
}

/// The stored identity, as a credential ready to sign with.
///
/// What every `auth` verb that reaches the server needs, from one read: the
/// server to address, and the account key to sign as. The management verbs
/// let `--server` stand in for the stored server, which is why they go through
/// [`RegisteredIdentity::resolve`] instead.
pub fn registered(work_dir: &Path) -> anyhow::Result<(ServerUrl, UserIdentity)> {
    // `resolve` already names `auth login`; add the other two ways to get an account.
    let stored = RegisteredIdentity::read(work_dir)?
        .resolve(None)
        .with_context(|| {
            format!(
                "{} holds no user identity. Run `pipette-ctl auth register` with a \
                 pre-auth key, or `pipette-ctl auth adopt` and have a machine already \
                 on the account add this key",
                identity_dir(work_dir).display()
            )
        })?;
    Ok((
        stored.server,
        UserIdentity {
            user_id: stored.user_id,
            user_key_id: stored.user_key_id,
            signer: signer(&stored.key)?,
        },
    ))
}

/// The four values a submission signs and sends with, each one resolved.
#[derive(Debug, PartialEq, Eq)]
pub struct SubmitIdentity {
    pub server: ServerUrl,
    pub user_id: UserId,
    pub user_key_id: UserKeyId,
    pub key: StoredKey,
}

impl RegisteredIdentity {
    /// Read the stored identity under `work_dir`.
    ///
    /// An absent `user.json` means a working directory that signs as nobody
    /// yet, and yields empty fields. A *present* one that will not parse is an
    /// error. Counting it as absent would fall back to [`DEFAULT_SERVER_URL`]
    /// and send requests to a server the operator never named.
    pub fn read(work_dir: &Path) -> anyhow::Result<Self> {
        let dir = identity_dir(work_dir);
        let record = read_user_record(&dir.join(USER_NAME))?;
        let key_file = dir.join(PRIVATE_KEY_NAME);
        let (user_id, user_key_id, server_url, ssh_key) =
            record.map_or((None, None, None, None), |rec| {
                (
                    Some(rec.user_id),
                    Some(rec.user_key_id),
                    Some(rec.server_url),
                    rec.ssh_key,
                )
            });
        // The record wins over the file: a directory can hold both (a generated
        // key later replaced by one of the machine's own), and the account is
        // registered against whichever `user.json` names.
        let key = match ssh_key {
            Some(record) => Some(StoredKey::System(record)),
            None => key_file.is_file().then_some(StoredKey::Local(key_file)),
        };
        Ok(Self {
            user_id,
            user_key_id,
            server_url,
            key,
        })
    }

    /// The stored identity as a credential to sign with.
    ///
    /// `server` wins over the stored server, and that over
    /// [`DEFAULT_SERVER_URL`]. The account, the key id and the key have no
    /// built-in stand-in, so a working directory missing any of them gets an
    /// error; so does one whose stored server is unusable with no `server` passed.
    pub fn resolve(self, server: Option<ServerUrl>) -> anyhow::Result<SubmitIdentity> {
        // Checked before the server, so a working directory missing the
        // credential hears about that first: it is the harder to supply.
        let user_id = self.user_id.map(UserId::try_new).transpose()?;
        let user_key_id = self.user_key_id.map(UserKeyId::try_new).transpose()?;
        let (Some(user_id), Some(user_key_id), Some(key)) = (user_id, user_key_id, self.key) else {
            anyhow::bail!(
                "no identity to sign the submission with: run `pipette-ctl auth login` to \
                 write one to .pipette-ctl/identity/"
            );
        };
        Ok(SubmitIdentity {
            server: server_or_stored(server, self.server_url.as_deref())?,
            user_id,
            user_key_id,
            key,
        })
    }

    /// No `user.json`: a working directory that signs as nobody yet.
    pub fn is_empty(&self) -> bool {
        self.user_id.is_none() && self.user_key_id.is_none() && self.server_url.is_none()
    }

    /// A key file or an ssh key record to sign with.
    pub fn has_key(&self) -> bool {
        self.key.is_some()
    }

    /// The key this directory signs with, whether or not an account is named.
    pub fn key(&self) -> Option<&StoredKey> {
        self.key.as_ref()
    }

    /// The server a request goes to: `named`, else the stored one, else
    /// [`DEFAULT_SERVER_URL`]. Needs no credential, so a working directory
    /// whose account does not resolve still names its own server.
    pub fn server(&self, named: Option<ServerUrl>) -> anyhow::Result<ServerUrl> {
        server_or_stored(named, self.server_url.as_deref())
    }
}

/// The stored URL is parsed only when it is the one being used, so `--server`
/// still reaches a server from a working directory whose `user.json` names
/// something no request can be built from.
fn server_or_stored(named: Option<ServerUrl>, stored: Option<&str>) -> anyhow::Result<ServerUrl> {
    match (named, stored) {
        (Some(named), _) => Ok(named),
        (None, Some(stored)) => ServerUrl::try_new(stored)
            .context("the stored server URL is unusable; pass --server to name another"),
        (None, None) => Ok(ServerUrl::try_new(DEFAULT_SERVER_URL)?),
    }
}

/// Read `user.json`, treating only an absent file as no identity.
fn read_user_record(path: &Path) -> anyhow::Result<Option<UserRecord>> {
    let raw = match fs::read_to_string(path) {
        Ok(raw) => raw,
        Err(error) if error.kind() == ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error).with_context(|| context(path)),
    };
    serde_json::from_str(&raw)
        .map(Some)
        .with_context(|| context(path))
}

fn context(path: &Path) -> String {
    format!("to read the stored identity from {}", path.display())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn work_dir_with(user: Option<&str>, private_key: bool) -> anyhow::Result<tempfile::TempDir> {
        let work = tempfile::tempdir()?;
        let dir = work.path().join(".pipette-ctl").join("identity");
        fs::create_dir_all(&dir)?;
        if let Some(json) = user {
            fs::write(dir.join(USER_NAME), json)?;
        }
        if private_key {
            fs::write(dir.join(PRIVATE_KEY_NAME), "abc123\n")?;
        }
        Ok(work)
    }

    const USER: &str = r#"{
        "user_id": "user_0192f0c1",
        "user_key_id": "userkey_9f2c",
        "server_url": "https://mgmt.example.com",
        "contact_email": "operator@example.com"
    }"#;

    #[test]
    fn a_prepared_work_dir_supplies_all_four() -> anyhow::Result<()> {
        let work = work_dir_with(Some(USER), true)?;
        let identity = RegisteredIdentity::read(work.path())?.resolve(None)?;
        assert_eq!(
            identity,
            SubmitIdentity {
                server: ServerUrl::try_new("https://mgmt.example.com")?,
                user_id: UserId::try_new("user_0192f0c1")?,
                user_key_id: UserKeyId::try_new("userkey_9f2c")?,
                key: StoredKey::Local(work.path().join(".pipette-ctl/identity/id_ed25519")),
            }
        );
        Ok(())
    }

    #[test]
    fn a_client_registration_is_not_borrowed() -> anyhow::Result<()> {
        let work = tempfile::tempdir()?;
        let dir = work.path().join(".pipette").join("identity");
        fs::create_dir_all(&dir)?;
        fs::write(
            dir.join("registration.json"),
            r#"{ "client_id": "ev1_9f2c", "server_url": "https://mgmt.example.com" }"#,
        )?;
        fs::write(dir.join("id_ed25519"), "abc123\n")?;

        assert_eq!(
            RegisteredIdentity::read(work.path())?,
            RegisteredIdentity::default()
        );
        Ok(())
    }

    #[test]
    fn a_bare_work_dir_falls_back_to_the_default_server() -> anyhow::Result<()> {
        let work = work_dir_with(None, false)?;
        let registered = RegisteredIdentity::read(work.path())?;
        assert_eq!(registered, RegisteredIdentity::default());
        assert_eq!(registered.server(None)?.to_string(), DEFAULT_SERVER_URL);
        Ok(())
    }

    #[test]
    fn a_work_dir_with_no_plan_workspace_reads_as_empty() -> anyhow::Result<()> {
        let work = tempfile::tempdir()?;
        assert_eq!(
            RegisteredIdentity::read(work.path())?,
            RegisteredIdentity::default()
        );
        Ok(())
    }

    #[rstest::rstest]
    #[case(Some(USER), false)]
    #[case(None, true)]
    #[case(None, false)]
    fn part_of_an_identity_does_not_resolve(
        #[case] user: Option<&str>,
        #[case] private_key: bool,
    ) -> anyhow::Result<()> {
        let work = work_dir_with(user, private_key)?;
        let error = RegisteredIdentity::read(work.path())?
            .resolve(None)
            .err()
            .ok_or_else(|| anyhow::anyhow!("part of an identity should not resolve"))?;
        anyhow::ensure!(
            error.to_string().contains("auth login"),
            "unexpected error: {error}"
        );
        Ok(())
    }

    #[rstest::rstest]
    #[case::a_client_id_as_the_user("ev1_9f2c", "userkey_9f2c", "user id")]
    #[case::a_user_id_as_the_key("user_0192f0c1", "user_0192f0c1", "key id")]
    fn an_id_of_the_wrong_kind_is_refused(
        #[case] user_id: &str,
        #[case] user_key_id: &str,
        #[case] expected: &str,
    ) -> anyhow::Result<()> {
        let stored = RegisteredIdentity {
            user_id: Some(user_id.to_string()),
            user_key_id: Some(user_key_id.to_string()),
            server_url: None,
            key: Some(StoredKey::Local(PathBuf::from("/tmp/other.key"))),
        };
        let error = stored
            .resolve(None)
            .err()
            .ok_or_else(|| anyhow::anyhow!("an id of the wrong kind should not resolve"))?;
        anyhow::ensure!(
            error.to_string().contains(expected),
            "unexpected error: {error}"
        );
        Ok(())
    }

    #[test]
    fn a_flag_overrides_a_stored_url_that_is_unusable() -> anyhow::Result<()> {
        let unusable = r#"{ "user_id": "user_0192f0c1", "user_key_id": "userkey_9f2c",
                            "server_url": "mgmt.example.com" }"#;
        let work = work_dir_with(Some(unusable), true)?;
        let registered = RegisteredIdentity::read(work.path())?;

        let named = ServerUrl::try_new("https://staging.example.com")?;
        assert_eq!(registered.resolve(Some(named.clone()))?.server, named);

        let work = work_dir_with(Some(unusable), true)?;
        let error = RegisteredIdentity::read(work.path())?
            .resolve(None)
            .err()
            .ok_or_else(|| anyhow::anyhow!("an unusable stored URL should not resolve"))?;
        anyhow::ensure!(
            format!("{error:#}").contains("--server"),
            "unexpected error: {error:#}"
        );
        Ok(())
    }

    #[rstest::rstest]
    #[case::not_json("{ not json")]
    #[case::no_server(r#"{ "user_id": "user_0192f0c1", "user_key_id": "userkey_9f2c" }"#)]
    #[case::no_key_id(r#"{ "user_id": "user_0192f0c1", "server_url": "https://m.example.com" }"#)]
    fn an_unreadable_identity_is_an_error(#[case] json: &str) -> anyhow::Result<()> {
        // Read as "no identity", it would default the server to production.
        let work = work_dir_with(Some(json), true)?;
        let error = RegisteredIdentity::read(work.path())
            .err()
            .ok_or_else(|| anyhow::anyhow!("an unreadable identity should be an error"))?;
        anyhow::ensure!(
            format!("{error:#}").contains(USER_NAME),
            "unexpected error: {error:#}"
        );
        Ok(())
    }

    #[test]
    fn a_system_key_is_recorded_rather_than_copied() -> anyhow::Result<()> {
        let work = tempfile::tempdir()?;
        let hex = "c8bef8049584357385e07db4dce82bca8e747cb5922bbda85cfbfb14230a55b3";
        let record = SshKeyRecord {
            public_key: hex.to_string(),
            found_at: "agent (1Password)".to_string(),
        };
        write(
            work.path(),
            &UserId::try_new("user_0192f0c1")?,
            &UserKeyId::try_new("userkey_9f2c")?,
            &ServerUrl::try_new("https://mgmt.example.com")?,
            &KeyToStore::System(record.clone()),
        )?;

        assert!(
            !key_file(work.path()).exists(),
            "no private key is written for a key the machine already holds"
        );
        assert_eq!(
            RegisteredIdentity::read(work.path())?.key,
            Some(StoredKey::System(record))
        );
        assert_eq!(public_key(work.path())?.as_ref(), hex);
        Ok(())
    }

    #[test]
    fn a_generated_identity_writes_no_ssh_key_field() -> anyhow::Result<()> {
        let work = tempfile::tempdir()?;
        write(
            work.path(),
            &UserId::try_new("user_0192f0c1")?,
            &UserKeyId::try_new("userkey_9f2c")?,
            &ServerUrl::try_new("https://mgmt.example.com")?,
            &KeyToStore::Generated("abc123"),
        )?;

        let json = fs::read_to_string(identity_dir(work.path()).join(USER_NAME))?;
        assert!(!json.contains("ssh_key"), "got {json}");
        assert_eq!(
            RegisteredIdentity::read(work.path())?.key,
            Some(StoredKey::Local(key_file(work.path())))
        );
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn a_rewrite_replaces_user_json_rather_than_truncating_it() -> anyhow::Result<()> {
        use std::os::unix::fs::MetadataExt;

        let work = tempfile::tempdir()?;
        let rewrite = || {
            write(
                work.path(),
                &UserId::try_new("user_0192f0c1")?,
                &UserKeyId::try_new("userkey_9f2c")?,
                &ServerUrl::try_new("https://mgmt.example.com")?,
                &KeyToStore::InPlace,
            )
        };
        let path = identity_dir(work.path()).join(USER_NAME);
        rewrite()?;
        let before = fs::metadata(&path)?.ino();
        rewrite()?;

        assert_ne!(fs::metadata(&path)?.ino(), before);
        assert!(!path.with_extension("json.tmp").exists());
        Ok(())
    }
}
