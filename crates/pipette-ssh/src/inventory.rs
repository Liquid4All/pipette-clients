//! Every Ed25519 key this host can sign with, from whichever place it lives.
//!
//! A key is often both in `~/.ssh` and loaded into an agent, so keys are merged
//! by public half and carry every place they were found. Usability is per
//! source: an agent can be asked, an unencrypted file can be read, and an
//! encrypted file can do neither on its own, though it is still listed because
//! the agent beside it usually holds the same key.

use std::fs;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};

use crate::error::Error;
use crate::key::Ed25519PublicKey;
use crate::known::{KnownAgent, discover};
use crate::private::PrivateKeyFile;

/// A ceiling on what is worth reading while looking for a key. An OpenSSH
/// private key is well under a kilobyte; `known_hosts` is megabytes.
const MAX_KEY_FILE: u64 = 64 * 1024;

/// One key, and everywhere on this host it was found.
#[derive(Debug, Clone)]
pub struct SystemKey {
    pub key: Ed25519PublicKey,
    /// The best name found for it: an agent often reports none, and the `.pub`
    /// file beside a key is then where the comment lives.
    pub comment: String,
    pub sources: Vec<KeySource>,
}

/// Where a key was found, and what that place can do with it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KeySource {
    /// Held by a running agent, which will sign without the key ever reaching
    /// this process.
    Agent {
        label: &'static str,
        socket: PathBuf,
    },
    /// An OpenSSH private key file. Signable here only when it is unencrypted.
    File { path: PathBuf, encrypted: bool },
}

/// What this host offers, and what it could not read while looking.
#[derive(Debug, Default)]
pub struct Inventory {
    /// Agent keys first, in the order the agents were discovered, then keys
    /// found only on disk.
    pub keys: Vec<SystemKey>,
    pub skipped: Vec<Skipped>,
}

/// A place that could not be read, kept so a listing can say why an expected
/// key is absent: a locked agent and an RSA key are both invisible otherwise.
#[derive(Debug)]
pub struct Skipped {
    pub source: String,
    pub reason: Error,
}

impl SystemKey {
    /// Whether some source can sign: an agent, or an unencrypted file.
    pub fn is_usable(&self) -> bool {
        self.sources.iter().any(KeySource::can_sign)
    }

    pub fn fingerprint(&self) -> String {
        self.key.fingerprint()
    }

    /// Every place it was found, in one column: `agent (1Password), ~/.ssh/id_ed25519`.
    pub fn found_at(&self) -> String {
        self.sources
            .iter()
            .map(KeySource::describe)
            .collect::<Vec<_>>()
            .join(", ")
    }
}

impl KeySource {
    pub fn can_sign(&self) -> bool {
        match self {
            Self::Agent { .. } => true,
            Self::File { encrypted, .. } => !encrypted,
        }
    }

    pub fn describe(&self) -> String {
        match self {
            Self::Agent { label, .. } => format!("agent ({label})"),
            Self::File {
                path,
                encrypted: false,
            } => path.display().to_string(),
            Self::File {
                path,
                encrypted: true,
            } => format!("{} (encrypted)", path.display()),
        }
    }
}

/// Every Ed25519 key the agents on this host hold, plus every one under
/// `~/.ssh`.
pub fn inventory() -> Inventory {
    collect(&discover(), ssh_dir().as_deref())
}

/// `~/.ssh`. Windows sets no `HOME` and its OpenSSH reads `%USERPROFILE%\.ssh`;
/// `HOME` is still consulted second for a git-bash or MSYS shell.
fn ssh_dir() -> Option<PathBuf> {
    home_variables()
        .iter()
        .find_map(|name| std::env::var_os(name).filter(|home| !home.is_empty()))
        .map(|home| PathBuf::from(home).join(".ssh"))
}

#[cfg(windows)]
fn home_variables() -> &'static [&'static str] {
    &["USERPROFILE", "HOME"]
}

#[cfg(unix)]
fn home_variables() -> &'static [&'static str] {
    &["HOME"]
}

/// The seam the tests drive: agents and a directory, rather than a host.
fn collect(agents: &[KnownAgent], ssh_dir: Option<&Path>) -> Inventory {
    let mut inventory = Inventory::default();
    agents
        .iter()
        .for_each(|known| match known.agent.ed25519_identities() {
            Ok(identities) => identities.into_iter().for_each(|identity| {
                inventory.add(
                    identity.key,
                    identity.comment,
                    KeySource::Agent {
                        label: known.label,
                        socket: known.agent.socket().to_path_buf(),
                    },
                );
            }),
            Err(reason) => inventory.skipped.push(Skipped {
                source: known.agent.socket().display().to_string(),
                reason,
            }),
        });
    if let Some(dir) = ssh_dir {
        inventory.read_ssh_dir(dir);
    }
    inventory
}

impl Inventory {
    /// Every key `selector` names: by fingerprint, hex, comment or file path.
    /// Case-insensitive on the comment, because a person typing a machine name
    /// should not have to match its capitalization.
    pub fn matching(&self, selector: &str) -> Vec<&SystemKey> {
        self.keys
            .iter()
            .filter(|key| key.matches(selector))
            .collect()
    }

    fn add(&mut self, key: Ed25519PublicKey, comment: String, source: KeySource) {
        match self.keys.iter_mut().find(|held| held.key == key) {
            Some(held) => {
                if held.comment.is_empty() {
                    held.comment = comment;
                }
                if !held.sources.contains(&source) {
                    held.sources.push(source);
                }
            }
            None => self.keys.push(SystemKey {
                key,
                comment,
                sources: vec![source],
            }),
        }
    }

    /// Private keys first, then the `.pub` files that name them: a `.pub` alone
    /// signs nothing, so it only ever contributes a comment. No `~/.ssh` at all
    /// is a fresh machine, not a fault.
    fn read_ssh_dir(&mut self, dir: &Path) {
        let Ok(entries) = fs::read_dir(dir) else {
            return;
        };
        let paths: Vec<PathBuf> = entries
            .flatten()
            .map(|entry| entry.path())
            .filter(|path| path.is_file() && fits(path))
            .collect();

        paths
            .iter()
            .filter(|path| !is_public(path))
            .for_each(|path| self.read_private_key(path));
        paths
            .iter()
            .filter(|path| is_public(path))
            .for_each(|path| self.read_comment(path));
    }

    /// `config` and `known_hosts` are passed over in silence; a file that calls
    /// itself a private key and cannot be used is reported, since "your id_rsa
    /// is not Ed25519" and "permission denied" are the answers to why a key is
    /// not on the list. Only a file that is not text is junk rather than a key.
    fn read_private_key(&mut self, path: &Path) {
        let text = match fs::read_to_string(path) {
            Ok(text) => text,
            Err(reason) if reason.kind() == ErrorKind::InvalidData => return,
            Err(reason) => {
                self.skipped.push(Skipped {
                    source: path.display().to_string(),
                    reason: Error::Unreadable(reason),
                });
                return;
            }
        };
        if !text.contains("PRIVATE KEY") {
            return;
        }
        match PrivateKeyFile::parse(&text) {
            Ok(file) => {
                let source = KeySource::File {
                    path: path.to_path_buf(),
                    encrypted: file.is_encrypted(),
                };
                self.add(file.public, file.comment, source);
            }
            Err(reason) => self.skipped.push(Skipped {
                source: path.display().to_string(),
                reason,
            }),
        }
    }

    /// Name a key already found from the `.pub` beside it, which is the only
    /// place an encrypted file's comment survives.
    fn read_comment(&mut self, path: &Path) {
        let Ok(text) = fs::read_to_string(path) else {
            return;
        };
        let Ok((key, comment)) = Ed25519PublicKey::from_openssh(text.trim()) else {
            return;
        };
        if let Some(held) = self.keys.iter_mut().find(|held| held.key == key)
            && held.comment.is_empty()
        {
            held.comment = comment;
        }
    }
}

impl SystemKey {
    fn matches(&self, selector: &str) -> bool {
        if self.key.to_hex() == selector.to_lowercase() {
            return true;
        }
        if self.fingerprint() == selector {
            return true;
        }
        if !self.comment.is_empty() && self.comment.eq_ignore_ascii_case(selector) {
            return true;
        }
        self.sources.iter().any(|source| match source {
            KeySource::File { path, .. } => path == Path::new(selector),
            KeySource::Agent { .. } => false,
        })
    }
}

/// A `.pub` file names a key; everything else in `~/.ssh` might hold one.
fn is_public(path: &Path) -> bool {
    path.extension().is_some_and(|extension| extension == "pub")
}

/// Skip what is too big to be a key before reading it, so listing keys never
/// pulls a large `known_hosts` into memory.
fn fits(path: &Path) -> bool {
    fs::metadata(path).is_ok_and(|meta| meta.len() <= MAX_KEY_FILE)
}

#[cfg(test)]
mod tests {
    use anyhow::Context;
    use rstest::rstest;

    use super::*;
    use crate::agent::Agent;
    use crate::fixtures::{
        LOCKED, LOCKED_FINGERPRINT, LOCKED_PUB, PLAIN, PLAIN_FINGERPRINT, PLAIN_PUB,
    };

    fn ssh_dir(files: &[(&str, &str)]) -> anyhow::Result<tempfile::TempDir> {
        let dir = tempfile::TempDir::new()?;
        files
            .iter()
            .try_for_each(|(name, contents)| std::fs::write(dir.path().join(name), contents))?;
        Ok(dir)
    }

    /// A passphrase key with no agent running is still listed and named from
    /// its `.pub`, so a person is sent to `ssh-add` rather than to a second key.
    #[test]
    fn an_encrypted_file_is_listed_but_cannot_sign() -> anyhow::Result<()> {
        let dir = ssh_dir(&[("id_ed25519", LOCKED), ("id_ed25519.pub", LOCKED_PUB)])?;

        let found = collect(&[], Some(dir.path()));

        let only = found.keys.first().context("the encrypted key is listed")?;
        assert_eq!(only.fingerprint(), LOCKED_FINGERPRINT);
        assert!(!only.is_usable(), "a passphrase stands in the way");
        assert_eq!(
            only.comment, "locked@fixture",
            "the comment is inside the encryption, so the .pub file carries it"
        );
        Ok(())
    }

    #[test]
    fn names_what_it_could_not_read_and_ignores_what_is_not_a_key() -> anyhow::Result<()> {
        let dir = ssh_dir(&[
            ("config", "Host *\n  AddKeysToAgent yes\n"),
            ("known_hosts", "github.com ssh-ed25519 AAAA\n"),
            (
                "id_rsa",
                "-----BEGIN RSA PRIVATE KEY-----\nMIIE\n-----END RSA PRIVATE KEY-----\n",
            ),
        ])?;

        let found = collect(&[], Some(dir.path()));

        assert!(found.keys.is_empty(), "got {:?}", found.keys);
        let skipped = found.skipped.first().context("the RSA key is reported")?;
        assert_eq!(found.skipped.len(), 1, "got {:?}", found.skipped);
        assert!(skipped.source.ends_with("id_rsa"), "got {}", skipped.source);
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn a_key_file_that_will_not_open_is_reported() -> anyhow::Result<()> {
        use std::os::unix::fs::PermissionsExt;

        let dir = ssh_dir(&[("id_ed25519", PLAIN)])?;
        let path = dir.path().join("id_ed25519");
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o000))?;
        if std::fs::read_to_string(&path).is_ok() {
            // Running as root, where no file is unopenable. Nothing to assert.
            return Ok(());
        }

        let found = collect(&[], Some(dir.path()));

        let skipped = found
            .skipped
            .first()
            .context("the unreadable key is named")?;
        assert!(found.keys.is_empty(), "got {:?}", found.keys);
        assert!(skipped.source.ends_with("id_ed25519"), "got {skipped:?}");
        Ok(())
    }

    #[test]
    fn an_unreachable_agent_is_reported_rather_than_dropped() -> anyhow::Result<()> {
        let dir = tempfile::TempDir::new()?;
        let known = vec![KnownAgent {
            label: "stale",
            agent: Agent::at(dir.path().join("gone.sock")),
        }];

        let found = collect(&known, None);

        let skipped = found.skipped.first().context("the dead socket is named")?;
        assert!(found.keys.is_empty(), "got {:?}", found.keys);
        assert!(
            matches!(skipped.reason, Error::Connect { .. }),
            "got {skipped:?}"
        );
        Ok(())
    }

    #[rstest]
    #[case::fingerprint(|_: &Path, _: &Ed25519PublicKey| PLAIN_FINGERPRINT.to_string(), 1)]
    #[case::hex(|_: &Path, key: &Ed25519PublicKey| key.to_hex(), 1)]
    #[case::comment_in_any_case(|_: &Path, _: &Ed25519PublicKey| "Plain@Fixture".to_string(), 1)]
    #[case::path(|dir: &Path, _: &Ed25519PublicKey| dir.join("id_ed25519").display().to_string(), 1)]
    #[case::someone_elses(|_: &Path, _: &Ed25519PublicKey| "someone-elses-key".to_string(), 0)]
    fn a_key_is_selectable_by_every_name_it_has(
        #[case] selector: fn(&Path, &Ed25519PublicKey) -> String,
        #[case] matches: usize,
    ) -> anyhow::Result<()> {
        let dir = ssh_dir(&[("id_ed25519", PLAIN), ("id_ed25519.pub", PLAIN_PUB)])?;
        let found = collect(&[], Some(dir.path()));
        let key = found.keys.first().context("the fixture key is found")?.key;

        let selector = selector(dir.path(), &key);
        assert_eq!(
            found.matching(&selector).len(),
            matches,
            "{selector} names {matches} key(s)"
        );
        Ok(())
    }

    /// The fake agent is a unix socket, so these cannot run on Windows.
    #[cfg(unix)]
    mod with_an_agent {
        use super::*;
        use crate::fake_agent::{FakeAgent, FakeKey};
        use crate::private::PrivateKeyFile;

        /// So an agent can hold the same key the file does.
        fn plain_seed() -> anyhow::Result<[u8; 32]> {
            let hex = PrivateKeyFile::parse(PLAIN)?
                .seed_hex()
                .context("the fixture is unencrypted")?;
            hex::decode(hex)?
                .try_into()
                .ok()
                .context("the seed is 32 bytes")
        }

        fn agents(agent: &FakeAgent) -> Vec<KnownAgent> {
            vec![KnownAgent {
                label: "test-agent",
                agent: Agent::at(agent.socket()),
            }]
        }

        #[test]
        fn merges_a_key_the_agent_and_the_disk_both_hold() -> anyhow::Result<()> {
            let agent = FakeAgent::start(vec![FakeKey::signing(plain_seed()?, "")?])?;
            let dir = ssh_dir(&[("id_ed25519", PLAIN), ("id_ed25519.pub", PLAIN_PUB)])?;

            let found = collect(&agents(&agent), Some(dir.path()));

            let only = found.keys.first().context("the fixture key is found")?;
            assert_eq!(found.keys.len(), 1, "got {:?}", found.keys);
            assert_eq!(only.fingerprint(), PLAIN_FINGERPRINT);
            assert!(only.is_usable());
            assert_eq!(
                only.sources.len(),
                2,
                "the agent and the file are both sources: {:?}",
                only.sources
            );
            assert_eq!(
                only.comment, "plain@fixture",
                "the agent reports no comment, so the file names the key"
            );
            Ok(())
        }
    }
}
