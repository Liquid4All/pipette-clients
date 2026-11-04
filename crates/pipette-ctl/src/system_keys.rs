//! The keys this machine already has, offered to `auth` instead of a new one.
//!
//! `auth` picks from what [`pipette_ssh::inventory()`] finds and stores which
//! key it chose, not the key: a key in an agent is never read, and a key in an
//! unencrypted file is read from its own path when the signer is resolved.

use std::sync::Arc;

use anyhow::Context;

use pipette_mgmt_client::{PublicKey, SIGNATURE_LEN, SignsPayload, UserSigner};
use pipette_ssh::inventory::{Inventory, KeySource, SystemKey, inventory};
use pipette_ssh::{Agent, Ed25519PublicKey, PrivateKeyFile};

/// Pick the key `--ssh-key` names, or the only usable one this host has.
///
/// Several usable keys is an error listing them: guessing which one the account
/// is registered against is not a decision this command can take back.
pub fn choose(selector: Option<&str>) -> anyhow::Result<SystemKey> {
    choose_from(inventory(), selector)
}

/// Over an inventory rather than a host, so tests never read the machine's keys.
fn choose_from(found: Inventory, selector: Option<&str>) -> anyhow::Result<SystemKey> {
    let usable: Vec<&SystemKey> = found.keys.iter().filter(|key| key.is_usable()).collect();

    let Some(selector) = selector else {
        return match usable.as_slice() {
            [only] => Ok((*only).clone()),
            [] => Err(nothing_usable(&found)),
            several => Err(anyhow::anyhow!(
                "this machine offers {} keys it can sign with, so --ssh-key has to say \
                 which one the account is registered against:\n{}",
                several.len(),
                describe(several.iter().copied())
            )),
        };
    };

    match found.matching(selector).as_slice() {
        [only] if only.is_usable() => Ok((*only).clone()),
        [only] => Err(anyhow::anyhow!(
            "{selector} names a key this machine cannot sign with:\n{}\n\n{}",
            describe([*only]),
            "Add it to an agent (`ssh-add <path>`) and run this again."
        )),
        [] => Err(anyhow::anyhow!(
            "no key in an agent or in ~/.ssh matches {selector}. A key kept elsewhere signs \
             once `ssh-add` holds it; `pipette-ctl auth keys available` lists what this \
             machine offers"
        )),
        several => Err(anyhow::anyhow!(
            "{selector} names {} keys; use a fingerprint to pick one:\n{}",
            several.len(),
            describe(several.iter().copied())
        )),
    }
}

/// An encrypted `~/.ssh/id_ed25519` with no agent running is the common shape,
/// and its fix is `ssh-add`, not a new key; only a machine with nothing at all
/// is told to generate one.
fn nothing_usable(found: &Inventory) -> anyhow::Error {
    if found.keys.is_empty() {
        return anyhow::anyhow!(
            "this machine has no Ed25519 key to register: nothing in ~/.ssh and no agent \
             holding one. Run `ssh-keygen -t ed25519` and `ssh-add` it, or pass --generate \
             to mint a key that lives with this working directory"
        );
    }
    anyhow::anyhow!(
        "this machine has {} Ed25519 key(s), and can sign with none of them:\n{}\n\n\
         Load one into an agent (`ssh-add <path>`), or pass --generate to mint a key that \
         lives with this working directory",
        found.keys.len(),
        describe(found.keys.iter())
    )
}

/// One key per line: what it is called, what names it, and where it was found.
fn describe<'a>(keys: impl IntoIterator<Item = &'a SystemKey>) -> String {
    keys.into_iter()
        .map(|key| {
            format!(
                "  {}  {:<20} {}",
                key.fingerprint(),
                if key.comment.is_empty() {
                    "(no comment)"
                } else {
                    &key.comment
                },
                key.found_at()
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// `auth keys available`. Unusable and unreadable entries are printed too:
/// whoever runs this is asking why the key they have in mind is missing.
pub fn print_available() {
    let found = inventory();
    if found.keys.is_empty() {
        println!("no Ed25519 keys found in ~/.ssh or in any agent on this machine");
    } else {
        println!("{}", describe(found.keys.iter()));
        let unusable = found.keys.iter().filter(|key| !key.is_usable()).count();
        if unusable > 0 {
            println!();
            println!(
                "{unusable} of these cannot sign here. `ssh-add` the file, or unlock the \
                 agent that holds it."
            );
        }
    }
    found
        .skipped
        .iter()
        .for_each(|skipped| println!("skipped {}: {}", skipped.source, skipped.reason));
}

/// The public key a chosen key registers as.
pub fn public_key_hex(key: &SystemKey) -> anyhow::Result<PublicKey> {
    PublicKey::try_new(key.key.to_hex()).context("the SSH key is not a usable public key")
}

/// A signer for the key this account registered, found again on this host.
///
/// Re-found at signing time because a key moves between agent and disk; an
/// identity that pinned a socket or path would break for no reason of its own.
pub fn signer_for(public_key_hex: &str) -> anyhow::Result<UserSigner> {
    signer_from(inventory(), public_key_hex)
}

/// Over an inventory rather than a host, so tests never read the machine's keys.
pub(crate) fn signer_from(found: Inventory, public_key_hex: &str) -> anyhow::Result<UserSigner> {
    let wanted = Ed25519PublicKey::from_hex(public_key_hex)
        .context("the stored identity does not name a usable SSH key")?;
    let key = found
        .keys
        .iter()
        .find(|key| key.key == wanted)
        .with_context(|| missing(&wanted, &found))?;
    signer_from_sources(key)
}

/// The registered key is gone: name it, and whatever could not be read.
/// Usually an agent that restarted empty, so the fix is to load it again.
fn missing(wanted: &Ed25519PublicKey, found: &Inventory) -> String {
    std::iter::once(format!(
        "this working directory signs as {}, and no agent or key file here offers that \
         key. Unlock the agent that had it, or `ssh-add` the key again",
        wanted.fingerprint()
    ))
    .chain(found.skipped.iter().map(|skipped| {
        format!(
            "\n  {} could not be read: {}",
            skipped.source, skipped.reason
        )
    }))
    .collect()
}

/// A readable file wins over an agent: no approval prompt per request, and a
/// submission keeps working when no agent is running.
fn signer_from_sources(key: &SystemKey) -> anyhow::Result<UserSigner> {
    let readable_file = key.sources.iter().find_map(|source| match source {
        KeySource::File {
            path,
            encrypted: false,
        } => Some(path),
        KeySource::File {
            encrypted: true, ..
        }
        | KeySource::Agent { .. } => None,
    });
    if let Some(path) = readable_file {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("to read the SSH key {}", path.display()))?;
        let seed = PrivateKeyFile::parse(&text)
            .with_context(|| format!("in the SSH key {}", path.display()))?
            .seed_hex()
            .with_context(|| format!("{} is encrypted", path.display()))?;
        return Ok(UserSigner::PrivateKey(
            pipette_mgmt_client::PrivateKey::try_new(seed)
                .with_context(|| format!("in the SSH key {}", path.display()))?,
        ));
    }

    let socket = key
        .sources
        .iter()
        .find_map(|source| match source {
            KeySource::Agent { socket, .. } => Some(socket.clone()),
            KeySource::File { .. } => None,
        })
        .with_context(|| {
            format!(
                "no agent holds {} and its file is encrypted; `ssh-add` it",
                key.fingerprint()
            )
        })?;
    Ok(UserSigner::External(Arc::new(AgentKey {
        agent: Agent::at(socket),
        key: key.key,
    })))
}

/// Signs through the agent that holds the key, one request at a time.
#[derive(Debug)]
struct AgentKey {
    agent: Agent,
    key: Ed25519PublicKey,
}

impl SignsPayload for AgentKey {
    fn sign_payload(
        &self,
        payload: &[u8],
    ) -> std::result::Result<[u8; SIGNATURE_LEN], Box<dyn std::error::Error + Send + Sync>> {
        Ok(self.agent.sign(&self.key, payload)?)
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use rstest::rstest;

    use pipette_ssh::inventory::Skipped;

    use super::*;

    fn key(seed: u8) -> Ed25519PublicKey {
        Ed25519PublicKey::from_bytes([seed; 32])
    }

    fn agent_key(seed: u8, comment: &str) -> SystemKey {
        SystemKey {
            key: key(seed),
            comment: comment.to_string(),
            sources: vec![KeySource::Agent {
                label: "1Password",
                socket: PathBuf::from("/tmp/agent.sock"),
            }],
        }
    }

    fn locked_file(seed: u8, comment: &str) -> SystemKey {
        SystemKey {
            key: key(seed),
            comment: comment.to_string(),
            sources: vec![KeySource::File {
                path: PathBuf::from("/home/someone/.ssh/id_ed25519"),
                encrypted: true,
            }],
        }
    }

    fn inventory_of(keys: Vec<SystemKey>) -> Inventory {
        Inventory {
            keys,
            skipped: Vec::new(),
        }
    }

    #[test]
    fn one_usable_key_needs_no_flag() -> anyhow::Result<()> {
        let found = inventory_of(vec![agent_key(1, "laptop"), locked_file(2, "old")]);

        let chosen = choose_from(found, None)?;

        assert_eq!(chosen.key, key(1), "the unusable one is not a candidate");
        Ok(())
    }

    /// An agent that could not be read, the locked-vault case.
    fn locked_agent() -> Skipped {
        Skipped {
            source: "/tmp/agent.sock".to_string(),
            reason: pipette_ssh::Error::Io(std::io::ErrorKind::ConnectionRefused.into()),
        }
    }

    #[rstest]
    #[case::several_usable(
        vec![agent_key(1, "laptop"), agent_key(2, "desktop")], Vec::new(), None,
        &["--ssh-key", "laptop", "desktop"]
    )]
    #[case::none_usable(vec![locked_file(1, "laptop")], Vec::new(), None, &["ssh-add", "--generate"])]
    #[case::no_keys_at_all(Vec::new(), Vec::new(), None, &["ssh-keygen", "--generate"])]
    #[case::no_keys_and_a_locked_agent(Vec::new(), vec![locked_agent()], None, &["no Ed25519 key"])]
    #[case::named_unusable(
        vec![agent_key(1, "laptop"), locked_file(2, "archive")], Vec::new(), Some("archive"),
        &["cannot sign with", "ssh-add"]
    )]
    #[case::named_nothing(vec![agent_key(1, "laptop")], Vec::new(), Some("desktop"), &["auth keys available"])]
    fn a_choice_that_cannot_be_made_says_what_would_make_it(
        #[case] keys: Vec<SystemKey>,
        #[case] skipped: Vec<Skipped>,
        #[case] selector: Option<&str>,
        #[case] needles: &[&str],
    ) -> anyhow::Result<()> {
        let found = Inventory { keys, skipped };

        let message = choose_from(found, selector)
            .err()
            .context("no key can be chosen")?
            .to_string();

        needles.iter().for_each(|needle| {
            assert!(message.contains(needle), "missing {needle:?} in {message}")
        });
        Ok(())
    }

    #[test]
    fn a_readable_file_signs_without_an_agent() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("id_ed25519");
        let pair = crate::ssh_keygen::generate("fixture")?;
        std::fs::write(&path, &pair.private_openssh)?;
        let (public, _) = Ed25519PublicKey::from_openssh(&pair.public_openssh)?;
        let found = inventory_of(vec![SystemKey {
            key: public,
            comment: "fixture".to_string(),
            sources: vec![KeySource::File {
                path,
                encrypted: false,
            }],
        }]);

        let signer = signer_from(found, &public.to_hex())?;

        let UserSigner::PrivateKey(private_key_hex) = signer else {
            anyhow::bail!("a readable file signs directly, with no agent round trip");
        };
        assert_eq!(
            pipette_mgmt_client::public_key_hex_from_private(&private_key_hex)?.as_ref(),
            public.to_hex(),
            "the key read from the file is the key the account registered"
        );
        Ok(())
    }

    #[test]
    fn a_readable_file_is_preferred_over_the_agent_beside_it() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("id_ed25519");
        let pair = crate::ssh_keygen::generate("fixture")?;
        std::fs::write(&path, &pair.private_openssh)?;
        let (public, _) = Ed25519PublicKey::from_openssh(&pair.public_openssh)?;
        let found = inventory_of(vec![SystemKey {
            key: public,
            comment: "fixture".to_string(),
            sources: vec![
                KeySource::Agent {
                    label: "1Password",
                    socket: PathBuf::from("/tmp/agent.sock"),
                },
                KeySource::File {
                    path,
                    encrypted: false,
                },
            ],
        }]);

        let signer = signer_from(found, &public.to_hex())?;

        assert!(
            matches!(signer, UserSigner::PrivateKey(_)),
            "the file signs, and the agent is never asked"
        );
        Ok(())
    }

    #[test]
    fn a_key_only_an_agent_holds_signs_through_it() -> anyhow::Result<()> {
        let found = inventory_of(vec![agent_key(3, "laptop")]);

        let signer = signer_from(found, &key(3).to_hex())?;

        assert!(
            matches!(signer, UserSigner::External(_)),
            "an agent-held key is never read into this process"
        );
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn an_agent_held_key_signs_a_payload_that_verifies() -> anyhow::Result<()> {
        use ed25519_dalek::{Signature, Verifier, VerifyingKey};

        use pipette_ssh::fake_agent::{FakeAgent, FakeKey};

        let seed = [7; 32];
        let agent = FakeAgent::start(vec![FakeKey::signing(seed, "laptop")?])?;
        let public = FakeKey::public_key(seed);
        let found = inventory_of(vec![SystemKey {
            key: public,
            comment: "laptop".to_string(),
            sources: vec![KeySource::Agent {
                label: "fake",
                socket: agent.socket().to_path_buf(),
            }],
        }]);
        let UserSigner::External(signer) = signer_from(found, &public.to_hex())? else {
            anyhow::bail!("an agent-held key is never read into this process");
        };
        let payload =
            b"\"@method\": GET\n\"@signature-params\": (\"@method\");keyid=\"userkey_9f2c\"";

        let signature = signer
            .sign_payload(payload)
            .map_err(|error| anyhow::anyhow!("{error}"))?;

        VerifyingKey::from_bytes(public.as_bytes())?
            .verify(payload, &Signature::from_bytes(&signature))?;
        Ok(())
    }

    #[test]
    fn a_key_that_is_no_longer_here_is_named_by_fingerprint() -> anyhow::Result<()> {
        let found = inventory_of(vec![agent_key(1, "laptop")]);

        let error = signer_from(found, &key(2).to_hex())
            .err()
            .context("the registered key is not on this machine")?;

        assert!(
            error.to_string().contains(&key(2).fingerprint()),
            "got {error}"
        );
        assert!(error.to_string().contains("ssh-add"), "got {error}");
        Ok(())
    }
}
