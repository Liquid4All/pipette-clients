//! A client for the local ssh-agent: list the Ed25519 keys it holds, and ask
//! it to sign bytes with one.
//!
//! Going through an agent keeps the private key out of this process. The agent
//! applies no hashing or prefixing of its own, so the 64 bytes it returns are
//! an ordinary Ed25519 signature over exactly the input.
//!
//! The protocol is `draft-miller-ssh-agent`: a big-endian `u32` length, a type
//! byte, then that type's fields as [`crate::wire`] strings.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use crate::error::{Error, Result};
use crate::key::{ED25519_ALGORITHM, Ed25519PublicKey};
use crate::wire::{Reader, push_string};

pub const AUTH_SOCK_ENV: &str = "SSH_AUTH_SOCK";

const REQUEST_IDENTITIES: u8 = 11;
const IDENTITIES_ANSWER: u8 = 12;
const SIGN_REQUEST: u8 = 13;
const SIGN_RESPONSE: u8 = 14;

/// Ed25519 defines no signature flags; the field exists for RSA's choice of
/// hash.
const NO_FLAGS: u32 = 0;

const SIGNATURE_LEN: usize = 64;

/// OpenSSH's own ceiling. A reply claiming more is a wedged or hostile agent,
/// and believing it would mean allocating whatever it asked for.
const MAX_MESSAGE: u32 = 256 * 1024;

/// A ceiling on a hung agent, not a latency budget: a key that requires
/// approval holds the reply until a human clicks, so anything tuned to a local
/// socket would abort exactly the case the prompt exists for.
///
/// Unix only. Windows reaches its agent through a named pipe opened as a file
/// handle, which has no timeout to set without a platform crate.
#[cfg(unix)]
const REPLY_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(120);

/// One Ed25519 key an agent holds.
#[derive(Debug, Clone)]
pub struct Identity {
    pub key: Ed25519PublicKey,
    /// Often empty, and always empty from 1Password, so
    /// [`Ed25519PublicKey::fingerprint`] is what tells keys apart.
    pub comment: String,
}

/// A handle on the agent listening at one socket.
///
/// Holds the path, not a connection: each request opens its own, because an
/// agent may close the socket after answering and because a signing identity
/// outlives any one request.
#[derive(Debug, Clone)]
pub struct Agent {
    socket: PathBuf,
}

impl Agent {
    pub fn at(socket: impl Into<PathBuf>) -> Self {
        Self {
            socket: socket.into(),
        }
    }

    pub fn socket(&self) -> &Path {
        &self.socket
    }

    /// The Ed25519 keys the agent holds, in the order it lists them.
    ///
    /// Other algorithms are skipped rather than reported: a developer's agent
    /// commonly holds an RSA key, and failing on it would make this unusable.
    pub fn ed25519_identities(&self) -> Result<Vec<Identity>> {
        let reply = self.request(&[REQUEST_IDENTITIES])?;
        let mut fields = Reader::new(&reply);
        let kind = fields.u8()?;
        if kind != IDENTITIES_ANSWER {
            return Err(Error::UnexpectedResponse {
                expected: IDENTITIES_ANSWER,
                found: kind,
            });
        }

        let count = fields.u32()?;
        (0..count).try_fold(Vec::new(), |mut identities, _| {
            let blob = fields.string()?;
            let comment = fields.string()?;
            // Read the algorithm first so a key of another kind is skipped,
            // while a blob that is malformed still fails.
            if Reader::new(blob).string()? == ED25519_ALGORITHM.as_bytes() {
                identities.push(Identity {
                    key: Ed25519PublicKey::from_blob(blob)?,
                    comment: String::from_utf8_lossy(comment).into_owned(),
                });
            }
            Ok(identities)
        })
    }

    /// Sign `data`, as given, with the key the agent holds for `key`.
    ///
    /// One round trip when it succeeds. An agent spends the same failure byte
    /// on every reason it might say no, so the listing that tells those apart
    /// is only paid for on the failure path; see [`Self::explain_refusal`].
    pub fn sign(&self, key: &Ed25519PublicKey, data: &[u8]) -> Result<[u8; SIGNATURE_LEN]> {
        let mut body = vec![SIGN_REQUEST];
        push_string(&mut body, &key.blob())?;
        push_string(&mut body, data)?;
        body.extend_from_slice(&NO_FLAGS.to_be_bytes());

        let reply = self.request(&body)?;
        let mut fields = Reader::new(&reply);
        let kind = fields.u8()?;
        if kind != SIGN_RESPONSE {
            return Err(self.explain_refusal(key, kind));
        }

        // Unwrapped from its `string algorithm || string signature` framing,
        // what remains is a plain Ed25519 signature over `data`.
        let mut signature = Reader::new(fields.string()?);
        let algorithm = signature.string()?;
        if algorithm != ED25519_ALGORITHM.as_bytes() {
            return Err(Error::SignatureAlgorithm {
                expected: ED25519_ALGORITHM,
                found: String::from_utf8_lossy(algorithm).into_owned(),
            });
        }
        let bytes = signature.string()?;
        let found = bytes.len();
        bytes.try_into().map_err(|_| Error::SignatureLength {
            expected: SIGNATURE_LEN,
            found,
        })
    }

    /// Turn a refusal into something the operator can act on, by asking the
    /// agent what it holds.
    ///
    /// A locked agent (verified with OpenSSH's `ssh-add -x`) lists zero keys,
    /// exactly like an empty one, so the empty case gets its own error rather
    /// than sending someone to `ssh-add` a key they may already have.
    ///
    /// Best effort: if the listing itself fails, the refusal is still reported.
    fn explain_refusal(&self, key: &Ed25519PublicKey, response: u8) -> Error {
        let fingerprint = key.fingerprint();
        match self.ed25519_identities() {
            Ok(identities) if identities.is_empty() => Error::AgentHoldsNothing { fingerprint },
            Ok(identities) if !identities.iter().any(|held| held.key == *key) => {
                Error::KeyNotHeld { fingerprint }
            }
            Ok(_) | Err(_) => Error::SignRefused {
                fingerprint,
                response,
            },
        }
    }

    /// Send one message and read the one that answers it, payload only: the
    /// length prefix is this function's, both directions.
    fn request(&self, body: &[u8]) -> Result<Vec<u8>> {
        let mut stream = connect(&self.socket)?;

        let len = u32::try_from(body.len()).map_err(|_| Error::RequestTooLarge(body.len()))?;
        stream.write_all(&len.to_be_bytes())?;
        stream.write_all(body)?;
        stream.flush()?;

        let mut header = [0u8; 4];
        stream.read_exact(&mut header)?;
        let len = u32::from_be_bytes(header);
        if len > MAX_MESSAGE {
            return Err(Error::MessageTooLarge {
                len,
                max: MAX_MESSAGE,
            });
        }
        let len = usize::try_from(len).map_err(|_| Error::Malformed("reply longer than memory"))?;
        let mut payload = vec![0u8; len];
        stream.read_exact(&mut payload)?;
        Ok(payload)
    }
}

/// The agent socket is a unix-domain socket everywhere but Windows, where
/// OpenSSH and 1Password both serve a named pipe instead.
/// A pipe opens for read and write as an ordinary file handle, and the protocol
/// above it is identical.
#[cfg(unix)]
fn connect(socket: &Path) -> Result<std::os::unix::net::UnixStream> {
    let stream =
        std::os::unix::net::UnixStream::connect(socket).map_err(|source| Error::Connect {
            socket: socket.to_path_buf(),
            source,
        })?;
    stream.set_read_timeout(Some(REPLY_TIMEOUT))?;
    stream.set_write_timeout(Some(REPLY_TIMEOUT))?;
    Ok(stream)
}

#[cfg(windows)]
fn connect(socket: &Path) -> Result<std::fs::File> {
    std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(socket)
        .map_err(|source| Error::Connect {
            socket: socket.to_path_buf(),
            source,
        })
}

#[cfg(all(test, unix))]
mod tests {
    use anyhow::Context;
    use ed25519_dalek::{Signature, Verifier, VerifyingKey};
    use rstest::rstest;

    use super::*;
    use crate::fake_agent::{CannedReply, FakeAgent, FakeKey};

    const HELD_SEED: [u8; 32] = [7; 32];
    const ABSENT_SEED: [u8; 32] = [9; 32];

    /// Shaped like a management payload, newlines and all.
    const PAYLOAD: &[u8] =
        b"v2\nGET\n/users/me\n2026-03-10T12:00:00Z\nuser_0192f0c1\nuserkey_9f2c\n0f1e";

    #[test]
    fn an_agent_signature_verifies_against_the_public_key() -> anyhow::Result<()> {
        let agent = FakeAgent::start(vec![FakeKey::signing(HELD_SEED, "")?])?;
        let key = FakeKey::public_key(HELD_SEED);

        let signature = Agent::at(agent.socket()).sign(&key, PAYLOAD)?;

        VerifyingKey::from_bytes(key.as_bytes())?
            .verify(PAYLOAD, &Signature::from_bytes(&signature))?;
        Ok(())
    }

    #[test]
    fn lists_only_the_ed25519_keys() -> anyhow::Result<()> {
        let agent = FakeAgent::start(vec![
            FakeKey::other_algorithm("work laptop")?,
            FakeKey::signing(HELD_SEED, "fleet key")?,
        ])?;

        let identities = Agent::at(agent.socket()).ed25519_identities()?;

        let only = identities.first().context("the Ed25519 key is listed")?;
        assert_eq!(identities.len(), 1, "got {identities:?}");
        assert_eq!(only.key, FakeKey::public_key(HELD_SEED));
        assert_eq!(only.comment, "fleet key");
        Ok(())
    }

    #[test]
    fn an_unreachable_socket_names_the_path() -> anyhow::Result<()> {
        let dir = tempfile::TempDir::new()?;
        let socket = dir.path().join("absent.sock");

        let err = Agent::at(&socket)
            .ed25519_identities()
            .err()
            .context("a socket that does not exist cannot be reached")?;

        assert!(matches!(err, Error::Connect { .. }), "got {err}");
        assert!(
            err.to_string().contains("absent.sock"),
            "message must name the socket: {err}"
        );
        Ok(())
    }

    #[rstest]
    #[case::a_key_it_does_not_hold(
        vec![FakeKey::signing(HELD_SEED, "")?],
        ABSENT_SEED,
        (|error| matches!(error, Error::KeyNotHeld { fingerprint } if *fingerprint == FakeKey::public_key(ABSENT_SEED).fingerprint())) as fn(&Error) -> bool
    )]
    #[case::an_agent_that_lists_nothing(
        vec![],
        HELD_SEED,
        (|error| matches!(error, Error::AgentHoldsNothing { .. })) as fn(&Error) -> bool
    )]
    #[case::a_key_it_holds_but_will_not_use(
        vec![FakeKey::refusing(HELD_SEED, "")?],
        HELD_SEED,
        (|error| matches!(error, Error::SignRefused { .. })) as fn(&Error) -> bool
    )]
    fn a_refusal_is_explained_by_what_the_agent_lists(
        #[case] held: Vec<FakeKey>,
        #[case] asked: [u8; 32],
        #[case] expected: fn(&Error) -> bool,
    ) -> anyhow::Result<()> {
        let agent = FakeAgent::start(held)?;

        let err = Agent::at(agent.socket())
            .sign(&FakeKey::public_key(asked), PAYLOAD)
            .err()
            .context("the agent cannot sign with this key")?;

        assert!(expected(&err), "got {err}");
        Ok(())
    }

    #[rstest]
    #[case::a_length_past_the_ceiling(
        CannedReply::oversized_length(),
        (|error| matches!(error, Error::MessageTooLarge { .. })) as fn(&Error) -> bool
    )]
    #[case::a_key_count_with_no_keys(
        CannedReply::missing_fields(),
        (|error| matches!(error, Error::Malformed(_))) as fn(&Error) -> bool
    )]
    #[case::a_signature_where_a_listing_was_asked_for(
        CannedReply::wrong_type(),
        (|error| matches!(error, Error::UnexpectedResponse { .. })) as fn(&Error) -> bool
    )]
    fn a_reply_that_lies_about_its_shape_is_a_typed_error(
        #[case] reply: CannedReply,
        #[case] expected: fn(&Error) -> bool,
    ) -> anyhow::Result<()> {
        let agent = FakeAgent::start_canned(reply)?;

        let err = Agent::at(agent.socket())
            .ed25519_identities()
            .err()
            .context("the canned reply is not a valid listing")?;

        assert!(expected(&err), "got {err}");
        Ok(())
    }
}
