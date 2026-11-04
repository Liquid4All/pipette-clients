//! Reading an `~/.ssh/id_ed25519` file: the same `openssh-key-v1` container
//! `pipette-ctl`'s generator writes, parsed back.
//!
//! An encrypted file still yields its public half, which the container holds in
//! the clear, so the key can be named and matched against an agent rather than
//! reported as unreadable.

use std::fmt;

use base64::{Engine, engine::general_purpose::STANDARD};

use crate::error::{Error, Result};
use crate::key::{ED25519_ALGORITHM, Ed25519PublicKey};
use crate::wire::Reader;

const BEGIN: &str = "-----BEGIN OPENSSH PRIVATE KEY-----";
const END: &str = "-----END OPENSSH PRIVATE KEY-----";
const MAGIC: &[u8] = b"openssh-key-v1\0";

/// The cipher name an unencrypted file carries. Anything else means a
/// passphrase this process does not have.
const NO_CIPHER: &[u8] = b"none";

const SEED_LEN: usize = 32;
/// OpenSSH stores the seed and the public key together as the "private key".
const SECRET_LEN: usize = 64;

/// One Ed25519 key read from an OpenSSH private key file.
pub struct PrivateKeyFile {
    pub public: Ed25519PublicKey,
    /// Empty when the file has none, and always for an encrypted file: the
    /// comment lives inside the encrypted section.
    pub comment: String,
    /// Absent when the file is encrypted, which is the only reason it can be.
    seed: Option<[u8; SEED_LEN]>,
}

impl PrivateKeyFile {
    /// Parse the text of an OpenSSH private key file.
    ///
    /// Ed25519 only, and one key per file: `ssh-keygen` writes no other shape,
    /// and a file holding something else is reported rather than guessed at.
    pub fn parse(text: &str) -> Result<Self> {
        let body = armored(text)?;
        let bytes = STANDARD.decode(body)?;

        let mut container = Reader::new(&bytes);
        if container.take(MAGIC.len())? != MAGIC {
            return Err(Error::NotOpenSshKey("the openssh-key-v1 magic is missing"));
        }
        let cipher = container.string()?;
        let _kdf = container.string()?;
        let _kdf_options = container.string()?;
        if container.u32()? != 1 {
            return Err(Error::NotOpenSshKey("the file holds more than one key"));
        }
        let public = Ed25519PublicKey::from_blob(container.string()?)?;
        let section = container.string()?;

        if cipher != NO_CIPHER {
            return Ok(Self {
                public,
                comment: String::new(),
                seed: None,
            });
        }
        let (seed, comment) = unlocked_section(section, &public)?;
        Ok(Self {
            public,
            comment,
            seed: Some(seed),
        })
    }

    pub fn is_encrypted(&self) -> bool {
        self.seed.is_none()
    }

    /// The seed as lowercase hex, the spelling the management client takes, or
    /// `None` when the file is encrypted.
    pub fn seed_hex(&self) -> Option<String> {
        self.seed.map(hex::encode)
    }
}

/// Never prints the seed: a `{:?}` in an error path is exactly where a private
/// key must not turn up.
impl fmt::Debug for PrivateKeyFile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PrivateKeyFile")
            .field("public", &self.public)
            .field("comment", &self.comment)
            .field("encrypted", &self.is_encrypted())
            .finish()
    }
}

/// The base64 between the two armor lines, stripped of the line breaks
/// `ssh-keygen` wraps it at.
fn armored(text: &str) -> Result<String> {
    let body = text
        .split_once(BEGIN)
        .ok_or(Error::NotOpenSshKey("the BEGIN line is missing"))?
        .1;
    let body = body
        .split_once(END)
        .ok_or(Error::NotOpenSshKey("the END line is missing"))?
        .0;
    Ok(body.split_whitespace().collect())
}

/// The private section of an unencrypted file: the seed, and the comment.
///
/// The two check-ints are OpenSSH's guard against a wrong passphrase. They are
/// checked even though nothing was decrypted: a `none` file whose guard fails
/// is corrupt, and its seed would sign for a key no server knows.
fn unlocked_section(section: &[u8], public: &Ed25519PublicKey) -> Result<([u8; SEED_LEN], String)> {
    let mut fields = Reader::new(section);
    if fields.u32()? != fields.u32()? {
        return Err(Error::NotOpenSshKey(
            "the check ints disagree, so the file is corrupt",
        ));
    }
    let algorithm = fields.string()?;
    if algorithm != ED25519_ALGORITHM.as_bytes() {
        return Err(Error::NotEd25519Blob {
            expected: ED25519_ALGORITHM,
            found: String::from_utf8_lossy(algorithm).into_owned(),
        });
    }
    let _public_again = fields.string()?;
    let secret = fields.string()?;
    if secret.len() != SECRET_LEN {
        return Err(Error::PrivateKeyLength {
            expected: SECRET_LEN,
            found: secret.len(),
        });
    }
    // The second half repeats the public key; it must agree with the header,
    // which is what everything else here matched on.
    let mut split = Reader::new(secret);
    let seed: [u8; SEED_LEN] = split
        .take(SEED_LEN)?
        .try_into()
        .map_err(|_| Error::NotOpenSshKey("the seed is not 32 bytes"))?;
    if split.take(SEED_LEN)? != public.as_bytes() {
        return Err(Error::NotOpenSshKey(
            "the public key inside disagrees with the one outside",
        ));
    }
    let comment = String::from_utf8_lossy(fields.string()?).into_owned();
    Ok((seed, comment))
}

#[cfg(test)]
mod tests {
    use anyhow::Context;
    use ed25519_dalek::SigningKey;
    use rstest::rstest;

    use super::*;

    use crate::fixtures::{LOCKED, LOCKED_FINGERPRINT, PLAIN, PLAIN_FINGERPRINT};

    #[test]
    fn an_unencrypted_file_yields_the_seed_its_public_key_names() -> anyhow::Result<()> {
        let file = PrivateKeyFile::parse(PLAIN)?;

        assert!(!file.is_encrypted());
        assert_eq!(file.comment, "plain@fixture");
        assert_eq!(file.public.fingerprint(), PLAIN_FINGERPRINT);

        let seed: [u8; SEED_LEN] = hex::decode(file.seed_hex().context("an unencrypted file")?)?
            .try_into()
            .ok()
            .context("the seed is 32 bytes")?;
        assert_eq!(
            SigningKey::from_bytes(&seed).verifying_key().to_bytes(),
            *file.public.as_bytes(),
            "the seed must derive the public key the file publishes"
        );
        Ok(())
    }

    #[test]
    fn an_encrypted_file_still_names_its_key() -> anyhow::Result<()> {
        let file = PrivateKeyFile::parse(LOCKED)?;

        assert!(file.is_encrypted());
        assert_eq!(file.seed_hex(), None, "no passphrase, no seed");
        assert_eq!(file.public.fingerprint(), LOCKED_FINGERPRINT);
        Ok(())
    }

    #[rstest]
    #[case::not_a_key(
        "nothing to see here",
        (|error| matches!(error, Error::NotOpenSshKey(_))) as fn(&Error) -> bool
    )]
    #[case::armor_without_an_end(
        PLAIN.trim_end_matches("-----END OPENSSH PRIVATE KEY-----\n"),
        (|error| matches!(error, Error::NotOpenSshKey(_))) as fn(&Error) -> bool
    )]
    #[case::truncated_container(
        "-----BEGIN OPENSSH PRIVATE KEY-----\nb3Blb\n-----END OPENSSH PRIVATE KEY-----\n",
        (|error| matches!(error, Error::Base64(_))) as fn(&Error) -> bool
    )]
    #[case::not_base64(
        "-----BEGIN OPENSSH PRIVATE KEY-----\n!!!!\n-----END OPENSSH PRIVATE KEY-----\n",
        (|error| matches!(error, Error::Base64(_))) as fn(&Error) -> bool
    )]
    fn a_file_that_is_not_a_key_is_a_typed_error(
        #[case] text: &str,
        #[case] expected: fn(&Error) -> bool,
    ) -> anyhow::Result<()> {
        let err = PrivateKeyFile::parse(text)
            .err()
            .with_context(|| format!("parsed {text:?} as a key"))?;
        assert!(expected(&err), "got {err}");
        Ok(())
    }

    #[test]
    fn a_corrupt_unencrypted_section_is_rejected() -> anyhow::Result<()> {
        let public = PrivateKeyFile::parse(PLAIN)?.public;
        let mut section = Vec::new();
        section.extend_from_slice(&1u32.to_be_bytes());
        section.extend_from_slice(&2u32.to_be_bytes());

        let err = unlocked_section(&section, &public)
            .err()
            .context("check ints that disagree are a corrupt file")?;

        assert!(matches!(err, Error::NotOpenSshKey(_)), "got {err}");
        Ok(())
    }
}
