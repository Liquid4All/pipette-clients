//! An Ed25519 public key in the three spellings this workspace moves it
//! between: the raw 32 bytes, the OpenSSH blob an agent and an
//! `~/.ssh/*.pub` line carry it in, and the hex the management server
//! registers it as.
//!
//! One type rather than a bare `[u8; 32]`, because the conversions are where
//! the mistakes live: the blob is not the key, and the fingerprint is over the
//! blob rather than over the key.

use std::fmt;

use base64::{
    Engine,
    engine::general_purpose::{STANDARD, STANDARD_NO_PAD},
};
use sha2::{Digest, Sha256};

use crate::error::{Error, Result};
use crate::wire::Reader;

/// The algorithm name an Ed25519 key is published and signed under. Every blob
/// this module reads or writes leads with it.
pub const ED25519_ALGORITHM: &str = "ssh-ed25519";

const KEY_LEN: usize = 32;

/// An Ed25519 public key: the value a server records, and the value that names
/// which key an agent should sign with.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Ed25519PublicKey([u8; KEY_LEN]);

impl Ed25519PublicKey {
    pub fn from_bytes(bytes: [u8; KEY_LEN]) -> Self {
        Self(bytes)
    }

    pub fn as_bytes(&self) -> &[u8; KEY_LEN] {
        &self.0
    }

    /// The key as lowercase hex, the form the management server registers a
    /// user key in.
    pub fn to_hex(&self) -> String {
        hex::encode(self.0)
    }

    pub fn from_hex(hex: &str) -> Result<Self> {
        let bytes = hex::decode(hex)?;
        let found = bytes.len();
        Ok(Self(bytes.try_into().map_err(|_| {
            Error::PublicKeyLength {
                expected: KEY_LEN,
                found,
            }
        })?))
    }

    /// The OpenSSH public-key blob: `string "ssh-ed25519" || string key`.
    ///
    /// This, not the bare key, is what the agent protocol names a key by and
    /// what a fingerprint is taken over. Both lengths are fixed, so the
    /// prefixes are written as literals and nothing here can fail.
    pub fn blob(&self) -> Vec<u8> {
        let mut blob = Vec::with_capacity(4 + ED25519_ALGORITHM.len() + 4 + KEY_LEN);
        blob.extend_from_slice(&[0, 0, 0, 11]);
        blob.extend_from_slice(ED25519_ALGORITHM.as_bytes());
        blob.extend_from_slice(&[0, 0, 0, 32]);
        blob.extend_from_slice(&self.0);
        blob
    }

    /// Read a key out of an OpenSSH public-key blob. Names the algorithm it
    /// rejected, so a caller can tell "not Ed25519" from "not a blob".
    pub fn from_blob(blob: &[u8]) -> Result<Self> {
        let mut fields = Reader::new(blob);
        let algorithm = fields.string()?;
        if algorithm != ED25519_ALGORITHM.as_bytes() {
            return Err(Error::NotEd25519Blob {
                expected: ED25519_ALGORITHM,
                found: String::from_utf8_lossy(algorithm).into_owned(),
            });
        }
        let key = fields.string()?;
        let found = key.len();
        Ok(Self(key.try_into().map_err(|_| {
            Error::PublicKeyLength {
                expected: KEY_LEN,
                found,
            }
        })?))
    }

    /// The `SHA256:<base64>` fingerprint `ssh-add -l` prints, which is how an
    /// operator tells an agent's keys apart when the comments are empty.
    ///
    /// Unpadded base64 over the blob, matching OpenSSH byte for byte: anything
    /// else would be unmatchable against what the operator can see.
    pub fn fingerprint(&self) -> String {
        format!(
            "SHA256:{}",
            STANDARD_NO_PAD.encode(Sha256::digest(self.blob()))
        )
    }

    /// The single `ssh-ed25519 <base64> <comment>` line of an `.pub` file. An
    /// empty comment yields no trailing space, the way `ssh-keygen` writes one.
    pub fn to_openssh(&self, comment: &str) -> String {
        let encoded = STANDARD.encode(self.blob());
        if comment.is_empty() {
            format!("{ED25519_ALGORITHM} {encoded}")
        } else {
            format!("{ED25519_ALGORITHM} {encoded} {comment}")
        }
    }

    /// Read one `ssh-ed25519 <base64> <comment>` line, as an `~/.ssh/*.pub`
    /// file or an `authorized_keys` entry holds it, with the comment beside it.
    ///
    /// The comment is returned because an agent often reports none (1Password
    /// always reports none), and the `.pub` file is then its only home.
    pub fn from_openssh(line: &str) -> Result<(Self, String)> {
        let mut fields = line.split_whitespace();
        let algorithm = fields
            .next()
            .ok_or(Error::NotOpenSshPublicKey("the line is empty"))?;
        if algorithm != ED25519_ALGORITHM {
            return Err(Error::NotEd25519Blob {
                expected: ED25519_ALGORITHM,
                found: algorithm.to_string(),
            });
        }
        let encoded = fields
            .next()
            .ok_or(Error::NotOpenSshPublicKey("the key itself is missing"))?;
        let key = Self::from_blob(&STANDARD.decode(encoded)?)?;
        Ok((key, fields.collect::<Vec<_>>().join(" ")))
    }
}

/// Renders as hex, the spelling that is also the server's. A public key is not
/// a secret, so it prints in full.
impl fmt::Debug for Ed25519PublicKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Ed25519PublicKey({})", self.to_hex())
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Context;

    use super::*;
    use crate::wire::push_string;

    /// A throwaway key, and the `.pub` line and fingerprint real OpenSSH tools
    /// print for it.
    const FIXTURE_HEX: &str = "c8bef8049584357385e07db4dce82bca8e747cb5922bbda85cfbfb14230a55b3";
    const FIXTURE_PUB: &str = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIMi++ASVhDVzheB9tNzoK8qOdHy1kiu9qFz7+xQjClWz pipette-ssh-fixture";
    const FIXTURE_FINGERPRINT: &str = "SHA256:oS2ZRCVQoQIfMmuVpoC897lnBXGwaDAEETqdXTUJwYo";
    /// The same key with its second half cut off, as a half-pasted one arrives.
    const TRUNCATED_HEX: &str = "c8bef8049584357385e07db4dce82bca";

    #[test]
    fn renders_the_openssh_line_and_fingerprint_openssh_prints() -> anyhow::Result<()> {
        let key = Ed25519PublicKey::from_hex(FIXTURE_HEX)?;
        assert_eq!(key.to_openssh("pipette-ssh-fixture"), FIXTURE_PUB);
        assert_eq!(key.fingerprint(), FIXTURE_FINGERPRINT);
        Ok(())
    }

    /// The blob writes its length prefixes as literals; this pins them to the
    /// wire encoder so the two can never drift.
    #[test]
    fn the_blob_is_what_the_wire_encoder_would_write() -> anyhow::Result<()> {
        let key = Ed25519PublicKey::from_hex(FIXTURE_HEX)?;
        let mut expected = Vec::new();
        push_string(&mut expected, ED25519_ALGORITHM.as_bytes())?;
        push_string(&mut expected, key.as_bytes())?;
        assert_eq!(key.blob(), expected);
        Ok(())
    }

    #[test]
    fn hex_and_blob_round_trip() -> anyhow::Result<()> {
        let key = Ed25519PublicKey::from_hex(FIXTURE_HEX)?;
        assert_eq!(Ed25519PublicKey::from_hex(&key.to_hex())?, key);
        assert_eq!(Ed25519PublicKey::from_blob(&key.blob())?, key);
        Ok(())
    }

    #[test]
    fn another_algorithm_is_rejected_by_name() -> anyhow::Result<()> {
        let mut blob = Vec::new();
        push_string(&mut blob, b"ssh-rsa")?;
        push_string(&mut blob, b"...")?;

        let err = Ed25519PublicKey::from_blob(&blob)
            .err()
            .context("an ssh-rsa blob is not an Ed25519 key")?;
        assert!(
            matches!(err, Error::NotEd25519Blob { ref found, .. } if found == "ssh-rsa"),
            "got {err}"
        );
        Ok(())
    }

    /// Half a pasted key is still valid hex, so only the length catches it.
    #[test]
    fn a_short_key_is_rejected_on_length() -> anyhow::Result<()> {
        let err = Ed25519PublicKey::from_hex(TRUNCATED_HEX)
            .err()
            .context("16 bytes is not an Ed25519 key")?;
        assert!(
            matches!(err, Error::PublicKeyLength { found: 16, .. }),
            "got {err}"
        );
        Ok(())
    }
}
