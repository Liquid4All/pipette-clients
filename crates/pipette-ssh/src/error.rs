use std::io;
use std::path::PathBuf;

/// Everything that can go wrong reading an SSH key or talking to a local agent.
///
/// The agent failures are split finely because each one asks the operator for
/// something different: start an agent, unlock it, add the key, or deal with an
/// agent that has the key and said no. An agent spends one failure byte on all
/// of them, so [`crate::Agent::sign`] reconstructs the difference.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("cannot reach the ssh-agent at {}: {source}", socket.display())]
    Connect {
        socket: PathBuf,
        #[source]
        source: io::Error,
    },

    #[error("ssh-agent I/O failed: {0}")]
    Io(#[from] io::Error),

    #[error(
        "the ssh-agent lists no keys at all, so it is either locked or empty; unlock it, or `ssh-add` a key (wanted {fingerprint})"
    )]
    AgentHoldsNothing { fingerprint: String },

    #[error(
        "the ssh-agent holds other keys but not {fingerprint}; `ssh-add` it, or use one the agent already has"
    )]
    KeyNotHeld { fingerprint: String },

    #[error(
        "the ssh-agent holds {fingerprint} and refused to sign with it (it answered message type {response}); an approval prompt may have been dismissed"
    )]
    SignRefused { fingerprint: String, response: u8 },

    #[error("the ssh-agent answered message type {found}, expected {expected}")]
    UnexpectedResponse { expected: u8, found: u8 },

    #[error("malformed ssh-agent message: {0}")]
    Malformed(&'static str),

    #[error("the ssh-agent announced a {len}-byte message, over the {max}-byte ceiling")]
    MessageTooLarge { len: u32, max: u32 },

    #[error("the ssh-agent signed with {found:?}, not {expected:?}")]
    SignatureAlgorithm {
        expected: &'static str,
        found: String,
    },

    #[error("an Ed25519 signature is {expected} bytes, the ssh-agent returned {found}")]
    SignatureLength { expected: usize, found: usize },

    #[error("not a hex Ed25519 public key: {0}")]
    PublicKeyNotHex(#[from] hex::FromHexError),

    #[error("an Ed25519 public key is {expected} bytes, got {found}")]
    PublicKeyLength { expected: usize, found: usize },

    #[error("not an {expected:?} public key blob: {found:?}")]
    NotEd25519Blob {
        expected: &'static str,
        found: String,
    },

    #[error("not an OpenSSH private key: {0}")]
    NotOpenSshKey(&'static str),

    /// Carries no path: it is reported against the file it was read from, and
    /// naming it twice in one line reads as two files.
    #[error("cannot be read: {0}")]
    Unreadable(#[source] io::Error),

    #[error("an SSH key is not valid base64: {0}")]
    Base64(#[from] base64::DecodeError),

    #[error("not an OpenSSH public key line: {0}")]
    NotOpenSshPublicKey(&'static str),

    #[error("an Ed25519 private key is {expected} bytes, got {found}")]
    PrivateKeyLength { expected: usize, found: usize },

    #[error("an SSH string of {0} bytes exceeds the u32 wire limit")]
    StringTooLong(usize),

    #[error("an ssh-agent request of {0} bytes exceeds the u32 wire limit")]
    RequestTooLarge(usize),
}

pub type Result<T> = std::result::Result<T, Error>;
