//! The SSH keys a host already has, and the local agent that signs with one.
//!
//! Lets a person authenticate with the key their machine already holds,
//! instead of a second private key written to disk.
//!
//! Ed25519 only: the management server registers a user key as 32 bytes of
//! hex and verifies an Ed25519 signature, so [`Agent::ed25519_identities`]
//! skips RSA and ECDSA keys.

pub mod agent;
pub mod error;
pub mod inventory;
pub mod key;
pub mod known;
pub mod private;
pub mod wire;

#[cfg(test)]
mod fixtures;

#[cfg(all(unix, any(test, feature = "test-support")))]
pub mod fake_agent;

pub use agent::{AUTH_SOCK_ENV, Agent, Identity};
pub use error::{Error, Result};
pub use inventory::{Inventory, KeySource, SystemKey, inventory};
pub use key::{ED25519_ALGORITHM, Ed25519PublicKey};
pub use known::{KnownAgent, discover};
pub use private::PrivateKeyFile;
