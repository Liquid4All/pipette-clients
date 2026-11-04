//! An ssh-agent stood up on a real unix socket, for the agent client's tests
//! and, behind `test-support`, a dependent's.
//!
//! CI runs no agent and a developer's holds arbitrary keys. A real socket also
//! exercises the framing end to end.

use std::io::{Read, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::{self, JoinHandle};

use ed25519_dalek::{Signer, SigningKey};
use tempfile::TempDir;

use crate::key::{ED25519_ALGORITHM, Ed25519PublicKey};
use crate::wire::{Reader, push_string};

/// Spelled out rather than imported from `agent`, so a change to the constants
/// there has to disagree with something.
const REQUEST_IDENTITIES: u8 = 11;
const IDENTITIES_ANSWER: u8 = 12;
const SIGN_REQUEST: u8 = 13;
const SIGN_RESPONSE: u8 = 14;
const FAILURE: u8 = 5;

/// A key the fake agent lists, and whether it will sign with it. A `None`
/// signer is what a dismissed approval prompt looks like on the wire: held,
/// but declined.
pub struct FakeKey {
    blob: Vec<u8>,
    comment: String,
    signer: Option<SigningKey>,
}

impl FakeKey {
    pub fn signing(seed: [u8; 32], comment: &str) -> crate::Result<Self> {
        let signer = SigningKey::from_bytes(&seed);
        Ok(Self {
            blob: Ed25519PublicKey::from_bytes(signer.verifying_key().to_bytes()).blob(),
            comment: comment.to_string(),
            signer: Some(signer),
        })
    }

    pub fn refusing(seed: [u8; 32], comment: &str) -> crate::Result<Self> {
        Ok(Self {
            signer: None,
            ..Self::signing(seed, comment)?
        })
    }

    pub fn other_algorithm(comment: &str) -> crate::Result<Self> {
        let mut blob = Vec::new();
        push_string(&mut blob, b"ssh-rsa")?;
        push_string(&mut blob, b"not-an-ed25519-key")?;
        Ok(Self {
            blob,
            comment: comment.to_string(),
            signer: None,
        })
    }

    pub fn public_key(seed: [u8; 32]) -> Ed25519PublicKey {
        Ed25519PublicKey::from_bytes(SigningKey::from_bytes(&seed).verifying_key().to_bytes())
    }
}

/// A frame the fake writes verbatim in place of a well-formed answer, length
/// prefix included, so a test can drive the client's guards against an agent
/// that lies about its own message.
pub struct CannedReply(Vec<u8>);

impl CannedReply {
    /// A length past the client's ceiling, with no body behind it. Written as a
    /// literal rather than derived from the client's constant, so raising that
    /// ceiling has to disagree with this.
    pub fn oversized_length() -> Self {
        Self((256 * 1024 + 1u32).to_be_bytes().to_vec())
    }

    /// A listing that promises a key it does not carry.
    pub fn missing_fields() -> Self {
        Self(vec![0, 0, 0, 5, IDENTITIES_ANSWER, 0, 0, 0, 1])
    }

    /// A signature where a listing was asked for.
    pub fn wrong_type() -> Self {
        Self(vec![0, 0, 0, 1, SIGN_RESPONSE])
    }
}

enum Behaviour {
    Keys(Vec<FakeKey>),
    Canned(CannedReply),
}

pub struct FakeAgent {
    socket: PathBuf,
    /// Held so the socket lives as long as the agent, and is removed with it.
    _dir: TempDir,
    stop: Arc<AtomicBool>,
    worker: Option<JoinHandle<()>>,
}

impl FakeAgent {
    pub fn start(keys: Vec<FakeKey>) -> anyhow::Result<Self> {
        Self::serving(Behaviour::Keys(keys))
    }

    pub fn start_canned(reply: CannedReply) -> anyhow::Result<Self> {
        Self::serving(Behaviour::Canned(reply))
    }

    fn serving(behaviour: Behaviour) -> anyhow::Result<Self> {
        let dir = TempDir::new()?;
        let socket = dir.path().join("agent.sock");
        let listener = UnixListener::bind(&socket)?;
        let stop = Arc::new(AtomicBool::new(false));

        let worker = {
            let stop = Arc::clone(&stop);
            thread::spawn(move || {
                listener
                    .incoming()
                    .take_while(|_| !stop.load(Ordering::SeqCst))
                    .for_each(|stream| {
                        if let Ok(mut stream) = stream {
                            let _ = serve(&mut stream, &behaviour);
                        }
                    });
            })
        };

        Ok(Self {
            socket,
            _dir: dir,
            stop,
            worker: Some(worker),
        })
    }

    pub fn socket(&self) -> &Path {
        &self.socket
    }
}

impl Drop for FakeAgent {
    /// `accept` blocks, so the flag alone would not end the worker: the extra
    /// connection is what wakes it to see the flag.
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        let _ = UnixStream::connect(&self.socket);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

/// Answer one request. Errors end the connection, which is what a real agent
/// does with a message it cannot parse.
fn serve(stream: &mut UnixStream, behaviour: &Behaviour) -> anyhow::Result<()> {
    let mut header = [0u8; 4];
    stream.read_exact(&mut header)?;
    let mut payload = vec![0u8; usize::try_from(u32::from_be_bytes(header))?];
    stream.read_exact(&mut payload)?;

    // The request is drained either way, so a canned answer still exercises the
    // client's write path.
    let keys = match behaviour {
        Behaviour::Canned(CannedReply(frame)) => {
            stream.write_all(frame)?;
            stream.flush()?;
            return Ok(());
        }
        Behaviour::Keys(keys) => keys,
    };

    let mut fields = Reader::new(&payload);
    let mut reply = Vec::new();
    match fields.u8()? {
        REQUEST_IDENTITIES => {
            reply.push(IDENTITIES_ANSWER);
            reply.extend_from_slice(&u32::try_from(keys.len())?.to_be_bytes());
            keys.iter().try_for_each(|key| {
                push_string(&mut reply, &key.blob)?;
                push_string(&mut reply, key.comment.as_bytes())
            })?;
        }
        SIGN_REQUEST => {
            let blob = fields.string()?;
            let data = fields.string()?;
            match keys
                .iter()
                .find(|key| key.blob == blob)
                .and_then(|key| key.signer.as_ref())
            {
                Some(signer) => {
                    let mut signature = Vec::new();
                    push_string(&mut signature, ED25519_ALGORITHM.as_bytes())?;
                    push_string(&mut signature, &signer.sign(data).to_bytes())?;
                    reply.push(SIGN_RESPONSE);
                    push_string(&mut reply, &signature)?;
                }
                None => reply.push(FAILURE),
            }
        }
        _ => reply.push(FAILURE),
    }

    stream.write_all(&u32::try_from(reply.len())?.to_be_bytes())?;
    stream.write_all(&reply)?;
    stream.flush()?;
    Ok(())
}
