//! The client's error: registration, identity files and catalog conversion.
//! Runtime-layer failures arrive through [`Error::Ops`].

use std::path::PathBuf;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("no local registration found at {0}")]
    RegistrationMissing(PathBuf),

    #[error("no usable local private key found")]
    PrivateKeyMissing,

    #[error("identity already exists ({0}); run `auth reset` before re-registering")]
    IdentityExists(String),

    #[error("the server registered this client under an unusable id, which cannot sign: {0}")]
    RegisteredId(#[source] pipette_plan_types::PrincipalIdError),

    #[error(
        "registered with the server as {client_id}, but saving the local \
             identity failed ({source}); any pre-auth key used is now spent. \
             Ask an admin to re-issue one before retrying"
    )]
    RegistrationPersisted {
        client_id: String,
        #[source]
        source: Box<Error>,
    },

    #[error(
        "set-device requires at least one of --device-name, --device-form-factor, or --client-details"
    )]
    SetDeviceEmpty,

    /// `sync <result-id>` on a workspace that cannot sign: a result id asks to
    /// submit, so this refuses rather than pulling and ignoring the argument.
    #[error(
        "submitting a result needs a registration; run `pipette auth register` \
             (or `auth import`) first, or run `pipette sync` with no result id to \
             pull the benchmark catalog"
    )]
    SyncSubmitNeedsRegistration,

    /// No default server on purpose: `auth register` records one, and a
    /// fallback here would have `sync` reach a host nobody named.
    #[error(
        "no management server to pull from: this workspace holds no \
             registration; pass --server-url, or run `pipette auth register` to \
             record one"
    )]
    SyncServerUrlUnknown,

    #[error(
        "--server-url {requested} disagrees with the registered server \
             {registered}, which this client's identity signs for; drop the flag, \
             or run `auth reset` to re-register elsewhere"
    )]
    SyncServerUrlConflict {
        requested: pipette_mgmt_client::ServerUrl,
        registered: pipette_mgmt_client::ServerUrl,
    },

    #[error(
        "{unsubmitted} unsubmitted result(s) would be lost; run `auth reset --force` \
             to delete them along with the identity, or `sync` them first"
    )]
    ResetNeedsForce { unsubmitted: usize },

    #[error("the bundle is not base64url; pass the whole token `auth export` printed")]
    BundleNotBase64url,

    #[error("the bundle decoded but is not an identity record: {0}")]
    BundleNotAnIdentity(String),

    #[error(
        "bundle version {found} is not readable by this build, which reads \
             version {expected}"
    )]
    BundleVersion { found: u32, expected: u32 },

    #[error("the bundle's `client_id` names no client to restore: {0}")]
    BundleClientId(#[source] pipette_plan_types::PrincipalIdError),

    #[error("the bundle's server URL is unusable: {0}")]
    BundleServerUrl(#[source] pipette_mgmt_client::ServerUrlError),

    #[error("the bundle's signing key is unusable: {0}")]
    BundlePrivateKey(#[source] pipette_mgmt_client::PrivateKeyError),

    #[error(
        "this workspace already holds {held}, which the bundle would replace; \
             run `auth reset` first, or pass --force to replace it and drop the \
             caches derived from it, {unsubmitted} unsubmitted result(s) among them"
    )]
    ImportIdentityConflict { held: String, unsubmitted: usize },

    /// A server benchmark that is not a valid
    /// [`pipette_plan_types::BenchmarkDefinition`]. `sync` skips it rather than
    /// aborting.
    #[error(
        "benchmark \"{}\" is not a known, valid definition: {source}",
        .benchmark_id.escape_default()
    )]
    BenchmarkConversion {
        benchmark_id: String,
        #[source]
        source: serde_json::Error,
    },

    #[error("removing {}: {source}", .path.display())]
    RemoveFile {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error(transparent)]
    Mgmt(#[from] pipette_mgmt_client::error::Error),

    #[error(transparent)]
    InvalidServerUrl(#[from] pipette_mgmt_client::ServerUrlError),

    #[error(transparent)]
    Ops(#[from] pipette_ops::Error),

    /// The console went away under the command: a pipe closed early, or a
    /// full terminal buffer.
    #[error("writing to the console: {0}")]
    Console(#[from] std::io::Error),

    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

pub type Result<T> = std::result::Result<T, Error>;
