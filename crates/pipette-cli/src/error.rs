//! Crate-level error for the client.
//!
//! Carries the vocabulary of talking to the management server — registration,
//! identity files, catalog conversion. Failures from the shared runtime layer
//! arrive through [`Error::Ops`]; anything that only needs to propagate flows
//! through [`Error::Other`].

use std::path::PathBuf;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("no local registration found at {0}")]
    RegistrationMissing(PathBuf),

    #[error("no local private key found")]
    PrivateKeyMissing,

    #[error("identity already exists ({0}); run `auth reset` before re-registering")]
    IdentityExists(String),

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

    #[error("set-device requires at least one of --device-name, --device-form-factor, or --client-details")]
    SetDeviceEmpty,

    #[error(
        "{unsubmitted} unsubmitted result(s) would be lost; run `auth reset --force` \
             to delete them along with the identity, or `sync` them first"
    )]
    ResetNeedsForce { unsubmitted: usize },

    /// A loose upstream benchmark didn't convert to a strict, known
    /// [`pipette_plan_types::benchmark::BenchmarkDefinition`] (unknown type or
    /// missing/mistyped required parameter). Callers skip the benchmark rather
    /// than abort.
    #[error("benchmark '{benchmark_id}' is not a known, valid definition: {source}")]
    BenchmarkConversion {
        benchmark_id: String,
        #[source]
        source: serde_json::Error,
    },

    #[error(transparent)]
    Mgmt(#[from] pipette_mgmt_client::error::Error),

    #[error(transparent)]
    Ops(#[from] pipette_ops::Error),

    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

pub type Result<T> = std::result::Result<T, Error>;
