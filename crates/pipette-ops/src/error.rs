//! Crate-level error for `pipette-ops`.
//!
//! `Error` is the single typed error the shared runtime layer returns. The
//! management-server vocabulary lives in the client's own error type; what
//! remains here is what every consumer can hit. Anything that only needs to
//! propagate flows through [`Error::Other`].

use std::path::PathBuf;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("I/O error at {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("failed to parse {path}: {source}")]
    ParseJson {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },

    #[error("failed to serialize registration JSON: {0}")]
    SerializeJson(#[source] serde_json::Error),

    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

pub type Result<T> = std::result::Result<T, Error>;
