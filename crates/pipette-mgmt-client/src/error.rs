use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("invalid base URL `{base_url}`: {source}")]
    InvalidBaseUrl {
        base_url: String,
        #[source]
        source: url::ParseError,
    },

    #[error("invalid request URL `{url}`: {source}")]
    InvalidUrl {
        url: String,
        #[source]
        source: url::ParseError,
    },

    #[error("failed to format auth timestamp: {0}")]
    TimestampFormat(#[source] time::error::Format),

    #[error("failed to decode ed25519 private key hex: {0}")]
    DecodePrivateKeyHex(#[source] hex::FromHexError),

    #[error("invalid ed25519 private key length")]
    InvalidPrivateKeyLength,

    #[error("failed to read OS entropy for keypair generation: {0}")]
    OsEntropy(getrandom::Error),

    #[error("failed to serialize request body: {0}")]
    EncodeJson(#[source] serde_json::Error),

    #[error("request failed for {method} {url}: {message}")]
    Transport {
        method: String,
        url: String,
        message: String,
    },

    #[error("request returned {status} for {method} {url}{body}")]
    HttpStatus {
        method: String,
        url: String,
        status: u16,
        body: String,
    },

    #[error("failed to parse JSON response for {method} {url}: {source}")]
    DecodeJson {
        method: String,
        url: String,
        #[source]
        source: serde_json::Error,
    },

    #[error("failed to build HTTP client: {0}")]
    BuildClient(String),
}

impl Error {
    /// The HTTP status when this is an [`Error::HttpStatus`], else `None`.
    pub fn http_status(&self) -> Option<u16> {
        match self {
            Self::HttpStatus { status, .. } => Some(*status),
            _ => None,
        }
    }

    /// True when the failure is a transport-level error (network/DNS/TLS),
    /// which the planner client should retry rather than treat as definitive.
    pub fn is_transient(&self) -> bool {
        match self {
            Self::Transport { .. } => true,
            Self::HttpStatus { status, .. } => *status >= 500,
            _ => false,
        }
    }
}

pub type Result<T> = std::result::Result<T, Error>;
