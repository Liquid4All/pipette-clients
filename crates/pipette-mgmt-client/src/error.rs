use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("invalid request URL `{url}`: {source}")]
    InvalidUrl {
        url: String,
        #[source]
        source: url::ParseError,
    },

    #[error("failed to format auth timestamp: {0}")]
    TimestampFormat(#[source] time::error::Format),

    /// A request timestamp this crate formatted that `Timestamp` will not
    /// accept. Unreachable while RFC 3339 stays a subset of what it parses;
    /// reported rather than asserted, per the panic ban.
    #[error("derived request timestamp `{derived}` is not a usable timestamp")]
    UnusableTimestamp { derived: String },

    #[error(transparent)]
    InvalidPrivateKey(#[from] crate::types::PrivateKeyError),

    /// Only reachable from key derivation over values this crate already
    /// validated, so it means a bug here, not bad caller input.
    #[error(transparent)]
    InvalidPublicKey(#[from] crate::types::PublicKeyError),

    /// A key id derived from a valid public key that `UserKeyId` will not
    /// accept. Unreachable while the prefix and the digest's alphabet agree
    /// with that type; reported rather than asserted, per the panic ban.
    #[error("derived key id `{derived}` is not a usable user key id")]
    UnusableKeyId { derived: String },

    #[error("failed to read OS entropy for keypair generation: {0}")]
    OsEntropy(getrandom::Error),

    /// The holder of the key (an ssh-agent, a vault) would not sign. Its own
    /// message is carried because "locked" and "prompt dismissed" are
    /// different instructions.
    #[error("the holder of this account's key would not sign the request: {source}")]
    Signer {
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    /// A path parameter the URL would resolve away, sending the request to
    /// another route; percent-encoding it does not help, since the URL parser
    /// decodes `%2E` the same way.
    #[error(
        "`{segment}` cannot be a path parameter: the URL would resolve it away and reach another route"
    )]
    DotSegment { segment: String },

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

    /// The progress stream of a call that files a plan ended with no `result`
    /// or `error` line. The server can still complete the request, so the
    /// caller cannot know if it took effect.
    #[error("the progress stream for {method} {url} ended before its last line ({cause})")]
    StreamEnded {
        method: String,
        url: String,
        cause: String,
    },

    #[error("failed to parse JSON response for {method} {url}: {source}{body}")]
    DecodeJson {
        method: String,
        url: String,
        /// An excerpt of the body, with its own separator so an empty one adds
        /// nothing. It gives serde's line and column something to point into.
        body: String,
        #[source]
        source: serde_json::Error,
    },

    #[error("failed to build HTTP client: {0}")]
    BuildClient(String),
}

impl Error {
    /// The HTTP status when this is an [`Error::HttpStatus`], else `None`.
    pub fn http_status(&self) -> Option<u16> {
        if let Self::HttpStatus { status, .. } = self {
            Some(*status)
        } else {
            None
        }
    }

    /// True for a transport error or a `5xx`, which a caller should retry
    /// rather than treat as definitive.
    pub fn is_transient(&self) -> bool {
        match self {
            Self::Transport { .. } => true,
            Self::HttpStatus { status, .. } => *status >= 500,
            Self::InvalidUrl { .. }
            | Self::TimestampFormat(_)
            | Self::UnusableTimestamp { .. }
            | Self::InvalidPrivateKey(_)
            | Self::InvalidPublicKey(_)
            | Self::UnusableKeyId { .. }
            | Self::OsEntropy(_)
            | Self::DotSegment { .. }
            | Self::EncodeJson(_)
            | Self::DecodeJson { .. }
            // The request can have taken effect, and a retry can file the
            // plan a second time.
            | Self::StreamEnded { .. }
            // A refusal to sign is a person's decision or a locked vault, and
            // retrying either just asks again.
            | Self::Signer { .. }
            | Self::BuildClient(_) => false,
        }
    }
}

pub type Result<T> = std::result::Result<T, Error>;
