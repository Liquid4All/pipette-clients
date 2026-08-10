use thiserror::Error;

#[derive(Debug, Error)]
pub enum PipetteError {
    #[error("model load error: {msg}")]
    ModelLoad { msg: String },
    #[error("out of memory: {msg}")]
    OutOfMemory { msg: String },
    #[error("tokenize error: {msg}")]
    Tokenize { msg: String },
    #[error("inference error: {msg}")]
    Inference { msg: String },
    #[error("benchmark error: {msg}")]
    Benchmark { msg: String },
    #[error("network error: {msg}")]
    Network { msg: String },
    #[error("io error: {msg}")]
    Io { msg: String },
    #[error("json error: {msg}")]
    Json { msg: String },
    #[error("auth error: {msg}")]
    Auth { msg: String },
    #[error("cancelled: {msg}")]
    Cancelled { msg: String },
    #[error("readiness error: {msg}")]
    Readiness { msg: String },
}

impl From<std::io::Error> for PipetteError {
    fn from(err: std::io::Error) -> Self {
        Self::Io {
            msg: err.to_string(),
        }
    }
}

impl From<serde_json::Error> for PipetteError {
    fn from(err: serde_json::Error) -> Self {
        Self::Json {
            msg: err.to_string(),
        }
    }
}
