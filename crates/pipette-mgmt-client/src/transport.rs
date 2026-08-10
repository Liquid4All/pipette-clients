//! Wire types for management HTTP (`HttpRequest` / `HttpResponse`) and a
//! leftover [`HttpTransport`] trait (tests / future hosts).
//!
//! Desktop [`crate::MgmtClient`] does **not** store a transport: each method
//! takes [`pipette_http::HttpClient`] and runs the request on that shared
//! blocking client. `User-Agent` is the client's default, not a mgmt header.

/// HTTP verbs used by the management API.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HttpMethod {
    Get,
    Post,
    Put,
    Patch,
}

impl HttpMethod {
    pub fn as_str(&self) -> &'static str {
        match self {
            HttpMethod::Get => "GET",
            HttpMethod::Post => "POST",
            HttpMethod::Put => "PUT",
            HttpMethod::Patch => "PATCH",
        }
    }
}

/// A fully-formed request: URL, headers (including `Accept`, `User-Agent`,
/// the signed auth headers, and any conditional `If-None-Match`), and a
/// serialized JSON body. Carries no reqwest types so the core compiles
/// without an HTTP stack.
#[derive(Debug, Clone)]
pub struct HttpRequest {
    pub method: HttpMethod,
    pub url: String,
    pub headers: Vec<(String, String)>,
    pub body: Option<Vec<u8>>,
}

/// The transport's reply: status code, response headers (the client only
/// reads `ETag`), and the raw body to be JSON-decoded by the client.
#[derive(Debug, Clone)]
pub struct HttpResponse {
    pub status: u16,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

impl HttpResponse {
    /// Case-insensitive header lookup (HTTP header names are
    /// case-insensitive; native hosts may preserve original casing).
    pub(crate) fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(key, _)| key.eq_ignore_ascii_case(name))
            .map(|(_, value)| value.as_str())
    }
}

/// A transport-level failure (connection refused, DNS, TLS, timeout, …).
/// The message is surfaced through [`crate::Error::Transport`].
#[derive(Debug, Clone)]
pub struct TransportError {
    pub message: String,
}

impl TransportError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl std::fmt::Display for TransportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for TransportError {}

/// Performs a single HTTP request on behalf of [`crate::MgmtClient`].
///
/// Implementors handle only the wire exchange — connecting, sending the
/// pre-built request, and returning status/headers/body. They do not
/// touch signing, serialization, or status interpretation; that all
/// stays in the client so every transport behaves identically.
pub trait HttpTransport: Send + Sync {
    fn execute(&self, request: HttpRequest) -> Result<HttpResponse, TransportError>;
}
