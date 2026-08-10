//! Workspace HTTPS client policy and the process-shared [`HttpClient`].
//!
//! All blocking HTTP goes through [`HttpClient::builder`] / [`HttpClient::new`].
//! There is no public reqwest builder. TLS is configured explicitly on the
//! builder (`use_preconfigured_tls` + webpki roots) — required under
//! `rustls-no-provider`, and safe on Android where the platform verifier needs
//! a JavaVM.
//!
//! Build once at the application edge and pass `&HttpClient` down.

use std::sync::Once;
use std::time::Duration;

use reqwest::header::{HeaderMap, ACCEPT};
use reqwest::Method;
use serde::de::DeserializeOwned;
use serde_json::Value;

/// Why TLS / client setup failed.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("failed to install a rustls crypto provider")]
    CryptoProviderInstall,
    /// `Client::build` failed after TLS was configured.
    #[error("failed to build HTTP client: {0}")]
    ClientBuild(String),
    /// Request send / status / transport failure.
    #[error("HTTP request failed: {0}")]
    Request(String),
    /// Response body was not valid JSON for the expected type.
    #[error("failed to parse JSON response: {0}")]
    Json(String),
}

pub type Result<T> = std::result::Result<T, Error>;

static INIT: Once = Once::new();

/// Default TCP connect timeout.
pub const DEFAULT_CONNECT_TIMEOUT: Duration = Duration::from_secs(30);

/// Default idle connection pool lifetime.
pub const DEFAULT_POOL_IDLE_TIMEOUT: Duration = Duration::from_secs(90);

/// Ensure a rustls default crypto provider is installed for this process.
fn ensure_default_crypto_provider() -> Result<()> {
    INIT.call_once(|| {
        if rustls::crypto::CryptoProvider::get_default().is_none() {
            let _ = rustls::crypto::ring::default_provider().install_default();
        }
    });
    if rustls::crypto::CryptoProvider::get_default().is_some() {
        Ok(())
    } else {
        Err(Error::CryptoProviderInstall)
    }
}

/// A rustls `ClientConfig` rooted in the bundled webpki Mozilla roots.
/// Used by [`HttpClientBuilder::preconfigured_tls`].
fn tls_client_config() -> Result<rustls::ClientConfig> {
    ensure_default_crypto_provider()?;
    Ok(rustls::ClientConfig::builder()
        .with_root_certificates(rustls::RootCertStore::from_iter(
            webpki_roots::TLS_SERVER_ROOTS.iter().cloned(),
        ))
        .with_no_client_auth())
}

/// Process-shared blocking reqwest client.
///
/// Clone is cheap (internal `Arc`); clones share the same pool.
///
/// Carries two pools with identical TLS, UA, and timeouts, differing only in
/// redirect policy. [`Self::client`] follows redirects — artifact and model
/// downloads rely on it, since registries routinely redirect to a CDN.
/// [`Self::client_no_redirects`] refuses them, for callers whose request
/// signature covers the request target and so cannot survive one.
#[derive(Clone, Debug)]
pub struct HttpClient {
    client: reqwest::blocking::Client,
    no_redirects: reqwest::blocking::Client,
}

impl HttpClient {
    /// Start a builder. TLS defaults are applied in [`HttpClientBuilder::build`]
    /// unless overridden.
    pub fn builder(user_agent: impl Into<String>) -> HttpClientBuilder {
        HttpClientBuilder {
            user_agent: user_agent.into(),
            connect_timeout: DEFAULT_CONNECT_TIMEOUT,
            request_timeout: None,
            tls: TlsMode::PreconfiguredWebpki,
        }
    }

    /// Process default: preconfigured TLS, default connect timeout, no body timeout.
    pub fn new(user_agent: impl Into<String>) -> Result<Self> {
        Self::builder(user_agent).build()
    }

    /// Preconfigured TLS client with an overall request/body timeout.
    pub fn with_request_timeout(user_agent: impl Into<String>, timeout: Duration) -> Result<Self> {
        Self::builder(user_agent)
            .preconfigured_tls()
            .request_timeout(timeout)
            .build()
    }

    /// Convenience: build [`Self::with_request_timeout`] and return a cloned
    /// blocking reqwest handle (same pool as the parent [`HttpClient`]).
    pub fn blocking_with_timeout(
        user_agent: impl Into<String>,
        timeout: Duration,
    ) -> Result<reqwest::blocking::Client> {
        Ok(Self::with_request_timeout(user_agent, timeout)?
            .client()
            .clone())
    }

    /// Borrow the shared blocking client (cloning it is cheap — internal `Arc`).
    pub fn client(&self) -> &reqwest::blocking::Client {
        &self.client
    }

    /// Borrow the shared blocking client that refuses redirects, surfacing a
    /// `3xx` as the response instead of following it.
    ///
    /// For requests whose signature covers the request target: following a
    /// redirect would present a signature over the pre-redirect target and get
    /// a `401`, and would forward the signing headers — which are not on
    /// reqwest's sensitive-header list — to the redirect host.
    pub fn client_no_redirects(&self) -> &reqwest::blocking::Client {
        &self.no_redirects
    }

    /// One-shot JSON request on this client (`Accept: application/json`).
    ///
    /// Non-success responses include the status and response body in
    /// [`Error::Request`] so callers can surface server diagnostics.
    pub fn json_request<T: DeserializeOwned>(
        &self,
        method: Method,
        url: &str,
        headers: Option<HeaderMap>,
        body: Option<Value>,
    ) -> Result<T> {
        let mut request = self
            .client
            .request(method, url)
            .header(ACCEPT, "application/json");
        if let Some(headers) = headers {
            request = request.headers(headers);
        }
        if let Some(body) = body {
            request = request.json(&body);
        }
        let response = request
            .send()
            .map_err(|e| Error::Request(format!("{e} ({url})")))?;
        let status = response.status();
        if !status.is_success() {
            let body = response.text().unwrap_or_default();
            return Err(Error::Request(format!("{status} ({url}): {body}")));
        }
        response
            .json()
            .map_err(|e| Error::Json(format!("{e} ({url})")))
    }
}

/// How TLS is configured on the reqwest client.
#[derive(Debug, Clone)]
enum TlsMode {
    /// `use_preconfigured_tls(tls_client_config())` — workspace default.
    PreconfiguredWebpki,
}

/// Builder for [`HttpClient`]. TLS, timeouts, and UA are set here explicitly.
#[derive(Debug, Clone)]
pub struct HttpClientBuilder {
    user_agent: String,
    connect_timeout: Duration,
    request_timeout: Option<Duration>,
    tls: TlsMode,
}

impl HttpClientBuilder {
    /// TCP connect timeout (default [`DEFAULT_CONNECT_TIMEOUT`]).
    pub fn connect_timeout(mut self, timeout: Duration) -> Self {
        self.connect_timeout = timeout;
        self
    }

    /// Overall request/body timeout. Default is none (downloads, long streams).
    pub fn request_timeout(mut self, timeout: Duration) -> Self {
        self.request_timeout = Some(timeout);
        self
    }

    /// Clear any overall request timeout (explicit; this is already the default).
    pub fn no_request_timeout(mut self) -> Self {
        self.request_timeout = None;
        self
    }

    /// Use bundled webpki roots via `ClientBuilder::use_preconfigured_tls`.
    ///
    /// This is the default and the only supported mode today. Kept as a method
    /// so call sites (and the CLI) can name TLS policy at the construction site.
    pub fn preconfigured_tls(mut self) -> Self {
        self.tls = TlsMode::PreconfiguredWebpki;
        self
    }

    /// Build the client with the selected TLS mode + timeouts + UA.
    pub fn build(self) -> Result<HttpClient> {
        Ok(HttpClient {
            client: self.pool(reqwest::redirect::Policy::default())?,
            no_redirects: self.pool(reqwest::redirect::Policy::none())?,
        })
    }

    /// One pool carrying this builder's TLS, UA, and timeouts under `redirect`.
    ///
    /// `reqwest::ClientBuilder` is not `Clone` and redirect policy is fixed at
    /// build time, so the two pools [`HttpClient`] holds are configured here
    /// rather than derived from one another.
    fn pool(&self, redirect: reqwest::redirect::Policy) -> Result<reqwest::blocking::Client> {
        let mut builder = reqwest::blocking::Client::builder()
            .user_agent(&self.user_agent)
            .connect_timeout(self.connect_timeout)
            .pool_idle_timeout(DEFAULT_POOL_IDLE_TIMEOUT)
            .tcp_nodelay(true)
            .redirect(redirect);

        builder = match self.tls {
            TlsMode::PreconfiguredWebpki => {
                // Load-bearing on Android (no JavaVM for rustls-platform-verifier).
                builder.use_preconfigured_tls(tls_client_config()?)
            }
        };

        // `None` = no overall body timeout (reqwest blocking accepts Option).
        builder = builder.timeout(self.request_timeout);

        builder
            .build()
            .map_err(|e| Error::ClientBuild(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ensure_default_crypto_provider_is_idempotent() -> Result<()> {
        ensure_default_crypto_provider()?;
        ensure_default_crypto_provider()?;
        Ok(())
    }

    #[test]
    fn tls_client_config_builds_with_roots() -> Result<()> {
        tls_client_config()?;
        Ok(())
    }

    #[test]
    fn builder_default_and_timeouts_build() -> Result<()> {
        let _ = HttpClient::new("pipette-test/0")?;
        let _ = HttpClient::builder("pipette-test/0")
            .preconfigured_tls()
            .request_timeout(Duration::from_secs(5))
            .build()?;
        let _ = HttpClient::builder("pipette-test/0")
            .preconfigured_tls()
            .connect_timeout(Duration::from_secs(5))
            .no_request_timeout()
            .build()?;
        Ok(())
    }
}
