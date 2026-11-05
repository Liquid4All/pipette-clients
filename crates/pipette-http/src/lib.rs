//! Workspace HTTPS client policy and the process-shared [`HttpClient`].
//!
//! Every request but `pipette-mgmt-client`'s goes out through [`send`];
//! `clippy.toml` enforces it. TLS is set explicitly (`use_preconfigured_tls`
//! with webpki roots): `rustls-no-provider` requires it, and the platform
//! verifier would need a JavaVM on Android.
//!
//! Build once at the application edge and pass `&HttpClient` down.

use std::io::Read;
use std::sync::Once;
use std::time::Duration;

use reqwest::blocking::{RequestBuilder, Response};
use reqwest::header::{ACCEPT, HeaderMap, RETRY_AFTER};
use reqwest::{Method, StatusCode};
use serde::de::DeserializeOwned;
use serde_json::Value;

pub mod range;
pub mod retry;

use retry::Disposition;

/// Client construction and request failures.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("failed to install a rustls crypto provider")]
    CryptoProviderInstall,
    #[error("failed to build HTTP client: {0}")]
    ClientBuild(String),
    /// No response: building, connecting or sending the request failed.
    #[error(transparent)]
    Request(reqwest::Error),
    /// No response before the client's timeout. `after` is how long the
    /// request ran, which is that limit: reqwest's error does not name it.
    #[error("timed out after {after:.0?}")]
    TimedOut {
        after: Duration,
        source: reqwest::Error,
    },
    /// A non-2xx response, with the head of the body the server sent.
    /// `retry_after` is the server's `Retry-After`, when it sent one in
    /// seconds.
    #[error("HTTP {status}{}", body_suffix(.body))]
    Status {
        status: StatusCode,
        body: String,
        retry_after: Option<Duration>,
    },
    #[error("failed to parse JSON response: {0}")]
    Json(String),
}

impl Error {
    /// Whether a retry can succeed where this request failed: no response
    /// came back, or the server answered `5xx` or `429`. A request that could
    /// not be built, or that redirected too often, fails the same way again.
    pub fn disposition(&self) -> Disposition {
        match self {
            Self::Request(e) if e.is_builder() || e.is_redirect() => Disposition::Permanent,
            Self::Request(_) | Self::TimedOut { .. } => Disposition::TRANSIENT,
            Self::Status {
                status,
                retry_after,
                ..
            } if status.is_server_error() || *status == StatusCode::TOO_MANY_REQUESTS => {
                Disposition::Transient {
                    retry_after: *retry_after,
                }
            }
            Self::Status { .. }
            | Self::CryptoProviderInstall
            | Self::ClientBuild(_)
            | Self::Json(_) => Disposition::Permanent,
        }
    }
}

fn body_suffix(body: &str) -> String {
    if body.is_empty() {
        String::new()
    } else {
        format!(": {body}")
    }
}

pub type Result<T> = std::result::Result<T, Error>;

static INIT: Once = Once::new();

pub const DEFAULT_CONNECT_TIMEOUT: Duration = Duration::from_secs(30);

pub const DEFAULT_POOL_IDLE_TIMEOUT: Duration = Duration::from_secs(90);

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
fn tls_client_config() -> Result<rustls::ClientConfig> {
    ensure_default_crypto_provider()?;
    Ok(rustls::ClientConfig::builder()
        .with_root_certificates(rustls::RootCertStore::from_iter(
            webpki_roots::TLS_SERVER_ROOTS.iter().cloned(),
        ))
        .with_no_client_auth())
}

/// Clone is cheap; clones share the pools.
///
/// Two pools with identical TLS, UA and timeouts, differing only in redirect
/// policy: [`Self::client`] follows redirects, which downloads need because
/// registries redirect to a CDN; [`Self::client_no_redirects`] refuses them.
#[derive(Clone, Debug)]
pub struct HttpClient {
    client: reqwest::blocking::Client,
    no_redirects: reqwest::blocking::Client,
}

impl HttpClient {
    /// Start a builder, preconfigured-TLS by default.
    pub fn builder(user_agent: impl Into<String>) -> HttpClientBuilder {
        HttpClientBuilder {
            user_agent: user_agent.into(),
            connect_timeout: DEFAULT_CONNECT_TIMEOUT,
            request_timeout: None,
            tls: TlsMode::PreconfiguredWebpki,
            no_proxy: false,
        }
    }

    /// Process default: preconfigured TLS, default connect timeout, no request timeout.
    pub fn new(user_agent: impl Into<String>) -> Result<Self> {
        Self::builder(user_agent).build()
    }

    /// How long to wait for the TCP connect on a streaming call. The server
    /// is on loopback and readiness-gated before the first request, so this
    /// only ever catches one that died between the gate and the call.
    const STREAM_CONNECT_TIMEOUT: Duration = Duration::from_secs(60);

    /// A client for a long-lived SSE stream from a loopback server: a connect
    /// timeout only, and no proxy, which can only misroute `127.0.0.1`.
    ///
    /// Returns the client with the bound its stream is read under; see
    /// [`StreamClient`] for why the two travel together.
    pub fn streaming(
        user_agent: impl Into<String>,
        idle_timeout: Duration,
    ) -> Result<StreamClient> {
        Ok(StreamClient {
            client: Self::builder(user_agent)
                .preconfigured_tls()
                .no_proxy()
                .connect_timeout(Self::STREAM_CONNECT_TIMEOUT)
                .no_request_timeout()
                .build()?
                .client()
                .clone(),
            idle_timeout,
        })
    }

    /// A client for a server on `127.0.0.1`, where a proxy can only misroute.
    /// Build it outside any timed span: a build spawns and a drop joins the
    /// pools' runtime threads.
    pub fn loopback(user_agent: impl Into<String>, request_timeout: Duration) -> Result<Self> {
        Self::builder(user_agent)
            .preconfigured_tls()
            .no_proxy()
            .request_timeout(request_timeout)
            .build()
    }

    pub fn with_request_timeout(user_agent: impl Into<String>, timeout: Duration) -> Result<Self> {
        Self::builder(user_agent)
            .preconfigured_tls()
            .request_timeout(timeout)
            .build()
    }

    /// The redirect-following pool.
    pub fn client(&self) -> &reqwest::blocking::Client {
        &self.client
    }

    /// The pool that refuses redirects, surfacing a `3xx` as the response.
    ///
    /// For requests whose signature covers the request target: following a
    /// redirect would present a signature over the old target and get a `401`,
    /// and would forward the signing headers (not on reqwest's sensitive-header
    /// list) to the redirect host.
    pub fn client_no_redirects(&self) -> &reqwest::blocking::Client {
        &self.no_redirects
    }

    /// One-shot JSON request on this client (`Accept: application/json`),
    /// sent through [`send`].
    pub fn json_request<T: DeserializeOwned>(
        &self,
        method: Method,
        url: &str,
        headers: Option<HeaderMap>,
        body: Option<Value>,
    ) -> Result<T> {
        let mut request = self
            .client
            .request(method.clone(), url)
            .header(ACCEPT, "application/json");
        if let Some(headers) = headers {
            request = request.headers(headers);
        }
        if let Some(body) = body {
            request = request.json(&body);
        }
        // Method, target and outcome at debug: a stalled or failing exchange
        // is otherwise invisible below the caller's error message.
        tracing::debug!(%method, %url, "sending request");
        let started = std::time::Instant::now();
        let response = send(request).inspect_err(|e| {
            tracing::debug!(
                %method,
                %url,
                elapsed = ?started.elapsed(),
                error = %e,
                "request failed"
            );
        })?;
        tracing::debug!(
            %method,
            %url,
            status = response.status().as_u16(),
            elapsed = ?started.elapsed(),
            "response"
        );
        response
            .json()
            .map_err(|e| Error::Json(format!("{e} ({url})")))
    }
}

/// The most of a refused request's body [`send`] reads: room for any server's
/// reason, not for a proxy's error page.
const REFUSAL_BODY_LIMIT: usize = 4096;

/// Send `request` and return the response unread, for the caller to stream.
/// A non-2xx is [`Error::Status`] with the head of the body, which says why;
/// reqwest's `error_for_status` drops it.
#[expect(clippy::disallowed_methods, reason = "this is the gate")]
pub fn send(request: RequestBuilder) -> Result<Response> {
    let started = std::time::Instant::now();
    let response = request.send().map_err(|source| {
        if source.is_timeout() {
            Error::TimedOut {
                after: started.elapsed(),
                source,
            }
        } else {
            Error::Request(source)
        }
    })?;
    let status = response.status();
    if status.is_success() {
        return Ok(response);
    }
    Err(Error::Status {
        status,
        retry_after: retry_after(&response),
        body: refusal_body(response),
    })
}

/// Whether `cause`, one link of an error chain, is a request that ran out of
/// time, sending it or reading its body. A body read reaches its reader as an
/// `io::Error` around reqwest's, whose own `source` skips the reqwest error.
pub fn is_timeout(cause: &(dyn std::error::Error + 'static)) -> bool {
    let request = cause
        .downcast_ref::<Error>()
        .and_then(|err| match err {
            Error::Request(err) | Error::TimedOut { source: err, .. } => Some(err),
            Error::CryptoProviderInstall
            | Error::ClientBuild(_)
            | Error::Status { .. }
            | Error::Json(_) => None,
        })
        .or_else(|| cause.downcast_ref::<reqwest::Error>())
        .or_else(|| {
            cause
                .downcast_ref::<std::io::Error>()?
                .get_ref()?
                .downcast_ref::<reqwest::Error>()
        });
    request.is_some_and(reqwest::Error::is_timeout)
}

/// The `Retry-After` a refusal carries, in its delta-seconds form. An
/// HTTP-date is not read: the retry then waits its own schedule.
fn retry_after(response: &Response) -> Option<Duration> {
    response
        .headers()
        .get(RETRY_AFTER)?
        .to_str()
        .ok()?
        .trim()
        .parse()
        .ok()
        .map(Duration::from_secs)
}

fn refusal_body(response: Response) -> String {
    let mut bytes = Vec::new();
    // A read that fails partway keeps what arrived, and the reason comes first.
    let _ = response
        .take(REFUSAL_BODY_LIMIT as u64 + 1)
        .read_to_end(&mut bytes);
    let text = String::from_utf8_lossy(&bytes);
    let text = text.trim();
    if bytes.len() <= REFUSAL_BODY_LIMIT {
        return text.to_owned();
    }
    let head = text
        .get(..text.floor_char_boundary(REFUSAL_BODY_LIMIT))
        .unwrap_or(text);
    format!("{}...", head.trim_end())
}

/// A streaming client and the bound its stream is read under, as one value so
/// a caller that has the client to send with also has the bound to read with.
#[derive(Debug, Clone)]
pub struct StreamClient {
    client: reqwest::blocking::Client,
    idle_timeout: Duration,
}

impl StreamClient {
    pub fn http(&self) -> &reqwest::blocking::Client {
        &self.client
    }

    /// How long the reader may wait between events before failing the stream.
    pub fn idle_timeout(&self) -> Duration {
        self.idle_timeout
    }
}

#[derive(Debug, Clone)]
enum TlsMode {
    PreconfiguredWebpki,
}

#[derive(Debug, Clone)]
pub struct HttpClientBuilder {
    user_agent: String,
    connect_timeout: Duration,
    request_timeout: Option<Duration>,
    tls: TlsMode,
    no_proxy: bool,
}

impl HttpClientBuilder {
    /// TCP connect timeout (default [`DEFAULT_CONNECT_TIMEOUT`]).
    pub fn connect_timeout(mut self, timeout: Duration) -> Self {
        self.connect_timeout = timeout;
        self
    }

    /// Bounds the wait for the response headers, then each read of the body;
    /// not an overall deadline. Default is none (downloads, long streams).
    pub fn request_timeout(mut self, timeout: Duration) -> Self {
        self.request_timeout = Some(timeout);
        self
    }

    /// Clear any request timeout (explicit; this is already the default).
    pub fn no_request_timeout(mut self) -> Self {
        self.request_timeout = None;
        self
    }

    /// Ignore `HTTP_PROXY`/`HTTPS_PROXY`/`ALL_PROXY` (and `NO_PROXY`) from the
    /// environment. For clients pinned to loopback servers the proxy can only
    /// misroute: CI runners set proxy vars without a localhost bypass, which
    /// turns every 127.0.0.1 poll into a hang. Remote clients must not use this.
    pub fn no_proxy(mut self) -> Self {
        self.no_proxy = true;
        self
    }

    /// Bundled webpki roots via `use_preconfigured_tls`: the default and only
    /// mode, kept as a method so a call site can name its TLS policy.
    pub fn preconfigured_tls(mut self) -> Self {
        self.tls = TlsMode::PreconfiguredWebpki;
        self
    }

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
                // Load-bearing on Android: the platform verifier needs a JavaVM.
                builder.use_preconfigured_tls(tls_client_config()?)
            }
        };

        builder = builder.timeout(self.request_timeout);

        if self.no_proxy {
            builder = builder.no_proxy();
        }

        builder
            .build()
            .map_err(|e| Error::ClientBuild(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use httpmock::MockServer;

    use super::*;

    fn loopback() -> Result<HttpClient> {
        HttpClient::builder("pipette-test/0").no_proxy().build()
    }

    #[test]
    fn a_refusal_keeps_the_servers_body() -> anyhow::Result<()> {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.path("/v1/completions");
            then.status(400)
                .body("{\"error\":\"prompt is too long\"}\n");
        });
        let http = loopback()?;
        let url = server.url("/v1/completions");
        let errors = [
            send(http.client().post(&url)).err(),
            http.json_request::<Value>(Method::POST, &url, None, None)
                .err(),
        ];
        let is_the_refusal = |error: &Option<Error>| {
            matches!(error, Some(Error::Status { status, body, .. })
                if *status == StatusCode::BAD_REQUEST
                    && body == r#"{"error":"prompt is too long"}"#)
        };
        assert!(errors.iter().all(is_the_refusal), "{errors:?}");
        Ok(())
    }

    #[test]
    fn a_refusal_quotes_its_body_only_when_it_has_one() {
        let message = |status, body: &str| {
            Error::Status {
                status,
                body: body.to_owned(),
                retry_after: None,
            }
            .to_string()
        };
        assert_eq!(
            message(StatusCode::BAD_REQUEST, "{}"),
            "HTTP 400 Bad Request: {}"
        );
        assert_eq!(
            message(StatusCode::SERVICE_UNAVAILABLE, ""),
            "HTTP 503 Service Unavailable"
        );
    }

    #[test]
    fn a_long_refusal_keeps_only_its_head() -> anyhow::Result<()> {
        let server = MockServer::start();
        let http = loopback()?;
        let x = |n| "x".repeat(n);
        let cases = [
            (x(REFUSAL_BODY_LIMIT), x(REFUSAL_BODY_LIMIT)),
            (
                x(REFUSAL_BODY_LIMIT + 1),
                format!("{}...", x(REFUSAL_BODY_LIMIT)),
            ),
            // `é` straddles the limit, so the cut lands before it.
            (
                format!("{}é", x(REFUSAL_BODY_LIMIT - 1)),
                format!("{}...", x(REFUSAL_BODY_LIMIT - 1)),
            ),
        ];
        cases.iter().enumerate().try_for_each(|(i, (sent, kept))| {
            let path = format!("/{i}");
            server.mock(|when, then| {
                when.path(path.as_str());
                then.status(502).body(sent);
            });
            match send(http.client().get(server.url(&path))) {
                Err(Error::Status { body, .. }) => {
                    anyhow::ensure!(&body == kept, "case {i}: kept {} bytes", body.len());
                    Ok(())
                }
                other => anyhow::bail!("case {i}: expected a refusal, got {other:?}"),
            }
        })
    }

    #[test]
    fn a_refused_connection_keeps_the_transport_error() -> anyhow::Result<()> {
        let port = std::net::TcpListener::bind("127.0.0.1:0")?
            .local_addr()?
            .port();
        let error = send(
            loopback()?
                .client()
                .get(format!("http://127.0.0.1:{port}/")),
        )
        .err();
        assert!(
            matches!(&error, Some(Error::Request(e)) if e.is_connect()),
            "{error:?}"
        );
        Ok(())
    }

    /// A refusal is retried only when the server could answer differently
    /// later, and keeps the wait the server asked for.
    #[rstest::rstest]
    #[case::not_found(404, None, Disposition::Permanent)]
    #[case::forbidden(403, Some("5"), Disposition::Permanent)]
    #[case::unavailable(503, None, Disposition::TRANSIENT)]
    #[case::unavailable_with_a_wait(503, Some("3"), Disposition::Transient {
        retry_after: Some(Duration::from_secs(3)),
    })]
    #[case::rate_limited(429, Some(" 7 "), Disposition::Transient {
        retry_after: Some(Duration::from_secs(7)),
    })]
    #[case::rate_limited_until_a_date(
        429,
        Some("Wed, 21 Oct 2026 07:28:00 GMT"),
        Disposition::TRANSIENT
    )]
    fn a_refusal_has_a_disposition(
        #[case] status: u16,
        #[case] retry_after: Option<&str>,
        #[case] expected: Disposition,
    ) -> anyhow::Result<()> {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.path("/artifact");
            let then = then.status(status);
            if let Some(retry_after) = retry_after {
                then.header("retry-after", retry_after);
            }
        });
        let error = send(loopback()?.client().get(server.url("/artifact")))
            .err()
            .ok_or_else(|| anyhow::anyhow!("a {status} was not refused"))?;
        assert_eq!(error.disposition(), expected, "{error:?}");
        Ok(())
    }

    #[test]
    fn a_refused_connection_is_transient() -> anyhow::Result<()> {
        let port = std::net::TcpListener::bind("127.0.0.1:0")?
            .local_addr()?
            .port();
        let error = send(
            loopback()?
                .client()
                .get(format!("http://127.0.0.1:{port}/")),
        )
        .err()
        .ok_or_else(|| anyhow::anyhow!("a closed port answered"))?;
        assert_eq!(error.disposition(), Disposition::TRANSIENT, "{error:?}");
        Ok(())
    }

    /// A late body reaches its reader as an `io::Error`; only a late head names the limit.
    #[rstest::rstest]
    #[case::no_head(Some(b"".as_slice()), true, true)]
    #[case::stalled_body(
        Some(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n".as_slice()),
        true,
        false
    )]
    #[case::refused(None, false, false)]
    fn a_timed_out_request_is_found_in_the_chain(
        #[case] head: Option<&'static [u8]>,
        #[case] timed_out: bool,
        #[case] names_the_limit: bool,
    ) -> anyhow::Result<()> {
        use std::io::Write;

        let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
        let url = format!("http://{}/completion", listener.local_addr()?);
        let peer = match head {
            None => {
                drop(listener);
                None
            }
            Some(head) => Some(std::thread::spawn(move || -> std::io::Result<()> {
                let (mut stream, _) = listener.accept()?;
                let _ = stream.read(&mut [0; 4096]);
                stream.write_all(head)?;
                std::thread::sleep(Duration::from_secs(1));
                Ok(())
            })),
        };
        // Windows refuses a connect to a closed port only after its SYN retries
        // (about 2 s), so the refused case waits past them to see the refusal.
        let limit = match peer {
            None => Duration::from_secs(10),
            Some(_) => Duration::from_millis(200),
        };
        let http = HttpClient::builder("pipette-test/0")
            .no_proxy()
            .request_timeout(limit)
            .build()?;
        let err = send(http.client().get(&url))
            .map_err(anyhow::Error::from)
            .and_then(|mut response| {
                let mut body = String::new();
                response.read_to_string(&mut body)?;
                Ok(body)
            })
            .err()
            .ok_or_else(|| anyhow::anyhow!("the request succeeded"))?
            .context("POST /completion");
        if let Some(peer) = peer {
            let _ = peer.join();
        }
        assert_eq!(err.chain().any(is_timeout), timed_out, "{err:#}");
        assert_eq!(
            format!("{err:#}").contains("timed out after "),
            names_the_limit,
            "{err:#}"
        );
        Ok(())
    }

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
