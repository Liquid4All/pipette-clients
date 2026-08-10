//! Management-server API client.
//!
//! Holds only the server base URL. The process-shared [`pipette_http::HttpClient`]
//! is passed into every method — this type never owns or stores an HTTP client.

use serde::de::DeserializeOwned;
use serde_json::Value;

use pipette_http::HttpClient;

use crate::{
    auth::{signed_headers, AuthIdentity},
    error::{Error, Result},
    transport::{HttpMethod, HttpRequest, HttpResponse, TransportError},
    types::{
        BatchSubmitResponse, BenchmarkSummary, ClaimedJob, ClientProfile, EntityTag, IfNoneMatch,
        JobResponse, RegisterRequest, RegisterResponse, RemoteBenchmark, SubmitResponse,
        UpdateClientRequest,
    },
};

/// Typed management API. Construct with [`Self::new`]; pass [`HttpClient`] on
/// every call so HTTP ownership stays at the process edge.
#[derive(Clone, Debug)]
pub struct MgmtClient {
    base_url: String,
}

#[derive(Debug, Clone)]
pub enum ConditionalResponse<T> {
    Modified { value: T, etag: Option<EntityTag> },
    NotModified { etag: Option<EntityTag> },
}

impl<T> ConditionalResponse<T> {
    pub fn etag(&self) -> Option<&EntityTag> {
        match self {
            Self::Modified { etag, .. } | Self::NotModified { etag } => etag.as_ref(),
        }
    }
}

impl MgmtClient {
    /// Parse and normalize `base_url`. Does not open a connection.
    pub fn new(base_url: impl Into<String>) -> Result<Self> {
        let base_url = base_url.into();
        let parsed = url::Url::parse(&base_url).map_err(|source| Error::InvalidBaseUrl {
            base_url: base_url.clone(),
            source,
        })?;
        Ok(Self {
            base_url: parsed.to_string().trim_end_matches('/').to_string(),
        })
    }

    pub fn register(
        &self,
        http: &HttpClient,
        request: RegisterRequest,
    ) -> Result<RegisterResponse> {
        self.request(
            http,
            HttpMethod::Post,
            "/clients/register",
            None,
            Some(request),
        )
    }

    pub fn me(&self, http: &HttpClient, auth: &AuthIdentity) -> Result<ClientProfile> {
        self.request(
            http,
            HttpMethod::Get,
            "/clients/me",
            Some(auth),
            Option::<Value>::None,
        )
    }

    /// `PATCH /clients/me` — update the caller's own server-side profile.
    pub fn update_me(
        &self,
        http: &HttpClient,
        auth: &AuthIdentity,
        request: UpdateClientRequest,
    ) -> Result<ClientProfile> {
        self.request(
            http,
            HttpMethod::Patch,
            "/clients/me",
            Some(auth),
            Some(request),
        )
    }

    pub fn list_benchmarks(
        &self,
        http: &HttpClient,
        auth: &AuthIdentity,
    ) -> Result<Vec<BenchmarkSummary>> {
        self.request(
            http,
            HttpMethod::Get,
            "/benchmarks",
            Some(auth),
            Option::<Value>::None,
        )
    }

    pub fn list_benchmarks_conditional(
        &self,
        http: &HttpClient,
        auth: &AuthIdentity,
        if_none_match: Option<&IfNoneMatch>,
    ) -> Result<ConditionalResponse<Vec<BenchmarkSummary>>> {
        self.request_conditional(
            http,
            HttpMethod::Get,
            "/benchmarks",
            Some(auth),
            Option::<Value>::None,
            if_none_match,
        )
    }

    pub fn get_benchmark(
        &self,
        http: &HttpClient,
        auth: &AuthIdentity,
        benchmark_id: &str,
    ) -> Result<RemoteBenchmark> {
        self.request(
            http,
            HttpMethod::Get,
            &format!("/benchmarks/{benchmark_id}"),
            Some(auth),
            Option::<Value>::None,
        )
    }

    pub fn get_benchmark_conditional(
        &self,
        http: &HttpClient,
        auth: &AuthIdentity,
        benchmark_id: &str,
        if_none_match: Option<&IfNoneMatch>,
    ) -> Result<ConditionalResponse<RemoteBenchmark>> {
        self.request_conditional(
            http,
            HttpMethod::Get,
            &format!("/benchmarks/{benchmark_id}"),
            Some(auth),
            Option::<Value>::None,
            if_none_match,
        )
    }

    pub fn submit_result<P: serde::Serialize>(
        &self,
        http: &HttpClient,
        auth: &AuthIdentity,
        payload: P,
    ) -> Result<SubmitResponse> {
        self.request(
            http,
            HttpMethod::Post,
            "/benchmarks",
            Some(auth),
            Some(payload),
        )
    }

    pub fn submit_result_batch<P: serde::Serialize>(
        &self,
        http: &HttpClient,
        auth: &AuthIdentity,
        payloads: Vec<P>,
    ) -> Result<BatchSubmitResponse> {
        self.request(
            http,
            HttpMethod::Post,
            "/benchmarks/batch",
            Some(auth),
            Some(serde_json::json!({ "submissions": payloads })),
        )
    }

    pub fn get_job(
        &self,
        http: &HttpClient,
        auth: &AuthIdentity,
        job_id: &str,
    ) -> Result<JobResponse> {
        self.request(
            http,
            HttpMethod::Get,
            &format!("/jobs/{job_id}"),
            Some(auth),
            Option::<Value>::None,
        )
    }

    /// `POST /plans/claim` — lease the next eligible job, or `None` on `204`.
    pub fn claim(&self, http: &HttpClient, auth: &AuthIdentity) -> Result<Option<ClaimedJob>> {
        let (response, url, method_name) = self.send(
            http,
            HttpMethod::Post,
            "/plans/claim",
            Some(auth),
            None,
            Option::<Value>::None,
        )?;
        match response.status {
            204 => Ok(None),
            status if (200..300).contains(&status) => {
                let job =
                    serde_json::from_slice(&response.body).map_err(|source| Error::DecodeJson {
                        method: method_name,
                        url,
                        source,
                    })?;
                Ok(Some(job))
            }
            _ => Err(self.status_error(response, url, method_name)),
        }
    }

    /// `PUT /plans/{job_id}/heartbeat` — renew the lease.
    pub fn heartbeat(&self, http: &HttpClient, auth: &AuthIdentity, job_id: &str) -> Result<()> {
        self.request_empty(
            http,
            HttpMethod::Put,
            &format!("/plans/{job_id}/heartbeat"),
            Some(auth),
            Option::<Value>::None,
        )
    }

    /// `POST /plans/{job_id}/reclaim` — re-acquire a previously held lease.
    pub fn reclaim(&self, http: &HttpClient, auth: &AuthIdentity, job_id: &str) -> Result<()> {
        self.request_empty(
            http,
            HttpMethod::Post,
            &format!("/plans/{job_id}/reclaim"),
            Some(auth),
            Option::<Value>::None,
        )
    }

    fn request<T: DeserializeOwned, B: serde::Serialize>(
        &self,
        http: &HttpClient,
        method: HttpMethod,
        path: &str,
        auth: Option<&AuthIdentity>,
        body: Option<B>,
    ) -> Result<T> {
        let (response, url, method_name) = self.send(http, method, path, auth, None, body)?;
        self.decode_success(response, url, method_name)
    }

    fn request_conditional<T: DeserializeOwned, B: serde::Serialize>(
        &self,
        http: &HttpClient,
        method: HttpMethod,
        path: &str,
        auth: Option<&AuthIdentity>,
        body: Option<B>,
        if_none_match: Option<&IfNoneMatch>,
    ) -> Result<ConditionalResponse<T>> {
        let (response, url, method_name) =
            self.send(http, method, path, auth, if_none_match, body)?;
        let etag = response
            .header("etag")
            .and_then(EntityTag::from_header_value);
        if response.status == 304 {
            return Ok(ConditionalResponse::NotModified { etag });
        }
        let value = self.decode_success(response, url, method_name)?;
        Ok(ConditionalResponse::Modified { value, etag })
    }

    fn send<B: serde::Serialize>(
        &self,
        http: &HttpClient,
        method: HttpMethod,
        path: &str,
        auth: Option<&AuthIdentity>,
        if_none_match: Option<&IfNoneMatch>,
        body: Option<B>,
    ) -> Result<(HttpResponse, String, String)> {
        let url = format!("{}{}", self.base_url, path);
        let method_name = method.as_str().to_string();
        let request = self.build_request(method, &url, auth, if_none_match, body)?;
        let response = execute(http, request).map_err(|source| Error::Transport {
            method: method_name.clone(),
            url: url.clone(),
            message: source.message,
        })?;
        Ok((response, url, method_name))
    }

    fn build_request<B: serde::Serialize>(
        &self,
        method: HttpMethod,
        url: &str,
        auth: Option<&AuthIdentity>,
        if_none_match: Option<&IfNoneMatch>,
        body: Option<B>,
    ) -> Result<HttpRequest> {
        // User-Agent comes from the shared reqwest client on `HttpClient`.
        let mut headers = vec![("Accept".to_string(), "application/json".to_string())];
        if let Some(auth) = auth {
            headers.extend(signed_headers(
                auth,
                method.as_str(),
                &request_target(url)?,
            )?);
        }
        if let Some(etag) = if_none_match {
            headers.push(("If-None-Match".to_string(), etag.header_value().to_string()));
        }
        let body = match body {
            Some(body) => {
                headers.push(("Content-Type".to_string(), "application/json".to_string()));
                Some(serde_json::to_vec(&body).map_err(Error::EncodeJson)?)
            }
            None => None,
        };
        Ok(HttpRequest {
            method,
            url: url.to_string(),
            headers,
            body,
        })
    }

    fn decode_success<T: DeserializeOwned>(
        &self,
        response: HttpResponse,
        url: String,
        method: String,
    ) -> Result<T> {
        if !(200..300).contains(&response.status) {
            return Err(self.status_error(response, url, method));
        }
        serde_json::from_slice(&response.body).map_err(|source| Error::DecodeJson {
            method,
            url,
            source,
        })
    }

    fn request_empty<B: serde::Serialize>(
        &self,
        http: &HttpClient,
        method: HttpMethod,
        path: &str,
        auth: Option<&AuthIdentity>,
        body: Option<B>,
    ) -> Result<()> {
        let (response, url, method_name) = self.send(http, method, path, auth, None, body)?;
        if !(200..300).contains(&response.status) {
            return Err(self.status_error(response, url, method_name));
        }
        Ok(())
    }

    fn status_error(&self, response: HttpResponse, url: String, method: String) -> Error {
        let body = String::from_utf8_lossy(&response.body);
        Error::HttpStatus {
            method,
            url,
            status: response.status,
            body: if body.is_empty() {
                String::new()
            } else {
                format!(": {body}")
            },
        }
    }
}

/// The request target the server sees: the URL's path plus its query. Parsed
/// out of the assembled URL rather than concatenated from the base URL and the
/// endpoint path, so it carries any path prefix on the base URL and reflects
/// whatever percent-encoding the URL parser applies — the same parse reqwest
/// performs to build the wire request, and the server signs over what it
/// receives.
fn request_target(url: &str) -> Result<String> {
    let parsed = url::Url::parse(url).map_err(|source| Error::InvalidUrl {
        url: url.to_string(),
        source,
    })?;
    Ok(match parsed.query() {
        Some(query) => format!("{}?{}", parsed.path(), query),
        None => parsed.path().to_string(),
    })
}

/// Run one request on the shared blocking client. `User-Agent` is the client's default.
///
/// Runs on the pool that refuses redirects: the `v1` signature covers the
/// request target, so following a redirect would present a signature over the
/// pre-redirect target and earn a `401`, and would forward `X-Signature` to the
/// redirect host. A redirecting deployment surfaces its `3xx` here instead.
fn execute(
    http: &HttpClient,
    request: HttpRequest,
) -> std::result::Result<HttpResponse, TransportError> {
    let method = match request.method {
        HttpMethod::Get => reqwest::Method::GET,
        HttpMethod::Post => reqwest::Method::POST,
        HttpMethod::Put => reqwest::Method::PUT,
        HttpMethod::Patch => reqwest::Method::PATCH,
    };
    let mut builder = http.client_no_redirects().request(method, &request.url);
    for (name, value) in &request.headers {
        builder = builder.header(name, value);
    }
    if let Some(body) = request.body {
        builder = builder.body(body);
    }
    let response = builder
        .send()
        .map_err(|source| TransportError::new(source.to_string()))?;
    let status = response.status().as_u16();
    let headers = response
        .headers()
        .iter()
        .filter_map(|(name, value)| {
            value
                .to_str()
                .ok()
                .map(|value| (name.as_str().to_string(), value.to_string()))
        })
        .collect();
    let body = response
        .bytes()
        .map_err(|source| TransportError::new(source.to_string()))?
        .to_vec();
    Ok(HttpResponse {
        status,
        headers,
        body,
    })
}

#[cfg(test)]
mod tests {
    use ed25519_dalek::{Signature, Verifier, VerifyingKey};
    use httpmock::prelude::*;

    use super::*;
    use crate::{auth::generate_keypair_hex, AuthIdentity};

    fn auth() -> Result<AuthIdentity> {
        let (private_key_hex, _) = generate_keypair_hex()?;
        Ok(AuthIdentity {
            client_id: "ev1_test".into(),
            private_key_hex,
        })
    }

    fn http_ctx() -> Result<HttpClient> {
        HttpClient::new("pipette-test/0").map_err(|e| Error::BuildClient(e.to_string()))
    }

    fn client(server: &MockServer) -> Result<MgmtClient> {
        MgmtClient::new(server.base_url())
    }

    /// The signature must cover the target the *server* sees, so a base URL
    /// carrying a path prefix has to appear in the signed payload — signing the
    /// bare endpoint path would 401 every request against such a deployment.
    #[test]
    fn signed_headers_cover_the_method_and_the_full_request_target() -> Result<()> {
        let (private_key_hex, public_key_hex) = generate_keypair_hex()?;
        let auth = AuthIdentity {
            client_id: "ev1_a3f8".into(),
            private_key_hex,
        };
        let c = MgmtClient::new("https://mgmt.example.com/api/")?;

        // Assembled the way `send` does, so the `/api` prefix asserted below
        // provably comes from the configured base URL rather than a literal.
        let url = format!("{}/clients/me", c.base_url);
        let request = c.build_request(
            HttpMethod::Patch,
            &url,
            Some(&auth),
            None,
            Option::<Value>::None,
        )?;

        let header = |name: &str| {
            request
                .headers
                .iter()
                .find(|(key, _)| key == name)
                .map(|(_, value)| value.clone())
                .ok_or_else(|| Error::BuildClient(format!("missing {name}")))
        };
        let timestamp = header("X-Timestamp")?;
        // Read off the header rather than recomputed: the server verifies the
        // signature against the nonce the request actually carried, so a nonce
        // that never reached the header would verify here and 401 in production.
        let nonce = header("X-Nonce")?;
        assert_eq!(header("X-Client-Id")?, "ev1_a3f8");

        let public: [u8; 32] = hex::decode(&public_key_hex)
            .map_err(Error::DecodePrivateKeyHex)?
            .try_into()
            .map_err(|_| Error::InvalidPrivateKeyLength)?;
        let signature: [u8; 64] = hex::decode(header("X-Signature")?)
            .map_err(Error::DecodePrivateKeyHex)?
            .try_into()
            .map_err(|_| Error::InvalidPrivateKeyLength)?;
        let payload = format!("v1\nPATCH\n/api/clients/me\n{timestamp}\nev1_a3f8\n{nonce}");
        VerifyingKey::from_bytes(&public)
            .map_err(|e| Error::BuildClient(e.to_string()))?
            .verify(payload.as_bytes(), &Signature::from_bytes(&signature))
            .map_err(|e| Error::BuildClient(e.to_string()))?;
        Ok(())
    }

    /// Every signed request carries its own nonce, so the server's replay cache
    /// can reject a repeat. Two requests built from one identity must not reuse
    /// one.
    #[test]
    fn each_signed_request_carries_a_fresh_nonce() -> Result<()> {
        let (private_key_hex, _) = generate_keypair_hex()?;
        let auth = AuthIdentity {
            client_id: "ev1_a3f8".into(),
            private_key_hex,
        };
        let c = MgmtClient::new("https://mgmt.example.com")?;

        let nonce = |request: &HttpRequest| {
            request
                .headers
                .iter()
                .find(|(key, _)| key == "X-Nonce")
                .map(|(_, value)| value.clone())
                .ok_or_else(|| Error::BuildClient("missing X-Nonce".into()))
        };
        let build = || {
            c.build_request(
                HttpMethod::Get,
                "https://mgmt.example.com/clients/me",
                Some(&auth),
                None,
                Option::<Value>::None,
            )
        };

        assert_ne!(nonce(&build()?)?, nonce(&build()?)?);
        Ok(())
    }

    #[test]
    fn request_target_keeps_the_query_string() -> Result<()> {
        assert_eq!(
            request_target("https://mgmt.example.com/clients/me?page=2")?,
            "/clients/me?page=2"
        );
        Ok(())
    }

    #[test]
    fn claim_200_returns_job() -> Result<()> {
        let server = MockServer::start();
        let body = r#"{
            "job_id": "job-1",
            "benchmark_id": "prefill_throughput_256",
            "time_window": "PT5M",
            "spec": {
                "benchmark": "prefill_throughput_256",
                "model": {
                    "type": "gguf_text",
                    "source": "huggingface",
                    "org": "o",
                    "repo_name": "r",
                    "path": "m-Q4_0.gguf"
                },
                "runtime": {
                    "type": "llamacpp_cli_stock_tools",
                    "source": "github_release",
                    "version": "b5000",
                    "flavor": "macos-arm64"
                }
            },
            "future_field": true
        }"#;
        let mock = server.mock(|when, then| {
            when.method(POST).path("/plans/claim");
            then.status(200).body(body);
        });
        let c = client(&server)?;
        let http = http_ctx()?;
        let job = c.claim(&http, &auth()?)?.ok_or_else(|| Error::HttpStatus {
            method: "POST".into(),
            url: "x".into(),
            status: 500,
            body: "missing job".into(),
        })?;
        assert_eq!(job.job_id, "job-1");
        assert_eq!(job.time_window, "PT5M");
        mock.assert();
        Ok(())
    }

    #[test]
    fn claim_204_returns_none() -> Result<()> {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(POST).path("/plans/claim");
            then.status(204);
        });
        let c = client(&server)?;
        let http = http_ctx()?;
        assert!(c.claim(&http, &auth()?)?.is_none());
        Ok(())
    }

    /// The `v1` signature covers the request target, so a followed redirect
    /// would present a signature over the pre-redirect target (a `401`) and
    /// leak `X-Signature` to the redirect host. The `3xx` has to surface as the
    /// response so a redirecting deployment is diagnosable.
    #[test]
    fn a_redirect_surfaces_instead_of_being_followed() -> Result<()> {
        let server = MockServer::start();
        let redirect = server.mock(|when, then| {
            when.method(POST).path("/plans/claim");
            then.status(308).header("Location", "/plans/claim/");
        });
        let followed = server.mock(|when, then| {
            when.method(POST).path("/plans/claim/");
            then.status(200).body("{}");
        });
        let c = client(&server)?;
        let http = http_ctx()?;
        let err = c
            .claim(&http, &auth()?)
            .err()
            .ok_or_else(|| Error::BuildClient("expected error".into()))?;
        assert_eq!(err.http_status(), Some(308));
        redirect.assert();
        followed.assert_hits(0);
        Ok(())
    }

    #[test]
    fn claim_403_is_http_status() -> Result<()> {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(POST).path("/plans/claim");
            then.status(403).body("pending");
        });
        let c = client(&server)?;
        let http = http_ctx()?;
        let err = c
            .claim(&http, &auth()?)
            .err()
            .ok_or_else(|| Error::BuildClient("expected error".into()))?;
        assert_eq!(err.http_status(), Some(403));
        Ok(())
    }

    #[test]
    fn heartbeat_uses_put_and_accepts_empty_200() -> Result<()> {
        let server = MockServer::start();
        let mock = server.mock(|when, then| {
            when.method(PUT).path("/plans/job-1/heartbeat");
            then.status(200);
        });
        let c = client(&server)?;
        let http = http_ctx()?;
        c.heartbeat(&http, &auth()?, "job-1")?;
        mock.assert();
        Ok(())
    }

    #[test]
    fn reclaim_uses_post_and_surfaces_409() -> Result<()> {
        let server = MockServer::start();
        let mock = server.mock(|when, then| {
            when.method(POST).path("/plans/job-1/reclaim");
            then.status(409).body("taken");
        });
        let c = client(&server)?;
        let http = http_ctx()?;
        let err = c
            .reclaim(&http, &auth()?, "job-1")
            .err()
            .ok_or_else(|| Error::BuildClient("expected error".into()))?;
        assert_eq!(err.http_status(), Some(409));
        mock.assert();
        Ok(())
    }

    #[test]
    fn http_status_5xx_is_transient_4xx_is_not() {
        let transient = Error::HttpStatus {
            method: "POST".into(),
            url: "u".into(),
            status: 503,
            body: String::new(),
        };
        let definitive = Error::HttpStatus {
            method: "POST".into(),
            url: "u".into(),
            status: 404,
            body: String::new(),
        };
        assert!(transient.is_transient());
        assert!(!definitive.is_transient());
        assert!(Error::Transport {
            method: "GET".into(),
            url: "u".into(),
            message: "reset".into(),
        }
        .is_transient());
    }
}
