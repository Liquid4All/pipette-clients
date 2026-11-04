//! Endpoints a **client** (a device doing the work) calls, signed with its
//! [`ClientIdentity`]: the routes `pipette-mgmt`'s access matrix admits a
//! client on (its `docs/authentication.md` §4).
//!
//! The catalog reads take an *optional* identity: they are served to anyone,
//! but a server that keeps some benchmarks to one principal shows them only to
//! a request signed by it.

use pipette_http::HttpClient;
use pipette_plan_types::{BatchSubmitted, JobAccepted, VaultResourceId};

use crate::{
    auth::ClientIdentity,
    call::{
        ByteStream, ConditionalResponse, ResumeFrom, decode_success, get_json,
        get_json_conditional, get_stream, send, send_empty, send_json, status_error,
    },
    error::Result,
    transport::HttpMethod,
    types::{
        BenchmarkSummary, ClaimedJob, ClientProfile, IfNoneMatch, JobResponse, RegisterRequest,
        RegisterResponse, RemoteBenchmark, ServerUrl, UpdateClientRequest, VaultResource,
    },
};

pub fn register(
    server: &ServerUrl,
    http: &HttpClient,
    request: RegisterRequest,
) -> Result<RegisterResponse> {
    send_json(
        server,
        http,
        HttpMethod::Post,
        "/clients/register",
        None,
        request,
    )
}

pub fn me(server: &ServerUrl, http: &HttpClient, auth: &ClientIdentity) -> Result<ClientProfile> {
    get_json(server, http, "/clients/me", Some(auth))
}

pub fn update_me(
    server: &ServerUrl,
    http: &HttpClient,
    auth: &ClientIdentity,
    request: UpdateClientRequest,
) -> Result<ClientProfile> {
    send_json(
        server,
        http,
        HttpMethod::Patch,
        "/clients/me",
        Some(auth),
        request,
    )
}

/// `GET /benchmarks`: the whole catalog, revalidated against `if_none_match`
/// (`None` is the unconditional request).
pub fn list_benchmarks_conditional(
    server: &ServerUrl,
    http: &HttpClient,
    auth: Option<&ClientIdentity>,
    if_none_match: Option<&IfNoneMatch>,
) -> Result<ConditionalResponse<Vec<BenchmarkSummary>>> {
    get_json_conditional(
        server,
        http,
        "/benchmarks",
        auth.map(|auth| auth as _),
        if_none_match,
    )
}

pub fn get_benchmark(
    server: &ServerUrl,
    http: &HttpClient,
    auth: Option<&ClientIdentity>,
    benchmark_id: &str,
) -> Result<RemoteBenchmark> {
    get_json(
        server,
        http,
        &format!("/benchmarks/{benchmark_id}"),
        auth.map(|auth| auth as _),
    )
}

pub fn get_benchmark_conditional(
    server: &ServerUrl,
    http: &HttpClient,
    auth: Option<&ClientIdentity>,
    benchmark_id: &str,
    if_none_match: Option<&IfNoneMatch>,
) -> Result<ConditionalResponse<RemoteBenchmark>> {
    get_json_conditional(
        server,
        http,
        &format!("/benchmarks/{benchmark_id}"),
        auth.map(|auth| auth as _),
        if_none_match,
    )
}

/// `GET {path}`: a benchmark's corpus as a packed tar, at the route the
/// catalog's `dataset.url` carries. The server owns that spelling, so it is not
/// rebuilt here.
///
/// Returns the bytes unverified: the caller checks the digest against the
/// catalog's pin and submits it either way, since a mismatch is a fact about
/// the run, not a reason to refuse one.
///
/// A `503` (artifact unbuilt, upstream check stale or unreachable) or `502`
/// (upstream no longer backs the pin) is [`crate::Error::is_transient`],
/// which tells a worker the corpus, not the model, stopped the job.
/// `Retry-After` is not surfaced: the client already holds the job, so it
/// fails and is rescheduled rather than slept through.
pub fn get_dataset(
    server: &ServerUrl,
    http: &HttpClient,
    auth: &ClientIdentity,
    path: &str,
) -> Result<Vec<u8>> {
    let (response, url, method_name) = send(server, http, HttpMethod::Get, path, Some(auth))?;
    if !(200..300).contains(&response.status) {
        return Err(status_error(response, url, method_name));
    }
    Ok(response.body)
}

/// [`get_dataset`], with the body left on the wire: an asr_eval corpus is
/// gigabytes, and the caller streams it to disk while hashing. The same
/// refusals, decided before the first byte.
pub fn get_dataset_stream(
    server: &ServerUrl,
    http: &HttpClient,
    auth: &ClientIdentity,
    path: &str,
) -> Result<ByteStream> {
    get_stream(server, http, path, Some(auth), Vec::new())
}

/// `GET /vault/resources/{resource_id}`: a private artifact's pinned `sha256`
/// and size. Read before the bytes so a cached copy can be recognized and a
/// refusal costs no download. `403` means the device's owner does not own the
/// resource, the same `403` [`get_vault_content`] would give.
pub fn get_vault_resource(
    server: &ServerUrl,
    http: &HttpClient,
    auth: &ClientIdentity,
    resource_id: &VaultResourceId,
) -> Result<VaultResource> {
    get_json(
        server,
        http,
        &format!("/vault/resources/{resource_id}"),
        Some(auth),
    )
}

/// `GET /vault/resources/{resource_id}/content`: the artifact's bytes, left on
/// the wire, since a checkpoint is gigabytes.
///
/// Unverified, like [`get_dataset`]: the caller hashes what it wrote against
/// the record's `sha256`, which the `ETag` also carries, quoted. Besides the
/// owner `403`, a `404` can mean the record names an object that is gone.
///
/// With `resume`, the server answers `206` from the offset if `If-Range`
/// matches, `200` with the whole object if it changed, or `416` if the offset
/// is at or past the end. `Range` and `If-Range` are outside the signed
/// payload; a range only narrows bytes the device can already fetch.
pub fn get_vault_content(
    server: &ServerUrl,
    http: &HttpClient,
    auth: &ClientIdentity,
    resource_id: &VaultResourceId,
    resume: Option<&ResumeFrom>,
) -> Result<ByteStream> {
    get_stream(
        server,
        http,
        &format!("/vault/resources/{resource_id}/content"),
        Some(auth),
        resume.map(ResumeFrom::headers).unwrap_or_default(),
    )
}

pub fn submit_result<P: serde::Serialize>(
    server: &ServerUrl,
    http: &HttpClient,
    auth: &ClientIdentity,
    payload: P,
) -> Result<JobAccepted> {
    send_json(
        server,
        http,
        HttpMethod::Post,
        "/benchmarks",
        Some(auth),
        payload,
    )
}

pub fn submit_result_batch<P: serde::Serialize>(
    server: &ServerUrl,
    http: &HttpClient,
    auth: &ClientIdentity,
    payloads: Vec<P>,
) -> Result<BatchSubmitted> {
    #[derive(serde::Serialize)]
    struct BatchSubmitRequest<P> {
        submissions: Vec<P>,
    }

    send_json(
        server,
        http,
        HttpMethod::Post,
        "/benchmarks/batch",
        Some(auth),
        BatchSubmitRequest {
            submissions: payloads,
        },
    )
}

pub fn get_job(
    server: &ServerUrl,
    http: &HttpClient,
    auth: &ClientIdentity,
    job_id: &str,
) -> Result<JobResponse> {
    get_json(server, http, &format!("/jobs/{job_id}"), Some(auth))
}

/// `POST /plans/claim`: lease the next eligible job, or `None` on `204`.
pub fn claim(
    server: &ServerUrl,
    http: &HttpClient,
    auth: &ClientIdentity,
) -> Result<Option<ClaimedJob>> {
    let (response, url, method_name) =
        send(server, http, HttpMethod::Post, "/plans/claim", Some(auth))?;
    match response.status {
        204 => Ok(None),
        _ => decode_success(response, url, method_name).map(Some),
    }
}

pub fn heartbeat(
    server: &ServerUrl,
    http: &HttpClient,
    auth: &ClientIdentity,
    job_id: &str,
) -> Result<()> {
    send_empty(
        server,
        http,
        HttpMethod::Put,
        &format!("/plans/{job_id}/heartbeat"),
        Some(auth),
    )
}

pub fn reclaim(
    server: &ServerUrl,
    http: &HttpClient,
    auth: &ClientIdentity,
    job_id: &str,
) -> Result<()> {
    send_empty(
        server,
        http,
        HttpMethod::Post,
        &format!("/plans/{job_id}/reclaim"),
        Some(auth),
    )
}

#[cfg(test)]
mod tests {
    use httpmock::Mock;
    use httpmock::prelude::*;

    use super::*;
    use crate::{
        ContentRange,
        call::test_support::{auth, http_ctx, server_url},
        error::Error,
        types::EntityTag,
    };

    /// A decoy that matches only signed requests, and the plain mock that
    /// serves the response.
    fn signed_decoy_and_plain<'s>(
        server: &'s MockServer,
        path: &str,
        body: &str,
    ) -> (Mock<'s>, Mock<'s>) {
        let signed = server.mock(|when, then| {
            when.method(GET).path(path).header_exists("X-Signature");
            then.status(500).body("signed");
        });
        let plain = server.mock(|when, then| {
            when.method(GET).path(path);
            then.status(200).body(body);
        });
        (signed, plain)
    }

    #[test]
    fn the_list_read_sends_no_credentials_without_an_identity() -> Result<()> {
        let server = MockServer::start();
        let (signed, plain) = signed_decoy_and_plain(&server, "/benchmarks", "[]");

        list_benchmarks_conditional(&server_url(&server)?, &http_ctx()?, None, None)?;

        signed.assert_calls(0);
        plain.assert();
        Ok(())
    }

    #[test]
    fn the_detail_read_sends_no_credentials_without_an_identity() -> Result<()> {
        let server = MockServer::start();
        let (signed, plain) = signed_decoy_and_plain(
            &server,
            "/benchmarks/b1",
            r#"{"benchmark_id":"b1","benchmark_type":"prefill_throughput"}"#,
        );

        get_benchmark(&server_url(&server)?, &http_ctx()?, None, "b1")?;

        signed.assert_calls(0);
        plain.assert();
        Ok(())
    }

    /// Matches only the full `v1` header set.
    fn signed_only<'s>(server: &'s MockServer, path: &str, body: &str) -> Mock<'s> {
        server.mock(|when, then| {
            when.method(GET)
                .path(path)
                .header_exists("X-Client-Id")
                .header_exists("X-Timestamp")
                .header_exists("X-Nonce")
                .header_exists("X-Signature");
            then.status(200).body(body);
        })
    }

    /// Unsigned, a registered device would see only the public catalog.
    #[test]
    fn the_list_read_signs_with_an_identity() -> Result<()> {
        let server = MockServer::start();
        let signed = signed_only(&server, "/benchmarks", "[]");

        list_benchmarks_conditional(&server_url(&server)?, &http_ctx()?, Some(&auth()?), None)?;

        signed.assert();
        Ok(())
    }

    #[test]
    fn the_detail_reads_sign_with_an_identity() -> Result<()> {
        let server = MockServer::start();
        let signed = signed_only(
            &server,
            "/benchmarks/b1",
            r#"{"benchmark_id":"b1","benchmark_type":"prefill_throughput"}"#,
        );
        let http = http_ctx()?;
        let auth = auth()?;

        get_benchmark(&server_url(&server)?, &http, Some(&auth), "b1")?;
        get_benchmark_conditional(&server_url(&server)?, &http, Some(&auth), "b1", None)?;

        signed.assert_calls(2);
        Ok(())
    }

    #[test]
    fn a_conditional_detail_read_reports_not_modified() -> Result<()> {
        let server = MockServer::start();
        let mock = server.mock(|when, then| {
            when.method(GET)
                .path("/benchmarks/b1")
                .header("If-None-Match", "\"v1\"");
            then.status(304).header("etag", "\"v1\"");
        });

        let http = http_ctx()?;
        let etag = crate::types::EntityTag::try_new("\"v1\"")
            .map_err(|e| Error::BuildClient(e.to_string()))?;
        let response = get_benchmark_conditional(
            &server_url(&server)?,
            &http,
            None,
            "b1",
            Some(&IfNoneMatch::from(&etag)),
        )?;
        assert!(matches!(response, ConditionalResponse::NotModified { .. }));
        mock.assert();
        Ok(())
    }

    #[test]
    fn claim_200_returns_job() -> Result<()> {
        let server = MockServer::start();
        let body = r#"{
            "job_id": "job-1",
            "benchmark_id": "prefill_throughput_256",
            "time_window": "PT5M",
            "client_versions": ["2026.10.02-430a0bf-internal"],
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
                    "repository_url": "github.com/ggml-org/llama.cpp",
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
        let http = http_ctx()?;
        let job =
            claim(&server_url(&server)?, &http, &auth()?)?.ok_or_else(|| Error::HttpStatus {
                method: "POST".into(),
                url: "x".into(),
                status: 500,
                body: "missing job".into(),
            })?;
        assert_eq!(job.job_id, "job-1");
        assert_eq!(job.time_window, "PT5M");
        assert_eq!(
            job.client_versions,
            Some(vec!["2026.10.02-430a0bf-internal".to_owned()])
        );
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
        let http = http_ctx()?;
        assert!(claim(&server_url(&server)?, &http, &auth()?)?.is_none());
        Ok(())
    }

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
        let http = http_ctx()?;
        let err = claim(&server_url(&server)?, &http, &auth()?)
            .err()
            .ok_or_else(|| Error::BuildClient("expected error".into()))?;
        assert_eq!(err.http_status(), Some(308));
        redirect.assert();
        followed.assert_calls(0);
        Ok(())
    }

    #[test]
    fn claim_403_is_http_status() -> Result<()> {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(POST).path("/plans/claim");
            then.status(403).body("pending");
        });
        let http = http_ctx()?;
        let err = claim(&server_url(&server)?, &http, &auth()?)
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
        let http = http_ctx()?;
        heartbeat(&server_url(&server)?, &http, &auth()?, "job-1")?;
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
        let http = http_ctx()?;
        let err = reclaim(&server_url(&server)?, &http, &auth()?, "job-1")
            .err()
            .ok_or_else(|| Error::BuildClient("expected error".into()))?;
        assert_eq!(err.http_status(), Some(409));
        mock.assert();
        Ok(())
    }

    const SHA_A: &str = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";

    fn vault_id() -> Result<VaultResourceId> {
        VaultResourceId::try_new("vault_0195c3e2").map_err(|e| Error::BuildClient(e.to_string()))
    }

    #[test]
    fn the_vault_record_read_is_signed_and_reads_the_pin() -> Result<()> {
        let server = MockServer::start();
        let mock = server.mock(|when, then| {
            when.method(GET)
                .path("/vault/resources/vault_0195c3e2")
                .header_exists("X-Signature")
                .header("X-Client-Id", "ev1_test");
            then.status(200).body(format!(
                r#"{{"resource_id":"vault_0195c3e2","name":"ckpt","owner":{{"type":"org","org_id":"org_1"}},"object_key":"org_1/ckpt.tar","sha256":"{SHA_A}","bytes":3,"created_at":"2026-09-01T00:00:00Z","updated_at":"2026-09-01T00:00:00Z"}}"#
            ));
        });
        let record =
            get_vault_resource(&server_url(&server)?, &http_ctx()?, &auth()?, &vault_id()?)?;
        mock.assert();
        assert_eq!(record.sha256.as_ref(), SHA_A);
        assert_eq!(record.bytes, 3);
        assert_eq!(record.name, "ckpt");
        Ok(())
    }

    #[test]
    fn the_vault_content_read_streams_the_body() -> Result<()> {
        let server = MockServer::start();
        let mock = server.mock(|when, then| {
            when.method(GET)
                .path("/vault/resources/vault_0195c3e2/content")
                .header_exists("X-Signature");
            then.status(200).header("Content-Length", "3").body("abc");
        });
        let mut stream = get_vault_content(
            &server_url(&server)?,
            &http_ctx()?,
            &auth()?,
            &vault_id()?,
            None,
        )?;
        mock.assert();
        let mut body = Vec::new();
        std::io::Read::read_to_end(&mut stream, &mut body)
            .map_err(|e| Error::BuildClient(e.to_string()))?;
        assert_eq!(body, b"abc");
        Ok(())
    }

    #[test]
    fn a_vault_content_resume_sends_range_and_if_range() -> Result<()> {
        let server = MockServer::start();
        let etag = format!("\"{SHA_A}\"");
        let mock = server.mock(|when, then| {
            when.method(GET)
                .path("/vault/resources/vault_0195c3e2/content")
                .header_exists("X-Signature")
                .header("Range", "bytes=2-")
                .header("If-Range", &etag);
            then.status(206)
                .header("Content-Range", "bytes 2-2/3")
                .header("ETag", &etag)
                .body("c");
        });
        let resume = ResumeFrom {
            offset: 2,
            if_range: EntityTag::try_new(etag.clone())
                .map_err(|e| Error::BuildClient(e.to_string()))?,
        };
        let stream = get_vault_content(
            &server_url(&server)?,
            &http_ctx()?,
            &auth()?,
            &vault_id()?,
            Some(&resume),
        )?;
        mock.assert();
        assert_eq!(stream.status(), 206);
        assert_eq!(
            stream.content_range(),
            Some(ContentRange {
                first: 2,
                last: 2,
                total: Some(3)
            })
        );
        assert_eq!(stream.strong_etag(), Some(etag));
        Ok(())
    }

    #[test]
    fn a_weak_etag_is_not_a_strong_etag() -> Result<()> {
        let server = MockServer::start();
        let etag = format!("W/\"{SHA_A}\"");
        server.mock(|when, then| {
            when.method(GET)
                .path("/vault/resources/vault_0195c3e2/content");
            then.status(200).header("ETag", &etag).body("abc");
        });
        let stream = get_vault_content(
            &server_url(&server)?,
            &http_ctx()?,
            &auth()?,
            &vault_id()?,
            None,
        )?;
        assert_eq!(stream.strong_etag(), None);
        Ok(())
    }

    #[rstest::rstest]
    #[case(403)]
    #[case(404)]
    #[case(416)]
    fn a_vault_content_refusal_is_a_status_error(#[case] status: u16) -> Result<()> {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(GET)
                .path("/vault/resources/vault_0195c3e2/content");
            then.status(status).body("not yours");
        });
        let resume = ResumeFrom {
            offset: 3,
            if_range: EntityTag::try_new(format!("\"{SHA_A}\""))
                .map_err(|e| Error::BuildClient(e.to_string()))?,
        };
        let err = get_vault_content(
            &server_url(&server)?,
            &http_ctx()?,
            &auth()?,
            &vault_id()?,
            (status == 416).then_some(&resume),
        )
        .err()
        .ok_or_else(|| Error::BuildClient("expected error".into()))?;
        assert_eq!(err.http_status(), Some(status));
        assert!(!err.is_transient());
        assert!(err.to_string().contains("not yours"), "{err}");
        Ok(())
    }
}
