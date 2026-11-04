//! Endpoints a **user** (a person or service job authoring work) calls,
//! signed with a [`crate::UserIdentity`] (`pipette-mgmt`'s
//! `docs/authentication.md` §5.11). [`register`] signs nothing,
//! [`me_by_key`] signs with a key whose account is not yet known, and the
//! untyped [`call()`] family signs only when given an identity.
//!
//! A host that both runs benchmarks and files plans holds one identity of each
//! kind: the lease verbs take an approved client, and plan permissions are a
//! user's. A user's routes live under `/api/v1`; a device's stay at the root.

use pipette_http::HttpClient;
use pipette_plan_types::UserKeyId;

use crate::{
    auth::{KeyOnlyIdentity, UserIdentity},
    call::{
        self, ByteStream, ImportProgress, RequestBody, decode_success, encode_target, get_json,
        get_stream, send, send_empty, send_json, send_value,
    },
    error::Result,
    transport::HttpMethod,
    types::{
        AddUserKeyRequest, AddUserKeyResponse, RegisterUserRequest, RegisterUserResponse,
        ServerUrl, UserProfile,
    },
};

/// `POST /api/v1/users/register`: bind a public key to an account (see
/// [`RegisterUserRequest`]). Unsigned, since no key is bound until this call
/// binds one.
pub fn register(
    server: &ServerUrl,
    http: &HttpClient,
    request: RegisterUserRequest,
) -> Result<RegisterUserResponse> {
    send_json(
        server,
        http,
        HttpMethod::Post,
        "/api/v1/users/register",
        None,
        request,
    )
}

pub fn me(server: &ServerUrl, http: &HttpClient, auth: &UserIdentity) -> Result<UserProfile> {
    get_json(server, http, "/api/v1/users/me", Some(auth))
}

/// Where a key stands, as [`me_by_key`] reads it.
#[derive(Debug)]
pub enum KeyApproval {
    Approved(UserProfile),
    /// `401 key not found` or `401 key expired`: no account holds the key yet,
    /// or the one that does has not renewed it.
    Pending,
    /// Any other `401`/`403` that carries mgmt's reason.
    Refused(RefusalError),
}

/// Why the server refused a key, for the reasons a caller acts on.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum RefusalError {
    #[error("user is disabled")]
    UserDisabled,
    #[error("timestamp expired")]
    TimestampExpired,
    #[error("{0}")]
    Other(String),
}

impl From<String> for RefusalError {
    fn from(reason: String) -> Self {
        match reason.as_str() {
            "user is disabled" => Self::UserDisabled,
            "timestamp expired" => Self::TimestampExpired,
            _ => Self::Other(reason),
        }
    }
}

/// [`me`] signed by the key alone, for a holder that does not know its account
/// yet. A 401/403 without mgmt's `{"error"}` body (a proxy's page) stays an
/// [`Error::HttpStatus`](crate::Error::HttpStatus), keeping its status and URL.
pub fn me_by_key(
    server: &ServerUrl,
    http: &HttpClient,
    auth: &KeyOnlyIdentity,
) -> Result<KeyApproval> {
    let (response, url, method) = send(
        server,
        http,
        HttpMethod::Get,
        "/api/v1/users/me",
        Some(auth),
    )?;
    if matches!(response.status, 401 | 403)
        && let Some(reason) = call::server_reason(&response.body)
    {
        let pending =
            response.status == 401 && matches!(reason.as_str(), "key not found" | "key expired");
        return Ok(if pending {
            KeyApproval::Pending
        } else {
            KeyApproval::Refused(reason.into())
        });
    }
    decode_success(response, url, method).map(KeyApproval::Approved)
}

/// `POST /api/v1/users/me/keys`: register another machine's key under this
/// account, signed by a key the account already holds. A second machine usually
/// cannot [`register`] itself: an invite onboards only an account nobody has
/// signed in as yet.
pub fn add_key(
    server: &ServerUrl,
    http: &HttpClient,
    auth: &UserIdentity,
    request: AddUserKeyRequest,
) -> Result<AddUserKeyResponse> {
    send_json(
        server,
        http,
        HttpMethod::Post,
        "/api/v1/users/me/keys",
        Some(auth),
        request,
    )
}

/// `DELETE /api/v1/users/me/keys/{key_id}`. Revoking the last key, or the one
/// signing, is permitted, so a compromised only key can be retired at once.
pub fn revoke_key(
    server: &ServerUrl,
    http: &HttpClient,
    auth: &UserIdentity,
    user_key_id: &UserKeyId,
) -> Result<()> {
    send_empty(
        server,
        http,
        HttpMethod::Delete,
        &format!("/api/v1/users/me/keys/{user_key_id}"),
        Some(auth),
    )
}

/// Any route, untyped, so a route mgmt adds needs no counterpart here.
/// `segments` are the path under the server root, one segment each. The answer
/// is JSON, or a string when the server answers another type. `auth` is
/// optional for the public catalog reads.
pub fn call(
    server: &ServerUrl,
    http: &HttpClient,
    auth: Option<&UserIdentity>,
    method: HttpMethod,
    segments: &[&str],
    query: &[(String, String)],
    body: Option<RequestBody<'_>>,
) -> Result<serde_json::Value> {
    send_value(
        server,
        http,
        method,
        &encode_target(segments, query)?,
        auth.map(|auth| auth as _),
        body,
    )
}

/// [`call`] for a route that files a plan, reading its progress stream: each
/// `progress` line goes to `on_progress`, and the `result` line is the answer.
#[expect(
    clippy::too_many_arguments,
    reason = "the arguments of `call`, plus the callback for the progress"
)]
pub fn call_ingest(
    server: &ServerUrl,
    http: &HttpClient,
    auth: Option<&UserIdentity>,
    method: HttpMethod,
    segments: &[&str],
    query: &[(String, String)],
    body: RequestBody<'_>,
    on_progress: &mut dyn FnMut(&ImportProgress),
) -> Result<serde_json::Value> {
    call::send_ingest(
        server,
        http,
        method,
        &encode_target(segments, query)?,
        auth.map(|auth| auth as _),
        body,
        on_progress,
    )
}

/// [`call`] as a `HEAD`: whether the resource exists, `404` being `false`.
pub fn exists(
    server: &ServerUrl,
    http: &HttpClient,
    auth: Option<&UserIdentity>,
    segments: &[&str],
) -> Result<bool> {
    call::exists(
        server,
        http,
        &encode_target(segments, &[])?,
        auth.map(|auth| auth as _),
    )
}

/// [`call`] as a `PUT` whose body is `file`, streamed from disk. `sha256` is
/// the file's digest: the signature covers it in place of the bytes, and the
/// server holds the upload to it.
pub fn upload(
    server: &ServerUrl,
    http: &HttpClient,
    auth: Option<&UserIdentity>,
    segments: &[&str],
    file: std::fs::File,
    len: u64,
    sha256: [u8; 32],
) -> Result<serde_json::Value> {
    call::send_file(
        server,
        http,
        HttpMethod::Put,
        &encode_target(segments, &[])?,
        auth.map(|auth| auth as _),
        call::FileBody { file, len, sha256 },
    )
}

/// [`call`] for a `GET` whose body is an artifact, streamed rather than read.
pub fn download(
    server: &ServerUrl,
    http: &HttpClient,
    auth: Option<&UserIdentity>,
    segments: &[&str],
    query: &[(String, String)],
) -> Result<ByteStream> {
    get_stream(
        server,
        http,
        &encode_target(segments, query)?,
        auth.map(|auth| auth as _),
        Vec::new(),
    )
}

#[cfg(test)]
pub(crate) mod tests {
    use httpmock::prelude::*;
    use rstest::rstest;
    use serde_json::Value;

    use super::*;
    use crate::{
        call::test_support::{http_ctx, server_url, user_auth, user_auth_and_key},
        error::Error,
        httpsig::test_support::signed_by,
        types::{ContactEmail, KeyDetails, PrivateKey, PublicKey, UserPreauthKey, UserStatus},
    };

    /// mgmt's whole `OwnUserView`, including fields the client drops.
    pub(crate) fn own_user_view() -> Value {
        serde_json::json!({
            "user_id": "user_0192f0c1",
            "contact_email": "ops@example.com",
            "account_type": "human",
            "status": "active",
            "created_at": "2026-01-02T03:04:05Z",
            "org_id": "org_0192f0c2",
            "org_slug": "acme",
            "permissions": ["read_clients", "read_plans", "submit_plans"],
            "may": {
                "read_roster": "none",
                "administer_roster": "none",
                "read_fleet": true,
                "administer_fleet": "none",
                "edit_catalog": "none",
                "read_plans": "own",
                "file_plans": true,
                "create_orgs": false,
                "grant": ["submit_plans", "read_plans", "read_clients"],
                "grant_service": ["submit_plans", "read_plans", "read_clients"],
            },
            "keys": [{
                "key_id": "userkey_9f2c",
                "public_key": "a3f8".repeat(16),
                "key_details": "work laptop",
                "added_at": "2026-01-02T03:04:05Z",
                "expires_at": "2026-02-02T03:04:05Z",
            }],
        })
    }

    fn user_preauth_key(raw: &str) -> anyhow::Result<UserPreauthKey> {
        Ok(UserPreauthKey::try_new(raw)?)
    }

    /// A valid public key named by a short seed ("a3f8").
    fn public_key(seed: &str) -> anyhow::Result<PublicKey> {
        Ok(PublicKey::try_new(seed.repeat(64 / seed.len()))?)
    }

    fn user_key_id(raw: &str) -> anyhow::Result<UserKeyId> {
        Ok(UserKeyId::try_new(raw)?)
    }

    fn register_request() -> anyhow::Result<RegisterUserRequest> {
        Ok(RegisterUserRequest {
            public_key: public_key("a3f8")?,
            user_preauth_key: Some(user_preauth_key("user_preauth_id.secret")?),
            contact_email: None,
            key_details: KeyDetails::try_new("work laptop")?,
        })
    }

    /// The whole body, so a renamed or dropped field fails.
    #[test]
    fn register_posts_the_key_and_the_user_preauth_key_it_is_admitted_by() -> anyhow::Result<()> {
        let server = MockServer::start();
        let mock = server.mock(|when, then| {
            when.method(POST)
                .path("/api/v1/users/register")
                .json_body(serde_json::json!({
                    "public_key": "a3f8".repeat(16),
                    "user_preauth_key": "user_preauth_id.secret",
                    "key_details": "work laptop",
                }));
            then.status(201).json_body(serde_json::json!({
                "user_id": "user_0192f0c1",
                "status": "active",
                "key_id": "userkey_9f2c",
            }));
        });
        let response = register(&server_url(&server)?, &http_ctx()?, register_request()?)?;
        assert_eq!(response.user_id.to_string(), "user_0192f0c1");
        assert_eq!(response.status, UserStatus::Active);
        assert_eq!(response.user_key_id.to_string(), "userkey_9f2c");
        mock.assert();
        Ok(())
    }

    /// The absent invite is omitted rather than `null`, which the server could
    /// not tell from a malformed invited registration.
    #[test]
    fn register_posts_a_contact_address_when_no_invite_admits_it() -> anyhow::Result<()> {
        let server = MockServer::start();
        let mock = server.mock(|when, then| {
            when.method(POST)
                .path("/api/v1/users/register")
                .json_body(serde_json::json!({
                    "public_key": "a3f8".repeat(16),
                    "contact_email": "ops@example.com",
                    "key_details": "work laptop",
                }));
            then.status(201).json_body(serde_json::json!({
                "user_id": "user_0192f0c1",
                "status": "pending",
                "key_id": "userkey_9f2c",
            }));
        });
        let response = register(
            &server_url(&server)?,
            &http_ctx()?,
            RegisterUserRequest {
                user_preauth_key: None,
                contact_email: Some(ContactEmail::try_new("ops@example.com")?),
                ..register_request()?
            },
        )?;
        assert_eq!(response.status, UserStatus::Pending);
        mock.assert();
        Ok(())
    }

    /// A signature by a key not yet bound to an account would be a `401`.
    #[test]
    fn register_sends_no_signature() -> anyhow::Result<()> {
        let server = MockServer::start();
        let mock = server.mock(|when, then| {
            when.method(POST)
                .path("/api/v1/users/register")
                .is_true(|request| {
                    let headers = request.headers();
                    ["Signature", "Signature-Input", "X-Signature"]
                        .iter()
                        .all(|name| !headers.contains_key(*name))
                });
            then.status(201).json_body(serde_json::json!({
                "user_id": "user_0192f0c1",
                "status": "active",
                "key_id": "userkey_9f2c",
            }));
        });
        register(&server_url(&server)?, &http_ctx()?, register_request()?)?;
        mock.assert();
        Ok(())
    }

    /// The holder compares the key id with the one it derived, so a missing
    /// one is a fault, not a default.
    #[test]
    fn register_refuses_a_receipt_with_no_key_id() -> anyhow::Result<()> {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(POST).path("/api/v1/users/register");
            then.status(201).json_body(serde_json::json!({
                "user_id": "user_0192f0c1",
                "status": "active",
            }));
        });
        let error = register(&server_url(&server)?, &http_ctx()?, register_request()?)
            .err()
            .ok_or_else(|| Error::BuildClient("expected a decode error".into()))?;
        // The wire name, not the Rust one (`user_key_id`).
        assert!(
            error.to_string().contains("missing field `key_id`"),
            "{error}"
        );
        Ok(())
    }

    #[test]
    fn me_signs_the_read_and_decodes_the_account_and_its_keys() -> anyhow::Result<()> {
        let (auth, public_key) = user_auth_and_key()?;
        let server = MockServer::start();
        let mock = server.mock(|when, then| {
            when.method(GET)
                .path("/api/v1/users/me")
                .is_true(signed_by(public_key, None));
            then.status(200).json_body(own_user_view());
        });
        let profile = me(&server_url(&server)?, &http_ctx()?, &auth)?;
        assert_eq!(profile.user_id.to_string(), "user_0192f0c1");
        assert_eq!(profile.contact_email.to_string(), "ops@example.com");
        assert_eq!(profile.status, UserStatus::Active);
        assert_eq!(
            profile.permissions,
            ["read_clients", "read_plans", "submit_plans"]
        );
        assert_eq!(profile.keys.unreadable, 0);
        assert_eq!(
            profile.keys.entries[0].user_key_id.to_string(),
            "userkey_9f2c"
        );
        assert_eq!(profile.keys.entries[0].key_details, "work laptop");
        mock.assert();
        Ok(())
    }

    fn key_only_identity(
        private_key_hex: PrivateKey,
        public_key_hex: &PublicKey,
    ) -> anyhow::Result<KeyOnlyIdentity> {
        Ok(KeyOnlyIdentity {
            user_key_id: crate::auth::key_id_for(public_key_hex)?,
            signer: crate::auth::UserSigner::PrivateKey(private_key_hex),
        })
    }

    #[test]
    fn me_by_key_signs_with_the_key_alone() -> anyhow::Result<()> {
        let (private_key_hex, public_key_hex) = crate::auth::generate_keypair()?;
        let auth = key_only_identity(private_key_hex, &public_key_hex)?;
        let verifies = signed_by(public_key_hex, Some(auth.user_key_id.to_string()));
        let server = MockServer::start();
        let mock = server.mock(|when, then| {
            when.method(GET).path("/api/v1/users/me").is_true(verifies);
            then.status(200).json_body(own_user_view());
        });
        let answer = me_by_key(&server_url(&server)?, &http_ctx()?, &auth)?;
        let KeyApproval::Approved(profile) = answer else {
            anyhow::bail!("expected an approval, got {answer:?}");
        };
        assert_eq!(profile.user_id.to_string(), "user_0192f0c1");
        mock.assert();
        Ok(())
    }

    #[rstest]
    #[case::key_not_found(401, r#"{"error":"key not found"}"#, "pending")]
    #[case::key_expired(401, r#"{"error":"key expired"}"#, "pending")]
    #[case::user_disabled(403, r#"{"error":"user is disabled"}"#, "refused: user is disabled")]
    #[case::timestamp_expired(
        401,
        r#"{"error":"timestamp expired"}"#,
        "refused: timestamp expired"
    )]
    #[case::unknown_reason(401, r#"{"error":"bad signature"}"#, "other: bad signature")]
    // Only a 401 means "not yet"; a 403 naming the same reason is a refusal.
    #[case::forbidden_key_not_found(403, r#"{"error":"key not found"}"#, "other: key not found")]
    #[case::empty_body(401, "", "status 401")]
    #[case::html_page(403, "<html>Forbidden</html>", "status 403")]
    fn me_by_key_reads_a_refusal_from_the_raw_body(
        #[case] status: u16,
        #[case] body: &str,
        #[case] expected: &str,
    ) -> anyhow::Result<()> {
        let (private_key_hex, public_key_hex) = crate::auth::generate_keypair()?;
        let auth = key_only_identity(private_key_hex, &public_key_hex)?;
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(GET).path("/api/v1/users/me");
            then.status(status).body(body);
        });
        let answer = me_by_key(&server_url(&server)?, &http_ctx()?, &auth);
        let got = match answer {
            Ok(KeyApproval::Pending) => "pending".to_string(),
            Ok(KeyApproval::Refused(
                refusal @ (RefusalError::UserDisabled | RefusalError::TimestampExpired),
            )) => format!("refused: {refusal}"),
            Ok(KeyApproval::Refused(RefusalError::Other(reason))) => format!("other: {reason}"),
            Err(Error::HttpStatus {
                status: refused,
                url,
                ..
            }) if url.ends_with("/users/me") => format!("status {refused}"),
            other => format!("{other:?}"),
        };
        assert_eq!(got, expected);
        Ok(())
    }

    #[test]
    fn add_key_posts_the_second_machines_key_without_a_user_preauth_key() -> anyhow::Result<()> {
        let (auth, signing_key) = user_auth_and_key()?;
        let server = MockServer::start();
        let mock = server.mock(|when, then| {
            when.method(POST)
                .path("/api/v1/users/me/keys")
                .is_true(signed_by(signing_key, None))
                .json_body(serde_json::json!({
                    "public_key": "b7c1".repeat(16),
                    "key_details": "desktop",
                }));
            // The register receipt, whose account fields must be tolerated.
            then.status(201).json_body(serde_json::json!({
                "user_id": "user_0192f0c1",
                "status": "active",
                "key_id": "userkey_4d0a",
            }));
        });
        let response = add_key(
            &server_url(&server)?,
            &http_ctx()?,
            &auth,
            AddUserKeyRequest {
                public_key: public_key("b7c1")?,
                key_details: KeyDetails::try_new("desktop")?,
                expires_at: None,
            },
        )?;
        assert_eq!(response.user_key_id.to_string(), "userkey_4d0a");
        mock.assert();
        Ok(())
    }

    #[test]
    fn revoke_key_deletes_the_named_registration_and_accepts_an_empty_204() -> anyhow::Result<()> {
        let server = MockServer::start();
        let mock = server.mock(|when, then| {
            when.method(DELETE)
                .path("/api/v1/users/me/keys/userkey_9f2c")
                .header_exists("Signature");
            then.status(204);
        });
        let key_id = user_key_id("userkey_9f2c")?;
        revoke_key(&server_url(&server)?, &http_ctx()?, &user_auth()?, &key_id)?;
        mock.assert();
        Ok(())
    }

    #[test]
    fn revoke_key_surfaces_a_refusal_status_and_body() -> anyhow::Result<()> {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(DELETE)
                .path("/api/v1/users/me/keys/userkey_9f2c");
            then.status(404).body("unknown key");
        });
        let key_id = user_key_id("userkey_9f2c")?;
        let error = revoke_key(&server_url(&server)?, &http_ctx()?, &user_auth()?, &key_id)
            .err()
            .ok_or_else(|| Error::BuildClient("expected HTTP error".into()))?;
        assert_eq!(error.http_status(), Some(404));
        assert!(error.to_string().contains("unknown key"));
        Ok(())
    }

    /// Body digest included, each segment and the query encoded.
    #[test]
    fn call_signs_and_relays_json() -> anyhow::Result<()> {
        let (auth, public_key) = user_auth_and_key()?;
        let server = MockServer::start();
        let mock = server.mock(|when, then| {
            when.method(PATCH)
                .path("/api/v1/orgs/a%2Fb")
                .query_param("dry", "1")
                .header_exists("Content-Digest")
                .is_true(signed_by(public_key, None))
                .json_body(serde_json::json!({ "display_name": "Acme" }));
            then.status(200)
                .json_body(serde_json::json!({ "org_id": "org_1" }));
        });
        let answer = call(
            &server_url(&server)?,
            &http_ctx()?,
            Some(&auth),
            HttpMethod::Patch,
            &["api", "v1", "orgs", "a/b"],
            &[("dry".into(), "1".into())],
            Some(RequestBody::Json(
                &serde_json::json!({ "display_name": "Acme" }),
            )),
        )?;
        assert_eq!(answer, serde_json::json!({ "org_id": "org_1" }));
        mock.assert();
        Ok(())
    }

    #[test]
    fn call_reads_an_empty_success_as_null() -> anyhow::Result<()> {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(DELETE).path("/api/v1/orgs/acme");
            then.status(204);
        });
        let answer = call(
            &server_url(&server)?,
            &http_ctx()?,
            Some(&user_auth()?),
            HttpMethod::Delete,
            &["api", "v1", "orgs", "acme"],
            &[],
            None,
        )?;
        assert_eq!(answer, Value::Null);
        Ok(())
    }

    #[test]
    fn call_sends_toml_under_its_own_type() -> anyhow::Result<()> {
        let document = "benchmarks = [\"decode_throughput_512_100\"]\n";
        let report = serde_json::json!({ "bodies": [] });
        let server = MockServer::start();
        let mock = server.mock(|when, then| {
            when.method(POST)
                .path("/api/v1/campaigns/campaign_1/plans")
                .query_param("dry_run", "true")
                .header("content-type", "application/toml")
                .body(document);
            then.status(200).json_body(report.clone());
        });
        let answer = call(
            &server_url(&server)?,
            &http_ctx()?,
            Some(&user_auth()?),
            HttpMethod::Post,
            &["api", "v1", "campaigns", "campaign_1", "plans"],
            &[("dry_run".into(), "true".into())],
            Some(RequestBody::Toml(document)),
        )?;
        assert_eq!(answer, report);
        mock.assert();
        Ok(())
    }

    #[test]
    fn call_reads_a_text_answer() -> anyhow::Result<()> {
        let document = "benchmarks = [\"decode_throughput_512_100\"]\n";
        let server = MockServer::start();
        let mock = server.mock(|when, then| {
            when.method(GET).path("/api/v1/plans/plan-1/document");
            then.status(200)
                .header("content-type", "application/toml; charset=utf-8")
                .body(document);
        });
        let answer = call(
            &server_url(&server)?,
            &http_ctx()?,
            Some(&user_auth()?),
            HttpMethod::Get,
            &["api", "v1", "plans", "plan-1", "document"],
            &[],
            None,
        )?;
        assert_eq!(answer, Value::String(document.into()));
        mock.assert();
        Ok(())
    }

    #[rstest]
    #[case(200, true)]
    #[case(404, false)]
    fn exists_reads_a_head_as_yes_or_no(
        #[case] status: u16,
        #[case] expected: bool,
    ) -> anyhow::Result<()> {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(httpmock::Method::HEAD)
                .path("/api/v1/vault/blobs/user_1/abc");
            then.status(status);
        });
        let found = exists(
            &server_url(&server)?,
            &http_ctx()?,
            Some(&user_auth()?),
            &["api", "v1", "vault", "blobs", "user_1", "abc"],
        )?;
        assert_eq!(found, expected);
        Ok(())
    }

    /// The signature covers the digest the streamed body must match.
    #[test]
    fn upload_streams_the_file_as_the_body() -> anyhow::Result<()> {
        let (auth, public_key) = user_auth_and_key()?;
        let server = MockServer::start();
        let mock = server.mock(|when, then| {
            when.method(PUT)
                .path("/api/v1/vault/blobs/user_1/abc")
                .header("content-type", "application/octet-stream")
                .is_true(signed_by(public_key, None))
                .body("weights");
            then.status(201)
                .json_body(serde_json::json!({ "sha256": "abc", "bytes": 7 }));
        });
        let mut file = tempfile::tempfile()?;
        std::io::Write::write_all(&mut file, b"weights")?;
        std::io::Seek::rewind(&mut file)?;
        let answer = upload(
            &server_url(&server)?,
            &http_ctx()?,
            Some(&auth),
            &["api", "v1", "vault", "blobs", "user_1", "abc"],
            file,
            7,
            crate::httpsig::sha256(b"weights"),
        )?;
        assert_eq!(answer, serde_json::json!({ "sha256": "abc", "bytes": 7 }));
        mock.assert();
        Ok(())
    }

    #[derive(Debug)]
    enum Ends {
        Answer(Value),
        Refused(u16, &'static str),
        Broken,
        Undecodable,
    }

    const PROGRESS: &str = r#"{"event":"progress","hf_ref":"org/m@abc","file":"m.gguf","bytes_done":5,"bytes_total":10}"#;
    const DONE: &str = r#"{"event":"progress","hf_ref":"org/m@abc","file":"m.gguf","bytes_done":10,"bytes_total":10}"#;
    const RESULT: &str = r#"{"event":"result","status":201,"body":{"plan_id":"plan_1"}}"#;

    /// The server holds back the `result` line until the callback has run.
    #[test]
    fn call_ingest_reports_progress_before_the_stream_ends() -> anyhow::Result<()> {
        use std::{
            io::{Read, Write},
            net::TcpListener,
            sync::mpsc,
            time::Duration,
        };

        let listener = TcpListener::bind("127.0.0.1:0")?;
        let server = ServerUrl::try_new(format!("http://{}", listener.local_addr()?))?;
        let (reported, heard) = mpsc::channel::<()>();
        std::thread::spawn(move || {
            let Ok((mut socket, _)) = listener.accept() else {
                return;
            };
            let mut request = Vec::new();
            let mut buf = [0; 1024];
            while !request.windows(4).any(|window| window == b"\r\n\r\n") {
                match socket.read(&mut buf) {
                    Ok(0) | Err(_) => return,
                    Ok(n) => request.extend_from_slice(buf.get(..n).unwrap_or_default()),
                }
            }
            let head = "HTTP/1.1 200 OK\r\ncontent-type: application/x-ndjson\r\n\
                        connection: close\r\n\r\n";
            let _ = socket.write_all(format!("{head}{PROGRESS}\n").as_bytes());
            // A client that buffers the body never calls back, and the stream
            // then ends with no last line.
            if heard.recv_timeout(Duration::from_secs(5)).is_ok() {
                let _ = socket.write_all(format!("{RESULT}\n").as_bytes());
            }
        });
        let mut seen = 0;
        let answer = call_ingest(
            &server,
            &http_ctx()?,
            None,
            HttpMethod::Put,
            &["api", "v1", "campaigns", "campaign_1", "document"],
            &[],
            RequestBody::Toml("x = 1\n"),
            &mut |_| {
                seen += 1;
                let _ = reported.send(());
            },
        );
        assert_eq!(answer?, serde_json::json!({ "plan_id": "plan_1" }));
        assert_eq!(seen, 1);
        Ok(())
    }

    /// Skips a `keepalive`, an unknown event and an unreadable `progress`
    /// line; a server that answers with no stream gets the plain decode.
    #[rstest]
    #[case::streamed(
        200, "application/x-ndjson",
        format!("{PROGRESS}\n{{\"event\":\"keepalive\"}}\n{{\"event\":\"future\",\"x\":1}}\n\n{DONE}\n{RESULT}\n"),
        2, Ends::Answer(serde_json::json!({ "plan_id": "plan_1" })),
    )]
    #[case::dry_run(
        200, "application/x-ndjson; charset=utf-8",
        r#"{"event":"result","status":200,"body":{"jobs":[]}}"#.to_string() + "\n",
        0, Ends::Answer(serde_json::json!({ "jobs": [] })),
    )]
    #[case::undecodable_progress(
        200, "application/x-ndjson",
        format!("{{\"event\":\"progress\",\"file\":\"m.gguf\",\"bytes_done\":\"five\"}}\n{RESULT}\n"),
        0, Ends::Answer(serde_json::json!({ "plan_id": "plan_1" })),
    )]
    #[case::error_line(
        200, "application/x-ndjson",
        format!("{PROGRESS}\n{{\"event\":\"error\",\"status\":409,\"error\":\"stale base\"}}\n"),
        1, Ends::Refused(409, "stale base"),
    )]
    #[case::no_last_line(200, "application/x-ndjson", format!("{PROGRESS}\n"), 1, Ends::Broken)]
    #[case::cut_mid_line(
        200, "application/x-ndjson", format!("{PROGRESS}\n{{\"event\":\"res"), 1, Ends::Broken,
    )]
    #[case::not_json(200, "application/x-ndjson", "not json\n".to_string(), 0, Ends::Undecodable)]
    #[case::plain_answer(
        201, "application/json", r#"{"plan_id":"plan_1"}"#.to_string(),
        0, Ends::Answer(serde_json::json!({ "plan_id": "plan_1" })),
    )]
    #[case::refused_before_the_stream(
        401, "application/json", r#"{"error":"bad signature"}"#.to_string(),
        0, Ends::Refused(401, "bad signature"),
    )]
    fn call_ingest_reads_the_progress_stream(
        #[case] status: u16,
        #[case] content_type: &str,
        #[case] body: String,
        #[case] progress: usize,
        #[case] ends: Ends,
    ) -> anyhow::Result<()> {
        let (auth, public_key) = user_auth_and_key()?;
        let server = MockServer::start();
        let mock = server.mock(|when, then| {
            when.method(PUT)
                .path("/api/v1/campaigns/campaign_1/document")
                .query_param("base_plan_id", "plan_0")
                .header("accept", "application/x-ndjson")
                .header("content-type", "application/toml")
                .is_true(signed_by(public_key, None))
                .body("x = 1\n");
            then.status(status)
                .header("content-type", content_type)
                .body(body);
        });
        let mut seen = Vec::new();
        let answer = call_ingest(
            &server_url(&server)?,
            &http_ctx()?,
            Some(&auth),
            HttpMethod::Put,
            &["api", "v1", "campaigns", "campaign_1", "document"],
            &[("base_plan_id".into(), "plan_0".into())],
            RequestBody::Toml("x = 1\n"),
            &mut |event| seen.push(event.clone()),
        );
        mock.assert();
        assert_eq!(seen.len(), progress, "{seen:?}");
        if let Some(first) = seen.first() {
            assert_eq!(
                first,
                &ImportProgress {
                    hf_ref: "org/m@abc".into(),
                    file: "m.gguf".into(),
                    bytes_done: 5,
                    bytes_total: 10,
                }
            );
        }
        match (ends, answer) {
            (Ends::Answer(expected), Ok(value)) => assert_eq!(value, expected),
            (Ends::Refused(expected, reason), Err(error)) => {
                assert_eq!(error.http_status(), Some(expected), "{error}");
                assert!(error.to_string().contains(reason), "{error}");
            }
            (Ends::Broken, Err(error @ Error::StreamEnded { .. })) => {
                assert!(!error.is_transient(), "{error}");
            }
            (Ends::Undecodable, Err(Error::DecodeJson { .. })) => {}
            (ends, answer) => anyhow::bail!("expected {ends:?}, got {answer:?}"),
        }
        Ok(())
    }
}
