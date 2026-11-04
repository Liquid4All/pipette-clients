//! The management-server calls, wrapped so the loop matches on a disposition
//! instead of on a status code.
//!
//! `pipette-mgmt-client` sends one request per function with no retries; the
//! retry policy here mirrors `crates/pipette-cli/src/client/worker/protocol.rs`,
//! because the lease rules are asymmetric: a `404` reclaims once (a submit then
//! resends the same body), a `409` aborts the run or drops the result.
//!
//! Nothing here builds or signs a request: every call goes through
//! `pipette_mgmt_client::client_api`, the same code the CLI signs with.

pub(crate) mod submit;

use std::time::Duration;

use anyhow::Context;
use serde::Serialize;

use pipette_http::HttpClient;
use pipette_mgmt_client::types::{
    ClaimedJob, ClientProfile, ClientStatus, DeviceProfileFields, ServerUrl, UpdateClientRequest,
};
use pipette_mgmt_client::{ClientIdentity, Error as MgmtError, client_api};
use pipette_plan_types::JobAccepted;

use crate::releases::catalog::RUNTIME_BUILD_FLAG_PREFIX;
use crate::timing;

const USER_AGENT: &str = "pipette-ios-proxy";

#[derive(Debug)]
pub(crate) struct Session {
    pub(crate) server: ServerUrl,
    pub(crate) http: HttpClient,
    pub(crate) auth: ClientIdentity,
}

/// A server exchange that ended without the answer the loop needs.
#[derive(Debug, thiserror::Error)]
enum ServerError {
    #[error("gave up submitting the result for job {job_id} after {attempts} attempts")]
    SubmitGaveUp { job_id: String, attempts: u32 },
    #[error(
        "client {client_id} is still `{status}` after {secs}s; approve it with `pipette-mgmt \
         clients approve {client_id}`. This worker acts as the phone's own registered client, so \
         the approval is that client's"
    )]
    NotApproved {
        client_id: String,
        status: String,
        secs: u64,
    },
    #[error(
        "client {client_id} is approved but its profile reindex is still pending after {secs}s; \
         the server would answer every claim with 204"
    )]
    ReindexPending { client_id: String, secs: u64 },
}

/// What one `POST /plans/claim` meant.
#[derive(Debug)]
pub(crate) enum ClaimPoll {
    Job(Box<ClaimedJob>),
    Idle,
    /// `403`: this client is not approved. Terminal for the loop.
    NotApproved,
    /// `401`: bad signature or timestamp outside the freshness window. Terminal.
    SignatureRejected,
    Transient(MgmtError),
    Fatal(MgmtError),
}

/// What one heartbeat meant for the run in flight.
#[derive(Debug)]
pub(crate) enum LeaseKeepalive {
    Ok,
    /// The lease is gone for good: terminate the run and skip submit.
    Abort,
    Retry,
}

/// What one submit attempt meant.
#[derive(Debug)]
enum SubmitDisposition {
    Accepted {
        job_id: String,
    },
    /// `409`: the server refuses this result. Do not retry.
    Dropped,
    RetryTransient,
    Fatal(MgmtError),
}

/// What one `reclaim` attempt meant. Its own type because the heartbeat and
/// submit paths read the same outcomes differently.
enum ReclaimOutcome {
    Renewed,
    /// `404` or `409`, or a definitive status the protocol does not define: the
    /// lease will not come back.
    Unrecoverable,
    Transient(MgmtError),
}

pub(crate) fn poll_claim(ctx: &Session) -> ClaimPoll {
    match client_api::claim(&ctx.server, &ctx.http, &ctx.auth) {
        Ok(Some(job)) => ClaimPoll::Job(Box::new(job)),
        Ok(None) => ClaimPoll::Idle,
        Err(e) if e.http_status() == Some(403) => ClaimPoll::NotApproved,
        // Unlike the CLI's `poll_claim`, 401 is its own disposition: this runs
        // unattended, and clock drift and non-approval are different fixes.
        Err(e) if e.http_status() == Some(401) => ClaimPoll::SignatureRejected,
        Err(e) if e.is_transient() => ClaimPoll::Transient(e),
        Err(e) => ClaimPoll::Fatal(e),
    }
}

/// Whether the server says the claim's benchmark does not exist. Signed as the phone, so a
/// benchmark private to its organization is found. Only a `404` is that answer: any other
/// failure is logged and left to the app, which fetches the benchmark itself.
pub(crate) fn benchmark_is_unknown(ctx: &Session, benchmark_id: &str) -> bool {
    match client_api::get_benchmark(&ctx.server, &ctx.http, Some(&ctx.auth), benchmark_id) {
        Ok(_) => false,
        Err(e) if e.http_status() == Some(404) => true,
        Err(e) => {
            tracing::warn!(
                error = %format!("{e:#}"),
                %benchmark_id,
                "could not look the benchmark up; launching anyway"
            );
            false
        }
    }
}

/// One `PUT /plans/{job_id}/heartbeat`, with the single `reclaim` retry folded in.
pub(crate) fn keepalive_lease(ctx: &Session, job_id: &str) -> LeaseKeepalive {
    match client_api::heartbeat(&ctx.server, &ctx.http, &ctx.auth, job_id) {
        Ok(()) => LeaseKeepalive::Ok,
        Err(e) if e.http_status() == Some(409) => {
            tracing::warn!("the lease was taken away (409); aborting the run");
            LeaseKeepalive::Abort
        }
        Err(e) if e.http_status() == Some(404) => {
            tracing::warn!("the lease is gone (404); reclaiming it once");
            match reclaim_once(ctx, job_id) {
                ReclaimOutcome::Renewed => {
                    tracing::info!("lease reclaimed");
                    LeaseKeepalive::Ok
                }
                ReclaimOutcome::Unrecoverable => {
                    tracing::warn!("the lease could not be reclaimed; aborting the run");
                    LeaseKeepalive::Abort
                }
                ReclaimOutcome::Transient(e) => {
                    tracing::warn!(
                        error = %e,
                        "reclaim failed transiently; keeping the run and retrying the heartbeat"
                    );
                    LeaseKeepalive::Retry
                }
            }
        }
        // Unlogged: the heartbeat thread reports the transient failure itself.
        Err(e) if e.is_transient() => LeaseKeepalive::Retry,
        // A status the protocol does not define: retry rather than abort a live
        // run over it, as the CLI does.
        Err(e) => {
            tracing::warn!(
                error = %e,
                "unexpected heartbeat status; retrying rather than aborting a live run"
            );
            LeaseKeepalive::Retry
        }
    }
}

fn reclaim_once(ctx: &Session, job_id: &str) -> ReclaimOutcome {
    match client_api::reclaim(&ctx.server, &ctx.http, &ctx.auth, job_id) {
        Ok(()) => ReclaimOutcome::Renewed,
        Err(e) if matches!(e.http_status(), Some(404 | 409)) => ReclaimOutcome::Unrecoverable,
        Err(e) if e.is_transient() => ReclaimOutcome::Transient(e),
        Err(_) => ReclaimOutcome::Unrecoverable,
    }
}

/// One `POST /benchmarks`, by reference so a retry resubmits identical bytes.
fn try_submit<P: Serialize>(ctx: &Session, job_id: &str, payload: &P) -> SubmitDisposition {
    match client_api::submit_result(&ctx.server, &ctx.http, &ctx.auth, payload) {
        Ok(response) => SubmitDisposition::Accepted {
            job_id: response.job_id,
        },
        Err(e) if e.http_status() == Some(409) => {
            tracing::warn!("the server refused the result (409); dropping it");
            SubmitDisposition::Dropped
        }
        // A 404 is a lost claim binding: reclaim once, resubmit the SAME body.
        Err(e) if e.http_status() == Some(404) => {
            tracing::warn!("the submission was not bound to a lease (404); reclaiming it once");
            match reclaim_once(ctx, job_id) {
                ReclaimOutcome::Renewed | ReclaimOutcome::Transient(_) => {
                    SubmitDisposition::RetryTransient
                }
                ReclaimOutcome::Unrecoverable => SubmitDisposition::Dropped,
            }
        }
        Err(e) if e.is_transient() => SubmitDisposition::RetryTransient,
        Err(e) => SubmitDisposition::Fatal(e),
    }
}

/// `POST /benchmarks` with the retry policy: 1 s doubling to a 30 s cap, 32
/// attempts (31 waits). `Ok(None)` is a result the server told us to drop;
/// exhausting the attempts is an error, since the measurement was taken.
pub(crate) fn submit_with_backoff<P: Serialize, S: FnMut(Duration)>(
    ctx: &Session,
    job_id: &str,
    payload: &P,
    mut sleep: S,
) -> anyhow::Result<Option<JobAccepted>> {
    let mut backoff = timing::SUBMIT_BACKOFF_BASE;
    for attempt in 1..=timing::SUBMIT_MAX_ATTEMPTS {
        match try_submit(ctx, job_id, payload) {
            SubmitDisposition::Accepted { job_id } => {
                return Ok(Some(JobAccepted { job_id }));
            }
            SubmitDisposition::Dropped => return Ok(None),
            SubmitDisposition::Fatal(e) => {
                return Err(describe(
                    e,
                    format!("submitting the result for job {job_id}"),
                ));
            }
            SubmitDisposition::RetryTransient => {
                if attempt == timing::SUBMIT_MAX_ATTEMPTS {
                    break;
                }
                tracing::warn!(
                    attempt = %attempt,
                    wait_secs = %backoff.as_secs(),
                    "submitting the result failed; retrying"
                );
                sleep(backoff);
                backoff = timing::next_backoff(backoff, timing::SUBMIT_BACKOFF_CAP);
            }
        }
    }
    Err(ServerError::SubmitGaveUp {
        job_id: job_id.to_owned(),
        attempts: timing::SUBMIT_MAX_ATTEMPTS,
    }
    .into())
}

/// `PATCH /clients/me` with the profile and capabilities this build advertises.
pub(crate) fn refresh_profile(
    ctx: &Session,
    client_details: Option<String>,
    device: DeviceProfileFields,
) -> anyhow::Result<ClientProfile> {
    let flags = device.capabilities.as_deref().unwrap_or_default();
    let build_flag_count = flags
        .iter()
        .filter(|flag| flag.starts_with(RUNTIME_BUILD_FLAG_PREFIX))
        .count();
    tracing::info!(
        capabilities = %flags.len(),
        runtime_build_flags = %build_flag_count,
        "refreshing the device profile"
    );
    client_api::update_me(
        &ctx.server,
        &ctx.http,
        &ctx.auth,
        // `capabilities` replaces the stored set wholesale, so send it complete.
        UpdateClientRequest {
            client_details,
            device,
        },
    )
    .map_err(|e| describe(e, "updating this client's server-side profile".to_string()))
}

pub(crate) fn fetch_profile(ctx: &Session) -> anyhow::Result<ClientProfile> {
    client_api::me(&ctx.server, &ctx.http, &ctx.auth)
        .map_err(|e| describe(e, "reading this client's server-side profile".to_string()))
}

#[derive(Debug)]
pub(crate) enum ClaimableWait {
    Claimable(Box<ClientProfile>),
    /// A signal arrived while waiting. Not a failure.
    Stopped,
}

/// Block until the client may claim: `status` Approved and `reindex_pending`
/// clear, within `crate::timing::APPROVAL_WAIT_BUDGET`.
///
/// Exhausting the budget is an error (unlike the CLI, which continues): an
/// unattended process says "nobody approved me" with a non-zero exit. The
/// budget counts polls and wall clock, so an injected no-op `sleep` ends and
/// the diagnostic's `after {}s` is true. `sleep` returning `false` means a
/// signal arrived.
pub(crate) fn wait_until_claimable<S: FnMut(Duration) -> bool>(
    ctx: &Session,
    mut sleep: S,
) -> anyhow::Result<ClaimableWait> {
    let mut waited = Duration::ZERO;
    let started = std::time::Instant::now();
    loop {
        let profile = fetch_profile(ctx)?;
        if profile.status == ClientStatus::Approved && !profile.reindex_pending {
            return Ok(ClaimableWait::Claimable(Box::new(profile)));
        }
        if waited >= timing::APPROVAL_WAIT_BUDGET
            || started.elapsed() >= timing::APPROVAL_WAIT_BUDGET
        {
            let secs = timing::APPROVAL_WAIT_BUDGET.as_secs();
            let client_id = profile.client_id.clone();
            return Err(if profile.status == ClientStatus::Approved {
                ServerError::ReindexPending { client_id, secs }
            } else {
                ServerError::NotApproved {
                    client_id,
                    status: profile.status.to_string(),
                    secs,
                }
            }
            .into());
        }
        let reason = if profile.status == ClientStatus::Approved {
            "waiting for the profile reindex to settle"
        } else {
            "waiting for this client to be approved"
        };
        tracing::info!(
            client_id = %profile.client_id,
            status = %profile.status,
            // The clock, not the count of intended sleeps.
            waited_secs = %started.elapsed().as_secs(),
            "{reason}"
        );
        if !sleep(timing::REINDEX_POLL_INTERVAL) {
            return Ok(ClaimableWait::Stopped);
        }
        waited += timing::REINDEX_POLL_INTERVAL;
    }
}

pub(crate) fn http_client() -> anyhow::Result<HttpClient> {
    // No overall request timeout, matching the CLI: the connect timeout bounds a
    // dead network. `pipette-mgmt-client` runs signed calls without redirects.
    HttpClient::new(USER_AGENT).context("building the HTTPS client for management calls")
}

/// Attach the operator hint a `401` (clock, replayed nonce, or unknown key) or a
/// `403` (not approved) deserves; every other status reads well as is.
fn describe(error: MgmtError, what: String) -> anyhow::Error {
    let hint = match error.http_status() {
        Some(401) => Some(
            "the server rejected this client's signature; check the host clock \
             (the timestamp must be within 5 minutes of the server's) and that this device's \
             `private_key` in the config is the one its `client_id` was registered with",
        ),
        Some(403) => Some(
            "this client is not approved; have an operator approve the phone's client row \
             (`pipette-mgmt clients approve <client_id>`)",
        ),
        _ => None,
    };
    let error = anyhow::Error::new(error);
    match hint {
        Some(hint) => error.context(format!("{what}: {hint}")),
        None => error.context(what),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use httpmock::Method::PATCH;
    use httpmock::prelude::*;

    use pipette_mgmt_client::public_key_hex_from_private;
    use pipette_mgmt_client::types::PrivateKey;
    use pipette_plan_types::ClientId;

    use super::*;
    use crate::worker::testing::profile_body;

    /// RFC 8032 test vector 1, whose public key is published.
    const SEED: &str = "9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60";
    const PUBLIC_KEY: &str = "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a";
    const CLIENT_ID: &str = "ev1_a3f8";

    /// The signed payload, spelled out: the bytes ARE the contract.
    const VECTOR_PAYLOAD: &str = "v1\nGET\n/clients/me?page=2\n2026-03-10T12:00:00Z\nev1_a3f8\n0f1e2d3c4b5a69788796a5b4c3d2e1f0";

    /// [`SEED`] signing [`VECTOR_PAYLOAD`], reproduced from a reference Ed25519.
    const VECTOR_SIGNATURE: &str = concat!(
        "943174b8c9e4318927e37ef05f34752225322762ac85355aac950b09c3450a7c",
        "504383c30b4e8d822a91e3bc8ec094751e642e124f0d25248cde35a50bc4a709"
    );

    /// One request's signed triple, as sent.
    #[derive(Debug, Clone)]
    struct Stamp {
        timestamp: String,
        nonce: String,
        signature: String,
    }

    fn identity() -> anyhow::Result<ClientIdentity> {
        Ok(ClientIdentity {
            client_id: ClientId::try_new(CLIENT_ID)
                .map_err(|e| anyhow::anyhow!("test client id: {e}"))?,
            private_key_hex: PrivateKey::try_new(SEED)
                .map_err(|e| anyhow::anyhow!("test private key: {e}"))?,
        })
    }

    fn ctx(server: &MockServer) -> anyhow::Result<Session> {
        ctx_at(&server.base_url())
    }

    fn ctx_at(base_url: &str) -> anyhow::Result<Session> {
        Ok(Session {
            server: ServerUrl::try_new(base_url)
                .map_err(|e| anyhow::anyhow!("test server url: {e}"))?,
            http: http_client()?,
            auth: identity()?,
        })
    }

    fn header(request: &HttpMockRequest, name: &str) -> Option<String> {
        request
            .headers()
            .get(name)?
            .to_str()
            .ok()
            .map(str::to_owned)
    }

    /// `X-Timestamp` as actually sent: `YYYY-MM-DDTHH:MM:SSZ`, optionally with
    /// fractional digits (`pipette_mgmt_client::RequestStamp` uses `time`'s
    /// `Rfc3339`), never a `+00:00` offset.
    fn is_rfc3339_utc_stamp(value: &str) -> bool {
        const SHAPE: &str = "0000-00-00T00:00:00";
        let Some(body) = value.strip_suffix('Z') else {
            return false;
        };
        let seconds = match body.split_once('.') {
            // A `.` with no digits after it is not a fraction.
            Some((seconds, fraction)) => {
                if fraction.is_empty() || !fraction.chars().all(|c| c.is_ascii_digit()) {
                    return false;
                }
                seconds
            }
            None => body,
        };
        seconds.len() == SHAPE.len()
            && seconds
                .chars()
                .zip(SHAPE.chars())
                .all(|(got, want)| match want {
                    '0' => got.is_ascii_digit(),
                    other => got == other,
                })
    }

    #[test]
    fn the_timestamp_shape_is_pinned_to_the_character() {
        assert!(is_rfc3339_utc_stamp("2026-09-13T16:19:37Z"));
        assert!(is_rfc3339_utc_stamp("2026-09-13T16:19:37.185747Z"));
        assert!(!is_rfc3339_utc_stamp("2026-09-13T16:19:37+00:00"));
        assert!(!is_rfc3339_utc_stamp("2026-09-13 16:19:37Z"));
        assert!(!is_rfc3339_utc_stamp("2026-09-13T16:19:37.Z"));
        assert!(!is_rfc3339_utc_stamp("2026-09-13T16:19:3Z"));
        assert!(!is_rfc3339_utc_stamp("TZ"));
        assert!(!is_rfc3339_utc_stamp(""));
    }

    fn is_lower_hex(value: &str, len: usize) -> bool {
        value.len() == len
            && value
                .chars()
                .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase())
    }

    /// The shapes the server checks, as a match condition. Hex matters: a nonce
    /// that could carry a `\n` could forge a field boundary in the payload.
    fn v1_signed_without_a_body(request: &HttpMockRequest) -> bool {
        let (Some(client_id), Some(timestamp), Some(nonce), Some(signature), Some(accept)) = (
            header(request, "X-Client-Id"),
            header(request, "X-Timestamp"),
            header(request, "X-Nonce"),
            header(request, "X-Signature"),
            header(request, "Accept"),
        ) else {
            return false;
        };
        client_id == CLIENT_ID
            && accept == "application/json"
            && is_rfc3339_utc_stamp(&timestamp)
            && is_lower_hex(&nonce, 32)
            && is_lower_hex(&signature, 128)
            && header(request, "Content-Type").is_none()
    }

    /// [`v1_signed_without_a_body`] plus a record of what was signed, into `log`.
    fn v1_signed_recording_into(request: &HttpMockRequest, log: &Mutex<Vec<Stamp>>) -> bool {
        if let (Some(timestamp), Some(nonce), Some(signature), Ok(mut seen)) = (
            header(request, "X-Timestamp"),
            header(request, "X-Nonce"),
            header(request, "X-Signature"),
            log.lock(),
        ) {
            seen.push(Stamp {
                timestamp,
                nonce,
                signature,
            });
        }
        v1_signed_without_a_body(request)
    }

    /// The 32 bytes of a 64-hex key, or an error naming the value.
    fn key_bytes(hex_value: &str) -> anyhow::Result<[u8; 32]> {
        hex::decode(hex_value)
            .with_context(|| format!("`{hex_value}` is not hex"))?
            .try_into()
            .map_err(|raw: Vec<u8>| anyhow::anyhow!("expected 32 bytes, got {}", raw.len()))
    }

    /// Does `signature_hex` really cover `payload`, under [`PUBLIC_KEY`]?
    fn verify_v1(payload: &str, signature_hex: &str) -> anyhow::Result<()> {
        let verifying = ed25519_dalek::VerifyingKey::from_bytes(&key_bytes(PUBLIC_KEY)?)
            .context("the pinned public key is not a valid Ed25519 point")?;
        let raw: [u8; 64] = hex::decode(signature_hex)
            .context("X-Signature is not hex")?
            .try_into()
            .map_err(|raw: Vec<u8>| {
                anyhow::anyhow!("X-Signature is {} bytes, not 64", raw.len())
            })?;
        ed25519_dalek::Verifier::verify(
            &verifying,
            payload.as_bytes(),
            &ed25519_dalek::Signature::from_bytes(&raw),
        )
        .with_context(|| format!("the signature sent does not cover `{payload}`"))
    }

    /// The same vector, through the derivation `pipette-mgmt-client` uses.
    #[test]
    fn a_known_seed_derives_the_pinned_public_key() -> anyhow::Result<()> {
        let private_key =
            PrivateKey::try_new(SEED).map_err(|e| anyhow::anyhow!("test private key: {e}"))?;
        let public_key = public_key_hex_from_private(&private_key)
            .context("deriving the public key from the pinned seed")?;
        assert_eq!(public_key.as_ref(), PUBLIC_KEY);
        Ok(())
    }

    #[test]
    fn the_v1_payload_and_its_signature_are_the_pinned_cross_language_vector() -> anyhow::Result<()>
    {
        // A trailing newline would be a seventh, empty field.
        assert_eq!(VECTOR_PAYLOAD.split('\n').count(), 6);
        assert_eq!(VECTOR_PAYLOAD.matches('\n').count(), 5);
        assert!(!VECTOR_PAYLOAD.ends_with('\n'));
        assert_eq!(
            VECTOR_PAYLOAD.split('\n').collect::<Vec<_>>(),
            vec![
                "v1",
                "GET",
                "/clients/me?page=2",
                "2026-03-10T12:00:00Z",
                CLIENT_ID,
                "0f1e2d3c4b5a69788796a5b4c3d2e1f0",
            ]
        );

        let signing = ed25519_dalek::SigningKey::from_bytes(&key_bytes(SEED)?);
        assert_eq!(hex::encode(signing.verifying_key().to_bytes()), PUBLIC_KEY);
        let signature: ed25519_dalek::Signature =
            ed25519_dalek::Signer::sign(&signing, VECTOR_PAYLOAD.as_bytes());
        assert_eq!(hex::encode(signature.to_bytes()), VECTOR_SIGNATURE);
        Ok(())
    }

    #[test]
    fn claims_are_signed_by_pipette_mgmt_client_with_a_fresh_nonce() -> anyhow::Result<()> {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let server = MockServer::start();
        let log = Arc::clone(&seen);
        let mock = server.mock(|when, then| {
            when.method(POST)
                .path("/plans/claim")
                .is_true(move |request| v1_signed_recording_into(request, &log));
            then.status(204);
        });
        let ctx = ctx(&server)?;

        assert!(matches!(poll_claim(&ctx), ClaimPoll::Idle));
        assert!(matches!(poll_claim(&ctx), ClaimPoll::Idle));
        mock.assert_calls(2);

        let stamps = seen
            .lock()
            .map_err(|_| anyhow::anyhow!("the stamp log was poisoned"))?;
        assert_eq!(stamps.len(), 2, "{stamps:?}");
        assert_ne!(
            stamps[0].nonce, stamps[1].nonce,
            "a nonce must not be reused"
        );

        stamps.iter().try_for_each(|stamp| {
            verify_v1(
                &format!(
                    "v1\nPOST\n/plans/claim\n{}\n{CLIENT_ID}\n{}",
                    stamp.timestamp, stamp.nonce
                ),
                &stamp.signature,
            )
        })
    }

    #[test]
    fn a_path_prefix_on_the_base_url_reaches_the_server() -> anyhow::Result<()> {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let server = MockServer::start();
        let log = Arc::clone(&seen);
        let mock = server.mock(|when, then| {
            when.method(POST)
                .path("/api/plans/claim")
                .is_true(move |request| v1_signed_recording_into(request, &log));
            then.status(204);
        });
        let ctx = ctx_at(&format!("{}/api/", server.base_url()))?;

        assert!(matches!(poll_claim(&ctx), ClaimPoll::Idle));
        mock.assert();

        // A signature over the bare `/plans/claim` would 401 every request.
        let stamps = seen
            .lock()
            .map_err(|_| anyhow::anyhow!("the stamp log was poisoned"))?;
        let stamp = stamps.first().context("the mock recorded no request")?;
        verify_v1(
            &format!(
                "v1\nPOST\n/api/plans/claim\n{}\n{CLIENT_ID}\n{}",
                stamp.timestamp, stamp.nonce
            ),
            &stamp.signature,
        )
    }

    #[rstest::rstest]
    #[case::not_approved(403, "NotApproved")]
    #[case::signature(401, "SignatureRejected")]
    #[case::server_error(503, "Transient")]
    #[case::unexpected(400, "Fatal")]
    fn claim_statuses_map_to_their_dispositions(
        #[case] status: u16,
        #[case] want: &str,
    ) -> anyhow::Result<()> {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(POST).path("/plans/claim");
            then.status(status);
        });
        let got = match poll_claim(&ctx(&server)?) {
            ClaimPoll::Job(_) => "Job",
            ClaimPoll::Idle => "Idle",
            ClaimPoll::NotApproved => "NotApproved",
            ClaimPoll::SignatureRejected => "SignatureRejected",
            ClaimPoll::Transient(_) => "Transient",
            ClaimPoll::Fatal(_) => "Fatal",
        };
        assert_eq!(got, want);
        Ok(())
    }

    /// The lookup is signed as the phone.
    #[rstest::rstest]
    #[case::known(200, false)]
    #[case::unknown(404, true)]
    #[case::signature(401, false)]
    #[case::server_error(503, false)]
    fn only_a_404_makes_a_benchmark_unknown(
        #[case] status: u16,
        #[case] unknown: bool,
    ) -> anyhow::Result<()> {
        let server = MockServer::start();
        let lookup = server.mock(|when, then| {
            when.method(GET)
                .path("/benchmarks/decode_throughput_256")
                .is_true(v1_signed_without_a_body);
            then.status(status).json_body(serde_json::json!({
                "benchmark_id": "decode_throughput_256",
                "benchmark_type": "decode_throughput",
            }));
        });

        assert_eq!(
            benchmark_is_unknown(&ctx(&server)?, "decode_throughput_256"),
            unknown
        );
        lookup.assert();
        Ok(())
    }

    #[test]
    fn a_claimed_job_arrives_typed_and_permissively() -> anyhow::Result<()> {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(POST).path("/plans/claim");
            then.status(200).json_body(serde_json::json!({
                "job_id": "job-abc",
                "benchmark_id": "prefill_throughput_256",
                "time_window": "PT10M",
                "expires_at": "not-a-timestamp",
                "spec": { "benchmark": "prefill_throughput_256" },
                "future_field": true,
            }));
        });
        match poll_claim(&ctx(&server)?) {
            ClaimPoll::Job(job) => {
                assert_eq!(job.job_id, "job-abc");
                assert_eq!(job.time_window, "PT10M");
                assert!(job.expires_at.is_none());
            }
            other @ (ClaimPoll::Idle
            | ClaimPoll::NotApproved
            | ClaimPoll::SignatureRejected
            | ClaimPoll::Transient(_)
            | ClaimPoll::Fatal(_)) => anyhow::bail!("expected a job, got {other:?}"),
        }
        Ok(())
    }

    #[rstest::rstest]
    #[case::renewed(200, None, "Ok")]
    #[case::taken_away(409, None, "Abort")]
    #[case::reclaimed(404, Some(200), "Ok")]
    #[case::reclaim_refused(404, Some(409), "Abort")]
    #[case::reclaim_missing(404, Some(404), "Abort")]
    #[case::reclaim_transient(404, Some(503), "Retry")]
    #[case::transient(500, None, "Retry")]
    // A status the lease protocol does not define must not kill a live run.
    #[case::undefined(418, None, "Retry")]
    fn heartbeat_outcomes(
        #[case] heartbeat_status: u16,
        #[case] reclaim_status: Option<u16>,
        #[case] want: &str,
    ) -> anyhow::Result<()> {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(PUT).path("/plans/job-abc/heartbeat");
            then.status(heartbeat_status);
        });
        let reclaim = reclaim_status.map(|status| {
            server.mock(|when, then| {
                when.method(POST).path("/plans/job-abc/reclaim");
                then.status(status);
            })
        });

        let got = match keepalive_lease(&ctx(&server)?, "job-abc") {
            LeaseKeepalive::Ok => "Ok",
            LeaseKeepalive::Abort => "Abort",
            LeaseKeepalive::Retry => "Retry",
        };
        assert_eq!(got, want);
        if let Some(reclaim) = reclaim {
            reclaim.assert_calls(1);
        }
        Ok(())
    }

    #[rstest::rstest]
    #[case::renewed(200, "Renewed")]
    #[case::refused(409, "Unrecoverable")]
    #[case::missing(404, "Unrecoverable")]
    #[case::transient(503, "Transient")]
    fn reclaim_reports_whether_the_lease_came_back(
        #[case] status: u16,
        #[case] want: &str,
    ) -> anyhow::Result<()> {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(POST).path("/plans/job-abc/reclaim");
            then.status(status);
        });
        let got = match reclaim_once(&ctx(&server)?, "job-abc") {
            ReclaimOutcome::Renewed => "Renewed",
            ReclaimOutcome::Unrecoverable => "Unrecoverable",
            ReclaimOutcome::Transient(_) => "Transient",
        };
        assert_eq!(got, want);
        Ok(())
    }

    #[test]
    fn a_submitted_result_carries_a_body_and_its_content_type() -> anyhow::Result<()> {
        let server = MockServer::start();
        let mock = server.mock(|when, then| {
            when.method(POST)
                .path("/benchmarks")
                .header("Content-Type", "application/json")
                .json_body_includes(r#"{"job_id": "job-abc"}"#);
            then.status(202)
                .json_body(serde_json::json!({ "job_id": "job-abc" }));
        });

        let accepted = submit_with_backoff(
            &ctx(&server)?,
            "job-abc",
            &serde_json::json!({ "job_id": "job-abc", "benchmark_id": "b" }),
            |_| (),
        )?;

        assert_eq!(
            accepted.map(|response| response.job_id).as_deref(),
            Some("job-abc")
        );
        mock.assert();
        Ok(())
    }

    #[test]
    fn a_refused_result_is_dropped_without_a_retry() -> anyhow::Result<()> {
        let server = MockServer::start();
        let mock = server.mock(|when, then| {
            when.method(POST).path("/benchmarks");
            then.status(409);
        });

        let dropped =
            submit_with_backoff(&ctx(&server)?, "job-abc", &serde_json::json!({}), |_| ())?;

        assert!(dropped.is_none());
        mock.assert_calls(1);
        Ok(())
    }

    #[test]
    fn a_submit_404_reclaims_once_and_resubmits() -> anyhow::Result<()> {
        let server = MockServer::start();
        let submit = server.mock(|when, then| {
            when.method(POST).path("/benchmarks");
            then.status(404);
        });
        let reclaim = server.mock(|when, then| {
            when.method(POST).path("/plans/job-abc/reclaim");
            then.status(200);
        });

        let outcome =
            submit_with_backoff(&ctx(&server)?, "job-abc", &serde_json::json!({}), |_| ());

        assert!(
            outcome.is_err(),
            "a permanent 404 must exhaust the attempts"
        );
        submit.assert_calls(timing::SUBMIT_MAX_ATTEMPTS as usize);
        reclaim.assert_calls(timing::SUBMIT_MAX_ATTEMPTS as usize);
        Ok(())
    }

    #[test]
    fn submit_backoff_doubles_to_the_cap_and_gives_up() -> anyhow::Result<()> {
        let server = MockServer::start();
        let mock = server.mock(|when, then| {
            when.method(POST).path("/benchmarks");
            then.status(503);
        });

        let mut waits = Vec::new();
        let error =
            submit_with_backoff(&ctx(&server)?, "job-abc", &serde_json::json!({}), |wait| {
                waits.push(wait.as_secs())
            })
            .err()
            .ok_or_else(|| anyhow::anyhow!("32 transient failures must be an error"))?;

        assert!(
            format!("{error:#}").contains("after 32 attempts"),
            "{error:#}"
        );
        mock.assert_calls(timing::SUBMIT_MAX_ATTEMPTS as usize);
        assert_eq!(waits.len(), timing::SUBMIT_MAX_ATTEMPTS as usize - 1);
        assert_eq!(&waits[..6], &[1, 2, 4, 8, 16, 30]);
        assert!(waits.iter().all(|secs| *secs <= 30));
        Ok(())
    }

    #[test]
    fn a_profile_refresh_patches_the_capabilities_wholesale() -> anyhow::Result<()> {
        let server = MockServer::start();
        let mock = server.mock(|when, then| {
            when.method(PATCH)
                .path("/clients/me")
                .json_body_includes(r#"{"device_form_factor": "phone"}"#)
                .body_includes(r#""capabilities":["runtime:mlx"]"#);
            then.status(200).json_body(profile_body("approved", false));
        });

        let profile = refresh_profile(
            &ctx(&server)?,
            Some("relaying iPhone16,1".to_string()),
            DeviceProfileFields {
                device_form_factor: Some("phone".to_string()),
                capabilities: Some(vec!["runtime:mlx".to_string()]),
                ..DeviceProfileFields::default()
            },
        )?;

        assert_eq!(profile.status, ClientStatus::Approved);
        mock.assert();
        Ok(())
    }

    #[test]
    fn an_approved_client_is_claimable_at_once() -> anyhow::Result<()> {
        let server = MockServer::start();
        let mock = server.mock(|when, then| {
            when.method(GET).path("/clients/me");
            then.status(200).json_body(profile_body("approved", false));
        });

        let profile = claimable(wait_until_claimable(&ctx(&server)?, |_| true)?)?;

        assert_eq!(profile.status, ClientStatus::Approved);
        mock.assert_calls(1);
        Ok(())
    }

    fn claimable(waited: ClaimableWait) -> anyhow::Result<ClientProfile> {
        match waited {
            ClaimableWait::Claimable(profile) => Ok(*profile),
            ClaimableWait::Stopped => {
                anyhow::bail!("the gate reported a signal where it should have lifted")
            }
        }
    }

    #[test]
    fn a_signal_during_the_wait_stops_the_gate_rather_than_spinning() -> anyhow::Result<()> {
        let server = MockServer::start();
        let me = server.mock(|when, then| {
            when.method(GET).path("/clients/me");
            then.status(200).json_body(profile_body("pending", false));
        });

        // What `timing::sleep_interruptible` degrades to once the flag is set.
        let waited = wait_until_claimable(&ctx(&server)?, |_| false)?;

        assert!(
            matches!(waited, ClaimableWait::Stopped),
            "a signalled gate must report the signal, not fail the budget"
        );
        me.assert_calls(1);
        Ok(())
    }

    #[rstest::rstest]
    #[case::pending("pending", false, "approve it with")]
    #[case::unknown("quarantined", false, "quarantined")]
    #[case::reindexing("approved", true, "reindex is still pending")]
    fn an_unclaimable_client_fails_the_budget_with_its_reason(
        #[case] status: &str,
        #[case] reindex_pending: bool,
        #[case] wanted_text: &str,
    ) -> anyhow::Result<()> {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(GET).path("/clients/me");
            then.status(200)
                .json_body(profile_body(status, reindex_pending));
        });

        let error = wait_until_claimable(&ctx(&server)?, |_| true)
            .err()
            .ok_or_else(|| anyhow::anyhow!("an unclaimable client must not return a profile"))?;

        assert!(format!("{error:#}").contains(wanted_text), "{error:#}");
        Ok(())
    }

    #[rstest::rstest]
    #[case::signature(401, "check the host clock")]
    #[case::not_approved(403, "not approved")]
    fn profile_reads_surface_the_operator_hint(
        #[case] status: u16,
        #[case] wanted_text: &str,
    ) -> anyhow::Result<()> {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(GET).path("/clients/me");
            then.status(status);
        });

        let error = fetch_profile(&ctx(&server)?)
            .err()
            .ok_or_else(|| anyhow::anyhow!("a {status} must fail the profile read"))?;

        assert!(format!("{error:#}").contains(wanted_text), "{error:#}");
        Ok(())
    }
}
