//! The planner wire protocol: claim a job, keep its lease alive, and submit
//! the outcome. Each step maps to one management-server endpoint and reports
//! back as an enum the loop matches on, so retry policy stays with the caller.

use std::time::Duration;

use time::format_description::well_known::Rfc3339;
use time::OffsetDateTime;

use pipette_http::HttpClient;
use pipette_mgmt_client::{
    types::{ClaimedJob, FailureSubmission},
    AuthIdentity, MgmtClient,
};

use crate::client::claim::UnrunnableClaim;
use crate::error::{Error, Result};

/// Bind a success submission payload (serialized
/// [`pipette_plan_types::result::BenchmarkSubmissionPayload`]) to the lease it came from by
/// stamping the claim's `job_id`.
///
/// Nothing else is overlaid. The payload's descriptors and flags come from the
/// run, which serialized the same `ClientRunSpec` values the claim carried — so
/// they already are the echo the server's claim-binding check looks for, in the
/// form with the local HuggingFace token stripped.
pub fn attach_claim_to_success_payload(
    payload: serde_json::Value,
    job: &ClaimedJob,
) -> Result<serde_json::Value> {
    let mut value = payload;
    let obj = value
        .as_object_mut()
        .ok_or_else(|| Error::Other(anyhow::anyhow!("submission payload is not a JSON object")))?;
    obj.insert(
        "job_id".into(),
        serde_json::Value::String(job.job_id.clone()),
    );
    Ok(value)
}

/// Build a plan-attached failure submission that echoes the claim.
pub fn failure_from_claim(
    job: &ClaimedJob,
    reason: impl Into<String>,
    retriable: bool,
) -> FailureSubmission {
    FailureSubmission::from_claim(job, reason, retriable, crate::CLIENT_VERSION)
}

/// Classify a run error into retriable vs. non-retriable for the failure
/// submission. Defaults to retriable (safe); only clearly unworkable jobs are
/// non-retriable (client-integration §5).
pub fn classify_run_error(err: &anyhow::Error) -> bool {
    // A claim that could not be read is permanent by construction — no need to
    // recognize it by its wording.
    if err.chain().any(|cause| cause.is::<UnrunnableClaim>()) {
        return false;
    }
    let msg = format!("{err:#}").to_ascii_lowercase();
    // Permanent: missing benchmark, on-device-only runtime. Matching on wording
    // decides that a job can never run anywhere, so the list stays narrow — the
    // claim-shaped markers that used to live here are now [`UnrunnableClaim`],
    // caught above by type.
    let non_retriable_markers = [
        "benchmark not found",
        "is not a known, valid definition",
        "runs on-device only",
    ];
    // Everything else (OOM, thermal, transient fetch) stays retriable.
    let looks_permanent = non_retriable_markers.iter().any(|m| msg.contains(m));
    !looks_permanent
}

/// Format a failure_reason with a leading UTC timestamp, matching the
/// convention in the client-integration examples.
pub fn format_failure_reason(detail: &str) -> String {
    let ts = OffsetDateTime::now_utc()
        .format(&Rfc3339)
        .unwrap_or_else(|_| "unknown-time".into());
    format!("[{ts}] {detail}")
}

// ---------------------------------------------------------------------------
// Claim / heartbeat / submit protocol decisions (client-integration §3–4)
// ---------------------------------------------------------------------------

/// One `POST /plans/claim` attempt, classified for the worker state machine.
#[derive(Debug)]
pub enum ClaimPoll {
    /// A job was leased (`200`).
    Job(Box<ClaimedJob>),
    /// No work right now (`204`) — idle-wait then retry.
    Idle,
    /// Client is pending (`403`) — stop the worker.
    NotApproved,
    /// Transient failure (`5xx` / network) — backoff and retry.
    Transient(pipette_mgmt_client::Error),
    /// Definitive unexpected error — surface and stop.
    Fatal(pipette_mgmt_client::Error),
}

/// Classify a single claim attempt (no retry loop).
pub fn poll_claim(client: &MgmtClient, http: &HttpClient, auth: &AuthIdentity) -> ClaimPoll {
    match client.claim(http, auth) {
        Ok(Some(job)) => ClaimPoll::Job(Box::new(job)),
        Ok(None) => ClaimPoll::Idle,
        Err(e) if e.http_status() == Some(403) => ClaimPoll::NotApproved,
        Err(e) if e.is_transient() => ClaimPoll::Transient(e),
        Err(e) => ClaimPoll::Fatal(e),
    }
}

/// Outcome of one heartbeat tick (client-integration §3–4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LeaseKeepalive {
    /// Lease renewed (or reclaimed after a 404).
    Ok,
    /// Lease is gone for good (`409`, or reclaim `404`/`409`) — abort the run.
    Abort,
    /// Transient failure — retry soon; do not abort yet.
    Retry,
}

/// One heartbeat, with reclaim recovery on `404`.
pub fn keepalive_lease(
    client: &MgmtClient,
    http: &HttpClient,
    auth: &AuthIdentity,
    job_id: &str,
) -> LeaseKeepalive {
    match client.heartbeat(http, auth, job_id) {
        Ok(()) => LeaseKeepalive::Ok,
        Err(e) if e.http_status() == Some(409) => LeaseKeepalive::Abort,
        Err(e) if e.http_status() == Some(404) => match client.reclaim(http, auth, job_id) {
            Ok(()) => LeaseKeepalive::Ok,
            Err(re) if matches!(re.http_status(), Some(404 | 409)) => LeaseKeepalive::Abort,
            Err(re) if re.is_transient() => LeaseKeepalive::Retry,
            Err(_) => LeaseKeepalive::Abort,
        },
        Err(e) if e.is_transient() => LeaseKeepalive::Retry,
        // Non-transient, non-404/409: retry rather than aborting a live run on
        // an unexpected 4xx the protocol doesn't define.
        Err(_) => LeaseKeepalive::Retry,
    }
}

/// Outcome of submitting a plan-attached result (success or failure body).
#[derive(Debug)]
pub enum SubmitDisposition {
    /// Server accepted (`202`).
    Accepted { job_id: String },
    /// Result must be dropped (`409`, or reclaim after `404` also failed).
    Dropped,
    /// Transient failure — caller should backoff and call again with the same body.
    RetryTransient,
    /// Unexpected definitive error.
    Fatal(pipette_mgmt_client::Error),
}

/// One submit attempt for a plan-attached body. On `404`, tries reclaim once
/// and returns [`SubmitDisposition::RetryTransient`] so the caller resubmits
/// (claim-binding is restored). Does not loop — the CLI owns the backoff.
pub fn try_submit_plan_result<P: serde::Serialize>(
    client: &MgmtClient,
    http: &HttpClient,
    auth: &AuthIdentity,
    job_id: &str,
    payload: &P,
) -> SubmitDisposition {
    match client.submit_result(http, auth, payload) {
        Ok(resp) => SubmitDisposition::Accepted {
            job_id: resp.job_id,
        },
        Err(e) if e.http_status() == Some(409) => SubmitDisposition::Dropped,
        Err(e) if e.http_status() == Some(404) => match client.reclaim(http, auth, job_id) {
            Ok(()) => SubmitDisposition::RetryTransient,
            Err(re) if matches!(re.http_status(), Some(404 | 409)) => SubmitDisposition::Dropped,
            Err(re) if re.is_transient() => SubmitDisposition::RetryTransient,
            Err(_) => SubmitDisposition::Dropped,
        },
        Err(e) if e.is_transient() => SubmitDisposition::RetryTransient,
        Err(e) => SubmitDisposition::Fatal(e),
    }
}

/// Drive [`try_submit_plan_result`] with exponential backoff until a terminal
/// disposition. `sleep` is injected so tests can no-op the waits.
pub fn submit_plan_result_with_backoff<P, S>(
    client: &MgmtClient,
    http: &HttpClient,
    auth: &AuthIdentity,
    job_id: &str,
    payload: &P,
    mut sleep: S,
) -> Result<SubmitDisposition>
where
    P: serde::Serialize,
    S: FnMut(Duration),
{
    let mut backoff = Duration::from_secs(1);
    const BACKOFF_CAP: Duration = Duration::from_secs(30);
    // Bound retries so a permanently-broken network can't spin forever in tests
    // or production without an external interrupt.
    for _ in 0..32 {
        match try_submit_plan_result(client, http, auth, job_id, payload) {
            SubmitDisposition::RetryTransient => {
                sleep(backoff);
                backoff = (backoff * 2).min(BACKOFF_CAP);
            }
            other => return Ok(other),
        }
    }
    Ok(SubmitDisposition::RetryTransient)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn attach_claim_stamps_the_job_id_and_leaves_the_run_payload_alone() -> anyhow::Result<()> {
        let job: ClaimedJob = serde_json::from_str(CLAIM_JOB_BODY)?;
        let payload = serde_json::json!({
            "benchmark_id": "prefill_throughput_256",
            "model_descriptor": "{\"type\":\"gguf_text\",\"other\":1}",
            "prefill_time_ms": 1.0
        });
        let out = attach_claim_to_success_payload(payload, &job)?;
        assert_eq!(out["job_id"], "job-abc");
        // The run's descriptor stands: it is the token-stripped serialization of
        // the same spec the claim carried.
        assert_eq!(out["model_descriptor"], r#"{"type":"gguf_text","other":1}"#);
        assert_eq!(out["prefill_time_ms"], 1.0);
        Ok(())
    }

    /// The whole chain a poison-pill job takes: it decodes, the runner refuses
    /// it, the refusal classifies as terminal, and the server learns which job
    /// died. Driven from a real bad payload rather than a hand-built error, so
    /// it breaks if the runner stops returning [`UnrunnableClaim`].
    #[test]
    fn an_unreadable_claim_is_reported_as_a_terminal_failure() -> anyhow::Result<()> {
        let mut body: serde_json::Value = serde_json::from_str(CLAIM_JOB_BODY)?;
        body.as_object_mut()
            .ok_or_else(|| anyhow::anyhow!("the fixture is an object"))?
            .insert("spec".into(), serde_json::json!({ "model": "not-a-model" }));
        let job: ClaimedJob = serde_json::from_value(body)?;

        let err = crate::client::claim::run_spec_from_claim(&job)
            .err()
            .ok_or_else(|| anyhow::anyhow!("an unreadable spec cannot produce a cell"))?;
        let retriable = classify_run_error(&err);
        assert!(!retriable, "an unreadable claim must not be retried");

        let wire = serde_json::to_value(failure_from_claim(&job, "[ts] bad spec", retriable))?;
        assert_eq!(wire["job_id"], "job-abc");
        assert_eq!(wire["benchmark_id"], "prefill_throughput_256");
        assert_eq!(wire["retriable"], false);
        Ok(())
    }

    #[test]
    fn classify_run_error_defaults_retriable() {
        assert!(classify_run_error(&anyhow::anyhow!("out of memory")));
        assert!(classify_run_error(&anyhow::anyhow!("connection reset")));
        assert!(!classify_run_error(&anyhow::anyhow!(
            "runtime `foo` runs on-device only and is not runnable from the CLI"
        )));
        assert!(!classify_run_error(&anyhow::anyhow!(
            "benchmark 'x' is not a known, valid definition: missing field"
        )));
    }

    // -----------------------------------------------------------------------
    // Protocol integration tests — httpmock covering claim / heartbeat /
    // reclaim / submit dispositions from client-integration.md.
    // -----------------------------------------------------------------------

    fn test_http() -> anyhow::Result<HttpClient> {
        Ok(HttpClient::new("pipette-test/0")?)
    }

    fn test_auth() -> anyhow::Result<AuthIdentity> {
        let (private_key_hex, _) = pipette_mgmt_client::generate_keypair_hex()?;
        Ok(AuthIdentity {
            client_id: "ev1_test".into(),
            private_key_hex,
        })
    }

    const CLAIM_JOB_BODY: &str = r#"{
        "job_id": "job-abc",
        "benchmark_id": "prefill_throughput_256",
        "time_window": "PT10M",
        "model_name": "m",
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
        }
    }"#;

    #[test]
    fn poll_claim_classifies_200() -> anyhow::Result<()> {
        use httpmock::prelude::*;
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(POST).path("/plans/claim");
            then.status(200).body(CLAIM_JOB_BODY);
        });
        let client = MgmtClient::new(server.base_url())?;
        let http = test_http()?;
        let auth = test_auth()?;
        match poll_claim(&client, &http, &auth) {
            ClaimPoll::Job(job) => {
                assert_eq!(job.job_id, "job-abc");
                assert_eq!(job.time_window, "PT10M");
            }
            other => anyhow::bail!("expected Job, got {other:?}"),
        }
        Ok(())
    }

    #[test]
    fn poll_claim_classifies_204_idle() -> anyhow::Result<()> {
        use httpmock::prelude::*;
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(POST).path("/plans/claim");
            then.status(204);
        });
        let client = MgmtClient::new(server.base_url())?;
        let http = test_http()?;
        assert!(matches!(
            poll_claim(&client, &http, &test_auth()?),
            ClaimPoll::Idle
        ));
        Ok(())
    }

    #[test]
    fn poll_claim_classifies_403_not_approved() -> anyhow::Result<()> {
        use httpmock::prelude::*;
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(POST).path("/plans/claim");
            then.status(403).body("pending");
        });
        let client = MgmtClient::new(server.base_url())?;
        let http = test_http()?;
        assert!(matches!(
            poll_claim(&client, &http, &test_auth()?),
            ClaimPoll::NotApproved
        ));
        Ok(())
    }

    #[test]
    fn keepalive_renews_on_200_and_aborts_on_409() -> anyhow::Result<()> {
        use httpmock::prelude::*;
        let server = MockServer::start();
        let ok = server.mock(|when, then| {
            when.method(PUT).path("/plans/job-abc/heartbeat");
            then.status(200);
        });
        let client = MgmtClient::new(server.base_url())?;
        let http = test_http()?;
        let auth = test_auth()?;
        assert!(matches!(
            keepalive_lease(&client, &http, &auth, "job-abc"),
            LeaseKeepalive::Ok
        ));
        ok.assert();

        let server2 = MockServer::start();
        server2.mock(|when, then| {
            when.method(PUT).path("/plans/job-abc/heartbeat");
            then.status(409).body("superseded");
        });
        let client2 = MgmtClient::new(server2.base_url())?;
        assert!(matches!(
            keepalive_lease(&client2, &http, &auth, "job-abc"),
            LeaseKeepalive::Abort
        ));
        Ok(())
    }

    #[test]
    fn keepalive_reclaims_after_heartbeat_404() -> anyhow::Result<()> {
        use httpmock::prelude::*;
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(PUT).path("/plans/job-abc/heartbeat");
            then.status(404).body("gone");
        });
        server.mock(|when, then| {
            when.method(POST).path("/plans/job-abc/reclaim");
            then.status(200);
        });
        let client = MgmtClient::new(server.base_url())?;
        let http = test_http()?;
        assert!(matches!(
            keepalive_lease(&client, &http, &test_auth()?, "job-abc"),
            LeaseKeepalive::Ok
        ));
        Ok(())
    }

    #[test]
    fn keepalive_aborts_when_reclaim_also_404() -> anyhow::Result<()> {
        use httpmock::prelude::*;
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(PUT).path("/plans/job-abc/heartbeat");
            then.status(404).body("gone");
        });
        server.mock(|when, then| {
            when.method(POST).path("/plans/job-abc/reclaim");
            then.status(404).body("still gone");
        });
        let client = MgmtClient::new(server.base_url())?;
        let http = test_http()?;
        assert!(matches!(
            keepalive_lease(&client, &http, &test_auth()?, "job-abc"),
            LeaseKeepalive::Abort
        ));
        Ok(())
    }

    #[test]
    fn keepalive_retries_on_transient_heartbeat() -> anyhow::Result<()> {
        use httpmock::prelude::*;
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(PUT).path("/plans/job-abc/heartbeat");
            then.status(503).body("down");
        });
        let client = MgmtClient::new(server.base_url())?;
        let http = test_http()?;
        assert!(matches!(
            keepalive_lease(&client, &http, &test_auth()?, "job-abc"),
            LeaseKeepalive::Retry
        ));
        Ok(())
    }

    #[test]
    fn try_submit_accepts_202() -> anyhow::Result<()> {
        use httpmock::prelude::*;
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(POST).path("/benchmarks");
            then.status(202).body(r#"{"job_id":"job-abc"}"#);
        });
        let client = MgmtClient::new(server.base_url())?;
        let http = test_http()?;
        let body = serde_json::json!({"job_id": "job-abc"});
        match try_submit_plan_result(&client, &http, &test_auth()?, "job-abc", &body) {
            SubmitDisposition::Accepted { job_id } => assert_eq!(job_id, "job-abc"),
            other => anyhow::bail!("expected Accepted, got {other:?}"),
        }
        Ok(())
    }

    #[test]
    fn try_submit_drops_on_409() -> anyhow::Result<()> {
        use httpmock::prelude::*;
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(POST).path("/benchmarks");
            then.status(409).body("superseded");
        });
        let client = MgmtClient::new(server.base_url())?;
        let http = test_http()?;
        assert!(matches!(
            try_submit_plan_result(
                &client,
                &http,
                &test_auth()?,
                "job-abc",
                &serde_json::json!({})
            ),
            SubmitDisposition::Dropped
        ));
        Ok(())
    }

    #[test]
    fn try_submit_reclaims_then_asks_for_retry_on_404() -> anyhow::Result<()> {
        use httpmock::prelude::*;
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(POST).path("/benchmarks");
            then.status(404).body("gone");
        });
        server.mock(|when, then| {
            when.method(POST).path("/plans/job-abc/reclaim");
            then.status(200);
        });
        let client = MgmtClient::new(server.base_url())?;
        let http = test_http()?;
        assert!(matches!(
            try_submit_plan_result(
                &client,
                &http,
                &test_auth()?,
                "job-abc",
                &serde_json::json!({})
            ),
            SubmitDisposition::RetryTransient
        ));
        Ok(())
    }

    #[test]
    fn submit_with_backoff_retries_then_accepts() -> anyhow::Result<()> {
        use httpmock::prelude::*;
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(POST).path("/benchmarks");
            then.status(202).body(r#"{"job_id":"job-abc"}"#);
        });
        let client = MgmtClient::new(server.base_url())?;
        let http = test_http()?;
        let auth = test_auth()?;
        let mut sleeps = 0;
        let disposition = submit_plan_result_with_backoff(
            &client,
            &http,
            &auth,
            "job-abc",
            &serde_json::json!({"job_id": "job-abc"}),
            |_| sleeps += 1,
        )?;
        match disposition {
            SubmitDisposition::Accepted { job_id } => assert_eq!(job_id, "job-abc"),
            other => anyhow::bail!("expected Accepted, got {other:?}"),
        }
        assert_eq!(sleeps, 0, "no backoff when first submit succeeds");
        Ok(())
    }

    #[test]
    fn failure_submission_identifies_the_job() -> anyhow::Result<()> {
        let job: ClaimedJob = serde_json::from_str(CLAIM_JOB_BODY)?;
        let failure = failure_from_claim(&job, "[ts] oom", true);
        let wire = serde_json::to_value(&failure)?;
        assert_eq!(wire["message_type"], "failure");
        assert_eq!(wire["job_id"], "job-abc");
        assert_eq!(wire["benchmark_id"], "prefill_throughput_256");
        assert_eq!(wire["retriable"], true);
        assert_eq!(wire["failure_reason"], "[ts] oom");
        // The server cannot recover this from the job body — a failure that
        // omitted it would leave "which build reported this" unanswerable.
        assert_eq!(wire["client_version"], crate::CLIENT_VERSION);
        Ok(())
    }

    #[test]
    fn end_to_end_claim_then_submit_success_path() -> anyhow::Result<()> {
        use httpmock::prelude::*;
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(POST).path("/plans/claim");
            then.status(200).body(CLAIM_JOB_BODY);
        });
        let submit = server.mock(|when, then| {
            when.method(POST).path("/benchmarks");
            then.status(202).body(r#"{"job_id":"job-abc"}"#);
        });
        let client = MgmtClient::new(server.base_url())?;
        let http = test_http()?;
        let auth = test_auth()?;

        let ClaimPoll::Job(job) = poll_claim(&client, &http, &auth) else {
            anyhow::bail!("expected a job");
        };
        let payload = attach_claim_to_success_payload(
            serde_json::json!({
                "benchmark_id": job.benchmark_id,
                "prefill_time_ms": 12.5,
                "model_descriptor": "{\"type\":\"gguf_text\",\"local\":true}",
                "device_name": "box"
            }),
            &job,
        )?;
        let disposition = try_submit_plan_result(&client, &http, &auth, &job.job_id, &payload);
        assert!(matches!(disposition, SubmitDisposition::Accepted { .. }));
        submit.assert();
        Ok(())
    }
}
