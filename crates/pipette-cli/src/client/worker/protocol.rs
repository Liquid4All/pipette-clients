//! The planner wire protocol: claim a job, keep its lease alive, and submit
//! the outcome. Each step maps to one management-server endpoint and reports
//! back as an enum the loop matches on, so retry policy stays with the caller.

use std::borrow::Cow;
use std::ops::ControlFlow;
use std::time::Duration;

use time::OffsetDateTime;
use time::format_description::well_known::Rfc3339;

use pipette_artifacts::VaultFetchError;
use pipette_http::HttpClient;
use pipette_mgmt_client::{
    ClientIdentity, ServerUrl, client_api,
    types::{ClaimedJob, FailureSubmission},
};
use pipette_subprocess::{ExitError, ReadyError, Stopped};

use crate::client::claim::UnrunnableClaimError;
use crate::client::worker::timing::retry_with_backoff;
use crate::error::{Error, Result};

/// Bind a success submission payload (serialized
/// [`pipette_plan_types::BenchmarkSubmissionPayload`]) to its lease by stamping
/// the claim's `job_id`. Nothing else is overlaid: the run serialized the same
/// `ClientRunSpec` the claim carried, so its descriptors already are the echo
/// the server's claim-binding check wants, with the local HuggingFace token
/// stripped.
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

/// Build a plan-attached failure submission that echoes the claim. The
/// submission carries `device_time_ms`: the time in milliseconds from the claim
/// to the moment the failure is ready to submit.
pub fn failure_from_claim(
    job: &ClaimedJob,
    reason: impl Into<String>,
    retriable: bool,
    device_time_ms: Option<u64>,
) -> FailureSubmission {
    FailureSubmission {
        device_time_ms,
        ..FailureSubmission::from_claim(job, reason, retriable, crate::CLIENT_VERSION)
    }
}

/// Whole milliseconds, saturating at `u64::MAX`.
pub fn device_time_ms(device_time: Duration) -> u64 {
    u64::try_from(device_time.as_millis()).unwrap_or(u64::MAX)
}

/// Each run of these is one warm-up or rep, so their limit bounds the measurement.
/// A probe or helper that overruns (`llama-server --help`, an image-token count)
/// says more about the host than the cell.
const MEASURING_PROGRAMS: [&str; 2] = ["llama-bench", "llama-batched-bench"];

/// llama.cpp's words (`llama-bench`, `llama_model_load`, `llama-server`) for a
/// model it would not load. Matched lower-cased.
const MODEL_LOAD_FAILED: [&str; 3] = [
    "failed to load model",
    "error loading model",
    "exiting due to model loading error",
];

/// A load failure that is not the model's: a missing, non-GGUF or truncated
/// file, which a fresh download repairs, or memory another process may hold (a
/// cancelled job's GPU allocation outlives it). The memory words are llama.cpp's,
/// `ENOMEM`'s, Windows' mapping errors and Vulkan's.
const LOAD_FAILED_ON_FILE_OR_MEMORY: [&str; 11] = [
    "failed to open",
    "failed to read magic",
    "corrupted or incomplete",
    "unable to allocate",
    "failed to allocate",
    "out of memory",
    "cannot allocate memory",
    "not enough memory resources",
    "paging file is too small",
    "outofdevicememory",
    "outofhostmemory",
];

/// llama.cpp's abort when the Hexagon DSP refuses an op (`HTP_STATUS_NO_SUPPORT`).
/// The model, the build and the chip decide it, not DSP memory other processes
/// hold. The cost: a refused cell fails for every phone, even one whose chip
/// could run it, until someone runs `campaigns rerun`.
const DSP_REFUSED_OP: &str = "dsp-error no-support";

/// A measuring program or a request to the engine's server overran. The server's
/// case is told from a download by the [`Stopped`] context only an engine
/// attaches, and from the model-load wait, which stays retriable, by the absence
/// of [`ReadyError`].
fn measurement_timed_out(err: &anyhow::Error) -> bool {
    let measuring = err.chain().any(|cause| {
        matches!(cause.downcast_ref::<ExitError>(),
            Some(ExitError::TimedOut { name, .. }) if MEASURING_PROGRAMS.contains(&name.as_str()))
    });
    measuring
        || (err.downcast_ref::<Stopped>().is_some()
            && !err.chain().any(|cause| cause.is::<ReadyError>())
            && err.chain().any(pipette_http::is_timeout))
}

/// Judged from the output tail the runtime's exit error quotes.
fn model_failed_to_load(err: &anyhow::Error) -> bool {
    err.chain().any(|cause| {
        let tail = match (
            cause.downcast_ref::<ExitError>(),
            cause.downcast_ref::<ReadyError>(),
        ) {
            (Some(ExitError::Failed { tail, .. }), _)
            | (_, Some(ReadyError::Exited { tail, .. })) => tail.to_ascii_lowercase(),
            _ => return false,
        };
        MODEL_LOAD_FAILED.iter().any(|m| tail.contains(m))
            && !LOAD_FAILED_ON_FILE_OR_MEMORY
                .iter()
                .any(|m| tail.contains(m))
    })
}

/// Whether a run error is retriable. Defaults to retriable; only jobs no client
/// is expected to run are not (the mgmt repo's `docs/client-integration.md`,
/// "Retriable vs. non-retriable failures"), which includes a cell whose
/// measurement ran out of time or whose model the runtime would not load: any
/// client meets the same limit or refusal.
pub fn classify_run_error(err: &anyhow::Error) -> bool {
    if err.chain().any(|cause| cause.is::<UnrunnableClaimError>()) {
        return false;
    }
    // A vault refusal every device with this owner would meet identically
    // (ownership, a deleted resource, drifted bytes) is not worth another claim;
    // a transport failure or a `5xx` on the same route still is.
    if err.chain().any(|cause| {
        cause
            .downcast_ref::<VaultFetchError>()
            .is_some_and(VaultFetchError::is_permanent)
    }) {
        return false;
    }
    // Only the engine raises these, after `prepare`'s downloads and digests.
    if measurement_timed_out(err) || model_failed_to_load(err) {
        return false;
    }
    let msg = format!("{err:#}").to_ascii_lowercase();
    // Matching on wording fails the job for every client, so the list stays narrow.
    let non_retriable_markers = [
        "benchmark not found",
        "is not a known, valid definition",
        "runs on-device only",
        DSP_REFUSED_OP,
    ];
    let looks_permanent = non_retriable_markers.iter().any(|m| msg.contains(m));
    !looks_permanent
}

/// Cap on the detail a failure carries. A llama-server failure can quote 64 KiB
/// of stderr, and a body the server refuses for its size is a reason nobody
/// reads: the submit settles as [`SubmitDisposition::Fatal`] and the job waits
/// out its lease unrecorded.
const MAX_DETAIL_BYTES: usize = 8 * 1024;

/// Keep both ends of an over-long detail, eliding the middle: the head is the
/// anyhow context chain (which cell, which step) and the tail is the engine's
/// own last words.
fn elide_middle(detail: &str) -> Cow<'_, str> {
    if detail.len() <= MAX_DETAIL_BYTES {
        return Cow::Borrowed(detail);
    }
    let head_budget = MAX_DETAIL_BYTES / 2;
    let tail_budget = MAX_DETAIL_BYTES - head_budget;
    let head = detail.floor_char_boundary(head_budget);
    let tail = detail.ceil_char_boundary(detail.len() - tail_budget);
    let dropped = tail - head;
    // Both cuts are char boundaries by construction; the fallback is for the lint.
    Cow::Owned(format!(
        "{}\n[... {dropped} bytes elided ...]\n{}",
        detail.get(..head).unwrap_or_default(),
        detail.get(tail..).unwrap_or_default()
    ))
}

/// A `failure_reason`: a leading UTC timestamp, then the detail held to
/// [`MAX_DETAIL_BYTES`]. Capped here because every submitted failure passes
/// through this function.
pub fn format_failure_reason(detail: &str) -> String {
    let ts = OffsetDateTime::now_utc()
        .format(&Rfc3339)
        .unwrap_or_else(|_| "unknown-time".into());
    format!("[{ts}] {}", elide_middle(detail))
}

/// One `POST /plans/claim` attempt, classified for the worker state machine.
#[derive(Debug)]
pub enum ClaimPoll {
    /// `200`.
    Job(Box<ClaimedJob>),
    /// `204`: idle-wait then retry.
    Idle,
    /// `403`: stop the worker. An unapproved client and one without
    /// `perform_work` get the same status; only the server's message says which.
    Refused(pipette_mgmt_client::Error),
    /// `5xx` or network: back off and retry.
    Transient(pipette_mgmt_client::Error),
    Fatal(pipette_mgmt_client::Error),
}

/// Classify a single claim attempt (no retry loop).
pub fn poll_claim(server: &ServerUrl, http: &HttpClient, auth: &ClientIdentity) -> ClaimPoll {
    match client_api::claim(server, http, auth) {
        Ok(Some(job)) => ClaimPoll::Job(Box::new(job)),
        Ok(None) => ClaimPoll::Idle,
        Err(e) if e.http_status() == Some(403) => ClaimPoll::Refused(e),
        Err(e) if e.is_transient() => ClaimPoll::Transient(e),
        Err(e) => ClaimPoll::Fatal(e),
    }
}

/// Outcome of one heartbeat tick.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LeaseKeepalive {
    /// Renewed, or reclaimed after a `404`.
    Ok,
    /// The lease is gone for good: abort the run.
    Abort,
    /// Retry soon; do not abort yet.
    Retry,
}

/// One heartbeat, with reclaim recovery on `404`.
pub fn keepalive_lease(
    server: &ServerUrl,
    http: &HttpClient,
    auth: &ClientIdentity,
    job_id: &str,
) -> LeaseKeepalive {
    match client_api::heartbeat(server, http, auth, job_id) {
        Ok(()) => LeaseKeepalive::Ok,
        Err(e) if e.http_status() == Some(409) => LeaseKeepalive::Abort,
        Err(e) if e.http_status() == Some(404) => {
            match client_api::reclaim(server, http, auth, job_id) {
                Ok(()) => LeaseKeepalive::Ok,
                Err(re) if matches!(re.http_status(), Some(404 | 409)) => LeaseKeepalive::Abort,
                Err(re) if re.is_transient() => LeaseKeepalive::Retry,
                Err(_) => LeaseKeepalive::Abort,
            }
        }
        Err(e) if e.is_transient() => LeaseKeepalive::Retry,
        // Non-transient, non-404/409: retry rather than aborting a live run on
        // an unexpected 4xx the protocol doesn't define.
        Err(_) => LeaseKeepalive::Retry,
    }
}

/// A submission that sending again would not change: the server refused it (a
/// `400`, a `403`, a `413`), or this client could not send it at all (an
/// unsigned request, an unencodable body).
///
/// A refused result is re-reported as a retriable failure. This reaching the
/// loop means that report failed too, and the live lease would hand the same
/// job back on the next claim, so the worker stops.
#[derive(Debug, thiserror::Error)]
#[error(transparent)]
pub struct SubmitRefusedError(pub pipette_mgmt_client::Error);

/// Outcome of submitting a plan-attached result (success or failure body).
#[derive(Debug)]
pub enum SubmitDisposition {
    /// `202`.
    Accepted { job_id: String },
    /// `409`, or the reclaim after a `404` also failed.
    Dropped,
    /// Back off and send the same body again.
    RetryTransient,
    /// Unexpected definitive error.
    Fatal(pipette_mgmt_client::Error),
}

/// One submit attempt for a plan-attached body. On `404`, reclaims once and
/// returns [`SubmitDisposition::RetryTransient`] so the caller resubmits.
pub fn try_submit_plan_result<P: serde::Serialize>(
    server: &ServerUrl,
    http: &HttpClient,
    auth: &ClientIdentity,
    job_id: &str,
    payload: &P,
) -> SubmitDisposition {
    match client_api::submit_result(server, http, auth, payload) {
        Ok(resp) => SubmitDisposition::Accepted {
            job_id: resp.job_id,
        },
        Err(e) if e.http_status() == Some(409) => SubmitDisposition::Dropped,
        Err(e) if e.http_status() == Some(404) => {
            match client_api::reclaim(server, http, auth, job_id) {
                Ok(()) => SubmitDisposition::RetryTransient,
                Err(re) if matches!(re.http_status(), Some(404 | 409)) => {
                    SubmitDisposition::Dropped
                }
                Err(re) if re.is_transient() => SubmitDisposition::RetryTransient,
                Err(_) => SubmitDisposition::Dropped,
            }
        }
        Err(e) if e.is_transient() => SubmitDisposition::RetryTransient,
        Err(e) => SubmitDisposition::Fatal(e),
    }
}

/// Drive [`try_submit_plan_result`] through [`retry_with_backoff`] until a
/// terminal disposition. `sleep` is injected so tests can no-op the waits, and
/// so a worker being shut down can break out of them: a `sleep` that breaks
/// settles as [`SubmitDisposition::RetryTransient`], the same answer as a
/// backoff that ran out, because the result is equally unsent either way.
pub fn submit_plan_result_with_backoff<P, S>(
    server: &ServerUrl,
    http: &HttpClient,
    auth: &ClientIdentity,
    job_id: &str,
    payload: &P,
    mut sleep: S,
) -> Result<SubmitDisposition>
where
    P: serde::Serialize,
    S: FnMut(Duration) -> ControlFlow<()>,
{
    const BACKOFF_FIRST: Duration = Duration::from_secs(1);
    const BACKOFF_CAP: Duration = Duration::from_secs(30);
    // Bound retries so a permanently-broken network can't spin forever in tests
    // or production without an external interrupt.
    const ATTEMPTS: u32 = 32;
    let settled = retry_with_backoff(
        1..=ATTEMPTS,
        BACKOFF_FIRST,
        BACKOFF_CAP,
        |_| match try_submit_plan_result(server, http, auth, job_id, payload) {
            SubmitDisposition::RetryTransient => ControlFlow::Continue(()),
            other @ (SubmitDisposition::Accepted { .. }
            | SubmitDisposition::Dropped
            | SubmitDisposition::Fatal(_)) => ControlFlow::Break(other),
        },
        &mut sleep,
    );
    Ok(settled.unwrap_or(SubmitDisposition::RetryTransient))
}

#[cfg(test)]
pub(crate) mod tests {
    use pipette_plan_types::ClientId;

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
        assert_eq!(out["model_descriptor"], r#"{"type":"gguf_text","other":1}"#);
        assert_eq!(out["prefill_time_ms"], 1.0);
        Ok(())
    }

    /// Driven from a real bad payload rather than a hand-built error, so it
    /// breaks if the runner stops returning [`UnrunnableClaimError`].
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

        let wire = serde_json::to_value(failure_from_claim(
            &job,
            "[ts] bad spec",
            retriable,
            Some(0),
        ))?;
        assert_eq!(wire["job_id"], "job-abc");
        assert_eq!(wire["benchmark_id"], "prefill_throughput_256");
        assert_eq!(wire["retriable"], false);
        Ok(())
    }

    /// A bracketed RFC 3339 timestamp, then the detail verbatim.
    #[test]
    fn a_failure_reason_is_timestamped() -> anyhow::Result<()> {
        let reason = format_failure_reason("out of memory");

        let (stamp, detail) = reason
            .strip_prefix('[')
            .and_then(|rest| rest.split_once("] "))
            .ok_or_else(|| anyhow::anyhow!("no bracketed timestamp: {reason}"))?;
        OffsetDateTime::parse(stamp, &Rfc3339)?;
        assert_eq!(detail, "out of memory");
        Ok(())
    }

    #[test]
    fn a_reason_within_the_cap_is_untouched() {
        let detail = "x".repeat(MAX_DETAIL_BYTES);
        let reason = format_failure_reason(&detail);

        assert!(reason.ends_with(&detail), "a short reason was rewritten");
        assert!(!reason.contains("elided"));
    }

    #[test]
    fn an_oversized_reason_keeps_both_ends() {
        const HEAD: &str = "running cell eval_ifbench: ";
        const TAIL: &str = ": ggml_metal_graph_compute: command buffer 0 failed";
        let detail = format!("{HEAD}{}{TAIL}", "llama_model_load noise\n".repeat(8_000));
        assert!(detail.len() > MAX_DETAIL_BYTES, "the fixture must overflow");

        let reason = format_failure_reason(&detail);

        assert!(reason.contains(HEAD), "the context chain was dropped");
        assert!(
            reason.ends_with(TAIL),
            "the engine's last words were dropped"
        );
        assert!(reason.contains("bytes elided"), "the loss went unmarked");
        // Bounded by the cap plus the timestamp and the marker, not by the input.
        assert!(
            reason.len() < MAX_DETAIL_BYTES + 128,
            "still {} bytes wide",
            reason.len()
        );
    }

    /// The budgets are even, so only the 3-byte scalar lands a cut
    /// mid-character; the 2- and 4-byte ones pin the exact-boundary path.
    #[rstest::rstest]
    #[case::two_bytes('\u{00e9}')]
    #[case::three_bytes('\u{20ac}')]
    #[case::four_bytes('\u{1f600}')]
    fn an_oversized_multibyte_reason_cuts_on_char_boundaries(#[case] scalar: char) {
        let detail = scalar.to_string().repeat(MAX_DETAIL_BYTES);
        let reason = format_failure_reason(&detail);

        assert!(reason.contains("bytes elided"));
        assert!(reason.ends_with(scalar), "{}", reason.len());
        assert!(
            reason.contains(&format!("{scalar}\n[... ")),
            "the head was cut mid-character"
        );
    }

    /// A vault refusal is found by type through either store's wrapping, the
    /// way the run path delivers it, and only the permanent kinds end the
    /// retries.
    #[rstest::rstest]
    #[case::refused(403, false, false)]
    #[case::drifted(0, false, false)]
    #[case::transient(503, true, false)]
    #[case::runtime_refused(403, false, true)]
    #[case::runtime_transient(503, true, true)]
    fn a_vault_refusal_is_classified_by_type(
        #[case] status: u16,
        #[case] retriable: bool,
        #[case] runtime: bool,
    ) -> anyhow::Result<()> {
        let resource_id = pipette_plan_types::VaultResourceId::try_new("vault_1")?;
        let cause = match status {
            403 => VaultFetchError::Refused {
                resource_id,
                status,
                detail: "not yours".into(),
            },
            0 => VaultFetchError::Drift {
                resource_id,
                expected: "aa".into(),
                actual: "bb".into(),
            },
            _ => VaultFetchError::Mgmt(pipette_mgmt_client::Error::HttpStatus {
                method: "GET".into(),
                url: "u".into(),
                status,
                body: String::new(),
            }),
        };
        let err = if runtime {
            anyhow::Error::from(pipette_artifacts::runtime::RuntimeStoreError::Fetch {
                runtime: "r".into(),
                source: anyhow::Error::from(cause),
            })
        } else {
            anyhow::Error::from(pipette_artifacts::model::ModelStoreError::Fetch {
                model: "m".into(),
                source: anyhow::Error::from(cause),
            })
        }
        .context("preparing the cell");
        assert_eq!(classify_run_error(&err), retriable, "{err:#}");
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

    fn overran(name: &str) -> anyhow::Error {
        anyhow::Error::from(ExitError::TimedOut {
            name: name.to_owned(),
            timeout: Duration::from_secs(3600),
            tail: "\nstderr tail:\nllama_context: n_ctx = 4196".to_owned(),
        })
        .context("decode_throughput_4096_100 warm-up")
    }

    fn exit_one() -> std::process::ExitStatus {
        #[cfg(unix)]
        let status =
            <std::process::ExitStatus as std::os::unix::process::ExitStatusExt>::from_raw(1 << 8);
        #[cfg(windows)]
        let status =
            <std::process::ExitStatus as std::os::windows::process::ExitStatusExt>::from_raw(1);
        status
    }

    fn bench_exited(stderr: &str) -> anyhow::Error {
        anyhow::Error::from(ExitError::Failed {
            name: "llama-bench".to_owned(),
            status: exit_one(),
            tail: format!("\nstderr tail:\n{stderr}"),
        })
        .context("decode_throughput_4096_100 warm-up")
    }

    fn server_exited(stderr: &str) -> anyhow::Error {
        anyhow::Error::from(ReadyError::Exited {
            name: "llama-server".to_owned(),
            status: exit_one(),
            tail: format!("\nstderr tail:\n{stderr}"),
        })
    }

    fn dsp_aborted(status: &str) -> anyhow::Error {
        anyhow::anyhow!("connection closed before message completed")
            .context("vl_throughput_768x768_text0_decode64_img1 warm-up: POST /v1/chat/completions")
            .context(format!(
                "llama-server was still running\nstderr tail:\n\
                 /__w/llama.cpp/llama.cpp/ggml/src/ggml-hexagon/ggml-hexagon.cpp:2444: \
                 ggml-hex: HTP0 synchronize failed : dsp-error {status}\n\
                 0.03.287.296 E ggml-hex: HTP0 dspcall : dsp-rsp {status}\nerror"
            ))
    }

    /// The tails are llama.cpp's own lines.
    #[rstest::rstest]
    #[case::bench_overran(overran("llama-bench"), false)]
    #[case::batched_bench_overran(overran("llama-batched-bench"), false)]
    #[case::a_probe_overran(overran("llama-server"), true)]
    #[case::bench_would_not_load(
        bench_exited("main: error: failed to load model '/m/q.gguf'"),
        false
    )]
    #[case::unknown_architecture(
        server_exited(
            "llama_model_load: error loading model: error loading model architecture: \
             unknown model architecture: 'lfm3'\nmain: exiting due to model loading error"
        ),
        false
    )]
    #[case::unknown_type(
        bench_exited(
            "gguf_init_from_file_impl: tensor 'blk.0.ffn_up.weight' has invalid ggml type 41\n\
             gguf_init_from_file_impl: failed to read tensor info\n\
             main: error: failed to load model '/m/q.gguf'"
        ),
        false
    )]
    #[case::missing_file(
        bench_exited(
            "gguf_init_from_file: failed to open GGUF file '/m/q.gguf' (No such file or directory)\n\
             main: error: failed to load model '/m/q.gguf'"
        ),
        true
    )]
    #[case::truncated_file(
        server_exited(
            "llama_model_load: error loading model: tensor 'output.weight' data is not within \
             the file bounds, model is corrupted or incomplete\n\
             main: exiting due to model loading error"
        ),
        true
    )]
    #[case::out_of_memory_at_load(
        server_exited(
            "llama_model_load: error loading model: unable to allocate Vulkan0 buffer\n\
             main: exiting due to model loading error"
        ),
        true
    )]
    #[case::mmap_out_of_memory(
        bench_exited(
            "llama_model_load: error loading model: mmap failed: Cannot allocate memory\n\
             main: error: failed to load model '/m/q.gguf'"
        ),
        true
    )]
    #[case::a_crash(bench_exited("Segmentation fault"), true)]
    #[case::the_dsp_refused_an_op(dsp_aborted("NO-SUPPORT"), false)]
    #[case::the_dsp_ran_short_of_vtcm(dsp_aborted("VTCM-TOO-SMALL"), true)]
    #[case::the_load_wait_overran(
        anyhow::Error::from(ReadyError::TimedOut {
            name: "llama-server".to_owned(),
            timeout: Duration::from_secs(120),
            tail: String::new(),
            source: "HTTP 503 Service Unavailable: loading model".into(),
        }),
        true
    )]
    fn an_engine_failure_is_final_for_its_limit_or_its_model(
        #[case] err: anyhow::Error,
        #[case] retriable: bool,
    ) {
        assert_eq!(classify_run_error(&err), retriable, "{err:#}");
    }

    /// A download's stall and the load wait's last check are the same timeout error.
    #[cfg(unix)]
    #[rstest::rstest]
    #[case::the_engines_request("request", false)]
    #[case::the_load_wait("load wait", true)]
    #[case::a_download("download", true)]
    fn a_request_timeout_is_final_only_in_the_engine(
        #[case] stage: &str,
        #[case] retriable: bool,
    ) -> anyhow::Result<()> {
        use anyhow::Context;

        let server = httpmock::MockServer::start();
        server.mock(|when, then| {
            when.path("/completion");
            then.delay(Duration::from_secs(2));
        });
        let http = HttpClient::loopback("pipette-test/0", Duration::from_millis(100))?;
        let timed_out = pipette_http::send(http.client().post(server.url("/completion")))
            .err()
            .context("the request did not time out")
            .map(anyhow::Error::from)?
            .context("failed to call /completion");
        let err = if stage == "download" {
            anyhow::Error::from(pipette_artifacts::model::ModelStoreError::Fetch {
                model: "m".into(),
                source: timed_out,
            })
        } else {
            let mut command = std::process::Command::new("sh");
            command.args(["-c", "exec sleep 30"]);
            let mut llama_server =
                pipette_subprocess::start_process("llama-server", command, Duration::ZERO)?;
            let failed = if stage == "load wait" {
                anyhow::Error::from(ReadyError::TimedOut {
                    name: "llama-server".to_owned(),
                    timeout: Duration::from_secs(120),
                    tail: String::new(),
                    source: timed_out.into(),
                })
            } else {
                timed_out.context("end_to_end_latency_256_256 rep 1/5")
            };
            failed.context(llama_server.stop_with_tail())
        }
        .context("running the cell");
        assert_eq!(classify_run_error(&err), retriable, "{err:#}");
        Ok(())
    }

    fn test_http() -> anyhow::Result<HttpClient> {
        Ok(HttpClient::new("pipette-test/0")?)
    }

    fn server_url(server: &httpmock::MockServer) -> anyhow::Result<ServerUrl> {
        Ok(ServerUrl::try_new(server.base_url())?)
    }

    pub(crate) fn test_auth() -> anyhow::Result<ClientIdentity> {
        let (private_key_hex, _) = pipette_mgmt_client::generate_keypair()?;
        Ok(ClientIdentity {
            client_id: ClientId::try_new("ev1_test")?,
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
                "repository_url": "github.com/ggml-org/llama.cpp",
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
        let server_url = server_url(&server)?;
        let http = test_http()?;
        let auth = test_auth()?;
        match poll_claim(&server_url, &http, &auth) {
            ClaimPoll::Job(job) => {
                assert_eq!(job.job_id, "job-abc");
                assert_eq!(job.time_window, "PT10M");
            }
            other @ (ClaimPoll::Idle
            | ClaimPoll::Refused(_)
            | ClaimPoll::Transient(_)
            | ClaimPoll::Fatal(_)) => anyhow::bail!("expected Job, got {other:?}"),
        }
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
        let renewing = server_url(&server)?;
        let http = test_http()?;
        let auth = test_auth()?;
        assert!(matches!(
            keepalive_lease(&renewing, &http, &auth, "job-abc"),
            LeaseKeepalive::Ok
        ));
        ok.assert();

        let server2 = MockServer::start();
        server2.mock(|when, then| {
            when.method(PUT).path("/plans/job-abc/heartbeat");
            then.status(409).body("superseded");
        });
        let superseded = server_url(&server2)?;
        assert!(matches!(
            keepalive_lease(&superseded, &http, &auth, "job-abc"),
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
        let server_url = server_url(&server)?;
        let http = test_http()?;
        assert!(matches!(
            keepalive_lease(&server_url, &http, &test_auth()?, "job-abc"),
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
        let server_url = server_url(&server)?;
        let http = test_http()?;
        assert!(matches!(
            keepalive_lease(&server_url, &http, &test_auth()?, "job-abc"),
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
        let server_url = server_url(&server)?;
        let http = test_http()?;
        assert!(matches!(
            keepalive_lease(&server_url, &http, &test_auth()?, "job-abc"),
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
        let server_url = server_url(&server)?;
        let http = test_http()?;
        let body = serde_json::json!({"job_id": "job-abc"});
        match try_submit_plan_result(&server_url, &http, &test_auth()?, "job-abc", &body) {
            SubmitDisposition::Accepted { job_id } => assert_eq!(job_id, "job-abc"),
            other @ (SubmitDisposition::Dropped
            | SubmitDisposition::RetryTransient
            | SubmitDisposition::Fatal(_)) => anyhow::bail!("expected Accepted, got {other:?}"),
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
        let server_url = server_url(&server)?;
        let http = test_http()?;
        assert!(matches!(
            try_submit_plan_result(
                &server_url,
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
        let server_url = server_url(&server)?;
        let http = test_http()?;
        assert!(matches!(
            try_submit_plan_result(
                &server_url,
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
    fn failure_submission_identifies_the_job() -> anyhow::Result<()> {
        let job: ClaimedJob = serde_json::from_str(CLAIM_JOB_BODY)?;
        let device_time = device_time_ms(Duration::from_micros(1_250_900));
        let failure = failure_from_claim(&job, "[ts] oom", true, Some(device_time));
        let wire = serde_json::to_value(&failure)?;
        assert_eq!(wire["message_type"], "failure");
        assert_eq!(wire["job_id"], "job-abc");
        assert_eq!(wire["benchmark_id"], "prefill_throughput_256");
        assert_eq!(wire["retriable"], true);
        assert_eq!(wire["failure_reason"], "[ts] oom");
        // The server cannot recover which build reported this from the job body.
        assert_eq!(wire["client_version"], crate::CLIENT_VERSION);
        // Whole milliseconds, truncated.
        assert_eq!(wire["device_time_ms"], 1_250);
        Ok(())
    }
}
