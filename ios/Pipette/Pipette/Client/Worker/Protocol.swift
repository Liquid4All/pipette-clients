import Foundation

// The planner wire protocol: claim a job, keep its lease alive, and submit the outcome:
// `pipette-cli`'s `client/worker/protocol.rs`. Each step reports an enum the caller
// (`PlannerWorker`) matches on, so retry policy, logging and UI state stay there, except
// the submit backoff, which the crate also keeps here. Where the crate's variants are unit
// cases, these carry the failure, for the caller's log line.

// MARK: - Payload helpers

/// Stamp the claim's `job_id` on a success payload: the crate's
/// `attach_claim_to_success_payload`. Nothing else is overlaid: the run serialized the same
/// cell the claim carried, so its descriptors and flags already are the echo the server's
/// claim-binding check looks for.
nonisolated func attachClaimToSuccessPayload(
    _ payload: [String: Any],
    job: ClaimedJob
) -> [String: Any] {
    var stamped = payload
    stamped["job_id"] = job.jobId
    return stamped
}

/// A failure submission that echoes the claim: the crate's `failure_from_claim`.
/// `deviceTime` runs from the claim to the moment the failure is ready to submit.
nonisolated func failureFromClaim(
    _ job: ClaimedJob,
    reason: String,
    retriable: Bool,
    clientVersion: String,
    deviceTime: Duration
) -> FailureSubmission {
    var body = FailureSubmission.fromClaim(
        job, failureReason: reason, retriable: retriable, clientVersion: clientVersion)
    body.deviceTimeMs = deviceTimeMs(deviceTime)
    return body
}

/// Whole milliseconds, truncated: the crate's `device_time_ms`.
nonisolated func deviceTimeMs(_ deviceTime: Duration) -> UInt64 {
    // A span on one monotonic clock is never negative; the guard keeps `UInt64(_:)` from
    // trapping if a caller passes one anyway.
    guard deviceTime > .zero else { return 0 }
    let (seconds, attoseconds) = deviceTime.components
    return UInt64(seconds) * 1000 + UInt64(attoseconds / 1_000_000_000_000_000)
}

/// Whether a run error is retriable: the crate's `classify_run_error`, and the one place
/// every failure path asks.
///
/// The terminal list stays narrow on purpose: a wrongly-non-retriable job is discarded for
/// the whole fleet on one device's say-so, where a wrongly-retriable one is bounded by its
/// `expires_at` (the mgmt repo's `docs/client-integration.md` §6). Type-based where the
/// crate matches wording: its missing-benchmark and unconvertible-definition markers are
/// `WorkerResolveError.unknownBenchmark` here, and no on-device-only refusal reaches a phone.
nonisolated func classifyRunError(_ error: Error) -> Bool {
    if error is UnrunnableClaimError { return false }
    if case .unknownBenchmark = error as? PlannerWorker.WorkerResolveError { return false }
    // A vault refusal every device with this owner would meet identically (ownership,
    // a deleted resource, drifted bytes) is not worth another claim; a transport
    // failure or a `5xx` on the same route still is.
    if let vault = error as? VaultFetchError, vault.isPermanent { return false }
    // Everything else (OOM, thermal, a failed fetch, a busy device) stays retriable.
    return true
}

/// The crate's `MAX_DETAIL_BYTES`: a body the server refuses for its size is a reason
/// nobody reads.
nonisolated let maxDetailBytes = 8 * 1024

/// Keep both ends of an over-long detail: the head is the error's context, the tail its
/// last words. The crate's `elide_middle`, over UTF-8 bytes so the two agree on the cap.
nonisolated func elideMiddle(_ detail: String) -> String {
    guard detail.utf8.count > maxDetailBytes else { return detail }
    let bytes = Array(detail.utf8)
    let headBudget = maxDetailBytes / 2
    let tailBudget = maxDetailBytes - headBudget
    // Step off a continuation byte (`10xxxxxx`) so neither cut splits a
    // scalar: the crate's `floor_char_boundary` / `ceil_char_boundary`.
    var head = headBudget
    while head > 0, bytes[head] & 0xC0 == 0x80 { head -= 1 }
    var tail = bytes.count - tailBudget
    while tail < bytes.count, bytes[tail] & 0xC0 == 0x80 { tail += 1 }
    let dropped = tail - head
    return String(decoding: bytes[..<head], as: UTF8.self)
        + "\n[... \(dropped) bytes elided ...]\n"
        + String(decoding: bytes[tail...], as: UTF8.self)
}

/// A `failure_reason` with a leading UTC timestamp and the detail held to
/// ``maxDetailBytes``: the crate's `format_failure_reason`.
nonisolated func formatFailureReason(_ detail: String) -> String {
    "[\(JobDateFormat.iso8601.string(from: Date()))] \(elideMiddle(detail))"
}

// MARK: - Claim

/// One `POST /plans/claim` attempt.
nonisolated enum ClaimPoll {
    case job(ClaimedJob)
    case idle
    /// `403`: not approved, or approved without `perform_work`. Only the server's message
    /// says which, so the error is carried rather than renamed.
    case refused(PipetteMgmtClient.Error)
    case transient(PipetteMgmtClient.Error)
    case fatal(PipetteMgmtClient.Error)
}

/// The crate's `poll_claim`.
nonisolated func pollClaim(
    server: ServerUrl,
    http: HttpClient = .shared,
    auth: ClientIdentity
) async -> ClaimPoll {
    do {
        guard let job = try await PipetteMgmtClient.ClientApi.claim(
            server: server, http: http, auth: auth)
        else { return .idle }
        return .job(job)
    } catch {
        if error.httpStatusCode == 403 { return .refused(error) }
        return error.isTransient ? .transient(error) : .fatal(error)
    }
}

// MARK: - Heartbeat

/// Outcome of one heartbeat tick (the mgmt repo's `docs/client-integration.md` §4-5).
nonisolated enum LeaseKeepalive {
    /// Renewed, or reclaimed after a `404`.
    case ok
    /// `409`, or a reclaim that answered `404`/`409` or refused definitively.
    case abort(PipetteMgmtClient.Error)
    case retry(PipetteMgmtClient.Error)
}

/// One heartbeat, with reclaim recovery on `404`: the crate's `keepalive_lease`. A reclaim
/// that failed in transit has not said the lease is gone, so it retries rather than costing
/// a live run.
nonisolated func keepaliveLease(
    server: ServerUrl,
    http: HttpClient = .shared,
    auth: ClientIdentity,
    jobId: String
) async -> LeaseKeepalive {
    do {
        try await PipetteMgmtClient.ClientApi.heartbeat(
            server: server, http: http, auth: auth, jobId: jobId)
        return .ok
    } catch {
        if error.httpStatusCode == 409 { return .abort(error) }
        // Retry rather than abort a live run on a status the protocol does not define.
        guard error.httpStatusCode == 404 else { return .retry(error) }
        do {
            try await PipetteMgmtClient.ClientApi.reclaim(
                server: server, http: http, auth: auth, jobId: jobId)
            return .ok
        } catch {
            return reclaimIsRetriable(error) ? .retry(error) : .abort(error)
        }
    }
}

// MARK: - Submit

/// Outcome of submitting a plan-attached result (success or failure body).
nonisolated enum SubmitDisposition {
    case accepted(jobId: String)
    /// `409`, or a reclaim after `404` that also failed definitively.
    case dropped
    /// The caller should back off and call again with the same body.
    case retryTransient
    case fatal(PipetteMgmtClient.Error)
    /// The submitting task was cancelled and nothing was decided: the caller must leave the
    /// result pending for a later sweep.
    ///
    /// No crate counterpart: the CLI dies with its process, where this loop is cancelled
    /// when the worker is switched off or the app backgrounds. Distinct from
    /// ``retryTransient`` because a cancelled `URLSession` request fails as a transient
    /// transport error, and the backoff would burn every attempt in milliseconds.
    case interrupted
}

/// One submit attempt: the crate's `try_submit_plan_result`. On `404`, reclaims once and
/// answers ``SubmitDisposition/retryTransient`` so the caller resubmits with the binding
/// restored.
nonisolated func trySubmitPlanResult(
    server: ServerUrl,
    http: HttpClient = .shared,
    auth: ClientIdentity,
    jobId: String,
    payload: Data
) async -> SubmitDisposition {
    do {
        let response = try await PipetteMgmtClient.ClientApi.submitResult(
            server: server, http: http, auth: auth, payloadJson: payload)
        return .accepted(jobId: response.jobId)
    } catch {
        if error.httpStatusCode == 409 { return .dropped }
        if error.httpStatusCode == 404 {
            do {
                try await PipetteMgmtClient.ClientApi.reclaim(
                    server: server, http: http, auth: auth, jobId: jobId)
                return .retryTransient
            } catch {
                return reclaimIsRetriable(error) ? .retryTransient : .dropped
            }
        }
        return error.isTransient ? .retryTransient : .fatal(error)
    }
}

/// Retry ``trySubmitPlanResult(server:http:auth:jobId:payload:)`` until a terminal
/// disposition: the crate's `submit_plan_result_with_backoff`. Exhaustion answers
/// ``SubmitDisposition/retryTransient``, and the caller decides what giving up means.
nonisolated func submitPlanResultWithBackoff(
    server: ServerUrl,
    http: HttpClient = .shared,
    auth: ClientIdentity,
    jobId: String,
    payload: Data,
    sleep: (TimeInterval) async -> Void
) async -> SubmitDisposition {
    // Bounded so a permanently broken network cannot spin forever.
    let backoffFirst: TimeInterval = 1
    let backoffCap: TimeInterval = 30
    let attempts = 32
    let settled: SubmitDisposition? = await retryWithBackoff(
        attempts: 1...attempts, first: backoffFirst, cap: backoffCap,
        attempt: { _ -> SubmitDisposition? in
            // Before each attempt: cancellation lands mid-backoff far more often than
            // mid-request, and a cancelled request is indistinguishable here from a 5xx.
            if Task.isCancelled { return .interrupted }
            let disposition = await trySubmitPlanResult(
                server: server, http: http, auth: auth, jobId: jobId, payload: payload)
            guard case .retryTransient = disposition else { return disposition }
            return nil
        },
        sleep: sleep)
    return settled ?? (Task.isCancelled ? .interrupted : .retryTransient)
}

/// Whether a failed reclaim leaves the lease worth keeping: only a transport failure or a
/// `5xx`, as a `404`/`409` says the job is someone else's now.
private nonisolated func reclaimIsRetriable(_ error: PipetteMgmtClient.Error) -> Bool {
    guard error.isTransient else { return false }
    guard let status = error.httpStatusCode else { return true }
    return status != 404 && status != 409
}
