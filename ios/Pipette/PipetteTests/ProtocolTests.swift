import Foundation
import Testing

@testable import Pipette

/// `Client/Worker/Protocol.swift`: the crate's `client/worker/protocol.rs` tests, over a
/// stubbed server where the crate starts `httpmock`.
struct ProtocolTests {
    private static let claimJobBody = """
        {"job_id": "job-abc", "benchmark_id": "prefill_throughput_256", "time_window": "PT10M",
         "model_name": "m",
         "spec": {"benchmark": "prefill_throughput_256",
                  "model": {"type": "gguf_text", "source": "huggingface", "org": "o",
                            "repo_name": "r", "path": "m-Q4_0.gguf"},
                  "runtime": {"type": "llamacpp_ios_pipette",
                              "repository_url": "github.com/ggml-org/llama.cpp",
                              "repository_version": "b5000", "flavor": "ios-arm64"}}}
        """

    private func job(_ body: String = Self.claimJobBody) throws -> ClaimedJob {
        try JSONDecoder().decode(ClaimedJob.self, from: Data(body.utf8))
    }

    private func auth() throws -> ClientIdentity { try testIdentity() }

    // MARK: - Payload helpers

    @Test func attachClaimStampsTheJobIdAndLeavesTheRunPayloadAlone() throws {
        let out = attachClaimToSuccessPayload([
            "benchmark_id": "prefill_throughput_256",
            "model_descriptor": #"{"type":"gguf_text","other":1}"#,
            "prefill_time_ms": 1.0,
        ], job: try job())
        #expect(out["job_id"] as? String == "job-abc")
        // The run's descriptor stands: it is the token-stripped serialization of the
        // same spec the claim carried.
        #expect(out["model_descriptor"] as? String == #"{"type":"gguf_text","other":1}"#)
        #expect(out["prefill_time_ms"] as? Double == 1.0)
    }

    /// The whole chain a poison-pill job takes: it decodes, the parse refuses it, the
    /// refusal classifies as terminal, and the server learns which job died. Driven
    /// from a real bad payload rather than a hand-built error, so it breaks if the
    /// parse stops throwing `UnrunnableClaimError`.
    @Test func anUnreadableClaimIsReportedAsATerminalFailure() throws {
        let job = try job("""
            {"job_id": "job-abc", "benchmark_id": "prefill_throughput_256",
             "time_window": "PT10M",
             "spec": {"benchmark": "prefill_throughput_256", "model": "not-a-model"}}
            """)
        let error: any Error
        do {
            _ = try runSpecFromClaim(job)
            Issue.record("an unreadable spec cannot produce a cell")
            return
        } catch let caught {
            error = caught
        }
        let retriable = classifyRunError(error)
        #expect(!retriable, "an unreadable claim must not be retried")

        let wire = try wireObject(failureFromClaim(
            job, reason: "[ts] bad spec", retriable: retriable, clientVersion: "9.9.9",
            deviceTime: .zero))
        #expect(wire["job_id"] as? String == "job-abc")
        #expect(wire["benchmark_id"] as? String == "prefill_throughput_256")
        #expect(wire["retriable"] as? Bool == false)
    }

    @Test func classifyRunErrorDefaultsRetriable() {
        // Retriable: specific to this device, so another may still succeed. A failed Hub
        // download arrives here as the cell's error, which is `deviceUnavailable`.
        #expect(classifyRunError(URLError(.timedOut)))
        #expect(classifyRunError(
            PlannerWorker.WorkerResolveError.deviceUnavailable("busy")))
        // Terminal: unreadable by construction, and a benchmark nobody knows.
        #expect(!classifyRunError(UnrunnableClaimError.noSpec))
        #expect(!classifyRunError(UnrunnableClaimError.unparseable))
        #expect(!classifyRunError(
            PlannerWorker.WorkerResolveError.unknownBenchmark("b")))
    }

    /// A refusal every device with this owner would meet is terminal on the wire, and a
    /// `5xx` or a transport failure on the same route stays retriable.
    @Test(arguments: [(403, false), (404, false), (503, true), (0, true)])
    func aVaultRefusalIsClassifiedByType(status: Int, retriable: Bool) throws {
        let resourceId = try VaultResourceId("vault_1")
        let url = "https://mgmt.test/vault/resources/vault_1/content"
        let error: VaultFetchError =
            switch status {
            case 403, 404: .refused(resourceId: resourceId, status: status, detail: "refused")
            case 0: .mgmt(.transport(method: "GET", url: url, message: "offline"))
            default: .mgmt(.httpStatus(method: "GET", url: url, status: status, body: ""))
            }
        #expect(classifyRunError(error) == retriable)
        let wire = try wireObject(failureFromClaim(
            try job(), reason: error.localizedDescription,
            retriable: classifyRunError(error), clientVersion: "9.9.9", deviceTime: .zero))
        #expect(wire["retriable"] as? Bool == retriable)
    }

    @Test func failureSubmissionIdentifiesTheJob() throws {
        let wire = try wireObject(failureFromClaim(
            try job(), reason: "[ts] oom", retriable: true, clientVersion: "9.9.9 (1234)",
            deviceTime: .milliseconds(1250) + .microseconds(900)))
        #expect(wire["message_type"] as? String == "failure")
        #expect(wire["job_id"] as? String == "job-abc")
        #expect(wire["benchmark_id"] as? String == "prefill_throughput_256")
        #expect(wire["retriable"] as? Bool == true)
        #expect(wire["failure_reason"] as? String == "[ts] oom")
        // The server cannot recover this from the job body; without it "which build
        // reported this" is unanswerable.
        #expect(wire["client_version"] as? String == "9.9.9 (1234)")
        // Whole milliseconds, truncated.
        #expect(wire["device_time_ms"] as? Int == 1250)
    }

    @Test func aFailureReasonIsTimestamped() throws {
        let reason = formatFailureReason("oom")
        #expect(reason.hasPrefix("["))
        #expect(reason.hasSuffix("] oom"))
        let stamp = String(reason.dropFirst().prefix { $0 != "]" })
        #expect(JobDateFormat.iso8601.date(from: stamp) != nil)
    }

    /// The elision exists for the engine dumps, and a normal failure must not gain a
    /// marker.
    @Test func aReasonWithinTheCapIsUntouched() {
        let detail = String(repeating: "x", count: maxDetailBytes)
        let reason = formatFailureReason(detail)
        #expect(reason.hasSuffix(detail))
        #expect(!reason.contains("elided"))
    }

    /// The context that opened the chain and the error that ended it both survive, and the
    /// result is bounded by the cap rather than by the input.
    @Test func anOversizedReasonKeepsBothEnds() {
        let head = "running cell eval_ifbench: "
        let tail = ": ggml_metal_graph_compute: command buffer 0 failed"
        let detail = head + String(repeating: "llama_model_load noise\n", count: 8_000) + tail
        #expect(detail.utf8.count > maxDetailBytes)

        let reason = formatFailureReason(detail)

        #expect(reason.contains(head))
        #expect(reason.hasSuffix(tail))
        #expect(reason.contains("bytes elided"))
        #expect(reason.utf8.count < maxDetailBytes + 128)
    }

    /// The cap is a byte count, so a cut that lands inside a scalar has to move off it: a
    /// replacement character would mean it did not. The budgets are even, so only the
    /// 3-byte scalar lands a cut mid-character.
    @Test(arguments: ["\u{00e9}", "\u{20ac}", "\u{1f600}"])
    func anOversizedMultibyteReasonCutsOnCharBoundaries(scalar: String) {
        let detail = String(repeating: scalar, count: maxDetailBytes)
        let reason = formatFailureReason(detail)
        #expect(reason.contains("bytes elided"))
        #expect(reason.hasSuffix(scalar))
        #expect(reason.contains(scalar + "\n[... "))
        #expect(!reason.contains("\u{FFFD}"))
    }

    // MARK: - Claim

    @Test func pollClaimClassifies200() async throws {
        let body = Self.claimJobBody
        let (http, server, _) = try RoutedURLProtocol.serve { _, _ in .reply(200, body) }
        guard case .job(let job) = await pollClaim(
            server: server, http: http, auth: try auth())
        else {
            Issue.record("expected a job")
            return
        }
        #expect(job.jobId == "job-abc")
        #expect(job.timeWindow == "PT10M")
    }

    /// A `5xx` is the server being down, which a daemon waits out; any other status the
    /// route does not define is definitive.
    @Test func pollClaimSplitsTransientFromFatal() async throws {
        let (downHttp, down, _) = try RoutedURLProtocol.serve { _, _ in .reply(503, "down") }
        guard case .transient = await pollClaim(
            server: down, http: downHttp, auth: try auth())
        else {
            Issue.record("expected transient")
            return
        }
        let (badHttp, bad, _) = try RoutedURLProtocol.serve { _, _ in .reply(400, "bad") }
        guard case .fatal = await pollClaim(
            server: bad, http: badHttp, auth: try auth())
        else {
            Issue.record("expected fatal")
            return
        }
    }

    // MARK: - Heartbeat

    @Test func keepaliveRenewsOn200AndAbortsOn409() async throws {
        let (okHttp, renewing, seen) = try RoutedURLProtocol.serve { _, _ in .reply(200) }
        guard case .ok = await keepaliveLease(
            server: renewing, http: okHttp, auth: try auth(), jobId: "job-abc")
        else {
            Issue.record("expected ok")
            return
        }
        #expect(seen.methodsAndPaths == ["PUT /plans/job-abc/heartbeat"])

        let (goneHttp, superseded, _) = try RoutedURLProtocol.serve { _, _ in
            .reply(409, "superseded")
        }
        guard case .abort = await keepaliveLease(
            server: superseded, http: goneHttp, auth: try auth(), jobId: "job-abc")
        else {
            Issue.record("expected abort")
            return
        }
    }

    @Test func keepaliveReclaimsAfterHeartbeat404() async throws {
        let (http, server, seen) = try RoutedURLProtocol.serve { request, _ in
            request.url?.path.hasSuffix("/heartbeat") == true ? .reply(404, "gone") : .reply(200)
        }
        guard case .ok = await keepaliveLease(
            server: server, http: http, auth: try auth(), jobId: "job-abc")
        else {
            Issue.record("expected ok")
            return
        }
        #expect(seen.methodsAndPaths == [
            "PUT /plans/job-abc/heartbeat", "POST /plans/job-abc/reclaim",
        ])
    }

    @Test func keepaliveAbortsWhenReclaimAlso404() async throws {
        let (http, server, _) = try RoutedURLProtocol.serve { _, _ in .reply(404, "gone") }
        guard case .abort = await keepaliveLease(
            server: server, http: http, auth: try auth(), jobId: "job-abc")
        else {
            Issue.record("expected abort")
            return
        }
    }

    @Test func keepaliveRetriesOnTransientHeartbeat() async throws {
        let (http, server, _) = try RoutedURLProtocol.serve { _, _ in .reply(503, "down") }
        guard case .retry = await keepaliveLease(
            server: server, http: http, auth: try auth(), jobId: "job-abc")
        else {
            Issue.record("expected retry")
            return
        }
    }

    /// A reclaim that failed in transit has not said the lease is gone.
    @Test func keepaliveRetriesWhenTheReclaimFailsInTransit() async throws {
        let (http, server, _) = try RoutedURLProtocol.serve { request, _ in
            request.url?.path.hasSuffix("/heartbeat") == true
                ? .reply(404, "gone") : .reply(503, "down")
        }
        guard case .retry = await keepaliveLease(
            server: server, http: http, auth: try auth(), jobId: "job-abc")
        else {
            Issue.record("expected retry")
            return
        }
    }

    // MARK: - Submit

    @Test func trySubmitAccepts202() async throws {
        let (http, server, _) = try RoutedURLProtocol.serve { _, _ in
            .reply(202, #"{"job_id":"job-abc"}"#)
        }
        guard case .accepted(let jobId) = await trySubmitPlanResult(
            server: server, http: http, auth: try auth(), jobId: "job-abc",
            payload: Data(#"{"job_id":"job-abc"}"#.utf8))
        else {
            Issue.record("expected accepted")
            return
        }
        #expect(jobId == "job-abc")
    }

    @Test func trySubmitDropsOn409() async throws {
        let (http, server, _) = try RoutedURLProtocol.serve { _, _ in .reply(409, "superseded") }
        guard case .dropped = await trySubmitPlanResult(
            server: server, http: http, auth: try auth(), jobId: "job-abc",
            payload: Data("{}".utf8))
        else {
            Issue.record("expected dropped")
            return
        }
    }

    @Test func trySubmitReclaimsThenAsksForRetryOn404() async throws {
        let (http, server, seen) = try RoutedURLProtocol.serve { request, _ in
            request.url?.path == "/benchmarks" ? .reply(404, "gone") : .reply(200)
        }
        guard case .retryTransient = await trySubmitPlanResult(
            server: server, http: http, auth: try auth(), jobId: "job-abc",
            payload: Data("{}".utf8))
        else {
            Issue.record("expected retryTransient")
            return
        }
        #expect(seen.methodsAndPaths == ["POST /benchmarks", "POST /plans/job-abc/reclaim"])
    }

    @Test func trySubmitIsFatalOnAnUndefinedRefusal() async throws {
        let (http, server, _) = try RoutedURLProtocol.serve { _, _ in .reply(400, "bad") }
        guard case .fatal = await trySubmitPlanResult(
            server: server, http: http, auth: try auth(), jobId: "job-abc",
            payload: Data("{}".utf8))
        else {
            Issue.record("expected fatal")
            return
        }
    }

    @Test func submitWithBackoffRetriesThenAccepts() async throws {
        // Accepted on the first attempt: no backoff at all.
        let (http, server, _) = try RoutedURLProtocol.serve { _, _ in
            .reply(202, #"{"job_id":"job-abc"}"#)
        }
        var sleeps: [TimeInterval] = []
        let first = await submitPlanResultWithBackoff(
            server: server, http: http, auth: try auth(), jobId: "job-abc",
            payload: Data("{}".utf8)
        ) { sleeps.append($0) }
        guard case .accepted = first else {
            Issue.record("expected accepted")
            return
        }
        #expect(sleeps.isEmpty, "no backoff when the first submit succeeds")

        // Two transients, then accepted: the wait doubles from one second.
        let (flakyHttp, flaky, _) = try RoutedURLProtocol.serve { _, index in
            index < 2 ? .reply(503, "down") : .reply(202, #"{"job_id":"job-abc"}"#)
        }
        let third = await submitPlanResultWithBackoff(
            server: flaky, http: flakyHttp, auth: try auth(), jobId: "job-abc",
            payload: Data("{}".utf8)
        ) { sleeps.append($0) }
        guard case .accepted = third else {
            Issue.record("expected accepted")
            return
        }
        #expect(sleeps == [1, 2])
    }

    /// Bounded, and exhaustion is a disposition rather than an error: the caller
    /// decides what giving up means.
    @Test func submitWithBackoffGivesUpAfterThirtyTwoAttempts() async throws {
        let (http, server, seen) = try RoutedURLProtocol.serve { _, _ in .reply(503, "down") }
        var sleeps: [TimeInterval] = []
        let disposition = await submitPlanResultWithBackoff(
            server: server, http: http, auth: try auth(), jobId: "job-abc",
            payload: Data("{}".utf8)
        ) { sleeps.append($0) }
        guard case .retryTransient = disposition else {
            Issue.record("expected retryTransient")
            return
        }
        #expect(seen.requests.count == 32)
        #expect(sleeps.prefix(6) == [1, 2, 4, 8, 16, 30])
        #expect(sleeps.max() == 30)
    }

    /// A cancelled task answers `interrupted`, never `retryTransient`. A cancelled
    /// `URLSession` request fails as a transient transport error, so without the check the
    /// backoff would burn all 32 attempts in milliseconds and the caller would read a
    /// stop as ordinary exhaustion, and file a result the server never saw as submitted.
    @Test func submitWithBackoffReportsCancellationAsInterrupted() async throws {
        let (http, server, seen) = try RoutedURLProtocol.serve { _, _ in .reply(503, "down") }
        let auth = try auth()
        let task = Task {
            await submitPlanResultWithBackoff(
                server: server, http: http, auth: auth, jobId: "job-abc",
                payload: Data("{}".utf8)
            ) { _ in }
        }
        task.cancel()
        guard case .interrupted = await task.value else {
            Issue.record("a cancelled submit must report interrupted")
            return
        }
        // Cancelled before the first attempt goes out: nothing is sent, so nothing about
        // the lease or the result has been decided.
        #expect(seen.requests.isEmpty)
    }

    private func wireObject(_ failure: FailureSubmission) throws -> [String: Any] {
        let data = try Coding.encoder.encode(failure)
        return try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
    }
}
