import Foundation
import Testing

@testable import Pipette

/// The claim loop: `pipette-cli`'s `commands/worker.rs` tests, and the loop's own.
@MainActor
struct PlannerWorkerTests {
    private func auth() throws -> ClientIdentity { try testIdentity() }

    private static let llamaRuntime: [String: Any] = [
        "type": "llamacpp_ios_pipette", "repository_url": "github.com/ggml-org/llama.cpp",
        "repository_version": "b1", "flavor": "ios-arm64",
    ]

    private func session(_ server: ServerUrl) throws -> MgmtSession {
        MgmtSession(
            registration: IdentityRegistration(
                clientId: try ClientId("client-1"),
                status: "approved",
                serverUrl: server,
                organization: "acme",
                contactEmail: "ops@example.com",
                registeredAt: "2026-01-01T00:00:00Z"),
            auth: try auth())
    }

    /// Polls rather than sleeping a fixed span, which a loaded simulator would overrun.
    private func waitUntil(
        _ what: String,
        sourceLocation: SourceLocation = #_sourceLocation,
        _ condition: () -> Bool
    ) async {
        let deadline = ContinuousClock.now.advanced(by: .seconds(10))
        while ContinuousClock.now < deadline {
            if condition() { return }
            try? await Task.sleep(for: .milliseconds(5))
        }
        Issue.record("timed out waiting for \(what)", sourceLocation: sourceLocation)
    }

    /// Heartbeat period the tests below run the loop at.
    private static let fastTick: TimeInterval = 0.02
    /// Long enough for many ``fastTick`` intervals to pass, so "nothing more happened" is
    /// a claim about the task rather than about the clock.
    private static let severalTicks: Duration = .milliseconds(300)

    // MARK: - How a session ends

    /// A 403 stops the loop as `max-jobs` does, but `pipette worker` exits non-zero there:
    /// a fleet script keyed on `BENCH_DONE` must not read "not approved" as "did its work".
    @Test func doingTheAskedWorkSucceeds() {
        #expect(PlannerWorker.SessionOutcome.completed(jobs: 3).isSuccess)
        #expect(PlannerWorker.SessionOutcome.stopped.isSuccess)
    }

    /// A fatal claim error and a failed profile refresh are the crate's `Err` from
    /// `execute`, so they exit non-zero like a refusal.
    @Test func aRefusedUnstartedOrFailedSessionFails() {
        #expect(!PlannerWorker.SessionOutcome.notApproved.isSuccess)
        #expect(!PlannerWorker.SessionOutcome.notStarted.isSuccess)
        #expect(!PlannerWorker.SessionOutcome.failed("fatal claim error: 400").isSuccess)
    }

    // MARK: - Claiming

    /// `claim_with_retry`: one second, doubling per transient, capped at a minute.
    @Test func claimTransientsBackOffDoublingToAMinute() async throws {
        let (http, server, seen) = try RoutedURLProtocol.serve { _, index in
            index < 8 ? .reply(503, "down") : .reply(204)
        }
        var sleeps: [TimeInterval] = []
        let poll = await PlannerWorker.claimWithRetry(
            server: server, http: http, auth: try auth()
        ) { sleeps.append($0) }
        guard case .idle = poll else {
            Issue.record("expected idle after the transients, got \(poll)")
            return
        }
        #expect(sleeps == [1, 2, 4, 8, 16, 32, 60, 60])
        #expect(seen.requests.count == 9)
    }

    /// Only a transient is retried: anything else is the loop's to act on at once.
    @Test func aClaimAnswerThatIsNotTransientIsHandedBackWithoutWaiting() async throws {
        var sleeps: [TimeInterval] = []
        let (refusedHttp, refused, _) = try RoutedURLProtocol.serve { _, _ in .reply(403) }
        let refusal = await PlannerWorker.claimWithRetry(
            server: refused, http: refusedHttp, auth: try auth()
        ) { sleeps.append($0) }
        guard case .refused = refusal else {
            Issue.record("expected refused, got \(refusal)")
            return
        }
        let (badHttp, bad, _) = try RoutedURLProtocol.serve { _, _ in .reply(400, "bad") }
        let failure = await PlannerWorker.claimWithRetry(
            server: bad, http: badHttp, auth: try auth()
        ) { sleeps.append($0) }
        guard case .fatal = failure else {
            Issue.record("expected fatal, got \(failure)")
            return
        }
        #expect(sleeps.isEmpty)
    }

    // MARK: - Submitting

    /// What max-jobs counts: every claim `runClaimedJob` returns from, as the crate
    /// counts every `Ok` of `run_claimed_job`. A dropped submit and an exhausted backoff
    /// are answers `post_completion` returns; only a definitive refusal throws, and only
    /// that one leaves the count alone.
    @Test func aDroppedOrAbandonedSubmitIsAnAnswerAndOnlyAFatalOneThrows() async throws {
        let body = Data("{}".utf8)
        let (droppedHttp, dropped, _) = try RoutedURLProtocol.serve { _, _ in .reply(409) }
        let droppedAnswer = try await PlannerWorker.postCompletion(
            server: dropped, http: droppedHttp, auth: try auth(), jobId: "job-1",
            payload: body) { _ in }
        guard case .dropped = droppedAnswer else {
            Issue.record("expected dropped")
            return
        }

        let (downHttp, down, _) = try RoutedURLProtocol.serve { _, _ in .reply(503) }
        let abandoned = try await PlannerWorker.postCompletion(
            server: down, http: downHttp, auth: try auth(), jobId: "job-1",
            payload: body) { _ in }
        guard case .retryTransient = abandoned else {
            Issue.record("expected the exhausted backoff to answer retryTransient")
            return
        }

        let (badHttp, bad, _) = try RoutedURLProtocol.serve { _, _ in .reply(400, "bad") }
        await #expect(throws: PipetteMgmtClient.Error.self) {
            try await PlannerWorker.postCompletion(
                server: bad, http: badHttp, auth: try self.auth(), jobId: "job-1",
                payload: body) { _ in }
        }
    }

    /// A failure goes through the same backoff a success does (`submit_failure` →
    /// `post_completion`): a failure lost to one `5xx` would leave the lease to lapse
    /// and the job to be re-served to every device that cannot run it.
    @Test func aFailureIsSubmittedThroughTheSameBackoff() async throws {
        let (http, server, seen) = try RoutedURLProtocol.serve { _, index in
            index == 0 ? .reply(503, "down") : .reply(202, #"{"job_id":"job-1"}"#)
        }
        let job = try ClaimTests.claimedJob(spec: nil)
        var sleeps: [TimeInterval] = []
        let disposition = try await PlannerWorker.submitFailure(
            job, reason: "[ts] oom", retriable: true, deviceTime: .seconds(3), server: server,
            http: http, auth: try auth(), clientVersion: "9.9.9"
        ) { sleeps.append($0) }
        guard case .accepted = disposition else {
            Issue.record("expected accepted")
            return
        }
        #expect(sleeps == [1])
        #expect(seen.methodsAndPaths == ["POST /benchmarks", "POST /benchmarks"])
        let lastBody = try #require(seen.bodies.last)
        let sent = try #require(JSONSerialization.jsonObject(with: lastBody) as? [String: Any])
        #expect(sent["message_type"] as? String == "failure")
        #expect(sent["job_id"] as? String == "job-1")
        #expect(sent["retriable"] as? Bool == true)
    }

    /// Filed where the submit sweep files an accepted result, so nothing posts it again.
    /// The crate's `a_successful_run_that_kept_its_lease_is_submitted_and_stored`.
    @Test func storeCompletedResultFilesUnderSyncedWithTheJobId() throws {
        let results = ResultsStore(root: FileManager.default.temporaryDirectory
            .appendingPathComponent("PipetteResults-\(UUID().uuidString)", isDirectory: true))
        defer { try? FileManager.default.removeItem(at: results.root) }
        let cellId = CellId(UUID().uuidString)
        try results.saveResult(
            .remotePending, cellId,
            payload: Data(#"{"benchmark_id":"prefill_throughput_512"}"#.utf8),
            extras: Data("{}".utf8))
        let collector = try ServerUrl("https://collector.example.com")

        PlannerWorker.storeCompletedResult(
            results: results, jobId: "job-1", cellId: cellId, serverJobId: "job-1",
            collector: collector)

        #expect(results.location(of: cellId) == .remoteSynced)
        let record = try #require(results.loadSubmission(cellId))
        #expect(record.status == .submitted)
        #expect(record.serverJobId == "job-1")
        #expect(record.collector == collector)
        let payload = try #require(results.loadPayload(cellId))
        #expect(String(decoding: payload, as: UTF8.self).contains("prefill_throughput_512"))

        // With the manifest's ack beside it, the cell is not offered for upload again.
        var acked = cell(id: cellId, benchmarkId: "prefill_throughput_512")
        acked.serverJobId = "job-1"
        #expect(!results.isUnsubmitted(acked))
    }

    // MARK: - The claim's benchmark

    /// One signed conditional read, answered `304`. `ClaimedBenchmarkCheckTests` covers every
    /// other answer.
    @Test func aCachedBenchmarkTheServerConfirmsIsNotFetched() async throws {
        let store = makeTemporaryBenchmarkStore()
        defer { removeStore(store) }
        let cached = BenchmarkDefinition.prefillThroughput(PrefillThroughputBenchmark(
            benchmarkId: BenchmarkId(validated: "prefill_throughput_512"),
            parameterPrefillTokens: 512))
        try store.cacheRemote(cached, etag: #""p""#)
        let (http, server, seen) = try RoutedURLProtocol.serve { request, _ in
            request.value(forHTTPHeaderField: "If-None-Match") == #""p""# ? .reply(304) : .reply(500)
        }

        let definition = try await PlannerWorker.ensureClaimBenchmarkCached(
            store: store, server: server, http: http, auth: try testIdentity(),
            benchmarkId: "prefill_throughput_512")

        #expect(definition == cached)
        #expect(seen.methodsAndPaths == ["GET /benchmarks/prefill_throughput_512"])
        #expect(seen.requests.first?.value(forHTTPHeaderField: "X-Client-Id") == "client-1")
        #expect(seen.requests.first?.value(forHTTPHeaderField: "X-Signature") != nil)
    }

    /// A benchmark the catalog does not hold is fetched and cached on the remote half,
    /// eval included: this is what lets an eval claim run rather than be retired. The
    /// fetch is signed as the claiming client.
    @Test func anUncachedBenchmarkIsFetchedAndCached() async throws {
        let store = makeTemporaryBenchmarkStore()
        defer { removeStore(store) }
        let served = BenchmarkDefinition.eval(EvalBenchmark(
            benchmarkId: BenchmarkId(validated: "eval_x"),
            parameterEvalId: EvalId("ifbench"),
            parameterDatasetName: "d",
            parameterMaxTokens: 16,
            parameterMcqChoices: nil,
            samples: nil))
        let body = String(decoding: try Coding.encoder.encode(served), as: UTF8.self)
        let (http, server, seen) = try RoutedURLProtocol.serve { _, _ in .reply(200, body) }

        let definition = try await PlannerWorker.ensureClaimBenchmarkCached(
            store: store, server: server, http: http, auth: try testIdentity(),
            benchmarkId: "eval_x")

        #expect(definition == served)
        #expect(seen.methodsAndPaths == ["GET /benchmarks/eval_x"])
        #expect(seen.requests.first?.value(forHTTPHeaderField: "X-Client-Id") == "client-1")
        #expect(seen.requests.first?.value(forHTTPHeaderField: "X-Signature") != nil)
        #expect(try store.get(.remote("eval_x")) == served)
    }

    /// A file cached before `parameter_num_images` has no image count, so running it would run
    /// and declare one image (`BenchmarkParams` is built from the definition), which mgmt
    /// refuses. Fetched once: the second claim revalidates what the first cached.
    @Test func aCacheFileAnOlderBuildWroteIsRefetchedNotRun() async throws {
        let store = makeTemporaryBenchmarkStore()
        defer { removeStore(store) }
        let legacy = try store.writeLegacyImg80()
        let (http, server, seen) = try RoutedURLProtocol.serve { request, _ in
            request.value(forHTTPHeaderField: "If-None-Match") == #""img80""#
                ? .reply(304) : .reply(img80Wire, etag: #""img80""#)
        }
        let claim = {
            try await PlannerWorker.ensureClaimBenchmarkCached(
                store: store, server: server, http: http, auth: try testIdentity(),
                benchmarkId: img80Id)
        }

        let definition = try await claim()

        guard case let .vlThroughput(vl) = definition else {
            Issue.record("expected vl_throughput, got \(definition)")
            return
        }
        #expect(vl.parameterNumImages.value == 80)
        guard case let .vlThroughput(_, _, _, _, numImages) = BenchmarkParams(definition) else {
            Issue.record("expected vl_throughput params")
            return
        }
        #expect(numImages.value == 80, "the result would declare it")
        #expect(seen.methodsAndPaths == ["GET /benchmarks/\(img80Id)"])
        #expect(FileManager.default.fileExists(atPath: legacy.path),
                "the older build's file went")

        #expect(try await claim() == definition)
        #expect(seen.methodsAndPaths == [
            "GET /benchmarks/\(img80Id)", "GET /benchmarks/\(img80Id)",
        ])
    }

    /// Only the server not knowing the id, or a body this build cannot read, is an
    /// unknown benchmark; both are terminal.
    @Test func aBenchmarkNobodyCanResolveIsTerminal() async throws {
        let store = makeTemporaryBenchmarkStore()
        defer { removeStore(store) }
        let (missingHttp, missing, _) = try RoutedURLProtocol.serve { _, _ in
            .reply(404, "benchmark not found")
        }
        await expectTerminal {
            try await PlannerWorker.ensureClaimBenchmarkCached(
                store: store, server: missing, http: missingHttp, auth: try testIdentity(),
                benchmarkId: "gone")
        }

        let (oddHttp, odd, _) = try RoutedURLProtocol.serve { _, _ in
            .reply(200, #"{"benchmark_id":"odd","benchmark_type":"not_a_type"}"#)
        }
        await expectTerminal {
            try await PlannerWorker.ensureClaimBenchmarkCached(
                store: store, server: odd, http: oddHttp, auth: try testIdentity(),
                benchmarkId: "odd")
        }
        #expect(try store.get(.remote("odd")) == nil)
        #expect(!FileManager.default.fileExists(atPath: store.remoteDetailPath("odd").path),
                "a body that does not convert was cached")
    }

    /// The cache is an optimization for the next claim, not a precondition for this one.
    @Test func aBenchmarkCacheWriteFailureIsAWarningNotARefusal() async throws {
        let store = makeTemporaryBenchmarkStore()
        defer { removeStore(store) }
        // A file where the remote half of the catalog goes: the read below misses on it,
        // and the write cannot create the directory over it.
        try FileManager.default.createDirectory(at: store.root, withIntermediateDirectories: true)
        try Data("not a directory".utf8).write(
            to: store.root.appendingPathComponent(BenchmarkSource.remote.rawValue))
        let served = BenchmarkDefinition.prefillThroughput(PrefillThroughputBenchmark(
            benchmarkId: BenchmarkId(validated: "prefill_throughput_512"),
            parameterPrefillTokens: 512))
        let body = String(decoding: try Coding.encoder.encode(served), as: UTF8.self)
        let (http, server, seen) = try RoutedURLProtocol.serve { _, _ in .reply(200, body) }

        let definition = try await PlannerWorker.ensureClaimBenchmarkCached(
            store: store, server: server, http: http, auth: try testIdentity(),
            benchmarkId: "prefill_throughput_512")

        #expect(definition == served)
        #expect(seen.methodsAndPaths == ["GET /benchmarks/prefill_throughput_512"])
        #expect(try store.get(.remote("prefill_throughput_512")) == nil, "nothing was cached")
    }

    /// A fetch that fails in transit says nothing about the benchmark.
    @Test func aBenchmarkFetchThatFailsInTransitStaysRetriable() async throws {
        let store = makeTemporaryBenchmarkStore()
        defer { removeStore(store) }
        let (http, server, _) = try RoutedURLProtocol.serve { _, _ in .reply(503, "down") }
        do {
            _ = try await PlannerWorker.ensureClaimBenchmarkCached(
                store: store, server: server, http: http, auth: try testIdentity(),
                benchmarkId: "b")
            Issue.record("expected the fetch to fail")
        } catch {
            #expect(classifyRunError(error))
        }
    }

    private func expectTerminal(
        _ body: () async throws -> BenchmarkDefinition,
        sourceLocation: SourceLocation = #_sourceLocation
    ) async {
        do {
            _ = try await body()
            Issue.record("expected a refusal", sourceLocation: sourceLocation)
        } catch {
            guard case .unknownBenchmark = error as? PlannerWorker.WorkerResolveError else {
                Issue.record("expected unknownBenchmark, got \(error)", sourceLocation: sourceLocation)
                return
            }
            #expect(!classifyRunError(error), sourceLocation: sourceLocation)
        }
    }

    // MARK: - The manifest a claim becomes

    /// The cell the worker runs is the claim's own, whole: its coordinate and pin (not a
    /// local copy of the repo) and its `benchmark_flags`. The coordinate is one this
    /// device does not hold; fetching it is `ensureModel`'s, covered by `EnsureModelTests`.
    @Test func theManifestCellIsTheClaimsOwn() throws {
        let job = try ClaimTests.claimedJob(spec: [
            "benchmark": "prefill_throughput_256",
            "runtime": Self.llamaRuntime,
            "model": [
                "type": "gguf_text", "source": "huggingface",
                "org": "o", "repo_name": "never-downloaded",
                "revision": "v1", "path": "m.gguf",
            ],
            "benchmark_flags": [
                "benchmark_type": "prefill_throughput",
                "runtime_type": "llamacpp_ios_pipette",
                "model_type": "gguf_text",
                "readiness": ["max_wait_secs": 1800, "skip_thermal": true],
            ],
        ])
        let spec = try ClientRunSpec.validated(try runSpecFromClaim(job))

        let manifest = PlannerWorker.shared.makeManifest(
            job: job, spec: spec,
            benchmark: .prefillThroughput(PrefillThroughputBenchmark(
                benchmarkId: BenchmarkId(validated: "prefill_throughput_256"),
                parameterPrefillTokens: 256)))

        let cell = try #require(manifest.cells.first)
        #expect(cell.spec.model.repo?.revision?.value == "v1")
        #expect(cell.spec.model.repo?.repoName.value == "never-downloaded")
        // The plan-authored readiness gate reaches `JobExecutor`.
        #expect(cell.spec.benchmarkFlags?.readiness?.maxWaitSecs == 1800)
        #expect(cell.spec.benchmarkFlags?.readiness?.skipThermal == true)
    }

    /// The cell is typed from the resolved benchmark, not from its id: an `eval` claim,
    /// which the id parse cannot classify, still becomes a cell instead of an empty job
    /// reported as an unknown benchmark.
    @Test func anEvalClaimBecomesACell() throws {
        let job = try ClaimTests.claimedJob(benchmarkId: "eval_ifbench", spec: [
            "benchmark": "eval_ifbench",
            "runtime": Self.llamaRuntime,
            "model": [
                "type": "gguf_text", "source": "huggingface",
                "org": "o", "repo_name": "r", "path": "m.gguf",
            ],
        ])
        let spec = try ClientRunSpec.validated(try runSpecFromClaim(job))

        let manifest = PlannerWorker.shared.makeManifest(
            job: job, spec: spec,
            benchmark: .eval(EvalBenchmark(
                benchmarkId: BenchmarkId(validated: "eval_ifbench"),
                parameterEvalId: EvalId("ifbench"),
                parameterDatasetName: "d",
                parameterMaxTokens: 16,
                parameterMcqChoices: nil,
                samples: nil)))

        let cell = try #require(manifest.cells.first)
        #expect(manifest.cells.count == 1)
        #expect(cell.benchmarkType == .eval)
        #expect(cell.benchmarkId == "eval_ifbench")
    }

    // MARK: - The heartbeat that holds the lease

    @Test func theHeartbeatTicksUntilItIsCancelled() async throws {
        let (http, server, seen) = try RoutedURLProtocol.serve { _, _ in .reply(200) }
        let leaseLost = PlannerWorker.LeaseLostFlag()
        let heartbeat = PlannerWorker.spawnHeartbeat(
            jobId: "job-1", every: Self.fastTick, session: try session(server), http: http,
            leaseLost: leaseLost)

        await waitUntil("two heartbeats") { seen.requests.count >= 2 }
        heartbeat.cancel()
        _ = await heartbeat.value

        let ticks = seen.requests.count
        try await Task.sleep(for: Self.severalTicks)
        #expect(seen.requests.count == ticks, "the loop kept beating after it was cancelled")
        let paths = Set(seen.methodsAndPaths)
        #expect(paths == ["PUT /plans/job-1/heartbeat"])
        #expect(!leaseLost.isLost, "a renewed lease was never lost")
    }

    /// A lease the server has given away sets the flag the run's submit consults, and the
    /// loop stops: there is no cancel path, so it has nothing left to do.
    @Test func aSupersededLeaseFlagsLeaseLostAndStopsBeating() async throws {
        let (http, server, seen) = try RoutedURLProtocol.serve { _, _ in .reply(409, "superseded") }
        let leaseLost = PlannerWorker.LeaseLostFlag()
        let heartbeat = PlannerWorker.spawnHeartbeat(
            jobId: "job-1", every: Self.fastTick, session: try session(server), http: http,
            leaseLost: leaseLost)

        await waitUntil("the lease-lost flag") { leaseLost.isLost }
        try await Task.sleep(for: Self.severalTicks)

        #expect(seen.requests.count == 1, "it went on beating on a lease it had lost")
        heartbeat.cancel()
    }

    /// A transient is not a lost lease: the loop retries after ``PlannerWorker/retryWait``.
    /// A cancel mid-wait still returns at once, so a finished run's submit never sits
    /// behind a retry timer.
    @Test func aTransientHeartbeatBacksOffAndStillCancelsAtOnce() async throws {
        #expect(PlannerWorker.retryWait > Self.fastTick, "the wait must be the longer one")
        let (http, server, seen) = try RoutedURLProtocol.serve { _, _ in .reply(503, "down") }
        let leaseLost = PlannerWorker.LeaseLostFlag()
        let heartbeat = PlannerWorker.spawnHeartbeat(
            jobId: "job-1", every: Self.fastTick, session: try session(server), http: http,
            leaseLost: leaseLost)

        await waitUntil("the first heartbeat") { seen.requests.count >= 1 }
        try await Task.sleep(for: Self.severalTicks)
        #expect(seen.requests.count == 1, "a transient retried at the tick interval")

        let stopping = ContinuousClock.now
        heartbeat.cancel()
        _ = await heartbeat.value
        #expect(stopping.duration(to: .now) < .seconds(1), "the cancel waited out the retry timer")
        #expect(!leaseLost.isLost, "a transient is not a lost lease")
    }

    // MARK: - What a finished run owes the server

    /// The server has already given the job to another device, so a lost lease is neither
    /// submitted nor filed as synced.
    @Test(arguments: [true, false])
    func aRunWhoseLeaseWasLostIsNotSubmitted(succeeded: Bool) async throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let (http, server, seen) = try RoutedURLProtocol.serve { _, _ in
            .reply(202, #"{"job_id":"job-1"}"#)
        }
        let (job, run) = try completedRun(ws: ws)

        try await report(
            job, runResult: succeeded ? .success(run) : .failure(URLError(.timedOut)),
            leaseLost: true, server: server, http: http, ws: ws)

        #expect(seen.requests.isEmpty, "a lost lease must not be submitted")
        #expect(ws.results.location(of: run.cellId) != .remoteSynced,
                "nothing was submitted, so nothing is filed as synced")
    }

    /// A run that kept its lease is posted against the claim and then filed as synced, so
    /// no later sweep sends it again.
    @Test func aSuccessfulRunThatKeptItsLeaseIsSubmittedAndFiled() async throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let (http, server, seen) = try RoutedURLProtocol.serve { _, _ in
            .reply(202, #"{"job_id":"job-1"}"#)
        }
        let (job, run) = try completedRun(ws: ws)

        try await report(
            job, runResult: .success(run), leaseLost: false, server: server, http: http,
            ws: ws)

        #expect(seen.methodsAndPaths == ["POST /benchmarks"])
        let body = try #require(seen.bodies.last)
        let sent = try #require(JSONSerialization.jsonObject(with: body) as? [String: Any])
        // The claim echo the server's binding check looks for.
        #expect(sent["job_id"] as? String == "job-1")
        #expect(ws.results.location(of: run.cellId) == .remoteSynced)
        let record = try #require(ws.results.loadSubmission(run.cellId))
        #expect(record.serverJobId == "job-1")
    }

    /// A run that failed, and the disposition its failure must reach the server with.
    enum FailedRun: CaseIterable, Sendable {
        case timedOut, unknownBenchmark, vaultRefused, vaultUnavailable

        var error: any Error {
            let url = "https://mgmt.test/vault/resources/vault_1/content"
            return switch self {
            case .timedOut: URLError(.timedOut)
            case .unknownBenchmark: PlannerWorker.WorkerResolveError.unknownBenchmark("gone")
            case .vaultRefused: VaultFetchError.refused(
                resourceId: VaultResourceId(validated: "vault_1"), status: 403, detail: "refused")
            case .vaultUnavailable: VaultFetchError.mgmt(
                .httpStatus(method: "GET", url: url, status: 503, body: ""))
            }
        }

        var retriable: Bool {
            switch self {
            case .timedOut, .vaultUnavailable: true
            case .unknownBenchmark, .vaultRefused: false
            }
        }
    }

    /// A run that failed is still an answer the server is owed, carrying the disposition
    /// ``classifyRunError(_:)`` decided: a local failure or a vault `503` is retriable, a
    /// benchmark nobody can resolve or a vault `403` is not. Whatever the run left on disk
    /// stays where it is: only a submitted result is filed as synced.
    @Test(arguments: FailedRun.allCases)
    func aFailedRunIsReportedAgainstTheClaim(failed: FailedRun) async throws {
        let (error, retriable) = (failed.error, failed.retriable)
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let (http, server, seen) = try RoutedURLProtocol.serve { _, _ in
            .reply(202, #"{"job_id":"job-1"}"#)
        }
        let (job, run) = try completedRun(ws: ws)

        try await report(
            job, runResult: .failure(error), leaseLost: false, server: server, http: http,
            ws: ws)

        #expect(seen.methodsAndPaths == ["POST /benchmarks"])
        let body = try #require(seen.bodies.last)
        let sent = try #require(JSONSerialization.jsonObject(with: body) as? [String: Any])
        #expect(sent["message_type"] as? String == "failure")
        #expect(sent["job_id"] as? String == "job-1")
        #expect(sent["retriable"] as? Bool == retriable)
        // The run's time from the claim, which `report` passes as three seconds.
        #expect(sent["device_time_ms"] as? Int == 3000)
        #expect(ws.results.location(of: run.cellId) != .remoteSynced,
                "a failure filed a result as submitted")
    }

    /// The checkpoint is retired once the server holds the result, and only then. A refused
    /// result or a lost lease leaves it, so the job's next run resumes every sample.
    @Test(arguments: [(202, false, false), (400, false, true), (202, true, true)])
    func anEvalCheckpointIsKeptUntilTheServerHasTheResult(
        status: Int, leaseLost: Bool, kept: Bool
    ) async throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let (http, server, _) = try RoutedURLProtocol.serve { _, _ in
            .reply(status, #"{"job_id":"job-1"}"#)
        }
        let request = payloadRequest(
            model: try ggufTextResolved(),
            benchmark: .eval(EvalBenchmark(
                benchmarkId: BenchmarkId(validated: "eval_smoke"),
                parameterEvalId: EvalId("math_500"),
                parameterDatasetName: "d",
                parameterMaxTokens: 8,
                samples: [EvalSample(id: "s0", messages: [])])))
        let checkpoint = try ws.evalCompletions.open(request)
        try checkpoint.append(.completed(id: "s0", completion: "answer", stopReason: .eos,
                                         stopDetail: nil, completionTokens: 1))
        checkpoint.close()
        let (job, run) = try completedRun(
            ws: ws, evalCheckpoint: try runRequestDigest(request))

        do {
            try await report(
                job, runResult: .success(run), leaseLost: leaseLost, server: server, http: http,
                ws: ws)
        } catch {
            // Where the crate reports a refusal as a retriable failure, this throws it.
            #expect(status == 400, "only a refused submit throws: \(error)")
        }

        let reopened = try ws.evalCompletions.open(request)
        defer { reopened.close() }
        #expect(reopened.contains("s0") == kept)
    }

    /// Hand a finished run to the worker's report step over the stub at `server`.
    private func report(
        _ job: ClaimedJob,
        runResult: Result<PlannerWorker.CompletedRun, any Error>,
        leaseLost: Bool,
        server: ServerUrl,
        http: HttpClient,
        ws: PipetteWorkspace
    ) async throws {
        try await PlannerWorker.shared.submitFinishedRun(
            job,
            runResult: runResult,
            leaseLost: leaseLost,
            elapsed: .seconds(3),
            session: try session(server),
            http: http,
            ws: ws,
            jobStore: JobStore(ws: ws)
        )
    }

    /// A finished run and the claim it answers: the manifest the worker would have built,
    /// with the payload the executor would have left in the store.
    private func completedRun(
        ws: PipetteWorkspace,
        evalCheckpoint: EvalRunDigest? = nil
    ) throws -> (job: ClaimedJob, run: PlannerWorker.CompletedRun) {
        let job = try ClaimTests.claimedJob(spec: [
            "benchmark": "prefill_throughput_256",
            "runtime": Self.llamaRuntime,
            "model": [
                "type": "gguf_text", "source": "huggingface",
                "org": "o", "repo_name": "r", "path": "m.gguf",
            ],
        ])
        let spec = try ClientRunSpec.validated(try runSpecFromClaim(job))
        let manifest = PlannerWorker.shared.makeManifest(
            job: job, spec: spec,
            benchmark: .prefillThroughput(PrefillThroughputBenchmark(
                benchmarkId: BenchmarkId(validated: "prefill_throughput_256"),
                parameterPrefillTokens: 256)))
        let cellId = try #require(manifest.cells.first).cellId
        // Where the executor leaves it: the submit files this, it does not write it.
        try writePayload(ws: ws, cellId: cellId, benchmarkId: "prefill_throughput_256")
        let stored = try #require(ws.results.loadPayload(cellId))
        let payload = try #require(JSONSerialization.jsonObject(with: stored) as? [String: Any])
        return (
            job,
            PlannerWorker.CompletedRun(manifest: manifest, cellId: cellId, payload: payload,
                                       evalCheckpoint: evalCheckpoint)
        )
    }
}
