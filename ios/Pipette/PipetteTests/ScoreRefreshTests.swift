import Foundation
import Testing

@testable import Pipette

/// The job ids a fake `GET /jobs/{id}` was asked for.
private actor AskedJobs {
    private(set) var ids: [String] = []
    func add(_ id: String) { ids.append(id) }
}

/// Ports the crate's `score_refresh_state` and `refresh_scored_job`.
struct ScoreRefreshTests {
    private let id = CellId("c-1")

    private func store() -> ResultsStore {
        ResultsStore(root: FileManager.default.temporaryDirectory
            .appendingPathComponent("PipetteResults-\(UUID().uuidString)", isDirectory: true))
    }

    private func remove(_ store: ResultsStore) { try? FileManager.default.removeItem(at: store.root) }

    /// A synced result for `benchmarkId`, with the submission record that names its job.
    private func synced(_ store: ResultsStore, benchmarkId: String,
                        serverJobId: String? = "job-1") throws {
        try store.saveResult(
            .remoteSynced, id,
            payload: JSONSerialization.data(withJSONObject: ["benchmark_id": benchmarkId]),
            extras: Data("{}".utf8))
        if let serverJobId { try store.saveSubmission(.submitted(serverJobId: serverJobId), id) }
    }

    private func job(_ json: String) throws -> JobResponse {
        try Coding.decoder.decode(JobResponse.self, from: Data(json.utf8))
    }

    // MARK: - score_refresh_state

    /// Only an eval without a valid score on disk needs asking about: a processed job keeps
    /// its score, so a saved one is never re-fetched, but a corrupt one reads as missing.
    @Test(arguments: [
        ("prefill_throughput_256", nil, ScoreRefreshState.notEval),
        ("eval_smoke", nil, .needsRefresh),
        ("eval_smoke", "{}", .alreadyScored),
        ("eval_smoke", "{not json", .needsRefresh),
        (nil, nil, .notEval),
    ] as [(String?, String?, ScoreRefreshState)])
    func theRefreshStateFollowsTypeAndMetrics(
        benchmarkId: String?, metrics: String?, expected: ScoreRefreshState
    ) throws {
        let store = store()
        defer { remove(store) }
        if let benchmarkId { try synced(store, benchmarkId: benchmarkId) }
        if let metrics {
            try Data(metrics.utf8).write(to: try #require(store.metricsPath(.remoteSynced, id)))
        }
        #expect(scoreRefreshState(store: store, id: id) == expected)
    }

    // MARK: - refresh_scored_job

    @Test func aProcessedJobStoresItsMetrics() async throws {
        let store = store()
        defer { remove(store) }
        try synced(store, benchmarkId: "eval_smoke")
        let asked = AskedJobs()
        let reply = try job(#"{"status":"processed","scored_at":"2026-09-01T00:00:00Z","metrics":[{"metric":"accuracy","value":0.75,"unit":"ratio"}]}"#)

        let scored = try await refreshScoredJob(
            store: store, server: try ServerUrl("https://collector.example.com"),
            auth: try testIdentity(), id: id,
            getJob: { _, _, jobId in
                await asked.add(jobId)
                return reply
            })

        #expect(scored)
        #expect(await asked.ids == ["job-1"])
        let saved = try Coding.decoder.decode(
            BenchmarkScoredResult.self,
            from: Data(contentsOf: try #require(store.metricsPath(.remoteSynced, id))))
        #expect(saved.scoredAt == "2026-09-01T00:00:00Z")
        #expect(saved.metrics == [BenchmarkJobMetric(metric: "accuracy", value: 0.75, unit: "ratio")])
        #expect(store.state(of: id) == .scored)
    }

    /// A scored result must never list without a time.
    @Test func aProcessedJobWithoutATimestampIsStampedNow() async throws {
        let store = store()
        defer { remove(store) }
        try synced(store, benchmarkId: "eval_smoke")
        let reply = try job(#"{"status":"processed","metrics":[]}"#)

        let scored = try await refreshScoredJob(
            store: store, server: try ServerUrl("https://collector.example.com"),
            auth: try testIdentity(), id: id, getJob: { _, _, _ in reply })

        #expect(scored)
        let path = try #require(store.metricsPath(.remoteSynced, id))
        let saved = try Coding.decoder.decode(BenchmarkScoredResult.self, from: Data(contentsOf: path))
        #expect(saved.scoredAt?.isEmpty == false)
    }

    @Test func anUnprocessedJobStoresNoScore() async throws {
        let store = store()
        defer { remove(store) }
        try synced(store, benchmarkId: "eval_smoke")
        let reply = try job(#"{"status":"scoring"}"#)

        let scored = try await refreshScoredJob(
            store: store, server: try ServerUrl("https://collector.example.com"),
            auth: try testIdentity(), id: id, getJob: { _, _, _ in reply })

        #expect(!scored)
        #expect(!store.hasValidMetrics(.remoteSynced, id))
        #expect(store.state(of: id) == .submitted)
    }

    /// Without the job id its submission record should carry there is nothing to ask for.
    @Test func aSyncedResultWithoutAJobIdCannotBeRefreshed() async throws {
        let store = store()
        defer { remove(store) }
        try synced(store, benchmarkId: "eval_smoke", serverJobId: nil)

        await #expect(throws: ScoreRefreshError.noServerJobId(resultId: "c-1")) {
            try await refreshScoredJob(
                store: store, server: try ServerUrl("https://collector.example.com"),
                auth: try testIdentity(), id: id,
                getJob: { _, _, _ in throw URLError(.cannotConnectToHost) })
        }
    }
}
