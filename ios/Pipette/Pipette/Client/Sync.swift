import Foundation

// `pipette-cli/src/client/sync.rs`: the submit preflight, the catalog pull
// (`pull_remote_benchmarks`) and the score refresh. The submit itself is
// `ResultSubmissionService`'s batch.

// MARK: - Submit

/// Refuse a stored payload that predates the model/runtime descriptors: the crate's
/// `require_descriptors`. Both submit paths forward the on-disk payload verbatim, so the
/// server would otherwise reject it with nothing said here.
nonisolated func requireDescriptors(_ resultId: String, _ payload: [String: Any]) throws {
    func present(_ key: String) -> Bool {
        guard let value = payload[key] as? String else { return false }
        return !value.isEmpty
    }
    guard present("model_descriptor"), present("runtime_descriptor") else {
        throw ResultSubmissionError.recordedBeforeDescriptors(resultId: resultId)
    }
}

/// Make an eval payload's completion ids unique before it goes up: the crate's
/// `dedupe_completion_ids` preflight. The scorer refuses duplicates with no per-sample
/// diagnostics, so repairing locally keeps a stale on-disk result submittable.
nonisolated func repairCompletionIds(_ resultId: String, _ payload: inout [String: Any]) throws {
    do {
        if let report = try dedupeCompletionIds(&payload) {
            AppLog.resultUploader.warning("result \(resultId): \(report.summary)")
        }
    } catch {
        throw ResultSubmissionError.preflightRepairFailed(
            resultId: resultId, reason: error.localizedDescription)
    }
}

// MARK: - Catalog pull

/// Why a catalog pull failed as a whole, worded as the crate words it.
nonisolated enum BenchmarkSyncError: LocalizedError, Equatable {
    case notModifiedWithoutCachedIndex
    /// A dead or unreachable server, not a bad benchmark.
    case allDetailFetchesFailed(total: Int)

    var errorDescription: String? {
        switch self {
        case .notModifiedWithoutCachedIndex:
            return "server returned 304 but cached benchmark index is unavailable"
        case .allDetailFetchesFailed(let total):
            return "all \(total) benchmark detail fetches failed; keeping the cached catalog untouched"
        }
    }
}

/// The crate's `BENCHMARK_DETAIL_CONCURRENCY_CAP`.
nonisolated let benchmarkDetailConcurrencyCap = 16

/// Conditional `GET /benchmarks`, injectable so tests drive the pull with no network. The
/// endpoint is public, so the pull needs no registration.
typealias BenchmarkListFetcher = (
    _ server: ServerUrl, _ ifNoneMatch: IfNoneMatch?
) async throws -> PipetteMgmtClient.ConditionalResponse<[PipetteMgmtClient.BenchmarkSummary]>

/// Conditional `GET /benchmarks/{id}`.
typealias BenchmarkDetailFetcher = (
    _ server: ServerUrl, _ benchmarkId: String, _ ifNoneMatch: IfNoneMatch?
) async throws -> PipetteMgmtClient.ConditionalResponse<RemoteBenchmark>

// MARK: - Pull

/// Pull the catalog into `store`; returns the state's `benchmarkCount`.
///
/// The list is stored as sent, every row, and rows this build cannot convert are filtered
/// wherever it is read, so the list ETag validates exactly the stored file. A detail that
/// does not convert is logged and not cached; a failed fetch leaves what is cached, and
/// every fetch failing aborts the pull with the store untouched.
@discardableResult
func pullRemoteBenchmarks(
    store: BenchmarkStore,
    server: ServerUrl,
    // Wrapped: the endpoints take an HTTP stack, and these fetchers do not.
    fetchList: BenchmarkListFetcher = { server, ifNoneMatch in
        try await PipetteMgmtClient.ClientApi.listBenchmarksConditional(
            server: server, http: .shared, ifNoneMatch: ifNoneMatch)
    },
    fetchDetail: @escaping BenchmarkDetailFetcher = { server, benchmarkId, ifNoneMatch in
        try await PipetteMgmtClient.ClientApi.getBenchmarkConditional(
            server: server, http: .shared, auth: nil, benchmarkId: benchmarkId,
            ifNoneMatch: ifNoneMatch)
    }
) async throws -> Int {
    // Only a cache from this server is revalidated: `sync server-url=` can aim an
    // unregistered device at another host, whose 304 to a foreign ETag would keep a catalog
    // it never served. A mismatch drops the ETags, not the cache; the pull refills it.
    let previousSync = try store.getSyncState().flatMap { $0.serverUrl == server ? $0 : nil }
    let cachedSummaries = try store.listRemoteIndex()
    let hasCachedIndex = !cachedSummaries.isEmpty || previousSync != nil
    let previousBenchmarksEtag = hasCachedIndex ? previousSync?.benchmarksEtag : nil

    let summaries: [PipetteMgmtClient.BenchmarkSummary]
    let benchmarksEtag: EntityTag?
    switch try await fetchList(server, previousBenchmarksEtag.map(IfNoneMatch.init)) {
    case .modified(let value, let etag):
        summaries = value
        benchmarksEtag = etag
    case .notModified(let etag):
        guard hasCachedIndex else { throw BenchmarkSyncError.notModifiedWithoutCachedIndex }
        summaries = cachedSummaries
        benchmarksEtag = etag ?? previousBenchmarksEtag
    }

    let fetchable = summaries.filter(fetchesDetail)
    try await pullRemoteBenchmarkDetails(
        store: store, server: server, summaries: fetchable,
        revalidates: previousSync != nil, fetch: fetchDetail)
    // Prune against every listed id, not just the rows fetched: a build that reads the
    // others may have cached their details, and a detail that failed or 304'd this round is
    // still in the catalog. After the fetch, so an all-failed abort leaves the store
    // untouched.
    try store.pruneRemoteDetails(keeping: summaries.map(\.benchmarkId))
    try store.putRemoteIndex(summaries)
    try store.putSyncState(RemoteSyncState(
        pulledAt: JobDateFormat.iso8601.string(from: Date()),
        benchmarkCount: fetchable.count,
        serverUrl: server,
        benchmarksEtag: benchmarksEtag))
    AppLog.benchmarkSync.info("synced \(fetchable.count) benchmarks")
    return fetchable.count
}

/// The crate's `fetches_detail`. A row lacks what only the detail carries (a corpus URL,
/// eval samples), so any check past type and id is left to the fetched body's conversion.
private func fetchesDetail(_ row: PipetteMgmtClient.BenchmarkSummary) -> Bool {
    guard BenchmarkDefinition.knownTypes.contains(row.benchmarkType) else {
        logUnrecognizedType(row.benchmarkId, rawType: row.benchmarkType)
        return false
    }
    do {
        _ = try BenchmarkId(row.benchmarkId)
        return true
    } catch {
        logSkippedBenchmark(row.benchmarkId, rawType: row.benchmarkType, error)
        return false
    }
}

/// The rows of the synced list this build converts; empty if nothing synced.
func storedDefinitions(store: BenchmarkStore) -> [BenchmarkDefinition] {
    let summaries: [PipetteMgmtClient.BenchmarkSummary]
    do {
        summaries = try store.listRemoteIndex()
    } catch {
        AppLog.benchmarkSync.error("unreadable remote benchmark index: \(error)")
        return []
    }
    return summaries.compactMap { try? benchmarkDefinitionFromSummary($0) }
}

/// One synced benchmark's full definition, eval `samples` included. `nil` until that
/// benchmark's detail has been fetched, or when the file on disk is unusable.
func storedBenchmark(id: String, store: BenchmarkStore) -> BenchmarkDefinition? {
    guard let benchmarkId = try? BenchmarkId(id) else { return nil }
    do {
        return try store.get(.remote(benchmarkId))
    } catch {
        AppLog.benchmarkSync.error("unreadable remote benchmark \(id): \(error)")
        return nil
    }
}

// MARK: - Details

/// Fetch, convert and cache each detail as the server sent it: the crate's
/// `pull_remote_benchmark_details`.
private func pullRemoteBenchmarkDetails(
    store: BenchmarkStore,
    server: ServerUrl,
    summaries: [PipetteMgmtClient.BenchmarkSummary],
    revalidates: Bool,
    fetch: @escaping BenchmarkDetailFetcher
) async throws {
    let requests = summaries.map { summary in
        BenchmarkDetailRequest(
            benchmarkId: summary.benchmarkId,
            previousEtag: cachedBenchmarkEtag(
                store: store, benchmarkId: summary.benchmarkId, revalidates: revalidates))
    }
    let responses = try await fetchRemoteBenchmarkDetails(
        server: server, requests: requests,
        concurrency: benchmarkDetailConcurrency(), fetch: fetch)

    for response in responses {
        // A failed fetch leaves the cached detail and its ETag; the list keeps its row
        // regardless.
        guard let conditional = response.response else { continue }
        switch conditional {
        case .modified(let value, let etag):
            // An id the `BenchmarkId` grammar refuses fails here, as a conversion.
            let converted: ConvertedRemote
            do {
                converted = try ConvertedRemote(value)
            } catch {
                logSkippedBenchmark(response.benchmarkId, rawType: value.benchmarkType, error)
                continue
            }
            try store.putRemote(converted, etag: etag)
        case .notModified:
            if !((try? BenchmarkId(response.benchmarkId)).map(store.hasRemoteDetail) ?? false) {
                // If-None-Match goes out only when the detail file exists, so the file was
                // deleted mid-sync: the next pull fetches it.
                AppLog.benchmarkSync.warning(
                    "remote benchmark \(escapedQuoted(response.benchmarkId)): 304 without a "
                        + "cached detail; will re-fetch on the next sync")
            }
        }
    }
}

/// The crate's `log_skipped_benchmark`. An unrecognized `benchmark_type` is version skew
/// (info); a known type that fails to convert is a schema disagreement (error). The id and
/// an unknown type are unchecked server text, so they are shown escaped.
private func logSkippedBenchmark(_ benchmarkId: String, rawType: String, _ error: Error) {
    guard BenchmarkDefinition.knownTypes.contains(rawType) else {
        return logUnrecognizedType(benchmarkId, rawType: rawType)
    }
    AppLog.benchmarkSync.error(
        "skipping remote benchmark \(escapedQuoted(benchmarkId)) (\(rawType)): "
            + "schema mismatch (\(error))")
}

private func logUnrecognizedType(_ benchmarkId: String, rawType: String) {
    AppLog.benchmarkSync.info(
        "skipping remote benchmark \(escapedQuoted(benchmarkId)): unrecognized "
            + "benchmark_type \(escapedQuoted(rawType))")
}

/// The ETag stored beside the cached detail, whichever pull or claim wrote it: the crate's
/// `cached_benchmark_etag`. None unless the last pull was from this server. An older build's
/// ETags, in its own `remote/sync.json`, are never read: a `304` to one could keep a lossy
/// or stale definition for good.
func cachedBenchmarkEtag(
    store: BenchmarkStore, benchmarkId: String, revalidates: Bool
) -> EntityTag? {
    guard revalidates, let id = try? BenchmarkId(benchmarkId) else { return nil }
    return store.remoteDetailEtag(id: id)
}

/// The crate's `benchmark_detail_concurrency`.
func benchmarkDetailConcurrency() -> Int {
    min(max(ProcessInfo.processInfo.activeProcessorCount, 1), benchmarkDetailConcurrencyCap)
}

private nonisolated struct BenchmarkDetailRequest: Sendable {
    let benchmarkId: String
    let previousEtag: EntityTag?
}

private nonisolated struct BenchmarkDetailResponse: Sendable {
    let benchmarkId: String
    /// `nil` when the fetch failed.
    let response: PipetteMgmtClient.ConditionalResponse<RemoteBenchmark>?
}

/// Fetch each detail with at most `concurrency` in flight: the crate's
/// `fetch_remote_benchmark_details`. One failure is a `nil` response; every fetch failing
/// is a dead server, so that throws before the caller rewrites anything.
private func fetchRemoteBenchmarkDetails(
    server: ServerUrl, requests: [BenchmarkDetailRequest], concurrency: Int,
    fetch: @escaping BenchmarkDetailFetcher
) async throws -> [BenchmarkDetailResponse] {
    let responses = await withTaskGroup(of: BenchmarkDetailResponse.self) { group in
        var pending = requests.makeIterator()
        func addNext() {
            guard let request = pending.next() else { return }
            group.addTask {
                do {
                    let response = try await fetch(
                        server, request.benchmarkId, request.previousEtag.map(IfNoneMatch.init))
                    return BenchmarkDetailResponse(
                        benchmarkId: request.benchmarkId, response: response)
                } catch {
                    AppLog.benchmarkSync.warning(
                        "sync: benchmark \(escapedQuoted(request.benchmarkId)) detail fetch "
                            + "failed (will retry next pull): \(error)")
                    return BenchmarkDetailResponse(
                        benchmarkId: request.benchmarkId, response: nil)
                }
            }
        }
        for _ in 0..<min(max(concurrency, 1), requests.count) { addNext() }
        var collected: [BenchmarkDetailResponse] = []
        for await response in group {
            collected.append(response)
            addNext()
        }
        return collected
    }
    if !responses.isEmpty, responses.allSatisfy({ $0.response == nil }) {
        throw BenchmarkSyncError.allDetailFetchesFailed(total: responses.count)
    }
    return responses
}

// MARK: - Score refresh
//
// A synced result is keyed by its cell rather than renamed to the server's job id, so the
// job id each request needs is read from the result's `submission.json`.

/// The crate's `ScoreRefreshState`.
nonisolated enum ScoreRefreshState: Equatable, Sendable {
    case notEval
    case alreadyScored
    case needsRefresh
}

/// `GET /jobs/{job_id}`, injectable so a test drives the refresh without a network.
typealias JobFetcher = @Sendable (ServerUrl, ClientIdentity, String) async throws -> JobResponse

nonisolated let fetchJobFromServer: JobFetcher = { server, auth, jobId in
    try await PipetteMgmtClient.ClientApi.getJob(
        server: server, http: .shared, auth: auth, jobId: jobId)
}

nonisolated enum ScoreRefreshError: LocalizedError, Equatable {
    case noServerJobId(resultId: String)

    var errorDescription: String? {
        switch self {
        case .noServerJobId(let resultId):
            return "result \(resultId) is synced but records no server job id"
        }
    }
}

/// The crate's `score_refresh_state`. A processed job's score never changes, so a result
/// with valid metrics on disk is never re-fetched.
nonisolated func scoreRefreshState(store: ResultsStore, id: CellId) -> ScoreRefreshState {
    let payload: [String: Any]
    do {
        payload = try store.loadPayload(.remoteSynced, id)
    } catch {
        AppLog.results.info("result \(id.value): cannot load payload, skipping refresh: \(error)")
        return .notEval
    }
    let benchmarkId = payload["benchmark_id"] as? String ?? ""
    guard BenchmarkType.fromId(benchmarkId) == .eval else { return .notEval }
    if store.hasValidMetrics(.remoteSynced, id) {
        AppLog.results.info("result \(id.value): already scored, skipping refresh")
        return .alreadyScored
    }
    return .needsRefresh
}

/// Save a synced result's score once its job is processed: the crate's
/// `refresh_scored_job`. Returns whether the result is now scored.
nonisolated func refreshScoredJob(
    store: ResultsStore,
    server: ServerUrl,
    auth: ClientIdentity,
    id: CellId,
    getJob: JobFetcher = fetchJobFromServer
) async throws -> Bool {
    guard let jobId = store.loadSubmission(id)?.serverJobId, !jobId.isEmpty else {
        throw ScoreRefreshError.noServerJobId(resultId: id.value)
    }
    AppLog.results.info("refreshing job \(jobId)")
    let job = try await getJob(server, auth, jobId)
    guard job.status == .processed else {
        AppLog.results.info("job \(jobId) not processed yet (status=\(job.status))")
        return false
    }
    let scored = BenchmarkScoredResult(
        scoredAt: job.scoredAt?.value ?? JobDateFormat.iso8601.string(from: Date()),
        metrics: job.metrics)
    try store.saveMetrics(.remoteSynced, id, scored)
    AppLog.results.info("job \(jobId) processed")
    return true
}
