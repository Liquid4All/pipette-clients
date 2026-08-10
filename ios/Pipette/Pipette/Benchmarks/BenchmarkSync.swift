import Foundation

/// Syncs the benchmark catalog from the management server into on-device storage,
/// mirroring the Rust client's two-level conditional pull
/// (`pipette-cli::client::sync::pull_remote_benchmarks`). The process — fetch flow, tolerance
/// policy, and on-disk layout — is documented on `sync(serverUrl:…)`.
///
/// Storage goes through an injected `BenchmarkStore` (file-backed in production,
/// in-memory in tests), so the sync logic doesn't touch `LocalStorage`/`FileManager`
/// directly. Network goes through injected `fetchList`/`fetchDetail` closures.
///
/// Consume side: persisted here and exposed via `storedCatalogEntries()`, which
/// `BenchmarkCatalog` parses as its sole source (there is no bundled catalog).
enum BenchmarkSync {
    /// Persisted sync state: the list-level ETag plus a per-benchmark ETag map,
    /// mirroring the Rust `RemoteSyncState`.
    nonisolated struct SyncState: Codable {
        var benchmarkCount: Int
        var benchmarksEtag: String?
        var benchmarkEtags: [String: String]

        enum CodingKeys: String, CodingKey {
            case benchmarkCount = "benchmark_count"
            case benchmarksEtag = "benchmarks_etag"
            case benchmarkEtags = "benchmark_etags"
        }
    }

    /// Max concurrent per-id detail fetches, mirroring the Rust bounded pool.
    private static let detailConcurrency = 6

    /// Conditional `GET /benchmarks`. Defaults to `ManagementClient.fetchBenchmarks`;
    /// injectable so tests drive the flow without a real network. The endpoint is
    /// public — no client id or signature — so sync needs no registration.
    typealias ListFetcher = (
        _ serverUrl: ServerURL, _ ifNoneMatch: String?
    ) async throws -> ManagementClient.ConditionalGet

    /// Conditional `GET /benchmarks/{id}`. Defaults to `ManagementClient.fetchBenchmark`.
    typealias DetailFetcher = (
        _ serverUrl: ServerURL, _ benchmarkId: String, _ ifNoneMatch: String?
    ) async throws -> ManagementClient.ConditionalGet

    // MARK: - Sync flow

    /// Pull the catalog and persist it; returns the number of stored
    /// (fully-parseable) benchmarks.
    ///
    /// **Two-level, ETag-conditional pull**
    /// 1. **List** — `GET /benchmarks` (definitions only) with the list `ETag` as
    ///    `If-None-Match`. Keep only entries that fully parse; a `304` reuses the
    ///    cached index.
    /// 2. **Per-id** — `GET /benchmarks/{id}` for each kept benchmark with its
    ///    stored `ETag`, bounded-concurrent. Only the per-id response carries the
    ///    eval `samples`, so each benchmark must be fetched individually; a `304`
    ///    keeps the cached detail.
    ///
    /// **Tolerant** — stores only definitions it can fully parse. An unrecognized
    /// `benchmark_type` is skipped quietly; a known type that fails to decode
    /// (schema mismatch) is skipped and logged as an error. Neither is surfaced to
    /// the user, and a per-id failure never aborts the sync.
    @discardableResult
    static func sync(
        serverUrl: ServerURL,
        fetchList: ListFetcher = ManagementClient.fetchBenchmarks,
        fetchDetail: @escaping DetailFetcher = ManagementClient.fetchBenchmark,
        store: BenchmarkStore
    ) async throws -> Int {
        let prior = store.getSyncState()

        // Level 1 — list (definitions only), ETag-conditional. Keep only the
        // entries that fully parse; persist that filtered view as the index.
        let list = try await fetchList(serverUrl, prior?.benchmarksEtag)

        let ids: [String]
        let listEtag: String?
        if let json = list.json {
            let kept = keepParseable(json)
            let data = (try? JSONSerialization.data(withJSONObject: kept.entries)) ?? Data("[]".utf8)
            try? store.putRemoteIndex(data)
            ids = kept.ids
            listEtag = list.etag
            AppLog.benchmarkSync.info("list modified: kept \(ids.count)/\(rawArray(json).count) definitions")
        } else if case let cached = store.listRemoteIndex(), !cached.isEmpty {
            // The stored index only ever holds rows that parsed on the way in, so its
            // ids need no re-filtering.
            ids = cached.map(\.benchmarkId)
            listEtag = list.etag ?? prior?.benchmarksEtag
            AppLog.benchmarkSync.info("list unchanged (304)")
        } else {
            // 304 with no cached index — nothing to work from.
            AppLog.benchmarkSync.warning("list 304 but no cached index")
            return 0
        }

        // Drop detail files for benchmarks no longer in the catalog (the ETag map
        // self-prunes since it's rebuilt from `ids`; the stored details don't).
        store.pruneRemoteDetails(keeping: ids)

        // Level 2 — per-id detail (carries eval samples), ETag-conditional,
        // bounded-concurrent. A 304 keeps the cached detail and its ETag; a detail
        // that doesn't parse (or errors) is skipped without aborting.
        let priorEtags = prior?.benchmarkEtags ?? [:]
        let fetched = await fetchDetails(
            ids: ids, serverUrl: serverUrl, priorEtags: priorEtags, fetch: fetchDetail, store: store)

        let newEtags = fetched.reduce(into: [String: String]()) { map, pair in
            if let etag = pair.1 { map[pair.0] = etag }
        }
        try store.putSyncState(
            SyncState(benchmarkCount: ids.count, benchmarksEtag: listEtag, benchmarkEtags: newEtags))

        AppLog.benchmarkSync.info("synced \(ids.count) benchmarks")
        return ids.count
    }

    /// The persisted server catalog as strict `BenchmarkDefinition`s, from the index.
    /// Empty if nothing synced.
    static func storedDefinitions(store: BenchmarkStore) -> [BenchmarkDefinition] {
        decode(store.listRemoteIndex().map(\.rawJson))
    }

    /// One synced benchmark's full definition, eval `samples` included. `nil` until
    /// that benchmark's detail has been fetched.
    static func storedBenchmark(id: String, store: BenchmarkStore) -> BenchmarkDefinition? {
        store.get(.remote(id))
    }

    // MARK: - Tolerant filtering

    /// Keep only the raw list entries that fully parse into a `BenchmarkDefinition`,
    /// preserving order. Returns the kept raw entries (to persist verbatim) and
    /// their ids (to drive the per-id fetch). Logging is the policy: a known
    /// `benchmark_type` that fails to decode is an error; an unrecognized type is
    /// skipped quietly. Neither is surfaced to the user.
    static func keepParseable(_ json: Data) -> (entries: [Any], ids: [String]) {
        var entries: [Any] = []
        var ids: [String] = []
        for entry in rawArray(json) {
            guard let object = entry as? [String: Any],
                let data = try? JSONSerialization.data(withJSONObject: object)
            else { continue }
            do {
                let def = try JSONDecoder().decode(BenchmarkDefinition.self, from: data)
                entries.append(entry)
                ids.append(def.benchmarkId)
            } catch {
                logSkip(object, error)
            }
        }
        return (entries, ids)
    }

    private static func logSkip(_ object: [String: Any], _ error: Error) {
        let id = object["benchmark_id"] as? String ?? "<no id>"
        let type = object["benchmark_type"] as? String
        if let type, BenchmarkDefinition.knownTypes.contains(type) {
            AppLog.benchmarkSync.warning("skipping '\(id)' (\(type)): schema mismatch: \(error)")
        } else {
            AppLog.benchmarkSync.warning("skipping '\(id)': unrecognized benchmark_type '\(type ?? "nil")'")
        }
    }

    // MARK: - Per-id fetch

    /// Fetch every kept benchmark's detail with at most `detailConcurrency` in
    /// flight. Each result is `(benchmarkId, etag?)`; never throws — a per-id
    /// failure returns the prior ETag so the rest of the sync still completes.
    private static func fetchDetails(
        ids: [String], serverUrl: ServerURL, priorEtags: [String: String],
        fetch: @escaping DetailFetcher, store: BenchmarkStore
    ) async -> [(String, String?)] {
        await withTaskGroup(of: (String, String?).self) { group in
            var iterator = ids.makeIterator()
            func addNext() {
                guard let id = iterator.next() else { return }
                group.addTask {
                    await fetchDetail(
                        id: id, serverUrl: serverUrl, priorEtag: priorEtags[id], fetch: fetch, store: store)
                }
            }

            for _ in 0..<min(detailConcurrency, ids.count) { addNext() }
            var results: [(String, String?)] = []
            for await result in group {
                results.append(result)
                addNext()
            }
            return results
        }
    }

    private static func fetchDetail(
        id: String, serverUrl: ServerURL, priorEtag: String?,
        fetch: DetailFetcher, store: BenchmarkStore
    ) async -> (String, String?) {
        do {
            let response = try await fetch(serverUrl, id, priorEtag)
            guard let json = response.json else {
                return (id, response.etag ?? priorEtag)  // 304 — keep cached detail
            }
            // Store the detail only if it fully parses (definition + samples). A
            // mismatch drops the ETag so a corrected detail is re-fetched next time.
            guard let definition = try? Coding.decoder.decode(BenchmarkDefinition.self, from: json) else {
                AppLog.benchmarkSync.warning("skipping detail '\(id)': schema mismatch")
                return (id, nil)
            }
            try? store.put(.remote, definition)
            return (id, response.etag)
        } catch {
            AppLog.benchmarkSync.error("detail fetch failed for '\(id)': \(error)")
            return (id, priorEtag)
        }
    }

    // MARK: - Decoding helpers

    private static func rawArray(_ json: Data) -> [Any] {
        (try? JSONSerialization.jsonObject(with: json)) as? [Any] ?? []
    }

    private static func decode(_ array: [Any]) -> [BenchmarkDefinition] {
        array.compactMap { entry in
            (try? JSONSerialization.data(withJSONObject: entry))
                .flatMap { try? JSONDecoder().decode(BenchmarkDefinition.self, from: $0) }
        }
    }

    // MARK: - Sync-state persistence

}
