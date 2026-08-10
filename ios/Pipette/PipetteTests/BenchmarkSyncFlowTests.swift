import Foundation
import Testing

@testable import Pipette

// MARK: - Fixtures

private let serverUrl = ServerURL("https://mgmt.test")

private let prefill = #"{"benchmark_id":"p","benchmark_type":"prefill_throughput","parameter_prefill_tokens":512}"#
private let decode = #"{"benchmark_id":"d","benchmark_type":"decode_throughput","parameter_prefill_tokens":256,"parameter_decode_tokens":100}"#
private let unknown = #"{"benchmark_id":"x","benchmark_type":"some_future_metric"}"#
// A `decode_throughput` missing its required `parameter_decode_tokens` — a known
// type with a schema mismatch.
private let decodeBad = #"{"benchmark_id":"d","benchmark_type":"decode_throughput","parameter_prefill_tokens":256}"#

/// A `304 Not Modified` response (no body) for either fetcher.
private let notModified = ManagementClient.ConditionalGet(json: nil, etag: nil)

private func jsonArray(_ entries: [String]) -> Data {
    Data("[\(entries.joined(separator: ","))]".utf8)
}

/// A `200` response carrying `json` with the given `ETag`.
private func body(_ json: String, etag: String) -> ManagementClient.ConditionalGet {
    .init(json: Data(json.utf8), etag: etag)
}

/// A `ListFetcher` returning a `200` with the given entries.
private func listing(_ entries: [String], etag: String) -> BenchmarkSync.ListFetcher {
    { _, _ in .init(json: jsonArray(entries), etag: etag) }
}

private let listUnchanged: BenchmarkSync.ListFetcher = { _, _ in notModified }

/// A `DetailFetcher` whose reply is chosen per benchmark id.
private func details(
    _ reply: @escaping (String) throws -> ManagementClient.ConditionalGet
) -> BenchmarkSync.DetailFetcher {
    { _, id, _ in try reply(id) }
}

/// Exercises the `BenchmarkSync` orchestration end-to-end with injected fetchers
/// and an in-memory `BenchmarkStore` — no real network, no files, no `LocalStorage`
/// global, so each test is fully isolated and runs in parallel.
@Suite
struct BenchmarkSyncFlowTests {
    private func sync(
        list: @escaping BenchmarkSync.ListFetcher, detail: @escaping BenchmarkSync.DetailFetcher,
        store: BenchmarkStore
    ) async throws -> Int {
        try await BenchmarkSync.sync(
            serverUrl: serverUrl, fetchList: list, fetchDetail: detail, store: store)
    }

    private func loadState(_ store: BenchmarkStore) -> BenchmarkSync.SyncState? {
        store.getSyncState()
    }

    // MARK: - List + per-id, all modified

    @Test func fullSyncPersistsIndexDetailsAndEtags() async throws {
        let store = makeTemporaryBenchmarkStore()
        let count = try await sync(
            list: listing([prefill, decode], etag: "\"list-v1\""),
            detail: details { body($0 == "p" ? prefill : decode, etag: "\"\($0)-v1\"") },
            store: store)

        #expect(count == 2)
        #expect(BenchmarkSync.storedDefinitions(store: store).map(\.benchmarkId).sorted() == ["d", "p"])
        #expect(BenchmarkSync.storedBenchmark(id: "p", store: store) != nil)
        #expect(BenchmarkSync.storedBenchmark(id: "d", store: store) != nil)

        let state = loadState(store)
        #expect(state?.benchmarksEtag == "\"list-v1\"")
        #expect(state?.benchmarkEtags["p"] == "\"p-v1\"")
        #expect(state?.benchmarkEtags["d"] == "\"d-v1\"")
    }

    // MARK: - Tolerant filtering through the real flow

    @Test func listStoresOnlyParseableAndSkipsTheRest() async throws {
        let store = makeTemporaryBenchmarkStore()
        // `bad` is a known type missing a required param, `x` an unknown type — both
        // must be dropped from the index and never fetched per-id.
        let bad = #"{"benchmark_id":"bad","benchmark_type":"decode_throughput","parameter_prefill_tokens":256}"#
        let count = try await sync(
            list: listing([prefill, unknown, bad, decode], etag: "\"l\""),
            detail: details { body($0 == "p" ? prefill : decode, etag: "\"\($0)\"") },
            store: store)

        #expect(count == 2)
        #expect(BenchmarkSync.storedDefinitions(store: store).map(\.benchmarkId).sorted() == ["d", "p"])
        #expect(BenchmarkSync.storedBenchmark(id: "x", store: store) == nil)
        #expect(BenchmarkSync.storedBenchmark(id: "bad", store: store) == nil)
    }

    // MARK: - List 304

    @Test func listNotModifiedReusesCachedIndexAndStillRunsPerId() async throws {
        let store = makeTemporaryBenchmarkStore()
        _ = try await sync(
            list: listing([prefill, decode], etag: "\"l\""),
            detail: details { body($0 == "p" ? prefill : decode, etag: "\"\($0)\"") },
            store: store)

        // Everything unchanged now: list + every detail 304.
        let count = try await sync(list: listUnchanged, detail: details { _ in notModified }, store: store)

        #expect(count == 2)
        #expect(BenchmarkSync.storedDefinitions(store: store).count == 2)
        #expect(BenchmarkSync.storedBenchmark(id: "p", store: store) != nil)
        // The per-id ETag survives a 304 (kept from the prior sync).
        #expect(loadState(store)?.benchmarkEtags["p"] == "\"p\"")
    }

    @Test func listNotModifiedWithoutCacheReturnsZero() async throws {
        let store = makeTemporaryBenchmarkStore()
        let count = try await sync(list: listUnchanged, detail: details { _ in notModified }, store: store)
        #expect(count == 0)
        #expect(BenchmarkSync.storedDefinitions(store: store).isEmpty)
    }

    // MARK: - Per-id failure modes

    @Test func perIdSchemaMismatchSkipsStoreAndDropsEtag() async throws {
        let store = makeTemporaryBenchmarkStore()
        // List entry parses (good `decode`); the DETAIL is malformed.
        let count = try await sync(
            list: listing([decode], etag: "\"l\""),
            detail: details { _ in body(decodeBad, etag: "\"d\"") },
            store: store)

        #expect(count == 1)  // kept in the index (list-level parse succeeded)
        #expect(BenchmarkSync.storedBenchmark(id: "d", store: store) == nil)  // detail not stored
        #expect(loadState(store)?.benchmarkEtags["d"] == nil)  // etag dropped → re-fetch next time
    }

    @Test func perIdNetworkErrorDoesNotAbortTheSync() async throws {
        let store = makeTemporaryBenchmarkStore()
        let count = try await sync(
            list: listing([prefill, decode], etag: "\"l\""),
            detail: details { id in
                if id == "p" { throw URLError(.timedOut) }
                return body(decode, etag: "\"d\"")
            },
            store: store)

        #expect(count == 2)
        #expect(BenchmarkSync.storedBenchmark(id: "d", store: store) != nil)
        #expect(BenchmarkSync.storedBenchmark(id: "p", store: store) == nil)  // errored, not stored
    }

    // MARK: - Pruning removed benchmarks

    @Test func orphanDetailsArePrunedWhenBenchmarkLeavesTheCatalog() async throws {
        let store = makeTemporaryBenchmarkStore()
        _ = try await sync(
            list: listing([prefill, decode], etag: "\"l1\""),
            detail: details { body($0 == "p" ? prefill : decode, etag: "\"\($0)\"") },
            store: store)
        #expect(BenchmarkSync.storedBenchmark(id: "p", store: store) != nil)

        // The server dropped "p" (new list ETag → modified).
        let count = try await sync(
            list: listing([decode], etag: "\"l2\""),
            detail: details { _ in body(decode, etag: "\"d2\"") },
            store: store)

        #expect(count == 1)
        #expect(BenchmarkSync.storedBenchmark(id: "p", store: store) == nil)  // orphan detail pruned
        #expect(BenchmarkSync.storedBenchmark(id: "d", store: store) != nil)
        #expect(loadState(store)?.benchmarkEtags["p"] == nil)  // ETag map self-pruned
    }

    @Test func storedReadbackReflectsStoredData() throws {
        let store = makeTemporaryBenchmarkStore()
        try store.putRemoteIndex(jsonArray([prefill]))
        try store.put(.remote, Coding.decoder.decode(BenchmarkDefinition.self, from: Data(prefill.utf8)))

        #expect(BenchmarkSync.storedDefinitions(store: store).map(\.benchmarkId) == ["p"])
        #expect(BenchmarkSync.storedBenchmark(id: "p", store: store) != nil)
        #expect(BenchmarkSync.storedBenchmark(id: "missing", store: store) == nil)
    }

    // MARK: - Catalog consumption (pure parse — no store, no global)

    @Test func mergedParsesSyncedEntriesWithNoBundledFallback() {
        // Empty synced input → empty catalog (no bundled fallback).
        #expect(BenchmarkCatalog.merged(syncedEntries: []).isEmpty)

        let entries: [[String: Any]] = [
            ["benchmark_id": "prefill_throughput_512", "benchmark_type": "prefill_throughput",
             "parameter_prefill_tokens": 512],
            ["benchmark_id": "eval_ifbench", "benchmark_type": "eval", "parameter_eval_id": "ifbench",
             "parameter_dataset_name": "d", "parameter_max_tokens": 64],
        ]
        let all = BenchmarkCatalog.merged(syncedEntries: entries)
        // Exactly the synced entries, sorted by id — nothing bundled bleeds in.
        #expect(all.map(\.benchmarkId) == ["eval_ifbench", "prefill_throughput_512"])
        #expect(!all.contains { $0.benchmarkId == "decode_throughput_256_100" })

        // Eval is in the catalog (resolves for history / lookups) but hidden from
        // the picker by the type filter.
        let selectable = BenchmarkCatalog.selectable(from: all)
        #expect(!selectable.contains { $0.benchmarkId == "eval_ifbench" })
    }

    // MARK: - Store layout

    /// Ids are the filename verbatim, as the crate writes them — so the same benchmark
    /// is the same file on both clients — and an id that could escape the directory is
    /// refused rather than encoded into something unrecognizable.
    @Test func idsAreWrittenVerbatimAndTraversalIsRefused() throws {
        let store = makeTemporaryBenchmarkStore()
        defer { removeStore(store) }

        let definition = try Coding.decoder.decode(
            BenchmarkDefinition.self, from: Data(prefill.utf8))
        try store.put(.remote, definition)
        #expect(FileManager.default.fileExists(
            atPath: store.root.appendingPathComponent("remote/p.json").path))

        let traversal = BenchmarkDefinition.prefillThroughput(
            benchmarkId: "../escape", prefillTokens: 8)
        #expect(throws: BenchmarkStoreError.unsafeBenchmarkId("../escape")) {
            try store.put(.local, traversal)
        }
    }

    /// `clearRemote` drops the synced half and leaves the generated one — the split is
    /// the whole reason the two halves are separate directories.
    @Test func clearRemoteKeepsLocal() throws {
        let store = makeTemporaryBenchmarkStore()
        defer { removeStore(store) }

        try StandardBenchmarks.seedLocal(into: store, kinds: [.prefillThroughput])
        try store.putRemoteIndex(Data("[]".utf8))
        try store.put(.remote, Coding.decoder.decode(
            BenchmarkDefinition.self, from: Data(prefill.utf8)))
        try store.putSyncState(
            BenchmarkSync.SyncState(benchmarkCount: 1, benchmarksEtag: "e", benchmarkEtags: [:]))

        store.clearRemote()

        #expect(store.listRemoteIndex().isEmpty)
        #expect(store.get(.remote("p")) == nil)
        #expect(store.getSyncState() == nil)
        #expect(!store.list(.local).isEmpty)
    }
}
