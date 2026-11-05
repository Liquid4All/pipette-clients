import Foundation
import Testing

@testable import Pipette

// MARK: - Fixtures

private func testServer() throws -> ServerUrl { try ServerUrl("https://mgmt.test") }

private typealias DetailResponse = PipetteMgmtClient.ConditionalResponse<RemoteBenchmark>

private let prefill = #"{"benchmark_id":"p","benchmark_type":"prefill_throughput","parameter_prefill_tokens":512}"#
private let decode = #"{"benchmark_id":"d","benchmark_type":"decode_throughput","parameter_prefill_tokens":256,"parameter_decode_tokens":100}"#
private let unknown = #"{"benchmark_id":"x","benchmark_type":"some_future_metric"}"#
// A known type missing its required `parameter_decode_tokens`.
private let decodeBad = #"{"benchmark_id":"d","benchmark_type":"decode_throughput","parameter_prefill_tokens":256}"#
private let bad = #"{"benchmark_id":"bad","benchmark_type":"decode_throughput","parameter_prefill_tokens":256}"#
// `BenchmarkId` refuses the whitespace, so the detail does not convert.
private let spacedId = #"{"benchmark_id":"prefill 512","benchmark_type":"prefill_throughput","parameter_prefill_tokens":512}"#
// A row of a known type that leaves out a parameter its detail carries.
private let partialRow = #"{"benchmark_id":"partial","benchmark_type":"prefill_throughput"}"#
private let partialBody = #"{"benchmark_id":"partial","benchmark_type":"prefill_throughput","parameter_prefill_tokens":128}"#

private func jsonArray(_ entries: [String]) -> Data {
    Data("[\(entries.joined(separator: ","))]".utf8)
}

private func tag(_ etag: String) throws -> EntityTag { try EntityTag(etag) }

private func summaries(_ entries: [String]) throws -> [PipetteMgmtClient.BenchmarkSummary] {
    try Coding.decoder.decode([PipetteMgmtClient.BenchmarkSummary].self, from: jsonArray(entries))
}

/// Decoded through the wire type the client returns, so a fixture cannot describe a
/// benchmark the real decoder would reject.
private func body(_ json: String, etag: String? = nil) throws -> DetailResponse {
    .modified(
        value: try Coding.decoder.decode(RemoteBenchmark.self, from: Data(json.utf8)),
        etag: try etag.map(tag))
}

private func listing(_ entries: [String], etag: String? = nil) -> BenchmarkListFetcher {
    { _, _ in .modified(value: try summaries(entries), etag: try etag.map(tag)) }
}

private let listUnchanged: BenchmarkListFetcher = { _, _ in .notModified(etag: nil) }

private func details(
    _ reply: @escaping (String) throws -> DetailResponse
) -> BenchmarkDetailFetcher {
    { _, id, _ in try reply(id) }
}

private func details(_ bodies: [String: String]) -> BenchmarkDetailFetcher {
    details { id in try body(try #require(bodies[id], "no fixture for \(id)")) }
}

/// What a fake server was asked, standing in for the `httpmock` hit counts the crate's
/// tests assert on.
private final class RequestLog: @unchecked Sendable {
    private let lock = NSLock()
    private var entries: [(path: String, ifNoneMatch: String?)] = []

    func record(_ path: String, _ ifNoneMatch: IfNoneMatch?) {
        lock.withLock { entries.append((path, ifNoneMatch?.headerValue)) }
    }

    func hits(_ path: String) -> Int { lock.withLock { entries.filter { $0.path == path }.count } }
    var conditionalHits: Int { lock.withLock { entries.filter { $0.ifNoneMatch != nil }.count } }
    var sentEtags: [String] { lock.withLock { entries.compactMap(\.ifNoneMatch) } }
}

/// Compares two lists by value, whatever their key order or number spelling.
private func jsonRows(_ data: Data) throws -> NSArray {
    try #require(JSONSerialization.jsonObject(with: data) as? NSArray)
}

/// The catalog pull (the crate's `pull_remote_benchmarks`) end to end with injected
/// fetchers.
@Suite
struct SyncTests {
    private func pull(
        list: @escaping BenchmarkListFetcher, detail: @escaping BenchmarkDetailFetcher,
        store: BenchmarkStore, server: ServerUrl? = nil
    ) async throws -> Int {
        try await pullRemoteBenchmarks(
            store: store, server: try server ?? testServer(), fetchList: list, fetchDetail: detail)
    }

    private func indexIds(_ store: BenchmarkStore) throws -> [String] {
        try store.listRemoteIndex().map(\.benchmarkId)
    }

    // MARK: - List + details, all modified

    @Test func aFullPullPersistsIndexDetailsAndSyncState() async throws {
        let store = makeTemporaryBenchmarkStore()
        defer { removeStore(store) }
        let count = try await pull(
            list: listing([prefill, decode], etag: "\"list-v1\""),
            detail: details { try body($0 == "p" ? prefill : decode, etag: "\"\($0)-v1\"") },
            store: store)

        #expect(count == 2)
        #expect(storedDefinitions(store: store).map(\.benchmarkId.value).sorted() == ["d", "p"])
        #expect(storedBenchmark(id: "p", store: store) != nil)
        #expect(storedBenchmark(id: "d", store: store) != nil)

        let state = try #require(try store.getSyncState())
        #expect(state.benchmarkCount == 2)
        #expect(state.serverUrl == (try testServer()))
        #expect(!state.pulledAt.isEmpty)
        #expect(state.benchmarksEtag == (try tag("\"list-v1\"")))
        #expect(store.remoteDetailEtag(id: "p") == (try tag("\"p-v1\"")))
        #expect(store.remoteDetailEtag(id: "d") == (try tag("\"d-v1\"")))
    }

    // MARK: - Tolerance, decided on the row's type and id

    /// The row stays in the stored list so the list ETag validates the server's whole list. A
    /// known-type row is fetched even when it lacks a parameter, since the list may leave out
    /// what only the detail carries.
    @Test func aRowOfAnUnknownTypeIsStoredButNeverFetched() async throws {
        let store = makeTemporaryBenchmarkStore()
        defer { removeStore(store) }
        let log = RequestLog()
        let count = try await pull(
            list: listing([prefill, unknown, bad, partialRow, decode]),
            detail: { _, id, ifNoneMatch in
                log.record("/benchmarks/\(id)", ifNoneMatch)
                let bodies = ["p": prefill, "d": decode, "bad": bad, "partial": partialBody]
                return try body(try #require(bodies[id], "\(id) was fetched"))
            },
            store: store)

        #expect(count == 4, "every row of a known type")
        #expect(try store.getSyncState()?.benchmarkCount == 4)
        #expect(try indexIds(store) == ["p", "x", "bad", "partial", "d"])
        #expect(log.hits("/benchmarks/x") == 0, "a row of an unknown type was fetched")
        for fetched in ["p", "bad", "partial", "d"] {
            #expect(log.hits("/benchmarks/\(fetched)") == 1, "\(fetched)")
        }
        #expect(storedBenchmark(id: "partial", store: store) != nil)
        for skipped in ["x", "bad"] {
            #expect(storedBenchmark(id: skipped, store: store) == nil)
            #expect(!FileManager.default.fileExists(atPath: store.remoteDetailPath(skipped).path),
                    "\(skipped) was cached")
        }
        // The listing filters the rows it cannot convert, the partial one among them.
        #expect(storedDefinitions(store: store).map(\.benchmarkId.value) == ["p", "d"])
    }

    /// It cannot be stored under a name a plan could name back, and listing it would offer
    /// it and then refuse it at the spec boundary.
    @Test func aRowWhoseIdIsNotABenchmarkIdIsNotFetched() async throws {
        let store = makeTemporaryBenchmarkStore()
        defer { removeStore(store) }
        let log = RequestLog()
        let count = try await pull(
            list: listing([spacedId, prefill]),
            detail: { _, id, ifNoneMatch in
                log.record("/benchmarks/\(id)", ifNoneMatch)
                return try body(prefill)
            },
            store: store)

        #expect(count == 1)
        #expect(try indexIds(store) == ["prefill 512", "p"])
        #expect(log.hits("/benchmarks/prefill 512") == 0)
        #expect(storedDefinitions(store: store).map(\.benchmarkId.value) == ["p"])
        #expect(storedBenchmark(id: "prefill 512", store: store) == nil)
    }

    /// No ETag, so a corrected detail is fetched in full next time. The row still counts:
    /// the count is the rows whose details the pull fetches.
    @Test func aDetailSchemaMismatchIsNotStoredAndLeavesNoEtag() async throws {
        let store = makeTemporaryBenchmarkStore()
        defer { removeStore(store) }
        let count = try await pull(
            list: listing([decode, prefill], etag: "\"l\""),
            detail: details { try body($0 == "d" ? decodeBad : prefill, etag: "\"\($0)\"") },
            store: store)

        #expect(count == 2)
        #expect(storedBenchmark(id: "d", store: store) == nil)
        #expect(store.remoteDetailEtag(id: "d") == nil)
    }

    /// Row for row, with rows this build cannot convert and keys it does not model.
    @Test func theStoredListIsWhatTheServerSent() async throws {
        let store = makeTemporaryBenchmarkStore()
        defer { removeStore(store) }
        let rows = [
            #"{"benchmark_id":"p","benchmark_type":"prefill_throughput","parameter_prefill_tokens":512,"server_only_key":"kept"}"#,
            unknown, bad, spacedId,
        ]

        #expect(try await pull(
            list: listing(rows, etag: "\"l\""), detail: details(["p": prefill, "bad": bad]),
            store: store) == 2)

        let onDisk = try jsonRows(Data(contentsOf: store.statePath("index.json")))
        #expect(onDisk == (try jsonRows(jsonArray(rows))))
    }

    // MARK: - What a detail is cached as

    /// Kept for a build that models them: a private entry's `org_id`, a parameter from a
    /// newer catalog.
    @Test func aPulledDetailKeepsKeysThisBuildDoesNotModel() async throws {
        let store = makeTemporaryBenchmarkStore()
        defer { removeStore(store) }
        let sent = #"{"benchmark_id":"p","benchmark_type":"prefill_throughput","parameter_prefill_tokens":512,"parameter_batch_size":4,"parameter_from_the_future":0.25,"org_id":"org_test"}"#

        #expect(try await pull(list: listing([prefill]), detail: details(["p": sent]), store: store)
            == 1)

        let onDisk = try jsonDocument(Data(contentsOf: store.remoteDetailPath("p")))
        #expect(onDisk == (try jsonDocument(Data(sent.utf8))))
    }

    /// The older build's `remote/<id>.json`, `remote/index.json` and `remote/sync.json` are
    /// never read, so no `304` can keep what it lost. Its files are left to it.
    @Test func aWorkspaceAnOlderBuildSyncedIsPulledCold() async throws {
        let store = makeTemporaryBenchmarkStore()
        defer { removeStore(store) }
        let legacy = try store.writeLegacyImg80()
        try store.writeOlderBuildBookkeeping(
            rows: [img80Wire], server: try testServer().value, listEtag: #""list-v1""#,
            benchmarkEtags: [img80Id: #""img80-v1""#])

        let log = RequestLog()
        #expect(try await pull(
            list: { _, ifNoneMatch in
                log.record("/benchmarks", ifNoneMatch)
                if ifNoneMatch != nil { return .notModified(etag: try tag("\"list-v1\"")) }
                return .modified(value: try summaries([img80Wire]), etag: try tag("\"list-v1\""))
            },
            detail: { _, id, ifNoneMatch in
                log.record("/benchmarks/\(id)", ifNoneMatch)
                if ifNoneMatch != nil { return .notModified(etag: try tag("\"img80-v1\"")) }
                return try body(img80Wire, etag: "\"img80-v1\"")
            },
            store: store) == 1)

        #expect(log.hits("/benchmarks") == 1)
        #expect(log.hits("/benchmarks/\(img80Id)") == 1)
        #expect(log.conditionalHits == 0, "an older build's ETag was sent")
        let olderBuildFiles = [legacy, store.root.appendingPathComponent("remote/index.json"),
                               store.root.appendingPathComponent("remote/sync.json")]
        for file in olderBuildFiles {
            #expect(FileManager.default.fileExists(atPath: file.path),
                    "\(file.lastPathComponent) went")
        }
        let onDisk = try jsonDocument(Data(contentsOf: store.remoteDetailPath(img80Id)))
        #expect(onDisk == (try jsonDocument(Data(img80Wire.utf8))))
        guard case let .vlThroughput(vl) = storedBenchmark(id: img80Id, store: store) else {
            Issue.record("the 80-image definition does not read back")
            return
        }
        #expect(vl.parameterNumImages.value == 80)
        #expect(store.remoteDetailEtag(id: BenchmarkId(validated: img80Id)) == (try tag("\"img80-v1\"")))
    }

    /// An older build writing after this pull (its filtered list and its own ETags, under the
    /// same list ETag) changes neither what the next `304` keeps nor the ETags it sends.
    @Test func anOlderBuildsFilesNeverReachState() async throws {
        let store = makeTemporaryBenchmarkStore()
        defer { removeStore(store) }
        #expect(try await pull(
            list: listing([prefill, unknown], etag: "\"list-v1\""),
            detail: details { _ in try body(prefill, etag: "\"p-v1\"") },
            store: store) == 1)
        try store.writeOlderBuildBookkeeping(
            rows: [prefill], server: try testServer().value, listEtag: #""list-v1""#,
            benchmarkEtags: ["p": #""p-older""#])

        let log = RequestLog()
        #expect(try await pull(
            list: { _, ifNoneMatch in
                log.record("/benchmarks", ifNoneMatch)
                return .notModified(etag: try tag("\"list-v1\""))
            },
            detail: { _, id, ifNoneMatch in
                log.record("/benchmarks/\(id)", ifNoneMatch)
                return .notModified(etag: nil)
            },
            store: store) == 1)

        #expect(log.sentEtags == ["\"list-v1\"", "\"p-v1\""])
        #expect(try indexIds(store) == ["p", "x"])
        let onDisk = try jsonRows(Data(contentsOf: store.statePath("index.json")))
        #expect(onDisk == (try jsonRows(jsonArray([prefill, unknown]))))
        #expect(store.remoteDetailEtag(id: "p") == (try tag("\"p-v1\"")))
        for name in ["index.json", "sync.json"] {
            #expect(FileManager.default.fileExists(
                atPath: store.root.appendingPathComponent("remote/\(name)").path),
                    "the older build's \(name) went")
        }
    }

    /// A `304` hands back every stored row: one this build cannot convert is not fetched,
    /// counted or listed, and the detail a build that converts it cached survives the prune.
    @Test func a304CannotShrinkTheCatalog() async throws {
        let store = makeTemporaryBenchmarkStore()
        defer { removeStore(store) }
        // What a build that converts `x` left, in the order a pull writes it.
        try store.cacheRemote(json: prefill, etag: "\"p-v1\"")
        try Data(unknown.utf8).write(to: store.remoteDetailPath("x"))
        try Data("\"x-v1\"".utf8).write(to: store.remoteEtagPath("x"))
        try store.putRemoteIndex(try summaries([prefill, unknown]))
        try store.putSyncState(RemoteSyncState(
            pulledAt: "2026-10-18T00:00:00Z", benchmarkCount: 2, serverUrl: try testServer(),
            benchmarksEtag: try tag("\"list-v1\"")))

        let log = RequestLog()
        let count = try await pull(
            list: { _, ifNoneMatch in
                log.record("/benchmarks", ifNoneMatch)
                return .notModified(etag: nil)
            },
            detail: { _, id, ifNoneMatch in
                log.record("/benchmarks/\(id)", ifNoneMatch)
                return .notModified(etag: nil)
            },
            store: store)

        #expect(count == 1)
        #expect(try store.getSyncState()?.benchmarkCount == 1)
        #expect(log.sentEtags == ["\"list-v1\"", "\"p-v1\""])
        #expect(log.hits("/benchmarks/x") == 0)
        #expect(try indexIds(store) == ["p", "x"])
        #expect(FileManager.default.fileExists(atPath: store.remoteDetailPath("x").path),
                "the prune deleted the detail of a row this build cannot convert")
        #expect(storedDefinitions(store: store).map(\.benchmarkId.value) == ["p"])
        #expect(BenchmarkCatalog.all(store: store).map(\.benchmarkId) == ["p"])
        #expect(BenchmarkCatalog.item(forId: "x", store: store) == nil)
        #expect(store.list(.remote).map(\.benchmarkId.value) == ["p"])
    }

    // MARK: - List 304

    @Test func aListNotModifiedReusesTheCachedIndexAndStillRunsDetails() async throws {
        let store = makeTemporaryBenchmarkStore()
        defer { removeStore(store) }
        _ = try await pull(
            list: listing([prefill, decode], etag: "\"l\""),
            detail: details { try body($0 == "p" ? prefill : decode, etag: "\"\($0)\"") },
            store: store)

        let count = try await pull(
            list: listUnchanged, detail: { _, _, _ in .notModified(etag: nil) }, store: store)

        #expect(count == 2)
        #expect(storedDefinitions(store: store).count == 2)
        // The per-id ETag survives a 304.
        #expect(store.remoteDetailEtag(id: "p") == (try tag("\"p\"")))
    }

    /// A 304 with nothing cached to reuse is not "pulled 0": the crate refuses it.
    @Test func a304ListWithoutACachedIndexIsAnError() async throws {
        let store = makeTemporaryBenchmarkStore()
        defer { removeStore(store) }
        await #expect(throws: BenchmarkSyncError.notModifiedWithoutCachedIndex) {
            try await pull(list: listUnchanged, detail: { _, _, _ in .notModified(etag: nil) },
                           store: store)
        }
        #expect(try store.getSyncState() == nil)
    }

    // MARK: - Failed detail fetches

    /// One failed fetch does not abort the pull, and no ETag means the next pull re-requests it.
    @Test func aFailedDetailFetchKeepsItsIdWithNoEtag() async throws {
        let store = makeTemporaryBenchmarkStore()
        defer { removeStore(store) }
        let count = try await pull(
            list: listing([prefill, decode], etag: "\"l\""),
            detail: details { id in
                if id == "p" { throw URLError(.timedOut) }
                return try body(decode, etag: "\"d\"")
            },
            store: store)

        #expect(count == 2)
        #expect(try indexIds(store) == ["p", "d"])
        #expect(storedBenchmark(id: "d", store: store) != nil)
        #expect(storedBenchmark(id: "p", store: store) == nil)
        #expect(store.remoteDetailEtag(id: "p") == nil)
    }

    /// Keeping the failed id in the index is what stops the drop becoming permanent behind
    /// the list ETag.
    @Test func transientDetailFailureRecoversOnNextSync() async throws {
        let store = makeTemporaryBenchmarkStore()
        defer { removeStore(store) }
        #expect(try await pull(
            list: listing([prefill, decode], etag: "\"v1\""),
            detail: details { id in
                if id == "d" { throw URLError(.badServerResponse) }
                return try body(prefill)
            },
            store: store) == 2)
        #expect(try indexIds(store).contains("d"), "failed id kept in index")
        #expect(store.remoteDetailEtag(id: "d") == nil, "no ETag stored for the failed fetch")

        let log = RequestLog()
        #expect(try await pull(
            list: { _, _ in .notModified(etag: try tag("\"v1\"")) },
            detail: { _, id, ifNoneMatch in
                log.record("/benchmarks/\(id)", ifNoneMatch)
                return id == "d" ? try body(decode) : .notModified(etag: nil)
            },
            store: store) == 2)
        #expect(log.hits("/benchmarks/d") == 1)
        #expect(store.hasRemoteDetail(id: "d"), "the failed benchmark recovered through the 304 path")
    }

    /// A pull aimed at a different server does not revalidate against the previous one's
    /// ETags, which a host answering 304 to would turn into a catalog it never served.
    @Test func aNewServerIsPulledCold() async throws {
        let store = makeTemporaryBenchmarkStore()
        defer { removeStore(store) }
        let first = try ServerUrl("https://first.example.test")
        let second = try ServerUrl("https://second.example.test")
        #expect(try await pull(
            list: listing([prefill], etag: "\"v1\""),
            detail: details { _ in try body(prefill, etag: "\"d1\"") },
            store: store, server: first) == 1)

        // The same ids and ETags on the second host, so a leaked `If-None-Match` would land
        // on a 304 rather than fail loudly.
        let log = RequestLog()
        #expect(try await pull(
            list: { _, ifNoneMatch in
                log.record("/benchmarks", ifNoneMatch)
                if ifNoneMatch != nil { return .notModified(etag: try tag("\"v1\"")) }
                return .modified(value: try summaries([prefill]), etag: try tag("\"v1\""))
            },
            detail: { _, id, ifNoneMatch in
                log.record("/benchmarks/\(id)", ifNoneMatch)
                if ifNoneMatch != nil { return .notModified(etag: try tag("\"d1\"")) }
                return try body(prefill, etag: "\"d1\"")
            },
            store: store, server: second) == 1)

        #expect(log.conditionalHits == 0)
        #expect(log.hits("/benchmarks") == 1)
        #expect(log.hits("/benchmarks/p") == 1)
        #expect(try store.getSyncState()?.serverUrl == second,
                "the cache now belongs to the server it was pulled from")
    }

    /// Every detail failing is a dead server, so the pull fails before it rewrites the
    /// index or the sync state.
    @Test func totalDetailOutageLeavesTheCachedCatalogUntouched() async throws {
        let store = makeTemporaryBenchmarkStore()
        defer { removeStore(store) }
        #expect(try await pull(
            list: listing([prefill], etag: "\"v0\""),
            detail: details { _ in try body(prefill) },
            store: store) == 1)

        await #expect(throws: BenchmarkSyncError.allDetailFetchesFailed(total: 1)) {
            try await pull(
                list: listing([prefill], etag: "\"v1\""),
                detail: details { _ in throw URLError(.cannotConnectToHost) },
                store: store)
        }
        #expect(try indexIds(store) == ["p"], "index untouched")
        #expect(store.hasRemoteDetail(id: "p"), "cached detail untouched")
        #expect(try store.getSyncState()?.benchmarksEtag == (try tag("\"v0\"")),
                "list ETag not advanced past the failed pull")
    }

    // MARK: - Pruning

    @Test func orphanDetailsArePrunedWhenABenchmarkLeavesTheCatalog() async throws {
        let store = makeTemporaryBenchmarkStore()
        defer { removeStore(store) }
        _ = try await pull(
            list: listing([prefill, decode], etag: "\"l1\""),
            detail: details { try body($0 == "p" ? prefill : decode, etag: "\"\($0)\"") },
            store: store)
        #expect(storedBenchmark(id: "p", store: store) != nil)

        let count = try await pull(
            list: listing([decode], etag: "\"l2\""),
            detail: details { _ in try body(decode, etag: "\"d2\"") },
            store: store)

        #expect(count == 1)
        #expect(storedBenchmark(id: "p", store: store) == nil)
        #expect(storedBenchmark(id: "d", store: store) != nil)
        #expect(!FileManager.default.fileExists(atPath: store.remoteEtagPath("p").path))
    }

    /// Not against the ETag the pull recorded for the body the claim replaced.
    @Test func aDetailAClaimRewroteIsRevalidatedAgainstItsOwnEtag() async throws {
        let store = makeTemporaryBenchmarkStore()
        defer { removeStore(store) }
        let log = RequestLog()
        let served: BenchmarkDetailFetcher = { _, id, ifNoneMatch in
            log.record("/benchmarks/\(id)", ifNoneMatch)
            return try body(prefill, etag: "\"p-v1\"")
        }
        #expect(try await pull(list: listing([prefill]), detail: served, store: store) == 1)
        #expect(log.sentEtags.isEmpty)

        // What a claim leaves after the server changed `p`.
        try store.cacheRemote(
            json: #"{"benchmark_id":"p","benchmark_type":"prefill_throughput","parameter_prefill_tokens":1024}"#,
            etag: "\"p-v2\"")

        #expect(try await pull(list: listing([prefill]), detail: served, store: store) == 1)

        #expect(log.sentEtags == ["\"p-v2\""])
        let onDisk = try jsonDocument(Data(contentsOf: store.remoteDetailPath("p")))
        #expect(onDisk == (try jsonDocument(Data(prefill.utf8))), "the claim's body was kept")
        #expect(store.remoteDetailEtag(id: "p") == (try tag("\"p-v1\"")))
    }

    // MARK: - Helpers the pull is built from

    /// Only the ETag stored beside a detail, and only after a pull from this server. An older
    /// build's `remote/<id>.json` does not count: a 304 would keep it, whatever it lost.
    @Test func cachedBenchmarkEtagRequiresValidDetailCache() throws {
        let store = makeTemporaryBenchmarkStore()
        defer { removeStore(store) }

        #expect(cachedBenchmarkEtag(store: store, benchmarkId: "p", revalidates: true) == nil)
        try FileManager.default.createDirectory(
            at: store.root.appendingPathComponent("remote"), withIntermediateDirectories: true)
        try Data(prefill.utf8).write(to: store.root.appendingPathComponent("remote/p.json"))
        #expect(cachedBenchmarkEtag(store: store, benchmarkId: "p", revalidates: true) == nil)
        try store.cacheRemote(json: prefill)
        #expect(cachedBenchmarkEtag(store: store, benchmarkId: "p", revalidates: true) == nil,
                "no ETag came with the body")
        try store.cacheRemote(json: prefill, etag: "\"detail\"")
        #expect(cachedBenchmarkEtag(store: store, benchmarkId: "p", revalidates: true)
                == (try tag("\"detail\"")))
        #expect(cachedBenchmarkEtag(store: store, benchmarkId: "p", revalidates: false) == nil,
                "another server's ETag")
    }

    @Test func benchmarkDetailConcurrencyUsesSystemLimitWithCap() {
        #expect((1...benchmarkDetailConcurrencyCap)
            .contains(benchmarkDetailConcurrency()))
    }

    @Test func storedReadbackReflectsStoredData() throws {
        let store = makeTemporaryBenchmarkStore()
        defer { removeStore(store) }
        try store.putRemoteIndex(try summaries([prefill]))
        try store.cacheRemote(json: prefill)

        #expect(storedDefinitions(store: store).map(\.benchmarkId.value) == ["p"])
        #expect(storedBenchmark(id: "p", store: store) != nil)
        #expect(storedBenchmark(id: "missing", store: store) == nil)
    }

    /// A malformed catalog never reaches the pull: it fails decoding into the wire type.
    @Test func aMalformedCatalogIsRefusedByTheWireType() {
        #expect(throws: (any Error).self) {
            try Coding.decoder.decode(
                [PipetteMgmtClient.BenchmarkSummary].self, from: Data("not json".utf8))
        }
    }

    // MARK: - RemoteSyncState

    /// The crate's `a_sync_state_reads_the_server_it_recorded`: older builds wrote `""` for
    /// unset, so it reads as "no server recorded", as a missing field does.
    @Test(arguments: [
        ("https://mgmt.example.com", "https://mgmt.example.com"),
        ("https://mgmt.example.com/", "https://mgmt.example.com"),
        ("", nil),
        (nil, nil),
    ] as [(String?, String?)])
    func aSyncStateReadsTheServerItRecorded(stored: String?, expected: String?) throws {
        var json: [String: Any] = ["pulled_at": "2026-01-01T00:00:00Z", "benchmark_count": 3]
        if let stored { json["server_url"] = stored }
        let state = try Coding.decoder.decode(
            RemoteSyncState.self, from: JSONSerialization.data(withJSONObject: json))
        #expect(state.pulledAt == "2026-01-01T00:00:00Z")
        #expect(state.benchmarkCount == 3)
        #expect(state.serverUrl?.value == expected)
    }

    /// This program wrote the file, so a server it cannot parse is corruption, and a cold
    /// pull would paper over it.
    @Test func anUnusableRecordedServerIsAnError() {
        let json = #"{"pulled_at":"2026-01-01T00:00:00Z","benchmark_count":3,"server_url":"mgmt.example.com"}"#
        do {
            _ = try Coding.decoder.decode(RemoteSyncState.self, from: Data(json.utf8))
            Issue.record("expected the read to fail")
        } catch {
            #expect("\(error)".contains("mgmt.example.com"), "\(error)")
        }
    }

    /// An iOS build once wrote no `pulled_at`; its file still reads, with no server, so the
    /// next pull goes cold.
    @Test func aSyncStateFromAnOlderBuildStillReads() throws {
        let json = #"{"benchmark_count":2,"benchmarks_etag":"\"abc\"","benchmark_etags":{"p":"\"e1\"","d":"\"e2\""}}"#
        let state = try Coding.decoder.decode(RemoteSyncState.self, from: Data(json.utf8))
        #expect(state.benchmarkCount == 2)
        #expect(state.pulledAt.isEmpty)
        #expect(state.serverUrl == nil)
        #expect(state.benchmarksEtag == (try tag("\"abc\"")))

        let roundTripped = try Coding.decoder.decode(
            RemoteSyncState.self, from: Coding.encoder.encode(state))
        #expect(roundTripped == state)
    }

    /// The crate's `skip_serializing_if`: unset fields are absent, not `null`.
    @Test func anUnsetFieldIsAbsentFromTheWireShape() throws {
        let bare = RemoteSyncState(
            pulledAt: "2026-01-01T00:00:00Z", benchmarkCount: 0, serverUrl: nil,
            benchmarksEtag: nil)
        let row = try #require(
            JSONSerialization.jsonObject(with: try Coding.encoder.encode(bare)) as? [String: Any])

        #expect(Set(row.keys) == ["pulled_at", "benchmark_count"])
    }
}
