import Foundation
import Testing

@testable import Pipette

/// A claim's revalidation of its cached benchmark against `GET /benchmarks/{id}` and its
/// stored ETag: pipette-cli's `ensure_claim_benchmark_cached` tests in `commands/worker.rs`.
/// Each test mints its own route, so the suite runs in parallel.
@Suite
struct ClaimedBenchmarkCheckTests {
    private nonisolated static let id = "prefill_throughput_x"

    private nonisolated static func prefill(_ tokens: NonNegativeCount) -> BenchmarkDefinition {
        .prefillThroughput(PrefillThroughputBenchmark(
            benchmarkId: BenchmarkId(validated: id), parameterPrefillTokens: tokens))
    }

    private static func body(_ definition: BenchmarkDefinition) throws -> String {
        String(decoding: try Coding.encoder.encode(definition), as: UTF8.self)
    }

    private nonisolated static func isSigned(_ request: URLRequest) -> Bool {
        ["X-Client-Id", "X-Timestamp", "X-Nonce", "X-Signature"]
            .allSatisfy { request.value(forHTTPHeaderField: $0) != nil }
    }

    private func serve(
        _ reply: @escaping @Sendable (URLRequest) -> StubResponse
    ) throws -> (http: HttpClient, server: ServerUrl, seen: StubRecorder) {
        try RoutedURLProtocol.serve { request, _ in
            Self.isSigned(request) ? reply(request) : .reply(401, #"{"error":"unsigned"}"#)
        }
    }

    private func claim(
        store: BenchmarkStore, server: ServerUrl, http: HttpClient
    ) async throws -> BenchmarkDefinition {
        try await PlannerWorker.ensureClaimBenchmarkCached(
            store: store, server: server, http: http, auth: try testIdentity(),
            benchmarkId: Self.id)
    }

    private func sentEtags(_ seen: StubRecorder) -> [String?] {
        seen.requests.map { $0.value(forHTTPHeaderField: "If-None-Match") }
    }

    /// Nothing is rewritten.
    @Test func a304RunsTheCache() async throws {
        let store = makeTemporaryBenchmarkStore()
        defer { removeStore(store) }
        let cached = Self.prefill(256)
        try store.cacheRemote(cached, etag: #""v1""#)
        let (http, server, seen) = try serve { _ in .reply(304) }

        #expect(try await claim(store: store, server: server, http: http) == cached)
        #expect(sentEtags(seen) == [#""v1""#])
        #expect(store.remoteDetailEtag(id: BenchmarkId(validated: Self.id)) == (try EntityTag(#""v1""#)))
    }

    /// The next claim revalidates against the new ETag.
    @Test func aChangedBodyReplacesTheCacheAndItsEtag() async throws {
        let store = makeTemporaryBenchmarkStore()
        defer { removeStore(store) }
        try store.cacheRemote(Self.prefill(256), etag: #""v1""#)
        let served = try Self.body(Self.prefill(512))
        let (http, server, seen) = try serve { request in
            request.value(forHTTPHeaderField: "If-None-Match") == #""v2""#
                ? .reply(304) : .reply(served, etag: #""v2""#)
        }

        #expect(try await claim(store: store, server: server, http: http) == Self.prefill(512))
        #expect(try store.get(.remote(BenchmarkId(validated: Self.id))) == Self.prefill(512))
        #expect(store.remoteDetailEtag(id: BenchmarkId(validated: Self.id)) == (try EntityTag(#""v2""#)))

        #expect(try await claim(store: store, server: server, http: http) == Self.prefill(512))
        #expect(sentEtags(seen) == [#""v1""#, #""v2""#])
    }

    /// What it returns replaces the cache.
    @Test func aCacheWithoutAnEtagIsReadUnconditionally() async throws {
        let store = makeTemporaryBenchmarkStore()
        defer { removeStore(store) }
        try store.cacheRemote(Self.prefill(256))
        let served = try Self.body(Self.prefill(512))
        let (http, server, seen) = try serve { _ in .reply(served, etag: #""v2""#) }

        #expect(try await claim(store: store, server: server, http: http) == Self.prefill(512))
        #expect(sentEtags(seen) == [nil])
        #expect(store.remoteDetailEtag(id: BenchmarkId(validated: Self.id)) == (try EntityTag(#""v2""#)))
    }

    /// The copy runs unchanged and keeps its ETag.
    @Test(arguments: [
        StubResponse.reply(403, #"{"error":"client is pending approval"}"#),
        .reply(503, "unavailable"),
        .reply("not json", etag: #""v2""#),
    ])
    func anyOtherFailureWithACacheRunsIt(answer: StubResponse) async throws {
        let store = makeTemporaryBenchmarkStore()
        defer { removeStore(store) }
        try store.cacheRemote(Self.prefill(256), etag: #""v1""#)
        let (http, server, _) = try serve { _ in answer }

        #expect(try await claim(store: store, server: server, http: http) == Self.prefill(256))
        #expect(try store.get(.remote(BenchmarkId(validated: Self.id))) == Self.prefill(256))
        #expect(store.remoteDetailEtag(id: BenchmarkId(validated: Self.id)) == (try EntityTag(#""v1""#)))
    }

    /// Fails it as an unknown benchmark, for any `404` body (a proxy's page among them): mgmt
    /// refuses a result whose definition is not the one it serves. The cache is left as it was.
    @Test(arguments: [
        StubResponse.reply(404, #"{"error":"benchmark not found"}"#),
        .reply(404, "<html>Not Found</html>"),
        .reply(#"{"benchmark_id":"prefill_throughput_x","benchmark_type":"mystery_metric"}"#,
               etag: #""v2""#),
    ])
    func a404OrAnUnrunnableChangeWithACacheFailsTheClaim(answer: StubResponse) async throws {
        let store = makeTemporaryBenchmarkStore()
        defer { removeStore(store) }
        try store.cacheRemote(Self.prefill(256), etag: #""v1""#)
        let (http, server, _) = try serve { _ in answer }
        do {
            _ = try await claim(store: store, server: server, http: http)
            Issue.record("expected the claim to fail on \(answer.status)")
        } catch {
            #expect(!classifyRunError(error), "\(error)")
        }
        #expect(try store.get(.remote(BenchmarkId(validated: Self.id))) == Self.prefill(256))
    }

    /// A `404` or a body this build cannot read is an unknown benchmark, which is terminal; a
    /// `5xx` stays retriable.
    @Test func aFailureWithNothingCachedFailsTheClaim() async throws {
        for (answer, unknown) in [
            (StubResponse.reply(404, #"{"error":"benchmark not found"}"#), true),
            (.reply(#"{"benchmark_id":"prefill_throughput_x","benchmark_type":"mystery_metric"}"#,
                    etag: #""v2""#), true),
            (.reply(503, "unavailable"), false),
        ] {
            let store = makeTemporaryBenchmarkStore()
            defer { removeStore(store) }
            let (http, server, _) = try serve { _ in answer }
            do {
                _ = try await claim(store: store, server: server, http: http)
                Issue.record("expected the claim to fail on \(answer.status)")
            } catch {
                #expect(classifyRunError(error) != unknown, "\(answer.status): \(error)")
            }
        }
    }

    /// Retriable, since a `304` says nothing about the benchmark.
    @Test func a304WithNothingCachedFailsRetriably() async throws {
        let store = makeTemporaryBenchmarkStore()
        defer { removeStore(store) }
        let (http, server, _) = try serve { _ in .reply(304) }

        await #expect(throws: URLError.self) {
            try await claim(store: store, server: server, http: http)
        }
    }
}
