import Foundation
import Testing

@testable import Pipette

/// `JobCell.resolvable`: fetching a benchmark the store and the structured id both miss.
@Suite
struct JobCellResolvableTests {
    /// Only the server describes it.
    private nonisolated static let visionId = "vl_throughput_256x256_text1024_decode256_img5"

    private nonisolated static let vision = BenchmarkDefinition.vlThroughput(VlThroughputBenchmark(
        benchmarkId: BenchmarkId(validated: visionId),
        parameterImageWidth: 256,
        parameterImageHeight: 256,
        parameterTextTokens: 1024,
        parameterDecodeTokens: 256,
        parameterNumImages: 5))

    /// A local definition of a type the id parse cannot describe.
    private nonisolated static let localSmoke = BenchmarkDefinition.vlThroughput(VlThroughputBenchmark(
        benchmarkId: BenchmarkId(validated: "vl_throughput_smoke"),
        parameterImageWidth: 224,
        parameterImageHeight: 224,
        parameterTextTokens: 8,
        parameterDecodeTokens: 8))

    private static func body(_ definition: BenchmarkDefinition) throws -> String {
        String(decoding: try Coding.encoder.encode(definition), as: UTF8.self)
    }

    /// Cached on the remote half. Bare and `remote/` are the same reference.
    @Test(arguments: [visionId, "remote/\(visionId)"])
    func aStoreMissTheServerAnswersBuildsACellAndIsCached(_ reference: String) async throws {
        let store = makeTemporaryBenchmarkStore()
        defer { removeStore(store) }
        #expect(BenchmarkDefinition(parsingId: Self.visionId) == nil)
        #expect(try store.get(.remote(BenchmarkId(validated: Self.visionId))) == nil)
        let body = try Self.body(Self.vision)
        let (http, server, seen) = try RoutedURLProtocol.serve { _, _ in .reply(200, body) }

        var skipped: [String] = []
        let resolved = try await JobCell.resolvable(
            benchmarkIds: [reference], store: store, server: server, http: http, auth: nil
        ) { reference, _ in skipped.append(reference) }

        #expect(resolved.references == [reference])
        #expect(resolved.claimed == [Self.vision])
        #expect(skipped.isEmpty)
        #expect(seen.methodsAndPaths == ["GET /benchmarks/\(Self.visionId)"])
        #expect(try store.get(.remote(BenchmarkId(validated: Self.visionId))) == Self.vision)

        let cells = JobCell.pending(benchmarkIds: resolved.references, for: .appleFoundation,
                                    store: store) { skipped.append($0) }
        #expect(skipped.isEmpty)
        #expect(cells.count == 1)
        #expect(cells.first?.benchmarkId == Self.visionId)
        #expect(cells.first?.benchmarkType == .vlThroughput)
        #expect(cells.first?.isSubmittable == true)
    }

    /// So an organization's private benchmark resolves.
    @Test func aDeviceWithAnIdentitySignsTheFetch() async throws {
        let store = makeTemporaryBenchmarkStore()
        defer { removeStore(store) }
        let body = try Self.body(Self.vision)
        let (http, server, seen) = try RoutedURLProtocol.serve { _, _ in .reply(200, body) }

        let resolved = try await JobCell.resolvable(
            benchmarkIds: [Self.visionId], store: store, server: server, http: http,
            auth: try testIdentity())

        #expect(resolved.references == [Self.visionId])
        let request = try #require(seen.requests.first)
        #expect(request.value(forHTTPHeaderField: "X-Client-Id") == "client-1")
        #expect(request.value(forHTTPHeaderField: "X-Signature") != nil)
    }

    /// Skipped with the server's reason; nothing is cached and no cell is built.
    @Test func aBenchmarkTheServerDoesNotKnowIsSkipped() async throws {
        let store = makeTemporaryBenchmarkStore()
        defer { removeStore(store) }
        let (http, server, seen) = try RoutedURLProtocol.serve { _, _ in
            .reply(404, "benchmark not found")
        }

        var skipped: [(reference: String, reason: String)] = []
        let resolved = try await JobCell.resolvable(
            benchmarkIds: [Self.visionId], store: store, server: server, http: http, auth: nil
        ) { skipped.append(($0, $1)) }

        #expect(resolved.references.isEmpty)
        #expect(resolved.claimed.isEmpty)
        #expect(skipped.map(\.reference) == [Self.visionId])
        #expect(skipped.first?.reason.contains("unrecognized benchmark id") == true)
        #expect(skipped.first?.reason.contains("benchmark not found") == true)
        #expect(seen.methodsAndPaths == ["GET /benchmarks/\(Self.visionId)"])
        #expect(try store.get(.remote(BenchmarkId(validated: Self.visionId))) == nil)
        #expect(JobCell.pending(benchmarkIds: resolved.references, for: .appleFoundation,
                                store: store).isEmpty)
    }

    /// Skipped like a `404`, and not cached.
    @Test func aDefinitionThisBuildCannotReadIsSkipped() async throws {
        let store = makeTemporaryBenchmarkStore()
        defer { removeStore(store) }
        let (http, server, _) = try RoutedURLProtocol.serve { _, _ in
            .reply(200, #"{"benchmark_id":"odd","benchmark_type":"not_a_type"}"#)
        }

        var skipped: [String] = []
        let resolved = try await JobCell.resolvable(
            benchmarkIds: ["odd"], store: store, server: server, http: http, auth: nil
        ) { reference, _ in skipped.append(reference) }

        #expect(resolved.references.isEmpty)
        #expect(skipped == ["odd"])
        #expect(try store.get(.remote("odd")) == nil)
    }

    @Test func aFetchThatFailsInTransitIsARetriableError() async throws {
        let store = makeTemporaryBenchmarkStore()
        defer { removeStore(store) }
        let server = try ServerUrl("https://collector.example.com")

        var skipped: [String] = []
        do {
            _ = try await JobCell.resolvable(
                benchmarkIds: [Self.visionId], store: store, server: server,
                http: UnreachableURLProtocol.httpClient, auth: nil
            ) { reference, _ in skipped.append(reference) }
            Issue.record("expected the fetch to fail")
        } catch {
            #expect(!(error is UnknownRemoteBenchmarkError))
            #expect(classifyRunError(error))
        }
        #expect(skipped.isEmpty)
    }

    @Test func aServerErrorIsARetriableError() async throws {
        let store = makeTemporaryBenchmarkStore()
        defer { removeStore(store) }
        let (http, server, _) = try RoutedURLProtocol.serve { _, _ in .reply(503, "down") }

        await #expect(throws: PipetteMgmtClient.Error.self) {
            try await JobCell.resolvable(
                benchmarkIds: [Self.visionId], store: store, server: server, http: http,
                auth: nil)
        }
    }

    @Test func aLocalReferenceIsNeverFetched() async throws {
        let store = makeTemporaryBenchmarkStore()
        defer { removeStore(store) }
        let body = try Self.body(Self.vision)
        let (http, server, seen) = try RoutedURLProtocol.serve { _, _ in .reply(200, body) }

        var skipped: [(reference: String, reason: String)] = []
        let resolved = try await JobCell.resolvable(
            benchmarkIds: ["local/\(Self.visionId)"], store: store, server: server, http: http,
            auth: nil
        ) { skipped.append(($0, $1)) }

        #expect(resolved.references.isEmpty)
        #expect(skipped.map(\.reference) == ["local/\(Self.visionId)"])
        #expect(skipped.first?.reason == "unrecognized benchmark id")
        #expect(seen.requests.isEmpty)
        #expect(try store.get(.remote(BenchmarkId(validated: Self.visionId))) == nil)
    }

    /// A remote miss is skipped like a local one.
    @Test func noServerFetchesNothing() async throws {
        let store = makeTemporaryBenchmarkStore()
        defer { removeStore(store) }

        var skipped: [(reference: String, reason: String)] = []
        let resolved = try await JobCell.resolvable(
            benchmarkIds: [Self.visionId], store: store, server: nil,
            http: UnreachableURLProtocol.httpClient, auth: nil
        ) { skipped.append(($0, $1)) }

        #expect(resolved.references.isEmpty)
        #expect(skipped.map(\.reference) == [Self.visionId])
        #expect(skipped.first?.reason == "unrecognized benchmark id")
    }

    /// Without a server, ladder rungs and stored definitions on either half resolve; an
    /// unparseable reference is skipped.
    @Test func anUnclaimedRunResolvesOnTheDevice() async throws {
        let store = makeTemporaryBenchmarkStore()
        defer { removeStore(store) }
        try store.cacheRemote(Self.vision)
        try store.putLocal(Self.localSmoke)

        var skipped: [String] = []
        let references = ["prefill_throughput_512", "local/decode_throughput_512_100",
                          Self.visionId, "local/vl_throughput_smoke"]
        let resolved = try await JobCell.resolvable(
            benchmarkIds: references + ["elsewhere/foo"], store: store, server: nil,
            http: UnreachableURLProtocol.httpClient, auth: nil
        ) { reference, _ in skipped.append(reference) }

        #expect(resolved.references == references)
        #expect(resolved.claimed.isEmpty)
        #expect(skipped == ["elsewhere/foo"])
    }

    /// As the planner worker's claim does: a revalidated cached definition runs as cached,
    /// and an uncached ladder rung is fetched rather than run from its id. A local reference
    /// is never asked about.
    @Test func aClaimedRunAsksTheServerAboutEachRemoteReference() async throws {
        let store = makeTemporaryBenchmarkStore()
        defer { removeStore(store) }
        try store.cacheRemote(Self.vision, etag: #""v""#)
        try store.putLocal(Self.localSmoke)
        let rung = BenchmarkDefinition.prefillThroughput(PrefillThroughputBenchmark(
            benchmarkId: BenchmarkId(validated: "prefill_throughput_512"),
            parameterPrefillTokens: 512))
        let rungBody = try Self.body(rung)
        let (http, server, seen) = try RoutedURLProtocol.serve { request, _ in
            switch request.url?.path {
            case "/benchmarks/\(Self.visionId)"
                where request.value(forHTTPHeaderField: "If-None-Match") == #""v""#: .reply(304)
            case "/benchmarks/prefill_throughput_512": .reply(200, rungBody)
            default: .reply(500)
            }
        }

        var skipped: [String] = []
        let references = ["prefill_throughput_512", "local/decode_throughput_512_100",
                          Self.visionId, "local/vl_throughput_smoke"]
        let resolved = try await JobCell.resolvable(
            benchmarkIds: references, store: store, server: server, http: http, auth: nil
        ) { reference, _ in skipped.append(reference) }

        #expect(resolved.references == references)
        #expect(resolved.claimed == [rung, Self.vision])
        #expect(skipped.isEmpty)
        #expect(seen.methodsAndPaths == [
            "GET /benchmarks/prefill_throughput_512", "GET /benchmarks/\(Self.visionId)",
        ])
    }

    /// Skipped because `pending` types its cell from the store.
    @Test func aDefinitionTheStoreWillNotTakeIsSkipped() async throws {
        let store = makeTemporaryBenchmarkStore()
        defer { removeStore(store) }
        // A file where the remote half's directory goes, so the write fails.
        try FileManager.default.createDirectory(at: store.root, withIntermediateDirectories: true)
        try Data("not a directory".utf8).write(
            to: store.root.appendingPathComponent(BenchmarkSource.remote.rawValue))
        let body = try Self.body(Self.vision)
        let (http, server, seen) = try RoutedURLProtocol.serve { _, _ in .reply(200, body) }

        var skipped: [(reference: String, reason: String)] = []
        let resolved = try await JobCell.resolvable(
            benchmarkIds: [Self.visionId], store: store, server: server, http: http, auth: nil
        ) { skipped.append(($0, $1)) }

        #expect(resolved.references.isEmpty)
        #expect(skipped.map(\.reference) == [Self.visionId])
        #expect(skipped.first?.reason.contains("would not store it") == true)
        #expect(seen.methodsAndPaths == ["GET /benchmarks/\(Self.visionId)"])
    }
}
