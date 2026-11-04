import Foundation
import Testing

@testable import Pipette

/// A claimed `BenchmarksCommands.run` fetches a benchmark its store lacks before it asks for
/// the model; a run without a claim fetches nothing, as `pipette-cli`'s `benchmarks run`.
///
/// The coordinator reaches the vault through the test's own route, so a model fetch is
/// recorded beside the benchmark fetch, in order.
@MainActor
struct BenchmarksRunFetchTests {
    /// Only the server describes it.
    private static let visionId = "vl_throughput_256x256_text1024_decode256_img5"
    private static let vaultId = "vault_0195c3e2"

    /// A workspace registered with `server`, as a proxied phone's is.
    private func registeredWorkspace(at server: ServerUrl) throws -> PipetteWorkspace {
        let ws = makeTemporaryWorkspace()
        let auth = try testIdentity()
        try ws.registration.putPrivateKey(auth.privateKeyHex)
        try ws.registration.put(IdentityRegistration(
            clientId: auth.clientId, status: "approved", serverUrl: server,
            organization: "test", contactEmail: "test@example.com",
            registeredAt: "2026-10-17T00:00:00Z"))
        return ws
    }

    /// A coordinator whose only way out is the vault over `vaultHttp`.
    private func coordinator(_ ws: PipetteWorkspace, vaultHttp: HttpClient) -> DownloadCoordinator {
        DownloadCoordinator(
            ws: ws, downloadHttpClient: UnreachableURLProtocol.httpClient,
            foregroundSession: true, reachability: FixedReachability(isConnected: true),
            vaultHttpClient: vaultHttp)
    }

    /// The proxy's invocation: one model the phone does not hold, one benchmark id, and
    /// the claim.
    private func run(
        _ ws: PipetteWorkspace, model: Model, http: HttpClient, coordinator: DownloadCoordinator,
        claim: String? = "job-1"
    ) async -> Bool {
        await BenchmarksCommands.run(
            spec: .model(model), modelMatch: nil, quant: nil, runtime: .llamacppIosPipette,
            batch: 512, benchmarks: [Self.visionId], metrics: [], offsets: [], submit: false,
            claim: claim, ws: ws, http: http, coordinator: coordinator)
    }

    /// An id the server does not know ends the run after one signed `GET`, and the model
    /// is never asked for.
    @Test func anUnknownBenchmarkEndsTheRunBeforeTheModelIsFetched() async throws {
        let (http, server, seen) = try RoutedURLProtocol.serve { _, _ in
            .reply(404, "benchmark not found")
        }
        let ws = try registeredWorkspace(at: server)
        defer { removeWorkspace(ws) }
        let model = try vaultTextSpec(Self.vaultId)
        let coordinator = coordinator(ws, vaultHttp: http)

        let ok = await run(ws, model: model, http: http, coordinator: coordinator)

        #expect(!ok)
        #expect(seen.methodsAndPaths == ["GET /benchmarks/\(Self.visionId)"])
        #expect(seen.requests.first?.value(forHTTPHeaderField: "X-Client-Id") == "client-1")
        #expect(!coordinator.hasTransfer(for: model))
    }

    @Test func aFetchThatFailsInTransitEndsTheRunBeforeTheModelIsFetched() async throws {
        let (vaultHttp, server, seen) = try RoutedURLProtocol.serve { _, _ in .reply(404) }
        let ws = try registeredWorkspace(at: server)
        defer { removeWorkspace(ws) }
        let model = try vaultTextSpec(Self.vaultId)
        let coordinator = coordinator(ws, vaultHttp: vaultHttp)

        let ok = await run(
            ws, model: model, http: UnreachableURLProtocol.httpClient, coordinator: coordinator)

        #expect(!ok)
        #expect(seen.requests.isEmpty, "the model fetch never started")
        #expect(!coordinator.hasTransfer(for: model))
    }

    /// A known id is fetched and cached before the model is asked for. The vault refuses
    /// the weights, so the run ends there.
    @Test func aBenchmarkTheServerKnowsIsFetchedBeforeTheModel() async throws {
        let vision = BenchmarkDefinition.vlThroughput(VlThroughputBenchmark(
            benchmarkId: BenchmarkId(validated: Self.visionId),
            parameterImageWidth: 256,
            parameterImageHeight: 256,
            parameterTextTokens: 1024,
            parameterDecodeTokens: 256,
            parameterNumImages: 5))
        let body = String(decoding: try Coding.encoder.encode(vision), as: UTF8.self)
        let (http, server, seen) = try RoutedURLProtocol.serve { request, _ in
            request.url?.path.hasPrefix("/benchmarks/") == true
                ? .reply(200, body) : .reply(404, "resource not found")
        }
        let ws = try registeredWorkspace(at: server)
        defer { removeWorkspace(ws) }
        let model = try vaultTextSpec(Self.vaultId)
        let coordinator = coordinator(ws, vaultHttp: http)

        let ok = await run(ws, model: model, http: http, coordinator: coordinator)

        #expect(!ok)
        #expect(seen.methodsAndPaths == [
            "GET /benchmarks/\(Self.visionId)", "GET \(StubVault.recordPath(Self.vaultId))",
        ])
        #expect(try ws.benchmarks.get(.remote(BenchmarkId(validated: Self.visionId))) == vision)
    }

    /// The body is not fetched again on a `304`.
    @Test func aCachedBenchmarkIsCheckedBeforeTheModel() async throws {
        let vision = BenchmarkDefinition.vlThroughput(VlThroughputBenchmark(
            benchmarkId: BenchmarkId(validated: Self.visionId),
            parameterImageWidth: 256,
            parameterImageHeight: 256,
            parameterTextTokens: 1024,
            parameterDecodeTokens: 256,
            parameterNumImages: 5))
        let visionId = Self.visionId
        let (http, server, seen) = try RoutedURLProtocol.serve { request, _ in
            switch request.url?.path {
            case "/benchmarks/\(visionId)"
                where request.value(forHTTPHeaderField: "If-None-Match") == #""v""#: .reply(304)
            case "/benchmarks/\(visionId)": .reply(500, "no body expected")
            default: .reply(404, "resource not found")
            }
        }
        let ws = try registeredWorkspace(at: server)
        defer { removeWorkspace(ws) }
        try ws.benchmarks.cacheRemote(vision, etag: #""v""#)
        let model = try vaultTextSpec(Self.vaultId)
        let coordinator = coordinator(ws, vaultHttp: http)

        let ok = await run(ws, model: model, http: http, coordinator: coordinator)

        #expect(!ok)
        #expect(seen.methodsAndPaths == [
            "GET /benchmarks/\(Self.visionId)", "GET \(StubVault.recordPath(Self.vaultId))",
        ])
        #expect(seen.requests.first?.value(forHTTPHeaderField: "X-Client-Id") == "client-1")
    }

    /// Without a claim nothing is fetched, and a benchmark the store lacks still ends the
    /// run before the model is asked for.
    @Test func aRunWithoutAClaimFetchesNothing() async throws {
        let (http, server, seen) = try RoutedURLProtocol.serve { _, _ in .reply(404) }
        let ws = try registeredWorkspace(at: server)
        defer { removeWorkspace(ws) }
        let model = try vaultTextSpec(Self.vaultId)
        let coordinator = coordinator(ws, vaultHttp: http)

        let ok = await run(ws, model: model, http: http, coordinator: coordinator, claim: nil)

        #expect(!ok)
        #expect(seen.requests.isEmpty)
        #expect(!coordinator.hasTransfer(for: model))
    }

    /// A claimed AFM run, which the proxy launches through `runAFM` rather than `run`,
    /// fetches a store miss too.
    @Test func aClaimedAfmRunFetchesAStoreMiss() async throws {
        let (http, server, seen) = try RoutedURLProtocol.serve { _, _ in
            .reply(404, "benchmark not found")
        }
        let ws = try registeredWorkspace(at: server)
        defer { removeWorkspace(ws) }

        let ok = await BenchmarksCommands.runAFM(
            benchmarkIds: [Self.visionId], submit: false, claim: "job-1", ws: ws, http: http)

        #expect(!ok)
        #expect(seen.methodsAndPaths == ["GET /benchmarks/\(Self.visionId)"])
        #expect(seen.requests.first?.value(forHTTPHeaderField: "X-Client-Id") == "client-1")
    }
}
