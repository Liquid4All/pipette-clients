import Foundation
import Testing

@testable import Pipette

/// Refusal cases pass a short `stallTimeout`, so a regression that waits for the deadline
/// fails rather than stalling the suite.
///
/// `.serialized`: these share `StubURLProtocol.answer` and the coordinator's
/// process-global background session.
@Suite(.serialized) @MainActor struct EnsureModelTests {

    private static var stubHttpClient: HttpClient {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [StubURLProtocol.self]
        return HttpClient(userAgent: "pipette-test/0", configuration: configuration)
    }

    /// What makes re-tapping Download on installed weights a no-op.
    @Test func aHitEnqueuesNoTransfer() async throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let spec = try ggufTextSpec("org/a-GGUF", "a-Q4_0.gguf")
        try installEntry(ws, spec)
        let coordinator = DownloadCoordinator(ws: ws)

        let bound = try await ensureModel(spec, ws: ws, coordinator: coordinator,
                                          stallTimeout: .seconds(2))

        #expect(!coordinator.hasTransfer(for: spec))
        #expect(bound.boundPaths?.payload == ws.models.payloadPath(for: spec))
    }

    /// Refused before anything is awaited: `startDownload` would decline it without
    /// registering a transfer, so waiting would only reach the same answer at the deadline.
    @Test func anUnfetchableArmIsRefusedByName() async throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let url = try ResourceUrl("https://example.com/weights.gguf")
        let spec = Model.ggufText(GgufText(source: .url(url: url, sha256: nil)))
        let coordinator = DownloadCoordinator(ws: ws)

        await #expect(throws: ModelStoreError.notFetchableHere(spec.artifactName)) {
            _ = try await ensureModel(spec, ws: ws, coordinator: coordinator,
                                      stallTimeout: .seconds(2))
        }
        #expect(!coordinator.hasTransfer(for: spec))
    }

    /// `availableBytes` equals the quota because the store is empty: nothing could be evicted.
    @Test func anOversizeArtifactIsRefusedBeforeAnyTransfer() async throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        try ws.setStorageQuotaBytes(1_000_000)
        let spec = try ggufTextSpec("org/big-GGUF", "big-Q4_0.gguf")
        let coordinator = DownloadCoordinator(ws: ws)

        await #expect(throws: ModelStoreError.admission(.exceedsQuota(
            artifact: spec.description, requiredBytes: 9_000_000, availableBytes: 1_000_000,
            protectedBytes: 0, quotaBytes: 1_000_000))) {
            _ = try await ensureModel(spec, ws: ws, coordinator: coordinator,
                                      declaredSizeBytes: 9_000_000, stallTimeout: .seconds(2))
        }
        #expect(!coordinator.hasTransfer(for: spec))
    }

    /// A caller that declares no size (a claim, `bench`, `models pull`) is sized from the
    /// Hub, as upstream sizes every miss.
    @Test func anUndeclaredSizeIsAskedOfTheHubAndRefusedWhenOversize() async throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        try ws.setStorageQuotaBytes(1_000_000)
        let spec = try ggufTextSpec("org/big-GGUF", "big-Q4_0.gguf")
        let seen = StubRecorder()
        StubURLProtocol.answer = { request in
            seen.record(request)
            return StubResponse(status: 200, body: Data(), headers: ["Content-Length": "9000000"])
        }
        let coordinator = DownloadCoordinator(ws: ws,
                                              downloadHttpClient: Self.stubHttpClient)

        await #expect(throws: ModelStoreError.admission(.exceedsQuota(
            artifact: spec.description, requiredBytes: 9_000_000, availableBytes: 1_000_000,
            protectedBytes: 0, quotaBytes: 1_000_000))) {
            _ = try await ensureModel(spec, ws: ws, coordinator: coordinator,
                                      stallTimeout: .seconds(2))
        }
        #expect(seen.methodsAndPaths == ["HEAD /org/big-GGUF/resolve/main/big-Q4_0.gguf"])
        #expect(!coordinator.hasTransfer(for: spec))
    }

    /// A hit costs no bytes, and a declared size is the catalog's authored figure.
    @Test func theHubIsNotAskedForAHitOrADeclaredSize() async throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        try ws.setStorageQuotaBytes(1_000_000)
        let held = try ggufTextSpec("org/a-GGUF", "a-Q4_0.gguf")
        try installEntry(ws, held)
        let declared = try ggufTextSpec("org/big-GGUF", "big-Q4_0.gguf")
        let seen = StubRecorder()
        StubURLProtocol.answer = { request in
            seen.record(request)
            return StubResponse(status: 404, body: Data(), headers: [:])
        }
        let coordinator = DownloadCoordinator(ws: ws,
                                              downloadHttpClient: Self.stubHttpClient)

        _ = try await ensureModel(held, ws: ws, coordinator: coordinator,
                                  stallTimeout: .seconds(2))
        await #expect(throws: ModelStoreError.self) {
            _ = try await ensureModel(declared, ws: ws, coordinator: coordinator,
                                      declaredSizeBytes: 9_000_000, stallTimeout: .seconds(2))
        }
        #expect(seen.requests.isEmpty)
    }

    /// Held weights cost no bytes, as upstream, where a cache hit neither pre-flights nor
    /// sweeps.
    @Test func aHitIsNotRefusedForItsSize() async throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let spec = try ggufTextSpec("org/big-GGUF", "big-Q4_0.gguf")
        try installEntry(ws, spec)
        try ws.setStorageQuotaBytes(1_000_000)
        let coordinator = DownloadCoordinator(ws: ws)

        let bound = try await ensureModel(spec, ws: ws, coordinator: coordinator,
                                          declaredSizeBytes: 9_000_000, stallTimeout: .seconds(2))

        #expect(bound.boundPaths?.payload == ws.models.payloadPath(for: spec))
    }

    /// It ships with the OS and has no store entry.
    @Test func appleFoundationIsNotEnsurable() async throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let coordinator = DownloadCoordinator(ws: ws)

        await #expect(throws: ModelStoreError.notStorable(Model.appleFoundationText.artifactName)) {
            _ = try await ensureModel(.appleFoundationText, ws: ws,
                                      coordinator: coordinator, stallTimeout: .seconds(2))
        }
    }

    /// Once a transfer leaves the coordinator nothing drives it (the reconcile sweep drops a
    /// restored row a foreground session won't adopt, a cancel removes one outright), so
    /// waiting out the 10-minute window would only delay an answer already on hand. Driven
    /// through the MLX arm because its `ModelTransport` is injectable.
    @Test func aTransferThatLeavesTheCoordinatorIsReportedNotPolledToTheDeadline() async throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let ref = try HfModelRef.parse(repo: "org/vanishing-MLX")
        let spec = Model.mlx(ref.asMlx())
        let coordinator = DownloadCoordinator(ws: ws,
                                              downloadHttpClient: UnreachableURLProtocol.httpClient)
        coordinator.transport = HangingDownloader()

        // A deadline far past the poll, so a regression times out rather than passing
        // slowly. Sized explicitly: the HTTP client is unreachable, and admission refuses an
        // artifact nobody can size.
        let fetch = Task {
            try await ensureModel(spec, ws: ws, coordinator: coordinator,
                                  declaredSizeBytes: 4096, stallTimeout: .seconds(600))
        }
        try await waitUntil(coordinator, ref.key) { $0 != nil }
        coordinator.cancel(key: ref.key)

        let failure = await #expect(throws: ModelStoreError.self) { _ = try await fetch.value }
        guard case let .fetch(_, reason) = try #require(failure) else {
            Issue.record("expected a fetch, got \(String(describing: failure))"); return
        }
        #expect(reason == "the transfer stopped without completing")
    }

    /// The window bounds a stall, not the fetch: it runs only once the bytes stop.
    @Test func aFetchThatKeepsMovingOutlastsTheStallWindow() async throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let ref = try HfModelRef.parse(repo: "org/trickling-MLX")
        let spec = Model.mlx(ref.asMlx())
        let coordinator = DownloadCoordinator(ws: ws,
                                              downloadHttpClient: UnreachableURLProtocol.httpClient)
        // Frames wider than the coordinator's 1 s progress debounce, so none is dropped, and a
        // window wide enough for one late frame plus a 1 s poll.
        coordinator.transport = TrickleThenStallDownloader(frames: 4, every: .milliseconds(1200))
        defer { coordinator.cancel(key: ref.key) }

        let started = ContinuousClock.now
        let failure = await #expect(throws: ModelStoreError.self) {
            _ = try await ensureModel(spec, ws: ws, coordinator: coordinator,
                                      declaredSizeBytes: 4096, stallTimeout: .seconds(4))
        }
        guard case .fetchTimedOut = try #require(failure) else {
            Issue.record("expected a fetchTimedOut, got \(String(describing: failure))"); return
        }
        // 3.6 s of progress, then four of nothing; a window run from the start ends near 4 s.
        #expect(ContinuousClock.now - started >= .seconds(6))
    }
}

/// Never completes, so a transfer stays in flight until the test disposes of it.
private nonisolated struct HangingDownloader: ModelTransport {
    func download(_ model: Model, ref: HfModelRef, token: AuthToken?,
                  admit: @escaping @Sendable (Int64) async throws -> Void,
                  progress: @escaping @Sendable (Double) -> Void) async throws -> TransferOutcome {
        try await Task.sleep(for: .seconds(30))
        return TransferOutcome(directory: URL(fileURLWithPath: "/never"), resolved: nil)
    }
}

private nonisolated struct TrickleThenStallDownloader: ModelTransport {
    let frames: Int
    let every: Duration
    func download(_ model: Model, ref: HfModelRef, token: AuthToken?,
                  admit: @escaping @Sendable (Int64) async throws -> Void,
                  progress: @escaping @Sendable (Double) -> Void) async throws -> TransferOutcome {
        for frame in 1...frames {
            progress(Double(frame) / Double(frames + 1))
            try await Task.sleep(for: every)
        }
        try await Task.sleep(for: .seconds(30))
        return TransferOutcome(directory: URL(fileURLWithPath: "/never"), resolved: nil)
    }
}
