import Foundation
import Testing

@testable import Pipette

/// Reports scripted progress, then either returns a directory populated with `files`
/// or throws. `nonisolated` so it's `Sendable` (required by `MLXModelDownloading`).
private nonisolated struct FakeDownloader: MLXModelDownloading {
    let outcome: Result<[String], DownloadError>
    var fractions: [Double] = [1.0]
    func download(_ ref: HFModelRef, token: AuthToken?,
                  progress: @escaping @Sendable (Double) -> Void) async throws -> URL {
        for f in fractions { progress(f) }
        switch outcome {
        case let .failure(error): throw error
        case let .success(files):
            let dir = FileManager.default.temporaryDirectory
                .appendingPathComponent("dl-\(UUID().uuidString)", isDirectory: true)
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            for f in files { try Data("x".utf8).write(to: dir.appendingPathComponent(f)) }
            return dir
        }
    }
}

/// Never completes — for cancellation.
private nonisolated struct HangingDownloader: MLXModelDownloading {
    func download(_ ref: HFModelRef, token: AuthToken?,
                  progress: @escaping @Sendable (Double) -> Void) async throws -> URL {
        try await Task.sleep(for: .seconds(30))
        return URL(fileURLWithPath: "/never")
    }
}

/// Reports one progress frame, then hangs — lets a test observe the transition
/// into `.downloading` without the download ever completing.
private nonisolated struct ProgressThenHangDownloader: MLXModelDownloading {
    let fraction: Double
    func download(_ ref: HFModelRef, token: AuthToken?,
                  progress: @escaping @Sendable (Double) -> Void) async throws -> URL {
        progress(fraction)
        try await Task.sleep(for: .seconds(30))
        return URL(fileURLWithPath: "/never")
    }
}

/// Records the token it was handed, then completes with a full bundle. `@unchecked
/// Sendable` + `NSLock`: `download` runs off the main actor, and the assertion reads the
/// capture back on it.
private nonisolated final class TokenCapturingDownloader: MLXModelDownloading, @unchecked Sendable {
    let files: [String]
    private let lock = NSLock()
    private var captured: AuthToken?

    init(files: [String]) { self.files = files }

    /// The token passed to the last `download`, or `nil` for an anonymous fetch.
    var capturedToken: AuthToken? { lock.withLock { captured } }

    func download(_ ref: HFModelRef, token: AuthToken?,
                  progress: @escaping @Sendable (Double) -> Void) async throws -> URL {
        lock.withLock { captured = token }
        progress(1.0)
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("dl-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        for f in files { try Data("x".utf8).write(to: dir.appendingPathComponent(f)) }
        return dir
    }
}

/// `true` iff the entry is in the `.failed` state (associated value ignored).
private func isFailed(_ dl: DownloadCoordinator.Download?) -> Bool {
    guard let state = dl?.state else { return false }
    if case .failed = state { return true }
    return false
}

/// The coordinator MLX flow, driven by a fake downloader (no network). Each test
/// builds its own `DownloadCoordinator` on a temporary `FileStorage`.
///
/// `.serialized`: the file-download path lazily creates a background `URLSession`
/// keyed by a process-global identifier, so these must not race each other.
@Suite(.serialized) @MainActor struct MLXDownloadCoordinatorTests {
    private static let completeModel = ["config.json", "model.safetensors", "tokenizer.json"]

    @Test func downloadsInstallsAndDiscoversRootModel() async throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }
        let ref = try HFModelRef.parse(repo: "LiquidAI/LFM2.5-350M-MLX-4bit")
        let coord = DownloadCoordinator(storage: storage)
        coord.mlxDownloader = FakeDownloader(outcome: .success(Self.completeModel), fractions: [0.5, 1.0])

        coord.startMLXDownload(ref, familyId: "lfm2.5-350m")
        try await waitUntil(coord, ref.key) { $0 == nil }

        let model = try #require(storage.availableModels().first { m in
            guard case .mlx = m.source else { return false }
            return m.hfRepo == ref.repo.description
        })
        #expect(model.name == "LFM2.5-350M-MLX-4bit")
        #expect(model.familyId == "lfm2.5-350m")
    }

    @Test func installsSubpathModel() async throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }
        let ref = try HFModelRef.parse(repo: "org/multi", subpath: "variant-4bit")
        let coord = DownloadCoordinator(storage: storage)
        coord.mlxDownloader = FakeDownloader(outcome: .success(Self.completeModel))

        coord.startMLXDownload(ref)
        try await waitUntil(coord, ref.key) { $0 == nil }

        let model = try #require(storage.availableModels().first { m in
            guard case .mlx = m.source else { return false }
            return m.hfRepo == ref.repo.description
        })
        #expect(model.name == "variant-4bit")
    }

    /// The plan's `auth_token` must reach the MLX transport. It used to stop at the
    /// coordinator: `HubMLXModelDownloader` read only the Keychain, which no plan-dispatched
    /// run writes, so every MLX pull fetched anonymously — 401 on a gated repo, and HF's
    /// per-IP anonymous rate limit on a public one.
    @Test func mlxDownloadPresentsTheCoordinateAuthToken() async throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }
        var repo = try HFRepo.parse("LiquidAI/LFM2.5-350M-MLX-4bit")
        repo.authToken = try AuthToken("hf_plan_token")
        let ref = HFModelRef(repo: repo, subpath: nil)
        let downloader = TokenCapturingDownloader(files: Self.completeModel)
        let coord = DownloadCoordinator(storage: storage)
        coord.mlxDownloader = downloader

        coord.startMLXDownload(ref)
        try await waitUntil(coord, ref.key) { $0 == nil }

        #expect(downloader.capturedToken?.value == "hf_plan_token")
    }

    /// The other half of the contract: a coordinate with no token fetches anonymously,
    /// which is what a public repo gets. Deterministic in a Simulator host — it cannot
    /// store Keychain items, so the two fallback tiers are empty by construction.
    @Test func mlxDownloadWithNoCoordinateTokenFetchesAnonymously() async throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }
        let ref = try HFModelRef.parse(repo: "LiquidAI/LFM2.5-350M-MLX-4bit")
        let downloader = TokenCapturingDownloader(files: Self.completeModel)
        let coord = DownloadCoordinator(storage: storage)
        coord.mlxDownloader = downloader

        coord.startMLXDownload(ref)
        try await waitUntil(coord, ref.key) { $0 == nil }

        #expect(downloader.capturedToken == nil)
    }

    /// PIP-281: the in-flight MLX download row must show its quant, never "unknown".
    /// An MLX download's `filename` is the repo/dir leaf (not a GGUF weight file), so a
    /// filename K-quant parse can't name it. Here the repo slug also doesn't encode a
    /// bit-width, so the source-derived quant is nil too — proving the row falls back to
    /// the catalog's explicitly-threaded `quant`.
    @Test func inFlightMlxDownloadRowUsesExplicitCatalogQuant() throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }
        let ref = try HFModelRef.parse(repo: "org/custom-mlx")
        let coord = DownloadCoordinator(storage: storage)
        coord.mlxDownloader = HangingDownloader()   // stays in-flight so the row is observable

        coord.startMLXDownload(ref, quant: "4bit")
        defer { coord.cancel(key: ref.key) }

        let dl = try #require(coord.downloads[ref.key])
        #expect(dl.source?.quant == nil, "precondition: this repo slug doesn't encode a bit-width")
        #expect(dl.quant == "4bit", "row falls back to the explicit catalog quant, not \"unknown\"")
    }

    /// Regression guard for the silent empty-download bug: an incomplete download
    /// fails the cell and records nothing.
    @Test func incompleteDownloadFailsAndRecordsNothing() async throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }
        let ref = try HFModelRef.parse(repo: "org/empty")
        let coord = DownloadCoordinator(storage: storage)
        coord.mlxDownloader = FakeDownloader(outcome: .success([]))  // empty dir

        coord.startMLXDownload(ref)
        try await waitUntil(coord, ref.key) { isFailed($0) }
        #expect(storage.availableModels().first { $0.hfRepo == ref.repo.description } == nil)
    }

    @Test func transportErrorFailsTheCell() async throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }
        let ref = try HFModelRef.parse(repo: "org/boom")
        let coord = DownloadCoordinator(storage: storage)
        coord.mlxDownloader = FakeDownloader(outcome: .failure(.transport("network")))

        coord.startMLXDownload(ref)
        try await waitUntil(coord, ref.key) { isFailed($0) }
        // The failure carries the transport error text in the state's reason.
        guard case .failed(let reason)? = coord.downloads[ref.key]?.state else {
            Issue.record("expected a .failed state with a reason"); return
        }
        #expect(reason.hasPrefix("Failed:"))
    }

    @Test func cancelStopsInFlightDownload() throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }
        let ref = try HFModelRef.parse(repo: "org/slow")
        let coord = DownloadCoordinator(storage: storage)
        coord.mlxDownloader = HangingDownloader()

        coord.startMLXDownload(ref)
        try #require(coord.downloads[ref.key] != nil, "startMLXDownload should register an in-flight entry")
        coord.cancel(key: ref.key)
        #expect(coord.downloads[ref.key] == nil)
    }

    /// State transition: an MLX pull registers in `.connecting` synchronously, then
    /// flips to `.downloading` on the first progress frame.
    @Test func mlxDownloadConnectingThenDownloading() async throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }
        let ref = try HFModelRef.parse(repo: "org/progress")
        let coord = DownloadCoordinator(storage: storage)
        coord.mlxDownloader = ProgressThenHangDownloader(fraction: 0.5)
        defer { coord.cancel(key: ref.key) }

        coord.startMLXDownload(ref)
        // Registered synchronously, before the download Task's first progress hop.
        #expect(coord.downloads[ref.key]?.state == .connecting)

        try await waitUntil(coord, ref.key) { $0?.state == .downloading }
        #expect(coord.downloads[ref.key]?.state == .downloading)
    }

    /// State transition on the `.file` (URLSession) path: `startDownload` registers
    /// in `.connecting`, and `pause(key:)` (gated to `.file`) moves it to `.pausing`.
    /// The states asserted are set synchronously, before any network I/O settles, so
    /// the test doesn't depend on the transfer making progress; `cancel` tears down
    /// the enqueued task.
    @Test func fileDownloadConnectingThenPausing() throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }
        let coord = DownloadCoordinator(storage: storage)
        let repo = try HFRepo.parse("org/gguf")
        let filename = try RepoSubpath("model.Q4_0.gguf")
        let key = LocalStorage.modelRelativePath(repo: repo.description, filename: filename.value)
        defer { coord.cancel(key: key) }

        coord.startDownload(.ggufText(GgufText(source: .huggingFace(repo: repo, path: filename, sha256: nil))))
        #expect(coord.downloads[key]?.state == .connecting)

        coord.pause(key: key)
        #expect(coord.downloads[key]?.state == .pausing)
    }

    /// PIP-369: the coordinator caps simultaneously-active transfers. Starting
    /// more than `maxConcurrentDownloads` leaves the surplus in `.queued`, and
    /// finishing an active one (here via cancel) promotes a queued download into
    /// flight. Uses the MLX path with a hanging downloader so every started pull
    /// stays in `.connecting` and the slot accounting is observable.
    @Test func concurrencyGateQueuesSurplusAndPromotesOnSlotFree() throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }
        let coord = DownloadCoordinator(storage: storage)
        coord.mlxDownloader = HangingDownloader()   // every pull stays in-flight

        let cap = DownloadCoordinator.maxConcurrentDownloads
        // cap + 1 distinct models, so exactly one must wait.
        let refs = try (0...cap).map { try HFModelRef.parse(repo: "org/model-\($0)") }
        for ref in refs { coord.startMLXDownload(ref) }
        defer { for ref in refs { coord.cancel(key: ref.key) } }

        let active = refs.filter { coord.downloads[$0.key]?.state == .connecting }
        let queued = refs.filter { coord.downloads[$0.key]?.state == .queued }
        #expect(active.count == cap, "only the cap should be in flight at once")
        #expect(queued.count == 1, "the surplus download should be queued")

        // Free a slot; the queued download should be promoted into flight.
        coord.cancel(key: active[0].key)
        #expect(queued.allSatisfy { coord.downloads[$0.key]?.state == .connecting },
                "a freed slot promotes the queued download into flight")
    }

    /// PIP-369 (review G2-2): a still-queued download must not be pausable —
    /// pausing it would flip it to `.paused` while its start closure lingers in
    /// `pendingStarts`, so `pumpQueue` could later resurrect it. Exercises the
    /// GGUF path so a real `.queued` entry exists. States are set synchronously,
    /// before any network I/O settles; `cancel` tears down the enqueued tasks.
    @Test func pausingAQueuedDownloadIsIgnored() throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }
        let coord = DownloadCoordinator(storage: storage)
        let cap = DownloadCoordinator.maxConcurrentDownloads

        // cap + 1 GGUF downloads: the surplus one is forced to wait in `.queued`.
        let coordinates = try (0...cap).map { i in
            (repo: try HFRepo.parse("org/gguf-\(i)"), path: try RepoSubpath("model.Q4_0.gguf"))
        }
        let models = coordinates.map {
            GgufText(source: .huggingFace(repo: $0.repo, path: $0.path, sha256: nil))
        }
        let keys = coordinates.map {
            LocalStorage.modelRelativePath(repo: $0.repo.description, filename: $0.path.value)
        }
        for m in models { coord.startDownload(.ggufText(m)) }
        defer { for k in keys { coord.cancel(key: k) } }

        let queuedKey = try #require(keys.first { coord.downloads[$0]?.state == .queued },
                                     "one GGUF download should wait in .queued past the cap")
        coord.pause(key: queuedKey)
        #expect(coord.downloads[queuedKey]?.state == .queued,
                "pausing a queued download is a no-op — it stays queued")
    }

    /// PIP-369 (review G2-1): `resume` is valid only from `.paused`. Calling it on
    /// an active download must be a no-op, so it can't spawn a duplicate transfer
    /// task alongside the running one.
    @Test func resumeIsIgnoredUnlessPaused() throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }
        let coord = DownloadCoordinator(storage: storage)
        let repo = try HFRepo.parse("org/gguf")
        let filename = try RepoSubpath("model.Q4_0.gguf")
        let key = LocalStorage.modelRelativePath(repo: repo.description, filename: filename.value)
        defer { coord.cancel(key: key) }

        coord.startDownload(.ggufText(GgufText(source: .huggingFace(repo: repo, path: filename, sha256: nil))))
        #expect(coord.downloads[key]?.state == .connecting)
        coord.resume(key: key)
        #expect(coord.downloads[key]?.state == .connecting,
                "resume on a non-paused download is ignored")
    }

    /// Poll until `done` holds for the entry (awaiting yields the main actor so the
    /// coordinator's completion hop can run) or a timeout elapses.
    private func waitUntil(_ coord: DownloadCoordinator, _ key: String, timeoutMs: Int = 5000,
                           _ done: (DownloadCoordinator.Download?) -> Bool) async throws {
        var waited = 0
        while !done(coord.downloads[key]), waited < timeoutMs {
            try await Task.sleep(for: .milliseconds(20))
            waited += 20
        }
    }
}
