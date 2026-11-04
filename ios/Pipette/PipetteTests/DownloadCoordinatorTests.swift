import Foundation
import Testing

@testable import Pipette

/// Returns a complete MLX bundle, so `finishDirectory` reaches the install and sweep.
private nonisolated struct CompleteMLXDownloader: ModelTransport {
    let payloadBytes: Int
    func download(_ model: Model, ref: HfModelRef, token: AuthToken?,
                  admit: @escaping @Sendable (Int64) async throws -> Void,
                  progress: @escaping @Sendable (Double) -> Void) async throws -> TransferOutcome {
        progress(1.0)
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("dl-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        for name in mlxBundleFiles {
            try Data(repeating: 0x78, count: payloadBytes).write(to: dir.appendingPathComponent(name))
        }
        return TransferOutcome(directory: dir, resolved: nil)
    }
}

/// Reports a frozen plan heavier than the caller declared, then completes. Stands in for
/// a catalog row whose `bytes` understates the package the Hub actually lists.
private nonisolated struct OversizePlanTransport: ModelTransport {
    let plannedBytes: Int64
    func download(_ model: Model, ref: HfModelRef, token: AuthToken?,
                  admit: @escaping @Sendable (Int64) async throws -> Void,
                  progress: @escaping @Sendable (Double) -> Void) async throws -> TransferOutcome {
        try await admit(plannedBytes)
        // Held open: returning would fail to install `/never` and hand the budget back
        // before a test could read it.
        try await Task.sleep(for: .seconds(30))
        return TransferOutcome(directory: URL(fileURLWithPath: "/never"), resolved: nil)
    }
}

/// A minimal ONNX `ModelProto` whose single initializer keeps its tensor in `sidecar`.
///
/// Real protobuf: `OnnxGenaiModels` walks `ModelProto.graph(7)` → `GraphProto.initializer(5)`
/// → `TensorProto.external_data(13)` → `StringStringEntryProto.key/value(1/2)` and rejects
/// anything else. Every length is one byte, which holds while the sidecar name stays under
/// ~100 characters.
func onnxGraphNaming(_ sidecar: String) -> Data {
    func lengthDelimited(field: UInt8, _ payload: Data) -> Data {
        var out = Data([field << 3 | 2, UInt8(payload.count)])
        out.append(payload)
        return out
    }
    var entry = lengthDelimited(field: 1, Data("location".utf8))
    entry.append(lengthDelimited(field: 2, Data(sidecar.utf8)))
    var tensor = lengthDelimited(field: 13, entry)
    tensor.append(contentsOf: [14 << 3 | 0, 1])  // data_location = EXTERNAL
    let graph = lengthDelimited(field: 5, tensor)
    return lengthDelimited(field: 7, graph)
}

/// Writes a complete ONNX Runtime GenAI package: the config, the graph it names, and the
/// external-data sidecar the graph references.
private nonisolated struct CompleteOnnxTransport: ModelTransport {
    func download(_ model: Model, ref: HfModelRef, token: AuthToken?,
                  admit: @escaping @Sendable (Int64) async throws -> Void,
                  progress: @escaping @Sendable (Double) -> Void) async throws -> TransferOutcome {
        progress(1.0)
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("dl-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let config: [String: Any] = ["model": ["decoder": ["filename": "model.onnx"]]]
        try JSONSerialization.data(withJSONObject: config)
            .write(to: dir.appendingPathComponent("genai_config.json"))
        try onnxGraphNaming("model.onnx_data")
            .write(to: dir.appendingPathComponent("model.onnx"))
        try Data(repeating: 0x77, count: 512)
            .write(to: dir.appendingPathComponent("model.onnx_data"))
        // Validation requires the tokenizer assets ORT Extensions loads before it publishes.
        try JSONSerialization.data(withJSONObject: [String: Any]())
            .write(to: dir.appendingPathComponent("tokenizer_config.json"))
        try JSONSerialization.data(withJSONObject: [String: Any]())
            .write(to: dir.appendingPathComponent("tokenizer.json"))
        return TransferOutcome(directory: dir, resolved: nil)
    }
}

/// Reports scripted progress, then returns a directory of `outcome`'s files or throws.
private nonisolated struct FakeDownloader: ModelTransport {
    let outcome: Result<[String], DownloadError>
    var fractions: [Double] = [1.0]
    func download(_ model: Model, ref: HfModelRef, token: AuthToken?,
                  admit: @escaping @Sendable (Int64) async throws -> Void,
                  progress: @escaping @Sendable (Double) -> Void) async throws -> TransferOutcome {
        for f in fractions { progress(f) }
        switch outcome {
        case let .failure(error): throw error
        case let .success(files):
            let dir = FileManager.default.temporaryDirectory
                .appendingPathComponent("dl-\(UUID().uuidString)", isDirectory: true)
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            for f in files { try Data("x".utf8).write(to: dir.appendingPathComponent(f)) }
            return TransferOutcome(directory: dir, resolved: nil)
        }
    }
}

private nonisolated struct HangingDownloader: ModelTransport {
    func download(_ model: Model, ref: HfModelRef, token: AuthToken?,
                  admit: @escaping @Sendable (Int64) async throws -> Void,
                  progress: @escaping @Sendable (Double) -> Void) async throws -> TransferOutcome {
        try await Task.sleep(for: .seconds(30))
        return TransferOutcome(directory: URL(fileURLWithPath: "/never"), resolved: nil)
    }
}

private nonisolated struct ProgressThenHangDownloader: ModelTransport {
    let fraction: Double
    func download(_ model: Model, ref: HfModelRef, token: AuthToken?,
                  admit: @escaping @Sendable (Int64) async throws -> Void,
                  progress: @escaping @Sendable (Double) -> Void) async throws -> TransferOutcome {
        progress(fraction)
        try await Task.sleep(for: .seconds(30))
        return TransferOutcome(directory: URL(fileURLWithPath: "/never"), resolved: nil)
    }
}

/// Records the token it was handed, then completes. `NSLock`: `download` runs off the main
/// actor, and the assertion reads the capture back on it.
private nonisolated final class TokenCapturingDownloader: ModelTransport, @unchecked Sendable {
    let files: [String]
    private let lock = NSLock()
    private var captured: AuthToken?

    init(files: [String]) { self.files = files }

    var capturedToken: AuthToken? { lock.withLock { captured } }

    func download(_ model: Model, ref: HfModelRef, token: AuthToken?,
                  admit: @escaping @Sendable (Int64) async throws -> Void,
                  progress: @escaping @Sendable (Double) -> Void) async throws -> TransferOutcome {
        lock.withLock { captured = token }
        progress(1.0)
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("dl-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        for f in files { try Data("x".utf8).write(to: dir.appendingPathComponent(f)) }
        return TransferOutcome(directory: dir, resolved: nil)
    }
}

private func isFailed(_ dl: DownloadCoordinator.Download?) -> Bool {
    guard let state = dl?.state else { return false }
    if case .failed = state { return true }
    return false
}

/// `DownloadCoordinator`: the transfers it owns and the state it answers about them.
///
/// The stateful suites serialize because the file-download path lazily creates a background `URLSession`
/// keyed by a process-global identifier, and constructing a coordinator registers the
/// process-wide `NetworkReachability.onReconnect` handler.
@Suite
struct DownloadCoordinatorTests {
    /// `captureProvenance` reads `downloads[key]` before the move, so a `cancel(key:)` racing
    /// the move cannot strip the repo/source the installer needs, which would orphan the file
    /// from the next scan. Writing the manifest is `GGUFModelInstallerTests`.
    @Suite(.serialized) @MainActor struct Provenance {
        private static let repo = "LiquidAI/LFM2.5-350M-GGUF"
        private static let filename = "LFM2.5-350M-Q4_0.gguf"
        private static let familyId = "lfm2.5-350m"

        @Test func captureProvenanceReadsInMemoryEntryWhenPresent() throws {
            let ws = makeTemporaryWorkspace()
            defer { removeWorkspace(ws) }

            let coordinator = DownloadCoordinator(ws: ws)
            let key = LocalStorage.modelRelativePath(repo: Self.repo, filename: Self.filename)
            #expect(coordinator.downloads[key] == nil, "test process should start with no in-flight download")

            let source = Model.ggufText(.init(source: .huggingFace(repo: try HfRepo.parseOrgRepo(Self.repo), path: try RepoSubpath(Self.filename), sha256: nil)))
            coordinator.startDownload(source, familyId: Self.familyId)
            defer { coordinator.cancel(key: key) }
            try #require(coordinator.downloads[key] != nil, "startDownload should populate the in-flight entry")

            let provenance = coordinator.captureProvenance(key: key)
            #expect(provenance.repo == Self.repo)
            #expect(provenance.filename == Self.filename)
            // Key reconstruction cannot recover the typed source.
            #expect(provenance.source == source)
        }

        @Test func captureProvenanceReconstructsFromKeyWhenNoRecord() throws {
            let ws = makeTemporaryWorkspace()
            defer { removeWorkspace(ws) }

            let coordinator = DownloadCoordinator(ws: ws)
            let key = LocalStorage.modelRelativePath(repo: Self.repo, filename: Self.filename)
            #expect(coordinator.downloads[key] == nil, "test process should start with no in-flight download")

            let provenance = coordinator.captureProvenance(key: key)
            #expect(provenance.repo == Self.repo)
            #expect(provenance.filename == Self.filename)
            #expect(provenance.source == nil)
        }

        @Test func sourceDrivenIgnitionCarriesTheSource() throws {
            let ws = makeTemporaryWorkspace()
            defer { removeWorkspace(ws) }

            let coordinator = DownloadCoordinator(ws: ws)
            let key = LocalStorage.modelRelativePath(repo: Self.repo, filename: Self.filename)
            #expect(coordinator.downloads[key] == nil, "test process should start with no in-flight download")

            let source = Model.ggufText(.init(source: .huggingFace(repo: try HfRepo.parseOrgRepo(Self.repo), path: try RepoSubpath(Self.filename), sha256: nil)))
            coordinator.startDownload(source, familyId: Self.familyId)
            defer { coordinator.cancel(key: key) }

            let provenance = coordinator.captureProvenance(key: key)
            #expect(provenance.source == source)
            #expect(provenance.repo == Self.repo)
        }

        @Test func captureProvenanceReconstructsSideloadKeyWithNilRepo() throws {
            let ws = makeTemporaryWorkspace()
            defer { removeWorkspace(ws) }

            let coordinator = DownloadCoordinator(ws: ws)
            let key = LocalStorage.modelRelativePath(repo: nil, filename: Self.filename)
            let provenance = coordinator.captureProvenance(key: key)
            #expect(provenance.repo == nil)
            #expect(provenance.filename == Self.filename)
        }

        /// `completedVersion` is what makes views re-scan.
        @Test func applyCompletionSuccessClearsEntryAndBumpsVersion() throws {
            let ws = makeTemporaryWorkspace()
            defer { removeWorkspace(ws) }

            let coordinator = DownloadCoordinator(ws: ws)
            let key = LocalStorage.modelRelativePath(repo: Self.repo, filename: Self.filename)
            let source = Model.ggufText(.init(source: .huggingFace(repo: try HfRepo.parseOrgRepo(Self.repo), path: try RepoSubpath(Self.filename), sha256: nil)))
            coordinator.startDownload(source, familyId: Self.familyId)
            try #require(coordinator.downloads[key] != nil)
            let versionBefore = coordinator.completedVersion

            coordinator.applyCompletion(key: key, result: .success(()))

            #expect(coordinator.downloads[key] == nil)
            #expect(coordinator.completedVersion == versionBefore + 1)
        }

        /// A `.failed` row stays visible, cancellable and detectable by the headless runner,
        /// as on the MLX path.
        @Test func applyCompletionFailureMarksRowFailed() throws {
            let ws = makeTemporaryWorkspace()
            defer { removeWorkspace(ws) }

            let coordinator = DownloadCoordinator(ws: ws)
            let key = LocalStorage.modelRelativePath(repo: Self.repo, filename: Self.filename)
            let source = Model.ggufText(.init(source: .huggingFace(repo: try HfRepo.parseOrgRepo(Self.repo), path: try RepoSubpath(Self.filename), sha256: nil)))
            coordinator.startDownload(source, familyId: Self.familyId)
            defer { coordinator.cancel(key: key) }

            coordinator.applyCompletion(key: key, result: .failure(URLError(.timedOut)))

            let isFailed: Bool = if case .failed = coordinator.downloads[key]?.state { true } else { false }
            #expect(isFailed)
        }

        // MARK: - The planned digest reaches the installer

        /// `sha256("abc")`, any valid digest: these pin the wiring, not the hashing.
        private static let digest = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"

        private func textSource(sha256: String?) throws -> Model {
            .ggufText(.init(source: .huggingFace(
                repo: try HfRepo.parseOrgRepo(Self.repo), path: try RepoSubpath(Self.filename),
                sha256: try sha256.map { try Sha256($0) })))
        }

        /// Without this the planner's digest could be silently dropped between the plan and
        /// the in-flight row, and every other test would still pass.
        @Test func ignitionPutsThePlannedDigestOnTheRow() throws {
            let ws = makeTemporaryWorkspace()
            defer { removeWorkspace(ws) }

            let coordinator = DownloadCoordinator(ws: ws)
            let key = LocalStorage.modelRelativePath(repo: Self.repo, filename: Self.filename)
            coordinator.startDownload(try textSource(sha256: Self.digest), familyId: Self.familyId)
            defer { coordinator.cancel(key: key) }

            #expect(coordinator.downloads[key]?.sha256?.value == Self.digest)
            #expect(coordinator.captureProvenance(key: key).sha256?.value == Self.digest)
        }

        @Test func ignitionCarriesNoDigestWhenTheSourceDeclaresNone() throws {
            let ws = makeTemporaryWorkspace()
            defer { removeWorkspace(ws) }

            let coordinator = DownloadCoordinator(ws: ws)
            let key = LocalStorage.modelRelativePath(repo: Self.repo, filename: Self.filename)
            coordinator.startDownload(try textSource(sha256: nil), familyId: Self.familyId)
            defer { coordinator.cancel(key: key) }

            #expect(coordinator.downloads[key]?.sha256 == nil)
        }

        @Test func eachVisionTransferCarriesItsOwnDigest() throws {
            let ws = makeTemporaryWorkspace()
            defer { removeWorkspace(ws) }
            let repo = "org/vl-GGUF"
            let mmprojDigest = "b5d4045c3f466fa91fe2cc6abe79232a1a57cdf104f7a26e716e0a1e2789df78"
            let source = Model.ggufVision(.init(source: .huggingFace(
                repo: try HfRepo.parseOrgRepo(repo),
                model: try RepoSubpath("vl-Q4_0.gguf"), modelSha256: try Sha256(Self.digest),
                mmproj: try RepoSubpath("mmproj-vl-F16.gguf"), mmprojSha256: try Sha256(mmprojDigest))))

            let coordinator = DownloadCoordinator(ws: ws)
            let modelKey = LocalStorage.modelRelativePath(repo: repo, filename: "vl-Q4_0.gguf")
            let mmprojKey = LocalStorage.modelRelativePath(repo: repo, filename: "mmproj-vl-F16.gguf")
            coordinator.startDownload(source)
            defer {
                coordinator.cancel(key: modelKey)
                coordinator.cancel(key: mmprojKey)
            }

            #expect(coordinator.downloads[modelKey]?.sha256?.value == Self.digest)
            #expect(coordinator.downloads[mmprojKey]?.sha256?.value == mmprojDigest)
        }

        /// A completion the system redelivers after relaunch has no in-memory row, so the
        /// persisted record is all that stands between those bytes and an unverified install.
        @Test func aPersistedRecordRestoresTheDigestOntoItsRow() throws {
            let ws = makeTemporaryWorkspace()
            defer { removeWorkspace(ws) }
            let key = LocalStorage.modelRelativePath(repo: Self.repo, filename: Self.filename)
            let source = try textSource(sha256: Self.digest)

            DownloadStore(dir: ws.downloadsDir).saveRecord(DownloadRecord(
                filename: Self.filename,
                urlString: "https://huggingface.co/\(Self.repo)/resolve/main/\(Self.filename)",
                repo: Self.repo, familyId: Self.familyId, source: source,
                sha256: try Sha256(Self.digest)))

            let coordinator = DownloadCoordinator(ws: ws)
            defer { coordinator.cancel(key: key) }

            #expect(coordinator.downloads[key]?.sha256?.value == Self.digest)
            #expect(coordinator.captureProvenance(key: key).sha256?.value == Self.digest)
        }
    }

    /// A finished install sweeps the store back under the cap without reclaiming what is still
    /// in flight. The oversize refusal is `ensureModel`'s: see `EnsureModelTests`.
    @Suite(.serialized) @MainActor struct QuotaGate {
        @Test func anMlxInstallSweepsAndKeepsWhatItJustInstalled() async throws {
            let ws = makeTemporaryWorkspace()
            defer { removeWorkspace(ws) }
            let stale = try installEntry(ws, try ggufTextSpec("org/stale-GGUF", "stale-Q4_0.gguf"),
                                         payloadBytes: 64_000,
                                         lastUsedAt: Date(timeIntervalSince1970: 1_600_000_000))
            try ws.setStorageQuotaBytes(entrySizeBytes(at: stale))
            let coord = DownloadCoordinator(ws: ws)
            coord.transport = CompleteMLXDownloader(payloadBytes: 64_000)
            let ref = try HfModelRef.parse(repo: "org/fresh-MLX-4bit")

            coord.startDirectoryDownload(.mlx(ref.asMlx()), ref: ref)
            try await waitUntil(coord, ref.key) { $0 == nil }

            let installed = try #require(ws.models.entryDir(for: .mlx(ref.asMlx())))
            #expect(FileManager.default.fileExists(atPath: installed.path))
            #expect(!FileManager.default.fileExists(atPath: stale.path), "the stale entry should be evicted")
        }

        /// A vision model's projector can land first and publish the shared entry before its
        /// weights do.
        @Test func anInFlightVisionTransferPinsTheSharedEntry() throws {
            let ws = makeTemporaryWorkspace()
            defer { removeWorkspace(ws) }
            let coord = DownloadCoordinator(ws: ws)
            let spec = try ggufVisionSpec("org/vl-GGUF", "vl-Q4_0.gguf", "mmproj-vl-F16.gguf")
            coord.startDownload(spec)
            defer {
                for filename in ["vl-Q4_0.gguf", "mmproj-vl-F16.gguf"] {
                    coord.cancel(key: LocalStorage.modelRelativePath(repo: "org/vl-GGUF", filename: filename))
                }
            }

            // A manifest but no weights yet: a manifest-less entry would be garbage anyway.
            let downloaded = FileManager.default.temporaryDirectory
                .appendingPathComponent("mmproj-\(UUID().uuidString).gguf")
            try Data(repeating: 0x6D, count: 64_000).write(to: downloaded)
            try GGUFModelInstaller(repo: "org/vl-GGUF", filename: "mmproj-vl-F16.gguf",
                                   source: spec, ws: ws).install(from: downloaded)

            let pins = coord.sweepPins(justInstalled: nil)
            #expect(pins.models.contains(try #require(ModelStorageKey.of(spec))))

            try ws.setStorageQuotaBytes(1)
            #expect(ws.sweepToQuota(pinning: pins).removed.isEmpty)
            let entry = try #require(ws.models.entryDir(for: spec))
            #expect((try? ModelArtifactStore.readManifest(at: ModelManifest.manifestURL(inEntryDir: entry))) != nil)
        }

        /// The sweep picks jobs by a peek at their status before decoding any.
        @Test func pinsAreTheModelsOfRunningAndPausedJobsOnly() throws {
            let ws = makeTemporaryWorkspace()
            defer { removeWorkspace(ws) }
            let specs = try ["running", "paused", "completed"].map {
                try ggufTextSpec("org/\($0)-GGUF", "\($0)-Q4_0.gguf")
            }
            for (spec, status) in zip(specs, [JobStatus.running, .paused, .completed]) {
                ws.saveJobManifest(JobManifest(
                    jobId: JobId("job-\(status.rawValue)"), createdAt: "2026-10-01T10:00:00Z",
                    nGpuLayers: 99, contextSize: 4096,
                    cells: [cell(id: CellId("cell-\(status.rawValue)"), benchmarkId: "prefill_throughput_512", source: spec)],
                    status: status))
            }

            let pins = DownloadCoordinator(ws: ws).sweepPins(justInstalled: nil)

            #expect(pins.models == Set(specs.prefix(2).compactMap { ModelStorageKey.of($0) }))
        }

    }

    /// Which `URLSession` a download runs on. A wrong pick is invisible until a cell fails on
    /// "stopped making progress" (see `DownloadCoordinator.sessionConfiguration`).
    @Suite("download session kind")
    struct SessionKind {
        @Test("a one-shot headless cell downloads on a foreground session", arguments: [
            ["Pipette", "headlessrun", "benchmarks", "run", "benchmark=decode_throughput_256"],
            ["Pipette", "headlessrun", "bench", "benchmarks=decode_throughput_256"],
            ["Pipette", "headlessrun", "models", "pull", "model={}"],
            ["Pipette", "headlessrun", "sync"],
        ])
        func oneShotUsesForeground(_ argv: [String]) {
            #expect(DownloadCoordinator.usesForegroundSession(argv))
        }

        /// `settings run` leaves the app resident and suspendable between claims, so its
        /// transfers need a background session to outlive suspension.
        @Test("the resident planner worker keeps the background session")
        func plannerWorkerStaysBackground() {
            #expect(!DownloadCoordinator.usesForegroundSession(
                ["Pipette", "headlessrun", "settings", "run"]))
        }

        @Test("only settings-run is excluded, not every verb containing run")
        func excludesOnlyTheWorkerVerb() {
            #expect(DownloadCoordinator.usesForegroundSession(
                ["Pipette", "headlessrun", "settings", "show"]))
            #expect(DownloadCoordinator.usesForegroundSession(
                ["Pipette", "headlessrun", "benchmarks", "run"]))
        }

        /// Someone holding the phone may background it mid-download.
        @Test("an interactive launch keeps the background session")
        func interactiveStaysBackground() {
            #expect(!DownloadCoordinator.usesForegroundSession(["Pipette"]))
            #expect(!DownloadCoordinator.usesForegroundSession(["Pipette", "bench", "benchmarks=x"]))
        }

        @Test("the foreground configuration is not a background session")
        func foregroundConfigurationHasNoIdentifier() {
            #expect(DownloadCoordinator.sessionConfiguration(foreground: true).identifier == nil)
            #expect(DownloadCoordinator.sessionConfiguration(foreground: false).identifier != nil)
        }

        @Test("launch events are requested only for the background session")
        func launchEventsOnlyWhenBackground() {
            #expect(!DownloadCoordinator.sessionConfiguration(foreground: true).sessionSendsLaunchEvents)
            #expect(DownloadCoordinator.sessionConfiguration(foreground: false).sessionSendsLaunchEvents)
        }

        @Test("shared settings survive the branch", arguments: [true, false])
        func sharedSettingsApplyToBoth(_ foreground: Bool) {
            let config = DownloadCoordinator.sessionConfiguration(foreground: foreground)
            #expect(config.allowsCellularAccess)
            #expect(config.waitsForConnectivity)
        }
    }

    @Suite(.serialized) @MainActor struct MlxFlow {
        private static let completeModel = ["config.json", "model.safetensors", "tokenizer.json"]

        @Test func downloadsInstallsAndDiscoversRootModel() async throws {
            let ws = makeTemporaryWorkspace()
            defer { removeWorkspace(ws) }
            let ref = try HfModelRef.parse(repo: "LiquidAI/LFM2.5-350M-MLX-4bit")
            let coord = DownloadCoordinator(ws: ws)
            coord.transport = FakeDownloader(outcome: .success(Self.completeModel), fractions: [0.5, 1.0])

            coord.startDirectoryDownload(.mlx(ref.asMlx()), ref: ref, familyId: "lfm2.5-350m")
            try await waitUntil(coord, ref.key) { $0 == nil }

            let model = try #require(ws.availableModels().first { m in
                guard case .mlx = m.source else { return false }
                return m.hfRepo == ref.repo.description
            })
            #expect(model.name == "LFM2.5-350M-MLX-4bit")
            #expect(model.familyId == "lfm2.5-350m")
        }

        @Test func installsSubpathModel() async throws {
            let ws = makeTemporaryWorkspace()
            defer { removeWorkspace(ws) }
            let ref = try HfModelRef.parse(repo: "org/multi", subpath: "variant-4bit")
            let coord = DownloadCoordinator(ws: ws)
            coord.transport = FakeDownloader(outcome: .success(Self.completeModel))

            coord.startDirectoryDownload(.mlx(ref.asMlx()), ref: ref)
            try await waitUntil(coord, ref.key) { $0 == nil }

            let model = try #require(ws.availableModels().first { m in
                guard case .mlx = m.source else { return false }
                return m.hfRepo == ref.repo.description
            })
            #expect(model.name == "variant-4bit")
        }

        /// Anonymous pulls get a 401 on a gated repo and HF's per-IP rate limit on a public one.
        @Test func mlxDownloadPresentsTheCoordinateAuthToken() async throws {
            let ws = makeTemporaryWorkspace()
            defer { removeWorkspace(ws) }
            var repo = try HfRepo.parseOrgRepo("LiquidAI/LFM2.5-350M-MLX-4bit")
            repo.authToken = try AuthToken("hf_plan_token")
            let ref = HfModelRef(repo: repo, subpath: nil)
            let downloader = TokenCapturingDownloader(files: Self.completeModel)
            let coord = DownloadCoordinator(ws: ws)
            coord.transport = downloader

            coord.startDirectoryDownload(.mlx(ref.asMlx()), ref: ref)
            try await waitUntil(coord, ref.key) { $0 == nil }

            #expect(downloader.capturedToken?.value == "hf_plan_token")
        }

        /// Deterministic in a Simulator host: it cannot store Keychain items, so the two
        /// Keychain fallback tiers of `resolveHfToken` are empty.
        @Test func mlxDownloadWithNoCoordinateTokenFetchesAnonymously() async throws {
            let ws = makeTemporaryWorkspace()
            defer { removeWorkspace(ws) }
            let ref = try HfModelRef.parse(repo: "LiquidAI/LFM2.5-350M-MLX-4bit")
            let downloader = TokenCapturingDownloader(files: Self.completeModel)
            let coord = DownloadCoordinator(ws: ws)
            coord.transport = downloader

            coord.startDirectoryDownload(.mlx(ref.asMlx()), ref: ref)
            try await waitUntil(coord, ref.key) { $0 == nil }

            #expect(downloader.capturedToken == nil)
        }

        /// An MLX download's `filename` is the repo leaf, so a K-quant parse cannot name it, and
        /// this slug encodes no bit-width either: only the catalog's `quant` is left.
        @Test func inFlightMlxDownloadRowUsesExplicitCatalogQuant() throws {
            let ws = makeTemporaryWorkspace()
            defer { removeWorkspace(ws) }
            let ref = try HfModelRef.parse(repo: "org/custom-mlx")
            let coord = DownloadCoordinator(ws: ws)
            coord.transport = HangingDownloader()

            coord.startDirectoryDownload(.mlx(ref.asMlx()), ref: ref, quant: "4bit")
            defer { coord.cancel(key: ref.key) }

            let dl = try #require(coord.downloads[ref.key])
            #expect(dl.source?.quant == nil, "precondition: this repo slug doesn't encode a bit-width")
            #expect(dl.quant == "4bit", "row falls back to the explicit catalog quant, not \"unknown\"")
        }

        @Test func incompleteDownloadFailsAndRecordsNothing() async throws {
            let ws = makeTemporaryWorkspace()
            defer { removeWorkspace(ws) }
            let ref = try HfModelRef.parse(repo: "org/empty")
            let coord = DownloadCoordinator(ws: ws)
            coord.transport = FakeDownloader(outcome: .success([]))

            coord.startDirectoryDownload(.mlx(ref.asMlx()), ref: ref)
            try await waitUntil(coord, ref.key) { isFailed($0) }
            #expect(ws.availableModels().first { $0.hfRepo == ref.repo.description } == nil)
        }

        @Test func transportErrorFailsTheCell() async throws {
            let ws = makeTemporaryWorkspace()
            defer { removeWorkspace(ws) }
            let ref = try HfModelRef.parse(repo: "org/boom")
            let coord = DownloadCoordinator(ws: ws)
            coord.transport = FakeDownloader(outcome: .failure(.transport("network")))

            coord.startDirectoryDownload(.mlx(ref.asMlx()), ref: ref)
            try await waitUntil(coord, ref.key) { isFailed($0) }
            guard case .failed(let reason)? = coord.downloads[ref.key]?.state else {
                Issue.record("expected a .failed state with a reason"); return
            }
            #expect(reason.hasPrefix("Failed:"))
        }

        @Test func cancelStopsInFlightDownload() async throws {
            let ws = makeTemporaryWorkspace()
            defer { removeWorkspace(ws) }
            let ref = try HfModelRef.parse(repo: "org/slow")
            let coord = DownloadCoordinator(ws: ws)
            coord.transport = HangingDownloader()

            coord.startDirectoryDownload(.mlx(ref.asMlx()), ref: ref)
            try #require(coord.downloads[ref.key] != nil, "startDirectoryDownload should register an in-flight entry")
            coord.cancel(key: ref.key)
            #expect(coord.downloads[ref.key]?.state == .cancelling)
            try await waitUntil(coord, ref.key) { $0 == nil }
        }

        @Test func mlxDownloadConnectingThenDownloading() async throws {
            let ws = makeTemporaryWorkspace()
            defer { removeWorkspace(ws) }
            let ref = try HfModelRef.parse(repo: "org/progress")
            let coord = DownloadCoordinator(ws: ws)
            coord.transport = ProgressThenHangDownloader(fraction: 0.5)
            defer { coord.cancel(key: ref.key) }

            coord.startDirectoryDownload(.mlx(ref.asMlx()), ref: ref)
            // Synchronously, before the download Task's first progress hop.
            #expect(coord.downloads[ref.key]?.state == .connecting)

            try await waitUntil(coord, ref.key) { $0?.state == .downloading }
            #expect(coord.downloads[ref.key]?.state == .downloading)
        }

        /// Both states are set synchronously, so the test does not depend on the transfer
        /// making progress.
        @Test func fileDownloadConnectingThenPausing() throws {
            let ws = makeTemporaryWorkspace()
            defer { removeWorkspace(ws) }
            let coord = DownloadCoordinator(ws: ws)
            let repo = try HfRepo.parseOrgRepo("org/gguf")
            let filename = try RepoSubpath("model.Q4_0.gguf")
            let key = LocalStorage.modelRelativePath(repo: repo.description, filename: filename.value)
            defer { coord.cancel(key: key) }

            coord.startDownload(.ggufText(GgufText(source: .huggingFace(repo: repo, path: filename, sha256: nil))))
            #expect(coord.downloads[key]?.state == .connecting)

            coord.pause(key: key)
            #expect(coord.downloads[key]?.state == .pausing)
        }

        @Test func concurrencyGateQueuesSurplusAndPromotesOnSlotFree() async throws {
            let ws = makeTemporaryWorkspace()
            defer { removeWorkspace(ws) }
            let coord = DownloadCoordinator(ws: ws)
            coord.transport = HangingDownloader()

            let cap = DownloadCoordinator.maxConcurrentDownloads
            let refs = try (0...cap).map { try HfModelRef.parse(repo: "org/model-\($0)") }
            for ref in refs { coord.startDirectoryDownload(.mlx(ref.asMlx()), ref: ref) }
            defer { for ref in refs { coord.cancel(key: ref.key) } }

            let active = refs.filter { coord.downloads[$0.key]?.state == .connecting }
            let queued = refs.filter { coord.downloads[$0.key]?.state == .queued }
            #expect(active.count == cap, "only the cap should be in flight at once")
            #expect(queued.count == 1, "the surplus download should be queued")

            coord.cancel(key: active[0].key)
            try await waitUntil(coord, active[0].key) { $0 == nil }
            #expect(queued.allSatisfy { coord.downloads[$0.key]?.state == .connecting },
                    "a freed slot promotes the queued download into flight")
        }

        /// Pausing would flip it to `.paused` while its start closure lingers in
        /// `pendingStarts`, so `pumpQueue` could later resurrect it.
        @Test func pausingAQueuedDownloadIsIgnored() throws {
            let ws = makeTemporaryWorkspace()
            defer { removeWorkspace(ws) }
            let coord = DownloadCoordinator(ws: ws)
            let cap = DownloadCoordinator.maxConcurrentDownloads

            let coordinates = try (0...cap).map { i in
                (repo: try HfRepo.parseOrgRepo("org/gguf-\(i)"), path: try RepoSubpath("model.Q4_0.gguf"))
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

        /// Otherwise it could spawn a duplicate transfer task alongside the running one.
        @Test func resumeIsIgnoredUnlessPaused() throws {
            let ws = makeTemporaryWorkspace()
            defer { removeWorkspace(ws) }
            let coord = DownloadCoordinator(ws: ws)
            let repo = try HfRepo.parseOrgRepo("org/gguf")
            let filename = try RepoSubpath("model.Q4_0.gguf")
            let key = LocalStorage.modelRelativePath(repo: repo.description, filename: filename.value)
            defer { coord.cancel(key: key) }

            coord.startDownload(.ggufText(GgufText(source: .huggingFace(repo: repo, path: filename, sha256: nil))))
            #expect(coord.downloads[key]?.state == .connecting)
            coord.resume(key: key)
            #expect(coord.downloads[key]?.state == .connecting,
                    "resume on a non-paused download is ignored")
        }

    }

    /// Launch-time ownership transfer for grants persisted by the previous process.
    @Suite(.serialized) @MainActor struct ReservationRestoration {
        private let repo = "org/a-GGUF"
        private let filename = "a-Q4_0.gguf"

        private func seedRecord(
            _ ws: PipetteWorkspace, source: Model, paused: Bool = false
        ) {
            let key = LocalStorage.modelRelativePath(repo: repo, filename: filename)
            DownloadStore(dir: ws.downloadsDir).saveRecord(DownloadRecord(
                filename: filename,
                urlString: "https://huggingface.co/\(repo)/resolve/main/\(filename)",
                repo: repo,
                familyId: nil,
                source: source
            ))
            if paused {
                DownloadStore(dir: ws.downloadsDir).saveResumeData(
                    Data("placeholder resume data".utf8), for: key)
            }
        }

        private func seedGrant(_ ws: PipetteWorkspace, source: Model) async throws {
            let key = try #require(ModelStorageKey.of(source))
            let admission = ModelAdmission(ws: ws, freeBytes: { _ in nil })
            _ = try await admission.admit(
                AdmissionRequest(key: key, label: source.description, declaredBytes: 4096),
                pins: SweepPins())
        }

        private func makeCoordinator(_ ws: PipetteWorkspace) -> DownloadCoordinator {
            DownloadCoordinator(
                ws: ws,
                downloadHttpClient: UnreachableURLProtocol.httpClient,
                foregroundSession: true,
                reachability: FixedReachability(isConnected: true),
                sessionConfiguration: .ephemeral)
        }

        private func waitForOutstanding(
            _ expected: Int64, _ coordinator: DownloadCoordinator
        ) async throws {
            for _ in 0..<250 where await coordinator.admission.outstandingBytes != expected {
                try await Task.sleep(for: .milliseconds(20))
            }
            #expect(await coordinator.admission.outstandingBytes == expected)
        }

        /// The restored row is provisional until URLSession reports its surviving tasks; the
        /// row has to go before the grant is reconciled, or the grant leaks.
        @Test func aStaleRestoredRowDoesNotKeepItsGrant() async throws {
            let ws = makeTemporaryWorkspace()
            defer { removeWorkspace(ws) }
            let source = try ggufTextSpec(repo, filename)
            let key = LocalStorage.modelRelativePath(repo: repo, filename: filename)
            seedRecord(ws, source: source)
            try await seedGrant(ws, source: source)

            let coordinator = makeCoordinator(ws)

            try await waitUntil(coordinator, key) { $0 == nil }
            try await waitForOutstanding(0, coordinator)
            #expect(DownloadStore(dir: ws.downloadsDir).allRecords().isEmpty)
        }

        /// Otherwise the completion finds an empty reservation map and leaves the ledger
        /// charged for good.
        @Test func aRestoredWriterCompletionReleasesItsGrant() async throws {
            let ws = makeTemporaryWorkspace()
            defer { removeWorkspace(ws) }
            let source = try ggufTextSpec(repo, filename)
            let entry = try #require(ModelStorageKey.of(source))
            let key = LocalStorage.modelRelativePath(repo: repo, filename: filename)
            seedRecord(ws, source: source, paused: true)
            try await seedGrant(ws, source: source)

            let coordinator = makeCoordinator(ws)
            for _ in 0..<250 where !coordinator.hasReservation(for: entry) {
                try await Task.sleep(for: .milliseconds(20))
            }
            #expect(coordinator.hasReservation(for: entry))
            #expect(await coordinator.admission.outstandingBytes == 4096)

            coordinator.applyCompletion(key: key, result: .success(()))

            try await waitForOutstanding(0, coordinator)
        }
    }

    /// Recovery from a transfer paused before the app relaunched, and where it stops.
    ///
    /// Nothing in a restored pause says what paused it. A caller needing the weights lifts it
    /// rather than reporting it as terminal, or the device stays stuck on that model until a
    /// person opens the app. Two pauses stand: one a person raised during the wait, and one
    /// with no network to resume onto. A resume the server will not honor is restarted from
    /// scratch once.
    @Suite(.serialized) @MainActor struct PauseRecovery {

        /// On its own the record restores as `.connecting`, which the reconcile sweep drops.
        private func seedRecord(
            _ ws: PipetteWorkspace, repo: String, filename: String, source: Model,
            etag: String? = nil
        ) {
            DownloadStore(dir: ws.downloadsDir).saveRecord(DownloadRecord(
                filename: filename,
                urlString: "https://huggingface.co/\(repo)/resolve/main/\(filename)",
                repo: repo,
                familyId: nil,
                source: source,
                etag: etag
            ))
        }

        /// The resume blob's presence is the paused state; `restoreState` only checks the file
        /// exists, so a placeholder serves. The ETag lets `performResume` spend the blob rather
        /// than start from zero.
        private func seedPausedTransfer(
            _ ws: PipetteWorkspace, repo: String, filename: String, source: Model
        ) {
            seedRecord(ws, repo: repo, filename: filename, source: source,
                       etag: "\"seeded\"")
            DownloadStore(dir: ws.downloadsDir).saveResumeData(
                Data("placeholder resume data".utf8),
                for: LocalStorage.modelRelativePath(repo: repo, filename: filename))
        }

        /// `foregroundSession` is forced: the test runner's arguments name no headless verb, so
        /// the default would use the process-global background session. `reachability` is
        /// fixed because the shared monitor reports offline until the app's launch path starts
        /// it.
        private func makeCoordinator(
            _ ws: PipetteWorkspace, online: Bool = true,
            reachability: NetworkReachable? = nil,
            sessionConfiguration: URLSessionConfiguration? = nil
        ) -> DownloadCoordinator {
            DownloadCoordinator(ws: ws,
                                downloadHttpClient: UnreachableURLProtocol.httpClient,
                                foregroundSession: true,
                                reachability: reachability ?? FixedReachability(isConnected: online),
                                sessionConfiguration: sessionConfiguration,
                                retryHooks: RetryHooks(sleep: { _ in }))
        }

        @Test func aPersistedResumeBlobRestoresAsPaused() async throws {
            let ws = makeTemporaryWorkspace()
            defer { removeWorkspace(ws) }
            let spec = try ggufTextSpec("org/a-GGUF", "a-Q4_0.gguf")
            seedPausedTransfer(ws, repo: "org/a-GGUF", filename: "a-Q4_0.gguf", source: spec)

            let coordinator = makeCoordinator(ws)

            #expect(coordinator.hasTransfer(for: spec))
            #expect(coordinator.isPaused(spec))
        }

        @Test func resumePausedAdvancesARestoredPause() async throws {
            let ws = makeTemporaryWorkspace()
            defer { removeWorkspace(ws) }
            let spec = try ggufTextSpec("org/a-GGUF", "a-Q4_0.gguf")
            seedPausedTransfer(ws, repo: "org/a-GGUF", filename: "a-Q4_0.gguf", source: spec)
            let coordinator = makeCoordinator(ws)

            await coordinator.resumePaused(for: spec)

            #expect(!coordinator.isPaused(spec))
            // Resumed, not discarded, so the fetch has something to wait on.
            #expect(coordinator.hasTransfer(for: spec))
        }

        /// A transfer conjured for a model never enqueued is one a waiting fetch would poll.
        @Test func resumePausedIsANoOpWhenNothingIsPaused() async throws {
            let ws = makeTemporaryWorkspace()
            defer { removeWorkspace(ws) }
            let spec = try ggufTextSpec("org/absent-GGUF", "absent-Q4_0.gguf")
            let coordinator = makeCoordinator(ws)

            await coordinator.resumePaused(for: spec)

            #expect(!coordinator.hasTransfer(for: spec))
        }

        /// A launch reaches the resume before the path monitor's first reading lands; taking the
        /// pessimistic default would leave the pause standing on an online device.
        @Test func aResumeWaitsForTheMonitorsFirstReadingRatherThanCallingItOffline() async throws {
            let ws = makeTemporaryWorkspace()
            defer { removeWorkspace(ws) }
            let spec = try ggufTextSpec("org/a-GGUF", "a-Q4_0.gguf")
            seedPausedTransfer(ws, repo: "org/a-GGUF", filename: "a-Q4_0.gguf", source: spec)
            // Long enough that the resume sees no reading yet, well inside its one-second wait.
            let coordinator = makeCoordinator(
                ws, reachability: WarmingReachability(warmsAfter: .milliseconds(200)))

            await coordinator.resumePaused(for: spec)

            #expect(!coordinator.isPaused(spec),
                    "the resume waited for the reading, found the device online, and took")
        }

        /// A pause raised once the wait is under way is a person's (`ModelsView` offers Pause on
        /// an in-flight row), so the fetch must not overrule it.
        @Test func aPauseRaisedDuringTheWaitIsReportedNotResumed() async throws {
            let ws = makeTemporaryWorkspace()
            defer { removeWorkspace(ws) }
            let spec = try ggufTextSpec("org/a-GGUF", "a-Q4_0.gguf")
            let coordinator = makeCoordinator(
                ws, sessionConfiguration: HangingURLProtocol.configuration)
            let key = LocalStorage.modelRelativePath(repo: "org/a-GGUF", filename: "a-Q4_0.gguf")

            // A wait that resumed this pause would run to the long deadline and fail the test.
            // Sized explicitly: admission refuses an unsized artifact, and the stub session
            // answers no HEAD.
            let fetch = Task {
                try await ensureModel(spec, ws: ws, coordinator: coordinator,
                                      declaredSizeBytes: 4096, stallTimeout: .seconds(600))
            }
            try await waitUntil(coordinator, key) { $0?.state == .connecting }
            coordinator.pause(key: key)

            let failure = await #expect(throws: ModelStoreError.self) { _ = try await fetch.value }
            guard case .fetchPaused = try #require(failure) else {
                Issue.record("expected a fetchPaused, got \(String(describing: failure))"); return
            }
            #expect(coordinator.isPaused(spec), "the pause stands; the wait left it alone")
        }

        /// Resuming offline would spend the blob on a task `waitsForConnectivity` parks, trading
        /// a recoverable multi-gigabyte partial for a wait that runs out the caller's window.
        @Test func offlineAResumeIsDeclinedAndTheResumeBlobKept() async throws {
            let ws = makeTemporaryWorkspace()
            defer { removeWorkspace(ws) }
            let spec = try ggufTextSpec("org/a-GGUF", "a-Q4_0.gguf")
            seedPausedTransfer(ws, repo: "org/a-GGUF", filename: "a-Q4_0.gguf", source: spec)
            let coordinator = makeCoordinator(ws, online: false)

            await coordinator.resumePaused(for: spec)

            #expect(coordinator.isPaused(spec))
            let store = DownloadStore(dir: ws.downloadsDir)
            #expect(store.hasResumeData(
                for: LocalStorage.modelRelativePath(repo: "org/a-GGUF", filename: "a-Q4_0.gguf")))
            #expect(store.allRecords().count == 1)
        }

        /// The fetch reports the pause in seconds rather than holding the device to its
        /// deadline. Only a resume that was attempted and failed clears the state.
        @Test func offlineAFetchReportsThePauseAndLeavesTheStateIntact() async throws {
            let ws = makeTemporaryWorkspace()
            defer { removeWorkspace(ws) }
            let spec = try ggufTextSpec("org/a-GGUF", "a-Q4_0.gguf")
            seedPausedTransfer(ws, repo: "org/a-GGUF", filename: "a-Q4_0.gguf", source: spec)
            let coordinator = makeCoordinator(ws, online: false)

            let failure = await #expect(throws: ModelStoreError.self) {
                _ = try await ensureModel(spec, ws: ws, coordinator: coordinator,
                                          stallTimeout: .seconds(30))
            }
            guard case .fetchPaused = try #require(failure) else {
                Issue.record("expected a fetchPaused, got \(String(describing: failure))"); return
            }

            let store = DownloadStore(dir: ws.downloadsDir)
            #expect(store.allRecords().count == 1)
            #expect(store.hasResumeData(
                for: LocalStorage.modelRelativePath(repo: "org/a-GGUF", filename: "a-Q4_0.gguf")))
        }

        /// The placeholder blob names no real partial, the shape of a blob the server will not
        /// carry on from. A phone driven from the CLI has nobody to clear that state by hand.
        @Test func aResumeThatCannotTakeIsRestartedFromScratchWithinTheSameRun() async throws {
            let ws = makeTemporaryWorkspace()
            defer { removeWorkspace(ws) }
            let spec = try ggufTextSpec("org/a-GGUF", "a-Q4_0.gguf")
            seedPausedTransfer(ws, repo: "org/a-GGUF", filename: "a-Q4_0.gguf", source: spec)
            ScriptedURLProtocol.serve(.body(Data("weights".utf8)))
            let coordinator = makeCoordinator(
                ws, sessionConfiguration: ScriptedURLProtocol.configuration)

            let bound = try await ensureModel(spec, ws: ws, coordinator: coordinator,
                                              stallTimeout: .seconds(30))

            #expect(bound.boundPaths != nil, "the fetch answers with launchable paths")
            // The spent resume, then exactly one fresh start, not a retry loop.
            #expect(ScriptedURLProtocol.requestCount == 2)
            #expect(!coordinator.hasTransfer(for: spec))
            let store = DownloadStore(dir: ws.downloadsDir)
            #expect(store.allRecords().isEmpty)
            #expect(!store.hasResumeData(
                for: LocalStorage.modelRelativePath(repo: "org/a-GGUF", filename: "a-Q4_0.gguf")))
        }

        /// The coordinate is the same every time, so re-igniting would only hold the device to
        /// its deadline to reach the answer it already had.
        @Test func aRestartThatAlsoFailsIsReportedRatherThanIgnitedAgain() async throws {
            let ws = makeTemporaryWorkspace()
            defer { removeWorkspace(ws) }
            let spec = try ggufTextSpec("org/a-GGUF", "a-Q4_0.gguf")
            seedPausedTransfer(ws, repo: "org/a-GGUF", filename: "a-Q4_0.gguf", source: spec)
            ScriptedURLProtocol.serve(.failure(.badServerResponse))
            let coordinator = makeCoordinator(
                ws, sessionConfiguration: ScriptedURLProtocol.configuration)

            let failure = await #expect(throws: ModelStoreError.self) {
                _ = try await ensureModel(spec, ws: ws, coordinator: coordinator,
                                          stallTimeout: .seconds(30))
            }
            guard case .fetch = try #require(failure) else {
                Issue.record("expected a fetch, got \(String(describing: failure))"); return
            }
            // A fetch that kept re-igniting would count higher and answer `.fetchTimedOut`.
            #expect(ScriptedURLProtocol.requestCount == 2)

            let store = DownloadStore(dir: ws.downloadsDir)
            #expect(store.allRecords().isEmpty)
            #expect(!store.hasResumeData(
                for: LocalStorage.modelRelativePath(repo: "org/a-GGUF", filename: "a-Q4_0.gguf")))
        }

        /// `interruptedByNetwork` marks a drop rather than a person, so the wait lifts it once
        /// more. Never restarted from scratch: a re-paused row is a partial worth keeping. The
        /// second drop ends the fetch.
        @Test func aResumeUndoneByANetworkDropIsLiftedOnceMoreThenReported() async throws {
            let ws = makeTemporaryWorkspace()
            defer { removeWorkspace(ws) }
            let spec = try ggufTextSpec("org/a-GGUF", "a-Q4_0.gguf")
            seedPausedTransfer(ws, repo: "org/a-GGUF", filename: "a-Q4_0.gguf", source: spec)
            // An offline code routes through `applyNetworkInterruption`, the only thing that
            // re-pauses a resumed row; a drop while online retries instead.
            ScriptedURLProtocol.serve(.failure(.notConnectedToInternet))
            let coordinator = makeCoordinator(
                ws, sessionConfiguration: ScriptedURLProtocol.configuration)

            let failure = await #expect(throws: ModelStoreError.self) {
                _ = try await ensureModel(spec, ws: ws, coordinator: coordinator,
                                          stallTimeout: .seconds(30))
            }
            guard case .fetchPaused = try #require(failure) else {
                Issue.record("expected a fetchPaused, got \(String(describing: failure))"); return
            }
            // The resume the wait inherited, and the one it spent lifting the drop.
            #expect(ScriptedURLProtocol.requestCount == 2)
            #expect(coordinator.isPaused(spec))
            #expect(DownloadStore(dir: ws.downloadsDir).allRecords().count == 1)
        }

        /// The transfer a restart would ignite parks on `waitsForConnectivity` until the
        /// caller's window closes.
        @Test func offlineARestartIsDeclinedAndThePartialKept() async throws {
            let ws = makeTemporaryWorkspace()
            defer { removeWorkspace(ws) }
            let spec = try ggufTextSpec("org/a-GGUF", "a-Q4_0.gguf")
            seedPausedTransfer(ws, repo: "org/a-GGUF", filename: "a-Q4_0.gguf", source: spec)
            let coordinator = makeCoordinator(ws, online: false)

            let restarted = await coordinator.restartStalled(for: spec)
            #expect(!restarted)

            #expect(coordinator.isPaused(spec))
            let store = DownloadStore(dir: ws.downloadsDir)
            #expect(store.allRecords().count == 1)
            #expect(store.hasResumeData(
                for: LocalStorage.modelRelativePath(repo: "org/a-GGUF", filename: "a-Q4_0.gguf")))
        }

        /// `startDownload` dedups on existing rows, so a restart that did not cancel first would
        /// enqueue nothing and leave the caller waiting on the dead row to its deadline.
        @Test func aRestartCancelsTheFailedTransferAndIgnitesTheSameCoordinate() async throws {
            let ws = makeTemporaryWorkspace()
            defer { removeWorkspace(ws) }
            let spec = try ggufTextSpec("org/a-GGUF", "a-Q4_0.gguf")
            ScriptedURLProtocol.serve(.failure(.badServerResponse))
            let coordinator = makeCoordinator(
                ws, sessionConfiguration: ScriptedURLProtocol.configuration)
            let key = LocalStorage.modelRelativePath(repo: "org/a-GGUF", filename: "a-Q4_0.gguf")

            // Reached through the transport rather than seeded: a row that stopped and holds
            // nothing.
            coordinator.startDownload(spec)
            try await waitUntil(coordinator, key) { if case .failed = $0?.state { true } else { false } }

            ScriptedURLProtocol.serve(.body(Data("weights".utf8)))
            let restarted = await coordinator.restartStalled(for: spec)
            #expect(restarted)

            #expect(coordinator.downloads.count == 1, "the dead row was replaced, not joined")
            try await waitUntil(coordinator, key) { $0 == nil }
            #expect(ws.models.bound(spec) != nil, "the restarted transfer installed")
        }

        /// A paused row holds a partial, and whether it may be lifted is the pause arm's call.
        /// Sweeping the whole model would throw away gigabytes and could overrule a person.
        @Test func aRestartLeavesAPausedSiblingAndItsPartialAlone() async throws {
            let ws = makeTemporaryWorkspace()
            defer { removeWorkspace(ws) }
            let spec = try ggufVisionSpec("org/v-GGUF", "v-Q4_0.gguf", "mmproj-v.gguf")
            seedPausedTransfer(ws, repo: "org/v-GGUF", filename: "mmproj-v.gguf", source: spec)
            ScriptedURLProtocol.serve(.failure(.badServerResponse))
            let coordinator = makeCoordinator(
                ws, sessionConfiguration: ScriptedURLProtocol.configuration)
            let weights = LocalStorage.modelRelativePath(repo: "org/v-GGUF", filename: "v-Q4_0.gguf")
            let projector = LocalStorage.modelRelativePath(repo: "org/v-GGUF", filename: "mmproj-v.gguf")

            coordinator.startDownload(spec)
            try await waitUntil(coordinator, weights) {
                if case .failed = $0?.state { true } else { false }
            }

            ScriptedURLProtocol.serve(.body(Data("weights".utf8)))
            let restarted = await coordinator.restartStalled(for: spec)
            #expect(restarted)

            try await waitUntil(coordinator, weights) { $0 == nil }
            let store = DownloadStore(dir: ws.downloadsDir)
            #expect(coordinator.downloads[projector]?.state == .paused, "the pause nobody lifted")
            #expect(store.hasResumeData(for: projector), "and the partial it is holding")
        }

        /// The resume reaches the paused weight file while the mmproj row is still
        /// `.connecting`. `.fetch` rather than `.fetchPaused` is the claim: a terminal pause
        /// would mean no resume was tried, which a headless run cannot get itself out of.
        @Test func aRestoredPauseIsResumedWhileTheModelsOtherFileIsUnpaused() async throws {
            let ws = makeTemporaryWorkspace()
            defer { removeWorkspace(ws) }
            let spec = try ggufVisionSpec("org/v-GGUF", "v-Q4_0.gguf", "mmproj-v.gguf")
            seedPausedTransfer(ws, repo: "org/v-GGUF", filename: "v-Q4_0.gguf", source: spec)
            seedRecord(ws, repo: "org/v-GGUF", filename: "mmproj-v.gguf", source: spec)
            ScriptedURLProtocol.serve(.failure(.badServerResponse))
            let coordinator = makeCoordinator(
                ws, sessionConfiguration: ScriptedURLProtocol.configuration)

            // One row paused, one not: a rule asking whether the model is paused lifts nothing.
            #expect(coordinator.hasTransfer(for: spec))
            #expect(!coordinator.isPaused(spec))

            let failure = await #expect(throws: ModelStoreError.self) {
                _ = try await ensureModel(spec, ws: ws, coordinator: coordinator,
                                          stallTimeout: .seconds(30))
            }
            guard case .fetch = try #require(failure) else {
                Issue.record("expected a fetch, got \(String(describing: failure))"); return
            }

            #expect(!DownloadStore(dir: ws.downloadsDir).hasResumeData(
                for: LocalStorage.modelRelativePath(repo: "org/v-GGUF", filename: "v-Q4_0.gguf")),
                "the blob was spent on a resume attempt, not left for a run that repeats this one")
        }

        /// The second coordinator over the same root stands in for the next process.
        @Test func aPartialKeptAtExitIsResumedByTheNextLaunch() async throws {
            let ws = makeTemporaryWorkspace()
            defer { removeWorkspace(ws) }
            let spec = try ggufTextSpec("org/a-GGUF", "a-Q4_0.gguf")
            let key = LocalStorage.modelRelativePath(repo: "org/a-GGUF", filename: "a-Q4_0.gguf")
            let file = Data((0..<(256 * 1024)).map { UInt8(truncatingIfNeeded: $0 &* 31) })
            ScriptedURLProtocol.serve(.stalledFile(file, etag: "\"a\"", sent: 64 * 1024))
            let exiting = makeCoordinator(
                ws, sessionConfiguration: ScriptedURLProtocol.configuration)

            exiting.startDownload(spec)
            try await waitUntil(exiting, key) { $0?.bytesDownloaded == 64 * 1024 }
            await exiting.keepPartialsForNextLaunch()

            #expect(exiting.downloads[key]?.state == .paused)
            #expect(DownloadStore(dir: ws.downloadsDir).hasResumeData(for: key))

            ScriptedURLProtocol.serve(.file(file, etag: "\"a\""))
            let next = makeCoordinator(ws, sessionConfiguration: ScriptedURLProtocol.configuration)
            let bound = try await ensureModel(spec, ws: ws, coordinator: next,
                                              stallTimeout: .seconds(30))

            let payload = try #require(bound.boundPaths?.payload)
            #expect(try Data(contentsOf: URL(fileURLWithPath: payload)) == file)
            #expect(ScriptedURLProtocol.ranges == ["bytes=65536-"],
                    "the next launch asked only for the bytes after the partial")
        }

        /// A background session's transfer outlives the process, so the exit leaves it running.
        @Test func aBackgroundSessionsTransferIsLeftRunningAtExit() async throws {
            let ws = makeTemporaryWorkspace()
            defer { removeWorkspace(ws) }
            let spec = try ggufTextSpec("org/a-GGUF", "a-Q4_0.gguf")
            let key = LocalStorage.modelRelativePath(repo: "org/a-GGUF", filename: "a-Q4_0.gguf")
            let file = Data(count: 256 * 1024)
            ScriptedURLProtocol.serve(.stalledFile(file, etag: "\"a\"", sent: 64 * 1024))
            let coordinator = DownloadCoordinator(
                ws: ws,
                downloadHttpClient: UnreachableURLProtocol.httpClient,
                foregroundSession: false,
                reachability: FixedReachability(isConnected: true),
                sessionConfiguration: ScriptedURLProtocol.configuration)
            defer { coordinator.cancel(key: key) }

            coordinator.startDownload(spec)
            try await waitUntil(coordinator, key) { $0?.state == .downloading }
            await coordinator.keepPartialsForNextLaunch()

            #expect(coordinator.downloads[key]?.state == .downloading)
            #expect(!DownloadStore(dir: ws.downloadsDir).hasResumeData(for: key))
        }
    }

    // MARK: - Retries

    /// A `.file` transfer that retries on the backoff schedule. It resumes only when the
    /// resumed `206` has the ETag the coordinator recorded for the partial.
    ///
    /// The tests record the waits instead of sleeping, with no jitter. `.serialized`: the
    /// `ScriptedURLProtocol` script is process-global.
    @Suite(.serialized) @MainActor struct Retries {
        private let repo = "org/a-GGUF"
        private let filename = "a-Q4_0.gguf"
        private var key: String { LocalStorage.modelRelativePath(repo: repo, filename: filename) }
        private var url: String { "https://huggingface.co/\(repo)/resolve/main/\(filename)" }

        /// Holds a retry wait open until the test releases it. The wait ignores
        /// cancellation, so the code after the wait runs whatever stopped the row.
        @MainActor final class Gate {
            private var continuation: CheckedContinuation<Void, Never>?
            var isWaiting: Bool { continuation != nil }
            func wait() async {
                await withCheckedContinuation { continuation = $0 }
            }
            func release() {
                continuation?.resume()
                continuation = nil
            }
        }

        private func makeCoordinator(
            _ ws: PipetteWorkspace, _ recorder: RetryRecorder, online: Bool = true,
            reachability: NetworkReachable? = nil, gate: Gate? = nil
        ) -> DownloadCoordinator {
            DownloadCoordinator(ws: ws,
                                downloadHttpClient: UnreachableURLProtocol.httpClient,
                                foregroundSession: true,
                                reachability: reachability ?? FixedReachability(isConnected: online),
                                sessionConfiguration: ScriptedURLProtocol.configuration,
                                retryHooks: recorder.hooks { await gate?.wait() })
        }

        private func download(
            _ coordinator: DownloadCoordinator, _ spec: Model
        ) async throws {
            coordinator.startDownload(spec)
            try await waitUntil(coordinator, key) { row in
                guard let row else { return true }
                if case .failed = row.state { return true }
                return false
            }
        }

        private func installedBytes(_ ws: PipetteWorkspace, _ spec: Model) throws -> Data {
            let path = try #require(ws.models.bound(spec)?.boundPaths?.payload)
            return try Data(contentsOf: URL(fileURLWithPath: path))
        }

        /// 256 KiB that differ from offset to offset, so a resume from the wrong offset or
        /// of the wrong file reads as different bytes.
        private static func payload(_ seed: UInt8) -> Data {
            Data((0..<(256 * 1024)).map { UInt8(truncatingIfNeeded: $0 &* 31) ^ seed })
        }

        /// N transient failures, then the file: installed, with one warning per failure, the
        /// schedule's waits, and one success line.
        @Test(arguments: [
            (failure: ScriptedURLProtocol.Answer.failure(.networkConnectionLost), times: 1),
            (failure: .failure(.cannotConnectToHost), times: 3),
            (failure: .failure(.timedOut), times: 2),
            (failure: .failure(.secureConnectionFailed), times: 1),
            (failure: .status(503), times: 4),
            (failure: .status(500), times: 2),
        ])
        func aTransientFailureIsRetriedUntilTheDownloadSucceeds(
            failure: ScriptedURLProtocol.Answer, times: Int
        ) async throws {
            let ws = makeTemporaryWorkspace()
            defer { removeWorkspace(ws) }
            let spec = try ggufTextSpec(repo, filename)
            ScriptedURLProtocol.serve(Array(repeating: failure, count: times)
                                      + [.body(Data("weights".utf8))])
            let recorder = RetryRecorder()
            let coordinator = makeCoordinator(ws, recorder)

            try await download(coordinator, spec)

            #expect(try installedBytes(ws, spec) == Data("weights".utf8))
            #expect(ScriptedURLProtocol.requestCount == times + 1)
            #expect(recorder.warnings == times)
            #expect(recorder.sleeps == Array([1.0, 2, 4, 8].prefix(times)))
            #expect(recorder.events.last == .recovered(url: url, attempts: times + 1))
        }

        @Test(arguments: [404, 401, 403])
        func aPermanentRefusalGetsOneAttempt(status: Int) async throws {
            let ws = makeTemporaryWorkspace()
            defer { removeWorkspace(ws) }
            let spec = try ggufTextSpec(repo, filename)
            ScriptedURLProtocol.serve(.status(status))
            let recorder = RetryRecorder()
            let coordinator = makeCoordinator(ws, recorder)

            try await download(coordinator, spec)

            guard case .failed = coordinator.downloads[key]?.state else {
                Issue.record("expected a failed row, got \(String(describing: coordinator.downloads[key]))")
                return
            }
            #expect(ScriptedURLProtocol.requestCount == 1)
            #expect(recorder.events.isEmpty)
            #expect(recorder.sleeps.isEmpty)
        }

        /// Every attempt fails: five requests, four waits, and the last failure is the row's.
        @Test func theAttemptsRunOutAndTheRowFails() async throws {
            let ws = makeTemporaryWorkspace()
            defer { removeWorkspace(ws) }
            let spec = try ggufTextSpec(repo, filename)
            ScriptedURLProtocol.serve(.status(503))
            let recorder = RetryRecorder()
            let coordinator = makeCoordinator(ws, recorder)

            try await download(coordinator, spec)

            guard case .failed = coordinator.downloads[key]?.state else {
                Issue.record("expected a failed row, got \(String(describing: coordinator.downloads[key]))")
                return
            }
            #expect(ScriptedURLProtocol.requestCount == 5)
            #expect(recorder.sleeps == [1, 2, 4, 8])
            #expect(recorder.warnings == 4)
        }

        /// Every attempt breaks part-way: the attempts run out, but the row is paused with
        /// its partial kept, and a later resume finishes the file from that partial. The
        /// device is online, so no reconnect comes, and the row is not flagged as the
        /// network's pause.
        @Test func theAttemptsRunOutOnABrokenBodyAndThePartialIsKept() async throws {
            let ws = makeTemporaryWorkspace()
            defer { removeWorkspace(ws) }
            let spec = try ggufTextSpec(repo, filename)
            let file = Self.payload(1)
            ScriptedURLProtocol.serve(.file(file, etag: "\"a\"", cutAfter: 16 * 1024))
            let recorder = RetryRecorder()
            let coordinator = makeCoordinator(ws, recorder)

            coordinator.startDownload(spec)
            try await waitUntil(coordinator, key) { $0?.state == .paused }

            #expect(ScriptedURLProtocol.requestCount == 5)
            #expect(recorder.sleeps == [1, 2, 4, 8])
            #expect(coordinator.downloads[key]?.interruptedByNetwork == false)
            let store = DownloadStore(dir: ws.downloadsDir)
            #expect(store.hasResumeData(for: key))
            #expect(store.allRecords().count == 1)

            ScriptedURLProtocol.serve(.file(file, etag: "\"a\""))
            coordinator.resume(key: key)
            try await waitUntil(coordinator, key) { $0 == nil }

            #expect(try installedBytes(ws, spec) == file)
            #expect(ScriptedURLProtocol.ranges == ["bytes=81920-"],
                    "five cuts of 16 KiB each, and the resume continues after them")
        }

        /// The last attempt is a resume that gets a different ETag. The restart from zero
        /// needs one more attempt, and none remain. With no partial left, the row fails.
        @Test func theAttemptsRunOutOnAChangedFileAndTheRowFails() async throws {
            let ws = makeTemporaryWorkspace()
            defer { removeWorkspace(ws) }
            let spec = try ggufTextSpec(repo, filename)
            let cutOld = ScriptedURLProtocol.Answer.file(Self.payload(1), etag: "\"a\"",
                                                         cutAfter: 16 * 1024)
            ScriptedURLProtocol.serve([cutOld, cutOld, cutOld, cutOld,
                                       .file(Self.payload(2), etag: "\"b\"")])
            let recorder = RetryRecorder()
            let coordinator = makeCoordinator(ws, recorder)

            try await download(coordinator, spec)

            guard case .failed = coordinator.downloads[key]?.state else {
                Issue.record("expected a failed row, got \(String(describing: coordinator.downloads[key]))")
                return
            }
            #expect(ScriptedURLProtocol.requestCount == 5)
            #expect(ScriptedURLProtocol.ranges.last != nil, "the last attempt is a resume")
            #expect(recorder.sleeps == [1, 2, 4, 8])
            let store = DownloadStore(dir: ws.downloadsDir)
            #expect(!store.hasResumeData(for: key))
        }

        /// A Pause or a Cancel during the wait between two attempts stops the retry. When
        /// the wait ends, no new attempt starts.
        @Test(arguments: [true, false])
        func aRowStoppedDuringTheWaitIsNotRetried(pause: Bool) async throws {
            let ws = makeTemporaryWorkspace()
            defer { removeWorkspace(ws) }
            let spec = try ggufTextSpec(repo, filename)
            ScriptedURLProtocol.serve(.status(503))
            let gate = Gate()
            let coordinator = makeCoordinator(ws, RetryRecorder(), gate: gate)

            coordinator.startDownload(spec)
            try await waitUntil(coordinator, key) { _ in gate.isWaiting }
            if pause {
                coordinator.pause(key: key)
                try await waitUntil(coordinator, key) { $0?.state == .paused }
            } else {
                coordinator.cancel(key: key)
                try await waitUntil(coordinator, key) { $0 == nil }
            }
            gate.release()
            try await Task.sleep(for: .milliseconds(200))

            #expect(ScriptedURLProtocol.requestCount == 1)
            #expect(coordinator.downloads[key]?.state == (pause ? .paused : nil))
        }

        /// A `429` waits what its `Retry-After` asks, up to two minutes. A longer wait is
        /// refused: one attempt, one warning, and the row fails.
        @Test(arguments: [
            (retryAfter: "2", requests: 2, sleeps: [2.0], installed: true),
            (retryAfter: "120", requests: 2, sleeps: [120], installed: true),
            (retryAfter: "121", requests: 1, sleeps: [], installed: false),
        ])
        func aRateLimitWaitsWhatItsRetryAfterAsks(
            retryAfter: String, requests: Int, sleeps: [TimeInterval], installed: Bool
        ) async throws {
            let ws = makeTemporaryWorkspace()
            defer { removeWorkspace(ws) }
            let spec = try ggufTextSpec(repo, filename)
            ScriptedURLProtocol.serve([.status(429, headers: ["Retry-After": retryAfter]),
                                       .body(Data("weights".utf8))])
            let recorder = RetryRecorder()
            let coordinator = makeCoordinator(ws, recorder)

            try await download(coordinator, spec)

            #expect((ws.models.bound(spec) != nil) == installed)
            #expect(ScriptedURLProtocol.requestCount == requests)
            #expect(recorder.sleeps == sleeps)
            #expect(recorder.warnings == 1)
        }

        /// A body that breaks part-way resumes at the byte where it stopped. The resumed
        /// `206` has the recorded ETag, so the complete file installs.
        @Test func aBrokenBodyResumesWhenTheETagMatches() async throws {
            let ws = makeTemporaryWorkspace()
            defer { removeWorkspace(ws) }
            let spec = try ggufTextSpec(repo, filename)
            let file = Self.payload(1)
            ScriptedURLProtocol.serve([.file(file, etag: "\"a\"", cutAfter: 100 * 1024),
                                       .file(file, etag: "\"a\"")])
            let recorder = RetryRecorder()
            let coordinator = makeCoordinator(ws, recorder)

            try await download(coordinator, spec)

            #expect(try installedBytes(ws, spec) == file)
            let ranges = ScriptedURLProtocol.ranges
            #expect(ranges == [nil, "bytes=102400-"], "the second request resumes at the cut")
            #expect(recorder.events.last == .recovered(url: url, attempts: 2))
        }

        /// The file changed on the server, and the server obeys `If-Range`: it answers the
        /// resume with a `200` and the whole new file. URLSession writes that `200` from
        /// zero, so the coordinator installs it without a second download.
        @Test func aResumeAnsweredWithTheWholeNewFileInstallsIt() async throws {
            let ws = makeTemporaryWorkspace()
            defer { removeWorkspace(ws) }
            let spec = try ggufTextSpec(repo, filename)
            let old = Self.payload(1)
            let new = Self.payload(2)
            ScriptedURLProtocol.serve([.file(old, etag: "\"a\"", cutAfter: 100 * 1024),
                                       .wholeFile(new, etag: "\"b\"")])
            let recorder = RetryRecorder()
            let coordinator = makeCoordinator(ws, recorder)

            try await download(coordinator, spec)

            #expect(try installedBytes(ws, spec) == new)
            #expect(ScriptedURLProtocol.ranges == [nil, "bytes=102400-"],
                    "one resume at the cut, and no second download")
        }

        /// The `200` of the new file also breaks. Its partial holds only the new file, so
        /// the next attempt resumes it and does not start from zero.
        @Test func aBrokenAnswerWithTheWholeNewFileResumes() async throws {
            let ws = makeTemporaryWorkspace()
            defer { removeWorkspace(ws) }
            let spec = try ggufTextSpec(repo, filename)
            let old = Self.payload(1)
            let new = Self.payload(2)
            ScriptedURLProtocol.serve([.file(old, etag: "\"a\"", cutAfter: 100 * 1024),
                                       .wholeFile(new, etag: "\"b\"", cutAfter: 50 * 1024),
                                       .file(new, etag: "\"b\"")])
            let recorder = RetryRecorder()
            let coordinator = makeCoordinator(ws, recorder)

            try await download(coordinator, spec)

            #expect(try installedBytes(ws, spec) == new)
            #expect(ScriptedURLProtocol.ranges == [nil, "bytes=102400-", "bytes=51200-"],
                    "the third request resumes the new file at its own cut")
        }

        /// The connection drops because the device went offline, before the path monitor
        /// says so. When the wait ends offline, the row is paused for the reconnect, with
        /// its partial kept, and no new task spends the partial.
        @Test func aRetryThatComesDueOfflinePausesTheRow() async throws {
            let ws = makeTemporaryWorkspace()
            defer { removeWorkspace(ws) }
            let spec = try ggufTextSpec(repo, filename)
            ScriptedURLProtocol.serve(.file(Self.payload(1), etag: "\"a\"", cutAfter: 100 * 1024))
            let recorder = RetryRecorder()
            let coordinator = makeCoordinator(ws, recorder, online: false)

            coordinator.startDownload(spec)
            try await waitUntil(coordinator, key) { $0?.state == .paused }

            #expect(coordinator.downloads[key]?.interruptedByNetwork == true)
            #expect(ScriptedURLProtocol.requestCount == 1)
            #expect(recorder.sleeps == [1])
            #expect(DownloadStore(dir: ws.downloadsDir).hasResumeData(for: key))
        }

        /// Wi-Fi goes off mid-download. The first error is a lost connection, and the path
        /// monitor still says online, so the transfer retries. The retry then fails offline
        /// and pauses the row for the reconnect. The reconnect resumes it at the cut.
        @Test func aLostConnectionThatTurnsOutOfflinePausesThenResumesOnReconnect()
            async throws {
            let ws = makeTemporaryWorkspace()
            defer { removeWorkspace(ws) }
            let spec = try ggufTextSpec(repo, filename)
            let file = Self.payload(1)
            ScriptedURLProtocol.serve([.file(file, etag: "\"a\"", cutAfter: 100 * 1024),
                                       .failure(.notConnectedToInternet),
                                       .file(file, etag: "\"a\"")])
            let coordinator = makeCoordinator(ws, RetryRecorder())

            coordinator.startDownload(spec)
            try await waitUntil(coordinator, key) { $0?.state == .paused }

            #expect(coordinator.downloads[key]?.interruptedByNetwork == true)
            #expect(DownloadStore(dir: ws.downloadsDir).hasResumeData(for: key))

            coordinator.resumeAfterReconnect()
            try await waitUntil(coordinator, key) { $0 == nil }

            #expect(try installedBytes(ws, spec) == file)
            #expect(ScriptedURLProtocol.ranges == [nil, "bytes=102400-", "bytes=102400-"],
                    "both requests after the cut resume at it")
        }

        /// The retry comes due before the path monitor has a reading. The coordinator waits
        /// for the reading, which says online, so the retry resumes and does not pause.
        @Test func aRetryThatComesDueBeforeAReadingWaitsForIt() async throws {
            let ws = makeTemporaryWorkspace()
            defer { removeWorkspace(ws) }
            let spec = try ggufTextSpec(repo, filename)
            let file = Self.payload(1)
            ScriptedURLProtocol.serve([.file(file, etag: "\"a\"", cutAfter: 100 * 1024),
                                       .file(file, etag: "\"a\"")])
            let coordinator = makeCoordinator(
                ws, RetryRecorder(),
                reachability: WarmingReachability(warmsAfter: .milliseconds(500)))

            try await download(coordinator, spec)

            #expect(try installedBytes(ws, spec) == file)
            #expect(ScriptedURLProtocol.ranges == [nil, "bytes=102400-"])
        }

        /// The resume fails before the server answers. URLSession restores the response
        /// from the resume data, so its ETag still matches and the next attempt resumes.
        @Test func aResumeThatFailsBeforeAResponseKeepsThePartial() async throws {
            let ws = makeTemporaryWorkspace()
            defer { removeWorkspace(ws) }
            let spec = try ggufTextSpec(repo, filename)
            let file = Self.payload(1)
            ScriptedURLProtocol.serve([.file(file, etag: "\"a\"", cutAfter: 100 * 1024),
                                       .failure(.cannotConnectToHost),
                                       .file(file, etag: "\"a\"")])
            let recorder = RetryRecorder()
            let coordinator = makeCoordinator(ws, recorder)

            try await download(coordinator, spec)

            #expect(try installedBytes(ws, spec) == file)
            let ranges = ScriptedURLProtocol.ranges
            #expect(ranges == [nil, "bytes=102400-", "bytes=102400-"],
                    "both requests after the cut resume at it")
        }

        /// The file changed on the server between the break and the resume. The coordinator
        /// stops the resume at its first response, before any bytes of the new file join the
        /// partial of the old file. It then fetches the new file from zero.
        @Test func aResumeWithADifferentETagStartsFromZero() async throws {
            let ws = makeTemporaryWorkspace()
            defer { removeWorkspace(ws) }
            let spec = try ggufTextSpec(repo, filename)
            let old = Self.payload(1)
            let new = Self.payload(2)
            ScriptedURLProtocol.serve([.file(old, etag: "\"a\"", cutAfter: 100 * 1024),
                                       .file(new, etag: "\"b\"")])
            let recorder = RetryRecorder()
            let coordinator = makeCoordinator(ws, recorder)

            try await download(coordinator, spec)

            #expect(try installedBytes(ws, spec) == new)
            let ranges = ScriptedURLProtocol.ranges
            #expect(ranges == [nil, "bytes=102400-", nil],
                    "the resume at the cut, and the fresh start that replaced it")
        }

        /// The resume of a changed file stops at its first response, before its body
        /// breaks. The fresh start follows with no retry wait, and it uses one attempt. The
        /// break of the old body uses no attempt.
        @Test func aChangedResumeWhoseBodyBreaksStartsFromZeroWithNoWait() async throws {
            let ws = makeTemporaryWorkspace()
            defer { removeWorkspace(ws) }
            let spec = try ggufTextSpec(repo, filename)
            let old = Self.payload(1)
            let new = Self.payload(2)
            ScriptedURLProtocol.serve([.file(old, etag: "\"a\"", cutAfter: 100 * 1024),
                                       .file(new, etag: "\"b\"", cutAfter: 16 * 1024),
                                       .file(new, etag: "\"b\"")])
            let recorder = RetryRecorder()
            let coordinator = makeCoordinator(ws, recorder)

            try await download(coordinator, spec)

            #expect(try installedBytes(ws, spec) == new)
            #expect(ScriptedURLProtocol.ranges == [nil, "bytes=102400-", nil])
            #expect(recorder.sleeps == [1], "only the first break waits")
            #expect(recorder.events.last == .recovered(url: url, attempts: 3))
        }

        /// The coordinator cancels the resume of a changed file and starts from zero. Then a
        /// late finish of the cancelled task arrives, with bytes of both file versions. The
        /// coordinator does not install those bytes, and the new transfer continues.
        @Test func aLateFinishOfTheCancelledResumeDoesNotInstall() async throws {
            let ws = makeTemporaryWorkspace()
            defer { removeWorkspace(ws) }
            let spec = try ggufTextSpec(repo, filename)
            let old = Self.payload(1)
            let new = Self.payload(2)
            ScriptedURLProtocol.serve([.file(old, etag: "\"a\"", cutAfter: 100 * 1024),
                                       .file(new, etag: "\"b\""),
                                       .hang])
            let coordinator = makeCoordinator(ws, RetryRecorder())

            coordinator.startDownload(spec)
            try await waitUntil(coordinator, key) { _ in ScriptedURLProtocol.requestCount == 3 }
            let cancelled = try #require(ScriptedURLProtocol.tasks[1] as? URLSessionDownloadTask)
            let joined = FileManager.default.temporaryDirectory
                .appendingPathComponent(UUID().uuidString)
            try (old.prefix(100 * 1024) + new.dropFirst(100 * 1024)).write(to: joined)
            defer { try? FileManager.default.removeItem(at: joined) }

            coordinator.urlSession(URLSession.shared, downloadTask: cancelled,
                                   didFinishDownloadingTo: joined)

            #expect(FileManager.default.fileExists(atPath: joined.path),
                    "nothing adopted the joined bytes")
            #expect(coordinator.downloads[key]?.state == .resuming)
            #expect(ws.models.bound(spec) == nil)

            coordinator.cancel(key: key)
            try await waitUntil(coordinator, key) { $0 == nil }
        }

        /// The last callback of a resume of a changed file. A background session that ends
        /// the transfer while the app is suspended delivers only this callback.
        enum ChangedResumeEnd: Sendable {
            case finish
            case brokenBody

            /// The body of the new file that the resume gets.
            var cutAfter: Int? {
                switch self {
                case .finish: nil
                case .brokenBody: 16 * 1024
                }
            }
        }

        /// A resume of a changed file reaches the app only at its end, and its `206` has a
        /// different ETag. The coordinator keeps no bytes of the resume. The next request
        /// starts from zero.
        @Test(arguments: [ChangedResumeEnd.finish, .brokenBody])
        func aChangedResumeSeenOnlyAtItsEndStartsFromZero(end: ChangedResumeEnd) async throws {
            let ws = makeTemporaryWorkspace()
            defer { removeWorkspace(ws) }
            let spec = try ggufTextSpec(repo, filename)
            let old = Self.payload(1)
            let new = Self.payload(2)
            ScriptedURLProtocol.serve([.file(old, etag: "\"a\"", cutAfter: 100 * 1024),
                                       .hang,
                                       .file(new, etag: "\"b\"", cutAfter: end.cutAfter),
                                       .hang])
            let coordinator = makeCoordinator(ws, RetryRecorder())

            coordinator.startDownload(spec)
            try await waitUntil(coordinator, key) { _ in ScriptedURLProtocol.requestCount == 2 }
            // The coordinator's resume hangs with `"a"` recorded; a task in a second session
            // gets the new file's `206` and stands in for it.
            let probeSession = URLSession(configuration: ScriptedURLProtocol.configuration)
            defer { probeSession.invalidateAndCancel() }
            var request = URLRequest(url: try #require(URL(string: url)))
            request.setValue("bytes=\(100 * 1024)-", forHTTPHeaderField: "Range")
            let resumed = probeSession.downloadTask(with: request)
            resumed.resume()
            try await waitUntil(coordinator, key) { _ in resumed.state == .completed }
            #expect((resumed.response as? HTTPURLResponse)?.statusCode == 206)
            resumed.taskDescription = key

            switch end {
            case .finish:
                let joined = FileManager.default.temporaryDirectory
                    .appendingPathComponent(UUID().uuidString)
                try (old.prefix(100 * 1024) + new.dropFirst(100 * 1024)).write(to: joined)
                defer { try? FileManager.default.removeItem(at: joined) }
                coordinator.urlSession(URLSession.shared, downloadTask: resumed,
                                       didFinishDownloadingTo: joined)
                #expect(FileManager.default.fileExists(atPath: joined.path),
                        "nothing adopted the joined bytes")
            case .brokenBody:
                let error = try #require(resumed.error)
                #expect((error as NSError).userInfo[NSURLSessionDownloadTaskResumeData] != nil)
                coordinator.urlSession(URLSession.shared, task: resumed,
                                       didCompleteWithError: error)
            }

            try await waitUntil(coordinator, key) { _ in ScriptedURLProtocol.requestCount == 4 }
            #expect(ScriptedURLProtocol.ranges.last == .some(nil),
                    "the request after the resume starts from zero")
            let store = DownloadStore(dir: ws.downloadsDir)
            #expect(!store.hasResumeData(for: key))
            #expect(ws.models.bound(spec) == nil)

            coordinator.cancel(key: key)
            try await waitUntil(coordinator, key) { $0 == nil }
        }

        /// A response with no strong ETag cannot confirm a resume. The next attempt therefore
        /// starts from zero, and does not append to bytes that nothing can check.
        @Test(arguments: [nil, "W/\"a\""])
        func aBodyWithNoStrongETagStartsAgainFromZero(etag: String?) async throws {
            let ws = makeTemporaryWorkspace()
            defer { removeWorkspace(ws) }
            let spec = try ggufTextSpec(repo, filename)
            let file = Self.payload(1)
            ScriptedURLProtocol.serve([.file(file, etag: etag, cutAfter: 100 * 1024),
                                       .file(file, etag: etag)])
            let recorder = RetryRecorder()
            let coordinator = makeCoordinator(ws, recorder)

            try await download(coordinator, spec)

            #expect(try installedBytes(ws, spec) == file)
            #expect(ScriptedURLProtocol.ranges == [nil, nil])
        }

        /// A transfer with no strong ETag that is left paused keeps its row across a
        /// relaunch. Its resume then starts from zero, because no ETag can check it. A
        /// record from before the `etag` field restores to the same state.
        @Test func aPausedRowWithNoStrongETagSurvivesARelaunch() async throws {
            let ws = makeTemporaryWorkspace()
            defer { removeWorkspace(ws) }
            let spec = try ggufTextSpec(repo, filename)
            let file = Self.payload(1)
            ScriptedURLProtocol.serve(.file(file, etag: nil, cutAfter: 16 * 1024))
            let coordinator = makeCoordinator(ws, RetryRecorder())
            coordinator.startDownload(spec)
            try await waitUntil(coordinator, key) { $0?.state == .paused }

            let relaunched = makeCoordinator(ws, RetryRecorder())

            #expect(relaunched.downloads[key]?.state == .paused)
            ScriptedURLProtocol.serve(.file(file, etag: nil))
            relaunched.resume(key: key)
            try await waitUntil(relaunched, key) { $0 == nil }
            #expect(try installedBytes(ws, spec) == file)
            #expect(ScriptedURLProtocol.ranges == [nil])
        }

        /// A `416`, or a `403` from an expired signed URL, refuses the partial, not the file.
        @Test(arguments: [416, 403])
        func aRefusedResumeStartsAgainFromZero(status: Int) async throws {
            let ws = makeTemporaryWorkspace()
            defer { removeWorkspace(ws) }
            let spec = try ggufTextSpec(repo, filename)
            let file = Self.payload(1)
            ScriptedURLProtocol.serve([.file(file, etag: "\"a\"", cutAfter: 100 * 1024),
                                       .status(status),
                                       .file(file, etag: "\"a\"")])
            let recorder = RetryRecorder()
            let coordinator = makeCoordinator(ws, recorder)

            try await download(coordinator, spec)

            #expect(try installedBytes(ws, spec) == file)
            let ranges = ScriptedURLProtocol.ranges
            #expect(ranges == [nil, "bytes=102400-", nil])
        }

        @Test func aRecordWrittenBeforeTheETagFieldStillDecodes() throws {
            struct OldRecord: Codable {
                let filename: String
                let urlString: String
                let repo: String?
            }
            let data = try PropertyListEncoder().encode(
                OldRecord(filename: filename, urlString: url, repo: repo))
            let record = try PropertyListDecoder().decode(DownloadRecord.self, from: data)
            #expect(record.etag == nil)
            #expect(record.filename == filename)
        }
    }

    // MARK: - The two meanings of cancellation

    /// `TransferRegistry` against the real coordinator and admission ledger, where the two
    /// cancellations differ: bytes still moving, and a byte budget still held.
    ///
    /// `HangingURLProtocol` transfers never finish, so each test must end its own, or the
    /// supervising task polls on against a `root` the test has already deleted.
    @Suite(.serialized) @MainActor struct Cancellation {
        private let repo = "org/a-GGUF"
        private let filename = "a-Q4_0.gguf"
        private var rowKey: String {
            LocalStorage.modelRelativePath(repo: repo, filename: filename)
        }

        /// Foreground and fixed online for the reasons `PauseRecovery.makeCoordinator` gives.
        private func makeCoordinator(_ ws: PipetteWorkspace) -> DownloadCoordinator {
            DownloadCoordinator(ws: ws,
                                downloadHttpClient: UnreachableURLProtocol.httpClient,
                                foregroundSession: true,
                                reachability: FixedReachability(isConnected: true),
                                sessionConfiguration: HangingURLProtocol.configuration)
        }

        /// The grant is handed back one hop after the cancellation surfaces in the
        /// supervising task, so it is polled for.
        private func untilOutstanding(
            _ bytes: Int64, _ coordinator: DownloadCoordinator
        ) async throws {
            for _ in 0..<500 where await coordinator.admission.outstandingBytes != bytes {
                try await Task.sleep(for: .milliseconds(10))
            }
            #expect(await coordinator.admission.outstandingBytes == bytes)
        }

        /// A waiter that released the budget on its way out, while the transfer went on
        /// writing, would let a second model be admitted into space already promised.
        @Test func aCallerThatStopsWaitingLeavesTheTransferAndItsGrantAlone() async throws {
            let ws = makeTemporaryWorkspace()
            defer { removeWorkspace(ws) }
            let spec = try ggufTextSpec(repo, filename)
            let coordinator = makeCoordinator(ws)

            let fetch = Task {
                try await ensureModel(spec, ws: ws, coordinator: coordinator,
                                      declaredSizeBytes: 4096, stallTimeout: .seconds(600))
            }
            try await waitUntil(coordinator, rowKey) { $0?.state == .connecting }
            #expect(await coordinator.admission.outstandingBytes == 4096)

            fetch.cancel()
            await #expect(throws: CancellationError.self) { _ = try await fetch.value }

            #expect(coordinator.hasTransfer(for: spec),
                    "the transfer is not the leaving caller's to stop")
            #expect(await coordinator.admission.outstandingBytes == 4096,
                    "the grant belongs to the transfer, which is still spending it")

            coordinator.cancelTransfers(for: spec)
            try await untilOutstanding(0, coordinator)
        }

        /// Re-admitted before any byte moves: the catalog's `bytes` is authored by hand, and
        /// the plan is what the source actually lists.
        @Test func aPlanHeavierThanItsGrantIsReadmitted() async throws {
            let ws = makeTemporaryWorkspace()
            defer { removeWorkspace(ws) }
            try ws.setStorageQuotaBytes(1_000_000)
            let coordinator = makeCoordinator(ws)
            coordinator.transport = OversizePlanTransport(plannedBytes: 400_000)
            let spec = try onnxGenaiSpec("org/onnx-repo")

            let fetch = Task {
                try await ensureModel(spec, ws: ws, coordinator: coordinator,
                                      declaredSizeBytes: 1000, stallTimeout: .seconds(10))
            }
            for _ in 0..<500 where await coordinator.admission.outstandingBytes != 400_000 {
                try await Task.sleep(for: .milliseconds(10))
            }
            #expect(await coordinator.admission.outstandingBytes == 400_000,
                    "the grant is resized to what the plan says the pull weighs")

            fetch.cancel()
            coordinator.cancelTransfers(for: spec)
            try await untilOutstanding(0, coordinator)
        }

        /// A paused transfer keeps its partial and resumes into the same entry, so handing the
        /// bytes back would offer them to the next admission twice.
        @Test func pausingTheDownloadFailsTheWaiterButKeepsTheGrant() async throws {
            let ws = makeTemporaryWorkspace()
            defer { removeWorkspace(ws) }
            let spec = try ggufTextSpec(repo, filename)
            let coordinator = makeCoordinator(ws)

            let fetch = Task {
                try await ensureModel(spec, ws: ws, coordinator: coordinator,
                                      declaredSizeBytes: 4096, stallTimeout: .seconds(600))
            }
            try await waitUntil(coordinator, rowKey) { $0?.state == .connecting }
            #expect(await coordinator.admission.outstandingBytes == 4096)

            coordinator.pause(key: rowKey)
            await #expect(throws: ModelStoreError.self) { _ = try await fetch.value }

            #expect(await coordinator.admission.outstandingBytes == 4096,
                    "the partial is still on disk and the resume writes into it")

            coordinator.cancelTransfers(for: spec)
            try await untilOutstanding(0, coordinator)
        }

        @Test func cancellingTheDownloadFailsTheWaiterAndGivesTheGrantBack() async throws {
            let ws = makeTemporaryWorkspace()
            defer { removeWorkspace(ws) }
            let spec = try ggufTextSpec(repo, filename)
            let coordinator = makeCoordinator(ws)

            let fetch = Task {
                try await ensureModel(spec, ws: ws, coordinator: coordinator,
                                      declaredSizeBytes: 4096, stallTimeout: .seconds(600))
            }
            try await waitUntil(coordinator, rowKey) { $0?.state == .connecting }

            coordinator.cancelTransfers(for: spec)

            let failure = await #expect(throws: ModelStoreError.self) { _ = try await fetch.value }
            guard case .fetchCancelled = try #require(failure) else {
                Issue.record("expected a fetchCancelled, got \(String(describing: failure))")
                return
            }
            try await waitUntil(coordinator, rowKey) { $0 == nil }
            #expect(!coordinator.hasTransfer(for: spec))
            try await untilOutstanding(0, coordinator)
        }

        /// Admitting each would promise 8 KiB for a 4 KiB transfer, and near the quota refuse
        /// the next download for space nothing will use.
        @Test func twoCallersShareOneTransferAndOneGrant() async throws {
            let ws = makeTemporaryWorkspace()
            defer { removeWorkspace(ws) }
            let spec = try ggufTextSpec(repo, filename)
            let coordinator = makeCoordinator(ws)
            let key = try #require(ModelStorageKey.of(spec))

            let first = Task {
                try await ensureModel(spec, ws: ws, coordinator: coordinator,
                                      declaredSizeBytes: 4096, stallTimeout: .seconds(600))
            }
            try await waitUntil(coordinator, rowKey) { $0?.state == .connecting }
            let second = Task {
                try await ensureModel(spec, ws: ws, coordinator: coordinator,
                                      declaredSizeBytes: 4096, stallTimeout: .seconds(600))
            }
            for _ in 0..<500 where coordinator.transfers.waiterCount(for: key) != 2 {
                try await Task.sleep(for: .milliseconds(10))
            }
            #expect(coordinator.transfers.waiterCount(for: key) == 2)

            #expect(await coordinator.admission.outstandingBytes == 4096,
                    "one transfer, one grant, however many callers want it")
            #expect(coordinator.downloads.count == 1)

            coordinator.cancelTransfers(for: spec)
            await #expect(throws: ModelStoreError.self) { _ = try await first.value }
            await #expect(throws: ModelStoreError.self) { _ = try await second.value }
            try await untilOutstanding(0, coordinator)
        }

        /// Otherwise every installed MLX or ONNX package permanently reduces the quota.
        @Test func aSuccessfulDirectoryDownloadGivesTheGrantBack() async throws {
            let ws = makeTemporaryWorkspace()
            defer { removeWorkspace(ws) }
            let spec = try onnxGenaiSpec("org/onnx-repo")
            let coordinator = makeCoordinator(ws)
            coordinator.transport = CompleteOnnxTransport()

            _ = try await ensureModel(
                spec, ws: ws, coordinator: coordinator,
                declaredSizeBytes: 4096, stallTimeout: .seconds(10))

            #expect(ws.models.bound(spec) != nil)
            try await untilOutstanding(0, coordinator)
        }
    }

    // MARK: - ONNX Runtime GenAI routing

    @Test func anOnnxPackageDownloadsThroughTheDirectoryTransport() async throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let spec = try onnxGenaiSpec("microsoft/Phi-3-mini-4k-instruct-onnx",
                                     subpath: "cpu_and_mobile/cpu-int4-rtn-block-32")
        let ref = try #require(HfModelRef.of(spec))
        let coord = DownloadCoordinator(ws: ws)
        coord.transport = CompleteOnnxTransport()

        coord.startDownload(spec)
        try await waitUntil(coord, ref.key) { $0 == nil }

        #expect(coord.errorMessage == nil, "the pull must not be refused as unfetchable")
        #expect(ws.models.bound(spec) != nil, "the package must publish")
    }

    /// Which files a repo root selects is per format: an ONNX root takes everything, an MLX
    /// root takes the allowlist.
    @Test func theTransportIsToldWhichFormatItIsMoving() async throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let seen = SeenModel()
        let coord = DownloadCoordinator(ws: ws)
        coord.transport = RecordingTransport(seen: seen)
        let spec = try onnxGenaiSpec("org/onnx-repo")
        let ref = try #require(HfModelRef.of(spec))

        coord.startDownload(spec)
        // `RecordingTransport` fails the pull, so the row settles in `.failed`.
        try await waitUntil(coord, ref.key) { if case .failed = $0?.state { true } else { false } }

        #expect(seen.globs == ["*"])
    }
}

private final class SeenModel: @unchecked Sendable {
    private let lock = NSLock()
    private var recorded: [String] = []
    func record(_ globs: [String]) { lock.lock(); recorded = globs; lock.unlock() }
    var globs: [String] { lock.lock(); defer { lock.unlock() }; return recorded }
}

/// Records the globs the coordinator's spec resolves to, then fails the pull.
private nonisolated struct RecordingTransport: ModelTransport {
    let seen: SeenModel
    func download(_ model: Model, ref: HfModelRef, token: AuthToken?,
                  admit: @escaping @Sendable (Int64) async throws -> Void,
                  progress: @escaping @Sendable (Double) -> Void) async throws -> TransferOutcome {
        seen.record(ref.globs(for: ModelType.of(model)))
        throw DownloadError.transport("recorded")
    }
}
