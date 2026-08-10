import Foundation
import Testing

@testable import Pipette

/// Returns a complete MLX bundle, so `finishMLX` reaches the install + sweep.
private nonisolated struct CompleteMLXDownloader: MLXModelDownloading {
    let payloadBytes: Int
    func download(_ ref: HFModelRef, token: AuthToken?,
                  progress: @escaping @Sendable (Double) -> Void) async throws -> URL {
        progress(1.0)
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("dl-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        for name in mlxBundleFiles {
            try Data(repeating: 0x78, count: payloadBytes).write(to: dir.appendingPathComponent(name))
        }
        return dir
    }
}

/// The quota's two coordinator-side rules: an oversize artifact is refused before the
/// fetch, and a finished install sweeps the store back under the cap without ever
/// reclaiming what is still in flight.
///
/// `.serialized`: the file-download path lazily creates a background `URLSession`
/// keyed by a process-global identifier, so these must not race each other.
@Suite(.serialized) @MainActor struct DownloadQuotaGateTests {
    @Test func refusesAnArtifactLargerThanTheWholeQuota() throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }
        try storage.setStorageQuotaBytes(1_000_000)
        let coord = DownloadCoordinator(storage: storage)
        let spec = try ggufTextSpec("org/big-GGUF", "big-Q4_0.gguf")

        coord.startDownload(spec, declaredSizeBytes: 9_000_000)

        #expect(coord.downloads.isEmpty, "nothing should be enqueued")
        let message = try #require(coord.errorMessage)
        #expect(message.contains(ByteFormat.fileSize(9_000_000)))
        #expect(message.contains(ByteFormat.storageLimit(1_000_000)))
    }

    @Test func aDeclaredSizeWithinTheQuotaEnqueuesNormally() throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }
        try storage.setStorageQuotaBytes(1_000_000)
        let coord = DownloadCoordinator(storage: storage)
        let repo = try HFRepo.parse("org/small-GGUF")
        let filename = try RepoSubpath("small-Q4_0.gguf")
        let key = LocalStorage.modelRelativePath(repo: repo.description, filename: filename.value)
        defer { coord.cancel(key: key) }

        coord.startDownload(.ggufText(GgufText(source: .huggingFace(repo: repo, path: filename, sha256: nil))),
                            declaredSizeBytes: 900_000)

        #expect(coord.downloads[key] != nil)
    }

    /// fetch → publish → sweep: the MLX install path evicts to fit, and what it just
    /// installed survives even though it is what pushed the store over.
    @Test func anMlxInstallSweepsAndKeepsWhatItJustInstalled() async throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }
        let stale = try installEntry(storage, try ggufTextSpec("org/stale-GGUF", "stale-Q4_0.gguf"),
                                     payloadBytes: 64_000,
                                     lastUsedAt: Date(timeIntervalSince1970: 1_600_000_000))
        try storage.setStorageQuotaBytes(DiskUsage.bytes(at: stale))
        let coord = DownloadCoordinator(storage: storage)
        coord.mlxDownloader = CompleteMLXDownloader(payloadBytes: 64_000)
        let ref = try HFModelRef.parse(repo: "org/fresh-MLX-4bit")

        coord.startMLXDownload(ref)
        try await waitUntilCleared(coord, ref.key)

        let installed = try #require(storage.modelStore.entryDir(for: .mlx(ref.asMlx())))
        #expect(FileManager.default.fileExists(atPath: installed.path))
        #expect(!FileManager.default.fileExists(atPath: stale.path), "the stale entry should be evicted")
    }

    /// A vision model's projector can land first: it publishes the shared entry, and
    /// the still-in-flight weights transfer pins that entry so a sweep can't take it.
    @Test func anInFlightVisionTransferPinsTheSharedEntry() throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }
        let coord = DownloadCoordinator(storage: storage)
        let spec = try ggufVisionSpec("org/vl-GGUF", "vl-Q4_0.gguf", "mmproj-vl-F16.gguf")
        coord.startDownload(spec)
        defer {
            for filename in ["vl-Q4_0.gguf", "mmproj-vl-F16.gguf"] {
                coord.cancel(key: LocalStorage.modelRelativePath(repo: "org/vl-GGUF", filename: filename))
            }
        }

        // The projector finished first, so the entry exists with a manifest but no
        // weights yet — manifest-less would have made it garbage.
        let downloaded = FileManager.default.temporaryDirectory
            .appendingPathComponent("mmproj-\(UUID().uuidString).gguf")
        try Data(repeating: 0x6D, count: 64_000).write(to: downloaded)
        try GGUFModelInstaller(repo: "org/vl-GGUF", filename: "mmproj-vl-F16.gguf",
                               source: spec, storage: storage).install(from: downloaded)

        let pins = coord.sweepPins(justInstalled: nil)
        #expect(pins.entries.contains(try #require(ModelStorageKey.of(spec))))

        try storage.setStorageQuotaBytes(1)
        #expect(storage.sweepToQuota(pinning: pins).removed.isEmpty)
        let entry = try #require(storage.modelStore.entryDir(for: spec))
        #expect(ModelManifest.forInstalledEntry(atDir: entry) != nil)
    }

    private func waitUntilCleared(_ coord: DownloadCoordinator, _ key: String,
                                  timeoutMs: Int = 5000) async throws {
        var waited = 0
        while coord.downloads[key] != nil, waited < timeoutMs {
            try await Task.sleep(for: .milliseconds(20))
            waited += 20
        }
    }
}
