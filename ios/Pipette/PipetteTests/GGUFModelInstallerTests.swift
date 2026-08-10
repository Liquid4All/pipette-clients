import Foundation
import Testing

@testable import Pipette

/// The GGUF completion seam (`GGUFModelInstaller`): a finished single-file download is
/// relocated into its store entry's `blobs/` and recorded with the entry manifest, so
/// the next disk scan rebuilds the model with its provenance. Each test installs into
/// its own temporary `FileStorage`, so the suite carries no shared global and runs in
/// parallel.
@MainActor struct GGUFModelInstallerTests {
    private static let repo = "LiquidAI/LFM2.5-350M-GGUF"
    private static let filename = "LFM2.5-350M-Q4_0.gguf"
    private static let familyId = "lfm2.5-350m"

    private func downloadedFile() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".gguf")
        try Data("gguf".utf8).write(to: url)
        return url
    }

    private func manifest(_ storage: FileStorage, _ spec: Model) throws -> ModelManifest {
        let entry = try #require(storage.modelStore.entryDir(for: spec))
        return try #require(ModelManifest.forInstalledEntry(atDir: entry))
    }

    /// The exact typed `Model` is written into the entry manifest — no string
    /// reconstruction when a source is present.
    @Test func installWritesExactManifestFromSource() throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }

        let source = try ggufTextSpec(Self.repo, Self.filename)
        try GGUFModelInstaller(repo: Self.repo, filename: Self.filename, source: source, storage: storage)
            .install(from: try downloadedFile())

        let blobs = try #require(storage.modelStore.blobsDir(for: source))
        #expect(FileManager.default.fileExists(atPath: blobs.appendingPathComponent(Self.filename).path))
        #expect(try manifest(storage, source).declared == source)
    }

    /// The recorded manifest makes the model discoverable with its provenance
    /// (repo + family) derived back from the written spec.
    @Test func installMakesModelDiscoverableFromProvenance() throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }

        let source = try ggufTextSpec(Self.repo, Self.filename)
        try GGUFModelInstaller(repo: Self.repo, filename: Self.filename, source: source, storage: storage)
            .install(from: try downloadedFile())

        let model = try #require(storage.availableModels().first { $0.name == Self.filename })
        #expect(model.hfRepo == Self.repo)
        #expect(model.familyId == Self.familyId)
    }

    /// No typed source (sideload / legacy record): the entry is keyed on a text model
    /// reconstructed from the repo + filename strings, so the file is still
    /// discoverable rather than orphaned.
    @Test func installFallsBackToStringManifestWithoutSource() throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }

        try GGUFModelInstaller(repo: Self.repo, filename: Self.filename, source: nil, storage: storage)
            .install(from: try downloadedFile())

        let expected = try ggufTextSpec(Self.repo, Self.filename)
        #expect(try manifest(storage, expected).declared == expected)
    }

    /// With no coordinate at all there is no entry to store the file under, so the
    /// install refuses rather than dropping the file where nothing will find it.
    @Test func installRefusesWithoutAnyCoordinate() throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }

        #expect(throws: DownloadError.self) {
            try GGUFModelInstaller(repo: nil, filename: "sideload.gguf", source: nil, storage: storage)
                .install(from: try downloadedFile())
        }
    }

    /// A vision model is two transfers and one entry: both files land in the same
    /// `blobs/`, and either order publishes the same single manifest — the projector
    /// finishing first must not leave the entry manifest-less for the sweeper.
    @Test func visionFilesShareOneEntryWhicheverLandsFirst() throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }
        let source = try ggufVisionSpec("org/vl-GGUF", "vl-Q4_0.gguf", "mmproj-vl-F16.gguf")

        try GGUFModelInstaller(repo: "org/vl-GGUF", filename: "mmproj-vl-F16.gguf",
                               source: source, storage: storage).install(from: try downloadedFile())
        #expect(try manifest(storage, source).declared == source, "the projector publishes the entry")

        try GGUFModelInstaller(repo: "org/vl-GGUF", filename: "vl-Q4_0.gguf",
                               source: source, storage: storage).install(from: try downloadedFile())

        let entry = try #require(storage.modelStore.entryDir(for: source))
        let blobs = try FileManager.default.contentsOfDirectory(atPath: entry.appendingPathComponent("blobs").path)
        #expect(Set(blobs) == ["vl-Q4_0.gguf", "mmproj-vl-F16.gguf"])
        let entryFiles = try FileManager.default.contentsOfDirectory(atPath: entry.path)
        #expect(Set(entryFiles) == ["blobs", Entry.manifestName])
        #expect(storage.availableModels().filter { $0.name == "vl-Q4_0.gguf" }.count == 1)
    }
}
