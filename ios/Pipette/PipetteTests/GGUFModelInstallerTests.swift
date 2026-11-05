import Foundation
import Testing

@testable import Pipette

/// A finished single-file download is relocated into its store entry's `blobs/` and
/// recorded with the entry manifest, so the next disk scan rebuilds the model with its
/// provenance.
@MainActor struct GGUFModelInstallerTests {
    private static let repo = "LiquidAI/LFM2.5-350M-GGUF"
    private static let filename = "LFM2.5-350M-Q4_0.gguf"
    private static let familyId = "lfm2.5-350m"

    private func downloadedFile() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".gguf")
        try Data("gguf".utf8).write(to: url)
        return url
    }

    private func manifest(_ ws: PipetteWorkspace, _ spec: Model) throws -> ModelManifest {
        let entry = try #require(ws.models.entryDir(for: spec))
        return try ModelArtifactStore.readManifest(at: ModelManifest.manifestURL(inEntryDir: entry))
    }

    /// No string reconstruction when a typed source is present.
    @Test func installWritesExactManifestFromSource() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }

        let source = try ggufTextSpec(Self.repo, Self.filename)
        try GGUFModelInstaller(repo: Self.repo, filename: Self.filename, source: source, ws: ws)
            .install(from: try downloadedFile())

        let blobs = try #require(ws.models.blobsDir(for: source))
        #expect(FileManager.default.fileExists(atPath: blobs.appendingPathComponent(Self.filename).path))
        #expect(try manifest(ws, source).declared == source)
    }

    @Test func installMakesModelDiscoverableFromProvenance() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }

        let source = try ggufTextSpec(Self.repo, Self.filename)
        try GGUFModelInstaller(repo: Self.repo, filename: Self.filename, source: source, ws: ws)
            .install(from: try downloadedFile())

        let model = try #require(ws.availableModels().first { $0.name == Self.filename })
        #expect(model.hfRepo == Self.repo)
        #expect(model.familyId == Self.familyId)
    }

    /// No typed source (sideload / legacy record): the entry is keyed on a text model
    /// reconstructed from the repo + filename strings, so the file is still
    /// discoverable rather than orphaned.
    @Test func installFallsBackToStringManifestWithoutSource() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }

        try GGUFModelInstaller(repo: Self.repo, filename: Self.filename, source: nil, ws: ws)
            .install(from: try downloadedFile())

        let expected = try ggufTextSpec(Self.repo, Self.filename)
        #expect(try manifest(ws, expected).declared == expected)
    }

    /// With no coordinate at all there is no entry to store the file under, so the
    /// install refuses rather than dropping the file where nothing will find it.
    @Test func installRefusesWithoutAnyCoordinate() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }

        #expect(throws: DownloadError.self) {
            try GGUFModelInstaller(repo: nil, filename: "sideload.gguf", source: nil, ws: ws)
                .install(from: try downloadedFile())
        }
    }

    /// A repo that publishes its quants in a subdirectory: the transfer knows only the
    /// URL's last component, while the manifest records `quants/...`. The file has to
    /// land where the manifest says, or the store reads the entry as a husk and fetches
    /// it again.
    @Test func aNestedRepoPathInstallsWhereTheManifestRecordsIt() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let source = try ggufTextSpec(Self.repo, "quants/\(Self.filename)")

        try GGUFModelInstaller(repo: Self.repo, filename: Self.filename, source: source,
                               ws: ws).install(from: try downloadedFile())

        let blobs = try #require(ws.models.blobsDir(for: source))
        #expect(FileManager.default.fileExists(
            atPath: blobs.appendingPathComponent("quants/\(Self.filename)").path))
        #expect(ws.models.bound(source) != nil, "the store resolves what it recorded")
    }

    /// A vision model is two transfers and one entry: both files land in the same
    /// `blobs/`, and either order publishes the same single manifest: the projector
    /// finishing first must not leave the entry manifest-less for the sweeper.
    @Test func visionFilesShareOneEntryWhicheverLandsFirst() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let source = try ggufVisionSpec("org/vl-GGUF", "vl-Q4_0.gguf", "mmproj-vl-F16.gguf")

        try GGUFModelInstaller(repo: "org/vl-GGUF", filename: "mmproj-vl-F16.gguf",
                               source: source, ws: ws).install(from: try downloadedFile())
        #expect(try manifest(ws, source).declared == source, "the projector publishes the entry")

        try GGUFModelInstaller(repo: "org/vl-GGUF", filename: "vl-Q4_0.gguf",
                               source: source, ws: ws).install(from: try downloadedFile())

        let entry = try #require(ws.models.entryDir(for: source))
        let blobs = try FileManager.default.contentsOfDirectory(atPath: entry.appendingPathComponent("blobs").path)
        #expect(Set(blobs) == ["vl-Q4_0.gguf", "mmproj-vl-F16.gguf"])
        let entryFiles = try FileManager.default.contentsOfDirectory(atPath: entry.path)
        #expect(Set(entryFiles) == ["blobs", Entry.manifestName])
        #expect(ws.availableModels().filter { $0.name == "vl-Q4_0.gguf" }.count == 1)
    }

    /// Bytes that do not match the digest the plan carried fail the download with the
    /// crate's `Sha256Mismatch` wording, and nothing lands in the store entry: the check
    /// runs before the relocate, which is the only point on this client where the bytes
    /// are still outside the entry.
    @Test func mismatchedBytesFailAndDoNotLandInTheStore() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let source = try ggufTextSpec(Self.repo, Self.filename)
        let blobs = try #require(ws.models.blobsDir(for: source))
        let dest = blobs.appendingPathComponent(Self.filename)

        // `downloadedFile` writes "gguf"; the digest declared here is sha256("abc").
        let expected = try Sha256("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
        let downloaded = try downloadedFile()
        let actual = try ModelFetch.sha256Hex(ofFileAt: downloaded)
        let installer = GGUFModelInstaller(repo: Self.repo, filename: Self.filename,
                                           source: source, sha256: expected, ws: ws)

        let error = #expect(throws: ModelFetchError.self) {
            try installer.install(from: downloaded)
        }
        #expect(error == .sha256Mismatch(dest: dest.path, expected: expected.value, actual: actual))
        #expect(!FileManager.default.fileExists(atPath: dest.path))
        #expect(try ws.models.find(source) == nil, "no manifest published for a refused download")
    }

    /// The transfer only knows the leaf (`enqueueFileDownload` takes
    /// `url.lastPathComponent`), so joining `blobs/<filename>` would miss where `toStored`
    /// records the file and `bound` would fail to resolve an intact entry.
    @Test func aNestedRepoPathLandsWhereTheManifestRecordsIt() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let source = try ggufTextSpec(Self.repo, "quants/\(Self.filename)")

        try GGUFModelInstaller(repo: Self.repo, filename: Self.filename, source: source,
                               ws: ws).install(from: try downloadedFile())

        let blobs = try #require(ws.models.blobsDir(for: source))
        #expect(FileManager.default.fileExists(
            atPath: blobs.appendingPathComponent("quants/\(Self.filename)").path))
        #expect(ws.models.bound(source) != nil, "the entry resolves after install")
    }

    /// The digest a plan carries is checked, not merely carried: bytes that match install
    /// exactly as an unverified download does.
    @Test func matchingBytesInstallNormally() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let source = try ggufTextSpec(Self.repo, Self.filename)
        let downloaded = try downloadedFile()
        let digest = try Sha256(try ModelFetch.sha256Hex(ofFileAt: downloaded))

        try GGUFModelInstaller(repo: Self.repo, filename: Self.filename, source: source,
                               sha256: digest, ws: ws).install(from: downloaded)

        let blobs = try #require(ws.models.blobsDir(for: source))
        #expect(FileManager.default.fileExists(atPath: blobs.appendingPathComponent(Self.filename).path))
        #expect(try manifest(ws, source).declared == source)
    }

    // MARK: - Adopting the bytes before hashing

    /// `adopt` takes the bytes out of the session's temp path without reading them, which
    /// is what lets the hash run off the main actor: the delegate can return, and its temp
    /// file can be deleted, while the install is still going.
    @Test func adoptMovesTheBytesOutOfTheSessionsTempPath() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let spec = try ggufTextSpec("org/a-GGUF", "a-Q4_0.gguf")
        let temp = FileManager.default.temporaryDirectory
            .appendingPathComponent("session-\(UUID().uuidString).gguf")
        try Data("weights".utf8).write(to: temp)
        let installer = GGUFModelInstaller(repo: "org/a-GGUF", filename: "a-Q4_0.gguf",
                                           source: spec, sha256: nil, ws: ws)

        let adopted = try installer.adopt(from: temp)

        #expect(!FileManager.default.fileExists(atPath: temp.path),
                "the session's file must be moved, not copied")
        #expect(FileManager.default.fileExists(atPath: adopted.path))
        #expect(adopted.path.contains(Entry.stagingDirName),
                "adopted bytes live where the sweep can reclaim them")
        #expect(try ws.models.find(spec) == nil, "adopting publishes nothing")
    }

    /// The two halves compose into the install the one-shot form does, and the scratch
    /// directory does not survive it.
    @Test func adoptThenFinishInstallsAndCleansUpItsScratch() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let spec = try ggufTextSpec("org/a-GGUF", "a-Q4_0.gguf")
        let temp = FileManager.default.temporaryDirectory
            .appendingPathComponent("session-\(UUID().uuidString).gguf")
        try Data("weights".utf8).write(to: temp)
        let installer = GGUFModelInstaller(repo: "org/a-GGUF", filename: "a-Q4_0.gguf",
                                           source: spec, sha256: nil, ws: ws)

        let adopted = try installer.adopt(from: temp)
        try installer.finish(adopted: adopted)

        #expect(ws.models.bound(spec) != nil)
        #expect(!FileManager.default.fileExists(atPath: adopted.deletingLastPathComponent().path),
                "the scratch directory is cleaned up")
    }

    /// Hashing off the main actor still refuses a mismatch and publishes nothing.
    @Test func aDigestMismatchStillRefusesAfterTheSplit() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let spec = try ggufTextSpec("org/a-GGUF", "a-Q4_0.gguf")
        let temp = FileManager.default.temporaryDirectory
            .appendingPathComponent("session-\(UUID().uuidString).gguf")
        try Data("weights".utf8).write(to: temp)
        let wrong = try Sha256(String(repeating: "a", count: 64))
        let installer = GGUFModelInstaller(repo: "org/a-GGUF", filename: "a-Q4_0.gguf",
                                           source: spec, sha256: wrong, ws: ws)

        let adopted = try installer.adopt(from: temp)
        #expect(throws: ModelFetchError.self) { try installer.finish(adopted: adopted) }

        #expect(ws.models.bound(spec) == nil, "nothing may be installed")
    }
}
