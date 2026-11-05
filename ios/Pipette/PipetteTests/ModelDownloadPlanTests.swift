import Foundation
import Testing

@testable import Pipette

/// `ModelFetch.planDownloads` is pure, so no case needs a network, a store or a `URLSession`.
/// Ported from `crates/pipette-artifacts/src/model/fetch.rs`'s tests, plus the refusals
/// where this client differs from the crate: a local source, a directory kind and Apple
/// Foundation, which the crate plans.
@MainActor struct ModelDownloadPlanTests {
    /// `sha256("abc")`, as the crate's `SHA_A` is.
    private static let shaA = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    private static let shaB = "b5d4045c3f466fa91fe2cc6abe79232a1a57cdf104f7a26e716e0a1e2789df78"

    /// The entry the planning cases re-home into, the crate's `BASE` split into the entry
    /// root and the `blobs/` name the store appends.
    private static let entry = URL(fileURLWithPath: "/store/entry")

    private func dest(_ relative: String) -> String { "/store/entry/blobs/\(relative)" }

    /// The bound form the store hands the planner: the same `toStored` → `underRoot`
    /// composition `ModelArtifactStore.fetchDestination` applies, re-homed onto a fixed
    /// entry so these cases need no store.
    private func bound(_ declared: Model) throws -> Model {
        try #require(declared.toStored(base: Entry.blobsDirName)?.underRoot(Self.entry))
    }

    private func repo(revision: String? = nil, auth: String? = nil) throws -> HfRepo {
        var repo = try HfRepo.parseOrgRepo("meta/llama")
        repo.revision = try revision.map { try HfRevision($0) }
        repo.authToken = try auth.map { try AuthToken($0) }
        return repo
    }

    private func hfText(revision: String? = nil, auth: String? = nil,
                        sha256: String? = Self.shaA) throws -> Model {
        .ggufText(GgufText(source: .huggingFace(
            repo: try repo(revision: revision, auth: auth),
            path: try RepoSubpath("Q4.gguf"),
            sha256: try sha256.map { try Sha256($0) })))
    }

    private func onlyDownload(_ declared: Model) throws -> ModelFetch.Download {
        let downloads = try ModelFetch.planDownloads(declared, into: try bound(declared))
        #expect(downloads.count == 1, "expected a single download")
        return try #require(downloads.first)
    }

    @Test func hfTextResolvesToAResolveUrlWithAuthAndSha() throws {
        let download = try onlyDownload(try hfText(auth: "hf_tok"))

        #expect(download.url.value == "https://huggingface.co/meta/llama/resolve/main/Q4.gguf")
        #expect(download.auth?.value == "hf_tok")
        #expect(download.sha256?.value == Self.shaA)
        #expect(download.dest.value == dest("Q4.gguf"))
    }

    @Test func revisionIsUsedWhenPinnedElseMain() throws {
        for (revision, expected) in [(nil, "main"), ("v1.0", "v1.0")] as [(String?, String)] {
            let download = try onlyDownload(try hfText(revision: revision))

            #expect(download.url.value == "https://huggingface.co/meta/llama/resolve/\(expected)/Q4.gguf")
            #expect(download.auth == nil, "no token on a public repo")
        }
    }

    /// A `url` source passes its URL through untouched and carries no credential: there is
    /// no repo to have one.
    @Test func urlSourcePassesThroughWithoutAuth() throws {
        let declared = Model.ggufText(GgufText(source: .url(
            url: try ResourceUrl("https://ex.com/dir/w.gguf"), sha256: nil)))
        // A `url` arm has no store form, so the bound dest is named directly rather than
        // through `toStored` (see `ModelArtifactStore.fetchDestination`).
        let into = Model.ggufText(GgufText(source: .absoluteFile(
            path: try AbsolutePath(dest("w.gguf")))))

        let downloads = try ModelFetch.planDownloads(declared, into: into)

        #expect(downloads.count == 1)
        #expect(downloads.first?.url.value == "https://ex.com/dir/w.gguf")
        #expect(downloads.first?.auth == nil)
        #expect(downloads.first?.sha256 == nil)
        #expect(downloads.first?.dest.value == dest("w.gguf"))
    }

    @Test func hfVisionPlansModelThenMmproj() throws {
        let declared = Model.ggufVision(GgufVision(source: .huggingFace(
            repo: try repo(auth: "hf_tok"),
            model: try RepoSubpath("model.gguf"), modelSha256: try Sha256(Self.shaA),
            mmproj: try RepoSubpath("mmproj.gguf"), mmprojSha256: try Sha256(Self.shaB))))

        let downloads = try ModelFetch.planDownloads(declared, into: try bound(declared))

        #expect(downloads.map(\.url.value) == [
            "https://huggingface.co/meta/llama/resolve/main/model.gguf",
            "https://huggingface.co/meta/llama/resolve/main/mmproj.gguf",
        ])
        #expect(downloads.map(\.dest.value) == [dest("model.gguf"), dest("mmproj.gguf")])
        #expect(downloads.map { $0.sha256?.value } == [Self.shaA, Self.shaB] as [String?])
        #expect(downloads.allSatisfy { $0.auth?.value == "hf_tok" }, "the repo token rides both files")
    }

    @Test func urlVisionPlansBothFilesWithoutAuth() throws {
        let declared = Model.ggufVision(GgufVision(source: .url(
            model: try ResourceUrl("https://ex.com/model.gguf"), modelSha256: nil,
            mmproj: try ResourceUrl("https://ex.com/mmproj.gguf"), mmprojSha256: nil)))
        let into = Model.ggufVision(GgufVision(source: .absoluteFiles(
            model: try AbsolutePath(dest("model.gguf")),
            mmproj: try AbsolutePath(dest("mmproj.gguf")))))

        let downloads = try ModelFetch.planDownloads(declared, into: into)

        #expect(downloads.map(\.url.value) == ["https://ex.com/model.gguf", "https://ex.com/mmproj.gguf"])
        #expect(downloads.allSatisfy { $0.auth == nil }, "URL sources carry no token")
        #expect(downloads.map(\.dest.value) == [dest("model.gguf"), dest("mmproj.gguf")])
    }

    /// A declared source that is already a path on disk is an import, not a download, for
    /// both the absolute and the relative spelling and for both GGUF shapes.
    @Test func aLocalOrRelativeDeclaredSourceIsAnImport() throws {
        let into = try bound(try hfText())
        let visionInto = Model.ggufVision(GgufVision(source: .absoluteFiles(
            model: try AbsolutePath(dest("model.gguf")),
            mmproj: try AbsolutePath(dest("mmproj.gguf")))))
        let declared: [(Model, Model, String)] = [
            (.ggufText(GgufText(source: .absoluteFile(path: try AbsolutePath("/authored/w.gguf")))),
             into, "local/relative gguf-text is imported, not downloaded"),
            (.ggufText(GgufText(source: .relativeFile(path: try RelativePath("w.gguf")))),
             into, "local/relative gguf-text is imported, not downloaded"),
            (.ggufVision(GgufVision(source: .absoluteFiles(
                model: try AbsolutePath("/authored/model.gguf"),
                mmproj: try AbsolutePath("/authored/mmproj.gguf")))),
             visionInto, "local/relative gguf-vision is imported, not downloaded"),
            (.ggufVision(GgufVision(source: .relativeFiles(
                model: try RelativePath("model.gguf"), mmproj: try RelativePath("mmproj.gguf")))),
             visionInto, "local/relative gguf-vision is imported, not downloaded"),
        ]

        for (source, target, why) in declared {
            #expect(throws: ModelFetchError.notFetchable(why)) {
                try ModelFetch.planDownloads(source, into: target)
            }
        }
    }

    /// A directory kind needs a network repo listing, so the pure planner defers it to the
    /// fetcher (here, the HubApi pull). Upstream plans a Hugging Face directory as a
    /// snapshot part instead.
    @Test func aDirectoryKindIsTheFetchersToPlan() throws {
        let declared = try mlxSpec("meta/llama")
        let into = try bound(declared)

        #expect(throws: ModelFetchError.notFetchable(
            "directory snapshots need a network repo listing; planned by the fetcher, not here")) {
            try ModelFetch.planDownloads(declared, into: into)
        }
    }

    /// Apple Foundation ships with the OS, so there is nothing to fetch for it.
    @Test func appleFoundationHasNoFetchableFiles() throws {
        #expect(throws: ModelFetchError.notFetchable(
            "the Apple Foundation model has no fetchable files")) {
            try ModelFetch.planDownloads(.appleFoundationText, into: .appleFoundationText)
        }
    }

    /// A bound dest that is not an absolute local arm is refused: the planner will not
    /// invent a destination, and the two relative/remote spellings are named apart so the
    /// log says which one arrived.
    @Test func aBoundDestinationOfTheWrongShapeIsRefused() throws {
        let declared = try hfText()
        let relative = Model.ggufText(GgufText(source: .relativeFile(
            path: try RelativePath("blobs/Q4.gguf"))))

        #expect(throws: ModelFetchError.notFetchable(
            "gguf-text fetch dest must be abs Local, got Relative")) {
            try ModelFetch.planDownloads(declared, into: relative)
        }
        #expect(throws: ModelFetchError.notFetchable("bound gguf-text is not an Absolute source")) {
            try ModelFetch.planDownloads(declared, into: declared)
        }

        let vision = Model.ggufVision(GgufVision(source: .huggingFace(
            repo: try repo(), model: try RepoSubpath("model.gguf"), modelSha256: nil,
            mmproj: try RepoSubpath("mmproj.gguf"), mmprojSha256: nil)))
        #expect(throws: ModelFetchError.notFetchable("gguf-vision fetch dest must be abs Local")) {
            try ModelFetch.planDownloads(vision, into: vision)
        }
    }

    /// A declared kind and a bound kind that disagree are refused rather than silently
    /// planned against whichever one the code reached first.
    @Test func declaredAndBoundKindsThatDisagreeAreRefused() throws {
        let declared = try hfText()
        let into = Model.ggufVision(GgufVision(source: .absoluteFiles(
            model: try AbsolutePath(dest("model.gguf")),
            mmproj: try AbsolutePath(dest("mmproj.gguf")))))

        #expect(throws: ModelFetchError.notFetchable("declared and localized model kinds disagree")) {
            try ModelFetch.planDownloads(declared, into: into)
        }
    }
}
