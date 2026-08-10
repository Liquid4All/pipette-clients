import Testing

@testable import Pipette

/// The run screen offers the whole catalog, not just what is on disk — a benchmark can be
/// started against a model that has not been fetched, and `RunCell.prepare` calls
/// `ensureModel` before the cell runs.
struct RunScreenCatalogTests {
    private func downloaded(_ entry: CatalogEntry, at path: String) -> DiscoveredModel {
        DiscoveredModel(source: entry.source, path: path, sizeBytes: 4096)
    }

    @Test func everyCatalogRowIsOfferedEvenWithNothingOnDisk() {
        let offered = NewJobView.offerable(downloaded: [])

        #expect(offered.count == CatalogEntry.catalog.count)
        #expect(offered.allSatisfy { !$0.isDownloaded })
    }

    /// A downloaded row and its catalog row are the same model, so the list must show one
    /// entry — and it has to be the downloaded one, which alone knows the real path and
    /// on-disk size. The catalog's figure is the declared download size.
    @Test func aDownloadedRowWinsItsCatalogDuplicate() throws {
        let entry = try #require(CatalogEntry.catalog.first)
        let offered = NewJobView.offerable(downloaded: [downloaded(entry, at: "/tmp/on-disk.gguf")])

        #expect(offered.count == CatalogEntry.catalog.count, "the duplicate is merged, not appended")
        let match = try #require(offered.first { $0.source == entry.source })
        #expect(match.isDownloaded)
        #expect(match.path == "/tmp/on-disk.gguf")
        #expect(match.sizeBytes == 4096)
    }

    /// SwiftUI keys rows by `id`. Catalog rows have no path, so an id derived from one
    /// would collapse the entire list into a single row.
    @Test func offeredRowsHaveDistinctIdentities() {
        let ids = Set(NewJobView.offerable(downloaded: []).map(\.id))
        #expect(ids.count == CatalogEntry.catalog.count)
    }

    /// A model downloaded for a claimed job carries the plan's `sha256`, which the
    /// catalog never states. Whole-`Model` equality would call those two different
    /// models and list the same weights twice — once "Downloaded", once not.
    @Test func aDownloadedRowWinsEvenWhenItCarriesIntegrityMetadata() throws {
        let entry = try #require(CatalogEntry.catalog.first { entry in
            if case .ggufText = entry.source { return true }
            return false
        })
        guard case let .ggufText(text) = entry.source,
              case let .huggingFace(repo, path, _) = text.source else {
            Issue.record("expected a huggingFace gguf_text catalog entry")
            return
        }
        let withSha = Model.ggufText(.init(source: .huggingFace(
            repo: repo, path: path, sha256: try Sha256(String(repeating: "a", count: 64)))))
        #expect(withSha != entry.source, "the premise: these are unequal as values")

        let offered = NewJobView.offerable(
            downloaded: [DiscoveredModel(source: withSha, path: "/tmp/x.gguf", sizeBytes: 1)])

        #expect(offered.count == CatalogEntry.catalog.count, "no duplicate row")
        #expect(offered.filter { $0.source.reference == entry.source.reference }.count == 1)
    }

    /// The hint marks what is present. Absent for a family with nothing on disk, because
    /// that is most of the catalog and a badge on every row says nothing.
    @Test func theHintMarksOnlyWhatIsOnDisk() throws {
        let entry = try #require(CatalogEntry.catalog.first)
        let onDisk = downloaded(entry, at: "/tmp/a.gguf")
        let absent = DiscoveredModel(catalog: entry)

        #expect(ModelGroup(key: "k", name: "n", paramLabel: nil, files: [absent]).downloadedHint == nil)
        #expect(ModelGroup(key: "k", name: "n", paramLabel: nil, files: [onDisk]).downloadedHint
                == "Downloaded")
        #expect(ModelGroup(key: "k", name: "n", paramLabel: nil, files: [onDisk, absent])
                .downloadedHint == "1 of 2 downloaded")
    }
}

/// A VL cell names both files through its source. Which coordinate that is depends on
/// how the projector was discovered — paired into the model's own entry, or as a
/// separate sideloaded row.
struct VisionCellSourceTests {
    private let projector = DiscoveredModel(
        source: ggufTextFixture("test/vl-GGUF", "mmproj-vl-F16.gguf"),
        path: "/tmp/mmproj-vl-F16.gguf", sizeBytes: 16)

    /// A `.ggufVision` coordinate already names its projector, so it is used as declared
    /// rather than re-derived from host paths — and it is one cell, not one per selected
    /// projector.
    @Test func aVisionCoordinateIsUsedAsDeclared() throws {
        let declared = Model.ggufVision(.init(source: .huggingFace(
            repo: try HFRepo.parse("test/vl-GGUF"), model: try RepoSubpath("vl-Q4_0.gguf"),
            modelSha256: nil, mmproj: try RepoSubpath("mmproj-vl-F16.gguf"), mmprojSha256: nil)))
        let model = DiscoveredModel(source: declared, path: "/tmp/vl-Q4_0.gguf", sizeBytes: 64)

        let sources = NewJobView.visionSources(for: model, mmprojFiles: [projector, projector])

        #expect(sources == [declared])
    }

    /// A sideloaded base has no vision coordinate, so the pair is named by both paths —
    /// one cell per selected projector.
    @Test func aSideloadedBasePairsWithEachSelectedProjector() throws {
        let model = DiscoveredModel(
            source: ggufTextFixture("test/vl-GGUF", "vl-Q4_0.gguf"),
            path: "/tmp/vl-Q4_0.gguf", sizeBytes: 64)

        let sources = NewJobView.visionSources(for: model, mmprojFiles: [projector])

        #expect(sources.count == 1)
        guard case let .ggufVision(vision) = try #require(sources.first),
              case let .absoluteFiles(weights, mmproj) = vision.source else {
            Issue.record("expected a .ggufVision/.absoluteFiles pair, got \(sources)")
            return
        }
        #expect(weights.value == "/tmp/vl-Q4_0.gguf")
        #expect(mmproj.value == "/tmp/mmproj-vl-F16.gguf")
    }

    /// Host paths can only name files that exist. A catalog row has none, so it yields
    /// no cell — rather than one that cannot name its own weights.
    @Test func anUndownloadedBaseYieldsNoPair() throws {
        let entry = try #require(CatalogEntry.catalog.first)
        let notOnDisk = DiscoveredModel(catalog: entry)

        #expect(NewJobView.visionSources(for: notOnDisk, mmprojFiles: [projector]).isEmpty)
    }
}
