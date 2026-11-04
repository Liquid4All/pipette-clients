import Testing

@testable import Pipette

/// The run screen offers the whole catalog, not just what is on disk: `RunCell.prepare`
/// fetches a missing model before the cell runs.
struct NewJobViewTests {
    // MARK: - The offered rows

    private func downloaded(_ entry: CatalogEntry, at path: String) -> DiscoveredModel {
        DiscoveredModel(source: entry.source, path: path, sizeBytes: 4096)
    }

    @Test func everyCatalogRowIsOfferedEvenWithNothingOnDisk() {
        let offered = NewJobView.offerable(downloaded: [])

        #expect(offered.count == CatalogEntry.catalog.count)
        #expect(offered.allSatisfy { !$0.isDownloaded })
    }

    /// Only the downloaded row knows the real path and on-disk size; the catalog's figure is
    /// the declared download size.
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

    /// A model downloaded for a claimed job carries the plan's `sha256`, which the catalog
    /// never states, so whole-`Model` equality would list the same weights twice.
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
        #expect(offered.filter { $0.source.description == entry.source.description }.count == 1)
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

    // MARK: - The vision sources

    private let projector = DiscoveredModel(
        source: ggufTextFixture("test/vl-GGUF", "mmproj-vl-F16.gguf"),
        path: "/tmp/mmproj-vl-F16.gguf", sizeBytes: 16)

    /// A `.ggufVision` coordinate already names its projector: one cell, not one per selected
    /// projector.
    @Test func aVisionCoordinateIsUsedAsDeclared() throws {
        let declared = Model.ggufVision(.init(source: .huggingFace(
            repo: try HfRepo.parseOrgRepo("test/vl-GGUF"), model: try RepoSubpath("vl-Q4_0.gguf"),
            modelSha256: nil, mmproj: try RepoSubpath("mmproj-vl-F16.gguf"), mmprojSha256: nil)))
        let model = DiscoveredModel(source: declared, path: "/tmp/vl-Q4_0.gguf", sizeBytes: 64)

        let sources = NewJobView.visionSources(for: model, mmprojFiles: [projector, projector])

        #expect(sources == [declared])
    }

    /// A sideloaded base has no vision coordinate, so each selected projector pairs with it
    /// by host path.
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

    /// Host paths name only files that exist, and a catalog row has none.
    @Test func anUndownloadedBaseYieldsNoPair() throws {
        let entry = try #require(CatalogEntry.catalog.first)
        let notOnDisk = DiscoveredModel(catalog: entry)

        #expect(NewJobView.visionSources(for: notOnDisk, mmprojFiles: [projector]).isEmpty)
    }
}
