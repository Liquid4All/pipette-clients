import Foundation
import Testing

@testable import Pipette

/// The store lookup `bench` relies on when the device holds more than one entry of a repo:
/// a pinned and an unpinned copy have identical artifact filenames, so only the coordinate
/// separates them.
@Suite(.serialized) @MainActor struct BenchModelResolutionTests {

    private func pinnedSpec(_ revision: String) throws -> Model {
        .ggufText(GgufText(source: .huggingFace(
            repo: HfRepo(org: HfOrg(validated: "LiquidAI"),
                         repoName: HfRepoName(validated: "LFM2.5-350M-GGUF"),
                         revision: HfRevision(validated: revision)),
            path: try RepoSubpath("LFM2.5-350M-Q4_0.gguf"), sha256: nil)))
    }

    /// Revision included, so a caller that keeps the resolved entry cannot land on the other.
    @Test func theStoreAnswersTheExactCoordinate() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let unpinned = try ggufTextSpec("LiquidAI/LFM2.5-350M-GGUF", "LFM2.5-350M-Q4_0.gguf")
        let pinned = try pinnedSpec("bb7ee58b243e4cede04187e323e760b04f8a0091")
        try installEntry(ws, unpinned)
        try installEntry(ws, pinned)

        let resolved = ws.availableModels().first { $0.source == pinned }

        #expect(resolved?.source == pinned)
        #expect(resolved?.source != unpinned)
        // And the descriptor a result would carry keeps the pin.
        let source = try #require(resolved?.source)
        #expect(try SubmissionRef.model(source)
            .contains("bb7ee58b243e4cede04187e323e760b04f8a0091"))
    }

    /// A run reports the model it was asked for. The manifest keeps the label of the fetch
    /// that created the entry and the lookup matches across labels, so the cell takes the
    /// request's: the label it named, or none.
    @Test func aRunReportsTheLabelItWasAskedWith() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        try installEntry(ws, try vaultTextSpec("vault_1", label: "first fetch"))

        for label in ["acme LFM3 q4", nil] {
            let requested = try vaultTextSpec("vault_1", label: label)
            let found = ws.availableModels().first { $0.source == requested }
            let entry = try #require(found)
            #expect(try reportedLabel(entry.source) == "first fetch")

            let carried = entry.asRequested(requested)
            let cells = JobCell.pending(benchmarkIds: ["prefill_throughput_512"], for: carried)
            let cell = try #require(cells.first)
            #expect(try reportedLabel(cell.source) == label)
            // The job manifest is where the cell waits for its upload.
            let saved = try JSONDecoder().decode(JobCell.self, from: JSONEncoder().encode(cell))
            #expect(try reportedLabel(saved.source) == label)
            // What the store measured stays the entry's.
            #expect(carried.path == entry.path)
            #expect(carried.sizeBytes == entry.sizeBytes)
        }
    }

    /// A claim's credential stops at the fetch.
    @Test func aRequestedTokenStaysOffTheCell() throws {
        var repo = try HfRepo.parseOrgRepo("LiquidAI/Gated-GGUF")
        let path = try RepoSubpath("Gated-Q4_0.gguf")
        let entry = DiscoveredModel(
            source: .ggufText(GgufText(source: .huggingFace(repo: repo, path: path, sha256: nil))),
            path: "/tmp/Gated-Q4_0.gguf", sizeBytes: 1)
        repo.authToken = try AuthToken("hf_secret")
        let requested = Model.ggufText(GgufText(source: .huggingFace(
            repo: repo, path: path, sha256: nil)))
        #expect(requested.authToken != nil)

        let carried = entry.asRequested(requested)
        let cells = JobCell.pending(benchmarkIds: ["prefill_throughput_512"], for: carried)
        let cell = try #require(cells.first)

        #expect(carried.source.authToken == nil)
        #expect(cell.source.authToken == nil)
    }
}
