import Foundation
import Testing

@testable import Pipette

/// The store lookup `bench` relies on when the device holds more than one entry of a repo.
///
/// A plan pins a revision; a device that also holds an unpinned copy has two entries whose
/// artifact *filenames* are identical, so only the coordinate separates them. This pins
/// that lookup — the run path that carries its answer through (`HeadlessRunner.run`) takes
/// no injectable seam, and is verified on a device instead.
@Suite(.serialized) @MainActor struct BenchModelResolutionTests {

    private func pinnedSpec(_ revision: String) throws -> Model {
        .ggufText(GgufText(source: .huggingFace(
            repo: HFRepo(org: HFOrg(validated: "LiquidAI"),
                         repoName: HFRepoName(validated: "LFM2.5-350M-GGUF"),
                         revision: HFRevision(validated: revision)),
            path: try RepoSubpath("LFM2.5-350M-Q4_0.gguf"), sha256: nil)))
    }

    /// The store answers the exact coordinate, revision included — so a caller that keeps
    /// the resolved entry cannot land on the other one.
    @Test func theStoreAnswersTheExactCoordinate() throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }
        let unpinned = try ggufTextSpec("LiquidAI/LFM2.5-350M-GGUF", "LFM2.5-350M-Q4_0.gguf")
        let pinned = try pinnedSpec("bb7ee58b243e4cede04187e323e760b04f8a0091")
        try installEntry(storage, unpinned)
        try installEntry(storage, pinned)

        let resolved = storage.availableModels().first { $0.source == pinned }

        #expect(resolved?.source == pinned)
        #expect(resolved?.source != unpinned)
        // And the descriptor a result would carry keeps the pin.
        let source = try #require(resolved?.source)
        #expect(try SubmissionRef.model(source)
            .contains("bb7ee58b243e4cede04187e323e760b04f8a0091"))
    }
}
