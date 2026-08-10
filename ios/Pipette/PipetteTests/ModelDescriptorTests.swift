import Foundation
import Testing

@testable import Pipette

/// The submitted `model_descriptor` — what the warehouse stores as this run's model
/// identity, and the only place a coordinate leaves the device.
///
/// The crate submits the whole `Model` (`results/record.rs` serializes
/// `without_auth_token()`), so anything the descriptor drops is a distinction the
/// warehouse cannot make. `revision` was dropped by all three arms.
struct ModelDescriptorTests {
    private func object(_ model: Model) throws -> [String: Any] {
        let json = try SubmissionRef.model(model)
        return try #require(
            try JSONSerialization.jsonObject(with: Data(json.utf8)) as? [String: Any])
    }

    private func pinned(_ slug: String, _ revision: String) throws -> HFRepo {
        var repo = try HFRepo.parse(slug)
        repo.revision = try HFRevision(revision)
        return repo
    }

    /// Two runs against different pins of one repo have to be distinguishable upstream.
    /// The storage key already treats the revision as identity; the descriptor did not,
    /// so the warehouse saw one model where the device saw two.
    @Test func aPinnedRevisionReachesTheDescriptor() throws {
        let model = Model.ggufText(GgufText(source: .huggingFace(
            repo: try pinned("org/repo", "v1.2.3"),
            path: try RepoSubpath("w.gguf"), sha256: nil)))

        #expect(try object(model)["revision"] as? String == "v1.2.3")
    }

    /// Absent, not null, matching the crate's `skip_serializing_if`.
    @Test func anUnpinnedModelOmitsTheKeyEntirely() throws {
        let model = Model.ggufText(GgufText(source: .huggingFace(
            repo: try HFRepo.parse("org/repo"),
            path: try RepoSubpath("w.gguf"), sha256: nil)))
        let json = try object(model)

        #expect(json["revision"] == nil)
        #expect(!json.keys.contains("revision"))
    }

    /// Every fetchable arm, because the pin was missing from all three and a per-arm
    /// spelling is what let that happen.
    @Test func everyFetchableArmCarriesTheCoordinate() throws {
        let repo = try pinned("org/repo", "v1")
        let models: [Model] = [
            .ggufText(GgufText(source: .huggingFace(
                repo: repo, path: try RepoSubpath("w.gguf"), sha256: nil))),
            .ggufVision(GgufVision(source: .huggingFace(
                repo: repo, model: try RepoSubpath("w.gguf"), modelSha256: nil,
                mmproj: try RepoSubpath("mm.gguf"), mmprojSha256: nil))),
            .mlx(Mlx(source: .huggingFace(repo: repo, prefix: try RepoSubpath("4bit")))),
        ]

        for model in models {
            let json = try object(model)
            #expect(json["org"] as? String == "org", "\(ModelType.of(model))")
            #expect(json["repo_name"] as? String == "repo", "\(ModelType.of(model))")
            #expect(json["revision"] as? String == "v1", "\(ModelType.of(model))")
            #expect(json["source"] as? String == "huggingface", "\(ModelType.of(model))")
        }
    }

    /// AFM ships with the OS: a bare tag, no coordinate to pin.
    @Test func appleFoundationStaysABareTag() throws {
        let json = try object(.appleFoundationText)

        #expect(json["type"] as? String == "apple_foundation_text")
        #expect(json.count == 1)
    }
}
