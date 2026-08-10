import Foundation
import Testing

@testable import Pipette

/// `Model` Codable round-trips and matches the crate's tagged serde shape
/// (`type` discriminator + flattened `org`/`repo_name`), so a manifest written here
/// is readable by the same tooling.
struct ModelSpecTests {
    private func roundTrip(_ m: Model) throws -> Model {
        try JSONDecoder().decode(Model.self, from: try JSONEncoder().encode(m))
    }

    @Test func ggufTextMatchesCrateShapeAndRoundTrips() throws {
        let model = Model.ggufText(.init(source: .huggingFace(repo: try HFRepo.parse("LiquidAI/LFM2.5-350M-GGUF"), path: try RepoSubpath("LFM2.5-350M-Q4_K_M.gguf"), sha256: nil)))
        let data = try JSONEncoder().encode(model)
        let json = try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any])
        #expect(json["type"] as? String == "gguf_text")
        #expect(json["org"] as? String == "LiquidAI")
        #expect(json["repo_name"] as? String == "LFM2.5-350M-GGUF")
        #expect(json["path"] as? String == "LFM2.5-350M-Q4_K_M.gguf")
        #expect(try roundTrip(model) == model)
    }

    @Test func visionRoundTripsWithMmproj() throws {
        let model = Model.ggufVision(.init(source: .huggingFace(repo: try HFRepo.parse("LiquidAI/LFM2.5-VL-3B-GGUF"), model: try RepoSubpath("model-Q4_K_M.gguf"), modelSha256: nil, mmproj: try RepoSubpath("mmproj-f16.gguf"), mmprojSha256: nil)))
        #expect(try roundTrip(model) == model)
    }

    @Test func mlxRootAndSubpathRoundTrip() throws {
        let root = Model.mlx(.init(source: .huggingFace(repo: try HFRepo.parse("LiquidAI/LFM2.5-350M-MLX-4bit"), prefix: nil)))
        #expect(try roundTrip(root) == root)
        #expect(root.repo?.description == "LiquidAI/LFM2.5-350M-MLX-4bit")
        // A root MLX manifest omits `subpath` (matches the crate's bare HfMlx).
        let rootJSON = try #require(try JSONSerialization.jsonObject(with: JSONEncoder().encode(root)) as? [String: Any])
        #expect(rootJSON["prefix"] == nil)
        #expect(rootJSON["type"] as? String == "mlx")

        let sub = Model.mlx(.init(source: .huggingFace(repo: try HFRepo.parse("org/multi"), prefix: try RepoSubpath("4bit"))))
        #expect(try roundTrip(sub) == sub)
        #expect(sub.mlxModel?.ref?.leaf == "4bit")
    }

    @Test func decodesCrateTaggedJson() throws {
        let json = Data(#"{"type":"mlx","source":"huggingface","org":"LiquidAI","repo_name":"LFM2.5-350M-MLX-4bit"}"#.utf8)
        let model = try JSONDecoder().decode(Model.self, from: json)
        #expect(model == .mlx(.init(source: .huggingFace(repo: try HFRepo.parse("LiquidAI/LFM2.5-350M-MLX-4bit"), prefix: nil))))
    }

    @Test func rejectsUnknownType() {
        let json = Data(#"{"type":"hf_torch","org":"x","repo_name":"y"}"#.utf8)
        #expect(throws: ModelError.unknownModelType("hf_torch")) {
            try JSONDecoder().decode(Model.self, from: json)
        }
    }

    // MARK: - Apple Foundation (the bare case: no coordinate, no disk footprint)

    @Test func appleFoundationEncodesBareDiscriminatorAndRoundTrips() throws {
        let model = Model.appleFoundationText
        let data = try JSONEncoder().encode(model)
        let json = try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any])
        // A bare case: only the discriminator, no org/repo_name/filename/flags.
        #expect(json["type"] as? String == "apple_foundation_text")
        #expect(json["org"] == nil)
        #expect(json["repo_name"] == nil)
        #expect(json["path"] == nil)
        #expect(json.count == 1)
        #expect(try roundTrip(model) == model)
    }

    /// The digests a plan pins survive the round trip. Before the source enums existed
    /// there was nowhere to put them, so a pinned plan silently lost its integrity check.
    @Test func digestsRoundTripForEveryFetchingArm() throws {
        let repo = try HFRepo.parse("org/repo")
        let text = Model.ggufText(GgufText(source: .huggingFace(
            repo: repo, path: try RepoSubpath("w.gguf"),
            sha256: try Sha256(String(repeating: "a", count: 64)))))
        let vision = Model.ggufVision(GgufVision(source: .huggingFace(
            repo: repo,
            model: try RepoSubpath("w.gguf"),
            modelSha256: try Sha256(String(repeating: "b", count: 64)),
            mmproj: try RepoSubpath("mm.gguf"),
            mmprojSha256: try Sha256(String(repeating: "c", count: 64)))))

        for model in [text, vision] {
            let restored = try JSONDecoder().decode(Model.self, from: JSONEncoder().encode(model))
            #expect(restored == model)
        }
    }

    @Test func decodesAppleFoundationTaggedJson() throws {
        let json = Data(#"{"type":"apple_foundation_text"}"#.utf8)
        #expect(try JSONDecoder().decode(Model.self, from: json) == .appleFoundationText)
    }

    @Test func appleFoundationHasNoCoordinateOrQuant() {
        let model = Model.appleFoundationText
        #expect(model.repo == nil)
        #expect(model.quant == nil)
        #expect(model.familyId == "apple-foundation")
        #expect(model.engineLabel == "Apple Foundation")
        #expect(model.authToken == nil)
    }

    /// AFM has no path to bind, so the pair is the same spec twice — what `alreadyBound`
    /// names, and the reason it needs no separate located form.
    @Test func appleFoundationIsAlreadyBound() {
        let pair = DeclaredBound.alreadyBound(Model.appleFoundationText)
        #expect(pair.declared == .appleFoundationText)
        #expect(pair.bound == .appleFoundationText)
    }
}

private extension Model {
    var mlxModel: Mlx? { if case let .mlx(m) = self { m } else { nil } }
}

/// `DiscoveredModel.appleFoundation` — the built-in row that surfaces Apple's
/// on-device model as a selectable, non-deletable `DiscoveredModel`. It wraps the bare
/// `.appleFoundation` core (no faked path/size footprint); path/size read as empty
/// because there is no file, and the display fields are derived from the spec.
struct AppleFoundationDiscoveredModelTests {
    @Test func sentinelFactoryExposesBuiltInIdentity() {
        let file = DiscoveredModel.appleFoundation
        #expect(file.source == .appleFoundationText)
        #expect(file.source == .appleFoundationText)
        // UI-created cells submit this as `model_name`, matching the Stage-1 headless
        // path — not "".
        #expect(file.hfRepo == AFMRuntime.submissionModelName)
        #expect(file.quant == nil)
        #expect(file.familyId == "apple-foundation")
        #expect(file.name == "Apple Foundation")
        #expect(file.displayName == "Apple Foundation")
        // Sentinels: a built-in system model has no file on disk.
        #expect(file.path == "")
        #expect(file.sizeBytes == 0)
    }

    /// AFM has no file, so deleting it is a no-op that must not trap or touch storage
    /// (the UI also hides the delete affordance for it).
    @Test @MainActor func deleteModelIsANoOpForBuiltIn() {
        // AFM has no file, so this touches no storage — a throwaway instance is fine.
        makeTemporaryStorage().deleteModel(.appleFoundation)
    }
}
