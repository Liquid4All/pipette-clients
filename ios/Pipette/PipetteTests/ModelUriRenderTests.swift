import Foundation
import Testing

@testable import Pipette

/// `ModelUri.uri(for:)` — the crate's `model_to_uri`, and the half that makes a listing
/// line usable: `models` prints this form, and `benchmarks run --model` reads it back.
///
/// The property under test is the round trip. A rendered URI that does not parse to the
/// model it came from would print a reference nothing can use.
struct ModelUriRenderTests {
    /// The canonical URIs `model_uri.rs`'s `round_trips` pins, for the schemes this client
    /// supports: parse → render must be byte-identical, so one model is one string
    /// whichever client wrote it. The crate's `url=` and `sha256=` cases are absent — this
    /// client refuses both on parse.
    @Test(arguments: [
        "gguf-text://repo=org/repo&path=Q4_K_M.gguf",
        "gguf-vision://repo=org/repo&model=m.gguf&mmproj=p.gguf",
        "mlx://repo=org/repo",
        "mlx://repo=org/repo&prefix=4bit&rev=v2",
    ])
    func renderingACanonicalUriIsByteIdentical(_ uri: String) throws {
        let model = try ModelUri.parse(uri)

        #expect(ModelUri.uri(for: model) == uri)
        #expect(try ModelUri.parse(#require(ModelUri.uri(for: model))) == model)
    }

    @Test func everyImportableArmRoundTrips() throws {
        let models: [Model] = [
            try ggufTextSpec("LiquidAI/LFM2-350M-GGUF", "LFM2-350M-Q4_K_M.gguf"),
            try ggufVisionSpec("LiquidAI/LFM2.5-VL-450M-GGUF", "vl-Q4_0.gguf", "mmproj-f16.gguf"),
            try mlxSpec("mlx-community/LFM2-350M-4bit"),
        ]

        for model in models {
            let uri = try #require(ModelUri.uri(for: model), "\(model.artifactName) should render")
            #expect(try ModelUri.parse(uri) == model, "round trip failed for \(uri)")
        }
    }

    /// A pinned revision is part of the coordinate, so it survives the trip.
    @Test func aPinnedRevisionSurvives() throws {
        var repo = try HFRepo.parse("LiquidAI/LFM2-350M-GGUF")
        repo.revision = try HFRevision("v1.2.3")
        let model = Model.ggufText(GgufText(source: .huggingFace(
            repo: repo, path: try RepoSubpath("m-Q4_0.gguf"), sha256: nil)))

        let uri = try #require(ModelUri.uri(for: model))
        #expect(uri.contains("rev=v1.2.3"))
        #expect(try ModelUri.parse(uri) == model)
    }

    /// One model renders one string, so a digest or a diff over the listing is stable.
    @Test func renderingIsDeterministic() throws {
        let model = try ggufVisionSpec("org/vl-GGUF", "vl.gguf", "proj.gguf")

        #expect(ModelUri.uri(for: model) == ModelUri.uri(for: model))
        // `model` before `mmproj`, the order `gguf_vision_to_uri` writes them, so both
        // clients render one model as one string.
        #expect(ModelUri.uri(for: model)
            == "gguf-vision://repo=org/vl-GGUF&model=vl.gguf&mmproj=proj.gguf")
    }

    /// What no URI can name on this client. `parse` refuses digests and `url=` sources, so
    /// rendering one would print a reference this client cannot read back; a store-relative
    /// arm is installed bytes rather than an importable coordinate; AFM ships with the OS.
    @Test func whatCannotBeNamedRendersNothing() throws {
        let pinned = Model.ggufText(GgufText(source: .huggingFace(
            repo: try HFRepo.parse("org/r"), path: try RepoSubpath("m.gguf"),
            sha256: try Sha256(String(repeating: "ab", count: 32)))))
        let stored = try #require(try ggufTextSpec("org/r", "m.gguf").toStored(base: "blobs"))
        let url = Model.ggufText(GgufText(source: .url(
            url: try ResourceUrl("https://example.com/m.gguf"), sha256: nil)))

        #expect(ModelUri.uri(for: pinned) == nil)
        #expect(ModelUri.uri(for: stored) == nil)
        #expect(ModelUri.uri(for: url) == nil)
        #expect(ModelUri.uri(for: .appleFoundationText) == nil)
    }
}
