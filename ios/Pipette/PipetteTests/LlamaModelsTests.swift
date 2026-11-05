import Foundation
import Testing

@testable import Pipette

/// The llama engine's `Resolved*` model types, the crate's `pipette-llamacpp/src/models.rs`:
/// each accepts one plan model kind and checks its files are on disk. A half-finished ensure
/// or bind would otherwise reach the engine as a load failure naming a file no one wrote.
struct LlamaModelsTests {
    /// A bound vision model, plain or speculative, its files written under `dir`.
    private func visionModel(_ dir: URL, speculative: Bool, separateDrafter: Bool = false)
        throws -> Model {
        let model = try writeFixtureFile(dir, "model.gguf")
        let mmproj = try writeFixtureFile(dir, "mmproj.gguf")
        let draft = separateDrafter ? try writeFixtureFile(dir, "draft.gguf") : nil
        let declared = speculative
            ? try ggufVisionSpeculativeSpec("org/vl-GGUF", "model.gguf", "mmproj.gguf",
                                            draft: separateDrafter ? "draft.gguf" : nil)
            : try ggufVisionSpec("org/vl-GGUF", "model.gguf", "mmproj.gguf")
        return try boundModel(declared, model, mmproj, draft)
    }

    @Test func resolvedGgufTextModelRejectsWrongVariant() throws {
        let mlx = try boundModel(try mlxSpec("org/m-MLX-4bit"), "/tmp/m", nil)
        #expect(throws: UnexpectedBoundModel.self) { try ResolvedGgufTextModel(mlx) }
    }

    /// A manifest can name weights a sweep has since reclaimed. The engine hears about it
    /// here, by path, rather than from llama.cpp.
    @Test(arguments: [true, false])
    func resolvedGgufTextModelChecksTheFile(_ present: Bool) throws {
        let dir = try makeFixtureDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let path = present ? try writeFixtureFile(dir, "model.gguf")
            : dir.appendingPathComponent("model.gguf").path
        let model = try boundModel(try ggufTextSpec("org/a-GGUF", "model.gguf"), path, nil)
        if present {
            #expect(try ResolvedGgufTextModel(model).path == path)
        } else {
            #expect(throws: BoundPathMissing(what: "GGUF text model", path: path)) {
                try ResolvedGgufTextModel(model)
            }
        }
    }

    @Test func resolvedGgufVisionModelTakesOnlyAPlainModel() throws {
        let dir = try makeFixtureDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let resolved = try ResolvedGgufVisionModel(try visionModel(dir, speculative: false))
        #expect(resolved.backbone == dir.appendingPathComponent("model.gguf").path)
        #expect(resolved.mmproj == dir.appendingPathComponent("mmproj.gguf").path)
        let speculative = try visionModel(dir, speculative: true)
        #expect(throws: UnexpectedBoundModel.self) { try ResolvedGgufVisionModel(speculative) }
    }

    /// Only a drafter shipped as its own gguf fills `draft`.
    @Test(arguments: [true, false])
    func resolvedGgufVisionSpeculativeModelTakesOnlyASpeculativeModel(_ separateDrafter: Bool)
        throws {
        let dir = try makeFixtureDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let resolved = try ResolvedGgufVisionSpeculativeModel(
            try visionModel(dir, speculative: true, separateDrafter: separateDrafter))
        #expect(resolved.vision.backbone == dir.appendingPathComponent("model.gguf").path)
        #expect(resolved.draft
            == (separateDrafter ? dir.appendingPathComponent("draft.gguf").path : nil))
        let plain = try visionModel(dir, speculative: false)
        #expect(throws: UnexpectedBoundModel.self) {
            try ResolvedGgufVisionSpeculativeModel(plain)
        }
    }

    /// A projector the manifest names but disk does not have. The `absoluteFiles` arm makes
    /// the *absent* projector unrepresentable, so what is left to catch is a path that no
    /// longer resolves, and it is caught by name.
    @Test func aVisionModelWithNoProjectorOnDiskIsRefused() throws {
        let dir = try makeFixtureDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let declared = try ggufVisionSpec("org/vl-GGUF", "vl-Q4_0.gguf", "mmproj-f16.gguf")
        let gone = dir.appendingPathComponent("mmproj-f16.gguf").path
        let model = try boundModel(declared, try writeFixtureFile(dir, "vl-Q4_0.gguf"), gone)

        #expect(throws: BoundPathMissing(what: "GGUF mmproj", path: gone)) {
            try ResolvedGgufVisionModel(model)
        }
    }
}
