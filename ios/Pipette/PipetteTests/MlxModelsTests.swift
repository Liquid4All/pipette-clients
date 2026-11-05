import Foundation
import Testing

@testable import Pipette

/// `ResolvedMlxModel` and the shared `requireBoundModelDir` it calls: the crate's
/// `pipette-mlx/src/models.rs` over `pipette-ops/src/models.rs`. The cases under the mark
/// port the ops crate's own tests.
struct MlxModelsTests {
    /// Dispatch routes on the runtime and never inspects the model, so this is where a
    /// mismatched pair is refused, by both kinds.
    @Test func anotherFormatIsRefusedByName() throws {
        let gguf = try boundRequest(try ggufTextSpec("org/a-GGUF", "a.gguf"), "/tmp/a.gguf")

        #expect(throws: BoundModelDirError.wrongKind(expected: .mlx, got: .ggufText)) {
            _ = try ResolvedMlxModel(gguf.model.bound).dir
        }
    }

    /// MLX binds a bundle directory, so a file at that path is not it.
    @Test func anMlxBundleMustBeADirectory() throws {
        let dir = try makeFixtureDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let bundle = dir.appendingPathComponent("m", isDirectory: true)
        try FileManager.default.createDirectory(at: bundle, withIntermediateDirectories: true)
        _ = try writeFixtureFile(bundle, "config.json")
        let spec = try mlxSpec("org/m-MLX-4bit")

        #expect(try ResolvedMlxModel(boundRequest(spec, bundle.path).model.bound).dir == bundle.path)

        let asFile = try writeFixtureFile(dir, "not-a-bundle")
        #expect(throws: BoundModelDirError.notADirectory(expected: .mlx, path: asFile)) {
            _ = try ResolvedMlxModel(boundRequest(spec, asFile).model.bound).dir
        }
    }

    // MARK: - The ids that end a generation

    /// `mlx_lm` collects its stop ids from the tokenizer *and* the bundle's configs. A
    /// model whose chat template ends on a second id (`<|im_end|>` beside `</s>`) would
    /// otherwise run every eval sample to the token cap and record it as truncated.
    @Test func theEndIdsCombineTheTokenizerAndTheConfigs() throws {
        let dir = try makeFixtureDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        try Data(#"{"eos_token_id": 2}"#.utf8)
            .write(to: dir.appendingPathComponent("config.json"))
        try Data(#"{"eos_token_id": [2, 7]}"#.utf8)
            .write(to: dir.appendingPathComponent("generation_config.json"))

        #expect(MlxModels.eosTokenIds(bundle: dir.path, tokenizerEos: 1) == [1, 2, 7])
    }

    /// A bundle that names none leaves the tokenizer's own id standing, and a config that
    /// will not parse contributes nothing rather than failing the cell.
    @Test func theTokenizersOwnIdIsTheFloor() throws {
        let dir = try makeFixtureDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        try Data("not json".utf8).write(to: dir.appendingPathComponent("config.json"))

        #expect(MlxModels.eosTokenIds(bundle: dir.path, tokenizerEos: 1) == [1])
        #expect(MlxModels.eosTokenIds(bundle: dir.path, tokenizerEos: nil).isEmpty)
    }

    // MARK: - requireBoundModelDir: the crate's `pipette-ops/src/models.rs` cases

    @Test func aBoundDirectoryOfTheExpectedKindIsReturned() throws {
        let dir = try makeFixtureDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let model = try boundModel(try mlxSpec("org/m-MLX-4bit"), dir.path, nil)

        #expect(try requireBoundModelDir(model, expected: .mlx, requiredFiles: []) == dir.path)
    }

    /// Another engine's model reaching this one is a dispatch bug; the error has to name
    /// both kinds or it reads as a missing file.
    @Test func anotherKindIsRefusedByName() throws {
        let gguf = try ggufTextSpec("org/a-GGUF", "a.gguf")
        let error = #expect(throws: BoundModelDirError.self) {
            _ = try requireBoundModelDir(gguf, expected: .mlx, requiredFiles: [])
        }
        let message = error?.localizedDescription ?? ""
        #expect(message.contains("mlx") && message.contains("gguf_text"), "got \(message)")
    }

    /// The omission people actually hit is a partial snapshot, so the error names the file
    /// rather than calling the directory incomplete. The MLX loader reads `config.json`
    /// first, so a bundle without one is refused before anything loads.
    @Test func aMissingRequiredFileIsNamed() throws {
        let dir = try makeFixtureDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let spec = try mlxSpec("org/m-MLX-4bit")

        let error = #expect(throws: BoundModelDirError.self) {
            _ = try ResolvedMlxModel(boundRequest(spec, dir.path).model.bound).dir
        }
        #expect(error == .missingFile(expected: .mlx, dir: dir.path, file: "config.json"))
        #expect(error?.localizedDescription.contains("config.json") == true)
    }

    /// An unbound model means ensure/bind never ran, which must not look like an empty
    /// directory.
    @Test func anUnboundModelIsRefused() throws {
        let error = #expect(throws: BoundModelDirError.self) {
            _ = try requireBoundModelDir(try mlxSpec("org/m-MLX-4bit"), expected: .mlx,
                                         requiredFiles: [])
        }
        #expect(error?.localizedDescription.contains("bindUnder") == true)
    }
}
