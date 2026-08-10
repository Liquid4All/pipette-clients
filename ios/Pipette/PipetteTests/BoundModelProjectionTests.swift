import Foundation
import Testing

@testable import Pipette

/// `LlamaModels` / `MLXModels` / `AFMModels` — the bound-model check each engine does
/// before it loads, the crate's per-engine `models.rs` (`require_gguf_text`,
/// `require_gguf_vision`, `require_mlx_model_dir`).
///
/// Two guarantees per call, and both matter: the arm is this engine's, and the path is
/// really on disk. A half-finished ensure or bind would otherwise reach the engine and
/// surface as a load failure naming a file no one wrote. Since dispatch routes on the
/// bound runtime alone, these are also the *only* place a model/runtime mismatch is
/// caught — upstream's arrangement.
struct BoundModelProjectionTests {
    /// The bound half a store would hand back: the same spec with its source rewritten to
    /// the `absolute*` arms, which is what `Model.bindUnder` yields and what `require*`
    /// matches. Built here rather than through the store so a projection can be tested
    /// against paths that do or do not exist.
    ///
    /// A vision fixture must state its projector. Defaulting it to the weights would hand
    /// back a path that exists, so a test meaning to exercise the missing-projector case
    /// would pass for the wrong reason — the same aliasing the `absoluteFiles` arm removed
    /// from production.
    private func bound(_ declared: Model, _ path: String, _ mmproj: String?) throws -> Model {
        switch declared {
        case .ggufText:
            .ggufText(.init(source: .absoluteFile(path: try AbsolutePath(path))))
        case .ggufVision:
            .ggufVision(.init(source: .absoluteFiles(
                model: try AbsolutePath(path),
                mmproj: try AbsolutePath(try #require(mmproj, "vision fixture needs a projector")))))
        case .mlx:
            .mlx(.init(source: .absoluteDir(dir: try AbsolutePath(path))))
        case .appleFoundationText:
            .appleFoundationText
        }
    }

    private func request(_ declared: Model, _ path: String = "/tmp/unused",
                         mmproj: String? = nil) throws -> RunRequest {
        RunRequest(
            runtime: .alreadyBound(Runtime.thisBuild(for: declared)),
            model: DeclaredBound(declared: declared,
                                 bound: try bound(declared, path, mmproj)),
            runtimeFlags: nil,
            benchmarkFlags: nil,
            benchmark: .decodeThroughput(benchmarkId: "decode_throughput_512_100",
                                         prefillTokens: 512, decodeTokens: 100))
    }

    private func temporaryDirectory() throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("bound-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    private func file(_ dir: URL, _ name: String) throws -> String {
        let url = dir.appendingPathComponent(name)
        try Data("gguf".utf8).write(to: url)
        return url.path
    }

    @Test func aBoundTextModelProjectsItsWeights() throws {
        let dir = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let path = try file(dir, "a-Q4_0.gguf")
        let declared = try ggufTextSpec("org/a-GGUF", "a-Q4_0.gguf")

        #expect(try LlamaModels.requireGgufText(request(declared, path)) == path)
    }

    @Test func aBoundVisionModelProjectsBothFiles() throws {
        let dir = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let weights = try file(dir, "vl-Q4_0.gguf")
        let mmproj = try file(dir, "mmproj-f16.gguf")
        let declared = try ggufVisionSpec("org/vl-GGUF", "vl-Q4_0.gguf", "mmproj-f16.gguf")

        let projected = try LlamaModels.requireGgufVision(
            request(declared, weights, mmproj: mmproj))
        #expect(projected.model == weights)
        #expect(projected.mmproj == mmproj)
    }

    /// A projector the manifest names but disk does not have. The `absoluteFiles` arm
    /// makes the *absent* projector unrepresentable — both paths or no arm — so what is
    /// left to catch is a path that no longer resolves, and it is caught by name.
    @Test func aVisionModelWithNoProjectorOnDiskIsRefused() throws {
        let dir = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let declared = try ggufVisionSpec("org/vl-GGUF", "vl-Q4_0.gguf", "mmproj-f16.gguf")
        let gone = dir.appendingPathComponent("mmproj-f16.gguf").path

        #expect(throws: BoundPathMissing(what: "GGUF mmproj", path: gone)) {
            _ = try LlamaModels.requireGgufVision(
                request(declared, try file(dir, "vl-Q4_0.gguf"), mmproj: gone))
        }
    }

    /// A manifest can name weights a sweep has since reclaimed. The engine hears about it
    /// here, by path, rather than from llama.cpp.
    @Test func weightsThatAreNotOnDiskAreRefused() throws {
        let declared = try ggufTextSpec("org/a-GGUF", "a-Q4_0.gguf")

        #expect(throws: BoundPathMissing(what: "GGUF text model",
                                         path: "/nowhere/a-Q4_0.gguf")) {
            _ = try LlamaModels.requireGgufText(request(declared, "/nowhere/a-Q4_0.gguf"))
        }
    }

    /// A text benchmark requires the *text* arm, as every text execute module upstream
    /// does. A vision model on one is refused here, matching the flags layer — plan-types
    /// gives gguf_vision a variant for `vl_throughput` alone.
    @Test func aVisionModelOnATextBenchmarkIsRefused() throws {
        let dir = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let declared = try ggufVisionSpec("org/vl-GGUF", "vl-Q4_0.gguf", "mmproj-f16.gguf")
        let vision = try request(declared, try file(dir, "vl-Q4_0.gguf"),
                                 mmproj: try file(dir, "mmproj-f16.gguf"))

        #expect(throws: UnexpectedBoundModel.self) {
            _ = try LlamaModels.requireGgufText(vision)
        }
    }

    /// Dispatch routes on the runtime and never inspects the model, so this is where a
    /// mismatched pair is refused. Each engine rejects what it cannot load, by name.
    @Test func anotherFormatIsRefusedByName() throws {
        let mlx = try request(try mlxSpec("org/m-MLX-4bit"), "/tmp/m")
        let gguf = try request(try ggufTextSpec("org/a-GGUF", "a.gguf"), "/tmp/a.gguf")

        #expect(throws: UnexpectedBoundModel.self) {
            _ = try LlamaModels.requireGgufText(mlx)
        }
        #expect(throws: UnexpectedBoundModel.self) {
            _ = try MLXModels.requireMlxModelDir(gguf)
        }
        // AFM has no path to project, so this pairing check is all it has — and without it
        // the runtime-routed dispatch would hand Apple Foundation a downloaded model.
        #expect(throws: UnexpectedBoundModel.self) {
            try AFMModels.requireAppleFoundation(gguf)
        }
        #expect(throws: Never.self) {
            try AFMModels.requireAppleFoundation(try request(.appleFoundationText))
        }
    }

    /// MLX binds a bundle directory, so a file at that path is not it.
    @Test func anMlxBundleMustBeADirectory() throws {
        let dir = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let bundle = dir.appendingPathComponent("m", isDirectory: true)
        try FileManager.default.createDirectory(at: bundle, withIntermediateDirectories: true)
        let spec = try mlxSpec("org/m-MLX-4bit")

        #expect(try MLXModels.requireMlxModelDir(request(spec, bundle.path)) == bundle.path)

        let asFile = try file(dir, "not-a-bundle")
        #expect(throws: BoundPathMissing(what: "MLX model directory", path: asFile)) {
            _ = try MLXModels.requireMlxModelDir(request(spec, asFile))
        }
    }
}
