import Foundation
import Testing

@testable import Pipette

// Fixtures for the per-engine bound-model checks (`LlamaModelsTests`, `MlxModelsTests`,
// `AFMModelsTests`): a declared model rewritten to the bound form a store hands back,
// pointed at paths a test controls.

/// The bound half a store would hand back: the same spec with its source rewritten to
/// the `absolute*` arms, as `Model.underRoot` yields. Built here rather than through the
/// store so a projection can be tested against paths that do or do not exist.
///
/// A vision fixture must state its projector: defaulting it to the weights would hand
/// back a path that exists, so a missing-projector test would pass for the wrong reason.
nonisolated func boundModel(_ declared: Model, _ path: String, _ mmproj: String?,
                            _ draft: String? = nil) throws -> Model {
    switch declared {
    case .ggufText:
        .ggufText(.init(source: .absoluteFile(path: try AbsolutePath(path))))
    case .ggufVision:
        .ggufVision(.init(source: .absoluteFiles(
            model: try AbsolutePath(path),
            mmproj: try AbsolutePath(try #require(mmproj, "vision fixture needs a projector")))))
    // The drafter stays optional here, unlike the projector: absent is a real arrangement
    // (a merged drafter), so defaulting it would make the two indistinguishable in a test.
    case .ggufVisionSpeculative:
        .ggufVisionSpeculative(.init(source: .absoluteFiles(
            model: try AbsolutePath(path),
            mmproj: try AbsolutePath(try #require(mmproj, "vision fixture needs a projector")),
            draft: try draft.map { try AbsolutePath($0) })))
    case .mlx:
        .mlx(.init(source: .absoluteDir(dir: try AbsolutePath(path))))
    case .onnxGenai:
        .onnxGenai(.init(source: .absoluteDir(dir: try AbsolutePath(path))))
    case .appleFoundationText:
        .appleFoundationText
    }
}

/// A decode cell for `declared`, bound to `path`: what an engine's model check is handed.
nonisolated func boundRequest(_ declared: Model, _ path: String = "/tmp/unused",
                              mmproj: String? = nil, draft: String? = nil) throws -> RunRequest {
    RunRequest(
        runtime: .alreadyBound(Runtime.thisBuild(for: declared)),
        model: DeclaredBound(declared: declared,
                             bound: try boundModel(declared, path, mmproj, draft)),
        runtimeFlags: nil,
        benchmarkFlags: nil,
        benchmark: .decodeThroughput(DecodeThroughputBenchmark(
            benchmarkId: BenchmarkId(validated: "decode_throughput_512_100"),
            parameterPrefillTokens: 512,
            parameterDecodeTokens: 100)))
}

/// A fresh scratch directory. The caller removes it.
nonisolated func makeFixtureDirectory() throws -> URL {
    let dir = FileManager.default.temporaryDirectory
        .appendingPathComponent("bound-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    return dir
}

/// A placeholder file named `name` in `dir`; returns its path.
nonisolated func writeFixtureFile(_ dir: URL, _ name: String) throws -> String {
    let url = dir.appendingPathComponent(name)
    try Data("gguf".utf8).write(to: url)
    return url.path
}
