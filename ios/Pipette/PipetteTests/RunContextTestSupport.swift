import Foundation

@testable import Pipette

/// A `RunContext` that checkpoints evals under `root` and has no corpus to lend: these
/// cells pull none.
nonisolated struct NoCorpus: RunContext {
    let root: URL

    func fetchDataset(_ dataset: Dataset) async throws -> Data {
        throw RuntimeError.engine("this test lends no corpus")
    }

    var datasetsDir: URL { root.appendingPathComponent("datasets", isDirectory: true) }

    var evalCompletions: EvalCompletionsStore {
        EvalCompletionsStore(root: root.appendingPathComponent("evals", isDirectory: true))
    }
}

/// A fresh eval checkpoint in its own scratch directory, for a cell driven without a
/// request to digest.
nonisolated func scratchCheckpoint() throws -> EvalCompletionSession {
    try EvalCompletionSession(
        root: FileManager.default.temporaryDirectory
            .appendingPathComponent("evals-\(UUID().uuidString)", isDirectory: true),
        digest: EvalRunDigest(value: String(repeating: "0", count: 64)),
        meta: EvalCompletionMeta(benchmarkId: "eval", runtime: "test", model: "test"))
}
