import Foundation

/// What the client lends a run besides the request: the crate's `RunContext`
/// (`crates/pipette-ops/src/run_context.rs`). Injected like `ReadinessGate`, so an engine
/// asks for a dataset and never holds the server or the key that fetches it.
nonisolated protocol RunContext {
    /// The bytes `dataset`'s route serves, fetched as this device.
    func fetchDataset(_ dataset: Dataset) async throws -> Data
    /// Where a run unpacks what it fetches, one directory per benchmark.
    var datasetsDir: URL { get }
    /// Where an eval checkpoints its samples, so a killed run resumes.
    var evalCompletions: EvalCompletionsStore { get }
}
