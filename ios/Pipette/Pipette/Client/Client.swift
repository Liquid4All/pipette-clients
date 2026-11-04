import Foundation

// The teardown `auth reset` and a forced `auth import` share: the crate's
// `pipette-cli/src/client/mod.rs`.

/// Wipe the caches derived from the server relationship (synced catalog, remote results,
/// eval state): the crate's `clear_remote_state`. Local benchmarks and results are kept.
///
/// Kept past a reset, a pending result would be submitted under the next identity's
/// `client_id` on the first `sync`.
nonisolated func clearRemoteState(
    benchmarks: BenchmarkStore,
    results: ResultsStore,
    evals: EvalCompletionsStore
) throws {
    benchmarks.clearRemote()
    for dir in [results.locationDir(.remotePending), results.locationDir(.remoteSynced)] {
        do {
            try FileManager.default.removeItem(at: dir)
        } catch CocoaError.fileNoSuchFile {
            // Nothing was ever filed there.
        }
    }
    try evals.clear()
}
