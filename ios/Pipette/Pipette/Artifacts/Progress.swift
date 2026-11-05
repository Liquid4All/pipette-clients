import Foundation

// Fetch progress, after `crates/pipette-artifacts/src/progress.rs`.

/// The crate's `FetchProgress`, but `fraction` is stored rather than derived and the byte
/// counts are optional: an MLX directory pull reports a fraction and no bytes. There is no
/// `file` because the coordinator's per-model totals do not say which part is moving.
nonisolated struct FetchProgress: Sendable {
    /// `Model.description`.
    let artifact: String
    let fraction: Double
    let doneBytes: Int64?
    let totalBytes: Int64?
}

typealias ProgressSink = (FetchProgress) -> Void
