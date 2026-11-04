import Foundation

/// Whether MLX shows in the UI: the MLX preset rows in the catalog and the MLX hint on the
/// New Job prefill-batch control. Gates visibility, not execution.
nonisolated enum MlxFeatureFlag {
    static let visibleInUI = true
}
