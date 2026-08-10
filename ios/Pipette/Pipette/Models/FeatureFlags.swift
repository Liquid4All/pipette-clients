import Foundation

/// Feature flag: surface MLX models in the UI (the MLX preset rows in the Models
/// tab). MLX models download as HF-repo directories via
/// `DownloadCoordinator.startMLXDownload`. A neutral flag now that there is no
/// runtime picker — the New Job flow shows all downloaded models regardless of engine.
/// Lives with the model domain because it gates catalog visibility, not execution.
nonisolated enum MLXFeatureFlag {
    static let visibleInUI = true
}
