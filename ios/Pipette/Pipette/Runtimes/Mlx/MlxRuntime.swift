import Foundation
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import MLXNN
import Tokenizers

/// MLX model lifecycle and memory breadcrumbs; the benchmarks are `MlxExecute`.
nonisolated enum MlxRuntime {
    /// Chunk size when a cell names none: MLX has no other load setting.
    static let defaultPrefillChunk: UInt32 = 512

    // MARK: - Milestone logging

    // `.info`, not `.debug`: permanent diagnostics of the memory ramp, visible without
    // debug output.

    static func footprintMB() -> Double {
        Double(ProcessMemory.physFootprintBytes()) / 1_048_576
    }

    static func logMem(_ label: String) {
        let mb = 1_048_576.0
        let fp = footprintMB()
        let active = Double(MLX.Memory.activeMemory) / mb
        let peak = Double(MLX.Memory.peakMemory) / mb
        AppLog.mlx.info(String(
            format: "%@ footprint=%.0fMB mlxActive=%.0fMB mlxPeak=%.0fMB",
            label, fp, active, peak))
    }

    // MARK: - Load / unload

    /// Pair with `releaseModel()`, or use `withFreshModel`.
    static func loadModel(path: String) async throws -> any LanguageModel {
        let dir = URL(fileURLWithPath: path)
        do {
            logMem("load.start \(dir.lastPathComponent)")
            let data = try Data(contentsOf: dir.appendingPathComponent("config.json"))
            let base = try JSONDecoder().decode(BaseConfiguration.self, from: data)
            let m = try await LLMTypeRegistry.shared.createModel(
                configuration: data, modelType: base.modelType)
            // `perLayerQuantization` supersedes the deprecated `quantization`. The async
            // overload keeps a multi-GB load from parking a cooperative thread.
            try await loadWeights(
                modelDirectory: dir, model: m,
                perLayerQuantization: base.perLayerQuantization
            )
            logMem("load.afterWeights")
            eval(m)
            logMem("load.afterEval")
            return m
        } catch {
            // Memoized by `JobExecutor` for the model's later cells.
            throw ModelLoadFailure(cause: .engine("MLX model load failed: \(error)"))
        }
    }

    /// A dropped model's buffers go to MLX's cache, not the OS; without this the next load
    /// stacks a second copy on top and the app is jetsam-killed.
    static func releaseModel() {
        MLX.Memory.clearCache()
    }

    static func withFreshModel<T>(path: String, _ body: (any LanguageModel) async throws -> T) async throws -> T {
        let model = try await loadModel(path: path)
        defer { releaseModel() }
        return try await body(model)
    }
}
