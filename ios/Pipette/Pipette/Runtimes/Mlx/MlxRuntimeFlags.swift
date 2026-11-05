import Foundation

// The MLX counterpart of `LlamaRuntimeFlags`. On desktop only the `vl_spec_decode` cell
// derives a flag (`speculative_enabled`), and iOS MLX refuses that kind, so the prefill
// chunk is the one flag filled in here. `pipette-mlx` has no flags module, so the shape
// follows `pipette-llamacpp`.

nonisolated enum MlxRuntimeFlags {
    /// Only the prefill chunk is filled in: other knobs reported here would look authored.
    static func forRun(_ req: RunRequest) throws -> RuntimeFlags {
        var r = try req.runtimeFlagsRef()
        r.nUbatch = r.nUbatch ?? MlxRuntime.defaultPrefillChunk
        return try r.resolve()
    }
}
