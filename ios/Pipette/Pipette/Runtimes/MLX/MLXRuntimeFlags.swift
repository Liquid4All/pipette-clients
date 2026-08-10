import Foundation

// The MLX counterpart of `LlamaRuntimeFlags`. The `pipette-mlx` crate has
// no flags module — a desktop MLX cell derives nothing — so the shape is borrowed from
// `pipette-llamacpp`, which is the crate this engine's one setting follows.

nonisolated enum MLXRuntimeFlags {
    /// The flags this cell runs with: the plan's, with the prefill chunk this engine
    /// supplies where the cell left it unset. MLX has no other load setting, so the two
    /// llama knobs stay absent — an MLX variant carries neither, and reporting one would
    /// look authored.
    static func forRun(_ req: RunRequest) throws -> RuntimeFlags {
        var r = try req.runtimeFlagsRef()
        r.nUbatch = r.nUbatch ?? MLXRuntime.defaultPrefillChunk
        return try r.resolve()
    }
}
