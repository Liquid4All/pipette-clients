import Foundation

/// Shared so every engine fails a cell the same way. No crate counterpart: each engine
/// crate fails through `anyhow`.
enum RuntimeError: LocalizedError, Equatable {
    case unsupported(String)
    case cancelled
    /// The device cannot run the cell as it is: the readiness gate timed out, the power state
    /// can move, or Apple's model is unavailable. Carries the observed condition.
    case readiness(String)
    case engine(String)
    /// The compute backend failed, and every later operation on this load fails with it.
    case backend(String)
    case outOfMemory(String)
    /// llama.cpp could not read the model file: missing, not a GGUF, or cut short.
    case unreadableModel(String)

    var errorDescription: String? {
        switch self {
        case .unsupported(let id):         return "Benchmark '\(id)' is not supported on this runtime."
        case .cancelled:                   return "Run cancelled."
        case .readiness(let observed):     return "Device not ready: \(observed)."
        case .engine(let detail):          return "Engine error: \(detail)."
        case .backend(let detail):         return "Engine backend failed: \(detail)."
        case .outOfMemory(let detail):     return "Out of memory: \(detail)."
        case .unreadableModel(let detail): return "Model file unreadable: \(detail)."
        }
    }
}

/// `JobExecutor` memoizes this failure, and nothing else, for the model's remaining cells:
/// a model that won't load won't load for the next cell either. No crate counterpart, as
/// the CLI runs each cell in its own process.
nonisolated struct ModelLoadFailure: Error, Equatable, LocalizedError {
    let cause: RuntimeError

    var errorDescription: String? { cause.errorDescription }
}
