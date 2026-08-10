import Foundation

/// Translate a job-execution error into text for `JobCell.errorMessage`. The
/// native runtimes (llama.cpp / MLX) surface plain `Error`s whose
/// `localizedDescription` is already user-readable, so this mostly unwraps that;
/// an OOM gets the actionable smaller-quant/model hint (the engine classifies it
/// from the captured llama.cpp log).
nonisolated func formatJobError(_ error: Error, contextSize: Int) -> String {
    if case RuntimeError.outOfMemory(let detail) = error {
        return """
            Model + context size \(contextSize) exceeded available device memory. \
            Try a smaller quant (e.g. Q4_0 instead of Q5_K_M), a smaller model, or \
            a smaller context length. \
            Underlying error: \(detail)
            """
    }
    return error.localizedDescription
}
