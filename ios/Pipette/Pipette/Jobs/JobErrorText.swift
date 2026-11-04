import Foundation

/// Text for `JobCell.errorMessage`. Runtime errors are already user-readable; an OOM
/// (classified from the captured engine log) gets a smaller-quant or smaller-model hint.
/// `contextSize` is nil when the window is only known inside the load.
nonisolated func formatJobError(_ error: Error, contextSize: Int?) -> String {
    // A load failure wraps the engine error that caused it, and the hint below is about
    // that cause rather than about the wrapper.
    let error = (error as? ModelLoadFailure)?.cause ?? error
    if case RuntimeError.outOfMemory(let detail) = error {
        let context = contextSize.map { "context size \($0)" } ?? "context"
        return """
            Model + \(context) exceeded available device memory. \
            Try a smaller quant (e.g. Q4_0 instead of Q5_K_M), a smaller model, or \
            a smaller context length. \
            Underlying error: \(detail)
            """
    }
    return error.localizedDescription
}
