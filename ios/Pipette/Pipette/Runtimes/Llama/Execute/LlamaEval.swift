import Foundation

// The crate's `pipette-llamacpp/src/execute/eval.rs`: one chat completion per sample, through
// `BenchmarkMeasurement.eval`, ungated.

nonisolated extension LlamaExecute {
    static func eval(
        _ benchmark: EvalBenchmark,
        cell: EvalCell,
        _ inf: Inference,
        checkpoint: EvalCompletionSession,
        isCancelled: @escaping () -> Bool,
        progress: @escaping (BenchmarkProgress) -> Void
    ) async throws -> BenchmarkResultData {
        // MCQ constrains to a single choice token (the crate's `n_predict: 1`).
        let maxTokens = cell.mcqChoices != nil ? 1 : Int(cell.maxTokens)
        return try await BenchmarkMeasurement.eval(
            benchmark, inFlight: .parallel(1), checkpoint: checkpoint, ctx: inf,
            progress: progress
        ) { inf, samples, record in
            for (at, sample) in samples.enumerated() {
                // A stop ends the cell rather than failing the sample.
                try cancellationGate(isCancelled)
                inf.resetContext()
                inf.resetSampler()
                let reading: EvalMeasurementReading
                do {
                    reading = try inf.chatCompletion(
                        messagesJSONString(sample.messages), maxTokens, cell.mcqChoices,
                        cell.sampling, cell.sampling.seedFor(sample.id)).reading
                } catch {
                    // A stop can fail the sample in flight (a backgrounded app loses the
                    // GPU), so a failure with one pending is the stop's.
                    try cancellationGate(isCancelled)
                    // A failed backend fails every later sample too: the cell ends, and its
                    // next run loads the model again.
                    if case .backend? = error as? RuntimeError { throw error }
                    reading = .failed(reason: "\(error)")
                }
                try record(at, reading)
            }
        }
    }
}

private nonisolated func messagesJSONString(_ messages: [[String: String]]) throws -> String {
    let data = try JSONSerialization.data(withJSONObject: messages, options: [])
    return String(decoding: data, as: UTF8.self)
}
