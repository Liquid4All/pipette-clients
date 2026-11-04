import Foundation
import MLXHuggingFace
import MLXLMCommon
import Tokenizers

// The crate's `pipette-mlx/src/execute/eval.rs`, through `BenchmarkMeasurement.eval`. A sample
// that fails to template or decode is a failed row, and the cell goes on.

nonisolated extension MlxExecute {
    static func eval(
        _ benchmark: EvalBenchmark,
        cell: EvalCell,
        checkpoint: EvalCompletionSession,
        model: ResolvedMlxModel,
        chunk: Int,
        isCancelled: @escaping () -> Bool,
        progress: @escaping (BenchmarkProgress) -> Void
    ) async throws -> BenchmarkResultData {
        let maxTokens = Int(cell.maxTokens)
        // Loaded before the model, so generation stays pure token space.
        let tokenizer = try await #huggingFaceTokenizerLoader().load(
            from: URL(fileURLWithPath: model.dir))
        // Every id that ends a generation for this bundle, not the tokenizer's alone.
        let eosIds = MlxModels.eosTokenIds(bundle: model.dir, tokenizerEos: tokenizer.eosTokenId)

        return try await MlxRuntime.withFreshModel(path: model.dir) { lm in
            MlxRuntime.logMem("eval.loaded samples=\(benchmark.samples?.count ?? 0) "
                + "maxTokens=\(maxTokens)")
            // `mlx_lm.server` serves one request at a time, and so does the model here.
            let result = try await BenchmarkMeasurement.eval(
                benchmark, inFlight: .parallel(1), checkpoint: checkpoint, ctx: lm,
                progress: progress
            ) { lm, samples, record in
                for (at, sample) in samples.enumerated() {
                    // A stop ends the cell rather than failing the sample; eval has no
                    // thermal gate.
                    try cancellationGate(isCancelled)
                    let reading: EvalMeasurementReading
                    do {
                        let messages = sample.messages.map { $0.mapValues { $0 as any Sendable } }
                        let promptIds = try tokenizer.applyChatTemplate(messages: messages)
                        let (outputIds, stoppedOnEos) = try MlxRuntime.generate(
                            lm, promptIds: promptIds, maxTokens: maxTokens,
                            prefillChunkSize: chunk, eosIds: eosIds, sampling: cell.sampling,
                            seed: cell.sampling.seedFor(sample.id))
                        reading = EvalGeneration(
                            text: tokenizer.decode(tokenIds: outputIds),
                            stopReason: evalStopReason(stoppedOnEnd: stoppedOnEos),
                            stopDetail: nil, completionTokens: UInt64(outputIds.count)).reading
                    } catch {
                        // A stop can fail the sample in flight, so a failure with one pending
                        // is the stop's.
                        try cancellationGate(isCancelled)
                        reading = .failed(reason: "\(error)")
                    }
                    try record(at, reading)
                }
            }
            MlxRuntime.logMem("eval.done")
            return result
        }
    }
}
