import Foundation

// ASR throughput: the crate's `pipette-ops/src/measurement/asr_throughput.rs`
// (`docs/methodology/asr-throughput.md`), on `vl_throughput`'s reps.

/// An audio count an ASR throughput cell cannot use.
nonisolated enum AsrThroughputMeasurementError: Error, Equatable, LocalizedError {
    case audioTokensFillPrompt(audioTokens: PositiveCount, textTokens: UInt32, promptTokens: PositiveCount)

    var errorDescription: String? {
        switch self {
        case let .audioTokensFillPrompt(audioTokens, textTokens, promptTokens):
            "engine counted audio_tokens \(audioTokens) of prompt_tokens \(promptTokens) beside "
                + "\(textTokens) text tokens: the prompt holds the template too, so the audio "
                + "and the text are less than all of it"
        }
    }
}

nonisolated extension BenchmarkMeasurement {
    /// `settleAudioTokens` runs once, after the reps. Audio plus text must stay below the
    /// prompt, since every chat template adds tokens of its own.
    static func asrThroughput<C>(
        _ benchmark: AsrThroughputBenchmark,
        reps: Int,
        readinessGate: ReadinessGate,
        observer: MeasurementObserver,
        warmup: MeasurementWarmup,
        ctx: C,
        attempt: (inout C) async throws -> VlThroughputMeasurementReading,
        settleAudioTokens: (PositiveCount) async throws -> PositiveCount
    ) async throws -> BenchmarkResultData {
        let marker = benchmark.benchmarkId.value
        let (first, prompt, predicted, total) = try await throughput(
            marker: marker, decodeTokens: benchmark.parameterDecodeTokens.value.magnitude,
            reps: reps, readinessGate: readinessGate, observer: observer, warmup: warmup,
            ctx: ctx, attempt: attempt)

        let audioTokens = try await settleAudioTokens(first.promptTokens)
        let textTokens = benchmark.parameterTextTokens.value.magnitude
        guard Int64(audioTokens.value) + Int64(textTokens) < Int64(first.promptTokens.value) else {
            throw AsrThroughputMeasurementError.audioTokensFillPrompt(
                audioTokens: audioTokens, textTokens: textTokens,
                promptTokens: first.promptTokens)
        }
        AppLog.jobRun.info("\(marker): asr token counts prompt_tokens=\(first.promptTokens) "
            + "audio_tokens=\(audioTokens)")
        // One rep has no spread to report.
        func stddev(_ stats: Stats) -> Double? { reps > 1 ? stats.stddevMs : nil }

        return .asrThroughput(promptTokens: UInt32(first.promptTokens.value),
                              audioTokens: UInt32(audioTokens.value),
                              promptMs: prompt.meanMs,
                              promptMsStddev: stddev(prompt),
                              predictedMs: predicted.meanMs,
                              predictedMsStddev: stddev(predicted),
                              totalTimeMs: total.meanMs,
                              totalTimeMsStddev: stddev(total))
    }
}
