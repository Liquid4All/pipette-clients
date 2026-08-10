import Foundation
import MLX
import MLXLMCommon

/// MLX benchmark methodology: maps a typed `BenchmarkDefinition` onto the
/// `MLXGenerate` primitives and the shared `BenchmarkMeasurement` rep loop, and
/// turns the result into the same `BenchmarkResult` the llama path emits. The
/// structural counterpart to `LlamaBenchmark` — the load and the single `run`
/// entry stay in `MLXRuntime`; this owns the per-definition dispatch.
extension MLXRuntime {
    /// Run one already-loaded benchmark and return the result (same schema as the
    /// llama path). Dispatches on the typed definition. `readiness` runs before
    /// each measured rep (the caller waits for the device to settle); a non-`.ready`
    /// outcome cancels the run or fails the cell. `eval` is handled by `run`; e2e
    /// takes `tokenizer` (it times tokenization in-window); `vl_throughput` isn't
    /// ported to MLX.
    nonisolated static func runOnModel(_ model: any LanguageModel, _ definition: BenchmarkDefinition,
                           prefillChunk chunk: Int, readiness: () -> ReadinessOutcome,
                           observer: RepObserver,
                           progress: (BenchmarkProgress) -> Void,
                           tokenizer: (any MLXLMCommon.Tokenizer)? = nil) async throws -> BenchmarkResult {
        // Release this cell's transient activation buffers when done; the resident
        // weights stay, so footprint resets to ~weights between same-model cells
        // instead of creeping up and OOM'ing on a larger-offset cell.
        defer { MLX.Memory.clearCache() }
        // Re-baseline MLX's peak-memory high-water mark so the per-cell `mlxPeak`
        // diagnostic (logged by `logMem`) reflects this cell, not a prior one. The
        // reported max_memory_usage no longer reads this counter — it samples the
        // process footprint in `MLXRuntime.maxMemory` — but the log line still helps
        // track the MLX-allocator ramp per cell.
        MLX.Memory.peakMemory = 0
        let id = definition.benchmarkId
        logMem("bench \(id) chunk=\(chunk)")
        switch definition {
        case .prefillThroughput(_, let prefill):
            let body = { prefillMillis(model, tokens: Int(prefill), chunk: chunk) }
            let (mean, sd) = try await BenchmarkMeasurement.measure(
                label: definition.benchmarkType, runs: Self.measurementRuns, warmup: { _ = body() },
                gate: { try readinessGate(readiness) }, observer: observer,
                onProgress: progress, body: body)
            return .prefillThroughput(timeMs: mean, stddev: sd)
        case .decodeThroughput(_, let prefill, let decode):
            let body = { generate(model, prefill: Int(prefill), decode: Int(decode), chunk: chunk) }
            let (mean, sd) = try await BenchmarkMeasurement.measure(
                label: definition.benchmarkType, runs: Self.measurementRuns, warmup: { _ = body() },
                gate: { try readinessGate(readiness) }, observer: observer,
                onProgress: progress, body: body)
            return .decodeThroughput(timeMs: mean, stddev: sd)
        case .endToEndLatency(_, let prefill, let decode):
            // e2e times tokenize + prefill + decode (CLI parity). Build the prompt
            // text once, untimed, to tokenize to exactly `prefill` tokens under this
            // model's tokenizer; each rep then tokenizes it inside the timed window.
            guard let tokenizer else { throw RuntimeError.engine("e2e requires a tokenizer") }
            let promptText = PromptSeed.buildPromptText(target: Int(prefill)) {
                tokenizer.encode(text: $0).count
            }
            let body = {
                endToEndMillis(model, tokenizer: tokenizer, promptText: promptText,
                               decode: Int(decode), chunk: chunk)
            }
            let (mean, sd) = try await BenchmarkMeasurement.measure(
                label: definition.benchmarkType, runs: Self.measurementRuns, warmup: { _ = body() },
                gate: { try readinessGate(readiness) }, observer: observer,
                onProgress: progress, body: body)
            return .endToEndLatency(timeMs: mean, stddev: sd)
        case .maxMemoryUsage, .vlThroughput, .eval:
            // max_memory_usage is owned by `run`/`maxMemory` (it brackets the whole
            // fresh-load with the footprint sampler); eval/vl_throughput aren't
            // dispatched here.
            throw RuntimeError.unsupported(id)
        }
    }

    /// `max_memory_usage`: drive one prefill + a single decode on a fresh load,
    /// bracketed by the shared process-footprint sampler (see
    /// `ProcessMemory.maxMemoryBracket`). `withFreshModel` releases the weights on
    /// exit; the generate timing is discarded — only the peak it provokes matters.
    nonisolated static func maxMemory(modelPath: String, prefillChunk: Int,
                          prefillTokens: Int) async throws -> BenchmarkResult {
        try await ProcessMemory.maxMemoryBracket(label: "mlx") {
            try await withFreshModel(path: modelPath) { model in
                _ = generate(model, prefill: prefillTokens, decode: 1, chunk: prefillChunk)
            }
        }
    }

    /// `eval`: complete every dataset sample on the loaded model and return the
    /// `{benchmark_id, completions: [{id, completion}]}` shape the unified payload
    /// builder ingests (parity with the llama eval path; the server scores). Each
    /// sample's chat messages go through the chat template and a greedy decode to
    /// `maxTokens` (or the first EOS). A sample that fails to template/decode
    /// becomes a failed completion, not an aborted run.
    nonisolated static func evalCompletions(_ model: any LanguageModel,
                                samples: [EvalSample], maxTokens: Int,
                                tokenizer: any MLXLMCommon.Tokenizer, prefillChunk: Int,
                                checkpoint: EvalCompletionSession?,
                                isCancelled: () -> Bool,
                                progress: (BenchmarkProgress) -> Void) throws -> BenchmarkResult {
        logMem("eval.loaded samples=\(samples.count) maxTokens=\(maxTokens)")
        // Per-sample cancel checkpoint (see `cancellationGate`); eval has no thermal gate.
        let completions = try evalSamples(samples,
            resuming: checkpoint,
            beforeSample: { _ in try cancellationGate(isCancelled) },
            progress: progress) { sample in
            let messages = sample.messages.map { $0.mapValues { $0 as any Sendable } }
            let promptIds = try tokenizer.applyChatTemplate(messages: messages)
            let (outputIds, stoppedOnEos) = generateGreedy(
                model, promptIds: promptIds, maxTokens: maxTokens,
                prefillChunkSize: prefillChunk, eosId: tokenizer.eosTokenId)
            // Use the engine's own stop signal: EOS token ⇒ `eos`, otherwise the
            // `maxTokens` cap bound it ⇒ `truncated`.
            let stopReason: BenchmarkEvalCompletionStopReason = stoppedOnEos ? .eos : .truncated
            return EvalGeneration(text: tokenizer.decode(tokenIds: outputIds),
                                  stopReason: stopReason, stopDetail: nil,
                                  completionTokens: outputIds.count)
        }
        logMem("eval.done")
        // See `LlamaBenchmark.evalOn` for why the session's own view is not what is
        // reported: the per-sample append is best-effort.
        checkpoint?.finalize()
        return .eval(completions: completions)
    }
}
