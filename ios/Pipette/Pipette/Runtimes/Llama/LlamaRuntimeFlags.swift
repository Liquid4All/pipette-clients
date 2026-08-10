import Foundation

// The flags a llama.cpp cell runs with: the plan's entry plus the defaults this engine
// overlays, kept in the same typed form so the result records what ran — the Swift
// counterpart of `pipette-llamacpp/src/runtime_flags.rs`, named for its engine as
// every file in this folder is (the target has one flat namespace).
//
// The crate splits this per invocation shape (`for_bench`, `for_server`) because a
// `llama-bench` cell and a `llama-server` cell overlay different values. This client has
// the one shape — in-process load, no argv — so it has the one function.
//
// A routing failure propagates, as `for_bench` lets it: plan-types defines gguf-vision
// flags for `vl_throughput` alone, so a vision model on a timing benchmark is not a cell
// upstream describes. `ClientRunSpec.validated` already refuses one at claim time; refusing it
// here too keeps a hand-authored `spec=` from quietly reporting base-weight timings under
// a vision descriptor.

nonisolated enum LlamaRuntimeFlags {
    /// The flags this cell runs with: the plan's, with this engine's defaults where the
    /// cell left a knob unset. The result is the same variant a plan authors, so it
    /// round-trips through the wire form and reaches the submission as the record of the
    /// load.
    static func forRun(_ req: RunRequest) throws -> RuntimeFlags {
        var r = try req.runtimeFlagsRef()
        r.numberGpuLayers = r.numberGpuLayers ?? LlamaCpp.defaultNumberGpuLayers
        r.ctxSize = r.ctxSize ?? Self.contextSize(for: req.benchmark)
        r.nUbatch = r.nUbatch ?? LlamaCpp.defaultNUbatch
        r.threads = r.threads ?? LlamaCpp.defaultThreads
        r.swaFull = r.swaFull ?? LlamaCpp.defaultSwaFull
        return try r.resolve()
    }

    /// The context a cell needs when it named none — sized as `pipette-llamacpp` sizes it.
    /// `llama-bench` fits its context to the workload it is handed (`n_prompt + n_gen +
    /// n_depth`, never `--ctx-size`), and the server cells pass a `default_ctx_size`
    /// derived the same way; both reduce to the benchmark's own window.
    ///
    /// Derived in the engine, as `for_server` derives it, rather than at the job layer: a
    /// cell states what it asked for, and what it left unset is the engine's to answer.
    /// Deliberately no broader floor — over-allocating inflates `max_memory_usage`, whose
    /// KV cache scales with the window, against the reference client.
    static func contextSize(for benchmark: BenchmarkDefinition) -> UInt32 {
        switch benchmark {
        case let .prefillThroughput(_, prefill):
            prefill
        case let .maxMemoryUsage(_, prefill):
            prefill.addingSaturating(1)
        case let .decodeThroughput(_, prefill, decode),
             let .endToEndLatency(_, prefill, decode):
            prefill.addingSaturating(decode)
        case let .eval(_, _, _, maxTokens, _, _):
            Self.evalPromptBudget.addingSaturating(maxTokens)
        case let .vlThroughput(_, width, height, text, decode):
            // ~1 token per 14x14 patch, floored as `for_server` floors it: the estimate
            // can undershoot, and a vision cell that overflows measures nothing.
            max(8192, (width / 14).multipliedSaturating(height / 14)
                .addingSaturating(text).addingSaturating(decode))
        }
    }

    /// Eval prompt budget, as the crate's eval cell states it (`8192 + max_tokens`).
    private static let evalPromptBudget: UInt32 = 8192
}

private nonisolated extension UInt32 {
    /// Clamp on overflow. Wrapping would silently produce a tiny context and defeat the
    /// sizing entirely.
    func addingSaturating(_ other: UInt32) -> UInt32 {
        let (sum, overflow) = addingReportingOverflow(other)
        return overflow ? .max : sum
    }

    func multipliedSaturating(_ other: UInt32) -> UInt32 {
        let (product, overflow) = multipliedReportingOverflow(by: other)
        return overflow ? .max : product
    }
}
