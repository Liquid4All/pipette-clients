import Foundation

// The plan's flags with this engine's defaults overlaid, kept typed so the result records
// what ran. A routing failure propagates: plan-types defines no gguf-vision flags for a timing
// cell, and `ClientRunSpec.validated` refuses one only when the claim carries `runtime_flags`,
// so this is what keeps a flagless claim or a hand-authored `spec=` from reporting
// base-weight timings under a vision descriptor.

nonisolated enum LlamaRuntimeFlags {
    /// `perImageTokens` is a vision cell's count for one frame, or an ASR cell's for its clip,
    /// once its projector has loaded; `recipeTextBytes` is an ASR recipe's `textBytes`.
    static func forRun(_ req: RunRequest, perImageTokens: UInt32? = nil,
                       recipeTextBytes: UInt32 = 0) throws -> RuntimeFlags {
        var r = try req.runtimeFlagsRef()
        // For a zero context `LlamaCpp.load` keeps llama's own default, which this side
        // can't report. Refused, as on Android.
        if r.ctxSize == 0 {
            throw KnobValueNotUsable(
                knob: "ctx_size", value: "0",
                reason: "the engine keeps llama's own default for a non-positive context, "
                    + "so the value this run used cannot be reported")
        }
        r.numberGpuLayers = r.numberGpuLayers ?? LlamaCpp.defaultNumberGpuLayers
        r.ctxSize = r.ctxSize ?? Self.ctxSize(for: req.benchmark, perImageTokens: perImageTokens,
                                              recipeTextBytes: recipeTextBytes)
        // Resolved exactly as `LlamaCpp.load` resolves it (a non-positive request is 512,
        // clamped to the context), so the report names the chunk the run used. As on Android.
        let asked = r.nUbatch ?? LlamaCpp.defaultNUbatch
        let requested = asked > 0 ? asked : LlamaCpp.defaultNUbatch
        r.nUbatch = (r.ctxSize ?? 0) > 0 ? min(requested, r.ctxSize ?? requested) : requested
        r.threads = r.threads ?? LlamaCpp.defaultThreads
        r.swaFull = r.swaFull ?? LlamaCpp.defaultSwaFull
        // Pinned for timing and memory cells as the crate's `get_load_mode` pins them. Eval
        // keeps llama.cpp's mmap default, as the crate's does: it runs long enough to
        // amortize the page faults.
        r.loadMode = r.loadMode ?? (req.benchmark.benchmarkType == .eval ? .mmap : LlamaCpp.defaultLoadMode)
        // Resolved against the model, so the row records what ran rather than what was unset.
        if req.benchmark.benchmarkType == .vlSpecDecode {
            r.speculativeEnabled = try resolveSpeculation(r.modelType, r.speculativeEnabled)
        }
        return try r.resolve()
    }

    /// What the run resolved, except on eval, where the crate reports the plan's flags
    /// unmodified since its overlay is the engine's business.
    static func reported(_ req: RunRequest, ran: RuntimeFlags) -> RuntimeFlags? {
        req.benchmark.benchmarkType == .eval ? req.runtimeFlags : ran
    }

    /// Sized as `pipette-llamacpp` sizes it: the benchmark's own window, as `llama-bench`
    /// fits its context to the workload. No broader floor, since a larger window inflates
    /// `max_memory_usage` against the reference client.
    static func ctxSize(for benchmark: BenchmarkDefinition,
                        perImageTokens: UInt32? = nil, recipeTextBytes: UInt32 = 0) -> UInt32 {
        switch benchmark {
        case let .prefillThroughput(b):
            b.parameterPrefillTokens.value.magnitude
        case let .maxMemoryUsage(b):
            b.parameterPrefillTokens.value.magnitude.addingSaturating(1)
        case let .decodeThroughput(b):
            b.parameterPrefillTokens.value.magnitude.addingSaturating(b.parameterDecodeTokens.value.magnitude)
        case let .endToEndLatency(b):
            b.parameterPrefillTokens.value.magnitude.addingSaturating(b.parameterDecodeTokens.value.magnitude)
        case let .eval(b):
            Self.evalPromptBudget.addingSaturating(b.parameterMaxTokens.value.magnitude)
        // A cell with more frames than any run takes fits no context: it is refused by name
        // before it loads, and sized here so that no picker offers it.
        case let .vlThroughput(b):
            // The estimate is floored: it can come up short, and an overflow measures nothing.
            (try? LlamaMtmd.CtxInputs(b))?
                .ctxSize(calibrated: perImageTokens, fallbackFloor: 8192) ?? .max
        case let .vlMaxMemory(b):
            // No floor: padding the window would inflate the peak it reports.
            (try? LlamaMtmd.CtxInputs(b))?
                .ctxSize(calibrated: perImageTokens, fallbackFloor: 0) ?? .max
        case .vlSpecDecode:
            // The window holds a whole conversation; an overflow fails the traversal rather
            // than truncating it.
            LlamaExecute.defaultCtxSizePerSlot
        // `perImageTokens` is the clip's audio count here; no floor, as the CLI. Before the
        // projector loads, the recipe's bytes are unknown and left out.
        case let .asrThroughput(b):
            asrCtxSize(b.parameterAudioDurationMs, b.parameterTextTokens, b.parameterDecodeTokens,
                       audioTokens: perImageTokens, recipeTextBytes: recipeTextBytes)
        case let .asrMaxMemory(b):
            asrCtxSize(b.parameterAudioDurationMs, b.parameterTextTokens, b.parameterDecodeTokens,
                       audioTokens: perImageTokens, recipeTextBytes: recipeTextBytes)
        }
    }

    /// The CLI's `CtxInputs::ctx_size`, the upper bound standing in for an uncounted clip.
    private static func asrCtxSize(_ duration: AudioDurationMillis, _ text: NonNegativeCount,
                                   _ decode: AsrDecodeTokens, audioTokens: UInt32?,
                                   recipeTextBytes: UInt32) -> UInt32 {
        (audioTokens ?? fallbackAudioTokens(durationMs: duration.value.magnitude))
            .addingSaturating(text.value.magnitude)
            .addingSaturating(recipeTextBytes)
            .addingSaturating(decode.value.magnitude)
            .addingSaturating(LlamaMtmd.ctxMargin)
    }

    /// The CLI's `fallback_audio_tokens`: the larger bound, so no encoder's context comes up short.
    static func fallbackAudioTokens(durationMs: UInt32) -> UInt32 {
        let byRate = ((durationMs + 999) / 1000).multipliedSaturating(25)
        let byWindow = (durationMs / 30_000 + 1).multipliedSaturating(750)
        return max(byRate, byWindow)
    }

    /// The crate's eval budget is `8192 + max_tokens`.
    private static let evalPromptBudget: UInt32 = 8192
}

/// An allowed knob with a value this engine can't act on (plan-types accepts any `u32`
/// context; this engine's load cannot honour a zero). Distinct from
/// `RuntimeFlagsError.knobNotAllowed`: blaming the knob rather than the value sends the
/// author to the wrong fix.
nonisolated struct KnobValueNotUsable: Error, Equatable, LocalizedError {
    let knob: String
    let value: String
    let reason: String

    var errorDescription: String? { "`\(knob) = \(value)` cannot be used: \(reason)" }
}
