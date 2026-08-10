import Foundation

/// Engine-agnostic benchmark measurement methodology — the crate's
/// `pipette-ops/src/measurement.rs`, shared by `LlamaBenchmark` and `MLXRuntime`. Owns
/// the rep loop, the population stats, and the timer — the parts that must stay identical
/// across runtimes for MLX and llama numbers to be comparable. The engine-specific kernels, the readiness→error mapping (`gate`),
/// and the warm-up *shape* stay in each runtime; this only orchestrates them.
nonisolated enum BenchmarkMeasurement {
    /// 1 warm-up (discarded) + `runs` timed reps, gating before each measured rep.
    /// Returns population `(mean_ms, stddev_ms)`.
    ///
    /// The single measurement entry point for all three runtimes. `async` is
    /// load-bearing only for AFM (its Foundation Models kernel awaits each
    /// streaming snapshot); llama and MLX pass synchronous closures, which run
    /// without suspending. One loop keeps the warm-up + gated-rep + `stats`
    /// discipline — the invariant that makes llama and MLX numbers comparable —
    /// in a single place, so it can't drift between runtimes. `gate` stays
    /// synchronous: it's a thermal reading, not I/O.
    ///
    /// `warmup` is a separate closure because the policy still differs. llama and MLX warm
    /// the cell's own shape — one untimed `body` — so the measured rep finds every Metal
    /// pipeline already compiled. AFM caps its warm-up at one token instead: the shapes do
    /// not match, but nothing there is ours to warm, and generation is the scarce resource.
    /// `gate` throws the runtime's *own* error on a non-ready outcome, so each runtime keeps
    /// its distinct user-facing error namespace with no mapping layer here.
    ///
    /// `label` names the cell in the per-rep progress log, as the crate's does — spelled
    /// `benchmark_type`, the same string `pipette-llamacpp` passes.
    static func measure(
        label: String,
        runs: Int,
        warmup: () async throws -> Void,
        gate: () throws -> Void,
        observer: RepObserver,
        onProgress: (BenchmarkProgress) -> Void = { _ in },
        body: () async throws -> Double
    ) async throws -> (mean: Double, stddev: Double) {
        // No crate counterpart — its warm-up is inside the driver invocation. Ours is a
        // separate untimed pass, long enough on a big model to look like a stalled load.
        AppLog.jobRun.info("\(label): warm-up")
        try await warmup()
        var samples: [Double] = []
        samples.reserveCapacity(runs)
        for i in 0 ..< runs {
            try gate()
            // Reported once the gate clears, so a sampler sees the rep's entry condition
            // rather than the wait's. `RepObserver` documents the bracketing contract.
            observer.repStarted()
            // After the gate, matching `measurement::run` — the line marks the rep
            // starting, not the wait that preceded it.
            AppLog.jobRun.info("\(label): measurement run \(i + 1)/\(runs)")
            samples.append(try await body())
            observer.repFinished()
            onProgress(.attempt(completed: i + 1, total: runs))
        }
        return stats(samples)
    }

    /// Mean and **sample** stddev — the crate's `mean_stddev`.
    ///
    /// Sample, not population: the estimator has to match the other clients or the same
    /// measurement reads differently depending on which one took it.
    static func stats(_ samples: [Double]) -> (mean: Double, stddev: Double) {
        guard !samples.isEmpty else { return (0, 0) }
        let n = Double(samples.count)
        let mean = samples.reduce(0, +) / n
        // A single rep has no spread to estimate.
        guard samples.count > 1 else { return (mean, 0) }
        let variance = samples.reduce(0) { $0 + ($1 - mean) * ($1 - mean) } / (n - 1)
        return (mean, variance.squareRoot())
    }

    /// Measure how long `body` takes, on a **monotonic** clock — wall-clock `Date`
    /// can step (NTP / manual time changes) mid-run and yield bogus or negative
    /// elapsed times. Returns a unit-agnostic `Duration`; the caller picks the
    /// precision it needs (the result schema uses `.milliseconds`).
    static func timed(_ body: () throws -> Void) rethrows -> Duration {
        let clock = ContinuousClock()
        let start = clock.now
        try body()
        return clock.now - start
    }
}

extension Duration {
    /// The precision the benchmark result schema (`prefill_time_ms`, …) reports,
    /// where the unit-agnostic `Duration` is pinned to the wire's milliseconds.
    nonisolated var milliseconds: Double {
        let c = components
        return Double(c.seconds) * 1_000 + Double(c.attoseconds) * 1e-15
    }
}

/// One eval sample's generation result: the completion text plus the stop
/// metadata captured at generation. `stopReason` is always classified
/// (`unknown` when a runtime can't tell); `completionTokens` is `nil` when the
/// runtime didn't report a count.
struct EvalGeneration {
    let text: String
    var stopReason: BenchmarkEvalCompletionStopReason
    var stopDetail: String?
    var completionTokens: Int?
}

/// Run each eval sample through `complete`, turning a per-sample failure into a
/// *failed completion* rather than aborting the whole run, and reporting `.sample`
/// progress per row. Shared by both runtimes so that contract can't drift. The
/// engine-specific bits stay in the closures: `beforeSample` runs *outside* the
/// failure-catch (so a readiness cancel there aborts — the llama path gates + resets
/// the KV here; MLX needs neither), and `complete` produces one `EvalGeneration`.
nonisolated func evalSamples(_ samples: [EvalSample],
                 resuming checkpoint: EvalCompletionSession? = nil,
                 beforeSample: (Int) throws -> Void = { _ in },
                 progress: (BenchmarkProgress) -> Void,
                 complete: (EvalSample) throws -> EvalGeneration) throws -> [BenchmarkEvalCompletion] {
    var completions: [BenchmarkEvalCompletion] = []
    for (index, sample) in samples.enumerated() {
        defer { progress(.sample(completed: index + 1, total: samples.count)) }
        // A sample an earlier attempt already completed is reused, not re-run: that is
        // what the checkpoint is for, and re-running would also pay for it twice.
        if let prior = checkpoint?.completion(for: sample.id) {
            completions.append(prior)
            continue
        }
        try beforeSample(index)
        let completion: BenchmarkEvalCompletion
        do {
            let gen = try complete(sample)
            completion = .completed(id: sample.id, text: gen.text, stopReason: gen.stopReason,
                                    stopDetail: gen.stopDetail, completionTokens: gen.completionTokens)
        } catch {
            completion = .failed(id: sample.id, reason: "\(error)")
        }
        completions.append(completion)
        // Recorded before the next sample starts, so a kill costs one sample rather than
        // the run. A failed checkpoint write is not worth losing the run over — the
        // completion still counts, it just will not survive a kill.
        try? checkpoint?.append(completion)
    }
    return completions
}
