import Foundation

/// The crate's `pipette-ops/src/measurement/mod.rs`; the typed entries are extensions in
/// `Ops/Measurement/`. Not named `Measurement`, which Foundation owns.
///
/// No `AttemptError`, whose one variant wraps any error: an attempt here throws untyped. No
/// `run` either; only engines this client lacks still call it.
nonisolated enum BenchmarkMeasurement {
    /// Repetitions every timing cell measures: the crate's `REPS`. One number because the
    /// result schema treats the per-rep arrays of different cells as one methodology.
    static let reps = 5

    /// `1` on a kind with no batch.
    static func batchSize(_ benchmark: BenchmarkDefinition) -> Int {
        (benchmark.batchSize ?? .one).inFlight()
    }

    /// Refuse a batched entry on an engine that runs one request at a time.
    ///
    /// Call it before anything loads: run serially, the entry would file the reference
    /// measurement under the batched id. `engine` says why, in the engine's words.
    static func requireOneInFlight(_ benchmark: BenchmarkDefinition, engine: String) throws {
        let batchSize = batchSize(benchmark)
        guard batchSize == 1 else {
            throw RuntimeError.engine(
                "\(benchmark.benchmarkId) keeps \(batchSize) requests in flight, and "
                    + "\(engine); a batched \(benchmark.benchmarkType.rawValue) entry needs a "
                    + "server with that many slots: vLLM or SGLang, or llama-server for "
                    + "end_to_end_latency")
        }
    }

    /// `work`'s wall time in milliseconds, beside its value.
    static func timed<T>(_ work: () async throws -> T) async rethrows -> (Double, T) {
        let clock = ContinuousClock()
        let started = clock.now
        let value = try await work()
        return (durationMs(clock.now - started), value)
    }

    /// A batch is one rep, so its prompts' readings reduce to one sample.
    static func perRequestMean(_ samples: [Double]) throws -> Double {
        guard !samples.isEmpty else { throw MeasurementError.noReadings }
        return samples.reduce(0, +) / Double(samples.count)
    }

    /// The crate's `Stats`.
    struct Stats: Equatable {
        let meanMs: Double
        let stddevMs: Double
    }

    /// Mean and **sample** stddev: the crate's `mean_stddev`. Empty is `(0, 0)`; a single
    /// sample has no spread, so `(mean, 0)`. Sample, not population: the estimator has to
    /// match the other clients.
    static func meanStddev(_ samples: [Double]) -> Stats {
        guard !samples.isEmpty else { return Stats(meanMs: 0, stddevMs: 0) }
        let n = Double(samples.count)
        let mean = samples.reduce(0, +) / n
        guard samples.count > 1 else { return Stats(meanMs: mean, stddevMs: 0) }
        let variance = samples.reduce(0) { $0 + ($1 - mean) * ($1 - mean) } / (n - 1)
        return Stats(meanMs: mean, stddevMs: variance.squareRoot())
    }

    /// Character for character the crate's, so logs compare across clients.
    static func reductionLine(label: String, samples: [Double], stats: Stats) -> String {
        let values = samples.map(format).joined(separator: ", ")
        return "\(label): samples [\(values)] ms -> mean \(format(stats.meanMs)) ms "
            + "stddev \(format(stats.stddevMs)) ms"
    }

    /// The crate's `{:.3}`. No `locale:`: `Locale.current` would change the separator and add
    /// grouping.
    static func format(_ value: Double) -> String {
        String(format: "%.3f", value)
    }
}

/// The rep hooks are called around each measured rep; for spec decode the rep is the whole
/// traversal. A throw abandons the cell. The other three drive the `attempt_timeout_seconds`
/// clock, and do nothing unless set, as for a caller with no limit.
nonisolated struct MeasurementObserver: Sendable {
    /// After the gate; `RunCell` samples the device.
    let onBeforeRepStart: @Sendable () throws -> Void
    /// After the rep's timed work; `RunCell` samples the device.
    let onAfterRepEnd: @Sendable () throws -> Void
    /// An attempt the rep hooks do not bracket starts (a peak-memory attempt), named for the
    /// overrun message. Throws where the crate's does not: an engine here cannot be stopped,
    /// so an attempt before it that ran past the limit fails the cell here.
    var onAttemptStart: @Sendable (String) throws -> Void = { _ in }
    /// A step the limit leaves out starts.
    var onHold: @Sendable () -> Void = {}
    /// It ended; the attempt goes on with the whole limit.
    var onResume: @Sendable () -> Void = {}

    /// Run `step`, which the limit leaves out, with the clock held.
    func outside<T>(_ step: () throws -> T) rethrows -> T {
        onHold()
        defer { onResume() }
        return try step()
    }
}

/// Where a timing entry's warm-up happens.
nonisolated enum MeasurementWarmup: Equatable, Sendable {
    /// One untimed, checked attempt before the reps, its reading not reduced, so a first-use
    /// cost (kernel selection, a lazily mapped weight) stays out of rep 1.
    case once
    /// Each attempt warms itself: a fresh `llama-bench` process upstream, a fresh Apple
    /// Foundation session here.
    case inline
}

/// Whether a peak-memory entry runs its attempt once before the measured one.
nonisolated enum MemoryMeasurementWarmup: Equatable, Sendable {
    /// The measured attempt is the cell's first.
    case cold
    /// One checked attempt first, its reading dropped, for an engine that compiles into a
    /// cache that outlives its process (a GPU driver's kernel cache, an OpenVINO blob dir).
    /// A peak is a high-water mark, so each attempt needs a process of its own. No iOS engine
    /// may pass it: every attempt runs in the app's process, so the warm-up would raise the
    /// measured peak.
    case once
}

/// The crate's `per_request_mean` bail, with its message.
nonisolated enum MeasurementError: Error, Equatable, LocalizedError {
    case noReadings

    var errorDescription: String? {
        switch self {
        case .noReadings: "a rep produced no readings to average"
        }
    }
}

/// `duration` in fractional milliseconds, the precision the benchmark result schema
/// (`prefill_time_ms`, …) reports.
nonisolated func durationMs(_ duration: Duration) -> Double {
    let c = duration.components
    return Double(c.seconds) * 1_000 + Double(c.attoseconds) * 1e-15
}
