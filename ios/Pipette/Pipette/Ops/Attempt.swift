import Foundation

/// What every client shares about `attempt_timeout_seconds`: the crate's
/// `pipette-ops/src/attempt.rs`. Each run keeps its own clock (`AttemptClock`), which the
/// engines mark through `MeasurementObserver`.
nonisolated enum Attempt {
    /// The limit on a cell whose flags set none, on the kinds that take a default.
    static let defaultLimit: Duration = .seconds(300)

    /// What a run's clock names the first attempt: the model load counts against it.
    static let loadAndWarmUp = "the model load and warm-up"

    /// The cell's limit: its flags' `attempt_timeout_seconds`, else the default for its
    /// kind. `nil` leaves it unbounded.
    static func limit(_ request: RunRequest) -> Duration? {
        request.benchmarkFlags?.attemptTimeout.map { Duration.fromSecs($0) }
            ?? defaultLimit(for: request.benchmark.benchmarkType)
    }

    /// Exhaustive, so a new kind has to say whether it gets one.
    static func defaultLimit(for benchmark: BenchmarkType) -> Duration? {
        switch benchmark {
        case .prefillThroughput, .decodeThroughput, .endToEndLatency, .maxMemoryUsage,
             .vlThroughput, .vlMaxMemory, .asrThroughput, .asrMaxMemory:
            defaultLimit
        // A corpus or an eval sets its own length, so a fixed default would cut healthy runs.
        case .vlSpecDecode, .eval:
            nil
        }
    }

    /// An attempt that ran past the limit.
    struct Overrun: Equatable, Sendable {
        let what: String
        let limit: Duration
        let elapsed: Duration
    }
}

/// The failure a run past its limit reports, final for every client (`classifyRunError`).
nonisolated struct AttemptTimeoutError: Error, Equatable, LocalizedError {
    let overrun: Attempt.Overrun
    /// What the engine failed with once it returned.
    let engineError: String?

    /// The crate's wording, so a failure reads the same whichever client ran the cell.
    var errorDescription: String? {
        let ran = "\(overrun.what) ran \(String(format: "%.1f", durationMs(overrun.elapsed) / 1_000))s, "
            + "past attempt_timeout_seconds (\(overrun.limit.components.seconds)s), and was stopped"
        return engineError.map { "\(ran); the engine then reported: \($0)" } ?? ran
    }
}
