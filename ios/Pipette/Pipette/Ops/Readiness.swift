import Foundation

// The readiness gate and the per-rep observer — the Swift counterpart of
// `crates/pipette-ops/src/readiness.rs`. Both are how a caller's policy reaches an engine
// without the engine knowing anything about thermal probes or device sensors.

/// Outcome of the host's pre-rep readiness gate. The non-`ready` variants fail
/// the cell rather than recording numbers under unknown thermal conditions.
public enum ReadinessOutcome: Equatable, Hashable {
    /// Device cooled below the threshold — proceed with the measured rep.
    case ready
    /// User cancelled during the wait.
    case cancelled
    /// Device did not cool below the threshold within the time budget.
    /// `observed` summarizes the last reading (temp / state / elapsed).
    case timedOut(observed: String)
}

extension ReadinessOutcome: Sendable {}

/// Translate a readiness outcome into control flow, throwing `RuntimeError` on a
/// non-ready outcome. Shared by both runtimes as the `gate` for
/// `BenchmarkMeasurement.measure`.
nonisolated func readinessGate(_ readiness: () -> ReadinessOutcome) throws {
    switch readiness() {
    case .ready: break
    case .cancelled: throw RuntimeError.cancelled
    case .timedOut(let observed): throw RuntimeError.readiness(observed)
    }
}

/// Cancel-only counterpart to `readinessGate`, for the eval path. Eval runs
/// flat-out — accuracy is thermal-invariant, so it never waits for a cool
/// baseline — but a long eval must still stop promptly when the user cancels.
/// Throws `RuntimeError.cancelled` if so; no thermal reading, no wait.
nonisolated func cancellationGate(_ isCancelled: () -> Bool) throws {
    if isCancelled() { throw RuntimeError.cancelled }
}

/// Notified at each end of a measured repetition — the crate's `RepObserver`
/// (`pipette-ops/src/readiness.rs`). The engine reports the event; what the caller makes
/// of it — sampling sensors, timing, nothing — is the caller's business, so an engine
/// needs no telemetry vocabulary and no device dependency of its own.
///
/// The two reports are positional: the caller pairs them by order, so the *n*-th start
/// belongs to the *n*-th end. A cell that measures no repetitions calls neither, which is
/// how the eval path reports no series at all.
///
/// Required, never defaulted, at every layer that forwards it — as upstream takes it: a
/// measurement path that fails to report is then a compile error rather than a silently
/// empty series at submission time. `ignore` is for callers that genuinely record nothing.
///
/// Non-throwing where upstream's hooks return `Result` — theirs can fail only on a
/// poisoned `Mutex` around the series, and this collector is confined to `measure`'s
/// sequential rep loop with no lock to poison.
nonisolated struct RepObserver: Sendable {
    private let started: @Sendable () -> Void
    private let finished: @Sendable () -> Void

    /// For paths that measure reps but record no series — tests, and callers that
    /// discard telemetry.
    static let ignore = RepObserver(started: {}, finished: {})

    init(started: @escaping @Sendable () -> Void, finished: @escaping @Sendable () -> Void) {
        self.started = started
        self.finished = finished
    }

    /// A repetition is about to begin. Reported **the moment the readiness gate clears**,
    /// before any of the rep's timed work: a caller sampling here records the rep's entry
    /// condition, and a report placed before the gate — or after the workload starts —
    /// records something else. Model load and warm-up happen earlier, outside this point.
    func repStarted() { started() }

    /// The repetition's timed work has completed. Reported immediately, so the pair
    /// brackets the timed region and nothing else.
    func repFinished() { finished() }
}

/// Pre-rep readiness gate. The host blocks until the device has settled to a
/// steady thermal baseline (polling the SoC die temperature via `socTemp()`)
/// and reports the `ReadinessOutcome` so the benchmark can proceed, abort on
/// cancel, or fail the cell when the device won't cool or the sensor is
/// unreadable.
public protocol ReadinessCallback: AnyObject, Sendable {
    func waitUntilReady() -> ReadinessOutcome
}
