import Foundation

// `ReadinessOutcome` is this client's `Result<(), ReadinessError>` from
// `pipette_readiness::wait_until_ready`.

/// The non-`ready` variants fail the cell rather than recording numbers under unknown
/// thermal conditions.
public enum ReadinessOutcome: Equatable, Hashable {
    case ready
    case cancelled
    /// The crate's `ReadinessError::TimedOut`. `observed` is the last poll's summary.
    case timedOut(maxWait: Duration, observed: String)
}

extension ReadinessOutcome: Sendable {}

extension ReadinessOutcome {
    /// Throws for a non-ready outcome, like `wait_until_ready`'s `?` upstream, so a
    /// `ReadinessGate` is one line over a wait.
    nonisolated func get() throws {
        switch self {
        case .ready: break
        case .cancelled: throw RuntimeError.cancelled
        case let .timedOut(maxWait, observed):
            throw RuntimeError.readiness("\(observed) after \(maxWait.components.seconds)s")
        }
    }
}

/// Cancel-only counterpart to a `ReadinessGate`, for the eval path: accuracy is
/// thermal-invariant, so eval never waits, but a long eval must still stop on cancel.
nonisolated func cancellationGate(_ isCancelled: () -> Bool) throws {
    if isCancelled() { throw RuntimeError.cancelled }
}

/// Pre-rep readiness wait. `nonisolated` because a run waits on its own thread, off the main
/// actor, and a caller holding the protocol must be able to say so too.
///
/// `observe` sees every poll beside the caller's own status handler, so the run records the
/// wait whatever the caller shows of it.
protocol ReadinessCallback: AnyObject, Sendable {
    nonisolated func waitUntilReady(observing observe: @escaping (ReadinessStatus) -> Void)
        -> ReadinessOutcome
}

extension ReadinessCallback {
    nonisolated func waitUntilReady() -> ReadinessOutcome {
        waitUntilReady(observing: { _ in })
    }
}

/// Invoked synchronously on the benchmark's background thread before each measured rep.
final class BenchmarkReadiness: @unchecked Sendable, ReadinessCallback {
    private let cancelFlag: CancelFlag
    private let readiness: ResolvedReadiness
    private let onStatus: @Sendable (ReadinessStatus) -> Void

    nonisolated init(cancelFlag: CancelFlag, readiness: ResolvedReadiness = ResolvedReadiness(),
                     onStatus: @escaping @Sendable (ReadinessStatus) -> Void) {
        self.cancelFlag = cancelFlag
        self.readiness = readiness
        self.onStatus = onStatus
    }

    nonisolated func waitUntilReady(observing observe: @escaping (ReadinessStatus) -> Void)
        -> ReadinessOutcome {
        let cancelFlag = cancelFlag
        let onStatus = onStatus
        return waitUntilReadyWith(readiness, callbacks: ReadinessCallbacks()
            .withStatus { status in
                observe(status)
                onStatus(status)
            }
            .withCancel { cancelFlag.isCancelled })
    }
}
