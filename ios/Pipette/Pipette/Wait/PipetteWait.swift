import Foundation

// `crates/pipette-wait`: the caller's side of a blocking wait. How each poll is narrated,
// when the wait gives up, and how it spends the gap between polls. Only what the readiness
// gate uses is ported.

/// Longest a wait sleeps before re-reading its cancel callback: the granularity a person
/// deserves after pressing cancel, where the poll interval is the one the sensors deserve.
nonisolated let cancelCheckInterval: Duration = .seconds(1)

/// How a status narrates itself when the caller has not said otherwise.
protocol WaitStatus {
    /// Called only when no `WaitCallbacks.withStatus` was installed.
    nonisolated func reportDefault()
}

nonisolated func neverCancelled() -> Bool {
    false
}

/// Rust's `ControlFlow<()>`: `break` means the caller cancelled and the rest of the sleep
/// was left unspent.
nonisolated enum ControlFlow: Equatable, Sendable {
    case `continue`
    case `break`

    func isBreak() -> Bool {
        self == .break
    }
}

/// The caller's side of a wait: how each poll is reported, whether the wait should give up
/// early, and how it sleeps between polls. None of the three describes what is waited for,
/// which is why they arrive as callbacks rather than as policy.
///
/// Installing `withStatus` takes over the narration: a wait reports through it and nowhere
/// else.
nonisolated struct WaitCallbacks<S> {
    private var onStatus: ((S) -> Void)?
    private var cancelled: () -> Bool = neverCancelled
    private var sleep: (Duration) -> Void = { Thread.sleep(forTimeInterval: $0 / .seconds(1)) }

    /// The domain's own narration, never cancel, sleep on the calling thread.
    init() {}

    /// Called once per poll on the calling thread, so a slow reporter lengthens the wait.
    func withStatus(_ onStatus: @escaping (S) -> Void) -> Self {
        var callbacks = self
        callbacks.onStatus = onStatus
        return callbacks
    }

    /// Read before the first probe, after each probe, and between the `cancelCheckInterval`
    /// pieces a sleep is cut into.
    func withCancel(_ cancelled: @escaping () -> Bool) -> Self {
        var callbacks = self
        callbacks.cancelled = cancelled
        return callbacks
    }

    /// What makes a wait testable: a test records the durations instead of spending them.
    func withSleep(_ sleep: @escaping (Duration) -> Void) -> Self {
        var callbacks = self
        callbacks.sleep = sleep
        return callbacks
    }

    func isCancelled() -> Bool {
        cancelled()
    }

    func sleepCancellably(_ total: Duration) -> ControlFlow {
        Pipette.sleepCancellably(total, cancelled: cancelled, sleep: sleep)
    }
}

/// The shorter of `interval` and what is left before `deadline`; zero once it has passed.
nonisolated func clampToDeadline(_ interval: Duration,
                                 _ deadline: ContinuousClock.Instant) -> Duration {
    min(interval, max(.zero, deadline - .now))
}

extension WaitCallbacks where S: WaitStatus {
    nonisolated func report(_ status: S) {
        if let onStatus {
            onStatus(status)
        } else {
            status.reportDefault()
        }
    }
}

/// Sleep `total` in pieces no longer than `cancelCheckInterval`, reading `cancelled` between
/// them. A zero `total` still reads the flag once, so a poll whose probe outran its interval
/// can be cancelled too.
///
/// The remainder is decremented rather than measured against a deadline, so an injected
/// `sleep` that does not sleep still terminates.
nonisolated func sleepCancellably(_ total: Duration, cancelled: () -> Bool,
                                  sleep: (Duration) -> Void) -> ControlFlow {
    var remaining = total
    while remaining > .zero {
        if cancelled() { return .break }
        let chunk = min(remaining, cancelCheckInterval)
        sleep(chunk)
        remaining -= chunk
    }
    return cancelled() ? .break : .continue
}
