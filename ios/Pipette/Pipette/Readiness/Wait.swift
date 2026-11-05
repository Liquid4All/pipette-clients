import Foundation

// Port of `crates/pipette-readiness/src/wait.rs`: the poll loop every platform shares, which
// differs only in the probe it is given.

nonisolated struct Probe {
    let ready: Bool
    let summary: String
}

/// Poll `probe` until it reports ready, the deadline passes, or the caller cancels. The
/// crate's probe returns a `Result`; this client's probes cannot fail (an unreadable signal is
/// part of the verdict), so this one returns the verdict.
nonisolated func waitLoop(platform: ReadinessPlatform, maxWait: Duration,
                          callbacks: ReadinessCallbacks, probe: () -> Probe) -> ReadinessOutcome {
    let started = ContinuousClock.now
    let deadline = started.advanced(by: maxWait)
    // Before the first probe: a caller that has already given up should not pay for a 1 s
    // CPU sample first.
    if callbacks.isCancelled() { return .cancelled }
    while true {
        let polled = ContinuousClock.now
        let probed = probe()
        // A probe spends a second or more sampling, long enough for a cancel to arrive.
        if callbacks.isCancelled() { return .cancelled }
        func status(_ phase: ReadinessPhase) -> ReadinessStatus {
            ReadinessStatus(phase: phase, platform: platform, observed: probed.summary,
                            elapsed: ContinuousClock.now - started, maxWait: maxWait)
        }
        if probed.ready {
            callbacks.report(status(.proceeding))
            return .ready
        }
        if ContinuousClock.now >= deadline {
            callbacks.report(status(.timedOut))
            return .timedOut(maxWait: maxWait, observed: probed.summary)
        }
        // The probe already spent part of this interval, so top up rather than sleep a
        // further full one; held to the deadline, or a timeout lands an interval late.
        let rest = clampToDeadline(max(.zero, sensorPollInterval - (ContinuousClock.now - polled)),
                                   deadline)
        callbacks.report(status(.waiting(pollIn: rest)))
        if callbacks.sleepCancellably(rest).isBreak() { return .cancelled }
    }
}
