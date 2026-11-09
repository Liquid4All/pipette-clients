import Foundation

/// The clock `attempt_timeout_seconds` runs on, one per run (`RunCell.dispatchRun`): the
/// crate's `pipette-cli/src/attempt_watchdog.rs`. An attempt is the model load and warm-up,
/// one measured rep, the work after a rep, or one peak-memory attempt, each timed on its own;
/// the clock stops while a readiness gate waits or a calibration runs.
///
/// The run marks it and the watchdog's thread reads it, so every field is under `lock`.
nonisolated final class AttemptClock: @unchecked Sendable {
    private let limit: Duration
    private let lock = NSLock()
    private var what = Attempt.loadAndWarmUp
    /// `nil` while held.
    private var started: ContinuousClock.Instant?
    /// The measured reps begun, which name theirs.
    private var reps = 0
    /// The overrun the watchdog caught and when its attempt started, kept for the next
    /// boundary: a `begin` restarts the attempt that ran over.
    private var caught: (overrun: Attempt.Overrun, started: ContinuousClock.Instant)?

    /// The first attempt, `Attempt.loadAndWarmUp`, starts at `now`.
    init(limit: Duration, at now: ContinuousClock.Instant = .now) {
        self.limit = limit
        started = now
    }

    /// A boundary: once the watchdog caught an overrun, the cell fails here, with how long the
    /// attempt that ran over has run by `now`: the engine ran on past the poll. `engineError` is
    /// what the engine returned after it, a boundary's own failure passed through as is.
    func check(
        after engineError: (any Error)? = nil, at now: ContinuousClock.Instant = .now
    ) throws {
        guard let caught = lock.withLock({ caught }) else { return }
        let overrun = Attempt.Overrun(
            what: caught.overrun.what, limit: caught.overrun.limit, elapsed: now - caught.started)
        throw (engineError as? AttemptTimeoutError)
            ?? AttemptTimeoutError(overrun: overrun, engineError: engineError?.localizedDescription)
    }

    /// A boundary, then a new attempt starts at `now` with the whole limit.
    func begin(_ what: String, at now: ContinuousClock.Instant = .now) throws {
        try check(at: now)
        lock.withLock {
            self.what = what
            started = now
        }
    }

    /// The next measured rep starts now.
    func beginRep() throws {
        let rep = lock.withLock {
            reps += 1
            return reps
        }
        try begin("rep \(rep)")
    }

    /// The rep just ended, and the work up to the next one is an attempt of its own.
    func beginAfterRep() throws {
        try begin("the work after rep \(lock.withLock { reps })")
    }

    /// The clock stops until the next `resume` or `begin`.
    func hold() {
        lock.withLock { started = nil }
    }

    /// The held attempt goes on with the whole limit.
    func resume(at now: ContinuousClock.Instant = .now) {
        lock.withLock { started = now }
    }

    /// The running attempt, once it is past the limit.
    func overrun(at now: ContinuousClock.Instant = .now) -> Attempt.Overrun? {
        lock.withLock {
            guard let started, now - started > limit else { return nil }
            return Attempt.Overrun(what: what, limit: limit, elapsed: now - started)
        }
    }

    /// Keep `overrun`, read at `now`, for the next boundary, unless one is kept already: it ran
    /// over first.
    func keep(_ overrun: Attempt.Overrun, at now: ContinuousClock.Instant = .now) {
        lock.withLock { caught = caught ?? (overrun, now - overrun.elapsed) }
    }
}

/// Watches a run's `AttemptClock` from a thread of its own, as the crate's watchdog does, so it
/// fires while an engine call holds every other one, the main thread included. Past the limit
/// the crate stops the run's children; an in-process engine has none and cannot be stopped
/// mid-call, so the launch ends through `abandon`, or without it the cell fails at its next
/// boundary.
nonisolated final class AttemptWatchdog: @unchecked Sendable {
    /// How late past the limit an overrun is caught.
    private static let poll = DispatchTimeInterval.milliseconds(250)

    private let clock: AttemptClock
    private let stop = DispatchSemaphore(value: 0)
    private let stopped = DispatchSemaphore(value: 0)

    /// `abandon` ends the process, so the clock never keeps that overrun for a boundary.
    init(watching clock: AttemptClock, abandon: (@Sendable (AttemptTimeoutError) -> Never)?) {
        self.clock = clock
        Thread { [self] in
            while stop.wait(timeout: .now() + Self.poll) == .timedOut {
                let now = ContinuousClock.now
                guard let overrun = clock.overrun(at: now) else { continue }
                if let abandon { abandon(AttemptTimeoutError(overrun: overrun, engineError: nil)) }
                AppLog.jobRun.error(
                    "\(overrun.what) is past attempt_timeout_seconds "
                        + "(\(overrun.limit.components.seconds)s); the engine cannot be stopped "
                        + "mid-call, so the cell fails when the engine next reaches a rep, an "
                        + "attempt or a gate, or returns")
                clock.keep(overrun, at: now)
                break
            }
            stopped.signal()
        }.start()
    }

    /// Stop watching. Once the limit fired, the run fails with `AttemptTimeoutError` whatever
    /// the engine said: it ran on past the limit. After an overrun `abandon` took, this waits
    /// for the exit.
    func finish(after engineError: (any Error)?) throws {
        stop.signal()
        stopped.wait()
        // An attempt past the limit now counts too, as in the crate: it ended before the poll.
        let now = ContinuousClock.now
        if let overrun = clock.overrun(at: now) { clock.keep(overrun, at: now) }
        try clock.check(after: engineError, at: now)
    }
}
