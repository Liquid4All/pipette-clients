import Foundation

/// Thread-safe cancellation flag shared between UI and benchmark runner.
nonisolated final class CancelFlag: @unchecked Sendable {
    /// Who tripped the flag. A user cancel is a deliberate pause; a
    /// background cancel is the app protecting benchmark validity when the
    /// scene leaves the foreground. The paused job records the difference.
    nonisolated enum Reason: Sendable {
        case user
        case background
    }

    nonisolated(unsafe) private var _cancelled = false
    nonisolated(unsafe) private var _reason: Reason?
    nonisolated private let lock = NSLock()

    nonisolated var isCancelled: Bool {
        lock.lock()
        defer { lock.unlock() }
        return _cancelled
    }

    /// Why the flag was tripped; nil until cancelled. The first cancel wins
    /// so a follow-up tap can't relabel an auto-pause (or vice versa).
    nonisolated var reason: Reason? {
        lock.lock()
        defer { lock.unlock() }
        return _reason
    }

    nonisolated func cancel(reason: Reason = .user) {
        lock.lock()
        if !_cancelled {
            _cancelled = true
            _reason = reason
        }
        lock.unlock()
    }
}
