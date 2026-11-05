import Foundation
@testable import Pipette

/// The waits that a retry asked for and the events that it reported, with no real delay.
@MainActor final class RetryRecorder {
    var sleeps: [TimeInterval] = []
    var events: [RetryEvent] = []

    /// Every event except a recovery.
    var warnings: Int {
        events.count { if case .recovered = $0 { false } else { true } }
    }

    /// No jitter. Each wait runs `afterSleep`, so a test can hold the retry between attempts.
    func hooks(afterSleep: @escaping () async -> Void = {}) -> RetryHooks {
        RetryHooks(
            jitter: { $0 },
            sleep: { [self] in
                sleeps.append($0)
                await afterSleep()
            },
            report: { [self] in events.append($0) })
    }
}
