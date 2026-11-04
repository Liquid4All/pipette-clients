@testable import Pipette

/// The path monitor as a launch-time read finds it: offline until it warms `warmsAfter`
/// later. The two properties move together on purpose, so an unwarmed monitor looks offline
/// to a caller reading only `isConnected`, which is the confusion under test.
nonisolated struct WarmingReachability: NetworkReachable {
    private let warmAt: ContinuousClock.Instant

    init(warmsAfter: Duration) {
        warmAt = ContinuousClock.now.advanced(by: warmsAfter)
    }

    var hasReading: Bool { ContinuousClock.now >= warmAt }
    var isConnected: Bool { hasReading }
}
