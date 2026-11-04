@testable import Pipette

/// The shared `NetworkReachability` reports offline until the app's launch path warms it,
/// so a test reading it would assert against the machine's network and launch order.
nonisolated struct FixedReachability: NetworkReachable {
    let isConnected: Bool
    /// A monitor with no reading yet is `WarmingReachability`'s case.
    var hasReading: Bool = true
}
