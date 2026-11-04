import Foundation

/// The active cooldown, for the UI's live timer. A `TimelineView` renders
/// `elapsed / deadline` from it, so no manual timer or per-second invalidation is needed.
nonisolated struct JobCoolingState: Equatable {
    let since: Date
    let deadline: Double
    /// The setpoint the device is cooling toward (°C), shown next to the current temperature.
    let targetC: Double

    /// Caption for a given render instant, e.g. `"Cooling 0:20 / 5:00 max"`.
    func caption(at now: Date) -> String {
        "Cooling \(Self.clock(now.timeIntervalSince(since))) / \(Self.clock(deadline)) max"
    }

    /// `m:ss`, held to `0...maxWaitCeiling` first so no value reaching the UI can trap the
    /// conversion to `Int`.
    private static func clock(_ seconds: Double) -> String {
        let ceiling = Double(maxWaitCeiling.components.seconds)
        let whole = seconds.isNaN ? 0 : Int(min(max(seconds, 0), ceiling).rounded(.down))
        return String(format: "%d:%02d", whole / 60, whole % 60)
    }
}
