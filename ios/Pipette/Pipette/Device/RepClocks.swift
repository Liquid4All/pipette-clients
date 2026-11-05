import Foundation

/// Logs each measured rep's CPU-cluster and GPU clock-state residency (`PipetteClocks.m`)
/// so a slow rep can be matched to a down-clock. Log only: nothing reaches the result. A
/// build without the private read logs nothing.
///
/// `@unchecked Sendable` as `ThermalSeries` is: the observer fires on the thread that runs
/// the reps, one rep at a time.
nonisolated final class RepClocks: @unchecked Sendable {
    private let marker: String
    private var start: UnsafeRawPointer?
    private var rep = 0

    init(marker: String) {
        self.marker = marker
        _ = Self.subscribe
    }

    /// The first sample subscribes, which walks every driver's report legend: done here,
    /// before the readiness gate, rather than between the gate and rep 1's timer.
    private static let subscribe: Void = {
        let sample = pipette_clock_sample()
        defer { pipette_clock_release(sample) }
        if sample == nil, pipette_private_thermal_build() != 0 {
            AppLog.jobRun.error("clocks: IOReport unreadable, no per-rep clock lines")
        }
    }()

    deinit {
        pipette_clock_release(start)
    }

    /// Last thing before the timed window opens.
    func repStarting() {
        pipette_clock_release(start)
        start = pipette_clock_sample()
    }

    /// First thing after the timed window closes.
    func repEnded() {
        let end = pipette_clock_sample()
        defer {
            pipette_clock_release(start)
            pipette_clock_release(end)
            start = nil
        }
        rep += 1
        guard let residency = pipette_clock_residency(start, end) else { return }
        AppLog.jobRun.info("\(marker) rep \(rep) clocks: \(residency)")
    }
}
