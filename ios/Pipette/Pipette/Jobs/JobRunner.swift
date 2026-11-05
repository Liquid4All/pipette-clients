import Foundation
import Observation

/// The one busy gate over benchmark work: UI jobs and the planner claim loop share
/// this instance, so a device runs one cell at a time.
@Observable
final class JobRunner {
    var runningJobId: JobId?
    /// "benchmark / model" label for the cell currently executing.
    var currentCellLabel: String = ""
    /// Fine-grained progress text fed by the benchmark runner's callback.
    var currentProgressText: String = ""
    /// Within-cell progress in [0, 1], so progress bars move during a long cell rather
    /// than only at cell boundaries.
    var currentCellFraction: Double = 0
    /// The thermal readiness gate's status while it holds a rep to let the device cool;
    /// `nil` otherwise (loading, measuring), which hides the cooling banner.
    var readinessStatus: ReadinessStatus?
    /// When the current cooldown began: the anchor a `TimelineView` ticks the elapsed
    /// count from. Cleared when cooling ends.
    var readinessStartedAt: Date?
    /// Set by a user-initiated start/resume so the job's detail page presents
    /// Pocket Mode the moment it appears. Cleared once presented (and on finish).
    var pocketModeRequestedJobId: JobId?
    /// Wall-clock time the current run started. Reset on each start/retry so
    /// Pocket Mode can show elapsed time for the active run only.
    var startedAt: Date?
    /// Manifest `completedCells` when this run started. The ETA extrapolates from cells
    /// finished during this run, or resuming a 90%-done job would read "1s left".
    private(set) var completedAtStart: Int = 0
    /// Number of cells this run will execute (the pending count at start). The
    /// run-relative denominator for the ETA.
    private(set) var totalToRun: Int = 0
    private(set) var cancelFlag: CancelFlag?

    /// Projected finish, re-anchored as progress advances (see `anchorETA`). The ETA label
    /// counts down toward it so it ticks every second between cell boundaries.
    private(set) var projectedFinish: Date?

    var isRunning: Bool { runningJobId != nil }

    @discardableResult
    func start(jobId: JobId, flag: CancelFlag, completedAtStart: Int = 0, totalToRun: Int = 0) -> Bool {
        guard runningJobId == nil else { return false }
        runningJobId = jobId
        cancelFlag = flag
        currentCellLabel = ""
        currentProgressText = ""
        currentCellFraction = 0
        readinessStatus = nil
        readinessStartedAt = nil
        startedAt = Date()
        projectedFinish = nil
        self.completedAtStart = completedAtStart
        self.totalToRun = totalToRun
        return true
    }

    /// Set the ETA baselines after a successful `start()`. The resume/retry
    /// path guards `start()` before resetting cell statuses, so its pending
    /// count — and therefore these baselines — is only known afterwards.
    func setRunBaselines(completedAtStart: Int, totalToRun: Int) {
        self.completedAtStart = completedAtStart
        self.totalToRun = totalToRun
    }

    /// The current cooldown for the UI's live timer, or `nil` when not cooling.
    var coolingState: JobCoolingState? {
        guard let status = readinessStatus, status.isWaiting(), let since = readinessStartedAt else { return nil }
        return JobCoolingState(since: since, deadline: status.maxWait / .seconds(1), targetC: maxDieC)
    }

    /// SoC die temperature in °C (the read the gate judges), or `nil` with no signal so
    /// the UI shows the thermal-state label instead. Only readable on a
    /// `PIPETTE_PRIVATE_THERMAL` build.
    var deviceTemperatureC: Double? {
        dieTempMaxC()
    }

    /// Human-readable "N min/s left" for the active run, counting down toward the
    /// projected finish anchored by `anchorETA`. Nil until the first projection
    /// lands, or when `jobId` isn't the running job.
    func estimatedTimeLeft(jobId: JobId, now: Date) -> String? {
        guard runningJobId == jobId, let projectedFinish else { return nil }
        return Self.formatTimeLeft(max(0, projectedFinish.timeIntervalSince(now)))
    }

    /// Re-anchor `projectedFinish` from progress made this run (a resumed job's
    /// already-done cells don't count). Holding the projection fixed between calls is what
    /// lets the ETA count down every second.
    func anchorETA(completedCells: Int, now: Date = Date()) {
        guard runningJobId != nil, let started = startedAt, totalToRun > 0 else {
            projectedFinish = nil
            return
        }
        let within = max(0, min(1, currentCellFraction))
        let doneThisRun = Double(max(0, completedCells - completedAtStart)) + within
        let progress = min(1, doneThisRun / Double(totalToRun))
        // Too early to project meaningfully — keep any prior projection ticking.
        guard progress > 0.02 else { return }
        let elapsed = now.timeIntervalSince(started)
        projectedFinish = started.addingTimeInterval(elapsed / progress)
    }

    private static func formatTimeLeft(_ seconds: TimeInterval) -> String {
        if seconds < 60 {
            return "\(Int(seconds))s left"
        }
        return "\(Int((seconds / 60).rounded())) min left"
    }

    func cancel(reason: CancelFlag.Reason = .user) {
        cancelFlag?.cancel(reason: reason)
    }

    func finish(jobId: JobId? = nil) {
        if let jobId, runningJobId != jobId { return }
        runningJobId = nil
        cancelFlag = nil
        currentCellLabel = ""
        currentProgressText = ""
        currentCellFraction = 0
        readinessStatus = nil
        readinessStartedAt = nil
        pocketModeRequestedJobId = nil
        startedAt = nil
        projectedFinish = nil
        completedAtStart = 0
        totalToRun = 0
    }
}
