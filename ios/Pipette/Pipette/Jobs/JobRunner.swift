import Foundation
import Observation
import SwiftUI

/// Shared, reused date formatters. Allocating a `DateFormatter` is expensive,
/// and scattering ad-hoc copies meant the `yyyy-MM-dd` formatter disagreed on
/// time zone (GMT vs current) across screens — so the same job could show a
/// different created-date depending on where you looked. One definition each.
///
/// `nonisolated(unsafe)` for the non-`Sendable` formatters: they're configured
/// once and only ever *read* (format/parse), which is thread-safe in practice, so
/// the nonisolated data layer (JobManifest/IdentityRegistration) and detached executor
/// tasks can use them without hopping to the main actor.
enum JobDateFormat {
    nonisolated(unsafe) static let relative: RelativeDateTimeFormatter = {
        let f = RelativeDateTimeFormatter()
        f.unitsStyle = .abbreviated
        return f
    }()

    nonisolated(unsafe) static let iso8601 = ISO8601DateFormatter()

    /// `yyyy-MM-dd` in the device's local time zone. `DateFormatter` is `Sendable`,
    /// so plain `nonisolated` (no `unsafe`) suffices here.
    nonisolated static let shortDate: DateFormatter = {
        let f = DateFormatter()
        f.calendar = Calendar(identifier: .gregorian)
        f.locale = Locale(identifier: "en_US_POSIX")
        f.timeZone = .current
        f.dateFormat = "yyyy-MM-dd"
        return f
    }()
}

/// SoC die-temperature ceiling (°C) the device must reach before a benchmark rep
/// proceeds (when the private SoC temp is available). 36°C: the deepest *reliable*
/// setpoint — the measured idle floor is ~35°C (room ambient + ~5–7°C), so 36°C is
/// ~1°C of margin above it for maximum thermal headroom. A 34–35°C gate sits at/below
/// the floor and times out; 40°C is faster but leaves less headroom. Floor is
/// ambient-dependent, so raise this in a warmer room. Only used when the private SoC
/// temp is available (PIPETTE_PRIVATE_THERMAL build); otherwise the gate uses thermalState.
nonisolated let thermalThresholdC: Double = 36.0

/// Setpoint for the public IMU-estimate fallback (°C). Higher than the soc_temp
/// setpoint on purpose: the IMU estimate is ~±1.5–2 °C, so we target `< 40 °C` with
/// margin (gate at 38 → true temp reliably under 40 on both test units, vs a tight
/// near-floor 36 °C where the IMU noise false-triggers). Used only when soc_temp is
/// unavailable AND the IMU has been calibrated.
nonisolated let imuThresholdC: Double = 38.0

/// How often the readiness gate re-reads the sensor while waiting.
///
/// Granularity, not criteria: the threshold decides whether a rep may start and has to
/// match every other client, while this only decides how soon that is noticed. A device
/// that cools in a second still waits a whole interval, so at the reference probes' 10 s a
/// measured cell spent 46 of 59 seconds gating — most of it after the die was already
/// under the setpoint. The Android client polls at 2 s for the same reason; the soc_temp
/// read is cheap enough that the interval buys nothing.
nonisolated let readinessPollInterval: Double = 3.0

/// How long the readiness gate will wait for the device to cool before giving up
/// and failing the rep. Sized so a hot device gets a realistic chance to reach the
/// setpoint (a serious/critical throttle can take minutes) without hanging a job
/// indefinitely on a device that never cools (e.g. a warm room above the setpoint).
/// Also the "allowed to cool" deadline the progress UI counts against.
///
/// This is the shared cross-platform default (`pipette_readiness::DEFAULT_MAX_WAIT`,
/// which documents the choice; Android carries it as `Readiness.COOLDOWN_MAX_MILLIS`).
/// Swift cannot read that constant, so this restates it: changing it here diverges
/// this client from every other, which is what PIP-278 was raised about.
nonisolated let readinessTimeoutSeconds: Double = 300

/// Classify one reading. `thermalState` and the temperature signal are evaluated
/// **at the same time** — both must pass to proceed:
/// - `thermalState != .nominal` (fair/serious/critical) is an **immediate** signal to
///   wait: the OS already sees thermal pressure, so we cool regardless of what the
///   temperature signal reads. Checked first.
/// - With `thermalState == .nominal`, gate on the temperature signal if we have one:
///   the exact private SoC temp (`PIPETTE_PRIVATE_THERMAL`, `temp > 0`) or the public
///   IMU estimate. A missing temperature signal (`temp <= 0`) is NOT a failure — a
///   nominal thermalState alone is enough to proceed.
/// What the gate actually runs with — the resolved form of the crate's
/// `ReadinessOverrides`, whose fields are optional because they describe a *request*.
/// Defaults are the built-in gate, so a caller that overrides neither is gated as before.
nonisolated struct ReadinessPolicy: Sendable, Equatable {
    var maxSeconds: Double = readinessTimeoutSeconds
    var skipThermal: Bool = false
}

nonisolated func isThermallyReady(temp: Double, state: ProcessInfo.ThermalState,
                                  thresholdC: Double, skipThermal: Bool = false) -> Bool {
    if state != .nominal { return false }          // thermal pressure → wait now
    // `skipThermal` waives the *temperature* criterion and keeps this one, matching the
    // crate's `--readiness-skip-thermal`: it changes what "ready" means, so a cell run with
    // it is not comparable to a gated one.
    if skipThermal { return true }
    if temp > 0 { return temp < thresholdC }
    // No temperature signal (no PIPETTE_PRIVATE_THERMAL build and no calibrated IMU):
    // `thermalState == .nominal` is the only gate. It is coarse — it can stay `.nominal`
    // well into throttle — so a real cool-gate needs the private flag or a calibrated IMU;
    // this branch is best-effort, not a guarantee that the device is at baseline.
    return true
}

/// One-line summary of a thermal reading, e.g. `thermal=temp:42.0°C(hot,<36) state:nominal`.
/// Reused both for the gate log and as the `observed` detail attached to a
/// timeout / sensor failure — mirroring the reference probes' `observed` string.
nonisolated func readinessSummary(temp: Double, state: ProcessInfo.ThermalState, thresholdC: Double) -> String {
    let verdict = temp <= 0 ? "unreadable" : (temp < thresholdC ? "ok" : "hot")
    return String(format: "thermal=temp:%.1f°C(%@,<%.0f) state:%@",
                  temp, verdict, thresholdC, state.shortLabel)
}

/// CLI-style one-line gate log, mirroring the `readiness: <platform> … → <action>`
/// lines the `pipette-llamacpp` readiness probes emit in `pipette-ops`.
nonisolated func readinessLog(temp: Double, state: ProcessInfo.ThermalState, thresholdC: Double, action: String) -> String {
    "readiness: ios \(readinessSummary(temp: temp, state: state, thresholdC: thresholdC)) → \(action)"
}

/// A single reading from the pre-rep thermal readiness gate, structured so the
/// UI can render a human-readable "Awaiting readiness" state instead of the raw
/// CLI log line. `description` reproduces that CLI line for logging.
nonisolated struct ReadinessStatus: Sendable, Equatable, CustomStringConvertible {
    enum Phase: Sendable, Equatable {
        case waiting        // gate is holding the rep while the device cools
        case proceeding     // device is cool enough; the rep is about to run
        case timedOut       // gave up waiting — the cell will fail
    }

    var phase: Phase
    /// Measured device temperature in °C, or `nil` when there is no temperature
    /// signal (no `PIPETTE_PRIVATE_THERMAL` build and no calibrated IMU). Kept
    /// optional on purpose: the raw `-1` sentinel must never reach the UI.
    var temperatureC: Double?
    /// OS thermal-state label (`ProcessInfo.ThermalState.shortLabel`), captured as
    /// a plain value so the status stays `Sendable` across the gate's background
    /// thread into the main-actor UI.
    var thermalStateLabel: String
    var thresholdC: Double
    var elapsedSeconds: Int
    /// How long the gate will wait before giving up and failing the rep — the
    /// "allowed to cool" deadline, surfaced so the UI can show elapsed vs. limit.
    var maxSeconds: Double
    /// The CLI action phrase (e.g. `"waiting 10s (30s)"`), for the debug log only.
    var action: String

    /// The gate is actively holding a rep — the only phase the UI banners.
    var isWaiting: Bool { phase == .waiting }

    var description: String {
        let verdict = temperatureC == nil ? "unreadable" : (temperatureC! < thresholdC ? "ok" : "hot")
        return String(format: "readiness: ios thermal=temp:%.1f°C(%@,<%.0f) state:%@ → %@",
                      temperatureC ?? -1, verdict, thresholdC, thermalStateLabel, action)
    }
}

/// The active cooldown, for the UI's live timer. Carries the start anchor and the
/// allowed-to-cool deadline so a `TimelineView` renders `elapsed / deadline` and
/// ticks itself once a second — no manual timer, no per-second view invalidation.
nonisolated struct JobCoolingState: Equatable {
    let since: Date
    let deadline: Double
    /// The setpoint the device is cooling toward (°C) — shown next to the current
    /// temperature so the user knows the target the gate is waiting for.
    let targetC: Double

    /// Caption for a given render instant, e.g. `"Cooling 0:20 / 5:00 max"`.
    func caption(at now: Date) -> String {
        let elapsed = max(0, Int(now.timeIntervalSince(since)))
        return "Cooling \(Self.clock(elapsed)) / \(Self.clock(Int(deadline.rounded()))) max"
    }

    private static func clock(_ seconds: Int) -> String {
        String(format: "%d:%02d", seconds / 60, seconds % 60)
    }
}

extension ProcessInfo.ThermalState {
    nonisolated var shortLabel: String {
        switch self {
        case .nominal:  return "nominal"
        case .fair:     return "fair"
        case .serious:  return "serious"
        case .critical: return "critical"
        @unknown default: return "unknown"
        }
    }
}

// MARK: - Job runner

@Observable
final class JobRunner {
    var runningJobId: JobId?
    /// "benchmark / model" label for the cell currently executing.
    var currentCellLabel: String = ""
    /// Fine-grained progress text fed by the benchmark runner's callback.
    var currentProgressText: String = ""
    /// Within-cell progress in [0, 1]. Combined with `completedCells/totalCells`
    /// so job progress bars move smoothly during a long-running cell instead of
    /// only ticking at cell boundaries.
    var currentCellFraction: Double = 0
    /// Live status of the thermal readiness gate while it holds a rep to let the
    /// device cool. `nil` whenever the gate isn't waiting (loading / measuring),
    /// which is what drives the "Awaiting readiness" banner on and off.
    var readinessStatus: ReadinessStatus?
    /// Wall-clock instant the current cooldown began — the anchor a `TimelineView`
    /// uses to tick the elapsed count once a second (no polling of its own). Set
    /// when cooling starts, cleared the moment it ends.
    var readinessStartedAt: Date?
    /// Set by a user-initiated start/resume so the job's detail page presents
    /// Pocket Mode the moment it appears. Cleared once presented (and on finish).
    var pocketModeRequestedJobId: JobId?
    /// Wall-clock time the current run started. Reset on each start/retry so
    /// Pocket Mode can show elapsed time for the active run only.
    var startedAt: Date?
    /// Manifest `completedCells` at the instant this run started. A resumed or
    /// retried job already has completed cells this run won't re-execute, so the
    /// ETA must extrapolate from cells finished *during this run* — not the whole
    /// manifest. Without this baseline, resuming a 90%-done job reads "1s left".
    private(set) var completedAtStart: Int = 0
    /// Number of cells this run will execute (the pending count at start). The
    /// run-relative denominator for the ETA.
    private(set) var totalToRun: Int = 0
    private(set) var cancelFlag: CancelFlag?

    /// Wall-clock instant this run is projected to finish, re-anchored whenever
    /// progress advances (see `anchorETA`). The ETA label counts `now` *down*
    /// toward it so the seconds tick every second, instead of sitting static —
    /// or creeping up — between cell boundaries.
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
    /// Carries the start anchor and the allowed-to-cool deadline so a `TimelineView`
    /// can render `elapsed / deadline` and tick it every second on its own.
    var coolingState: JobCoolingState? {
        guard let status = readinessStatus, status.isWaiting, let since = readinessStartedAt else { return nil }
        return JobCoolingState(since: since, deadline: status.maxSeconds, targetC: status.thresholdC)
    }

    /// Device temperature in °C for display, or `nil` when there's no signal — the
    /// UI then shows the thermal-state label instead of a bogus `-1`. Prefers the
    /// gate's live reading while cooling; otherwise a direct SoC read, which is
    /// only nonzero on a `PIPETTE_PRIVATE_THERMAL` build (the sensor-ON case).
    var deviceTemperatureC: Double? {
        if let cooling = readinessStatus?.temperatureC { return cooling }
        let soc = socTemp()
        return soc > 0 ? soc : nil
    }

    /// Human-readable "N min/s left" for the active run, counting down toward the
    /// projected finish anchored by `anchorETA`. Nil until the first projection
    /// lands, or when `jobId` isn't the running job.
    func estimatedTimeLeft(jobId: JobId, now: Date) -> String? {
        guard runningJobId == jobId, let projectedFinish else { return nil }
        return Self.formatTimeLeft(max(0, projectedFinish.timeIntervalSince(now)))
    }

    /// Re-anchor `projectedFinish` from progress made so far *this run*. Called on
    /// the main actor whenever the completed-cell count or within-cell fraction
    /// advances; holding the projection fixed between calls is what lets the ETA
    /// count down second-by-second rather than only stepping at cell boundaries.
    ///
    /// Progress is run-relative, not whole-manifest: a resumed/retried job's
    /// already-done cells this run won't re-execute, so counting them would read a
    /// near-instant ETA.
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

// MARK: - Cancellation & progress

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

/// The single iOS thermal readiness gate, invoked from the benchmark via
/// `ReadinessCallback` before each measured rep (the gate before a cell's first
/// rep also serves as the between-cell cooldown). Blocks until the SoC die
/// temperature is below `thresholdC` and the thermal state is `.nominal`,
/// re-reading every `readinessPollInterval`, then returns a [`ReadinessOutcome`].
/// A timeout or unreadable sensor fails the cell rather than measuring under
/// unknown thermal conditions.
nonisolated func waitUntilThermallyReady(
    thresholdC: Double = thermalThresholdC,
    maxSeconds: Double = readinessTimeoutSeconds,
    skipThermal: Bool = false,
    cancelFlag: CancelFlag,
    progress: (ReadinessStatus) -> Void
) -> ReadinessOutcome {
    if cancelFlag.isCancelled { return .cancelled }

    #if targetEnvironment(simulator)
    // The simulator has no SoC die-temperature sensor, so there is nothing to
    // gate on — proceed (mirrors the no-op readiness path that `pipette-ops`
    // uses for platforms without probes).
    progress(ReadinessStatus(phase: .proceeding, temperatureC: nil,
                             thermalStateLabel: ProcessInfo.ThermalState.nominal.shortLabel,
                             thresholdC: thresholdC, elapsedSeconds: 0, maxSeconds: maxSeconds,
                             action: "simulator: no thermal sensor, proceeding"))
    return .ready
    #else
    // Monotonic clock — wall-clock `Date` could step (NTP / time change) mid-wait
    // and skew the cooldown timeout.
    let clock = ContinuousClock()
    let start = clock.now

    while true {
        if cancelFlag.isCancelled { return .cancelled }
        // `thermalState` is always-on (see isThermallyReady: != .nominal ⇒ wait
        // immediately). Alongside it we pick the temperature dimension: the exact
        // private soc_temp if available, else the public IMU estimate (if calibrated).
        // `effThreshold` matches whichever temp signal is driving (soc_temp uses
        // `thresholdC`, the ~±1.5 °C IMU uses the margin-padded `imuThresholdC`).
        let socT = socTemp()                                   // max die temp C, or -1
        let temp: Double
        let effThreshold: Double
        if socT > 0 {
            temp = socT
            effThreshold = thresholdC
        } else if let est = IMUThermometer.estimate() {
            temp = est
            effThreshold = imuThresholdC
        } else {
            temp = -1
            effThreshold = thresholdC
        }
        let state = ProcessInfo.processInfo.thermalState
        let elapsed = clock.now - start
        let elapsedSeconds = Int(elapsed.components.seconds)
        // Hide the -1 sentinel from anything user-facing: a non-positive reading
        // means no signal, surfaced as a nil temperature.
        let reportedTemp: Double? = temp > 0 ? temp : nil
        let stateLabel = state.shortLabel
        if isThermallyReady(temp: temp, state: state, thresholdC: effThreshold,
                            skipThermal: skipThermal) {
            progress(ReadinessStatus(phase: .proceeding, temperatureC: reportedTemp,
                                     thermalStateLabel: stateLabel,
                                     thresholdC: effThreshold, elapsedSeconds: elapsedSeconds,
                                     maxSeconds: maxSeconds,
                                     action: "proceeding (\(elapsedSeconds)s)"))
            return .ready
        } else {
            if elapsed >= .seconds(maxSeconds) {
                progress(ReadinessStatus(phase: .timedOut, temperatureC: reportedTemp,
                                         thermalStateLabel: stateLabel,
                                         thresholdC: effThreshold, elapsedSeconds: elapsedSeconds,
                                         maxSeconds: maxSeconds,
                                         action: "timed out after \(Int(maxSeconds))s, failing rep"))
                return .timedOut(observed:
                    "\(readinessSummary(temp: temp, state: state, thresholdC: effThreshold)) after \(elapsedSeconds)s")
            }
            progress(ReadinessStatus(phase: .waiting, temperatureC: reportedTemp,
                                     thermalStateLabel: stateLabel,
                                     thresholdC: effThreshold, elapsedSeconds: elapsedSeconds,
                                     maxSeconds: maxSeconds,
                                     action: "waiting \(Int(readinessPollInterval))s (\(elapsedSeconds)s)"))
            // Wake every second so a cancel is honored promptly, not only at the
            // end of the interval. The UI's live cooldown timer is driven by a
            // `TimelineView` off the cooling start, not by re-emitting here.
            let wakeAt = clock.now.advanced(by: .seconds(readinessPollInterval))
            while clock.now < wakeAt {
                if cancelFlag.isCancelled { return .cancelled }
                Thread.sleep(forTimeInterval: 1.0)
            }
        }
    }
    #endif
}

/// Bridges the `ReadinessCallback` protocol to `waitUntilThermallyReady`, invoked
/// synchronously on the benchmark's background thread before each measured rep.
final class BenchmarkReadiness: @unchecked Sendable, ReadinessCallback {
    private let cancelFlag: CancelFlag
    private let thresholdC: Double
    private let maxSeconds: Double
    private let skipThermal: Bool
    private let onStatus: @Sendable (ReadinessStatus) -> Void

    nonisolated init(cancelFlag: CancelFlag, thresholdC: Double = thermalThresholdC,
                     maxSeconds: Double = readinessTimeoutSeconds, skipThermal: Bool = false,
                     onStatus: @escaping @Sendable (ReadinessStatus) -> Void) {
        self.cancelFlag = cancelFlag
        self.thresholdC = thresholdC
        self.maxSeconds = maxSeconds
        self.skipThermal = skipThermal
        self.onStatus = onStatus
    }

    nonisolated func waitUntilReady() -> ReadinessOutcome {
        waitUntilThermallyReady(thresholdC: thresholdC, maxSeconds: maxSeconds,
                                skipThermal: skipThermal, cancelFlag: cancelFlag,
                                progress: onStatus)
    }
}
