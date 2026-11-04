import Foundation

/// The readings a run takes as each rep starts and finishes, stored as pairs: the crate's
/// `ThermalSeries` (`pipette-ops/src/thermal_series.rs`). Pairs, because two separate
/// `before`/`after` arrays drift silently once one rep contributes to one and not the other.
/// Owned by `RunCell.dispatchRun`; engines see only the `MeasurementObserver` over it.
///
/// `@unchecked Sendable`: written only from the observer hooks inside the sequential rep
/// loop, and read only after the run completes.
nonisolated final class ThermalSeries<R: Sendable>: @unchecked Sendable {
    /// `(start, end)` per finished rep, in rep order.
    private var finishedReps: [(start: R, end: R)] = []
    /// The start of the rep in flight; `nil` between reps.
    private var openRepStart: R?

    /// Opening a second rep before closing the first would pair the wrong readings, so it
    /// throws.
    func start(_ reading: R) throws {
        guard openRepStart == nil else { throw ThermalSeriesError.startedWhileOpen }
        openRepStart = reading
    }

    func finish(_ reading: R) throws {
        guard let start = openRepStart else { throw ThermalSeriesError.endedWithoutStart }
        openRepStart = nil
        finishedReps.append((start, reading))
    }

    /// The collected pairs, leaving the collector empty. A rep still open here has a start
    /// and no end, the one misreporting `start` and `finish` cannot see.
    func take() throws -> [(start: R, end: R)] {
        guard openRepStart == nil else { throw ThermalSeriesError.finishedWithOpenRep }
        defer { finishedReps = [] }
        return finishedReps
    }
}

/// A rep sequence that cannot be paired: the crate's `ThermalSeries` ensures, with their
/// messages. Array position is the iteration index, so a misaligned sequence is refused.
nonisolated enum ThermalSeriesError: Error, Equatable, LocalizedError {
    case startedWhileOpen
    case endedWithoutStart
    case finishedWithOpenRep

    var errorDescription: String? {
        switch self {
        case .startedWhileOpen:
            "a repetition was started while the previous one was still open"
        case .endedWithoutStart:
            "a repetition ended that was never started"
        case .finishedWithOpenRep:
            "the run finished with a repetition still open"
        }
    }
}
