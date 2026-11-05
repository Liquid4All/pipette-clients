import Foundation
import Testing

@testable import Pipette

/// `ThermalSeries`: the crate's `pipette-ops/src/thermal_series.rs`. Array position is the
/// iteration index, so pairing is the contract, and a sequence that breaks it is refused
/// rather than silently misaligned. The first four cases port the crate tests of the same
/// names.
struct ThermalSeriesTests {
    /// Opening a second repetition without closing the first would pair the wrong two
    /// readings, so it fails the run instead.
    @Test func aSecondStartWithoutAnEndIsRefused() throws {
        let series = ThermalSeries<Int>()
        try series.start(0)

        #expect(throws: ThermalSeriesError.startedWhileOpen) { try series.start(1) }
    }

    @Test func anEndWithoutAStartIsRefused() {
        #expect(throws: ThermalSeriesError.endedWithoutStart) {
            try ThermalSeries<Int>().finish(0)
        }
    }

    @Test func matchedReportsPairUp() throws {
        let series = ThermalSeries<Int>()
        for _ in 0 ..< 3 {
            try series.start(0)
            try series.finish(0)
        }

        #expect(try series.take().count == 3)
    }

    /// A start with no end is invisible to the two reports (nothing follows it), so the
    /// drain is where that run has to fail.
    @Test func aRunThatEndsMidRepetitionIsRefused() throws {
        let series = ThermalSeries<Int>()
        try series.start(0)
        try series.finish(0)
        try series.start(1)

        #expect(throws: ThermalSeriesError.finishedWithOpenRep) { _ = try series.take() }
    }

    /// Each repetition's two readings stay together, in rep order.
    @Test func eachRepetitionKeepsItsOwnPair() throws {
        let series = ThermalSeries<Int>()
        try series.start(0)
        try series.finish(1)
        try series.start(2)
        try series.finish(3)

        let pairs = try series.take()
        #expect(pairs.map(\.start) == [0, 2])
        #expect(pairs.map(\.end) == [1, 3])
    }

    /// The drain empties the collector, as the crate's `take` does, so a second read
    /// reports no reps rather than the previous run's.
    @Test func takingTheSeriesEmptiesIt() throws {
        let series = ThermalSeries<Int>()
        try series.start(0)
        try series.finish(0)

        #expect(try series.take().count == 1)
        #expect(try series.take().isEmpty)
    }
}
