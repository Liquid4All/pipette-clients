import Foundation
import Testing

@testable import Pipette

/// The byte/rate rendering a headless download line carries.
///
/// These pin the CLI's rules (`crates/pipette-cli/src/progress.rs`) on this side of the
/// language boundary — the two cannot share code, so the agreement is only as good as
/// what is asserted here.
struct TransferFormatTests {

    /// Both halves in one unit, chosen from the *larger*. Scaled to the done side, a
    /// nearly-finished 2.9 GB download would read `1400.0/2.9` and a barely-started one
    /// would flip units between lines.
    @Test func bothHalvesRenderInTheTotalsUnit() {
        #expect(TransferFormat.bytes(done: 1_400_000_000, total: 2_900_000_000) == "1.4/2.9 GB")
        #expect(TransferFormat.bytes(done: 48_000_000, total: 96_000_000) == "48.0/96.0 MB")
    }

    /// The case the single-unit rule exists for: a small artifact on a GB scale reads
    /// `0.0/0.1 GB` and looks stalled for its whole download.
    @Test func aSmallTotalDoesNotRenderOnAGigabyteScale() {
        #expect(TransferFormat.bytes(done: 1_000_000, total: 48_000_000) == "1.0/48.0 MB")
    }

    /// An unsized transfer still reports what has landed — that is what distinguishes a
    /// slow download from a hung one, which is the whole reason the line exists.
    @Test func anUnknownTotalReportsTheBytesSoFar() {
        #expect(TransferFormat.bytes(done: 48_000_000, total: nil) == "48.0 MB")
        // A declared zero is not a size, and dividing by it would render `inf`.
        #expect(TransferFormat.bytes(done: 512, total: 0) == "512.0 B")
    }

    @Test func rateRendersInTheUnitItFills() {
        #expect(TransferFormat.rate(bytesPerSecond: 22_500_000) == "22.5 MB/s")
        #expect(TransferFormat.rate(bytesPerSecond: 1_500) == "1.5 KB/s")
    }

    /// `Int64(_:)` traps on a NaN or an out-of-range Double, and this runs inside a log
    /// line mid-benchmark. Rendering nothing is the correct answer; crashing the run over
    /// a progress line is not.
    @Test func anUnrepresentableRateRendersNothingRatherThanTrapping() {
        #expect(TransferFormat.rate(bytesPerSecond: .nan).isEmpty)
        #expect(TransferFormat.rate(bytesPerSecond: .infinity).isEmpty)
        #expect(TransferFormat.rate(bytesPerSecond: -1).isEmpty)
        #expect(TransferFormat.rate(bytesPerSecond: Double(Int64.max)).isEmpty)
    }
}
