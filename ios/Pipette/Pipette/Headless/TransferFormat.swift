import Foundation

/// Byte and rate rendering for a transfer in flight.
///
/// A deliberate port of the CLI's `crates/pipette-cli/src/progress.rs` (`bytes_of`,
/// `rate_of`, `unit_for`), so a download reads the same whichever client ran it. Swift
/// cannot share the Rust, so the rules are restated here and the tests pin them; change
/// one side and change the other.
nonisolated enum TransferFormat {
    /// `1.4/2.9 GB`, or `48.0 MB` when the total is unknown.
    ///
    /// Both halves render in one unit, chosen from the larger, so the pair reads as a
    /// single quantity — a 48 MB file on a GB scale would show `0.0/0.1 GB` and look
    /// stalled for its whole download.
    static func bytes(done: Int64, total: Int64?) -> String {
        if let total, total > 0 {
            let (scale, unit) = self.unit(for: total)
            return String(format: "%.1f/%.1f %@", Double(done) / scale, Double(total) / scale, unit)
        }
        let (scale, unit) = self.unit(for: done)
        return String(format: "%.1f %@", Double(done) / scale, unit)
    }

    /// `22.5 MB/s`, or empty for a figure that is not a number.
    ///
    /// The guard is not theoretical: `Int64(_:)` traps on a NaN or an out-of-range Double,
    /// and this runs inside a log line during a benchmark — a crash there would lose the
    /// run to a cosmetic.
    static func rate(bytesPerSecond: Double) -> String {
        guard bytesPerSecond.isFinite, bytesPerSecond >= 0,
              bytesPerSecond < Double(Int64.max)
        else { return "" }
        let (scale, unit) = unit(for: Int64(bytesPerSecond))
        return String(format: "%.1f %@/s", bytesPerSecond / scale, unit)
    }

    /// Divisor and name for the largest unit `bytes` fills. Decimal, as storage and
    /// transfer rates are quoted — a download is not a memory allocation.
    static func unit(for bytes: Int64) -> (scale: Double, name: String) {
        let kb = 1_000.0, mb = 1_000_000.0, gb = 1_000_000_000.0
        switch Double(bytes) {
        case gb...: return (gb, "GB")
        case mb..<gb: return (mb, "MB")
        case kb..<mb: return (kb, "KB")
        default: return (1, "B")
        }
    }
}
