import Foundation

/// What a headless run says while `ensureModel` is fetching.
///
/// One line when the bytes start moving, one per decile, one at the end. The callback
/// fires per whole percent, which through the plan runner is thousands of lines per run,
/// but a silent minute looks like the hung launch `devicectl` produces.
///
/// Lines read like the CLI's renderer (`1.4/2.9 GB 22.5 MB/s`). The cadence differs on
/// purpose: the CLI redraws every 15 s for one run, while this shares a console with every
/// cell a plan runs. A cache hit prints nothing.
nonisolated final class DownloadProgressLog: @unchecked Sendable {
    private let label: String
    private let what: String
    private let clock = ContinuousClock()
    private let start: ContinuousClock.Instant
    /// Starts at the first decile, not before it: 0% is what the opening line already says.
    private var lastDecile = 0
    private var started = false
    /// Not construction time: counting the store lookup and queue wait made the first rate
    /// read a fraction of the real one.
    private var firstByteAt: ContinuousClock.Instant?
    private var lastBytes: Int64?
    private var lastTotal: Int64?

    /// `label` is the verb's own prefix (`bench`, `models pull`), `what` the artifact.
    init(label: String, what: String) {
        self.label = label
        self.what = what
        self.start = clock.now
    }

    /// Serial by construction (`awaitFetch` polls on one actor), so the counters need no lock.
    func report(_ progress: FetchProgress) {
        let percent = Int((progress.fraction * 100).rounded())
        // Only the opening line names the artifact: a model URI is long enough to crowd out
        // the number that moved.
        if !started {
            started = true
            HeadlessRunner.log("\(label) downloading \(what)")
        }
        if let done = progress.doneBytes {
            if firstByteAt == nil, done > 0 { firstByteAt = clock.now }
            lastBytes = done
            lastTotal = progress.totalBytes
        }
        // Completion lands between polls, so the last sample is short of the tail and the
        // closing line would read `503.4/522.2 MB` on a whole download.
        if progress.fraction >= 1, let total = lastTotal { lastBytes = total }
        let decile = percent / 10
        guard decile > lastDecile else { return }
        lastDecile = decile
        HeadlessRunner.log("\(label) downloading \(percent)%\(transferred)\(rate) \(elapsed)")
    }

    /// Silent when nothing was transferred, so a cache hit reports no zero-byte download.
    func finish() {
        guard started else { return }
        HeadlessRunner.log("\(label) downloaded\(transferred)\(rate) \(elapsed)")
    }

    /// Empty for a transfer that reports no bytes: an MLX directory pull, where HubApi
    /// exposes only a percentage.
    private var transferred: String {
        guard let done = lastBytes else { return "" }
        return " " + bytesOf(done: done, total: lastTotal)
    }

    /// Averaged over the transfer, since a single per-second poll interval swings too much.
    /// Blank below 0.25 s of transfer (the CLI's floor): one buffer over a few microseconds
    /// divides out to gigabytes per second.
    private var rate: String {
        guard let done = lastBytes, let firstByteAt else { return "" }
        // `components.seconds` alone truncates, dividing the first report by zero.
        let moving = clock.now - firstByteAt
        let seconds = Double(moving.components.seconds)
            + Double(moving.components.attoseconds) / 1e18
        guard seconds >= 0.25 else { return "" }
        let formatted = rateOf(Double(done) / seconds)
        return formatted.isEmpty ? "" : " " + formatted
    }

    private var elapsed: String {
        "(\(Int((clock.now - start).components.seconds))s)"
    }
}

// The CLI's `bytes_of`, `unit_for` and `rate_of` (crates/pipette-cli/src/progress.rs),
// restated so a download reads the same on either client. Change both sides together.

/// `1.4/2.9 GB`, or `48.0 MB` when the total is unknown. Both halves share the total's
/// unit, so the pair reads as one quantity.
nonisolated func bytesOf(done: Int64, total: Int64?) -> String {
    if let total, total > 0 {
        let (scale, unit) = unitFor(total)
        return String(format: "%.1f/%.1f %@", Double(done) / scale, Double(total) / scale, unit)
    }
    let (scale, unit) = unitFor(done)
    return String(format: "%.1f %@", Double(done) / scale, unit)
}

/// Decimal units, as storage and transfer rates are quoted.
nonisolated func unitFor(_ bytes: Int64) -> (scale: Double, name: String) {
    let kb = 1_000.0, mb = 1_000_000.0, gb = 1_000_000_000.0
    switch Double(bytes) {
    case gb...: return (gb, "GB")
    case mb..<gb: return (mb, "MB")
    case kb..<mb: return (kb, "KB")
    default: return (1, "B")
    }
}

/// `22.5 MB/s`, or empty for a figure that is not a number. The guard matters:
/// `Int64(_:)` traps on NaN or out-of-range, which would crash a benchmark over a log line.
nonisolated func rateOf(_ bytesPerSec: Double) -> String {
    guard bytesPerSec.isFinite, bytesPerSec >= 0, bytesPerSec < Double(Int64.max)
    else { return "" }
    let (scale, unit) = unitFor(Int64(bytesPerSec))
    return String(format: "%.1f %@/s", bytesPerSec / scale, unit)
}
