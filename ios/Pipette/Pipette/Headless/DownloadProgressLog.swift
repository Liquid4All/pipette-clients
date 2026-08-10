import Foundation

/// What a headless run says while `ensureModel` is fetching.
///
/// One line when the bytes start moving, one per decile after that, one when it ends —
/// where the callback behind it fires per whole percent, which is ~100 lines per model
/// and, through the plan runner, thousands per run. A silent minute is indistinguishable
/// from the hung launch `devicectl` produces, so the transfer has to be visible; it does
/// not have to be visible a hundred times.
///
/// Each line carries what the CLI's progress renderer carries — bytes against the total
/// and the rate they are moving at — so `1.4/2.9 GB 22.5 MB/s` reads the same on a phone
/// as on a desktop. The cadence deliberately does not match: the CLI redraws a log every
/// 15 s because it renders one run's own output, while this shares a console with every
/// cell a plan runs.
///
/// A cache hit fires none of them, which makes the first line the signal that this cell
/// is downloading rather than already resolved.
nonisolated final class DownloadProgressLog: @unchecked Sendable {
    private let label: String
    private let what: String
    private let clock = ContinuousClock()
    private let start: ContinuousClock.Instant
    /// Starts at the first decile, not before it: 0% is what the opening line already says.
    private var lastDecile = 0
    private var started = false
    /// When the first byte landed, which is not when this was constructed: the store
    /// lookup and the queue wait run in between, and counting them made the first rate
    /// read a fraction of the real one — at exactly the moment someone is watching.
    private var firstByteAt: ContinuousClock.Instant?
    private var lastBytes: Int64?
    private var lastTotal: Int64?

    /// `label` is the verb's own prefix (`bench`, `models pull`), `what` the artifact.
    init(label: String, what: String) {
        self.label = label
        self.what = what
        self.start = clock.now
    }

    /// Feed to `ensureModel(progress:)`. Serial by construction — `awaitFetch` polls on one
    /// actor — so the counters need no lock.
    func report(_ progress: FetchProgress) {
        let percent = Int((progress.fraction * 100).rounded())
        // Only the opening line names the artifact — a model URI carries a repo, a path
        // and a revision, and repeating it a dozen times crowds out the number that moved.
        if !started {
            started = true
            HeadlessRunner.log("\(label) downloading \(what)")
        }
        if let done = progress.doneBytes {
            if firstByteAt == nil, done > 0 { firstByteAt = clock.now }
            lastBytes = done
            lastTotal = progress.totalBytes
        }
        // Completion is the store resolving, which happens between polls — so the last
        // sample is short of the tail, and without this the closing line reads
        // `503.4/522.2 MB` on a download that finished whole.
        if progress.fraction >= 1, let total = lastTotal { lastBytes = total }
        let decile = percent / 10
        guard decile > lastDecile else { return }
        lastDecile = decile
        HeadlessRunner.log("\(label) downloading \(percent)%\(transferred)\(rate) \(elapsed)")
    }

    /// Call once the ensure returns. Says nothing when nothing was transferred, so a hit
    /// stays silent rather than reporting a download of zero bytes.
    func finish() {
        guard started else { return }
        HeadlessRunner.log("\(label) downloaded\(transferred)\(rate) \(elapsed)")
    }

    /// ` 1.4/2.9 GB`, or empty for a transfer that reports no bytes — an MLX directory
    /// pull, whose percentage is all HubApi exposes.
    private var transferred: String {
        guard let done = lastBytes else { return "" }
        return " " + TransferFormat.bytes(done: done, total: lastTotal)
    }

    /// ` 22.5 MB/s`, averaged over the transfer rather than sampled between reports: the
    /// poll behind this is per-second and a single interval's delta swings with it.
    ///
    /// Blank below a quarter-second of transfer, matching the CLI's floor — one buffer
    /// over a few microseconds divides out to gigabytes per second, which is the first
    /// figure a reader would see.
    private var rate: String {
        guard let done = lastBytes, let firstByteAt else { return "" }
        // Sub-second precision on purpose: `components.seconds` alone truncates, which
        // would divide the first report's bytes by zero or by a second it didn't take.
        let moving = clock.now - firstByteAt
        let seconds = Double(moving.components.seconds)
            + Double(moving.components.attoseconds) / 1e18
        guard seconds >= 0.25 else { return "" }
        let formatted = TransferFormat.rate(bytesPerSecond: Double(done) / seconds)
        return formatted.isEmpty ? "" : " " + formatted
    }

    private var elapsed: String {
        "(\(Int((clock.now - start).components.seconds))s)"
    }
}
