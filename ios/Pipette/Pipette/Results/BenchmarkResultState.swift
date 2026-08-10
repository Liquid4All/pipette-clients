import Foundation

/// A stored result's address: `<jobId>/<cellId>`, the spelling `results show result=…`
/// takes and the CLI's `--result` analogue.
///
/// iOS has no single flat result id the way the desktop store does — a result lives under
/// its job — so the pair *is* the id rather than something derived from one.
nonisolated struct ResultId: Hashable, Sendable, CustomStringConvertible {
    let jobId: JobId
    let cellId: CellId

    var description: String { "\(jobId.value)/\(cellId.value)" }

    /// Parse the `<jobId>/<cellId>` form. Exactly two non-empty segments: a job id alone
    /// addresses a job, not a result, and `job=` is the parameter for that.
    static func parse(_ raw: String) -> ResultId? {
        let parts = raw.split(separator: "/", omittingEmptySubsequences: false)
        guard parts.count == 2, !parts[0].isEmpty, !parts[1].isEmpty else { return nil }
        return ResultId(jobId: JobId(String(parts[0])), cellId: CellId(String(parts[1])))
    }
}

/// How far a stored result has travelled — the counterpart of the CLI's
/// `BenchmarkResultState` in `pipette-cli/src/results/types.rs`.
///
/// Three rungs, as the crate has: a result that has not gone up, one the collector took,
/// and one that came back scored. ``BenchmarkResultLocation`` is what a result's
/// directory says, and `store.rs:168` maps it here — both `local` and `remotePending`
/// are `recorded`, because "generated here" and "waiting its turn" are the same rung.
/// What separates them is the benchmark's provenance, which the cell already carries.
///
/// The crate spells this rung `Local` after its directory; iOS keeps `recorded`, since
/// `local` here would collide with the catalog half a benchmark came from.
///
/// Ordered, so "most advanced wins" is the enum's own comparison rather than a rule each
/// caller re-implements.
nonisolated enum BenchmarkResultState: String, CaseIterable, Comparable, Sendable {
    /// `payload.json` written — the run finished and produced measurements.
    case recorded
    /// `submission.json` written — the collector accepted it.
    case submitted
    /// `metrics.json` written — the server scored it and the score came back.
    case scored

    private var rank: Int {
        switch self {
        case .recorded: 0
        case .submitted: 1
        case .scored: 2
        }
    }

    static func < (lhs: BenchmarkResultState, rhs: BenchmarkResultState) -> Bool { lhs.rank < rhs.rank }
}
