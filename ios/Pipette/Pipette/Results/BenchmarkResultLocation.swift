import Foundation

/// Where a result lives, which is also how far it has travelled — the crate's
/// `BenchmarkResultLocation` (`pipette-cli/src/results/types.rs:14`).
///
/// The directory *is* the status. A result recorded from a `local/` benchmark lands in
/// `local/` and is never submitted; one from the synced catalog lands in
/// `remote/pending/` and moves to `remote/synced/` when the collector accepts it. That
/// is the crate's `BenchmarkResultLocation::from(BenchmarkSource)` plus its
/// `move_result_dir`, and it replaces deriving the ladder from which files happen to
/// exist beside each other.
///
/// TODO: review — mirrors `types.rs:14`; three variants and the directory names checked
/// against `location_dir`.
nonisolated enum BenchmarkResultLocation: String, CaseIterable, Sendable {
    /// Generated benchmark, not submittable — the crate's `results/local/`.
    case local
    /// Recorded and awaiting submission — `results/remote/pending/`.
    case remotePending
    /// The collector accepted it — `results/remote/synced/`.
    case remoteSynced

    /// The path under the store root, as `location_dir` spells it.
    var directory: String {
        switch self {
        case .local: return "local"
        case .remotePending: return "remote/pending"
        case .remoteSynced: return "remote/synced"
        }
    }

    /// Where a freshly recorded result goes, given the half its benchmark came from —
    /// the crate's `From<BenchmarkSource>`. A body from the synced catalog is
    /// submittable, so its result waits for the next sweep; one only this device has is
    /// not, so it stays local.
    init(recordedFrom source: BenchmarkSource) {
        switch source {
        case .local: self = .local
        case .remote: self = .remotePending
        }
    }

    /// The lifecycle state this location implies. The crate spells the first rung
    /// `Local` after the directory; iOS calls it `recorded`, and that name is kept
    /// because it is what the UI and CSV export already show.
    var state: BenchmarkResultState {
        switch self {
        case .local, .remotePending: return .recorded
        case .remoteSynced: return .submitted
        }
    }
}
