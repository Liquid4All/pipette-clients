import Foundation

// Pre-flight repair of eval submission payloads, mirroring `pipette-cli/src/score_validation.rs`.
// The scorer refuses a request with duplicate completion ids, and by the time that shows
// server-side the run is lost. Checkpointed evals are unique by construction; this repairs
// what slips through anyway (a `payload.json` written by an earlier build, say).

/// Why a payload could not be repaired — the crate's `ScoreValidationError`.
nonisolated enum ScoreValidationError: LocalizedError, Equatable {
    /// A `completions` entry without a string `id`. The id is the join key against the
    /// dataset, so an unkeyed entry cannot be deduplicated; dropping it would mask a bug.
    case completionMissingId(benchmarkId: String)

    var errorDescription: String? {
        switch self {
        case .completionMissingId(let benchmarkId):
            return "benchmark \(benchmarkId): completions entry missing string `id`; "
                + "refusing to dedupe /score payload"
        }
    }
}

/// Duplicate ids named verbatim in ``DedupReport/summary`` — the crate's
/// `MAX_LISTED_DUPLICATES`.
nonisolated let maxListedDuplicates = 10

/// What ``dedupeCompletionIds(_:)`` removed from a payload — the crate's `DedupReport`.
/// The caller logs ``summary``; nothing else consumes it.
nonisolated struct DedupReport: Equatable, Sendable {
    let benchmarkId: String
    let before: Int
    let after: Int
    /// The first ``maxListedDuplicates`` distinct duplicate ids, sorted.
    let listedDuplicateIds: [String]
    /// Total distinct duplicate ids; may exceed `listedDuplicateIds.count`.
    let distinctDuplicates: Int

    var summary: String {
        let removed = before - after
        let listed = listedDuplicateIds.joined(separator: ", ")
        let remainder = max(0, distinctDuplicates - listedDuplicateIds.count)
        let remainderClause = remainder == 0 ? "" : " (and \(remainder) more)"
        return "benchmark \(benchmarkId): deduplicated /score payload, removed \(removed) duplicate "
            + "completion entries across \(distinctDuplicates) distinct id(s) "
            + "[\(listed)\(remainderClause)]; kept last occurrence of each id "
            + "(\(after) of \(before) completions)"
    }
}

/// The crate's `dedupe_completion_ids`. Keeps the last entry per id (as the checkpoint's
/// append-then-load reads it) in original order; `nil` when nothing was removed.
nonisolated func dedupeCompletionIds(
    _ payload: inout [String: Any]
) throws(ScoreValidationError) -> DedupReport? {
    let benchmarkId = payload["benchmark_id"] as? String ?? "<unknown>"
    guard let completions = payload["completions"] as? [Any] else { return nil }

    let before = completions.count
    // Validate ids up front: an entry the server cannot key to a sample is a bug to report,
    // not a row to drop.
    var ids: [String] = []
    ids.reserveCapacity(before)
    for completion in completions {
        guard let id = (completion as? [String: Any])?["id"] as? String else {
            throw .completionMissingId(benchmarkId: benchmarkId)
        }
        ids.append(id)
    }

    // Last wins: walk backwards keeping the first sighting of each id, then flip back.
    var seen = Set<String>()
    var duplicateIds: [String] = []
    var kept: [Any] = []
    for (completion, id) in zip(completions, ids).reversed() {
        if seen.insert(id).inserted {
            kept.append(completion)
        } else {
            duplicateIds.append(id)
        }
    }
    guard !duplicateIds.isEmpty else { return nil }
    payload["completions"] = Array(kept.reversed())

    let distinct = Array(Set(duplicateIds)).sorted()
    return DedupReport(
        benchmarkId: benchmarkId,
        before: before,
        after: kept.count,
        listedDuplicateIds: Array(distinct.prefix(maxListedDuplicates)),
        distinctDuplicates: distinct.count)
}
