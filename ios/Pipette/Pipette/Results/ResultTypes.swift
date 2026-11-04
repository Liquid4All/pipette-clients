import Foundation

// A stored result's location, state, address and `extras.json` sidecar: `pipette-cli`'s
// `results/types.rs`. The wire payload is in `PlanTypes/Submission.swift`.

/// The crate's `BenchmarkResultLocation`. The directory is the status: a result from a
/// `local/` benchmark is never submitted; one from the synced catalog lands in
/// `remote/pending/` and moves to `remote/synced/` when the collector accepts it.
nonisolated enum BenchmarkResultLocation: String, CaseIterable, Sendable {
    /// Generated benchmark, not submittable — the crate's `results/local/`.
    case local
    /// Recorded and awaiting submission — `results/remote/pending/`.
    case remotePending
    /// The collector accepted it — `results/remote/synced/`.
    case remoteSynced

    var label: String {
        switch self {
        case .local: return "local"
        case .remotePending: return "pending"
        case .remoteSynced: return "synced"
        }
    }

    /// The crate's `From<BenchmarkSource>`: only a synced-catalog benchmark is submittable.
    init(_ source: BenchmarkSource) {
        switch source {
        case .local: self = .local
        case .remote: self = .remotePending
        }
    }
}

/// A stored result's address, `<jobId>/<cellId>` (`results show result=`). iOS has no flat
/// result id like the desktop store: a result lives under its job, so the pair is the id.
nonisolated struct ResultId: Hashable, Sendable, CustomStringConvertible {
    let jobId: JobId
    let cellId: CellId

    var description: String { "\(jobId.value)/\(cellId.value)" }

    /// A job id alone addresses a job, not a result: `job=` is the parameter for that.
    static func parse(_ raw: String) -> ResultId? {
        let parts = raw.split(separator: "/", omittingEmptySubsequences: false)
        guard parts.count == 2, !parts[0].isEmpty, !parts[1].isEmpty else { return nil }
        return ResultId(jobId: JobId(String(parts[0])), cellId: CellId(String(parts[1])))
    }
}

/// The crate's `BenchmarkResultState`. Both `local` and `remotePending` locations map to
/// `local`: what separates them is the benchmark's provenance, which the cell carries.
/// Raw values are the crate's printed spellings, used by `results list state=`.
nonisolated enum BenchmarkResultState: String, CaseIterable, Comparable, Sendable {
    case local
    case submitted
    /// `metrics.json` came back from the server.
    case scored

    private var rank: Int {
        switch self {
        case .local: 0
        case .submitted: 1
        case .scored: 2
        }
    }

    static func < (lhs: BenchmarkResultState, rhs: BenchmarkResultState) -> Bool { lhs.rank < rhs.rank }
}

/// The crate's `BenchmarkScoredResult`, saved as `metrics.json`. Both fields are written
/// even when absent, as serde writes an `Option`.
nonisolated struct BenchmarkScoredResult: Codable, Equatable, Sendable {
    var scoredAt: String?
    var metrics: [BenchmarkJobMetric]?

    enum CodingKeys: String, CodingKey {
        case scoredAt = "scored_at"
        case metrics
    }

    init(scoredAt: String?, metrics: [BenchmarkJobMetric]?) {
        self.scoredAt = scoredAt
        self.metrics = metrics
    }

    init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        scoredAt = try c.decodeIfPresent(String.self, forKey: .scoredAt)
        metrics = try c.decodeIfPresent([BenchmarkJobMetric].self, forKey: .metrics)
    }

    func encode(to encoder: any Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(scoredAt, forKey: .scoredAt)
        try c.encode(metrics, forKey: .metrics)
    }
}

/// The crate declares `BenchmarkJobMetric` twice (wire and stored, same fields); Swift has
/// one namespace, so the wire type in `PipetteMgmtClient/Types.swift` is the stored type too.
nonisolated extension BenchmarkJobMetric: Encodable {
    private enum EncodingKeys: String, CodingKey {
        case metric, value, unit
    }

    func encode(to encoder: any Encoder) throws {
        var c = encoder.container(keyedBy: EncodingKeys.self)
        try c.encode(metric, forKey: .metric)
        try c.encode(value, forKey: .value)
        try c.encode(unit, forKey: .unit)
    }
}

/// The crate's `BenchmarkResultExtras`, stored as `extras.json` and never submitted.
/// `executable` and `command` stay, empty, so the artifact reads the same as the desktop's.
nonisolated struct BenchmarkResultExtras: Codable, Equatable {
    var executable: String?
    var command: [String]
    var stdout: String
    var stderr: String
    /// One line per turn of a corpus traversal; what `observation_output_digest` summarises.
    /// `nil` elsewhere, eliding the key as the crate's `skip_serializing_if` does.
    var turns: [TurnObservation]?
}
