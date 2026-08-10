import Foundation

/// The kind of measurement a benchmark performs — the Swift mirror of plan-types'
/// `BenchmarkType`, a typed view over the `benchmark_type` string the catalog and job
/// payloads carry. Storage stays a `String`, so the server and persisted-manifest
/// contract is unchanged, and an unknown or legacy spelling decodes as `nil` rather than
/// failing.
///
/// Labels and ordering live in `BenchmarkType+Display`: the crate renders this
/// `snake_case` and has no opinion about how a screen shows it.
nonisolated enum BenchmarkType: String, CaseIterable, Identifiable, Codable {
    case endToEndLatency = "end_to_end_latency"
    case prefillThroughput = "prefill_throughput"
    case decodeThroughput = "decode_throughput"
    case maxMemoryUsage = "max_memory_usage"
    case vlThroughput = "vl_throughput"
    case eval

    var id: String { rawValue }

    /// Parse from the stored `benchmark_type` string (nil for unknown/legacy).
    init?(type: String?) {
        guard let type, let v = BenchmarkType(rawValue: type) else { return nil }
        self = v
    }

    /// The type a *benchmark id* belongs to — the id is this type's wire spelling, either
    /// bare (`eval`) or with the workload appended (`prefill_throughput_256`).
    ///
    /// Longest match first: no type is a prefix of another today, but matching short-first
    /// would silently pick the wrong one if that ever changed.
    init(benchmarkId id: String) throws {
        let match = BenchmarkType.allCases
            .sorted { $0.rawValue.count > $1.rawValue.count }
            .first { id == $0.rawValue || id.hasPrefix("\($0.rawValue)_") }
        guard let match else { throw UnknownBenchmarkId(id: id) }
        self = match
    }

    /// Whether this benchmark waits on device readiness before each measured rep.
    ///
    /// The timing kinds do; `eval` and `max_memory_usage` do not, which is why the crate
    /// refuses `--readiness-max-wait-secs` and `--readiness-skip-thermal` on them — the
    /// cell carries no readiness knob for the value to reach.
    var gatesOnReadiness: Bool {
        switch self {
        case .prefillThroughput, .decodeThroughput, .endToEndLatency, .vlThroughput:
            return true
        case .eval, .maxMemoryUsage:
            return false
        }
    }

    /// Whether this benchmark must load the model fresh rather than reusing a
    /// loaded handle — only `max_memory_usage`, which measures load+run in
    /// isolation. Drives the runner's reuse-vs-fresh-load dispatch.
    var requiresFreshLoad: Bool { self == .maxMemoryUsage }

    /// True for vision-language benchmarks (need an mmproj).
    var isVisionLanguage: Bool { self == .vlThroughput }
}

/// A benchmark id whose leading segment names no `benchmark_type`.
nonisolated struct UnknownBenchmarkId: Error, Equatable, LocalizedError {
    let id: String
    var errorDescription: String? { "`\(id)` does not name a benchmark_type" }
}
