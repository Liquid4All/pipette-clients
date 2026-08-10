import Foundation

/// How a benchmark type is shown and ordered. Separate from the type itself, which
/// mirrors the crate's `benchmark_type` discriminant and has no opinion about labels —
/// the crate renders it `snake_case` and leaves presentation to whoever displays it.
extension BenchmarkType {
    /// Full, human-facing name — section headers and detail pages.
    var displayName: String {
        switch self {
        case .endToEndLatency:   return "End-to-End Latency"
        case .prefillThroughput: return "Prefill Throughput"
        case .decodeThroughput:  return "Decode Throughput"
        case .maxMemoryUsage:    return "Max Memory Usage"
        case .vlThroughput:      return "Vision-Language Throughput"
        case .eval:              return "Eval Accuracy"
        }
    }

    /// Compact name for dense table column headers, where the full name is too wide.
    var shortName: String {
        switch self {
        case .endToEndLatency:   return "E2E Latency"
        case .prefillThroughput: return "Prefill Throughput"
        case .decodeThroughput:  return "Decode Throughput"
        case .maxMemoryUsage:    return "Max Memory"
        case .vlThroughput:      return "VL Throughput"
        case .eval:              return "Eval"
        }
    }

    /// One-line description of what the benchmark measures.
    var summary: String {
        switch self {
        case .endToEndLatency:
            return "Total time to complete a request, from prompt to final token."
        case .prefillThroughput:
            return "How fast the model ingests input tokens during prompt processing."
        case .decodeThroughput:
            return "The rate at which the model generates output tokens, in tok/s."
        case .maxMemoryUsage:
            return "Peak memory used while loading and running the benchmark."
        case .vlThroughput:
            return "Prompt-processing and decode speed for image + text inputs."
        case .eval:
            return "Task accuracy measured against expected answers."
        }
    }

    /// Canonical ordering across every screen (lower sorts first).
    var rank: Int {
        switch self {
        case .endToEndLatency:   return 0
        case .prefillThroughput: return 1
        case .decodeThroughput:  return 2
        case .maxMemoryUsage:    return 3
        case .vlThroughput:      return 4
        case .eval:              return 5
        }
    }
}
