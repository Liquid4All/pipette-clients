import Foundation

/// How a benchmark type is labelled and ordered. The type itself mirrors the crate's
/// `snake_case` `benchmark_type` discriminant and leaves presentation to the client.
extension BenchmarkType {
    /// Full name, for section headers and detail pages.
    var displayName: String {
        switch self {
        case .endToEndLatency:   return "End-to-End Latency"
        case .prefillThroughput: return "Prefill Throughput"
        case .decodeThroughput:  return "Decode Throughput"
        case .maxMemoryUsage:    return "Max Memory Usage"
        case .vlThroughput:      return "Vision-Language Throughput"
        case .vlMaxMemory:       return "Vision-Language Max Memory"
        case .vlSpecDecode:    return "Vision-Language Speculative Decoding"
        case .asrThroughput:     return "Speech Recognition Throughput"
        case .asrMaxMemory:      return "Speech Recognition Max Memory"
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
        case .vlMaxMemory:       return "VL Max Memory"
        case .vlSpecDecode:    return "VL Spec Decode"
        case .asrThroughput:     return "ASR Throughput"
        case .asrMaxMemory:      return "ASR Max Memory"
        case .eval:              return "Eval"
        }
    }

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
        case .vlMaxMemory:
            return "Peak memory used while loading a vision model and encoding its images."
        case .vlSpecDecode:
            return "A vision model over a real image + text corpus, with its speculative "
                + "drafter on or off: generation speed, and accept length when on."
        case .asrThroughput:
            return "Audio-encode, prefill and decode speed for a synthetic clip + text."
        case .asrMaxMemory:
            return "Peak memory used while loading a speech model and transcribing its clip."
        case .eval:
            return "Task accuracy measured against expected answers."
        }
    }

    /// Ordering across every screen (lower sorts first).
    var rank: Int {
        switch self {
        case .endToEndLatency:   return 0
        case .prefillThroughput: return 1
        case .decodeThroughput:  return 2
        case .maxMemoryUsage:    return 3
        case .vlThroughput:      return 4
        case .vlMaxMemory:       return 5
        case .vlSpecDecode:    return 6
        case .asrThroughput:     return 7
        case .asrMaxMemory:      return 8
        case .eval:              return 9
        }
    }
}
