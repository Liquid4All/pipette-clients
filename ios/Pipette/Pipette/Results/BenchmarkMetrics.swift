import Foundation

/// The payload helpers return `nil` for an absent value and throw this only when
/// data is present but unusable, so callers stay silent on the empty case and log
/// the broken one.
nonisolated enum PayloadError: Error, CustomStringConvertible {
    case unreadableFile(path: String, underlying: Error)
    case corruptJSON(path: String, underlying: Error)
    case notAJSONObject(path: String)
    case malformedField(key: String, value: String)

    var description: String {
        switch self {
        case let .unreadableFile(path, underlying):
            return "unreadable file at \(path): \(underlying)"
        case let .corruptJSON(path, underlying):
            return "corrupt JSON at \(path): \(underlying)"
        case let .notAJSONObject(path):
            return "not a JSON object at \(path)"
        case let .malformedField(key, value):
            return "field '\(key)' is not a number: \(value)"
        }
    }
}

/// Reads a result artifact (`payload.json` / `metrics.json`) from disk.
nonisolated enum ResultPayload {
    /// `nil` when the file doesn't exist (the cell hasn't produced it yet); throws
    /// when it exists but is unreadable or not a JSON object.
    static func readObject(at url: URL) throws -> [String: Any]? {
        let data: Data
        do {
            data = try Data(contentsOf: url)
        } catch let error as CocoaError where error.code == .fileReadNoSuchFile {
            return nil
        } catch {
            throw PayloadError.unreadableFile(path: url.path, underlying: error)
        }
        let object: Any
        do {
            object = try JSONSerialization.jsonObject(with: data)
        } catch {
            throw PayloadError.corruptJSON(path: url.path, underlying: error)
        }
        guard let dict = object as? [String: Any] else {
            throw PayloadError.notAJSONObject(path: url.path)
        }
        return dict
    }
}

/// Type-tolerant accessors for values pulled out of a benchmark `payload.json`
/// (or a benchmark's catalog params). `JSONSerialization` hands back
/// `NSNumber`/`String`/`Int` interchangeably for the same field across runs, so
/// every reader coerces through here rather than casting inline.
nonisolated enum PayloadScalars {
    /// `nil` when the key is absent; throws when the value is present but not a
    /// number, so a corrupt field doesn't masquerade as a missing metric.
    static func double(_ payload: [String: Any], _ key: String) throws -> Double? {
        guard let raw = payload[key], !(raw is NSNull) else { return nil }
        if let value = raw as? Double { return value }
        if let value = raw as? Int { return Double(value) }
        if let value = raw as? NSNumber { return value.doubleValue }
        if let value = raw as? String, let parsed = Double(value) { return parsed }
        throw PayloadError.malformedField(key: key, value: String(describing: raw))
    }

    /// Defaults to `0` rather than throwing: params come from the trusted synced
    /// catalog, not the untrusted result payload.
    static func uint(_ payload: [String: Any], _ key: String) -> UInt32 {
        if let value = payload[key] as? UInt32 { return value }
        if let value = payload[key] as? Int, value >= 0 { return UInt32(value) }
        if let value = payload[key] as? NSNumber { return value.uint32Value }
        if let value = payload[key] as? String, let parsed = UInt32(value) { return parsed }
        return 0
    }

    /// Nil for `NSNull` and non-scalars. `Bool` is matched before the numeric
    /// branches because a JSON boolean also casts as `Int`.
    static func string(_ value: Any?) -> String? {
        guard let value, !(value is NSNull) else { return nil }
        if let value = value as? String { return value }
        if let value = value as? Bool { return value ? "true" : "false" }
        if let value = value as? Int { return "\(value)" }
        if let value = value as? Int64 { return "\(value)" }
        if let value = value as? UInt32 { return "\(value)" }
        if let value = value as? Double { return number(value) }
        if let value = value as? NSNumber { return number(value.doubleValue) }
        return nil
    }

    /// Locale-independent so CSV cells stay parseable whatever the device locale.
    static func number(_ value: Double) -> String {
        guard value.isFinite else { return "" }
        if abs(value.rounded() - value) < 0.0001 {
            return "\(Int64(value.rounded()))"
        }
        return String(format: "%.6g", locale: Locale(identifier: "en_US_POSIX"), value)
    }
}

/// A benchmark's headline metric. Formatting-free on purpose: the CSV exporter and
/// the cell detail view render `numericValue` differently.
nonisolated struct BenchmarkMetric {
    let name: String
    let unit: String
    let numericValue: Double
    let higherIsBetter: Bool
}

nonisolated enum BenchmarkMetrics {
    /// Shared by the CSV exporter and the cell detail view. `nil` when the payload
    /// carries no metric; throws when a field it needs is present but not numeric.
    static func compute(
        payload: [String: Any],
        params: [String: Any],
        type: BenchmarkType
    ) throws -> BenchmarkMetric? {
        switch type {
        case .prefillThroughput:
            guard let ms = try PayloadScalars.double(payload, "prefill_time_ms"), ms > 0 else { return nil }
            let tokens = PayloadScalars.uint(params, "parameter_prefill_tokens")
            let throughput = Double(tokens) / (ms / 1000)
            return BenchmarkMetric(name: "Prefill throughput", unit: "tok/s", numericValue: throughput, higherIsBetter: true)
        case .decodeThroughput:
            guard let ms = try PayloadScalars.double(payload, "decode_time_ms"), ms > 0 else { return nil }
            let tokens = PayloadScalars.uint(params, "parameter_decode_tokens")
            let throughput = Double(tokens) / (ms / 1000)
            return BenchmarkMetric(name: "Decode throughput", unit: "tok/s", numericValue: throughput, higherIsBetter: true)
        case .endToEndLatency:
            guard let ms = try PayloadScalars.double(payload, "total_time_ms") else { return nil }
            return BenchmarkMetric(name: "E2E latency", unit: "ms", numericValue: ms, higherIsBetter: false)
        // The media kinds report through the max-memory result, as the crate's do.
        case .maxMemoryUsage, .vlMaxMemory, .asrMaxMemory:
            guard let bytes = try PayloadScalars.double(payload, "max_ram_bytes") else { return nil }
            return BenchmarkMetric(name: "Max memory", unit: "bytes", numericValue: bytes, higherIsBetter: false)
        // TTFT (`prompt_ms`, see docs/methodology/vl-throughput.md): the image-dominated
        // prefill is what a vision cell measures. Decode throughput barely moves with
        // resolution, so every rung of the VL ladder would read the same.
        case .vlThroughput:
            guard let promptMs = try PayloadScalars.double(payload, "prompt_ms"), promptMs > 0
            else { return nil }
            return BenchmarkMetric(name: "TTFT", unit: "ms", numericValue: promptMs,
                                   higherIsBetter: false)
        // TTFT, as for VL: the clip's encode and prefill is what an ASR cell measures.
        case .asrThroughput:
            guard let promptMs = try PayloadScalars.double(payload, "asr_prompt_ms"), promptMs > 0
            else { return nil }
            return BenchmarkMetric(name: "TTFT", unit: "ms", numericValue: promptMs,
                                   higherIsBetter: false)
        // No speedup: that needs the other run's row, a server-side join. The rate uses the
        // engine's `predicted_ms`, not the client clock, so the vision prefill and
        // readiness waits don't smear it.
        case .vlSpecDecode:
            guard let generated = try PayloadScalars.double(
                    payload, "vl_spec_decode_generated_tokens")
            else { return nil }
            // Speculation on: accept length, which shows whether the drafter is wired up
            // and doesn't move with the host. Averaged per turn because the wire's pooled
            // counters include the prefill and the drafts the stop token discarded.
            if let sum = try PayloadScalars.double(
                payload, "vl_spec_decode_turn_accept_length_sum"),
                let turns = try PayloadScalars.double(payload, "vl_spec_decode_turns"),
                turns > 0 {
                return BenchmarkMetric(name: "Accept length", unit: "tok",
                                       numericValue: sum / turns,
                                       higherIsBetter: true)
            }
            guard let ms = try PayloadScalars.double(payload, "vl_spec_decode_predicted_ms"),
                  ms > 0
            else { return nil }
            return BenchmarkMetric(name: "Decode", unit: "tok/s",
                                   numericValue: generated / ms * 1000, higherIsBetter: true)
        case .eval:
            return nil
        }
    }
}
