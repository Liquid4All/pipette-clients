import Foundation

/// Errors surfaced by the result-payload layer. The parsing helpers return `nil`
/// for a value that is simply absent (a normal, expected state) and throw one of
/// these only when data is *present but unusable* — a corrupt result file, or a
/// field whose JSON value can't be coerced to the type a metric needs. The
/// distinction lets callers stay silent on the common empty case and log the
/// genuinely broken one.
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
    /// Returns the parsed object, or `nil` when the file doesn't exist — a cell
    /// that hasn't produced this artifact yet, which is not an error. Throws
    /// `PayloadError` when the file is present but unreadable or not a JSON
    /// object, so the caller surfaces a corrupt result instead of silently
    /// treating it as empty.
    static func readObject(at url: URL) throws -> [String: Any]? {
        let data: Data
        do {
            data = try Data(contentsOf: url)
        } catch let error as CocoaError where error.code == .fileReadNoSuchFile {
            // No artifact yet (cell not run or not synced) — absence is not an error.
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
    /// Coerces `payload[key]` to `Double`. Returns `nil` when the key is absent;
    /// throws `PayloadError.malformedField` when the value is present but not a
    /// number, so a corrupt payload field surfaces rather than masquerading as a
    /// missing metric.
    static func double(_ payload: [String: Any], _ key: String) throws -> Double? {
        guard let raw = payload[key], !(raw is NSNull) else { return nil }
        if let value = raw as? Double { return value }
        if let value = raw as? Int { return Double(value) }
        if let value = raw as? NSNumber { return value.doubleValue }
        if let value = raw as? String, let parsed = Double(value) { return parsed }
        throw PayloadError.malformedField(key: key, value: String(describing: raw))
    }

    /// Coerces a catalog param to `UInt32`, defaulting to `0`. Params come from
    /// the trusted synced catalog (not the untrusted result payload), so a
    /// missing or odd value degrades to `0` rather than throwing.
    static func uint(_ payload: [String: Any], _ key: String) -> UInt32 {
        if let value = payload[key] as? UInt32 { return value }
        if let value = payload[key] as? Int, value >= 0 { return UInt32(value) }
        if let value = payload[key] as? NSNumber { return value.uint32Value }
        if let value = payload[key] as? String, let parsed = UInt32(value) { return parsed }
        return 0
    }

    /// Renders a scalar JSON value for display/CSV. Returns nil for `NSNull` and
    /// non-scalar values (so callers can skip them). `Bool` is matched before the
    /// numeric branches because a JSON boolean also casts as `Int`.
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

    /// Locale-independent numeric rendering: integral values print without a
    /// decimal point, fractional values with up to 6 significant digits. Used for
    /// CSV cells (must stay parseable regardless of device locale) and detail rows.
    static func number(_ value: Double) -> String {
        guard value.isFinite else { return "" }
        if abs(value.rounded() - value) < 0.0001 {
            return "\(Int64(value.rounded()))"
        }
        return String(format: "%.6g", locale: Locale(identifier: "en_US_POSIX"), value)
    }
}

/// A benchmark's headline metric, computed once from a result payload plus the
/// benchmark's catalog params. Formatting-free on purpose: the CSV exporter and
/// the cell detail view render `numericValue` differently (compact vs. fixed
/// precision), so display strings stay with the consumer.
nonisolated struct BenchmarkMetric {
    let name: String
    let unit: String
    let numericValue: Double
    let higherIsBetter: Bool
}

nonisolated enum BenchmarkMetrics {
    /// The single source of truth for turning a result payload into its primary
    /// metric. Both the CSV exporter and the cell detail view call this, so a
    /// metric fix lands in both places at once.
    ///
    /// Returns `nil` when the payload legitimately carries no metric (the timing
    /// field is absent or non-positive, or the type is `.eval`). Throws
    /// `PayloadError` when a required payload field is present but not numeric, so
    /// a corrupt result reaches the caller instead of rendering as a blank value.
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
        case .maxMemoryUsage:
            guard let bytes = try PayloadScalars.double(payload, "max_ram_bytes") else { return nil }
            return BenchmarkMetric(name: "Max memory", unit: "bytes", numericValue: bytes, higherIsBetter: false)
        case .vlThroughput:
            guard let promptMs = try PayloadScalars.double(payload, "prompt_ms"),
                  let predictedMs = try PayloadScalars.double(payload, "predicted_ms"),
                  promptMs + predictedMs > 0
            else { return nil }
            let promptTokens = try PayloadScalars.double(payload, "prompt_tokens")
                ?? Double(PayloadScalars.uint(params, "parameter_text_tokens"))
            let decodeTokens = Double(PayloadScalars.uint(params, "parameter_decode_tokens"))
            let throughput = (promptTokens + decodeTokens) / ((promptMs + predictedMs) / 1000)
            return BenchmarkMetric(name: "VL throughput", unit: "tok/s", numericValue: throughput, higherIsBetter: true)
        case .eval:
            // Eval accuracy isn't a single-cell payload metric.
            return nil
        }
    }
}
