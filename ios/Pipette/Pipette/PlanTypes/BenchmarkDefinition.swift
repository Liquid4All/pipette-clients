import Foundation

/// A benchmark definition decoded from the server-shaped payload — the typed
/// Swift mirror of the Rust `BenchmarkDefinition` (`pipette_plan_types::benchmark`),
/// a sum type tagged on `benchmark_type` with each variant's typed parameters.
///
/// This replaces the per-runtime minimal JSON decoders (`MLXRuntime.BenchDef` /
/// `EvalDef` and `HeadlessRunner.Def`): the JSON is decoded once at the boundary
/// and the runtime dispatches on the case, so params are typed and there's no
/// re-parsing per kernel. Decoding rejects an unknown `benchmark_type` — the
/// client operates strictly internally (see PIP-235's loose→strict principle).
nonisolated enum BenchmarkDefinition: Equatable {
    case prefillThroughput(benchmarkId: String, prefillTokens: UInt32)
    case decodeThroughput(benchmarkId: String, prefillTokens: UInt32, decodeTokens: UInt32)
    case endToEndLatency(benchmarkId: String, prefillTokens: UInt32, decodeTokens: UInt32)
    case maxMemoryUsage(benchmarkId: String, prefillTokens: UInt32)
    case eval(
        benchmarkId: String, evalId: EvalId, datasetName: String,
        maxTokens: UInt32, mcqChoices: [String]?, samples: [EvalSample]?)
    case vlThroughput(
        benchmarkId: String, imageWidth: UInt32, imageHeight: UInt32,
        textTokens: UInt32, decodeTokens: UInt32)

    var benchmarkId: String {
        switch self {
        case .prefillThroughput(let id, _), .maxMemoryUsage(let id, _),
            .decodeThroughput(let id, _, _), .endToEndLatency(let id, _, _),
            .eval(let id, _, _, _, _, _), .vlThroughput(let id, _, _, _, _):
            return id
        }
    }

    /// Reconstruct a definition from a structured benchmark id, for the four
    /// ladder types — the execution path's fallback when the synced catalog has
    /// no entry for an id (e.g. a historical job whose benchmark left the
    /// catalog). Mirrors the Rust `BenchmarkType::from_id`: the type is the
    /// matched prefix and the workload numbers are parsed straight out of the id,
    /// so no ladder of values is hardcoded.
    ///
    /// - `prefill_throughput_<P>`    → `.prefillThroughput(id, P)`
    /// - `max_memory_usage_<P>`      → `.maxMemoryUsage(id, P)`
    /// - `decode_throughput_<P>_<D>` → `.decodeThroughput(id, P, D)`
    /// - `end_to_end_latency_<P>_<D>`→ `.endToEndLatency(id, P, D)`
    ///
    /// Returns `nil` for `eval`, `vl_throughput`, smoke, or anything that doesn't
    /// match a known type with the expected trailing integers. iOS benchmark ids
    /// here, so none is stripped.
    init?(parsingId id: String) {
        // Match on the type rawValue prefix (types contain underscores, so we
        // can't just split on `_`), then parse the trailing `_<P>` / `_<P>_<D>`.
        func suffixInts(after type: String) -> [UInt32]? {
            guard id.hasPrefix(type + "_") else { return nil }
            let tail = id.dropFirst(type.count + 1)
            let parts = tail.split(separator: "_", omittingEmptySubsequences: false)
            let nums = parts.compactMap { UInt32($0) }
            return nums.count == parts.count ? nums : nil
        }

        if let n = suffixInts(after: "prefill_throughput"), n.count == 1 {
            self = .prefillThroughput(benchmarkId: id, prefillTokens: n[0])
        } else if let n = suffixInts(after: "max_memory_usage"), n.count == 1 {
            self = .maxMemoryUsage(benchmarkId: id, prefillTokens: n[0])
        } else if let n = suffixInts(after: "decode_throughput"), n.count == 2 {
            self = .decodeThroughput(benchmarkId: id, prefillTokens: n[0], decodeTokens: n[1])
        } else if let n = suffixInts(after: "end_to_end_latency"), n.count == 2 {
            self = .endToEndLatency(benchmarkId: id, prefillTokens: n[0], decodeTokens: n[1])
        } else {
            return nil
        }
    }
}

extension BenchmarkDefinition {
    /// The typed benchmark kind for this definition — the canonical tag that lets
    /// the two enums stop drifting via a hand-written string bridge. `BenchmarkType`
    /// is the single source of truth for the `benchmark_type` vocabulary; a parsed
    /// definition maps to its case here, the inverse of the `init(from:)` tag switch.
    nonisolated var type: BenchmarkType {
        switch self {
        case .prefillThroughput: return .prefillThroughput
        case .decodeThroughput: return .decodeThroughput
        case .endToEndLatency: return .endToEndLatency
        case .maxMemoryUsage: return .maxMemoryUsage
        case .eval: return .eval
        case .vlThroughput: return .vlThroughput
        }
    }

    /// The `benchmark_type` wire string — `type.rawValue`, kept as a named
    /// accessor for the string-keyed catalog / CSV readers that still index by it.
    nonisolated var benchmarkType: String { type.rawValue }

    /// A minimal `parameter_*` dict reconstructed from the typed params — the shape the
    /// string-keyed readers index (the live cell label, CSV export), so a parsed-only cell
    /// presents the same fields a catalog-backed one does.
    nonisolated var parameterFields: [String: Any] {
        switch self {
        case .prefillThroughput(_, let p), .maxMemoryUsage(_, let p):
            return ["parameter_prefill_tokens": Int(p)]
        case .decodeThroughput(_, let p, let d), .endToEndLatency(_, let p, let d):
            return ["parameter_prefill_tokens": Int(p), "parameter_decode_tokens": Int(d)]
        case .vlThroughput(_, let w, let h, let t, let d):
            return [
                "parameter_image_width": Int(w), "parameter_image_height": Int(h),
                "parameter_text_tokens": Int(t), "parameter_decode_tokens": Int(d),
            ]
        case .eval(_, _, _, let maxTokens, _, _):
            return ["parameter_max_tokens": Int(maxTokens)]
        }
    }
}

/// One eval dataset row: a stable `id` and the chat `messages` to complete.
struct EvalSample: Codable, Equatable {
    let id: String
    let messages: [[String: String]]
}

nonisolated extension BenchmarkDefinition: Decodable {
    /// The flat wire fields (`benchmark_type` tag + `parameter_*` params), matching
    /// the Rust serde shape so the same JSON deserializes on both sides.
    private enum Key: String, CodingKey {
        case benchmarkId = "benchmark_id"
        case benchmarkType = "benchmark_type"
        case prefillTokens = "parameter_prefill_tokens"
        case decodeTokens = "parameter_decode_tokens"
        case evalId = "parameter_eval_id"
        case datasetName = "parameter_dataset_name"
        case maxTokens = "parameter_max_tokens"
        case mcqChoices = "parameter_mcq_choices"
        case samples
        case imageWidth = "parameter_image_width"
        case imageHeight = "parameter_image_height"
        case textTokens = "parameter_text_tokens"
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: Key.self)
        let id = try c.decode(String.self, forKey: .benchmarkId)
        // Decode the tag as the enum — an unknown `benchmark_type` throws via the
        // synthesized rawValue decode (DecodingError), preserving reject-unknown.
        // Switching over `BenchmarkType` is exhaustive, so it can't drift from the
        // enum's cases (no literal-string switch, no `default:`).
        let type = try c.decode(BenchmarkType.self, forKey: .benchmarkType)
        switch type {
        case .prefillThroughput:
            self = .prefillThroughput(
                benchmarkId: id, prefillTokens: try c.decode(UInt32.self, forKey: .prefillTokens))
        case .decodeThroughput:
            self = .decodeThroughput(
                benchmarkId: id,
                prefillTokens: try c.decode(UInt32.self, forKey: .prefillTokens),
                decodeTokens: try c.decode(UInt32.self, forKey: .decodeTokens))
        case .endToEndLatency:
            self = .endToEndLatency(
                benchmarkId: id,
                prefillTokens: try c.decode(UInt32.self, forKey: .prefillTokens),
                decodeTokens: try c.decode(UInt32.self, forKey: .decodeTokens))
        case .maxMemoryUsage:
            self = .maxMemoryUsage(
                benchmarkId: id, prefillTokens: try c.decode(UInt32.self, forKey: .prefillTokens))
        case .eval:
            self = .eval(
                benchmarkId: id,
                evalId: try c.decode(EvalId.self, forKey: .evalId),
                datasetName: try c.decode(String.self, forKey: .datasetName),
                maxTokens: try c.decode(UInt32.self, forKey: .maxTokens),
                mcqChoices: try c.decodeIfPresent([String].self, forKey: .mcqChoices),
                samples: try c.decodeIfPresent([EvalSample].self, forKey: .samples))
        case .vlThroughput:
            self = .vlThroughput(
                benchmarkId: id,
                imageWidth: try c.decode(UInt32.self, forKey: .imageWidth),
                imageHeight: try c.decode(UInt32.self, forKey: .imageHeight),
                textTokens: try c.decode(UInt32.self, forKey: .textTokens),
                decodeTokens: try c.decode(UInt32.self, forKey: .decodeTokens))
        }
    }

    /// The `benchmark_type` tags this client understands — derived from
    /// `BenchmarkType` so there's no parallel string list to drift (one source of
    /// truth, mirroring Rust `KnownEvalId`). Lets a consumer tell an unrecognized
    /// type (skip quietly) from a known type that failed to decode (a schema
    /// mismatch worth logging).
    static let knownTypes: Set<String> = Set(BenchmarkType.allCases.map(\.rawValue))
}

nonisolated extension BenchmarkDefinition: Encodable {
    /// The inverse of `init(from:)`, emitting the same flat tag + `parameter_*` shape.
    ///
    /// Needed because the local catalog half is *written* by this client — the crate
    /// serializes the same `BenchmarkDefinition` into `benchmarks/local/<id>.json`. Each
    /// case writes exactly the keys its decode requires, so a definition round-trips;
    /// `BenchmarkDefinitionCodingTests` pins that.
    func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: Key.self)
        try c.encode(benchmarkId, forKey: .benchmarkId)
        try c.encode(type, forKey: .benchmarkType)
        switch self {
        case .prefillThroughput(_, let prefill), .maxMemoryUsage(_, let prefill):
            try c.encode(prefill, forKey: .prefillTokens)
        case .decodeThroughput(_, let prefill, let decode),
             .endToEndLatency(_, let prefill, let decode):
            try c.encode(prefill, forKey: .prefillTokens)
            try c.encode(decode, forKey: .decodeTokens)
        case .eval(_, let evalId, let dataset, let maxTokens, let mcqChoices, let samples):
            try c.encode(evalId, forKey: .evalId)
            try c.encode(dataset, forKey: .datasetName)
            try c.encode(maxTokens, forKey: .maxTokens)
            // Absent, not null — the decode side uses `decodeIfPresent`.
            try c.encodeIfPresent(mcqChoices, forKey: .mcqChoices)
            try c.encodeIfPresent(samples, forKey: .samples)
        case .vlThroughput(_, let width, let height, let textTokens, let decode):
            try c.encode(width, forKey: .imageWidth)
            try c.encode(height, forKey: .imageHeight)
            try c.encode(textTokens, forKey: .textTokens)
            try c.encode(decode, forKey: .decodeTokens)
        }
    }
}
