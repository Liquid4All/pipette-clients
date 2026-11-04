import Foundation

// The submission schema: pipette-mgmt's
// `crates/pipette-plan-types/src/submission/mod.rs`.
//
// `BenchmarkResultData` is an untagged union on the wire; decoding resolves to exactly
// one variant or throws. Swift `Codable` has no `flatten`, so the payload carries the
// result fields directly, filled by an exhaustive `switch` in its init. The
// `max_memory_usage` names are the methodology terms (`maxHostBytes`); the wire keys stay
// `max_ram_bytes` / `max_vram_bytes`, as the crate renames them.

/// The crate's `BenchmarkEvalCompletionStopReason`, owned by mgmt (pipette-mgmt's
/// `docs/scoring-service.md`). `unknown` is the fallback when a runtime cannot classify
/// the stop.
///
/// mgmt marks every client value `recorded`, whether it came from an engine signal
/// (llama.cpp EOG, MLX EOS) or a heuristic (AFM: generated tokens against the cap).
enum BenchmarkEvalCompletionStopReason: String, Codable, Equatable {
    case eos
    case truncated
    case failure
    case unknown
}

/// The crate's `BenchmarkEvalCompletion`, as a sum so a failed sample cannot carry a
/// completion. On the wire a failure is `{id, completion: "", failed: true, failed_reason,
/// stop_reason: "failure", stop_detail}`, where `stop_detail` repeats `failed_reason`.
nonisolated enum BenchmarkEvalCompletion: Equatable {
    case completed(id: String, completion: String, stopReason: BenchmarkEvalCompletionStopReason, stopDetail: String?, completionTokens: UInt64?)
    case failed(id: String, failedReason: String)

    var id: String {
        switch self {
        case .completed(let id, _, _, _, _), .failed(let id, _): return id
        }
    }
}

nonisolated extension BenchmarkEvalCompletion: Codable {
    private enum CodingKeys: String, CodingKey {
        case id, completion, failed
        case failedReason = "failed_reason"
        case stopReason = "stop_reason"
        case stopDetail = "stop_detail"
        case completionTokens = "completion_tokens"
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let id = try c.decode(String.self, forKey: .id)
        // Required on every row, as the crate requires it; a failed one's is empty.
        let completion = try c.decode(String.self, forKey: .completion)
        if try c.decodeIfPresent(Bool.self, forKey: .failed) == true {
            self = .failed(id: id, failedReason: try c.decodeIfPresent(String.self, forKey: .failedReason) ?? "")
        } else {
            self = .completed(
                id: id,
                completion: completion,
                // A payload that predates the field reads as `unknown`.
                stopReason: try c.decodeIfPresent(BenchmarkEvalCompletionStopReason.self, forKey: .stopReason) ?? .unknown,
                stopDetail: try c.decodeIfPresent(String.self, forKey: .stopDetail),
                completionTokens: try c.decodeIfPresent(UInt64.self, forKey: .completionTokens))
        }
    }

    func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(id, forKey: .id)
        switch self {
        case .completed(_, let text, let stopReason, let stopDetail, let completionTokens):
            try c.encode(text, forKey: .completion)
            try c.encode(stopReason, forKey: .stopReason)
            try c.encodeIfPresent(stopDetail, forKey: .stopDetail)
            try c.encodeIfPresent(completionTokens, forKey: .completionTokens)
        case .failed(_, let reason):
            try c.encode("", forKey: .completion)   // the schema requires the field
            try c.encode(true, forKey: .failed)
            try c.encode(reason, forKey: .failedReason)
            // `failed` / `failed_reason` are dual-written for legacy consumers.
            try c.encode(BenchmarkEvalCompletionStopReason.failure, forKey: .stopReason)
            try c.encode(reason, forKey: .stopDetail)
        }
    }
}

/// The crate's `MemoryObservation`: a run's memory on any benchmark kind, qualifying a
/// result rather than scoring it. `nil` means not observed, never zero. Nothing on this
/// client produces one yet.
nonisolated struct MemoryObservation: Codable, Equatable, Sendable {
    /// Peak resident memory. Swapped-out pages are not counted; `maxSwapBytes` says the
    /// figure was suppressed by reclaim.
    var maxHostBytes: UInt64?
    /// Peak swap. Not additive with `maxHostBytes`: the two need not coincide. `0` is a
    /// real reading.
    var maxSwapBytes: UInt64?

    enum CodingKeys: String, CodingKey {
        case maxHostBytes = "observation_max_host_bytes"
        case maxSwapBytes = "observation_max_swap_bytes"
    }

    init(maxHostBytes: UInt64? = nil, maxSwapBytes: UInt64? = nil) {
        self.maxHostBytes = maxHostBytes
        self.maxSwapBytes = maxSwapBytes
    }

    /// The crate's `host_only`. Zero is how every sampler spells "no read landed", so it
    /// is withheld.
    static func hostOnly(_ hostBytes: UInt64) -> MemoryObservation {
        MemoryObservation(maxHostBytes: hostBytes > 0 ? hostBytes : nil)
    }

    /// The crate's `with_swap`. A withheld peak withholds the swap reading too.
    static func withSwap(_ hostBytes: UInt64, _ swapBytes: UInt64) -> MemoryObservation {
        hostBytes > 0 ? MemoryObservation(maxHostBytes: hostBytes, maxSwapBytes: swapBytes)
            : MemoryObservation()
    }

    /// The crate's `merge_max`, term by term: the largest peak and the largest swap need
    /// not come from the same rep.
    func mergeMax(_ other: MemoryObservation) -> MemoryObservation {
        func larger(_ a: UInt64?, _ b: UInt64?) -> UInt64? {
            switch (a, b) {
            case let (a?, b?): max(a, b)
            case let (found?, nil), let (nil, found?): found
            case (nil, nil): nil
            }
        }
        return MemoryObservation(maxHostBytes: larger(maxHostBytes, other.maxHostBytes),
                                 maxSwapBytes: larger(maxSwapBytes, other.maxSwapBytes))
    }

    /// Omits an unobserved term, as the crate does.
    func encode(to encoder: any Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encodeIfPresent(maxHostBytes, forKey: .maxHostBytes)
        try c.encodeIfPresent(maxSwapBytes, forKey: .maxSwapBytes)
    }
}

/// Milliseconds a metric divides by, so zero is refused; `Float`, its warehouse column's
/// width.
nonisolated struct PositiveMillis: Hashable, Comparable, Sendable, CustomStringConvertible {
    let value: Float

    init(_ value: Float) throws(PositiveMillisError) {
        guard value.isFinite, value > 0 else { throw PositiveMillisError(value: value) }
        self.value = value
    }

    var description: String { "\(value)" }

    static func < (lhs: PositiveMillis, rhs: PositiveMillis) -> Bool { lhs.value < rhs.value }
}

nonisolated struct PositiveMillisError: Error, Equatable, LocalizedError {
    let value: Float

    var errorDescription: String? { "time must be finite and above 0 ms, got \(value)" }
}

/// A size or peak in bytes. Zero is a real reading, such as a run that touched no swap.
nonisolated struct ByteCount: Hashable, Comparable, Sendable, CustomStringConvertible {
    let value: Int64

    init(_ value: Int64) throws(ByteCountError) {
        guard value >= 0 else { throw ByteCountError(value: value) }
        self.value = value
    }

    var description: String { "\(value)" }

    static func < (lhs: ByteCount, rhs: ByteCount) -> Bool { lhs.value < rhs.value }
}

nonisolated struct ByteCountError: Error, Equatable, LocalizedError {
    let value: Int64

    var errorDescription: String? { "byte count must not be negative, got \(value)" }
}

/// The most of `device_hardware_readings` or `device_os_throttling_readings` the server
/// keeps, in bytes, cut with `keepLastElements`. Five reps fit on every platform with room
/// to spare.
nonisolated let maxDeviceReadingsBytes = 128 * 1024

/// The most of `readiness_gate_waits` the server keeps, in bytes, cut the same way. Five
/// reps' waits are about 1 KiB.
nonisolated let maxReadinessGateWaitsBytes = 8 * 1024

/// The crate's `MAX_CLIENT_EXTRAS_BYTES`. Refused, not cut, since cut JSON is no longer JSON.
nonisolated let maxClientExtrasBytes = 16 * 1024

/// The crate's `AsrContextSizing`. Only an `exact` peak is the clip's own.
nonisolated enum AsrContextSizing: String, Sendable, Encodable {
    /// From the audio tokens the projector counted for this clip.
    case exact
    /// From the upper bound, when no count could be had.
    case upperBound = "upper_bound"
    /// The cell set `ctx_size`.
    case pinned

    /// The CLI's `asr::context_sizing`.
    init(pinned: Bool, calibrated: Bool) {
        self = pinned ? .pinned : calibrated ? .exact : .upperBound
    }
}

/// Where `asr_audio_tokens` came from: the crate's `AsrAudioTokensSource`.
nonisolated enum AsrAudioTokensSource: String, Sendable, Encodable {
    /// The projector's own count, marker-free.
    case calibrated
    /// The prompt with the clip less the same prompt without it, markers included.
    case measured
}

/// The ASR keys of `client_extras`, which mgmt requires on every ASR success.
nonisolated struct AsrClientExtras: Sendable, Encodable {
    var contextSizing: AsrContextSizing
    /// `nil` on `asr_max_memory`, which does not send it.
    var audioTokensSource: AsrAudioTokensSource?
    /// The clip is longer than the model documents taking.
    var beyondModelAudioLimit: Bool

    enum CodingKeys: String, CodingKey {
        case contextSizing = "asr_context_sizing"
        case audioTokensSource = "asr_audio_tokens_source"
        case beyondModelAudioLimit = "asr_beyond_model_audio_limit"
    }

    /// Keys sorted, as the CLI's `serde_json::Map` (a `BTreeMap`) writes them.
    func json() throws -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        return String(decoding: try encoder.encode(self), as: UTF8.self)
    }
}

/// `keepLastElements` was handed something other than a JSON array.
nonisolated struct NotAJsonArray: Error, Equatable {}

/// Cut the JSON array in `s` to at most `maxBytes` by dropping elements from its front, so
/// what is left still parses: the crate's `keep_last_elements`.
nonisolated func keepLastElements(_ s: String, maxBytes: Int) throws -> String {
    guard try JSONSerialization.jsonObject(with: Data(s.utf8), options: .fragmentsAllowed)
        is [Any] else { throw NotAJsonArray() }
    if s.utf8.count <= maxBytes { return s }
    let elements = topLevelElements(s)
    var len = 2
    var kept = 0
    for element in elements.reversed() {
        len += element.utf8.count + (kept > 0 ? 1 : 0)
        guard len <= maxBytes else { break }
        kept += 1
    }
    return "[\(elements.suffix(kept).joined(separator: ","))]"
}

/// The elements of a JSON array already known to parse, each as written, without the
/// whitespace around it: what serde's `RawValue` hands the crate.
private nonisolated func topLevelElements(_ s: String) -> [String] {
    let bytes = Array(s.utf8)
    var elements: [String] = []
    var depth = 0
    var inString = false
    var escaped = false
    var start = 0
    for (i, byte) in bytes.enumerated() {
        if inString {
            if escaped {
                escaped = false
            } else if byte == UInt8(ascii: "\\") {
                escaped = true
            } else if byte == UInt8(ascii: "\"") {
                inString = false
            }
            continue
        }
        switch byte {
        case UInt8(ascii: "\""):
            inString = true
        case UInt8(ascii: "["), UInt8(ascii: "{"):
            depth += 1
            if depth == 1 { start = i + 1 }
        case UInt8(ascii: ","), UInt8(ascii: "]"), UInt8(ascii: "}"):
            if depth == 1 {
                let element = String(decoding: bytes[start ..< i], as: UTF8.self)
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                if !element.isEmpty { elements.append(element) }
                start = i + 1
            }
            if byte != UInt8(ascii: ",") { depth -= 1 }
        default:
            break
        }
    }
    return elements
}

/// The crate's `RuntimeOutput`. `nil` means not captured, `""` that nothing was printed.
/// Not submitted from this client, so the crate's caps and `keep_tail` have no port.
nonisolated struct RuntimeOutput: Equatable, Sendable {
    var runtimeStdout: String?
    var runtimeStderr: String?
}

/// The six sums every `vl_spec_decode` row carries, over every timed turn.
nonisolated struct CorpusTotals: Equatable, Sendable {
    /// The engine's own prefill time, vision encoder included: the reference harness's TTFT.
    var promptMs: Double = 0
    /// The engine's own generation time.
    var predictedMs: Double = 0
    /// Client clock, request to first token.
    var ttftMs: Double = 0
    /// Client clock, first token to last token.
    var decodeMs: Double = 0
    /// Client clock, request to end of generation. Measured, not reconstructed.
    var e2eMs: Double = 0
    var generatedTokens: UInt64 = 0
}

/// The crate's `BenchmarkResultData`, decoded by which fields are present. Not one-to-one
/// with `BenchmarkType`: a `vl_max_memory` run reports through `maxMemoryUsage`.
nonisolated enum BenchmarkResultData: Decodable, Equatable {
    case prefillThroughput(prefillTimeMs: Double, prefillTimeMsStddev: Double?)
    case decodeThroughput(decodeTimeMs: Double, decodeTimeMsStddev: Double?)
    case endToEndLatency(totalTimeMs: Double, totalTimeMsStddev: Double?)
    case maxMemoryUsage(maxHostBytes: UInt64, maxGpuBytes: UInt64?, maxNpuBytes: UInt64?)
    case eval(completions: [BenchmarkEvalCompletion])
    /// `imageTokens` is the image-only share of `promptTokens`, `nil` when the engine
    /// cannot count it. `totalTimeMs` is the whole request on this client's clock.
    case vlThroughput(promptTokens: UInt32, imageTokens: UInt32? = nil, promptMs: Double,
                      promptMsStddev: Double?, predictedMs: Double, predictedMsStddev: Double?,
                      totalTimeMs: Double? = nil, totalTimeMsStddev: Double? = nil)
    /// The crate's `VlSpecDecode`: one corpus traversal, speculation on or off.
    ///
    /// Counters, not rates, and a basis: none follows from the rest. `e2eMs` is measured,
    /// never `ttft + decode`; the server rejects one below that sum. Target forwards are
    /// not carried, since the server derives its own denominator.
    ///
    /// `draftedTokens` and `acceptedTokens` are present exactly when speculation was on;
    /// zeros would put `0 / 0` in the server's acceptance rate. `turnAcceptLengthSum` (per
    /// turn, the decode loop's tokens over its forwards, prefill excluded) comes with them:
    /// divided by `turns` it is the request-mean accept length the pooled counters cannot
    /// give.
    case vlSpecDecode(turns: UInt32, totals: CorpusTotals,
                        draftedTokens: UInt64?, acceptedTokens: UInt64?,
                        turnAcceptLengthSum: Double?)
    /// `audioTokens` is the audio-only share of `promptTokens`.
    case asrThroughput(promptTokens: UInt32, audioTokens: UInt32, promptMs: Double,
                       promptMsStddev: Double?, predictedMs: Double, predictedMsStddev: Double?,
                       totalTimeMs: Double?, totalTimeMsStddev: Double?)

    enum CodingKeys: String, CodingKey {
        case prefillTimeMs = "prefill_time_ms"
        case prefillTimeMsStddev = "prefill_time_ms_stddev"
        case decodeTimeMs = "decode_time_ms"
        case decodeTimeMsStddev = "decode_time_ms_stddev"
        case totalTimeMs = "total_time_ms"
        case totalTimeMsStddev = "total_time_ms_stddev"
        case maxHostBytes = "max_ram_bytes"
        case maxGpuBytes = "max_vram_bytes"
        // The methodology names the crate accepts as `alias`.
        case maxHostBytesAlias = "max_host_bytes"
        case maxGpuBytesAlias = "max_gpu_bytes"
        case maxNpuBytes = "max_npu_bytes"
        case completions
        case promptTokens = "prompt_tokens"
        case imageTokens = "image_tokens"
        case promptMs = "prompt_ms"
        case promptMsStddev = "prompt_ms_stddev"
        case predictedMs = "predicted_ms"
        case predictedMsStddev = "predicted_ms_stddev"
        case vlSpecDecodeTurns = "vl_spec_decode_turns"
        case vlSpecDecodePromptMs = "vl_spec_decode_prompt_ms"
        case vlSpecDecodePredictedMs = "vl_spec_decode_predicted_ms"
        case vlSpecDecodeTtftMs = "vl_spec_decode_ttft_ms"
        case vlSpecDecodeDecodeMs = "vl_spec_decode_decode_ms"
        case vlSpecDecodeE2eMs = "vl_spec_decode_e2e_ms"
        case vlSpecDecodeGeneratedTokens = "vl_spec_decode_generated_tokens"
        case vlSpecDecodeDraftedTokens = "vl_spec_decode_drafted_tokens"
        case vlSpecDecodeAcceptedTokens = "vl_spec_decode_accepted_tokens"
        case vlSpecDecodeTurnAcceptLengthSum = "vl_spec_decode_turn_accept_length_sum"
        case asrPromptTokens = "asr_prompt_tokens"
        case asrAudioTokens = "asr_audio_tokens"
        case asrPromptMs = "asr_prompt_ms"
        case asrPromptMsStddev = "asr_prompt_ms_stddev"
        case asrPredictedMs = "asr_predicted_ms"
        case asrPredictedMsStddev = "asr_predicted_ms_stddev"
    }

    // Resolve by each variant's primary field, in the crate's declaration order. The
    // remaining required fields use `decode`, so an incomplete variant throws rather than
    // producing a payload the server cannot deserialize.
    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        if let value = try c.decodeIfPresent(Double.self, forKey: .prefillTimeMs) {
            self = .prefillThroughput(
                prefillTimeMs: value,
                prefillTimeMsStddev: try c.decodeIfPresent(Double.self, forKey: .prefillTimeMsStddev))
        } else if let value = try c.decodeIfPresent(Double.self, forKey: .decodeTimeMs) {
            self = .decodeThroughput(
                decodeTimeMs: value,
                decodeTimeMsStddev: try c.decodeIfPresent(Double.self, forKey: .decodeTimeMsStddev))
        } else if let promptTokens = try c.decodeIfPresent(UInt32.self, forKey: .promptTokens) {
            // Ahead of `endToEndLatency`: a VL row carries `total_time_ms` too.
            self = .vlThroughput(
                promptTokens: promptTokens,
                imageTokens: try c.decodeIfPresent(UInt32.self, forKey: .imageTokens),
                promptMs: try c.decode(Double.self, forKey: .promptMs),
                promptMsStddev: try c.decodeIfPresent(Double.self, forKey: .promptMsStddev),
                predictedMs: try c.decode(Double.self, forKey: .predictedMs),
                predictedMsStddev: try c.decodeIfPresent(Double.self, forKey: .predictedMsStddev),
                totalTimeMs: try c.decodeIfPresent(Double.self, forKey: .totalTimeMs),
                totalTimeMsStddev: try c.decodeIfPresent(Double.self, forKey: .totalTimeMsStddev))
        } else if let promptTokens = try c.decodeIfPresent(UInt32.self, forKey: .asrPromptTokens) {
            // Ahead of `endToEndLatency` too, for the same reason.
            self = .asrThroughput(
                promptTokens: promptTokens,
                audioTokens: try c.decode(UInt32.self, forKey: .asrAudioTokens),
                promptMs: try c.decode(Double.self, forKey: .asrPromptMs),
                promptMsStddev: try c.decodeIfPresent(Double.self, forKey: .asrPromptMsStddev),
                predictedMs: try c.decode(Double.self, forKey: .asrPredictedMs),
                predictedMsStddev: try c.decodeIfPresent(Double.self, forKey: .asrPredictedMsStddev),
                totalTimeMs: try c.decodeIfPresent(Double.self, forKey: .totalTimeMs),
                totalTimeMsStddev: try c.decodeIfPresent(Double.self, forKey: .totalTimeMsStddev))
        } else if let value = try c.decodeIfPresent(Double.self, forKey: .totalTimeMs) {
            self = .endToEndLatency(
                totalTimeMs: value,
                totalTimeMsStddev: try c.decodeIfPresent(Double.self, forKey: .totalTimeMsStddev))
        } else if let hostBytes = try c.decodeIfPresent(UInt64.self, forKey: .maxHostBytes)
                    ?? c.decodeIfPresent(UInt64.self, forKey: .maxHostBytesAlias) {
            self = .maxMemoryUsage(
                maxHostBytes: hostBytes,
                maxGpuBytes: try c.decodeIfPresent(UInt64.self, forKey: .maxGpuBytes)
                    ?? c.decodeIfPresent(UInt64.self, forKey: .maxGpuBytesAlias),
                maxNpuBytes: try c.decodeIfPresent(UInt64.self, forKey: .maxNpuBytes))
        } else if let completions = try c.decodeIfPresent([BenchmarkEvalCompletion].self, forKey: .completions) {
            self = .eval(completions: completions)
        } else if let turns = try c.decodeIfPresent(UInt32.self, forKey: .vlSpecDecodeTurns) {
            // The drafter counters are absent on a row with speculation off.
            self = .vlSpecDecode(
                turns: turns,
                totals: CorpusTotals(
                    promptMs: try c.decode(Double.self, forKey: .vlSpecDecodePromptMs),
                    predictedMs: try c.decode(Double.self, forKey: .vlSpecDecodePredictedMs),
                    ttftMs: try c.decode(Double.self, forKey: .vlSpecDecodeTtftMs),
                    decodeMs: try c.decode(Double.self, forKey: .vlSpecDecodeDecodeMs),
                    e2eMs: try c.decode(Double.self, forKey: .vlSpecDecodeE2eMs),
                    generatedTokens: try c.decode(UInt64.self, forKey: .vlSpecDecodeGeneratedTokens)),
                draftedTokens: try c.decodeIfPresent(UInt64.self, forKey: .vlSpecDecodeDraftedTokens),
                acceptedTokens: try c.decodeIfPresent(UInt64.self, forKey: .vlSpecDecodeAcceptedTokens),
                turnAcceptLengthSum: try c.decodeIfPresent(Double.self,
                                                           forKey: .vlSpecDecodeTurnAcceptLengthSum))
        } else {
            throw DecodingError.dataCorrupted(.init(
                codingPath: decoder.codingPath,
                debugDescription: "benchmark result has no recognized metric fields"))
        }
    }
}

/// A runtime's result: `{"benchmark_id": …, <metric fields>}`.
struct BenchmarkRun: Decodable, Equatable {
    let benchmarkId: String
    let result: BenchmarkResultData

    enum CodingKeys: String, CodingKey {
        case benchmarkId = "benchmark_id"
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        benchmarkId = try container.decode(String.self, forKey: .benchmarkId)
        result = try BenchmarkResultData(from: decoder)  // the same flat object
    }
}

/// The crate's `BenchmarkSubmissionPayload`. `encode(to:)` is synthesized so no field can
/// be dropped by a hand-written encoder; nil optionals elide.
///
/// No `MemoryObservation` (nothing here produces one) or `RuntimeOutput`: the run's output
/// stays in `extras.json`, and `pipette-ios-proxy` adds the app's console as
/// `runtime_stdout` / `runtime_stderr` when it submits.
nonisolated struct BenchmarkSubmissionPayload: Encodable, Sendable {
    let benchmarkId: String
    /// The server checks these against its catalog; the tag rides as `benchmark_type`.
    let benchmarkType: BenchmarkType?
    let benchmarkParams: BenchmarkParams?
    let deviceName: String
    let deviceFormFactor: DeviceFormFactor
    let deviceOsName: String
    let deviceOsVersion: String
    let deviceOsBuild: String?
    let deviceOsSecurityPatch: String?
    let deviceCpuArch: String?
    let deviceOsFamily: String?
    let deviceChipModel: String
    let deviceRamBytes: UInt64
    let deviceGpuModel: String?
    let deviceGpuVramBytes: UInt64?
    let deviceNpuModel: String?
    let deviceNpuVramBytes: UInt64?
    let deviceBatteryLevel: Int32?
    let devicePowerState: DevicePowerState?
    let devicePowerSaveMode: Bool
    /// A JSON array in a string, one `ReadingsElement` per rep and sample point.
    let deviceHardwareReadings: String?
    let deviceOsThrottlingReadings: String?
    /// A JSON array in a string, one `GateWaitElement` per gated rep.
    let readinessGateWaits: String?
    // Model and runtime identity is the descriptors alone; a reader decodes
    // `model_descriptor` as a `Model` rather than reading a second field.
    let modelFlags: String?
    /// The crate's `Model` as a JSON string.
    let modelDescriptor: String
    let runtimeFlags: String?
    /// The only place a cell that waived the thermal criterion is told apart from a gated
    /// one.
    let benchmarkFlags: String?
    // The crate's `Runtime` as a JSON string.
    let runtimeDescriptor: String
    let runtimeCpuVariant: String?
    // This app build, not the runtime it drove, so a shift in the numbers can be pinned
    // on an app change. The same string Settings shows and Sentry tags as `app_version`.
    let clientVersion: String
    /// Milliseconds on a monotonic clock from receiving the claim to the result being
    /// ready. Set only for a claimed run, after the payload is built.
    var deviceTimeMs: UInt64?
    /// A JSON object in a string, at most `maxClientExtrasBytes`; only ASR rows send one.
    let clientExtras: String?
    let submittedAt: String
    // Flattened from `result` in init.
    let prefillTimeMs: Double?
    let prefillTimeMsStddev: Double?
    let decodeTimeMs: Double?
    let decodeTimeMsStddev: Double?
    let totalTimeMs: Double?
    let totalTimeMsStddev: Double?
    let maxHostBytes: UInt64?
    let maxGpuBytes: UInt64?
    let maxNpuBytes: UInt64?
    let completions: [BenchmarkEvalCompletion]?
    let promptTokens: UInt32?
    let imageTokens: UInt32?
    let promptMs: Double?
    let promptMsStddev: Double?
    let predictedMs: Double?
    let predictedMsStddev: Double?
    let vlSpecDecodeTurns: UInt32?
    let vlSpecDecodePromptMs: Double?
    let vlSpecDecodePredictedMs: Double?
    let vlSpecDecodeTtftMs: Double?
    let vlSpecDecodeDecodeMs: Double?
    let vlSpecDecodeE2eMs: Double?
    let vlSpecDecodeGeneratedTokens: UInt64?
    let vlSpecDecodeDraftedTokens: UInt64?
    let vlSpecDecodeAcceptedTokens: UInt64?
    let vlSpecDecodeTurnAcceptLengthSum: Double?
    let asrPromptTokens: UInt32?
    let asrAudioTokens: UInt32?
    let asrPromptMs: Double?
    let asrPromptMsStddev: Double?
    let asrPredictedMs: Double?
    let asrPredictedMsStddev: Double?
    /// sha256 of the corpus tar, submitted even when it does not match the catalog's pin:
    /// the server records the verdict rather than rejecting, so this says which corpus a
    /// row measured.
    let observationDatasetDigest: String?
    /// sha256 over the replies in order, each turn as `sha256(text)`, a tab, its token
    /// count, a newline. Equal digests mean two rows matched on every turn without either's
    /// text leaving the device.
    let observationOutputDigest: String?
    /// No `CodingKeys` case: the wire carries only the flattened fields.
    let result: BenchmarkResultData

    enum CodingKeys: String, CodingKey {
        case benchmarkId = "benchmark_id"
        case benchmarkType = "benchmark_type"
        case benchmarkParams = "benchmark_params"
        case deviceName = "device_name"
        case deviceFormFactor = "device_form_factor"
        case deviceOsName = "device_os_name"
        case deviceOsVersion = "device_os_version"
        case deviceOsBuild = "device_os_build"
        case deviceOsSecurityPatch = "device_os_security_patch"
        case deviceCpuArch = "device_cpu_arch"
        case deviceOsFamily = "device_os_family"
        case deviceChipModel = "device_chip_model"
        case deviceRamBytes = "device_ram_bytes"
        case deviceGpuModel = "device_gpu_model"
        case deviceGpuVramBytes = "device_gpu_vram_bytes"
        case deviceNpuModel = "device_npu_model"
        case deviceNpuVramBytes = "device_npu_vram_bytes"
        case deviceBatteryLevel = "device_battery_level"
        case devicePowerState = "device_power_state"
        case devicePowerSaveMode = "device_power_save_mode"
        case deviceHardwareReadings = "device_hardware_readings"
        case deviceOsThrottlingReadings = "device_os_throttling_readings"
        case readinessGateWaits = "readiness_gate_waits"
        case modelFlags = "model_flags"
        case modelDescriptor = "model_descriptor"
        case runtimeFlags = "runtime_flags"
        case benchmarkFlags = "benchmark_flags"
        case runtimeDescriptor = "runtime_descriptor"
        case runtimeCpuVariant = "runtime_cpu_variant"
        case clientVersion = "client_version"
        case deviceTimeMs = "device_time_ms"
        case clientExtras = "client_extras"
        case submittedAt = "submitted_at"
        case prefillTimeMs = "prefill_time_ms"
        case prefillTimeMsStddev = "prefill_time_ms_stddev"
        case decodeTimeMs = "decode_time_ms"
        case decodeTimeMsStddev = "decode_time_ms_stddev"
        case totalTimeMs = "total_time_ms"
        case totalTimeMsStddev = "total_time_ms_stddev"
        case maxHostBytes = "max_ram_bytes"
        case maxGpuBytes = "max_vram_bytes"
        case maxNpuBytes = "max_npu_bytes"
        case completions
        case promptTokens = "prompt_tokens"
        case imageTokens = "image_tokens"
        case promptMs = "prompt_ms"
        case promptMsStddev = "prompt_ms_stddev"
        case predictedMs = "predicted_ms"
        case predictedMsStddev = "predicted_ms_stddev"
        case vlSpecDecodeTurns = "vl_spec_decode_turns"
        case vlSpecDecodePromptMs = "vl_spec_decode_prompt_ms"
        case vlSpecDecodePredictedMs = "vl_spec_decode_predicted_ms"
        case vlSpecDecodeTtftMs = "vl_spec_decode_ttft_ms"
        case vlSpecDecodeDecodeMs = "vl_spec_decode_decode_ms"
        case vlSpecDecodeE2eMs = "vl_spec_decode_e2e_ms"
        case vlSpecDecodeGeneratedTokens = "vl_spec_decode_generated_tokens"
        case vlSpecDecodeDraftedTokens = "vl_spec_decode_drafted_tokens"
        case vlSpecDecodeAcceptedTokens = "vl_spec_decode_accepted_tokens"
        case vlSpecDecodeTurnAcceptLengthSum = "vl_spec_decode_turn_accept_length_sum"
        case asrPromptTokens = "asr_prompt_tokens"
        case asrAudioTokens = "asr_audio_tokens"
        case asrPromptMs = "asr_prompt_ms"
        case asrPromptMsStddev = "asr_prompt_ms_stddev"
        case asrPredictedMs = "asr_predicted_ms"
        case asrPredictedMsStddev = "asr_predicted_ms_stddev"
        case observationDatasetDigest = "observation_dataset_digest"
        case observationOutputDigest = "observation_output_digest"
    }

    /// `device` and `power` arrive whole: fifteen loose arguments, eight of them `String`
    /// or `String?`, is where a transposition hides.
    init(
        benchmarkId: String,
        benchmarkParams: BenchmarkParams? = nil,
        device: DeviceInfo,
        power: PowerReading,
        deviceHardwareReadings: String? = nil,
        deviceOsThrottlingReadings: String? = nil,
        readinessGateWaits: String? = nil,
        modelFlags: String?,
        modelDescriptor: String,
        runtimeFlags: String?,
        benchmarkFlags: String?,
        runtimeDescriptor: String,
        runtimeCpuVariant: String?,
        clientVersion: String = Bundle.main.appVersionDisplayString,
        clientExtras: String? = nil,
        submittedAt: String,
        observationDatasetDigest: String? = nil,
        observationOutputDigest: String? = nil,
        result: BenchmarkResultData
    ) {
        self.benchmarkId = benchmarkId
        self.benchmarkType = benchmarkParams?.benchmarkType
        self.benchmarkParams = benchmarkParams
        self.deviceName = device.deviceName
        self.deviceFormFactor = device.deviceFormFactor
        self.deviceOsName = device.deviceOsName
        self.deviceOsVersion = device.deviceOsVersion
        self.deviceOsBuild = device.deviceOsBuild
        self.deviceOsSecurityPatch = device.deviceOsSecurityPatch
        self.deviceCpuArch = device.deviceCpuArch
        self.deviceOsFamily = device.deviceOsFamily
        self.deviceChipModel = device.deviceChipModel
        self.deviceRamBytes = device.deviceRamBytes
        self.deviceGpuModel = device.deviceGpuModel
        self.deviceGpuVramBytes = device.deviceGpuVramBytes
        self.deviceNpuModel = device.deviceNpuModel
        self.deviceNpuVramBytes = device.deviceNpuVramBytes
        self.deviceBatteryLevel = power.batteryLevel
        self.devicePowerState = power.powerState
        self.devicePowerSaveMode = power.powerSaveMode
        self.deviceHardwareReadings = deviceHardwareReadings
        self.deviceOsThrottlingReadings = deviceOsThrottlingReadings
        self.readinessGateWaits = readinessGateWaits
        self.modelFlags = modelFlags
        self.modelDescriptor = modelDescriptor
        self.runtimeFlags = runtimeFlags
        self.benchmarkFlags = benchmarkFlags
        self.runtimeDescriptor = runtimeDescriptor
        self.runtimeCpuVariant = runtimeCpuVariant
        self.clientVersion = clientVersion
        self.clientExtras = clientExtras
        self.submittedAt = submittedAt
        self.observationDatasetDigest = observationDatasetDigest
        self.observationOutputDigest = observationOutputDigest
        self.result = result

        var prefillTimeMs: Double?
        var prefillTimeMsStddev: Double?
        var decodeTimeMs: Double?
        var decodeTimeMsStddev: Double?
        var totalTimeMs: Double?
        var totalTimeMsStddev: Double?
        var maxHostBytes: UInt64?
        var maxGpuBytes: UInt64?
        var maxNpuBytes: UInt64?
        var completions: [BenchmarkEvalCompletion]?
        var promptTokens: UInt32?
        var imageTokens: UInt32?
        var promptMs: Double?
        var promptMsStddev: Double?
        var predictedMs: Double?
        var predictedMsStddev: Double?
        var vlSpecDecodeTurns: UInt32?
        var vlSpecDecodePromptMs: Double?
        var vlSpecDecodePredictedMs: Double?
        var vlSpecDecodeTtftMs: Double?
        var vlSpecDecodeDecodeMs: Double?
        var vlSpecDecodeE2eMs: Double?
        var vlSpecDecodeGeneratedTokens: UInt64?
        var vlSpecDecodeDraftedTokens: UInt64?
        var vlSpecDecodeAcceptedTokens: UInt64?
        var vlSpecDecodeTurnAcceptLengthSum: Double?
        var asrPromptTokens: UInt32?
        var asrAudioTokens: UInt32?
        var asrPromptMs: Double?
        var asrPromptMsStddev: Double?
        var asrPredictedMs: Double?
        var asrPredictedMsStddev: Double?
        switch result {
        case let .prefillThroughput(timeMs, stddev):
            prefillTimeMs = timeMs; prefillTimeMsStddev = stddev
        case let .decodeThroughput(timeMs, stddev):
            decodeTimeMs = timeMs; decodeTimeMsStddev = stddev
        case let .endToEndLatency(timeMs, stddev):
            totalTimeMs = timeMs; totalTimeMsStddev = stddev
        case let .maxMemoryUsage(hostBytes, gpuBytes, npuBytes):
            maxHostBytes = hostBytes; maxGpuBytes = gpuBytes; maxNpuBytes = npuBytes
        case let .eval(samples):
            completions = samples
        case let .vlThroughput(tokens, image, promptMsValue, promptStddev, predictedMsValue,
                               predictedStddev, totalTimeMsValue, totalStddev):
            promptTokens = tokens
            imageTokens = image
            promptMs = promptMsValue
            promptMsStddev = promptStddev
            predictedMs = predictedMsValue
            predictedMsStddev = predictedStddev
            totalTimeMs = totalTimeMsValue
            totalTimeMsStddev = totalStddev
        case let .vlSpecDecode(turns, totals, drafted, accepted, acceptLengthSum):
            vlSpecDecodeTurns = turns
            vlSpecDecodePromptMs = totals.promptMs
            vlSpecDecodePredictedMs = totals.predictedMs
            vlSpecDecodeTtftMs = totals.ttftMs
            vlSpecDecodeDecodeMs = totals.decodeMs
            vlSpecDecodeE2eMs = totals.e2eMs
            vlSpecDecodeGeneratedTokens = totals.generatedTokens
            vlSpecDecodeDraftedTokens = drafted
            vlSpecDecodeAcceptedTokens = accepted
            vlSpecDecodeTurnAcceptLengthSum = acceptLengthSum
        case let .asrThroughput(tokens, audio, promptMsValue, promptStddev, predictedMsValue,
                                predictedStddev, totalTimeMsValue, totalStddev):
            asrPromptTokens = tokens
            asrAudioTokens = audio
            asrPromptMs = promptMsValue
            asrPromptMsStddev = promptStddev
            asrPredictedMs = predictedMsValue
            asrPredictedMsStddev = predictedStddev
            totalTimeMs = totalTimeMsValue
            totalTimeMsStddev = totalStddev
        }
        self.prefillTimeMs = prefillTimeMs
        self.prefillTimeMsStddev = prefillTimeMsStddev
        self.decodeTimeMs = decodeTimeMs
        self.decodeTimeMsStddev = decodeTimeMsStddev
        self.totalTimeMs = totalTimeMs
        self.totalTimeMsStddev = totalTimeMsStddev
        self.maxHostBytes = maxHostBytes
        self.maxGpuBytes = maxGpuBytes
        self.maxNpuBytes = maxNpuBytes
        self.completions = completions
        self.promptTokens = promptTokens
        self.imageTokens = imageTokens
        self.promptMs = promptMs
        self.promptMsStddev = promptMsStddev
        self.predictedMs = predictedMs
        self.predictedMsStddev = predictedMsStddev
        self.vlSpecDecodeTurns = vlSpecDecodeTurns
        self.vlSpecDecodePromptMs = vlSpecDecodePromptMs
        self.vlSpecDecodePredictedMs = vlSpecDecodePredictedMs
        self.vlSpecDecodeTtftMs = vlSpecDecodeTtftMs
        self.vlSpecDecodeDecodeMs = vlSpecDecodeDecodeMs
        self.vlSpecDecodeE2eMs = vlSpecDecodeE2eMs
        self.vlSpecDecodeGeneratedTokens = vlSpecDecodeGeneratedTokens
        self.vlSpecDecodeDraftedTokens = vlSpecDecodeDraftedTokens
        self.vlSpecDecodeAcceptedTokens = vlSpecDecodeAcceptedTokens
        self.vlSpecDecodeTurnAcceptLengthSum = vlSpecDecodeTurnAcceptLengthSum
        self.asrPromptTokens = asrPromptTokens
        self.asrAudioTokens = asrAudioTokens
        self.asrPromptMs = asrPromptMs
        self.asrPromptMsStddev = asrPromptMsStddev
        self.asrPredictedMs = asrPredictedMs
        self.asrPredictedMsStddev = asrPredictedMsStddev
    }
}
