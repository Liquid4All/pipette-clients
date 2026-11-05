import Foundation

// pipette-mgmt's `crates/pipette-plan-types/src/benchmark/mod.rs`, fields under its wire
// names.

/// Opaque benchmark identifier as a plan writes it: the crate's `BenchmarkId`.
///
/// An id, never an address: a catalog side (`local/…`, `remote/…`) is this client's way
/// of *finding* a definition, and plans distribute ids alone, so a `/` is refused rather
/// than swallowed into the id. The grammar is the server's,
/// `[A-Za-z0-9][A-Za-z0-9_.-]*`, which it mints and interpolates into its keys.
nonisolated struct BenchmarkId: ValidatedString, Comparable {
    let value: String
    init(validated value: String) { self.value = value }
    static func validate(_ value: String) -> Bool { check(value) == nil }
    static let rejection = ModelError.invalidBenchmarkId
    static func < (lhs: BenchmarkId, rhs: BenchmarkId) -> Bool { lhs.value < rhs.value }

    /// Why `id` is not a benchmark id, naming the character refused, or `nil` when it is
    /// one: the crate's `check_benchmark_id`.
    static func check(_ id: String) -> BenchmarkIdError? {
        func isAsciiAlphanumeric(_ scalar: Unicode.Scalar) -> Bool {
            ("A"..."Z").contains(scalar) || ("a"..."z").contains(scalar)
                || ("0"..."9").contains(scalar)
        }
        var scalars = id.unicodeScalars.makeIterator()
        guard let first = scalars.next() else { return .empty }
        guard isAsciiAlphanumeric(first) else { return .invalidFirst(first) }
        while let scalar = scalars.next() {
            guard isAsciiAlphanumeric(scalar) || "_.-".unicodeScalars.contains(scalar) else {
                return .invalidChar(scalar)
            }
        }
        return nil
    }
}

nonisolated enum BenchmarkIdError: Error, Equatable, LocalizedError {
    case empty
    case invalidFirst(Unicode.Scalar)
    case invalidChar(Unicode.Scalar)

    var errorDescription: String? {
        switch self {
        case .empty:
            "benchmark id must not be empty"
        case let .invalidFirst(scalar):
            "benchmark id must start with [A-Za-z0-9], not '\(scalar.escaped(asASCII: true))'"
        case let .invalidChar(scalar):
            "benchmark id contains '\(scalar.escaped(asASCII: true))'; allowed: "
                + "[A-Za-z0-9][A-Za-z0-9_.-]*"
        }
    }
}

/// The crate's `BenchmarkType`, declared in the crate's order.
///
/// `fromId` is total (`nil` for an id outside the set) so a client can list a benchmark of
/// a kind it does not implement; `init(parsing:)` is the fallible parse of a type string.
/// Labels and display order live in `BenchmarkType+Display`.
nonisolated enum BenchmarkType: String, CaseIterable, Identifiable, Codable {
    case prefillThroughput = "prefill_throughput"
    case decodeThroughput = "decode_throughput"
    case endToEndLatency = "end_to_end_latency"
    case maxMemoryUsage = "max_memory_usage"
    case eval
    case vlThroughput = "vl_throughput"
    case vlMaxMemory = "vl_max_memory"
    case vlSpecDecode = "vl_spec_decode"
    case asrThroughput = "asr_throughput"
    case asrMaxMemory = "asr_max_memory"

    var id: String { rawValue }

    /// The crate's `FromStr`: `kebab-case` parses too, as the `--type` filter spells it.
    init(parsing string: String) throws(BenchmarkTypeError) {
        guard let type = BenchmarkType(rawValue: string.replacingOccurrences(of: "-", with: "_"))
        else { throw BenchmarkTypeError(string) }
        self = type
    }

    /// The type a benchmark id belongs to, by prefix: the crate's `from_id`. A heuristic:
    /// prefer a definition's `benchmarkType` wherever a catalog is at hand.
    static func fromId(_ benchmarkId: String) -> BenchmarkType? {
        let id = withoutCatalogSide(benchmarkId)
        if id.hasPrefix("prefill_throughput") { return .prefillThroughput }
        if id.hasPrefix("decode_throughput") { return .decodeThroughput }
        if id.hasPrefix("end_to_end_latency") { return .endToEndLatency }
        if id.hasPrefix("max_memory_usage") { return .maxMemoryUsage }
        if id.hasPrefix("eval_") || id == "eval" { return .eval }
        if id.hasPrefix("vl_max_memory") { return .vlMaxMemory }
        if id.hasPrefix("vl_throughput") { return .vlThroughput }
        if id.hasPrefix("vl_spec_decode") { return .vlSpecDecode }
        if id.hasPrefix("asr_throughput") { return .asrThroughput }
        if id.hasPrefix("asr_max_memory") { return .asrMaxMemory }
        return nil
    }

    /// Shared with `BenchmarkDefinition(parsingId:)` so it reads an id as `fromId` does.
    static func withoutCatalogSide(_ benchmarkId: String) -> Substring {
        let id = Substring(benchmarkId)
        for side in ["local/", "remote/"] where id.hasPrefix(side) {
            return id.dropFirst(side.count)
        }
        return id
    }

    /// Whether this benchmark waits on device readiness before each measured rep.
    ///
    /// The timing kinds do; `eval` and the two memory kinds do not, which is why the crate
    /// has no readiness field on their cells and refuses `--readiness-max-wait-secs` and
    /// `--readiness-skip-thermal` on them.
    var gatesOnReadiness: Bool {
        switch self {
        case .prefillThroughput, .decodeThroughput, .endToEndLatency, .vlThroughput,
             .vlSpecDecode, .asrThroughput:
            return true
        case .eval, .maxMemoryUsage, .vlMaxMemory, .asrMaxMemory:
            return false
        }
    }

    /// The memory kinds measure load and run in isolation. `vl_spec_decode` owns its load
    /// instead: whether the drafter is attached is the cell's `speculative_enabled`, so a
    /// preloaded handle could be the wrong one.
    var requiresFreshLoad: Bool {
        self == .maxMemoryUsage || self == .vlMaxMemory || self == .vlSpecDecode
            || self == .asrMaxMemory
    }

    /// True for vision-language benchmarks (need an mmproj).
    var isVisionLanguage: Bool {
        self == .vlThroughput || self == .vlMaxMemory || self == .vlSpecDecode
    }

    /// True for speech-recognition benchmarks (need an audio mmproj).
    var isAudio: Bool { self == .asrThroughput || self == .asrMaxMemory }

    var needsProjector: Bool { isVisionLanguage || isAudio }

    /// Whether this kind runs over a corpus the management server serves (the crate's
    /// `BenchmarkDefinition::dataset()`). Decides what a job fetches and leases before its
    /// first cell, and deletes when it ends.
    var needsCorpus: Bool { self == .vlSpecDecode }
}

/// A benchmark-type string, in either spelling, that names no `BenchmarkType`.
nonisolated struct BenchmarkTypeError: Error, Equatable, LocalizedError {
    let value: String
    init(_ value: String) { self.value = value }
    var errorDescription: String? { "unknown benchmark type '\(value)'" }
}

/// Which half of the catalog a benchmark came from. A result from a server definition is
/// submittable and waits for the next sync; one from a device-generated definition stays
/// on the device, as `BenchmarkResultLocation::from` files it
/// (`crates/pipette-cli/src/results/types.rs`).
nonisolated enum BenchmarkSource: String, Sendable, Codable, CaseIterable {
    /// Generated on this device by ``StandardBenchmarks``; never submitted.
    case local
    /// Synced from the management server; submittable.
    case remote
}

/// Requests a cell keeps in flight at once. Zero is refused rather than run as `1`, which
/// would file a serial measurement under a batched id; past `Int32.max`, its warehouse
/// column's, it is refused too.
nonisolated struct BatchSize: Hashable, Sendable, Codable, CustomStringConvertible {
    let value: UInt32

    init(_ value: UInt32) throws(BatchSizeError) {
        guard value >= 1 else { throw .zero }
        guard value <= UInt32(Int32.max) else { throw .tooLarge(value) }
        self.value = value
    }

    /// Trusted constructor, for a value already known to be at least `1`.
    init(validated value: UInt32) { self.value = value }

    static let one = BatchSize(validated: 1)

    var asU32: UInt32 { value }

    /// The count as its Int32 warehouse column holds it; the bound makes the fallback
    /// unreachable.
    var asI32: Int32 { Int32(exactly: value) ?? .max }

    func inFlight() -> Int { Int(value) }

    var description: String { "\(value)" }

    /// Out of range is a decoding error: the entry is malformed.
    init(from decoder: any Decoder) throws {
        let container = try decoder.singleValueContainer()
        do {
            try self.init(try container.decode(UInt32.self))
        } catch let error as BatchSizeError {
            throw DecodingError.dataCorruptedError(
                in: container, debugDescription: error.errorDescription ?? "")
        }
    }

    func encode(to encoder: any Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(value)
    }
}

nonisolated enum BatchSizeError: Error, Equatable, LocalizedError {
    case zero
    case tooLarge(UInt32)

    var errorDescription: String? {
        switch self {
        case .zero: "parameter_batch_size must be at least 1, got 0"
        case let .tooLarge(value): "parameter_batch_size must be at most \(Int32.max), got \(value)"
        }
    }
}

/// An `as*` accessor was called on the wrong kind.
nonisolated struct BenchmarkDefinitionError: Error, Equatable, LocalizedError {
    let expected: String
    let gotId: String
    let gotType: String

    var errorDescription: String? { "expected \(expected) benchmark, got `\(gotId)` (\(gotType))" }
}

/// The timing kinds carry the batch as a parameter, so the id spells it (`_bs<n>`). This
/// client runs one sequence and refuses more by name.
nonisolated struct PrefillThroughputBenchmark: Equatable, Sendable, Codable {
    var benchmarkId: BenchmarkId
    var parameterPrefillTokens: NonNegativeCount
    var parameterBatchSize: BatchSize = .one

    private enum CodingKeys: String, CodingKey {
        case benchmarkId = "benchmark_id"
        case parameterPrefillTokens = "parameter_prefill_tokens"
        case parameterBatchSize = "parameter_batch_size"
    }
}

nonisolated struct DecodeThroughputBenchmark: Equatable, Sendable, Codable {
    var benchmarkId: BenchmarkId
    var parameterPrefillTokens: NonNegativeCount
    var parameterDecodeTokens: PositiveCount
    var parameterBatchSize: BatchSize = .one

    private enum CodingKeys: String, CodingKey {
        case benchmarkId = "benchmark_id"
        case parameterPrefillTokens = "parameter_prefill_tokens"
        case parameterDecodeTokens = "parameter_decode_tokens"
        case parameterBatchSize = "parameter_batch_size"
    }
}

nonisolated struct EndToEndLatencyBenchmark: Equatable, Sendable, Codable {
    var benchmarkId: BenchmarkId
    var parameterPrefillTokens: NonNegativeCount
    var parameterDecodeTokens: PositiveCount
    var parameterBatchSize: BatchSize = .one

    private enum CodingKeys: String, CodingKey {
        case benchmarkId = "benchmark_id"
        case parameterPrefillTokens = "parameter_prefill_tokens"
        case parameterDecodeTokens = "parameter_decode_tokens"
        case parameterBatchSize = "parameter_batch_size"
    }
}

nonisolated struct MaxMemoryUsageBenchmark: Equatable, Sendable, Codable {
    var benchmarkId: BenchmarkId
    var parameterPrefillTokens: NonNegativeCount

    private enum CodingKeys: String, CodingKey {
        case benchmarkId = "benchmark_id"
        case parameterPrefillTokens = "parameter_prefill_tokens"
    }
}

/// Every sampling knob is optional: an entry that omits one leaves the client's default in
/// place, and one that pins any is its own benchmark id.
nonisolated struct EvalBenchmark: Equatable, Sendable, Codable {
    var benchmarkId: BenchmarkId
    var parameterEvalId: EvalId
    var parameterDatasetName: String
    var parameterMaxTokens: PositiveCount
    /// Where a sample's prompt plus output stops; echoed into results. mgmt runs a bounded
    /// entry only on vLLM and llama-server.
    var parameterMaxContext: PositiveCount?
    var parameterMcqChoices: [String]?
    var parameterTemperature: SamplingTemperature?
    var parameterTopP: TopP?
    var parameterTopK: TopK?
    var parameterMinP: MinP?
    var parameterSeed: SamplingSeed?
    /// Typed where the crate carries `Vec<Value>` opaquely: the engines here read each sample.
    var samples: [EvalSample]?

    private enum CodingKeys: String, CodingKey {
        case benchmarkId = "benchmark_id"
        case parameterEvalId = "parameter_eval_id"
        case parameterDatasetName = "parameter_dataset_name"
        case parameterMaxTokens = "parameter_max_tokens"
        case parameterMaxContext = "parameter_max_context"
        case parameterMcqChoices = "parameter_mcq_choices"
        case parameterTemperature = "parameter_temperature"
        case parameterTopP = "parameter_top_p"
        case parameterTopK = "parameter_top_k"
        case parameterMinP = "parameter_min_p"
        case parameterSeed = "parameter_seed"
        case samples
    }

    func samplingTemperature() throws -> SamplingTemperature {
        try parameterTemperature ?? SamplingTemperature(parameterEvalId.samplingTemperature)
    }

    /// The base seed offset by the repeat index `k` of `<id>#<k>`, so repeats are independent
    /// draws; wraps within llama.cpp's 32-bit seed.
    func sampleSeed(_ sampleId: String) -> UInt32? {
        guard let seed = parameterSeed else { return nil }
        let repeatIndex = sampleId.lastIndex(of: "#")
            .flatMap { UInt32(sampleId[sampleId.index(after: $0)...]) } ?? 0
        return UInt32(truncatingIfNeeded: UInt64(seed.value) + UInt64(repeatIndex))
    }
}

/// One eval row. Anything but `{id, messages}` fails to decode here; the crate carries it.
nonisolated struct EvalSample: Codable, Equatable, Sendable {
    let id: String
    let messages: [[String: String]]
}

/// The crate's `default_num_images`: an entry that omits the count holds one image.
nonisolated let defaultNumImages = NonNegativeCount(validated: 1)

nonisolated struct VlThroughputBenchmark: Equatable, Sendable, Codable {
    var benchmarkId: BenchmarkId
    var parameterImageWidth: PositiveCount
    var parameterImageHeight: PositiveCount
    var parameterTextTokens: NonNegativeCount
    var parameterDecodeTokens: PositiveCount
    /// More than one is multi-frame or video.
    var parameterNumImages: NonNegativeCount = defaultNumImages

    private enum CodingKeys: String, CodingKey {
        case benchmarkId = "benchmark_id"
        case parameterImageWidth = "parameter_image_width"
        case parameterImageHeight = "parameter_image_height"
        case parameterTextTokens = "parameter_text_tokens"
        case parameterDecodeTokens = "parameter_decode_tokens"
        case parameterNumImages = "parameter_num_images"
    }
}

nonisolated struct VlMaxMemoryBenchmark: Equatable, Sendable, Codable {
    var benchmarkId: BenchmarkId
    var parameterImageWidth: PositiveCount
    var parameterImageHeight: PositiveCount
    var parameterTextTokens: NonNegativeCount = .zero
    /// `nil` is one token and stays off the wire, so an entry that omits it keeps its body
    /// and params.
    var parameterDecodeTokens: PositiveCount?
    var parameterNumImages: NonNegativeCount = defaultNumImages

    private enum CodingKeys: String, CodingKey {
        case benchmarkId = "benchmark_id"
        case parameterImageWidth = "parameter_image_width"
        case parameterImageHeight = "parameter_image_height"
        case parameterTextTokens = "parameter_text_tokens"
        case parameterDecodeTokens = "parameter_decode_tokens"
        case parameterNumImages = "parameter_num_images"
    }
}

nonisolated struct AudioSampleRate: SamplingKnob {
    let value: Int32
    init(validated value: Int32) { self.value = value }
    static let bounds: ClosedRange<Int32> = 8000 ... 48000
    static let field = "parameter_audio_sample_rate_hz"
}

nonisolated struct AudioDurationMillis: SamplingKnob {
    let value: Int32
    init(validated value: Int32) { self.value = value }
    static let bounds: ClosedRange<Int32> = 1 ... 600_000
    static let field = "parameter_audio_duration_ms"
}

/// At least two, as the crate requires: one token would leave its server-side decode window empty.
nonisolated struct AsrDecodeTokens: SamplingKnob {
    let value: Int32
    init(validated value: Int32) { self.value = value }
    static let bounds: ClosedRange<Int32> = 2 ... 8192
    static let field = "parameter_decode_tokens"
}

/// `UInt16`, the width of the WAV `fmt ` channel field.
nonisolated struct AudioChannels: SamplingKnob {
    let value: UInt16
    init(validated value: UInt16) { self.value = value }
    static let bounds: ClosedRange<UInt16> = 1 ... 2
    static let field = "parameter_audio_channels"
}

nonisolated struct AsrThroughputBenchmark: Equatable, Sendable, Codable {
    var benchmarkId: BenchmarkId
    var parameterAudioDurationMs: AudioDurationMillis
    var parameterAudioSampleRateHz: AudioSampleRate
    var parameterAudioChannels: AudioChannels
    var parameterTextTokens: NonNegativeCount = .zero
    var parameterDecodeTokens: AsrDecodeTokens

    private enum CodingKeys: String, CodingKey {
        case benchmarkId = "benchmark_id"
        case parameterAudioDurationMs = "parameter_audio_duration_ms"
        case parameterAudioSampleRateHz = "parameter_audio_sample_rate_hz"
        case parameterAudioChannels = "parameter_audio_channels"
        case parameterTextTokens = "parameter_text_tokens"
        case parameterDecodeTokens = "parameter_decode_tokens"
    }
}

/// The peak of `AsrThroughputBenchmark`'s workload, under the same parameters.
nonisolated struct AsrMaxMemoryBenchmark: Equatable, Sendable, Codable {
    var benchmarkId: BenchmarkId
    var parameterAudioDurationMs: AudioDurationMillis
    var parameterAudioSampleRateHz: AudioSampleRate
    var parameterAudioChannels: AudioChannels
    var parameterTextTokens: NonNegativeCount = .zero
    var parameterDecodeTokens: AsrDecodeTokens

    private enum CodingKeys: String, CodingKey {
        case benchmarkId = "benchmark_id"
        case parameterAudioDurationMs = "parameter_audio_duration_ms"
        case parameterAudioSampleRateHz = "parameter_audio_sample_rate_hz"
        case parameterAudioChannels = "parameter_audio_channels"
        case parameterTextTokens = "parameter_text_tokens"
        case parameterDecodeTokens = "parameter_decode_tokens"
    }
}

/// A VL model over a real image and text corpus, its drafter on or off as the run's
/// `speculative_enabled` says. Both arms share the id and the server joins their rows; the
/// draft length is a parameter because it changes the measurement on a fixed model.
nonisolated struct VlSpecDecodeBenchmark: Equatable, Sendable, Codable {
    var benchmarkId: BenchmarkId
    /// Tokens the drafter proposes per verification step.
    var parameterDraftLength: PositiveCount
    /// Generation cap per turn.
    var parameterMaxTokens: PositiveCount
    var parameterBatchSize: BatchSize = .one
    /// Greedy unless the entry pins one: sampling moves the acceptance rate.
    var parameterTemperature: SamplingTemperature = .greedy
    var dataset: Dataset

    private enum CodingKeys: String, CodingKey {
        case benchmarkId = "benchmark_id"
        case parameterDraftLength = "parameter_draft_length"
        case parameterMaxTokens = "parameter_max_tokens"
        case parameterBatchSize = "parameter_batch_size"
        case parameterTemperature = "parameter_temperature"
        case dataset
    }
}

// The crate's `#[serde(default)]` fields, which a synthesized decode would require.

nonisolated extension PrefillThroughputBenchmark {
    init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        benchmarkId = try c.decode(BenchmarkId.self, forKey: .benchmarkId)
        parameterPrefillTokens = try c.decode(NonNegativeCount.self, forKey: .parameterPrefillTokens)
        parameterBatchSize = try c.decodeIfPresent(BatchSize.self, forKey: .parameterBatchSize) ?? .one
    }
}

nonisolated extension DecodeThroughputBenchmark {
    init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        benchmarkId = try c.decode(BenchmarkId.self, forKey: .benchmarkId)
        parameterPrefillTokens = try c.decode(NonNegativeCount.self, forKey: .parameterPrefillTokens)
        parameterDecodeTokens = try c.decode(PositiveCount.self, forKey: .parameterDecodeTokens)
        parameterBatchSize = try c.decodeIfPresent(BatchSize.self, forKey: .parameterBatchSize) ?? .one
    }
}

nonisolated extension EndToEndLatencyBenchmark {
    init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        benchmarkId = try c.decode(BenchmarkId.self, forKey: .benchmarkId)
        parameterPrefillTokens = try c.decode(NonNegativeCount.self, forKey: .parameterPrefillTokens)
        parameterDecodeTokens = try c.decode(PositiveCount.self, forKey: .parameterDecodeTokens)
        parameterBatchSize = try c.decodeIfPresent(BatchSize.self, forKey: .parameterBatchSize) ?? .one
    }
}

nonisolated extension VlThroughputBenchmark {
    init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        benchmarkId = try c.decode(BenchmarkId.self, forKey: .benchmarkId)
        parameterImageWidth = try c.decode(PositiveCount.self, forKey: .parameterImageWidth)
        parameterImageHeight = try c.decode(PositiveCount.self, forKey: .parameterImageHeight)
        parameterTextTokens = try c.decode(NonNegativeCount.self, forKey: .parameterTextTokens)
        parameterDecodeTokens = try c.decode(PositiveCount.self, forKey: .parameterDecodeTokens)
        parameterNumImages = try c.decodeIfPresent(NonNegativeCount.self, forKey: .parameterNumImages) ?? defaultNumImages
    }
}

nonisolated extension VlMaxMemoryBenchmark {
    init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        benchmarkId = try c.decode(BenchmarkId.self, forKey: .benchmarkId)
        parameterImageWidth = try c.decode(PositiveCount.self, forKey: .parameterImageWidth)
        parameterImageHeight = try c.decode(PositiveCount.self, forKey: .parameterImageHeight)
        parameterTextTokens = try c.decodeIfPresent(NonNegativeCount.self, forKey: .parameterTextTokens) ?? .zero
        parameterDecodeTokens = try c.decodeIfPresent(PositiveCount.self, forKey: .parameterDecodeTokens)
        parameterNumImages = try c.decodeIfPresent(NonNegativeCount.self, forKey: .parameterNumImages) ?? defaultNumImages
    }
}

nonisolated extension VlSpecDecodeBenchmark {
    init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        benchmarkId = try c.decode(BenchmarkId.self, forKey: .benchmarkId)
        parameterDraftLength = try c.decode(PositiveCount.self, forKey: .parameterDraftLength)
        parameterMaxTokens = try c.decode(PositiveCount.self, forKey: .parameterMaxTokens)
        parameterBatchSize = try c.decodeIfPresent(BatchSize.self, forKey: .parameterBatchSize) ?? .one
        // Absent, not null: the crate's field is a plain `SamplingTemperature`.
        parameterTemperature = c.contains(.parameterTemperature)
            ? try c.decode(SamplingTemperature.self, forKey: .parameterTemperature) : .greedy
        dataset = try c.decode(Dataset.self, forKey: .dataset)
    }
}

nonisolated extension AsrThroughputBenchmark {
    init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        benchmarkId = try c.decode(BenchmarkId.self, forKey: .benchmarkId)
        parameterAudioDurationMs = try c.decode(AudioDurationMillis.self, forKey: .parameterAudioDurationMs)
        parameterAudioSampleRateHz = try c.decode(AudioSampleRate.self, forKey: .parameterAudioSampleRateHz)
        parameterAudioChannels = try c.decode(AudioChannels.self, forKey: .parameterAudioChannels)
        parameterTextTokens = try c.decodeIfPresent(NonNegativeCount.self, forKey: .parameterTextTokens) ?? .zero
        parameterDecodeTokens = try c.decode(AsrDecodeTokens.self, forKey: .parameterDecodeTokens)
    }
}

nonisolated extension AsrMaxMemoryBenchmark {
    init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        benchmarkId = try c.decode(BenchmarkId.self, forKey: .benchmarkId)
        parameterAudioDurationMs = try c.decode(AudioDurationMillis.self, forKey: .parameterAudioDurationMs)
        parameterAudioSampleRateHz = try c.decode(AudioSampleRate.self, forKey: .parameterAudioSampleRateHz)
        parameterAudioChannels = try c.decode(AudioChannels.self, forKey: .parameterAudioChannels)
        parameterTextTokens = try c.decodeIfPresent(NonNegativeCount.self, forKey: .parameterTextTokens) ?? .zero
        parameterDecodeTokens = try c.decode(AsrDecodeTokens.self, forKey: .parameterDecodeTokens)
    }
}

/// Tagged on `benchmark_type`; an unknown type fails to decode.
nonisolated enum BenchmarkDefinition: Equatable, Sendable {
    case prefillThroughput(PrefillThroughputBenchmark)
    case decodeThroughput(DecodeThroughputBenchmark)
    case endToEndLatency(EndToEndLatencyBenchmark)
    case maxMemoryUsage(MaxMemoryUsageBenchmark)
    case eval(EvalBenchmark)
    case vlThroughput(VlThroughputBenchmark)
    case vlMaxMemory(VlMaxMemoryBenchmark)
    case vlSpecDecode(VlSpecDecodeBenchmark)
    case asrThroughput(AsrThroughputBenchmark)
    case asrMaxMemory(AsrMaxMemoryBenchmark)

    var benchmarkId: BenchmarkId {
        switch self {
        case let .prefillThroughput(b): b.benchmarkId
        case let .decodeThroughput(b): b.benchmarkId
        case let .endToEndLatency(b): b.benchmarkId
        case let .maxMemoryUsage(b): b.benchmarkId
        case let .eval(b): b.benchmarkId
        case let .vlThroughput(b): b.benchmarkId
        case let .vlMaxMemory(b): b.benchmarkId
        case let .vlSpecDecode(b): b.benchmarkId
        case let .asrThroughput(b): b.benchmarkId
        case let .asrMaxMemory(b): b.benchmarkId
        }
    }

    /// Authoritative, unlike `BenchmarkType.fromId`, which guesses from an id.
    var benchmarkType: BenchmarkType {
        switch self {
        case .prefillThroughput: .prefillThroughput
        case .decodeThroughput: .decodeThroughput
        case .endToEndLatency: .endToEndLatency
        case .maxMemoryUsage: .maxMemoryUsage
        case .eval: .eval
        case .vlThroughput: .vlThroughput
        case .vlMaxMemory: .vlMaxMemory
        case .vlSpecDecode: .vlSpecDecode
        case .asrThroughput: .asrThroughput
        case .asrMaxMemory: .asrMaxMemory
        }
    }

    /// Greedy for every kind but eval.
    func evalTemperature() throws -> SamplingTemperature {
        guard case let .eval(b) = self else { return .greedy }
        return try b.samplingTemperature()
    }

    private func unexpectedType(_ expected: String) -> BenchmarkDefinitionError {
        BenchmarkDefinitionError(expected: expected, gotId: benchmarkId.value,
                                 gotType: benchmarkType.rawValue)
    }

    func asPrefillThroughput() throws(BenchmarkDefinitionError) -> PrefillThroughputBenchmark {
        guard case let .prefillThroughput(b) = self else { throw unexpectedType("prefill_throughput") }
        return b
    }

    func asDecodeThroughput() throws(BenchmarkDefinitionError) -> DecodeThroughputBenchmark {
        guard case let .decodeThroughput(b) = self else { throw unexpectedType("decode_throughput") }
        return b
    }

    func asEndToEndLatency() throws(BenchmarkDefinitionError) -> EndToEndLatencyBenchmark {
        guard case let .endToEndLatency(b) = self else { throw unexpectedType("end_to_end_latency") }
        return b
    }

    func asMaxMemoryUsage() throws(BenchmarkDefinitionError) -> MaxMemoryUsageBenchmark {
        guard case let .maxMemoryUsage(b) = self else { throw unexpectedType("max_memory_usage") }
        return b
    }

    func asEval() throws(BenchmarkDefinitionError) -> EvalBenchmark {
        guard case let .eval(b) = self else { throw unexpectedType("eval") }
        return b
    }

    func asVlMaxMemory() throws(BenchmarkDefinitionError) -> VlMaxMemoryBenchmark {
        guard case let .vlMaxMemory(b) = self else { throw unexpectedType("vl_max_memory") }
        return b
    }

    func asVlThroughput() throws(BenchmarkDefinitionError) -> VlThroughputBenchmark {
        guard case let .vlThroughput(b) = self else { throw unexpectedType("vl_throughput") }
        return b
    }

    func asVlSpecDecode() throws(BenchmarkDefinitionError) -> VlSpecDecodeBenchmark {
        guard case let .vlSpecDecode(b) = self else { throw unexpectedType("vl_spec_decode") }
        return b
    }

    func asAsrThroughput() throws(BenchmarkDefinitionError) -> AsrThroughputBenchmark {
        guard case let .asrThroughput(b) = self else { throw unexpectedType("asr_throughput") }
        return b
    }

    func asAsrMaxMemory() throws(BenchmarkDefinitionError) -> AsrMaxMemoryBenchmark {
        guard case let .asrMaxMemory(b) = self else { throw unexpectedType("asr_max_memory") }
        return b
    }

    /// The corpus this benchmark downloads before it runs.
    var dataset: Dataset? {
        guard case let .vlSpecDecode(b) = self else { return nil }
        return b.dataset
    }

    var batchSize: BatchSize? {
        switch self {
        case let .prefillThroughput(b): b.parameterBatchSize
        case let .decodeThroughput(b): b.parameterBatchSize
        case let .endToEndLatency(b): b.parameterBatchSize
        case let .vlSpecDecode(b): b.parameterBatchSize
        case .maxMemoryUsage, .eval, .vlThroughput, .vlMaxMemory, .asrThroughput, .asrMaxMemory:
            nil
        }
    }
}

extension BenchmarkDefinition {
    /// A ladder definition rebuilt from an id such as `decode_throughput_<P>_<D>`: the
    /// fallback when the synced catalog no longer has the id (a historical job). No crate
    /// counterpart. `nil` for the kinds whose parameters an id does not spell (`eval`, the
    /// vision kinds, a `_smoke` id). The definition names the bare id, catalog side stripped.
    nonisolated init?(parsingId id: String) {
        guard let type = BenchmarkType.fromId(id),
              let benchmarkId = try? BenchmarkId(String(BenchmarkType.withoutCatalogSide(id)))
        else { return nil }
        let bare = benchmarkId.value
        // The type's own spelling, then `_<P>` or `_<P>_<D>`. Types contain underscores, so
        // the id is matched on the whole type rather than split on `_`.
        func trailingIntegers() -> [Int]? {
            guard bare.hasPrefix(type.rawValue + "_") else { return nil }
            let parts = bare.dropFirst(type.rawValue.count + 1)
                .split(separator: "_", omittingEmptySubsequences: false)
            let numbers = parts.compactMap { Int($0) }
            return numbers.count == parts.count ? numbers : nil
        }
        guard let n = trailingIntegers() else { return nil }
        // A count out of its type's range names no definition the crate could hold.
        switch type {
        case .prefillThroughput where n.count == 1:
            guard let prefill = try? NonNegativeCount(n[0]) else { return nil }
            self = .prefillThroughput(PrefillThroughputBenchmark(
                benchmarkId: benchmarkId, parameterPrefillTokens: prefill))
        case .maxMemoryUsage where n.count == 1:
            guard let prefill = try? NonNegativeCount(n[0]) else { return nil }
            self = .maxMemoryUsage(MaxMemoryUsageBenchmark(
                benchmarkId: benchmarkId, parameterPrefillTokens: prefill))
        case .decodeThroughput where n.count == 2:
            guard let prefill = try? NonNegativeCount(n[0]),
                  let decode = try? PositiveCount(n[1]) else { return nil }
            self = .decodeThroughput(DecodeThroughputBenchmark(
                benchmarkId: benchmarkId, parameterPrefillTokens: prefill,
                parameterDecodeTokens: decode))
        case .endToEndLatency where n.count == 2:
            guard let prefill = try? NonNegativeCount(n[0]),
                  let decode = try? PositiveCount(n[1]) else { return nil }
            self = .endToEndLatency(EndToEndLatencyBenchmark(
                benchmarkId: benchmarkId, parameterPrefillTokens: prefill,
                parameterDecodeTokens: decode))
        default:
            return nil
        }
    }

    /// A minimal `parameter_*` dict reconstructed from the typed params: the shape the
    /// string-keyed readers index (the live cell label, CSV export), so a parsed-only cell
    /// presents the same fields a catalog-backed one does. No crate counterpart.
    nonisolated var parameterFields: [String: Any] {
        switch self {
        case let .maxMemoryUsage(b):
            return ["parameter_prefill_tokens": Int(b.parameterPrefillTokens.value)]
        case let .prefillThroughput(b):
            return [
                "parameter_prefill_tokens": Int(b.parameterPrefillTokens.value),
                "parameter_batch_size": Int(b.parameterBatchSize.value),
            ]
        case let .decodeThroughput(b):
            return [
                "parameter_prefill_tokens": Int(b.parameterPrefillTokens.value),
                "parameter_decode_tokens": Int(b.parameterDecodeTokens.value),
                "parameter_batch_size": Int(b.parameterBatchSize.value),
            ]
        case let .endToEndLatency(b):
            return [
                "parameter_prefill_tokens": Int(b.parameterPrefillTokens.value),
                "parameter_decode_tokens": Int(b.parameterDecodeTokens.value),
                "parameter_batch_size": Int(b.parameterBatchSize.value),
            ]
        case let .vlThroughput(b):
            return [
                "parameter_image_width": Int(b.parameterImageWidth.value),
                "parameter_image_height": Int(b.parameterImageHeight.value),
                "parameter_text_tokens": Int(b.parameterTextTokens.value),
                "parameter_decode_tokens": Int(b.parameterDecodeTokens.value),
                "parameter_num_images": Int(b.parameterNumImages.value),
            ]
        case let .vlMaxMemory(b):
            var fields: [String: Any] = [
                "parameter_image_width": Int(b.parameterImageWidth.value),
                "parameter_image_height": Int(b.parameterImageHeight.value),
                "parameter_text_tokens": Int(b.parameterTextTokens.value),
                "parameter_num_images": Int(b.parameterNumImages.value),
            ]
            fields["parameter_decode_tokens"] = b.parameterDecodeTokens.map { Int($0.value) }
            return fields
        case let .eval(b):
            var fields: [String: Any] = ["parameter_max_tokens": Int(b.parameterMaxTokens.value)]
            fields["parameter_temperature"] = b.parameterTemperature?.value
            fields["parameter_top_p"] = b.parameterTopP?.value
            fields["parameter_top_k"] = b.parameterTopK.map { Int($0.value) }
            fields["parameter_min_p"] = b.parameterMinP?.value
            fields["parameter_seed"] = b.parameterSeed.map { Int($0.value) }
            return fields
        case let .vlSpecDecode(b):
            // The corpus is not a parameter: it is a pin, and the readers that index this
            // dict render short cell labels.
            return [
                "parameter_draft_length": Int(b.parameterDraftLength.value),
                "parameter_max_tokens": Int(b.parameterMaxTokens.value),
                "parameter_batch_size": Int(b.parameterBatchSize.value),
                "parameter_temperature": b.parameterTemperature.value,
            ]
        case let .asrThroughput(b):
            return asrParameterFields(
                b.parameterAudioDurationMs, b.parameterAudioSampleRateHz, b.parameterAudioChannels,
                b.parameterTextTokens, b.parameterDecodeTokens)
        case let .asrMaxMemory(b):
            return asrParameterFields(
                b.parameterAudioDurationMs, b.parameterAudioSampleRateHz, b.parameterAudioChannels,
                b.parameterTextTokens, b.parameterDecodeTokens)
        }
    }

    private nonisolated func asrParameterFields(
        _ durationMs: AudioDurationMillis, _ sampleRate: AudioSampleRate, _ channels: AudioChannels,
        _ text: NonNegativeCount, _ decode: AsrDecodeTokens
    ) -> [String: Any] {
        [
            "parameter_audio_duration_ms": Int(durationMs.value),
            "parameter_audio_sample_rate_hz": Int(sampleRate.value),
            "parameter_audio_channels": Int(channels.value),
            "parameter_text_tokens": Int(text.value),
            "parameter_decode_tokens": Int(decode.value),
        ]
    }

    /// Lets a consumer tell an unrecognized type (skip quietly) from a known type that
    /// failed to decode (a schema mismatch worth logging).
    nonisolated static let knownTypes: Set<String> = Set(BenchmarkType.allCases.map(\.rawValue))
}

nonisolated extension BenchmarkDefinition: Codable {
    private enum Tag: String, CodingKey {
        case benchmarkType = "benchmark_type"
    }

    /// The tag, then the kind's struct from the same flat object, as serde reads an
    /// internally tagged enum.
    init(from decoder: any Decoder) throws {
        let type = try decoder.container(keyedBy: Tag.self)
            .decode(BenchmarkType.self, forKey: .benchmarkType)
        switch type {
        case .prefillThroughput: self = .prefillThroughput(try .init(from: decoder))
        case .decodeThroughput: self = .decodeThroughput(try .init(from: decoder))
        case .endToEndLatency: self = .endToEndLatency(try .init(from: decoder))
        case .maxMemoryUsage: self = .maxMemoryUsage(try .init(from: decoder))
        case .eval: self = .eval(try .init(from: decoder))
        case .vlThroughput: self = .vlThroughput(try .init(from: decoder))
        case .vlMaxMemory: self = .vlMaxMemory(try .init(from: decoder))
        case .vlSpecDecode: self = .vlSpecDecode(try .init(from: decoder))
        case .asrThroughput: self = .asrThroughput(try .init(from: decoder))
        case .asrMaxMemory: self = .asrMaxMemory(try .init(from: decoder))
        }
    }

    /// Defaulted fields included, so the local catalog round-trips.
    func encode(to encoder: any Encoder) throws {
        var tag = encoder.container(keyedBy: Tag.self)
        try tag.encode(benchmarkType, forKey: .benchmarkType)
        switch self {
        case let .prefillThroughput(b): try b.encode(to: encoder)
        case let .decodeThroughput(b): try b.encode(to: encoder)
        case let .endToEndLatency(b): try b.encode(to: encoder)
        case let .maxMemoryUsage(b): try b.encode(to: encoder)
        case let .eval(b): try b.encode(to: encoder)
        case let .vlThroughput(b): try b.encode(to: encoder)
        case let .vlMaxMemory(b): try b.encode(to: encoder)
        case let .vlSpecDecode(b): try b.encode(to: encoder)
        case let .asrThroughput(b): try b.encode(to: encoder)
        case let .asrMaxMemory(b): try b.encode(to: encoder)
        }
    }
}
