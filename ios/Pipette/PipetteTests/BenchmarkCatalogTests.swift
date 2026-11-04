import Foundation
import Testing

@testable import Pipette

private func item(_ id: String, _ type: String, prefill: Int?, decode: Int? = nil) -> BenchmarkItem {
    var raw: [String: Any] = [:]
    if let prefill { raw["parameter_prefill_tokens"] = prefill }
    if let decode { raw["parameter_decode_tokens"] = decode }
    // `BenchmarkCatalog.parse` drops entries that don't decode, so every real item carries
    // a definition and a fixture without one is a shape production never produces.
    let definition: BenchmarkDefinition? = switch (type, prefill, decode) {
    case let ("prefill_throughput", .some(p), _):
        .prefillThroughput(PrefillThroughputBenchmark(
            benchmarkId: BenchmarkId(validated: id), parameterPrefillTokens: NonNegativeCount(validated: Int32(p))))
    case let ("max_memory_usage", .some(p), _):
        .maxMemoryUsage(MaxMemoryUsageBenchmark(
            benchmarkId: BenchmarkId(validated: id), parameterPrefillTokens: NonNegativeCount(validated: Int32(p))))
    case let ("decode_throughput", .some(p), .some(d)):
        .decodeThroughput(DecodeThroughputBenchmark(
            benchmarkId: BenchmarkId(validated: id), parameterPrefillTokens: NonNegativeCount(validated: Int32(p)),
            parameterDecodeTokens: PositiveCount(validated: Int32(d))))
    case let ("end_to_end_latency", .some(p), .some(d)):
        .endToEndLatency(EndToEndLatencyBenchmark(
            benchmarkId: BenchmarkId(validated: id), parameterPrefillTokens: NonNegativeCount(validated: Int32(p)),
            parameterDecodeTokens: PositiveCount(validated: Int32(d))))
    default:
        // eval fixtures exist to be excluded by type, before the window is ever read.
        nil
    }
    return BenchmarkItem(
        benchmarkId: id, benchmarkType: type, sampleCount: nil, rawJson: raw,
        definition: definition)
}

/// A vision rung's window comes from its image geometry, not a token count, so the fixture
/// states the geometry the picker's filter estimates from.
private func vlItem(_ id: String, width: PositiveCount, height: PositiveCount,
                    text: NonNegativeCount = 0, decode: PositiveCount = 64,
                    images: NonNegativeCount = 1,
                    peakMemory: Bool = false) -> BenchmarkItem {
    let definition: BenchmarkDefinition = peakMemory
        ? .vlMaxMemory(VlMaxMemoryBenchmark(
            benchmarkId: BenchmarkId(validated: id),
            parameterImageWidth: width,
            parameterImageHeight: height,
            parameterTextTokens: text,
            parameterNumImages: images))
        : .vlThroughput(VlThroughputBenchmark(
            benchmarkId: BenchmarkId(validated: id),
            parameterImageWidth: width,
            parameterImageHeight: height,
            parameterTextTokens: text,
            parameterDecodeTokens: decode,
            parameterNumImages: images))
    return BenchmarkItem(
        benchmarkId: id, benchmarkType: definition.benchmarkType.rawValue, sampleCount: nil,
        rawJson: definition.parameterFields, definition: definition)
}

@Suite
struct BenchmarkCatalogTests {
    // MARK: - Benchmark-type presentation

    @Test func benchmarkTypePresentationAndOrdering() {
        #expect(BenchmarkCatalog.displayName(for: "end_to_end_latency") == "End-to-End Latency")
        #expect(BenchmarkCatalog.shortName(for: "max_memory_usage") == "Max Memory")
        #expect(
            BenchmarkCatalog.description(for: "vl_throughput")
                == "Prompt-processing and decode speed for image + text inputs.")
        #expect(BenchmarkCatalog.displayName(for: "custom_benchmark_type") == "Custom Benchmark Type")

        let ordered = [
            "end_to_end_latency", "prefill_throughput", "decode_throughput",
            "max_memory_usage", "vl_throughput", "vl_spec_decode", "asr_throughput",
            "asr_max_memory", "eval", "unknown",
        ]
        #expect(
            ordered.sorted { BenchmarkCatalog.typeRank(for: $0) < BenchmarkCatalog.typeRank(for: $1) }
                == ordered)
    }

    // MARK: - selectable(from:) filter

    @Test func selectableAdvertisesSupportedRungsUnderTheContextCap() {
        let items = [
            item("prefill_throughput_100", "prefill_throughput", prefill: 100),
            item("prefill_throughput_4096", "prefill_throughput", prefill: 4096),   // ctx 4096 < 5k
            item("prefill_throughput_8192", "prefill_throughput", prefill: 8192),   // ctx 8192 → hidden
            item("max_memory_usage_4096", "max_memory_usage", prefill: 4096),       // ctx 4097 < 5k
            item("max_memory_usage_8192", "max_memory_usage", prefill: 8192),       // ctx 8193 → hidden
            item("decode_throughput_4096_100", "decode_throughput", prefill: 4096, decode: 100),  // 4196 < 5k
            item("decode_throughput_8192_100", "decode_throughput", prefill: 8192, decode: 100),  // hidden
            item("end_to_end_latency_4096_256", "end_to_end_latency", prefill: 4096, decode: 256), // 4352 < 5k
            item("eval_ifbench", "eval", prefill: nil),
        ]

        let selectable = BenchmarkCatalog.selectable(from: items)
        #expect(
            selectable.map(\.benchmarkId) == [
                "decode_throughput_4096_100",
                "end_to_end_latency_4096_256",
                "max_memory_usage_4096",
                "prefill_throughput_100", "prefill_throughput_4096",
            ])
        #expect(!selectable.contains { $0.benchmarkType == "eval" })
    }

    /// Vision rungs are capped on the engine's context estimate, not the text cap: the two
    /// measure different things, so a shared number would hide every vision cell or none.
    /// An uncalibrated throughput estimate floors at 8192, which is the cap itself.
    @Test func selectableAdvertisesVisionRungsUnderTheVisionCap() {
        let items = [
            // 224² → 256/frame + 8 text + 8 decode + 64 + 8 = 344.
            vlItem("vl_throughput_224x224_t8_d8_f1", width: 224, height: 224, text: 8, decode: 8),
            // 512² → 1296 + 64 + 64 + 8 = 1432.
            vlItem("vl_throughput_512x512_t0_d64_f1", width: 512, height: 512),
            // 512² × 8 frames → 1296·8 + 64 + 64 + 8·8 = 10 560 → over the cap, hidden.
            vlItem("vl_throughput_512x512_t0_d64_f8", width: 512, height: 512, images: 8),
            // One 2048² frame → 146² = 21 316 → hidden.
            vlItem("vl_throughput_2048x2048_t0_d64_f1", width: 2048, height: 2048),
        ]
        #expect(
            BenchmarkCatalog.selectable(from: items).map(\.benchmarkId)
                == ["vl_throughput_224x224_t8_d8_f1", "vl_throughput_512x512_t0_d64_f1"])
    }

    /// Peak-memory vision rungs take the same cap on their estimate: one decode token and
    /// no 8192 floor.
    @Test func selectableAdvertisesVisionMemoryRungsUnderTheVisionCap() {
        let items = [
            // 512² → 1296 + 0 text + 1 decode + 64 + 8 = 1369.
            vlItem("vl_max_memory_512x512_t0_f1", width: 512, height: 512, peakMemory: true),
            // 256² × 5 frames → 18²·5 + 1024 text + 1 + 64 + 8·5 = 2749.
            vlItem("vl_max_memory_256x256_t1024_f5", width: 256, height: 256, text: 1024,
                   images: 5, peakMemory: true),
            // One 1280² frame → 91² = 8281, + 1 + 64 + 8 = 8354 → over the cap, hidden.
            vlItem("vl_max_memory_1280x1280_t0_f1", width: 1280, height: 1280, peakMemory: true),
        ]
        #expect(
            BenchmarkCatalog.selectable(from: items).map(\.benchmarkId)
                == ["vl_max_memory_256x256_t1024_f5", "vl_max_memory_512x512_t0_f1"])
    }

    @Test func selectableGatesDecodeAndE2EOnTotalContext() {
        let items = [
            item("prefill_throughput_4900", "prefill_throughput", prefill: 4900),               // ctx 4900 < 5k
            item("decode_throughput_4900_100", "decode_throughput", prefill: 4900, decode: 100),  // 5000 → hidden (strict <)
            item("decode_throughput_4800_100", "decode_throughput", prefill: 4800, decode: 100),  // 4900 < 5k
            item("end_to_end_latency_4800_256", "end_to_end_latency", prefill: 4800, decode: 256), // 5056 → hidden
        ]
        #expect(
            BenchmarkCatalog.selectable(from: items).map(\.benchmarkId)
                == ["decode_throughput_4800_100", "prefill_throughput_4900"])
    }

    /// The ASR kinds run from plans only, however small the clip.
    @Test func selectableHidesTheAsrKinds() throws {
        let throughput = BenchmarkDefinition.asrThroughput(AsrThroughputBenchmark(
            benchmarkId: BenchmarkId(validated: "asr_throughput_1000ms_text0_decode8"),
            parameterAudioDurationMs: try AudioDurationMillis(1000),
            parameterAudioSampleRateHz: try AudioSampleRate(16000),
            parameterAudioChannels: try AudioChannels(1),
            parameterDecodeTokens: try AsrDecodeTokens(8)))
        let memory = BenchmarkDefinition.asrMaxMemory(AsrMaxMemoryBenchmark(
            benchmarkId: BenchmarkId(validated: "asr_max_memory_1000ms_text0_decode8"),
            parameterAudioDurationMs: try AudioDurationMillis(1000),
            parameterAudioSampleRateHz: try AudioSampleRate(16000),
            parameterAudioChannels: try AudioChannels(1),
            parameterDecodeTokens: try AsrDecodeTokens(8)))
        let items = [throughput, memory].map {
            BenchmarkItem(benchmarkId: $0.benchmarkId.value, benchmarkType: $0.benchmarkType.rawValue,
                          sampleCount: nil, rawJson: $0.parameterFields, definition: $0)
        }
        #expect(BenchmarkCatalog.selectable(from: items).isEmpty)
    }

    @Test func selectableOnEmptyIsEmpty() {
        #expect(BenchmarkCatalog.selectable(from: []).isEmpty)
    }

    // MARK: - BenchmarkDefinition(parsingId:)

    @Test func parsingIdRoundTripsTheFourLadderTypes() {
        #expect(
            BenchmarkDefinition(parsingId: "prefill_throughput_512")
                == .prefillThroughput(PrefillThroughputBenchmark(
                    benchmarkId: BenchmarkId(validated: "prefill_throughput_512"),
                    parameterPrefillTokens: 512)))
        #expect(
            BenchmarkDefinition(parsingId: "max_memory_usage_1024")
                == .maxMemoryUsage(MaxMemoryUsageBenchmark(
                    benchmarkId: BenchmarkId(validated: "max_memory_usage_1024"),
                    parameterPrefillTokens: 1024)))
        #expect(
            BenchmarkDefinition(parsingId: "decode_throughput_256_100")
                == .decodeThroughput(DecodeThroughputBenchmark(
                    benchmarkId: BenchmarkId(validated: "decode_throughput_256_100"),
                    parameterPrefillTokens: 256,
                    parameterDecodeTokens: 100)))
        #expect(
            BenchmarkDefinition(parsingId: "end_to_end_latency_8192_256")
                == .endToEndLatency(EndToEndLatencyBenchmark(
                    benchmarkId: BenchmarkId(validated: "end_to_end_latency_8192_256"),
                    parameterPrefillTokens: 8192,
                    parameterDecodeTokens: 256)))
    }

    @Test func parsingIdRejectsUnsupportedAndGarbage() {
        #expect(BenchmarkDefinition(parsingId: "eval_ifbench") == nil)
        #expect(BenchmarkDefinition(parsingId: "vl_throughput_256x256_32_128") == nil)
        #expect(BenchmarkDefinition(parsingId: "smoke") == nil)
        #expect(BenchmarkDefinition(parsingId: "prefill_throughput") == nil)        // no number
        #expect(BenchmarkDefinition(parsingId: "prefill_throughput_abc") == nil)    // non-numeric
        #expect(BenchmarkDefinition(parsingId: "prefill_throughput_512_100") == nil) // too many ints
        #expect(BenchmarkDefinition(parsingId: "decode_throughput_256") == nil)     // too few ints
        #expect(BenchmarkDefinition(parsingId: "") == nil)
        #expect(BenchmarkDefinition(parsingId: "totally_unknown_42") == nil)
    }

    // MARK: - Tolerance: ignore benchmarks we can't parse

    @Test func catalogIgnoresUnparseableSyncedEntries() {
        let entries: [[String: Any]] = [
            ["benchmark_id": "prefill_throughput_256", "benchmark_type": "prefill_throughput",
             "parameter_prefill_tokens": 256],
            ["benchmark_id": "speculative_decode_256", "benchmark_type": "speculative_decode",
             "parameter_prefill_tokens": 256],
            // Missing the decode-tokens param its type requires.
            ["benchmark_id": "decode_broken", "benchmark_type": "decode_throughput",
             "parameter_prefill_tokens": 256],
            ["benchmark_type": "prefill_throughput", "parameter_prefill_tokens": 256],
        ]
        #expect(BenchmarkCatalog.merged(syncedEntries: entries).map(\.benchmarkId) == ["prefill_throughput_256"])
    }

    // MARK: - item(forId:) resolver

    @Test func itemForIdPrefersCatalogThenParsesThenNil() {
        let catalog = ["prefill_throughput_256": item("prefill_throughput_256", "prefill_throughput", prefill: 256)]
        #expect(BenchmarkCatalog.item(forId: "prefill_throughput_256", in: catalog)?.benchmarkId == "prefill_throughput_256")
        let parsed = BenchmarkCatalog.item(forId: "decode_throughput_512_100", in: catalog)
        #expect(parsed?.benchmarkType == "decode_throughput")
        #expect(parsed?.rawJson["parameter_prefill_tokens"] as? Int == 512)
        #expect(BenchmarkCatalog.item(forId: "eval_ifbench", in: catalog) == nil)
    }

    // MARK: - CSV export resolves params without a catalog

    @Test func parameterSummaryResolvesFromIdWhenCatalogEmpty() {
        #expect(CompletedResultsCSVExporter.parameterSummary("prefill_throughput_512", in: [:]) == "512")
        #expect(CompletedResultsCSVExporter.parameterSummary("decode_throughput_256_100", in: [:]) == "256-100")
        #expect(CompletedResultsCSVExporter.parameterSummary("vl_throughput_256x512_32_128", in: [:]) == "256x512")
        #expect(CompletedResultsCSVExporter.parameterSummary("totally_unknown_42", in: [:]) == nil)
    }

    // MARK: - Context sizing

    /// A wrapped sum would size a tiny context the run overflows. A count fits `Int32`, so
    /// a text window's sum stays exact; the vision estimate's product saturates to `.max`.
    @Test func contextSizingSaturatesOnOverflow() {
        let most = UInt32(Int32.max)
        #expect(
            LlamaRuntimeFlags.ctxSize(for: .decodeThroughput(DecodeThroughputBenchmark(
                benchmarkId: BenchmarkId(validated: "d"),
                parameterPrefillTokens: NonNegativeCount(validated: .max),
                parameterDecodeTokens: 10))) == most + 10)
        #expect(
            LlamaRuntimeFlags.ctxSize(for: .maxMemoryUsage(MaxMemoryUsageBenchmark(
                benchmarkId: BenchmarkId(validated: "m"),
                parameterPrefillTokens: NonNegativeCount(validated: .max)))) == most + 1)
        #expect(
            LlamaRuntimeFlags.ctxSize(for: .vlThroughput(VlThroughputBenchmark(
                benchmarkId: BenchmarkId(validated: "vl"),
                parameterImageWidth: PositiveCount(validated: .max),
                parameterImageHeight: PositiveCount(validated: .max),
                parameterTextTokens: 32,
                parameterDecodeTokens: 128))) == .max)
    }

    // MARK: - Both halves

    /// Preferring `local/` would silently make the duplicate's results unsubmittable.
    @Test func remoteWinsADuplicateId() throws {
        let store = makeTemporaryBenchmarkStore()
        defer { removeStore(store) }

        try StandardBenchmarks.seedStandardLocal(store, kinds: [.prefillThroughput])
        let row = """
        [{"benchmark_id":"prefill_throughput_512","benchmark_type":"prefill_throughput",\
        "parameter_prefill_tokens":512,"server_only_key":"kept"}]
        """
        try store.putRemoteIndex(try Coding.decoder.decode([PipetteMgmtClient.BenchmarkSummary].self, from: Data(row.utf8)))

        let all = BenchmarkCatalog.all(store: store)
        let duplicated = all.filter { $0.benchmarkId == "prefill_throughput_512" }
        #expect(duplicated.count == 1)
        // The remote row's unmodelled key survives for the raw readers.
        #expect(duplicated.first?.rawJson["server_only_key"] as? String == "kept")
        #expect(all.count == StandardBenchmarks.standardLocalBenchmarks(kinds: [.prefillThroughput]).count)
    }

    // MARK: - Catalog consumption (pure parse — no store, no global)

    @Test func mergedParsesSyncedEntriesWithNoBundledFallback() {
        #expect(BenchmarkCatalog.merged(syncedEntries: []).isEmpty)

        let entries: [[String: Any]] = [
            ["benchmark_id": "prefill_throughput_512", "benchmark_type": "prefill_throughput",
             "parameter_prefill_tokens": 512],
            ["benchmark_id": "eval_ifbench", "benchmark_type": "eval", "parameter_eval_id": "ifbench",
             "parameter_dataset_name": "d", "parameter_max_tokens": 64],
        ]
        let all = BenchmarkCatalog.merged(syncedEntries: entries)
        #expect(all.map(\.benchmarkId) == ["eval_ifbench", "prefill_throughput_512"])
        #expect(!all.contains { $0.benchmarkId == "decode_throughput_256_100" })

        // Eval resolves for history and lookups, but the picker hides it.
        let selectable = BenchmarkCatalog.selectable(from: all)
        #expect(!selectable.contains { $0.benchmarkId == "eval_ifbench" })
    }

    // MARK: - BenchmarkItem

    @Test func aCatalogItemSurfacesItsTypedEvalId() {
        let eval = BenchmarkItem(
            benchmarkId: "eval_ifbench", benchmarkType: "eval",
            sampleCount: nil, rawJson: ["parameter_eval_id": "ifbench"], definition: nil)
        #expect(eval.evalId == .known(.ifbench))

        let unknown = BenchmarkItem(
            benchmarkId: "eval_future", benchmarkType: "eval",
            sampleCount: nil, rawJson: ["parameter_eval_id": "some_future_eval"], definition: nil)
        #expect(unknown.evalId == .unknown("some_future_eval"))

        let prefill = BenchmarkItem(
            benchmarkId: "prefill_throughput_512", benchmarkType: "prefill_throughput",
            sampleCount: nil, rawJson: ["parameter_prefill_tokens": 512], definition: nil)
        #expect(prefill.evalId == nil)
    }

    private let afm = RuntimeKind.afm
    private let mlx = RuntimeKind.mlx
    private let llama = RuntimeKind.llamaCpp

    // MARK: - The support matrix

    @Test func afmSupportsOnlyEvalDecodeAndE2E() {
        #expect(isBenchmarkSupported(.eval, on: afm))
        #expect(isBenchmarkSupported(.decodeThroughput, on: afm))
        #expect(isBenchmarkSupported(.endToEndLatency, on: afm))
        #expect(!isBenchmarkSupported(.prefillThroughput, on: afm))
        #expect(!isBenchmarkSupported(.maxMemoryUsage, on: afm))
        #expect(!isBenchmarkSupported(.vlThroughput, on: afm))
    }

    @Test func mlxSupportsEverythingButTheProjectorKinds() {
        for bt in BenchmarkType.allCases {
            // This app links no MLX vision or audio engine.
            #expect(isBenchmarkSupported(bt, on: mlx) == !bt.needsProjector,
                    "mlx \(bt.rawValue)")
        }
    }

    @Test func llamaSupportsEveryBenchmarkType() {
        for bt in BenchmarkType.allCases {
            #expect(isBenchmarkSupported(bt, on: llama), "llama \(bt.rawValue)")
        }
    }

    // MARK: - Drift guard: AFM's accepted set == isBenchmarkSupported(_, on: .afm)

    /// `AFMRuntime.run` guards on this predicate, so pinning the intended set here catches
    /// a drift before it reaches a device.
    @Test func afmAcceptedSetMatchesSupportPredicate() {
        let intendedAccepted: Set<BenchmarkType> = [.eval, .decodeThroughput, .endToEndLatency]
        let predicateAccepted = Set(BenchmarkType.allCases.filter { isBenchmarkSupported($0, on: .afm) })
        #expect(predicateAccepted == intendedAccepted)
    }
}
