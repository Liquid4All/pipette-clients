import Foundation
import Testing

@testable import Pipette

/// Helper: a `BenchmarkItem` carrying just the fields the selection filter and
/// context sizing read (`benchmarkType` + `parameter_prefill_tokens`).
private func item(_ id: String, _ type: String, prefill: Int?, decode: Int? = nil) -> BenchmarkItem {
    var raw: [String: Any] = [:]
    if let prefill { raw["parameter_prefill_tokens"] = prefill }
    if let decode { raw["parameter_decode_tokens"] = decode }
    // Every catalog item carries a typed definition — `BenchmarkCatalog.parse` drops the
    // entries that don't decode — and the picker's filter reads it, so a fixture without
    // one would exercise a shape production never produces.
    let definition: BenchmarkDefinition? = switch (type, prefill, decode) {
    case let ("prefill_throughput", .some(p), _):
        .prefillThroughput(benchmarkId: id, prefillTokens: UInt32(p))
    case let ("max_memory_usage", .some(p), _):
        .maxMemoryUsage(benchmarkId: id, prefillTokens: UInt32(p))
    case let ("decode_throughput", .some(p), .some(d)):
        .decodeThroughput(benchmarkId: id, prefillTokens: UInt32(p), decodeTokens: UInt32(d))
    case let ("end_to_end_latency", .some(p), .some(d)):
        .endToEndLatency(benchmarkId: id, prefillTokens: UInt32(p), decodeTokens: UInt32(d))
    default:
        // eval / vl_throughput fixtures exist to be excluded by type, before the window
        // is ever read.
        nil
    }
    return BenchmarkItem(
        benchmarkId: id, benchmarkType: type, sampleCount: nil, rawJson: raw,
        definition: definition)
}

@Suite
struct BenchmarkCatalogAndContextTests {
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
            "max_memory_usage", "vl_throughput", "eval", "unknown",
        ]
        #expect(
            ordered.sorted { BenchmarkCatalog.typeRank(for: $0) < BenchmarkCatalog.typeRank(for: $1) }
                == ordered)
    }

    // MARK: - selectable(from:) filter

    /// The picker advertises only the four supported types, and within each only
    /// the rungs whose required context stays under the 5k cap — the heaviest
    /// (8192) rung and the unsupported types are hidden, not removed from `all`.
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
            // Unsupported types — never advertised regardless of context.
            item("eval_ifbench", "eval", prefill: nil),
            item("vl_throughput_256x256_32_128", "vl_throughput", prefill: 256, decode: 128),
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
        #expect(!selectable.contains { $0.benchmarkType == "vl_throughput" })
    }

    /// Decode/e2e gate on prefill + decode, not prefill alone: a prefill that fits
    /// on its own can still push the *total* context over the cap.
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

    @Test func selectableOnEmptyIsEmpty() {
        #expect(BenchmarkCatalog.selectable(from: []).isEmpty)
    }

    // MARK: - BenchmarkDefinition(parsingId:)

    @Test func parsingIdRoundTripsTheFourLadderTypes() {
        #expect(
            BenchmarkDefinition(parsingId: "prefill_throughput_512")
                == .prefillThroughput(benchmarkId: "prefill_throughput_512", prefillTokens: 512))
        #expect(
            BenchmarkDefinition(parsingId: "max_memory_usage_1024")
                == .maxMemoryUsage(benchmarkId: "max_memory_usage_1024", prefillTokens: 1024))
        #expect(
            BenchmarkDefinition(parsingId: "decode_throughput_256_100")
                == .decodeThroughput(
                    benchmarkId: "decode_throughput_256_100", prefillTokens: 256, decodeTokens: 100))
        #expect(
            BenchmarkDefinition(parsingId: "end_to_end_latency_8192_256")
                == .endToEndLatency(
                    benchmarkId: "end_to_end_latency_8192_256", prefillTokens: 8192, decodeTokens: 256))
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

    /// The catalog drops any synced entry it can't fully decode — an unknown
    /// `benchmark_type`, a known type missing required params, or a missing id —
    /// rather than carrying a half-formed entry. Only the parseable one survives.
    @Test func catalogIgnoresUnparseableSyncedEntries() {
        let entries: [[String: Any]] = [
            ["benchmark_id": "prefill_throughput_256", "benchmark_type": "prefill_throughput",
             "parameter_prefill_tokens": 256],
            // Unknown type the client has never heard of.
            ["benchmark_id": "speculative_decode_256", "benchmark_type": "speculative_decode",
             "parameter_prefill_tokens": 256],
            // Known type, but missing the required decode-tokens param.
            ["benchmark_id": "decode_broken", "benchmark_type": "decode_throughput",
             "parameter_prefill_tokens": 256],
            // Missing benchmark_id entirely.
            ["benchmark_type": "prefill_throughput", "parameter_prefill_tokens": 256],
        ]
        #expect(BenchmarkCatalog.merged(syncedEntries: entries).map(\.benchmarkId) == ["prefill_throughput_256"])
    }

    // MARK: - item(forId:) resolver

    @Test func itemForIdPrefersCatalogThenParsesThenNil() {
        let catalog = ["prefill_throughput_256": item("prefill_throughput_256", "prefill_throughput", prefill: 256)]
        // Listed in the synced catalog → that entry wins.
        #expect(BenchmarkCatalog.item(forId: "prefill_throughput_256", in: catalog)?.benchmarkId == "prefill_throughput_256")
        // Absent but parseable → reconstructed from the id.
        let parsed = BenchmarkCatalog.item(forId: "decode_throughput_512_100", in: catalog)
        #expect(parsed?.benchmarkType == "decode_throughput")
        #expect(parsed?.rawJson["parameter_prefill_tokens"] as? Int == 512)
        // Neither listed nor parseable (eval isn't a ladder type) → nil.
        #expect(BenchmarkCatalog.item(forId: "eval_ifbench", in: catalog) == nil)
    }

    // MARK: - CSV export resolves params without a catalog

    /// With an empty catalog, the exporter resolves a historical result's
    /// parameter summary by parsing the id — explicit coverage of the parse-path
    /// fallback the metric tests exercise only incidentally.
    @Test func parameterSummaryResolvesFromIdWhenCatalogEmpty() {
        #expect(CompletedResultsCSVExporter.parameterSummary("prefill_throughput_512", in: [:]) == "512")
        #expect(CompletedResultsCSVExporter.parameterSummary("decode_throughput_256_100", in: [:]) == "256-100")
        #expect(CompletedResultsCSVExporter.parameterSummary("vl_throughput_256x512_32_128", in: [:]) == "256x512")
        #expect(CompletedResultsCSVExporter.parameterSummary("totally_unknown_42", in: [:]) == nil)
    }

    // MARK: - Context sizing

    /// The window arithmetic saturates rather than wrapping. A wrapped sum would produce a
    /// tiny context and a run that overflows it, which is worse than refusing the cell.
    @Test func contextSizingSaturatesOnOverflow() {
        #expect(
            LlamaRuntimeFlags.contextSize(for: .decodeThroughput(
                benchmarkId: "d", prefillTokens: .max, decodeTokens: 10)) == .max)
        #expect(
            LlamaRuntimeFlags.contextSize(for: .maxMemoryUsage(
                benchmarkId: "m", prefillTokens: .max)) == .max)
        #expect(
            LlamaRuntimeFlags.contextSize(for: .vlThroughput(
                benchmarkId: "vl", imageWidth: .max, imageHeight: .max,
                textTokens: 32, decodeTokens: 128)) == .max)
    }
}
