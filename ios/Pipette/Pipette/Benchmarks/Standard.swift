import Foundation

/// The built-in local ladder and smoke definitions, entry for entry the crate's
/// `crates/pipette-cli/src/benchmarks/standard.rs`.
///
/// Generated on the device, so their results are never submitted (see ``BenchmarkSource``).
/// They let a device measure before it has a registration, and let an operator run the same
/// ladder the desktop clients run.
nonisolated enum StandardBenchmarks {
    /// A methodology value: kept verbatim from the crate.
    static let ladderTokens: [Int32] = [100, 256, 512, 1024, 2048, 4096, 8192]

    static func standardLocalBenchmarks(kinds: [BenchmarkType]) -> [BenchmarkDefinition] {
        ladderLocalBenchmarks(kinds: kinds) + smokeLocalBenchmarks(kinds: kinds)
    }

    private static func ladderLocalBenchmarks(kinds: [BenchmarkType]) -> [BenchmarkDefinition] {
        ladderTokens.flatMap { tokens in
            selectedKinds(kinds).compactMap { ladderEntry($0, tokens: tokens) }
        }
    }

    private static func smokeLocalBenchmarks(kinds: [BenchmarkType]) -> [BenchmarkDefinition] {
        selectedKinds(kinds).compactMap(smokeEntry)
    }

    /// Driven from `allCases`, so the output ignores the order and duplicates in `kinds`.
    private static func selectedKinds(_ kinds: [BenchmarkType]) -> [BenchmarkType] {
        BenchmarkType.allCases.filter(kinds.contains)
    }

    /// Exhaustive, so a new `BenchmarkType` forces a decision here.
    private static func ladderEntry(_ kind: BenchmarkType, tokens: Int32) -> BenchmarkDefinition? {
        switch kind {
        case .prefillThroughput:
            return .prefillThroughput(PrefillThroughputBenchmark(
                benchmarkId: BenchmarkId(validated: "prefill_throughput_\(tokens)"),
                parameterPrefillTokens: NonNegativeCount(validated: tokens)))
        case .decodeThroughput:
            return .decodeThroughput(DecodeThroughputBenchmark(
                benchmarkId: BenchmarkId(validated: "decode_throughput_\(tokens)_100"),
                parameterPrefillTokens: NonNegativeCount(validated: tokens),
                parameterDecodeTokens: PositiveCount(validated: 100)))
        case .endToEndLatency:
            return .endToEndLatency(EndToEndLatencyBenchmark(
                benchmarkId: BenchmarkId(validated: "end_to_end_latency_\(tokens)_256"),
                parameterPrefillTokens: NonNegativeCount(validated: tokens),
                parameterDecodeTokens: PositiveCount(validated: 256)))
        case .maxMemoryUsage:
            return .maxMemoryUsage(MaxMemoryUsageBenchmark(
                benchmarkId: BenchmarkId(validated: "max_memory_usage_\(tokens)"),
                parameterPrefillTokens: NonNegativeCount(validated: tokens)))
        case .eval, .vlThroughput, .vlMaxMemory, .vlSpecDecode, .asrThroughput, .asrMaxMemory:
            return nil
        }
    }

    /// `vl_spec_decode` has no smoke (or ladder) entry: its workload is a corpus the
    /// management server pins and serves, so there is no local definition to seed.
    private static func smokeEntry(_ kind: BenchmarkType) -> BenchmarkDefinition? {
        switch kind {
        case .prefillThroughput:
            return .prefillThroughput(PrefillThroughputBenchmark(
                benchmarkId: BenchmarkId(validated: "prefill_throughput_smoke"),
                parameterPrefillTokens: NonNegativeCount(validated: 8)))
        case .decodeThroughput:
            return .decodeThroughput(DecodeThroughputBenchmark(
                benchmarkId: BenchmarkId(validated: "decode_throughput_smoke"),
                parameterPrefillTokens: NonNegativeCount(validated: 8),
                parameterDecodeTokens: PositiveCount(validated: 8)))
        case .endToEndLatency:
            return .endToEndLatency(EndToEndLatencyBenchmark(
                benchmarkId: BenchmarkId(validated: "end_to_end_latency_smoke"),
                parameterPrefillTokens: NonNegativeCount(validated: 8),
                parameterDecodeTokens: PositiveCount(validated: 8)))
        case .maxMemoryUsage:
            return .maxMemoryUsage(MaxMemoryUsageBenchmark(
                benchmarkId: BenchmarkId(validated: "max_memory_usage_smoke"),
                parameterPrefillTokens: NonNegativeCount(validated: 8)))
        case .eval:
            return .eval(EvalBenchmark(
                benchmarkId: BenchmarkId(validated: "eval_smoke"),
                parameterEvalId: EvalId("eval_smoke"),
                parameterDatasetName: "local",
                parameterMaxTokens: PositiveCount(validated: 4),
                samples: [EvalSample(
                    id: "smoke-1",
                    messages: [["role": "user", "content": "Reply with exactly OK."]])]))
        case .vlThroughput:
            return .vlThroughput(VlThroughputBenchmark(
                benchmarkId: BenchmarkId(validated: "vl_throughput_smoke"),
                parameterImageWidth: PositiveCount(validated: 224),
                parameterImageHeight: PositiveCount(validated: 224),
                parameterTextTokens: NonNegativeCount(validated: 8),
                parameterDecodeTokens: PositiveCount(validated: 8),
                parameterNumImages: NonNegativeCount(validated: 1)))
        case .vlMaxMemory:
            return .vlMaxMemory(VlMaxMemoryBenchmark(
                benchmarkId: BenchmarkId(validated: "vl_max_memory_smoke"),
                parameterImageWidth: PositiveCount(validated: 224),
                parameterImageHeight: PositiveCount(validated: 224),
                parameterTextTokens: NonNegativeCount(validated: 8),
                parameterNumImages: NonNegativeCount(validated: 1)))
        // ASR runs from plans only, on the separate ASR build of llama.cpp.
        case .vlSpecDecode, .asrThroughput, .asrMaxMemory:
            return nil
        }
    }

    /// Idempotent: an existing entry is counted as updated.
    @discardableResult
    static func seedStandardLocal(
        _ store: BenchmarkStore, kinds: [BenchmarkType]
    ) throws -> LocalBenchmarkInitSummary {
        var summary = LocalBenchmarkInitSummary()
        for definition in standardLocalBenchmarks(kinds: kinds) {
            let existed = try store.get(.local(definition.benchmarkId)) != nil
            try store.putLocal(definition)
            if existed { summary.updated += 1 } else { summary.created += 1 }
        }
        return summary
    }
}

nonisolated struct LocalBenchmarkInitSummary: Equatable {
    var created = 0
    var updated = 0
}
