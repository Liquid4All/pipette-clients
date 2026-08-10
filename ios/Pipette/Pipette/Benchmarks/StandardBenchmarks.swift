import Foundation

/// The built-in local ladder + smoke definitions — the crate's
/// `pipette-cli/src/benchmarks/standard.rs`.
///
/// These are generated on the device, not synced, so their results are never submitted
/// (see ``BenchmarkSource``). They exist so a device can measure before it has a
/// registration, and so an operator can run the same ladder the desktop clients run.
///
/// The caller declares which kinds it can execute; entries are emitted in
/// `BenchmarkType` declaration order so the catalog is stable regardless of the order
/// or duplicates in `kinds`. Note the declaration order differs from the crate's, so
/// the two clients emit the same *set* in a different order — the store sorts by id on
/// read, which is where order is actually observable.
///
/// TODO: review — mirrors `standard.rs`; the ladder values, the id format strings and
/// the smoke parameters were checked against it entry by entry.
nonisolated enum StandardBenchmarks {
    /// Token ladder shared across every kind that has a ladder form. Tuning these is a
    /// methodology question; the port keeps the crate's values verbatim.
    static let ladderTokens: [UInt32] = [100, 256, 512, 1024, 2048, 4096, 8192]

    /// Counts from ``seedLocal(into:kinds:)`` — the crate's `LocalBenchmarkInitSummary`.
    struct SeedSummary: Equatable {
        var created = 0
        var updated = 0
    }

    /// The standard set for `kinds`: the ladder plus one smoke entry per kind.
    static func all(kinds: [BenchmarkType]) -> [BenchmarkDefinition] {
        ladder(kinds: kinds) + smoke(kinds: kinds)
    }

    static func ladder(kinds: [BenchmarkType]) -> [BenchmarkDefinition] {
        ladderTokens.flatMap { tokens in
            selected(kinds).compactMap { ladderEntry($0, tokens: tokens) }
        }
    }

    static func smoke(kinds: [BenchmarkType]) -> [BenchmarkDefinition] {
        selected(kinds).map(smokeEntry)
    }

    /// The kinds in canonical (declaration) order, restricted to those selected —
    /// driving from `allCases` makes the output independent of `kinds`' order and
    /// tolerant of duplicates.
    private static func selected(_ kinds: [BenchmarkType]) -> [BenchmarkType] {
        BenchmarkType.allCases.filter(kinds.contains)
    }

    /// Ladder entry for `kind` at `tokens`, or nil for a kind with no ladder form.
    /// The switch is exhaustive, so a new `BenchmarkType` forces a decision here.
    private static func ladderEntry(_ kind: BenchmarkType, tokens: UInt32) -> BenchmarkDefinition? {
        switch kind {
        case .prefillThroughput:
            return .prefillThroughput(
                benchmarkId: "prefill_throughput_\(tokens)", prefillTokens: tokens)
        case .decodeThroughput:
            return .decodeThroughput(
                benchmarkId: "decode_throughput_\(tokens)_100",
                prefillTokens: tokens, decodeTokens: 100)
        case .endToEndLatency:
            return .endToEndLatency(
                benchmarkId: "end_to_end_latency_\(tokens)_256",
                prefillTokens: tokens, decodeTokens: 256)
        case .maxMemoryUsage:
            return .maxMemoryUsage(
                benchmarkId: "max_memory_usage_\(tokens)", prefillTokens: tokens)
        case .eval, .vlThroughput:
            return nil
        }
    }

    /// Smoke entry for `kind`. Every kind has one; exhaustive for the same reason.
    private static func smokeEntry(_ kind: BenchmarkType) -> BenchmarkDefinition {
        switch kind {
        case .prefillThroughput:
            return .prefillThroughput(benchmarkId: "prefill_throughput_smoke", prefillTokens: 8)
        case .decodeThroughput:
            return .decodeThroughput(
                benchmarkId: "decode_throughput_smoke", prefillTokens: 8, decodeTokens: 8)
        case .endToEndLatency:
            return .endToEndLatency(
                benchmarkId: "end_to_end_latency_smoke", prefillTokens: 8, decodeTokens: 8)
        case .maxMemoryUsage:
            return .maxMemoryUsage(benchmarkId: "max_memory_usage_smoke", prefillTokens: 8)
        case .eval:
            return .eval(
                benchmarkId: "eval_smoke",
                evalId: EvalId("eval_smoke"),
                datasetName: "local",
                maxTokens: 4,
                mcqChoices: nil,
                samples: [EvalSample(
                    id: "smoke-1",
                    messages: [["role": "user", "content": "Reply with exactly OK."]])])
        case .vlThroughput:
            return .vlThroughput(
                benchmarkId: "vl_throughput_smoke",
                imageWidth: 224, imageHeight: 224, textTokens: 8, decodeTokens: 8)
        }
    }

    /// Write the standard set for `kinds` into the store's local half — the crate's
    /// `seed_standard_local`. Catalog policy lives here; the store only receives ready
    /// definitions. Idempotent: a second call updates rather than double-creating.
    @discardableResult
    static func seedLocal(into store: BenchmarkStore, kinds: [BenchmarkType]) throws -> SeedSummary {
        var summary = SeedSummary()
        for definition in all(kinds: kinds) {
            let existed = store.get(.local(definition.benchmarkId)) != nil
            try store.put(.local, definition)
            if existed { summary.updated += 1 } else { summary.created += 1 }
        }
        return summary
    }
}
