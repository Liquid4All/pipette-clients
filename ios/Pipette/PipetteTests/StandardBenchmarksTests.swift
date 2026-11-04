import Foundation
import Testing

@testable import Pipette

/// The generated local half, mirroring `pipette-cli`'s `benchmarks/standard.rs`.
struct StandardBenchmarksTests {

    // MARK: - The standard local set

    /// What `benchmarks init-local` seeds on this client: the ladder over the four ladder
    /// kinds and a smoke entry for each of the seven it can run (7×4 + 7). The crate pins its
    /// own `init-local` the same way, over every kind, in `every_kind_seeds_thirty_eight`.
    @MainActor @Test func initLocalSeedsThirtyFive() {
        #expect(StandardBenchmarks.standardLocalBenchmarks(kinds: BenchmarksCommands.runnableKinds)
            .count == 35)
    }

    @Test func kindsWithoutALadderContributeOnlySmoke() {
        #expect(StandardBenchmarks.standardLocalBenchmarks(kinds: [.eval, .vlThroughput, .vlMaxMemory])
            .map(\.benchmarkId.value) == ["eval_smoke", "vl_throughput_smoke", "vl_max_memory_smoke"])
    }

    /// The corpus comes from the server, so there is nothing local to seed for
    /// `vl_spec_decode`.
    @Test func vlSpecDecodeSeedsNothing() {
        #expect(StandardBenchmarks.standardLocalBenchmarks(kinds: [.vlSpecDecode]).isEmpty)
    }

    /// The crate's `kinds_are_a_set`.
    @Test func outputIsIndependentOfTheKindsOrder() {
        let a = StandardBenchmarks.standardLocalBenchmarks(kinds: [.decodeThroughput, .prefillThroughput])
        let b = StandardBenchmarks.standardLocalBenchmarks(kinds: [.prefillThroughput, .decodeThroughput, .prefillThroughput])
        #expect(a.map(\.benchmarkId) == b.map(\.benchmarkId))
    }

    /// The warehouse groups on these ids, so a divergence from the crate's format strings
    /// would split one benchmark into two rows (`the_ladder_ids_are_stable`).
    @Test func ladderIdsMatchTheCratesFormatStrings() {
        let ids = Set(StandardBenchmarks.standardLocalBenchmarks(kinds: BenchmarkType.allCases)
            .map(\.benchmarkId.value))
        for expected in ["prefill_throughput_512", "decode_throughput_512_100",
                         "end_to_end_latency_512_256", "max_memory_usage_512",
                         "eval_smoke", "vl_throughput_smoke"] {
            #expect(ids.contains(expected), "missing \(expected)")
        }
    }

    // MARK: - Seeding

    /// The crate's `seed_standard_then_get_local`.
    @Test func seedingTwiceUpdatesRatherThanDuplicating() throws {
        let store = makeTemporaryBenchmarkStore()
        defer { removeStore(store) }

        let first = try StandardBenchmarks.seedStandardLocal(store, kinds: [.prefillThroughput])
        #expect(first.created > 0)
        #expect(first.updated == 0)

        let again = try StandardBenchmarks.seedStandardLocal(store, kinds: [.prefillThroughput])
        #expect(again.created == 0)
        #expect(again.updated == first.created)
        #expect(store.list(.local).count == first.created)
    }

    /// The local half's encoder has to produce what the decoder reads.
    @Test func aSeededDefinitionRoundTrips() throws {
        let store = makeTemporaryBenchmarkStore()
        defer { removeStore(store) }

        try StandardBenchmarks.seedStandardLocal(store, kinds: [.decodeThroughput, .eval])

        #expect(try store.get(.local("decode_throughput_512_100"))
            == .decodeThroughput(DecodeThroughputBenchmark(
                benchmarkId: BenchmarkId(validated: "decode_throughput_512_100"),
                parameterPrefillTokens: 512,
                parameterDecodeTokens: 100)))
        // The eval smoke entry keeps its sample through the round trip.
        let eval = try #require(try store.get(.local("eval_smoke"))).asEval()
        #expect(eval.parameterEvalId.rawValue == "eval_smoke")
        #expect(eval.parameterDatasetName == "local")
        #expect(eval.parameterMaxTokens == 4)
        #expect(eval.samples?.count == 1)
    }
}
