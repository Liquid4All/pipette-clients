import Foundation
import Testing

@testable import Pipette

/// The sync filter consumes the server's `GET /benchmarks` array loosely:
/// recognized benchmarks become strict `BenchmarkDefinition`s, and an entry with
/// a `benchmark_type` this client doesn't know is skipped (not a hard failure),
/// so a newer server catalog never breaks the sync.
@Suite
struct BenchmarkSyncTests {
    /// Only fully-parseable entries are kept/stored: an unrecognized `benchmark_type`
    /// (`mystery`) and a known type missing a required parameter (`bad`, a schema
    /// mismatch) are both dropped. The per-id fetch is driven by the kept ids.
    @Test func keepParseableDropsUnknownTypeAndSchemaMismatch() {
        let json = #"""
            [
              {"benchmark_id":"p","benchmark_type":"prefill_throughput","parameter_prefill_tokens":512},
              {"benchmark_id":"mystery","benchmark_type":"some_future_metric","parameter_x":1},
              {"benchmark_id":"bad","benchmark_type":"decode_throughput","parameter_prefill_tokens":256},
              {"benchmark_id":"d","benchmark_type":"decode_throughput","parameter_prefill_tokens":256,"parameter_decode_tokens":100}
            ]
            """#
        let kept = BenchmarkSync.keepParseable(Data(json.utf8))
        #expect(kept.ids == ["p", "d"])
        #expect(kept.entries.count == 2)
    }

    @Test func keepParseableEmptyForMalformed() {
        #expect(BenchmarkSync.keepParseable(Data("not json".utf8)).ids.isEmpty)
    }

    /// The persisted state mirrors the Rust `RemoteSyncState` snake_case shape.
    @Test func syncStateDecodesSnakeCase() throws {
        let json = #"""
            {"benchmark_count":2,"benchmarks_etag":"\"abc\"","benchmark_etags":{"p":"\"e1\"","d":"\"e2\""}}
            """#
        let state = try JSONDecoder().decode(BenchmarkSync.SyncState.self, from: Data(json.utf8))
        #expect(state.benchmarkCount == 2)
        #expect(state.benchmarksEtag == "\"abc\"")
        #expect(state.benchmarkEtags["d"] == "\"e2\"")

        let roundTripped = try JSONDecoder().decode(
            BenchmarkSync.SyncState.self, from: JSONEncoder().encode(state))
        #expect(roundTripped.benchmarkEtags == state.benchmarkEtags)
    }
}
