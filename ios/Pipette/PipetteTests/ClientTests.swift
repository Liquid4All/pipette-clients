import Foundation
import Testing

@testable import Pipette

/// `Client/Client.swift`: the crate's `client/mod.rs`, whose one function both `auth
/// reset` and a forced `auth import` run.
struct ClientTests {

    @Test func clearRemoteStateRemovesServerCachesAndKeepsLocal() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let definition = BenchmarkDefinition.maxMemoryUsage(MaxMemoryUsageBenchmark(
            benchmarkId: BenchmarkId(validated: "max_memory_usage_512"),
            parameterPrefillTokens: 512))
        try ws.benchmarks.cacheRemote(definition)
        try ws.benchmarks.putLocal(definition)
        let filed: [(BenchmarkResultLocation, String)] =
            [(.remotePending, "r-pending"), (.remoteSynced, "r-synced"), (.local, "r-local")]
        for (location, id) in filed {
            let payload = try #require(ws.results.payloadPath(location, CellId(id)))
            try FileManager.default.createDirectory(
                at: payload.deletingLastPathComponent(), withIntermediateDirectories: true)
            try Data("{}".utf8).write(to: payload)
        }
        let evals = ws.evalCompletions.root
        try FileManager.default.createDirectory(at: evals, withIntermediateDirectories: true)
        try Data("x".utf8).write(to: evals.appendingPathComponent("job-1.jsonl"))

        try clearRemoteState(benchmarks: ws.benchmarks, results: ws.results,
                             evals: ws.evalCompletions)

        #expect(ws.benchmarks.list(.remote).isEmpty)
        #expect(!FileManager.default.fileExists(
            atPath: ws.results.locationDir(.remotePending).path))
        #expect(!FileManager.default.fileExists(
            atPath: ws.results.locationDir(.remoteSynced).path))
        #expect(!FileManager.default.fileExists(atPath: evals.path))
        // Local benchmarks and local results are server-independent and kept.
        #expect(try ws.benchmarks.get(.local("max_memory_usage_512")) != nil)
        #expect(ws.results.location(of: CellId("r-local")) == .local)
    }

    /// Idempotent, as the crate's is: a device that never synced has nothing to clear.
    @Test func clearRemoteStateOnAnEmptyStoreSucceeds() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        try clearRemoteState(benchmarks: ws.benchmarks, results: ws.results,
                             evals: ws.evalCompletions)
    }
}
