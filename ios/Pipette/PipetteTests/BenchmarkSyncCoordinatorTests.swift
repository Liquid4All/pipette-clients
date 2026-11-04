import Foundation
import Testing

@testable import Pipette

@Suite
struct BenchmarkSyncCoordinatorTests {
    @Test func coalescesConcurrentSameKeySyncs() async throws {
        let probe = CoordinatorProbe(mode: .delayedSuccess(3))
        let coordinator = BenchmarkSyncCoordinator(syncRunner: { serverUrl, store in
            try await probe.run(serverUrl: serverUrl, store: store)
        })
        let store = makeTemporaryBenchmarkStore()
        let key = BenchmarkSyncCoordinator.Key(serverURL: try testServer().value, storageRoot: "same")

        let tasks = (0..<8).map { _ in
            Task {
                try await coordinator.sync(serverUrl: try testServer(), store: store, key: key)
            }
        }

        var results: [Int] = []
        for task in tasks {
            results.append(try await task.value)
        }

        #expect(results.allSatisfy { $0 == 3 })
        #expect(await probe.callCount() == 1)
    }

    @Test func failedSyncCanRetry() async throws {
        let probe = CoordinatorProbe(mode: .failThenSucceed(5))
        let coordinator = BenchmarkSyncCoordinator(syncRunner: { serverUrl, store in
            try await probe.run(serverUrl: serverUrl, store: store)
        })
        let store = makeTemporaryBenchmarkStore()
        let key = BenchmarkSyncCoordinator.Key(serverURL: try testServer().value, storageRoot: "retry")

        do {
            _ = try await coordinator.sync(serverUrl: try testServer(), store: store, key: key)
            Issue.record("first sync should fail")
        } catch {
            #expect((error as? URLError)?.code == .timedOut)
        }

        let result = try await coordinator.sync(serverUrl: try testServer(), store: store, key: key)
        #expect(result == 5)
        #expect(await probe.callCount() == 2)
    }

    @Test func differentKeysDoNotCoalesce() async throws {
        let probe = CoordinatorProbe(mode: .delayedSuccess(7))
        let coordinator = BenchmarkSyncCoordinator(syncRunner: { serverUrl, store in
            try await probe.run(serverUrl: serverUrl, store: store)
        })
        let store = makeTemporaryBenchmarkStore()
        let firstKey = BenchmarkSyncCoordinator.Key(serverURL: try testServer().value, storageRoot: "first")
        let secondKey = BenchmarkSyncCoordinator.Key(serverURL: try testServer().value, storageRoot: "second")

        async let first = coordinator.sync(serverUrl: try testServer(), store: store, key: firstKey)
        async let second = coordinator.sync(serverUrl: try testServer(), store: store, key: secondKey)
        let firstResult = try await first
        let secondResult = try await second

        #expect(firstResult == 7)
        #expect(secondResult == 7)
        #expect(await probe.callCount() == 2)
    }

    @Test func differentServersOnSameStorageSerializeWrites() async throws {
        let probe = CoordinatorProbe(mode: .delayedSuccess(11))
        let coordinator = BenchmarkSyncCoordinator(syncRunner: { serverUrl, store in
            try await probe.run(serverUrl: serverUrl, store: store)
        })
        let store = makeTemporaryBenchmarkStore()
        let firstKey = BenchmarkSyncCoordinator.Key(serverURL: try testServer().value, storageRoot: "shared")
        let secondServerURL = try ServerUrl("https://other-benchmarks.example")
        let secondKey = BenchmarkSyncCoordinator.Key(serverURL: secondServerURL.value, storageRoot: "shared")

        async let first = coordinator.sync(serverUrl: try testServer(), store: store, key: firstKey)
        async let second = coordinator.sync(serverUrl: secondServerURL, store: store, key: secondKey)
        let firstResult = try await first
        let secondResult = try await second

        #expect(firstResult == 11)
        #expect(secondResult == 11)
        #expect(await probe.callCount() == 2)
        #expect(await probe.maxConcurrentCount() == 1)
    }

    @Test func clearAndSyncWaitsForPriorStorageWork() async throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let firstServerURL = try ServerUrl("https://first-benchmarks.example")
        let secondServerURL = try ServerUrl("https://second-benchmarks.example")
        let probe = CacheClearProbe()
        let coordinator = BenchmarkSyncCoordinator(syncRunner: { serverUrl, store in
            if serverUrl.value == firstServerURL.value {
                try store.putRemoteIndex(try Coding.decoder.decode([PipetteMgmtClient.BenchmarkSummary].self, from: Data(#"[{"benchmark_id":"stale","benchmark_type":"prefill_throughput"}]"#.utf8)))
                await probe.markFirstStarted()
                try await Task.sleep(nanoseconds: 50_000_000)
                return 1
            }

            await probe.recordSecondSawCleared(try store.listRemoteIndex().isEmpty)
            return 2
        })

        let firstTask = Task {
            try await coordinator.sync(serverUrl: firstServerURL, ws: ws)
        }
        await probe.waitForFirstStarted()
        let secondTask = Task {
            try await coordinator.syncAfterClearingCache(serverUrl: secondServerURL, ws: ws)
        }

        #expect(try await firstTask.value == 1)
        #expect(try await secondTask.value == 2)
        #expect(await probe.secondSawCleared())
    }
}

/// The collector these tests sync from. A function, not a constant: ``ServerUrl``
/// validates what it is given, and a global cannot throw.
private func testServer() throws -> ServerUrl { try ServerUrl("https://benchmarks.example") }

private actor CoordinatorProbe {
    enum Mode {
        case delayedSuccess(Int)
        case failThenSucceed(Int)
    }

    private let mode: Mode
    private var calls = 0
    private var activeCalls = 0
    private var maxActiveCalls = 0

    init(mode: Mode) {
        self.mode = mode
    }

    func run(serverUrl _: ServerUrl, store _: BenchmarkStore) async throws -> Int {
        calls += 1
        activeCalls += 1
        maxActiveCalls = max(maxActiveCalls, activeCalls)
        defer { activeCalls -= 1 }
        switch mode {
        case .delayedSuccess(let value):
            try await Task.sleep(nanoseconds: 50_000_000)
            return value
        case .failThenSucceed(let value):
            if calls == 1 {
                throw URLError(.timedOut)
            }
            return value
        }
    }

    func callCount() -> Int {
        calls
    }

    func maxConcurrentCount() -> Int {
        maxActiveCalls
    }
}

private actor CacheClearProbe {
    private var firstStarted = false
    private var firstStartedContinuation: CheckedContinuation<Void, Never>?
    private var secondCleared = false

    func markFirstStarted() {
        firstStarted = true
        firstStartedContinuation?.resume()
        firstStartedContinuation = nil
    }

    func waitForFirstStarted() async {
        guard !firstStarted else { return }
        await withCheckedContinuation { continuation in
            firstStartedContinuation = continuation
        }
    }

    func recordSecondSawCleared(_ value: Bool) {
        secondCleared = value
    }

    func secondSawCleared() -> Bool {
        secondCleared
    }
}
