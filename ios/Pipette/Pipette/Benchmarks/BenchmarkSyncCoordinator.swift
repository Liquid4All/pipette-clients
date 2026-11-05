import Foundation

actor BenchmarkSyncCoordinator {
    static let shared = BenchmarkSyncCoordinator()

    struct Key: Hashable {
        let serverURL: String
        let storageRoot: String
    }

    typealias SyncRunner = @Sendable (ServerUrl, BenchmarkStore) async throws -> Int

    private struct InFlight {
        let id: Int
        let task: Task<Int, Error>
    }

    private struct StorageTail {
        let id: Int
        let task: Task<Void, Never>
    }

    private var nextTaskId = 0
    private var inFlight: [Key: InFlight] = [:]
    private var storageTails: [String: StorageTail] = [:]
    private let syncRunner: SyncRunner

    init(syncRunner: @escaping SyncRunner = { serverUrl, store in
        try await pullRemoteBenchmarks(store: store, server: serverUrl)
    }) {
        self.syncRunner = syncRunner
    }

    /// Pull the catalog opportunistically on a UI event, such as a login completing.
    ///
    /// Never on a headless run: as in the crate, a headless sync happens only when asked
    /// for (the `sync` command). A headless launch still builds the SwiftUI scene, so
    /// without this guard a UI event could fire a network fetch and a disk write partway
    /// through a measured benchmark.
    nonisolated static func syncBestEffortInBackground(
        serverUrl: ServerUrl,
        ws: PipetteWorkspace,
        reason: String,
        clearCacheFirst: Bool = false
    ) {
        guard !HeadlessRunner.isHeadless else { return }
        Task.detached {
            await shared.syncBestEffort(
                serverUrl: serverUrl,
                ws: ws,
                reason: reason,
                clearCacheFirst: clearCacheFirst
            )
        }
    }

    @discardableResult
    func sync(serverUrl: ServerUrl, ws: PipetteWorkspace) async throws -> Int {
        try await sync(
            serverUrl: serverUrl,
            store: ws.benchmarks,
            key: Self.key(serverUrl: serverUrl, ws: ws)
        )
    }

    @discardableResult
    func syncAfterClearingCache(serverUrl: ServerUrl, ws: PipetteWorkspace) async throws -> Int {
        try await sync(
            serverUrl: serverUrl,
            store: ws.benchmarks,
            key: Self.key(serverUrl: serverUrl, ws: ws),
            clearCache: { ws.benchmarks.clearRemote() },
            coalesceExisting: false
        )
    }

    @discardableResult
    func sync(serverUrl: ServerUrl, store: BenchmarkStore, key: Key) async throws -> Int {
        try await sync(
            serverUrl: serverUrl,
            store: store,
            key: key,
            clearCache: nil,
            coalesceExisting: true
        )
    }

    @discardableResult
    private func sync(
        serverUrl: ServerUrl,
        store: BenchmarkStore,
        key: Key,
        clearCache: (@Sendable () -> Void)?,
        coalesceExisting: Bool
    ) async throws -> Int {
        if coalesceExisting, let existing = inFlight[key] {
            return try await existing.task.value
        }

        let runner = syncRunner
        let priorStorageTail = storageTails[key.storageRoot]?.task
        let taskId = makeTaskId()
        let task = Task.detached {
            await priorStorageTail?.value
            clearCache?()
            return try await runner(serverUrl, store)
        }
        let storageTail = Task {
            _ = try? await task.value
        }
        inFlight[key] = InFlight(id: taskId, task: task)
        storageTails[key.storageRoot] = StorageTail(id: taskId, task: storageTail)

        do {
            let count = try await task.value
            finish(key: key, taskId: taskId)
            return count
        } catch {
            finish(key: key, taskId: taskId)
            throw error
        }
    }

    func syncBestEffort(
        serverUrl: ServerUrl,
        ws: PipetteWorkspace,
        reason: String,
        clearCacheFirst: Bool = false
    ) async {
        do {
            if clearCacheFirst {
                _ = try await syncAfterClearingCache(serverUrl: serverUrl, ws: ws)
            } else {
                _ = try await sync(serverUrl: serverUrl, ws: ws)
            }
        } catch {
            AppLog.benchmarkSync.error("\(reason) sync failed: \(error)")
        }
    }

    private static func key(serverUrl: ServerUrl, ws: PipetteWorkspace) -> Key {
        Key(
            serverURL: serverUrl.value,
            storageRoot: ws.root.standardizedFileURL.path
        )
    }

    private func makeTaskId() -> Int {
        nextTaskId += 1
        return nextTaskId
    }

    private func finish(key: Key, taskId: Int) {
        if inFlight[key]?.id == taskId {
            inFlight[key] = nil
        }
        if storageTails[key.storageRoot]?.id == taskId {
            storageTails[key.storageRoot] = nil
        }
    }
}
